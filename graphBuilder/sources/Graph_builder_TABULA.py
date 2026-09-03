"""
TABULA JSON → enriched TTL. Pipeline step: python graphBuilder/run_pipeline.py --tabula

Renovation-aware layer model
----------------------------
* existing              → one layer from type_of_construction (as-built description)
* usual_refurbishment   → existing layer URI(s) linked + new refurbishment layer(s)
* advanced_refurbishment→ existing layer URI(s) linked + new refurbishment layer(s)
* doors / windows       → no LayerSet; bmp:hasLayerFunction on the element state directly
"""

import json
import os
import re
import sys
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef, XSD
from rdflib.namespace import OWL, RDF, RDFS

BASE_DIR = Path(__file__).resolve().parents[2]
_GRAPH_BUILDER_DIR = Path(__file__).resolve().parents[1]
if str(_GRAPH_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_BUILDER_DIR))

from ontology_reasoning import load_ontology_corpus, load_ontology_graph, process_description
from adapters.tabula import iter_tabula_stage_descriptions
from ontology_reasoning.material_axioms import ensure_layer_material_pair, resolve_category_and_type
from ontology_reasoning.rdf_layers import emit_enforced_layer, nlp_layer_json_to_tabula_layerset
from ontology_reasoning.validation_report import begin_report, finalize_report, record_ctx

# Set to a building id to process/print one building only; None for full dataset.
#TEST_BUILDING_ID = "DE.N.MFH.03.Gen.ReEx.001"
TEST_BUILDING_ID = None

TABULA = Namespace("https://namedgraphs.org/tabula#")
AT = Namespace("https://w3id.org/at#")
LCA = Namespace("https://w3id.org/lca#")
BOT = Namespace("https://w3id.org/bot#")
BEO = Namespace("https://w3id.org/beo#")
BMP = Namespace("https://w3id.org/bmp#")
UNIT = Namespace("http://qudt.org/vocab/unit/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
DC_IDENTIFIER = URIRef("http://purl.org/dc/elements/1.1/identifier")

BMP_LOAD_BEARING = URIRef("https://w3id.org/bmp#LoadBearing")
BMP_INSULATING = URIRef("https://w3id.org/bmp#Insulating")
BMP_GLAZING = URIRef("https://w3id.org/bmp#Glazing")
BMP_OPENING = URIRef("https://w3id.org/bmp#Opening")
BMP_FINISHING = URIRef("https://w3id.org/bmp#Finishing")
BMP_SINGLE_LAYER = URIRef("https://w3id.org/bmp#SingleLayer")
BMP_MULTI_LAYER = URIRef("https://w3id.org/bmp#MultiLayer")

INSULATION_PATTERN = re.compile(
    r"\b(insulation|insulated|dämmung|daemmung|dämmkern|daemmkern|"
    r"wärmedämm|waermedaemm|dämmstoff|daemmstoff|gedämmt|gedaemmt)\b",
    re.IGNORECASE,
)
GLAZING_PATTERN = re.compile(
    r"\b(glazing|glazed|verglasung|scheiben|fenster|window|"
    r"triple glazing|double glazing|isolierverglasung)\b",
    re.IGNORECASE,
)
OPENING_PATTERN = re.compile(r"\b(door|tür|tur|haustür|entry door|metal door)\b", re.IGNORECASE)
THICKNESS_CM_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*cm", re.IGNORECASE)
THICKNESS_MM_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm", re.IGNORECASE)

BRICK_PATTERN = re.compile(r"\b(brick|brickwork|ziegel|mauerwerk)\b", re.IGNORECASE)
CONCRETE_PATTERN = re.compile(r"\b(concrete|beton|ortbeton|stahlbeton)\b", re.IGNORECASE)
WOOD_PATTERN = re.compile(
    r"\b(wood|timber|holz|clt|sparren|balken|faserplatte|fibreboard|fiberboard)\b|holz",
    re.IGNORECASE,
)
GLASS_PATTERN = re.compile(r"\b(glass|glas|glazing|verglasung|scheiben)\b", re.IGNORECASE)
POLYMER_PATTERN = re.compile(r"\b(plastic|kunststoff|pvc|polymer)\b", re.IGNORECASE)
STEEL_PATTERN = re.compile(r"\b(steel|stahl)\b", re.IGNORECASE)


def combined_description(german_desc, english_desc):
    return " ".join(part for part in (german_desc or "", english_desc or "") if part).strip()


def mentions_insulation(text):
    return bool(text and INSULATION_PATTERN.search(text))


def extract_thickness_cm(text):
    if not text:
        return None
    cm_match = THICKNESS_CM_PATTERN.search(text)
    if cm_match:
        return float(cm_match.group(1).replace(",", "."))
    mm_match = THICKNESS_MM_PATTERN.search(text)
    if mm_match:
        return float(mm_match.group(1).replace(",", ".")) / 10.0
    return None


def resolve_material_from_text(text, element_label=""):
    """Map construction/measure text to bmp:MaterialCategory (e.g. bmp:Brick)."""
    combined = f"{text} {element_label}".lower()
    if BRICK_PATTERN.search(combined):
        return BMP.Brick
    if CONCRETE_PATTERN.search(combined):
        return BMP.Concrete
    if WOOD_PATTERN.search(combined):
        return BMP.WoodTimber
    if POLYMER_PATTERN.search(combined):
        return BMP.Polymer
    if GLASS_PATTERN.search(combined):
        return BMP.Glass
    if STEEL_PATTERN.search(combined):
        return BMP.Concrete
    if mentions_insulation(combined):
        return BMP.MineralFibre
    return None


def resolve_existing_layer_function(text, element_label):
    combined = f"{text} {element_label}".lower()
    label_lower = element_label.lower()

    if "window" in label_lower or ("fenster" in combined and "door" not in label_lower):
        return BMP_GLAZING
    if "door" in label_lower or OPENING_PATTERN.search(combined):
        return BMP_OPENING
    if mentions_insulation(combined):
        return BMP_INSULATING
    if any(token in combined for token in ("cladding", "verputz", "coating")):
        return BMP_FINISHING
    return BMP_LOAD_BEARING


def make_layer(
    layer_index,
    function_iri,
    material_category_iri=None,
    material_type_iri=None,
    thickness_cm=None,
    *,
    material_iri=None,
    ontology_graph=None,
):
    layer = {
        "layer_index": layer_index,
        "function_iri": function_iri,
        "thickness_cm": thickness_cm,
    }
    if material_category_iri is None and material_iri is not None:
        material_category_iri = material_iri
    if ontology_graph is not None and (
        material_category_iri is not None or material_type_iri is not None or material_iri is not None
    ):
        cat, typ = ensure_layer_material_pair(
            ontology_graph,
            category_iri=material_category_iri or material_iri,
            type_iri=material_type_iri,
        )
        if cat is not None:
            layer["material_category_iri"] = cat
            layer["material_type_iri"] = typ
    elif material_category_iri is not None:
        layer["material_category_iri"] = material_category_iri
        layer["material_type_iri"] = material_type_iri or material_category_iri
    return layer


def build_existing_layerset(construction_de, construction_eng, element_label, ontology_graph=None):
    """
    Existing renovation state: exactly one layer from the as-built description.
    Example: 'Vollziegel-Mauerwerk' / 'brickwork' → LoadBearing + Brick.
    """
    combined = combined_description(construction_de, construction_eng)
    if not combined:
        return None

    function_iri = resolve_existing_layer_function(combined, element_label)
    material_iri = resolve_material_from_text(combined, element_label)
    thickness_cm = extract_thickness_cm(combined) if mentions_insulation(combined) else None

    return {
        "topology": BMP_SINGLE_LAYER,
        "layers": [
            make_layer(
                1,
                function_iri,
                material_iri,
                thickness_cm=thickness_cm,
                ontology_graph=ontology_graph,
            )
        ],
    }


def build_renovation_delta_layer(measure_de, measure_eng, element_label, layer_index, ontology_graph=None):
    """Extract the refurbishment addition from usual/advanced measure text."""
    combined = combined_description(measure_de, measure_eng)
    if not combined:
        return None

    label_lower = element_label.lower()

    if "window" in label_lower or "fenster" in combined.lower():
        if GLAZING_PATTERN.search(combined):
            return make_layer(
                layer_index,
                BMP_GLAZING,
                BMP.Glass,
                ontology_graph=ontology_graph,
            )
        if mentions_insulation(combined):
            return make_layer(
                layer_index,
                BMP_INSULATING,
                BMP.MineralFibre,
                thickness_cm=extract_thickness_cm(combined),
                ontology_graph=ontology_graph,
            )

    if mentions_insulation(combined):
        return make_layer(
            layer_index,
            BMP_INSULATING,
            BMP.MineralFibre,
            thickness_cm=extract_thickness_cm(combined),
            ontology_graph=ontology_graph,
        )

    if GLAZING_PATTERN.search(combined):
        return make_layer(
            layer_index,
            BMP_GLAZING,
            BMP.Glass,
            ontology_graph=ontology_graph,
        )

    return None


def build_refurbishment_layerset(
    existing_layer_uris, measure_de, measure_eng, element_label, ontology_graph=None
):
    """
    Usual/Advanced: link to existing layer URI(s), then append new refurbishment layer(s).
    The as-built layer is not duplicated — only the delta layer is minted.
    """
    if not existing_layer_uris:
        return None

    layers = [
        {"layer_index": idx, "reuse_uri": layer_uri}
        for idx, layer_uri in enumerate(existing_layer_uris, start=1)
    ]

    delta_layer = build_renovation_delta_layer(
        measure_de,
        measure_eng,
        element_label,
        layer_index=len(layers) + 1,
        ontology_graph=ontology_graph,
    )
    if delta_layer:
        layers.append(delta_layer)

    topology = BMP_MULTI_LAYER if len(layers) > 1 else BMP_SINGLE_LAYER
    return {"topology": topology, "layers": layers}


def layerset_from_nlp(desc_de, desc_eng, subject_uri, ontology_corpus, profile="tabula", ontology_graph=None):
    """Run OWL NLP when descriptions exist; return TABULA layerset dict or None."""
    if not combined_description(desc_de, desc_eng):
        return None
    ctx = process_description(
        subject_uri=subject_uri,
        german_desc=desc_de,
        english_desc=desc_eng,
        ontology_corpus=ontology_corpus,
        profile=profile,
        ontology_graph=ontology_graph,
    )
    record_ctx(subject_uri, ctx, source="tabula")
    if not ctx or not ctx.layer_json:
        return None
    layer_json = ctx.layer_json
    if isinstance(layer_json, list):
        layer_json = next((item for item in layer_json if item), None)
    if not isinstance(layer_json, dict):
        return None
    return nlp_layer_json_to_tabula_layerset(layer_json, ontology_graph=ontology_graph)


def nlp_delta_layer(desc_de, desc_eng, subject_uri, ontology_corpus, layer_index, ontology_graph=None):
    layerset = layerset_from_nlp(
        desc_de, desc_eng, subject_uri, ontology_corpus, ontology_graph=ontology_graph
    )
    if not layerset or not layerset.get("layers"):
        return None
    delta = layerset["layers"][-1]
    delta["layer_index"] = layer_index
    return delta


def build_refurbishment_layerset_with_nlp(
    existing_layer_uris,
    measure_de,
    measure_eng,
    element_label,
    subject_uri,
    ontology_corpus,
    ontology_graph=None,
):
    layerset = build_refurbishment_layerset(
        existing_layer_uris,
        measure_de,
        measure_eng,
        element_label,
        ontology_graph=ontology_graph,
    )
    if not layerset:
        return layerset

    nlp_layer = nlp_delta_layer(
        measure_de,
        measure_eng,
        subject_uri,
        ontology_corpus,
        layer_index=len(layerset["layers"]),
        ontology_graph=ontology_graph,
    )
    if nlp_layer and layerset["layers"] and not layerset["layers"][-1].get("reuse_uri"):
        layerset["layers"][-1] = nlp_layer
    elif nlp_layer:
        layerset["layers"].append(nlp_layer)
        layerset["topology"] = BMP_MULTI_LAYER if len(layerset["layers"]) > 1 else BMP_SINGLE_LAYER
    return layerset


def init_graph():
    g = Graph()
    g.bind("tabula", TABULA)
    g.bind("at", AT)
    g.bind("lca", LCA)
    g.bind("bot", BOT)
    g.bind("beo", BEO)
    g.bind("bmp", BMP)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("unit", UNIT)
    g.bind("dc1", DC_IDENTIFIER)
    g.bind("xsd", XSD)
    g.bind("qudt", QUDT)

    g.add((AT.RenovationStatus, RDF.type, OWL.Class))
    g.add((AT.NoRenovation, RDF.type, OWL.Class))
    g.add((AT.MinorRenovation, RDF.type, OWL.Class))
    g.add((AT.MajorRenovation, RDF.type, OWL.Class))
    g.add((AT.NoRenovation, RDFS.subClassOf, AT.RenovationStatus))
    g.add((AT.MinorRenovation, RDFS.subClassOf, AT.RenovationStatus))
    g.add((AT.MajorRenovation, RDFS.subClassOf, AT.RenovationStatus))
    return g


def load_tabula_buildings():
    ab_path = BASE_DIR / "data" / "data-json-TABULA" / "apartment" / "ab_buildings_combined.json"
    mfh_path = BASE_DIR / "data" / "data-json-TABULA" / "multifamilyhouse" / "mfh_buildings_combined.json"
    data_ab = json.loads(ab_path.read_text(encoding="utf-8"))
    data_mfh = json.loads(mfh_path.read_text(encoding="utf-8"))
    buildings = data_ab["buildings"] + data_mfh["buildings"]

    test_id = TEST_BUILDING_ID
    if test_id is None and os.environ.get("TABULA_TEST_BUILDING_ID"):
        test_id = os.environ["TABULA_TEST_BUILDING_ID"]
    if test_id:
        buildings = [b for b in buildings if b["building"]["id"] == test_id]
        if not buildings:
            raise ValueError(f"TABULA test building not found: {test_id!r}")
    return buildings


def extract_building_kv_pairs(building_node):
    b_info = building_node["building"]
    kv = {
        "building_id": b_info["id"],
        "building_json": b_info["building_json"],
        "building_image": b_info["building_image"],
        "size_class": b_info["size_class"],
        "construction_period": b_info["construction_period"],
        "reference_floor_area_m2": float(b_info["reference_floor_area_m2"]),
        "heat_supply_type": b_info["heat_supply_system"]["type"],
        "climate_region": b_info["climate_region"],
    }

    if "energy_need_for_heating" in building_node:
        energy_data = building_node["energy_need_for_heating"]
        kv["heating_demand_existing"] = energy_data["existing_state"]
        kv["heating_demand_usual"] = energy_data["usual_refurbishment"]
        advanced = energy_data["advanced_refurbishment"]
        kv["heating_demand_advanced"] = advanced["total"] if isinstance(advanced, dict) else advanced
    elif "energy" in building_node and "heating_demand" in building_node["energy"]:
        energy_data = building_node["energy"]["heating_demand"]["states"]
        kv["heating_demand_existing"] = energy_data["existing"]
        kv["heating_demand_usual"] = energy_data["usual_refurbishment"]
        advanced = energy_data["advanced_refurbishment"]
        kv["heating_demand_advanced"] = advanced["total"] if isinstance(advanced, dict) else advanced

    kv["elements"] = []
    for key, data in building_node["elements"].items():
        kv["elements"].append(
            {
                "label": data.get("name", key).title(),
                "surface_area_m2": float(data["surface_area_m2"]),
                "construction_type_eng": data.get("type_of_construction_eng")
                or data.get("type_of_construction", ""),
                "construction_type_de": data.get("type_of_construction_de")
                or data.get("type_of_construction", ""),
                "existing_u_value": float(data["states"]["existing"]["u_value_W_m2K"]),
                "usual_u_value": float(data["states"]["usual_refurbishment"]["u_value_W_m2K"]),
                "usual_measure": data["states"]["usual_refurbishment"].get("measure_eng")
                or data["states"]["usual_refurbishment"].get("measure", ""),
                "usual_measure_de": data["states"]["usual_refurbishment"].get("measure_de")
                or data["states"]["usual_refurbishment"].get("measure", ""),
                "advanced_u_value": float(data["states"]["advanced_refurbishment"]["u_value_W_m2K"]),
                "advanced_measure": data["states"]["advanced_refurbishment"].get("measure_eng")
                or data["states"]["advanced_refurbishment"].get("measure", ""),
                "advanced_measure_de": data["states"]["advanced_refurbishment"].get("measure_de")
                or data["states"]["advanced_refurbishment"].get("measure", ""),
            }
        )
    return kv


def resolve_beo_element_type(element_label):
    label_lower = element_label.lower()
    if "roof" in label_lower:
        return BEO.Roof
    if "wall" in label_lower:
        return BEO.Wall
    if "floor" in label_lower or "slab" in label_lower:
        return BEO.Slab
    if "door" in label_lower:
        return BEO.Door
    if "window" in label_lower:
        return BEO.Window
    return BEO.BuiltElement


def is_opening_element(element_label):
    beo_type = resolve_beo_element_type(element_label)
    return beo_type in (BEO.Door, BEO.Window)


def opening_layer_function(element_label):
    """Doors and windows have no layers — function is attached to the element state."""
    if "window" in element_label.lower():
        return BMP_GLAZING
    if "door" in element_label.lower():
        return BMP_OPENING
    return None


def assign_element_types(g, target_uri, element_label):
    g.add((target_uri, RDF.type, OWL.NamedIndividual))
    g.add((target_uri, RDF.type, BEO.BuiltElement))
    g.add((target_uri, RDF.type, AT.ElementArchetype))
    g.add((target_uri, RDF.type, resolve_beo_element_type(element_label)))


def layer_uri(safe_id, state_suffix, layer_index):
    return URIRef(TABULA[f"Building_{safe_id}_Layer_{state_suffix}_L{layer_index}"])


def emit_layer_triples(g, layer_uri_ref, layer, ontology_graph=None):
    g.add((layer_uri_ref, RDF.type, OWL.NamedIndividual))
    if ontology_graph is None:
        raise ValueError("ontology_graph is required to emit enforced layer/material RDF")

    material_uri = URIRef(f"{layer_uri_ref}_Mat")
    emit_enforced_layer(
        g,
        layer_uri=layer_uri_ref,
        material_uri=material_uri,
        function_iri=layer["function_iri"],
        category_iri=layer.get("material_category_iri") or layer.get("material_iri"),
        type_iri=layer.get("material_type_iri"),
        ontology_graph=ontology_graph,
        layer_types=(BMP.Layer,),
    )

    if layer.get("thickness_cm") is not None:
        thick_node = BNode()
        g.add((layer_uri_ref, BMP.hasThickness, thick_node))
        g.add((thick_node, RDF.value, Literal(float(layer["thickness_cm"]), datatype=XSD.float)))
        g.add((thick_node, QUDT.unit, UNIT["CentiM"]))


def emit_layerset(g, safe_id, state_suffix, element_state_uri, layerset, ontology_graph=None):
    layerset_uri = TABULA[f"Building_{safe_id}_LayerSet_{state_suffix}"]
    g.add((element_state_uri, BMP.hasLayerSet, layerset_uri))
    g.add((layerset_uri, RDF.type, OWL.NamedIndividual))
    g.add((layerset_uri, RDF.type, BMP.LayerSet))
    g.add((layerset_uri, RDF.type, layerset["topology"]))

    emitted_layer_uris = []
    for layer in layerset["layers"]:
        if layer.get("reuse_uri"):
            layer_uri_ref = layer["reuse_uri"]
            g.add((layerset_uri, BMP.hasLayer, layer_uri_ref))
            emitted_layer_uris.append(layer_uri_ref)
            continue

        idx = layer["layer_index"]
        layer_uri_ref = layer_uri(safe_id, state_suffix, idx)
        g.add((layerset_uri, BMP.hasLayer, layer_uri_ref))
        emit_layer_triples(g, layer_uri_ref, layer, ontology_graph=ontology_graph)
        emitted_layer_uris.append(layer_uri_ref)

    return emitted_layer_uris


def layer_summary_text(layers):
    parts = []
    for layer in layers:
        if layer.get("reuse_uri"):
            parts.append(f"L{layer['layer_index']}:reuse({layer['reuse_uri'].split('_')[-1]})")
            continue
        fn = str(layer["function_iri"]).split("#")[-1]
        cat = layer.get("material_category_iri") or layer.get("material_iri")
        mat = str(cat).split("#")[-1] if cat else "-"
        typ = layer.get("material_type_iri")
        if typ and str(typ) != str(cat):
            mat = f"{mat}/{str(typ).split('#')[-1]}"
        thick = f"@{layer['thickness_cm']}cm" if layer.get("thickness_cm") is not None else ""
        parts.append(f"L{layer['layer_index']}:{fn}/{mat}{thick}")
    return ", ".join(parts)


def process_building(g, building_node, ontology_corpus=None, ontology_graph=None):
    kv = extract_building_kv_pairs(building_node)
    building_id = kv["building_id"]
    safe_id = building_id.replace(".", "_")
    building_uri = URIRef(TABULA[f"Building_{safe_id}"])

    print(f"\nProcessing building: {building_id}")

    g.add((building_uri, RDF.type, OWL.NamedIndividual))
    g.add((building_uri, RDF.type, BOT.Building))
    g.add((building_uri, RDF.type, AT.BuildingArchetype))
    if kv["size_class"] == "AB":
        g.add((building_uri, AT.hasResidentialType, AT.ApartmentHouse))
    elif kv["size_class"] == "MFH":
        g.add((building_uri, AT.hasResidentialType, AT.MultiFamilyHouse))

    g.add((building_uri, RDFS.label, Literal(building_id, datatype=XSD.string)))
    g.add((building_uri, DC_IDENTIFIER, Literal(building_id, datatype=XSD.string)))
    g.add((building_uri, AT.hasSourceJSON, Literal(kv["building_json"], datatype=XSD.string)))
    if kv["building_image"]:
        g.add((building_uri, AT.hasImage, Literal(kv["building_image"], datatype=XSD.string)))
    g.add((building_uri, AT.hasConstructionPeriod, Literal(kv["construction_period"], datatype=XSD.string)))
    g.add((building_uri, LCA.hasClimateRegion, Literal(kv["climate_region"], datatype=XSD.string)))
    g.add((building_uri, AT.hasReferenceFloorArea, Literal(kv["reference_floor_area_m2"], datatype=XSD.float)))
    g.add((building_uri, AT.hasReferenceEnergySupply, Literal(kv["heat_supply_type"], datatype=XSD.string)))

    for suffix, label, value, renovation_class in (
        ("Status_Existing", "Existing State", kv.get("heating_demand_existing"), AT.NoRenovation),
        ("Status_Usual", "Usual Refurbishment", kv.get("heating_demand_usual"), AT.MinorRenovation),
        ("Status_Advanced", "Advanced Refurbishment", kv.get("heating_demand_advanced"), AT.MajorRenovation),
    ):
        if value is None:
            continue
        state_uri = URIRef(TABULA[f"Building_{safe_id}_{suffix}"])
        g.add((building_uri, AT.hasRenovationStatus, state_uri))
        g.add((state_uri, RDF.type, OWL.NamedIndividual))
        g.add((state_uri, RDF.type, renovation_class))
        g.add((state_uri, RDFS.label, Literal(f"{building_id} - {label}", datatype=XSD.string)))
        demand_node = BNode()
        g.add((state_uri, AT.hasReferenceEnergyDemand, demand_node))
        g.add((demand_node, AT.hasValue, Literal(float(value), datatype=XSD.float)))
        g.add((demand_node, QUDT.unit, UNIT["KiloW-HR-PER-M2"]))

    for element in kv["elements"]:
        element_label = element["label"]
        safe_element_label = element_label.replace(" ", "_")
        element_uri = URIRef(TABULA[f"Building_{safe_id}_Element_{safe_element_label}"])

        assign_element_types(g, element_uri, element_label)
        g.add((element_uri, RDFS.label, Literal(element_label, datatype=XSD.string)))
        g.add((building_uri, AT.hasElementArchetype, element_uri))

        area_node = BNode()
        g.add((element_uri, AT.hasReferenceArea, area_node))
        g.add((area_node, AT.hasValue, Literal(element["surface_area_m2"], datatype=XSD.float)))
        g.add((area_node, QUDT.unit, UNIT["M2"]))

        use_layers = not is_opening_element(element_label)
        existing_layerset = None
        existing_layer_uris = []

        stages = (
            {
                "suffix": "Existing",
                "label": "Existing State",
                "u_value": element["existing_u_value"],
                "desc_de": element["construction_type_de"],
                "desc_eng": element["construction_type_eng"],
                "renovation_class": AT.NoRenovation,
                "mode": "existing",
            },
            {
                "suffix": "Usual",
                "label": "Usual Refurbishment",
                "u_value": element["usual_u_value"],
                "desc_de": element["usual_measure_de"],
                "desc_eng": element["usual_measure"],
                "renovation_class": AT.MinorRenovation,
                "mode": "refurbishment",
            },
            {
                "suffix": "Advanced",
                "label": "Advanced Refurbishment",
                "u_value": element["advanced_u_value"],
                "desc_de": element["advanced_measure_de"],
                "desc_eng": element["advanced_measure"],
                "renovation_class": AT.MajorRenovation,
                "mode": "refurbishment",
            },
        )

        if use_layers:
            record = iter_tabula_stage_descriptions(
                building_id,
                safe_id,
                element_label,
                stages[0],
            )
            nlp_existing = None
            if ontology_corpus and record:
                nlp_existing = layerset_from_nlp(
                    stages[0]["desc_de"],
                    stages[0]["desc_eng"],
                    record.subject_uri,
                    ontology_corpus,
                    ontology_graph=ontology_graph,
                )
            existing_layerset = nlp_existing or build_existing_layerset(
                element["construction_type_de"],
                element["construction_type_eng"],
                element_label,
                ontology_graph=ontology_graph,
            )

        for stage in stages:
            if stage["u_value"] is None:
                continue

            state_suffix = f"{safe_element_label}_{stage['suffix']}"
            element_state_uri = URIRef(TABULA[f"Building_{safe_id}_Element_{state_suffix}"])

            g.add((element_uri, AT.hasRenovationStatus, element_state_uri))
            assign_element_types(g, element_state_uri, element_label)
            g.add(
                (
                    element_state_uri,
                    RDFS.label,
                    Literal(f"{building_id} - {element_label} ({stage['label']})", datatype=XSD.string),
                )
            )
            g.add((element_state_uri, AT.hasRenovationStatus, stage["renovation_class"]))

            u_value_node = BNode()
            g.add((element_state_uri, AT.hasUValue, u_value_node))
            g.add((u_value_node, AT.hasValue, Literal(stage["u_value"], datatype=XSD.float)))
            g.add((u_value_node, QUDT.unit, UNIT["W-PER-M2-K"]))

            if stage["desc_de"]:
                g.add((element_state_uri, AT.hasArchetypeDescription, Literal(stage["desc_de"], lang="de")))
            if stage["desc_eng"]:
                g.add((element_state_uri, AT.hasArchetypeDescription, Literal(stage["desc_eng"], lang="en")))

            if not use_layers:
                layer_fn = opening_layer_function(element_label)
                g.add((element_state_uri, BMP.hasLayerFunction, layer_fn))
                print(f"  {element_label} ({stage['suffix']}): {str(layer_fn).split('#')[-1]} (element-level)")
                continue

            if stage["mode"] == "existing":
                layerset = existing_layerset
            elif ontology_corpus and (stage["desc_de"] or stage["desc_eng"]):
                stage_record = iter_tabula_stage_descriptions(
                    building_id, safe_id, element_label, stage
                )
                layerset = build_refurbishment_layerset_with_nlp(
                    existing_layer_uris,
                    stage["desc_de"],
                    stage["desc_eng"],
                    element_label,
                    stage_record.subject_uri if stage_record else "",
                    ontology_corpus,
                    ontology_graph=ontology_graph,
                )
            else:
                layerset = build_refurbishment_layerset(
                    existing_layer_uris,
                    stage["desc_de"],
                    stage["desc_eng"],
                    element_label,
                    ontology_graph=ontology_graph,
                )

            if layerset and layerset.get("layers"):
                print(f"  {element_label} ({stage['suffix']}): {layer_summary_text(layerset['layers'])}")
                layer_uris = emit_layerset(
                    g, safe_id, state_suffix, element_state_uri, layerset, ontology_graph=ontology_graph
                )
                if stage["mode"] == "existing":
                    existing_layer_uris = layer_uris


def build_tabula_graph():
    ontology_g = load_ontology_graph(BASE_DIR / "owl")
    print("Loading shared ontology corpus for TABULA NLP...")
    ontology_corpus = load_ontology_corpus(ontology_g)
    begin_report("tabula")

    g = init_graph()
    for building_node in load_tabula_buildings():
        process_building(g, building_node, ontology_corpus=ontology_corpus, ontology_graph=ontology_g)
    return g


def main():
    print(f"Base directory: {BASE_DIR}")
    test_id = TEST_BUILDING_ID or os.environ.get("TABULA_TEST_BUILDING_ID")
    test_mode = bool(test_id)
    if test_mode:
        print(f"Test mode: single building {test_id}")
    else:
        print("Full dataset mode: all TABULA buildings")

    g = build_tabula_graph()
    out_dir = BASE_DIR / "ttl"
    out_dir.mkdir(parents=True, exist_ok=True)

    building_count = sum(1 for _ in g.subjects(RDF.type, AT.BuildingArchetype))
    element_count = sum(1 for _ in g.subjects(RDF.type, AT.ElementArchetype))
    layerset_count = sum(1 for _ in g.subjects(RDF.type, BMP.LayerSet))
    layer_count = sum(1 for _ in g.subjects(RDF.type, BMP.Layer))

    if test_mode:
        out_path = out_dir / f"tabula_{test_id.replace('.', '_')}_test.ttl"
    else:
        out_path = out_dir / "tabula_buildings-enriched.ttl"
        if out_path.is_file():
            out_path.unlink()

    g.serialize(destination=out_path, format="turtle")
    ontology_g = load_ontology_graph()
    finalize_report(data_graph=g, ontology_graph=ontology_g, ttl_path=out_path)

    print(f"\nSaved: {out_path}")
    print(
        f"Summary: {building_count} buildings | "
        f"{element_count} element archetypes | "
        f"{layerset_count} layer sets | "
        f"{layer_count} layers | "
        f"{len(g)} triples"
    )


if __name__ == "__main__":
    main()
