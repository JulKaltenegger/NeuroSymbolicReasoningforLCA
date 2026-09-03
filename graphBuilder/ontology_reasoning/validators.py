"""Validate LLM JSON output against allowed ontology IRIs."""

from __future__ import annotations

import re

from .chunking import combined_description
from .material_axioms import (
    resolve_category_and_type,
    sanitize_layer_prediction,
)
from .layer_axioms import (
    BMP_AIR,
    BMP_FINISHING,
    BMP_LOAD_BEARING,
    BMP_MULTI_LAYER,
    default_layerset_topology,
    resolve_layerset_topology,
    sanitize_layer_function_iri,
)

BMP = "https://w3id.org/bmp#"
AIR = str(BMP_AIR)
FINISHING = str(BMP_FINISHING)
LOAD_BEARING = str(BMP_LOAD_BEARING)
BINDERS = f"{BMP}Binders"
GYPSUM = f"{BMP}GypsumPlaster"
CONCRETE = f"{BMP}Concrete"
LWC = f"{BMP}LightWeightConcrete"
BRICK = f"{BMP}Brick"
BMU = f"{BMP}BrickMasonryUnit"

# Type cues first so "Leichtbeton" does not collapse to bare Concrete.
# "hollow concrete" is intentionally not a type cue.
_TYPE_CUES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"leichtbeton|lightweight\s+concrete", re.I), CONCRETE, LWC),
    (re.compile(r"hohllochziegel|hollow[- ]hole\s+brick", re.I), BRICK, BMU),
    (re.compile(r"stahlbeton|reinforced\s+concrete", re.I), CONCRETE, f"{BMP}ReinforcedConcrete"),
    (re.compile(r"geputzt|\bputz\b|plastered|gypsum\s+plaster", re.I), BINDERS, GYPSUM),
]
_CATEGORY_CUES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bbricks?\b|\bziegel(?!element)", re.I), BRICK),
    (re.compile(r"\bconcrete\b|\bbeton\b", re.I), CONCRETE),
]
_FUNCTION_WORDS = {
    "Facade": ("facade", "fassade", "outer", "außen", "aussen"),
    "Air": ("air gap", "airgap", "luftschicht", "in-between"),
    "LoadBearing": ("load bearing", "load-bearing", "loadbearing", "tragend", "inner"),
    "Finishing": ("finishing", "putz", "geputzt", "plaster"),
    "Insulating": ("insulation", "dämm", "daemm"),
}
_ALT_SPLIT = re.compile(r"\s+oder\s+|\s+or\s+", re.I)

THICKNESS_CM_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*cm", re.IGNORECASE)
THICKNESS_MM_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm", re.IGNORECASE)


def allowed_iris_from_matches(matches) -> set[str]:
    return {match["entity_uri"] for match in matches}


def expand_allowed_iris(matches, ontology_graph=None, profile: str = "bbsr") -> set[str]:
    """Post-LLM whitelist: retrieval hits + OWL composition classes (all from graph)."""
    from .config import get_profile
    from .owl_schema import composition_allowed_iris

    allowed = allowed_iris_from_matches(matches)
    if ontology_graph is None:
        return allowed

    cfg = get_profile(profile)
    if cfg.get("llm_mode") == "material_only":
        return allowed

    return allowed | composition_allowed_iris(ontology_graph, profile)


def _clean_iri(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    value = re.sub(r"\[.*?\]\(.*?\)", "", value)
    value = value.strip("[]() ")
    if value.startswith("http"):
        return value
    return None


def extract_thickness_cm(text: str | None) -> float | None:
    if not text:
        return None
    cm_match = THICKNESS_CM_PATTERN.search(text)
    if cm_match:
        return float(cm_match.group(1).replace(",", "."))
    mm_match = THICKNESS_MM_PATTERN.search(text)
    if mm_match:
        return float(mm_match.group(1).replace(",", ".")) / 10.0
    return None


def validate_layer_json(
    layer_json: dict | None,
    allowed_iris: set[str] | None = None,
    ontology_graph=None,
) -> dict | None:
    if not layer_json or "layers" not in layer_json:
        return None

    topology = _clean_iri(layer_json.get("layer_topology"))
    if allowed_iris and topology and topology not in allowed_iris:
        topology = None

    if ontology_graph is not None:
        topo_resolved = resolve_layerset_topology(ontology_graph, topology)
        topology = (
            str(topo_resolved)
            if topo_resolved is not None
            else default_layerset_topology(ontology_graph)
        )
    else:
        topology = topology or "https://w3id.org/bmp#SingleLayer"

    cleaned = {
        "layer_topology": topology,
        "layer_set_description": layer_json.get("layer_set_description"),
        "thickness_cm": layer_json.get("thickness_cm"),
        "layers": [],
    }

    for layer in layer_json.get("layers", []):
        layer_out = {
            "layer_index": layer.get("layer_index", 1),
            "predicted_function_iri": _clean_iri(layer.get("predicted_function_iri")),
            "predicted_category_iri": _clean_iri(layer.get("predicted_category_iri")),
            "predicted_type_iri": _clean_iri(layer.get("predicted_type_iri")),
            # [UNUSED] no emitter reads predicted_material_iri; sanitize_layer_prediction
            # ignores it, so a model answering only here would lose its material silently
            # "predicted_material_iri": _clean_iri(layer.get("predicted_material_iri")),
            "thickness_cm": layer.get("thickness_cm"),
        }
        if allowed_iris:
            for key in (
                "predicted_function_iri",
                "predicted_category_iri",
                "predicted_type_iri",
                # [UNUSED] "predicted_material_iri",
            ):
                if layer_out[key] and layer_out[key] not in allowed_iris:
                    layer_out[key] = None

        if ontology_graph is not None:
            fn, cat, typ = sanitize_layer_prediction(
                ontology_graph,
                function_iri=layer_out["predicted_function_iri"],
                category_iri=layer_out["predicted_category_iri"],
                type_iri=layer_out["predicted_type_iri"],
            )
            layer_out["predicted_function_iri"] = fn
            layer_out["predicted_category_iri"] = cat
            layer_out["predicted_type_iri"] = typ
            if fn:
                layer_out["predicted_function_iri"] = sanitize_layer_function_iri(
                    ontology_graph, fn
                )
            cat_uri, typ_uri = resolve_category_and_type(
                ontology_graph,
                category_iri=cat,
                type_iri=typ,
            )
            if cat_uri is not None:
                layer_out["predicted_category_iri"] = str(cat_uri)
                layer_out["predicted_type_iri"] = str(typ_uri) if typ_uri is not None else None
            else:
                layer_out["predicted_category_iri"] = None
                layer_out["predicted_type_iri"] = None

        has_signal = layer_out.get("thickness_cm") is not None or any(
            layer_out.get(k) for k in layer_out if k.startswith("predicted_") and layer_out.get(k)
        )
        if has_signal:
            cleaned["layers"].append(layer_out)

    for idx, layer in enumerate(cleaned["layers"], start=1):
        layer["layer_index"] = idx

    return cleaned if cleaned["layers"] else None


def audit_graph(data_graph, ontology_graph) -> dict[str, list[tuple[str, str]]]:
    """Report emitted triples whose object violates the OWL range of its slot.

    Returns a mapping of issue kind -> [(subject, offending object), ...]. Used to verify
    that a generated TTL only carries proper LayerFunction subclasses in
    bmp:hasLayerFunction, category anchors in bmp:hasMaterialCategory, and types
    subordinate to their category in bmp:hasMaterialType.
    """
    from rdflib import URIRef
    from rdflib.namespace import RDF

    from .layer_axioms import BMP_HAS_LAYER_FUNCTION, BMP_LAYER, is_layer_function_iri
    from .material_axioms import (
        BMP_HAS_MATERIAL_CATEGORY,
        BMP_HAS_MATERIAL_TYPE,
        anchor_for_material_iri,
        material_category_uris,
        material_type_uris,
    )

    anchors = material_category_uris(ontology_graph)
    types = material_type_uris(ontology_graph)
    issues: dict[str, list[tuple[str, str]]] = {
        "invalid_layer_function": [],
        "invalid_material_category": [],
        "invalid_material_type": [],
        "orphan_material_type": [],
        "material_on_layer": [],
    }

    for subject, obj in data_graph.subject_objects(BMP_HAS_LAYER_FUNCTION):
        if not is_layer_function_iri(ontology_graph, obj):
            issues["invalid_layer_function"].append((str(subject), str(obj)))

    for subject, obj in data_graph.subject_objects(BMP_HAS_MATERIAL_CATEGORY):
        if str(obj) not in anchors:
            issues["invalid_material_category"].append((str(subject), str(obj)))
        if (subject, RDF.type, URIRef(str(BMP_LAYER))) in data_graph:
            issues["material_on_layer"].append((str(subject), str(obj)))

    # An IRI in the hasMaterialType range whose ancestry reaches no category anchor is
    # unusable: resolve_category_and_type() discards the pair and no material is emitted.
    for subject, obj in data_graph.subject_objects(BMP_HAS_MATERIAL_TYPE):
        if anchor_for_material_iri(ontology_graph, obj, anchors) is None:
            kind = "orphan_material_type" if str(obj) in types else "invalid_material_type"
            issues[kind].append((str(subject), str(obj)))

    return {kind: found for kind, found in issues.items() if found}


def normalize_composition_payload(raw) -> tuple[float | None, list[dict]]:
    if not raw or not isinstance(raw, dict):
        if isinstance(raw, list):
            return None, [item for item in raw if isinstance(item, dict)]
        return None, []

    element_thickness = raw.get("element_thickness_cm")
    if element_thickness is not None:
        try:
            element_thickness = float(element_thickness)
        except (TypeError, ValueError):
            element_thickness = None

    if "layer_sets" in raw and isinstance(raw["layer_sets"], list):
        return element_thickness, [item for item in raw["layer_sets"] if isinstance(item, dict)]
    if "layers" in raw:
        return element_thickness, [raw]
    return element_thickness, []


def validate_composition(raw, allowed_iris=None, ontology_graph=None):
    element_thickness, layer_sets = normalize_composition_payload(raw)
    validated = []
    for layer_set in layer_sets:
        cleaned = validate_layer_json(layer_set, allowed_iris, ontology_graph)
        if cleaned:
            validated.append(cleaned)
    return element_thickness, validated


def _short(iri: str | None) -> str:
    if not iri:
        return ""
    return iri.rsplit("#", 1)[-1]


def _mentions(text: str) -> list[dict]:
    found: list[dict] = []
    used: list[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        return any(not (span[1] <= start or span[0] >= end) for start, end in used)

    for pattern, cat, typ in _TYPE_CUES:
        for match in pattern.finditer(text):
            span = match.span()
            if overlaps(span):
                continue
            used.append(span)
            found.append({"category": cat, "type": typ, "span": span, "text": match.group(0)})
    for pattern, cat in _CATEGORY_CUES:
        for match in pattern.finditer(text):
            span = match.span()
            if overlaps(span):
                continue
            used.append(span)
            found.append({"category": cat, "type": None, "span": span, "text": match.group(0)})
    found.sort(key=lambda item: item["span"][0])
    return found


def _unique_mentions(mentions: list[dict]) -> list[dict]:
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[dict] = []
    for item in mentions:
        key = (item["category"], item["type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _assign_mention(layer: dict, mention: dict, ontology_graph) -> None:
    cat, typ = resolve_category_and_type(
        ontology_graph,
        category_iri=mention["category"],
        type_iri=mention["type"],
    )
    if cat is None:
        return
    layer["predicted_category_iri"] = str(cat)
    layer["predicted_type_iri"] = str(typ) if typ is not None else None


def _clause_matches_layer(clause: str, layer: dict) -> bool:
    fn = _short(layer.get("predicted_function_iri"))
    clause_l = clause.lower()
    if any(word in clause_l for word in _FUNCTION_WORDS.get(fn, ())):
        return True
    thick = layer.get("thickness_cm")
    if thick is None:
        return False
    value = int(thick) if float(thick).is_integer() else thick
    return bool(re.search(rf"\b{re.escape(str(value))}\s*cm\b", clause_l))


def _fillable_layers(layers: list[dict]) -> list[dict]:
    return [layer for layer in layers if layer.get("predicted_function_iri") != AIR]


def _strip_air_materials(layers: list[dict]) -> None:
    for layer in layers:
        if layer.get("predicted_function_iri") == AIR:
            layer["predicted_category_iri"] = None
            layer["predicted_type_iri"] = None


def _assign_materials_to_layers(layer_set: dict, text: str, mentions: list[dict], ontology_graph) -> None:
    masonry = [item for item in mentions if item["category"] != BINDERS]
    if not masonry:
        return
    clauses = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    layers = _fillable_layers(layer_set.get("layers") or [])
    used: set[int] = set()
    for layer in layers:
        if layer.get("predicted_category_iri"):
            continue
        for idx, mention in enumerate(masonry):
            if idx in used:
                continue
            local = text[max(0, mention["span"][0] - 80) : mention["span"][1] + 80]
            in_clause = any(
                mention["text"].lower() in clause.lower() and _clause_matches_layer(clause, layer)
                for clause in clauses
            )
            if in_clause or _clause_matches_layer(local, layer):
                _assign_mention(layer, mention, ontology_graph)
                used.add(idx)
                break
    unset = [layer for layer in layers if not layer.get("predicted_category_iri")]
    leftover = [mention for idx, mention in enumerate(masonry) if idx not in used]
    if len(unset) == 1 and leftover:
        _assign_mention(unset[0], leftover[0], ontology_graph)
    elif len(unset) == 1 and len(masonry) == 1:
        _assign_mention(unset[0], masonry[0], ontology_graph)


def _finishing_layer(index: int, mention: dict, ontology_graph) -> dict:
    layer = {
        "layer_index": index,
        "predicted_function_iri": FINISHING,
        "predicted_category_iri": None,
        "predicted_type_iri": None,
        "thickness_cm": None,
    }
    _assign_mention(layer, mention, ontology_graph)
    return layer


def _ensure_plaster_finishes(layer_set: dict, plaster: dict, ontology_graph) -> None:
    layers = list(layer_set.get("layers") or [])
    if any(layer.get("predicted_function_iri") == FINISHING for layer in layers):
        return
    core = [layer for layer in layers if layer.get("predicted_function_iri") != AIR]
    if not core:
        return
    layer_set["layers"] = [
        _finishing_layer(1, plaster, ontology_graph),
        *core,
        _finishing_layer(len(core) + 2, plaster, ontology_graph),
    ]
    for idx, layer in enumerate(layer_set["layers"], start=1):
        layer["layer_index"] = idx
    layer_set["layer_topology"] = str(BMP_MULTI_LAYER)


def _leaf_template(layer_set: dict) -> list[dict]:
    layers = [dict(layer) for layer in (layer_set.get("layers") or [])]
    load_bearing_count = sum(
        1 for item in layers if item.get("predicted_function_iri") == LOAD_BEARING
    )
    structural = []
    for layer in layers:
        fn = layer.get("predicted_function_iri")
        if fn in {None, FINISHING}:
            continue
        empty_extra_leaf = (
            fn == LOAD_BEARING
            and not layer.get("predicted_category_iri")
            and layer.get("thickness_cm") is None
            and load_bearing_count > 1
        )
        if empty_extra_leaf:
            continue
        structural.append(layer)
    if structural:
        return structural
    if layers:
        return [layers[0]]
    return [
        {
            "layer_index": 1,
            "predicted_function_iri": LOAD_BEARING,
            "predicted_category_iri": None,
            "predicted_type_iri": None,
            "thickness_cm": layer_set.get("thickness_cm"),
        }
    ]


def _ensure_alternative_sets(
    layer_sets: list[dict],
    masonry_alts: list[dict],
    plaster: dict | None,
    ontology_graph,
) -> list[dict]:
    template = layer_sets[0]
    rebuilt = []
    for alt in masonry_alts[:2]:
        core = [dict(layer) for layer in _leaf_template(template)]
        leaf = next(
            (layer for layer in core if layer.get("predicted_function_iri") == LOAD_BEARING),
            core[0],
        )
        if leaf.get("thickness_cm") is None and template.get("thickness_cm") is not None:
            leaf["thickness_cm"] = template["thickness_cm"]
        _assign_mention(leaf, alt, ontology_graph)
        new_set = {
            "layer_topology": template.get("layer_topology"),
            "layer_set_description": None,
            "thickness_cm": template.get("thickness_cm"),
            "layers": core,
        }
        if plaster:
            _ensure_plaster_finishes(new_set, plaster, ontology_graph)
        rebuilt.append(new_set)
    return rebuilt


def _drop_unattested_types(layer_sets: list[dict], mentions: list[dict]) -> None:
    attested = {item["type"] for item in mentions if item.get("type")}
    for layer_set in layer_sets:
        for layer in layer_set.get("layers") or []:
            typ = layer.get("predicted_type_iri")
            if typ and typ not in attested:
                layer["predicted_type_iri"] = None


def apply_named_material_rules(
    layer_sets: list[dict],
    *,
    german_desc: str | None,
    english_desc: str | None,
    ontology_graph,
) -> list[dict]:
    """Fill named materials and split 'oder'/'or' after LLM JSON is structurally valid.

    Cues must appear in the source text. Category words stay category-only.
    German wins when DE and EN conflict (geputzt ≠ cleaned).
    """
    if not layer_sets or ontology_graph is None:
        return layer_sets
    text = combined_description(german_desc, english_desc)
    if not text:
        return layer_sets

    mentions = _unique_mentions(_mentions(text))
    plaster = next((item for item in mentions if item["category"] == BINDERS), None)
    masonry = [item for item in mentions if item["category"] != BINDERS]
    if _ALT_SPLIT.search(text) and len(masonry) >= 2:
        layer_sets = _ensure_alternative_sets(layer_sets, masonry, plaster, ontology_graph)
    else:
        for layer_set in layer_sets:
            _strip_air_materials(layer_set.get("layers") or [])
            _assign_materials_to_layers(layer_set, text, mentions, ontology_graph)
            if plaster:
                _ensure_plaster_finishes(layer_set, plaster, ontology_graph)

    for layer_set in layer_sets:
        _strip_air_materials(layer_set.get("layers") or [])
    _drop_unattested_types(layer_sets, mentions)
    return [item for item in layer_sets if item.get("layers")]
