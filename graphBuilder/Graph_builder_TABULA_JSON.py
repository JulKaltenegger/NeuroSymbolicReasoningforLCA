import json
import re
from collections import Counter
from pathlib import Path
from rdflib import Graph, Literal, Namespace, URIRef, XSD, BNode
from rdflib.namespace import RDF, RDFS
from sentence_transformers import SentenceTransformer, util
import torch

# Model Setup & Hardware Acceleration Routing
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent.parent
print(f"Base Directory: {BASE_DIR}")


########################################################
#### GRAPH BUILDING CONFIGURATION & NAMESPACES
########################################################
ontology_g = Graph()
ontology_path = (BASE_DIR / "owl" / "KB-LCA-merged.ttl").resolve()
print(f"Ontology path: {ontology_path}")
ontology_g.parse(location=ontology_path.as_uri(), format="ttl")

# LOAD TABULA DATA
data_ab = json.load(open(BASE_DIR / "data" / "data-json-TABULA" / "apartment" / "ab_buildings_combined.json", "r", encoding="utf-8"))
data_mfh = json.load(open(BASE_DIR / "data" / "data-json-TABULA" / "multifamilyhouse" / "mfh_buildings_combined.json", "r", encoding="utf-8"))

# Get data from JSON
def extract_building_kv_pairs(building_node):
    b_info = building_node["building"]
    
    # 1. Standardize Core Building Info
    kv_pairs = {
        "building_id": b_info["id"],
        "building_json": b_info["building_json"],
        "building_image": b_info["building_image"],
        "size_class": b_info["size_class"],
        "construction_period": b_info["construction_period"],
        "reference_floor_area_m2": float(b_info["reference_floor_area_m2"]),
        "heat_supply_type": b_info["heat_supply_system"]["type"],
        "climate_region": b_info["climate_region"],
    }
    
    # 2. Handle Energy Demand Key Discrepancies & Extract Unit Upfront
    if "energy_need_for_heating" in building_node:
        energy_data = building_node["energy_need_for_heating"]
        kv_pairs["heating_demand_unit"] = energy_data.get("unit", "kWh/(m2a)")
        kv_pairs["heating_demand_existing"] = energy_data["existing_state"]
        kv_pairs["heating_demand_usual"] = energy_data["usual_refurbishment"]
        kv_pairs["heating_demand_advanced"] = energy_data["advanced_refurbishment"]["total"] if isinstance(energy_data["advanced_refurbishment"], dict) else energy_data["advanced_refurbishment"]
    elif "energy" in building_node and "heating_demand" in building_node["energy"]:
        energy_data = building_node["energy"]["heating_demand"]["states"]
        kv_pairs["heating_demand_unit"] = building_node["energy"]["heating_demand"].get("unit", "kWh/(m2a)")
        kv_pairs["heating_demand_existing"] = energy_data["existing"]
        kv_pairs["heating_demand_usual"] = energy_data["usual_refurbishment"]
        kv_pairs["heating_demand_advanced"] = energy_data["advanced_refurbishment"]["total"] if isinstance(energy_data["advanced_refurbishment"], dict) else energy_data["advanced_refurbishment"]
    else:
        kv_pairs["heating_demand_unit"] = "kWh/(m2a)"

    # 3. Standardize and Flatten Elements
    kv_pairs["elements"] = []
    for key, data in building_node["elements"].items():
        element_label = data.get("name", key).title() 
        
        element_kv = {
            "label": element_label,
            "surface_area_m2": float(data["surface_area_m2"]),
            "construction_type_eng": data.get("type_of_construction_eng") or data.get("type_of_construction", ""),
            "construction_type_de": data.get("type_of_construction_de") or data.get("type_of_construction", ""),
            "states": data["states"],
                        
            # Existing State
            "existing_u_value": float(data["states"]["existing"]["u_value_W_m2K"]),
            
            # Usual Refurbishment State
            "usual_u_value": float(data["states"]["usual_refurbishment"]["u_value_W_m2K"]),
            "usual_measure": data["states"]["usual_refurbishment"].get("measure_eng") or data["states"]["usual_refurbishment"].get("measure", ""),
            "usual_measure_de": data["states"]["usual_refurbishment"].get("measure_de") or data["states"]["usual_refurbishment"].get("measure", ""),
           
            # Advanced Refurbishment State
            "advanced_u_value": float(data["states"]["advanced_refurbishment"]["u_value_W_m2K"]),
            "advanced_measure": data["states"]["advanced_refurbishment"].get("measure_eng") or data["states"]["advanced_refurbishment"].get("measure", ""),
            "advanced_measure_de": data["states"]["advanced_refurbishment"].get("measure_de") or data["states"]["advanced_refurbishment"].get("measure", "")
        }
        kv_pairs["elements"].append(element_kv)
        
    return kv_pairs


# Namespaces & Graph Setup
TABULA = Namespace("https://namedgraphs.org/tabula#")
AT = Namespace("https://w3id.org/at#")
LCA = Namespace("https://w3id.org/lca#")
BOT = Namespace("https://w3id.org/bot#")
BEO = Namespace("https://w3id.org/beo#")
BMP = Namespace("https://w3id.org/bmp#")
UNIT = Namespace("http://qudt.org/2.1/vocab/unit#")  # <-- Official W3C QUDT Unit Namespace
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
DC_IDENTIFIER = URIRef("http://purl.org/dc/elements/1.1/identifier")
OM = Namespace("http://ontology.eil.utoronto.ca/icity/OM#")
QUDT = Namespace("http://qudt.org/schema/qudt/")

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
g.bind("unit", UNIT)  # Bound to clean prefix
g.bind("dc1", Namespace("http://purl.org/dc/elements/1.1/identifier"))
g.bind("xsd", XSD)
g.bind("om", OM)
g.bind("qudt", QUDT)

# Schema taxonomy definitions
g.add((AT.RenovationStatus, RDF.type, OWL.Class))
g.add((AT.NoRenovation, RDF.type, OWL.Class))
g.add((AT.MinorRenovation, RDF.type, OWL.Class))
g.add((AT.MajorRenovation, RDF.type, OWL.Class))
g.add((AT.NoRenovation, RDFS.subClassOf, AT.RenovationStatus))
g.add((AT.MinorRenovation, RDFS.subClassOf, AT.RenovationStatus))
g.add((AT.MajorRenovation, RDFS.subClassOf, AT.RenovationStatus))


# Main Graph Processing Loop
for building_node in data_ab["buildings"] + data_mfh["buildings"]:

    kv = extract_building_kv_pairs(building_node)    
    safe_id = kv["building_id"].replace(".", "_")
    building_uri = URIRef(TABULA[f"Building_{safe_id}"])
    
    g.add((building_uri, RDF.type, OWL.NamedIndividual))
    g.add((building_uri, RDF.type, BOT.Building))
    g.add((building_uri, RDF.type, AT.BuildingArchetype))

    # Conditional Residential Type Assignment
    if kv["size_class"] == "AB":
        g.add((building_uri, AT.hasResidentialType, AT.ApartmentHouse))
    elif kv["size_class"] == "MFH":
        g.add((building_uri, AT.hasResidentialType, AT.MultiFamilyHouse))

    g.add((building_uri, RDFS.label, Literal(kv["building_id"], datatype=XSD.string)))
    g.add((building_uri, DC_IDENTIFIER, Literal(kv["building_id"], datatype=XSD.string)))
    
    # Links to raw source code assets and tracking logs for structural context mapping
    g.add((building_uri, AT.hasSourceJSON, Literal(kv["building_json"], datatype=XSD.string)))
    
    if kv["building_image"]:
        g.add((building_uri, AT.hasImage, Literal(kv["building_image"], datatype=XSD.string)))
        
    g.add((building_uri, AT.hasConstructionPeriod, Literal(kv["construction_period"], datatype=XSD.string)))
    g.add((building_uri, LCA.hasClimateRegion, Literal(kv["climate_region"], datatype=XSD.string)))
    g.add((building_uri, AT.hasReferenceFloorArea, Literal(kv["reference_floor_area_m2"], datatype=XSD.float)))
    g.add((building_uri, AT.hasReferenceEnergySupply, Literal(kv["heat_supply_type"], datatype=XSD.string)))


    # Renovation Status Process Layer (Macro Building Metrics)
    state_mappings = [
        {"suffix": "Status_Existing", "label": "Existing State", "key": "heating_demand_existing", "subclass": AT.NoRenovation},
        {"suffix": "Status_Usual", "label": "Usual Refurbishment", "key": "heating_demand_usual", "subclass": AT.MinorRenovation},
        {"suffix": "Status_Advanced", "label": "Advanced Refurbishment", "key": "heating_demand_advanced", "subclass": AT.MajorRenovation}
    ]

    for mapping in state_mappings:
        if mapping["key"] in kv and kv[mapping["key"]] is not None:
            state_uri = URIRef(TABULA[f"Building_{safe_id}_{mapping['suffix']}"])
            demand_value = float(kv[mapping["key"]])
            
            # Connect Building and renovation status individual
            g.add((building_uri, AT.hasRenovationStatus, state_uri))
            g.add((state_uri, RDF.type, OWL.NamedIndividual))
            g.add((state_uri, RDF.type, mapping["subclass"]))
            g.add((state_uri, RDFS.label, Literal(f"{kv['building_id']} - {mapping['label']}", datatype=XSD.string)))
            
            # Create a Blank Node for the structured dimension
            quantity_bnode = BNode()
            g.add((state_uri, AT.hasReferenceEnergyDemand, quantity_bnode))
            g.add((quantity_bnode, AT.hasValue, Literal(demand_value, datatype=XSD.float)))
            
            # CORRECTED: True semantic QUDT reference for Kilowatt Hours Per Square Meter Year (Anno)
            g.add((quantity_bnode, QUDT.unit, UNIT["KiloW-HR-PER-M2"]))


    # Building Elements Layer
    for element in kv["elements"]:
        element_label_raw = element["label"]
        safe_element_label = element_label_raw.replace(" ", "_")
        element_uri = URIRef(TABULA[f"Building_{safe_id}_Element_{safe_element_label}"])
        
        g.add((element_uri, RDF.type, OWL.NamedIndividual))
        g.add((element_uri, RDF.type, BEO.BuildingElement))
        g.add((element_uri, RDF.type, AT.ElementArchetype))
        g.add((element_uri, RDFS.label, Literal(element_label_raw, datatype=XSD.string)))
        
        # Conditional Specific Class Assignment Layer
        label_lower = element_label_raw.lower()
        if "roof" in label_lower:
            g.add((element_uri, RDF.type, BEO.Roof))
        elif "wall" in label_lower:
            g.add((element_uri, RDF.type, BEO.Wall))
        elif "floor" in label_lower:
            g.add((element_uri, RDF.type, BEO.Slab))
        elif "door" in label_lower:
            g.add((element_uri, RDF.type, BEO.Door))
        elif "window" in label_lower:
            g.add((element_uri, RDF.type, BEO.Window))

        g.add((building_uri, TABULA.hasElement, element_uri))
        
        # Surface Area Blank Node
        area_bnode = BNode()
        g.add((element_uri, AT.hasSurfaceArea, area_bnode))
        g.add((area_bnode, AT.hasValue, Literal(element["surface_area_m2"], datatype=XSD.float)))
        g.add((area_bnode, QUDT.unit, UNIT["M2"]))  # Uniform QUDT Namespace URI Object

        element_stages = [
            {"suffix": "Existing", "label": "Existing State", "u_value": element["existing_u_value"], "desc_eng": element["construction_type_eng"], "desc_de": element["construction_type_de"], "subclass": AT.NoRenovation},
            {"suffix": "Usual", "label": "Usual Refurbishment", "u_value": element["usual_u_value"], "desc_eng": element["usual_measure"], "desc_de": element["usual_measure_de"], "subclass": AT.MinorRenovation},
            {"suffix": "Advanced", "label": "Advanced Refurbishment", "u_value": element["advanced_u_value"], "desc_eng": element["advanced_measure"], "desc_de": element["advanced_measure_de"], "subclass": AT.MajorRenovation}
        ]

        for stage in element_stages:
            if stage["u_value"] is None:
                continue
                
            state_suffix = f"{safe_element_label}_{stage['suffix']}"
            element_state_uri = URIRef(TABULA[f"Building_{safe_id}_Element_{state_suffix}"])
            
            g.add((element_uri, AT.hasRenovationStatus, element_state_uri))
            g.add((element_state_uri, RDF.type, OWL.NamedIndividual))
            g.add((element_state_uri, RDF.type, stage["subclass"]))
            g.add((element_state_uri, RDFS.label, Literal(f"{kv['building_id']} - {element_label_raw} ({stage['label']})", datatype=XSD.string)))
            
            if stage["desc_eng"]:
                g.add((element_state_uri, AT.hasArchetypeDescription, Literal(stage["desc_eng"], datatype=XSD.string)))
            if stage["desc_de"]:
                g.add((element_state_uri, AT.hasArchetypeDescription, Literal(stage["desc_de"], datatype=XSD.string)))
                
            # U-Value Blank Node
            u_value_bnode = BNode()
            g.add((element_state_uri, AT.hasUValue, u_value_bnode))
            g.add((u_value_bnode, AT.hasValue, Literal(stage["u_value"], datatype=XSD.float)))
            g.add((u_value_bnode, QUDT.unit, UNIT["W-PER-M2-K"]))  # Uniform QUDT Namespace URI Object

print(f"Successfully generated building-tier statements. Total triples: {len(g)}")

ttl_dir = BASE_DIR / "ttl"
out_path_enriched = ttl_dir / "tabula_buildings-enriched.ttl"

if out_path_enriched.is_file():
    out_path_enriched.unlink()

g.serialize(destination=str(out_path_enriched), format="turtle")
print(f"Graph successfully saved to: {out_path_enriched}")