import json
import re
from collections import Counter
from pathlib import Path
from rdflib import Graph, Literal, Namespace, URIRef, XSD, BNode
from rdflib.namespace import RDF, RDFS
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import ollama
from pyvis.network import Network
import torch

# Model Setup & Hardware Acceleration Routing
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
print("Using device:", device)

LLM_MODEL = "llama3"  # Change this to "llama3.1", "mistral", etc. depending on local setup
BASE_DIR = Path(__file__).resolve().parent.parent
print(f"Base Directory: {BASE_DIR}")

# TRANSLATOR 
translator = GoogleTranslator(source='de', target='en')

def translate_safely(text):
    """Safely handles extraction/translation of strings with fallback."""
    if not text:
        return None
    if isinstance(text, dict):
        return None
    try:
        return translator.translate(str(text))
    except Exception as e:
        print(f"Translation warning: {e}. Falling back to original.")


########################################################
#### GET DATA FROM JSON FILE
########################################################
json_files = [BASE_DIR / "data" / "data_text_BBSR" / "json_outputs" / "page_boxes.json"]
all_data = []

for file_path in json_files:
    if not file_path.is_file():
        raise FileNotFoundError(f"OCR JSON not found:\n  {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        chunk = json.load(f)
        
    if isinstance(chunk, dict) and "pages" in chunk:
        records = chunk["pages"]
    elif isinstance(chunk, list):
        records = chunk
    else:
        raise ValueError(f"Expected a JSON array or object with 'pages', got {type(chunk).__name__}")
        
    all_data.extend(records)
    print(f"Loaded {len(records)} BBSR page record(s) from {file_path.relative_to(BASE_DIR)}")

print(f"Total BBSR referenced buildings: {len(all_data)}")


########################################################
#### GRAPH BUILDING CONFIGURATION & NAMESPACES
########################################################
ontology_g = Graph()
ontology_path = (BASE_DIR / "owl" / "KB-LCA-merged.ttl").resolve()
print(f"Ontology path: {ontology_path}")
ontology_g.parse(location=ontology_path.as_uri(), format="ttl")

# Namespaces & Graph Setup
BBSR = Namespace("https://namedgraphs.org/bbsr#")
AT = Namespace("https://w3id.org/at#")
LCA = Namespace("https://w3id.org/lca#")
BOT = Namespace("https://w3id.org/bot#")
BEO = Namespace("https://w3id.org/beo#")
BMP = Namespace("https://w3id.org/bmp#")
UNIT = Namespace("http://qudt.org/2.1/vocab/unit#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
DC_IDENTIFIER = URIRef("http://purl.org/dc/elements/1.1/identifier")
OM = Namespace("http://ontology.eil.utoronto.ca/icity/OM#")
QUDT = Namespace("http://qudt.org/schema/qudt/")

g = Graph()
g.bind("bbsr", BBSR)
g.bind("at", AT)
g.bind("lca", LCA)
g.bind("bot", BOT)
g.bind("beo", BEO)
g.bind("bmp", BMP)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)
g.bind("unit", UNIT)
g.bind("dc1", Namespace("http://purl.org/dc/elements/1.1/identifier"))
g.bind("xsd", XSD)
g.bind("om", OM)
g.bind("qudt", QUDT)

instance_counter = 1

############################################################
# STEP 1: RDF Graph creation (initial graph)
############################################################
for page_record in all_data:
    page_header = page_record.get("page_header", {})
    main_section = page_header.get("main_section") or {}
    main_title = main_section.get("title")
    subseries = page_header.get("subseries") or {}
    subseries_title = subseries.get("title")

    building_uri = BBSR[f"building{instance_counter:03d}"]

    g.add((building_uri, RDF.type, BOT.Building))
    g.add((building_uri, RDF.type, AT.BuildingArchetype))
    g.add((building_uri, AT.hasResidentialType, AT.ApartmentHouse))

    g.add((building_uri, AT.hasDescription, Literal(main_title, lang="de")))
    if main_title_en := translate_safely(main_title):
        g.add((building_uri, AT.hasDescription, Literal(main_title_en, lang="en")))
        
    g.add((building_uri, AT.hasDescription, Literal(subseries_title, lang="de")))
    if subseries_title_en := translate_safely(subseries_title):
        g.add((building_uri, AT.hasDescription, Literal(subseries_title_en, lang="en")))

    structured_sections = page_record.get("structured_sections", [])
    for section in structured_sections:
        properties = section.get("properties", {})
        slab = properties.get("Decke")
        floor_thickness = properties.get("Fußbodendicke")
        external_wall = properties.get("Außenwände")
        internal_wall = properties.get("Innenwände")
        partition_wall = properties.get("Trennwände")

        ### Slab Processing
        if slab is not None:
            slab_uri = BBSR[f"slab_{instance_counter:03d}"]
            g.add((building_uri, AT.hasElementArchetype, slab_uri))
            g.add((slab_uri, RDF.type, BEO.Slab))
            g.add((slab_uri, RDF.type, AT.ElementArchetype))

            if isinstance(slab, dict):
                slab_construction = slab.get("bauweise")
                slab_thickness = slab.get("dicke")
                if slab_construction is not None:
                    g.add((slab_uri, AT.hasDescription, Literal(slab_construction, lang="de")))
                    if slab_construction_en := translate_safely(slab_construction):
                        g.add((slab_uri, AT.hasDescription, Literal(slab_construction_en, lang="en")))
                if slab_thickness is not None:
                    g.add((slab_uri, BMP.hasThickness, Literal(slab_thickness)))
            elif isinstance(slab, str):
                g.add((slab_uri, AT.hasDescription, Literal(slab, lang="de")))
                if slab_en := translate_safely(slab):
                    g.add((slab_uri, AT.hasDescription, Literal(slab_en, lang="en")))
            
            fallback_thickness = properties.get("Deckendicke") or properties.get("Dicke")
            if fallback_thickness is not None:
                g.add((slab_uri, BMP.hasThickness, Literal(fallback_thickness)))
        
        ### Floor Processing
        if floor_thickness is not None:
            floor_uri = BBSR[f"floor_{instance_counter:03d}"]
            g.add((building_uri, AT.hasElementArchetype, floor_uri))
            g.add((floor_uri, RDF.type, BEO.Floor))
            g.add((floor_uri, BMP.hasThickness, Literal(floor_thickness)))

        ### External Wall Processing
        if external_wall is not None:
            external_wall_uri = BBSR[f"external_wall_{instance_counter:03d}"]
            g.add((building_uri, AT.hasElementArchetype, external_wall_uri))
            g.add((external_wall_uri, RDF.type, BEO.Wall))
            g.add((external_wall_uri, AT.hasDescription, Literal(external_wall, lang="de")))
            if ext_wall_en := translate_safely(external_wall):
                g.add((external_wall_uri, AT.hasDescription, Literal(ext_wall_en, lang="en")))

        ### Internal Wall Processing
        if internal_wall is not None:
            internal_wall_uri = BBSR[f"internal_wall_{instance_counter:03d}"]
            g.add((building_uri, AT.hasElementArchetype, internal_wall_uri))
            g.add((internal_wall_uri, RDF.type, BEO.Wall))
            g.add((internal_wall_uri, AT.hasDescription, Literal(internal_wall, lang="de")))
            if int_wall_en := translate_safely(internal_wall):
                g.add((internal_wall_uri, AT.hasDescription, Literal(int_wall_en, lang="en")))

        ### Partition Wall Processing
        if partition_wall is not None:
            partition_wall_uri = BBSR[f"partition_wall_{instance_counter:03d}"]
            g.add((building_uri, AT.hasElementArchetype, partition_wall_uri))
            g.add((partition_wall_uri, RDF.type, BEO.WallPARTITIONING))
            g.add((partition_wall_uri, AT.hasDescription, Literal(partition_wall, lang="de")))
            if part_wall_en := translate_safely(partition_wall):
                g.add((partition_wall_uri, AT.hasDescription, Literal(part_wall_en, lang="en")))

        # Dimensional properties
        building_type = properties.get("Gebäudetyp")
        builiding_length = properties.get("Gebäudelängen")
        builiding_width = properties.get("Gebäubreite")
        roof_shape = properties.get("Dachform und -art")
        level_number = properties.get("Geschoßanzahl")
        level_height = properties.get("Geschoßhöhe")

        if building_type is not None:
            g.add((building_uri, AT.hasBuildingType, Literal(building_type, lang="de")))
            if b_type_en := translate_safely(building_type):
                g.add((building_uri, AT.hasBuildingType, Literal(b_type_en, lang="en")))
                
            g.add((building_uri, OM.hasLength, Literal(builiding_length, lang="de")))
            g.add((building_uri, OM.hasWidth, Literal(builiding_width, lang="de")))
            g.add((building_uri, AT.hasRoofShape, Literal(roof_shape, lang="de")))
            if roof_en := translate_safely(roof_shape):
                g.add((building_uri, AT.hasRoofShape, Literal(roof_en, lang="en")))
                
            g.add((building_uri, BOT.hasStorey, Literal(level_number, lang="de")))            
            g.add((building_uri, OM.hasStoreyHeight, Literal(level_height, lang="de")))
   
    instance_counter += 1


############################################################
# STEP 2 & 3: RDF Graph embedding & Chunking
############################################################
def create_graph_embeddings(graph_instance):
    element_corpus = []
    allowed_types = [BEO.Wall, BEO.Slab, BEO.Floor, BEO.Roof, BEO.WallPARTITIONING]
    element_subjects = set()

    for allowed_type in allowed_types:
        for subject in graph_instance.subjects(RDF.type, allowed_type):
            element_subjects.add(subject)

    for s in element_subjects:
        de_descs = [str(d) for d in graph_instance.objects(s, AT.hasDescription) if getattr(d, 'language', None) == 'de']
        en_descs = [str(d) for d in graph_instance.objects(s, AT.hasDescription) if getattr(d, 'language', None) == 'en']
        thicknesses = [f"{thick}" for thick in graph_instance.objects(s, BMP.hasThickness)]

        combined_text = " | ".join(de_descs + en_descs + thicknesses)
        text_to_decompose = " | ".join(en_descs if en_descs else de_descs)
        if thicknesses:
            text_to_decompose += " | " + " | ".join(thicknesses)

        textual_chunks = textual_decomposition_llm(text_to_decompose)
        embedding_text = " | ".join(textual_chunks)

        element_corpus.append({
            "element_uri": str(s),
            "raw_description": combined_text,
            "textual_chunks": textual_chunks,
            "embedding_text": embedding_text,
            "embedding": None
        })

    for item in element_corpus:
        item["chunk_embeddings"] = []
        for chunk in item["textual_chunks"]:
            chunk_embedding = embedder_model.encode(chunk, convert_to_tensor=True)
            item["chunk_embeddings"].append({
                "chunk": chunk,
                "embedding": chunk_embedding
            })

    print(f"Created chunk embeddings for {len(element_corpus)} infrastructure elements.")
    return element_corpus


############################################################
# STEP 4: YOUR PREFERRED EXPLICIT LANGUAGE EMBEDDING METHOD
############################################################
def create_ontology_embeddings(ontology_graph):
    """
    Dynamically parses the OWL model graph to discover hierarchical relations
    (subClassOf, domain, range) to enable a scoped multi-tier retrieval engine.
    """
    ontology_corpus = []
    
    # 1. Dynamically extract broad parent categories (Top-Level Classes)
    parent_categories = set()
    for child, parent in ontology_graph.subject_objects(RDFS.subClassOf):
        if isinstance(parent, URIRef) and "http" in str(parent):
            parent_categories.add(parent)

    # 2. Extract every concept node to build the dynamic index maps
    all_entities = set(ontology_graph.subjects(RDF.type, OWL.Class)) | set(ontology_graph.subjects(RDF.type, RDF.Property))

    for entity in all_entities:
        entity_uri = str(entity)
        local_name = entity_uri.split("#")[-1].split("/")[-1]
        
        # Base token structure always starts with the localized class ID string
        text_parts = [local_name] 
        
        # --- STRICT LANGUAGE FILTERED LITERAL EXTRACTION ---
        # Only harvest literals that are implicitly untagged or explicitly marked as English (@en)
        for label in ontology_graph.objects(entity, RDFS.label):
            lang = getattr(label, 'language', None)
            if lang is None or lang == 'en':
                text_parts.append(str(label))
                
        for comment in ontology_graph.objects(entity, RDFS.comment):
            lang = getattr(comment, 'language', None)
            if lang is None or lang == 'en':
                text_parts.append(str(comment))
                
        dcterms_description = URIRef("http://purl.org/dc/terms/description")
        for desc in ontology_graph.objects(entity, dcterms_description):
            lang = getattr(desc, 'language', None)
            if lang is None or lang == 'en':
                text_parts.append(str(desc))
                
        # If no English textual annotations exist, fall back onto formatting the local name fragment split
        if len(text_parts) == 1:
            # e.g., converts CamelCase "LightWeightConcrete" to "Light Weight Concrete"
            spaced_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', local_name)
            text_parts.append(spaced_name)
            
        combined_text = " | ".join(text_parts)
        
        # Mine structural graph relationships dynamically
        superclasses = [str(parent) for parent in ontology_graph.objects(entity, RDFS.subClassOf)]
        subclasses = [str(child) for child in ontology_graph.subjects(RDFS.subClassOf, entity)]
        
        # Categorize node characteristics based purely on OWL topology
        is_category = entity in parent_categories
        is_material_type = any("Material" in str(s) or "Concrete" in str(s) or "Insulation" in str(s) for s in superclasses)

        ontology_corpus.append({
            "entity_uri": entity_uri,
            "text": combined_text,
            "superclasses": superclasses,
            "subclasses": subclasses,
            "is_category": is_category,
            "is_material_type": is_material_type,
            "embedding": None
        })

    # Encode texts using the cross-lingual sentence model
    texts = [item["text"] for item in ontology_corpus]
    embeddings = embedder_model.encode(texts, convert_to_tensor=True)

    for idx, emb in enumerate(embeddings):
        ontology_corpus[idx]["embedding"] = emb

    print(f"Dynamically generated graph-mined ontology embeddings for {len(ontology_corpus)} entities.")
    return ontology_corpus


############################################################
# STEP 5: TEXTUAL SEMANTIC DECOMPOSITION
############################################################
def textual_decomposition_llm(text):
    chunks = []
    text_lower = text.lower()

    # --- Structural Element Core Systems ---
    if "slab" in text_lower or "decke" in text_lower: 
        chunks.append("slab")
    if "hohlraumdecke" in text_lower or "cavity slab" in text_lower or "hollow-core" in text_lower:
        chunks.append("hollow-core slab")
    if "vollbetondecke" in text_lower or "solid slab" in text_lower:
        chunks.append("solid concrete slab")
        
    if "wall" in text_lower or "wand" in text_lower: 
        chunks.append("wall")
    if "floor" in text_lower or "fußboden" in text_lower: 
        chunks.append("floor")

    # --- Advanced Material & Concrete Classifications ---
    if "reinforced concrete" in text_lower or "stahlbeton" in text_lower: 
        chunks.append("reinforced concrete")
    if "spannbeton" in text_lower or "prestressed concrete" in text_lower:
        chunks.append("prestressed concrete")
    if "leichtbeton" in text_lower or "lightweight concrete" in text_lower:
        chunks.append("lightweight concrete")
    if "leichtzuschlagstoffbeton" in text_lower or "lzs-beton" in text_lower:
        chunks.append("lightweight aggregate concrete")
    if "porenbeton" in text_lower or "aerated concrete" in text_lower:
        chunks.append("aerated concrete")
    if "haufwerksporig" in text_lower or "no-fines concrete" in text_lower:
        chunks.append("no-fines porous concrete")
    if "normalbeton" in text_lower or "normal concrete" in text_lower: 
        chunks.append("normal concrete")
    elif "concrete" in text_lower or "beton" in text_lower: 
        chunks.append("concrete")

    # --- Insulation & Specialized Layer Materials ---
    if "mineral wool" in text_lower or "mineralwolle" in text_lower: 
        chunks.append("mineral wool")
    if "schaumpolystyrol" in text_lower or "polystyrene" in text_lower or "eps" in text_lower:
        chunks.append("expanded polystyrene insulation")
    if "dämmung" in text_lower or "insulation" in text_lower:
        chunks.append("thermal insulation")
        if "kerndämmung" in text_lower: chunks.append("core insulation")
        if "außendämmung" in text_lower: chunks.append("exterior insulation")
        if "innendämmung" in text_lower: chunks.append("interior insulation")
        
    if "plaster" in text_lower or "gips" in text_lower or "putz" in text_lower: 
        chunks.append("plaster coating")

    # --- Structural Layer Topology Configurations ---
    if "single layer" in text_lower or "einschichtig" in text_lower: 
        chunks.append("single layer")
    if "two layer" in text_lower or "zweischichtig" in text_lower: 
        chunks.append("two layer")
    if "multi layer" in text_lower or "dreischichtig" in text_lower or "3-layered" in text_lower: 
        chunks.append("multi layer")

    # --- Finishes and Processing Surface Status ---
    if "surface finished" in text_lower or "oberflächenfertig" in text_lower: 
        chunks.append("surface finished")
    if "geputzt" in text_lower or "finished plaster" in text_lower:
        chunks.append("plaster finished surface")

    # --- Physical Dimensions ---
    thickness_match = re.findall(r"\d+(?:,\d+)?\s*cm", text_lower)
    for thickness in thickness_match:
        chunks.append(f"{thickness} thickness")

    return list(dict.fromkeys(chunks))


##########################################################
# STEP 6: SCOPED HIERARCHICAL MATCH RETRIEVAL LOGIC
##########################################################
def retrieve_hierarchical_matches(chunk_embedding, ontology_corpus, top_k=6):
    """
    Executes a two-tiered taxonomic lookup matching parent categories first, 
    and then finding valid sub-ordinary classes mapped under that category.
    """
    category_candidates = []
    for item in ontology_corpus:
        if item["is_category"]:
            score = util.cos_sim(chunk_embedding, item["embedding"]).item()
            category_candidates.append((item, score))
    
    category_candidates.sort(key=lambda x: x[1], reverse=True)
    if not category_candidates:
        return []
        
    best_category, best_cat_score = category_candidates[0]
    matched_category_uri = best_category["entity_uri"]

    scoped_type_candidates = []
    for item in ontology_corpus:
        if matched_category_uri in item["superclasses"] or item["entity_uri"] == matched_category_uri:
            score = util.cos_sim(chunk_embedding, item["embedding"]).item()
            scoped_type_candidates.append({
                "entity_uri": item["entity_uri"],
                "text": item["text"],
                "score": score,
                "parent_category": matched_category_uri
            })

    scoped_type_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scoped_type_candidates[:top_k]


############################################################
# STEP 7: LLM reasoning
############################################################
def llm_reasoning_to_rdf(element_uri, german_desc, english_desc, ontology_matches):
    de_variants = [v.strip() for v in str(german_desc).split(";") if v.strip()]
    en_variants = [v.strip() for v in str(english_desc).split(";") if v.strip()]

    if len(de_variants) != len(en_variants):
        en_variants = de_variants

    ontology_context = ""
    for match in ontology_matches:
        ontology_context += f"- IRI: {match['entity_uri']} [Subordinary to: {match.get('parent_category', 'None')}] | Metadata: {match['text']}\n"

    system_prompt = """
You are a deterministic mapping agent for a Semantic Web engine. Map the architectural description fragments precisely to the provided scoped IRIs.
Always select the specific MaterialType subclass IRI for 'predicted_material_iri' and its broader parent class for 'predicted_function_iri'.

Output a single, raw JSON object and NOTHING else. Do not wrap in markdown blocks or output introductory text.

Expected JSON Structure:
{
  "layer_topology": "https://w3id.org/bmp#SingleLayer",
  "layer_set_description": "German structural variant label description short",
  "thickness_cm": 29.0,
  "layers": [
    {
      "layer_index": 1,
      "predicted_function_iri": "BROADER_PARENT_CATEGORY_IRI",
      "predicted_material_iri": "SPECIFIC_SUBORDINARY_TYPE_IRI"
    }
  ]
}
"""
    compiled_variants_results = []

    for idx, (de_var, en_var) in enumerate(zip(de_variants, en_variants), start=1):
        user_prompt = f"""
Analyze Element Configuration:
- Subject Element URI: <{element_uri}>
- German Source Label: "{de_var}"
- English Source Label: "{en_var}"

Valid Constraints Vocabulary (You must pull from these exact options):
{ontology_context}
"""
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.0}
            )
            raw_content = response["message"]["content"].strip()
            raw_content = re.sub(r"```json", "", raw_content, flags=re.IGNORECASE).strip()
            raw_content = re.sub(r"```", "", raw_content).strip()
            
            compiled_variants_results.append(json.loads(raw_content))
        except Exception as e:
            print(f"LLM processing failed for {element_uri} [Variant {idx}]: {e}")
            continue

    return compiled_variants_results if compiled_variants_results else None


##########################################################
# GRAPH GENERATION LOGIC TIER
##########################################################
def compile_json_to_graph(graph_instance, element_uri, variants_data):
    if not variants_data:
        return

    element_ref = URIRef(element_uri)
    instance_suffix = element_uri.split("_")[-1]
    existing_types = [str(t) for t in graph_instance.objects(element_ref, RDF.type)]
    
    archetype_class = AT.ElementArchetype
    prefix = "element"

    if "https://w3id.org/beo#WallPARTITIONING" in existing_types:
        archetype_class = AT.WallArchetype
        prefix = "part_wall"
    elif "https://w3id.org/beo#Wall" in existing_types:
        archetype_class = AT.WallArchetype
        prefix = "ex_wall" if "external_wall" in str(element_uri) else ("int_wall" if "internal_wall" in str(element_uri) else "wall")
    elif "https://w3id.org/beo#Slab" in existing_types:
        archetype_class = AT.SlabArchetype
        prefix = "slab"
    elif "https://w3id.org/beo#Floor" in existing_types:
        archetype_class = AT.FloorArchetype
        prefix = "floor"

    graph_instance.add((element_ref, RDF.type, archetype_class))

    for idx, json_data in enumerate(variants_data, start=1):
        layerset_uri = BBSR[f"{prefix}_layerset_{instance_suffix}_var{idx}"]
        graph_instance.add((element_ref, BMP.hasLayerSet, layerset_uri))
        graph_instance.add((layerset_uri, RDF.type, BMP.LayerSet))
        
        if "layer_topology" in json_data:
            graph_instance.add((layerset_uri, RDF.type, URIRef(json_data["layer_topology"])))
        if "layer_set_description" in json_data:
            graph_instance.add((layerset_uri, AT.hasDescription, Literal(json_data["layer_set_description"], lang="de")))

        if "thickness_cm" in json_data and json_data["thickness_cm"]:
            thick_node = BNode()
            graph_instance.add((layerset_uri, BMP.hasThickness, thick_node))
            graph_instance.add((thick_node, RDF.value, Literal(float(json_data["thickness_cm"]), datatype=XSD.float)))
            graph_instance.add((thick_node, QUDT.unit, UNIT.CentiM))

        for i, layer_data in enumerate(json_data.get("layers", []), start=1):
            layer_uri = BBSR[f"{prefix}_layer_{instance_suffix}_var{idx}_0{i}"]
            graph_instance.add((layerset_uri, BMP.hasLayer, layer_uri))
            graph_instance.add((layer_uri, RDF.type, BMP.Layer))
            
            if layer_data.get("predicted_function_iri"):
                graph_instance.add((layer_uri, BMP.hasLayerFunction, URIRef(layer_data["predicted_function_iri"])))
            if layer_data.get("predicted_material_iri"):
                graph_instance.add((layer_uri, BMP.hasMaterialCategory, URIRef(layer_data["predicted_material_iri"])))


##########################################################
# DETERMINISTIC PIPELINE EXECUTION ENGINE
##########################################################

# 1. Serialize the baseline initialization graph configuration
out_path_init = BASE_DIR / "ttl" / "bbsr_buildings-init.ttl"
out_path_init.parent.mkdir(parents=True, exist_ok=True)
if out_path_init.is_file():
    out_path_init.unlink()
g.serialize(destination=out_path_init, format="turtle")
print(f"Successfully generated clean initialization graph: {out_path_init.name}")

# 2. Extract targets using our graph-mined taxonomy structures
element_corpus = create_graph_embeddings(g)
ontology_corpus = create_ontology_embeddings(ontology_g)

print("\n==================================================")
print("RUNNING HIERARCHICAL COGNITIVE TAXONOMY LOOP")
print("==================================================")

for element in element_corpus:
    uri_str = element['element_uri']
    element_uri_ref = URIRef(uri_str)
    
    de_descriptions = [str(d) for d in g.objects(element_uri_ref, AT.hasDescription) if getattr(d, 'language', None) == 'de']
    en_descriptions = [str(d) for d in g.objects(element_uri_ref, AT.hasDescription) if getattr(d, 'language', None) == 'en']
    
    german_txt = de_descriptions[0] if de_descriptions else "No German description found"
    english_txt = en_descriptions[0] if en_descriptions else "No English translation found"

    print(f"Processing structural entity: {uri_str}")
    
    # Query our tiered lookup system for context assembly
    aggregated_matches = []
    for chunk_data in element["chunk_embeddings"]:
        matches = retrieve_hierarchical_matches(chunk_data["embedding"], ontology_corpus, top_k=4)
        aggregated_matches.extend(matches)
    
    seen_uris = set()
    unique_matches = []
    for m in sorted(aggregated_matches, key=lambda x: x["score"], reverse=True):
        if m["entity_uri"] not in seen_uris:
            seen_uris.add(m["entity_uri"])
            unique_matches.append(m)
            
    final_matches = unique_matches[:10]
    reasoned_json = llm_reasoning_to_rdf(uri_str, german_txt, english_txt, final_matches)
    
    if reasoned_json:
        compile_json_to_graph(g, uri_str, reasoned_json)

# 3. Save out the enriched structural master file once execution is complete
out_path_enriched = BASE_DIR / "ttl" / "bbsr_buildings-enriched.ttl"
if out_path_enriched.is_file():
    out_path_enriched.unlink()
g.serialize(destination=out_path_enriched, format="turtle")

print("==================================================")
print(f"COMPLETE: Saved completely enriched graph to {out_path_enriched.name}")
print("==================================================\n")


##########################################################
# INTERACTIVE GEOMETRIC NETWORK VISUALIZATION
##########################################################
OUTPUT_HTML = BASE_DIR / "ttl" / "graph_visualization.html"

net = Network(height="850px", width="100%", bgcolor="#222222", font_color="white", directed=True)
net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4)

COLOR_MAP = {
    "Building": "#FF5733", "Wall": "#33FF57", "Slab": "#3357FF", 
    "Floor": "#F3FF33", "LayerSet": "#9B59B6", "Layer": "#1ABC9C", 
    "Default_Class": "#E67E22", "Literal": "#BDC3C7"
}

def get_node_style(node, graph_instance):
    if not isinstance(node, str) and hasattr(node, 'datatype'):
        return str(node), COLOR_MAP["Literal"], 15
        
    node_str = str(node)
    label = node_str.split("#")[-1].split("/")[-1]
    node_types = [str(t) for t in graph_instance.objects(node, RDF.type)]
    
    if str(BOT.Building) in node_types:
        return f"🏢 {label}", COLOR_MAP["Building"], 35
    elif str(BEO.Wall) in node_types or str(BEO.WallPARTITIONING) in node_types:
        return f"🧱 {label}", COLOR_MAP["Wall"], 28
    elif str(BEO.Slab) in node_types:
        return f"🥞 {label}", COLOR_MAP["Slab"], 28
    elif str(BEO.Floor) in node_types:
        return f"📐 {label}", COLOR_MAP["Floor"], 28
    elif str(BMP.LayerSet) in node_types:
        return f"📦 {label}", COLOR_MAP["LayerSet"], 22
    elif str(BMP.Layer) in node_types:
        return f"🥞 {label}", COLOR_MAP["Layer"], 18
    elif "https://w3id.org/" in node_str or "http://" in node_str:
        return label, COLOR_MAP["Default_Class"], 20
        
    return label, "#7F8C8D", 15

# Populate layout straight using our updated clean memory reference
for s, p, o in g:
    s_label, s_color, s_size = get_node_style(s, g)
    net.add_node(str(s), label=s_label, color=s_color, size=s_size, title=str(s))
    
    o_label, o_color, o_size = get_node_style(o, g)
    net.add_node(str(o), label=o_color, color=o_color, size=o_size, title=str(o))
    
    edge_label = p.split("#")[-1].split("/")[-1]
    net.add_edge(str(s), str(o), label=edge_label, color="#95A5A6", weight=1)

net.write_html(str(OUTPUT_HTML))
print(f"Successfully generated dynamic HTML serialization layout visualization at:\n -> {OUTPUT_HTML.resolve()}")