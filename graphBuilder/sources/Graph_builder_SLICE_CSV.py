"""SLiCE CSV → enriched TTL. Pipeline step: python graphBuilder/run_pipeline.py --slice"""
import json
import os
import re
import sys
from pathlib import Path

try:
    import ollama
except ImportError:
    ollama = None

import pandas as pd
from rdflib import BNode, Graph, Literal, Namespace, URIRef, XSD
from rdflib.namespace import OWL, RDF, RDFS

try:
    import torch
    from sentence_transformers import SentenceTransformer, util

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    torch = None
    util = None
    embedder_model = None
    device = "cpu"
    EMBEDDINGS_AVAILABLE = False

LLM_MODEL = "llama3"

BASE_DIR = Path(__file__).resolve().parents[2]
_GRAPH_BUILDER_DIR = Path(__file__).resolve().parents[1]
if str(_GRAPH_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_BUILDER_DIR))

from ontology_reasoning import load_ontology_corpus, load_ontology_graph, process_description
from ontology_reasoning.material_axioms import ensure_layer_material_pair
from ontology_reasoning.rdf_layers import emit_enforced_layer, material_pair_from_nlp
from ontology_reasoning.validation_report import begin_report, finalize_report, record_ctx

# Smoke test: one element name from the CSV; None = full dataset.
TEST_ELEMENT_NAME = None

########################################################
# NAMESPACES & GRAPH INITIALIZATION
########################################################
LCA = Namespace("https://w3id.org/lca#")
BOT = Namespace("https://w3id.org/bot#")
BEO = Namespace("https://w3id.org/beo#")
BMP = Namespace("https://w3id.org/bmp#")
AT = Namespace("https://w3id.org/at#")
BPO = Namespace("https://w3id.org/bpo#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
SLICE = Namespace("https://w3id.org/slice#")

g = Graph()
g.bind("lca", LCA)
g.bind("bot", BOT)
g.bind("beo", BEO)
g.bind("bmp", BMP)
g.bind("at", AT)
g.bind("bpo", BPO)
g.bind("qudt", QUDT)
g.bind("unit", UNIT)
g.bind("slice", SLICE)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)
g.bind("xsd", XSD)

INDICATOR_MAP = {
    "indicator_GWP": LCA.hasGWP,
    "indicator_PM": LCA.hasPM,
    "indicator_EP": LCA.hasEP,
    "indicator_aci": LCA.hasAP,
    "indicator_pof": LCA.hasPOCP,
    "indicator_HTc": LCA.hasHT_c,
    "indicator_HTnc": LCA.hasHT_nc,
    "indicator_irhh": LCA.hasIR,
    "indicator_ecofw": LCA.hasET_fw,
    "indicator_ws": LCA.hasWU,
    "indicator_luosom": LCA.hasLU_osom,
    "indicator_luobio": LCA.hasLU_bio,
    "indicator_luourb": LCA.hasLU_urb,
    "indicator_luoagr": LCA.hasLU_agr,
    "indicator_luofor": LCA.hasLU_for,
    "indicator_lutsom": LCA.hasLU_tsom,
    "indicator_lutbio": LCA.hasLU_tbio,
    "indicator_luturb": LCA.hasLU_turb,
    "indicator_lutagr": LCA.hasLU_tagr,
}

ACTIVITY_TYPE_MAP = {
    "Material in": LCA.MaterialIn,
    "Transport to site": LCA.TransportToSite,
    "Transport EOL": LCA.TransportEoL,
    "Process": LCA.Process,
    "Material loss": LCA.MaterialLoss,
    "Material out": LCA.MaterialOut,
}

UNIT_MAP = {
    "kg": UNIT.KiloGM,
    "kg ": UNIT.KiloGM,
    "tkm": UNIT["TONNE-KM"],
    "m3": UNIT.M3,
    "m2": UNIT.M2,
    "hr": UNIT.HR,
}

LCS_SUFFIX_MAP = {
    "A1-3": "LCSA1A2A3",
    "A4": "LCSA4",
    "A5": "LCSA5",
    "B2.1": "LCSB2_1",
    "B2.2": "LCSB2_2",
    "B2.3": "LCSB2_3",
    "B4": "LCSB4",
    "C1": "LCSC1",
    "C2": "LCSC2",
    "C3": "LCSC3",
    "C4": "LCSC4",
}

THICKNESS_PATTERN = re.compile(
    r"\([^)]*?(\d+(?:\s*-\s*\d+)?)\s*(mm|cm)\)",
    re.IGNORECASE,
)

# SLiCE worksection_name is pipe-delimited; segment 1 is the functional category.
SLICE_SEGMENT1_LAYER_FUNCTION = {
    "demolition": BMP.LoadBearing,
    "primary part": BMP.LoadBearing,
    "slab": BMP.LoadBearing,
    "cladding": BMP.Finishing,
    "cladding - horizontal surfaces": BMP.Finishing,
    "screed": BMP.Finishing,
    "sloping layer": BMP.Finishing,
    "infrastructure": URIRef("https://w3id.org/bmp#Infrastructure"),
    "counter battens": URIRef("https://w3id.org/bmp#Infrastructure"),
    "support structure": URIRef("https://w3id.org/bmp#Infrastructure"),
    "boarding": URIRef("https://w3id.org/bmp#Infrastructure"),
    "vapour barrier": URIRef("https://w3id.org/bmp#Infrastructure"),
    "thermal insulation": BMP.Insulating,
    "air cavity": BMP.NonLoadBearing,
}

LAYER_FUNCTION_URI = "https://w3id.org/bmp#LayerFunction"
DCTERMS_DESCRIPTION = URIRef("http://purl.org/dc/terms/description")

worksection_axiom_cache = {}

########################################################
# ONTOLOGY INDEXING (BBSR-style full TBox mining)
########################################################
ontology_g = load_ontology_graph(BASE_DIR / "owl")

MATERIAL_ANCHOR_URIS = {
    str(uri) for uri in ontology_g.objects(BMP.hasMaterialCategory, RDFS.range)
}


def index_taxonomy_branch(parent_uri):
    if not EMBEDDINGS_AVAILABLE:
        return []
    corpus = []
    for child in ontology_g.subjects(RDFS.subClassOf, URIRef(parent_uri)):
        if not isinstance(child, URIRef):
            continue
        local_name = child.split("#")[-1].split("/")[-1]
        text_parts = [local_name]
        for label in ontology_g.objects(child, RDFS.label):
            if getattr(label, "language", None) in (None, "en", "de"):
                text_parts.append(str(label))
        for comment in ontology_g.objects(child, RDFS.comment):
            if getattr(comment, "language", None) in (None, "en"):
                text_parts.append(str(comment))
        combined_text = " | ".join(text_parts)
        embedding = embedder_model.encode(combined_text, convert_to_tensor=True, device=device)
        corpus.append({"uri": str(child), "text": combined_text, "embedding": embedding})
    return corpus


def is_layer_function_entity(entity_uri):
    return entity_uri == LAYER_FUNCTION_URI or LAYER_FUNCTION_URI in str(entity_uri)


def is_material_entity(item):
    entity_uri = item["entity_uri"]
    if is_layer_function_entity(entity_uri):
        return False
    if entity_uri in MATERIAL_ANCHOR_URIS:
        return True
    if any(anchor in item.get("superclasses", []) for anchor in MATERIAL_ANCHOR_URIS):
        return True
    if "/bmp#" not in entity_uri:
        return False
    local_name = entity_uri.split("#")[-1]
    return any(
        token in local_name
        for token in (
            "Wood",
            "Concrete",
            "Brick",
            "Glass",
            "Polymer",
            "Stone",
            "Fibre",
            "Fiber",
            "Hemp",
            "Cork",
            "Straw",
            "Timber",
            "Insulation",
            "Polystyrene",
        )
    )


def build_class_corpus_item(ontology_graph, entity):
    entity_uri = str(entity)
    local_name = entity_uri.split("#")[-1].split("/")[-1]
    text_parts = [local_name]
    for label in ontology_graph.objects(entity, RDFS.label):
        lang = getattr(label, "language", None)
        if lang is None or lang == "en":
            text_parts.append(str(label))
    for comment in ontology_graph.objects(entity, RDFS.comment):
        lang = getattr(comment, "language", None)
        if lang is None or lang == "en":
            text_parts.append(str(comment))
    for desc in ontology_graph.objects(entity, DCTERMS_DESCRIPTION):
        lang = getattr(desc, "language", None)
        if lang is None or lang == "en":
            text_parts.append(str(desc))
    if len(text_parts) == 1:
        text_parts.append(re.sub(r"(?<!^)(?=[A-Z])", " ", local_name))
    return {
        "entity_uri": entity_uri,
        "text": " | ".join(text_parts),
        "superclasses": [str(parent) for parent in ontology_graph.objects(entity, RDFS.subClassOf)],
        "embedding": None,
    }


def create_material_ontology_corpus(ontology_graph):
    material_corpus = []
    for entity in ontology_graph.subjects(RDF.type, OWL.Class):
        if not str(entity).startswith("https://w3id.org/bmp#"):
            continue
        item = build_class_corpus_item(ontology_graph, entity)
        if not is_material_entity(item):
            continue
        material_corpus.append(item)

    if material_corpus:
        embeddings = embedder_model.encode(
            [item["text"] for item in material_corpus],
            convert_to_tensor=True,
            device=device,
        )
        for idx, embedding in enumerate(embeddings):
            material_corpus[idx]["embedding"] = embedding
    return material_corpus


if EMBEDDINGS_AVAILABLE:
    print("Indexing LayerFunction branch and shared ontology corpus...")
    LAYER_FUNCTION_CORPUS = index_taxonomy_branch(LAYER_FUNCTION_URI)
    ONTOLOGY_CORPUS = load_ontology_corpus(ontology_g)
    print(
        f"  LayerFunction items: {len(LAYER_FUNCTION_CORPUS)} | "
        f"Ontology corpus: {len(ONTOLOGY_CORPUS)} | "
        f"Material anchors: {len(MATERIAL_ANCHOR_URIS)}"
    )
else:
    print("Embedding models unavailable; layer-function segment map only.")
    LAYER_FUNCTION_CORPUS = []
    ONTOLOGY_CORPUS = []


def match_text_to_corpus(hint, corpus, fallback_uri, threshold=0.35):
    if not EMBEDDINGS_AVAILABLE or not hint or not corpus:
        return URIRef(fallback_uri)
    hint_emb = embedder_model.encode(hint, convert_to_tensor=True, device=device)
    best_match = fallback_uri
    max_score = -1.0
    for item in corpus:
        score = util.cos_sim(hint_emb, item["embedding"]).item()
        if score > max_score:
            max_score = score
            best_match = item["uri"] if "uri" in item else item["entity_uri"]
    return URIRef(best_match) if max_score > threshold else URIRef(fallback_uri)


def retrieve_material_hierarchical_matches(material_hint, worksection_name="", top_k=8):
    """Step 1: anchor MaterialCategory, then scope to material subclasses/types."""
    text_chunk = material_hint or worksection_name
    if not EMBEDDINGS_AVAILABLE or not text_chunk or not ONTOLOGY_CORPUS:
        return []

    chunk_embedding = embedder_model.encode(text_chunk, convert_to_tensor=True, device=device)
    anchor_scores = []
    for item in ONTOLOGY_CORPUS:
        if item["entity_uri"] not in MATERIAL_ANCHOR_URIS:
            continue
        score = util.cos_sim(chunk_embedding, item["embedding"]).item()
        anchor_scores.append((item, score))
    anchor_scores.sort(key=lambda pair: pair[1], reverse=True)

    scoped_matches = []
    if anchor_scores:
        matched_anchor = anchor_scores[0][0]["entity_uri"]
        for item in ONTOLOGY_CORPUS:
            if not is_material_entity(item):
                continue
            in_branch = (
                item["entity_uri"] == matched_anchor
                or matched_anchor in item.get("superclasses", [])
            )
            if not in_branch:
                local_name = item["entity_uri"].split("#")[-1].lower()
                anchor_name = matched_anchor.split("#")[-1].lower()
                if anchor_name not in local_name and local_name not in text_chunk.lower():
                    continue
            score = util.cos_sim(chunk_embedding, item["embedding"]).item()
            scoped_matches.append(
                {
                    "entity_uri": item["entity_uri"],
                    "text": item["text"],
                    "score": score,
                    "parent_category": matched_anchor,
                }
            )

    if not scoped_matches:
        for item in ONTOLOGY_CORPUS:
            if not is_material_entity(item):
                continue
            score = util.cos_sim(chunk_embedding, item["embedding"]).item()
            scoped_matches.append(
                {
                    "entity_uri": item["entity_uri"],
                    "text": item["text"],
                    "score": score,
                    "parent_category": None,
                }
            )

    scoped_matches.sort(key=lambda match: match["score"], reverse=True)
    return augment_material_matches(text_chunk, scoped_matches, top_k)


MATERIAL_HINT_BOOSTS = {
    "straw": ["https://w3id.org/bmp#Hemp", "https://w3id.org/bmp#HempFibre"],
    "cork": ["https://w3id.org/bmp#Cork"],
    "clt": ["https://w3id.org/bmp#WoodTimber"],
    "wood": ["https://w3id.org/bmp#WoodTimber"],
}


def augment_material_matches(material_hint, matches, top_k=8):
    hint_lower = material_hint.lower().strip()
    seen = {match["entity_uri"] for match in matches}
    for token, boosted_uris in MATERIAL_HINT_BOOSTS.items():
        if token not in hint_lower:
            continue
        if token == "wood" and "fibre" in hint_lower:
            continue
        for boosted_uri in boosted_uris:
            if boosted_uri in seen:
                continue
            corpus_item = next(
                (item for item in ONTOLOGY_CORPUS if item["entity_uri"] == boosted_uri),
                None,
            )
            if corpus_item is None:
                continue
            matches.append(
                {
                    "entity_uri": corpus_item["entity_uri"],
                    "text": corpus_item["text"],
                    "score": 0.95,
                    "parent_category": next(
                        (anchor for anchor in MATERIAL_ANCHOR_URIS if anchor in corpus_item.get("superclasses", [])),
                        None,
                    ),
                }
            )
            seen.add(boosted_uri)
    for item in ONTOLOGY_CORPUS:
        if not is_material_entity(item):
            continue
        if item["entity_uri"] in seen:
            continue
        local_name = item["entity_uri"].split("#")[-1]
        spaced_name = re.sub(r"(?<!^)(?=[A-Z])", " ", local_name)
        name_tokens = [token.lower() for token in re.findall(r"[A-Za-z]{3,}", spaced_name)]
        if not name_tokens:
            continue
        overlap = sum(1 for token in name_tokens if token in hint_lower)
        if overlap == 0:
            continue
        matches.append(
            {
                "entity_uri": item["entity_uri"],
                "text": item["text"],
                "score": 0.5 + overlap * 0.1,
                "parent_category": next(
                    (anchor for anchor in MATERIAL_ANCHOR_URIS if anchor in item.get("superclasses", [])),
                    None,
                ),
            }
        )
        seen.add(item["entity_uri"])

    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:top_k]


########################################################
# HELPERS
########################################################
def make_safe_id(text):
    if not text or (isinstance(text, float) and pd.isna(text)):
        return "Unknown"
    text = re.sub(r"[\s|/\\:\*\?\"<>\|\+\-\.]+", "_", str(text).strip())
    return re.sub(r"_+", "_", text).strip("_")


def parse_thickness_from_text(text):
    match = THICKNESS_PATTERN.search(text or "")
    if not match:
        return None
    value_raw = match.group(1).replace(" ", "")
    unit_token = match.group(2).lower()
    if "-" in value_raw:
        value = value_raw
        datatype = XSD.string
    else:
        value = float(value_raw.replace(",", "."))
        datatype = XSD.float
    unit_iri = UNIT.MilliM if unit_token == "mm" else UNIT.CentiM
    return value, unit_iri, datatype


def parse_slice_worksection(worksection_name):
    parts = [part.strip() for part in str(worksection_name).split("|")]
    return {
        "context": parts[0] if parts else "",
        "function_segment": parts[1] if len(parts) > 1 else "",
        "component_segment": parts[2] if len(parts) > 2 else "",
        "material_segment": parts[3] if len(parts) > 3 else "",
    }


def layer_function_from_slice_segments(worksection_name):
    """Map SLiCE pipe-segment 1 to bmp:LayerFunction using the dataset taxonomy."""
    parsed = parse_slice_worksection(worksection_name)
    function_key = parsed["function_segment"].lower()
    if function_key in SLICE_SEGMENT1_LAYER_FUNCTION:
        return SLICE_SEGMENT1_LAYER_FUNCTION[function_key]

    context = parsed["context"].lower().replace("-", " ")
    if "loadbearing" in context or "load bearing" in context:
        if function_key in {"", "demolition", "primary part", "slab"}:
            return BMP.LoadBearing

    text_lower = str(worksection_name).lower()
    if "thermal insulation" in text_lower:
        return BMP.Insulating
    if "cladding" in text_lower or "thick coating" in text_lower:
        return BMP.Finishing
    if "infrastructure" in text_lower:
        return URIRef("https://w3id.org/bmp#Infrastructure")
    return None


def retrieve_layer_function_matches(text_chunk, top_k=8):
    if not EMBEDDINGS_AVAILABLE or not text_chunk:
        return []
    chunk_embedding = embedder_model.encode(text_chunk, convert_to_tensor=True, device=device)
    matches = []
    for item in LAYER_FUNCTION_CORPUS:
        score = util.cos_sim(chunk_embedding, item["embedding"]).item()
        matches.append({"entity_uri": item["uri"], "text": item["text"], "score": score})
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:top_k]


def llm_resolve_layer_function(worksection_name):
    ontology_matches = retrieve_layer_function_matches(worksection_name)
    ontology_context = "\n".join(
        f"- IRI: {match['entity_uri']} | {match['text']}"
        for match in ontology_matches
    )
    parsed = parse_slice_worksection(worksection_name)
    system_prompt = """
You map SLiCE worksection descriptions to exactly one bmp:LayerFunction class IRI.

The description uses pipe-separated segments:
segment 0 = building part context
segment 1 = functional category (PRIMARY signal)
segment 2+ = component / material detail

Use these rules for segment 1:
- Demolition, Primary part, Slab -> bmp:LoadBearing
- Cladding, Screed, Sloping layer -> bmp:Finishing
- Infrastructure, Boarding, Counter battens, Support structure, Vapour barrier -> bmp:Infrastructure
- Thermal insulation -> bmp:Insulating
- Air cavity -> bmp:NonLoadBearing

Return only raw JSON: {"predicted_function_iri": "https://w3id.org/bmp#Finishing"}
The IRI must be one of the valid options below (or bmp:Infrastructure).
"""
    try:
        if ollama is None:
            raise RuntimeError("ollama package not installed")
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Worksection: '{worksection_name}'\n"
                        f"Parsed segment 1: '{parsed['function_segment']}'\n\n"
                        f"Valid LayerFunction options:\n{ontology_context}"
                    ),
                },
            ],
            options={"temperature": 0.0},
            format="json",
        )
        payload = json.loads(response["message"]["content"].strip())
        function_iri = payload.get("predicted_function_iri")
        return URIRef(function_iri) if function_iri else None
    except Exception as exc:
        print(f"Layer-function LLM failed for '{worksection_name}': {exc}")
        return None


def resolve_layer_function(worksection_name):
    function_iri = layer_function_from_slice_segments(worksection_name)
    if function_iri is None:
        function_iri = llm_resolve_layer_function(worksection_name)
    if function_iri is None and LAYER_FUNCTION_CORPUS:
        function_iri = match_text_to_corpus(
            parse_slice_worksection(worksection_name)["function_segment"] or worksection_name,
            LAYER_FUNCTION_CORPUS,
            str(BMP.NonLoadBearing),
            threshold=0.30,
        )
    if function_iri is None:
        function_iri = BMP.NonLoadBearing
    return function_iri


def extract_material_hint(worksection_name):
    parsed = parse_slice_worksection(worksection_name)
    raw = parsed["material_segment"] or parsed["component_segment"] or worksection_name
    return re.sub(r"\([^)]*\)", "", raw).strip(" |")


def llm_resolve_material_category(worksection_name, function_iri, material_hint, ontology_matches):
    context_items = {match["entity_uri"]: match for match in ontology_matches}
    for item in ONTOLOGY_CORPUS:
        if item["entity_uri"] in MATERIAL_ANCHOR_URIS:
            context_items.setdefault(
                item["entity_uri"],
                {
                    "entity_uri": item["entity_uri"],
                    "text": item["text"],
                    "parent_category": None,
                },
            )
    ontology_context = "\n".join(
        f"- IRI: {match['entity_uri']} [anchor: {match.get('parent_category', 'n/a')}] | {match['text']}"
        for match in context_items.values()
    )
    ontology_matches = list(context_items.values())
    parsed = parse_slice_worksection(worksection_name)
    system_prompt = """
You map SLiCE layer descriptions to bmp material classes from the scoped ontology vocabulary.

Inputs include:
- the resolved bmp:LayerFunction (structural role of the layer)
- the material text segment extracted from the SLiCE pipe description
- candidate material class IRIs mined from the ontology

Rules:
1. predicted_category_iri: broad material anchor (bmp:WoodTimber, bmp:MineralFibre, bmp:NaturalStone, etc.)
2. predicted_type_iri: most specific material class from the scoped list (e.g. bmp:WoodFibre, bmp:WoodSoft)
3. Use the material segment as the primary signal, not the layer function name.
4. Both IRIs MUST come from the scoped ontology options list.
5. Examples:
   - Infrastructure + "Bituminised wood fibre" -> type WoodFibre (category WoodTimber)
   - Cladding + "Cork" -> category Cork
   - Insulating + "Straw" -> category Hemp
   - LoadBearing + "CLT" -> category WoodTimber
   - Demolition + "Wood" -> category WoodTimber

Return only raw JSON:
{
  "predicted_category_iri": "https://w3id.org/bmp#WoodTimber",
  "predicted_type_iri": "https://w3id.org/bmp#WoodFibre"
}
"""
    try:
        if ollama is None:
            raise RuntimeError("ollama package not installed")
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Worksection: '{worksection_name}'\n"
                        f"Resolved layer function: <{function_iri}>\n"
                        f"Material segment: '{material_hint}'\n"
                        f"Component segment: '{parsed['component_segment']}'\n\n"
                        f"Scoped ontology material options:\n{ontology_context}"
                    ),
                },
            ],
            options={"temperature": 0.0},
            format="json",
        )
        payload = json.loads(response["message"]["content"].strip())
        type_iri = payload.get("predicted_type_iri")
        category_iri = payload.get("predicted_category_iri")
        allowed = {match["entity_uri"] for match in ontology_matches}
        for candidate in (type_iri, category_iri):
            if candidate in allowed:
                return URIRef(candidate)
    except Exception as exc:
        print(f"Material-category LLM failed for '{worksection_name}': {exc}")
    return None


def resolve_material_assignment(worksection_name, function_iri):
    material_hint = extract_material_hint(worksection_name)
    if not EMBEDDINGS_AVAILABLE or not ONTOLOGY_CORPUS:
        return None, None

    ctx = process_description(
        subject_uri=f"https://w3id.org/slice#material/{abs(hash(worksection_name)) % 10**8}",
        german_desc=worksection_name,
        english_desc=material_hint,
        ontology_corpus=ONTOLOGY_CORPUS,
        profile="slice",
        extra_text=material_hint,
        ontology_graph=ontology_g,
    )
    record_ctx(worksection_name, ctx, source="slice")
    cat, typ = material_pair_from_nlp(ctx, ontology_g) if ctx else (None, None)
    pair = ensure_layer_material_pair(ontology_g, category_iri=cat, type_iri=typ)
    if pair[0] is None:
        print(
            f"  WARNING: no material mapped for '{worksection_name[:60]}' "
            f"(layer function {function_iri} is not a material — check NLP/retrieval)"
        )
    return pair


# Kept for reference — previous single-IRI resolver
# def resolve_material_category(worksection_name, function_iri):
#     ...


def infer_element_type(element_name, worksections, sfb_class):
    combined = " ".join(
        part for part in [str(element_name or ""), " ".join(worksections[:3]), str(sfb_class or "")] if part
    ).lower()
    if any(token in combined for token in ("roof", "fr0", "ceiling")) or str(sfb_class).startswith("27"):
        return BEO.Roof
    if any(token in combined for token in ("slab", "floor", "if0", "deck")) or str(sfb_class).startswith("23"):
        return BEO.Slab
    return BEO.Wall


def lcm_group_key(row):
    for column in ("LCM_nest1_code", "LCM_code"):
        if column in row and pd.notna(row[column]):
            return str(row[column]).strip()
    return "Unknown"


def lcs_suffix(lcm_code):
    code = str(lcm_code).strip()
    if code in LCS_SUFFIX_MAP:
        return LCS_SUFFIX_MAP[code]
    return f"LCS_{make_safe_id(code)}"


def lcm_to_stage_types(lcm_code):
    code = str(lcm_code).strip()
    if code == "A1-3":
        return [LCA.A1, LCA.A2, LCA.A3]
    base = code.split(".")[0]
    if hasattr(LCA, base):
        return [getattr(LCA, base)]
    return []


def map_activity_type(activity_name):
    if pd.isna(activity_name):
        return None
    return ACTIVITY_TYPE_MAP.get(str(activity_name).strip())


def map_unit(unit_name):
    if pd.isna(unit_name):
        return None
    return UNIT_MAP.get(str(unit_name).strip().lower())


def aggregate_indicator_values(rows, column):
    total = 0.0
    has_value = False
    for _, row in rows.iterrows():
        if column in row and pd.notna(row[column]):
            total += float(row[column])
            has_value = True
    return total if has_value else None


########################################################
# OWL-CONSTRAINED LLM WORKSECTION PARSING
########################################################
def mine_worksection_axioms(worksection_name):
    if worksection_name in worksection_axiom_cache:
        return worksection_axiom_cache[worksection_name]

    function_iri = resolve_layer_function(worksection_name)
    category_iri, type_iri = resolve_material_assignment(worksection_name, function_iri)

    parsed_thickness = parse_thickness_from_text(worksection_name)
    if parsed_thickness:
        thickness_value, thickness_unit, thickness_datatype = parsed_thickness
    else:
        thickness_value = None
        thickness_unit = None
        thickness_datatype = None

    result = {
        "function_iri": function_iri if isinstance(function_iri, URIRef) else URIRef(function_iri),
        "category_iri": category_iri,
        "type_iri": type_iri,
        "thickness_value": thickness_value,
        "thickness_unit": thickness_unit if thickness_value is not None else None,
        "thickness_datatype": thickness_datatype if thickness_value is not None else None,
    }
    worksection_axiom_cache[worksection_name] = result
    return result


def add_layer_axiom_triples(layer_uri, worksection_name, axioms):
    if axioms["thickness_value"] is not None:
        thick_node = BNode()
        g.add((layer_uri, BMP.hasThickness, thick_node))
        g.add(
            (
                thick_node,
                BMP.hasValue,
                Literal(axioms["thickness_value"], datatype=axioms["thickness_datatype"]),
            )
        )
        g.add((thick_node, BMP.hasUnit, axioms["thickness_unit"]))


def add_lcs_impact_node(material_uri, layer_idx, el_idx, lcm_code, rows):
    stage_suffix = lcs_suffix(lcm_code)
    lcs_uri = SLICE[f"material_inst_layer{el_idx}_{layer_idx}_{stage_suffix}"]

    g.add((lcs_uri, RDF.type, OWL.NamedIndividual))
    for stage_type in lcm_to_stage_types(lcm_code):
        g.add((lcs_uri, RDF.type, stage_type))

    first_row = rows.iloc[0]
    lcs_label = str(first_row["LCS_name"]).strip() if pd.notna(first_row.get("LCS_name")) else lcm_code
    g.add((lcs_uri, LCA.hasLifeCycleStage, Literal(lcs_label, lang="en")))

    techflows = sorted(
        {
            str(value).strip()
            for value in rows["techflow_name"]
            if pd.notna(value) and str(value).strip()
        }
    )
    if techflows:
        g.add((lcs_uri, LCA.hasTechflowDescription, Literal(" | ".join(techflows), lang="en")))

    activity_iri = map_activity_type(first_row.get("activity_type"))
    if activity_iri:
        g.add((lcs_uri, LCA.hasActivityType, activity_iri))

    unit_iri = map_unit(first_row.get("techflow_unit"))
    if unit_iri:
        g.add((lcs_uri, LCA.hasUnit, unit_iri))

    amount = aggregate_indicator_values(rows, "techflow_amount")
    if amount is not None:
        g.add((lcs_uri, LCA.declaredUnitAmount, Literal(amount, datatype=XSD.float)))

    for column, ontology_prop in INDICATOR_MAP.items():
        value = aggregate_indicator_values(rows, column)
        if value is not None:
            g.add((lcs_uri, ontology_prop, Literal(value, datatype=XSD.float)))

    g.add((material_uri, BMP.hasLifeCycleStage, lcs_uri))


########################################################
# MASTER PARSING ENGINE
########################################################
def process_slice_dataset(csv_path):
    global g
    g = Graph()
    g.bind("lca", LCA)
    g.bind("bot", BOT)
    g.bind("beo", BEO)
    g.bind("bmp", BMP)
    g.bind("at", AT)
    g.bind("bpo", BPO)
    g.bind("qudt", QUDT)
    g.bind("unit", UNIT)
    g.bind("slice", SLICE)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    print(f"Beginning SLiCE CSV to TTL pipeline: {csv_path.name}")
    begin_report("slice")
    df = pd.read_csv(csv_path)
    df = df[df["element_name"].notna()].copy()

    element_names = list(dict.fromkeys(df["element_name"].tolist()))

    test_element = TEST_ELEMENT_NAME
    if test_element is None and os.environ.get("SLICE_TEST_ELEMENT"):
        test_element = os.environ["SLICE_TEST_ELEMENT"]
    if test_element:
        element_names = [n for n in element_names if n == test_element]
        if not element_names:
            raise ValueError(f"SLICE test element not found in CSV: {test_element!r}")
        print(f"TEST MODE: processing SLiCE element {test_element!r} only")

    print(f"Found {len(element_names)} building element(s) to process.")

    for el_idx, element_name in enumerate(element_names, start=1):
        element_df = df[df["element_name"] == element_name]
        worksections = list(dict.fromkeys(element_df["worksection_name"].dropna().tolist()))
        sfb_class = element_df["element_class_sfb_mmg"].iloc[0] if "element_class_sfb_mmg" in element_df else None

        element_uri = SLICE[f"element_inst_{el_idx:02d}"]
        layerset_uri = SLICE[f"layerSet_{el_idx:02d}"]
        element_type = infer_element_type(element_name, worksections, sfb_class)

        g.add((element_uri, RDF.type, OWL.NamedIndividual))
        g.add((element_uri, RDF.type, BEO.BuiltElement))
        g.add((element_uri, RDF.type, AT.ElementArchetype))
        g.add((element_uri, RDF.type, element_type))
        g.add((element_uri, RDFS.label, Literal(str(element_name), datatype=XSD.string)))
        g.add((element_uri, BMP.hasLayerSet, layerset_uri))

        for worksection_name in worksections:
            g.add(
                (
                    element_uri,
                    AT.hasArchetypeDescription,
                    Literal(worksection_name, datatype=XSD.string),
                )
            )

        g.add((layerset_uri, RDF.type, BPO.Product))
        g.add((layerset_uri, RDF.type, BMP.LayerSet))

        for layer_idx, worksection_name in enumerate(worksections, start=1):
            layer_uri = SLICE[f"Layer_inst_{el_idx:02d}_{layer_idx:02d}"]
            material_uri = SLICE[f"material_inst_layer{el_idx}_{layer_idx}"]

            g.add((layerset_uri, BMP.hasLayer, layer_uri))
            g.add((layer_uri, RDF.type, BPO.Component))
            g.add((layer_uri, RDF.type, BMP.Layer))
            g.add((layer_uri, BPO.isPartOf, layerset_uri))
            g.add(
                (
                    layer_uri,
                    AT.hasArchetypeDescription,
                    Literal(worksection_name, datatype=XSD.string),
                )
            )

            axioms = mine_worksection_axioms(worksection_name)
            add_layer_axiom_triples(layer_uri, worksection_name, axioms)
            cat_label = str(axioms["category_iri"]).split("#")[-1] if axioms["category_iri"] else "-"
            typ_label = str(axioms["type_iri"]).split("#")[-1] if axioms["type_iri"] else "-"
            ws_short = worksection_name if len(worksection_name) <= 60 else worksection_name[:60] + "..."
            print(f"    layer {layer_idx}: {ws_short} -> bmp:{cat_label} / bmp:{typ_label}")

            emit_enforced_layer(
                g,
                layer_uri=layer_uri,
                material_uri=material_uri,
                function_iri=axioms["function_iri"],
                category_iri=axioms["category_iri"],
                type_iri=axioms["type_iri"],
                ontology_graph=ontology_g,
                layer_types=(BPO.Component, BMP.Layer),
            )

            layer_rows = element_df[element_df["worksection_name"] == worksection_name]
            for lcm_code, lcm_rows in layer_rows.groupby(
                layer_rows.apply(lcm_group_key, axis=1),
                sort=False,
            ):
                add_lcs_impact_node(material_uri, layer_idx, el_idx, lcm_code, lcm_rows)

        print(
            f"  [{el_idx:02d}/{len(element_names)}] {element_name}: "
            f"{len(worksections)} layers, {len(element_df)} impact rows"
        )

    out_dir = BASE_DIR / "ttl"
    out_dir.mkdir(parents=True, exist_ok=True)
    if test_element:
        safe_name = re.sub(r"[^\w]+", "_", test_element).strip("_")[:40]
        out_path = out_dir / f"slice_{safe_name}_test.ttl"
    else:
        out_path = out_dir / "slice_data_instantiated.ttl"
    g.serialize(destination=out_path, format="turtle")
    finalize_report(data_graph=g, ontology_graph=ontology_g, ttl_path=out_path)
    layerset_count = sum(1 for _ in g.subjects(RDF.type, BMP.LayerSet))
    layer_count = sum(1 for _ in g.subjects(RDF.type, BMP.Layer))
    print(f"\nProcessing complete. SLiCE graph saved to: {out_path}")
    print(f"Summary: {layerset_count} layer sets | {layer_count} layers | {len(g)} triples")
    return out_path


if __name__ == "__main__":
    csv_file_path = BASE_DIR / "data" / "data_csv_SLiCE" / "SLiCE_Mouton2022a_dropNAN.csv"
    process_slice_dataset(csv_file_path)
