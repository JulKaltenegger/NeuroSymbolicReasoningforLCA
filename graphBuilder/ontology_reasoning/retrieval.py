"""Step 5: hierarchical ontology match retrieval."""

from __future__ import annotations

import re

from .chunking import combined_description
from .config import get_profile
from .layer_axioms import LAYER_FUNCTION_URI, BMP_LAYER_SET
from .material_axioms import BMP_MATERIAL_CATEGORY, BMP_MATERIAL_TYPE
from .corpus import encode_text

# [UNUSED — REVIVE CANDIDATE] Deterministic keyword → LayerFunction fallback, never wired in.
# This is the missing safety net for descriptions where no LayerFunction clears
# similarity_threshold (see ttl/text_examples/material_alternatives.json, where zero Function
# matches were retrieved and the LLM invented bmp:Facade unsupported by any evidence).
# INSULATION_PATTERN = re.compile(
#     r"\b(insulation|insulated|dämmung|daemmung|dämmkern|daemmkern|"
#     r"wärmedämm|waermedaemm|dämmstoff|daemmstoff|gedämmt|gedaemmt)\b",
#     re.IGNORECASE,
# )
# GLAZING_PATTERN = re.compile(
#     r"\b(glazing|glazed|verglasung|scheiben|fenster|window|triple glazing|double glazing)\b",
#     re.IGNORECASE,
# )
# OPENING_PATTERN = re.compile(r"\b(door|tür|tur|haustür|entry door|metal door)\b", re.IGNORECASE)
# AIR_GAP_PATTERN = re.compile(
#     r"\b(air gap|airgap|air layer|luftschicht|hohlraum|cavity)\b",
#     re.IGNORECASE,
# )
# LOAD_BEARING_PATTERN = re.compile(
#     r"\b(load[- ]bearing|loadbearing|tragend|tragwerk|bearing layer)\b",
#     re.IGNORECASE,
# )
# FACADE_PATTERN = re.compile(r"\b(facade|fassade|cladding|verkleidung)\b", re.IGNORECASE)

MATERIAL_HINT_BOOSTS = {
    "straw": ["https://w3id.org/bmp#Hemp", "https://w3id.org/bmp#HempFibre"],
    "cork": ["https://w3id.org/bmp#Cork"],
    "clt": ["https://w3id.org/bmp#WoodTimber"],
    "wood": ["https://w3id.org/bmp#WoodTimber"],
    "hohlloch": ["https://w3id.org/bmp#Brick", "https://w3id.org/bmp#BrickMasonryUnit"],
    "hollow-hole": ["https://w3id.org/bmp#Brick", "https://w3id.org/bmp#BrickMasonryUnit"],
    "leichtbeton": ["https://w3id.org/bmp#Concrete", "https://w3id.org/bmp#LightWeightConcrete"],
    "lightweight concrete": ["https://w3id.org/bmp#Concrete", "https://w3id.org/bmp#LightWeightConcrete"],
}

# IRI local name (bmp:Cavity → "Cavity") is scored on its own so a long rdfs:label
# cannot bury the class name. Final score is the best of name vs label/comment mix.
_LABEL_WEIGHT = 0.70
_COMMENT_WEIGHT = 0.30
LAYER_SET_URI = str(BMP_LAYER_SET)
MATERIAL_CATEGORY_URI = str(BMP_MATERIAL_CATEGORY)
MATERIAL_TYPE_URI = str(BMP_MATERIAL_TYPE)


def _cos_sim(a, b):
    from sentence_transformers import util

    return util.cos_sim(a, b).item()


def _item_score(chunk_embedding, item) -> float:
    full = _cos_sim(chunk_embedding, item["embedding"])
    label_emb = item.get("label_embedding")
    name_emb = item.get("name_embedding")
    label = _cos_sim(chunk_embedding, label_emb) if label_emb is not None else full
    mixed = _LABEL_WEIGHT * label + _COMMENT_WEIGHT * full
    if name_emb is None:
        return mixed
    return max(_cos_sim(chunk_embedding, name_emb), mixed)

# [UNUSED — REVIVE CANDIDATE] see the pattern block above
# KEYWORD_MATERIAL_IRIS = {
#     "cork": "https://w3id.org/bmp#Cork",
#     "straw": "https://w3id.org/bmp#Wood",
#     "clt": "https://w3id.org/bmp#WoodTimber",
#     "hemp": "https://w3id.org/bmp#Hemp",
# }
#
# KEYWORD_FUNCTION_IRIS = {
#     "insulation": "https://w3id.org/bmp#Insulating",
#     "glazing": "https://w3id.org/bmp#Glazing",
#     "door": "https://w3id.org/bmp#Opening",
#     "air_gap": "https://w3id.org/bmp#Air",
#     "load_bearing": "https://w3id.org/bmp#LoadBearing",
#     "facade": "https://w3id.org/bmp#Facade",
# }
#
#
# def keyword_material_iri(text: str) -> str | None:
#     if not text:
#         return None
#     text_lower = text.lower()
#     for token, iri in KEYWORD_MATERIAL_IRIS.items():
#         if re.search(rf"\b{re.escape(token)}\b", text_lower):
#             return iri
#     return None
#
#
# def keyword_layer_function_iris(text: str) -> list[str]:
#     """All layer-function IRIs implied by keywords in the text (outer → inner order)."""
#     if not text:
#         return []
#     found: list[str] = []
#     if FACADE_PATTERN.search(text):
#         found.append(KEYWORD_FUNCTION_IRIS["facade"])
#     if INSULATION_PATTERN.search(text):
#         found.append(KEYWORD_FUNCTION_IRIS["insulation"])
#     if GLAZING_PATTERN.search(text):
#         found.append(KEYWORD_FUNCTION_IRIS["glazing"])
#     if OPENING_PATTERN.search(text):
#         found.append(KEYWORD_FUNCTION_IRIS["door"])
#     if AIR_GAP_PATTERN.search(text):
#         found.append(KEYWORD_FUNCTION_IRIS["air_gap"])
#     if LOAD_BEARING_PATTERN.search(text):
#         found.append(KEYWORD_FUNCTION_IRIS["load_bearing"])
#     return list(dict.fromkeys(found))
#
#
# def keyword_layer_function_iri(text: str) -> str | None:
#     iris = keyword_layer_function_iris(text)
#     return iris[0] if iris else None


def _is_layer_function_candidate(item) -> bool:
    """Phase A `is_layer_function` flag; drop the abstract bmp:LayerFunction class."""
    return bool(item.get("is_layer_function")) and item["entity_uri"] != LAYER_FUNCTION_URI


def _is_layerset_topology_candidate(item) -> bool:
    """Phase A `is_layerset_topology` flag; drop the abstract bmp:LayerSet class."""
    return bool(item.get("is_layerset_topology")) and item["entity_uri"] != LAYER_SET_URI


def _is_material_category_candidate(item) -> bool:
    """Phase A `is_material_category` flag; drop the abstract bmp:MaterialCategory class."""
    return bool(item.get("is_material_category")) and item["entity_uri"] != MATERIAL_CATEGORY_URI


def _is_material_type_candidate(item) -> bool:
    """Phase A `is_material_type` flag; never a LayerFunction or the abstract type class."""
    return (
        bool(item.get("is_material_type"))
        and not item.get("is_layer_function")
        and item["entity_uri"] != MATERIAL_TYPE_URI
    )


def _score_corpus(chunk_embedding, ontology_corpus, predicate):
    matches = []
    for item in ontology_corpus:
        if not predicate(item):
            continue
        matches.append(
            {
                "entity_uri": item["entity_uri"],
                "text": item["text"],
                "score": _item_score(chunk_embedding, item),
                "parent_category": None,
                "taxonomy_branch": None,
            }
        )
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


def retrieve_matches_for_chunk(chunk_embedding, ontology_corpus, *, profile: str, top_k: int = 5):
    cfg = get_profile(profile)
    branches = cfg["branches"]
    threshold = cfg["similarity_threshold"]
    results = []

    if "layer_function" in branches:
        # Only proper bmp:LayerFunction subclasses may be scored for this slot: the
        # abstract anchor and every material/property class stay out of the candidate list.
        function_matches = _score_corpus(
            chunk_embedding,
            ontology_corpus,
            _is_layer_function_candidate,
        )
        for match in function_matches[:3]:
            match["parent_category"] = LAYER_FUNCTION_URI
            match["taxonomy_branch"] = "Function"
            if match["score"] >= threshold:
                results.append(match)

    if "layerset" in branches:
        topology_matches = _score_corpus(
            chunk_embedding,
            ontology_corpus,
            _is_layerset_topology_candidate,
        )
        for match in topology_matches[:3]:
            match["parent_category"] = LAYER_SET_URI
            match["taxonomy_branch"] = "LayerSet"
            if match["score"] >= threshold:
                results.append(match)

    if "material" in branches:
        material_categories = _score_corpus(
            chunk_embedding,
            ontology_corpus,
            _is_material_category_candidate,
        )
        scoped_material_types = []
        if material_categories and material_categories[0]["score"] >= threshold:
            matched_anchor = material_categories[0]["entity_uri"]
            for item in ontology_corpus:
                under_anchor = (
                    matched_anchor in item.get("superclasses", [])
                    or item["entity_uri"] == matched_anchor
                )
                if not under_anchor or not _is_material_type_candidate(item):
                    continue
                scoped_material_types.append(
                    {
                        "entity_uri": item["entity_uri"],
                        "text": item["text"],
                        "score": _item_score(chunk_embedding, item),
                        "parent_category": matched_anchor,
                        "taxonomy_branch": "Material",
                    }
                )
            scoped_material_types.sort(key=lambda m: m["score"], reverse=True)
            results.extend(scoped_material_types[:top_k])
        else:
            material_matches = _score_corpus(
                chunk_embedding,
                ontology_corpus,
                _is_material_type_candidate,
            )
            results.extend([m for m in material_matches[:top_k] if m["score"] >= threshold])

    return results


def retrieve_matches(
    chunks: list[str],
    ontology_corpus,
    *,
    profile: str = "bbsr",
    german_desc: str | None = None,
    english_desc: str | None = None,
    top_k: int | None = None,
):
    cfg = get_profile(profile)
    top_k = top_k or cfg["retrieval_top_k"]
    combined = combined_description(german_desc, english_desc)
    aggregated = []

    queries = list(chunks)
    if combined and combined not in queries:
        queries.append(combined)

    for chunk in queries:
        chunk_embedding = encode_text(chunk)
        aggregated.extend(
            retrieve_matches_for_chunk(chunk_embedding, ontology_corpus, profile=profile, top_k=top_k)
        )

    if profile == "slice" and combined:
        aggregated = augment_material_matches(combined, aggregated, ontology_corpus, top_k=cfg["top_k"])

    return deduplicate_matches(aggregated, limit=cfg["top_k"])


def deduplicate_matches(matches, limit: int = 10):
    seen: set[str] = set()
    unique = []
    for match in sorted(matches, key=lambda m: m["score"], reverse=True):
        if match["entity_uri"] in seen:
            continue
        seen.add(match["entity_uri"])
        unique.append(match)
    return unique[:limit]


def augment_material_matches(material_hint: str, matches: list[dict], ontology_corpus, top_k: int = 8):
    hint_lower = (material_hint or "").lower().strip()
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
                (item for item in ontology_corpus if item["entity_uri"] == boosted_uri),
                None,
            )
            if corpus_item is None:
                continue
            matches.append(
                {
                    "entity_uri": corpus_item["entity_uri"],
                    "text": corpus_item["text"],
                    "score": 0.95,
                    "parent_category": None,
                    "taxonomy_branch": "Material",
                }
            )
            seen.add(boosted_uri)

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:top_k]
