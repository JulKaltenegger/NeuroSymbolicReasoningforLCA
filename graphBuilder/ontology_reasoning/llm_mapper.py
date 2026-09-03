"""Step 6: LLM schema reasoning with OWL-derived prompts and retrieval context."""

from __future__ import annotations

from .chunking import combined_description
from .config import get_llm_config, get_profile
from .llm_backends import chat_json, is_llm_available, log_llm_provider_once
from .element_hints import element_hint_for
from .owl_schema import COMPOSITION_TTL_SKELETON, build_scoped_vocabulary

COMPOSITION_JSON_RULES = """
Return raw JSON only. Use ONLY IRIs that appear in the user prompt (OWL axioms or retrieval matches).

Two parallel OWL role chains (never mix slots):

STRUCTURAL (LayerSet + Layer):
  Element → layer_sets[] → each LayerSet:
    layer_topology  → rdf:type on LayerSet: one subclass of bmp:LayerSet (Cavity, MultiLayer, SingleLayer, Frame)
    layers[]        → each Layer:
      predicted_function_iri → bmp:hasLayerFunction: one subclass of bmp:LayerFunction (Facade, Air, LoadBearing, …)

MATERIAL (optional per layer when text names a material):
  predicted_category_iri → bmp:hasMaterialCategory (category anchor from OWL range)
  predicted_type_iri     → bmp:hasMaterialType (type rdfs:subClassOf that category)

Structure:
1. element_thickness_cm — when stated for the whole element
2. layer_sets[] — one entry per assembly or material alternative ("oder" / "or")
   - layer_topology — exactly ONE LayerSet topology subclass (see OWL list in user prompt)
   - thickness_cm — total assembly thickness
   - layers[] — ordered outer → inner; one object per stated layer
3. Each layer: layer_index, thickness_cm, predicted_function_iri (required), optional material IRIs

LayerSet topology — follow explicit assembly wording in the text:
  * "cavity (wall)" / "Hohlraum" → bmp:Cavity
  * "multi-layer(ed)" / "mehrschichtig" → bmp:MultiLayer
  * "single-layer" / "einschichtig" → bmp:SingleLayer
  * "frame" → bmp:Frame
  An air-gap layer (bmp:Air function) does NOT by itself mean bmp:Cavity topology.

Example A — text says "cavity wall" (structure only):
{
  "element_thickness_cm": 25,
  "layer_sets": [{
    "layer_topology": "https://w3id.org/bmp#Cavity",
    "thickness_cm": 25,
    "layers": [
      {"layer_index": 1, "thickness_cm": 5, "predicted_function_iri": "https://w3id.org/bmp#Facade"},
      {"layer_index": 2, "thickness_cm": 5, "predicted_function_iri": "https://w3id.org/bmp#Air"},
      {"layer_index": 3, "thickness_cm": 15, "predicted_function_iri": "https://w3id.org/bmp#LoadBearing"}
    ]
  }]
}

Example B — same layers but text says "multi-layered wall" (not "cavity wall"):
{
  "element_thickness_cm": 25,
  "layer_sets": [{
    "layer_topology": "https://w3id.org/bmp#MultiLayer",
    "thickness_cm": 25,
    "layers": [
      {"layer_index": 1, "thickness_cm": 5, "predicted_function_iri": "https://w3id.org/bmp#Facade"},
      {"layer_index": 2, "thickness_cm": 5, "predicted_function_iri": "https://w3id.org/bmp#Air"},
      {"layer_index": 3, "thickness_cm": 15, "predicted_function_iri": "https://w3id.org/bmp#LoadBearing"}
    ]
  }]
}

When the text names a material for a layer, you MUST set predicted_category_iri:
  * a category word only (concrete, brick, wood, …) → predicted_category_iri only.
    Leave predicted_type_iri null. Do NOT pick a default subtype.
    "hollow concrete" is still bmp:Concrete with null type — not LightWeightConcrete.
  * a named type (Leichtbeton, Hohllochziegel, GypsumPlaster, …) → both category and type.
  * no material named (air gap, unnamed insulation, function-only facade) → omit both material IRIs.

Example C — "30 cm … hollow concrete blocks" (category only):
{
  "element_thickness_cm": 30,
  "layer_sets": [{
    "layer_topology": "https://w3id.org/bmp#SingleLayer",
    "thickness_cm": 30,
    "layers": [
      {"layer_index": 1, "thickness_cm": 30, "predicted_function_iri": "https://w3id.org/bmp#LoadBearing", "predicted_category_iri": "https://w3id.org/bmp#Concrete"}
    ]
  }]
}

If German and English conflict, follow German ("geputzt" = plastered, not "cleaned").
"beiderseitig geputzt" / plastered on both sides → two Finishing layers
(Binders + GypsumPlaster), topology MultiLayer, even if the masonry leaf is einschichtig.
Each plaster face is its own layer (do not share one material instance).

Do not invent thickness. Set thickness_cm only when the text states it for that layer or assembly.

\"oder\" / \"or\" / \"alternatively\" / \" / \" between materials or placements → one layer_sets[] entry per alternative. Do not collapse them into one LayerSet.
""".strip()


def build_ontology_context_string(
    matches,
    *,
    ontology_graph=None,
    ontology_corpus=None,
    profile: str = "bbsr",
) -> str:
    """OWL axiom structure + cosine retrieval matches (user prompt vocabulary block)."""
    if ontology_graph is None:
        from .owl_schema import format_retrieval_matches

        return format_retrieval_matches(matches, ontology_corpus)
    return build_scoped_vocabulary(
        matches,
        ontology_graph=ontology_graph,
        ontology_corpus=ontology_corpus,
        profile=profile,
    ).prompt_block()


def _system_prompt(llm_mode: str) -> str:
    if llm_mode == "material_only":
        return (
            "You map construction layer descriptions to bmp material IRIs.\n"
            "The user prompt contains OWL material hierarchy axioms and cosine retrieval matches.\n"
            "Use ONLY IRIs from that prompt.\n"
            "Return JSON only:\n"
            '{"material_category_iri": "https://w3id.org/bmp#...", '
            '"material_type_iri": "https://w3id.org/bmp#..."}\n'
            "Category must be the anchor; type must be a valid subtype per OWL subClassOf."
        )
    # [UNUSED] no profile sets llm_mode="layers_only" (only "full" and "material_only")
    # if llm_mode == "layers_only":
    #     return (
    #         f"{COMPOSITION_TTL_SKELETON}\n\n"
    #         "Map the description to LayerSet JSON using only IRIs from the user prompt.\n"
    #         "Return raw JSON with keys layer_topology, thickness_cm, layers[]."
    #     )
    return f"{COMPOSITION_TTL_SKELETON}\n\n{COMPOSITION_JSON_RULES}"


def map_layers_with_llm(
    *,
    subject_uri: str,
    german_desc: str,
    english_desc: str,
    ontology_matches,
    profile: str = "bbsr",
    llm_model: str | None = None,
    ontology_graph=None,
    ontology_corpus=None,
    element_type_iri: str | None = None,
):
    cfg = get_profile(profile)
    if not cfg.get("use_llm") or not is_llm_available():
        return None

    if ontology_graph is None:
        return None

    log_llm_provider_once()
    llm_cfg = get_llm_config()
    model = llm_model or llm_cfg["model"]

    vocab = build_scoped_vocabulary(
        ontology_matches,
        ontology_graph=ontology_graph,
        ontology_corpus=ontology_corpus,
        profile=profile,
    )
    if not vocab.axiom_structure.strip() and not ontology_matches:
        return None

    combined = combined_description(german_desc, english_desc)
    element_hint = element_hint_for(element_type_iri)
    user_prompt = f"""
Analyze Element Configuration:
- Subject Element URI: <{subject_uri}>
- Element type IRI: {element_type_iri or "(not specified)"}
- German Description: "{german_desc or ''}"
- English Description: "{english_desc or ''}"
- Combined: "{combined}"

Element-type guidance:
{element_hint}

Read the FULL description. Map it to the target RDF shape in the system prompt.
Use IRIs only from the OWL axioms and retrieval matches below.

{vocab.prompt_block()}
"""
    try:
        return chat_json(_system_prompt(cfg["llm_mode"]), user_prompt, model=model)
    except Exception as exc:
        print(f"LLM processing failed for {subject_uri} ({llm_cfg['provider']}/{model}): {exc}")
        return None
