"""
Per-element-type guidance for the LLM user prompt.

Edit ELEMENT_HINTS below in your own words. Keys are beo: element class IRIs.
The shared system prompt and OWL vocabulary stay the same; only this slice
changes per wall / slab / roof / floor.

Used when process_description(..., element_type_iri=...) is set.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reformulate these hints yourself — they are injected into the LLM user prompt
# ---------------------------------------------------------------------------

ELEMENT_HINTS: dict[str, str] = {
    "https://w3id.org/beo#Wall": """
Element class: Wall (beo:Wall)

Structural chain (every layer_set):
  layer_topology → rdf:type bmp:LayerSet + one topology (Cavity, MultiLayer, SingleLayer, …)
  each layer → predicted_function_iri → bmp:hasLayerFunction (Facade, Air, LoadBearing, …)

Material chain (required when the text names a material):
  predicted_category_iri must be set for that layer; predicted_type_iri only if a subtype is named.

Topology from assembly wording ("cavity wall" → Cavity; "multi-layered" → MultiLayer).
Air-gap layer = bmp:Air function, not automatically bmp:Cavity topology; Air has no material.
Material alternatives ("oder" / "or") → separate layer_sets[].
"geputzt" / plastered both sides → two Finishing layers (Binders/GypsumPlaster) and MultiLayer.
""".strip(),
    "https://w3id.org/beo#WallPARTITIONING": """
Element class: Partition wall (beo:WallPARTITIONING)
Often single-layer or thin multi-layer; lighter structure than external walls.
""".strip(),
    "https://w3id.org/beo#Slab": """
Element class: Slab (beo:Slab)
Typical assemblies: multi-layer deck (structural + insulation + finish).
""".strip(),
    "https://w3id.org/beo#Floor": """
Element class: Floor (beo:Floor)
Finish layers, screed, structural slab — describe per stated thicknesses.
If the text names concrete/brick, set predicted_category_iri; leave type null unless a subtype is named.
""".strip(),
    "https://w3id.org/beo#Roof": """
Element class: Roof (beo:Roof)
Typical assemblies: membrane, insulation, structure; cold or warm roof build-ups.
""".strip(),
}

DEFAULT_ELEMENT_HINT = """
Element class: generic building element.
Apply the shared composition schema: element → layer_sets[] → layers[] → materials.
""".strip()


def element_hint_for(element_type_iri: str | None) -> str:
    if not element_type_iri:
        return DEFAULT_ELEMENT_HINT
    return ELEMENT_HINTS.get(element_type_iri.strip(), DEFAULT_ELEMENT_HINT)
