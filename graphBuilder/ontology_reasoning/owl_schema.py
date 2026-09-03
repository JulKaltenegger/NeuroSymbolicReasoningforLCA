"""OWL-derived composition schema for LLM prompts and validation whitelists.

Reads object properties, class hierarchies, and ranges directly from the loaded
ontology graph — no hand-maintained category/type lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDFS

from .config import get_profile
from .layer_axioms import (
    BMP_LAYER_FUNCTION,
    LAYER_COMPOSITION_PROPERTIES,
    layer_function_uris,
    layerset_topology_uris,
)
from .material_axioms import (
    anchor_for_material_iri,
    is_material_category_iri,
    is_material_type_iri,
    material_category_type_tree,
    material_category_uris,
    material_type_uris,
    MATERIAL_COMPOSITION_PROPERTIES,
)

# Composition object properties for OWL axiom summaries (layer + material links + thickness).
COMPOSITION_PROPERTIES = (
    *LAYER_COMPOSITION_PROPERTIES,
    *MATERIAL_COMPOSITION_PROPERTIES,
    URIRef("https://w3id.org/bmp#hasThickness"),
)

COMPOSITION_TTL_SKELETON = """
@prefix beo: <https://w3id.org/beo#> .
@prefix bmp: <https://w3id.org/bmp#> .
@prefix at: <https://w3id.org/at#> .

# TARGET RDF SHAPE — structural + material roles (both from OWL subclasses / ranges).

:element a beo:Wall ;
  at:hasLayerSet :layerset_1 .                 # object property: Element → LayerSet

:layerset_1 a bmp:LayerSet, bmp:MultiLayer ;  # classes: base + topology subclass
  bmp:hasLayer :layer_1, :layer_2 .           # object property: LayerSet → Layer

:layer_1 a bmp:Layer ;                        # class: Layer instance
  bmp:hasLayerFunction bmp:Facade .             # object property: Layer → LayerFunction class

:layer_2 a bmp:Layer ;
  bmp:hasLayerFunction bmp:LoadBearing ;
  at:hasMaterial :mat_1 .                       # object property: Layer → Material

:mat_1 a bmp:Material ;                         # class: Material instance
  bmp:hasMaterialCategory bmp:Brick ;           # object property → category class
  bmp:hasMaterialType bmp:BrickMasonryUnit .    # object property → type class

# JSON mapping:
#   layer_topology         → layerset rdf:type topology (subClassOf bmp:LayerSet)
#   predicted_function_iri → bmp:hasLayerFunction on bmp:Layer
#   predicted_category_iri → bmp:hasMaterialCategory on bmp:Material
#   predicted_type_iri     → bmp:hasMaterialType on bmp:Material
""".strip()


def short_iri(iri: str | URIRef) -> str:
    text = str(iri)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rsplit("/", 1)[-1]


# [UNUSED] superseded by format_layer_functions / format_layerset_topologies
# def _format_class_set(uris: set[str]) -> str:
#     return ", ".join(f"bmp:{short_iri(u)}" for u in sorted(uris)) or "(none in ontology)"

#Domain Range Axioms
def format_object_property_axioms(ontology_graph: Graph) -> str:
    """Summarize composition-related object properties from OWL domain/range."""
    lines: list[str] = []
    seen: set[str] = set()
    for prop in COMPOSITION_PROPERTIES:
        key = str(prop)
        if key in seen:
            continue
        domains = [short_iri(d) for d in ontology_graph.objects(prop, RDFS.domain)]
        ranges = [short_iri(r) for r in ontology_graph.objects(prop, RDFS.range)]
        if not domains and not ranges:
            continue
        seen.add(key)
        prefix = "bmp" if key.startswith("https://w3id.org/bmp#") else "at"
        label = short_iri(prop)
        lines.append(
            f"- {prefix}:{label}"
            f"  domain: {', '.join(domains) or '—'}"
            f"  |  range: {', '.join(ranges) or '—'}"
        )
    return "\n".join(lines) if lines else "(no composition properties found in ontology)"

#Material Hierarchy Axioms
def format_material_hierarchy(ontology_graph: Graph) -> str:
    """Full category → type tree from OWL (hasMaterialCategory/Type ranges + subClassOf)."""
    tree = material_category_type_tree(ontology_graph)
    lines: list[str] = []
    for category, types in tree.items():
        cat_label = f"bmp:{short_iri(category)}"
        if not types:
            lines.append(f"- {cat_label}")
            continue
        type_labels = ", ".join(f"bmp:{short_iri(t)}" for t in types)
        lines.append(f"- {cat_label}")
        lines.append(f"    types: {type_labels}")
    return "\n".join(lines) if lines else "(no material hierarchy in ontology)"

#Layer Functions Axioms
def format_layer_functions(ontology_graph: Graph) -> str:
    from rdflib.namespace import RDFS

    lines: list[str] = []
    for uri in sorted(layer_function_uris(ontology_graph)):
        if uri == str(BMP_LAYER_FUNCTION):
            continue
        label = next(ontology_graph.objects(URIRef(uri), RDFS.label), None)
        if label:
            lines.append(f"- bmp:{short_iri(uri)} — {label}")
        else:
            lines.append(f"- bmp:{short_iri(uri)}")
    return "\n".join(lines) if lines else "(none in ontology)"

#Layer Composition Hierarchy Axioms
def format_layer_composition_hierarchy(ontology_graph: Graph) -> str:
    """Structural assembly rules — classes (rdf:type) vs object properties (links)."""
    return "\n".join(
        [
            "Classes (owl:Class — entity types assigned via rdf:type):",
            "  bmp:LayerSet + one topology subclass (Cavity, MultiLayer, SingleLayer, Frame)",
            "  bmp:Layer",
            "  bmp:LayerFunction subclass per layer (Facade, Air, LoadBearing, …)",
            "",
            "Object properties (owl:ObjectProperty — instance links):",
            f"  at:hasLayerSet   — Element → LayerSet",
            f"  bmp:hasLayer     — LayerSet → Layer",
            f"  bmp:hasLayerFunction — Layer → LayerFunction class",
            "",
            "Valid LayerSet topologies (rdfs:subClassOf bmp:LayerSet):",
            format_layerset_topologies(ontology_graph),
            "",
            "Valid layer functions (rdfs:subClassOf bmp:LayerFunction):",
            format_layer_functions(ontology_graph),
            "",
            "Optional per layer — Material (when text names a material):",
            "  at:hasMaterial / bmp:hasMaterial — Layer → Material",
            "  bmp:hasMaterialCategory — Material → category anchor",
            "  bmp:hasMaterialType — Material → type subClassOf category",
        ]
    )

#Layer Set Topologies Axioms
def format_layerset_topologies(ontology_graph: Graph) -> str:
    from rdflib.namespace import RDFS

    lines: list[str] = []
    for uri in sorted(layerset_topology_uris(ontology_graph)):
        label = next(ontology_graph.objects(URIRef(uri), RDFS.label), None)
        if label:
            lines.append(f"- bmp:{short_iri(uri)} — {label}")
        else:
            lines.append(f"- bmp:{short_iri(uri)}")
    return "\n".join(lines) if lines else "(none in ontology)"


def build_owl_axiom_structure(ontology_graph: Graph, *, profile: str = "bbsr") -> str:
    """OWL axiom block for the user prompt — entirely derived from the loaded graph."""
    cfg = get_profile(profile)
    sections = [
        "### Object properties (from OWL)",
        format_object_property_axioms(ontology_graph),
    ]
    if cfg.get("llm_mode") != "full":
        sections.extend(
            [
                "",
                "### Material hierarchy (bmp:hasMaterialCategory → bmp:hasMaterialType via rdfs:subClassOf)",
                format_material_hierarchy(ontology_graph),
            ]
        )
    if cfg.get("llm_mode") == "full":
        sections.extend(
            [
                "",
                "### Layer / LayerSet composition (structural — from OWL)",
                format_layer_composition_hierarchy(ontology_graph),
                "",
                "### Material hierarchy (bmp:hasMaterialCategory → bmp:hasMaterialType via rdfs:subClassOf)",
                format_material_hierarchy(ontology_graph),
            ]
        )
    return "\n".join(sections)


def format_retrieval_matches(matches, ontology_corpus=None) -> str:
    """Cosine retrieval hits with scores and corpus metadata."""
    if not matches:
        return "(no retrieval matches above threshold)"
    corpus_by_uri = {item["entity_uri"]: item for item in (ontology_corpus or [])}
    lines: list[str] = []
    for match in matches:
        uri = match["entity_uri"]
        branch = match.get("taxonomy_branch") or "Retrieved"
        score = match.get("score")
        score_txt = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        parent = match.get("parent_category")
        parent_txt = f" parent={short_iri(parent)}" if parent else ""
        meta = corpus_by_uri.get(uri, {}).get("text", short_iri(uri))
        lines.append(f"- {uri} [{branch}{score_txt}{parent_txt}]")
        lines.append(f"    {meta}")
    return "\n".join(lines)


@dataclass
class ScopedVocabulary:
    """OWL-derived schema + retrieval-scoped IRIs for one description."""

    axiom_structure: str
    retrieval_matches: str
    scoped_material: str = ""
    # [UNUSED] never read; the real post-LLM whitelist is validators.expand_allowed_iris
    # all_iris: set[str] = field(default_factory=set)

    def prompt_block(self) -> str:
        parts = [
            "## OWL composition axioms (read from ontology)",
            self.axiom_structure,
            "",
            "## Cosine retrieval matches (text-relevant entities)",
            self.retrieval_matches,
        ]
        if self.scoped_material.strip():
            parts.extend(["", "## Material scope (retrieval × OWL hierarchy)", self.scoped_material])
        return "\n".join(parts)


def _scoped_categories_from_matches(matches, ontology_graph: Graph) -> set[str]:
    categories: set[str] = set()
    anchors = material_category_uris(ontology_graph)
    for match in matches:
        uri = match["entity_uri"]
        if is_material_category_iri(ontology_graph, uri):
            categories.add(uri)
        elif is_material_type_iri(ontology_graph, uri):
            anchor = anchor_for_material_iri(ontology_graph, uri, anchors)
            if anchor:
                categories.add(anchor)
    return categories


def format_scoped_material_tree(ontology_graph: Graph, categories: set[str]) -> str:
    """Retrieval-scoped slice of the OWL material hierarchy."""
    if not categories:
        return ""
    tree = material_category_type_tree(ontology_graph)
    lines: list[str] = []
    for category in sorted(categories):
        types = tree.get(category, [])
        lines.append(f"- bmp:{short_iri(category)}")
        if types:
            lines.append(f"    types: {', '.join(f'bmp:{short_iri(t)}' for t in types)}")
    return "\n".join(lines)


def build_scoped_vocabulary(
    matches,
    *,
    ontology_graph: Graph,
    ontology_corpus=None,
    profile: str = "bbsr",
) -> ScopedVocabulary:
    """Combine OWL axiom structure with cosine retrieval for the LLM user prompt."""
    axiom_structure = build_owl_axiom_structure(ontology_graph, profile=profile)
    retrieval_section = format_retrieval_matches(matches, ontology_corpus)
    scoped_categories = _scoped_categories_from_matches(matches, ontology_graph)

    # [UNUSED] all_iris was computed here per description and never read by any caller.
    # Removing the walk also drops a material_category_type_tree() pass per description.
    # all_iris: set[str] = {m["entity_uri"] for m in matches}
    # tree = material_category_type_tree(ontology_graph)
    # for category in scoped_categories:
    #     all_iris.add(category)
    #     all_iris.update(tree.get(category, []))
    #
    # cfg = get_profile(profile)
    # if cfg.get("llm_mode") == "full":
    #     all_iris |= layer_function_uris(ontology_graph)
    #     all_iris |= layerset_topology_uris(ontology_graph)
    # else:
    #     for match in matches:
    #         uri = match["entity_uri"]
    #         if is_material_category_iri(ontology_graph, uri) or is_material_type_iri(
    #             ontology_graph, uri
    #         ):
    #             all_iris.add(uri)
    #             anchor = anchor_for_material_iri(ontology_graph, uri)
    #             if anchor:
    #                 all_iris.add(anchor)

    return ScopedVocabulary(
        axiom_structure=axiom_structure,
        retrieval_matches=retrieval_section,
        scoped_material=format_scoped_material_tree(ontology_graph, scoped_categories),
    )


def composition_allowed_iris(ontology_graph: Graph, profile: str = "bbsr") -> set[str]:
    """All composition class IRIs from OWL — used for post-LLM whitelist in full mode."""
    cfg = get_profile(profile)
    allowed = set(material_category_uris(ontology_graph))
    allowed |= material_type_uris(ontology_graph)
    if cfg.get("llm_mode") == "full":
        allowed |= layer_function_uris(ontology_graph)
        allowed |= layerset_topology_uris(ontology_graph)
    return allowed


# [UNUSED] duplicate of llm_mapper.build_ontology_context_string, which is the one imported
# def build_ontology_context_string(
#     matches,
#     *,
#     ontology_graph=None,
#     ontology_corpus=None,
#     profile: str = "bbsr",
# ) -> str:
#     """Backward-compatible alias: full user vocabulary block."""
#     if ontology_graph is None:
#         return format_retrieval_matches(matches, ontology_corpus)
#     vocab = build_scoped_vocabulary(
#         matches,
#         ontology_graph=ontology_graph,
#         ontology_corpus=ontology_corpus,
#         profile=profile,
#     )
#     return vocab.prompt_block()
