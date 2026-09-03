"""Resolve bmp material category/type pairs against the OWL hierarchy.

Material instances use hasMaterialCategory (anchor) + hasMaterialType (subClassOf).
Layer / LayerSet rules live in layer_axioms.py.
"""

from __future__ import annotations

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

from .layer_axioms import is_layer_function_iri
from .ontology_utils import is_subclass_of

BMP_NS = "https://w3id.org/bmp#"
AT_NS = "https://w3id.org/at#"

# ---------------------------------------------------------------------------
# owl:Class — material entity type
# ---------------------------------------------------------------------------
BMP_MATERIAL = URIRef(f"{BMP_NS}Material")

# ---------------------------------------------------------------------------
# owl:ObjectProperty — material links on Material instances
# ---------------------------------------------------------------------------
AT_HAS_MATERIAL = URIRef(f"{AT_NS}hasMaterial")  # Layer → Material (at: domain/range)
BMP_HAS_MATERIAL = URIRef(f"{BMP_NS}hasMaterial")  # used in emitted RDF graphs
BMP_HAS_MATERIAL_CATEGORY = URIRef(f"{BMP_NS}hasMaterialCategory")  # Material → category anchor
BMP_HAS_MATERIAL_TYPE = URIRef(f"{BMP_NS}hasMaterialType")  # Material → type subclass
BMP_MATERIAL_CATEGORY = URIRef(f"{BMP_NS}MaterialCategory")
BMP_MATERIAL_TYPE = URIRef(f"{BMP_NS}MaterialType")

# [UNUSED] documentation-only constant
# MATERIAL_ENTITY_CLASSES: tuple[URIRef, ...] = (BMP_MATERIAL,)

MATERIAL_COMPOSITION_PROPERTIES: tuple[URIRef, ...] = (
    AT_HAS_MATERIAL,
    BMP_HAS_MATERIAL,
    BMP_HAS_MATERIAL_CATEGORY,
    BMP_HAS_MATERIAL_TYPE,
)


def _typed_iris(ontology_graph: Graph, marker: URIRef) -> set[str]:
    """IRIs punned as individuals of `marker`, excluding the abstract class itself."""
    return {
        str(uri)
        for uri in ontology_graph.subjects(RDF.type, marker)
        if isinstance(uri, URIRef) and uri != marker
    }


def material_category_uris(ontology_graph: Graph) -> set[str]:
    """Fillers for hasMaterialCategory.

    Prefers OWL 2 DL punning (`rdf:type bmp:MaterialCategory` on the class IRI).
    Falls back to direct `rdfs:subClassOf bmp:MaterialCategory` so the tree still
    works before punning is authored. The abstract anchor is never a filler.
    """
    typed = _typed_iris(ontology_graph, BMP_MATERIAL_CATEGORY)
    if typed:
        return typed
    return {
        str(uri)
        for uri in ontology_graph.subjects(RDFS.subClassOf, BMP_MATERIAL_CATEGORY)
        if isinstance(uri, URIRef) and uri != BMP_MATERIAL_CATEGORY
    }


def material_type_uris(ontology_graph: Graph) -> set[str]:
    """Fillers for hasMaterialType.

    Prefers punning (`rdf:type bmp:MaterialType`). Falls back to descendants of
    category anchors that are not themselves categories.
    """
    typed = _typed_iris(ontology_graph, BMP_MATERIAL_TYPE)
    if typed:
        return typed
    cats = material_category_uris(ontology_graph)
    found: set[str] = set()
    stack = [URIRef(cat) for cat in cats]
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        key = str(node)
        if key in visited:
            continue
        visited.add(key)
        for child in ontology_graph.subjects(RDFS.subClassOf, node):
            if not isinstance(child, URIRef):
                continue
            child_key = str(child)
            if child_key not in cats:
                found.add(child_key)
            stack.append(child)
    return found


def material_types_under_category(ontology_graph: Graph, category_iri: str) -> list[str]:
    """Direct and indirect material types under a category anchor (OWL subClassOf)."""
    category_ref = URIRef(str(category_iri))
    type_uris = material_type_uris(ontology_graph)
    found: list[str] = []
    stack = [category_ref]
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        key = str(node)
        if key in visited:
            continue
        visited.add(key)
        if key != str(category_iri) and key in type_uris:
            found.append(key)
        for child in ontology_graph.subjects(RDFS.subClassOf, node):
            if isinstance(child, URIRef):
                stack.append(child)
    return sorted(found)


def material_category_type_tree(ontology_graph: Graph) -> dict[str, list[str]]:
    """Category anchor → material types, derived from OWL ranges + subClassOf."""
    tree: dict[str, list[str]] = {}
    for category in sorted(material_category_uris(ontology_graph)):
        types = material_types_under_category(ontology_graph, category)
        if category in material_type_uris(ontology_graph):
            types = sorted(set(types) | {category})
        tree[category] = types
    return tree


# Boolean material category IRIs
def is_material_category_iri(ontology_graph: Graph, iri) -> bool:
    return bool(iri) and str(iri) in material_category_uris(ontology_graph)


# Boolean material type IRIs
def is_material_type_iri(ontology_graph: Graph, iri) -> bool:
    if not iri:
        return False
    iri_str = str(iri)
    if iri_str in material_type_uris(ontology_graph):
        return True
    return anchor_for_material_iri(ontology_graph, iri) is not None


def anchor_for_material_iri(ontology_graph: Graph, iri, anchors: set[str] | None = None) -> str | None:
    if not iri:
        return None
    if is_layer_function_iri(ontology_graph, iri):
        return None
    anchors = anchors or material_category_uris(ontology_graph)
    iri_str = str(iri)
    if iri_str in anchors:
        return iri_str

    visited: set[str] = set()
    stack = [URIRef(iri_str)]
    while stack:
        node = stack.pop()
        node_key = str(node)
        if node_key in visited:
            continue
        visited.add(node_key)
        if node_key in anchors:
            return node_key
        for parent in ontology_graph.objects(node, RDFS.subClassOf):
            if isinstance(parent, URIRef) and not is_layer_function_iri(ontology_graph, parent):
                stack.append(parent)
    return None


def default_type_for_category(ontology_graph: Graph, category_iri, type_uris: set[str] | None = None) -> URIRef:
    category_ref = URIRef(str(category_iri))
    type_uris = type_uris or material_type_uris(ontology_graph)
    category_str = str(category_ref)
    if category_str in type_uris:
        return category_ref
    for child in ontology_graph.subjects(RDFS.subClassOf, category_ref):
        if str(child) in type_uris:
            return child
    return category_ref


def resolve_category_and_type(
    ontology_graph: Graph,
    *,
    category_iri=None,
    type_iri=None,
) -> tuple[URIRef | None, URIRef | None]:
    """Return (category, type) with type subordinate to category per OWL.

    A type is kept only when it is a real material type under the category.
    Naming a category alone (\"concrete\", \"brick\") must not invent a default type.
    """
    anchors = material_category_uris(ontology_graph)
    type_uris = material_type_uris(ontology_graph)

    cat = str(category_iri) if category_iri else None
    typ = str(type_iri) if type_iri else None

    if cat and is_layer_function_iri(ontology_graph, cat):
        cat = None
    if typ and is_layer_function_iri(ontology_graph, typ):
        typ = None

    if cat and cat not in anchors:
        anchor = anchor_for_material_iri(ontology_graph, cat, anchors)
        if anchor:
            if not typ:
                typ = cat
            cat = anchor
        else:
            cat = None

    if typ and not cat:
        cat = anchor_for_material_iri(ontology_graph, typ, anchors)
    if not cat and not typ:
        return None, None

    cat = anchor_for_material_iri(ontology_graph, cat, anchors) if cat else None
    if not cat:
        return None, None

    if typ == cat:
        typ = None
    elif typ and typ in anchors and typ not in type_uris:
        typ = None
    elif typ:
        under_cat = is_subclass_of(ontology_graph, typ, cat) and typ != cat
        if not under_cat:
            anchor = anchor_for_material_iri(ontology_graph, typ, anchors)
            if anchor and typ != anchor:
                cat = anchor
            else:
                typ = None

    return URIRef(cat), URIRef(typ) if typ else None


def ensure_layer_material_pair(
    ontology_graph: Graph,
    *,
    category_iri=None,
    type_iri=None,
    **_ignored,
) -> tuple[URIRef | None, URIRef | None]:
    """Resolve material category + type from material IRIs only (never from layer function)."""
    return resolve_category_and_type(
        ontology_graph,
        category_iri=category_iri,
        type_iri=type_iri,
    )


def sanitize_layer_prediction(
    ontology_graph: Graph,
    *,
    function_iri=None,
    category_iri=None,
    type_iri=None,
) -> tuple[str | None, str | None, str | None]:
    """Keep layer function and material IRIs in their correct OWL slots."""
    fn = str(function_iri) if function_iri else None
    cat = str(category_iri) if category_iri else None
    typ = str(type_iri) if type_iri else None
    anchors = material_category_uris(ontology_graph)
    types = material_type_uris(ontology_graph)

    if cat and is_layer_function_iri(ontology_graph, cat):
        cat = None
    if typ and is_layer_function_iri(ontology_graph, typ):
        typ = None

    # A subtype (e.g. bmp:LightWeightConcrete) in the category slot is demoted to the type
    # slot and replaced by its anchor, so hasMaterialCategory only ever carries a range anchor.
    if cat and cat not in anchors:
        anchor = anchor_for_material_iri(ontology_graph, cat, anchors)
        if anchor:
            if not typ:
                typ = cat
            cat = anchor
        else:
            cat = None

    if fn and not is_layer_function_iri(ontology_graph, fn):
        material_anchor = anchor_for_material_iri(ontology_graph, fn, anchors)
        if material_anchor or fn in types or fn in anchors:
            if not typ:
                typ = fn
            if not cat:
                cat = material_anchor or anchor_for_material_iri(ontology_graph, fn, anchors) or fn
            fn = None

    if fn and is_layer_function_iri(ontology_graph, fn):
        pass
    elif fn:
        fn = None

    return fn, cat, typ
