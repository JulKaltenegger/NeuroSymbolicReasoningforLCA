"""LayerSet topology and LayerFunction resolution from OWL.

Composition chain (classes vs object properties):

  Element --[at:hasLayerSet]--> LayerSet instance
    LayerSet rdf:type bmp:LayerSet + one topology class (Cavity, MultiLayer, …)
    LayerSet --[bmp:hasLayer]--> Layer instance(s)
      Layer rdf:type bmp:Layer
      Layer --[bmp:hasLayerFunction]--> LayerFunction class (Facade, Air, …)

owl:Class constants name entities; owl:ObjectProperty constants name links.
Runtime validation still reads subclass trees from the loaded OWL graph.
"""

from __future__ import annotations

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

from .ontology_utils import is_subclass_of

BMP_NS = "https://w3id.org/bmp#"
AT_NS = "https://w3id.org/at#"

# ---------------------------------------------------------------------------
# owl:Class — structural entity types (instances in RDF data)
# ---------------------------------------------------------------------------
BMP_LAYER = URIRef(f"{BMP_NS}Layer")
BMP_LAYER_SET = URIRef(f"{BMP_NS}LayerSet")
BMP_LAYER_FUNCTION = URIRef(f"{BMP_NS}LayerFunction")

# String aliases (back-compat for retrieval / corpus string comparisons)
# [UNUSED] LAYER_SET_URI = str(BMP_LAYER_SET)
LAYER_FUNCTION_URI = str(BMP_LAYER_FUNCTION)

# LayerSet topology subclasses — rdfs:subClassOf bmp:LayerSet
BMP_CAVITY = URIRef(f"{BMP_NS}Cavity")
BMP_FRAME = URIRef(f"{BMP_NS}Frame")
BMP_MULTI_LAYER = URIRef(f"{BMP_NS}MultiLayer")
BMP_SINGLE_LAYER = URIRef(f"{BMP_NS}SingleLayer")

LAYERSET_TOPOLOGY_CLASSES: tuple[URIRef, ...] = (
    BMP_CAVITY,
    BMP_FRAME,
    BMP_MULTI_LAYER,
    BMP_SINGLE_LAYER,
)
LAYERSET_TOPOLOGY_IRIS: frozenset[str] = frozenset(str(u) for u in LAYERSET_TOPOLOGY_CLASSES)

# LayerFunction subclasses — rdfs:subClassOf bmp:LayerFunction
BMP_AIR = URIRef(f"{BMP_NS}Air")
BMP_FACADE = URIRef(f"{BMP_NS}Facade")
BMP_FINISHING = URIRef(f"{BMP_NS}Finishing")
BMP_GLAZING = URIRef(f"{BMP_NS}Glazing")
BMP_INSULATING = URIRef(f"{BMP_NS}Insulating")
BMP_LOAD_BEARING = URIRef(f"{BMP_NS}LoadBearing")
BMP_NON_LOAD_BEARING = URIRef(f"{BMP_NS}NonLoadBearing")
BMP_OPENING = URIRef(f"{BMP_NS}Opening")

LAYER_FUNCTION_CLASSES: tuple[URIRef, ...] = (
    BMP_AIR,
    BMP_FACADE,
    BMP_FINISHING,
    BMP_GLAZING,
    BMP_INSULATING,
    BMP_LOAD_BEARING,
    BMP_NON_LOAD_BEARING,
    BMP_OPENING,
)
LAYER_FUNCTION_IRIS: frozenset[str] = frozenset(str(u) for u in LAYER_FUNCTION_CLASSES)

# ---------------------------------------------------------------------------
# owl:ObjectProperty — composition links (domain → range in OWL)
# ---------------------------------------------------------------------------
AT_HAS_LAYER_SET = URIRef(f"{AT_NS}hasLayerSet")  # Element → LayerSet
BMP_HAS_LAYER = URIRef(f"{BMP_NS}hasLayer")  # LayerSet → Layer
BMP_HAS_LAYER_FUNCTION = URIRef(f"{BMP_NS}hasLayerFunction")  # Layer → LayerFunction

# [UNUSED] documentation-only constant
# LAYER_ENTITY_CLASSES: tuple[URIRef, ...] = (
#     BMP_LAYER,
#     BMP_LAYER_SET,
#     BMP_LAYER_FUNCTION,
#     *LAYERSET_TOPOLOGY_CLASSES,
#     *LAYER_FUNCTION_CLASSES,
# )

LAYER_COMPOSITION_PROPERTIES: tuple[URIRef, ...] = (
    AT_HAS_LAYER_SET,
    BMP_HAS_LAYER,
    BMP_HAS_LAYER_FUNCTION,
)


def layer_function_uris(ontology_graph: Graph) -> set[str]:
    """Proper LayerFunction subclasses from OWL.

    The rdfs:range of bmp:hasLayerFunction is bmp:LayerFunction itself. That anchor is
    abstract and must never be emitted as a filler, so only strict subclasses qualify.
    """
    uris: set[str] = set(LAYER_FUNCTION_IRIS)
    for uri in ontology_graph.objects(BMP_HAS_LAYER_FUNCTION, RDFS.range):
        if isinstance(uri, URIRef):
            uris.update(_strict_subclasses(ontology_graph, uri))
    uris.update(_strict_subclasses(ontology_graph, BMP_LAYER_FUNCTION))
    uris.discard(str(BMP_LAYER_FUNCTION))
    return uris


def _strict_subclasses(ontology_graph: Graph, parent: URIRef) -> set[str]:
    """Transitive rdfs:subClassOf descendants of `parent`, excluding `parent` itself."""
    found: set[str] = set()
    stack = [parent]
    while stack:
        node = stack.pop()
        for child in ontology_graph.subjects(RDFS.subClassOf, node):
            if isinstance(child, URIRef) and str(child) not in found and child != parent:
                found.add(str(child))
                stack.append(child)
    return found


def layerset_topology_uris(ontology_graph: Graph) -> set[str]:
    """LayerSet topology subclasses from OWL (rdfs:subClassOf bmp:LayerSet)."""
    uris: set[str] = set(LAYERSET_TOPOLOGY_IRIS)
    for child in ontology_graph.subjects(RDFS.subClassOf, BMP_LAYER_SET):
        if isinstance(child, URIRef):
            uris.add(str(child))
    return uris


# Boolean layer function IRIs
def is_layer_function_iri(ontology_graph: Graph, iri) -> bool:
    if not iri:
        return False
    iri_str = str(iri)
    if iri_str == str(BMP_LAYER_FUNCTION):
        return False
    if iri_str in LAYER_FUNCTION_IRIS:
        return True
    return is_subclass_of(ontology_graph, iri_str, BMP_LAYER_FUNCTION)


# Boolean layer set IRIs
def is_layerset_topology_iri(ontology_graph: Graph, iri) -> bool:
    if not iri:
        return False
    iri_str = str(iri)
    if iri_str in LAYERSET_TOPOLOGY_IRIS:
        return True
    return is_subclass_of(ontology_graph, iri_str, BMP_LAYER_SET) and iri_str != str(BMP_LAYER_SET)


def default_layerset_topology(ontology_graph: Graph) -> str:
    topologies = layerset_topology_uris(ontology_graph)
    preferred = str(BMP_SINGLE_LAYER)
    if preferred in topologies:
        return preferred
    return sorted(topologies)[0] if topologies else preferred


def resolve_layerset_topology(ontology_graph: Graph, topology_iri=None) -> URIRef | None:
    """Return a valid LayerSet topology subclass, or None if the IRI is illegal."""
    from .material_axioms import is_material_category_iri, is_material_type_iri

    if not topology_iri:
        return URIRef(default_layerset_topology(ontology_graph))

    topo = str(topology_iri).strip()
    if is_layerset_topology_iri(ontology_graph, topo):
        return URIRef(topo)

    if is_layer_function_iri(ontology_graph, topo):
        return None
    if is_material_category_iri(ontology_graph, topo) or is_material_type_iri(ontology_graph, topo):
        return None

    if is_subclass_of(ontology_graph, topo, BMP_LAYER_SET) and topo != str(BMP_LAYER_SET):
        return URIRef(topo)

    return None


def resolve_layer_function(ontology_graph: Graph, function_iri=None) -> URIRef | None:
    """Return a valid LayerFunction subclass for bmp:hasLayerFunction."""
    from .material_axioms import is_material_category_iri, is_material_type_iri

    if not function_iri:
        return None

    fn = str(function_iri).strip()
    if is_layer_function_iri(ontology_graph, fn):
        return URIRef(fn)

    if is_layerset_topology_iri(ontology_graph, fn):
        return None
    if is_material_category_iri(ontology_graph, fn) or is_material_type_iri(ontology_graph, fn):
        return None

    return None


def default_layer_function(ontology_graph: Graph) -> URIRef:
    for candidate in (BMP_LOAD_BEARING, *LAYER_FUNCTION_CLASSES):
        resolved = resolve_layer_function(ontology_graph, candidate)
        if resolved is not None:
            return resolved
    functions = layer_function_uris(ontology_graph)
    return URIRef(sorted(functions)[0]) if functions else BMP_LOAD_BEARING


# [UNUSED] validators.validate_layer_json calls resolve_layerset_topology directly
# def sanitize_layerset_topology(ontology_graph: Graph, topology_iri=None) -> str | None:
#     resolved = resolve_layerset_topology(ontology_graph, topology_iri)
#     return str(resolved) if resolved is not None else None


def sanitize_layer_function_iri(ontology_graph: Graph, function_iri=None) -> str | None:
    resolved = resolve_layer_function(ontology_graph, function_iri)
    return str(resolved) if resolved is not None else None
