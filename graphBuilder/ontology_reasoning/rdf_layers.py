"""Convert shared NLP layer JSON into builder-specific layer dicts and RDF triples."""

from __future__ import annotations

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from .layer_axioms import (
    BMP_HAS_LAYER,
    BMP_HAS_LAYER_FUNCTION,
    BMP_LAYER,
    BMP_MULTI_LAYER,
    BMP_SINGLE_LAYER,
    default_layer_function,
    resolve_layer_function,
)
from .material_axioms import (
    BMP_HAS_MATERIAL,
    BMP_HAS_MATERIAL_CATEGORY,
    BMP_HAS_MATERIAL_TYPE,
    BMP_MATERIAL,
    ensure_layer_material_pair,
    resolve_category_and_type,
    sanitize_layer_prediction,
)

def emit_material_instance_triples(
    graph: Graph,
    *,
    layer_uri,
    material_uri,
    category_iri,
    type_iri,
    ontology_graph: Graph | None = None,
):
    """
    Canonical pattern:
      Layer --bmp:hasMaterial--> Material
      Material --bmp:hasMaterialCategory--> anchor category
      Material --bmp:hasMaterialType--> subordinate type
    """
    if ontology_graph is not None:
        category_iri, type_iri = resolve_category_and_type(
            ontology_graph,
            category_iri=category_iri,
            type_iri=type_iri,
        )
    if category_iri is None:
        return
    graph.add((layer_uri, BMP_HAS_MATERIAL, material_uri))
    graph.add((material_uri, RDF.type, BMP_MATERIAL))
    graph.add((material_uri, BMP_HAS_MATERIAL_CATEGORY, URIRef(str(category_iri))))
    if type_iri is not None and str(type_iri) != str(category_iri):
        graph.add((material_uri, BMP_HAS_MATERIAL_TYPE, URIRef(str(type_iri))))


def emit_enforced_layer(
    graph: Graph,
    *,
    layer_uri,
    material_uri,
    function_iri,
    category_iri=None,
    type_iri=None,
    ontology_graph: Graph,
    layer_types: tuple | None = None,
):
    """
    Mandatory RDF bundle for one layer inside a LayerSet:
      Layer (bmp:Layer) + bmp:hasLayerFunction
      Layer + bmp:hasMaterial -> Material (optional)
      Material + bmp:hasMaterialCategory + bmp:hasMaterialType
    """
    function = URIRef(str(function_iri)) if function_iri else default_layer_function(ontology_graph)
    fn, cat_iri, typ_iri = sanitize_layer_prediction(
        ontology_graph,
        function_iri=function,
        category_iri=category_iri,
        type_iri=type_iri,
    )
    # Only `fn` may be re-resolved: sanitize_layer_prediction drops IRIs that belong in the
    # material slots, and falling back to the raw `function` would re-admit them.
    resolved_fn = resolve_layer_function(ontology_graph, fn)
    function = resolved_fn if resolved_fn is not None else default_layer_function(ontology_graph)

    for rdf_type in layer_types or (BMP_LAYER,):
        graph.add((layer_uri, RDF.type, rdf_type))
    graph.add((layer_uri, BMP_HAS_LAYER_FUNCTION, function))

    cat, typ = ensure_layer_material_pair(
        ontology_graph,
        category_iri=cat_iri,
        type_iri=typ_iri,
    )
    if cat is None:
        return

    emit_material_instance_triples(
        graph,
        layer_uri=layer_uri,
        material_uri=material_uri,
        category_iri=cat,
        type_iri=typ,
        ontology_graph=None,
    )


def material_pair_from_layer_dict(layer: dict, ontology_graph: Graph | None = None):
    category = layer.get("material_category_iri") or layer.get("predicted_category_iri")
    typ = layer.get("material_type_iri") or layer.get("predicted_type_iri")
    if not category and layer.get("material_iri") is not None:
        category = layer["material_iri"]
    if ontology_graph is not None:
        return ensure_layer_material_pair(
            ontology_graph,
            category_iri=category,
            type_iri=typ,
        )
    if category:
        return URIRef(str(category)), URIRef(str(typ)) if typ else None
    return None, None


def nlp_layer_json_to_tabula_layerset(layer_json: dict | None, *, default_topology=None, ontology_graph=None):
    if not layer_json or not layer_json.get("layers"):
        return None

    default_fn = default_layer_function(ontology_graph) if ontology_graph is not None else BMP_LAYER

    layers = []
    for layer in layer_json["layers"]:
        function_iri = layer.get("predicted_function_iri")
        if function_iri and ontology_graph is not None:
            resolved = resolve_layer_function(ontology_graph, function_iri)
            function_iri = resolved if resolved is not None else None
        layer_out = {
            "layer_index": layer.get("layer_index", len(layers) + 1),
            "function_iri": URIRef(str(function_iri)) if function_iri else default_fn,
            "thickness_cm": layer.get("thickness_cm"),
        }
        if ontology_graph is not None:
            cat, typ = material_pair_from_layer_dict(
                {
                    "predicted_category_iri": layer.get("predicted_category_iri"),
                    "predicted_type_iri": layer.get("predicted_type_iri"),
                },
                ontology_graph,
            )
            layer_out["material_category_iri"] = cat
            layer_out["material_type_iri"] = typ
        else:
            category_iri, type_iri = material_pair_from_layer_dict(layer, ontology_graph)
            if category_iri is not None:
                layer_out["material_category_iri"] = category_iri
                layer_out["material_type_iri"] = type_iri
        layers.append(layer_out)

    topology_raw = layer_json.get("layer_topology")
    topology = URIRef(topology_raw) if topology_raw else (default_topology or BMP_SINGLE_LAYER)
    if len(layers) > 1 and topology == BMP_SINGLE_LAYER:
        topology = BMP_MULTI_LAYER

    return {"topology": topology, "layers": layers}


# [UNUSED] only reachable from the material_pair_from_nlp tail below, which never runs
# because every caller passes a real ontology_graph.
# def top_material_iri_from_nlp(ctx) -> URIRef | None:
#     if ctx is None:
#         return None
#     if ctx.material_type_iri:
#         return URIRef(ctx.material_type_iri)
#     if ctx.material_category_iri:
#         return URIRef(ctx.material_category_iri)
#     if ctx.layer_json and isinstance(ctx.layer_json, dict):
#         for layer in ctx.layer_json.get("layers", []):
#             mat = layer.get("predicted_type_iri") or layer.get("predicted_category_iri")
#             if mat:
#                 return URIRef(mat)
#     for match in getattr(ctx, "matches", []):
#         if match.get("taxonomy_branch") == "Material":
#             return URIRef(match["entity_uri"])
#     return None


def material_pair_from_nlp(ctx, ontology_graph: Graph | None = None) -> tuple[URIRef | None, URIRef | None]:
    if ctx is None:
        return None, None
    category = getattr(ctx, "material_category_iri", None)
    typ = getattr(ctx, "material_type_iri", None)
    if ontology_graph is not None:
        return ensure_layer_material_pair(
            ontology_graph,
            category_iri=category,
            type_iri=typ,
        )
    if category or typ:
        cat = URIRef(category) if category else URIRef(typ)
        return cat, URIRef(typ) if typ else cat
    # [UNUSED] unreachable: the ontology_graph branch above always returns first
    # single = top_material_iri_from_nlp(ctx)
    # if single is not None and ontology_graph is not None:
    #     return resolve_category_and_type(ontology_graph, type_iri=single)
    # return (single, single) if single is not None else (None, None)
    return None, None
