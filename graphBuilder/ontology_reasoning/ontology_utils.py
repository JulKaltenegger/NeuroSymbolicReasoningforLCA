"""Shared OWL graph helpers."""

from __future__ import annotations

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS


def is_subclass_of(ontology_graph: Graph, child, parent) -> bool:
    child_ref = URIRef(str(child))
    parent_ref = URIRef(str(parent))
    if child_ref == parent_ref:
        return True
    visited: set[str] = set()
    stack = [child_ref]
    while stack:
        node = stack.pop()
        node_key = str(node)
        if node_key in visited:
            continue
        visited.add(node_key)
        for ancestor in ontology_graph.objects(node, RDFS.subClassOf):
            if ancestor == parent_ref:
                return True
            if isinstance(ancestor, URIRef):
                stack.append(ancestor)
    return False
