"""Load OWL graph, build embedded retrieval corpus."""
# parse KB-LCA-merged.ttl with RDF lib,  build corpus of all classes and properties

from __future__ import annotations

import re
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from .config import (
    DEFAULT_EMBEDDER,
    DCTERMS_DESCRIPTION,
    resolve_ontology_overlay_path,
    resolve_ontology_path,
)
from .layer_axioms import (
    is_layer_function_iri,
    is_layerset_topology_iri,
    LAYER_FUNCTION_URI,
)
from .material_axioms import (
    is_material_category_iri,
    is_material_type_iri,
)

_embedder = None
_device = None


def _get_embedder():
    global _embedder, _device
    if _embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(DEFAULT_EMBEDDER, device=_device)
    return _embedder, _device


def encode_texts(texts, convert_to_tensor=True):
    model, device = _get_embedder()
    return model.encode(texts, convert_to_tensor=convert_to_tensor, device=device)


def encode_text(text, convert_to_tensor=True):
    return encode_texts([text], convert_to_tensor=convert_to_tensor)[0]


# corpus.load_ontology_graph() parses owl/KB-LCA-merged.ttl into an rdflib graph. 
# This is your symbolic side, and it is the object passed around as ontology_graph for every later validation.

def load_ontology_graph(owl_dir: Path | None = None) -> Graph:
    """Load the main KB from owl/KB-LCA-merged.ttl (+ optional ONTOLOGY_OVERLAY_PATH)."""
    main_path = resolve_ontology_path(owl_dir)
    graph = Graph()
    graph.parse(location=main_path.as_uri(), format="ttl")

    overlay_path = resolve_ontology_overlay_path(owl_dir)
    if overlay_path is not None:
        graph.parse(location=overlay_path.as_uri(), format="ttl")

    return graph


def ontology_source_label(owl_dir: Path | None = None) -> str:
    """Human-readable path of the loaded ontology source(s)."""
    main = resolve_ontology_path(owl_dir)
    overlay = resolve_ontology_overlay_path(owl_dir)
    if overlay is not None:
        return f"{main} + overlay {overlay}"
    return str(main)



def _class_text_profile(ontology_graph, entity) -> dict:
    entity_uri = str(entity)
    local_name = entity_uri.split("#")[-1].split("/")[-1]
    pretty_name = re.sub(r"(?<!^)(?=[A-Z])", " ", local_name)
    name_parts = [local_name, pretty_name]
    rdfs_labels: list[str] = []
    comment_parts: list[str] = []

    for label in ontology_graph.objects(entity, RDFS.label):
        lang = getattr(label, "language", None)
        if lang in (None, "en", "de"):
            rdfs_labels.append(str(label))

    for comment in ontology_graph.objects(entity, RDFS.comment):
        lang = getattr(comment, "language", None)
        if lang in (None, "en", "de"):
            comment_parts.append(str(comment))

    for desc in ontology_graph.objects(entity, DCTERMS_DESCRIPTION):
        lang = getattr(desc, "language", None)
        if lang in (None, "en", "de"):
            comment_parts.append(str(desc))

    name_text = " | ".join(dict.fromkeys(part for part in name_parts if part))
    label_text = " | ".join(dict.fromkeys([name_text, *rdfs_labels] if rdfs_labels else [name_text]))
    full_parts = [label_text, *comment_parts]
    uri = str(entity)
    return {
        "entity_uri": uri,
        "name_text": name_text,
        "label_text": label_text,
        "text": " | ".join(part for part in full_parts if part),
        "superclasses": [str(p) for p in ontology_graph.objects(entity, RDFS.subClassOf)],
        "subclasses": [str(c) for c in ontology_graph.subjects(RDFS.subClassOf, entity)],
        "is_layer_function": is_layer_function_iri(ontology_graph, uri),
        "is_layerset_topology": is_layerset_topology_iri(ontology_graph, uri),
        "is_material_category": is_material_category_iri(ontology_graph, uri),
        "is_material_type": is_material_type_iri(ontology_graph, uri),
        "embedding": None,
        "label_embedding": None,
        "name_embedding": None,
    }


# corpus.load_ontology_corpus() walks every owl:Class, builds a text profile for each (local name, rdfs:label, rdfs:comment, dcterms:description), 
# tags it with is_layer_function / is_material_category / is_material_type, and embeds all those texts with 
# SentenceTransformer. This is your neural side. It is expensive and happens once per run.

def load_ontology_corpus(ontology_graph: Graph | None = None, ontology_path: Path | str | None = None):
    if ontology_graph is None:
        if ontology_path is None:
            ontology_graph = load_ontology_graph()
        else:
            ontology_graph = Graph()
            ontology_graph.parse(location=Path(ontology_path).resolve().as_uri(), format="ttl")

    entities = set(ontology_graph.subjects(RDF.type, OWL.Class)) | set(
        ontology_graph.subjects(RDF.type, RDF.Property)
    )
    corpus = [_class_text_profile(ontology_graph, entity) for entity in entities]
    if corpus:
        name_texts = [item["name_text"] for item in corpus]
        label_texts = [item["label_text"] for item in corpus]
        full_texts = [item["text"] for item in corpus]
        embeddings = encode_texts(name_texts + label_texts + full_texts)
        n = len(corpus)
        for idx in range(n):
            corpus[idx]["name_embedding"] = embeddings[idx]
            corpus[idx]["label_embedding"] = embeddings[n + idx]
            corpus[idx]["embedding"] = embeddings[2 * n + idx]
    return corpus
