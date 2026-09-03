"""Shared OWL + LLM pipeline for description → layer structure."""

from .config import DEFAULT_ONTOLOGY_PATH, resolve_ontology_path
from .corpus import load_ontology_corpus, load_ontology_graph, ontology_source_label
from .pipeline import OntologyNLPContext, process_description, run_owl_nlp_pipeline

__all__ = [
    "DEFAULT_ONTOLOGY_PATH",
    "OntologyNLPContext",
    "load_ontology_corpus",
    "load_ontology_graph",
    "ontology_source_label",
    "process_description",
    "resolve_ontology_path",
    "run_owl_nlp_pipeline",
]
