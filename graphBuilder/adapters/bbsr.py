"""Yield BBSR element descriptions from an initialized RDF graph."""

from __future__ import annotations

from rdflib import URIRef
from rdflib.namespace import RDF

from . import DescriptionRecord
from ontology_reasoning.chunking import decompose_description

AT_HAS_DESCRIPTION = URIRef("https://w3id.org/at#hasDescription")
AT_HAS_ARCHETYPE_DESCRIPTION = URIRef("https://w3id.org/at#hasArchetypeDescription")
BEO_TYPES = (
    URIRef("https://w3id.org/beo#Wall"),
    URIRef("https://w3id.org/beo#Slab"),
    URIRef("https://w3id.org/beo#Floor"),
    URIRef("https://w3id.org/beo#Roof"),
    URIRef("https://w3id.org/beo#WallPARTITIONING"),
)


def iter_bbsr_descriptions(graph) -> list[DescriptionRecord]:
    records: list[DescriptionRecord] = []
    seen: set[str] = set()

    for beo_type in BEO_TYPES:
        for subject in graph.subjects(RDF.type, beo_type):
            uri_str = str(subject)
            if uri_str in seen:
                continue
            seen.add(uri_str)

            de_descs = [
                str(d)
                for pred in (AT_HAS_ARCHETYPE_DESCRIPTION, AT_HAS_DESCRIPTION)
                for d in graph.objects(subject, pred)
                if getattr(d, "language", None) == "de"
            ]
            en_descs = [
                str(d)
                for pred in (AT_HAS_ARCHETYPE_DESCRIPTION, AT_HAS_DESCRIPTION)
                for d in graph.objects(subject, pred)
                if getattr(d, "language", None) == "en"
            ]
            if not de_descs and not en_descs:
                continue

            german = de_descs[0] if de_descs else ""
            english = en_descs[0] if en_descs else ""
            chunks = decompose_description(german, english, profile="bbsr")

            records.append(
                DescriptionRecord(
                    subject_uri=uri_str,
                    german_desc=german,
                    english_desc=english,
                    extra_text=" | ".join(chunks) if chunks else None,
                    metadata={"profile": "bbsr", "element_type_iri": str(beo_type)},
                )
            )
    return records
