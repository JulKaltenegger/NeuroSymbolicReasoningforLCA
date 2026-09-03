#!/usr/bin/env python3
"""
Run ad-hoc construction descriptions through the shared ontology_reasoning pipeline.

Edit EXAMPLE_CASES below with your own engineer text, then:

    python graphBuilder/run_pipeline.py                 # all gold examples (default)
    python graphBuilder/sources/run_text_examples.py --id cavity_wall
    python graphBuilder/sources/run_text_examples.py --list
    python graphBuilder/sources/run_text_examples.py --no-ttl

Outputs TTL under ttl/text_examples/<id>.ttl and JSON under ttl/text_examples/<id>.json.
Compares that TTL to ttl/text_examples/gold/<id>.ttl by composition shape.
Renovation cases (existing / usual / advanced) emit three linked element states
that reuse the existing layer URIs, then compare against gold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef, XSD
from rdflib.namespace import RDF, RDFS

GRAPH_BUILDER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GRAPH_BUILDER_DIR.parent
if str(GRAPH_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_BUILDER_DIR))

from ontology_reasoning import load_ontology_corpus, load_ontology_graph, ontology_source_label, process_description
from ontology_reasoning.config import get_llm_config
from ontology_reasoning.gold_eval import compare_ttl_to_gold
from ontology_reasoning.llm_backends import is_llm_available, log_llm_provider_once
from ontology_reasoning.rdf_layers import emit_enforced_layer
from ontology_reasoning.validation_report import begin_report, finalize_report, record_ctx
from ontology_reasoning.validators import extract_thickness_cm

# ---------------------------------------------------------------------------
# Add your own test descriptions here
# ---------------------------------------------------------------------------
EXAMPLE_CASES = [
    {
        "id": "cavity_wall",
        "label": "Cavity wall with air gap, functions only (EN)",
        "element_uri": "https://namedgraphs.org/test#wall_cavity_001",
        "rdf_type": "https://w3id.org/beo#Wall",
        "german": "",
        "english": (
            "The wall element is overall 25 cm thick, with a cavity wall, where the "
            "in-between layer is an air gap of 5 cm, the outer layer is 5 cm facade, "
            "and inner layer is 15 cm load bearing."
        ),
    },
    {
        "id": "multi_layered_wall",
        "label": "Multi-layered wall with air gap, functions only (EN)",
        "element_uri": "https://namedgraphs.org/test#wall_multilayer_001",
        "rdf_type": "https://w3id.org/beo#Wall",
        "german": "",
        "english": (
            "The wall element is overall 25 cm thick, with a multi-layered wall, where "
            "the in-between layer is an air gap of 5 cm, the outer layer is 5 cm facade, "
            "and inner layer is 15 cm load bearing."
        ),
    },
    {
        "id": "concrete_floor",
        "label": "Hollow concrete block floor (EN)",
        "element_uri": "https://namedgraphs.org/test#floor_01",
        "rdf_type": "https://w3id.org/beo#Floor",
        "german": "",
        "english": (
            "The floor element is overall 30 cm thick, made out of hollow concrete blocks."
        ),
    },
    {
        "id": "cavity_wall_materials",
        "label": "Cavity wall with air gap + brick and concrete (EN)",
        "element_uri": "https://namedgraphs.org/test#wall_cavity_002",
        "rdf_type": "https://w3id.org/beo#Wall",
        "german": "",
        "english": (
            "The wall element is overall 25 cm thick, with a cavity wall, where the "
            "in-between layer is an air gap of 5 cm, the outer layer is 5 cm facade "
            "made out of bricks, and inner layer is 15 cm load bearing concrete panels."
        ),
    },
    {
        "id": "material_alternatives",
        "label": "BBSR-style material alternatives (DE+EN)",
        "element_uri": "https://namedgraphs.org/test#wall_alternatives_001",
        "rdf_type": "https://w3id.org/beo#Wall",
        "german": (
            "Leichtbeton- oder Hohllochziegelelemente, inschichtig, 29 cm dick; "
            "beiderseitig geputzt"
        ),
        "english": (
            "Lightweight concrete or hollow-hole brick elements, single-layer, "
            "29 cm thick; cleaned on both sides"
        ),
    },
    {
        "id": "tabula_floor_renovation",
        "label": "TABULA floor: existing + usual/advanced insulation add-on (EN)",
        "element_uri": "https://namedgraphs.org/test#floor_renovation_001",
        "rdf_type": "https://w3id.org/beo#Slab",
        "element_label": "Floor 1",
        "gold": "ttl/text_examples/gold/tabula_floor_renovation.ttl",
        "german": "",
        "english": (
            "Existing: concrete ceiling with 1 cm insulation. "
            "Usual: add 8 cm insulation below / alternatively: on top of ceiling "
            "(in case of floor renovation). "
            "Advanced: add 12 cm insulation below / alternatively: on top of ceiling "
            "/ combination of both."
        ),
        "renovation_states": [
            {
                "id": "existing",
                "status_iri": "https://w3id.org/at#NoRenovation",
                "english": "concrete ceiling with 1 cm insulation",
            },
            {
                "id": "usual",
                "status_iri": "https://w3id.org/at#MinorRenovation",
                "english": (
                    "add 8 cm insulation below / alternatively: on top of ceiling "
                    "(in case of floor renovation)"
                ),
            },
            {
                "id": "advanced",
                "status_iri": "https://w3id.org/at#MajorRenovation",
                "english": (
                    "add 12 cm insulation below / alternatively: on top of ceiling "
                    "/ combination of both"
                ),
            },
        ],
    },
]

TEST = Namespace("https://namedgraphs.org/test#")
AT = Namespace("https://w3id.org/at#")
BEO = Namespace("https://w3id.org/beo#")
BMP = Namespace("https://w3id.org/bmp#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/2.1/vocab/unit#")

BMP_INSULATING = "https://w3id.org/bmp#Insulating"
BMP_LOAD_BEARING = "https://w3id.org/bmp#LoadBearing"
BMP_CONCRETE = "https://w3id.org/bmp#Concrete"
BMP_MULTI_LAYER = "https://w3id.org/bmp#MultiLayer"
BMP_SINGLE_LAYER = "https://w3id.org/bmp#SingleLayer"

STATE_LABEL_SUFFIX = {
    "existing": " (Existing State)",
    "usual": " (Usual Refurbishment)",
    "advanced": " (Advanced Refurbishment)",
}


def short_iri(iri: str | None) -> str:
    if not iri:
        return "—"
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def emit_thickness(graph: Graph, subject, cm: float) -> None:
    node = BNode()
    graph.add((subject, BMP.hasThickness, node))
    graph.add((node, RDF.value, Literal(float(cm), datatype=XSD.float)))
    graph.add((node, QUDT.unit, UNIT.CentiM))


def emit_composition_rdf(
    *,
    element_uri: str,
    rdf_type: str,
    german_desc: str,
    english_desc: str,
    element_thickness_cm: float | None,
    layer_sets: list[dict],
    ontology_graph: Graph,
) -> Graph:
    g = Graph()
    g.bind("test", TEST)
    g.bind("at", AT)
    g.bind("beo", BEO)
    g.bind("bmp", BMP)
    g.bind("qudt", QUDT)
    g.bind("unit", UNIT)
    g.bind("xsd", XSD)

    element_ref = URIRef(element_uri)
    g.add((element_ref, RDF.type, URIRef(rdf_type)))
    g.add((element_ref, RDF.type, AT.ElementArchetype))

    if german_desc:
        g.add((element_ref, AT.hasDescription, Literal(german_desc, lang="de")))
    if english_desc:
        g.add((element_ref, AT.hasDescription, Literal(english_desc, lang="en")))

    if element_thickness_cm:
        emit_thickness(g, element_ref, element_thickness_cm)

    slug = element_uri.rsplit("#", 1)[-1]
    for var_idx, layer_set in enumerate(layer_sets, start=1):
        layerset_uri = TEST[f"{slug}_layerset_var{var_idx}"]
        g.add((element_ref, BMP.hasLayerSet, layerset_uri))
        g.add((layerset_uri, RDF.type, BMP.LayerSet))

        topology = layer_set.get("layer_topology")
        if topology:
            g.add((layerset_uri, RDF.type, URIRef(topology)))

        if layer_set.get("layer_set_description"):
            g.add(
                (
                    layerset_uri,
                    AT.hasDescription,
                    Literal(layer_set["layer_set_description"], lang="en"),
                )
            )

        if layer_set.get("thickness_cm"):
            emit_thickness(g, layerset_uri, layer_set["thickness_cm"])

        for layer_idx, layer in enumerate(layer_set.get("layers", []), start=1):
            layer_uri = TEST[f"{slug}_layer_var{var_idx}_{layer_idx:02d}"]
            g.add((layerset_uri, BMP.hasLayer, layer_uri))
            material_uri = TEST[f"{slug}_mat_var{var_idx}_{layer_idx:02d}"]

            emit_enforced_layer(
                g,
                layer_uri=layer_uri,
                material_uri=material_uri,
                function_iri=layer.get("predicted_function_iri"),
                category_iri=layer.get("predicted_category_iri"),
                type_iri=layer.get("predicted_type_iri"),
                ontology_graph=ontology_graph,
                layer_types=(BMP.Layer,),
            )

            if layer.get("thickness_cm"):
                emit_thickness(g, layer_uri, layer["thickness_cm"])

    return g


def _layer_identity(layer: dict) -> tuple:
    return (
        layer.get("predicted_function_iri"),
        layer.get("predicted_category_iri"),
        layer.get("predicted_type_iri"),
        layer.get("thickness_cm"),
    )


def _copy_layer(layer: dict | None = None, **overrides) -> dict:
    src = layer or {}
    out = {
        "predicted_function_iri": src.get("predicted_function_iri"),
        "predicted_category_iri": src.get("predicted_category_iri"),
        "predicted_type_iri": src.get("predicted_type_iri"),
        "thickness_cm": src.get("thickness_cm"),
    }
    out.update(overrides)
    return out


def _is_insulating(layer: dict) -> bool:
    iri = layer.get("predicted_function_iri") or ""
    return iri.rsplit("#", 1)[-1] == "Insulating"


def _layers_from_ctx(ctx) -> list[dict]:
    if not ctx or not ctx.layer_json:
        return []
    sets = ctx.layer_json if isinstance(ctx.layer_json, list) else [ctx.layer_json]
    layer_set = next((item for item in sets if isinstance(item, dict) and item.get("layers")), None)
    if not layer_set:
        return []
    return [_copy_layer(layer) for layer in layer_set.get("layers") or []]


def _align_existing_layers(layers: list[dict], english: str) -> list[dict]:
    """Keep NLP layers, then fill gaps the existing text actually names."""
    text = (english or "").lower()
    thick = extract_thickness_cm(english)
    out = [_copy_layer(layer) for layer in layers]
    names_insulation = "insulation" in text or "daemm" in text or "dämm" in text

    if names_insulation:
        for layer in out:
            if _is_insulating(layer):
                layer["predicted_category_iri"] = None
                layer["predicted_type_iri"] = None
                if thick is not None:
                    layer["thickness_cm"] = thick
        for layer in out:
            if not _is_insulating(layer) and thick is not None and layer.get("thickness_cm") == thick:
                layer["thickness_cm"] = None
        if not any(_is_insulating(layer) for layer in out):
            out.append(_copy_layer(predicted_function_iri=BMP_INSULATING, thickness_cm=thick))

    if "concrete" in text:
        for layer in out:
            cat = (layer.get("predicted_category_iri") or "").rsplit("#", 1)[-1]
            if "Concrete" in cat and not _is_insulating(layer):
                layer["predicted_category_iri"] = BMP_CONCRETE
                layer["predicted_type_iri"] = None
                if not layer.get("predicted_function_iri"):
                    layer["predicted_function_iri"] = BMP_LOAD_BEARING
        has_concrete = any(
            ((layer.get("predicted_category_iri") or "").rsplit("#", 1)[-1] == "Concrete")
            and not _is_insulating(layer)
            for layer in out
        )
        if not has_concrete:
            out.insert(
                0,
                _copy_layer(
                    predicted_function_iri=BMP_LOAD_BEARING,
                    predicted_category_iri=BMP_CONCRETE,
                ),
            )
    return out


def _placement_kinds(text: str) -> list[str]:
    low = (text or "").lower()
    kinds: list[str] = []
    if re.search(r"\bbelow\b", low):
        kinds.append("below")
    if "on top" in low:
        kinds.append("top")
    if "combination" in low:
        kinds.append("both")
    return kinds or ["add"]


def _insulation_layer(cm: float | None) -> dict:
    return _copy_layer(predicted_function_iri=BMP_INSULATING, thickness_cm=cm)


def _delta_thickness(english: str, ctx) -> float | None:
    for layer in _layers_from_ctx(ctx):
        if _is_insulating(layer) and layer.get("thickness_cm") is not None:
            return layer["thickness_cm"]
    return extract_thickness_cm(english)


def _assemble_placement(existing_layers: list[dict], kind: str, cm: float | None) -> list[dict]:
    added = _insulation_layer(cm)
    if kind == "top":
        return [added] + existing_layers
    if kind == "both":
        return [added] + existing_layers + [_insulation_layer(cm)]
    return existing_layers + [added]


def _run_state_nlp(case: dict, state: dict, *, ontology_corpus, ontology_graph: Graph):
    return process_description(
        subject_uri=f"{case['element_uri']}_{state.get('id', 'state')}",
        german_desc=state.get("german") or None,
        english_desc=state.get("english") or None,
        ontology_corpus=ontology_corpus,
        profile=case.get("profile", "bbsr"),
        ontology_graph=ontology_graph,
        element_type_iri=case.get("rdf_type"),
    )


def process_renovation_case(case: dict, *, ontology_corpus, ontology_graph: Graph) -> dict:
    """Existing NLP first; usual/advanced reuse those layers and add placement variants."""
    states = {item.get("id"): item for item in case.get("renovation_states") or []}
    existing = states.get("existing") or {}
    existing_ctx = _run_state_nlp(
        case, existing, ontology_corpus=ontology_corpus, ontology_graph=ontology_graph
    )
    existing_layers = _align_existing_layers(
        _layers_from_ctx(existing_ctx),
        existing.get("english") or "",
    )

    bundle = {
        "existing_ctx": existing_ctx,
        "existing_layers": existing_layers,
        "states": {
            "existing": {
                **existing,
                "ctx": existing_ctx,
                "layer_sets": [{"kind": "given", "layers": existing_layers}],
            }
        },
    }
    for state_id in ("usual", "advanced"):
        state = states.get(state_id)
        if not state:
            continue
        ctx = _run_state_nlp(
            case, state, ontology_corpus=ontology_corpus, ontology_graph=ontology_graph
        )
        cm = _delta_thickness(state.get("english") or "", ctx)
        bundle["states"][state_id] = {
            **state,
            "ctx": ctx,
            "layer_sets": [
                {"kind": kind, "layers": _assemble_placement(existing_layers, kind, cm)}
                for kind in _placement_kinds(state.get("english") or "")
            ],
        }
    return bundle


def _emit_one_layer(graph: Graph, layer_uri, material_uri, layer: dict, ontology_graph: Graph) -> None:
    emit_enforced_layer(
        graph,
        layer_uri=layer_uri,
        material_uri=material_uri,
        function_iri=layer.get("predicted_function_iri"),
        category_iri=layer.get("predicted_category_iri"),
        type_iri=layer.get("predicted_type_iri"),
        ontology_graph=ontology_graph,
        layer_types=(BMP.Layer,),
    )
    if layer.get("thickness_cm") is not None:
        emit_thickness(graph, layer_uri, layer["thickness_cm"])


def _new_insulation_uri(slug: str, state_id: str, kind: str, new_idx: int):
    if kind == "both":
        face = "both_top" if new_idx == 1 else "both_below"
        return TEST[f"{slug}_{state_id}_layer_insul_{face}"]
    if kind in {"below", "top", "add"}:
        suffix = "below" if kind == "add" else kind
        return TEST[f"{slug}_{state_id}_layer_insul_{suffix}"]
    return TEST[f"{slug}_{state_id}_layer_{kind}_{new_idx:02d}"]


def emit_renovation_rdf(case: dict, bundle: dict, ontology_graph: Graph) -> Graph:
    g = Graph()
    g.bind("test", TEST)
    g.bind("at", AT)
    g.bind("beo", BEO)
    g.bind("bmp", BMP)
    g.bind("qudt", QUDT)
    g.bind("unit", UNIT)
    g.bind("xsd", XSD)

    element_uri = case["element_uri"]
    slug = element_uri.rsplit("#", 1)[-1]
    rdf_type = URIRef(case.get("rdf_type", "https://w3id.org/beo#Slab"))
    element_label = case.get("element_label") or "Floor 1"
    parent = URIRef(element_uri)
    g.add((parent, RDF.type, AT.ElementArchetype))
    g.add((parent, RDF.type, rdf_type))
    g.add((parent, RDFS.label, Literal(element_label)))

    existing_layers = bundle.get("existing_layers") or []
    existing_uri_by_identity = {}
    for idx, layer in enumerate(existing_layers, start=1):
        existing_uri_by_identity[_layer_identity(layer)] = TEST[f"{slug}_existing_layer_{idx:02d}"]

    for state_id, state in bundle.get("states", {}).items():
        state_ref = TEST[f"{slug}_{state_id}"]
        g.add((parent, AT.hasRenovationStatus, state_ref))
        g.add((state_ref, RDF.type, AT.ElementArchetype))
        g.add((state_ref, RDF.type, rdf_type))
        g.add((state_ref, RDFS.label, Literal(element_label + STATE_LABEL_SUFFIX.get(state_id, ""))))
        if state.get("english"):
            g.add((state_ref, AT.hasDescription, Literal(state["english"], lang="en")))
        if state.get("german"):
            g.add((state_ref, AT.hasDescription, Literal(state["german"], lang="de")))
        if state.get("status_iri"):
            g.add((state_ref, AT.hasRenovationStatus, URIRef(state["status_iri"])))

        for layer_set in state.get("layer_sets") or []:
            kind = layer_set.get("kind") or "given"
            layerset_uri = (
                TEST[f"{slug}_existing_layerset"]
                if state_id == "existing"
                else TEST[f"{slug}_{state_id}_layerset_{kind}"]
            )
            g.add((state_ref, BMP.hasLayerSet, layerset_uri))
            g.add((layerset_uri, RDF.type, BMP.LayerSet))
            n_layers = len(layer_set.get("layers") or [])
            g.add(
                (
                    layerset_uri,
                    RDF.type,
                    URIRef(BMP_MULTI_LAYER if n_layers > 1 else BMP_SINGLE_LAYER),
                )
            )

            new_idx = 0
            for layer in layer_set.get("layers") or []:
                ident = _layer_identity(layer)
                reuse = existing_uri_by_identity.get(ident) if state_id != "existing" else None
                if reuse is not None:
                    g.add((layerset_uri, BMP.hasLayer, reuse))
                    continue
                if state_id == "existing":
                    layer_uri = existing_uri_by_identity[ident]
                    mat_n = list(existing_uri_by_identity).index(ident) + 1
                    material_uri = TEST[f"{slug}_existing_mat_{mat_n:02d}"]
                else:
                    new_idx += 1
                    layer_uri = _new_insulation_uri(slug, state_id, kind, new_idx)
                    material_uri = TEST[f"{slug}_{state_id}_mat_{kind}_{new_idx:02d}"]
                g.add((layerset_uri, BMP.hasLayer, layer_uri))
                _emit_one_layer(g, layer_uri, material_uri, layer, ontology_graph)

    return g


def print_renovation_summary(case: dict, bundle: dict) -> None:
    print_summary(case, bundle.get("existing_ctx"))
    for state_id, state in bundle.get("states", {}).items():
        sets = state.get("layer_sets") or []
        print(f"\n  State {state_id}: {len(sets)} layer set(s)")
        for layer_set in sets:
            layers = layer_set.get("layers") or []
            print(f"    [{layer_set.get('kind')}] {len(layers)} layer(s)")
            for idx, layer in enumerate(layers, start=1):
                fn = short_iri(layer.get("predicted_function_iri"))
                cat = short_iri(layer.get("predicted_category_iri"))
                typ = short_iri(layer.get("predicted_type_iri"))
                lt = layer.get("thickness_cm", "—")
                print(f"      layer {idx}: {lt} cm | {fn} | {cat}/{typ}")


def print_summary(case: dict, ctx) -> None:
    print(f"\n{'=' * 60}")
    print(f"Case: {case.get('id', '?')} — {case.get('label', '')}")
    print(f"{'=' * 60}")
    states = case.get("renovation_states") or []
    if states:
        for state in states:
            print(f"  {state.get('id', '?')}: {state.get('english', '')}")
    else:
        if case.get("german"):
            print(f"DE: {case['german']}")
        if case.get("english"):
            print(f"EN: {case['english']}")

    if ctx is None:
        if not states:
            print("\n  -> Pipeline returned no result (empty description or LLM/retrieval failed)")
        return

    print(f"\nRetrieval: {len(ctx.matches)} ontology match(es)")
    for match in ctx.matches[:8]:
        print(
            f"  - {short_iri(match['entity_uri'])} "
            f"[{match.get('taxonomy_branch', '')}]"
        )
    if len(ctx.matches) > 8:
        print(f"  ... and {len(ctx.matches) - 8} more")

    if ctx.element_thickness_cm:
        print(f"\nElement thickness: {ctx.element_thickness_cm} cm")

    layer_sets = ctx.layer_json
    if not layer_sets:
        print("\n  -> No layer_sets produced")
        return

    if not isinstance(layer_sets, list):
        layer_sets = [layer_sets]

    print(f"\nLayer sets: {len(layer_sets)}")
    for var_idx, layer_set in enumerate(layer_sets, start=1):
        topo = short_iri(layer_set.get("layer_topology"))
        thick = layer_set.get("thickness_cm")
        print(f"\n  [{var_idx}] topology={topo}, thickness={thick or '—'} cm")
        for layer in layer_set.get("layers", []):
            fn = short_iri(layer.get("predicted_function_iri"))
            cat = short_iri(layer.get("predicted_category_iri"))
            typ = short_iri(layer.get("predicted_type_iri"))
            lt = layer.get("thickness_cm", "—")
            print(f"      layer {layer.get('layer_index', '?')}: {lt} cm | {fn} | {cat}/{typ}")


def gold_path_for(case: dict) -> Path:
    if case.get("gold"):
        return REPO_ROOT / case["gold"]
    return REPO_ROOT / "ttl" / "text_examples" / "gold" / f"{case.get('id') or 'example'}.ttl"


def compare_case_to_gold(case: dict, predicted_ttl: Path | None) -> int:
    gold_path = gold_path_for(case)
    if not gold_path.exists():
        print(f"  -> GOLD: missing ({gold_path.relative_to(REPO_ROOT)})")
        return 0
    if predicted_ttl is None or not predicted_ttl.exists():
        print(f"  -> GOLD: {gold_path.relative_to(REPO_ROOT)} (pipeline TTL not produced; not scored)")
        return 0
    diffs = compare_ttl_to_gold(predicted_ttl, gold_path)
    if not diffs:
        print(f"  -> GOLD: match {gold_path.relative_to(REPO_ROOT)}")
        return 0
    print(f"  -> GOLD: mismatch {gold_path.relative_to(REPO_ROOT)}")
    for line in diffs:
        print(f"      {line}")
    return 1


def _matches_payload(ctx) -> list[dict]:
    return [
        {
            "iri": m["entity_uri"],
            "branch": m.get("taxonomy_branch"),
            "text": m.get("text"),
        }
        for m in (ctx.matches if ctx else [])
    ]


def run_renovation_case(
    case: dict, *, ontology_corpus, ontology_graph: Graph, write_ttl: bool
) -> int:
    bundle = process_renovation_case(
        case, ontology_corpus=ontology_corpus, ontology_graph=ontology_graph
    )
    print_renovation_summary(case, bundle)

    out_dir = REPO_ROOT / "ttl" / "text_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = case.get("id") or "example"
    existing_ctx = bundle.get("existing_ctx")
    payload = {
        "id": case_id,
        "label": case.get("label"),
        "english": case.get("english"),
        "renovation_states": [
            {
                "id": state_id,
                "english": state.get("english"),
                "layer_sets": state.get("layer_sets"),
            }
            for state_id, state in bundle.get("states", {}).items()
        ],
        "matches": _matches_payload(existing_ctx),
    }
    json_path = out_dir / f"{case_id}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> JSON: {json_path.relative_to(REPO_ROOT)}")

    ttl_path = None
    if write_ttl and bundle.get("existing_layers"):
        rdf_g = emit_renovation_rdf(case, bundle, ontology_graph)
        ttl_path = out_dir / f"{case_id}.ttl"
        ttl_path.write_text(rdf_g.serialize(format="turtle"), encoding="utf-8")
        print(f"  -> TTL:  {ttl_path.relative_to(REPO_ROOT)} ({len(rdf_g)} triples)")
    elif write_ttl:
        print("  -> TTL:  skipped (no layer_sets to emit)")

    gold_code = compare_case_to_gold(case, ttl_path)
    record_ctx(
        case.get("id") or "example",
        existing_ctx,
        gold_fidelity=100.0 if gold_code == 0 and ttl_path else (0.0 if ttl_path else None),
        source="gold",
    )
    if gold_code:
        return 1
    return 0 if bundle.get("existing_layers") else 1


def run_case(case: dict, *, ontology_corpus, ontology_graph: Graph, write_ttl: bool) -> int:
    if case.get("renovation_states"):
        return run_renovation_case(
            case,
            ontology_corpus=ontology_corpus,
            ontology_graph=ontology_graph,
            write_ttl=write_ttl,
        )

    ctx = process_description(
        subject_uri=case["element_uri"],
        german_desc=case.get("german") or None,
        english_desc=case.get("english") or None,
        ontology_corpus=ontology_corpus,
        profile=case.get("profile", "bbsr"),
        ontology_graph=ontology_graph,
        element_type_iri=case.get("rdf_type"),
    )
    print_summary(case, ctx)

    out_dir = REPO_ROOT / "ttl" / "text_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = case.get("id") or "example"

    payload = {
        "id": case_id,
        "label": case.get("label"),
        "german": case.get("german"),
        "english": case.get("english"),
        "element_thickness_cm": getattr(ctx, "element_thickness_cm", None) if ctx else None,
        "layer_json": ctx.layer_json if ctx else None,
        "matches": _matches_payload(ctx),
    }
    json_path = out_dir / f"{case_id}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> JSON: {json_path.relative_to(REPO_ROOT)}")

    ttl_path = None
    if write_ttl and ctx and ctx.layer_json:
        layer_sets = ctx.layer_json if isinstance(ctx.layer_json, list) else [ctx.layer_json]
        rdf_g = emit_composition_rdf(
            element_uri=case["element_uri"],
            rdf_type=case.get("rdf_type", "https://w3id.org/beo#Wall"),
            german_desc=case.get("german", ""),
            english_desc=case.get("english", ""),
            element_thickness_cm=ctx.element_thickness_cm,
            layer_sets=layer_sets,
            ontology_graph=ontology_graph,
        )
        ttl_path = out_dir / f"{case_id}.ttl"
        ttl_path.write_text(rdf_g.serialize(format="turtle"), encoding="utf-8")
        print(f"  -> TTL:  {ttl_path.relative_to(REPO_ROOT)} ({len(rdf_g)} triples)")
    elif write_ttl:
        print("  -> TTL:  skipped (no layer_sets to emit)")

# compare case to gold standard
    gold_code = compare_case_to_gold(case, ttl_path)
    record_ctx(
        case.get("id") or "example",
        ctx if not case.get("renovation_states") else None,
        gold_fidelity=100.0 if gold_code == 0 and ttl_path else (0.0 if ttl_path else None),
        source="gold",
    )
    if gold_code:
        return 1
    return 0 if ctx and ctx.layer_json else 1


def load_cases_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a list or {{'examples': [...]}} in {path}")


def select_cases(cases: list[dict], *, case_num: int | None, case_id: str | None) -> list[dict]:
    if case_id:
        selected = [c for c in cases if c.get("id") == case_id]
        if not selected:
            raise SystemExit(f"No case with id={case_id!r}. Use --list to see ids.")
        return selected
    if case_num is not None:
        if case_num < 1 or case_num > len(cases):
            raise SystemExit(f"--case must be 1..{len(cases)}, got {case_num}")
        return [cases[case_num - 1]]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Test free-text descriptions via ontology_reasoning")
    parser.add_argument("--case", type=int, metavar="N", help="Run example N (1-based index)")
    parser.add_argument("--id", dest="case_id", metavar="ID", help="Run example by id field")
    parser.add_argument("--json", type=Path, metavar="PATH", help="Load examples from JSON file")
    parser.add_argument("--list", action="store_true", help="List available examples and exit")
    parser.add_argument("--no-ttl", action="store_true", help="Skip TTL output (JSON + console only)")
    args = parser.parse_args()

    cases = load_cases_from_json(args.json) if args.json else EXAMPLE_CASES

    if args.list:
        print("Available text examples:\n")
        for idx, case in enumerate(cases, start=1):
            print(f"  {idx}. {case.get('id', '?')} — {case.get('label', '')}")
        return 0

    selected = select_cases(cases, case_num=args.case, case_id=args.case_id)

    print(f"Repo: {REPO_ROOT}")
    log_llm_provider_once()
    llm_cfg = get_llm_config()
    print(f"LLM: {llm_cfg['provider']} / {llm_cfg['model']} | available: {is_llm_available()}")

    print(f"Loading ontology: {ontology_source_label(REPO_ROOT / 'owl')}")
    ontology_graph = load_ontology_graph(REPO_ROOT / "owl")
    ontology_corpus = load_ontology_corpus(ontology_graph)

    begin_report("gold")
    failures = 0
    for case in selected:
        code = run_case(
            case,
            ontology_corpus=ontology_corpus,
            ontology_graph=ontology_graph,
            write_ttl=not args.no_ttl,
        )
        failures += code

    finalize_report(data_graph=None, ontology_graph=ontology_graph, ttl_path=None)
    print(f"\nDone: {len(selected) - failures}/{len(selected)} case(s) matched gold / produced layer_sets")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
