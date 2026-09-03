"""Compare pipeline TTL against gold TTL by composition shape, not by URI minting."""

from __future__ import annotations

from collections import Counter

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

BMP_HAS_LAYER_SET = URIRef("https://w3id.org/bmp#hasLayerSet")
BMP_HAS_LAYER = URIRef("https://w3id.org/bmp#hasLayer")
BMP_HAS_LAYER_FUNCTION = URIRef("https://w3id.org/bmp#hasLayerFunction")
BMP_HAS_MATERIAL = URIRef("https://w3id.org/bmp#hasMaterial")
BMP_HAS_MATERIAL_CATEGORY = URIRef("https://w3id.org/bmp#hasMaterialCategory")
BMP_HAS_MATERIAL_TYPE = URIRef("https://w3id.org/bmp#hasMaterialType")
BMP_HAS_THICKNESS = URIRef("https://w3id.org/bmp#hasThickness")


def _short(iri) -> str | None:
    if iri is None:
        return None
    text = str(iri)
    return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _thickness_cm(graph: Graph, subject) -> float | None:
    for node in graph.objects(subject, BMP_HAS_THICKNESS):
        for value in graph.objects(node, RDF.value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _layer_record(graph: Graph, layer) -> dict:
    fn = next((_short(obj) for obj in graph.objects(layer, BMP_HAS_LAYER_FUNCTION)), None)
    cat = None
    typ = None
    for material in graph.objects(layer, BMP_HAS_MATERIAL):
        cat = next((_short(obj) for obj in graph.objects(material, BMP_HAS_MATERIAL_CATEGORY)), None)
        typ = next((_short(obj) for obj in graph.objects(material, BMP_HAS_MATERIAL_TYPE)), None)
        break
    return {
        "function": fn,
        "category": cat,
        "type": typ,
        "thickness_cm": _thickness_cm(graph, layer),
    }


def composition_signature(graph: Graph) -> list[dict]:
    """Bag of layer-sets per element that has bmp:hasLayerSet."""
    signatures = []
    for element in sorted(graph.subjects(BMP_HAS_LAYER_SET, None), key=str):
        layer_sets = []
        for layerset in graph.objects(element, BMP_HAS_LAYER_SET):
            topologies = sorted(_short(t) for t in graph.objects(layerset, RDF.type) if _short(t))
            layers = [_layer_record(graph, layer) for layer in graph.objects(layerset, BMP_HAS_LAYER)]
            layers.sort(
                key=lambda row: (
                    row["function"] or "",
                    row["category"] or "",
                    row["type"] or "",
                    row["thickness_cm"] if row["thickness_cm"] is not None else -1,
                )
            )
            layer_sets.append(
                {
                    "topology": topologies,
                    "thickness_cm": _thickness_cm(graph, layerset),
                    "layers": layers,
                }
            )
        layer_sets.sort(
            key=lambda row: (
                tuple(row["topology"]),
                len(row["layers"]),
                row["thickness_cm"] if row["thickness_cm"] is not None else -1,
                tuple(
                    (
                        layer["function"] or "",
                        layer["category"] or "",
                        layer["type"] or "",
                    )
                    for layer in row["layers"]
                ),
            )
        )
        signatures.append(
            {
                "element": _short(element),
                "element_thickness_cm": _thickness_cm(graph, element),
                "layer_sets": layer_sets,
            }
        )
    return signatures


def describe_mismatch(predicted: list[dict], gold: list[dict]) -> list[str]:
    lines: list[str] = []
    if len(predicted) != len(gold):
        lines.append(f"element count {len(predicted)} != gold {len(gold)}")
    n = max(len(predicted), len(gold))
    for idx in range(n):
        pred = predicted[idx] if idx < len(predicted) else None
        exp = gold[idx] if idx < len(gold) else None
        label = (exp or pred or {}).get("element", f"#{idx}")
        if pred is None:
            lines.append(f"{label}: missing in pipeline output")
            continue
        if exp is None:
            lines.append(f"{label}: extra in pipeline output")
            continue
        if pred.get("element_thickness_cm") != exp.get("element_thickness_cm"):
            lines.append(
                f"{label}: element thickness {pred.get('element_thickness_cm')} "
                f"!= gold {exp.get('element_thickness_cm')}"
            )
        if pred["layer_sets"] != exp["layer_sets"]:
            lines.append(f"{label}: layer-set shape differs")
            lines.append(f"    gold:      {exp['layer_sets']}")
            lines.append(f"    predicted: {pred['layer_sets']}")
    return lines


def reused_layer_count(graph: Graph) -> int:
    """How many layer individuals appear in more than one LayerSet (URI reuse)."""
    counts: Counter = Counter()
    for layerset in graph.objects(None, BMP_HAS_LAYER_SET):
        for layer in graph.objects(layerset, BMP_HAS_LAYER):
            counts[layer] += 1
    return sum(1 for n in counts.values() if n > 1)


def compare_ttl_to_gold(predicted_path, gold_path) -> list[str]:
    predicted = Graph()
    gold = Graph()
    predicted.parse(location=predicted_path.resolve().as_uri(), format="ttl")
    gold.parse(location=gold_path.resolve().as_uri(), format="ttl")
    diffs = describe_mismatch(composition_signature(predicted), composition_signature(gold))
    gold_reused = reused_layer_count(gold)
    pred_reused = reused_layer_count(predicted)
    if gold_reused and pred_reused != gold_reused:
        diffs.append(
            f"reused layer URIs {pred_reused} != gold {gold_reused} "
            "(usual/advanced should point at the existing layer individuals)"
        )
    return diffs
