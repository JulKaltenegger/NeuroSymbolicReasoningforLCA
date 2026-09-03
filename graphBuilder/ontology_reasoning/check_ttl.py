"""Audit generated TTL: OWL property ranges + SHACL composition shapes.

    python graphBuilder/run_pipeline.py --validate
    python -m ontology_reasoning.check_ttl
    python -m ontology_reasoning.check_ttl ttl/bbsr_buildings-enriched.ttl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from .config import OWL_DIR, REPO_ROOT
from .corpus import load_ontology_graph
from .validators import audit_graph

SHAPES_PATH = OWL_DIR / "shapes" / "composition.shacl.ttl"
REPORTS_DIR = REPO_ROOT / "ttl" / "reports"
SH_RESULT = URIRef("http://www.w3.org/ns/shacl#result")
SH_RESULT_SEVERITY = URIRef("http://www.w3.org/ns/shacl#resultSeverity")
SH_WARNING = URIRef("http://www.w3.org/ns/shacl#Warning")
SH_VIOLATION = URIRef("http://www.w3.org/ns/shacl#Violation")
SH_RESULT_MESSAGE = URIRef("http://www.w3.org/ns/shacl#resultMessage")
SH_FOCUS_NODE = URIRef("http://www.w3.org/ns/shacl#focusNode")
SH_SOURCE_CONSTRAINT = URIRef("http://www.w3.org/ns/shacl#sourceConstraint")
SH_SEVERITY = URIRef("http://www.w3.org/ns/shacl#severity")
SH_VALIDATION_RESULT = URIRef("http://www.w3.org/ns/shacl#ValidationResult")

DATASET_TTL = (
    ("bbsr", "ttl/bbsr_buildings-enriched.ttl"),
    ("tabula", "ttl/tabula_buildings-enriched.ttl"),
    ("slice", "ttl/slice_data_instantiated.ttl"),
)

ISSUE_LABELS = {
    "invalid_layer_function": "bmp:hasLayerFunction is not a proper bmp:LayerFunction subclass",
    "invalid_material_category": "bmp:hasMaterialCategory is not a category anchor",
    "invalid_material_type": "bmp:hasMaterialType is outside the OWL range",
    "orphan_material_type": "bmp:hasMaterialType is in range but reaches no category anchor",
    "material_on_layer": "material asserted on bmp:Layer instead of bmp:Material",
}


def short(iri) -> str:
    text = str(iri)
    return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def default_targets() -> list[Path]:
    return [REPO_ROOT / rel for _, rel in DATASET_TTL]


def _source_for(path: Path) -> str | None:
    rel = path.as_posix().replace("\\", "/")
    for name, dataset_rel in DATASET_TTL:
        if rel.endswith(dataset_rel):
            return name
    return None


def audit_owl_ranges(data_graph: Graph, ontology_graph: Graph) -> dict[str, list[tuple[str, str]]]:
    return audit_graph(data_graph, ontology_graph)


def _print_owl_issues(issues: dict[str, list[tuple[str, str]]]) -> int:
    if not issues:
        print("  OWL ranges: clean")
        return 0
    total = 0
    for kind, found in issues.items():
        counts = Counter(short(obj) for _, obj in found)
        total += len(found)
        print(f"  OWL ranges: {ISSUE_LABELS.get(kind, kind)}: {len(found)}")
        for name, count in counts.most_common():
            print(f"      {name} x{count}")
    return total


def validate_shacl(data_graph: Graph, shapes_graph: Graph) -> tuple[bool, Graph, str]:
    try:
        from pyshacl import validate
    except ImportError as exc:
        raise RuntimeError("pyshacl is not installed (pip install pyshacl)") from exc
    conforms, report_graph, report_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    return bool(conforms), report_graph, report_text


def _result_severity(report_graph: Graph, result) -> URIRef:
    """pyshacl SPARQL results often omit sh:resultSeverity; read the constraint."""
    severity = next(report_graph.objects(result, SH_RESULT_SEVERITY), None)
    if severity == SH_WARNING:
        return SH_WARNING
    constraint = next(report_graph.objects(result, SH_SOURCE_CONSTRAINT), None)
    if constraint is not None:
        declared = next(report_graph.objects(constraint, SH_SEVERITY), None)
        if declared == SH_WARNING:
            return SH_WARNING
    return severity or SH_VIOLATION


def _iter_results(report_graph: Graph):
    results = list(report_graph.subjects(RDF.type, SH_VALIDATION_RESULT))
    if results:
        return results
    return list(report_graph.objects(None, SH_RESULT))


def _shacl_rows(report_graph: Graph) -> list[dict]:
    rows = []
    for result in _iter_results(report_graph):
        severity = _result_severity(report_graph, result)
        focus = next(report_graph.objects(result, SH_FOCUS_NODE), None)
        message = next(report_graph.objects(result, SH_RESULT_MESSAGE), "")
        rows.append(
            {
                "severity": short(severity).lower(),
                "focus": short(focus),
                "focus_iri": str(focus) if focus is not None else None,
                "message": str(message),
            }
        )
    rows.sort(key=lambda row: (row["severity"] != "violation", row["focus"] or ""))
    return rows


def _print_shacl_results(rows: list[dict], *, limit: int = 12) -> tuple[int, int]:
    violations = sum(1 for row in rows if row["severity"] == "violation")
    warnings = sum(1 for row in rows if row["severity"] == "warning")
    if not rows:
        print("  SHACL: conforms")
        return 0, 0
    print(f"  SHACL: {violations} violation(s), {warnings} warning(s)")
    for row in rows[:limit]:
        print(f"      [{row['severity']}] {row['focus']}: {row['message']}")
    remaining = len(rows) - limit
    if remaining > 0:
        print(f"      ... {remaining} more")
    return violations, warnings


def audit_file(path: Path, ontology_graph: Graph, shapes_graph: Graph | None) -> dict:
    data_graph = Graph()
    data_graph.parse(location=path.resolve().as_uri(), format="ttl")
    rel = str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path).replace("\\", "/")
    print(f"\n=== {rel}")

    owl_issues = audit_owl_ranges(data_graph, ontology_graph)
    owl_n = _print_owl_issues(owl_issues)

    shacl_violations = 0
    shacl_warnings = 0
    shacl_conforms = True
    shacl_results: list[dict] = []
    if shapes_graph is not None:
        try:
            _conforms, report_graph, _text = validate_shacl(data_graph, shapes_graph)
            shacl_results = _shacl_rows(report_graph)
            shacl_violations, shacl_warnings = _print_shacl_results(shacl_results)
            shacl_conforms = shacl_violations == 0
        except RuntimeError as exc:
            print(f"  SHACL: skipped ({exc})")
            shacl_conforms = True

    return {
        "source": _source_for(path),
        "path": rel,
        "owl_range_issues": owl_n,
        "owl_issue_kinds": {kind: len(found) for kind, found in owl_issues.items()},
        "shacl_conforms": shacl_conforms,
        "shacl_violations": shacl_violations,
        "shacl_warnings": shacl_warnings,
        "shacl_results": shacl_results,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    targets = [Path(arg) for arg in argv] or default_targets()
    if not targets:
        print("No TTL files to validate.")
        return 0

    print(f"Ontology: {OWL_DIR / 'KB-LCA-merged.ttl'}")
    print(f"SHACL:    {SHAPES_PATH}")
    ontology_graph = load_ontology_graph()
    shapes_graph = None
    if SHAPES_PATH.is_file():
        shapes_graph = Graph()
        shapes_graph.parse(location=SHAPES_PATH.resolve().as_uri(), format="ttl")
    else:
        print(f"SHACL shapes missing: {SHAPES_PATH}")

    rows = []
    owl_total = 0
    shacl_fail = 0
    for target in targets:
        if not target.exists():
            print(f"\n=== {target}\n  missing — skipped")
            continue
        row = audit_file(target, ontology_graph, shapes_graph)
        rows.append(row)
        owl_total += row["owl_range_issues"]
        if not row["shacl_conforms"]:
            shacl_fail += 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "ttl_validation.json"
    report_path.write_text(
        json.dumps(
            {
                "shapes": str(SHAPES_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "files": rows,
                "owl_range_issues": owl_total,
                "shacl_files_with_violations": shacl_fail,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nOWL range issues: {owl_total}")
    print(f"SHACL files with violations: {shacl_fail}/{len(rows)}")
    print(f"  -> {report_path}")
    return 1 if owl_total or shacl_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
