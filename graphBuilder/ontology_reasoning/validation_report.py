"""Collect gold-fidelity, retrieval cosine, and OWL-validity scores for a build.

Gold text-example TTLs that match are the 100% baseline. Dataset GraphBuilders
record cosine per description and OWL validity of the emitted graph. Writes
JSON + a PNG/SVG under ttl/reports/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rdflib import Graph

from .config import REPO_ROOT
from .validators import audit_graph

REPORTS_DIR = REPO_ROOT / "ttl" / "reports"


@dataclass
class ScoreRow:
    source: str
    case_id: str
    cosine: float | None = None
    gold_fidelity: float | None = None
    owl_validity: float | None = None

    @property
    def composite(self) -> float | None:
        """Gold matches sit at 100. New TTL uses OWL validity scaled by cosine."""
        if self.gold_fidelity is not None:
            return self.gold_fidelity
        if self.owl_validity is None:
            return None
        if self.cosine is None:
            return self.owl_validity
        return round(self.owl_validity * self.cosine, 1)


_ROWS: list[ScoreRow] = []
_SOURCE = "pipeline"


def cosine_from_matches(matches) -> float | None:
    scores = [
        float(m["score"])
        for m in (matches or [])
        if isinstance(m, dict) and isinstance(m.get("score"), (int, float))
    ]
    if not scores:
        return None
    scores.sort(reverse=True)
    top = scores[:3]
    return sum(top) / len(top)


def begin_report(source: str) -> None:
    global _SOURCE, _ROWS
    _SOURCE = source
    _ROWS = []


def record_ctx(case_id: str, ctx, *, gold_fidelity: float | None = None, source: str | None = None) -> None:
    matches = getattr(ctx, "matches", None) if ctx is not None else None
    _ROWS.append(
        ScoreRow(
            source=source or _SOURCE,
            case_id=str(case_id).rsplit("#", 1)[-1][:80],
            cosine=cosine_from_matches(matches),
            gold_fidelity=gold_fidelity,
        )
    )


def owl_validity_pct(data_graph: Graph, ontology_graph: Graph) -> float:
    issues = audit_graph(data_graph, ontology_graph)
    n_bad = sum(len(v) for v in issues.values())
    from .layer_axioms import BMP_HAS_LAYER_FUNCTION
    from .material_axioms import BMP_HAS_MATERIAL_CATEGORY, BMP_HAS_MATERIAL_TYPE

    n_slots = (
        len(list(data_graph.subject_objects(BMP_HAS_LAYER_FUNCTION)))
        + len(list(data_graph.subject_objects(BMP_HAS_MATERIAL_CATEGORY)))
        + len(list(data_graph.subject_objects(BMP_HAS_MATERIAL_TYPE)))
    )
    if n_slots == 0:
        return 100.0 if n_bad == 0 else 0.0
    return max(0.0, round(100.0 * (1.0 - n_bad / n_slots), 1))


def finalize_report(
    *,
    data_graph: Graph | None,
    ontology_graph: Graph | None,
    ttl_path: Path | None,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    owl_pct = None
    if data_graph is not None and ontology_graph is not None:
        owl_pct = owl_validity_pct(data_graph, ontology_graph)
        for row in _ROWS:
            if row.owl_validity is None:
                row.owl_validity = owl_pct

    payload = {
        "source": _SOURCE,
        "ttl": str(ttl_path) if ttl_path else None,
        "owl_validity_pct": owl_pct,
        "gold_is_100_when_matched": True,
        "rows": [asdict(row) | {"composite": row.composite} for row in _ROWS],
    }
    json_path = out_dir / f"{_SOURCE}_validation.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    chart_path = _write_chart(out_dir / f"{_SOURCE}_validation.png", out_dir / f"{_SOURCE}_validation.svg")
    print(f"  -> validation JSON: {json_path}")
    if chart_path:
        print(f"  -> validation chart: {chart_path}")
    return json_path


def _write_chart(png_path: Path, svg_path: Path) -> Path | None:
    gold_rows = [r for r in _ROWS if r.gold_fidelity is not None]
    other_rows = [r for r in _ROWS if r.gold_fidelity is None]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_svg_fallback(svg_path, gold_rows, other_rows)
        return svg_path if svg_path.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if gold_rows:
        labels = [r.case_id for r in gold_rows]
        fidelity = [r.gold_fidelity or 0 for r in gold_rows]
        cosine = [(r.cosine or 0) * 100 for r in gold_rows]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], fidelity, width=0.4, label="Gold fidelity (match=100%)", color="#2ca02c")
        ax.bar([i + 0.2 for i in x], cosine, width=0.4, label="Retrieval cosine %", color="#1f77b4")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.axhline(100, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 110)
    ax.set_ylabel("score %")
    ax.set_title("Gold text examples (baseline = 100%)")
    ax.legend(fontsize=8)

    ax = axes[1]
    sources = sorted({r.source for r in other_rows})
    if sources:
        means_cos = []
        means_owl = []
        for src in sources:
            rows = [r for r in other_rows if r.source == src]
            cos_vals = [(r.cosine or 0) * 100 for r in rows]
            owl_vals = [r.owl_validity for r in rows if r.owl_validity is not None]
            means_cos.append(sum(cos_vals) / len(cos_vals) if cos_vals else 0)
            means_owl.append(sum(owl_vals) / len(owl_vals) if owl_vals else 0)
        x = range(len(sources))
        ax.bar([i - 0.2 for i in x], means_cos, width=0.4, label="Mean cosine %", color="#1f77b4")
        ax.bar([i + 0.2 for i in x], means_owl, width=0.4, label="OWL validity %", color="#ff7f0e")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sources)
        ax.axhline(100, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.7, label="Gold = 100%")
    else:
        ax.text(0.5, 0.5, "No dataset rows in this run", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylim(0, 110)
    ax.set_title("New TTL vs gold baseline")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return png_path


def _write_svg_fallback(svg_path: Path, gold_rows: list[ScoreRow], other_rows: list[ScoreRow]) -> None:
    bars = []
    y = 20
    for row in gold_rows:
        w = 3 * (row.gold_fidelity or 0)
        bars.append(f'<text x="10" y="{y}" font-size="11">{row.case_id} gold</text>')
        bars.append(f'<rect x="180" y="{y-10}" width="{w}" height="12" fill="#2ca02c"/>')
        y += 22
    for src in sorted({r.source for r in other_rows}):
        rows = [r for r in other_rows if r.source == src]
        cos = [r.cosine or 0 for r in rows]
        mean = (sum(cos) / len(cos) * 100) if cos else 0
        bars.append(f'<text x="10" y="{y}" font-size="11">{src} cosine</text>')
        bars.append(f'<rect x="180" y="{y-10}" width="{3 * mean}" height="12" fill="#1f77b4"/>')
        y += 22
    svg_path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="520" height="{max(y + 20, 80)}">'
        + "".join(bars)
        + "</svg>",
        encoding="utf-8",
    )
