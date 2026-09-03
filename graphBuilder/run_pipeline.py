#!/usr/bin/env python3
"""Single entry for gold text examples + BBSR / TABULA / SLiCE graph builders.

The four conversion steps live in graphBuilder/sources/. This file only
selects which of them to run:

    python graphBuilder/run_pipeline.py                 # gold examples only
    python graphBuilder/run_pipeline.py --bbsr          # gold + BBSR
    python graphBuilder/run_pipeline.py --tabula --slice
    python graphBuilder/run_pipeline.py --all           # gold + all three datasets
    python graphBuilder/run_pipeline.py --skip-gold --bbsr
    python graphBuilder/run_pipeline.py --smoke --all   # one-item dataset tests
    python graphBuilder/run_pipeline.py --check-llm --live
    python graphBuilder/run_pipeline.py --validate      # SHACL + OWL ranges on existing TTL
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

GRAPH_BUILDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = GRAPH_BUILDER_DIR.parent
if str(GRAPH_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_BUILDER_DIR))

from sources import SOURCES


def run_step(label: str, script: Path, env: dict | None = None) -> int:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    merged = os.environ.copy()
    merged.setdefault("PYTHONUNBUFFERED", "1")
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=GRAPH_BUILDER_DIR,
        env=merged,
    ).returncode


def selected_sources(args) -> list[dict]:
    run_ids = set()
    if not args.skip_gold:
        run_ids.add("gold")
    if args.all or args.bbsr:
        run_ids.add("bbsr")
    if args.all or args.tabula:
        run_ids.add("tabula")
    if args.all or args.slice:
        run_ids.add("slice")
    if args.smoke and not (args.bbsr or args.tabula or args.slice or args.all):
        run_ids.update(item["id"] for item in SOURCES if item["group"] == "dataset")
        run_ids.discard("gold")
    return [item for item in SOURCES if item["id"] in run_ids]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gold text examples + BBSR / TABULA / SLiCE graph builders"
    )
    parser.add_argument("--bbsr", action="store_true", help="Run sources/Graph_builder_BBSR_JSON.py")
    parser.add_argument("--tabula", action="store_true", help="Run sources/Graph_builder_TABULA.py")
    parser.add_argument("--slice", action="store_true", help="Run sources/Graph_builder_SLICE_CSV.py")
    parser.add_argument("--all", action="store_true", help="Run gold + BBSR + TABULA + SLiCE")
    parser.add_argument("--skip-gold", action="store_true", help="Skip gold text-example baseline")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Dataset builders process one test item (env filters already in each builder)",
    )
    parser.add_argument("--check-llm", action="store_true", help="Check .env / API keys and exit")
    parser.add_argument("--live", action="store_true", help="With --check-llm, ping the configured LLM")
    parser.add_argument("--validate", action="store_true", help="Validate existing TTL with SHACL + OWL ranges (no rebuild)")
    parser.add_argument("ttl_files", nargs="*", help="Optional TTL paths (used with --validate)")
    args = parser.parse_args()

    if args.live and not args.check_llm:
        parser.error("--live requires --check-llm")

    if args.check_llm:
        from ontology_reasoning.check_llm import main as check_llm_main

        return check_llm_main(["--live"] if args.live else [])

    if args.validate:
        from ontology_reasoning.check_ttl import main as check_ttl_main

        return check_ttl_main(list(args.ttl_files))

    steps = selected_sources(args)
    if not steps:
        parser.error("Nothing to run. Use default gold, or --bbsr / --tabula / --slice / --all / --smoke")

    print(f"Repo: {REPO_ROOT}")
    print("Pipeline sources (graphBuilder/sources/):")
    for item in SOURCES:
        marker = "*" if item in steps else " "
        print(f"  [{marker}] {item['id']:7}  {item['script'].name}")
    print("TTL out: ttl/")
    print("Reports: ttl/reports/<source>_validation.json|.svg")

    code = 0
    for item in steps:
        extra_env = item.get("smoke_env") if args.smoke else None
        code |= run_step(item["label"], item["script"], env=extra_env)

    print("\nTTL files under ttl/")
    print("Charts under ttl/reports/")
    return 1 if code else 0


if __name__ == "__main__":
    raise SystemExit(main())
