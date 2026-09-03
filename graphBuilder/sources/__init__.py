"""Conversion sources invoked by run_pipeline.py.

Each entry is one graph-construction step: gold free-text examples, then the
three dataset builders. run_pipeline.py is the only orchestrator.
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent

SOURCES = (
    {
        "id": "gold",
        "script": _DIR / "run_text_examples.py",
        "label": "Gold text examples -> ttl/text_examples/",
        "group": "gold",
    },
    {
        "id": "bbsr",
        "script": _DIR / "Graph_builder_BBSR_JSON.py",
        "label": "BBSR JSON -> ttl/bbsr_buildings-enriched.ttl",
        "group": "dataset",
        "smoke_env": {"BBSR_TEST_INDEX": "1"},
    },
    {
        "id": "tabula",
        "script": _DIR / "Graph_builder_TABULA.py",
        "label": "TABULA JSON -> ttl/tabula_buildings-enriched.ttl",
        "group": "dataset",
        "smoke_env": {"TABULA_TEST_BUILDING_ID": "DE.East.AB.06.Gen.ReEx.001"},
    },
    {
        "id": "slice",
        "script": _DIR / "Graph_builder_SLICE_CSV.py",
        "label": "SLiCE CSV -> ttl/slice_data_instantiated.ttl",
        "group": "dataset",
        "smoke_env": {"SLICE_TEST_ELEMENT": "EW05 CLT + str blown"},
    },
)
