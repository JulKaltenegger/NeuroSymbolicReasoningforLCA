"""Yield SLiCE worksection descriptions for material mapping."""

from __future__ import annotations

import re

from . import DescriptionRecord


def parse_slice_worksection(worksection_name: str) -> dict[str, str]:
    parts = [part.strip() for part in worksection_name.split("|")]
    function_segment = parts[0] if parts else worksection_name
    component_segment = parts[1] if len(parts) > 1 else ""
    material_segment = parts[2] if len(parts) > 2 else ""
    return {
        "function_segment": function_segment,
        "component_segment": component_segment,
        "material_segment": material_segment,
    }


def extract_material_hint(worksection_name: str) -> str:
    parsed = parse_slice_worksection(worksection_name)
    raw = parsed["material_segment"] or parsed["component_segment"] or worksection_name
    return re.sub(r"\([^)]*\)", "", raw).strip(" |")


def iter_slice_worksection_description(
    worksection_name: str,
    *,
    element_idx: int,
    layer_idx: int,
) -> DescriptionRecord:
    hint = extract_material_hint(worksection_name)
    return DescriptionRecord(
        subject_uri=f"https://namedgraphs.org/slice#Layer_inst_{element_idx:02d}_{layer_idx:02d}",
        german_desc=worksection_name,
        english_desc=hint,
        extra_text=hint,
        metadata={
            "profile": "slice",
            "worksection_name": worksection_name,
            "material_hint": hint,
        },
    )
