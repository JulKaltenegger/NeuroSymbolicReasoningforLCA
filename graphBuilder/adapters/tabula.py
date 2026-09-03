"""Yield TABULA renovation-stage descriptions."""

from __future__ import annotations

from . import DescriptionRecord


def iter_tabula_stage_descriptions(
    building_id: str,
    safe_id: str,
    element_label: str,
    stage: dict,
) -> DescriptionRecord | None:
    desc_de = stage.get("desc_de") or ""
    desc_eng = stage.get("desc_eng") or ""
    if not desc_de and not desc_eng:
        return None

    safe_element = element_label.replace(" ", "_")
    state_suffix = f"{safe_element}_{stage['suffix']}"
    subject_uri = f"https://namedgraphs.org/tabula#Building_{safe_id}_Element_{state_suffix}"

    return DescriptionRecord(
        subject_uri=subject_uri,
        german_desc=desc_de,
        english_desc=desc_eng,
        metadata={
            "profile": "tabula",
            "building_id": building_id,
            "element_label": element_label,
            "stage_suffix": stage["suffix"],
            "stage_mode": stage["mode"],
        },
    )
