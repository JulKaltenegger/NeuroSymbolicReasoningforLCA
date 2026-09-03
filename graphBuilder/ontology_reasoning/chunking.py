"""Step 2: normalize and decompose description text into retrieval chunks."""

from __future__ import annotations

import re


def combined_description(german_desc: str | None, english_desc: str | None) -> str:
    return " ".join(part for part in (german_desc or "", english_desc or "") if part).strip()


def decompose_description(
    german_desc: str | None,
    english_desc: str | None,
    *,
    profile: str = "bbsr",
    extra_text: str | None = None,
) -> list[str]:
    """Return deduplicated textual chunks for embedding retrieval."""
    text = combined_description(german_desc, english_desc)
    if extra_text:
        text = f"{text} {extra_text}".strip()
    if not text:
        return []

    chunks = _rule_chunks(text.lower())

    if profile == "tabula" and not chunks:
        chunks = [text]

    if not chunks:
        chunks = [text]

    return list(dict.fromkeys(chunks))


def _rule_chunks(text_lower: str) -> list[str]:
    chunks: list[str] = []

    if "slab" in text_lower or "decke" in text_lower:
        chunks.append("slab")
    if "hohlraumdecke" in text_lower or "cavity slab" in text_lower or "hollow-core" in text_lower:
        chunks.append("hollow-core slab")
    if "vollbetondecke" in text_lower or "solid slab" in text_lower:
        chunks.append("solid concrete slab")
    if "wall" in text_lower or "wand" in text_lower:
        chunks.append("wall")
    if "floor" in text_lower or "fußboden" in text_lower or "fussboden" in text_lower:
        chunks.append("floor")
    if "roof" in text_lower or "dach" in text_lower:
        chunks.append("roof")

    if "reinforced concrete" in text_lower or "stahlbeton" in text_lower:
        chunks.append("reinforced concrete")
    if "spannbeton" in text_lower or "prestressed concrete" in text_lower:
        chunks.append("prestressed concrete")
    if "leichtbeton" in text_lower or "lightweight concrete" in text_lower:
        chunks.append("lightweight concrete")
    if "leichtzuschlagstoffbeton" in text_lower or "lzs-beton" in text_lower:
        chunks.append("lightweight aggregate concrete")
    if "porenbeton" in text_lower or "aerated concrete" in text_lower:
        chunks.append("aerated concrete")
    if "haufwerksporig" in text_lower or "no-fines concrete" in text_lower:
        chunks.append("no-fines porous concrete")
    if "normalbeton" in text_lower or "normal concrete" in text_lower:
        chunks.append("normal concrete")
    elif "concrete" in text_lower or "beton" in text_lower:
        chunks.append("concrete")

    if "mineral wool" in text_lower or "mineralwolle" in text_lower:
        chunks.append("mineral wool")
    if "schaumpolystyrol" in text_lower or "polystyrene" in text_lower or "eps" in text_lower:
        chunks.append("expanded polystyrene insulation")
    if re.search(r"\b(dämmung|daemmung|insulation|insulated|gedämmt|gedaemmt)\b", text_lower):
        chunks.append("thermal insulation")
        if "kerndämmung" in text_lower or "kerndaemmung" in text_lower:
            chunks.append("core insulation")
        if "außendämmung" in text_lower or "aussendaemmung" in text_lower:
            chunks.append("exterior insulation")
        if "innendämmung" in text_lower or "innendaemmung" in text_lower:
            chunks.append("interior insulation")

    if "brick" in text_lower or "ziegel" in text_lower or "mauerwerk" in text_lower:
        chunks.append("brickwork")
    if "hohlloch" in text_lower or "hollow-hole" in text_lower or "hollow hole" in text_lower:
        chunks.append("hollow-hole brick elements")
    if "wood" in text_lower or "timber" in text_lower or "holz" in text_lower:
        chunks.append("wood timber")
    if re.search(r"\bcork\b", text_lower):
        chunks.append("cork")
    if re.search(r"\bstraw\b", text_lower):
        chunks.append("straw plant fiber")
    if "glazing" in text_lower or "verglasung" in text_lower or "fenster" in text_lower:
        chunks.append("glazing")
    if "door" in text_lower or "tür" in text_lower or "tur" in text_lower:
        chunks.append("door opening")

    if "plaster" in text_lower or "gips" in text_lower or "putz" in text_lower or "geputzt" in text_lower:
        chunks.append("gypsum plaster binders")
    if "single layer" in text_lower or "einschichtig" in text_lower:
        chunks.append("single layer")
    if "two layer" in text_lower or "zweischichtig" in text_lower:
        chunks.append("two layer")
    if re.search(r"\bmulti[- ]?layer", text_lower) or "mehrschichtig" in text_lower or "dreischichtig" in text_lower or "3-layered" in text_lower:
        chunks.append("multi layer")
    if "cavity wall" in text_lower or re.search(r"\bkellerwand\b|\bhohlraumwand\b", text_lower):
        chunks.append("cavity wall")

    for thickness in re.findall(r"\d+(?:,\d+)?\s*cm", text_lower):
        chunks.append(f"{thickness} thickness")
    for thickness in re.findall(r"\d+(?:,\d+)?\s*mm", text_lower):
        chunks.append(f"{thickness} thickness")

    return chunks
