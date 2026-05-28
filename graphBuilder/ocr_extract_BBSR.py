import json
import re
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pdf2image.exceptions import PDFInfoNotInstalledError

import os
import re
import json
from pathlib import Path

import pandas as pd

from pdf2image import convert_from_path

# TESSERACT SETUP 
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# CONFIG

REPO_ROOT = Path(__file__).resolve().parent

PDF_PATH = "data-text-BBSR/BBSR.pdf"
POPPLER_PATH = r"C:\Users\go46wic\Dropbox\_PhD\11_KnowledgeBaseLCA\NeuroSymbolicReasoningforLCA\tools\Release-26.02.0-0\poppler-26.02.0\Library\bin"

INDEX_PAGE = 5      # page with Inhaltsverzeichnis (1-based PDF index)
LOGICAL_FIRST_PAGE_ON_PDF = 7

DEFAULT_PAGE_BOX_NOTE = (
    "BBSR / iEMB typology specification sheets: many Bauweisen and IW/P series; each page "
    "typically three sections (Gebäudekonstruktion, Gebäudecharakteristik, Wohnungscharakteristik). "
    "Printed booklet page (Seite n) is not the same as PDF page index."
)
CONTENT_PDF_PAGES = [11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 57, 61, 65, 69]  # add more PDF 1-based indices, e.g. [11, 15, 19, 23, 27, 31, ...]
CONTENT_PDF_PAGE_NOTES = []  # optional per-page strings (same order as CONTENT_PDF_PAGES); use for §/Typenserie hints

# OCR quality
CONTENT_PAGE_OCR_DPI = 300
PAGE_BOX_LAYOUT_Y_TOL = 22
PAGE_BOX_JSON_PATH = "data-text-BBSR/page_boxes.json"
WRITE_PAGE_BOXES_JSON = True

# Section 2. OCR HELPER - convert  pdf to image, tesseract ocr and pandas

def ocr_page(pdf_path, page_number, dpi=300):
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
    )

    image = images[0]

    data = pytesseract.image_to_data(
        image,
        lang="deu",
        output_type=Output.DATAFRAME
    )

    data = data.dropna(subset=["text"])
    data = data[data["text"].str.strip() != ""]

    return data

# Section 3. INDEX PARSER (TOC)

def group_rows(df, y_tol=10):
    rows = []
    current = []
    current_y = None

    df = df.sort_values("top")

    for _, r in df.iterrows():
        if current_y is None:
            current_y = r["top"]

        # stricter grouping
        if abs(r["top"] - current_y) <= y_tol:
            current.append(r)
        else:
            rows.append(current)
            current = [r]
            current_y = r["top"]

    if current:
        rows.append(current)

    return rows



def parse_index_from_layout(df):
    entries = []

    # define columns (tune if needed)
    LEFT_MAX = 350
    RIGHT_MIN = 1200

    rows = group_rows(df)

    for row in rows:
        row = sorted(row, key=lambda r: r["left"])

        section = None
        title_parts = []
        page = None

        for r in row:
            txt = r["text"]
            x = r["left"]

            # LEFT COLUMN → section
            if x < LEFT_MAX and re.match(r"\d+(\.\d+)?", txt):
                section = txt

            # RIGHT COLUMN → page
            elif x > RIGHT_MIN and re.match(r"\d+", txt):
                page = int(txt)

            # MIDDLE → title
            else:
                title_parts.append(txt)

        title = " ".join(title_parts).strip()

        if section or title:
            entries.append({
                "section": section,
                "title": title,
                "page": page
            })

    return entries


# CONTENT parsing (KEY-VALUE)


def extract_key_values(data):
    """
    Legacy: one flat dict for the whole page via Tesseract ``line_num``.

    Problem: ignores layout columns and section headings (2.1.1 / 2.1.2 …), so keys
    collide and stdout looks like “everything in one object”. Prefer
    :func:`layout_df_to_lines` + :func:`layout_lines_to_structured_sections`.
    """
    rows = []

    for _, g in data.groupby("line_num"):
        words = g.sort_values("left")["text"].tolist()
        line = " ".join(words)
        rows.append(line)

    result = {}
    current_key = None

    for line in rows:

        # normalize bullet OCR noise
        line = re.sub(r"^[•\-\*\s]+", "", line)

        if ":" in line:
            key, val = line.split(":", 1)

            current_key = key.strip()
            result[current_key] = val.strip()

        else:
            # multiline continuation
            if current_key:
                result[current_key] += " " + line.strip()

    return result


#Layout lines (geometry) + boxed specs structured JSON


def layout_df_to_lines(df, y_tol=22):
    """Cluster OCR words into reading-order lines (avoids dumping the whole page as one KV map)."""
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()

    df_sorted = df.sort_values("top")
    lines_out = []
    current = []
    anchor_y = None

    for _, row in df_sorted.iterrows():
        if anchor_y is None:
            anchor_y = row["top"]
        if abs(row["top"] - anchor_y) <= y_tol:
            current.append(row)
        else:
            if current:
                sub = pd.DataFrame(current).sort_values("left")
                text = " ".join(sub["text"].tolist())
                x0 = int(sub["left"].min())
                y0 = int(sub["top"].min())
                x1 = int((sub["left"] + sub["width"]).max())
                y1 = int((sub["top"] + sub["height"]).max())
                lines_out.append({"text": text, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}})
            current = [row]
            anchor_y = row["top"]

    if current:
        sub = pd.DataFrame(current).sort_values("left")
        text = " ".join(sub["text"].tolist())
        x0 = int(sub["left"].min())
        y0 = int(sub["top"].min())
        x1 = int((sub["left"] + sub["width"]).max())
        y1 = int((sub["top"] + sub["height"]).max())
        lines_out.append({"text": text, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}})
    return lines_out


def layout_strip_bullet_noise(text):
    t = text.strip()
    t = re.sub(r"^[\*\u2022\•\-\s]+", "", t)
    t = re.sub(r"^[eEsS©®]\s*", "", t)
    return t.strip()


def layout_split_key_value(line_text):
    if ":" in line_text:
        key, _, rest = line_text.partition(":")
        if key.strip() and rest.strip():
            return key.strip(), rest.strip()
    return None, None


def _normalize_ocr_measurement_display(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"m\?", "m²", s)


def _normalize_property_values(props):
    if isinstance(props, dict):
        return {k: _normalize_property_values(v) for k, v in props.items()}
    if isinstance(props, list):
        return [_normalize_property_values(x) for x in props]
    if isinstance(props, str):
        return _normalize_ocr_measurement_display(props)
    return props


def _normalize_key_for_lookup(raw):
    if not raw:
        return ""
    t = raw.lower().strip()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    t = re.sub(r"[^\w\s()\-]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _sanitize_property_key(raw_key):
    k = layout_strip_bullet_noise(raw_key or "")
    k = re.sub(r"^[»°\*\+\-\s]+\s*", "", k).strip()
    k = re.sub(r"^[a-zäöüß]\s+", "", k, flags=re.I)
    return k.strip()


def _fields_to_flat_kv(fields):
    out = {}
    for f in fields or []:
        kk = _sanitize_property_key(f.get("key") or "")
        if not kk:
            continue
        vv = (f.get("value") or "").strip()
        if kk in out:
            out[kk] = (out[kk] + " " + vv).strip()
        else:
            out[kk] = vv
    return out


def _take_field_once(mapping, needles, reject_contains=()):
    needles = [_normalize_key_for_lookup(n) for n in needles]
    reject_contains = [_normalize_key_for_lookup(r) for r in reject_contains]
    for k in list(mapping.keys()):
        nk = _normalize_key_for_lookup(k)
        if any(rc and rc in nk for rc in reject_contains):
            continue
        if needles and all(n in nk for n in needles):
            return mapping.pop(k)
    return None



############## parse element descriptinos

def _parse_deckenspannweiten(val):
    chunks = re.findall(r"\d+,\d+\s*m", val)
    if chunks:
        return chunks
    chunks_d = re.findall(r"\d+\.\d+\s*m", val)
    return chunks_d if chunks_d else [val.strip()]

def _fix_ausfuehrung_typos(s):
    t = (s or "").strip()
    if re.match(r"^in.?schichtig$", t, flags=re.I):
        return "einschichtig"
    return t

def _parse_materials_fragment(s):
    s = (s or "").strip()
    parts = re.split(r"\s+oder\s+", s, flags=re.IGNORECASE)
    parts = [p.strip().rstrip("-").strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    return parts[0] if parts else s

def _parse_aussenwaende(val):
    val = (val or "").strip().replace(".", ",")
    m = re.match(
        r"^(?P<mats>.+?),\s*(?P<ausf>[^,]+),\s*(?P<dnum>[\d,]+)\s*cm\s*dick\s*;\s*(?P<ober>.+)$",
        val,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return {"_raw": val} if val else {}
    mats = _parse_materials_fragment(m.group("mats"))
    dicke = "{} cm".format(m.group("dnum").strip())
    return {
        "material": mats,
        "ausführung": _fix_ausfuehrung_typos(m.group("ausf")),
        "dicke": dicke,
        "oberfläche": m.group("ober").strip(),
    }


def _parse_innenwaende(val):
    val = (val or "").strip().replace(".", ",")
    m = re.match(
        r"^(?P<mat>.+?),\s*(?P<dnum>[\d,]+)\s*cm\s*dick\s*;\s*(?P<ober>.+)$",
        val,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return {"_raw": val} if val else {}
    return {
        "material": m.group("mat").strip(),
        "dicke": "{} cm".format(m.group("dnum").strip()),
        "oberfläche": m.group("ober").strip(),
    }


def _parse_trennwaende(val):
    val = (val or "").strip().replace(".", ",")
    m = re.match(
        r"^(?P<mat>.+?),\s*(?P<dnum>[\d,]+)\s*cm\s*dick$",
        val,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return {"_raw": val} if val else {}
    return {
        "material": m.group("mat").strip(),
        "dicke": "{} cm".format(m.group("dnum").strip()),
    }


def _parse_decke(val):
    val = (val or "").strip().replace(".", ",")
    m = re.match(
        r"^(?P<typ>.+?),\s*(?P<bau>[^,]+),\s*(?P<dnum>[\d,]+)\s*cm\s*dick\s*;\s*Unterseite\s+(?P<unt>.+)$",
        val,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return {"_raw": val} if val else {}
    return {
        "typ": m.group("typ").strip(),
        "bauweise": m.group("bau").strip(),
        "dicke": "{} cm".format(m.group("dnum").strip()),
        "unterseite": m.group("unt").strip(),
    }


def build_properties_gebaeudekonstruktion_211(flat_kv):
    d = dict(flat_kv)
    props = {}

    v_ls = _take_field_once(d, ["laststufe"])
    if v_ls is not None:
        props["Laststufe (Montage)"] = v_ls

    v_ks = _take_field_once(d, ["konstruktionssystem"])
    if v_ks is not None:
        props["Konstruktionssystem"] = v_ks

    v_dw = _take_field_once(d, ["deckenspannweiten"])
    if v_dw is not None:
        props["Deckenspannweiten"] = _parse_deckenspannweiten(v_dw)

    v_aw = _take_field_once(d, ["aussen", "wand"])
    if v_aw is not None:
        aw = _parse_aussenwaende(v_aw)
        props["Außenwände"] = aw.get("_raw") if len(aw) == 1 and "_raw" in aw else aw

    v_iw = _take_field_once(d, ["innen", "wand"])
    if v_iw is not None:
        iw = _parse_innenwaende(v_iw)
        props["Innenwände"] = iw.get("_raw") if len(iw) == 1 and "_raw" in iw else iw

    v_tw = _take_field_once(d, ["trenn", "wand"])
    if v_tw is not None:
        tw = _parse_trennwaende(v_tw)
        props["Trennwände"] = tw.get("_raw") if len(tw) == 1 and "_raw" in tw else tw

    v_dc = _take_field_once(d, ["decke"], reject_contains=("spannweit",))
    if v_dc is not None:
        dc = _parse_decke(v_dc)
        props["Decke"] = dc.get("_raw") if len(dc) == 1 and "_raw" in dc else dc

    v_fb = _take_field_once(d, ["fussboden", "dicke"]) or _take_field_once(
        d, ["fussbodendicke"]
    )
    if v_fb is not None:
        props["Fußbodendicke"] = v_fb

    for k, v in d.items():
        props[k] = v

    return props


def parse_h3_sections_from_lines(all_lines):
    """
    Scan sorted layout lines: each ``2.1.x Title`` opens a boxed section;
    bullet lines ``key: value`` and continuations attach to it.
    """
    lines_sorted = sorted(all_lines, key=lambda L: (L["bbox"]["y0"], L["bbox"]["x0"]))
    sections = []
    current = None

    for line in lines_sorted:
        text = line.get("text") or ""
        t = layout_strip_bullet_noise(text)

        m = re.match(r"^(\d+\.\d+\.\d+)\s*(.*)$", t.strip())
        if m:
            if current is not None:
                sections.append(current)
            title_tail = (m.group(2) or "").strip()
            current = {"section": m.group(1), "title": title_tail or None, "fields": []}
            continue

        if current is None:
            continue

        ts = t.strip()
        if re.match(r"^\d{1,2}\s+RWE\b", ts, re.I):
            if current["fields"]:
                last = current["fields"][-1]
                sep = ", " if last.get("value") and not last["value"].rstrip().endswith(",") else " "
                last["value"] = (last["value"] + sep + ts).strip()
            continue

        if ":" in ts:
            k, v = layout_split_key_value(ts)
            if k and v:
                current["fields"].append({"key": k, "value": v})
            continue

        if current["fields"]:
            last = current["fields"][-1]
            last["value"] = (last["value"] + " " + layout_strip_bullet_noise(text)).strip()

    if current is not None:
        sections.append(current)
    return sections


def layout_lines_to_structured_sections(all_lines):
    """One JSON object per ``2.1.x`` boxed block; §2.1.1 gets nested walls/deck semantics."""
    out = []
    for s in parse_h3_sections_from_lines(all_lines):
        sid = s["section"]
        title = (s.get("title") or "").strip()
        flat = _fields_to_flat_kv(s["fields"])

        if sid == "2.1.1":
            props = build_properties_gebaeudekonstruktion_211(flat)
        else:
            props = dict(flat)

        props = _normalize_property_values(props)
        out.append({"section": sid, "title": title or None, "properties": props})
    return out


def resolve_pdf_path() -> Path:
    p = Path(PDF_PATH)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _json_rel_path(abs_path: Path) -> str:
    abs_path = abs_path.resolve()
    try:
        return abs_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return abs_path.as_posix()


def content_pdf_pages_to_extract():
    """
    1-based PDF page indices for the page_boxes bundle.
    Env PDF_EXTRACT_PAGES (comma-separated) or PDF_EXTRACT_PAGE overrides CONTENT_PDF_PAGES.
    """
    env_raw = os.environ.get("PDF_EXTRACT_PAGES")
    if env_raw is not None and str(env_raw).strip() != "":
        pages = []
        for part in str(env_raw).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                n = int(part)
                if n > 0:
                    pages.append(n)
            except ValueError:
                pass
        return pages
    lone = os.environ.get("PDF_EXTRACT_PAGE", "").strip()
    if lone:
        try:
            n = int(lone)
            return [n] if n > 0 else []
        except ValueError:
            return []
    return [p for p in CONTENT_PDF_PAGES if isinstance(p, int) and p > 0]


def default_empty_outline():
    """Placeholder outline matching hand-edited page_boxes.json (no outline OCR yet)."""
    return {
        "section_id": None,
        "title": None,
        "source_line": None,
        "fields": [],
    }


def extract_single_page_for_bundle(
    pdf_abs: Path,
    pdf_page_1based: int,
    *,
    dpi: int,
    y_tol: int,
    note,
):
    """
    OCR one PDF page → debug PNG + structured_sections (2.1.x boxes).
    """
    images = convert_from_path(
        str(pdf_abs),
        first_page=pdf_page_1based,
        last_page=pdf_page_1based,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
    )
    image = images[0]
    debug_dir = REPO_ROOT / "data-text-BBSR"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_name = f"debug_page_{pdf_page_1based}.png"
    debug_abs = debug_dir / debug_name
    image.save(str(debug_abs))

    data = pytesseract.image_to_data(image, lang="deu", output_type=Output.DATAFRAME)
    layout_lines = layout_df_to_lines(data, y_tol=y_tol)
    structured_sections = layout_lines_to_structured_sections(layout_lines)

    debug_rel = (Path("data-text-BBSR") / debug_name).as_posix()

    return {
        "pdf_page_index": pdf_page_1based,
        "note": note,
        "dpi": dpi,
        "debug_image": debug_rel,
        "outline": default_empty_outline(),
        "structured_sections": structured_sections,
    }


def build_page_boxes_bundle():
    pdf_abs = resolve_pdf_path()
    if not pdf_abs.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_abs}")

    pages = content_pdf_pages_to_extract()
    if not pages:
        return None

    logical_first = int(
        os.environ.get("LOGICAL_FIRST_PAGE_ON_PDF", str(LOGICAL_FIRST_PAGE_ON_PDF))
    )
    raw_default = os.environ.get("DEFAULT_PAGE_BOX_NOTE")
    if raw_default is not None:
        default_note = raw_default.strip() or None
    else:
        default_note = DEFAULT_PAGE_BOX_NOTE

    extracted = []
    for i, pg in enumerate(pages):
        nnote = None
        if i < len(CONTENT_PDF_PAGE_NOTES):
            raw = CONTENT_PDF_PAGE_NOTES[i]
            nnote = raw if raw else None
        extracted.append(
            extract_single_page_for_bundle(
                pdf_abs,
                pg,
                dpi=CONTENT_PAGE_OCR_DPI,
                y_tol=PAGE_BOX_LAYOUT_Y_TOL,
                note=nnote,
            )
        )

    return {
        "pdf_path": _json_rel_path(pdf_abs),
        "logical_first_page_on_pdf": logical_first,
        "default_note": default_note,
        "pages_requested_pdf_index": list(pages),
        "page_count": len(extracted),
        "pages": extracted,
    }


# RUN PIPELINE

def main():

    pdf_abs = resolve_pdf_path()

    # INDEX 
    print("\n INDEX \n")

    index_df = ocr_page(str(pdf_abs), INDEX_PAGE)
    index_entries = parse_index_from_layout(index_df)

    print(json.dumps(index_entries, indent=2, ensure_ascii=False))

    # Page-box bundle: same JSON shape as data-text-BBSR/page_boxes.json (multi-page)
    if WRITE_PAGE_BOXES_JSON:
        try:
            bundle = build_page_boxes_bundle()
        except FileNotFoundError as exc:
            print(f"\nSkipping page_boxes.json: {exc}\n")
        else:
            if bundle is None:
                print(
                    "\nSkipping page_boxes.json: no pages in CONTENT_PDF_PAGES "
                    "and no PDF_EXTRACT_PAGES / PDF_EXTRACT_PAGE.\n"
                )
            else:
                out_path = REPO_ROOT / PAGE_BOX_JSON_PATH
                out_path.parent.mkdir(parents=True, exist_ok=True)
                dumped = json.dumps(bundle, ensure_ascii=False, indent=2)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(dumped)
                    if not dumped.endswith("\n"):
                        f.write("\n")
                print(
                    f"\nWrote {out_path.relative_to(REPO_ROOT)} — "
                    f"{bundle['page_count']} page(s), "
                    f"pdf indices {bundle['pages_requested_pdf_index']}\n"
                )


if __name__ == "__main__":
    main()





# ########################################################
# # CONFIG

# BASE_DIR = Path(__file__).resolve().parent.parent

# # PDF_PATH = (
# #     BASE_DIR
# #     / "data_source"
# #     / "data_text_BBSR"
# #     / "bbsr_buildings.pdf"
# # )
# LOCAL_DATA_DIR = Path(r"C:\Users\go46wic\intermediat")
# PDF_PATH = LOCAL_DATA_DIR / "bbsr_building.pdf"


# OUTPUT_DIR = (
#     BASE_DIR
#     / "data-text-BBSR"
# )

# DEBUG_DIR = OUTPUT_DIR / "debug_pages"
# JSON_OUTPUT = OUTPUT_DIR / "json_outputs" / "page_boxes.json"

# DEBUG_DIR.mkdir(parents=True, exist_ok=True)
# JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# DPI = 300

# OCR_CONFIG = "--oem 3 --psm 6"

# LANG = "deu"

# # 1-based PDF page numbers to process
# TARGET_PAGES = [11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 57, 61, 65, 69]

# # Optional local path to Poppler bin (contains pdfinfo.exe / pdftoppm.exe)
# # Example: BASE_DIR / "tools" / "poppler" / "Library" / "bin"
# POPPLER_PATH = r"C:\Users\go46wic\Dropbox\_PhD\11_KnowledgeBaseLCA\NeuroSymbolicReasoningforLCA\tools\Release-26.02.0-0\poppler-26.02.0\Library\bin"

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# if not PDF_PATH.exists():
#     raise FileNotFoundError(
#         f"\n[ERROR] PDF File missing!\n"
#         f"Could not find 'bbsr_buildings.pdf' at: {PDF_PATH}\n"
#         f"Please verify the file has been copied out of Dropbox into that exact path."
#     )

# if not Path(POPPLER_PATH).exists():
#     raise FileNotFoundError(
#         f"\n[ERROR] Poppler path path is broken!\n"
#         f"The folder does not exist: {POPPLER_PATH}\n"
#         f"Please verify your extraction layout."
#     )


# ########################################################
# # OCR HELPERS


# def clean_text(text):

#     if text is None:
#         return None

#     text = text.replace("\x0c", " ")
#     text = text.replace("|", "I")

#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     return text.strip()


# def image_to_text(image):

#     text = pytesseract.image_to_string(
#         image,
#         lang=LANG,
#         config=OCR_CONFIG
#     )

#     return clean_text(text)


# def image_to_data(image):

#     return pytesseract.image_to_data(
#         image,
#         lang=LANG,
#         config=OCR_CONFIG,
#         output_type=pytesseract.Output.DICT
#     )


# ########################################################
# # PAGE LAYOUT DETECTION


# def preprocess_for_layout(cv_img):

#     gray = cv2.cvtColor(
#         cv_img,
#         cv2.COLOR_BGR2GRAY
#     )

#     blur = cv2.GaussianBlur(
#         gray,
#         (3, 3),
#         0
#     )

#     thresh = cv2.threshold(
#         blur,
#         180,
#         255,
#         cv2.THRESH_BINARY_INV
#     )[1]

#     return thresh


# def detect_layout_boxes(cv_img):

#     thresh = preprocess_for_layout(cv_img)

#     kernel = cv2.getStructuringElement(
#         cv2.MORPH_RECT,
#         (25, 8)
#     )

#     dilated = cv2.dilate(
#         thresh,
#         kernel,
#         iterations=2
#     )

#     contours, _ = cv2.findContours(
#         dilated,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     boxes = []

#     for contour in contours:

#         x, y, w, h = cv2.boundingRect(contour)

#         area = w * h

#         if area < 2500:
#             continue

#         if h < 20:
#             continue

#         boxes.append({
#             "x": int(x),
#             "y": int(y),
#             "w": int(w),
#             "h": int(h),
#             "area": int(area)
#         })

#     boxes = sorted(
#         boxes,
#         key=lambda b: (b["y"], b["x"])
#     )

#     return boxes


# ########################################################
# # OCR LINE GROUPING


# def group_ocr_lines(ocr_dict):

#     lines = {}

#     n = len(ocr_dict["text"])

#     for i in range(n):

#         text = clean_text(
#             ocr_dict["text"][i]
#         )

#         if not text:
#             continue

#         block = ocr_dict["block_num"][i]
#         par = ocr_dict["par_num"][i]
#         line = ocr_dict["line_num"][i]

#         key = (block, par, line)

#         if key not in lines:

#             lines[key] = {
#                 "text": [],
#                 "x": [],
#                 "y": [],
#                 "w": [],
#                 "h": []
#             }

#         lines[key]["text"].append(text)

#         lines[key]["x"].append(
#             ocr_dict["left"][i]
#         )

#         lines[key]["y"].append(
#             ocr_dict["top"][i]
#         )

#         lines[key]["w"].append(
#             ocr_dict["width"][i]
#         )

#         lines[key]["h"].append(
#             ocr_dict["height"][i]
#         )

#     grouped = []

#     for key, val in lines.items():

#         grouped.append({
#             "text": clean_text(
#                 " ".join(val["text"])
#             ),
#             "x": min(val["x"]),
#             "y": min(val["y"]),
#             "w": max(val["w"]),
#             "h": max(val["h"])
#         })

#     grouped = sorted(
#         grouped,
#         key=lambda l: (l["y"], l["x"])
#     )

#     return grouped


# ########################################################
# # PAGE HEADER EXTRACTION


# MAIN_SECTION_PATTERN = re.compile(
#     r"^(\d+)\s+(.*)$"
# )

# SUBSERIES_PATTERN = re.compile(
#     r"^(\d+\.\d+)\s+(.*)$"
# )


# def extract_page_header(lines):

#     result = {
#         "main_section": {
#             "section": None,
#             "title": None
#         },
#         "subseries": {
#             "section": None,
#             "title": None
#         }
#     }

#     for line in lines:

#         text = line["text"]

#         main_match = MAIN_SECTION_PATTERN.match(text)

#         if main_match:

#             result["main_section"] = {
#                 "section": main_match.group(1),
#                 "title": clean_text(
#                     main_match.group(2)
#                 )
#             }

#             continue

#         sub_match = SUBSERIES_PATTERN.match(text)

#         if sub_match:

#             result["subseries"] = {
#                 "section": sub_match.group(1),
#                 "title": clean_text(
#                     sub_match.group(2)
#                 )
#             }

#     return result


# ########################################################
# # STRUCTURED SECTION PARSER


# SECTION_PATTERN = re.compile(
#     r"^(\d+\.\d+\.\d+)\s+(.*)$"
# )

# PROPERTY_PATTERN = re.compile(
#     r"^([^:]+):\s*(.+)$"
# )


# def parse_property_value(value):

#     if value is None:
#         return None

#     value = clean_text(value)

#     if ";" in value and len(value) > 50:

#         return [
#             clean_text(v)
#             for v in value.split(";")
#             if clean_text(v)
#         ]

#     if "," in value and re.search(r"\d,\d+\s*m", value):

#         return [
#             clean_text(v)
#             for v in value.split(",")
#             if clean_text(v)
#         ]

#     return value


# def parse_nested_decke(value):

#     if value is None:
#         return None

#     result = {}

#     value = clean_text(value)

#     result["raw"] = value

#     thickness_match = re.search(
#         r"(\d+[,\.\d]*)\s*cm",
#         value
#     )

#     if thickness_match:

#         result["dicke"] = (
#             thickness_match.group(1)
#             + " cm"
#         )

#     if "zweischalig" in value.lower():

#         result["bauweise"] = "zweischalig"

#     if "stahlbeton" in value.lower():

#         result["material"] = "Stahlbeton"

#     return result


# def parse_structured_sections(lines):

#     sections = []

#     current_section = None

#     current_property = None

#     for line in lines:

#         text = clean_text(line["text"])

#         if not text:
#             continue

#         section_match = SECTION_PATTERN.match(text)

#         if section_match:

#             if current_section:
#                 sections.append(current_section)

#             current_section = {
#                 "section": section_match.group(1),
#                 "title": clean_text(
#                     section_match.group(2)
#                 ),
#                 "properties": {}
#             }

#             current_property = None

#             continue

#         prop_match = PROPERTY_PATTERN.match(text)

#         if prop_match and current_section:

#             key = clean_text(
#                 prop_match.group(1)
#             )

#             value = clean_text(
#                 prop_match.group(2)
#             )

#             if key.lower() == "decke":

#                 parsed = parse_nested_decke(value)

#                 current_section["properties"][key] = parsed

#             else:

#                 current_section["properties"][key] = (
#                     parse_property_value(value)
#                 )

#             current_property = key

#             continue

#         if current_property and current_section:

#             prev = current_section["properties"][
#                 current_property
#             ]

#             if isinstance(prev, str):

#                 current_section["properties"][
#                     current_property
#                 ] = clean_text(
#                     prev + " " + text
#                 )

#     if current_section:

#         sections.append(current_section)

#     return sections


# ########################################################
# # FINAL JSON SHAPING


# DATE_PATTERN = re.compile(r"\b\d{2}/\d{2}\b")
# PAGE_PATTERN = re.compile(r"\bSeite\s+\d+\b", re.IGNORECASE)


# def parse_document_info(lines):

#     clean_lines = [
#         clean_text(line["text"])
#         for line in lines
#         if clean_text(line["text"])
#     ]

#     organization = None
#     title = None
#     subtitle = None
#     date = None
#     page = None

#     if clean_lines:
#         organization = clean_lines[0]

#     # Keep top metadata lines before first section id (e.g. 6.3.1 ...)
#     section_start_index = None
#     for idx, line in enumerate(clean_lines):
#         if SECTION_PATTERN.match(line):
#             section_start_index = idx
#             break

#     head_lines = clean_lines[:section_start_index] if section_start_index else clean_lines[:6]

#     # Extract date/page markers from header lines
#     for line in head_lines:
#         date_match = DATE_PATTERN.search(line)
#         if date_match and date is None:
#             date = date_match.group(0)

#         page_match = PAGE_PATTERN.search(line)
#         if page_match and page is None:
#             page = page_match.group(0)

#     # Candidate lines for title/subtitle
#     filtered = []
#     for line in head_lines:
#         if PAGE_PATTERN.search(line) or DATE_PATTERN.search(line):
#             continue
#         if MAIN_SECTION_PATTERN.match(line):
#             continue
#         if SUBSERIES_PATTERN.match(line):
#             continue
#         filtered.append(line)

#     if len(filtered) >= 2:
#         title = filtered[1]
#     elif len(filtered) == 1:
#         title = filtered[0]

#     if len(filtered) >= 3:
#         subtitle = filtered[2]

#     return {
#         "organization": organization,
#         "title": title,
#         "subtitle": subtitle,
#         "date": date,
#         "page": page
#     }


# def to_target_json(document_info, structured_sections):

#     sections = []

#     for sec in structured_sections:
#         sections.append({
#             "id": sec["section"],
#             "title": sec["title"],
#             "data": sec["properties"]
#         })

#     return {
#         "document_info": document_info,
#         "sections": sections
#     }


# ########################################################
# # MAIN OCR PIPELINE


# print(f"Loading PDF:\n{PDF_PATH}")

# results: List[Dict[str, Any]] = []

# for pdf_page_number in TARGET_PAGES:

#     print(f"\nProcessing PDF page {pdf_page_number}")

#     try:
#         page_images = convert_from_path(
#             PDF_PATH,
#             dpi=DPI,
#             first_page=pdf_page_number,
#             last_page=pdf_page_number,
#             poppler_path=str(POPPLER_PATH) if POPPLER_PATH else None
#         )
#     except PDFInfoNotInstalledError as e:
#         raise SystemExit(
#             "pdf2image could not find Poppler (pdfinfo.exe).\n"
#             "Install Poppler and either add its bin directory to PATH,\n"
#             "or set POPPLER_PATH in this script."
#         ) from e

#     if not page_images:
#         print(f"Skipped PDF page {pdf_page_number}: no image generated.")
#         continue

#     pil_page = page_images[0]

#     cv_img = cv2.cvtColor(
#         np.array(pil_page),
#         cv2.COLOR_RGB2BGR
#     )

#     page_debug_path = (
#         DEBUG_DIR
#         / f"page_{pdf_page_number}_raw.png"
#     )

#     pil_page.save(page_debug_path)

#     layout_boxes = detect_layout_boxes(
#         cv_img
#     )

#     page_lines = []

#     page_boxes = []

#     debug_img = cv_img.copy()

#     for box_id, box in enumerate(layout_boxes):

#         x = box["x"]
#         y = box["y"]
#         w = box["w"]
#         h = box["h"]

#         crop = cv_img[
#             y:y+h,
#             x:x+w
#         ]

#         crop_pil = Image.fromarray(
#             cv2.cvtColor(
#                 crop,
#                 cv2.COLOR_BGR2RGB
#             )
#         )

#         ocr_dict = image_to_data(
#             crop_pil
#         )

#         grouped_lines = group_ocr_lines(
#             ocr_dict
#         )

#         box_text = "\n".join([
#             l["text"]
#             for l in grouped_lines
#         ])

#         page_lines.extend(grouped_lines)

#         page_boxes.append({

#             "box_id": box_id,

#             "bbox": {
#                 "x": x,
#                 "y": y,
#                 "w": w,
#                 "h": h
#             },

#             "text": clean_text(
#                 box_text
#             )
#         })

#         cv2.rectangle(
#             debug_img,
#             (x, y),
#             (x+w, y+h),
#             (0, 255, 0),
#             2
#         )

#     page_lines = sorted(
#         page_lines,
#         key=lambda l: (l["y"], l["x"])
#     )

#     page_header = extract_page_header(
#         page_lines
#     )

#     structured_sections = parse_structured_sections(
#         page_lines
#     )

#     debug_boxed_path = (
#         DEBUG_DIR
#         / f"page_{pdf_page_number}_boxed.png"
#     )

#     cv2.imwrite(
#         str(debug_boxed_path),
#         debug_img
#     )

#     document_info = parse_document_info(page_lines)

#     page_record = to_target_json(
#         document_info=document_info,
#         structured_sections=structured_sections
#     )
#     page_record["pdf_page"] = pdf_page_number
#     page_record["debug_image_raw"] = str(page_debug_path.relative_to(BASE_DIR))
#     page_record["debug_image_boxed"] = str(debug_boxed_path.relative_to(BASE_DIR))
#     page_record["detected_box_count"] = len(page_boxes)
#     page_record["page_header"] = page_header

#     results.append(page_record)


# ########################################################
# # EXPORT JSON


# bundle = {

#     "source_pdf": str(
#         PDF_PATH.relative_to(BASE_DIR)
#     ),

#     "processed_pdf_pages": TARGET_PAGES,
#     "pages": results
# }

# with open(
#     JSON_OUTPUT,
#     "w",
#     encoding="utf-8"
# ) as f:

#     json.dump(
#         bundle,
#         f,
#         indent=2,
#         ensure_ascii=False
#     )

# print(
#     f"\nWrote JSON:\n{JSON_OUTPUT}"
# )

# print(
#     f"\nProcessed {len(results)} pages."
# )