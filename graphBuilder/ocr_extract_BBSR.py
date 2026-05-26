import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


########################################################
# CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = (
    BASE_DIR
    / "data-text-BBSR"
    / "pdf"
    / "bbsr_buildings.pdf"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data-text-BBSR"
)

DEBUG_DIR = OUTPUT_DIR / "debug_pages"
JSON_OUTPUT = OUTPUT_DIR / "json_outputs" / "page_boxes.json"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

DPI = 300

OCR_CONFIG = "--oem 3 --psm 6"

LANG = "deu"


########################################################
# OCR HELPERS


def clean_text(text):

    if text is None:
        return None

    text = text.replace("\x0c", " ")
    text = text.replace("|", "I")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def image_to_text(image):

    text = pytesseract.image_to_string(
        image,
        lang=LANG,
        config=OCR_CONFIG
    )

    return clean_text(text)


def image_to_data(image):

    return pytesseract.image_to_data(
        image,
        lang=LANG,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT
    )


########################################################
# PAGE LAYOUT DETECTION


def preprocess_for_layout(cv_img):

    gray = cv2.cvtColor(
        cv_img,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    thresh = cv2.threshold(
        blur,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    return thresh


def detect_layout_boxes(cv_img):

    thresh = preprocess_for_layout(cv_img)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 8)
    )

    dilated = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        if area < 2500:
            continue

        if h < 20:
            continue

        boxes.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area)
        })

    boxes = sorted(
        boxes,
        key=lambda b: (b["y"], b["x"])
    )

    return boxes


########################################################
# OCR LINE GROUPING


def group_ocr_lines(ocr_dict):

    lines = {}

    n = len(ocr_dict["text"])

    for i in range(n):

        text = clean_text(
            ocr_dict["text"][i]
        )

        if not text:
            continue

        block = ocr_dict["block_num"][i]
        par = ocr_dict["par_num"][i]
        line = ocr_dict["line_num"][i]

        key = (block, par, line)

        if key not in lines:

            lines[key] = {
                "text": [],
                "x": [],
                "y": [],
                "w": [],
                "h": []
            }

        lines[key]["text"].append(text)

        lines[key]["x"].append(
            ocr_dict["left"][i]
        )

        lines[key]["y"].append(
            ocr_dict["top"][i]
        )

        lines[key]["w"].append(
            ocr_dict["width"][i]
        )

        lines[key]["h"].append(
            ocr_dict["height"][i]
        )

    grouped = []

    for key, val in lines.items():

        grouped.append({
            "text": clean_text(
                " ".join(val["text"])
            ),
            "x": min(val["x"]),
            "y": min(val["y"]),
            "w": max(val["w"]),
            "h": max(val["h"])
        })

    grouped = sorted(
        grouped,
        key=lambda l: (l["y"], l["x"])
    )

    return grouped


########################################################
# PAGE HEADER EXTRACTION


MAIN_SECTION_PATTERN = re.compile(
    r"^(\d+)\s+(.*)$"
)

SUBSERIES_PATTERN = re.compile(
    r"^(\d+\.\d+)\s+(.*)$"
)


def extract_page_header(lines):

    result = {
        "main_section": {
            "section": None,
            "title": None
        },
        "subseries": {
            "section": None,
            "title": None
        }
    }

    for line in lines:

        text = line["text"]

        main_match = MAIN_SECTION_PATTERN.match(text)

        if main_match:

            result["main_section"] = {
                "section": main_match.group(1),
                "title": clean_text(
                    main_match.group(2)
                )
            }

            continue

        sub_match = SUBSERIES_PATTERN.match(text)

        if sub_match:

            result["subseries"] = {
                "section": sub_match.group(1),
                "title": clean_text(
                    sub_match.group(2)
                )
            }

    return result


########################################################
# STRUCTURED SECTION PARSER


SECTION_PATTERN = re.compile(
    r"^(\d+\.\d+\.\d+)\s+(.*)$"
)

PROPERTY_PATTERN = re.compile(
    r"^([^:]+):\s*(.+)$"
)


def parse_property_value(value):

    if value is None:
        return None

    value = clean_text(value)

    if ";" in value and len(value) > 50:

        return [
            clean_text(v)
            for v in value.split(";")
            if clean_text(v)
        ]

    if "," in value and re.search(r"\d,\d+\s*m", value):

        return [
            clean_text(v)
            for v in value.split(",")
            if clean_text(v)
        ]

    return value


def parse_nested_decke(value):

    if value is None:
        return None

    result = {}

    value = clean_text(value)

    result["raw"] = value

    thickness_match = re.search(
        r"(\d+[,\.\d]*)\s*cm",
        value
    )

    if thickness_match:

        result["dicke"] = (
            thickness_match.group(1)
            + " cm"
        )

    if "zweischalig" in value.lower():

        result["bauweise"] = "zweischalig"

    if "stahlbeton" in value.lower():

        result["material"] = "Stahlbeton"

    return result


def parse_structured_sections(lines):

    sections = []

    current_section = None

    current_property = None

    for line in lines:

        text = clean_text(line["text"])

        if not text:
            continue

        section_match = SECTION_PATTERN.match(text)

        if section_match:

            if current_section:
                sections.append(current_section)

            current_section = {
                "section": section_match.group(1),
                "title": clean_text(
                    section_match.group(2)
                ),
                "properties": {}
            }

            current_property = None

            continue

        prop_match = PROPERTY_PATTERN.match(text)

        if prop_match and current_section:

            key = clean_text(
                prop_match.group(1)
            )

            value = clean_text(
                prop_match.group(2)
            )

            if key.lower() == "decke":

                parsed = parse_nested_decke(value)

                current_section["properties"][key] = parsed

            else:

                current_section["properties"][key] = (
                    parse_property_value(value)
                )

            current_property = key

            continue

        if current_property and current_section:

            prev = current_section["properties"][
                current_property
            ]

            if isinstance(prev, str):

                current_section["properties"][
                    current_property
                ] = clean_text(
                    prev + " " + text
                )

    if current_section:

        sections.append(current_section)

    return sections


########################################################
# MAIN OCR PIPELINE


print(f"Loading PDF:\n{PDF_PATH}")

pages = convert_from_path(
    PDF_PATH,
    dpi=DPI
)

results = []

for page_index, pil_page in enumerate(pages):

    print(f"\nProcessing page {page_index}")

    cv_img = cv2.cvtColor(
        np.array(pil_page),
        cv2.COLOR_RGB2BGR
    )

    page_debug_path = (
        DEBUG_DIR
        / f"debug_page_{page_index}.png"
    )

    pil_page.save(page_debug_path)

    layout_boxes = detect_layout_boxes(
        cv_img
    )

    page_lines = []

    page_boxes = []

    debug_img = cv_img.copy()

    for box_id, box in enumerate(layout_boxes):

        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        crop = cv_img[
            y:y+h,
            x:x+w
        ]

        crop_pil = Image.fromarray(
            cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )
        )

        ocr_dict = image_to_data(
            crop_pil
        )

        grouped_lines = group_ocr_lines(
            ocr_dict
        )

        box_text = "\n".join([
            l["text"]
            for l in grouped_lines
        ])

        page_lines.extend(grouped_lines)

        page_boxes.append({

            "box_id": box_id,

            "bbox": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            },

            "text": clean_text(
                box_text
            )
        })

        cv2.rectangle(
            debug_img,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    page_lines = sorted(
        page_lines,
        key=lambda l: (l["y"], l["x"])
    )

    page_header = extract_page_header(
        page_lines
    )

    structured_sections = parse_structured_sections(
        page_lines
    )

    debug_boxed_path = (
        DEBUG_DIR
        / f"boxed_page_{page_index}.png"
    )

    cv2.imwrite(
        str(debug_boxed_path),
        debug_img
    )

    page_record = {

        "pdf_page_index": page_index,

        "note": None,

        "dpi": DPI,

        "page_header": page_header,

        "debug_image": str(
            debug_boxed_path.relative_to(BASE_DIR)
        ),

        "outline": {
            "section_id": None,
            "title": None,
            "source_line": None,
            "fields": []
        },

        "boxes": page_boxes,

        "structured_sections": structured_sections
    }

    results.append(page_record)


########################################################
# EXPORT JSON


bundle = {

    "source_pdf": str(
        PDF_PATH.relative_to(BASE_DIR)
    ),

    "pages": results
}

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        bundle,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"\nWrote JSON:\n{JSON_OUTPUT}"
)

print(
    f"\nProcessed {len(results)} pages."
)