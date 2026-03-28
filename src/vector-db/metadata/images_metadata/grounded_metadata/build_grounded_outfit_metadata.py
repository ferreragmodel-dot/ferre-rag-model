#!/usr/bin/env python3
"""Build grounded (expert) metadata by linking Technical descriptions PDFs to season images.

Goal
----
Given a season folder like:
  input-datasets/ferre-designs/Dataset DataShack 2026/ALTA MODA 1987 SS
this script finds each pair:
  Technical descriptions/<id>.pdf        (text fields)
  Technical descriptions/<id>_1.pdf      (images grouped by red section headers)

It extracts:
- structured fields from <id>.pdf using the fact that labels are red
- images from <id>_1.pdf grouped by section headers (also red)
- matches those embedded images to the real images stored in the season folders
  (even with subfolders) using a perceptual hash (dHash)

Outputs:
- outfit-centric JSON: list of outfit records
- optional image-centric JSON: mapping from image rel_path -> outfit metadata

This is designed to run inside the existing ferre-rag-model repo (fitz is already used there).
"""

from __future__ import annotations

import argparse
import io
import json
from operator import inv
import os
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image


# -------------------------
# Canonical fields (xxx.pdf)
# -------------------------

FIELD_ALIASES: Dict[str, str] = {
    # left: normalized label text -> canonical key
    "file": "file",
    "picture": "picture",
    "object": "object",
    "designer": "designer",
    "designe": "designer",
    "design": "designer",
    "label": "label",
    "inventory": "inventory",
    "year": "year",
    "season": "season",
    "collection": "collection",
    "colleciton": "collection",
    "look": "look",
    "size": "size",
    "materials": "materials",
    "material": "materials",
    "working process": "working_process",
    "workingprocess": "working_process",
    "description": "description",
    "remark": "remark",
    "remarks": "remark",
    "source": "source",
    "bibliography": "bibliography",
    "exhibitions": "exhibitions",
    "exhibition": "exhibitions",
    "echibitions": "exhibitions",
    "acquisition": "acquisition",
    "acquistion": "acquisition",
    "author of file": "author_of_file",
    "present location": "present_location",
    "date of file": "date_of_file",
    "condition": "condition",
    "working processe": "working_process",
    "working proces": "working_process",
    "sources": "source",
    "sourc": "source",
    # --- Italian labels -> canonical English keys ---
    "acquisizione": "acquisition",
    "autore scheda": "author_of_file",
    "bibliografia": "bibliography",
    "collezione": "collection",
    "stato conservazione": "condition",
    "data scheda": "date_of_file",
    "descrizione": "description",
    "stilista": "designer",
    "mostre": "exhibitions",
    "scheda": "file",          # "File" label in Italian PDFs
    "inventario": "inventory",
    "etichetta": "label",
    "materiali": "materials",
    "oggetto": "object",
    "immagine": "picture",
    "luogo conservazione": "present_location",
    "note": "remark",
    "stagione": "season",
    "taglia": "size",
    "fonti": "source",
    "lavorazioni": "working_process",
    "anno": "year",
}

ITALIAN_FIELD_LABELS = {
    "acquisizione", "autore scheda", "bibliografia", "collezione", "stato conservazione",
    "data scheda", "descrizione", "stilista", "mostre", "scheda", "inventario", "etichetta",
    "materiali", "oggetto", "immagine", "luogo conservazione", "note", "stagione",
    "taglia", "fonti", "lavorazioni", "anno",
}

# --------------------------
# Section fields (xxx_1.pdf)
# --------------------------

SECTION_ALIASES: Dict[str, str] = {
    # red headings in xxx_1.pdf -> canonical section key
    "runway shots": "runway_shots",
    "runaway shots": "runway_shots",  # common typo
    "technical file": "technical_files",
    "technical files": "technical_files",
    "technical drawings": "technical_drawings",
    "fashion show drawings": "fashion_show_drawings",
    "ad image": "ad_image",
    "advertising": "ad_image",
    "editorial image": "editorial_image",
    "press coverage": "editorial_image",
    "catwalk drawings": "fashion_show_drawings",
    "catwalk drawing": "fashion_show_drawings",
}

# Folder name (in season) -> section key
FOLDER_TO_SECTION: Dict[str, str] = {
    "fashion show photos": "runway_shots",
    "technical sheets": "technical_files",
    "technical drawings": "technical_drawings",
    "catwalk drawings": "fashion_show_drawings",
    "advertising": "ad_image",
    "press coverage": "editorial_image",
    "fashion show drawings": "fashion_show_drawings",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PRIMARY_DATASET_ROOT_NAME = "dataset datashack 2026"
DATASET_ROOT_NAMES = {PRIMARY_DATASET_ROOT_NAME}


@dataclass
class ImageMatch:
    rel_path: str
    id: str
    dist: int


# -------------------------
# Utility: colors and text
# -------------------------

def _norm(s: str) -> str:
    """Normalize text for robust matching.

    Handles common PDF ligatures like "ﬁ" (fi) and "ﬂ" (fl).
    """
    s = s.replace("\u00a0", " ")  # nbsp
    # common ligatures
    s = (
        s.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("ﬀ", "ff")
        .replace("ﬃ", "ffi")
        .replace("ﬄ", "ffl")
    )
    return re.sub(r"\s+", " ", s.strip().lower())


def _is_red_color_int(color: int, *, r_min: int = 180, g_max: int = 90, b_max: int = 90) -> bool:
    """Heuristic for 'red' text in PDF spans.

    PyMuPDF span['color'] is usually 0xRRGGBB.
    """
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF
    return (r >= r_min) and (g <= g_max) and (b <= b_max)


def _iter_lines_with_style(page_dict: Dict[str, Any]) -> Iterable[Tuple[str, bool, Tuple[float, float, float, float]]]:
    """Yield (line_text, is_red_majority, bbox) for each line in a page."""
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(sp.get("text", "") for sp in spans).strip()
            if not text:
                continue
            red_votes = 0
            total_votes = 0
            for sp in spans:
                t = sp.get("text", "")
                if not t.strip():
                    continue
                total_votes += 1
                if _is_red_color_int(int(sp.get("color", 0))):
                    red_votes += 1
            is_red = (total_votes > 0) and (red_votes / total_votes >= 0.6)
            bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
            yield text, is_red, bbox

def _cluster_x_positions(xs: List[float], gap: float = 90.0) -> List[float]:
    """Cluster x positions into column centers (works for 1/2/3 columns)."""
    if not xs:
        return [0.0]
    xs = sorted(xs)
    clusters: List[List[float]] = []
    for x in xs:
        if not clusters or abs(x - clusters[-1][-1]) > gap:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    return [sum(c) / len(c) for c in clusters]

def _infer_column_centers_for_doc(doc: fitz.Document) -> List[float]:
    """Infer column centers from the x position of red LABEL lines across the whole doc."""
    xs: List[float] = []
    for page in doc:
        page_dict = page.get_text("dict")
        for text, is_red, bbox in _iter_lines_with_style(page_dict):
            if not is_red:
                continue
            norm_label = re.sub(r"[:\-]+$", "", _norm(text)).strip()
            if norm_label in FIELD_ALIASES:
                xs.append(float(bbox[0]))
    centers = _cluster_x_positions(xs, gap=90.0)
    return centers if centers else [0.0]

# -------------------------
# Perceptual hashing (dHash)
# -------------------------

def dhash_from_pil(img: Image.Image, hash_size: int = 8) -> int:
    """Difference hash (dHash). Returns an integer bitstring."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    try:
        pixels = list(img.get_flattened_data())
    except Exception:
        pixels = list(img.getdata())
    # row-major access
    h = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            h = (h << 1) | int(left > right)
    return h


def dhash_from_file(path: str) -> int:
    with Image.open(path) as im:
        return dhash_from_pil(im)


def dhash_from_bytes(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as im:
        return dhash_from_pil(im)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# -------------------------
# Extract fields from xxx.pdf
# -------------------------

def extract_fields_from_technical_pdf(pdf_path: str) -> Tuple[Dict[str, Optional[str]], str, str]:
    """Extract key-value fields from <id>.pdf.

    Strategy:
    - identify red label lines
    - treat all subsequent non-red lines as content for the current label
      until the next label appears (even across pages)

    Returns (fields, raw_text).
    """
    doc = fitz.open(pdf_path)

    fields: Dict[str, List[str]] = {v: [] for v in set(FIELD_ALIASES.values())}

    col_centers = _infer_column_centers_for_doc(doc)
    current_key_by_col: Dict[int, Optional[str]] = {i: None for i in range(len(col_centers))}

    # raw text (debugging / fallback)
    raw_text_parts: List[str] = []
    found_it_label = False

    for page in doc:
        page_dict = page.get_text("dict")

        # keep raw text as a fallback
        raw_text_parts.append(page.get_text("text"))

        # build a list of lines in reading order (top->bottom)
        lines = list(_iter_lines_with_style(page_dict))
        lines.sort(key=lambda x: (x[2][1], x[2][0]))  # y0 then x0

        for text, is_red, bbox in lines:
            norm_txt = _norm(text)
            norm_label = re.sub(r"[:\-]+$", "", norm_txt).strip()

            x0 = float(bbox[0]) if bbox and len(bbox) >= 1 else 0.0
            col = min(range(len(col_centers)), key=lambda i: abs(x0 - col_centers[i]))

            # label line (red)
            if is_red and norm_label in FIELD_ALIASES:
                current_key_by_col[col] = FIELD_ALIASES[norm_label]
                if norm_label in ITALIAN_FIELD_LABELS:
                    found_it_label = True
                continue
            
            # ignore other red lines
            if is_red:
                continue
            
            current_key = current_key_by_col[col]
            if current_key is None:
                continue
            
            cleaned = text.strip()

            # footer / noise filters (EN + IT)
            if re.search(
                r"\bPrinted on\b|\bStampato il\b|Fondazione\s+Gianfranco\s+Ferr[ée]|All rights reserved|Tutti i diritti riservati|©",
                cleaned,
                flags=re.IGNORECASE,
            ):
                continue
            # page markers (EN + IT)
            if re.match(r"^\s*(Pg\.?|Pag\.?)\s*#?\s*\d+\s*$", cleaned, flags=re.IGNORECASE):
                continue
            # sometimes page number is captured as just "1" (usually bottom margin)
            if re.match(r"^\s*\d+\s*$", cleaned) and (bbox[1] > page.rect.height - 80):
                continue
            
            if cleaned:
                fields[current_key].append(cleaned)

    # convert list-of-lines to string / None
    out: Dict[str, Optional[str]] = {}
    singleline_keys = {"description", "remark", "materials", "working_process"}

    for k, parts in fields.items():
        cleaned_parts = [p.strip() for p in parts if p and p.strip()]
        if k in singleline_keys:
            joined = " ".join(cleaned_parts)
            joined = re.sub(r"\s+", " ", joined).strip()
        else:
            joined = "\n".join(cleaned_parts).strip()

        out[k] = joined if joined else None

    raw_text = "\n".join(raw_text_parts).strip()
    lang = "it" if found_it_label else "en"
    return out, raw_text, lang    

def cleanup_fields(fields: Dict[str, Optional[str]], season_folder_name: str) -> None:
    # helper: join non-empty lines
    def lines(v: Optional[str]) -> List[str]:
        return [x.strip() for x in (v or "").splitlines() if x.strip()]

    # Remove footer/page noise that sometimes slips into fields (EN + IT)
    noise_line = re.compile(
        r"(Printed on|Stampato il|All rights reserved|Tutti i diritti riservati|Fondazione\s+Gianfranco\s+Ferr[ée]|©)",
        flags=re.IGNORECASE,
    )
    page_line = re.compile(r"^\s*(Pg\.?|Pag\.?)\s*#?\s*\d+\s*$", flags=re.IGNORECASE)

    for k, v in list(fields.items()):
        if not v:
            continue
        kept = []
        for ln in v.splitlines():
            s = ln.strip()
            if not s:
                continue
            if noise_line.search(s):
                continue
            if page_line.match(s):
                continue
            kept.append(s)
        fields[k] = "\n".join(kept).strip() or None

    # FILE: keep only the numeric id (e.g., "N. 301\nN. 301" -> "301")
    if fields.get("file"):
        m = re.search(r"\b(\d{1,6})\b", fields["file"])
        fields["file"] = m.group(1) if m else (lines(fields["file"])[0] if lines(fields["file"]) else fields["file"])

    # YEAR: if it contains designer + year, keep year numeric and move the other part to designer if missing
    if fields.get("year"):
        yy = re.search(r"\b(19\d{2}|20\d{2})\b", fields["year"])
        if yy:
            year_val = yy.group(1)
            other = [l for l in lines(fields["year"]) if not re.search(r"\b(19\d{2}|20\d{2})\b", l)]
            fields["year"] = year_val
            if other and not fields.get("designer"):
                fields["designer"] = " ".join(other)

    # SEASON normalization (Spring - Summer -> Spring-Summer, Fall - Winter -> Fall-Winter)
    if fields.get("season"):
        t = _norm(fields["season"])
        if "spring" in t or "summer" in t:
            fields["season"] = "Spring-Summer"
        elif "fall" in t or "winter" in t:
            fields["season"] = "Fall-Winter"

    # If season missing, infer from season folder like "SS1987" / "FW1986-87"
    if not fields.get("season"):
        sn = _norm(season_folder_name)
        if sn.startswith("ss"):
            fields["season"] = "Spring-Summer"
        elif sn.startswith("fw"):
            fields["season"] = "Fall-Winter"

    # LABEL: if label begins with season line, move that to season and keep the rest as label
    if fields.get("label"):
        lns = lines(fields["label"])
        if lns:
            s0 = _norm(lns[0])
            if ("spring" in s0 or "summer" in s0):
                fields["season"] = "Spring-Summer"
                fields["label"] = "\n".join(lns[1:]).strip() or None
            elif ("fall" in s0 or "winter" in s0):
                fields["season"] = "Fall-Winter"
                fields["label"] = "\n".join(lns[1:]).strip() or None

    # SIZE vs INVENTORY semantics (your desired meaning):
    # if inventory is digits and size is text like "made to measure", swap them
    inv = (fields.get("inventory") or "").strip()
    sz = (fields.get("size") or "").strip()
    if inv and re.fullmatch(r"\d+", inv) and sz and re.search(r"[A-Za-z]", sz):
        fields["inventory"], fields["size"] = inv, sz

    # Also strip accidental page markers from size/inventory
    # (but keep real size codes like 10301)
    for k in ("inventory", "size"):
        if fields.get(k):
            lns = []
            for x in lines(fields[k]):
                # drop only small page-number-like tokens (1-2 digits), optionally preceded by "Pg"
                if re.match(r"^(Pg\.?\s*#?\s*)?\d{1,2}$", x, flags=re.IGNORECASE):
                    continue
                lns.append(x)
            fields[k] = "\n".join(lns).strip() or None

    # REMARK vs SOURCE: if source missing but last remark line looks like a source, move it
    if fields.get("remark") and not fields.get("source"):
        txt = fields["remark"].strip()

        m = re.search(
            r"^(.*?)(\s+((?:S/S|F/W)\s*\d{4}.*?(?:design specs|design specifications).*))$",
            txt,
            flags=re.IGNORECASE,
        )
        if m:
            fields["remark"] = m.group(1).strip() or None
            fields["source"] = m.group(3).strip() or None

# ---------------------------------
# Extract sections+images from xxx_1
# ---------------------------------

def _extract_image_blocks_with_xref(page: fitz.Page) -> List[Dict[str, Any]]:
    """Try to get image blocks with bbox and xref.

    Works on many PDFs with page.get_text('rawdict').
    """
    try:
        d = page.get_text("rawdict")
    except Exception:
        d = page.get_text("dict")

    blocks = d.get("blocks", [])
    img_blocks = [b for b in blocks if b.get("type") == 1]
    return img_blocks


def extract_sections_and_embedded_images(pdf_path: str, *, force_all: bool = False) -> Dict[str, List[bytes]]:
    """Return section_key -> list of embedded image bytes extracted from <id>_1.pdf."""
    doc = fitz.open(pdf_path)

    section_to_images: Dict[str, List[bytes]] = {v: [] for v in set(SECTION_ALIASES.values())}

    current_section: Optional[str] = "all_images" if force_all else None

    for page in doc:
        page_dict = page.get_text("dict")

        # detect section headers (red lines)
        lines = list(_iter_lines_with_style(page_dict))
        lines.sort(key=lambda x: (x[2][1], x[2][0]))

        # build header markers on this page
        headers: List[Tuple[float, str]] = []  # (y0, section_key)
        if not force_all:
            for text, is_red, bbox in lines:
                if not is_red:
                    continue
                norm = _norm(text)
                norm = re.sub(r"[:\-]+$", "", norm).strip()
                if norm in SECTION_ALIASES:
                    headers.append((bbox[1], SECTION_ALIASES[norm]))

        headers.sort(key=lambda x: x[0])

        # image blocks with bbox
        img_blocks = _extract_image_blocks_with_xref(page)

        # Sort images top->bottom
        img_blocks_sorted = sorted(img_blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))

        # If we have headers, assign each image to the nearest previous header on the same page.
        # Otherwise, keep using current_section (from previous pages).
        for b in img_blocks_sorted:
            bbox = b.get("bbox", [0, 0, 0, 0])
            y0 = bbox[1] if len(bbox) >= 2 else 0

            if headers:
                # find last header above this image
                sec = None
                for hy, hsec in headers:
                    if hy <= y0:
                        sec = hsec
                    else:
                        break
                if sec is not None:
                    current_section = sec

            if current_section is None:
                continue  # in force_all mode this never happens

            # Extract bytes. Prefer xref if available.
            xref = b.get("xref")
            if xref is None:
                # Some rawdict formats store image in b['image'] as dict with 'xref'
                imginfo = b.get("image")
                if isinstance(imginfo, dict) and "xref" in imginfo:
                    xref = imginfo.get("xref")

            data = None
            if xref is not None:
                try:
                    base = doc.extract_image(int(xref))
                    data = base.get("image")
                except Exception:
                    data = None

            if data is None:
                # Fallback: try to render the image block area.
                # This is slower but avoids corner cases.
                try:
                    rect = fitz.Rect(bbox)
                    pix = page.get_pixmap(clip=rect, dpi=200)
                    data = pix.tobytes("png")
                except Exception:
                    data = None

            if data:
                # filter out tiny decorative images/logos/etc.
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        w, h = im.size
                    min_side = 150
                    if current_section in {"technical_drawings", "fashion_show_drawings"}:
                        min_side = 80  # drawings can be smaller
                    
                    if w < min_side or h < min_side:
                        continue
                except Exception:
                    pass

                section_to_images.setdefault(current_section, []).append(data)

    # remove empty sections
    return {k: v for k, v in section_to_images.items() if v}


# -------------------------
# Index season images
# -------------------------

def find_named_folders(season_dir: str, folder_name: str) -> List[str]:
    """Find directories within season_dir whose basename matches folder_name (case-insensitive).

    Returns list of absolute paths.
    """
    folder_name_n = _norm(folder_name)
    out = []
    for root, dirs, _files in os.walk(season_dir):
        for d in dirs:
            if _norm(d) == folder_name_n:
                out.append(os.path.join(root, d))
    return out


def list_image_files(root_dir: str) -> List[str]:
    out = []
    for r, _dirs, files in os.walk(root_dir):
        for f in files:
            p = os.path.join(r, f)
            ext = Path(p).suffix.lower()
            if ext in IMAGE_EXTS:
                out.append(p)
    return out


def _cache_paths_match_season(paths: Iterable[str], season_dir: str, *, sample_size: int = 25) -> bool:
    season_dir_abs = os.path.abspath(season_dir)
    checked = 0

    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            return False
        if checked >= sample_size:
            break

        abs_path = os.path.abspath(raw_path)
        try:
            common = os.path.commonpath([season_dir_abs, abs_path])
        except ValueError:
            return False

        if common != season_dir_abs:
            return False
        if not os.path.exists(abs_path):
            return False
        checked += 1

    return checked > 0


def build_candidate_index(season_dir: str, *, cache_path: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Return section_key -> {abs_path -> dhash}.

    If cache_path exists, loads it; otherwise builds and writes cache.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, dict):
            cached_paths: List[str] = []
            for mp in cached.values():
                if isinstance(mp, dict):
                    cached_paths.extend(str(p) for p in mp.keys())
            if _cache_paths_match_season(cached_paths, season_dir):
                # cached format: {section_key: {abs_path: hash_int_as_str}}
                return {
                    sec: {p: int(h) for p, h in mp.items()}
                    for sec, mp in cached.items()
                    if isinstance(mp, dict)
                }

    index: Dict[str, Dict[str, int]] = {}

    for folder_name, section_key in FOLDER_TO_SECTION.items():
        dirs = find_named_folders(season_dir, folder_name)
        if not dirs:
            continue
        paths = []
        for d in dirs:
            paths.extend(list_image_files(d))

        mp: Dict[str, int] = {}
        for p in paths:
            # Safer than fitz.Pixmap(path) in some environments
            try:
                mp[p] = dhash_from_file(p)
            except Exception:
                # ignore unreadable
                continue
        if mp:
            index[section_key] = mp

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        serializable = {sec: {p: str(h) for p, h in mp.items()} for sec, mp in index.items()}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    return index

def build_global_image_index(season_dir: str, *, cache_path: Optional[str] = None) -> Dict[str, int]:
    """Return {abs_path -> dhash} for ALL images under the season_dir."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, dict) and _cache_paths_match_season(cached.keys(), season_dir):
            return {p: int(h) for p, h in cached.items()}

    paths = list_image_files(season_dir)
    mp: Dict[str, int] = {}
    for p in paths:
        try:
            mp[p] = dhash_from_file(p)
        except Exception:
            continue

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({p: str(h) for p, h in mp.items()}, f, ensure_ascii=False)

    return mp

# -------------------------
# Matching
# -------------------------

def _to_db_dir(seg: str) -> str:
    # normalize exported path segments -> lowercase + underscores
    return re.sub(r"\s+", "_", seg.strip()).lower()

def _to_db_file(seg: str) -> str:
    return _to_db_dir(seg)


def _find_dataset_root_part(parts: Iterable[str]) -> Optional[str]:
    parts_list = list(parts)
    for part in parts_list:
        if _norm(part) == PRIMARY_DATASET_ROOT_NAME:
            return part
    for part in reversed(parts_list):
        if _norm(part) in DATASET_ROOT_NAMES:
            return part
    return None


def dataset_root_slug_from_path(path: str) -> str:
    parts = Path(os.path.abspath(path)).parts
    dataset_root = _find_dataset_root_part(parts)
    if dataset_root is not None:
        return _to_db_file(dataset_root)
    return "dataset"


def season_output_slug(season_dir: str) -> str:
    season_name = os.path.basename(os.path.abspath(season_dir).rstrip("/\\"))
    season_slug = _to_db_file(season_name)
    dataset_slug = dataset_root_slug_from_path(season_dir)
    return f"{dataset_slug}_{season_slug}"


def cache_root_dir() -> Path:
    return Path(__file__).resolve().parent / ".cache"


def season_cache_key(season_dir: str) -> str:
    abs_path = os.path.abspath(season_dir)
    parts = list(Path(abs_path).parts)
    dataset_root = _find_dataset_root_part(parts)
    if dataset_root is not None:
        anchor = parts.index(dataset_root)
        rel_parts = parts[anchor:]
    else:
        rel_parts = parts[-2:]
    return "__".join(_to_db_file(p) for p in rel_parts)


def season_cache_paths(season_dir: str, *, global_match: bool) -> Tuple[str, Optional[str]]:
    root = cache_root_dir()
    root.mkdir(parents=True, exist_ok=True)
    key = season_cache_key(season_dir)
    cache_path = str(root / f"{key}__sections.json")
    cache_all = str(root / f"{key}__global.json") if global_match else None
    return cache_path, cache_all

def _relpath(path: str, base: str) -> str:
    """
    Return a DB path starting at the dataset root folder.
    Lowercase every segment and replace spaces with underscores.
    """
    abs_path = os.path.abspath(path)
    parts = Path(abs_path).parts

    # find the dataset-root anchor (case-insensitive)
    dataset_root = _find_dataset_root_part(parts)
    start = parts.index(dataset_root) if dataset_root is not None else None

    if start is None:
        rel = os.path.relpath(abs_path, base)
        rel_parts = list(Path(rel).parts)
        return "/".join(_to_db_file(s) for s in rel_parts)

    rel_parts = list(parts[start:])  # starts with dataset root, then season, then folders..., then filename
    return "/".join(_to_db_file(s) for s in rel_parts)


def _image_id_from_path(path: str) -> str:
    return Path(path).stem


def best_match(extracted_hash: int, candidates: Dict[str, int]) -> Tuple[str, int]:
    best_p = ""
    best_d = 10**9
    for p, h in candidates.items():
        d = hamming(extracted_hash, h)
        if d < best_d:
            best_p, best_d = p, d
    return best_p, best_d


def match_section_images(
    embedded_images: List[bytes],
    candidates: Dict[str, int],
    season_dir: str,
    *,
    exact_max_dist: int,
    near_max_dist: int,
) -> Dict[str, Any]:
    """Match embedded images to candidate images for a section."""
    exact: List[ImageMatch] = []
    near: List[ImageMatch] = []
    unmatched = 0

    used_paths = set()

    for data in embedded_images:
        try:
            eh = dhash_from_bytes(data)
        except Exception:
            unmatched += 1
            continue

        if not candidates:
            unmatched += 1
            continue

        p, d = best_match(eh, candidates)
        if not p:
            unmatched += 1
            continue

        if d <= exact_max_dist:
            if p not in used_paths:
                used_paths.add(p)
                exact.append(ImageMatch(_relpath(p, season_dir), _image_id_from_path(p), d))
        elif d <= near_max_dist:
            if p not in used_paths:
                used_paths.add(p)
                near.append(ImageMatch(_relpath(p, season_dir), _image_id_from_path(p), d))
        else:
            unmatched += 1

    return {
        "exact": [m.__dict__ for m in exact],
        "near": [m.__dict__ for m in near],
        "unmatched": unmatched,
    }


def expand_similar(
    seed_matches: List[Dict[str, Any]],
    candidates: Dict[str, int],
    season_dir: str,
    *,
    k: int,
    expand_max_dist: int,
) -> List[Dict[str, Any]]:
    """Find additional similar images around each seed match (by hash distance)."""
    if not seed_matches or not candidates:
        return []

    # reverse lookup for candidate hash
    cand_items = list(candidates.items())
    out: List[Dict[str, Any]] = []
    seen = set(m.get("rel_path") for m in seed_matches)

    for seed in seed_matches:
        seed_rel = seed.get("rel_path")
        # find abs path from rel by scanning candidates
        seed_abs = None
        for p, _h in cand_items:
            if _relpath(p, season_dir) == seed_rel:
                seed_abs = p
                break
        if seed_abs is None:
            continue
        seed_hash = candidates.get(seed_abs)
        if seed_hash is None:
            continue

        scored: List[Tuple[int, str]] = []
        for p, h in cand_items:
            rel = _relpath(p, season_dir)
            if rel in seen:
                continue
            d = hamming(seed_hash, h)
            if d <= expand_max_dist:
                scored.append((d, p))

        scored.sort(key=lambda x: x[0])
        for d, p in scored[:k]:
            rel = _relpath(p, season_dir)
            if rel in seen:
                continue
            seen.add(rel)
            out.append({
                "rel_path": rel,
                "id": _image_id_from_path(p),
                "dist": d,
                "seed": seed.get("id"),
            })

    return out


# -------------------------
# Season processing
# -------------------------

def find_technical_description_dirs(season_dir: str) -> List[str]:
    out = []
    allowed = {"technical descriptions", "tech descriptions"}
    for root, dirs, _files in os.walk(season_dir):
        for d in dirs:
            if _norm(d) in allowed:
                out.append(os.path.join(root, d))
    return out


def list_pdf_pairs(tech_desc_dir: str) -> List[Tuple[str, str, str]]:
    """Return list of (outfit_id, pdf_main, pdf_images)."""
    files = sorted([f for f in os.listdir(tech_desc_dir) if f.lower().endswith(".pdf")])
    main = {}
    images = {}
    for f in files:
        stem = Path(f).stem
        m = re.fullmatch(r"(\d+)(?:_(\d+))?", stem)
        if not m:
            continue
        oid = m.group(1)
        suffix = m.group(2)
        if suffix is None:
            main[oid] = os.path.join(tech_desc_dir, f)
        else:
            # we only care _1 as the companion
            if suffix == "1":
                images[oid] = os.path.join(tech_desc_dir, f)

    pairs = []
    for oid, p_main in main.items():
        p_img = images.get(oid)
        if p_img:
            pairs.append((oid, p_main, p_img))
    return sorted(pairs, key=lambda x: int(x[0]))


def record_to_image_map(
    outfit_record: Dict[str, Any],
    season_dir: str,
    *,
    include_near: bool,
    include_expanded: bool,
) -> Dict[str, Any]:
    """Convert outfit-centric record to image-centric mapping."""
    fields = outfit_record.get("fields", {})
    exclude = {"author_of_file", "date_of_file"}

    compact: Dict[str, Any] = {}

    for k, v in fields.items():
        if k in exclude:
            continue
        if k == "season":
            compact["season"] = v
        else:
            compact[k] = v

    out: Dict[str, Any] = {}

    linked = outfit_record.get("linked_images", {})
    for section, payload in linked.items():
        for m in payload.get("exact", []):
            rel = m.get("rel_path")
            if rel:
                out[rel] = compact | {"match_type": "exact"}
        if include_near:
            for m in payload.get("near", []):
                rel = m.get("rel_path")
                if rel and rel not in out:
                    out[rel] = compact | {"match_type": "near"}

    if include_expanded:
        expanded = outfit_record.get("expanded_similar", {})
        for section, lst in expanded.items():
            for m in lst:
                rel = m.get("rel_path")
                if rel and rel not in out:
                    out[rel] = compact | {"match_type": "expanded", "seed": m.get("seed")}

    return out


def flatten_outfit_record(outfit_record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the grounded_outfit payload with selected top-level metadata."""
    fields = dict(outfit_record.get("fields", {}))

    return {
        "author_of_file": fields.get("author_of_file"),
        "date_of_file": fields.get("date_of_file"),
        "technical_description_pdf": outfit_record.get("technical_description_pdf"),
        "technical_images_pdf": outfit_record.get("technical_images_pdf"),
        "fields": fields,
        "linked_images": outfit_record.get("linked_images", {}),
        "expanded_similar": outfit_record.get("expanded_similar", {}),
        "raw_text": outfit_record.get("raw_text"),
        "language": outfit_record.get("language"),
    }


def find_season_dirs(dataset_root_dir: str) -> List[str]:
    """Return direct child folders of the dataset root sorted by name."""
    base = Path(dataset_root_dir)
    return sorted(str(p) for p in base.iterdir() if p.is_dir())


def merge_image_map_files(image_map_paths: Iterable[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in image_map_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def season_output_paths(
    season_dir: str,
    *,
    outfits_dir: Path,
    imgmap_dir: Path,
    explicit_image_map_output: Optional[str] = None,
) -> Tuple[Path, Path]:
    season_slug = season_output_slug(season_dir)
    out_path = outfits_dir / f"{season_slug}_all.json"
    map_path = Path(explicit_image_map_output) if explicit_image_map_output else (imgmap_dir / f"{season_slug}_all.json")
    return out_path, map_path


def season_already_computed(
    season_dir: str,
    *,
    outfits_dir: Path,
    imgmap_dir: Path,
    explicit_image_map_output: Optional[str] = None,
) -> bool:
    out_path, map_path = season_output_paths(
        season_dir,
        outfits_dir=outfits_dir,
        imgmap_dir=imgmap_dir,
        explicit_image_map_output=explicit_image_map_output,
    )
    if not (out_path.exists() and map_path.exists()):
        return False

    expected_root = f"{dataset_root_slug_from_path(season_dir)}/"
    try:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        return False

    for key in payload.keys():
        if isinstance(key, str) and key:
            return key.startswith(expected_root)

    return False


def process_season(
    season_dir: str,
    *,
    only_outfit_ids: Optional[List[str]],
    max_pairs: Optional[int],
    exact_max_dist: int,
    near_max_dist: int,
    expand_similar_images: bool,
    expand_k: int,
    expand_max_dist: int,
    cache_hashes: bool,
    global_image_match: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (outfit_records, image_map)."""

    season_dir = os.path.abspath(season_dir)
    season_name = os.path.basename(season_dir.rstrip("/\\"))

    tech_dirs = find_technical_description_dirs(season_dir)
    if not tech_dirs:
        raise FileNotFoundError(f"No 'Technical descriptions' directory found under {season_dir}")

    cache_path = None
    cache_all = None
    if cache_hashes:
        cache_path, cache_all = season_cache_paths(season_dir, global_match=global_image_match)

    candidate_index = build_candidate_index(season_dir, cache_path=cache_path)

    outfit_records: List[Dict[str, Any]] = []
    image_map: Dict[str, Any] = {}

    global_index = None
    if global_image_match:
        global_index = build_global_image_index(season_dir, cache_path=cache_all)

    for tech_dir in tech_dirs:
        pairs = list_pdf_pairs(tech_dir)

        if only_outfit_ids:
            pairs = [p for p in pairs if p[0] in set(only_outfit_ids)]

        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        for oid, pdf_main, pdf_img in pairs:
            fields, raw_text, doc_lang = extract_fields_from_technical_pdf(pdf_main)
            cleanup_fields(fields, season_name)

            outfit_id = fields.get("file") or oid

            sections_to_embedded = extract_sections_and_embedded_images(pdf_img, force_all=global_image_match)

            linked_images: Dict[str, Any] = {}
            expanded: Dict[str, Any] = {}

            for section_key, embedded_imgs in sections_to_embedded.items():
                candidates = global_index if global_image_match else candidate_index.get(section_key, {})

                # fallback: catwalk/fashion-show drawings sometimes live under technical drawings
                if section_key == "fashion_show_drawings":
                    candidates2 = candidate_index.get("technical_drawings", {})
                    if candidates2:
                        candidates = dict(candidates)  # copy
                        candidates.update(candidates2)

                linked = match_section_images(
                    embedded_imgs,
                    candidates,
                    season_dir,
                    exact_max_dist=exact_max_dist,
                    near_max_dist=near_max_dist,
                )
                linked_images[section_key] = linked

                if expand_similar_images and linked.get("exact"):
                    expanded_list = expand_similar(
                        linked.get("exact", []),
                        candidates,
                        season_dir,
                        k=expand_k,
                        expand_max_dist=expand_max_dist,
                    )
                    if expanded_list:
                        expanded[section_key] = expanded_list

            record = {
                "outfit_id": outfit_id,
                "season": season_name,
                "technical_description_pdf": _relpath(pdf_main, season_dir),
                "technical_images_pdf": _relpath(pdf_img, season_dir),
                "fields": fields,
                "linked_images": linked_images,
                "expanded_similar": expanded,
                "raw_text": raw_text,
                "language": doc_lang,
            }
            outfit_records.append(record)

            # build image-centric mapping
            image_map.update(
                record_to_image_map(
                    record,
                    season_dir,
                    include_near=True,
                    include_expanded=expand_similar_images,
                )
            )

    return outfit_records, image_map


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build grounded metadata from Technical descriptions PDFs")

    p.add_argument("--per-outfit", action="store_true", help="Write one JSON per outfit into the output folders")
    p.add_argument(
        "--season-dir",
        default=None,
        help="Path to one season folder, e.g. .../Dataset DataShack 2026/ALTA MODA 1987 SS",
    )
    p.add_argument(
        "--dataset-root-dir",
        dest="dataset_root_dir",
        default=None,
        help="Path to the dataset root containing all seasons, e.g. .../ferre-designs/Dataset DataShack 2026",
    )
    p.add_argument("--output", default=None, help="Output base dir or JSON path. Defaults to this script folder.")
    p.add_argument("--output-image-mapping", default=None, help="Optional explicit output JSON path for the image-centric mapping")
    p.add_argument("--merge-image-maps-output", default=None, help="Optional output JSON path for a single merged image map")
    p.add_argument("--global-image-match", action="store_true", help="Ignore section headings and match all embedded images against ALL season images")

    p.add_argument(
        "--only-outfit-ids",
        default=None,
        help="Comma-separated list of outfit numeric ids to process (e.g. 301,302).",
    )
    p.add_argument("--max-pairs", type=int, default=None, help="Limit how many pairs to process")

    p.add_argument("--exact-max-dist", type=int, default=6, help="Max Hamming distance for EXACT match")
    p.add_argument("--near-max-dist", type=int, default=14, help="Max Hamming distance for NEAR match")

    p.add_argument("--expand-similar", action="store_true", help="Also include additional similar images (hash-NN)")
    p.add_argument("--expand-k", type=int, default=5, help="How many similar images to add per seed")
    p.add_argument("--expand-max-dist", type=int, default=10, help="Max Hamming distance for expansion")

    p.add_argument("--cache-hashes", action="store_true", help="Cache season image hashes to speed up reruns")

    return p.parse_args()

def _base_output_dir(output_arg: Optional[str]) -> Path:
    if not output_arg:
        return Path(__file__).resolve().parent
    p = Path(output_arg)
    # if user passes a .json file, use its parent as the base directory
    if p.suffix.lower() == ".json":
        return p.parent
    return p


def _write_season_outputs(
    records: List[Dict[str, Any]],
    image_map: Dict[str, Any],
    *,
    season_dir: str,
    outfits_dir: Path,
    imgmap_dir: Path,
    per_outfit: bool,
    include_expanded: bool,
    explicit_image_map_output: Optional[str] = None,
) -> List[Path]:
    season_slug = season_output_slug(season_dir)
    written_image_map_paths: List[Path] = []

    if per_outfit:
        for rec in records:
            oid = rec["outfit_id"]
            out_path = outfits_dir / f"{season_slug}_{oid}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(flatten_outfit_record(rec), f, ensure_ascii=False, indent=2)

            per_map = record_to_image_map(
                rec,
                os.path.abspath(season_dir),
                include_near=True,
                include_expanded=include_expanded,
            )
            map_path = imgmap_dir / f"{season_slug}_{oid}.json"
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(per_map, f, ensure_ascii=False, indent=2)
            written_image_map_paths.append(map_path)

        print(f"Wrote {len(records)} outfit files -> {outfits_dir}")
        print(f"Wrote {len(records)} image-map files -> {imgmap_dir}")
        return written_image_map_paths

    tag = "all"
    out_path = outfits_dir / f"{season_slug}_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([flatten_outfit_record(rec) for rec in records], f, ensure_ascii=False, indent=2)

    map_path = Path(explicit_image_map_output) if explicit_image_map_output else (imgmap_dir / f"{season_slug}_{tag}.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(image_map, f, ensure_ascii=False, indent=2)
    written_image_map_paths.append(map_path)

    print(f"Wrote {len(records)} outfit records -> {out_path}")
    print(f"Wrote {len(image_map)} image mappings -> {map_path}")
    return written_image_map_paths

def main() -> int:
    args = parse_args()

    if bool(args.season_dir) == bool(args.dataset_root_dir):
        raise SystemExit("Use exactly one of --season-dir or --dataset-root-dir")

    only = None
    if args.only_outfit_ids:
        only = [x.strip() for x in args.only_outfit_ids.split(",") if x.strip()]

    base_dir = _base_output_dir(args.output)
    outfits_dir = base_dir / "grounded_outfit"
    imgmap_dir = base_dir / "grounded_image_map"
    outfits_dir.mkdir(parents=True, exist_ok=True)
    imgmap_dir.mkdir(parents=True, exist_ok=True)

    season_dirs = [args.season_dir] if args.season_dir else find_season_dirs(args.dataset_root_dir)
    written_image_map_paths: List[Path] = []

    for season_dir in season_dirs:
        explicit_map_output = args.output_image_mapping if len(season_dirs) == 1 else None
        if not args.per_outfit and season_already_computed(
            season_dir,
            outfits_dir=outfits_dir,
            imgmap_dir=imgmap_dir,
            explicit_image_map_output=explicit_map_output,
        ):
            print(f"Skipping already computed season -> {season_dir}")
            _out_path, existing_map_path = season_output_paths(
                season_dir,
                outfits_dir=outfits_dir,
                imgmap_dir=imgmap_dir,
                explicit_image_map_output=explicit_map_output,
            )
            written_image_map_paths.append(existing_map_path)
            continue

        records, image_map = process_season(
            season_dir,
            only_outfit_ids=only,
            max_pairs=args.max_pairs,
            exact_max_dist=args.exact_max_dist,
            near_max_dist=args.near_max_dist,
            expand_similar_images=bool(args.expand_similar),
            expand_k=args.expand_k,
            expand_max_dist=args.expand_max_dist,
            cache_hashes=bool(args.cache_hashes),
            global_image_match=bool(args.global_image_match),
        )

        written_image_map_paths.extend(
            _write_season_outputs(
                records,
                image_map,
                season_dir=season_dir,
                outfits_dir=outfits_dir,
                imgmap_dir=imgmap_dir,
                per_outfit=bool(args.per_outfit),
                include_expanded=bool(args.expand_similar),
                explicit_image_map_output=explicit_map_output,
            )
        )

    if args.merge_image_maps_output:
        merged_image_map = merge_image_map_files(written_image_map_paths)
        merge_path = Path(args.merge_image_maps_output)
        merge_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merge_path, "w", encoding="utf-8") as f:
            json.dump(merged_image_map, f, ensure_ascii=False, indent=2)
        print(f"Wrote merged image map -> {merge_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#docker compose run --rm --entrypoint /bin/bash llm-rag-cli -lc "cd /app && source /.venv/bin/activate && python metadata/images_metadata/grounded_metadata/build_grounded_outfit_metadata.py --dataset-root-dir 'input-datasets/ferre-designs/Dataset DataShack 2026' --cache-hashes --global-image-match --merge-image-maps-output 'metadata/images_metadata/grounded_metadata/grounded_image_map/all_collections_merged.json'"
