#!/usr/bin/env python3
"""Normalize the grounded image-map archive into a JSONL stream.

Phase 01 in the compact-facets pipeline:
- read image-centric grounded metadata from metadata/grounded_image_map/*.json
- preserve the original record under raw
- add stable top-level routing fields used later by derivation and loading
- normalize unicode, whitespace, and a small repair dictionary for known OCR/noise
- write one JSON object per image to metadata/normalized_grounded/grounded_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

INPUT_DIR = Path("metadata/grounded_image_map")
OUTPUT_DIR = Path("metadata/normalized_grounded")
OUTPUT_JSONL = OUTPUT_DIR / "grounded_raw.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "grounded_raw_report.json"
SCHEMA_VERSION = "grounded_raw_v1"

TEXT_FIELDS = {
    "acquisition",
    "bibliography",
    "collection",
    "condition",
    "description",
    "designer",
    "exhibitions",
    "label",
    "materials",
    "object",
    "present_location",
    "remark",
    "season",
    "season_label",
    "size",
    "source",
    "working_process",
}

KNOWN_REPAIRS = {
    ("condition", "Goog"): "Good",
    ("object", "Completo tre prezzi"): "Completo tre pezzi",
    ("object", "Three-piece evenng dress"): "Three-piece evening dress",
}

COLLECTION_MAP = {
    "alta moda": "Couture",
    "couture": "Couture",
}

SEASON_FILE_RE = re.compile(r"^(FW\d{4}(?:-\d{2})?|SS\d{4})_all\.json$", re.IGNORECASE)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def infer_asset_type(source_path: str) -> str:
    path = source_path.replace("\\", "/").lower()
    if "/fashion_show_photos" in path:
        return "fashion_show_photos"
    if "/fashion_show_drawings" in path:
        return "fashion_show_drawings"
    if "/technical_sheets" in path:
        return "technical_sheets"
    if "/technical_drawings" in path:
        return "technical_drawings"
    if "/material_sheets" in path:
        return "material_sheets"
    if "/advertising" in path:
        return "advertising"
    if "/press_coverage" in path:
        return "press_coverage"
    if "/embroidery" in path:
        return "embroidery"
    return "unknown"


def infer_collection_line(source_path: str) -> Optional[str]:
    parts = source_path.replace("\\", "/").split("/")
    return parts[0] if parts else None


def parse_year(value: Any, season: Optional[str]) -> Optional[int]:
    if value is not None:
        m = re.search(r"(19|20)\d{2}", str(value))
        if m:
            return int(m.group(0))
    if season:
        m = re.match(r"^(FW|SS)(\d{4})", season)
        if m:
            return int(m.group(2))
    return None


def normalize_collection(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = normalize_unicode(str(value))
    return COLLECTION_MAP.get(cleaned.lower(), cleaned)


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_unicode(value)
    return value


def apply_repairs(field: str, value: Any) -> Tuple[Any, Optional[Dict[str, str]]]:
    if value is None or not isinstance(value, str):
        return value, None
    cleaned = normalize_unicode(value)
    repaired = KNOWN_REPAIRS.get((field, cleaned), cleaned)
    if repaired != cleaned:
        return repaired, {"field": field, "original": cleaned, "repaired": repaired}
    return cleaned, None


def normalize_record(source_path: str, record: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    raw = dict(record)
    normalized_fields: Dict[str, Any] = {}
    repair_log: List[Dict[str, str]] = []

    for key, value in record.items():
        if key in TEXT_FIELDS and isinstance(value, str):
            repaired_value, repair = apply_repairs(key, value)
            normalized_fields[key] = repaired_value
            if repair:
                repair_log.append(repair)
        else:
            normalized_fields[key] = clean_scalar(value)

    season = normalize_unicode(str(record.get("season") or source_file.removesuffix("_all.json")))
    collection = normalize_collection(record.get("collection"))
    year = parse_year(record.get("year"), season)
    asset_type = infer_asset_type(source_path)

    completeness_signals = [
        normalized_fields.get("object"),
        normalized_fields.get("description"),
        normalized_fields.get("materials"),
    ]
    non_null_core = sum(x is not None for x in completeness_signals)
    if non_null_core == 3:
        raw_completeness = "full"
    elif non_null_core >= 1:
        raw_completeness = "partial"
    else:
        raw_completeness = "minimal"

    normalized = {
        "record_id": normalize_unicode(source_path),
        "source_path": normalize_unicode(source_path),
        "source_file": source_file,
        "season": season,
        "year": year,
        "collection": collection,
        "collection_line": infer_collection_line(source_path),
        "look": clean_scalar(record.get("look")),
        "outfit_id": clean_scalar(record.get("outfit_id")),
        "file": clean_scalar(record.get("file")),
        "asset_type": asset_type,
        "metadata_source": "grounded",
        "raw_completeness": raw_completeness,
        "schema_version": SCHEMA_VERSION,
        "match_type": clean_scalar(record.get("match_type")),
        "section": clean_scalar(record.get("section")),
        "raw": raw,
        "normalized": normalized_fields,
        "repair_log": repair_log,
    }
    return normalized


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*_all.json")):
        if SEASON_FILE_RE.match(path.name):
            yield path


def build_archive(input_dir: Path = INPUT_DIR, output_jsonl: Path = OUTPUT_JSONL, output_report: Path = OUTPUT_REPORT) -> Tuple[Path, Path]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    season_counts = Counter()
    asset_type_counts = Counter()
    completeness_counts = Counter()
    repair_counts = Counter()
    repaired_examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    total_records = 0

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for json_path in iter_input_files(input_dir):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            for source_path, record in payload.items():
                normalized = normalize_record(source_path, record, json_path.name)
                f_out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                total_records += 1
                season_counts[normalized["season"]] += 1
                asset_type_counts[normalized["asset_type"]] += 1
                completeness_counts[normalized["raw_completeness"]] += 1
                for repair in normalized["repair_log"]:
                    key = f"{repair['field']}::{repair['original']}->{repair['repaired']}"
                    repair_counts[key] += 1
                    if len(repaired_examples[key]) < 3:
                        repaired_examples[key].append({
                            "record_id": normalized["record_id"],
                            **repair,
                        })

    report = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "output_jsonl": str(output_jsonl),
        "total_records": total_records,
        "season_counts": dict(season_counts),
        "asset_type_counts": dict(asset_type_counts),
        "raw_completeness_counts": dict(completeness_counts),
        "repair_counts": dict(repair_counts),
        "repair_examples": dict(repaired_examples),
    }
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_jsonl, output_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize the grounded Ferré image-map archive into JSONL.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()

    out_jsonl, out_report = build_archive(args.input_dir, args.output_jsonl, args.output_report)
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_report}")
