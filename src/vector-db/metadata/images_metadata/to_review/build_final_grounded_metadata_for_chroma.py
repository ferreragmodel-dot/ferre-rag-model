#!/usr/bin/env python3
"""Build final grounded image metadata ready for Chroma loading.

This script is the practical bridge from:
    phase 04 output (text-derived + image-filled compact facets)
into:
    a minimal final metadata layer you can load into Chroma now,
    without generating metadata for non-grounded images.

Design goals
------------
1. Keep only useful fields.
2. Avoid repeating the same information in multiple places.
3. Preserve both:
   - compact retrieval facets (for filtering / reranking / UI chips)
   - key archival raw text (for answer generation after retrieval)
4. Produce a *flat* Chroma-ready JSONL, because Chroma metadata works best
   with scalar fields and string arrays rather than nested dicts.
5. Also produce a richer nested JSONL for offline inspection.

Typical current usage in this repo:
- input:  metadata/image_filled_grounded_facets/grounded_compact_filled_llm.jsonl
- output: metadata/final_for_chroma/final_image_metadata_for_chroma.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_PHASE4_INPUT = "metadata/image_filled_grounded_facets/grounded_compact_filled_llm.jsonl"
DEFAULT_NORMALIZED_INPUT = "metadata/normalized_grounded/grounded_raw.jsonl"
DEFAULT_OUTPUT_JSONL = "metadata/final_for_chroma/final_image_metadata_for_chroma.jsonl"
DEFAULT_OUTPUT_MAP = "metadata/final_for_chroma/final_image_metadata_for_chroma_map.json"
DEFAULT_OUTPUT_ARCHIVE_JSONL = "metadata/final_for_chroma/final_image_metadata_archive.jsonl"
DEFAULT_REPORT = "metadata/final_for_chroma/final_image_metadata_for_chroma_report.json"
DEFAULT_ALLOWED_ASSET_TYPES = "fashion_show_photos"
SCHEMA_VERSION = "ferre_final_for_chroma_v1"

FACET_ARRAY_FIELDS = [
    "garments",
    "colors",
    "material_families",
    "patterns",
    "silhouette_tags",
    "length_tags",
    "neckline_tags",
    "sleeve_tags",
    "closure_tags",
    "embellishment_tags",
    "style_tags",
]

FACET_SCALAR_FIELDS = [
    "season",
    "year",
    "collection",
    "collection_line",
    "look",
    "outfit_id",
    "asset_type",
    "metadata_source",
    "raw_completeness",
    "facet_source",
    "facet_confidence",
    "needs_review",
    "phase4_indexable",
    "schema_version",
]

RAW_USEFUL_FIELDS = [
    "object",
    "description",
    "materials",
    "remark",
    "working_process",
    "source",
]

PROVENANCE_FIELDS = [
    "source_file",
    "match_type",
    "repair_log",
    "image_fill_applied",
    "image_fill_strategy",
    "image_fill_model",
    "image_fill_prompt_version",
    "image_fill_group_key",
    "image_fill_group_size",
    "image_fill_group_image_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final grounded image metadata for Chroma loading."
    )
    parser.add_argument(
        "--phase4-input",
        default=DEFAULT_PHASE4_INPUT,
        help="Phase-04 JSONL file (heuristic or LLM version).",
    )
    parser.add_argument(
        "--normalized-input",
        default=DEFAULT_NORMALIZED_INPUT,
        help="Phase-01 normalized grounded archive JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Flat Chroma-ready JSONL output.",
    )
    parser.add_argument(
        "--output-map",
        default=DEFAULT_OUTPUT_MAP,
        help="Flat Chroma-ready JSON map keyed by source_path.",
    )
    parser.add_argument(
        "--output-archive-jsonl",
        default=DEFAULT_OUTPUT_ARCHIVE_JSONL,
        help="Nested archive-oriented JSONL output.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="Summary report JSON.",
    )
    parser.add_argument(
        "--asset-types",
        default=DEFAULT_ALLOWED_ASSET_TYPES,
        help="Comma-separated asset types to include. Default: fashion_show_photos",
    )
    parser.add_argument(
        "--include-non-indexable",
        action="store_true",
        help="Include records even when phase4_indexable is false.",
    )
    parser.add_argument(
        "--keep-empty-fields",
        action="store_true",
        help="Keep empty/null fields instead of dropping them from final outputs.",
    )
    return parser.parse_args()


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)



def write_json(path: str, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count



def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).replace("\u00a0", " ").strip()
    value = " ".join(value.split())
    return value or None



def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        seen = set()
        for item in value:
            text = normalize_text(item)
            if text and text not in seen:
                out.append(text)
                seen.add(text)
        return out
    text = normalize_text(value)
    return [text] if text else []



def maybe_drop_empty(d: Dict[str, Any], keep_empty_fields: bool) -> Dict[str, Any]:
    if keep_empty_fields:
        return d
    cleaned: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, dict) and len(v) == 0:
            continue
        cleaned[k] = v
    return cleaned



def load_normalized_index(path: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        index[row["record_id"]] = row
    return index



def extract_raw_subset(normalized_row: Dict[str, Any]) -> Dict[str, Any]:
    raw = normalized_row.get("raw", {}) or {}
    subset = {f: normalize_text(raw.get(f)) for f in RAW_USEFUL_FIELDS}
    return subset



def build_flat_record(
    phase4_row: Dict[str, Any],
    normalized_row: Dict[str, Any],
    keep_empty_fields: bool,
) -> Dict[str, Any]:
    raw_subset = extract_raw_subset(normalized_row)

    flat: Dict[str, Any] = {
        "id": phase4_row.get("record_id") or phase4_row.get("source_path"),
        "source_path": phase4_row.get("source_path"),
        "description_short": normalize_text(phase4_row.get("description_short")),
        "remark_short": normalize_text(phase4_row.get("remark_short")),
        # useful archival text kept flat so it is Chroma-friendly
        "archive_object": raw_subset.get("object"),
        "archive_description": raw_subset.get("description"),
        "archive_materials": raw_subset.get("materials"),
        "archive_remark": raw_subset.get("remark"),
        "archive_working_process": raw_subset.get("working_process"),
        "archive_source": raw_subset.get("source"),
        "final_schema_version": SCHEMA_VERSION,
    }

    for field in FACET_ARRAY_FIELDS:
        flat[field] = normalize_list(phase4_row.get(field))

    for field in FACET_SCALAR_FIELDS:
        value = phase4_row.get(field)
        if field in {"season", "collection", "collection_line", "look", "outfit_id", "asset_type", "metadata_source", "raw_completeness", "facet_source", "schema_version"}:
            flat[field] = normalize_text(value)
        else:
            flat[field] = value

    return maybe_drop_empty(flat, keep_empty_fields)



def build_archive_record(
    phase4_row: Dict[str, Any],
    normalized_row: Dict[str, Any],
    keep_empty_fields: bool,
) -> Dict[str, Any]:
    raw_subset = extract_raw_subset(normalized_row)

    retrieval_facets: Dict[str, Any] = {
        "description_short": normalize_text(phase4_row.get("description_short")),
        "remark_short": normalize_text(phase4_row.get("remark_short")),
    }
    for field in FACET_ARRAY_FIELDS:
        retrieval_facets[field] = normalize_list(phase4_row.get(field))
    for field in FACET_SCALAR_FIELDS:
        value = phase4_row.get(field)
        if field in {"season", "collection", "collection_line", "look", "outfit_id", "asset_type", "metadata_source", "raw_completeness", "facet_source", "schema_version"}:
            retrieval_facets[field] = normalize_text(value)
        else:
            retrieval_facets[field] = value

    provenance = {
        key: phase4_row.get(key, normalized_row.get(key))
        for key in PROVENANCE_FIELDS
        if key in phase4_row or key in normalized_row
    }
    provenance["final_schema_version"] = SCHEMA_VERSION

    archive_row = {
        "record_id": phase4_row.get("record_id"),
        "source_path": phase4_row.get("source_path"),
        "retrieval_facets": maybe_drop_empty(retrieval_facets, keep_empty_fields),
        "archive_raw": maybe_drop_empty(raw_subset, keep_empty_fields),
        "provenance": maybe_drop_empty(provenance, keep_empty_fields),
    }
    return maybe_drop_empty(archive_row, keep_empty_fields)



def main() -> None:
    args = parse_args()

    allowed_asset_types = {
        item.strip() for item in args.asset_types.split(",") if item.strip()
    }

    normalized_index = load_normalized_index(args.normalized_input)

    flat_rows: List[Dict[str, Any]] = []
    archive_rows: List[Dict[str, Any]] = []
    flat_map: Dict[str, Dict[str, Any]] = {}

    stats = {
        "phase4_input": args.phase4_input,
        "normalized_input": args.normalized_input,
        "allowed_asset_types": sorted(allowed_asset_types),
        "input_rows": 0,
        "kept_rows": 0,
        "skipped_non_indexable": 0,
        "skipped_asset_type": 0,
        "missing_normalized_match": 0,
        "asset_type_counts": {},
        "raw_completeness_counts": {},
        "facet_source_counts": {},
    }

    for row in iter_jsonl(args.phase4_input):
        stats["input_rows"] += 1

        asset_type = normalize_text(row.get("asset_type"))
        if allowed_asset_types and asset_type not in allowed_asset_types:
            stats["skipped_asset_type"] += 1
            continue

        if not args.include_non_indexable and row.get("phase4_indexable") is False:
            stats["skipped_non_indexable"] += 1
            continue

        record_id = row.get("record_id")
        normalized_row = normalized_index.get(record_id)
        if not normalized_row:
            stats["missing_normalized_match"] += 1
            continue

        flat = build_flat_record(row, normalized_row, args.keep_empty_fields)
        nested = build_archive_record(row, normalized_row, args.keep_empty_fields)

        flat_rows.append(flat)
        archive_rows.append(nested)
        if flat.get("source_path"):
            flat_map[flat["source_path"]] = flat

        stats["kept_rows"] += 1
        stats["asset_type_counts"][asset_type] = stats["asset_type_counts"].get(asset_type, 0) + 1

        raw_completeness = normalize_text(row.get("raw_completeness")) or "unknown"
        stats["raw_completeness_counts"][raw_completeness] = stats["raw_completeness_counts"].get(raw_completeness, 0) + 1

        facet_source = normalize_text(row.get("facet_source")) or "unknown"
        stats["facet_source_counts"][facet_source] = stats["facet_source_counts"].get(facet_source, 0) + 1

    flat_rows.sort(key=lambda x: x.get("source_path") or "")
    archive_rows.sort(key=lambda x: x.get("source_path") or "")
    flat_map = dict(sorted(flat_map.items(), key=lambda kv: kv[0]))

    write_jsonl(args.output_jsonl, flat_rows)
    write_jsonl(args.output_archive_jsonl, archive_rows)
    write_json(args.output_map, flat_map)
    write_json(args.report, stats)

    print(f"Wrote {len(flat_rows)} flat Chroma-ready rows to {args.output_jsonl}")
    print(f"Wrote {len(archive_rows)} nested archive rows to {args.output_archive_jsonl}")
    print(f"Wrote metadata map ({len(flat_map)} keys) to {args.output_map}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
