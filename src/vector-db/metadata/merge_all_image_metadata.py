#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SEASON_LABELS = {
    "SS": "Spring-Summer",
    "FW": "Fall-Winter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge fashion_show_photos metadata JSON files with all grounded_image_map JSON files "
            "into a single JSON keyed by image path."
        )
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the metadata directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <metadata-dir>/ferre_images_all_metadata_merged.json).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process the first N image ids after sorting. Useful for testing.",
    )
    parser.add_argument(
        "--only-overlap",
        action="store_true",
        help="Only merge images that exist in both metadata sources. Useful for sanity checks.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print merge statistics without writing an output file.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON with indentation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {path}, found {type(data).__name__}")
    return data


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned)
    if isinstance(value, (dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def infer_collection_line(image_id: str) -> str:
    parts = image_id.split("/")
    return parts[0] if len(parts) >= 1 else ""


def infer_season_from_path(image_id: str) -> str:
    parts = image_id.split("/")
    return parts[1] if len(parts) >= 2 else ""


def infer_asset_type(image_id: str) -> str:
    parts = image_id.split("/")
    return parts[2] if len(parts) >= 3 else ""


def infer_season_label(season: str) -> str:
    if season.startswith("SS"):
        return SEASON_LABELS["SS"]
    if season.startswith("FW"):
        return SEASON_LABELS["FW"]
    return ""


def infer_year(season: str) -> str:
    if season.startswith("SS"):
        return season[2:]
    if season.startswith("FW"):
        return season[2:6]
    return ""


def normalize_metadata_entry(image_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)

    season = infer_season_from_path(image_id) or normalized.get("season", "")
    normalized["season"] = season
    normalized["season_label"] = infer_season_label(season)
    normalized.pop("season_code", None)

    normalized["collection_line"] = normalized.get("collection_line") or infer_collection_line(image_id)
    normalized["asset_type"] = normalized.get("asset_type") or infer_asset_type(image_id)
    normalized["source_path"] = normalized.get("source_path") or image_id
    normalized["year"] = normalized.get("year") or infer_year(season)

    for field in ["garments", "colors", "materials", "patterns", "silhouette", "notes"]:
        normalized[field] = stringify(normalized.get(field))

    return normalized


def normalize_grounded_entry(image_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)

    season = normalized.get("season") or infer_season_from_path(image_id)
    normalized["season"] = season
    normalized["season_label"] = infer_season_label(season) or stringify(normalized.get("season_label"))

    normalized.pop("section", None)
    normalized.pop("picture", None)

    normalized["collection_line"] = infer_collection_line(image_id)
    normalized["asset_type"] = infer_asset_type(image_id)
    normalized["source_path"] = image_id
    normalized["year"] = stringify(normalized.get("year")) or infer_year(season)
    normalized["materials"] = stringify(normalized.get("materials"))

    return normalized


def merge_entry(
    image_id: str,
    meta_entry: dict[str, Any] | None,
    grounded_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    if meta_entry:
        merged.update(meta_entry)
    if grounded_entry:
        merged.update(grounded_entry)

    # Preserve/infer the mandatory path-derived fields in the final record.
    merged["source_path"] = (meta_entry or {}).get("source_path") or image_id
    merged["collection_line"] = (meta_entry or {}).get("collection_line") or infer_collection_line(image_id)
    merged["asset_type"] = (meta_entry or {}).get("asset_type") or infer_asset_type(image_id)

    # Keep grounded materials when available; otherwise use the metadata version.
    grounded_materials = stringify((grounded_entry or {}).get("materials"))
    metadata_materials = stringify((meta_entry or {}).get("materials"))
    merged["materials"] = grounded_materials or metadata_materials

    # Force a single season format everywhere.
    season = (grounded_entry or {}).get("season") or (meta_entry or {}).get("season") or infer_season_from_path(image_id)
    merged["season"] = season
    merged["season_label"] = infer_season_label(season)
    merged.pop("season_code", None)
    merged.pop("section", None)
    merged.pop("picture", None)

    # Guarantee string fields from the metadata JSONs stay strings in the final JSON.
    for field in ["garments", "colors", "materials", "patterns", "silhouette", "notes"]:
        if field in merged:
            merged[field] = stringify(merged.get(field))

    if not stringify(merged.get("year")):
        merged["year"] = infer_year(season)
    else:
        merged["year"] = stringify(merged.get("year"))

    return merged


def collect_metadata_files(metadata_dir: Path) -> tuple[list[Path], list[Path]]:
    metadata_files = sorted(metadata_dir.glob("ferre_images_fashion_show_photos_metadata_*.json"))
    grounded_files = sorted((metadata_dir / "grounded_image_map").glob("*.json"))

    if not metadata_files:
        raise FileNotFoundError(
            f"No ferre_images_fashion_show_photos_metadata_*.json files found in {metadata_dir}"
        )
    if not grounded_files:
        raise FileNotFoundError(
            f"No JSON files found in {metadata_dir / 'grounded_image_map'}"
        )

    return metadata_files, grounded_files


def build_merged_dataset(
    metadata_dir: Path,
    sample: int | None = None,
    only_overlap: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    metadata_files, grounded_files = collect_metadata_files(metadata_dir)

    metadata_index: dict[str, dict[str, Any]] = {}
    grounded_index: dict[str, dict[str, Any]] = {}

    for path in metadata_files:
        raw = load_json(path)
        for image_id, entry in raw.items():
            metadata_index[image_id] = normalize_metadata_entry(image_id, entry)

    for path in grounded_files:
        raw = load_json(path)
        for image_id, entry in raw.items():
            grounded_index[image_id] = normalize_grounded_entry(image_id, entry)

    metadata_ids = set(metadata_index)
    grounded_ids = set(grounded_index)
    overlap_ids = metadata_ids & grounded_ids

    if only_overlap:
        ordered_ids = sorted(overlap_ids)
    else:
        ordered_ids = sorted(metadata_ids | grounded_ids)

    if sample is not None:
        ordered_ids = ordered_ids[:sample]

    merged: dict[str, dict[str, Any]] = {}
    for image_id in ordered_ids:
        merged[image_id] = merge_entry(
            image_id,
            metadata_index.get(image_id),
            grounded_index.get(image_id),
        )

    stats = {
        "metadata_images": len(metadata_ids),
        "grounded_images": len(grounded_ids),
        "overlap_images": len(overlap_ids),
        "metadata_only_images": len(metadata_ids - grounded_ids),
        "grounded_only_images": len(grounded_ids - metadata_ids),
        "final_images": len(merged),
    }
    return merged, stats


def main() -> None:
    args = parse_args()

    metadata_dir = args.metadata_dir.resolve()
    output_path = args.output or (metadata_dir / "ferre_images_all_metadata_merged.json")
    output_path = output_path.resolve()

    merged, stats = build_merged_dataset(
        metadata_dir=metadata_dir,
        sample=args.sample,
        only_overlap=args.only_overlap,
    )

    print("Merge stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    if args.stats_only:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        else:
            json.dump(merged, f, ensure_ascii=False)
    print(f"\nWritten merged JSON to: {output_path}")


if __name__ == "__main__":
    main()



# -----------------------------------------------------------------------------
# HOW TO USE THIS SCRIPT
# -----------------------------------------------------------------------------
# The script merges:
#   1) all JSON files named like:
#        ferre_images_fashion_show_photos_metadata_*.json
#   2) all JSON files inside:
#        metadata/grounded_image_map/
#
# It matches records by image id / image path, for example:
#   ALTA-MODA/SS1987/fashion_show_photos/12345.jpg
#
# The final output is one single big JSON containing all images from all years.
#
# -----------------------------------------------------------------------------
# BASIC USAGE
# -----------------------------------------------------------------------------
# Open a terminal, move into the metadata folder, and run:
#
#   cd metadata
#   python merge_all_image_metadata.py --pretty --output ferre_images_all_metadata_merged.json
#
# This will create a merged JSON with all records.
#
# -----------------------------------------------------------------------------
# TEST BEFORE RUNNING ON ALL FILES
# -----------------------------------------------------------------------------
# To test the script on only a small number of images first:
#
#   cd metadata
#   python merge_all_image_metadata.py --sample 20 --pretty --output merged_sample.json
#
# Meaning:
#   --sample 20
#       process only the first 20 image ids after sorting
#   --pretty
#       save the JSON with indentation so it is easier to read
#   --output merged_sample.json
#       write the result to a test file instead of the final full file
#
# -----------------------------------------------------------------------------
# TEST ONLY THE IMAGES PRESENT IN BOTH SOURCES
# -----------------------------------------------------------------------------
# Useful to check if the merge between the two metadata sources works correctly:
#
#   cd metadata
#   python merge_all_image_metadata.py --only-overlap --sample 20 --pretty --output merged_overlap_sample.json
#
# Meaning:
#   --only-overlap
#       only merge images that are present both in:
#       - ferre_images_fashion_show_photos_metadata_*.json
#       - grounded_image_map/*.json
#
# -----------------------------------------------------------------------------
# SEE ONLY THE STATISTICS, WITHOUT WRITING A FILE
# -----------------------------------------------------------------------------
# This is useful to quickly check how many images are in each source and how many
# will end up in the final merged dataset:
#
#   cd metadata
#   python merge_all_image_metadata.py --stats-only
#
# -----------------------------------------------------------------------------
# RUN ON ALL FILES
# -----------------------------------------------------------------------------
# When the test looks good, run the full merge:
#
#   cd metadata
#   python merge_all_image_metadata.py --pretty --output ferre_images_all_metadata_merged.json
#
# -----------------------------------------------------------------------------
# WHAT THE SCRIPT NORMALIZES IN THE FINAL JSON
# -----------------------------------------------------------------------------
# - keeps: source_path, collection_line, asset_type
# - removes: season_code, section, picture
# - normalizes season to format like: SS1987 or FW1986-87
# - adds season_label in the grounded format
# - converts these fields to strings:
#     garments, colors, materials, patterns, silhouette, notes
# - if materials conflict, it keeps the materials from grounded_image_map
# - includes also grounded-only images, even if they do not appear in the main
#   metadata JSON files
# -----------------------------------------------------------------------------
