#!/usr/bin/env python3
"""
Build a registry-style JSON for grounded outfit clusters matched via dHash.

Input:
- grounded_outfit/*.json produced by build_grounded_outfit_metadata.py

Output:
- a single JSON array where each row represents one grounded outfit cluster

Notes
-----
- Cluster IDs are derived from the collection code plus the grounded `fields.file`
  value, e.g. `FW1986-1987_file_202`.
- `image_paths` contains all images found under `linked_images` (`exact` + `near`),
  de-duplicated while preserving the original order.
- The three metric fields use the per-image dHash match distance as a proxy:
  `avg_similarity = mean(1 - dist / 64)`,
  `max_pairwise_distance = max(dist / 64)`,
  `cluster_confidence` is a simple rule-based label derived from those distances.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "grounded_outfit"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "grounded_outfit_dhash_clusters_registry.json"


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")

    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected item {idx} in {path} to be an object, got {type(row).__name__}"
            )
        rows.append(row)
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def extract_collection_code(path: str) -> str:
    norm_path = path.replace("\\", "/").lower()

    match_fw = re.search(r"alta_moda_(\d{4})-(\d{2})_fw", norm_path)
    if match_fw:
        start_year = match_fw.group(1)
        end_suffix = match_fw.group(2)
        end_year = start_year[:2] + end_suffix
        return f"FW{start_year}-{end_year}"

    match_ss = re.search(r"alta_moda_(\d{4})_ss", norm_path)
    if match_ss:
        return f"SS{match_ss.group(1)}"

    raise ValueError(f"Could not extract collection code from path: {path}")


def normalize_file_code(value: Optional[str], pdf_path: str) -> str:
    if value:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if digits:
            return digits if len(digits) >= 3 else digits.zfill(3)

    fallback = Path(pdf_path).stem
    fallback = re.sub(r"_\d+$", "", fallback)
    digits = "".join(ch for ch in fallback if ch.isdigit())
    if digits:
        return digits if len(digits) >= 3 else digits.zfill(3)

    raise ValueError(f"Could not derive file code from record with pdf path: {pdf_path}")


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def collect_linked_images_and_distances(record: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    image_paths: List[str] = []
    distances: List[int] = []

    linked_images = record.get("linked_images") or {}
    if not isinstance(linked_images, dict):
        return [], []

    for payload in linked_images.values():
        if not isinstance(payload, dict):
            continue

        for bucket_name in ("exact", "near"):
            bucket = payload.get(bucket_name) or []
            if not isinstance(bucket, list):
                continue

            for item in bucket:
                if not isinstance(item, dict):
                    continue

                rel_path = item.get("rel_path")
                dist = item.get("dist")
                if not rel_path:
                    continue

                image_paths.append(rel_path)
                if isinstance(dist, int):
                    distances.append(dist)
                else:
                    distances.append(0)

    ordered_paths = dedupe_preserve_order(image_paths)
    if len(ordered_paths) == len(image_paths):
        return ordered_paths, distances

    kept_distances: List[int] = []
    seen = set()
    for path, dist in zip(image_paths, distances):
        if path in seen:
            continue
        seen.add(path)
        kept_distances.append(dist)
    return ordered_paths, kept_distances


def compute_proxy_metrics(distances: List[int]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not distances:
        return None, None, None

    similarities = [max(0.0, 1.0 - (dist / 64.0)) for dist in distances]
    avg_similarity = round(sum(similarities) / len(similarities), 4)
    max_distance = round(max(distances) / 64.0, 4)

    max_raw = max(distances)
    if max_raw <= 2:
        confidence = "high"
    elif max_raw <= 6:
        confidence = "medium"
    else:
        confidence = "low"

    return avg_similarity, max_distance, confidence


def build_cluster_row(record: Dict[str, Any]) -> Dict[str, Any]:
    fields = record.get("fields") or {}
    if not isinstance(fields, dict):
        fields = {}

    pdf_description = record.get("technical_description_pdf")
    pdf_images = record.get("technical_images_pdf")
    if not isinstance(pdf_description, str) or not pdf_description:
        raise ValueError("Record is missing technical_description_pdf")

    collection_code = extract_collection_code(pdf_description)
    file_code = normalize_file_code(fields.get("file"), pdf_description)
    cluster_id = f"{collection_code}_file_{file_code}"

    pdf_paths = dedupe_preserve_order(
        path for path in (pdf_description, pdf_images) if isinstance(path, str) and path
    )
    image_paths, distances = collect_linked_images_and_distances(record)
    avg_similarity, max_pairwise_distance, cluster_confidence = compute_proxy_metrics(distances)

    return {
        "cluster_id": cluster_id,
        "cluster_type": "dhash",
        "season": None,
        "outfit_id": None,
        "pdf_status": "present",
        "pdf_paths": pdf_paths,
        "image_paths": image_paths,
        "llm_input_mode": None,
        "cluster_size": len(image_paths),
        "reclustered": None,
        "avg_similarity": avg_similarity,
        "max_pairwise_distance": max_pairwise_distance,
        "cluster_confidence": cluster_confidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a grounded outfit cluster registry JSON from dHash-linked metadata."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing grounded_outfit/*.json files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    rows: List[Dict[str, Any]] = []
    for json_path in sorted(input_dir.glob("*.json")):
        records = load_json_array(json_path)
        rows.extend(build_cluster_row(record) for record in records)

    rows.sort(key=lambda row: row["cluster_id"])
    write_json(output_path, rows)

    print(f"Wrote {len(rows)} grounded outfit clusters -> {output_path}")


if __name__ == "__main__":
    main()
