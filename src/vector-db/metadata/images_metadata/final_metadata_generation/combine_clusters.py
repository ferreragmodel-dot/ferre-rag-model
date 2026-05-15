from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_FILE = BASE_DIR / "../missing_pdf_clustering/missing_pdf_clusters_registry_final.jsonl"
GROUNDED_FILE = BASE_DIR / "../grounded_clustering/grounded_outfit_dhash_clusters_registry_final.json"
GROUNDED_DIR = BASE_DIR / "../grounded_clustering/grounded_outfit"
OUTPUT_FILE = BASE_DIR / "all_outfit_clusters_registry_final.jsonl"

SEASON_FILES = [
    "dataset_datashack_2026_alta_moda_1986-87_fw_all.json",
    "dataset_datashack_2026_alta_moda_1987_ss_all.json",
    "dataset_datashack_2026_alta_moda_1987-88_fw_all.json",
    "dataset_datashack_2026_alta_moda_1988_ss_all.json",
    "dataset_datashack_2026_alta_moda_1988-89_fw_all.json",
    "dataset_datashack_2026_alta_moda_1989_ss_all.json",
]

PDF_COMPLETENESS_FIELDS = ["description", "materials", "remark"]


def is_nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def normalize_season_from_filename(filename: str) -> str:
    name = Path(filename).name.replace(".json", "")
    parts = name.split("_")
    year_part = parts[-3]
    season_part = parts[-2].lower()

    if season_part == "fw":
        if "-" in year_part:
            start, end2 = year_part.split("-")
            return f"FW{start}-{start[:2]}{end2}"
        return f"FW{year_part}"
    if season_part == "ss":
        return f"SS{year_part}"
    return year_part


def extract_outfit_id(cluster_id: str) -> str | None:
    match = re.search(r"_(\d+)$", cluster_id)
    return match.group(1) if match else None


def extract_season(cluster_id: str) -> str | None:
    match = re.match(r"^((?:FW\d{4}-\d{4})|(?:SS\d{4}))_", cluster_id)
    return match.group(1) if match else None


def classify_grounded_entry(entry):
    fields = entry.get("fields", {}) or {}

    has_any_pdf_field = any(
        is_nonempty(fields.get(field))
        for field in PDF_COMPLETENESS_FIELDS
    )
    has_complete_pdf_fields = all(
        is_nonempty(fields.get(field))
        for field in PDF_COMPLETENESS_FIELDS
    )

    if has_complete_pdf_fields:
        return "available"
    if has_any_pdf_field:
        return "incomplete"
    return "empty"


def build_pdf_status_map() -> Dict[Tuple[str, str], str]:
    pdf_status_map: Dict[Tuple[str, str], str] = {}

    for fname in SEASON_FILES:
        season_file = GROUNDED_DIR / fname
        season_key = normalize_season_from_filename(fname)
        with open(season_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        for entry in data:
            status = classify_grounded_entry(entry)
            for pdf_key in ["technical_description_pdf", "technical_images_pdf"]:
                pdf_path = entry.get(pdf_key)
                if is_nonempty(pdf_path):
                    basename = os.path.basename(pdf_path)
                    pdf_status_map[(season_key, basename)] = status

    return pdf_status_map


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def repair_embedding_cluster(cluster: Dict[str, Any]) -> Dict[str, Any]:
    repaired = dict(cluster)
    repaired["pdf_status"] = "missing"
    repaired["llm_input_mode"] = "image_only"
    repaired["outfit_id"] = repaired.get("outfit_id") or repaired.get("cluster_id")
    return repaired


def repair_grounded_cluster(
    cluster: Dict[str, Any],
    pdf_status_map: Dict[Tuple[str, str], str],
) -> Dict[str, Any]:
    repaired = dict(cluster)
    cluster_id = str(repaired.get("cluster_id") or "")
    season = extract_season(cluster_id) or repaired.get("season")

    statuses: List[str] = []
    for pdf_path in repaired.get("pdf_paths", []) or []:
        basename = os.path.basename(str(pdf_path))
        status = pdf_status_map.get((season, basename))
        if status:
            statuses.append(status)

    repaired["season"] = season
    repaired["outfit_id"] = extract_outfit_id(cluster_id) or repaired.get("outfit_id")
    repaired["llm_input_mode"] = "image+pdf"
    repaired["pdf_status"] = "available" if "available" in statuses else "incomplete"
    return repaired


def main() -> None:
    pdf_status_map = build_pdf_status_map()
    rows: List[Dict[str, Any]] = []

    for cluster in iter_jsonl(EMBEDDING_FILE.resolve()):
        rows.append(repair_embedding_cluster(cluster))

    with open(GROUNDED_FILE.resolve(), "r", encoding="utf-8") as handle:
        grounded_data = json.load(handle)

    for cluster in grounded_data:
        rows.append(repair_grounded_cluster(cluster, pdf_status_map))

    with open(OUTPUT_FILE.resolve(), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Loaded {len(rows)} clusters")
    print(f"Combined file saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
