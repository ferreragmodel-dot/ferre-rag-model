import json
import os

CONTENT_FIELDS = ["description", "materials", "source", "remark", "working_process"]

season_files = [
    "grounded_outfit/dataset_datashack_2026_alta_moda_1986-87_fw_all.json",
    "grounded_outfit/dataset_datashack_2026_alta_moda_1987_ss_all.json",
    "grounded_outfit/dataset_datashack_2026_alta_moda_1987-88_fw_all.json",
    "grounded_outfit/dataset_datashack_2026_alta_moda_1988_ss_all.json",
    "grounded_outfit/dataset_datashack_2026_alta_moda_1988-89_fw_all.json",
    "grounded_outfit/dataset_datashack_2026_alta_moda_1989_ss_all.json",
]

def is_nonempty(x):
    return x is not None and str(x).strip() != ""

def normalize_season_from_filename(filename: str) -> str:
    """
    Example:
    dataset_datashack_2026_alta_moda_1986-87_fw_all.json -> FW1986-1987
    dataset_datashack_2026_alta_moda_1987_ss_all.json    -> SS1987
    """
    name = os.path.basename(filename).replace(".json", "")
    parts = name.split("_")

    # expected tail patterns:
    # ... 1986-87 fw all
    # ... 1987 ss all
    year_part = parts[-3]
    season_part = parts[-2].lower()

    if season_part == "fw":
        if "-" in year_part:
            start, end2 = year_part.split("-")
            end_full = start[:2] + end2
            return f"FW{start}-{end_full}"
        return f"FW{year_part}"
    elif season_part == "ss":
        return f"SS{year_part}"
    else:
        return year_part

# Build a lookup table: (season, basename) -> "available" / "empty"
pdf_status_map = {}

for season_file in season_files:
    with open(season_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    season_key = normalize_season_from_filename(season_file)

    for entry in data:
        fields = entry.get("fields", {})
        description = fields.get("description")

        non_empty_count = sum(
            1 for field in CONTENT_FIELDS if is_nonempty(fields.get(field))
        )

        if not is_nonempty(description) or non_empty_count < 2:
            status = "empty"
        else:
            status = "available"

        for pdf_key in ["technical_description_pdf", "technical_images_pdf"]:
            pdf_path = entry.get(pdf_key)
            if is_nonempty(pdf_path):
                basename = os.path.basename(pdf_path)
                pdf_status_map[(season_key, basename)] = status

# Update grounded registry
input_registry = "grounded_outfit_dhash_clusters_registry_fixed_paths.json"
output_registry = "grounded_outfit_dhash_clusters_registry_final.json"

with open(input_registry, "r", encoding="utf-8") as f:
    registry = json.load(f)

for cluster in registry:
    cluster_season = cluster.get("season")
    pdf_paths = cluster.get("pdf_paths", [])

    statuses = []
    for p in pdf_paths:
        basename = os.path.basename(p)
        status = pdf_status_map.get((cluster_season, basename))
        if status is not None:
            statuses.append(status)

    # Since this grounded registry only contains PDF-backed clusters,
    # assign available if any linked PDF is available; otherwise empty.
    if "available" in statuses:
        cluster["pdf_status"] = "available"
    else:
        cluster["pdf_status"] = "empty"

with open(output_registry, "w", encoding="utf-8") as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

print(f"Done. Updated registry saved to: {output_registry}")