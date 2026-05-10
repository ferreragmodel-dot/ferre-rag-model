import json
import re

# Normalize a single path string into the real filesystem-style path
def fix_path(path: str) -> str:
    # Add repo prefix if missing
    if not path.startswith("ferre-rag-model/"):
        path = "ferre-rag-model/" + path

    # Apply exact folder name replacements
    replacements = {
        "dataset_datashack_2026": "Dataset DataShack 2026",
        "alta_moda_": "ALTA MODA ",
        "womenswear": "Womenswear",
        "technical_sheets": "Technical sheets",
        "technical_drawings": "Technical drawings",
        "technical_descriptions": "Technical descriptions",
        "fashion_show_drawings": "Fashion show drawings",
        "fashion_show_photos_–_final": "Fashion show photos – final",
        "fashion_show_photos_–_milan": "Fashion show photos – Milan",
        "fashion_show_photos": "Fashion show photos",
        "fashion_show_photos_–_milan_castello_sforzesco": "Fashion show photos – Milan Castello Sforzesco",
        "material_sheets": "Material sheets",
    }

    for old, new in replacements.items():
        path = path.replace(old, new)

    # Fix season formatting in folder names
    path = path.replace("_fw", " FW")
    path = path.replace("_ss", " SS")

    return path


# Extract outfit_id from cluster_id
# Example: "FW1986-1987_file_202" -> "202"
def extract_outfit_id(cluster_id: str):
    match = re.search(r'_(\d+)$', cluster_id)
    return match.group(1) if match else None


# Extract season from cluster_id
# Examples:
# "FW1986-1987_file_202" -> "FW1986-1987"
# "SS1987_file_301" -> "SS1987"
def extract_season(cluster_id: str):
    match = re.match(r'^((?:FW\d{4}-\d{4})|(?:SS\d{4}))_', cluster_id)
    return match.group(1) if match else None


input_file = "grounded_outfit_dhash_clusters_registry.json"
output_file = "grounded_outfit_dhash_clusters_registry_fixed_paths.json"


def main():
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for obj in data:
        cluster_id = obj.get("cluster_id", "")

        # Update outfit_id based on the trailing numeric part of cluster_id
        obj["outfit_id"] = extract_outfit_id(cluster_id)

        # Update season based on the prefix of cluster_id
        obj["season"] = extract_season(cluster_id)

        # Set llm_input_mode for grounded clusters
        obj["llm_input_mode"] = "image+pdf"

        # Update image_paths only if present and valid
        if "image_paths" in obj and isinstance(obj["image_paths"], list):
            obj["image_paths"] = [
                fix_path(p) if isinstance(p, str) else p
                for p in obj["image_paths"]
            ]

        # Update pdf_paths only if present and valid
        if "pdf_paths" in obj and isinstance(obj["pdf_paths"], list):
            obj["pdf_paths"] = [
                fix_path(p) if isinstance(p, str) else p
                for p in obj["pdf_paths"]
            ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done. Fixed file saved to: {output_file}")


if __name__ == "__main__":
    main()
