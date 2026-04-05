import json

def fix_path(path: str) -> str:
    # === Step 1: add prefix if missing ===
    if not path.startswith("ferre-rag-model/"):
        path = "ferre-rag-model/" + path

    # === Step 2: replace normalized folder names ===
    replacements = {
        "dataset_datashack_2026": "Dataset DataShack 2026",
        "alta_moda_": "ALTA MODA ",
        "fashion_show_photos": "Fashion show photos",
        "womenswear": "Womenswear",
        "technical_drawings": "Technical drawings",
    }

    for old, new in replacements.items():
        path = path.replace(old, new)

    # === Step 3: fix season formatting ===
    path = path.replace("_fw", " FW")
    path = path.replace("_ss", " SS")

    return path


input_file = "missing_pdf_clusters_registry_reclustered.jsonl"
output_file = "missing_pdf_clusters_registry_reclustered_fixed_paths.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    for line in f_in:
        obj = json.loads(line)

        # ✅ only modify image_paths
        if "image_paths" in obj and isinstance(obj["image_paths"], list):
            obj["image_paths"] = [
                fix_path(p) if isinstance(p, str) else p
                for p in obj["image_paths"]
            ]

        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ Done! All paths fixed.")