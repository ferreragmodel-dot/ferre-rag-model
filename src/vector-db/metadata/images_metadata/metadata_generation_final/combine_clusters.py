import json

# Input files
embedding_file = "../generated_metadata/missing_pdf_clusters_registry_reclustered_fixed_paths.jsonl"
grounded_file = "../grounded_metadata/grounded_outfit_dhash_clusters_registry_final.json"

# Output file
output_file = "all_outfit_clusters_registry.jsonl"


with open(output_file, "w", encoding="utf-8") as f_out:

    # === 1. Load embedding clusters (jsonl) ===
    with open(embedding_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("✅ Loaded embedding clusters")

    # === 2. Load grounded clusters (json list) ===
    with open(grounded_file, "r", encoding="utf-8") as f_in:
        grounded_data = json.load(f_in)

        for obj in grounded_data:
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("✅ Loaded grounded clusters")

print(f"🎉 Done! Combined file saved to: {output_file}")