#!/usr/bin/env python3
"""
Generate final Ferré image metadata from the combined cluster registry.

Key behavior
------------
- Reads all_outfit_clusters_registry.jsonl
- For pdf_status in {"available", "incomplete"}:
    * sends all existing cluster images + matched PDF text to the LLM
    * uses merged archive_keywords + external_seed_keywords
    * copies PDF-derived fields directly from the matched grounded season JSON entry
- For pdf_status == "missing":
    * sends all existing cluster images only
    * uses merged archive_keywords + external_seed_keywords
    * fills PDF-derived fields with null
- Skips image paths that do not exist on disk
- Does NOT generate output rows for skipped/missing images
- Writes one output record per existing image path
- Metadata is generated per image, but cluster_confidence controls how strongly
  the model should keep tags consistent across images in the same cluster
- llm_description is generated per image
- Writes skipped image info to a separate skipped_images.jsonl file

Batching behavior
-----------------
- Supports easier batching with:
    * --batch-size
    * --batch-index   (1-based)
- Example for 970 clusters:
    * batch 1 with size 100 => clusters[0:100]
    * batch 10 with size 100 => clusters[900:970]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types


# ---------------------------
# Paths / defaults
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_CLUSTERS = BASE_DIR / "all_outfit_clusters_registry.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "generated_image_metadata_final.jsonl"
DEFAULT_SKIPPED_OUTPUT = BASE_DIR / "skipped_images.jsonl"
DEFAULT_ONTOLOGY = (BASE_DIR / "../fashion_ontology/ferre_retrieval_ontology.json").resolve()
DEFAULT_GROUNDED_DIR = (BASE_DIR / "../grounded_metadata/grounded_outfit").resolve()

DEFAULT_MODEL = "gemini-2.0-flash-001"
DEFAULT_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

SEASON_FILES = [
    "dataset_datashack_2026_alta_moda_1986-87_fw_all.json",
    "dataset_datashack_2026_alta_moda_1987_ss_all.json",
    "dataset_datashack_2026_alta_moda_1987-88_fw_all.json",
    "dataset_datashack_2026_alta_moda_1988_ss_all.json",
    "dataset_datashack_2026_alta_moda_1988-89_fw_all.json",
    "dataset_datashack_2026_alta_moda_1989_ss_all.json",
]

TAG_FIELDS = [
    "garments_tags",
    "colors_tags",
    "material_tags",
    "patterns_tags",
    "silhouette_tags",
    "length_tags",
    "neckline_tags",
    "sleeve_tags",
    "closure_tags",
    "embellishment_tags",
    "style_tags",
]

PDF_FIELDS = [
    "season",
    "label",
    "acquisition",
    "look",
    "file",
    "inventory",
    "object",
    "source",
    "description",
    "exhibitions",
    "size",
    "materials",
    "present_location",
    "remark",
    "bibliography",
    "designer",
    "working_process",
    "condition",
    "collection",
    "year",
]

MAX_RETRIES = 3
INITIAL_BACKOFF = 2
PDF_BACKED_STATUSES = {"available", "incomplete"}
PDF_BACKED_TAG_LIMIT = 10
IMAGE_ONLY_TAG_LIMIT = 5


# ---------------------------
# Small utilities
# ---------------------------

def is_nonempty(x: Any) -> bool:
    return x is not None and str(x).strip() != ""


def normalize_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def resolve_repo_relative_path(rel_path: str) -> Path:
    """
    Cluster registry paths look like:
    ferre-rag-model/Dataset DataShack 2026/...

    Convert them into absolute paths under the repo root.
    """
    rel_path = rel_path.strip()
    if rel_path.startswith("ferre-rag-model/"):
        rel_path = rel_path[len("ferre-rag-model/"):]
    return REPO_ROOT / rel_path


def filter_existing_image_paths(image_paths: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Keep only images that exist on disk. Missing images are collected in skipped rows.
    """
    existing = []
    skipped = []

    for image_path in image_paths:
        abs_path = resolve_repo_relative_path(image_path)
        if abs_path.exists():
            existing.append(image_path)
        else:
            skipped.append({
                "source_path": image_path,
                "resolved_path": str(abs_path),
                "reason": "missing_file",
            })

    return existing, skipped


def normalize_season_from_filename(filename: str) -> str:
    """
    Example:
    dataset_datashack_2026_alta_moda_1986-87_fw_all.json -> FW1986-1987
    dataset_datashack_2026_alta_moda_1987_ss_all.json    -> SS1987
    """
    name = Path(filename).name.replace(".json", "")
    parts = name.split("_")
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
    return year_part


def extract_year_path(season_path: Optional[str]) -> Optional[str]:
    """
    FW1986-1987 -> 1986
    SS1987      -> 1987
    """
    m = re.search(r"(\d{4})", season_path or "")
    return m.group(1) if m else None


def extract_asset_type(image_path: str) -> str:
    """
    Use the parent folder name exactly as it appears in the fixed path.
    """
    return Path(image_path).parent.name


def extract_collection_line(_: str) -> str:
    """
    Based on your project, this is ALTA MODA.
    """
    return "ALTA MODA"


def make_no_pdf_fields() -> Dict[str, Any]:
    return {k: None for k in PDF_FIELDS}


def shorten(text: Optional[str], max_len: int = 320) -> Optional[str]:
    text = normalize_text(text)
    if not text:
        return None
    if len(text) <= max_len:
        return text
    clipped = text[: max_len - 1].rstrip(" ,;:-")
    return clipped + "…"


# ---------------------------
# Ontology / vocab
# ---------------------------

def load_ontology(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_allowed_keywords(ontology: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Use the full retrieval vocabulary for every run mode.

    The mode argument is accepted for backward compatibility with older callers.
    """
    allowed = {}
    for field in TAG_FIELDS:
        merged: List[str] = []
        seen = set()
        for key_name in ["archive_keywords", "external_seed_keywords"]:
            for keyword in ontology[field].get(key_name, []):
                normalized = normalize_text(keyword)
                if normalized and normalized not in seen:
                    merged.append(normalized)
                    seen.add(normalized)
        allowed[field] = merged
    return allowed


# ---------------------------
# Grounded season JSON lookup
# ---------------------------

def build_grounded_pdf_lookup(grounded_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Build lookup:
      (season_key, basename(pdf_path)) -> season JSON entry
    """
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for fname in SEASON_FILES:
        season_file = grounded_dir / fname
        with open(season_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        season_key = normalize_season_from_filename(fname)

        for entry in data:
            for pdf_key in ["technical_description_pdf", "technical_images_pdf"]:
                pdf_path = entry.get(pdf_key)
                if is_nonempty(pdf_path):
                    basename = os.path.basename(pdf_path)
                    lookup[(season_key, basename)] = entry

    return lookup


def get_grounded_entry_for_cluster(
    cluster: Dict[str, Any],
    grounded_lookup: Dict[Tuple[str, str], Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    season_key = cluster.get("season")
    for pdf_path in cluster.get("pdf_paths", []):
        basename = os.path.basename(pdf_path)
        entry = grounded_lookup.get((season_key, basename))
        if entry is not None:
            return entry
    return None


def extract_pdf_fields_from_entry(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if entry is None:
        return make_no_pdf_fields()

    fields = entry.get("fields", {}) or {}
    out = {}
    for k in PDF_FIELDS:
        v = fields.get(k)
        out[k] = normalize_text(v) or None
    return out


def build_pdf_text_blob(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    if entry is None:
        return None

    fields = entry.get("fields", {}) or {}
    payload = {
        "technical_description_pdf": entry.get("technical_description_pdf"),
        "technical_images_pdf": entry.get("technical_images_pdf"),
        "fields": {k: fields.get(k) for k in PDF_FIELDS},
        "raw_text": entry.get("raw_text"),
        "language": entry.get("language"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------
# LLM schema / prompt
# ---------------------------

def build_response_schema(allowed_keywords: Dict[str, List[str]]) -> Dict[str, Any]:
    image_item_properties = {
        "source_path": {"type": "STRING"},
        "llm_description": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "A concise archival-style description grounded only in what is visible in this specific image "
                "and/or explicitly supported by the PDF when available. Avoid generic openings such as "
                "'A model on a runway wears...'. Prefer garment-focused, factual phrasing."
            ),
        },
    }

    for field in TAG_FIELDS:
        image_item_properties[field] = {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": allowed_keywords[field],
            },
        }

    return {
        "type": "OBJECT",
        "required": ["images_metadata"],
        "properties": {
            "images_metadata": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": ["source_path", *TAG_FIELDS, "llm_description"],
                    "properties": image_item_properties,
                },
            }
        },
    }


def build_cluster_prompt(
    cluster: Dict[str, Any],
    allowed_keywords: Dict[str, List[str]],
    grounded_entry: Optional[Dict[str, Any]],
) -> str:
    mode = cluster["llm_input_mode"]
    image_paths = cluster["image_paths"]
    cluster_confidence = cluster.get("cluster_confidence")

    instructions = {
        "task": (
            "Generate strict structured fashion metadata for one Ferré outfit cluster. "
            "Return one metadata object per source_path."
        ),
        "critical_rules": [
            "Use ONLY values from the supplied vocabulary for all tag arrays.",
            "Do NOT invent new tag values.",
            "If a tag is not clearly supported, return an empty array for that field.",
            "Be conservative and anti-hallucination.",
            "Do not infer unseen garment parts, colors, materials, patterns, silhouettes, or embellishments.",
            "Return one images_metadata item for EVERY source_path provided.",
            "The source_path in images_metadata must exactly match the provided source_path string.",
            "llm_description must be grounded only in that specific image and, when available, the PDF text.",
        ],
        "llm_description_rules": [
            "Write llm_description as a structured, garment-focused visual description.",
            "Describe the key garments, silhouette, construction, and visible details.",
            "Include specific visible elements such as cut, shape, layering, accessories, surface treatment, or materials if clearly visible.",
            "Use 1–2 sentences or one compact but sufficiently detailed descriptive phrase.",
            "Do not collapse the description into a short label.",
            "The description should be more specific than 'Technical drawing of a dress' or 'Material swatches including lace and black fabric'.",
            "Avoid generic storytelling such as 'A model on a runway wears...'.",
            "Instead, directly describe the outfit, drawing, sheet, swatch, or object in an archival-style but still concrete way.",
            "Good style examples: "
            "'Gray tailored jacket with structured shoulders, paired with a fitted skirt and wide belt; matching gloves and headpiece.' "
            "'Technical drawing of a sleeveless dress with defined waist and flared skirt, annotated with construction details.' "
            "'Fabric swatches showing black lace and textured wool with stitching and material references.'",
            "Bad style examples: "
            "'Dress.' "
            "'Technical drawing.' "
            "'A model wearing clothes.'",
        ],
        "mode_specific_rules": [
            (
                "This cluster is image+pdf. Use the PDF text as grounding support. Use ONLY the supplied vocabulary."
                if mode == "image+pdf"
                else
                "This cluster is image_only. There is no PDF support. Use ONLY the supplied vocabulary."
            )
        ],
        "tag_selection_rules": [
            "Choose only high-confidence tags that are clearly visible in the image or explicitly supported by the PDF text.",
            "Prefer a short, precise list over a broad list of plausible but weakly supported tags.",
            f"For PDF-backed clusters, aim for at most {PDF_BACKED_TAG_LIMIT} values per tag field.",
            f"For image-only clusters, aim for at most {IMAGE_ONLY_TAG_LIMIT} values per tag field.",
            "If many vocabulary terms could apply, keep the strongest and most specific ones.",
        ],
        "cluster_confidence_rules": [
            "You must take cluster_confidence seriously when deciding whether metadata should stay consistent across images.",
            "If cluster_confidence is high, assume the images most likely belong to the same outfit cluster. Keep tags consistent across images whenever the evidence supports it.",
            "If cluster_confidence is medium, keep only stable outfit-identity fields consistent when clearly supported across the cluster or PDF. Allow image-specific differences for fields that may vary by crop, angle, visibility, or uncertainty.",
            "If cluster_confidence is low, do not force consistency across images. Treat each image more independently and only assign tags supported by that image and available PDF evidence.",
            "Stable outfit-identity fields that should be kept consistent when confidence is high or medium, if supported: garments_tags, silhouette_tags, length_tags, neckline_tags, sleeve_tags, closure_tags, style_tags.",
            "Fields that may vary more across images and can differ when confidence is medium or low: colors_tags, material_tags, patterns_tags, embellishment_tags.",
            "When uncertain, prefer empty arrays rather than forcing a shared answer.",
        ],
    }

    payload = {
        "cluster_id": cluster.get("cluster_id"),
        "cluster_type": cluster.get("cluster_type"),
        "season": cluster.get("season"),
        "outfit_id": cluster.get("outfit_id"),
        "pdf_status": cluster.get("pdf_status"),
        "llm_input_mode": cluster.get("llm_input_mode"),
        "cluster_confidence": cluster_confidence,
        "image_paths": image_paths,
        "allowed_keywords": allowed_keywords,
        "pdf_text": build_pdf_text_blob(grounded_entry) if mode == "image+pdf" else None,
    }

    return (
        json.dumps(instructions, ensure_ascii=False, indent=2)
        + "\n\nCluster payload:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def call_llm_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    image_paths: List[str],
    response_schema: Dict[str, Any],
    temperature: float,
    key: str,
) -> Optional[Dict[str, Any]]:
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            contents: List[Any] = [prompt]

            for image_path in image_paths:
                abs_path = resolve_repo_relative_path(image_path)
                img_bytes = abs_path.read_bytes()
                mime_type = infer_mime_type(abs_path)
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                if isinstance(parsed, dict):
                    return parsed
                return json.loads(json.dumps(parsed, ensure_ascii=False))

            text_resp = getattr(response, "text", None)
            if text_resp:
                return json.loads(text_resp)

            return None

        except Exception as e:
            err = str(e)
            retryable = any(x in err for x in ["429", "500", "503", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED"])
            if attempt < MAX_RETRIES - 1 and retryable:
                print(f"[retry] {key}: {attempt + 1}/{MAX_RETRIES} failed, sleeping {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue

            print(f"[error] {key}: {err[:500]}")
            return None

    return None


# ---------------------------
# Post-processing
# ---------------------------

def normalize_tag_list(values: Any, allowed: List[str], max_items: Optional[int] = None) -> List[str]:
    if not isinstance(values, list):
        return []

    allowed_set = set(allowed)
    out = []
    for v in values:
        if isinstance(v, str):
            vv = normalize_text(v)
            if vv and vv in allowed_set and vv not in out:
                out.append(vv)
                if max_items is not None and len(out) >= max_items:
                    break
    return out


def tag_limit_for_cluster(cluster: Dict[str, Any]) -> int:
    return (
        PDF_BACKED_TAG_LIMIT
        if cluster.get("pdf_status") in PDF_BACKED_STATUSES
        else IMAGE_ONLY_TAG_LIMIT
    )


def cluster_to_output_rows(
    cluster: Dict[str, Any],
    llm_output: Optional[Dict[str, Any]],
    allowed_keywords: Dict[str, List[str]],
    grounded_entry: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pdf_fields = extract_pdf_fields_from_entry(grounded_entry) \
        if cluster["pdf_status"] in PDF_BACKED_STATUSES else make_no_pdf_fields()
    tag_limit = tag_limit_for_cluster(cluster)

    image_result_map = {}
    for image_path in cluster["image_paths"]:
        image_result_map[image_path] = {
            "llm_description": None,
            **{field: [] for field in TAG_FIELDS}
        }

    if llm_output is not None and isinstance(llm_output.get("images_metadata"), list):
        for item in llm_output["images_metadata"]:
            if not isinstance(item, dict):
                continue
            source_path = item.get("source_path")
            if source_path not in image_result_map:
                continue

            image_result_map[source_path]["llm_description"] = shorten(
                item.get("llm_description"),
                max_len=320
            )

            for field in TAG_FIELDS:
                image_result_map[source_path][field] = normalize_tag_list(
                    item.get(field, []),
                    allowed_keywords[field],
                    max_items=tag_limit,
                )

    season_path = cluster.get("season") or None
    year_path = extract_year_path(season_path) if season_path else None

    rows = []
    for image_path in cluster["image_paths"]:
        per_image = image_result_map[image_path]

        row = {
            "source_path": image_path,
            "collection_line": extract_collection_line(image_path),
            "season_path": season_path,
            "year_path": year_path,
            "asset_type": extract_asset_type(image_path),

            "season": pdf_fields["season"],
            "label": pdf_fields["label"],
            "acquisition": pdf_fields["acquisition"],
            "look": pdf_fields["look"],
            "file": pdf_fields["file"],
            "inventory": pdf_fields["inventory"],
            "object": pdf_fields["object"],
            "source": pdf_fields["source"],
            "description": pdf_fields["description"],
            "exhibitions": pdf_fields["exhibitions"],
            "size": pdf_fields["size"],
            "materials": pdf_fields["materials"],
            "present_location": pdf_fields["present_location"],
            "remark": pdf_fields["remark"],
            "bibliography": pdf_fields["bibliography"],
            "designer": pdf_fields["designer"],
            "working_process": pdf_fields["working_process"],
            "condition": pdf_fields["condition"],
            "collection": pdf_fields["collection"],
            "year": pdf_fields["year"],

            "garments_tags": per_image["garments_tags"],
            "colors_tags": per_image["colors_tags"],
            "material_tags": per_image["material_tags"],
            "patterns_tags": per_image["patterns_tags"],
            "silhouette_tags": per_image["silhouette_tags"],
            "length_tags": per_image["length_tags"],
            "neckline_tags": per_image["neckline_tags"],
            "sleeve_tags": per_image["sleeve_tags"],
            "closure_tags": per_image["closure_tags"],
            "embellishment_tags": per_image["embellishment_tags"],
            "style_tags": per_image["style_tags"],

            "llm_description": per_image["llm_description"],

            "pdf_available": cluster["pdf_status"],
            "cluster_id": cluster.get("cluster_id"),
            "outfit_id": cluster.get("outfit_id"),
        }
        rows.append(row)

    return rows


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final Ferré image metadata from cluster registry.")
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTERS), help="Path to all_outfit_clusters_registry.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    parser.add_argument("--skipped-output", default=str(DEFAULT_SKIPPED_OUTPUT), help="Skipped images JSONL path")
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY), help="Path to ferre_retrieval_ontology.json")
    parser.add_argument("--grounded-dir", default=str(DEFAULT_GROUNDED_DIR), help="Directory containing the 6 grounded season JSON files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Vertex AI location")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")

    # Original selection style
    parser.add_argument("--offset-clusters", type=int, default=0, help="Start from this cluster index")
    parser.add_argument("--limit-clusters", type=int, default=None, help="Only process this many clusters")

    # Easier batching style
    parser.add_argument("--batch-size", type=int, default=None, help="Number of clusters per batch, e.g. 100")
    parser.add_argument("--batch-index", type=int, default=None, help="1-based batch index, e.g. 1, 2, 3 ...")

    parser.add_argument("--cluster-id", default=None, help="Run only one specific cluster_id")
    parser.add_argument("--resume", action="store_true", help="Skip clusters already present in the output")
    return parser.parse_args()


def load_existing_source_paths(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    for row in iter_jsonl(path):
        source_path = row.get("source_path")
        if source_path:
            seen.add(source_path)
    return seen


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    args = parse_args()

    if "GCP_PROJECT" not in os.environ and "GOOGLE_CLOUD_PROJECT" not in os.environ:
        raise SystemExit("Set GCP_PROJECT or GOOGLE_CLOUD_PROJECT before running.")
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise SystemExit("Set GOOGLE_APPLICATION_CREDENTIALS before running.")

    project = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

    clusters_path = Path(args.clusters)
    output_path = Path(args.output)
    skipped_output_path = Path(args.skipped_output)
    ontology_path = Path(args.ontology)
    grounded_dir = Path(args.grounded_dir)

    if args.batch_index is not None and args.batch_size is not None:
        skipped_output_path = skipped_output_path.with_name(
            f"{skipped_output_path.stem}_batch_{args.batch_index}{skipped_output_path.suffix}"
        )

    ontology = load_ontology(ontology_path)
    grounded_lookup = build_grounded_pdf_lookup(grounded_dir)

    all_clusters = list(iter_jsonl(clusters_path))
    total_clusters = len(all_clusters)

    total_batches = None
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise SystemExit("--batch-size must be > 0")
        total_batches = (total_clusters + args.batch_size - 1) // args.batch_size
        print(f"Total clusters: {total_clusters}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total batches: {total_batches}")

    if args.cluster_id is not None:
        selected_clusters = [c for c in all_clusters if c.get("cluster_id") == args.cluster_id]
    else:
        if (args.batch_size is None) ^ (args.batch_index is None):
            raise SystemExit("Use --batch-size and --batch-index together.")

        if args.batch_size is not None and args.batch_index is not None:
            if args.batch_index <= 0:
                raise SystemExit("--batch-index must be >= 1")

            start = (args.batch_index - 1) * args.batch_size
            end = start + args.batch_size

            if start >= total_clusters:
                raise SystemExit(
                    f"--batch-index {args.batch_index} is out of range. "
                    f"There are only {total_batches} batches for {total_clusters} clusters."
                )

            selected_clusters = all_clusters[start:end]

            print(
                f"Running batch {args.batch_index}/{total_batches}: "
                f"clusters[{start}:{min(end, total_clusters)}]"
            )
        else:
            selected_clusters = all_clusters[args.offset_clusters:]
            if args.limit_clusters is not None:
                selected_clusters = selected_clusters[: args.limit_clusters]

    if not selected_clusters:
        print("No clusters selected. Check --cluster-id or offset/limit.")
        return

    seen_source_paths = load_existing_source_paths(output_path) if args.resume else set()
    output_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    skipped_clusters = 0

    client = genai.Client(
        vertexai=True,
        project=project,
        location=args.location,
        http_options=types.HttpOptions(api_version="v1"),
    )

    try:
        for idx, cluster in enumerate(selected_clusters, start=1):
            cluster_id = cluster.get("cluster_id", f"cluster_{idx}")

            if args.resume and all(p in seen_source_paths for p in cluster["image_paths"]):
                skipped_clusters += 1
                print(f"[skip] {cluster_id} already present in output")
                continue

            allowed_keywords = get_allowed_keywords(ontology, cluster["llm_input_mode"])
            grounded_entry = (
                get_grounded_entry_for_cluster(cluster, grounded_lookup)
                if cluster["pdf_status"] in PDF_BACKED_STATUSES
                else None
            )

            existing_image_paths, skipped_images = filter_existing_image_paths(cluster["image_paths"])
            for row in skipped_images:
                row["cluster_id"] = cluster_id
                row["outfit_id"] = cluster.get("outfit_id")
                skipped_rows.append(row)

            if not existing_image_paths:
                skipped_rows.append({
                    "cluster_id": cluster_id,
                    "outfit_id": cluster.get("outfit_id"),
                    "source_path": None,
                    "resolved_path": None,
                    "reason": "no_existing_images_left_in_cluster",
                })
                print(f"[skip] {cluster_id} has no existing images left")
                continue

            cluster_for_run = dict(cluster)
            cluster_for_run["image_paths"] = existing_image_paths

            prompt = build_cluster_prompt(cluster_for_run, allowed_keywords, grounded_entry)
            response_schema = build_response_schema(allowed_keywords)

            llm_output = call_llm_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                image_paths=cluster_for_run["image_paths"],
                response_schema=response_schema,
                temperature=args.temperature,
                key=cluster_id,
            )

            rows = cluster_to_output_rows(
                cluster=cluster_for_run,
                llm_output=llm_output,
                allowed_keywords=allowed_keywords,
                grounded_entry=grounded_entry,
            )
            output_rows.extend(rows)

            print(f"[ok] {cluster_id} -> {len(rows)} image records")

    finally:
        client.close()

    if args.resume and output_path.exists():
        with open(output_path, "a", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        save_jsonl(output_path, output_rows)

    # Overwrite skipped report for this run (or this batch-specific file)
    save_jsonl(skipped_output_path, skipped_rows)

    print(f"Wrote {len(output_rows)} records to: {output_path}")
    print(f"Wrote {len(skipped_rows)} skipped image records to: {skipped_output_path}")
    if args.resume:
        print(f"Skipped {skipped_clusters} clusters already present in output.")


if __name__ == "__main__":
    main()
