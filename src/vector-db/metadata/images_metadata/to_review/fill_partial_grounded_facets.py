#!/usr/bin/env python3
"""Phase 04 — fill partial grounded compact facets with multimodal Gemini.

Updated behavior
----------------
This version groups records by outfit before multimodal generation.
Records sharing the same (season, collection_line, outfit_id) are sent together
as a single multimodal prompt so Gemini sees multiple views of the same outfit
and returns one shared facet block. That shared block is then reused for each
processable record in the group, which avoids inconsistent per-image outputs for
identical outfits photographed from different angles.

The script works with either:
- heuristic phase 03 output: metadata/derived_grounded_facets/grounded_compact.jsonl
- LLM phase 03 output:       metadata/derived_grounded_facets/grounded_compact_llm.jsonl

The script follows the phase-04 decision table:
1. if description/materials exist -> keep text-derived facets only (no image fill by default)
2. if object exists but description/materials are missing -> keep confirmed text garments,
   fill only the missing facets from the grouped outfit images
3. if object, description, and materials are all missing -> generate the facet block from the grouped outfit images

The script never generates archival fields such as acquisition, bibliography, inventory,
present_location, or condition.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
except Exception:  # allows --dry-run in environments without google-genai
    genai = None
    types = None


DEFAULT_INPUT = "metadata/derived_grounded_facets/grounded_compact.jsonl"
DEFAULT_NORMALIZED = "metadata/normalized_grounded/grounded_raw.jsonl"
DEFAULT_VOCAB = "metadata/config/ferre_facet_vocabulary_v1.json"
DEFAULT_IMAGES_ROOT = "input-datasets/ferre-designs"
DEFAULT_OUTPUT = "metadata/image_filled_grounded_facets/grounded_compact_filled.jsonl"
DEFAULT_REPORT = "metadata/image_filled_grounded_facets/grounded_compact_filled_report.json"
DEFAULT_MODEL = "gemini-2.0-flash-001"
DEFAULT_LOCATION = "us-central1"
PROMPT_VERSION = "phase04_image_fill_grouped_v2"
SCHEMA_VERSION = "ferre_facets_v1"
FACET_SOURCE_FILLED = "grounded_plus_image_fill_v1"

MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_BACKOFF = 32

FACET_FIELDS = [
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

TEXT_SUMMARY_FIELDS = ["description_short", "remark_short"]

ASSET_TYPES_VISUAL = {
    "fashion_show_photos",
    "fashion_show_drawings",
    "technical_sheets",
    "technical_drawings",
    "material_sheets",
    "advertising",
    "press_coverage",
    "embroidery",
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_whitespace(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def shorten(text: Optional[str], max_len: int = 220) -> Optional[str]:
    text = normalize_whitespace(text)
    if not text:
        return None
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip(" ,;:-") + "…"


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            item = normalize_whitespace(str(item))
            if item:
                out.append(item)
        return out
    if isinstance(value, str):
        value = normalize_whitespace(value)
        if not value:
            return []
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value]
    return []


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        item = normalize_whitespace(item)
        if item and item not in out:
            out.append(item)
    return out


def confidence_bucket(score: float) -> Tuple[bool, bool]:
    """Return (accepted, needs_review)."""
    if score >= 0.75:
        return True, False
    if score >= 0.50:
        return True, True
    return False, True


def resolve_image_path(images_root: str, source_path: str) -> str:
    return str((Path(images_root) / Path(source_path)).resolve())


def guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def make_group_key(record: Dict[str, Any]) -> str:
    season = normalize_whitespace(record.get("season")) or "unknown"
    collection_line = normalize_whitespace(record.get("collection_line")) or "unknown"
    outfit_id = normalize_whitespace(record.get("outfit_id")) or "unknown"
    return f"{season}::{collection_line}::{outfit_id}"


def build_response_schema(vocab: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    facet_fields = vocab["facet_fields"]
    properties: Dict[str, Any] = {}

    for field in FACET_FIELDS:
        properties[field] = {
            "type": "ARRAY",
            "items": {"type": "STRING", "enum": facet_fields[field]},
        }

    properties["description_short"] = {
        "type": "STRING",
        "nullable": True,
        "description": "One concise factual sentence grounded in the outfit visible across the provided images.",
    }
    properties["remark_short"] = {
        "type": "STRING",
        "nullable": True,
        "description": "One concise style-oriented note grounded in the shared outfit across the provided images, or null.",
    }
    properties["facet_confidence"] = {
        "type": "NUMBER",
        "description": "Confidence from 0.0 to 1.0 for the newly generated grouped image-based facets.",
    }

    if strategy == "object_plus_image_fill":
        description = "Fill only missing facets from multiple images of the same outfit. Preserve already confirmed text-derived facets."
    else:
        description = "Generate one shared compact facet block from multiple images of the same outfit using only allowed vocabulary values."

    return {
        "type": "OBJECT",
        "description": description,
        "propertyOrdering": [*FACET_FIELDS, *TEXT_SUMMARY_FIELDS, "facet_confidence"],
        "required": [*FACET_FIELDS, *TEXT_SUMMARY_FIELDS, "facet_confidence"],
        "properties": properties,
    }


def to_json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def aggregate_group_grounded_context(group_compact_records: List[Dict[str, Any]], normalized_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    archive_context = {
        "season": group_compact_records[0].get("season"),
        "year": group_compact_records[0].get("year"),
        "collection": group_compact_records[0].get("collection"),
        "collection_line": group_compact_records[0].get("collection_line"),
        "look": group_compact_records[0].get("look"),
        "outfit_id": group_compact_records[0].get("outfit_id"),
        "raw_completeness_values": unique_preserve_order(str(r.get("raw_completeness")) for r in group_compact_records if r.get("raw_completeness")),
        "source_paths": [r.get("source_path") for r in group_compact_records if r.get("source_path")],
        "record_ids": [r.get("record_id") for r in group_compact_records if r.get("record_id")],
        "asset_types": unique_preserve_order(str(r.get("asset_type")) for r in group_compact_records if r.get("asset_type")),
    }

    normalized_fields: Dict[str, List[str]] = {k: [] for k in ["object", "description", "materials", "working_process", "remark"]}
    raw_fields: Dict[str, List[str]] = {k: [] for k in ["object", "description", "materials", "working_process", "remark"]}

    for record in group_compact_records:
        grounded = normalized_by_id.get(record["record_id"], {})
        raw = grounded.get("raw", {}) or {}
        normalized = grounded.get("normalized", {}) or {}
        for field in normalized_fields:
            if normalized.get(field):
                normalized_fields[field].append(normalized[field])
            if raw.get(field):
                raw_fields[field].append(raw[field])

    aggregated = {
        "archive_context": archive_context,
        "grounded_text_available": {field: unique_preserve_order(values) for field, values in normalized_fields.items()},
        "raw_text_traceability": {field: unique_preserve_order(values) for field, values in raw_fields.items()},
    }
    return aggregated


def compute_group_strategy(group_compact_records: List[Dict[str, Any]], normalized_by_id: Dict[str, Dict[str, Any]]) -> str:
    strategies = []
    for record in group_compact_records:
        grounded = normalized_by_id.get(record["record_id"], {})
        strategies.append(detect_phase4_strategy(grounded, record))
    if "image_only" in strategies:
        return "image_only"
    if "object_plus_image_fill" in strategies:
        return "object_plus_image_fill"
    return "text_only"


def build_group_locked_non_empty(group_compact_records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    locked: Dict[str, List[str]] = {}
    for field in FACET_FIELDS:
        values: List[str] = []
        for record in group_compact_records:
            values.extend(normalize_list(record.get(field)))
        values = unique_preserve_order(values)
        if values:
            locked[field] = values
    return locked


def build_group_missing_fields(group_compact_records: List[Dict[str, Any]], strategy: str) -> List[str]:
    missing: List[str] = []
    for field in FACET_FIELDS:
        if strategy == "object_plus_image_fill" and field == "garments":
            garments_locked = any(normalize_list(record.get("garments")) for record in group_compact_records)
            if garments_locked:
                continue
        if any(not normalize_list(record.get(field)) for record in group_compact_records):
            missing.append(field)

    if any(not normalize_whitespace(record.get("description_short")) for record in group_compact_records):
        missing.append("description_short")
    if any(not normalize_whitespace(record.get("remark_short")) for record in group_compact_records):
        missing.append("remark_short")
    return missing


def build_group_prompt(
    group_key: str,
    group_compact_records: List[Dict[str, Any]],
    normalized_by_id: Dict[str, Dict[str, Any]],
    vocab: Dict[str, Any],
    strategy: str,
    missing_fields: List[str],
) -> str:
    compact_vocab = {field: vocab["facet_fields"][field] for field in FACET_FIELDS}
    locked_non_empty = build_group_locked_non_empty(group_compact_records)
    aggregated = aggregate_group_grounded_context(group_compact_records, normalized_by_id)

    strategy_rules = {
        "text_only": [
            "Do not use image completion. This grouped record should remain text-derived only.",
        ],
        "object_plus_image_fill": [
            "The grouped outfit has object-level textual evidence but description/materials are missing for at least one record.",
            "You are seeing multiple images of the SAME outfit_id.",
            "Return one shared facet block that is consistent across the whole outfit group.",
            "Preserve already confirmed text-derived facets when they are present.",
            "Use the images only to fill missing facet arrays and optional short summaries.",
            "If garments are already present in locked fields, do not replace them.",
            "If images differ by crop or angle, infer only the common outfit, not image-specific details.",
        ],
        "image_only": [
            "The grouped outfit is too incomplete textually; derive one shared compact facet block from the provided images.",
            "You are seeing multiple images of the SAME outfit_id.",
            "Use the common visible outfit across the image set and ignore background people or unrelated garments.",
            "If images differ by crop or angle, infer only the common outfit, not image-specific details.",
            "Stay conservative: prefer [] over uncertain guesses.",
        ],
    }

    record_briefs = []
    for record in group_compact_records:
        record_briefs.append(
            {
                "record_id": record.get("record_id"),
                "source_path": record.get("source_path"),
                "asset_type": record.get("asset_type"),
                "raw_completeness": record.get("raw_completeness"),
                "existing_compact_record": {field: record.get(field) for field in [*FACET_FIELDS, *TEXT_SUMMARY_FIELDS]},
            }
        )

    instructions = {
        "task": "Fill missing Ferré retrieval facets from multiple archive images of the same outfit.",
        "rules": [
            "Use only values from the supplied controlled vocabulary.",
            "Return arrays for facet fields and never comma-joined strings.",
            "Return ONE shared facet block for the whole outfit group, not one block per image.",
            "Do not generate archival fields such as acquisition, bibliography, inventory, present_location, condition, or exhibitions.",
            "Do not invent textile supplier names, article codes, percentages, or hidden garment details.",
            "Focus on the primary outfit or main visible subject shared across the images.",
            "If a facet is not visually supportable, return an empty array for that facet.",
            "description_short and remark_short must be concise and factual.",
            *strategy_rules[strategy],
        ],
    }

    payload = {
        "group_key": group_key,
        "group_size": len(group_compact_records),
        **aggregated,
        "group_records": record_briefs,
        "locked_non_empty_fields": locked_non_empty,
        "missing_fields_to_fill": missing_fields,
        "allowed_vocabulary": compact_vocab,
    }

    return f"{to_json_text(instructions)}\n\nArchive payload:\n{to_json_text(payload)}\n"


def call_multimodal_llm_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    image_paths: List[str],
    response_schema: Dict[str, Any],
    key: str,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    backoff = INITIAL_BACKOFF

    parts: List[Any] = [{"text": prompt}]
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime_type = guess_mime(image_path)
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=types.Content(role="user", parts=parts),
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
                try:
                    return json.loads(json.dumps(parsed, ensure_ascii=False))
                except Exception:
                    pass

            text_resp = getattr(response, "text", None)
            if text_resp:
                return json.loads(text_resp)

            return None

        except Exception as e:
            error_str = str(e)
            is_retryable = (
                "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "500" in error_str
                or "503" in error_str
                or "DEADLINE_EXCEEDED" in error_str
            )
            if attempt < MAX_RETRIES - 1 and is_retryable:
                print(f"[retry] {key}: attempt {attempt + 1}/{MAX_RETRIES} failed, retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            print(f"[error] {key}: {error_str[:400]}")
            return None

    return None


def apply_synonyms(field: str, items: List[str], vocab: Dict[str, Any]) -> List[str]:
    synonym_map = vocab.get("synonym_maps", {}).get(field, {})
    allowed = set(vocab["facet_fields"][field])
    out: List[str] = []
    for item in items:
        norm = normalize_whitespace(item)
        if not norm:
            continue
        mapped = synonym_map.get(norm, norm)
        if mapped in allowed and mapped not in out:
            out.append(mapped)
    return out


def validate_generated_block(block: Dict[str, Any], vocab: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    errors: Dict[str, List[str]] = {}
    normalized: Dict[str, Any] = {}

    for field in FACET_FIELDS:
        items = normalize_list(block.get(field))
        items = apply_synonyms(field, items, vocab)
        normalized[field] = items
        allowed = set(vocab["facet_fields"][field])
        invalid = [item for item in items if item not in allowed]
        if invalid:
            errors[field] = invalid

    normalized["description_short"] = shorten(block.get("description_short"), 180)
    normalized["remark_short"] = shorten(block.get("remark_short"), 180)

    score = block.get("facet_confidence", 0.0)
    try:
        score = float(score)
    except Exception:
        score = 0.0
    normalized["facet_confidence"] = max(0.0, min(1.0, score))

    return normalized, errors


def detect_phase4_strategy(grounded_record: Dict[str, Any], compact_record: Dict[str, Any]) -> str:
    normalized = grounded_record.get("normalized", {}) or {}
    has_description = bool(normalize_whitespace(normalized.get("description")))
    has_materials = bool(normalize_whitespace(normalized.get("materials")))
    has_object = bool(normalize_whitespace(normalized.get("object")))

    if has_description or has_materials:
        return "text_only"
    if has_object:
        return "object_plus_image_fill"
    return "image_only"


def missing_fields_for_strategy(compact_record: Dict[str, Any], strategy: str) -> List[str]:
    missing: List[str] = []
    for field in FACET_FIELDS:
        if strategy == "object_plus_image_fill" and field == "garments" and compact_record.get("garments"):
            continue
        if not compact_record.get(field):
            missing.append(field)

    if not compact_record.get("description_short"):
        missing.append("description_short")
    if not compact_record.get("remark_short"):
        missing.append("remark_short")
    return missing


def merge_records(
    compact_record: Dict[str, Any],
    generated: Dict[str, Any],
    strategy: str,
    accepted: bool,
    needs_review: bool,
    prompt_model: str,
    missing_fields: List[str],
    group_key: Optional[str] = None,
    group_size: int = 1,
) -> Dict[str, Any]:
    merged = dict(compact_record)
    filled_fields: List[str] = []

    if accepted:
        for field in FACET_FIELDS:
            existing = normalize_list(merged.get(field))
            new_values = normalize_list(generated.get(field))

            if strategy == "object_plus_image_fill":
                if existing:
                    continue
                if new_values:
                    merged[field] = new_values
                    filled_fields.append(field)
            else:
                if not existing and new_values:
                    merged[field] = new_values
                    filled_fields.append(field)

        for field in TEXT_SUMMARY_FIELDS:
            existing_text = normalize_whitespace(merged.get(field))
            new_text = normalize_whitespace(generated.get(field))
            if not existing_text and new_text:
                merged[field] = new_text
                filled_fields.append(field)

        merged["facet_confidence"] = max(float(merged.get("facet_confidence", 0.0) or 0.0), float(generated["facet_confidence"]))
        merged["facet_source"] = FACET_SOURCE_FILLED
        merged["needs_image_fill"] = False if filled_fields or strategy == "image_only" else merged.get("needs_image_fill", False)
    else:
        merged["needs_image_fill"] = True

    merged["image_fill_applied"] = bool(accepted and filled_fields)
    merged["image_fill_strategy"] = strategy
    merged["image_fill_fields"] = filled_fields
    merged["image_fill_missing_fields_requested"] = missing_fields
    merged["image_fill_model"] = prompt_model
    merged["image_fill_prompt_version"] = PROMPT_VERSION
    merged["image_fill_group_key"] = group_key
    merged["image_fill_group_size"] = group_size
    merged["needs_review"] = needs_review
    merged["phase4_indexable"] = accepted

    return merged


def summarize_plan(records: List[Dict[str, Any]], normalized_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_records": len(records),
        "needs_image_fill_records": 0,
        "needs_image_fill_groups": 0,
        "by_asset_type": {},
        "by_strategy": {},
        "sample_candidates": [],
    }

    by_asset_type: Dict[str, int] = {}
    by_strategy: Dict[str, int] = {}
    groups: set[str] = set()

    for record in records:
        if not record.get("needs_image_fill"):
            continue
        summary["needs_image_fill_records"] += 1
        groups.add(make_group_key(record))
        asset_type = record.get("asset_type", "unknown")
        by_asset_type[asset_type] = by_asset_type.get(asset_type, 0) + 1
        grounded = normalized_by_id.get(record["record_id"], {})
        strategy = detect_phase4_strategy(grounded, record)
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        if len(summary["sample_candidates"]) < 10:
            summary["sample_candidates"].append(
                {
                    "record_id": record.get("record_id"),
                    "source_path": record.get("source_path"),
                    "asset_type": asset_type,
                    "strategy": strategy,
                    "facet_confidence": record.get("facet_confidence"),
                    "group_key": make_group_key(record),
                }
            )

    summary["needs_image_fill_groups"] = len(groups)
    summary["by_asset_type"] = by_asset_type
    summary["by_strategy"] = by_strategy
    return summary


def choose_group_image_paths(
    group_compact_records: List[Dict[str, Any]],
    images_root: str,
    max_group_images: int,
) -> Tuple[List[str], List[str]]:
    image_paths: List[str] = []
    missing: List[str] = []
    for record in group_compact_records:
        asset_type = record.get("asset_type", "unknown")
        if asset_type not in ASSET_TYPES_VISUAL:
            continue
        source_path = record.get("source_path")
        if not source_path:
            continue
        resolved = resolve_image_path(images_root, source_path)
        if Path(resolved).exists():
            if resolved not in image_paths:
                image_paths.append(resolved)
        else:
            missing.append(resolved)
    return image_paths[:max_group_images], missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 04 — fill partial grounded facets with image support")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Phase-03 compact facet JSONL input")
    parser.add_argument("--normalized-input", default=DEFAULT_NORMALIZED, help="Normalized grounded JSONL input")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="Frozen Ferré vocabulary JSON")
    parser.add_argument("--images-root", default=DEFAULT_IMAGES_ROOT, help="Root folder containing ALTA-MODA/... image files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Phase-04 filled JSONL output")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Phase-04 report JSON output")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Vertex AI location")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records")
    parser.add_argument("--only-needs-fill", action="store_true", help="Process only records already flagged with needs_image_fill=true")
    parser.add_argument("--resume", action="store_true", help="Append missing records to an existing output file")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Gemini. Only write a plan report.")
    parser.add_argument(
        "--fill-low-confidence-text",
        action="store_true",
        help="Also image-fill text-rich records with strategy text_only when they were flagged for low confidence.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip records whose image path does not exist instead of failing.",
    )
    parser.add_argument(
        "--max-group-images",
        type=int,
        default=6,
        help="Maximum number of images from the same outfit group to send in one Gemini prompt.",
    )
    args = parser.parse_args()

    vocab = load_json(args.vocab)
    compact_records = list(iter_jsonl(args.input))
    if args.limit is not None:
        compact_records = compact_records[: args.limit]

    normalized_by_id = {record["record_id"]: record for record in iter_jsonl(args.normalized_input)}
    group_members_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in compact_records:
        group_members_by_key[make_group_key(record)].append(record)

    plan_summary = summarize_plan(compact_records, normalized_by_id)

    if args.dry_run:
        report = {
            "phase": "phase04",
            "mode": "dry_run",
            "input": args.input,
            "normalized_input": args.normalized_input,
            "images_root": args.images_root,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "summary": plan_summary,
        }
        write_json(args.report, report)
        print(f"[ok] dry-run report written to {args.report}")
        return

    if genai is None or types is None:
        raise SystemExit("google-genai is not installed in this environment. Install project dependencies before running phase 04.")
    if "GCP_PROJECT" not in os.environ:
        raise SystemExit("GCP_PROJECT is not set.")
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS is not set.")

    client = genai.Client(vertexai=True, project=os.environ["GCP_PROJECT"], location=args.location)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    if args.resume and Path(args.output).exists():
        for record in iter_jsonl(args.output):
            already_done.add(record["record_id"])

    mode = "a" if args.resume and Path(args.output).exists() else "w"

    report_stats: Dict[str, Any] = {
        "phase": "phase04",
        "mode": "run",
        "input": args.input,
        "normalized_input": args.normalized_input,
        "images_root": args.images_root,
        "output": args.output,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "processed": 0,
        "group_calls": 0,
        "group_reused": 0,
        "skipped_resume": 0,
        "skipped_text_only": 0,
        "missing_image": 0,
        "llm_errors": 0,
        "applied": 0,
        "accepted_without_review": 0,
        "accepted_with_review": 0,
        "rejected_low_confidence": 0,
        "by_strategy": {},
        "by_asset_type": {},
        "plan_summary": plan_summary,
        "samples": [],
    }

    group_result_cache: Dict[str, Dict[str, Any]] = {}

    with open(args.output, mode, encoding="utf-8") as out_f:
        for compact_record in compact_records:
            record_id = compact_record["record_id"]
            if record_id in already_done:
                report_stats["skipped_resume"] += 1
                continue

            if args.only_needs_fill and not compact_record.get("needs_image_fill"):
                continue

            grounded_record = normalized_by_id.get(record_id)
            if grounded_record is None:
                merged = dict(compact_record)
                merged["needs_review"] = True
                merged["phase4_indexable"] = False
                merged["image_fill_error"] = "normalized_record_missing"
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                report_stats["llm_errors"] += 1
                continue

            strategy = detect_phase4_strategy(grounded_record, compact_record)
            asset_type = compact_record.get("asset_type", "unknown")
            report_stats["by_strategy"][strategy] = report_stats["by_strategy"].get(strategy, 0) + 1
            report_stats["by_asset_type"][asset_type] = report_stats["by_asset_type"].get(asset_type, 0) + 1

            if strategy == "text_only" and not args.fill_low_confidence_text:
                merged = dict(compact_record)
                merged["image_fill_applied"] = False
                merged["image_fill_strategy"] = strategy
                merged["image_fill_fields"] = []
                merged["image_fill_missing_fields_requested"] = []
                merged["image_fill_model"] = args.model
                merged["image_fill_prompt_version"] = PROMPT_VERSION
                merged["image_fill_group_key"] = make_group_key(compact_record)
                merged["image_fill_group_size"] = len(group_members_by_key[make_group_key(compact_record)])
                merged["needs_review"] = bool(merged.get("facet_confidence", 0.0) < 0.75)
                merged["phase4_indexable"] = True
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                report_stats["processed"] += 1
                report_stats["skipped_text_only"] += 1
                continue

            group_key = make_group_key(compact_record)
            group_compact_records = group_members_by_key[group_key]
            group_size = len(group_compact_records)
            group_strategy = compute_group_strategy(group_compact_records, normalized_by_id)

            if group_key in group_result_cache:
                cache_entry = group_result_cache[group_key]
                generated = cache_entry.get("generated")
                validation_errors = cache_entry.get("validation_errors", {})
                accepted = cache_entry.get("accepted", False)
                needs_review = cache_entry.get("needs_review", True)
                image_paths = cache_entry.get("image_paths", [])
                missing_fields_group = cache_entry.get("missing_fields", [])
                report_stats["group_reused"] += 1
            else:
                image_paths, missing_paths = choose_group_image_paths(group_compact_records, args.images_root, args.max_group_images)
                if not image_paths:
                    merged = dict(compact_record)
                    merged["needs_review"] = True
                    merged["phase4_indexable"] = False
                    merged["image_fill_error"] = f"missing_group_images:{group_key}"
                    merged["image_fill_group_key"] = group_key
                    merged["image_fill_group_size"] = group_size
                    out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    report_stats["processed"] += 1
                    report_stats["missing_image"] += 1
                    if not args.skip_missing_images:
                        print(f"[warn] missing images for group {group_key}: {missing_paths[:3]}")
                    continue

                missing_fields_group = build_group_missing_fields(group_compact_records, group_strategy)
                response_schema = build_response_schema(vocab, group_strategy)
                prompt = build_group_prompt(
                    group_key=group_key,
                    group_compact_records=group_compact_records,
                    normalized_by_id=normalized_by_id,
                    vocab=vocab,
                    strategy=group_strategy,
                    missing_fields=missing_fields_group,
                )
                generated_raw = call_multimodal_llm_with_retry(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    image_paths=image_paths,
                    response_schema=response_schema,
                    key=group_key,
                    temperature=args.temperature,
                )
                report_stats["group_calls"] += 1

                if generated_raw is None:
                    group_result_cache[group_key] = {
                        "generated": None,
                        "validation_errors": {},
                        "accepted": False,
                        "needs_review": True,
                        "image_paths": image_paths,
                        "missing_fields": missing_fields_group,
                    }
                    merged = dict(compact_record)
                    merged["needs_review"] = True
                    merged["phase4_indexable"] = False
                    merged["image_fill_error"] = "llm_generation_failed"
                    merged["image_fill_strategy"] = group_strategy
                    merged["image_fill_group_key"] = group_key
                    merged["image_fill_group_size"] = group_size
                    out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    report_stats["processed"] += 1
                    report_stats["llm_errors"] += 1
                    continue

                generated, validation_errors = validate_generated_block(generated_raw, vocab)
                accepted, needs_review = confidence_bucket(generated["facet_confidence"])
                group_result_cache[group_key] = {
                    "generated": generated,
                    "validation_errors": validation_errors,
                    "accepted": accepted,
                    "needs_review": needs_review,
                    "image_paths": image_paths,
                    "missing_fields": missing_fields_group,
                }

            if generated is None:
                merged = dict(compact_record)
                merged["needs_review"] = True
                merged["phase4_indexable"] = False
                merged["image_fill_error"] = "llm_generation_failed"
                merged["image_fill_strategy"] = group_strategy
                merged["image_fill_group_key"] = group_key
                merged["image_fill_group_size"] = group_size
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                report_stats["processed"] += 1
                report_stats["llm_errors"] += 1
                continue

            missing_fields_record = missing_fields_for_strategy(compact_record, strategy)
            merged = merge_records(
                compact_record=compact_record,
                generated=generated,
                strategy=strategy,
                accepted=accepted,
                needs_review=needs_review,
                prompt_model=args.model,
                missing_fields=missing_fields_record,
                group_key=group_key,
                group_size=group_size,
            )

            if validation_errors:
                merged.setdefault("validation_errors", {})
                merged["validation_errors"]["phase04_generated"] = validation_errors
                merged["validation_passed"] = False
            else:
                merged["validation_passed"] = bool(merged.get("validation_passed", True))

            merged["image_fill_generated_facet_confidence"] = generated["facet_confidence"]
            merged["image_fill_generated_preview"] = {
                field: generated.get(field)
                for field in [*FACET_FIELDS, *TEXT_SUMMARY_FIELDS]
                if generated.get(field)
            }
            merged["image_fill_group_image_count"] = len(group_result_cache[group_key].get("image_paths", []))

            if accepted and not needs_review:
                report_stats["accepted_without_review"] += 1
            elif accepted and needs_review:
                report_stats["accepted_with_review"] += 1
            else:
                report_stats["rejected_low_confidence"] += 1

            if merged.get("image_fill_applied"):
                report_stats["applied"] += 1

            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            report_stats["processed"] += 1

            if len(report_stats["samples"]) < 10:
                report_stats["samples"].append(
                    {
                        "record_id": record_id,
                        "group_key": group_key,
                        "group_size": group_size,
                        "strategy": strategy,
                        "group_strategy": group_strategy,
                        "asset_type": asset_type,
                        "applied": merged.get("image_fill_applied"),
                        "generated_confidence": generated["facet_confidence"],
                        "needs_review": merged.get("needs_review"),
                        "image_fill_fields": merged.get("image_fill_fields"),
                    }
                )

    write_json(args.report, report_stats)
    print(f"[ok] phase-04 output written to {args.output}")
    print(f"[ok] phase-04 report written to {args.report}")


if __name__ == "__main__":
    main()

# cd C:\Users\filip\Desktop\Polimi\ASP\AI4Fashion\ferre-rag-model\src\vector-db

# $env:GCP_PROJECT = "idyllic-psyche-487701-u7"
# $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\filip\Desktop\Polimi\ASP\AI4Fashion\secrets\llm-service-account.json"

#Remove-Item metadata\derived_grounded_facets\grounded_compact_outfit245.jsonl -ErrorAction Ignore
#Get-Content metadata\derived_grounded_facets\grounded_compact.jsonl | ForEach-Object {
#    $obj = $_ | ConvertFrom-Json
#    if ($obj.season -eq "FW1986-87" -and $obj.collection_line -eq "ALTA-MODA" -and $obj.outfit_id -eq "245") {
#        $_ | Add-Content metadata\derived_grounded_facets\grounded_compact_outfit245.jsonl
#    }
#}

#Test
#py -3 metadata\fill_partial_grounded_facets.py `
#  --input metadata\derived_grounded_facets\grounded_compact_outfit245.jsonl `
#  --output metadata\image_filled_grounded_facets\grounded_compact_filled_outfit245.jsonl `
#  --report metadata\image_filled_grounded_facets\grounded_compact_filled_outfit245_report.json `
#  --images-root "C:\Users\filip\Desktop\Polimi\ASP\AI4Fashion\ferre-rag-model\src\vector-db\input-datasets\ferre-designs" `
#  --model gemini-2.0-flash-001 `
#  --location us-central1 `
#  --max-group-images 6

#Whole pipeline
#py -3 metadata\fill_partial_grounded_facets.py `
#  --input metadata\derived_grounded_facets\grounded_compact_llm.jsonl `
#  --output metadata\image_filled_grounded_facets\grounded_compact_filled_llm.jsonl `
#  --report metadata\image_filled_grounded_facets\grounded_compact_filled_llm_report.json `
#  --images-root "C:\Users\filip\Desktop\Polimi\ASP\AI4Fashion\ferre-rag-model\src\vector-db\input-datasets\ferre-designs" `
#  --model gemini-2.0-flash-001 `
#  --location us-central1 `
#  --max-group-images 20