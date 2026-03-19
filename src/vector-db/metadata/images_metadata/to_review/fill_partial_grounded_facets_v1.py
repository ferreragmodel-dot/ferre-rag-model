#!/usr/bin/env python3
"""Phase 04 — fill partial grounded compact facets with multimodal Gemini.

Purpose
-------
This script takes the phase-03 compact facet output and resolves *partial* grounded
records using the corresponding image when text evidence is missing.

It is designed to work with either:
- heuristic phase 03 output: metadata/derived_grounded_facets/grounded_compact.jsonl
- LLM phase 03 output:       metadata/derived_grounded_facets/grounded_compact_llm.jsonl

The script follows the phase-04 decision table:
1. if description/materials exist -> keep text-derived facets only (no image fill by default)
2. if object exists but description/materials are missing -> keep confirmed text garments,
   fill only the missing facets from image
3. if object, description, and materials are all missing -> generate the facet block from image

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
PROMPT_VERSION = "phase04_image_fill_v1"
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
        "description": "One concise factual sentence grounded in visible evidence.",
    }
    properties["remark_short"] = {
        "type": "STRING",
        "nullable": True,
        "description": "One concise style-oriented note grounded in visible evidence, or null.",
    }
    properties["facet_confidence"] = {
        "type": "NUMBER",
        "description": "Confidence from 0.0 to 1.0 for the newly generated image-based facets.",
    }

    if strategy == "object_plus_image_fill":
        description = "Fill only missing facets from the image. Preserve already confirmed text-derived facets."
    else:
        description = "Generate a compact facet block from the image using only allowed vocabulary values."

    return {
        "type": "OBJECT",
        "description": description,
        "propertyOrdering": [*FACET_FIELDS, *TEXT_SUMMARY_FIELDS, "facet_confidence"],
        "required": [*FACET_FIELDS, *TEXT_SUMMARY_FIELDS, "facet_confidence"],
        "properties": properties,
    }


def to_json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_prompt(
    compact_record: Dict[str, Any],
    grounded_record: Dict[str, Any],
    vocab: Dict[str, Any],
    strategy: str,
    missing_fields: List[str],
) -> str:
    raw = grounded_record.get("raw", {}) or {}
    normalized = grounded_record.get("normalized", {}) or {}

    compact_vocab = {field: vocab["facet_fields"][field] for field in FACET_FIELDS}
    locked_non_empty = {field: compact_record.get(field, []) for field in FACET_FIELDS if compact_record.get(field)}

    strategy_rules = {
        "text_only": [
            "Do not use image completion. This record should remain text-derived only.",
        ],
        "object_plus_image_fill": [
            "The object field exists but description/materials are missing.",
            "Preserve already confirmed text-derived facets.",
            "Use the image only to fill missing facet arrays and optional short summaries.",
            "If garments are already present, do not replace them.",
            "If garments are empty, you may add garments only when visually clear.",
        ],
        "image_only": [
            "The textual record is too incomplete; derive the compact facet block from the image.",
            "Stay conservative: prefer [] over uncertain guesses.",
        ],
    }

    instructions = {
        "task": "Fill missing Ferré retrieval facets from one archive image.",
        "rules": [
            "Use only values from the supplied controlled vocabulary.",
            "Return arrays for facet fields and never comma-joined strings.",
            "Do not generate archival fields such as acquisition, bibliography, inventory, present_location, condition, or exhibitions.",
            "Do not invent textile supplier names, article codes, percentages, or hidden garment details.",
            "Focus on the primary outfit or main visible subject in the image.",
            "If a facet is not visually supportable, return an empty array for that facet.",
            "description_short and remark_short must be concise and factual.",
            *strategy_rules[strategy],
        ],
    }

    payload = {
        "archive_context": {
            "record_id": compact_record.get("record_id"),
            "source_path": compact_record.get("source_path"),
            "source_file": compact_record.get("source_file"),
            "season": compact_record.get("season"),
            "year": compact_record.get("year"),
            "collection": compact_record.get("collection"),
            "collection_line": compact_record.get("collection_line"),
            "look": compact_record.get("look"),
            "asset_type": compact_record.get("asset_type"),
            "raw_completeness": compact_record.get("raw_completeness"),
            "image_fill_strategy": strategy,
        },
        "grounded_text_available": {
            "object": normalized.get("object"),
            "description": normalized.get("description"),
            "materials": normalized.get("materials"),
            "working_process": normalized.get("working_process"),
            "remark": normalized.get("remark"),
        },
        "raw_text_traceability": {
            "object": raw.get("object"),
            "description": raw.get("description"),
            "materials": raw.get("materials"),
            "working_process": raw.get("working_process"),
            "remark": raw.get("remark"),
        },
        "existing_compact_record": {field: compact_record.get(field) for field in [*FACET_FIELDS, *TEXT_SUMMARY_FIELDS]},
        "locked_non_empty_fields": locked_non_empty,
        "missing_fields_to_fill": missing_fields,
        "allowed_vocabulary": compact_vocab,
    }

    return f"{to_json_text(instructions)}\n\nArchive payload:\n{to_json_text(payload)}\n"


def call_multimodal_llm_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    image_path: str,
    response_schema: Dict[str, Any],
    key: str,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    backoff = INITIAL_BACKOFF

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    mime_type = guess_mime(image_path)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=types.Content(
                    role="user",
                    parts=[
                        {"text": prompt},
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                ),
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
    merged["needs_review"] = needs_review
    merged["phase4_indexable"] = accepted

    return merged


def summarize_plan(records: List[Dict[str, Any]], normalized_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_records": len(records),
        "needs_image_fill_records": 0,
        "by_asset_type": {},
        "by_strategy": {},
        "sample_candidates": [],
    }

    by_asset_type: Dict[str, int] = {}
    by_strategy: Dict[str, int] = {}

    for record in records:
        if not record.get("needs_image_fill"):
            continue
        summary["needs_image_fill_records"] += 1
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
                }
            )

    summary["by_asset_type"] = by_asset_type
    summary["by_strategy"] = by_strategy
    return summary


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
    args = parser.parse_args()

    vocab = load_json(args.vocab)
    compact_records = list(iter_jsonl(args.input))
    if args.limit is not None:
        compact_records = compact_records[: args.limit]

    normalized_by_id = {record["record_id"]: record for record in iter_jsonl(args.normalized_input)}

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
                merged["needs_review"] = bool(merged.get("facet_confidence", 0.0) < 0.75)
                merged["phase4_indexable"] = True
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                report_stats["processed"] += 1
                report_stats["skipped_text_only"] += 1
                continue

            if asset_type not in ASSET_TYPES_VISUAL:
                merged = dict(compact_record)
                merged["needs_review"] = True
                merged["phase4_indexable"] = False
                merged["image_fill_error"] = f"unsupported_asset_type:{asset_type}"
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                report_stats["processed"] += 1
                report_stats["llm_errors"] += 1
                continue

            image_path = resolve_image_path(args.images_root, compact_record["source_path"])
            if not Path(image_path).exists():
                merged = dict(compact_record)
                merged["needs_review"] = True
                merged["phase4_indexable"] = False
                merged["image_fill_error"] = f"missing_image:{image_path}"
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                report_stats["processed"] += 1
                report_stats["missing_image"] += 1
                if not args.skip_missing_images:
                    print(f"[warn] missing image for {record_id}: {image_path}")
                continue

            missing_fields = missing_fields_for_strategy(compact_record, strategy)
            response_schema = build_response_schema(vocab, strategy)
            prompt = build_prompt(compact_record, grounded_record, vocab, strategy, missing_fields)
            generated_raw = call_multimodal_llm_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                image_path=image_path,
                response_schema=response_schema,
                key=record_id,
                temperature=args.temperature,
            )

            if generated_raw is None:
                merged = dict(compact_record)
                merged["needs_review"] = True
                merged["phase4_indexable"] = False
                merged["image_fill_error"] = "llm_generation_failed"
                merged["image_fill_strategy"] = strategy
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                report_stats["processed"] += 1
                report_stats["llm_errors"] += 1
                continue

            generated, validation_errors = validate_generated_block(generated_raw, vocab)
            accepted, needs_review = confidence_bucket(generated["facet_confidence"])
            merged = merge_records(
                compact_record=compact_record,
                generated=generated,
                strategy=strategy,
                accepted=accepted,
                needs_review=needs_review,
                prompt_model=args.model,
                missing_fields=missing_fields,
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
                        "strategy": strategy,
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



#Remove-Item metadata\derived_grounded_facets\grounded_compact_test3.jsonl -ErrorAction Ignore
#
#$i = 0
#Get-Content metadata\derived_grounded_facets\grounded_compact.jsonl | ForEach-Object {
#    $obj = $_ | ConvertFrom-Json
#    if ($obj.needs_image_fill -eq $true -and $i -lt 3) {
#        $_ | Add-Content metadata\derived_grounded_facets\grounded_compact_test3.jsonl
#        $i++
#    }
#} 


#py -3 metadata\fill_partial_grounded_facets.py `
#  --input metadata\derived_grounded_facets\grounded_compact_test3.jsonl `
#  --output metadata\image_filled_grounded_facets\grounded_compact_filled_test3.jsonl `
#  --report metadata\image_filled_grounded_facets\grounded_compact_filled_test3_report.json `
#  --images-root "C:\Users\filip\Desktop\Polimi\ASP\AI4Fashion\ferre-rag-model\src\vector-db\input-datasets\ferre-designs" `
#  --model gemini-2.0-flash-001 `
#  --location us-central1




