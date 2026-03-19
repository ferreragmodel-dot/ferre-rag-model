#!/usr/bin/env python3
"""Derive compact Ferré retrieval facets from grounded archive text using Vertex AI.

Why this exists
---------------
Phase 03 already has a deterministic heuristic extractor. This LLM variant is a drop-in
replacement when you want better handling of:
- bilingual Italian/English descriptions
- OCR noise and unusual phrasing
- archive-specific style tags and nuanced silhouettes
- less rigid garment decomposition

Design goals
------------
- Reuse the same stack already present in the repo: google-genai + Vertex AI / Gemini.
- Keep the output shape as close as possible to the heuristic phase-03 file so phase 04
  can consume it with minimal or zero code changes.
- Use structured output + enums from the frozen Ferré vocabulary.
- Stay conservative: if text does not support a facet, return [] or null rather than guess.

Input
-----
- metadata/normalized_grounded/grounded_raw.jsonl
- metadata/config/ferre_facet_vocabulary_v1.json

Output
------
- metadata/derived_grounded_facets/grounded_compact_llm.jsonl
- metadata/derived_grounded_facets/grounded_compact_llm_report.json

Optional hybrid mode
--------------------
You can pass the heuristic phase-03 output as a fallback source. In that mode, if the LLM
returns an empty facet field but the heuristic extractor found something valid, the
heuristic value is copied in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types


DEFAULT_INPUT = "metadata/normalized_grounded/grounded_raw.jsonl"
DEFAULT_VOCAB = "metadata/config/ferre_facet_vocabulary_v1.json"
DEFAULT_OUTPUT = "metadata/derived_grounded_facets/grounded_compact_llm.jsonl"
DEFAULT_REPORT = "metadata/derived_grounded_facets/grounded_compact_llm_report.json"
DEFAULT_MODEL = "gemini-2.0-flash-001"
DEFAULT_LOCATION = "us-central1"
SCHEMA_VERSION = "ferre_facets_v1"
FACET_SOURCE = "derived_from_grounded_llm_v1"
PROMPT_VERSION = "phase03_llm_prompt_v1"

MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_BACKOFF = 32

ENUM_FACET_FIELDS = [
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

TEXT_FIELD_MAP = {
    "object": "object",
    "description": "description",
    "materials": "materials",
    "working_process": "working_process",
    "remark": "remark",
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
    clipped = text[: max_len - 1].rstrip(" ,;:-")
    return clipped + "…"


def to_json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_response_schema(vocab: Dict[str, Any]) -> Dict[str, Any]:
    facet_fields = vocab["facet_fields"]

    properties: Dict[str, Any] = {}
    for field in ENUM_FACET_FIELDS:
        properties[field] = {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": facet_fields[field],
            },
        }

    properties["description_short"] = {
        "type": "STRING",
        "nullable": True,
        "description": "One concise sentence, max about 25 words, grounded only in the record.",
    }
    properties["remark_short"] = {
        "type": "STRING",
        "nullable": True,
        "description": "One concise sentence, max about 25 words, grounded only in the remark if present.",
    }

    return {
        "type": "OBJECT",
        "propertyOrdering": [
            *ENUM_FACET_FIELDS,
            "description_short",
            "remark_short",
        ],
        "required": [
            *ENUM_FACET_FIELDS,
            "description_short",
            "remark_short",
        ],
        "properties": properties,
    }


def build_prompt(record: Dict[str, Any], vocab: Dict[str, Any]) -> str:
    raw = record.get("raw", {}) or {}
    normalized = record.get("normalized", {}) or {}

    compact_vocab = {field: vocab["facet_fields"][field] for field in ENUM_FACET_FIELDS}

    instructions = {
        "task": (
            "Extract compact retrieval facets from one grounded Ferré archive record. "
            "Use only evidence from the textual record. Do not infer visual details not stated in text."
        ),
        "rules": [
            "Use only values from the supplied vocabulary.",
            "If a facet is not supported by the text, return an empty array for that facet.",
            "Do not include supplier names, article codes, percentages, inventory data, bibliography, acquisition, or present location inside facet arrays.",
            "Decompose outfit labels into actual garments only when the text supports the decomposition.",
            "Keep description_short factual and concise.",
            "Keep remark_short factual and concise; use null if remark is absent or uninformative.",
            "Do not invent materials, colors, or patterns.",
            "The archive is bilingual; Italian and English may appear in the same record.",
            "The main goal is retrieval facets, not literary prose.",
        ],
    }

    payload = {
        "archive_context": {
            "record_id": record.get("record_id"),
            "source_path": record.get("source_path"),
            "source_file": record.get("source_file"),
            "season": record.get("season"),
            "year": record.get("year"),
            "collection": record.get("collection"),
            "collection_line": record.get("collection_line"),
            "look": record.get("look"),
            "file": record.get("file"),
            "asset_type": record.get("asset_type"),
            "raw_completeness": record.get("raw_completeness"),
        },
        "normalized_text_fields": {
            "object": normalized.get("object"),
            "description": normalized.get("description"),
            "materials": normalized.get("materials"),
            "working_process": normalized.get("working_process"),
            "remark": normalized.get("remark"),
        },
        "raw_text_fields_for_traceability": {
            "object": raw.get("object"),
            "description": raw.get("description"),
            "materials": raw.get("materials"),
            "working_process": raw.get("working_process"),
            "remark": raw.get("remark"),
        },
        "allowed_vocabulary": compact_vocab,
    }

    return (
        f"{to_json_text(instructions)}\n\n"
        f"Archive record:\n{to_json_text(payload)}\n"
    )


def call_llm_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    response_schema: Dict[str, Any],
    key: str,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
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


def normalize_scalar_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                item = item.strip()
                if item:
                    out.append(item)
        return out
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        # Defensive fallback only. Structured output should already return arrays.
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value]
    return []


def apply_synonyms(field: str, items: List[str], vocab: Dict[str, Any]) -> List[str]:
    synonym_map = vocab.get("synonym_maps", {}).get(field, {})
    allowed = set(vocab["facet_fields"][field])

    out: List[str] = []
    for item in items:
        norm = normalize_whitespace(item)
        if not norm:
            continue
        low = norm.lower()
        mapped = synonym_map.get(low, low)

        if isinstance(mapped, list):
            candidates = mapped
        else:
            candidates = [mapped]

        for cand in candidates:
            cand = normalize_whitespace(cand)
            if not cand:
                continue
            cand = cand.lower()
            if cand in allowed and cand not in out:
                out.append(cand)
    return out


def validate_facets(compact: Dict[str, Any], vocab: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    errors: Dict[str, Any] = {}
    for field in ENUM_FACET_FIELDS:
        values = compact.get(field, [])
        if not isinstance(values, list):
            errors[field] = {"error": "not_a_list", "value": values}
            continue
        bad = [v for v in values if v not in vocab["facet_fields"][field]]
        if bad:
            errors[field] = {"error": "invalid_enum_values", "values": bad}
    return (len(errors) == 0, errors)


def merge_with_heuristic_fallback(
    compact: Dict[str, Any],
    heuristic_map: Dict[str, Dict[str, Any]],
    record_id: str,
) -> None:
    if not heuristic_map or record_id not in heuristic_map:
        return

    base = heuristic_map[record_id]

    for field in ENUM_FACET_FIELDS:
        llm_vals = list(compact.get(field, []) or [])
        heur_vals = list(base.get(field, []) or [])

        llm_set = set(llm_vals)
        heur_set = set(heur_vals)

        # Case 1: LLM empty -> use heuristic
        if not llm_vals:
            compact[field] = heur_vals
            continue

        # Case 2: LLM output is a subset of heuristic output -> use heuristic
        if llm_set and llm_set.issubset(heur_set):
            compact[field] = heur_vals

    if not compact.get("description_short"):
        compact["description_short"] = base.get("description_short")

    if not compact.get("remark_short"):
        compact["remark_short"] = base.get("remark_short")


def compute_needs_image_fill(record: Dict[str, Any], compact: Dict[str, Any]) -> bool:
    raw = record.get("raw", {}) or {}
    raw_completeness = record.get("raw_completeness")

    has_object = bool(normalize_whitespace(raw.get("object")))
    has_description = bool(normalize_whitespace(raw.get("description")))
    has_materials = bool(normalize_whitespace(raw.get("materials")))

    if raw_completeness in {"partial", "minimal"}:
        return True
    if not has_description and not has_materials:
        return True
    if not has_object and not compact.get("garments"):
        return True
    if not compact.get("garments") and not compact.get("material_families"):
        return True
    return False


def compute_confidence(record: Dict[str, Any], compact: Dict[str, Any], validation_passed: bool) -> float:
    if not validation_passed:
        return 0.40

    raw = record.get("raw", {}) or {}
    signal_fields = [
        raw.get("object"),
        raw.get("description"),
        raw.get("materials"),
        raw.get("working_process"),
        raw.get("remark"),
    ]
    signal_count = sum(1 for x in signal_fields if normalize_whitespace(x))

    populated_facets = sum(1 for field in ENUM_FACET_FIELDS if compact.get(field))
    score = 0.45
    score += min(signal_count, 5) * 0.08
    score += min(populated_facets, 6) * 0.03
    if record.get("raw_completeness") == "full":
        score += 0.08
    if compact.get("description_short"):
        score += 0.03
    if compact.get("remark_short"):
        score += 0.02
    return round(min(score, 0.98), 2)


def load_heuristic_map(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        rid = row.get("record_id")
        if rid:
            out[rid] = row
    return out


def maybe_resume_existing(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    existing: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        rid = row.get("record_id")
        if rid:
            existing[rid] = row
    return existing


def compact_from_llm_output(
    record: Dict[str, Any],
    llm_output: Dict[str, Any],
    vocab: Dict[str, Any],
    heuristic_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "record_id": record.get("record_id"),
        "source_path": record.get("source_path"),
        "source_file": record.get("source_file"),
        "outfit_id": record.get("outfit_id"),
        "season": record.get("season"),
        "year": record.get("year"),
        "collection": record.get("collection"),
        "collection_line": record.get("collection_line"),
        "look": record.get("look"),
        "file": record.get("file"),
        "asset_type": record.get("asset_type"),
        "metadata_source": record.get("metadata_source", "grounded"),
        "raw_completeness": record.get("raw_completeness"),
    }

    for field in ENUM_FACET_FIELDS:
        values = normalize_scalar_to_list(llm_output.get(field, []))
        values = apply_synonyms(field, values, vocab)
        compact[field] = values

    compact["description_raw"] = normalize_whitespace((record.get("raw") or {}).get("description"))
    compact["materials_raw"] = normalize_whitespace((record.get("raw") or {}).get("materials"))
    compact["remark_raw"] = normalize_whitespace((record.get("raw") or {}).get("remark"))

    compact["description_short"] = shorten(llm_output.get("description_short"), max_len=180)
    compact["remark_short"] = shorten(llm_output.get("remark_short"), max_len=180)

    merge_with_heuristic_fallback(compact, heuristic_map, compact["record_id"])

    validation_passed, validation_errors = validate_facets(compact, vocab)
    compact["needs_image_fill"] = compute_needs_image_fill(record, compact)
    compact["facet_confidence"] = compute_confidence(record, compact, validation_passed)
    compact["facet_source"] = FACET_SOURCE
    compact["validation_passed"] = validation_passed
    compact["validation_errors"] = validation_errors
    compact["schema_version"] = SCHEMA_VERSION
    compact["llm_prompt_version"] = PROMPT_VERSION
    return compact


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive Ferré compact facets from grounded text with Vertex AI")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input grounded_raw.jsonl")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="Facet vocabulary JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output grounded_compact_llm.jsonl")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Output report JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION), help="Vertex AI location")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--resume", action="store_true", help="Skip records already present in the output file")
    parser.add_argument(
        "--heuristic-fallback",
        default=None,
        help="Optional grounded_compact.jsonl from the heuristic phase-03 run. Empty LLM fields fall back to it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "GCP_PROJECT" not in os.environ and "GOOGLE_CLOUD_PROJECT" not in os.environ:
        raise SystemExit("Set GCP_PROJECT or GOOGLE_CLOUD_PROJECT before running.")
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise SystemExit("Set GOOGLE_APPLICATION_CREDENTIALS before running.")

    project = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    vocab = load_json(args.vocab)
    response_schema = build_response_schema(vocab)
    heuristic_map = load_heuristic_map(args.heuristic_fallback)
    existing = maybe_resume_existing(args.output) if args.resume else {}

    client = genai.Client(
        vertexai=True,
        project=project,
        location=args.location,
        http_options=types.HttpOptions(api_version="v1"),
    )

    rows: List[Dict[str, Any]] = []
    skipped_existing = 0
    llm_failures = 0
    total = 0

    try:
        for record in iter_jsonl(args.input):
            total += 1
            if args.limit is not None and len(rows) >= args.limit:
                break

            record_id = record.get("record_id")
            if args.resume and record_id in existing:
                rows.append(existing[record_id])
                skipped_existing += 1
                continue

            prompt = build_prompt(record, vocab)
            llm_output = call_llm_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                response_schema=response_schema,
                key=record_id or f"row-{total}",
                temperature=args.temperature,
            )

            if llm_output is None:
                llm_failures += 1
                # Minimal record preserving phase-03 interface.
                compact = {
                    "record_id": record.get("record_id"),
                    "source_path": record.get("source_path"),
                    "source_file": record.get("source_file"),
                    "outfit_id": record.get("outfit_id"),
                    "season": record.get("season"),
                    "year": record.get("year"),
                    "collection": record.get("collection"),
                    "collection_line": record.get("collection_line"),
                    "look": record.get("look"),
                    "file": record.get("file"),
                    "asset_type": record.get("asset_type"),
                    "metadata_source": record.get("metadata_source", "grounded"),
                    "raw_completeness": record.get("raw_completeness"),
                    **{field: [] for field in ENUM_FACET_FIELDS},
                    "description_raw": normalize_whitespace((record.get("raw") or {}).get("description")),
                    "materials_raw": normalize_whitespace((record.get("raw") or {}).get("materials")),
                    "remark_raw": normalize_whitespace((record.get("raw") or {}).get("remark")),
                    "description_short": None,
                    "remark_short": None,
                    "facet_confidence": 0.0,
                    "facet_source": FACET_SOURCE,
                    "needs_image_fill": True,
                    "validation_passed": False,
                    "validation_errors": {"llm": {"error": "generation_failed"}},
                    "schema_version": SCHEMA_VERSION,
                    "llm_prompt_version": PROMPT_VERSION,
                }
                merge_with_heuristic_fallback(compact, heuristic_map, compact["record_id"])
                rows.append(compact)
                print(f"[fail] {record_id}")
                continue

            compact = compact_from_llm_output(record, llm_output, vocab, heuristic_map)
            rows.append(compact)
            print(f"[ok] {record_id}")

    finally:
        client.close()

    save_jsonl(args.output, rows)

    validation_passed_count = sum(1 for r in rows if r.get("validation_passed"))
    needs_image_fill_count = sum(1 for r in rows if r.get("needs_image_fill"))
    avg_conf = round(sum(float(r.get("facet_confidence", 0.0)) for r in rows) / max(len(rows), 1), 3)

    report = {
        "input": args.input,
        "output": args.output,
        "report": args.report,
        "model": args.model,
        "location": args.location,
        "temperature": args.temperature,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "facet_source": FACET_SOURCE,
        "records_written": len(rows),
        "records_scanned": total,
        "skipped_existing": skipped_existing,
        "llm_failures": llm_failures,
        "validation_passed": validation_passed_count,
        "needs_image_fill": needs_image_fill_count,
        "avg_facet_confidence": avg_conf,
        "heuristic_fallback_used": bool(args.heuristic_fallback),
    }
    write_json(args.report, report)

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()



# run a small test
# python metadata/derive_grounded_compact_facets_llm.py \
#  --limit 20 \
#  --heuristic-fallback metadata/derived_grounded_facets/grounded_compact.jsonl

# run full
# python metadata/derive_grounded_compact_facets_llm.py \
#  --heuristic-fallback metadata/derived_grounded_facets/grounded_compact.jsonl