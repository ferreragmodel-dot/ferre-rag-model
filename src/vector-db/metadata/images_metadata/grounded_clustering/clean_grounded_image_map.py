#!/usr/bin/env python3
"""Clean the merged grounded image map while preserving its current schema.

The input is expected to be a JSON object shaped like:
    {image_rel_path: {field: value, ...}, ...}

The output keeps the same structure and the same metadata fields for each image,
adding only one extra boolean flag per image record to indicate whether that
record was modified by the cleaning pass.

Current cleaning scope:
- repair common mojibake / encoding corruption in keys and string fields
- normalize Unicode whitespace
- normalize text punctuation in metadata values
- preserve the normalized path style for image keys
- apply a small set of safe known repairs carried over from earlier cleanup work
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "grounded_image_map"
    / "all_collections_merged.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "grounded_image_map"
    / "all_collections_merged_cleaned.json"
)
DEFAULT_FLAG_FIELD = "cleaned"

SUSPECT_MOJIBAKE_TOKENS = (
    "\u00c3",
    "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u0153",
    "\u00e2\u20ac\u009d",
    "\u00e2\u20ac\u02dc",
    "\u00e2\u20ac\u201c",
    "\u00e2\u20ac\u201d",
    "\u00e2\u20ac\u00a6",
    "\u00e2\u201a\u00ac",
    "\u00c2",
    "\u00ef\u00ac",
)

KNOWN_REPAIRS = {
    ("condition", "Goog"): "Good",
    ("object", "Completo tre prezzi"): "Completo tre pezzi",
    ("object", "Three-piece evenng dress"): "Three-piece evening dress",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the merged grounded image map and add a modified flag."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flag-field", default=DEFAULT_FLAG_FIELD)
    return parser.parse_args()


def suspicious_score(text: str) -> int:
    return sum(text.count(token) for token in SUSPECT_MOJIBAKE_TOKENS)


def try_utf8_redecode(text: str, encoding: str) -> str:
    try:
        return text.encode(encoding).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def repair_mojibake(text: str) -> str:
    current = text
    for _ in range(2):
        current_score = suspicious_score(current)
        best = current
        best_score = current_score
        for encoding in ("latin-1", "cp1252"):
            candidate = try_utf8_redecode(current, encoding)
            candidate_score = suspicious_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        if best == current:
            break
        current = best
    return current


def normalize_text_value(text: str) -> str:
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalize_path_key(path: str) -> str:
    path = repair_mojibake(path)
    path = unicodedata.normalize("NFKC", path)
    path = path.replace("\\", "/").strip()
    parts = [part for part in path.split("/") if part]
    cleaned_parts = []
    for part in parts:
        part = part.replace("\u00a0", " ")
        part = re.sub(r"\s+", "_", part.strip()).lower()
        cleaned_parts.append(part)
    return "/".join(cleaned_parts)


def clean_scalar(field: str, value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    cleaned = normalize_text_value(value)
    cleaned = KNOWN_REPAIRS.get((field, cleaned), cleaned)
    if cleaned == "":
        return None
    return cleaned


def clean_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    cleaned: Dict[str, Any] = {}
    changed = False

    for field, value in record.items():
        cleaned_value = clean_scalar(field, value)
        cleaned[field] = cleaned_value
        if cleaned_value != value:
            changed = True

    return cleaned, changed


def clean_payload(
    payload: Dict[str, Dict[str, Any]],
    flag_field: str,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    cleaned_payload: Dict[str, Dict[str, Any]] = {}
    modified_count = 0

    for original_key, record in payload.items():
        cleaned_key = normalize_path_key(original_key)
        cleaned_record, record_changed = clean_record(record)
        key_changed = cleaned_key != original_key
        modified = key_changed or record_changed
        cleaned_record[flag_field] = modified
        if modified:
            modified_count += 1

        if cleaned_key in cleaned_payload:
            raise ValueError(
                f"Cleaning produced a duplicate image key: {cleaned_key!r} "
                f"(from {original_key!r})"
            )

        cleaned_payload[cleaned_key] = cleaned_record

    return cleaned_payload, modified_count


def main() -> int:
    args = parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cleaned_payload, modified_count = clean_payload(payload, flag_field=args.flag_field)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cleaned_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Read {len(payload)} image records from {args.input}")
    print(f"Wrote cleaned archive to {args.output}")
    print(f"Modified {modified_count} image records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# docker compose run --rm --entrypoint /bin/bash llm-rag-cli -lc "cd /app && source /.venv/bin/activate && python metadata/images_metadata/grounded_clustering/clean_grounded_image_map.py"
