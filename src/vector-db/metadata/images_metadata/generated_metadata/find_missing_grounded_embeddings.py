#!/usr/bin/env python3
"""Find image embeddings that are missing from the grounded cleaned archive."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs"
DEFAULT_GROUNDED_JSON = (
    Path(__file__).resolve().parents[1]
    / "grounded_metadata"
    / "grounded_image_map"
    / "all_collections_merged_cleaned.json"
)
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "embeddings_missing_from_grounded.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List embedded images that are not present in the grounded cleaned archive."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory containing embeddings-images*.jsonl files.",
    )
    parser.add_argument(
        "--grounded-json",
        type=Path,
        default=DEFAULT_GROUNDED_JSON,
        help="Path to all_collections_merged_cleaned.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the missing-image JSON.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> str:
    parts = [part for part in path.replace("\\", "/").strip().split("/") if part]
    return "/".join(re.sub(r"\s+", "_", part.strip()).lower() for part in parts)


def iter_embedding_files(outputs_dir: Path) -> Iterable[Path]:
    yield from sorted(outputs_dir.glob("embeddings-images*.jsonl"))


def load_grounded_paths(path: Path) -> Set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {normalize_path(key) for key in payload.keys()}


def build_missing_entries(outputs_dir: Path, grounded_json: Path) -> List[Dict[str, Any]]:
    grounded_paths = load_grounded_paths(grounded_json)
    missing: List[Dict[str, Any]] = []

    for jsonl_path in iter_embedding_files(outputs_dir):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                raw_path = record.get("path")
                embedding = record.get("embedding")
                if not isinstance(raw_path, str) or not isinstance(embedding, list):
                    continue

                normalized_path = normalize_path(raw_path)
                if normalized_path in grounded_paths:
                    continue

                missing.append(
                    {
                        "path": normalized_path,
                        "embedding": embedding,
                        "pdf_status": "missing",
                    }
                )

    return missing


def main() -> int:
    args = parse_args()
    missing = build_missing_entries(args.outputs_dir, args.grounded_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Checked embeddings in {args.outputs_dir}")
    print(f"Compared against grounded archive {args.grounded_json}")
    print(f"Wrote {len(missing)} missing image embeddings -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
