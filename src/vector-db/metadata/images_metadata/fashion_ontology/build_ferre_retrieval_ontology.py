#!/usr/bin/env python3
"""Build a compact Ferre retrieval ontology from grounded archive metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - depends on local runtime
    genai = None
    types = None


SLOT_NAMES = [
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

BUNDLE_FIELDS = ["object", "description", "materials", "working_process", "remark", "label"]

SLOT_LIMITS_ARCHIVE = {
    "garments_tags": 28,
    "colors_tags": 20,
    "material_tags": 24,
    "patterns_tags": 18,
    "silhouette_tags": 24,
    "length_tags": 16,
    "neckline_tags": 18,
    "sleeve_tags": 16,
    "closure_tags": 16,
    "embellishment_tags": 24,
    "style_tags": 18,
}

SLOT_LIMITS_EXTERNAL = {
    "garments_tags": 40,
    "colors_tags": 24,
    "material_tags": 32,
    "patterns_tags": 24,
    "silhouette_tags": 24,
    "length_tags": 18,
    "neckline_tags": 24,
    "sleeve_tags": 18,
    "closure_tags": 18,
    "embellishment_tags": 24,
    "style_tags": 16,
}

SUPPORTED_SEED_SUFFIXES = {".json", ".jsonl", ".txt", ".owl", ".xml", ".rdf"}
DEFAULT_MODEL = "gemini-2.0-flash-001"
DEFAULT_LOCATION = "us-central1"
MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_BACKOFF = 24
DEFAULT_SINGLE_PASS_CHAR_LIMIT = 40_000
DEFAULT_BATCH_SIZE = 20

MOJIBAKE_MARKERS = ("Ã", "â", "Â", "Ë", "È", "É", "�")

BASE_DIR = Path(__file__).resolve().parent
IMAGES_METADATA_DIR = BASE_DIR.parent
VECTOR_DB_DIR = BASE_DIR.parents[2]


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(BASE_DIR)

DEFAULT_ARCHIVE_INPUT = (
    IMAGES_METADATA_DIR
    / "grounded_clustering"
    / "grounded_image_map"
    / "all_collections_merged_cleaned.json"
)
DEFAULT_SEED_ROOT = BASE_DIR / "public_fashion_onologies"
DEFAULT_OUTPUT = BASE_DIR / "ferre_retrieval_ontology.json"

SEED_PRIORITY_NAMES = {
    "category_attributes_descriptions.json": 200,
    "list_category_cloth.txt": 190,
    "list_attr_cloth.txt": 190,
    "clothingtypes.json": 170,
    "clothingmaterials.json": 170,
    "clothingstyles.json": 155,
    "colors.json": 150,
    "clothing-materials.json": 150,
}

SEED_PATH_BLACKLIST = (
    "/eval/",
    "/anno_fine/train",
    "/anno_fine/test",
    "/anno_fine/val",
    "/fashion landmark detection benchmark/",
    "/weatherconditions",
    "/events",
    "/religions",
    "/seasons",
    "/date.xml",
    "/clothingsizes",
    "/readme",
    "/requirements.txt",
    "/license.txt",
    "/project-notes.txt",
    "/project-steps.txt",
    "/useful-queries.txt",
    "/sample.json",
    "/fake_results.json",
    "/list_eval_partition.txt",
    "/list_bbox.txt",
    "/list_landmarks.txt",
    "/list_joints.txt",
    "/dbpedia-triples.ttl",
    "/triples-with-problems.ttl",
    "/semantic-clothing.ttl",
    "/semantic-clothing.turtle.owl",
    "/semantic-clothing.functional.owl",
    "/semantic-clothing.latex.owl",
    "/semantic-clothing.manchester.owl",
)

GENERIC_STOP_TERMS = {"", "n/a", "na", "none", "null", "unknown", "other"}

SEED_REJECT_TERMS = {
    "no dress",
    "no neckline",
    "no opening",
    "no special manufacturing technique",
    "no non textile material",
    "no waistline",
    "no fastening",
    "no closure",
    "conventional",
    "regular fit",
    "normal waist",
    "regular collar",
}

STYLE_WHITELIST = {
    "boho chic",
    "bohemian style",
    "business casual",
    "casual",
    "clubwear",
    "couture",
    "daywear",
    "dress clothes",
    "empire silhouette",
    "eveningwear",
    "formal",
    "gothic fashion",
    "ivy league",
    "military",
    "preppy",
    "utility",
    "wilderness chic",
}

STYLE_BANLIST = {
    "anti fashion",
    "bogan",
    "chav",
    "cycle chic",
    "flogger",
    "heroin chic",
    "nazi chic",
    "pokemon subculture",
    "pokémon subculture",
    "size zero",
    "sloane ranger",
    "style tribe",
    "swenkas",
    "thrift store chic",
    "young fogey",
}

COLORS_VOCAB = {
    "amber",
    "beige",
    "black",
    "blue",
    "bronze",
    "brown",
    "burgundy",
    "camel",
    "champagne",
    "charcoal",
    "copper",
    "coral",
    "cream",
    "crimson",
    "cyan",
    "ecru",
    "gold",
    "gray",
    "green",
    "grey",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lilac",
    "magenta",
    "maroon",
    "mauve",
    "navy",
    "ocher",
    "ochre",
    "olive",
    "orange",
    "peach",
    "pink",
    "purple",
    "red",
    "rose",
    "rosso corsa",
    "rust",
    "saffron",
    "silver",
    "sienna",
    "taupe",
    "teal",
    "turquoise",
    "ultramarine",
    "violet",
    "white",
    "wine",
    "yellow",
}

CORE_COLOR_ALLOWLIST = {
    "beige",
    "black",
    "blue",
    "bronze",
    "brown",
    "camel",
    "champagne",
    "charcoal",
    "coral",
    "cream",
    "ecru",
    "gold",
    "gray",
    "green",
    "grey",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lilac",
    "magenta",
    "maroon",
    "mauve",
    "navy",
    "ochre",
    "olive",
    "orange",
    "pink",
    "purple",
    "red",
    "rose",
    "rust",
    "saffron",
    "silver",
    "taupe",
    "teal",
    "turquoise",
    "violet",
    "white",
    "wine",
    "yellow",
}


@dataclass
class OutfitBundle:
    outfit_id: str
    identity: Dict[str, str]
    bundle_text: str
    field_texts: Dict[str, List[str]]
    source_paths: List[str]
    asset_count: int


@dataclass
class SeedTermInfo:
    slot: str
    term: str
    sources: set[str] = field(default_factory=set)
    source_groups: set[str] = field(default_factory=set)
    base_score: int = 0


def maybe_fix_mojibake(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text).replace("\u00a0", " ")
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        for encoding in ("latin1", "cp1252"):
            try:
                candidate = text.encode(encoding).decode("utf-8")
            except Exception:
                continue
            old_hits = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
            new_hits = sum(candidate.count(marker) for marker in MOJIBAKE_MARKERS)
            if new_hits < old_hits:
                text = candidate
                break
    return text


def normalize_whitespace(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text or None


def simplify_for_compare(text: Optional[str]) -> str:
    if not text:
        return ""
    text = maybe_fix_mojibake(text) or ""
    text = text.lower().replace("_", " ").replace("-", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text) or ""


def slugify(text: Optional[str]) -> str:
    normalized = simplify_for_compare(text)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "na"


def clean_archive_field(field_name: str, value: Optional[str]) -> Optional[str]:
    text = normalize_whitespace(maybe_fix_mojibake(value))
    if not text or text.lower() in GENERIC_STOP_TERMS:
        return None

    if field_name == "materials":
        text = re.sub(r"\([^)]*\)", " ", text)

    text = re.sub(r"\bart\.?\s*[\w./-]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdes\.?\s*[\w./-]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcol\.?\s*[\w./-]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvar\.?\s*[\w./-]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,3}\s*%\s*[a-zà-ÿ]*\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(?:[.,]\d+)?\s*cm\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcm\s*\d+(?:[.,]\d+)?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return normalize_whitespace(text)


def unique_texts(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_whitespace(value)
        if not cleaned:
            continue
        key = simplify_for_compare(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rstrip(" ,;:-")
    return f"{clipped}..."


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_archive_records(path: Path) -> Dict[str, Dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}, found {type(data).__name__}.")
    return {str(source_path): record for source_path, record in data.items() if isinstance(record, dict)}


def build_outfit_groups(records: Dict[str, Dict[str, Any]]) -> List[OutfitBundle]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

    for source_path, record in records.items():
        key = tuple(
            normalize_whitespace(maybe_fix_mojibake(record.get(field))) or ""
            for field in ("year", "season", "collection", "file", "look")
        )
        if not any(key):
            field_fallback = "|".join(simplify_for_compare(record.get(field)) for field in BUNDLE_FIELDS)
            key = ("fallback", hashlib.sha1(field_fallback.encode("utf-8")).hexdigest(), "", "", "")
        grouped[key].append((source_path, record))

    outfits: List[OutfitBundle] = []
    for key, items in sorted(grouped.items()):
        year, season, collection, file_id, look = key
        identity = {
            "year": year,
            "season": season,
            "collection": collection,
            "file": file_id,
            "look": look,
        }
        collected_fields: Dict[str, List[str]] = {field: [] for field in BUNDLE_FIELDS}
        source_paths: List[str] = []

        for source_path, record in items:
            source_paths.append(source_path)
            for field_name in BUNDLE_FIELDS:
                cleaned = clean_archive_field(field_name, record.get(field_name))
                if cleaned:
                    collected_fields[field_name].append(cleaned)

        field_texts: Dict[str, List[str]] = {}
        sections: List[str] = []
        for field_name in BUNDLE_FIELDS:
            unique_values = unique_texts(collected_fields[field_name])
            if not unique_values:
                continue
            field_texts[field_name] = unique_values
            joined = " || ".join(truncate(value, 900) for value in unique_values)
            sections.append(f"{field_name}: {joined}")

        outfit_id = "__".join(
            [
                slugify(year),
                slugify(season),
                slugify(collection),
                slugify(file_id),
                slugify(look),
            ]
        )
        outfits.append(
            OutfitBundle(
                outfit_id=outfit_id,
                identity=identity,
                bundle_text="\n".join(sections),
                field_texts=field_texts,
                source_paths=source_paths,
                asset_count=len(items),
            )
        )

    return outfits


def build_archive_response_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "propertyOrdering": SLOT_NAMES,
        "required": SLOT_NAMES,
        "properties": {
            slot: {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            }
            for slot in SLOT_NAMES
        },
    }


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_archive_prompt(outfits: Sequence[OutfitBundle]) -> str:
    instructions = {
        "task": (
            "Build a compact Ferre-specific retrieval ontology from deduplicated archive outfits. "
            "Use only the archive text bundles below."
        ),
        "goal": (
            "Produce compact, controlled retrieval keywords. Precision and consistency matter more "
            "than completeness. This is not a full scholarly description."
        ),
        "rules": [
            "Return only archive-grounded terms or short archive-grounded phrases.",
            "Translate basic colors and direct material constructions into English when the translation is obvious and standard for retrieval.",
            "Preserve fashion-native Italian or French terms when they are the standard fashion word or there is no cleaner direct English replacement.",
            "Include all distinct supported retrieval-useful terms, even if they appear only once in the archive.",
            "Merge near-synonyms, plural-singular variants, and trivial wording variants.",
            "Keep arrays retrieval-useful, but do not drop supported specific material variants just because they are rare.",
            "Ignore supplier names, article codes, percentages, bibliography, acquisition, present location, inventory, and irrelevant prose.",
            "Do not output full sentences, explanations, or unsupported terms.",
            "Use lowercase output.",
            "If evidence is weak for a slot, return an empty array.",
            "Because there is no dedicated collar or lapel slot, collar and lapel descriptors may go into neckline_tags when useful.",
            "Because there is no dedicated waistline slot, useful waistline or fit descriptors may go into silhouette_tags when helpful.",
        ],
        "slot_guidance": {
            "garments_tags": "Garments, outfit components, and accessories explicitly present in the archive text.",
            "colors_tags": "Explicit color names.",
            "material_tags": "Fabric or material names, including textile families and non-textile materials used in garments.",
            "patterns_tags": "Surface patterns and print terms.",
            "silhouette_tags": "Shape, fit, line, waistline, and outline descriptors useful for retrieval.",
            "length_tags": "Garment length or hemline length descriptors.",
            "neckline_tags": "Neckline, collar, or lapel descriptors useful for retrieval.",
            "sleeve_tags": "Sleeve presence, sleeve length, or sleeve construction descriptors.",
            "closure_tags": "Opening or fastening descriptors.",
            "embellishment_tags": "Decorative or construction details useful for retrieval.",
            "style_tags": "Archive-grounded style or use-context descriptors such as couture, daywear, eveningwear, military, utility, and gala.",
        },
    }

    payload = {
        "outfit_count": len(outfits),
        "outfits": [
            {
                "outfit_id": outfit.outfit_id,
                "identity": outfit.identity,
                "field_texts": outfit.field_texts,
                "bundle_text": outfit.bundle_text,
            }
            for outfit in outfits
        ],
    }

    return (
        f"{to_pretty_json(instructions)}\n\n"
        f"Archive outfits:\n{to_pretty_json(payload)}\n"
    )


def build_archive_consolidation_prompt(
    candidates_by_slot: Dict[str, List[Dict[str, Any]]]
) -> str:
    instructions = {
        "task": "Consolidate archive-grounded ontology candidates into a compact final ontology.",
        "rules": [
            "Use only the candidate terms and support metadata provided below.",
            "Merge trivial wording variants and near-synonyms.",
            "Preserve meaningful archive-specific phrasing when it is retrieval-useful.",
            "Keep all distinct supported retrieval-useful terms, including rare but specific archive terms.",
            "Translate basic colors and direct material constructions into English when that translation is obvious and standard for retrieval.",
            "Preserve fashion-native Italian or French terms when they are the standard fashion word.",
            "Return only a JSON object with the exact required slots.",
        ],
    }
    return (
        f"{to_pretty_json(instructions)}\n\n"
        f"Candidate terms:\n{to_pretty_json(candidates_by_slot)}\n"
    )


def parse_model_json_response(response: Any) -> Dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, dict):
            return parsed
        return json.loads(json.dumps(parsed, ensure_ascii=False))

    text = getattr(response, "text", None)
    if not text:
        try:
            text = response.candidates[0].content.parts[0].text
        except Exception as exc:  # pragma: no cover - depends on remote response
            raise ValueError("Gemini response did not contain parsable JSON.") from exc

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in ("429", "500", "503", "resource_exhausted", "deadline", "timeout")
    )


def call_llm_with_retry(
    client: Any,
    model: str,
    prompt: str,
    response_schema: Dict[str, Any],
    key: str,
) -> Dict[str, Any]:
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            parsed = parse_model_json_response(response)
            if not isinstance(parsed, dict):
                raise ValueError(f"Gemini returned a non-object response for {key}.")
            return parsed
        except Exception as exc:  # pragma: no cover - depends on remote response
            if attempt < MAX_RETRIES - 1 and is_retryable_error(exc):
                print(f"[retry] {key}: attempt {attempt + 1} failed, retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue
            raise

    raise RuntimeError(f"Gemini call failed for {key}.")


BASIC_COLOR_PHRASE_TRANSLATIONS = {
    "blu scuro": "dark blue",
    "bleu scuro": "dark blue",
    "grigio scuro": "dark gray",
    "rosso vino": "wine red",
    "rosso bordeaux": "burgundy",
}

BASIC_COLOR_TOKEN_TRANSLATIONS = {
    "nero": "black",
    "nera": "black",
    "neri": "black",
    "nere": "black",
    "black": "black",
    "bianco": "white",
    "bianca": "white",
    "bianchi": "white",
    "bianche": "white",
    "white": "white",
    "blu": "blue",
    "bleu": "blue",
    "blue": "blue",
    "azzurro": "blue",
    "azzurra": "blue",
    "rosso": "red",
    "rossa": "red",
    "rossi": "red",
    "rosse": "red",
    "red": "red",
    "verde": "green",
    "verdi": "green",
    "green": "green",
    "giallo": "yellow",
    "gialla": "yellow",
    "yellow": "yellow",
    "marrone": "brown",
    "marroni": "brown",
    "brown": "brown",
    "grigio": "gray",
    "grigia": "gray",
    "grigi": "gray",
    "grigie": "gray",
    "gray": "gray",
    "grey": "gray",
    "oro": "gold",
    "dorato": "gold",
    "dorata": "gold",
    "gold": "gold",
    "argento": "silver",
    "silver": "silver",
    "avorio": "ivory",
    "ivory": "ivory",
    "rosa": "pink",
    "pink": "pink",
    "viola": "purple",
    "purple": "purple",
    "beige": "beige",
    "fucsia": "fuchsia",
    "fuchsia": "fuchsia",
    "bordeaux": "burgundy",
    "ocra": "ochre",
    "crema": "cream",
    "paglia": "straw",
}

ARCHIVE_MATERIAL_PATTERN_MAP = {
    "silk faille": [r"\bsilk faille\b", r"\bfaille di seta\b"],
    "silk georgette": [r"\bsilk georgette\b", r"\bgeorgette di seta\b"],
    "silk satin": [r"\bsilk satin\b", r"\braso di seta\b"],
    "silk organza": [r"\bsilk organza\b", r"\borganza di seta\b"],
    "silk chiffon": [r"\bsilk chiffon\b", r"\bchiffon di seta\b"],
    "silk gazar": [r"\bsilk gaza+r\b", r"\bgaza+r di seta\b"],
    "silk cady": [r"\bsilk cady\b", r"\bcady di seta\b"],
    "silk taffeta": [r"\bsilk taffeta\b", r"\btaffeta di seta\b"],
    "silk velvet": [r"\bsilk velvet\b", r"\bvelluto di seta\b"],
    "silk serge": [r"\bsilk serge\b"],
    "silk crepe de chine": [
        r"\bsilk crepe de chine\b",
        r"\bcrepe de chine di seta\b",
        r"\bcr\w*pe de chine di seta\b",
    ],
    "silk mousseline": [r"\bsilk mousseline\b", r"\bmousseline di seta\b", r"\bprint silk mousseline\b"],
    "silk jacquard": [r"\bsilk jacquard\b", r"\bjacquard di seta\b"],
    "silk marocain": [r"\bsilk marocain\b", r"\bmarocain di seta\b"],
    "silk duchesse": [r"\bsilk duchesse\b", r"\bduchesse di seta\b"],
    "tussah silk": [r"\btussah silk\b"],
    "silk tape": [r"\bsilk tape\b", r"\bnastro di seta\b"],
    "silk crepe": [r"\bsilk crepe\b", r"\bprint silk crepe\b"],
    "silk gros grain": [r"\bsilk gros ?grain\b"],
    "silk print fabric": [r"\bsilk print fabric\b"],
    "wool/silk satin": [r"\bwool/silk satin\b", r"\braso di lana e seta\b"],
    "wool/silk blend": [r"\bwool/silk blend\b", r"\bwool/silk blend fabric\b", r"\blana e seta\b"],
    "wool jersey": [r"\bwool jersey\b", r"\bjersey di lana\b"],
    "silk jersey": [r"\bsilk jersey\b", r"\bjersey di seta\b"],
    "silk tulle": [r"\bsilk tulle\b", r"\btulle di seta\b"],
    "silk lace": [r"\bsilk lace\b", r"\bsilk chantilly lace\b"],
    "cotton velvet": [r"\bcotton velvet\b"],
    "cashmere knit": [r"\bcashmere knit\b"],
    "wool satin": [r"\bwool satin\b"],
    "viscose ribbon": [r"\bviscose ribbon\b"],
    "nylon crinoline": [r"\bnylon crinoline\b"],
    "crystal jet": [r"\bcrystal jets?\b", r"\bcrystal jet\b"],
    "lurex": [r"\blurex\b"],
    "mesh": [r"\bmesh\b"],
    "raffia": [r"\braffia\b"],
    "organza": [r"\borganza\b"],
    "faille": [r"\bfaille\b"],
    "gazar": [r"\bgaza+r\b"],
    "georgette": [r"\bgeorgette\b"],
    "taffeta": [r"\btaffeta\b"],
    "velvet": [r"\bvelvet\b", r"\bvelluto\b"],
    "serge": [r"\bserge\b"],
    "crepe de chine": [r"\bcrepe de chine\b", r"\bcr\w*pe de chine\b"],
    "mousseline": [r"\bmousseline\b"],
    "jacquard": [r"\bjacquard\b"],
    "marocain": [r"\bmarocain\b"],
    "duchesse": [r"\bduchesse\b"],
    "jersey": [r"\bjersey\b"],
    "tulle": [r"\btulle\b"],
    "lace": [r"\blace\b", r"\bpizzo\b"],
    "silk": [r"\bsilk\b", r"\bseta\b"],
    "wool": [r"\bwool\b", r"\blana\b"],
    "cotton": [r"\bcotton\b", r"\bcotone\b"],
    "nylon": [r"\bnylon\b"],
    "cashmere": [r"\bcashmere\b", r"\bcachemire\b"],
    "fur": [r"\bfur\b", r"\bpelliccia\b"],
}

ARCHIVE_GARMENT_PATTERN_MAP = {
    "jacket": [r"\bgiacca\b", r"\bjacket\b"],
    "skirt": [r"\bgonna\b", r"\bskirt\b"],
    "t-shirt": [r"\bt-?shirt\b"],
    "dress": [r"\babito\b", r"\bdress\b"],
    "evening dress": [r"\bevening dress\b"],
    "coat": [r"\bcappotto\b", r"\bcoat\b"],
    "belt": [r"\bcintura\b", r"\bbelt\b"],
    "stole": [r"\bstola\b", r"\bstole\b"],
    "shawl": [r"\bshawl\b"],
    "cape": [r"\bcape\b", r"\bcappa\b", r"\bmantella\b"],
    "top": [r"\btop\b"],
    "shirt": [r"\bcamicia\b", r"\bshirt\b"],
    "blouse": [r"\bcamicetta\b", r"\bblouse\b"],
    "bodice": [r"\bcorpetto\b", r"\bbodice\b"],
    "bustier": [r"\bbustier\b"],
    "pants": [r"\bpantaloni\b", r"\bpants\b", r"\btrousers\b"],
    "tunic": [r"\btunica\b", r"\btunic\b"],
    "ballgown": [r"\bballgown\b"],
}

ARCHIVE_PATTERN_PATTERN_MAP = {
    "paisley": [r"\bpaisley\b"],
    "chalkstripe": [r"\bchalkstripe\b"],
    "stripe": [r"\bstripes?\b", r"\brighe?\b"],
    "tartan": [r"\btartan\b"],
    "polkadot": [r"\bpolkadot\b"],
    "zigzag": [r"\bzigzag\b"],
    "herringbone": [r"\bherringbone\b"],
    "birdseye": [r"\bbird'?s[- ]eye\b"],
    "melange": [r"\bmelange\b"],
    "plaid": [r"\bplaid\b"],
    "floral": [r"\bfloral\b"],
    "foulard": [r"\bfoulard\b"],
    "print": [r"\bprint(?:ed)?\b", r"\bstampa(?:ta|to|te|ti)?\b"],
}


def normalize_archive_keyword_common(term: str) -> Optional[str]:
    if not term:
        return None
    cleaned = normalize_whitespace(maybe_fix_mojibake(term))
    if not cleaned:
        return None
    cleaned = cleaned.lower().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^[,;:.()\-\s]+|[,;:.()\-\s]+$", "", cleaned)
    if not cleaned or cleaned in GENERIC_STOP_TERMS:
        return None
    if len(cleaned) > 70:
        return None
    if re.search(r"\b(?:art|col|des|var)\.?\s*[\w./-]+\b", cleaned):
        return None
    if re.search(r"\b\d{1,3}\s*%\b", cleaned):
        return None
    return cleaned


def normalize_archive_color_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    for source, target in BASIC_COLOR_PHRASE_TRANSLATIONS.items():
        simplified = re.sub(rf"\b{re.escape(source)}\b", target, simplified)
    tokens = [
        BASIC_COLOR_TOKEN_TRANSLATIONS.get(token, token)
        for token in simplified.split()
    ]
    cleaned = " ".join(tokens).strip()
    cleaned = re.sub(r"\bcolor\b", "", cleaned).strip()
    cleaned = normalize_whitespace(cleaned)
    if cleaned in {"burnt", "bruciato"}:
        return None
    exact_map = {
        "bluette": "bright blue",
        "bois de rose": "rosewood",
        "carne": "nude",
        "cipria": "powder pink",
    }
    if cleaned in exact_map:
        return exact_map[cleaned]
    return cleaned


def normalize_archive_material_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    normalized = normalize_whitespace(maybe_fix_mojibake(cleaned))
    if not normalized:
        return None
    normalized = normalized.lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    replacements = {
        "gazaar": "gazar",
        "grosgrain": "gros grain",
        "raso": "satin",
        "velluto": "velvet",
        "seta": "silk",
        "lana": "wool",
        "cotone": "cotton",
        "pizzo": "lace",
        "taffeta": "taffeta",
        "cachemire": "cashmere",
        "rafia": "raffia",
        "paillettes": "sequins",
        "piqué": "pique",
    }
    for source, target in replacements.items():
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
    normalized = re.sub(r"\bwool\s*(?:/|e|and)\s*silk\b", "wool/silk", normalized)
    normalized = re.sub(r"\bsilk\s*(?:/|e|and)\s*wool\b", "wool/silk", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    simplified = simplify_for_compare(normalized)
    exact_map = {
        "faille di silk": "silk faille",
        "silk faille": "silk faille",
        "georgette di silk": "silk georgette",
        "silk georgette": "silk georgette",
        "organza di silk": "silk organza",
        "silk organza": "silk organza",
        "chiffon di silk": "silk chiffon",
        "silk chiffon": "silk chiffon",
        "gazar di silk": "silk gazar",
        "silk gazar": "silk gazar",
        "cady di silk": "silk cady",
        "silk cady": "silk cady",
        "organza silk": "silk organza",
        "taffeta di silk": "silk taffeta",
        "silk taffeta": "silk taffeta",
        "velvet di silk": "silk velvet",
        "silk velvet": "silk velvet",
        "silk serge": "silk serge",
        "serge di silk": "silk serge",
        "crepe de chine di silk": "silk crepe de chine",
        "silk crepe de chine": "silk crepe de chine",
        "mousseline di silk": "silk mousseline",
        "silk mousseline": "silk mousseline",
        "jacquard di silk": "silk jacquard",
        "silk jacquard": "silk jacquard",
        "marocain di silk": "silk marocain",
        "silk marocain": "silk marocain",
        "duchesse di silk": "silk duchesse",
        "silk duchesse": "silk duchesse",
        "jersey di wool": "wool jersey",
        "wool jersey": "wool jersey",
        "jersey di silk": "silk jersey",
        "silk jersey": "silk jersey",
        "tulle di silk": "silk tulle",
        "silk tulle": "silk tulle",
        "tussah silk": "tussah silk",
        "silk tape": "silk tape",
        "nastro di silk": "silk tape",
        "silk crepe": "silk crepe",
        "silk gros grain": "silk gros grain",
        "silk print fabric": "silk print fabric",
        "satin di silk": "silk satin",
        "cotton piquet": "cotton pique",
        "piquet di cotton": "cotton pique",
        "wool silk satin": "wool/silk satin",
        "satin di wool silk": "wool/silk satin",
        "wool silk blend": "wool/silk blend",
        "wool silk": "wool/silk blend",
        "wool silk blend fabric": "wool/silk blend",
        "crinolina di nylon": "nylon crinoline",
        "jacquard wool": "wool jacquard",
        "legno": "wood",
        "lurex oro": "gold lurex",
        "pelliccia di volpe": "fox fur",
        "tulle elasticizzato": "elasticized tulle",
        "cannete di silk": "silk cannete",
        "canottiglia": "bugle beads",
        "canottiglie": "bugle beads",
        "filo di viscosa": "viscose yarn",
        "jais": "jet",
        "lace macrame di cotton": "cotton macrame lace",
        "natte di silk": "silk natte",
        "paglia di vienna": "vienna straw",
        "panno di wool": "wool cloth",
        "tobolari di organza": "organza tubing",
        "zibellino": "sable",
    }
    if simplified in exact_map:
        return exact_map[simplified]
    if simplified in {"pique", "piqu"}:
        return "pique"
    return simplified


def normalize_archive_garment_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    simplified = simplified.replace("t shirt", "t-shirt")
    exact_map = {
        "abito": "dress",
        "abito da sera": "evening dress",
        "ball gown": "ballgown",
        "cabane": "caban",
        "camicia": "shirt",
        "canotta": "tank top",
        "cappotto": "coat",
        "completo": "ensemble",
        "completo due pezzi": "two piece outfit",
        "giacca": "jacket",
        "giacchino": "jacket",
        "gonna": "skirt",
        "lana coat": "wool coat",
        "pantaloni": "pants",
        "scialle": "shawl",
        "soprabito": "overcoat",
        "stola": "stole",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_pattern_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    replacements = {
        "stripes": "stripe",
        "striping": "stripe",
        "righe": "stripe",
        "bird s eye": "birdseye",
        "bird eye": "birdseye",
    }
    for source, target in replacements.items():
        simplified = re.sub(rf"\b{re.escape(source)}\b", target, simplified)
    exact_map = {
        "disegno a volute": "scroll pattern",
        "disegno cachemire": "paisley pattern",
        "disegno floreale": "floral pattern",
        "floreale": "floral",
        "ricamo": "embroidered",
        "stampa paisley": "paisley print",
        "volute barocche": "baroque volutes",
    }
    if simplified in exact_map:
        return exact_map[simplified]
    return simplified


def normalize_archive_silhouette_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    exact_map = {
        "aderente": "body hugging",
        "ampia": "ample",
        "ampio": "ample",
        "arricciate": "gathered",
        "corta": "short",
        "effetto drappeggio": "draped effect",
        "fasciante": "body hugging",
        "linea ampia": "ample line",
        "linea arrotondata": "rounded line",
        "linea asciutta": "slim line",
        "linea diritta": "straight line",
        "linea dritta": "straight line",
        "linea morbida": "soft line",
        "svasata": "flared",
        "taglio maschile": "masculine cut",
        "vestibilita asciutta": "slim fit",
        "vita alta": "high waist",
        "voluminose": "voluminous",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_length_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    exact_map = {
        "3 4": "three quarter length",
        "al ginocchio": "knee length",
        "alla caviglia": "ankle length",
        "corta": "short",
        "lunga alla caviglia": "ankle length",
        "lunga a terra": "floor length",
        "lunghe": "long",
        "lunghezza ai fianchi": "hip length",
        "lunghezza al ginocchio": "knee length",
        "primo fianco": "high hip",
        "sopra il ginocchio": "above the knee",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_neckline_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    exact_map = {
        "ampio arrotondato a scialle": "wide rounded shawl lapel",
        "arrotondato a scialle": "shawl lapel",
        "crewneck": "crew neck",
        "girocollo": "crew neck",
        "revers": "lapel",
        "revers a lancia": "peak lapel",
        "revers a uomo": "menswear lapel",
        "scollo a barca": "boat neck",
        "scollo a v": "v neckline",
        "scollo all americana": "halter neck",
        "v neck": "v neckline",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_sleeve_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    exact_map = {
        "3 4 sleeves": "three quarter sleeves",
        "manica lunga a tubo": "long tubular sleeves",
        "maniche arricciate": "gathered sleeves",
        "maniche a kimono": "kimono sleeves",
        "maniche corte": "short sleeves",
        "maniche lunghe": "long sleeves",
        "maniche voluminose": "voluminous sleeves",
        "no sleeves": "sleeveless",
        "senza maniche": "sleeveless",
        "spalle importanti": "pronounced shoulders",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_closure_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    exact_map = {
        "allacciatura a un bottone": "one button closure",
        "automatic button": "snap button",
        "automatico": "snap button",
        "asola": "buttonhole",
        "automatici": "snap buttons",
        "bottone": "button",
        "bottoni automatici ricoperti": "covered snap buttons",
        "bottoni rivestiti": "covered buttons",
        "chiusura con due bottoni rivestiti": "two covered button closure",
        "chiusura con zip sul dorso": "back zip closure",
        "chiusura lampo": "zip closure",
        "chiusura posteriore": "back closure",
        "gancio": "hook",
        "gancio in metallo": "metal hook closure",
        "lateral zip": "side zip closure",
        "monopetto": "single breasted",
        "senza chiusure": "no closure",
        "zip laterale": "side zip closure",
        "zipper": "zip closure",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_embellishment_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    exact_map = {
        "applicazione": "applique",
        "applicazioni": "appliques",
        "applicazioni all over di fiori ricamati": "all over embroidered floral appliques",
        "balza": "flounce",
        "balza di tulle plissettato": "pleated tulle flounce",
        "canottiglia": "bugle beads",
        "canottiglie": "bugle beads",
        "ciuffi": "tassels",
        "drappeggio": "draping",
        "fiori": "flowers",
        "fiori ricamati": "embroidered flowers",
        "fiocchi": "bows",
        "fori": "cutouts",
        "imbottito": "padded",
        "impunturati": "topstitched",
        "impunturato": "topstitched",
        "impunture": "topstitching",
        "impunture orizzontali": "horizontal topstitching",
        "jais": "jet",
        "motivi floreali": "floral motifs",
        "motivo a pieghe": "pleated motif",
        "nastri": "ribbons",
        "orlo festonato": "scalloped hem",
        "orlo irregolare a smerli": "irregular scalloped hem",
        "paillette": "sequin",
        "paillettes": "sequins",
        "perline": "beads",
        "pieghe": "pleats",
        "plissettatura": "pleating",
        "punto catenella": "chain stitch",
        "ramoscelli": "sprigs",
        "ricoperto": "covered",
        "ricamato": "embroidered",
        "ricamo": "embroidery",
        "ricamo a mano": "hand embroidery",
        "righe": "stripes",
        "tasche rettangolari applicate": "rectangular patch pockets",
        "tassles": "tassels",
        "traforature": "openwork",
        "trapuntatura": "quilted",
        "volute": "scrollwork",
        "volute barocche": "baroque volutes",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_style_keyword(term: str) -> Optional[str]:
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    exact_map = {
        "alta moda": "couture",
        "taglio maschile": "mannish",
    }
    return exact_map.get(simplified, simplified)


def normalize_archive_keyword(slot: str, term: str) -> Optional[str]:
    if slot == "garments_tags":
        return normalize_archive_garment_keyword(term)
    if slot == "colors_tags":
        return normalize_archive_color_keyword(term)
    if slot == "material_tags":
        return normalize_archive_material_keyword(term)
    if slot == "patterns_tags":
        return normalize_archive_pattern_keyword(term)
    if slot == "silhouette_tags":
        return normalize_archive_silhouette_keyword(term)
    if slot == "length_tags":
        return normalize_archive_length_keyword(term)
    if slot == "neckline_tags":
        return normalize_archive_neckline_keyword(term)
    if slot == "sleeve_tags":
        return normalize_archive_sleeve_keyword(term)
    if slot == "closure_tags":
        return normalize_archive_closure_keyword(term)
    if slot == "embellishment_tags":
        return normalize_archive_embellishment_keyword(term)
    if slot == "style_tags":
        return normalize_archive_style_keyword(term)
    cleaned = normalize_archive_keyword_common(term)
    if not cleaned:
        return None
    simplified = simplify_for_compare(cleaned)
    simplified = simplified.replace("t shirt", "t-shirt")
    return simplified


def empty_keyword_map() -> Dict[str, List[str]]:
    return {slot: [] for slot in SLOT_NAMES}


def merge_archive_keyword_dicts(
    *keyword_dicts: Dict[str, List[str]],
    outfits: Sequence[OutfitBundle],
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for slot in SLOT_NAMES:
        values: List[str] = []
        seen: set[str] = set()
        for keyword_dict in keyword_dicts:
            for value in keyword_dict.get(slot, []):
                normalized = normalize_archive_keyword(slot, str(value))
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                values.append(normalized)
        merged[slot] = sorted(values, key=lambda term: (-archive_support_count(term, outfits), term))
    return merged


def collect_outfit_texts(outfits: Sequence[OutfitBundle], fields: Optional[Sequence[str]] = None) -> List[str]:
    texts: List[str] = []
    for outfit in outfits:
        if fields is None:
            iterables = outfit.field_texts.values()
        else:
            iterables = [outfit.field_texts.get(field, []) for field in fields]
        for values in iterables:
            texts.extend(values)
    return texts


def harvest_terms_from_pattern_map(texts: Sequence[str], pattern_map: Dict[str, Sequence[str]]) -> List[str]:
    found: set[str] = set()
    simplified_texts = [simplify_for_compare(text) for text in texts if text]
    for canonical, patterns in pattern_map.items():
        for text in simplified_texts:
            if any(re.search(pattern, text) for pattern in patterns):
                found.add(canonical)
                break
    return sorted(found)


def harvest_archive_colors(texts: Sequence[str]) -> List[str]:
    found: set[str] = set()
    simplified_texts = [simplify_for_compare(text) for text in texts if text]
    for text in simplified_texts:
        for source, target in BASIC_COLOR_PHRASE_TRANSLATIONS.items():
            if re.search(rf"\b{re.escape(source)}\b", text):
                found.add(target)
        for source, target in BASIC_COLOR_TOKEN_TRANSLATIONS.items():
            if re.search(rf"\b{re.escape(source)}\b", text):
                found.add(target)
    return sorted(found)


def harvest_archive_keywords_deterministically(outfits: Sequence[OutfitBundle]) -> Dict[str, List[str]]:
    all_texts = collect_outfit_texts(outfits)
    material_texts = collect_outfit_texts(outfits, fields=["materials", "description", "working_process"])
    garment_texts = collect_outfit_texts(outfits, fields=["object", "description", "label"])

    harvested: Dict[str, List[str]] = {slot: [] for slot in SLOT_NAMES}
    harvested["colors_tags"] = harvest_archive_colors(all_texts)
    harvested["material_tags"] = harvest_terms_from_pattern_map(material_texts, ARCHIVE_MATERIAL_PATTERN_MAP)
    harvested["garments_tags"] = harvest_terms_from_pattern_map(garment_texts, ARCHIVE_GARMENT_PATTERN_MAP)
    harvested["patterns_tags"] = harvest_terms_from_pattern_map(all_texts, ARCHIVE_PATTERN_PATTERN_MAP)

    return merge_archive_keyword_dicts(harvested, outfits=outfits)


def archive_support_count(term: str, outfits: Sequence[OutfitBundle]) -> int:
    simplified_term = simplify_for_compare(term)
    if not simplified_term:
        return 0
    compact_term = simplified_term.replace(" ", "")
    phrase_pattern = re.compile(rf"\b{re.escape(simplified_term)}\b")
    compact_pattern = re.compile(rf"\b{re.escape(compact_term)}\b")
    count = 0
    for outfit in outfits:
        haystack = simplify_for_compare(outfit.bundle_text)
        compact_haystack = haystack.replace(" ", "")
        if phrase_pattern.search(haystack) or compact_pattern.search(compact_haystack):
            count += 1
    return count


def sample_supporting_outfits(
    term: str,
    outfits: Sequence[OutfitBundle],
    max_items: int = 3,
) -> List[str]:
    simplified_term = simplify_for_compare(term)
    if not simplified_term:
        return []
    compact_term = simplified_term.replace(" ", "")
    phrase_pattern = re.compile(rf"\b{re.escape(simplified_term)}\b")
    compact_pattern = re.compile(rf"\b{re.escape(compact_term)}\b")
    matches: List[str] = []
    for outfit in outfits:
        haystack = simplify_for_compare(outfit.bundle_text)
        compact_haystack = haystack.replace(" ", "")
        if phrase_pattern.search(haystack) or compact_pattern.search(compact_haystack):
            matches.append(outfit.outfit_id)
        if len(matches) >= max_items:
            break
    return matches


def finalize_archive_keywords(
    raw_result: Dict[str, Any],
    outfits: Sequence[OutfitBundle],
) -> Dict[str, List[str]]:
    return merge_archive_keyword_dicts(raw_result, outfits=outfits)


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def consolidate_archive_candidates(
    client: Any,
    model: str,
    batch_results: Sequence[Dict[str, List[str]]],
    outfits: Sequence[OutfitBundle],
) -> Dict[str, List[str]]:
    candidates_by_slot: Dict[str, List[Dict[str, Any]]] = {slot: [] for slot in SLOT_NAMES}

    for slot in SLOT_NAMES:
        candidate_terms = unique_texts(term for batch in batch_results for term in batch.get(slot, []))
        ranked_candidates = sorted(
            candidate_terms,
            key=lambda term: (-archive_support_count(term, outfits), term),
        )[:48]

        for term in ranked_candidates:
            candidates_by_slot[slot].append(
                {
                    "term": term,
                    "support_count": archive_support_count(term, outfits),
                    "sample_outfit_ids": sample_supporting_outfits(term, outfits),
                }
            )

    prompt = build_archive_consolidation_prompt(candidates_by_slot)
    raw = call_llm_with_retry(
        client=client,
        model=model,
        prompt=prompt,
        response_schema=build_archive_response_schema(),
        key="archive_consolidation",
    )
    return finalize_archive_keywords(raw, outfits)


def discover_default_project_id() -> Optional[str]:
    if os.environ.get("GCP_PROJECT"):
        return os.environ["GCP_PROJECT"]

    docker_shell_path = VECTOR_DB_DIR / "docker-shell.sh"
    if docker_shell_path.exists():
        content = docker_shell_path.read_text(encoding="utf-8")
        match = re.search(r'export\s+GCP_PROJECT="([^"]+)"', content)
        if match:
            return match.group(1)
    return None


def discover_default_credentials_path() -> Optional[Path]:
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        REPO_ROOT.parent / "secrets" / "llm-service-account.json",
        REPO_ROOT / "secrets" / "llm-service-account.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_archive_keywords_with_llm(
    outfits: Sequence[OutfitBundle],
    model: str,
    location: str,
    single_pass_char_limit: int,
    batch_size: int,
) -> Dict[str, List[str]]:
    harvested_keywords = harvest_archive_keywords_deterministically(outfits)

    if genai is None or types is None:
        raise RuntimeError(
            "google-genai is not installed. Install repo dependencies or add google-genai to your runtime."
        )

    project_id = discover_default_project_id()
    if not project_id:
        raise RuntimeError("Could not resolve GCP_PROJECT from the environment or docker-shell.sh.")

    credentials_path = discover_default_credentials_path()
    if not credentials_path:
        raise RuntimeError(
            "Could not resolve GOOGLE_APPLICATION_CREDENTIALS. Expected a sibling secrets/llm-service-account.json."
        )

    os.environ.setdefault("GCP_PROJECT", project_id)
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_path))

    client = genai.Client(vertexai=True, project=project_id, location=location)
    prompt = build_archive_prompt(outfits)

    if len(prompt) <= single_pass_char_limit:
        raw = call_llm_with_retry(
            client=client,
            model=model,
            prompt=prompt,
            response_schema=build_archive_response_schema(),
            key="archive_single_pass",
        )
        return merge_archive_keyword_dicts(
            harvested_keywords,
            finalize_archive_keywords(raw, outfits),
            outfits=outfits,
        )

    batch_results: List[Dict[str, List[str]]] = []
    for batch_index, batch in enumerate(chunked(outfits, batch_size), start=1):
        batch_prompt = build_archive_prompt(batch)
        raw_batch = call_llm_with_retry(
            client=client,
            model=model,
            prompt=batch_prompt,
            response_schema=build_archive_response_schema(),
            key=f"archive_batch_{batch_index}",
        )
        batch_results.append(finalize_archive_keywords(raw_batch, batch))

    return merge_archive_keyword_dicts(
        harvested_keywords,
        *batch_results,
        outfits=outfits,
    )


GARMENT_HINTS = (
    "anorak",
    "bag",
    "belt",
    "blazer",
    "blouse",
    "bodice",
    "bolero",
    "bustier",
    "cape",
    "cardigan",
    "coat",
    "coverup",
    "dress",
    "duster",
    "gown",
    "glove",
    "gloves",
    "hat",
    "hoodie",
    "jacket",
    "jeans",
    "jumpsuit",
    "kaftan",
    "kimono",
    "leggings",
    "nightdress",
    "nightgown",
    "onesie",
    "pants",
    "parka",
    "poncho",
    "raincoat",
    "robe",
    "romper",
    "scarf",
    "shawl",
    "shirt",
    "shoe",
    "shorts",
    "skirt",
    "stole",
    "suit",
    "sundress",
    "sweater",
    "sweatshirt",
    "tee",
    "t-shirt",
    "tie",
    "top",
    "trousers",
    "tunic",
    "vest",
)

MATERIAL_HINTS = (
    "acrylic",
    "bead",
    "bone",
    "cashmere",
    "chiffon",
    "cloth",
    "cotton",
    "crystal",
    "denim",
    "duchesse",
    "fabric",
    "faille",
    "feather",
    "fur",
    "gazar",
    "gem",
    "jersey",
    "knit",
    "lace",
    "leather",
    "linen",
    "mohair",
    "metal",
    "nylon",
    "organza",
    "paillette",
    "polyester",
    "rubber",
    "satin",
    "silk",
    "spandex",
    "suede",
    "tulle",
    "velvet",
    "wool",
    "wood",
)

PATTERN_HINTS = (
    "abstract",
    "animal",
    "argyle",
    "camouflage",
    "camo",
    "check",
    "cheetah",
    "chevron",
    "dot",
    "fair isle",
    "floral",
    "geometric",
    "giraffe",
    "graphic",
    "herringbone",
    "houndstooth",
    "lattice",
    "leopard",
    "letters",
    "numbers",
    "paisley",
    "pattern",
    "plain",
    "print",
    "printed",
    "solid",
    "snake",
    "snakeskin",
    "stripe",
    "striped",
    "toile de jouy",
    "zebra",
)

SILHOUETTE_HINTS = (
    "a-line",
    "asymmetrical",
    "baggy",
    "balloon",
    "bell",
    "bell bottom",
    "bootcut",
    "boxy",
    "circle",
    "circular",
    "curved fit",
    "dropped waist",
    "empire waist",
    "fit and flare",
    "flare",
    "high low",
    "high waist",
    "loose fit",
    "low waist",
    "mermaid",
    "oversized",
    "peplum",
    "pencil",
    "peg",
    "regular fit",
    "silhouette",
    "slim",
    "straight",
    "tent",
    "tight fit",
    "trumpet",
    "tubular",
    "voluminous",
    "wide leg",
)

NECKLINE_HINTS = (
    "asymmetric neckline",
    "banded collar",
    "boat neck",
    "bow collar",
    "chelsea collar",
    "choker neck",
    "collarless",
    "crew neck",
    "crossover neck",
    "cowl neck",
    "high neck",
    "illusion neck",
    "keyhole neck",
    "lapel",
    "mandarin collar",
    "neckline",
    "notched lapel",
    "off the shoulder",
    "one shoulder",
    "oval neck",
    "peak lapel",
    "peter pan collar",
    "plunging neckline",
    "polo collar",
    "queen anne",
    "round neck",
    "sailor collar",
    "scoop neck",
    "shawl lapel",
    "shirt collar",
    "square neckline",
    "stand away collar",
    "straight across",
    "strapless",
    "sweetheart",
    "surplice neck",
    "turtle neck",
    "u-neck",
    "v-neck",
)

SLEEVE_HINTS = (
    "batwing sleeve",
    "bell sleeve",
    "bishop sleeve",
    "cap sleeve",
    "circular flounce sleeve",
    "dolman sleeve",
    "dropped shoulder sleeve",
    "elbow length",
    "kimono sleeve",
    "leg of mutton sleeve",
    "long sleeve",
    "poet sleeve",
    "puff sleeve",
    "raglan sleeve",
    "set in sleeve",
    "short sleeve",
    "sleeveless",
    "three quarter",
    "tulip sleeve",
    "wrist length",
)

CLOSURE_HINTS = (
    "back zip",
    "buckle",
    "buckled",
    "button",
    "double breasted",
    "fly opening",
    "lace up",
    "no opening",
    "single breasted",
    "toggle",
    "toggled",
    "wrapping",
    "zip",
    "zip up",
    "zipper",
)

EMBELLISHMENT_HINTS = (
    "applique",
    "bead",
    "beading",
    "bow",
    "cutout",
    "embossed",
    "embroidery",
    "flower",
    "frayed",
    "fringe",
    "gathering",
    "intarsia",
    "jet",
    "lining",
    "paillette",
    "patch",
    "perforated",
    "pleat",
    "pleated",
    "quilted",
    "ribbon",
    "rivet",
    "ruffle",
    "ruched",
    "sequin",
    "slit",
    "smocking",
    "sunburst",
    "tassel",
    "tiered",
    "topstitch",
)

STYLE_HINTS = (
    "bohemian",
    "boho",
    "business casual",
    "casual",
    "chic",
    "clubwear",
    "couture",
    "daywear",
    "dress clothes",
    "eveningwear",
    "formal",
    "gala",
    "gothic",
    "grunge",
    "ivy league",
    "military",
    "preppy",
    "utility",
    "wilderness",
)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_seed_path(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SEED_SUFFIXES:
        return 0

    normalized_path = str(path).replace("\\", "/").lower()
    if any(token in normalized_path for token in SEED_PATH_BLACKLIST):
        return 0

    basename = path.name.lower()
    score = SEED_PRIORITY_NAMES.get(basename, 0)
    if score <= 0:
        return 0

    if basename in {"list_attr_cloth.txt", "list_category_cloth.txt"} and "/anno_fine/" not in normalized_path:
        return 0

    if "fashionpedia" in normalized_path:
        score += 20
    if "deepfashion" in normalized_path:
        score += 15
    if "semcloth" in normalized_path:
        score += 10
    if "ontology" in normalized_path and suffix in {".owl", ".xml", ".rdf"}:
        score += 8
    return score


def select_seed_files(seed_root: Path) -> List[Path]:
    best_by_hash: Dict[str, Tuple[int, Path]] = {}
    for path in seed_root.rglob("*"):
        if not path.is_file():
            continue
        score = score_seed_path(path)
        if score <= 0:
            continue
        digest = sha1_file(path)
        current = best_by_hash.get(digest)
        if current is None or score > current[0] or (
            score == current[0] and len(str(path)) < len(str(current[1]))
        ):
            best_by_hash[digest] = (score, path)

    selected = [item[1] for item in best_by_hash.values()]
    selected.sort(key=lambda path: (-score_seed_path(path), str(path).lower()))
    return selected


def split_outside_parens(text: str, delimiter: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == delimiter and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def normalize_seed_piece(piece: str) -> str:
    piece = normalize_whitespace(maybe_fix_mojibake(piece)) or ""
    piece = piece.lower().replace("_", " ")
    piece = piece.replace("–", "-").replace("—", "-")
    piece = piece.replace("’", "'").replace("“", '"').replace("”", '"')
    piece = re.sub(r"\s+", " ", piece)
    return piece.strip(" ,;:.")


def expand_seed_term(raw_term: str, slot: str) -> List[str]:
    text = normalize_seed_piece(raw_term)
    if not text:
        return []

    pieces = split_outside_parens(text, ",")
    expanded: List[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        match = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", piece)
        if match:
            core = normalize_seed_piece(match.group(1))
            context = normalize_seed_piece(match.group(2))
            if context == "a":
                piece = core
            elif context in {"pattern", "length", "neck", "neckline", "collar", "lapel", "sleeve", "skirt", "dress", "coat", "jacket", "pants", "shorts", "opening"}:
                if slot in {"length_tags", "patterns_tags"} and context in {"length", "pattern"}:
                    piece = core
                elif slot == "neckline_tags" and context == "neck":
                    piece = f"{core} neck"
                elif slot == "closure_tags" and context == "opening":
                    piece = core
                else:
                    piece = f"{core} {context}"
            else:
                piece = core

        piece = piece.replace("/", " ")
        piece = piece.replace("t shirt", "t-shirt")
        piece = piece.replace("tee shirt", "t-shirt")
        piece = piece.replace("zip up", "zip")
        piece = piece.replace("zipper", "zip")
        piece = piece.replace("off-the-shoulder", "off the shoulder")
        piece = normalize_seed_piece(piece)
        if piece:
            expanded.append(piece)

    return expanded


def reject_seed_term(term: str, slot: str) -> bool:
    if not term or term in GENERIC_STOP_TERMS or term in SEED_REJECT_TERMS:
        return True
    if len(term) > 60:
        return True
    if re.fullmatch(r"[0-9 .%-]+", term):
        return True
    if term.startswith("no "):
        return True
    if slot == "style_tags" and term.replace("-", " ") in STYLE_BANLIST:
        return True
    if slot == "garments_tags" and term in {"collar", "lapel", "neckline", "pocket", "sleeve", "hood"}:
        return True
    return False


def add_seed_term(
    store: Dict[str, Dict[str, SeedTermInfo]],
    slot: str,
    raw_term: str,
    source_path: Path,
    source_group: str,
    base_score: int,
) -> None:
    for term in expand_seed_term(raw_term, slot):
        if reject_seed_term(term, slot):
            continue
        info = store[slot].get(term)
        if info is None:
            info = SeedTermInfo(slot=slot, term=term)
            store[slot][term] = info
        info.sources.add(str(source_path))
        info.source_groups.add(source_group)
        info.base_score = max(info.base_score, base_score)


def parse_semcloth_bindings(path: Path, slot: str, base_score: int, store: Dict[str, Dict[str, SeedTermInfo]]) -> None:
    data = read_json(path)
    bindings = data.get("results", {}).get("bindings", []) if isinstance(data, dict) else []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        raw_term = None
        if "label" in binding and isinstance(binding["label"], dict):
            raw_term = binding["label"].get("value")
        if not raw_term:
            for value in binding.values():
                if isinstance(value, dict):
                    iri = value.get("value")
                    if iri:
                        raw_term = iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                        break
        if raw_term:
            add_seed_term(store, slot, str(raw_term), path, path.stem.lower(), base_score)


def parse_deepfashion_category_list(path: Path, store: Dict[str, Dict[str, SeedTermInfo]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[2:]:
        line = normalize_whitespace(line)
        if not line:
            continue
        match = re.match(r"^(.*?)\s+(\d+)$", line)
        if match:
            add_seed_term(store, "garments_tags", match.group(1), path, "deepfashion_categories", 7)


def parse_deepfashion_attribute_list(path: Path, store: Dict[str, Dict[str, SeedTermInfo]]) -> None:
    type_to_slot = {
        "1": "patterns_tags",
        "2": "sleeve_tags",
        "3": "length_tags",
        "4": "neckline_tags",
        "5": "material_tags",
        "6": "silhouette_tags",
    }
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[2:]:
        line = normalize_whitespace(line)
        if not line:
            continue
        match = re.match(r"^(.*?)\s+(\d+)$", line)
        if match and match.group(2) in type_to_slot:
            add_seed_term(
                store,
                type_to_slot[match.group(2)],
                match.group(1),
                path,
                "deepfashion_attributes",
                7,
            )


def map_fashionpedia_attribute(slot_name: str, raw_term: str) -> Optional[str]:
    slot_name = normalize_seed_piece(slot_name)
    raw_term = normalize_seed_piece(raw_term)

    if slot_name in {"textile pattern", "animal"}:
        return "patterns_tags"
    if slot_name in {"silhouette", "waistline"}:
        return "silhouette_tags"
    if slot_name == "length":
        if any(token in raw_term for token in ("sleeve", "sleeveless", "elbow", "wrist", "three quarter")):
            return "sleeve_tags"
        return "length_tags"
    if slot_name == "neckline type":
        return "neckline_tags"
    if slot_name in {"opening type", "closures"}:
        return "closure_tags"
    if slot_name in {"textile finishing, manufacturing techniques", "decorations"}:
        return "embellishment_tags"
    if slot_name in {"non-textile material type", "leather"}:
        return "material_tags"
    if slot_name == "nickname":
        if any(token in raw_term for token in ("dress", "coat", "jacket", "pants", "shorts", "skirt", "top", "shirt", "tee", "t-shirt", "hoodie", "blazer", "anorak", "bolero", "kimono", "robe", "gown", "parka")):
            return "garments_tags"
        if any(token in raw_term for token in ("neck", "neckline", "collar", "lapel", "off the shoulder", "one shoulder", "collarless")):
            return "neckline_tags"
        if "sleeve" in raw_term:
            return "sleeve_tags"
        if "opening" in raw_term:
            return "closure_tags"
    return None


def parse_fashionpedia(path: Path, store: Dict[str, Dict[str, SeedTermInfo]]) -> None:
    data = read_json(path)
    categories = data.get("categories", []) if isinstance(data, dict) else []
    attributes = data.get("attributes", []) if isinstance(data, dict) else []

    for category in categories:
        if not isinstance(category, dict):
            continue
        raw_name = category.get("name")
        supercategory = normalize_seed_piece(category.get("supercategory"))
        if raw_name and supercategory not in {"garment parts", "closures", "decorations"}:
            add_seed_term(store, "garments_tags", str(raw_name), path, "fashionpedia_categories", 9)

        if supercategory == "closures":
            add_seed_term(store, "closure_tags", str(raw_name), path, "fashionpedia_categories", 6)
        elif supercategory == "decorations":
            add_seed_term(store, "embellishment_tags", str(raw_name), path, "fashionpedia_categories", 6)

    for attribute in attributes:
        if not isinstance(attribute, dict):
            continue
        raw_name = attribute.get("name")
        raw_supercategory = attribute.get("supercategory")
        if not raw_name or not raw_supercategory:
            continue
        slot = map_fashionpedia_attribute(str(raw_supercategory), str(raw_name))
        if slot:
            add_seed_term(store, slot, str(raw_name), path, f"fashionpedia_{slot}", 8)


def infer_slot_from_term(term: str) -> Optional[str]:
    simplified = normalize_seed_piece(term)
    if not simplified:
        return None

    if simplified in COLORS_VOCAB or simplified.endswith(" blue") or simplified.endswith(" green"):
        return "colors_tags"
    if any(hint in simplified for hint in CLOSURE_HINTS):
        return "closure_tags"
    if any(hint in simplified for hint in SLEEVE_HINTS):
        return "sleeve_tags"
    if any(hint in simplified for hint in NECKLINE_HINTS):
        return "neckline_tags"
    if any(hint in simplified for hint in SILHOUETTE_HINTS):
        return "silhouette_tags"
    if any(hint in simplified for hint in PATTERN_HINTS):
        return "patterns_tags"
    if any(hint in simplified for hint in EMBELLISHMENT_HINTS):
        return "embellishment_tags"
    if any(hint in simplified for hint in MATERIAL_HINTS):
        return "material_tags"
    if any(hint in simplified for hint in STYLE_HINTS):
        return "style_tags"
    if any(hint in simplified for hint in GARMENT_HINTS):
        return "garments_tags"
    return None


def parse_generic_json_or_jsonl(path: Path, store: Dict[str, Dict[str, SeedTermInfo]]) -> None:
    items: List[Any]
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        items = [read_json(path)]

    def walk(node: Any) -> Iterable[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in {"name", "label", "term", "value"} and isinstance(value, str):
                    yield value
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    for item in items:
        for raw_term in walk(item):
            slot = infer_slot_from_term(raw_term)
            if slot:
                add_seed_term(store, slot, raw_term, path, path.stem.lower(), 4)


def parse_generic_txt(path: Path, store: Dict[str, Dict[str, SeedTermInfo]]) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = normalize_whitespace(maybe_fix_mojibake(line))
        if not cleaned or cleaned.isdigit():
            continue
        slot = infer_slot_from_term(cleaned)
        if slot:
            add_seed_term(store, slot, cleaned, path, path.stem.lower(), 3)


def parse_generic_owl_or_xml(path: Path, store: Dict[str, Dict[str, SeedTermInfo]]) -> None:
    text = path.read_text(encoding="utf-8")
    terms: List[str] = []

    try:
        root = ET.fromstring(text)
        for elem in root.iter():
            tag = elem.tag.rsplit("}", 1)[-1].lower()
            if tag == "literal" and elem.text:
                terms.append(elem.text)
            for attr_value in elem.attrib.values():
                if "#" in attr_value or ":" in attr_value:
                    local = attr_value.rsplit("#", 1)[-1].rsplit("/", 1)[-1].rsplit(":", 1)[-1]
                    terms.append(local)
    except ET.ParseError:
        terms.extend(re.findall(r"<Literal[^>]*>([^<]+)</Literal>", text))
        terms.extend(re.findall(r'(?:IRI|abbreviatedIRI)="([^"]+)"', text))

    for raw_term in terms:
        slot = infer_slot_from_term(raw_term)
        if slot:
            add_seed_term(store, slot, raw_term, path, path.stem.lower(), 2)


def extract_external_seed_candidates(seed_files: Sequence[Path]) -> Dict[str, Dict[str, SeedTermInfo]]:
    store: Dict[str, Dict[str, SeedTermInfo]] = {slot: {} for slot in SLOT_NAMES}

    for path in seed_files:
        basename = path.name.lower()
        if basename == "category_attributes_descriptions.json":
            parse_fashionpedia(path, store)
        elif basename == "list_category_cloth.txt":
            parse_deepfashion_category_list(path, store)
        elif basename == "list_attr_cloth.txt":
            parse_deepfashion_attribute_list(path, store)
        elif basename == "clothingtypes.json":
            parse_semcloth_bindings(path, "garments_tags", 7, store)
        elif basename in {"clothingmaterials.json", "clothing-materials.json"}:
            parse_semcloth_bindings(path, "material_tags", 7, store)
        elif basename == "colors.json":
            parse_semcloth_bindings(path, "colors_tags", 6, store)
        elif basename == "clothingstyles.json":
            parse_semcloth_bindings(path, "style_tags", 5, store)
        elif path.suffix.lower() in {".json", ".jsonl"}:
            parse_generic_json_or_jsonl(path, store)
        elif path.suffix.lower() == ".txt":
            parse_generic_txt(path, store)
        elif path.suffix.lower() in {".owl", ".xml", ".rdf"}:
            parse_generic_owl_or_xml(path, store)

    return store


def rank_external_seed_terms(
    candidates: Dict[str, Dict[str, SeedTermInfo]],
    archive_keywords: Dict[str, List[str]],
    outfits: Sequence[OutfitBundle],
) -> Dict[str, List[str]]:
    archive_keyword_set = {
        slot: {normalize_seed_piece(term) for term in terms}
        for slot, terms in archive_keywords.items()
    }

    ranked_output: Dict[str, List[str]] = {}
    for slot in SLOT_NAMES:
        scored: List[Tuple[int, str]] = []
        for term, info in candidates.get(slot, {}).items():
            normalized = normalize_seed_piece(term)
            if reject_seed_term(normalized, slot):
                continue

            support = archive_support_count(normalized, outfits)
            if (
                slot == "colors_tags"
                and normalized not in archive_keyword_set.get(slot, set())
                and normalized not in CORE_COLOR_ALLOWLIST
            ):
                continue

            score = info.base_score
            score += len(info.source_groups) * 3
            score += len(info.sources)
            score += min(support, 6)
            if normalized in archive_keyword_set.get(slot, set()):
                score += 5
            elif support > 0:
                score += 3
            if slot == "style_tags":
                if normalized in STYLE_WHITELIST:
                    score += 2
                elif support == 0 and normalized not in archive_keyword_set.get(slot, set()):
                    continue
            scored.append((score, normalized))

        ranked_terms = []
        seen: set[str] = set()
        for _, term in sorted(scored, key=lambda item: (-item[0], item[1])):
            if term in seen:
                continue
            seen.add(term)
            ranked_terms.append(term)

        ranked_output[slot] = ranked_terms[: SLOT_LIMITS_EXTERNAL[slot]]

    return ranked_output


def empty_final_ontology() -> Dict[str, Dict[str, List[str]]]:
    return {
        slot: {
            "archive_keywords": [],
            "external_seed_keywords": [],
        }
        for slot in SLOT_NAMES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Ferre-specific retrieval ontology.")
    parser.add_argument("--archive-input", type=Path, default=DEFAULT_ARCHIVE_INPUT)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION))
    parser.add_argument("--single-pass-char-limit", type=int, default=DEFAULT_SINGLE_PASS_CHAR_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--skip-archive-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.archive_input.exists():
        raise FileNotFoundError(f"Archive input not found: {args.archive_input}")
    if not args.seed_root.exists():
        raise FileNotFoundError(f"Seed root not found: {args.seed_root}")

    print(f"[load] archive input: {args.archive_input}")
    archive_records = load_archive_records(args.archive_input)
    print(f"[load] archive records: {len(archive_records)}")

    outfits = build_outfit_groups(archive_records)
    print(f"[dedupe] distinct outfits: {len(outfits)}")

    seed_files = select_seed_files(args.seed_root)
    print(f"[load] selected seed files: {len(seed_files)}")
    for path in seed_files:
        print(f"  - {path.relative_to(args.seed_root)}")

    if args.skip_archive_llm:
        print("[archive] extracting archive-grounded keywords deterministically (LLM skipped)")
        archive_keywords = harvest_archive_keywords_deterministically(outfits)
    else:
        print("[archive] extracting archive-grounded keywords with Gemini on Vertex AI")
        archive_keywords = build_archive_keywords_with_llm(
            outfits=outfits,
            model=args.model,
            location=args.location,
            single_pass_char_limit=args.single_pass_char_limit,
            batch_size=args.batch_size,
        )

    print("[seed] extracting external seed keywords")
    seed_candidates = extract_external_seed_candidates(seed_files)
    external_seed_keywords = rank_external_seed_terms(
        candidates=seed_candidates,
        archive_keywords=archive_keywords,
        outfits=outfits,
    )

    final = empty_final_ontology()
    for slot in SLOT_NAMES:
        final[slot]["archive_keywords"] = archive_keywords.get(slot, [])
        final[slot]["external_seed_keywords"] = external_seed_keywords.get(slot, [])

    write_json(args.output, final)
    print(f"[write] ontology written to: {args.output}")


if __name__ == "__main__":
    main()
