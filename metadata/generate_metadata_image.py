import os
from google import genai
from google.genai.types import Content
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path
import re

# Image root folder
IMAGES_DIR = "input-datasets/ferre-designs"
OUTPUT_DIR = "metadata"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 32  # seconds

# Supported asset types (folder-name substring routing)
ASSET_FASHION_SHOW = "fashion_show_photos"
ASSET_TECHNICAL_DRAWINGS = "technical_drawings"
# TODO: add other asset types here


def call_llm_with_retry(llm_client, prompt: str, key: str) -> Optional[str]:
    """Call LLM with exponential backoff retry logic."""
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=Content(role="user", parts=[{"text": prompt}]),
            )

            text_resp = getattr(response, "text", None)
            if text_resp is None:
                try:
                    text_resp = response.candidates[0].content.parts[0].text
                except Exception:
                    text_resp = str(response)

            return text_resp

        except Exception as e:
            error_str = str(e)
            is_billing_error = "BILLING_DISABLED" in error_str or "billing" in error_str.lower()
            is_retryable = (
                "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "500" in error_str
                or "503" in error_str
            )

            if is_billing_error:
                print(f"  Billing error (not retryable): {error_str[:100]}")
                return None

            if attempt < MAX_RETRIES - 1 and is_retryable:
                print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed for {key}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                print(f"  LLM call failed for {key}: {error_str[:200]}")
                return None

    return None


def detect_asset_type(rel_path: str) -> Optional[str]:
    """
    Route by folder-name substring checks.
    Example desired structure:
      ferre-designs/ALTA-MODA/FW1986-87/fashion_show_photos/xxx.jpg
    """
    p = rel_path.replace("\\", "/").lower()
    if f"/{ASSET_FASHION_SHOW}/" in f"/{p}/" or ASSET_FASHION_SHOW in p:
        return ASSET_FASHION_SHOW
    if f"/{ASSET_TECHNICAL_DRAWINGS}/" in f"/{p}/" or ASSET_TECHNICAL_DRAWINGS in p:
        return ASSET_TECHNICAL_DRAWINGS

    # TODO: add other asset types here
    return None


def parse_known_context(rel_path: str, asset_type: str) -> Dict[str, Any]:
    """
    Extract known context from folder structure, e.g.
      ALTA-MODA/FW1986-87/fashion_show_photos/61344.jpg

    Returns keys:
      - year: "1986-1987" or "1987"
      - season: "Fall/Winter" or "Spring/Summer"
      - season_code: "FW" or "SS"
      - collection_line: e.g. "ALTA-MODA" if available
      - asset_type: "fashion_show_photos" or "technical_drawings" etc.
      - source_path: rel_path (stable key)
    """
    p = rel_path.replace("\\", "/")
    parts = [x for x in p.split("/") if x]

    collection_line = parts[0] if len(parts) >= 1 else None

    # Find season folder token like FW1986-87 or SS1987 anywhere in the path
    season_folder = None
    for token in parts:
        if re.match(r"^(FW|SS)\d{4}(-\d{2})?$", token):
            season_folder = token
            break

    season_code = None
    season = None
    year = None

    if season_folder:
        season_code = season_folder[:2]  # FW / SS
        season = "Fall/Winter" if season_code == "FW" else "Spring/Summer"

        # Examples:
        #   FW1986-87 -> year "1986-1987"
        #   SS1987    -> year "1987"
        m = re.match(r"^(FW|SS)(\d{4})(-\d{2})?$", season_folder)
        if m:
            start_year = m.group(2)  # "1986"
            end_two = m.group(3)     # "-87" or None
            if end_two:
                end_year = start_year[:2] + end_two.replace("-", "")  # "1987"
                year = f"{start_year}-{end_year}"
            else:
                year = start_year

    ctx = {
        "year": year,
        "season": season,
        "season_code": season_code,
        "collection_line": collection_line,
        "asset_type": asset_type,
        "source_path": p,
    }

    return ctx


def build_prompt(asset_type: str, rel_path: str, known_ctx: Dict[str, Any]) -> str:
    """
    - request ONLY valid JSON
    - null for missing fields
    - keep strings short
    """
    if asset_type == ASSET_FASHION_SHOW:
        return f"""You are extracting metadata for a Ferré fashion archive image.

Return ONLY a valid JSON object (no extra text). Use null if not available.

Known context (already parsed from folder structure; DO NOT change these values):
{{
  "year": {json.dumps(known_ctx.get("year"))},
  "season": {json.dumps(known_ctx.get("season"))},
  "season_code": {json.dumps(known_ctx.get("season_code"))},
  "collection_line": {json.dumps(known_ctx.get("collection_line"))},
  "asset_type": {json.dumps(known_ctx.get("asset_type"))},
  "source_path": {json.dumps(known_ctx.get("source_path"))}
}}

Now extract additional metadata for this fashion show photo. Return ONLY this JSON shape:

{{
  "year": {json.dumps(known_ctx.get("year"))},
  "season": {json.dumps(known_ctx.get("season"))},
  "season_code": {json.dumps(known_ctx.get("season_code"))},
  "collection_line": {json.dumps(known_ctx.get("collection_line"))},
  "asset_type": "fashion_show_photos",
  "source_path": {json.dumps(known_ctx.get("source_path"))},

  "garments": [],      // array of lowercase strings (e.g., ["coat", "dress", "trousers"])
  "colors": [],        // array of lowercase strings (e.g., ["black", "white"])
  "materials": [],     // array of lowercase strings (e.g., ["silk", "wool"])
  "patterns": []       // array of lowercase strings (e.g., ["stripes", "floral"]) or []
  "silhouette": "...",    // short phrase (e.g., "tailored, structured") or null
  "notes": "..."          // short extra details or null
}}

Rules:
- Keep values concise.
- Do not invent or hallucinate details that are not clearly supported by the image or provided context.
- If a field cannot be confidently determined, use null (for single-value fields) or [] (for array fields).
- If none detected, return an empty array [].
- Use arrays of short lowercase strings for garments, colors, materials, and patterns.
- Do NOT return comma-separated strings.
- Focus ONLY on the primary model in the foreground.
- If multiple models appear in the image, ignore background models and extract metadata only for the main, central figure.
- Keep descriptions concise and objective.
- Return ONLY valid JSON.

Image path:
{rel_path}
"""

    if asset_type == ASSET_TECHNICAL_DRAWINGS:
        # Placeholder prompt 
        return f"""TODO: Add technical drawing metadata extraction prompt.

Return ONLY a valid JSON object (no extra text). Use null if not available.

Return ONLY this JSON shape:
{{
  "asset_type": "technical_drawings",
  "source_path": "{rel_path}"
}}

Rules:
- Return ONLY valid JSON.

Image path:
{rel_path}
"""

    # TODO: add other asset types here
    raise ValueError(f"Unknown asset_type: {asset_type}")


def parse_json_response(text_resp: str) -> dict:
    """Parse JSON; fallback to extracting an embedded JSON object."""
    try:
        return json.loads(text_resp)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", text_resp, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass
        return {"raw_response": text_resp}


def enforce_known_context(parsed: dict, known_ctx: Dict[str, Any]) -> dict:
    """
    Force known context values into parsed output to avoid LLM drift.
    """
    for k in ["year", "season", "season_code", "collection_line", "asset_type", "source_path"]:
        parsed[k] = known_ctx.get(k)
    return parsed


def main():
    # Basic env checks
    if "GCP_PROJECT" not in os.environ:
        print("Environment variable GCP_PROJECT is not set. Please export it or load .env before running.")
        return
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set. Please export it before running.")
        return

    # Initialize Vertex AI client
    try:
        llm_client = genai.Client(vertexai=True, project=os.environ["GCP_PROJECT"], location="us-central1")
    except Exception as e:
        print("Failed to initialize Vertex AI client:", str(e))
        return

    if not os.path.isdir(IMAGES_DIR):
        print(f"Images directory not found: {IMAGES_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_by_type: Dict[str, Dict[str, Any]] = {}

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

    for root, _, files in os.walk(IMAGES_DIR):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in image_extensions:
                continue

            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, IMAGES_DIR).replace("\\", "/")

            asset_type = detect_asset_type(rel_path)
            if asset_type is None:
                continue

            if asset_type not in metadata_by_type:
                metadata_by_type[asset_type] = {}

            key = rel_path  # use relative path as the key (stable)

            known_ctx = parse_known_context(rel_path, asset_type)
            prompt = build_prompt(asset_type, rel_path, known_ctx)

            text_resp = call_llm_with_retry(llm_client, prompt, key)
            if text_resp is None:
                metadata_by_type[asset_type][key] = {"error": "Failed after retries", **known_ctx}
                continue

            parsed = parse_json_response(text_resp)
            parsed = enforce_known_context(parsed, known_ctx)

            metadata_by_type[asset_type][key] = parsed
            print(f"Successfully extracted metadata for {key}")


    # Write one file per asset type (name includes asset type)
    for asset_type, metadata in metadata_by_type.items():
        output_file = os.path.join(OUTPUT_DIR, f"ferre_images_{asset_type}_metadata.json")
        try:
            with open(output_file, "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"LLM-extracted metadata written to {output_file}")
        except Exception as e:
            print(f"Failed to write output file {output_file}: {e}")


if __name__ == "__main__":
    main()