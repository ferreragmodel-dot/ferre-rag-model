import os
import fitz  # PyMuPDF
from google import genai
from google.genai.types import Content
import json
import time
from typing import Optional

PDF_DIR = "input-datasets/ferre-notes-lessons"
OUTPUT_FILE = "ferre_archive_metadata.json"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 32  # seconds


def call_llm_with_retry(llm_client, prompt: str, filename: str) -> Optional[str]:
    """Call LLM with exponential backoff retry logic."""
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=Content(role="user", parts=[{"text": prompt}])
            )

            # Parse response
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
            is_retryable = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "500" in error_str or "503" in error_str

            if is_billing_error:
                # Billing errors are not retryable
                print(f"  Billing error (not retryable): {error_str[:100]}")
                return None

            if attempt < MAX_RETRIES - 1 and is_retryable:
                print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed for {filename}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                print(f"  LLM call failed for {filename}: {error_str[:200]}")
                return None

    return None


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

    metadata = {}

    if not os.path.isdir(PDF_DIR):
        print(f"PDF directory not found: {PDF_DIR}")
        return

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, filename)
            try:
                doc = fitz.open(file_path)
            except Exception as e:
                print(f"Failed to open PDF {file_path}: {e}")
                continue

            # Extract first page text (or more if needed)
            try:
                text = doc[0].get_text()
            except Exception:
                text = ""

            prompt = f"""Extract the following metadata from this document and return ONLY a valid JSON object, no other text:
{{
  "title": "...",
  "year": "...",
  "main_topics": "...",
  "author": "...",
  "type": "..."
}}

If a field is not available, use null. Make sure to return only valid JSON.

Document text (first page):
{text}
"""

            text_resp = call_llm_with_retry(llm_client, prompt, filename)
            if text_resp is None:
                metadata[filename] = {"error": "Failed after retries"}
                continue

            # Try to parse as JSON
            try:
                parsed_metadata = json.loads(text_resp)
                metadata[filename] = parsed_metadata
                print(f"Successfully extracted metadata for {filename}")
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                try:
                    # Look for JSON object in the response
                    import re
                    json_match = re.search(r'\{[^{}]*\}', text_resp, re.DOTALL)
                    if json_match:
                        parsed_metadata = json.loads(json_match.group())
                        metadata[filename] = parsed_metadata
                        print(f"Successfully extracted metadata for {filename} (from embedded JSON)")
                    else:
                        # Store raw response if no JSON found
                        metadata[filename] = {"raw_response": text_resp}
                        print(f"Warning: Could not parse JSON for {filename}, storing raw response")
                except Exception as parse_error:
                    metadata[filename] = {"error": f"Failed to parse response: {str(parse_error)}", "raw_response": text_resp}
                    print(f"Error parsing response for {filename}: {parse_error}")

    # Save to JSON
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"LLM-extracted metadata written to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to write output file: {e}")


if __name__ == "__main__":
    main()
