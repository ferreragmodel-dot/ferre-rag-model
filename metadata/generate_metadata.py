import os
import fitz  # PyMuPDF
from google import genai
from google.genai.types import Content
import json

PDF_DIR = "input-datasets/ferre-notes-lessons"
OUTPUT_FILE = "ferre_mappings_llm.json"


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

            prompt = f"""
Extract the following metadata from this document, if available:
- Title
- Year
- Main topics (comma separated)
- Author (if available)
- Type (e.g. lesson, note, article)

Document text (first page):
{text}
"""

            try:
                response = llm_client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=Content(role="user", parts=[{"text": prompt}])
                )
            except Exception as e:
                print(f"LLM call failed for {filename}: {e}")
                metadata[filename] = {"error": str(e)}
                continue

            # Parse LLM response (assume JSON or structured text)
            text_resp = getattr(response, "text", None)
            if text_resp is None:
                # try candidates
                try:
                    text_resp = response.candidates[0].content.parts[0].text
                except Exception:
                    text_resp = str(response)

            print(f"LLM response for {filename}:\n{text_resp}\n")
            metadata[filename] = text_resp

    # Save to JSON
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"LLM-extracted metadata written to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to write output file: {e}")


if __name__ == "__main__":
    main()
