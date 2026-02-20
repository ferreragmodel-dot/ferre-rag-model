import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import chromadb

# Import from the lightweight module (NOT cli.py)
from retrieval_core import load_text_embeddings, ferre_mappings


def main():
    # Pick a key that exists in your mapping JSON
    doc_key = "Lesson_Jewelry"
    assert doc_key in ferre_mappings, f"{doc_key} not found in ferre_mappings"

    # Dummy embedding df (no PDFs / no GCP required)
    df = pd.DataFrame({
        "chunk": ["chunk A", "chunk B"],
        "doc": [doc_key, doc_key],
        "embedding": [np.random.randn(8).tolist(), np.random.randn(8).tolist()],
    })

    # Local in-memory Chroma collection
    client = chromadb.Client()
    col = client.create_collection("merge_test")

    # Load into Chroma using your real loader
    load_text_embeddings(df, col, batch_size=10)

    out = col.get(limit=5)
    print("Metadatas:")
    for m in out["metadatas"]:
        print(m)

    # Validate merged metadata
    m0 = out["metadatas"][0]
    assert m0["doc"] == doc_key
    assert "type" in m0, "missing 'type' in metadata"
    assert "title" in m0, "missing 'title' in metadata"
    assert "filename" in m0, "missing 'filename' in metadata"

    print("PASS: mapping metadata merged into chunk metadata")


if __name__ == "__main__":
    main()