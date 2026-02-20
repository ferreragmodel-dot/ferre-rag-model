import chromadb
import numpy as np

def main():
    client = chromadb.Client()
    col = client.create_collection("smoke_meta")

    docs = ["a", "b", "c", "d"]
    metas = [
        {"doc": "Notes_Fashion", "type": "note"},
        {"doc": "Notes_Fashion", "type": "note"},
        {"doc": "Lesson_Jewelry", "type": "lesson"},
        {"doc": "Lesson_Jewelry", "type": "lesson"},
    ]
    embs = np.random.randn(4, 8).tolist()
    col.add(ids=[f"id{i}" for i in range(4)], documents=docs, metadatas=metas, embeddings=embs)

    q = np.random.randn(8).tolist()
    r = col.query(query_embeddings=[q], n_results=10, where={"type": "lesson"})
    assert all(m["type"] == "lesson" for m in r["metadatas"][0])

    print("PASS: metadata filtering works. Returned:", r["ids"][0])

if __name__ == "__main__":
    main()