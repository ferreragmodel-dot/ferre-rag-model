import json
import os
import hashlib

ferre_mappings = {}
for candidate in (
    "ferre_mappings.json",
    "ferre_mappings_llm.json",
    os.path.join("metadata", "ferre_archive_metadata.json"),
):
    if os.path.exists(candidate):
        with open(candidate, "r") as f:
            ferre_mappings = json.load(f)
        break


def parse_filters(filter_args):
    if not filter_args:
        return None
    where = {}
    for s in filter_args:
        if "=" not in s:
            raise ValueError(f"Invalid --filter '{s}'. Use key=value.")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if v.isdigit():
            v = int(v)
        where[k] = v
    return where


def retrieve_chunks(collection, query_embedding, top_k=10, filters=None, contains=None):
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if filters:
        kwargs["where"] = filters
    if contains:
        kwargs["where_document"] = {"$contains": contains}
    return collection.query(**kwargs)


def load_text_embeddings(df, collection, batch_size=500):
    df["id"] = df.index.astype(str)

    key_col = "doc" if "doc" in df.columns else "book" if "book" in df.columns else None
    if key_col is None:
        raise ValueError("Dataframe must contain either 'doc' or 'book' column")

    hashed_keys = df[key_col].astype(str).apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
    )
    df["id"] = hashed_keys + "-" + df["id"]

    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

        ids = batch["id"].tolist()
        documents = batch["chunk"].tolist()
        embeddings = batch["embedding"].tolist()

        metadatas = []
        for key in batch[key_col].tolist():
            item_meta = {key_col: key}
            if key in ferre_mappings:
                item_meta.update(ferre_mappings[key])
            metadatas.append(item_meta)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        total_inserted += len(batch)

    print(f"Inserted {total_inserted} items")