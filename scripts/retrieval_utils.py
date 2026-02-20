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