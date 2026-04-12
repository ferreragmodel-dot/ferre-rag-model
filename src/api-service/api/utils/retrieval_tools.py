from google.genai import types

# ── Image archive constants ────────────────────────────────────────────────────
IMAGE_SEASONS = ["FW1986-1987", "FW1987-1988", "FW1988-1989", "SS1987", "SS1988", "SS1989"]
IMAGE_YEARS   = ["1986", "1987", "1988", "1989"]

# Single FunctionDeclaration used to extract structured visual filters from a
# free-text query.  All filter parameters are optional — the LLM only fills in
# what the query explicitly mentions.
image_search_func = types.FunctionDeclaration(
    name="search_images",
    description=(
        "Search the Gianfranco Ferré fashion archive image collection. "
        "Extract any visual or temporal filters mentioned in the query "
        "(season, year, garments, colors, materials, etc.). "
        "Always provide a search_query even if filters are present."
    ),
    parameters={
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": (
                    "Semantic search query for image similarity — rephrase the user's "
                    "request as a descriptive visual sentence."
                ),
            },
            "season_path": {
                "type": "string",
                "enum": IMAGE_SEASONS,
                "description": "Season filter, e.g. 'FW1986-1987' or 'SS1988'.",
            },
            "year_path": {
                "type": "string",
                "enum": IMAGE_YEARS,
                "description": "Year filter, e.g. '1987'.",
            },
            "garments_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Garment types present in the image (e.g. 'dress', 'jacket', 'coat').",
            },
            "colors_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Colors present in the image (e.g. 'black', 'red', 'ivory').",
            },
            "material_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fabric or material (e.g. 'silk', 'wool', 'leather').",
            },
            "patterns_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Patterns or prints (e.g. 'stripe', 'floral', 'solid').",
            },
            "silhouette_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Silhouette descriptors (e.g. 'fitted', 'loose', 'structured').",
            },
            "length_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Garment length (e.g. 'knee', 'midi', 'floor').",
            },
            "neckline_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Neckline style (e.g. 'v neckline', 'round neck', 'high neck').",
            },
            "sleeve_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Sleeve style (e.g. 'long sleeve', 'sleeveless', 'short sleeve').",
            },
            "closure_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Closure type (e.g. 'button', 'zipper', 'buckle').",
            },
            "embellishment_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Embellishments or details (e.g. 'bow', 'lace', 'embroidery').",
            },
            "style_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Overall style (e.g. 'formal', 'casual', 'dress clothes').",
            },
        },
        "required": ["search_query"],
    },
)

image_search_tool = types.Tool(function_declarations=[image_search_func])

# ── Text archive constants ─────────────────────────────────────────────────────
# All documents available in the archive (filename without .pdf extension).
# These match the 'doc' metadata field stored in ChromaDB.
ARCHIVE_YEARS = ["1973", "1996", "1997", "1998", "1999", "2000", "2001", "2003", "2005", "2006", "2007", "2023"]

ARCHIVE_DOCUMENTS = [
    "Archivio Fondazione Ferre (Server)",
    "Lesson_Composition and Fashion",
    "Lesson_Creating a style",
    "Lesson_Creativity and Working Method",
    "Lesson_Design in Fashion",
    "Lesson_Designing the Material",
    "Lesson_Exotic Inspirations",
    "Lesson_Fashion Design and Creativity",
    "Lesson_Jewelry",
    "Lesson_Men's Fashion",
    "Lesson_Tailor of two cities",
    "Lesson_The form of Emotions",
    "Lesson_The Itinerary of Design",
    "Notes_Accessory",
    "Notes_Authenticity",
    "Notes_China",
    "Notes_Colors and Feelings",
    "Notes_Creative Path",
    "Notes_Details",
    "Notes_Dior",
    "Notes_Elegance",
    "Notes_Emotions",
    "Notes_Fashion",
    "Notes_Fashion and Architecture",
    "Notes_Fashion and Art",
    "Notes_Fashion and Ethics",
    "Notes_Femininity",
    "Notes_India",
    "Notes_Inspiration",
    "Notes_Jewels",
    "Notes_Lexicon",
    "Notes_Materials",
    "Notes_My Woman",
    "Notes_Past and Future",
    "Notes_Reference Figures",
    "Notes_Shapes",
    "Notes_Sizes+",
    "Notes_Values",
    "Notes_White shirt",
]

# Tool 1: General semantic search across the full archive
search_archive_func = types.FunctionDeclaration(
    name="search_archive",
    description=(
        "Search the Gianfranco Ferre archive by topic or content across all documents. "
        "Use this for general questions about Ferre's work, fashion philosophy, design process, "
        "materials, elegance, creativity, and other broad topics."
    ),
    parameters={
        "type": "object",
        "properties": {
            "search_content": {
                "type": "string",
                "description": (
                    "The topic or question to search for. Expand into a descriptive sentence "
                    "for better semantic matching (e.g. 'Ferre's view on the relationship "
                    "between architecture and fashion design')."
                ),
            },
        },
        "required": ["search_content"],
    },
)

# Tool 2: Search within a specific document
search_by_document_func = types.FunctionDeclaration(
    name="search_by_document",
    description=(
        "Search within a specific document from the Gianfranco Ferre archive. "
        "Use when the query explicitly refers to a known document, lesson, or topic area "
        "(e.g. India travel notes, white shirt essay, jewelry lesson, Dior notes)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "search_content": {
                "type": "string",
                "description": "The topic or question to search for within the document.",
            },
            "doc": {
                "type": "string",
                "description": "The specific document to search within.",
                "enum": ARCHIVE_DOCUMENTS,
            },
        },
        "required": ["search_content", "doc"],
    },
)

# Tool 3: Search filtered by year
search_by_year_func = types.FunctionDeclaration(
    name="search_by_year",
    description=(
        "Search the Gianfranco Ferre archive filtered to documents from a specific year. "
        "Use when the query asks about a particular period or year."
    ),
    parameters={
        "type": "object",
        "properties": {
            "search_content": {
                "type": "string",
                "description": "The topic or question to search for within documents from that year.",
            },
            "year": {
                "type": "string",
                "description": "The year to filter by.",
                "enum": ARCHIVE_YEARS,
            },
        },
        "required": ["search_content", "year"],
    },
)

ferre_archive_tool = types.Tool(
    function_declarations=[search_archive_func, search_by_document_func, search_by_year_func]
)


def search_archive(search_content, collection, embed_func, top_k=10):
    """Search the full archive by semantic similarity.

    Returns:
        tuple: (formatted_text, sources_list)
    """
    query_embedding = embed_func(search_content)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    sources = []
    for doc, meta in zip(docs, metadatas):
        sources.append({
            "document": meta.get("doc", "Unknown"),
            "year": meta.get("year", "Unknown"),
            "excerpt": doc[:400] + "..." if len(doc) > 400 else doc,
        })

    text = "\n\n".join(docs) if docs else "No results found."
    return text, sources


def search_by_year(search_content, year, collection, embed_func, top_k=10):
    """Search the archive filtered to documents from a specific year.

    Returns:
        tuple: (formatted_text, sources_list)
    """
    query_embedding = embed_func(search_content)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"year": year},
    )
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    sources = []
    for doc, meta in zip(docs, metadatas):
        sources.append({
            "document": meta.get("doc", "Unknown"),
            "year": meta.get("year", "Unknown"),
            "excerpt": doc[:400] + "..." if len(doc) > 400 else doc,
        })

    text = "\n\n".join(docs) if docs else f"No results found for year '{year}'."
    return text, sources


def search_by_document(search_content, doc, collection, embed_func, top_k=10):
    """Search within a specific document by semantic similarity.

    Returns:
        tuple: (formatted_text, sources_list)
    """
    query_embedding = embed_func(search_content)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"doc": doc},
    )
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    sources = []
    for doc_text, meta in zip(docs, metadatas):
        sources.append({
            "document": meta.get("doc", "Unknown"),
            "year": meta.get("year", "Unknown"),
            "excerpt": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
        })

    text = "\n\n".join(docs) if docs else f"No results found in '{doc}'."
    return text, sources


def execute_function_calls(function_calls, collection, embed_func, top_k=10):
    """Execute LLM-requested function calls and return function response Parts and sources.

    Returns:
        tuple: (parts_list, sources_list) - sources limited to top 5 most relevant
    """
    parts = []
    all_sources = []

    for function_call in function_calls:
        name = function_call.name
        args = dict(function_call.args)
        print(f"  Calling: {name}({args})")

        if name == "search_archive":
            response_text, sources = search_archive(args["search_content"], collection, embed_func, top_k=top_k)
        elif name == "search_by_document":
            response_text, sources = search_by_document(
                args["search_content"], args["doc"], collection, embed_func, top_k=top_k
            )
        elif name == "search_by_year":
            response_text, sources = search_by_year(
                args["search_content"], args["year"], collection, embed_func, top_k=top_k
            )
        else:
            print(f"  Unknown function: {name}")
            continue

        print(f"  Retrieved {len(response_text)} chars from {len(sources)} sources")

        # Collect unique sources (deduplicate by document + excerpt)
        for source in sources:
            if source not in all_sources:
                all_sources.append(source)

        parts.append(
            types.Part.from_function_response(
                name=name,
                response={"content": response_text},
            )
        )

    # Limit to top 5 most relevant sources
    return parts, all_sources[:5]
