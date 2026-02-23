# TODO - Adapt this file for the Ferre Archive Project. [DONE]
# This file contains example code for implementing agent tools and function calling with LLMs.
# The code is currently commented out and serves as a template for how to define tools, functions, and execute them based on LLM responses.

# import json
# from google import genai
# from google.genai import types

# # import vertexai
# # from vertexai.generative_models import FunctionDeclaration, Tool, Part

# # Specify a function declaration and parameters for an API request
# get_book_by_author_func = types.FunctionDeclaration(
#     name="get_book_by_author",
#     description="Get the book chunks filtered by author name",
#     # Function parameters are specified in OpenAPI JSON schema format
#     parameters={
#         "type": "object",
#         "properties": {
#             "author": {
#                 "type": "string",
#                 "description": "The author name",
#                 "enum": [
#                     "C. F. Langworthy and Caroline Louisa Hunt",
#                     "J. Twamley",
#                     "George E. Newell",
#                     "T. D. Curtis",
#                     "Charles Thom and W. W. Fisk",
#                     "Thomas Wilson Reid",
#                     "Bob Brown",
#                     "Charles S. Brooks",
#                     "Pavlos Protopapas",
#                 ],
#             },
#             "search_content": {
#                 "type": "string",
#                 "description": "The search text to filter content from books. The search term is compared against the book text based on cosine similarity. Expand the search term to a a sentence or two to get better matches",
#             },
#         },
#         "required": ["author", "search_content"],
#     },
# )


# def get_book_by_author(author, search_content, collection, embed_func):

#     query_embedding = embed_func(search_content)

#     # Query based on embedding value
#     results = collection.query(
#         query_embeddings=[query_embedding], n_results=10, where={"author": author}
#     )
#     return "\n".join(results["documents"][0])


# get_book_by_search_content_func = types.FunctionDeclaration(
#     name="get_book_by_search_content",
#     description="Get the book chunks filtered by search terms",
#     # Function parameters are specified in OpenAPI JSON schema format
#     parameters={
#         "type": "object",
#         "properties": {
#             "search_content": {
#                 "type": "string",
#                 "description": "The search text to filter content from books. The search term is compared against the book text based on cosine similarity. Expand the search term to a a sentence or two to get better matches",
#             },
#         },
#         "required": ["search_content"],
#     },
# )


# def get_book_by_search_content(search_content, collection, embed_func):

#     query_embedding = embed_func(search_content)

#     # Query based on embedding value
#     results = collection.query(query_embeddings=[query_embedding], n_results=10)
#     return "\n".join(results["documents"][0])


# # Define all functions available to the cheese expert
# cheese_expert_tool = types.Tool(
#     function_declarations=[get_book_by_author_func, get_book_by_search_content_func]
# )


# def execute_function_calls(function_calls, collection, embed_func):
#     parts = []
#     for function_call in function_calls:
#         print("Function:", function_call.name)
#         if function_call.name == "get_book_by_author":
#             print(
#                 "Calling function with args:",
#                 function_call.args["author"],
#                 function_call.args["search_content"],
#             )
#             response = get_book_by_author(
#                 function_call.args["author"],
#                 function_call.args["search_content"],
#                 collection,
#                 embed_func,
#             )
#             print("Response:", response)
#             # function_responses.append({"function_name":function_call.name, "response": response})
#             parts.append(
#                 types.Part.from_function_response(
#                     name=function_call.name,
#                     response={
#                         "content": response,
#                     },
#                 ),
#             )
#         if function_call.name == "get_book_by_search_content":
#             print("Calling function with args:", function_call.args["search_content"])
#             response = get_book_by_search_content(
#                 function_call.args["search_content"], collection, embed_func
#             )
#             print("Response:", response)
#             # function_responses.append({"function_name":function_call.name, "response": response})
#             parts.append(
#                 types.Part.from_function_response(
#                     name=function_call.name,
#                     response={
#                         "content": response,
#                     },
#                 ),
#             )

#     return parts

# ---------------------------------------------------------------------------
# Ferre Archive Agent Tools
# ---------------------------------------------------------------------------

from google.genai import types

# All documents available in the archive (filename without .pdf extension).
# These match the 'doc' metadata field stored in ChromaDB.
# Years with documents in the archive (documents with null year are excluded).
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
        "Search the Gianfranco Ferré archive by topic or content across all documents. "
        "Use this for general questions about Ferré's work, fashion philosophy, design process, "
        "materials, elegance, creativity, and other broad topics."
    ),
    parameters={
        "type": "object",
        "properties": {
            "search_content": {
                "type": "string",
                "description": (
                    "The topic or question to search for. Expand into a descriptive sentence "
                    "for better semantic matching (e.g. 'Ferré's view on the relationship "
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
        "Search within a specific document from the Gianfranco Ferré archive. "
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
        "Search the Gianfranco Ferré archive filtered to documents from a specific year. "
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
    """Search the full archive by semantic similarity."""
    query_embedding = embed_func(search_content)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else "No results found."


def search_by_year(search_content, year, collection, embed_func, top_k=10):
    """Search the archive filtered to documents from a specific year."""
    query_embedding = embed_func(search_content)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"year": year},
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else f"No results found for year '{year}'."


def search_by_document(search_content, doc, collection, embed_func, top_k=10):
    """Search within a specific document by semantic similarity."""
    query_embedding = embed_func(search_content)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"doc": doc},
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else f"No results found in '{doc}'."


def execute_function_calls(function_calls, collection, embed_func, top_k=10):
    """Execute LLM-requested function calls and return function response Parts."""
    parts = []
    for function_call in function_calls:
        name = function_call.name
        args = dict(function_call.args)
        print(f"  Calling: {name}({args})")

        if name == "search_archive":
            response = search_archive(args["search_content"], collection, embed_func, top_k=top_k)
        elif name == "search_by_document":
            response = search_by_document(
                args["search_content"], args["doc"], collection, embed_func, top_k=top_k
            )
        elif name == "search_by_year":
            response = search_by_year(
                args["search_content"], args["year"], collection, embed_func, top_k=top_k
            )
        else:
            print(f"  Unknown function: {name}")
            continue

        print(f"  Retrieved {len(response)} chars")
        parts.append(
            types.Part.from_function_response(
                name=name,
                response={"content": response},
            )
        )
    return parts
