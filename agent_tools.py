# Ferre Archive Agent Tools
# Implements function calling tools for the Gianfranco Ferré archive RAG agent.
# The LLM uses these tools to decide how to filter/retrieve relevant chunks automatically.

from google.genai import types

# ---------------------------------------------------------------------------
# Enum values derived from metadata/ferre_archive_metadata.json
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = ["lesson", "note", "archive"]

DOCUMENT_NAMES = [
    "Archivio Fondazione Ferré (Server)",
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
    "Lesson_The Itinerary of Design",
    "Lesson_The form of Emotions",
    "Notes_Accessory",
    "Notes_Authenticity",
    "Notes_China",
    "Notes_Colors and Feelings",
    "Notes_Creative Path",
    "Notes_Details",
    "Notes_Dior",
    "Notes_Elegance",
    "Notes_Emotions",
    "Notes_Fashion and Architecture",
    "Notes_Fashion and Art",
    "Notes_Fashion and Ethics",
    "Notes_Fashion",
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

# ---------------------------------------------------------------------------
# Function declarations (schema exposed to the LLM)
# ---------------------------------------------------------------------------

search_archive_func = types.FunctionDeclaration(
    name="search_archive",
    description=(
        "Search the Gianfranco Ferré archive for relevant content using semantic similarity. "
        "Use this for general queries not tied to a specific document type or title."
    ),
    parameters={
        "type": "object",
        "properties": {
            "search_content": {
                "type": "string",
                "description": (
                    "The search query. Expand to a sentence or two for better semantic matches."
                ),
            },
        },
        "required": ["search_content"],
    },
)

search_by_type_func = types.FunctionDeclaration(
    name="search_by_type",
    description=(
        "Search the Ferré archive filtered by document type. "
        "Use 'lesson' for lecture transcripts, 'note' for written notes/reflections, "
        "'archive' for the general archive index document."
    ),
    parameters={
        "type": "object",
        "properties": {
            "doc_type": {
                "type": "string",
                "description": "Document type to filter by.",
                "enum": DOCUMENT_TYPES,
            },
            "search_content": {
                "type": "string",
                "description": (
                    "The search query. Expand to a sentence or two for better semantic matches."
                ),
            },
        },
        "required": ["doc_type", "search_content"],
    },
)

search_by_document_func = types.FunctionDeclaration(
    name="search_by_document",
    description=(
        "Search within a specific Ferré archive document by its exact title. "
        "Use when the user references a specific lesson or note by name."
    ),
    parameters={
        "type": "object",
        "properties": {
            "document_name": {
                "type": "string",
                "description": "The exact document key from the archive.",
                "enum": DOCUMENT_NAMES,
            },
            "search_content": {
                "type": "string",
                "description": (
                    "The search query. Expand to a sentence or two for better semantic matches."
                ),
            },
        },
        "required": ["document_name", "search_content"],
    },
)

ferre_expert_tool = types.Tool(
    function_declarations=[
        search_archive_func,
        search_by_type_func,
        search_by_document_func,
    ]
)

# ---------------------------------------------------------------------------
# Tool implementation functions
# ---------------------------------------------------------------------------

def search_archive(search_content, collection, embed_func, top_k=10):
    query_embedding = embed_func(search_content)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return "\n\n".join(results["documents"][0])


def search_by_type(doc_type, search_content, collection, embed_func, top_k=10):
    query_embedding = embed_func(search_content)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"type": doc_type},
    )
    return "\n\n".join(results["documents"][0])


def search_by_document(document_name, search_content, collection, embed_func, top_k=10):
    query_embedding = embed_func(search_content)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"doc": document_name},
    )
    return "\n\n".join(results["documents"][0])


def execute_function_calls(function_calls, collection, embed_func, top_k=10):
    """
    Execute function calls returned by the LLM and return a list of
    Part.from_function_response objects to feed back into the next LLM turn.
    """
    parts = []
    for function_call in function_calls:
        print("Function call:", function_call.name, dict(function_call.args))
        response = None

        if function_call.name == "search_archive":
            response = search_archive(
                function_call.args["search_content"],
                collection,
                embed_func,
                top_k=top_k,
            )
        elif function_call.name == "search_by_type":
            response = search_by_type(
                function_call.args["doc_type"],
                function_call.args["search_content"],
                collection,
                embed_func,
                top_k=top_k,
            )
        elif function_call.name == "search_by_document":
            response = search_by_document(
                function_call.args["document_name"],
                function_call.args["search_content"],
                collection,
                embed_func,
                top_k=top_k,
            )

        if response is not None:
            print("Retrieved content (first 300 chars):", response[:300], "...")
            parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={"content": response},
                )
            )

    return parts


# ---------------------------------------------------------------------------
# Reference: original cheese-book agent tools (kept for reference)
# ---------------------------------------------------------------------------

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
