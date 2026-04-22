import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import base64
from pathlib import Path
import traceback
import chromadb

# Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from google.genai import errors
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel

from api.utils.retrieval_tools import ferre_archive_tool, execute_function_calls, image_search_tool

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
MULTIMODAL_EMBEDDING_DIMENSION = 1408  # For image embeddings
GENERATIVE_MODEL = "gemini-2.0-flash-001"
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
CHROMADB_PORT = os.environ["CHROMADB_PORT"]

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

# Initialize Vertex AI for multimodal embeddings
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
multimodal_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in Gianfranco Ferre and fashion archive research. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response using only the information found in the given chunks.
4. When citing information, use inline citations in the format [1], [2], [3], etc. that reference the numbered sources.
5. **Only cite the most important and unique sources (maximum 5 citations).** Do not over-cite; focus on the most relevant information sources.
6. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
7. Always maintain a professional and knowledgeable tone, befitting a Ferre archive expert.
8. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.

Remember:
- You are an expert in Ferre and fashion, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- If asked about topics unrelated to Ferre or fashion, politely redirect the conversation back to archive-related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.
- Always cite your sources with [N] markers when making factual claims, but keep citations minimal and meaningful.

Your goal is to provide accurate, helpful information about Ferre and fashion based solely on the content of the text chunks you receive with each query.
"""

# Connect to ChromaDB (optional for local non-RAG runs)
try:
    chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
except Exception as chroma_error:
    chroma_client = None
    print(f"ChromaDB unavailable; running agent without retrieval tools: {chroma_error}")
COLLECTION_NAME = "semantic-split-collection"

# Initialize agent chat sessions
chat_sessions: Dict[str, "AgentChatSession"] = {}


class AgentChatSession:
    """Stateful wrapper for the agentic conversation history.

    The agent calls generate_content() directly (stateless API), so we track the
    conversation history manually as a list of Content objects.
    """

    def __init__(self, history: List[Content] = None):
        self.history: List[Content] = history or []


def create_chat_session(past_history: List[Content] = None) -> AgentChatSession:
    """Create a new agent chat session, optionally with pre-existing history."""
    return AgentChatSession(history=list(past_history) if past_history else [])


def generate_query_embedding(query: str) -> List[float]:
    """Generate an embedding vector for a query string."""
    response = llm_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
    )
    return response.embeddings[0].values


def generate_image_query_embedding(query: str) -> List[float]:
    """Generate a multimodal embedding vector for image search.

    Uses the multimodalembedding@001 model to generate text embeddings that are
    in the same 1408-dimensional space as the image embeddings, allowing for
    proper semantic similarity search.
    """
    try:
        response = multimodal_model.get_embeddings(
            contextual_text=query,
            dimension=MULTIMODAL_EMBEDDING_DIMENSION
        )
        if response and response.text_embedding:
            return response.text_embedding
        else:
            raise ValueError("No text embeddings returned from multimodal model")
    except Exception as e:
        print(f"Error generating multimodal embedding: {e}")
        raise


def _build_user_content(message: Dict) -> Content:
    """Build a Content object from a message dict (text and/or image)."""
    user_parts = []

    if message.get("image"):
        base64_string = message["image"]
        if "," in base64_string:
            header, base64_data = base64_string.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
        else:
            base64_data = base64_string
            mime_type = "image/jpeg"
        image_bytes = base64.b64decode(base64_data)
        user_parts.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))
        user_parts.append(
            Part.from_text(
                text=message.get("content")
                or "Describe what you see in this image in the context of Gianfranco Ferre fashion archive research"
            )
        )
    elif message.get("image_path"):
        image_path = os.path.join("chat-history", "llm-agent", message["image_path"])
        with Path(image_path).open("rb") as f:
            image_bytes = f.read()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
        }.get(Path(image_path).suffix.lower(), "image/jpeg")
        user_parts.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))
        user_parts.append(
            Part.from_text(
                text=message.get("content")
                or "Describe what you see in this image in the context of Gianfranco Ferre fashion archive research"
            )
        )
    else:
        if message.get("content"):
            user_parts.append(Part.from_text(text=message["content"]))

    if not user_parts:
        raise ValueError("Message must contain either text content or image")

    return Content(role="user", parts=user_parts)


_IMAGE_TAG_FIELDS = [
    "garments_tags", "colors_tags", "material_tags", "patterns_tags",
    "silhouette_tags", "length_tags", "neckline_tags", "sleeve_tags",
    "closure_tags", "embellishment_tags", "style_tags",
]


def _build_chroma_filters(args: dict) -> tuple[Optional[dict], Optional[dict]]:
    """Convert LLM-extracted filter args into ChromaDB where + where_document clauses.

    - Scalar fields (season_path, year_path): exact-match via `where` metadata filter.
    - Tag fields: all tag values are stored as a single concatenated string in the
      ChromaDB document field, so they are filtered via `where_document` $contains.
      Multiple tag values are AND-ed (each must appear in the document).

    Returns:
        (where, where_document) — either may be None if no filters of that type.
    """
    # Scalar metadata filters
    scalar_conditions = []
    for field in ("season_path", "year_path"):
        value = args.get(field)
        if value:
            scalar_conditions.append({field: value})

    where: Optional[dict] = None
    if len(scalar_conditions) == 1:
        where = scalar_conditions[0]
    elif len(scalar_conditions) > 1:
        where = {"$and": scalar_conditions}

    # Tag filters → where_document $contains (all tags are in the document text)
    tag_values = []
    for field in _IMAGE_TAG_FIELDS:
        tag_values.extend(v for v in (args.get(field) or []) if v)

    where_document: Optional[dict] = None
    if len(tag_values) == 1:
        where_document = {"$contains": tag_values[0]}
    elif len(tag_values) > 1:
        where_document = {"$and": [{"$contains": v} for v in tag_values]}

    return where, where_document


def extract_image_filters(query: str) -> tuple[str, Optional[dict], Optional[dict]]:
    """Use Gemini to extract visual filters from a free-text query.

    Returns:
        (refined_search_query, where, where_document)
        where       — ChromaDB metadata filter (season_path, year_path)
        where_document — ChromaDB document filter (tag values via $contains)
        Both are None when no filters of that type were identified.
    """
    try:
        response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=[Content(role="user", parts=[Part.from_text(text=query)])],
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[image_search_tool],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="any")
                ),
            ),
        )
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]
        if not function_calls:
            return query, None, None

        args = dict(function_calls[0].args)
        refined_query = args.get("search_query") or query
        where, where_document = _build_chroma_filters(args)
        print(f"Image filters — query: '{refined_query}', where: {where}, where_document: {where_document}")
        return refined_query, where, where_document

    except Exception as e:
        print(f"extract_image_filters failed, falling back to plain query: {e}")
        return query, None, None


def retrieve_text_chunks(session: AgentChatSession, message: Dict) -> tuple:
    """
    Steps 1 & 2 of the agentic pipeline: tool selection + ChromaDB retrieval.

    Does NOT modify session.history — call generate_final_answer() next.

    Returns:
        tuple: (user_content, tool_call_content, function_responses_content, sources)
        tool_call_content and function_responses_content are None when ChromaDB is
        unavailable or no tool was selected.
    """
    try:
        user_content = _build_user_content(message)

        if chroma_client is None:
            return user_content, None, None, []

        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        # Step 1: LLM selects which tool(s) to call
        tool_selection_response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=session.history + [user_content],
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[ferre_archive_tool],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="any")
                ),
            ),
        )

        function_calls = [
            part.function_call
            for part in tool_selection_response.candidates[0].content.parts
            if part.function_call
        ]
        print("Function calls:", function_calls)

        if not function_calls:
            return user_content, None, None, []

        tool_call_content = tool_selection_response.candidates[0].content

        # Step 2: Execute function calls against ChromaDB
        function_responses, sources = execute_function_calls(
            function_calls, collection, embed_func=generate_query_embedding
        )

        return user_content, tool_call_content, Content(parts=function_responses), sources

    except Exception as e:
        print(f"Error retrieving text chunks: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve text chunks: {str(e)}",
        )


def generate_final_answer(
    session: AgentChatSession,
    user_content: Content,
    tool_call_content: Optional[Content],
    function_responses_content: Optional[Content],
    sources: List,
    image_context: Optional[str] = None,
) -> tuple:
    """
    Step 3: Generate the grounded final answer.

    Optionally injects image metadata context so the LLM can reference the
    images being shown to the user alongside the response.
    Updates session.history.

    Returns:
        tuple: (response_text, sources)
    """
    try:
        system = SYSTEM_INSTRUCTION
        if image_context:
            system = system + "\n\n" + image_context

        contents = list(session.history) + [user_content]

        if tool_call_content is not None:
            contents.extend([tool_call_content, function_responses_content])
            config = types.GenerateContentConfig(
                system_instruction=system,
                tools=[ferre_archive_tool],
            )
        else:
            config = types.GenerateContentConfig(system_instruction=system)

        final_response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=contents,
            config=config,
        )
        final_text = final_response.text

        # Append the full exchange to history
        session.history.append(user_content)
        if tool_call_content is not None:
            session.history.append(tool_call_content)
            session.history.append(function_responses_content)
        session.history.append(
            Content(role="model", parts=[Part.from_text(text=final_text)])
        )

        return final_text, sources

    except Exception as e:
        print(f"Error generating final answer: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}",
        )


def generate_chat_response(session: AgentChatSession, message: Dict) -> tuple:
    """
    Convenience wrapper: retrieve_text_chunks → generate_final_answer.

    Preserves the original interface without image metadata context injection.
    Use retrieve_text_chunks + generate_final_answer directly when image
    context integration is needed.
    """
    user_content, tool_call_content, function_responses_content, sources = retrieve_text_chunks(
        session, message
    )
    return generate_final_answer(
        session, user_content, tool_call_content, function_responses_content, sources
    )


def rebuild_chat_session(chat_history: List[Dict], history_dir: str = None) -> AgentChatSession:
    """Rebuild an agent chat session from stored chat history.

    Only user text/image and assistant text turns are stored on disk.
    The intermediate tool-call/function-response content is not persisted,
    so the rebuilt history contains simplified user<->assistant pairs.
    """
    formatted_history = []
    for message in chat_history:
        if message["role"] == "user":
            parts = []
            has_image = False
            if message.get("image_path") and history_dir:
                image_full_path = os.path.join(history_dir, message["image_path"])
                try:
                    with Path(image_full_path).open("rb") as f:
                        image_bytes = f.read()
                    mime_type = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                    }.get(Path(image_full_path).suffix.lower(), "image/jpeg")
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                    has_image = True
                except Exception as e:
                    print(f"Error loading image for history rebuild: {str(e)}")
            if message.get("content"):
                parts.append(types.Part.from_text(text=message["content"]))
            elif has_image:
                parts.append(
                    types.Part.from_text(
                        text="Describe what you see in this image in the context of Gianfranco Ferre fashion archive research"
                    )
                )
            if parts:
                formatted_history.append(types.UserContent(parts=parts))
        elif message["role"] == "assistant":
            if message.get("content"):
                formatted_history.append(
                    types.ModelContent(
                        parts=[types.Part.from_text(text=message["content"])]
                    )
                )

    return create_chat_session(formatted_history)
