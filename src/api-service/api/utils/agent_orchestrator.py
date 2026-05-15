import os
import google.auth
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import base64
import mimetypes
from pathlib import Path
import traceback
import chromadb

from google import genai
from google.genai import types
from google.genai.types import Content, Part
from google.genai import errors
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel

from api.utils.retrieval_tools import ferre_archive_tool, execute_function_calls, image_search_tool
from api.utils.gcs_utils import DATASET_PREFIX, fetch_image_bytes, resolve_design_images_dir

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
MULTIMODAL_EMBEDDING_DIMENSION = 1408  # For image embeddings
GENERATIVE_MODEL = os.environ.get("GENERATIVE_MODEL", "gemini-2.0-flash")
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))
CHROMADB_SSL = os.environ.get("CHROMADB_SSL", "false").lower() == "true"

_google_api_key = os.environ.get("GOOGLE_API_KEY")
if _google_api_key:
    llm_client = genai.Client(api_key=_google_api_key)
else:
    _credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    _credentials = _credentials.with_quota_project(GCP_PROJECT)
    llm_client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        credentials=_credentials,
    )

# Initialize Vertex AI for multimodal embeddings (lazy-loaded on first use)
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
_multimodal_model = None

def _get_multimodal_model():
    global _multimodal_model
    if _multimodal_model is None:
        _multimodal_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    return _multimodal_model

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in Gianfranco Ferré and fashion archive research. Each response you produce is accompanied by 3 archive images shown to the user alongside your text. You receive both retrieved text chunks from the archive and metadata/visuals of those images as context.

When answering a query:
1. Carefully read all the text chunks and image context provided.
2. Identify the most relevant information to address the user's question.
3. Formulate your response using only the information found in the provided context.
4. When citing text sources, use inline citations in the format [1], [2], [3], etc.
5. **Only cite the most important and unique sources (maximum 5 citations).** Do not over-cite.
6. If the provided context does not contain sufficient information, state that clearly.
7. Always maintain a professional and knowledgeable tone, befitting a Ferré archive expert.
8. If there are contradictions in the provided context, mention and explain them.

Calibrate the balance between text and images based on the nature of the query:

- **Text-focused queries** (Ferré's philosophy, creative process, biography, opinions on materials, elegance, architecture, travel, etc.): Ground your answer primarily in the retrieved text chunks and cite them. Reference the shown images only briefly when they directly illustrate a point — do not make them the focus.

- **Image-focused queries** (requests to see specific garments, colors, styles, silhouettes, seasons, or collections): Focus your response on describing and contextualizing what is visible in the retrieved images. Keep the textual response concise; use the text chunks only for brief background. Do not cite text sources heavily.

- **Mixed queries**: Balance both accordingly — use text for conceptual depth and images for visual illustration.

When the user asks for more details, background, or specific information about one of the shown images (e.g. "tell me more about image 2", "what materials is image 1 made of?", "who wore image 3?"): respond with exactly this, replacing N with the image number:
"For more details about this piece, click on Image N to open its dedicated archive page and start a conversation directly about it."
Do not attempt to answer detail questions about specific shown images beyond what is already visible in the image context provided.

The labels Image 1, Image 2, Image 3 always refer to the three images shown in the most recent response. If the user refers to images from a previous response or an earlier round (e.g. "the first image from two rounds ago", "the second image you showed before", "the image before that", "the ones from earlier"):
- Do NOT perform any image search.
- Do NOT call any tool.
- Respond only with: "I can only reference the images from the most recent response. Please refer to Image 1, 2, or 3 from the last set shown."

If the user references more than one shown image at once for any kind of search or filtering — including visual similarity, color, material, garment, season, or any other attribute — (e.g. "similar to image 1 and image 2", "same color as the first and second one", "like image 1 but also image 3", "images with the colors of both image 1 and image 2"):
- Do NOT perform any image search.
- Do NOT call any tool.
- Respond only with: "I can only use one image as a reference at a time. Which one would you like me to use — Image 1, 2, or 3?"

Remember:
- Do not invent information or draw from knowledge outside of the provided context.
- If asked about topics unrelated to Ferré or fashion, politely redirect the conversation.
- Be concise while covering all relevant information.
"""

def _make_chroma_client():
    return chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT, ssl=CHROMADB_SSL)

# Connect to ChromaDB (optional for local non-RAG runs)
try:
    chroma_client = _make_chroma_client()
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



def _build_retrieved_images_content(selected_images: Optional[List[Dict[str, str]]]) -> Optional[Content]:
    if not selected_images:
        return None

    design_images_dir = resolve_design_images_dir()
    if design_images_dir is None:
        return None

    parts: List[Part] = [
        Part.from_text(
            text=(
                "These are the top archive images retrieved for this query. "
                "Use the actual visuals together with the retrieved text chunks when forming your answer."
            )
        )
    ]

    for image in selected_images[:3]:
        source_path = image.get("source_path")
        if not source_path:
            continue

        # Try GCS first, fall back to local disk
        gcs_result = fetch_image_bytes(source_path)
        if gcs_result:
            image_data, mime_type = gcs_result
            parts.append(Part.from_bytes(data=image_data, mime_type=mime_type))
            continue

        # Local fallback
        relative_path = source_path.removeprefix(DATASET_PREFIX)
        image_path = (design_images_dir / relative_path).resolve() if design_images_dir else None

        if image_path is None:
            continue
        try:
            image_path.relative_to(design_images_dir.resolve())
        except ValueError:
            continue

        if not image_path.exists():
            continue

        mime_type, _ = mimetypes.guess_type(str(image_path))
        with image_path.open("rb") as image_file:
            parts.append(
                Part.from_bytes(
                    data=image_file.read(),
                    mime_type=mime_type or "image/jpeg",
                )
            )

    if len(parts) == 1:
        return None

    return Content(role="user", parts=parts)


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
        response = _get_multimodal_model().get_embeddings(
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
    elif message.get("content"):
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


def extract_image_filters(query: str) -> tuple[str, Optional[dict], Optional[dict], Optional[int], list]:
    """Use Gemini to extract visual filters from a free-text query.

    Returns:
        (refined_search_query, where, where_document, reference_image_index, filter_attributes)
        where                — ChromaDB metadata filter (season_path, year_path)
        where_document       — ChromaDB document filter (tag values via $contains)
        reference_image_index — 1/2/3 if the user referred to a specific shown image, else None
        filter_attributes    — list of attribute names (e.g. ["collection", "year"]) to filter
                               by from the referenced image's metadata; empty list if not applicable
        where and where_document are None when no filters of that type were identified.
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
            return query, None, None, None, []

        args = dict(function_calls[0].args)
        refined_query = args.get("search_query") or query
        where, where_document = _build_chroma_filters(args)

        raw_ref = args.get("reference_image_index")
        reference_image_index = int(raw_ref) if raw_ref is not None else None

        raw_attrs = args.get("filter_attributes") or []
        filter_attributes = [str(a) for a in raw_attrs] if raw_attrs else []

        print(
            f"Image filters — query: '{refined_query}', where: {where}, "
            f"where_document: {where_document}, ref_image: {reference_image_index}, "
            f"filter_attrs: {filter_attributes}"
        )
        return refined_query, where, where_document, reference_image_index, filter_attributes

    except Exception as e:
        print(f"extract_image_filters failed, falling back to plain query: {e}")
        return query, None, None, None, []


def get_image_embedding_from_chroma(source_path: str) -> Optional[List[float]]:
    """Retrieve the stored multimodal embedding for an image from ChromaDB.

    Used when the user refers to a specific shown image for visual similarity search,
    so we can use its embedding directly instead of generating a text-based one.
    """
    try:
        client = _make_chroma_client()
        collection = client.get_collection(name="images-fashion-show-photos")
        results = collection.get(
            where={"source_path": source_path},
            include=["embeddings"],
        )
        embeddings = results.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            return list(embeddings[0])
    except Exception as e:
        print(f"get_image_embedding_from_chroma failed for '{source_path}': {e}")
    return None


def retrieve_text_chunks(
    session: AgentChatSession,
    message: Dict,
    retrieval_hint: Optional[str] = None,
) -> tuple:
    """
    Steps 1 & 2 of the agentic pipeline: tool selection + ChromaDB retrieval.

    Does NOT modify session.history — call generate_final_answer() next.

    Args:
        retrieval_hint: Optional context prepended to the user message only during
            tool selection (Step 1) to steer RAG towards relevant chunks.
            Not stored in session history.

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

        # Step 1: LLM selects which tool(s) to call.
        # If a retrieval_hint is provided, prepend it to the user content so the
        # LLM can use item metadata to enrich its tool call, without storing
        # the hint in session history.
        if retrieval_hint:
            hint_content = Content(
                role="user",
                parts=[Part.from_text(text=retrieval_hint)] + list(user_content.parts),
            )
            tool_selection_contents = session.history + [hint_content]
        else:
            tool_selection_contents = session.history + [user_content]

        tool_selection_response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=tool_selection_contents,
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
    selected_images: Optional[List[Dict[str, str]]] = None,
    system_instruction: Optional[str] = None,
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
        contents = list(session.history) + [user_content]

        if tool_call_content is not None:
            contents.extend([tool_call_content, function_responses_content])

        # Inject image metadata and bytes as a user-turn Content so they are
        # part of the reasoning context (not the behavioral system instruction)
        # and can be persisted in history for follow-up turns.
        if image_context:
            contents.append(Content(role="user", parts=[Part.from_text(text=image_context)]))

        retrieved_images_content = _build_retrieved_images_content(selected_images)
        if retrieved_images_content is not None:
            contents.append(retrieved_images_content)

        system = system_instruction if system_instruction is not None else SYSTEM_INSTRUCTION
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[ferre_archive_tool] if tool_call_content is not None else [],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="none")
            ) if tool_call_content is not None else None,
        )

        final_response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=contents,
            config=config,
        )
        final_text = final_response.text

        # Append the exchange to history.
        # Image bytes (retrieved_images_content) are intentionally NOT stored —
        # they are large and not needed in past turns. The text image_context is
        # sufficient for follow-up questions about previously shown images.
        # A sliding window caps history at ~10 turns to bound memory usage.
        session.history.append(user_content)
        if tool_call_content is not None:
            session.history.append(tool_call_content)
            session.history.append(function_responses_content)
        if image_context:
            session.history.append(Content(role="user", parts=[Part.from_text(text=image_context)]))
        session.history.append(
            Content(role="model", parts=[Part.from_text(text=final_text)])
        )

        # Sliding window: keep at most 50 Content items (~10 turns at ~5 items/turn)
        MAX_HISTORY_ITEMS = 50
        if len(session.history) > MAX_HISTORY_ITEMS:
            session.history = session.history[-MAX_HISTORY_ITEMS:]

        return final_text, sources

    except Exception as e:
        print(f"Error generating final answer: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}",
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
