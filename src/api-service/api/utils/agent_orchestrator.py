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

from api.utils.retrieval_tools import ferre_archive_tool, execute_function_calls

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-2.0-flash-001"
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
CHROMADB_PORT = os.environ["CHROMADB_PORT"]

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in Gianfranco Ferre and fashion archive research. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response using only the information found in the given chunks.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a Ferre archive expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.

Remember:
- You are an expert in Ferre and fashion, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- If asked about topics unrelated to Ferre or fashion, politely redirect the conversation back to archive-related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.

Your goal is to provide accurate, helpful information about Ferre and fashion based solely on the content of the text chunks you receive with each query.
"""

# Connect to ChromaDB (optional for local non-RAG runs)
try:
    chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
except Exception as chroma_error:
    chroma_client = None
    print(f"ChromaDB unavailable; running agent without retrieval tools: {chroma_error}")
COLLECTION_NAME = "recursive-split-collection"

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


def generate_chat_response(session: AgentChatSession, message: Dict) -> str:
    """
    Generate a response using the 3-step agentic RAG pipeline.

    Step 1 - Tool selection: LLM decides which archive search tool to call.
    Step 2 - Execution: run the selected tool(s) against ChromaDB.
    Step 3 - Answer generation: LLM generates a grounded final answer.

    Args:
        session: AgentChatSession holding the conversation history.
        message: Dict with 'content' (text) and optionally 'image' (base64).

    Returns:
        str: The model's final response text.
    """
    try:

        # Build user content parts (text + optional image)
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

        user_content = Content(role="user", parts=user_parts)

        # If ChromaDB is unavailable, skip tool-based retrieval and do direct generation.
        if chroma_client is None:
            final_response = llm_client.models.generate_content(
                model=GENERATIVE_MODEL,
                contents=session.history + [user_content],
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            )
            final_text = final_response.text
            session.history.append(user_content)
            session.history.append(
                Content(role="model", parts=[Part.from_text(text=final_text)])
            )
            return final_text

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
            # Fallback: no tool selected, answer directly
            final_response = llm_client.models.generate_content(
                model=GENERATIVE_MODEL,
                contents=session.history + [user_content],
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            )
            final_text = final_response.text
            session.history.append(user_content)
            session.history.append(
                Content(role="model", parts=[Part.from_text(text=final_text)])
            )
            return final_text

        tool_call_content = tool_selection_response.candidates[0].content

        # Step 2: Execute function calls against ChromaDB
        function_responses = execute_function_calls(
            function_calls, collection, embed_func=generate_query_embedding
        )

        # Step 3: LLM generates final grounded answer
        final_response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=session.history + [
                user_content,
                tool_call_content,
                Content(parts=function_responses),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[ferre_archive_tool],
            ),
        )
        final_text = final_response.text

        # Append the full exchange to history
        session.history.append(user_content)
        session.history.append(tool_call_content)
        session.history.append(Content(parts=function_responses))
        session.history.append(
            Content(role="model", parts=[Part.from_text(text=final_text)])
        )

        return final_text

    except Exception as e:
        print(f"Error generating agent response: {str(e)}")
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
