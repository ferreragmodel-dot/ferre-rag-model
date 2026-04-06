import os
import uuid
import time
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Header, Request, HTTPException
from fastapi.responses import FileResponse
import chromadb

from api.utils.agent_orchestrator import (
    chat_sessions,
    create_chat_session,
    generate_chat_response,
    rebuild_chat_session,
    generate_image_query_embedding,
    CHROMADB_HOST,
    CHROMADB_PORT,
)
from api.utils.chat_utils import ChatHistoryManager, ChatMessage

router = APIRouter()

chat_manager = ChatHistoryManager(model="llm-agent")

# Must match the prefix used in source_path (DB and ChromaDB metadata)
DATASET_PREFIX = "Dataset DataShack 2026/"


def _search_similar_images(query: str, request: Request, count: int = 3) -> List[Dict[str, str]]:
    chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = chroma_client.get_collection(name="images-fashion-show-photos")
    query_embedding = generate_image_query_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=count)

    images = []
    if results and results.get("metadatas") and results["metadatas"][0]:
        for metadata in results["metadatas"][0]:
            image_path = metadata.get("path")
            if not image_path:
                continue
            relative = image_path.removeprefix(DATASET_PREFIX)
            image_url = str(request.base_url).rstrip("/") + f"/design-images/{relative}"
            images.append({"source_path": image_path, "image_url": image_url})
    return images


@router.get("/chats")
async def get_chats(
    x_session_id: str = Header(None, alias="X-Session-ID"),
    limit: Optional[int] = None,
):
    if not x_session_id:
        x_session_id = "default-session"
    return chat_manager.get_recent_chats(x_session_id, limit)


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    x_session_id: str = Header(None, alias="X-Session-ID"),
):
    if not x_session_id:
        x_session_id = "default-session"
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/chats")
async def start_chat_with_llm(
    request: Request,
    message: ChatMessage,
    x_session_id: str = Header(None, alias="X-Session-ID"),
):
    if not x_session_id:
        x_session_id = "default-session"

    message_dict = message.model_dump()
    chat_id = str(uuid.uuid4())
    current_time = int(time.time())

    chat_session = create_chat_session()
    chat_sessions[chat_id] = chat_session

    message_dict["message_id"] = str(uuid.uuid4())
    message_dict["role"] = "user"

    assistant_response_text, response_sources = generate_chat_response(chat_session, message_dict)

    title = (message_dict.get("content") or "Image chat")[:50] + "..."
    images = _search_similar_images(message_dict.get("content", ""), request, count=3)

    chat_response = {
        "chat_id": chat_id,
        "title": title,
        "dts": current_time,
        "messages": [
            message_dict,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": assistant_response_text,
            },
        ],
        "images": images,
        "sources": response_sources,
    }

    chat_manager.save_chat(chat_response, x_session_id)
    return chat_response


@router.post("/chats/{chat_id}")
async def continue_chat_with_llm(
    request: Request,
    chat_id: str,
    message: ChatMessage,
    x_session_id: str = Header(None, alias="X-Session-ID"),
):
    if not x_session_id:
        x_session_id = "default-session"

    message_dict = message.model_dump()
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat_session = chat_sessions.get(chat_id)
    if not chat_session:
        chat_session = rebuild_chat_session(chat["messages"], history_dir=chat_manager.history_dir)
        chat_sessions[chat_id] = chat_session

    message_dict["message_id"] = str(uuid.uuid4())
    message_dict["role"] = "user"

    assistant_response_text, response_sources = generate_chat_response(chat_session, message_dict)

    chat["dts"] = int(time.time())
    chat["messages"].append(message_dict)
    chat["messages"].append({
        "message_id": str(uuid.uuid4()),
        "role": "assistant",
        "content": assistant_response_text,
    })
    chat["images"] = _search_similar_images(message_dict.get("content", ""), request, count=3)
    chat["sources"] = response_sources

    chat_manager.save_chat(chat, x_session_id)
    return chat


@router.get("/images/{chat_id}/{message_id}.png")
async def get_chat_image(chat_id: str, message_id: str):
    image_path = Path(os.path.join(chat_manager.images_dir, chat_id, f"{message_id}.png")).resolve()
    images_dir = Path(chat_manager.images_dir).resolve()

    if not str(image_path).startswith(str(images_dir)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    content_type, _ = mimetypes.guess_type(str(image_path))
    return FileResponse(path=image_path, media_type=content_type or "application/octet-stream")
