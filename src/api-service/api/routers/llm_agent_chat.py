import os
import random
from fastapi import APIRouter, Header, Query, Body, HTTPException, Request
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
import uuid
import time
from datetime import datetime
import mimetypes
from pathlib import Path
from api.utils.agent_orchestrator import (
    chat_sessions,
    create_chat_session,
    generate_chat_response,
    rebuild_chat_session,
)
from api.utils.chat_utils import ChatHistoryManager, ChatMessage


# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}

# Define Router
router = APIRouter()

# Initialize chat history manager and sessions
chat_manager = ChatHistoryManager(model="llm-agent")


def _safe_parents(path: Path, n: int):
    """Return path.parents[n] or None if out of range."""
    return path.parents[n] if n < len(path.parents) else None


def _find_designs_dir() -> Optional[Path]:
    """Resolve the ferre-designs dataset directory."""
    _base = Path(__file__).resolve()
    candidates = [
        c for c in [
            Path("/input-datasets/ferre-designs"),
            _safe_parents(_base, 4) and _safe_parents(_base, 4) / "input-datasets" / "ferre-designs",
            _safe_parents(_base, 3) and _safe_parents(_base, 3) / "input-datasets" / "ferre-designs",
            _safe_parents(_base, 1) and _safe_parents(_base, 1) / "input-datasets" / "ferre-designs",
            Path.cwd() / "input-datasets" / "ferre-designs",
        ] if c
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _get_first_season_images() -> list[Path]:
    """Return image paths from first season folder."""
    designs_dir = _find_designs_dir()
    if not designs_dir:
        return []

    season_dirs = sorted([path for path in designs_dir.glob("*/*") if path.is_dir()])
    if not season_dirs:
        return []

    first_season_dir = season_dirs[0]
    return sorted([
        file_path
        for file_path in first_season_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ])


def _get_sample_images(request: Request, count: int = 3) -> List[Dict[str, str]]:
    """Sample random images from first season and build URLs."""
    designs_dir = _find_designs_dir()
    if not designs_dir:
        return []

    image_files = _get_first_season_images()
    if not image_files:
        return []

    selected = random.sample(image_files, k=min(count, len(image_files)))
    images = []
    for image_path in selected:
        try:
            relative_path = image_path.relative_to(designs_dir).as_posix()
            image_url = str(request.base_url).rstrip("/") + f"/design-images/{relative_path}"
            images.append({
                "source_path": relative_path,
                "image_url": image_url,
            })
        except Exception:
            continue
    return images


@router.get("/chats")
async def get_chats(
    x_session_id: str = Header(None, alias="X-Session-ID"), limit: Optional[int] = None
):
    """Get all chats, optionally limited to a specific number"""
    print("x_session_id:", x_session_id)
    # Generate a default session ID if none provided
    if not x_session_id:
        x_session_id = "default-session"
    return chat_manager.get_recent_chats(x_session_id, limit)


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str, x_session_id: str = Header(None, alias="X-Session-ID")
):
    """Get a specific chat by ID"""
    print("x_session_id:", x_session_id)
    # Generate a default session ID if none provided
    if not x_session_id:
        x_session_id = "default-session"
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/chats")
async def start_chat_with_llm(
    request: Request,
    message: ChatMessage, x_session_id: str = Header(None, alias="X-Session-ID")
):
    message_dict = message.model_dump()
    print("content:", message_dict["content"])
    print("x_session_id:", x_session_id)

    # Generate a default session ID if none provided
    if not x_session_id:
        x_session_id = "default-session"

    """Start a new chat with an initial message"""
    chat_id = str(uuid.uuid4())
    current_time = int(time.time())

    # Create a new agent chat session
    chat_session = create_chat_session()
    chat_sessions[chat_id] = chat_session

    # Add ID and role to the user message
    message_dict["message_id"] = str(uuid.uuid4())
    message_dict["role"] = "user"

    # Generate response
    assistant_response_text, response_sources = generate_chat_response(chat_session, message_dict)

    # Create chat response
    title = message_dict.get("content")
    if not title:
        title = "Image chat"
    title = title[:50] + "..."

    # Get sample images
    images = _get_sample_images(request, count=3)

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

    # Save chat
    chat_manager.save_chat(chat_response, x_session_id)
    return chat_response


@router.post("/chats/{chat_id}")
async def continue_chat_with_llm(
    request: Request,
    chat_id: str,
    message: ChatMessage,
    x_session_id: str = Header(None, alias="X-Session-ID"),
):
    message_dict = message.model_dump()
    print("content:", message_dict["content"])
    print("x_session_id:", x_session_id)

    # Generate a default session ID if none provided
    if not x_session_id:
        x_session_id = "default-session"

    """Add a message to an existing chat"""
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Get or rebuild agent chat session
    chat_session = chat_sessions.get(chat_id)
    if not chat_session:
        chat_session = rebuild_chat_session(chat["messages"], history_dir=chat_manager.history_dir)
        chat_sessions[chat_id] = chat_session

    # Update timestamp
    current_time = int(time.time())
    chat["dts"] = current_time

    # Add message ID and role
    message_dict["message_id"] = str(uuid.uuid4())
    message_dict["role"] = "user"

    # Generate response
    assistant_response_text, response_sources = generate_chat_response(chat_session, message_dict)

    # Add messages
    chat["messages"].append(message_dict)
    chat["messages"].append(
        {
            "message_id": str(uuid.uuid4()),
            "role": "assistant",
            "content": assistant_response_text,
        }
    )

    # Get sample images (refresh on each query)
    images = _get_sample_images(request, count=3)
    chat["images"] = images
    chat["sources"] = response_sources

    # Save updated chat
    chat_manager.save_chat(chat, x_session_id)
    return chat


@router.get("/images/{chat_id}/{message_id}.png")
async def get_chat_image(chat_id: str, message_id: str):
    """
    Serve an image from the chat history.

    Args:
        chat_id: The chat ID
        message_id: The message ID

    Returns:
        FileResponse: The image file with appropriate content type
    """
    try:
        # Construct the image path
        image_path = os.path.join(chat_manager.images_dir, chat_id, f"{message_id}.png")

        # Verify the path exists and is within the images directory
        image_path = Path(image_path).resolve()
        images_dir = Path(chat_manager.images_dir).resolve()

        # Security check: ensure the requested file is within the images directory
        if not str(image_path).startswith(str(images_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(image_path))
        if not content_type:
            content_type = "application/octet-stream"

        return FileResponse(path=image_path, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")
