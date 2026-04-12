import os
import uuid
import time
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, Request, HTTPException
from fastapi.responses import FileResponse
import chromadb
from sqlmodel import Session, select

from api.db import get_session
from api.models.fashion_item import FashionItem
from api.utils.agent_orchestrator import (
    chat_sessions,
    create_chat_session,
    generate_chat_response,
    rebuild_chat_session,
    generate_image_query_embedding,
    extract_image_filters,
    retrieve_text_chunks,
    generate_final_answer,
    CHROMADB_HOST,
    CHROMADB_PORT,
)
from api.utils.chat_utils import ChatHistoryManager, ChatMessage

router = APIRouter()

chat_manager = ChatHistoryManager(model="llm-agent")

# Must match the prefix used in source_path (DB and ChromaDB metadata)
DATASET_PREFIX = "Dataset DataShack 2026/"

# Fetch more image candidates than the final count to enable metadata-based re-ranking
IMAGE_CANDIDATES_COUNT = 15
# Final number of images returned to the frontend
TOP_IMAGES_COUNT = 3
# Minimum filtered results before falling back to unfiltered candidates.
# Higher than TOP_IMAGES_COUNT so re-ranking always has enough candidates to choose from,
# compensating for LLM-generated tags that may be incomplete or inaccurate.
MIN_FILTERED_RESULTS = 5


def _search_similar_images(
    query: str,
    request: Request,
    count: int = 3,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
    min_results: int = 3,
) -> List[Dict[str, str]]:
    chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = chroma_client.get_collection(name="images-fashion-show-photos")
    query_embedding = generate_image_query_embedding(query)

    def _run_query(
        n_results: int,
        where_clause: Optional[dict],
        where_doc_clause: Optional[dict],
    ) -> List[Dict[str, str]]:
        kwargs: dict = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where_clause:
            kwargs["where"] = where_clause
        if where_doc_clause:
            kwargs["where_document"] = where_doc_clause
        results = collection.query(**kwargs)
        images = []
        if results and results.get("metadatas") and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                image_path = metadata.get("source_path")
                if not image_path:
                    continue
                relative = image_path.removeprefix(DATASET_PREFIX)
                image_url = str(request.base_url).rstrip("/") + f"/design-images/{relative}"
                images.append({"source_path": image_path, "image_url": image_url})
        return images

    images = _run_query(count, where, where_document)

    # If filters produced fewer results than the minimum needed, fill the gap
    # with unfiltered results (excluding duplicates already retrieved).
    if (where or where_document) and len(images) < min_results:
        print(f"Filtered search returned {len(images)} results (< {min_results}), adding unfiltered results.")
        existing_paths = {img["source_path"] for img in images}
        unfiltered = _run_query(count, where_clause=None, where_doc_clause=None)
        for img in unfiltered:
            if img["source_path"] not in existing_paths:
                images.append(img)
                existing_paths.add(img["source_path"])
                if len(images) >= count:
                    break

    return images


def _fetch_images_metadata(source_paths: List[str], db: Session) -> Dict[str, Any]:
    """Fetch FashionItem records from Postgres for the given source paths."""
    if not source_paths:
        return {}
    items = db.exec(
        select(FashionItem).where(FashionItem.source_path.in_(source_paths))
    ).all()
    return {item.source_path: item for item in items}


def _score_image(item: Any, text_context: str) -> int:
    """Score an image by how many of its tags and description match the text context."""
    text_lower = text_context.lower()
    score = 0
    for tag_list in [
        item.garments_tags,
        item.colors_tags,
        item.material_tags,
        item.patterns_tags,
        item.style_tags,
        item.silhouette_tags,
        item.embellishment_tags,
    ]:
        for tag in (tag_list or []):
            if tag.lower() in text_lower:
                score += 1
    if item.year_path and item.year_path in text_context:
        score += 2
    if item.llm_description:
        desc_words = set(item.llm_description.lower().split())
        context_words = set(text_lower.split())
        score += len(desc_words & context_words) // 5
    return score


def _build_image_context(ranked_items: List[Any]) -> str:
    """Format image metadata as a context string to inject into the final LLM call."""
    if not ranked_items:
        return ""
    lines = [
        "The following archive images are shown to the user alongside your response. "
        "Reference them naturally in your answer when they support your points:\n"
    ]
    for i, item in enumerate(ranked_items, 1):
        tags: List[str] = []
        for tag_list in [item.garments_tags, item.colors_tags, item.material_tags]:
            tags.extend(tag_list or [])
        desc = item.llm_description or "(no description available)"
        lines.append(f"Image {i} \u2014 {item.season_path} ({item.year_path}), {item.collection_line}")
        if tags:
            lines.append(f"  Tags: {', '.join(tags[:10])}")
        lines.append(f"  Description: {desc}")
    return "\n".join(lines)


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
    db: Session = Depends(get_session),
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

    # Step 1+2: Tool selection + ChromaDB text chunk retrieval (parallel with image filters)
    user_content, tool_call_content, function_responses_content, sources = retrieve_text_chunks(
        chat_session, message_dict
    )

    # Step 3: Extract visual filters from query, then run image similarity search
    raw_query = message_dict.get("content", "")
    refined_query, image_where, image_where_document = extract_image_filters(raw_query)
    image_candidates = _search_similar_images(
        refined_query, request, count=IMAGE_CANDIDATES_COUNT,
        where=image_where, where_document=image_where_document,
        min_results=MIN_FILTERED_RESULTS,
    )

    # Step 4: Fetch Postgres metadata for all image candidates
    candidate_paths = [img["source_path"] for img in image_candidates]
    metadata_map = _fetch_images_metadata(candidate_paths, db)

    # Step 5: Re-rank images by tag/description overlap with retrieved text context
    text_context = (
        raw_query + " "
        + " ".join(s.get("excerpt", "") for s in sources)
    )
    ranked_items = sorted(
        metadata_map.values(),
        key=lambda item: _score_image(item, text_context),
        reverse=True,
    )[:TOP_IMAGES_COUNT]

    ranked_paths = {item.source_path for item in ranked_items}
    top_images = [img for img in image_candidates if img["source_path"] in ranked_paths]
    if not top_images:
        top_images = image_candidates[:TOP_IMAGES_COUNT]

    # Step 6: Generate final answer with text chunks + image metadata context
    image_context = _build_image_context(ranked_items) if ranked_items else None
    assistant_response_text, response_sources = generate_final_answer(
        chat_session, user_content, tool_call_content, function_responses_content,
        sources, image_context
    )

    title = (message_dict.get("content") or "Image chat")[:50] + "..."
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
        "images": top_images,
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
    db: Session = Depends(get_session),
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

    # Step 1+2: Tool selection + ChromaDB text chunk retrieval
    user_content, tool_call_content, function_responses_content, sources = retrieve_text_chunks(
        chat_session, message_dict
    )

    # Step 3: Extract visual filters from query, then run image similarity search
    raw_query = message_dict.get("content", "")
    refined_query, image_where, image_where_document = extract_image_filters(raw_query)
    image_candidates = _search_similar_images(
        refined_query, request, count=IMAGE_CANDIDATES_COUNT,
        where=image_where, where_document=image_where_document,
        min_results=MIN_FILTERED_RESULTS,
    )

    # Step 4: Fetch Postgres metadata for all image candidates
    candidate_paths = [img["source_path"] for img in image_candidates]
    metadata_map = _fetch_images_metadata(candidate_paths, db)

    # Step 5: Re-rank images by tag/description overlap with retrieved text context
    text_context = (
        raw_query + " "
        + " ".join(s.get("excerpt", "") for s in sources)
    )
    ranked_items = sorted(
        metadata_map.values(),
        key=lambda item: _score_image(item, text_context),
        reverse=True,
    )[:TOP_IMAGES_COUNT]

    ranked_paths = {item.source_path for item in ranked_items}
    top_images = [img for img in image_candidates if img["source_path"] in ranked_paths]
    if not top_images:
        top_images = image_candidates[:TOP_IMAGES_COUNT]

    # Step 6: Generate final answer with text chunks + image metadata context
    image_context = _build_image_context(ranked_items) if ranked_items else None
    assistant_response_text, response_sources = generate_final_answer(
        chat_session, user_content, tool_call_content, function_responses_content,
        sources, image_context
    )

    chat["dts"] = int(time.time())
    chat["messages"].append(message_dict)
    chat["messages"].append({
        "message_id": str(uuid.uuid4()),
        "role": "assistant",
        "content": assistant_response_text,
    })
    chat["images"] = top_images
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
