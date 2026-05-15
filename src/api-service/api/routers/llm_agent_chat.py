import os
import uuid
import time
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import chromadb
from sqlmodel import Session, select

from api.db import get_session
from api.models.fashion_item import FashionItem
from api.utils.agent_orchestrator import (
    chat_sessions,
    create_chat_session,
    rebuild_chat_session,
    generate_image_query_embedding,
    extract_image_filters,
    get_image_embedding_from_chroma,
    retrieve_text_chunks,
    generate_final_answer,
    CHROMADB_HOST,
    CHROMADB_PORT,
    CHROMADB_SSL,
)
from api.utils.chat_utils import ChatHistoryManager, ChatMessage
from api.utils.gcs_utils import build_image_url

router = APIRouter()

# ── Chat history managers ─────────────────────────────────────────────────────
chat_manager = ChatHistoryManager(model="llm-agent")
item_chat_manager = ChatHistoryManager(model="llm-agent-item")

# In-memory session store for item chats (separate from general chat_sessions)
item_chat_sessions: Dict[str, Any] = {}

# ── Constants ─────────────────────────────────────────────────────────────────
# DATASET_PREFIX and build_proxy_url are imported from gcs_utils (shared constants)

# Fetch more image candidates than the final count to enable metadata-based re-ranking
IMAGE_CANDIDATES_COUNT = 15
# Final number of images returned to the frontend
TOP_IMAGES_COUNT = 3
# Minimum filtered results before falling back to unfiltered candidates.
# Higher than TOP_IMAGES_COUNT so re-ranking always has enough candidates to choose from,
# compensating for LLM-generated tags that may be incomplete or inaccurate.
MIN_FILTERED_RESULTS = 5


# ── Request models ────────────────────────────────────────────────────────────

class ItemChatMessage(BaseModel):
    """Request body for starting an item-specific chat."""
    source_path: str = Field(..., description="source_path of the archive fashion item")
    content: Optional[str] = Field(None, description="User message text")


# ── Image search helpers (general chat) ───────────────────────────────────────

def _build_image_url(request: Request, source_path: str) -> str:
    return build_image_url(str(request.base_url), source_path)


def _search_similar_images(
    query: str,
    request: Request,
    count: int = TOP_IMAGES_COUNT,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
    min_results: int = MIN_FILTERED_RESULTS,
    query_embedding: Optional[List[float]] = None,
) -> List[Dict[str, str]]:
    """Query ChromaDB for similar images, with fallback to unfiltered results.

    If query_embedding is provided (e.g. retrieved from a shown image), it is used
    directly instead of generating a new text-based embedding from query.
    """
    chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT, ssl=CHROMADB_SSL)
    collection = chroma_client.get_collection(name="images-fashion-show-photos")
    if query_embedding is None:
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
                source_path = metadata.get("source_path")
                if not source_path:
                    continue
                images.append({
                    "source_path": source_path,
                    "image_url": _build_image_url(request, source_path),
                })
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
    """Score an image candidate by tag/description overlap with the text context."""
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
    """Serialize top-ranked image metadata as context for the final LLM call."""
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


# ── Item chat helpers ─────────────────────────────────────────────────────────

def _build_item_system_instruction(item: FashionItem) -> str:
    """Build a system instruction scoped to a specific archive fashion item."""
    tag_fields = [
        ("garments_tags", "Garments"),
        ("colors_tags", "Colors"),
        ("material_tags", "Materials"),
        ("patterns_tags", "Patterns"),
        ("silhouette_tags", "Silhouette"),
        ("style_tags", "Style"),
        ("embellishment_tags", "Embellishments"),
        ("length_tags", "Length"),
        ("neckline_tags", "Neckline"),
        ("sleeve_tags", "Sleeves"),
        ("closure_tags", "Closure"),
    ]
    tag_lines = []
    for field, label in tag_fields:
        values = getattr(item, field) or []
        if values:
            tag_lines.append(f"  {label}: {', '.join(values)}")

    # Prefer archival description over LLM-generated one
    description = item.description or item.llm_description

    metadata_lines = list(filter(None, [
        f"  Season: {item.season_path}" if item.season_path else None,
        f"  Year: {item.year_path}" if item.year_path else None,
        f"  Collection: {item.collection_line}" if item.collection_line else None,
        f"  Look: {item.look}" if item.look else None,
        f"  Description: {description}" if description else None,
        f"  Materials: {item.materials}" if item.materials else None,
        f"  Working process: {item.working_process}" if item.working_process else None,
        f"  Acquisition: {item.acquisition}" if item.acquisition else None,
        f"  Exhibitions: {item.exhibitions}" if item.exhibitions else None,
        f"  Present location: {item.present_location}" if item.present_location else None,
        f"  Condition: {item.condition}" if item.condition else None,
        *tag_lines,
    ]))

    metadata_block = "\n".join(metadata_lines)

    return (
        "You are a specialist assistant scoped EXCLUSIVELY to one specific Gianfranco Ferré archive item. "
        "The item's complete metadata is provided below — treat these values as direct facts.\n\n"
        f"ITEM METADATA:\n{metadata_block}\n\n"
        "DECISION RULE — evaluate this BEFORE consulting any retrieved text chunks:\n"
        "Ask yourself: 'Is this question answerable solely from the item metadata listed above, "
        "or from what is directly visible in this item's image?' "
        "If YES → answer using the metadata and retrieved context that directly supports this item. "
        "If NO → respond with exactly: \"For broader archive questions, please use the general Archive Chat.\" "
        "Do not add anything else to that response.\n\n"
        "IMPORTANT: The presence of relevant retrieved text chunks does NOT make a question on-topic. "
        "If answering requires going beyond the item metadata above — even if the archive contains "
        "relevant information — you must still redirect.\n\n"
        "ALLOWED — answer these:\n"
        "- The item's physical attributes: garments, materials, colors, silhouette, embellishments, patterns\n"
        "- The item's metadata fields: season, year, collection, look, condition, exhibitions, acquisition\n"
        "- How to style or interpret this specific item\n"
        "- What season/collection this item belongs to (based on its season/year metadata)\n\n"
        "NOT ALLOWED — redirect these with the exact phrase above:\n"
        "- Ferré's biography, education, personal history, travels, or life events\n"
        "- Ferré's opinions, philosophy, or views on any topic (color, materials, design, etc.)\n"
        "- Other collections, seasons, or items not represented by this specific item's metadata\n"
        "- Design history, cultural topics, or anything requiring general archive knowledge\n\n"
        "If the user sends a greeting, respond with a brief welcome and invite questions about this item.\n\n"
        "When answering allowed questions:\n"
        "- Use metadata values directly — never write '[see item metadata]' or similar placeholders.\n"
        "- Cite retrieved text with inline citations [1], [2], [3] (maximum 3, only if directly about this item).\n"
        "- Do not invent information beyond what is in the provided metadata and retrieved context."
    )


def _build_item_retrieval_hint(item: FashionItem) -> str:
    """Build a hint to steer RAG tool selection towards the item's context.

    Prepended to the user message only during Step 1 (tool selection) so Gemini
    generates a richer search_content. Not stored in conversation history.
    """
    all_tags: List[str] = []
    for field in ["garments_tags", "colors_tags", "material_tags",
                  "patterns_tags", "style_tags", "embellishment_tags"]:
        all_tags.extend(getattr(item, field) or [])

    parts = [
        f"Item context: {item.season_path or ''} {item.year_path or ''}".strip()
    ]
    if all_tags:
        parts.append(f"Tags: {', '.join(all_tags[:15])}")
    if item.llm_description:
        parts.append(f"Description: {item.llm_description}")

    return " | ".join(parts)


# ── Shared pipeline logic ─────────────────────────────────────────────────────

def _run_general_chat_pipeline(
    chat_session: Any,
    message_dict: dict,
    request: Request,
    db: Session,
    current_images: Optional[List[Dict[str, Any]]] = None,
) -> tuple:
    """Execute Steps 1–6 of the general chat pipeline.

    Args:
        current_images: Images shown to the user in the previous response (if any).
            Used to resolve reference_image_index when the user asks for visually
            similar results to a specific shown image.

    Returns:
        (assistant_text, response_sources, top_images)
    """
    # Step 1+2: Tool selection + ChromaDB text retrieval
    user_content, tool_call_content, function_responses_content, sources = retrieve_text_chunks(
        chat_session, message_dict
    )

    # Step 3: Extract visual filters + ChromaDB image search
    raw_query = message_dict.get("content", "")
    refined_query, image_where, image_where_document, ref_image_index, filter_attributes = extract_image_filters(raw_query)

    # If the user referred to a specific shown image, resolve its data from ChromaDB/Postgres.
    visual_embedding: Optional[List[float]] = None
    if ref_image_index is not None and current_images:
        idx = ref_image_index - 1  # convert 1-based to 0-based
        if 0 <= idx < len(current_images):
            ref_source_path = current_images[idx].get("source_path")
            if ref_source_path:
                # Visual embedding — use for visual/tag-based similarity queries.
                # Skip only when filter_attributes contains exclusively scalar filters
                # (season, year), where the metadata filter alone is sufficient and
                # the visual embedding would undesirably bias towards the referenced image.
                _SCALAR_ONLY_ATTRS = {"season", "year"}
                use_visual = not filter_attributes or bool(
                    set(filter_attributes) - _SCALAR_ONLY_ATTRS
                )
                if use_visual:
                    visual_embedding = get_image_embedding_from_chroma(ref_source_path)
                    if visual_embedding:
                        print(f"Using visual embedding of shown image {ref_image_index} ({ref_source_path})")
                    else:
                        print(f"Visual embedding not found for image {ref_image_index}, falling back to text query")

                # Metadata/tag filters from the referenced image
                if filter_attributes:
                    ref_item = db.exec(
                        select(FashionItem).where(FashionItem.source_path == ref_source_path)
                    ).first()
                    if ref_item:
                        # Scalar metadata → ChromaDB `where` filter
                        extra_where = []
                        if "season" in filter_attributes and ref_item.season_path:
                            extra_where.append({"season_path": ref_item.season_path})
                        if "year" in filter_attributes and ref_item.year_path:
                            extra_where.append({"year_path": ref_item.year_path})

                        if extra_where:
                            if image_where:
                                existing = (
                                    image_where["$and"]
                                    if "$and" in image_where
                                    else [image_where]
                                )
                                image_where = {"$and": extra_where + existing}
                            elif len(extra_where) == 1:
                                image_where = extra_where[0]
                            else:
                                image_where = {"$and": extra_where}
                            print(f"Scalar filters from image {ref_image_index}: {image_where}")

                        # Tag-based → ChromaDB `where_document` ($contains on first tag)
                        extra_doc = []
                        tag_map = [
                            ("color",    ref_item.colors_tags),
                            ("material", ref_item.material_tags),
                            ("garment",  ref_item.garments_tags),
                        ]
                        for attr, tag_list in tag_map:
                            if attr in filter_attributes and tag_list:
                                extra_doc.append({"$contains": tag_list[0]})

                        if extra_doc:
                            if image_where_document:
                                existing_doc = (
                                    image_where_document["$and"]
                                    if "$and" in image_where_document
                                    else [image_where_document]
                                )
                                image_where_document = {"$and": extra_doc + existing_doc}
                            elif len(extra_doc) == 1:
                                image_where_document = extra_doc[0]
                            else:
                                image_where_document = {"$and": extra_doc}
                            print(f"Tag filters from image {ref_image_index}: {image_where_document}")

    image_candidates = _search_similar_images(
        refined_query, request, count=IMAGE_CANDIDATES_COUNT,
        where=image_where, where_document=image_where_document,
        min_results=MIN_FILTERED_RESULTS,
        query_embedding=visual_embedding,
    )

    # Step 4: Fetch Postgres metadata for candidates
    candidate_paths = [img["source_path"] for img in image_candidates]
    metadata_map = _fetch_images_metadata(candidate_paths, db)

    # Step 5: Re-rank by tag/description overlap with retrieved text context
    text_context = raw_query + " " + " ".join(s.get("excerpt", "") for s in sources)
    ranked_items = sorted(
        metadata_map.values(),
        key=lambda item: _score_image(item, text_context),
        reverse=True,
    )[:TOP_IMAGES_COUNT]

    # Build top_images in the same order as ranked_items so that Image 1/2/3 in the
    # LLM context matches the left-to-right order shown in the frontend.
    candidate_url_map = {img["source_path"]: img for img in image_candidates}
    top_images = [
        candidate_url_map[item.source_path]
        for item in ranked_items
        if item.source_path in candidate_url_map
    ]
    if not top_images:
        top_images = image_candidates[:TOP_IMAGES_COUNT]

    # Step 6: Generate final answer with text + image context
    image_context = _build_image_context(ranked_items) if ranked_items else None
    assistant_text, response_sources = generate_final_answer(
        chat_session, user_content, tool_call_content, function_responses_content,
        sources, image_context=image_context, selected_images=top_images,
    )

    return assistant_text, response_sources, top_images


def _run_item_chat_pipeline(
    chat_session: Any,
    message_dict: dict,
    item: FashionItem,
    item_system_instruction: str,
) -> tuple:
    """Execute the RAG pipeline for item-specific chat.

    Differences from general chat:
    - retrieval_hint enriches tool selection with item metadata (Option A)
    - No ChromaDB image search — uses the item's own image
    - item_system_instruction scopes responses to the item

    Returns:
        (assistant_text, response_sources)
    """
    retrieval_hint = _build_item_retrieval_hint(item)

    # Step 1+2: Tool selection (enriched with item context) + ChromaDB text retrieval
    user_content, tool_call_content, function_responses_content, sources = retrieve_text_chunks(
        chat_session, message_dict, retrieval_hint=retrieval_hint,
    )

    # Step 3 (item chat): Always pass the item's own image for multimodal context
    selected_images = [{"source_path": item.source_path}]

    # Step 4: Generate final answer scoped to the item
    assistant_text, response_sources = generate_final_answer(
        chat_session, user_content, tool_call_content, function_responses_content,
        sources,
        image_context=None,              # item metadata already in system instruction
        selected_images=selected_images,
        system_instruction=item_system_instruction,
    )

    return assistant_text, response_sources


# ── General chat routes ───────────────────────────────────────────────────────

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

    chat_id = str(uuid.uuid4())
    current_time = int(time.time())

    chat_session = create_chat_session()
    chat_sessions[chat_id] = chat_session

    message_dict = {
        **message.model_dump(),
        "message_id": str(uuid.uuid4()),
        "role": "user",
    }

    assistant_text, response_sources, top_images = _run_general_chat_pipeline(
        chat_session, message_dict, request, db
    )

    chat_response = {
        "chat_id": chat_id,
        "title": (message_dict.get("content") or "Chat")[:50] + "...",
        "dts": current_time,
        "messages": [
            message_dict,
            {"message_id": str(uuid.uuid4()), "role": "assistant", "content": assistant_text},
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

    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat_session = chat_sessions.get(chat_id)
    if not chat_session:
        chat_session = rebuild_chat_session(chat["messages"], history_dir=chat_manager.history_dir)
        chat_sessions[chat_id] = chat_session

    message_dict = {
        **message.model_dump(),
        "message_id": str(uuid.uuid4()),
        "role": "user",
    }

    assistant_text, response_sources, top_images = _run_general_chat_pipeline(
        chat_session, message_dict, request, db,
        current_images=chat.get("images"),
    )

    chat["dts"] = int(time.time())
    chat["messages"].append(message_dict)
    chat["messages"].append({
        "message_id": str(uuid.uuid4()), "role": "assistant", "content": assistant_text,
    })
    chat["images"] = top_images
    chat["sources"] = response_sources

    chat_manager.save_chat(chat, x_session_id)
    return chat


# ── Item chat routes ──────────────────────────────────────────────────────────

@router.post("/item-chats")
async def start_item_chat(
    request: Request,
    message: ItemChatMessage,
    x_session_id: str = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_session),
):
    if not x_session_id:
        x_session_id = "default-session"

    item = db.exec(
        select(FashionItem).where(FashionItem.source_path == message.source_path)
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Item not found: {message.source_path}")

    chat_id = str(uuid.uuid4())
    current_time = int(time.time())

    chat_session = create_chat_session()
    item_chat_sessions[chat_id] = chat_session

    message_dict = {
        "message_id": str(uuid.uuid4()),
        "role": "user",
        "content": message.content,
        "image": None,
    }

    item_system_instruction = _build_item_system_instruction(item)
    item_image = {
        "source_path": item.source_path,
        "image_url": _build_image_url(request, item.source_path),
    }

    assistant_text, response_sources = _run_item_chat_pipeline(
        chat_session, message_dict, item, item_system_instruction,
    )

    chat_response = {
        "chat_id": chat_id,
        "item_source_path": item.source_path,
        "title": (message.content or "Item chat")[:50] + "...",
        "dts": current_time,
        "messages": [
            message_dict,
            {"message_id": str(uuid.uuid4()), "role": "assistant", "content": assistant_text},
        ],
        "images": [item_image],
        "sources": response_sources,
    }

    item_chat_manager.save_chat(chat_response, x_session_id)
    return chat_response


@router.post("/item-chats/{chat_id}")
async def continue_item_chat(
    request: Request,
    chat_id: str,
    message: ChatMessage,
    x_session_id: str = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_session),
):
    if not x_session_id:
        x_session_id = "default-session"

    chat = item_chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Item chat not found")

    source_path = chat.get("item_source_path")
    item = db.exec(
        select(FashionItem).where(FashionItem.source_path == source_path)
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Item not found: {source_path}")

    chat_session = item_chat_sessions.get(chat_id)
    if not chat_session:
        chat_session = rebuild_chat_session(
            chat["messages"], history_dir=item_chat_manager.history_dir
        )
        item_chat_sessions[chat_id] = chat_session

    message_dict = {
        **message.model_dump(),
        "message_id": str(uuid.uuid4()),
        "role": "user",
    }

    item_system_instruction = _build_item_system_instruction(item)
    item_image = {
        "source_path": item.source_path,
        "image_url": _build_image_url(request, item.source_path),
    }

    assistant_text, response_sources = _run_item_chat_pipeline(
        chat_session, message_dict, item, item_system_instruction,
    )

    chat["dts"] = int(time.time())
    chat["messages"].append(message_dict)
    chat["messages"].append({
        "message_id": str(uuid.uuid4()), "role": "assistant", "content": assistant_text,
    })
    chat["images"] = [item_image]
    chat["sources"] = response_sources

    item_chat_manager.save_chat(chat, x_session_id)
    return chat


# ── Chat image serving ────────────────────────────────────────────────────────

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
