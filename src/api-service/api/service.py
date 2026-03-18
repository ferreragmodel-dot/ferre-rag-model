import random
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select
from starlette.middleware.cors import CORSMiddleware

from api.db import create_db_and_tables, get_session
import api.models.fashion_item  # noqa: F401 — registers table with SQLModel metadata
from api.models.fashion_item import FashionItem
from api.routers import llm_agent_chat
from api.seeds.seed import seed
from api.utils.agent_orchestrator import create_chat_session, generate_chat_response

app = FastAPI(title="API Server", description="API Server", version="v1")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    seed()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}


def _safe_parents(path: Path, n: int):
    return path.parents[n] if n < len(path.parents) else None


def _find_dir(*candidates: Path) -> Path:
    for c in candidates:
        if c and c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(f"None of the candidate directories exist: {candidates}")


_base = Path(__file__).resolve()

IMAGES_DIR = _find_dir(
    Path("/images"),
    _safe_parents(_base, 3) and _safe_parents(_base, 3) / "images",
    Path.cwd() / "images",
)

DESIGNS_DIR = _find_dir(
    Path("/input-datasets/ferre-designs"),
    _safe_parents(_base, 3) and _safe_parents(_base, 3) / "input-datasets" / "ferre-designs",
    Path.cwd() / "input-datasets" / "ferre-designs",
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/design-images", StaticFiles(directory=str(DESIGNS_DIR)), name="design-images")


class ConversationRequest(BaseModel):
    message: str


# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_image_url(request: Request, source_path: str) -> str:
    return str(request.base_url).rstrip("/") + f"/design-images/{source_path}"


def _resolve_source_path(source_path: str | None, fallback: str) -> str:
    if not source_path:
        return fallback
    candidate = (DESIGNS_DIR / source_path).resolve()
    try:
        candidate.relative_to(DESIGNS_DIR.resolve())
    except ValueError:
        return fallback
    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
        return candidate.relative_to(DESIGNS_DIR).as_posix()
    return fallback


# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to Gianfranco Ferré research assistant"}


@app.get("/archive/landing-feed")
async def get_landing_feed(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=100),
    session: Session = Depends(get_session),
):
    items = session.exec(select(FashionItem).order_by(func.random()).offset(offset).limit(limit)).all()

    return {
        "items": [
            {
                "id": item.id,
                "title": Path(item.source_path).stem.replace("_", " "),
                "image_url": _build_image_url(request, item.source_path),
                "source_path": item.source_path,
            }
            for item in items
        ],
        "pagination": {
            "offset": offset,
            "limit": limit,
            "next_offset": offset + limit,
            "has_more": len(items) == limit,
        },
    }


@app.get("/archive/item-detail")
async def get_archive_item_detail(
    request: Request,
    source_path: str | None = Query(default=None),
    session: Session = Depends(get_session),
):
    item: FashionItem | None = None

    if source_path:
        item = session.exec(
            select(FashionItem).where(FashionItem.source_path == source_path)
        ).first()

    if item is None:
        item = session.exec(select(FashionItem)).first()

    if item is None:
        raise HTTPException(status_code=404, detail="No fashion items found in database")

    valid_path = _resolve_source_path(source_path, item.source_path)

    return {
        "id": item.id,
        "image_url": _build_image_url(request, valid_path),
        "metadata": item.model_dump(),
    }


@app.post("/archive/conversation")
async def archive_conversation(
    request: Request,
    payload: ConversationRequest,
    session: Session = Depends(get_session),
):
    rag_session = create_chat_session()
    response_text = generate_chat_response(rag_session, {"content": payload.message})

    all_items = session.exec(select(FashionItem)).all()
    sampled = random.sample(all_items, k=min(3, len(all_items)))

    return {
        "query": payload.message,
        "response": response_text,
        "images": [
            {
                "source_path": item.source_path,
                "image_url": _build_image_url(request, item.source_path),
            }
            for item in sampled
        ],
        "tags": [],
    }


# Additional routers here
app.include_router(llm_agent_chat.router, prefix="/llm-agent")
