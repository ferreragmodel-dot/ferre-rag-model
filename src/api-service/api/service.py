import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlmodel import Session, select
from starlette.middleware.cors import CORSMiddleware

from api.db import create_db_and_tables, get_session
import api.models.fashion_item  # noqa: F401 — registers table with SQLModel metadata
from api.models.fashion_item import FashionItem
from api.routers import llm_agent_chat
from api.seeds.seed import seed

app = FastAPI(title="API Server", description="API Server", version="v1")

# The source_path stored in the DB starts with this prefix (e.g.
# "Dataset DataShack 2026/ALTA MODA 1986-87 FW/...").
# The Docker volume mounts the "Dataset DataShack 2026" folder directly at
# /design-images, so this prefix must be stripped when building image URLs.
DATASET_PREFIX = "Dataset DataShack 2026/"


def _resolve_design_images_dir() -> str:
    env_path = os.environ.get("DESIGN_IMAGES_DIR")
    if env_path and Path(env_path).exists():
        return env_path

    docker_path = Path("/design-images")
    if docker_path.exists():
        return str(docker_path)

    # Local dev fallback: repo root contains "Dataset DataShack 2026"
    local_repo_dataset = Path(__file__).resolve().parents[3] / "Dataset DataShack 2026"
    if local_repo_dataset.exists():
        return str(local_repo_dataset)

    # Keep startup informative if neither Docker mount nor local dataset is present.
    raise RuntimeError(
        "Design images directory not found. Set DESIGN_IMAGES_DIR or mount /design-images."
    )


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    seed()


app.mount(
    "/design-images",
    StaticFiles(directory=_resolve_design_images_dir()),
    name="design-images",
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_image_url(request: Request, source_path: str) -> str:
    relative = source_path.removeprefix(DATASET_PREFIX)
    return str(request.base_url).rstrip("/") + f"/design-images/{relative}"


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
    items = session.exec(
        select(FashionItem).order_by(func.random()).offset(offset).limit(limit)
    ).all()

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

    return {
        "id": item.id,
        "image_url": _build_image_url(request, item.source_path),
        "metadata": item.model_dump(),
    }


app.include_router(llm_agent_chat.router, prefix="/llm-agent")
