import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware


from fastapi import Depends
from sqlmodel import Session, select

from api.routers import llm_agent_chat
from api.db import create_db_and_tables, get_session
import api.models.fashion_item  # noqa: F401 — registers table with SQLModel metadata
from api.models.fashion_item import FashionItem
from api.seeds.seed import seed

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    seed()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}


def _safe_parents(path: Path, n: int):
    """Return path.parents[n] or None if out of range."""
    return path.parents[n] if n < len(path.parents) else None


def _find_images_dir() -> Path:
    """Resolve the images directory in both local and containerized layouts."""
    _base = Path(__file__).resolve()
    candidates = [
        c for c in [
            Path("/images"),
            _safe_parents(_base, 3) and _safe_parents(_base, 3) / "images",
            _safe_parents(_base, 1) and _safe_parents(_base, 1) / "images",
            Path.cwd() / "images",
        ] if c
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError("Unable to find an images directory for static serving")


def _find_designs_dir() -> Path:
    """Resolve the ferre-designs dataset directory in local/container layouts."""
    _base = Path(__file__).resolve()
    candidates = [
        c for c in [
            Path("/input-datasets/ferre-designs"),
            _safe_parents(_base, 3) and _safe_parents(_base, 3) / "input-datasets" / "ferre-designs",
            _safe_parents(_base, 1) and _safe_parents(_base, 1) / "input-datasets" / "ferre-designs",
            Path.cwd() / "input-datasets" / "ferre-designs",
        ] if c
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError("Unable to find input-datasets/ferre-designs for static serving")


def _collect_design_images(designs_dir: Path) -> list[Path]:
    return [
        file_path
        for file_path in designs_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _build_design_image_url(request: Request, source_path: str) -> str:
    return str(request.base_url).rstrip("/") + f"/design-images/{source_path}"


def _resolve_valid_source_path(source_path: str | None, fallback_path: str) -> str:
    if not source_path:
        return fallback_path

    candidate = (DESIGNS_DIR / source_path).resolve()
    try:
        candidate.relative_to(DESIGNS_DIR.resolve())
    except ValueError:
        return fallback_path

    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
        return candidate.relative_to(DESIGNS_DIR).as_posix()

    return fallback_path


IMAGES_DIR = _find_images_dir()
DESIGNS_DIR = _find_designs_dir()
DESIGN_IMAGE_FILES = _collect_design_images(DESIGNS_DIR)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/design-images", StaticFiles(directory=str(DESIGNS_DIR)), name="design-images")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to Gianfranco Ferré research assistant"}


@app.get("/archive/landing-feed")
async def get_landing_feed(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=100),
):
    """Temporary paginated feed that returns random design images."""
    if not DESIGN_IMAGE_FILES:
        raise HTTPException(status_code=500, detail="No design images found in input-datasets/ferre-designs")

    items = []
    selected_images = random.choices(DESIGN_IMAGE_FILES, k=limit)

    for index in range(offset, offset + limit):
        selected_path = selected_images[index - offset]
        relative_path = selected_path.relative_to(DESIGNS_DIR).as_posix()
        image_url = str(request.base_url).rstrip("/") + f"/design-images/{relative_path}"

        items.append(
            {
                "id": f"ferre-{index}",
                "title": selected_path.stem.replace("_", " "),
                "image_url": image_url,
                "source_path": relative_path,
            }
        )

    next_offset = offset + limit
    return {
        "items": items,
        "pagination": {
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "has_more": True,
        },
    }


@app.get("/archive/item-detail")
async def get_archive_item_detail(
    request: Request,
    source_path: str | None = Query(default=None),
    session: Session = Depends(get_session),
):
    """Return metadata for a fashion item by source_path, falling back to first DB item."""
    item: FashionItem | None = None

    if source_path:
        item = session.exec(
            select(FashionItem).where(FashionItem.source_path == source_path)
        ).first()

    if item is None:
        item = session.exec(select(FashionItem)).first()

    if item is None:
        raise HTTPException(status_code=404, detail="No fashion items found in database")

    valid_source_path = _resolve_valid_source_path(source_path, item.source_path)
    image_url = _build_design_image_url(request, valid_source_path)

    return {
        "id": valid_source_path,
        "image_url": image_url,
        "metadata": item.model_dump(),
    }


# Additional routers here
app.include_router(llm_agent_chat.router, prefix="/llm-agent")