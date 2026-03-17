import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware


from api.routers import llm_agent_chat

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}

STUB_METADATA = {
    "outfit_id": "266",
    "season": "FW1986-87",
    "season_label": "Fall-Winter",
    "label": "Gianfranco Ferré couture (gold lettering)",
    "acquisition": None,
    "look": "67",
    "file": "266",
    "inventory": "10267",
    "object": "Evening dress",
    "source": "F/W 1986-87 Couture Collection design specs",
    "description": "Long dress with black velvet bodice and slim b&w chalkstripe silk cady skirt. Silk taffeta belt complete with decorative topstitching. Stole in chalkstripe cady with black organza on the underside. A stunning wrap construction with all the flair of a giant ribbon.",
    "exhibitions": "Italian Style since 1945, Victoria and Albert Museum, London, 2015",
    "size": "made to measure",
    "materials": "Black&white chalkstripe print silk cady (Taroni, Como; art. 6422, des. 25, col. 1), silk velvet (Redaelli velvet; art. Prestige super; col. black), silk taffeta, silk organza",
    "present_location": "FGF space c/o Open Care - Milan",
    "remark": "Bordering between sleek feminine severity and slinky sari type silhouette, this dress makes a strong impact on both graphic and chromatic levels. Another point of interest is the revisitation in evening version of a supremely mannish chalkstripe pattern.",
    "bibliography": None,
    "designer": "Gianfranco Ferré",
    "working_process": "Parallel topstitching with 1 cm spacing for bow",
    "condition": "Good",
    "collection": "Couture",
    "year": "1986",
    "match_type": "exact",
    "collection_line": "ALTA-MODA",
    "asset_type": "fashion_show_drawings",
    "source_path": "ALTA-MODA/FW1986-87/fashion_show_drawings/16537.jpg",
}


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


def _resolve_valid_source_path(source_path: str | None) -> str:
    fallback_path = STUB_METADATA["source_path"]
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
):
    """Stub endpoint for image detail popup metadata."""
    valid_source_path = _resolve_valid_source_path(source_path)
    metadata = {**STUB_METADATA, "source_path": valid_source_path}
    image_url = _build_design_image_url(request, valid_source_path)

    return {
        "id": valid_source_path,
        "image_url": image_url,
        "metadata": metadata,
    }


# Additional routers here
app.include_router(llm_agent_chat.router, prefix="/llm-agent")