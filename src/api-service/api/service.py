from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware


from api.routers import llm_agent_chat

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")


def _find_images_dir() -> Path:
    """Resolve the images directory in both local and containerized layouts."""
    candidates = [
        Path(__file__).resolve().parents[3] / "images",
        Path(__file__).resolve().parents[1] / "images",
        Path.cwd() / "images",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError("Unable to find an images directory for static serving")


IMAGES_DIR = _find_images_dir()
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

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


@app.get("/square_root/")
async def square_root(x: float = 1, y: float = 2):
    z = x**2 + y**2
    return z**0.5


@app.get("/archive/landing-feed")
async def get_landing_feed(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=100),
):
    """Temporary paginated feed that repeats a placeholder image."""
    image_url = str(request.base_url).rstrip("/") + "/images/ferre.png"

    items = []
    for index in range(offset, offset + limit):
        items.append(
            {
                "id": f"ferre-{index}",
                "title": "Gianfranco Ferre Archive Fragment",
                "image_url": image_url,
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


# Additional routers here
app.include_router(llm_agent_chat.router, prefix="/llm-agent")