import json
import os
import unicodedata
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import String, case, cast, func, or_, text as sa_text
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
    from api.utils.gcs_utils import build_proxy_url
    proxy = build_proxy_url(str(request.base_url), source_path)
    if proxy:
        return proxy
    relative = unicodedata.normalize("NFC", source_path.removeprefix(DATASET_PREFIX))
    return str(request.base_url).rstrip("/") + f"/design-images/{quote(relative, safe='/')}"


@app.get("/")
async def get_index():
    return {"message": "Welcome to Gianfranco Ferré research assistant"}


@app.get("/archive/filter-options")
async def get_filter_options(session: Session = Depends(get_session)):
    seasons = [
        row[0]
        for row in session.execute(
            sa_text(
                "SELECT DISTINCT season_path FROM fashionitem "
                "WHERE season_path IS NOT NULL ORDER BY season_path"
            )
        ).fetchall()
    ]

    garments = [
        row[0]
        for row in session.execute(
            sa_text(
                "SELECT DISTINCT value FROM fashionitem, "
                "json_array_elements_text(garments_tags) AS value "
                "WHERE value IS NOT NULL AND value <> '' ORDER BY value"
            )
        ).fetchall()
    ]

    colors = [
        row[0]
        for row in session.execute(
            sa_text(
                "SELECT DISTINCT value FROM fashionitem, "
                "json_array_elements_text(colors_tags) AS value "
                "WHERE value IS NOT NULL AND value <> '' ORDER BY value"
            )
        ).fetchall()
    ]

    materials = [
        row[0]
        for row in session.execute(
            sa_text(
                "SELECT DISTINCT value FROM fashionitem, "
                "json_array_elements_text(material_tags) AS value "
                "WHERE value IS NOT NULL AND value <> '' ORDER BY value"
            )
        ).fetchall()
    ]

    return {
        "seasons": seasons,
        "garments": garments,
        "colors": colors,
        "materials": materials,
    }


@app.get("/archive/landing-feed")
async def get_landing_feed(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=100),
    season_path: str | None = Query(default=None),
    garments: list[str] = Query(default=[]),
    colors: list[str] = Query(default=[]),
    materials: list[str] = Query(default=[]),
    session: Session = Depends(get_session),
):
    # Build filter conditions (OR within each category, AND across categories)
    conditions = []
    if season_path:
        conditions.append(FashionItem.season_path == season_path)
    if garments:
        conditions.append(or_(*[
            sa_text("fashionitem.garments_tags::jsonb @> CAST(:g_{i} AS jsonb)".replace("{i}", str(i)))
            .bindparams(**{f"g_{i}": json.dumps([g])})
            for i, g in enumerate(garments)
        ]))
    if colors:
        conditions.append(or_(*[
            sa_text("fashionitem.colors_tags::jsonb @> CAST(:c_{i} AS jsonb)".replace("{i}", str(i)))
            .bindparams(**{f"c_{i}": json.dumps([c])})
            for i, c in enumerate(colors)
        ]))
    if materials:
        conditions.append(or_(*[
            sa_text("fashionitem.material_tags::jsonb @> CAST(:m_{i} AS jsonb)".replace("{i}", str(i)))
            .bindparams(**{f"m_{i}": json.dumps([m])})
            for i, m in enumerate(materials)
        ]))

    # Deduplication: for PDF-sourced items, show one image per cluster_id.
    # For LLM-clustered items (pdf_available="missing"), cluster may be wrong
    # so every image gets its own partition key (id::text) and all are shown.
    partition_key = case(
        (FashionItem.pdf_available == "missing", cast(FashionItem.id, String())),
        else_=FashionItem.cluster_id,
    )
    row_num = func.row_number().over(
        partition_by=partition_key,
        order_by=FashionItem.source_path,
    ).label("rn")

    inner = select(FashionItem.id, row_num)
    for cond in conditions:
        inner = inner.where(cond)
    subq = inner.subquery("deduped")

    rep_ids = select(subq.c.id).where(subq.c.rn == 1)

    items = session.exec(
        select(FashionItem)
        .where(FashionItem.id.in_(rep_ids))
        .order_by(func.random())
        .offset(offset)
        .limit(limit)
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


@app.get("/archive/item-cluster")
async def get_item_cluster(
    request: Request,
    source_path: str = Query(...),
    session: Session = Depends(get_session),
):
    item = session.exec(
        select(FashionItem).where(FashionItem.source_path == source_path)
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # LLM-clustered items (pdf_available="missing") may group different garments together —
    # only show multiple angles when items come from the same PDF (available/empty).
    if item.pdf_available == "missing":
        cluster_items = [item]
    else:
        cluster_items = session.exec(
            select(FashionItem)
            .where(FashionItem.cluster_id == item.cluster_id)
            .where(FashionItem.pdf_available != "missing")
            .order_by(FashionItem.source_path)
        ).all() or [item]

    return {
        "cluster_id": item.cluster_id,
        "items": [
            {
                "source_path": ci.source_path,
                "image_url": _build_image_url(request, ci.source_path),
            }
            for ci in cluster_items
        ],
    }


@app.get("/archive/image")
async def proxy_image(source_path: str = Query(...)):
    """Proxy an image from GCS, streaming bytes back to the client."""
    from api.utils.gcs_utils import GCS_BUCKET, _get_gcs_client, source_path_to_gcs_object
    if not GCS_BUCKET:
        raise HTTPException(status_code=404, detail="GCS not configured")
    try:
        client = _get_gcs_client()
        blob = client.bucket(GCS_BUCKET).blob(source_path_to_gcs_object(source_path))
        data = blob.download_as_bytes()
        content_type = blob.content_type or "image/jpeg"
        return StreamingResponse(iter([data]), media_type=content_type)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {e}")


app.include_router(llm_agent_chat.router, prefix="/llm-agent")
