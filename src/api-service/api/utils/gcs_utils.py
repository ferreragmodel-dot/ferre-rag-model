"""
GCS utilities for image byte fetching and proxy URL generation.

GCS_BUCKET env var must be set. If not set, functions return None and callers
fall back to local static file serving.
"""
import mimetypes
import os
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

DATASET_PREFIX = "Dataset DataShack 2026/"
GCS_BUCKET = os.environ.get("GCS_BUCKET")


def resolve_design_images_dir() -> Optional[Path]:
    """Return the local path to the design images directory, or None if not found.

    Resolution order:
    1. DESIGN_IMAGES_DIR env var (explicit override)
    2. /design-images (Docker volume mount path)
    3. Repo root / "Dataset DataShack 2026" (local dev fallback)
    """
    env_path = os.environ.get("DESIGN_IMAGES_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    docker_path = Path("/design-images")
    if docker_path.exists():
        return docker_path

    try:
        local_repo_dataset = Path(__file__).resolve().parents[3] / "Dataset DataShack 2026"
        if local_repo_dataset.exists():
            return local_repo_dataset
    except IndexError:
        pass

    return None


@lru_cache(maxsize=1)
def _get_gcs_client():
    from google.cloud import storage
    import google.auth

    project = os.environ.get("GCP_PROJECT")

    # In local dev, ADC user credentials don't carry billing context properly.
    # Setting USE_GCLOUD_TOKEN=true in .env uses the gcloud CLI token instead.
    # In production (Cloud Run), google.auth.default() uses the metadata server.
    if os.environ.get("USE_GCLOUD_TOKEN", "").lower() == "true":
        import subprocess
        import google.oauth2.credentials
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
        creds = google.oauth2.credentials.Credentials(token=token)
    else:
        creds, _ = google.auth.default()
        # The ADC file has quota_project_id set, which makes the SDK send
        # x-goog-user-project on every request. GCS then tries to bill
        # ferre-rag-model, which has no billing account → 403. The bucket's
        # owning project has billing and is billed directly when no
        # x-goog-user-project header is present, so we clear it here.
        if hasattr(creds, "with_quota_project"):
            creds = creds.with_quota_project(None)

    return storage.Client(project=project, credentials=creds)


def source_path_to_gcs_object(source_path: str) -> str:
    """Strip the local dataset prefix to get the GCS object name."""
    return source_path.removeprefix(DATASET_PREFIX)


def build_proxy_url(base_url: str, source_path: str) -> Optional[str]:
    """Return a backend proxy URL for the given source_path, or None if GCS not configured."""
    if not GCS_BUCKET:
        return None
    return base_url.rstrip("/") + "/archive/image?" + urlencode({"source_path": source_path})


def build_image_url(base_url: str, source_path: str) -> str:
    """Return the best available URL for an archive image.

    Prefers the GCS proxy endpoint; falls back to the local static-file URL
    when GCS is not configured.
    """
    proxy = build_proxy_url(base_url, source_path)
    if proxy:
        return proxy
    relative = unicodedata.normalize("NFC", source_path.removeprefix(DATASET_PREFIX))
    return base_url.rstrip("/") + f"/design-images/{quote(relative, safe='/')}"


def fetch_image_bytes(source_path: str) -> Optional[tuple[bytes, str]]:
    """
    Download image bytes from GCS.
    Returns (bytes, mime_type) or None if unavailable.
    """
    if not GCS_BUCKET:
        return None
    try:
        client = _get_gcs_client()
        object_name = source_path_to_gcs_object(source_path)
        blob = client.bucket(GCS_BUCKET).blob(object_name)
        data = blob.download_as_bytes()
        mime_type, _ = mimetypes.guess_type(object_name)
        return data, mime_type or "image/jpeg"
    except Exception as e:
        print(f"[gcs] Failed to fetch image bytes for {source_path}: {e}")
        return None
