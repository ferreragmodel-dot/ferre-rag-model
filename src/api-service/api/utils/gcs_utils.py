"""
GCS utilities for image byte fetching and proxy URL generation.

GCS_BUCKET env var must be set. If not set, functions return None and callers
fall back to local static file serving.
"""
import os
import mimetypes
from functools import lru_cache
from typing import Optional

DATASET_PREFIX = "Dataset DataShack 2026/"
GCS_BUCKET = os.environ.get("GCS_BUCKET")


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

    return storage.Client(project=project, credentials=creds)


def source_path_to_gcs_object(source_path: str) -> str:
    """Strip the local dataset prefix to get the GCS object name."""
    return source_path.removeprefix(DATASET_PREFIX)


def build_signed_url(source_path: str) -> Optional[str]:
    """
    Returns None — signed URLs require a service account key.
    Use the /archive/image proxy endpoint instead.
    """
    return None


def build_proxy_url(base_url: str, source_path: str) -> Optional[str]:
    """Return a backend proxy URL for the given source_path, or None if GCS not configured."""
    if not GCS_BUCKET:
        return None
    from urllib.parse import urlencode
    return base_url.rstrip("/") + "/archive/image?" + urlencode({"source_path": source_path})


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
