from __future__ import annotations

import hashlib
import json
import mimetypes
from importlib import resources

from .pipeline_version import get_pipeline_version

PIPELINE_VERSION_PLACEHOLDER = b'"__MAP_BOUNDARY_PIPELINE_VERSION__"'
ASSET_VERSION_PLACEHOLDER = b"__MAP_BOUNDARY_ASSET_VERSION__"
WEB_ASSET_VERSION_FILES = (
    "app.css",
    "app.js",
    "boundary-builder-icon.png",
    "openfreemap-boundary.json",
    "openfreemap-dark.json",
)

_WEB_ASSET_VERSION: str | None = None


def web_asset_response(name: str) -> tuple[bytes, str]:
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("invalid asset name")
    asset = resources.files("map_boundary_builder").joinpath("web_assets", name)
    if not asset.is_file():
        raise FileNotFoundError(name)

    data = asset.read_bytes()
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    if name.endswith(".js"):
        mime = "text/javascript; charset=utf-8"
    elif name.endswith(".css"):
        mime = "text/css; charset=utf-8"
    elif name.endswith(".html"):
        mime = "text/html; charset=utf-8"
        data = data.replace(
            PIPELINE_VERSION_PLACEHOLDER,
            json.dumps(get_pipeline_version()).encode("utf-8"),
        )
        data = data.replace(ASSET_VERSION_PLACEHOLDER, web_asset_version().encode("utf-8"))
    return data, mime


def web_asset_version() -> str:
    global _WEB_ASSET_VERSION
    if _WEB_ASSET_VERSION is not None:
        return _WEB_ASSET_VERSION

    digest = hashlib.sha256()
    asset_root = resources.files("map_boundary_builder").joinpath("web_assets")
    for filename in WEB_ASSET_VERSION_FILES:
        asset = asset_root.joinpath(filename)
        digest.update(filename.encode("utf-8"))
        digest.update(asset.read_bytes())
    _WEB_ASSET_VERSION = f"asset-{digest.hexdigest()[:16]}"
    return _WEB_ASSET_VERSION
