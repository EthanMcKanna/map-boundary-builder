from __future__ import annotations

import json
import mimetypes
from importlib import resources

from .pipeline_version import get_pipeline_version

PIPELINE_VERSION_PLACEHOLDER = b'"__MAP_BOUNDARY_PIPELINE_VERSION__"'


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
    return data, mime
