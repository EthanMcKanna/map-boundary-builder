from __future__ import annotations

import hashlib
import os
from importlib import resources

from . import __version__

PIPELINE_VERSION_ENV = "MAP_BOUNDARY_PIPELINE_VERSION"
PIPELINE_VERSION_FILES = (
    "extract.py",
    "geocoder.py",
    "geocoder_seed.json",
    "georeference.py",
    "georef_transform.py",
    "geojson.py",
    "image_io.py",
    "ocr.py",
    "osm_places.py",
    "osm_roads.py",
    "pipeline_version.py",
    "runner.py",
)

_PIPELINE_VERSION: str | None = None


def get_pipeline_version() -> str:
    configured = os.environ.get(PIPELINE_VERSION_ENV)
    if configured:
        return configured

    global _PIPELINE_VERSION
    if _PIPELINE_VERSION is not None:
        return _PIPELINE_VERSION

    digest = hashlib.sha256()
    digest.update(__version__.encode("utf-8"))
    for filename in PIPELINE_VERSION_FILES:
        source = resources.files("map_boundary_builder").joinpath(filename)
        digest.update(filename.encode("utf-8"))
        digest.update(source.read_bytes())
    _PIPELINE_VERSION = f"pipeline-{digest.hexdigest()[:16]}"
    return _PIPELINE_VERSION
