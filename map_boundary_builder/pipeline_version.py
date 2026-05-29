from __future__ import annotations

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from importlib import resources

from . import __version__

PIPELINE_VERSION_ENV = "MAP_BOUNDARY_PIPELINE_VERSION"
PIPELINE_VERSION_PACKAGES = (
    "numpy",
    "onnxruntime",
    "opencv-python-headless",
    "pillow",
    "rapidocr-onnxruntime",
    "shapely",
)
PIPELINE_VERSION_FILES = (
    "catalog_match.py",
    "extract.py",
    "geocoder.py",
    "geocoder_seed.json",
    "georeference.py",
    "georef_transform.py",
    "geojson.py",
    "image_io.py",
    "network_policy.py",
    "ocr.py",
    "osm_places.py",
    "osm_places_seed.json.gz",
    "osm_road_points_seed.npz",
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
    for filename, source in pipeline_version_sources():
        digest.update(filename.encode("utf-8"))
        digest.update(source.read_bytes())
    for package, package_version in pipeline_version_dependency_versions():
        digest.update(package.encode("utf-8"))
        digest.update(package_version.encode("utf-8"))
    _PIPELINE_VERSION = f"pipeline-{digest.hexdigest()[:16]}"
    return _PIPELINE_VERSION


def pipeline_version_sources():
    package_root = resources.files("map_boundary_builder")
    for filename in PIPELINE_VERSION_FILES:
        yield filename, package_root.joinpath(filename)
    catalog_dir = package_root.joinpath("service_area_catalog")
    for source in sorted(catalog_dir.iterdir(), key=lambda item: item.name):
        if source.name.endswith(".json"):
            yield f"service_area_catalog/{source.name}", source


def pipeline_version_dependency_versions():
    for package in PIPELINE_VERSION_PACKAGES:
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "missing"
        yield package, package_version
