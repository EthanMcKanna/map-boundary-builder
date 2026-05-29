from __future__ import annotations

import time
from typing import Any


def should_prewarm_generation_runtime(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "ocr", "generation", "runtime"}


def prewarm_generation_runtime() -> dict[str, Any]:
    started = time.perf_counter()
    profile: dict[str, Any] = {}
    try:
        catalog_started = time.perf_counter()
        from .catalog_match import load_catalog_entries

        profile["catalog_entries"] = len(load_catalog_entries())
        profile["catalog_s"] = elapsed_seconds(catalog_started)

        seed_started = time.perf_counter()
        from .geocoder import load_geocoder_seed
        from .osm_places import load_osm_places_seed
        from .osm_roads import load_road_points_seed

        profile["geocoder_seed_entries"] = len(load_geocoder_seed())
        profile["osm_place_seed_entries"] = len(load_osm_places_seed())
        profile["road_seed_entries"] = len(load_road_points_seed())
        profile["seed_s"] = elapsed_seconds(seed_started)

        ocr_started = time.perf_counter()
        from .ocr import rapidocr_engine

        rapidocr_engine()
        profile["rapidocr_s"] = elapsed_seconds(ocr_started)
        profile["status"] = "ok"
    except Exception as exc:
        profile["status"] = "error"
        profile["error"] = str(exc)
    profile["total_s"] = elapsed_seconds(started)
    return profile


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started), 6)
