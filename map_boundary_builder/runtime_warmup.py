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

        extraction_started = time.perf_counter()
        extraction_profile = warm_extraction_runtime()
        profile["extraction_warmed"] = True
        profile["extraction_style"] = extraction_profile["style"]
        profile["extraction_contour_count"] = extraction_profile["contour_count"]
        profile["extraction_s"] = elapsed_seconds(extraction_started)

        ocr_started = time.perf_counter()
        from .ocr import warm_rapidocr_runtime

        profile["rapidocr_inference_warmed"] = warm_rapidocr_runtime()
        profile["rapidocr_s"] = elapsed_seconds(ocr_started)
        profile["status"] = "ok"
    except Exception as exc:
        profile["status"] = "error"
        profile["error"] = str(exc)
    profile["total_s"] = elapsed_seconds(started)
    return profile


def warm_extraction_runtime() -> dict[str, Any]:
    import numpy as np

    from .extract import extract_service_area_from_rgb

    rgb = np.full((256, 256, 3), 255, dtype=np.uint8)
    rgb[64:192, 64:192] = (40, 150, 230)
    result = extract_service_area_from_rgb(rgb)
    return {
        "style": result.style,
        "coverage_ratio": round(result.coverage_ratio, 6),
        "contour_count": result.contour_count,
        "confidence": round(result.confidence, 6),
    }


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started), 6)
