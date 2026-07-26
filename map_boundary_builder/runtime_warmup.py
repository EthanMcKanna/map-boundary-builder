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
        seed_started = time.perf_counter()
        from .geocoder import load_geocoder_seed
        from .osm_places import load_osm_places_seed
        from .osm_roads import load_road_points_seed, seeded_road_points_source_digest

        profile["geocoder_seed_entries"] = len(load_geocoder_seed())
        profile["osm_place_seed_entries"] = len(load_osm_places_seed())
        road_seed = load_road_points_seed()
        profile["road_seed_entries"] = len(road_seed)
        profile["road_seed_digest_entries"] = sum(
            1 for key in road_seed if seeded_road_points_source_digest(key) is not None
        )
        profile["seed_s"] = elapsed_seconds(seed_started)

        segment_started = time.perf_counter()
        segment_profile = warm_segmentation_runtime()
        profile["segmentation_warmed"] = True
        profile["segmentation_engine"] = segment_profile["engine"]
        profile["segmentation_contour_count"] = segment_profile["contour_count"]
        profile["segmentation_s"] = elapsed_seconds(segment_started)

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


def warm_segmentation_runtime() -> dict[str, Any]:
    import numpy as np

    from .segment import segment_image

    # A textured basemap with a translucent-style blue fill sub-region, so the
    # warmup loads the ONNX session and exercises the same threshold, repair,
    # and polygonization code a real screenshot hits.
    rgb = np.full((256, 256, 3), 236, dtype=np.uint8)
    rgb[::8, :] = (212, 212, 212)
    rgb[:, ::8] = (212, 212, 212)
    fill = np.zeros((256, 256), dtype=bool)
    fill[64:192, 64:192] = True
    blended = rgb.astype(np.float32)
    blended[fill] = blended[fill] * 0.45 + np.array([40, 150, 230], dtype=np.float32) * 0.55
    rgb = np.clip(blended, 0, 255).astype(np.uint8)
    result = segment_image(rgb)
    diagnostics = result.diagnostics or {}
    return {
        "engine": diagnostics.get("segmentation_engine", result.style),
        "coverage_ratio": round(result.coverage_ratio, 6),
        "contour_count": result.contour_count,
        "confidence": round(result.confidence, 6),
    }


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started), 6)
