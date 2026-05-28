from __future__ import annotations

import time
from typing import Any
from urllib.parse import parse_qs


def parse_warm_targets(query: str) -> set[str]:
    raw_targets = parse_qs(query).get("warm", [])
    targets: set[str] = set()
    for raw in raw_targets:
        for value in raw.split(","):
            value = value.strip().lower()
            if value:
                targets.add(value)
    if "1" in targets or "true" in targets or "all" in targets:
        targets.add("generation")
    return targets & {"generation", "ocr"}


def warm_generation_runtime(targets: set[str]) -> dict[str, Any]:
    started = time.perf_counter()
    warmed: dict[str, Any] = {"warm_targets": sorted(targets)}
    errors: dict[str, str] = {}
    if "generation" in targets:
        try:
            from map_boundary_builder import extract as _extract  # noqa: F401
            from map_boundary_builder import georeference as _georeference  # noqa: F401
            from map_boundary_builder import runner as _runner  # noqa: F401

            warmed["generation"] = True
        except Exception as exc:
            errors["generation"] = str(exc)
    if targets & {"generation", "ocr"}:
        try:
            from map_boundary_builder.ocr import rapidocr_engine

            rapidocr_engine()
            warmed["ocr"] = True
        except Exception as exc:
            errors["ocr"] = str(exc)
    warmed["warm_elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    if errors:
        warmed["warm_errors"] = errors
    return warmed
