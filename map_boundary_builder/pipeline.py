"""Unified boundary pipeline.

Linear stages — load → segment → read labels → georeference → export — with
one typed result. Replaces the branching ``runner.build_boundary`` path.

The pipeline never hard-fails just because the city could not be inferred:
it returns ``status="needs_city"`` with the extracted pixel boundary intact
so callers can prompt the user and finish with ``complete_with_city``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .extract import DEFAULT_SIMPLIFY_PX, ExtractionHints, ExtractionResult, load_rgb, write_mask_png, write_overlay_png
from .geojson import feature_collection, target_selection_confidence, write_geojson
from .georeference import (
    GeoreferenceResult,
    georeference_from_city_context,
    georeference_from_labels,
    resolve_city_contexts,
)
from .image_io import normalize_image_for_processing
from .ocr import OcrLabel, extract_ocr_labels_from_rgb
from .segment import segment_image

PipelineStatus = Literal["complete", "needs_city", "failed"]

ProgressCallback = Callable[[str, int, str], None]

NEEDS_CITY_SAMPLE_LABELS = 8


@dataclass(frozen=True)
class PipelineOptions:
    simplify_px: float = DEFAULT_SIMPLIFY_PX
    min_control_points: int = 3
    min_confidence: float = 0.0
    seed_point: tuple[float, float] | None = None
    target_rgb: tuple[int, int, int] | None = None
    model_path: Path | None = None
    model_threshold: float | None = None
    ocr_cache: bool = True
    allow_tesseract_fallback: bool = True

    def extraction_hints(self) -> ExtractionHints:
        return ExtractionHints(seed_point=self.seed_point, target_rgb=self.target_rgb)


@dataclass(frozen=True)
class NeedsCityDetail:
    reason: Literal["no_city_context", "georeference_failed"]
    ocr_label_count: int
    sample_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "ocr_label_count": self.ocr_label_count,
            "sample_labels": list(self.sample_labels),
        }


@dataclass(frozen=True)
class PipelineResult:
    status: PipelineStatus
    reason: str | None
    summary: dict[str, Any]
    geojson: dict[str, Any] | None = None
    extraction: ExtractionResult | None = None
    labels: tuple[OcrLabel, ...] | None = None
    georeference: GeoreferenceResult | None = None
    needs_city: NeedsCityDetail | None = None
    output_path: Path | None = None
    mask_path: Path | None = None
    overlay_path: Path | None = None
    # False for transient failures (network, engine init) that callers must
    # not memoize; True for deterministic outcomes.
    cacheable: bool = True


def run_pipeline(
    image_path: str | Path,
    *,
    city: str | None = None,
    output_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
    options: PipelineOptions | None = None,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    opts = options or PipelineOptions()
    emit = progress or (lambda stage, percent, detail: None)
    started = time.monotonic()
    timings: dict[str, float] = {}

    emit("load", 5, "Reading image")
    stage_start = time.monotonic()
    try:
        normalized_path, rgb = _load(image_path, debug_dir=debug_dir)
    except Exception as exc:
        return _failed_result(
            "invalid_image",
            f"Could not read image: {exc}",
            summary=_base_summary(image_path, city, started, timings),
        )
    height, width = rgb.shape[:2]
    timings["load_s"] = round(time.monotonic() - stage_start, 3)

    emit("extract", 20, "Extracting boundary and reading labels")
    stage_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        extraction_future = executor.submit(
            segment_image,
            rgb,
            model_path=opts.model_path,
            simplify_px=opts.simplify_px,
            threshold=opts.model_threshold,
            hints=opts.extraction_hints(),
        )
        labels_future = executor.submit(
            extract_ocr_labels_from_rgb,
            str(normalized_path),
            rgb,
            allow_tesseract_fallback=opts.allow_tesseract_fallback,
            cache=opts.ocr_cache,
        )
        extraction_error: Exception | None = None
        try:
            extraction = extraction_future.result()
        except ValueError as exc:
            extraction = None
            extraction_error = exc
        try:
            labels = tuple(labels_future.result())
        except Exception:
            labels = ()
    timings["extract_s"] = round(time.monotonic() - stage_start, 3)

    summary = _base_summary(image_path, city, started, timings)
    summary["image_width"] = width
    summary["image_height"] = height
    summary["ocr_label_count"] = len(labels)

    if extraction is None:
        return _failed_result(
            "no_boundary_found",
            str(extraction_error or "No service-area polygon could be extracted"),
            summary=summary,
            labels=labels,
        )
    summary["extraction"] = _extraction_summary(extraction)

    emit("georeference", 55, "Inferring location and fitting georeference")
    return _georeference_and_export(
        normalized_path=normalized_path,
        source_path=Path(image_path),
        rgb=rgb,
        extraction=extraction,
        labels=labels,
        city=city,
        output_path=Path(output_path) if output_path is not None else None,
        debug_dir=Path(debug_dir) if debug_dir is not None else None,
        options=opts,
        summary=summary,
        timings=timings,
        started=started,
        emit=emit,
    )


def complete_with_city(
    image_path: str | Path,
    prior: PipelineResult,
    city: str,
    *,
    output_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
    options: PipelineOptions | None = None,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    """Finish a ``needs_city`` run with a user-supplied city.

    Reuses the prior extraction and OCR labels; only georeference and export
    are recomputed.
    """
    if prior.extraction is None:
        raise ValueError("prior result carries no extraction to reuse")
    opts = options or PipelineOptions()
    emit = progress or (lambda stage, percent, detail: None)
    started = time.monotonic()
    timings: dict[str, float] = {}
    normalized_path, rgb = _load(image_path, debug_dir=debug_dir)
    summary = _base_summary(image_path, city, started, timings)
    summary["image_width"] = rgb.shape[1]
    summary["image_height"] = rgb.shape[0]
    summary["ocr_label_count"] = len(prior.labels or ())
    summary["extraction"] = _extraction_summary(prior.extraction)
    emit("georeference", 55, "Fitting georeference with provided city")
    return _georeference_and_export(
        normalized_path=normalized_path,
        source_path=Path(image_path),
        rgb=rgb,
        extraction=prior.extraction,
        labels=prior.labels or (),
        city=city,
        output_path=Path(output_path) if output_path is not None else None,
        debug_dir=Path(debug_dir) if debug_dir is not None else None,
        options=opts,
        summary=summary,
        timings=timings,
        started=started,
        emit=emit,
    )


def _load(image_path: str | Path, *, debug_dir: str | Path | None) -> tuple[Path, np.ndarray]:
    normalized_path = normalize_image_for_processing(
        image_path,
        output_dir=debug_dir,
    )
    return Path(normalized_path), load_rgb(normalized_path)


def _georeference_and_export(
    *,
    normalized_path: Path,
    source_path: Path,
    rgb: np.ndarray,
    extraction: ExtractionResult,
    labels: tuple[OcrLabel, ...],
    city: str | None,
    output_path: Path | None,
    debug_dir: Path | None,
    options: PipelineOptions,
    summary: dict[str, Any],
    timings: dict[str, float],
    started: float,
    emit: ProgressCallback,
) -> PipelineResult:
    height, width = rgb.shape[:2]
    stage_start = time.monotonic()
    try:
        georeference = georeference_from_labels(
            list(labels),
            str(normalized_path),
            city,
            width,
            height,
            rgb=rgb,
            min_control_points=options.min_control_points,
        )
    except Exception as exc:
        summary["georeference_error"] = str(exc)
        return _failed_result(
            "georeference_error",
            f"Georeferencing failed: {exc}",
            summary=summary,
            extraction=extraction,
            labels=labels,
            cacheable=False,
        )
    if georeference is None:
        # Label control points were insufficient (few distinct street names,
        # garbled OCR). If a city is known — supplied or inferred from labels
        # like "Tampa FL" — try matching detected road structure against
        # public OSM road geometry around that city.
        georeference = _road_search_fallback(rgb, extraction, labels, city)
    timings["georeference_s"] = round(time.monotonic() - stage_start, 3)

    if georeference is None:
        if city is not None:
            return _failed_result(
                "georeference_failed_with_city",
                "Could not fit a reliable georeference even with the provided city. "
                "The screenshot may not contain enough readable map labels.",
                summary=summary,
                extraction=extraction,
                labels=labels,
            )
        detail = _needs_city_detail(labels, city)
        summary["needs_city"] = detail.to_dict()
        summary["status"] = "needs_city"
        mask_path = overlay_path = None
        if debug_dir is not None:
            # The caller will show the extracted boundary while prompting for
            # a city, so the debug artifacts are written even without a fit.
            mask_path = debug_dir / "mask.png"
            overlay_path = debug_dir / "overlay.png"
            write_mask_png(extraction.mask, mask_path)
            write_overlay_png(normalized_path, extraction.mask, overlay_path, rgb=rgb)
        return PipelineResult(
            status="needs_city",
            reason=detail.reason,
            summary=summary,
            extraction=extraction,
            labels=labels,
            needs_city=detail,
            mask_path=mask_path,
            overlay_path=overlay_path,
        )

    transform = georeference.transform
    combined_confidence = round(float(extraction.confidence) * float(transform.confidence), 3)
    summary["georeference"] = {
        "source": transform.source,
        "city": transform.city,
        "confidence": transform.confidence,
        "control_points": len(georeference.control_points),
        "residual_median_m": round(georeference.residual_median_m, 1),
        "residual_p90_m": round(georeference.residual_p90_m, 1),
        "meters_per_pixel": transform.meters_per_pixel,
        "rotation_degrees": round(transform.rotation_radians * 180.0 / np.pi, 3),
    }
    summary["combined_confidence"] = combined_confidence
    if options.min_confidence > 0 and combined_confidence < options.min_confidence:
        return _failed_result(
            "low_confidence",
            f"Combined confidence {combined_confidence} is below the required minimum "
            f"{options.min_confidence}.",
            summary=summary,
            extraction=extraction,
            labels=labels,
            georeference=georeference,
        )

    emit("export", 90, "Writing artifacts")
    geojson = feature_collection(
        extraction,
        width,
        height,
        transform,
        str(source_path),
        city or "",
    )
    mask_path = overlay_path = None
    if output_path is not None:
        write_geojson(geojson, output_path)
    if debug_dir is not None:
        mask_path = debug_dir / "mask.png"
        overlay_path = debug_dir / "overlay.png"
        write_mask_png(extraction.mask, mask_path)
        write_overlay_png(normalized_path, extraction.mask, overlay_path, rgb=rgb)
    summary["status"] = "complete"
    summary["city"] = transform.city
    summary["target_selection_confidence"] = target_selection_confidence(extraction)
    summary["total_s"] = round(time.monotonic() - started, 3)
    emit("complete", 100, "Boundary complete")
    return PipelineResult(
        status="complete",
        reason=None,
        summary=summary,
        geojson=geojson,
        extraction=extraction,
        labels=labels,
        georeference=georeference,
        output_path=output_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
    )


ROAD_SEARCH_MAX_DIMENSION = 500


def _road_search_fallback(
    rgb: np.ndarray,
    extraction: ExtractionResult,
    labels: tuple[OcrLabel, ...],
    city: str | None,
) -> GeoreferenceResult | None:
    if city is not None:
        candidates = [city]
    else:
        try:
            contexts = resolve_city_contexts(list(labels), None)
        except Exception:
            return None
        candidates = [context.query for context in contexts[:2]]
    if not candidates:
        return None

    # The road-structure search works on low-resolution street grids (its
    # line-feature matcher only activates below ~520 px), so search on a
    # downscaled copy and rescale the fitted transform back to source pixels.
    import cv2
    from dataclasses import replace as dataclass_replace

    from shapely.affinity import scale as scale_geometry

    height, width = rgb.shape[:2]
    factor = min(1.0, ROAD_SEARCH_MAX_DIMENSION / max(height, width))
    if factor < 1.0:
        small = cv2.resize(
            rgb,
            (max(1, round(width * factor)), max(1, round(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
        geometry = scale_geometry(extraction.pixel_geometry, xfact=factor, yfact=factor, origin=(0, 0))
        pixel_scale = small.shape[1] / float(width)
    else:
        small = rgb
        geometry = extraction.pixel_geometry
        pixel_scale = 1.0

    for candidate in candidates:
        try:
            result = georeference_from_city_context(small, candidate, geometry)
        except Exception:
            continue
        if result is None:
            continue
        transform = dataclass_replace(
            result.transform,
            meters_per_pixel=result.transform.meters_per_pixel * pixel_scale,
        )
        return GeoreferenceResult(
            transform=transform,
            control_points=result.control_points,
            residual_median_m=result.residual_median_m,
            residual_p90_m=result.residual_p90_m,
        )
    return None


def _needs_city_detail(labels: tuple[OcrLabel, ...], city: str | None) -> NeedsCityDetail:
    contexts = resolve_city_contexts(list(labels), city)
    reason: Literal["no_city_context", "georeference_failed"] = (
        "no_city_context" if not contexts else "georeference_failed"
    )
    ranked = sorted(labels, key=lambda label: label.confidence, reverse=True)
    samples: list[str] = []
    for label in ranked:
        text = label.text.strip()
        if text and text not in samples:
            samples.append(text)
        if len(samples) >= NEEDS_CITY_SAMPLE_LABELS:
            break
    return NeedsCityDetail(
        reason=reason,
        ocr_label_count=len(labels),
        sample_labels=tuple(samples),
    )


def _extraction_summary(extraction: ExtractionResult) -> dict[str, Any]:
    diagnostics = extraction.diagnostics or {}
    return {
        "engine": diagnostics.get("segmentation_engine", extraction.style),
        "style": extraction.style,
        "coverage_ratio": round(extraction.coverage_ratio, 6),
        "contour_count": extraction.contour_count,
        "confidence": extraction.confidence,
    }


def _base_summary(
    image_path: str | Path,
    city: str | None,
    started: float,
    timings: dict[str, float],
) -> dict[str, Any]:
    return {
        "image": str(image_path),
        "city_input": city,
        "timings": timings,
        "status": "in_progress",
    }


def _failed_result(
    reason: str,
    message: str,
    *,
    summary: dict[str, Any],
    extraction: ExtractionResult | None = None,
    labels: tuple[OcrLabel, ...] = (),
    georeference: GeoreferenceResult | None = None,
    cacheable: bool = True,
) -> PipelineResult:
    summary = dict(summary)
    summary["status"] = "failed"
    summary["reason"] = reason
    summary["message"] = message
    return PipelineResult(
        status="failed",
        reason=reason,
        summary=summary,
        extraction=extraction,
        labels=tuple(labels),
        georeference=georeference,
        cacheable=cacheable,
    )
