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
from .georef_transform import GeoreferenceTransform
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
RESCUE_MAX_ROTATION_DEGREES = 8.0
RESCUE_MIN_METERS_PER_PIXEL = 1.5
RESCUE_MAX_METERS_PER_PIXEL = 120.0
RESCUE_MAX_LABEL_ERROR_FRACTION = 0.15
RESCUE_MAX_CITY_DISTANCE_M = 80_000.0


def _label_anchors(
    labels: tuple[OcrLabel, ...],
    city_center_mercator: tuple[float, float] | None,
    city_query: str = "",
) -> list[tuple[OcrLabel, tuple[float, float], bool]]:
    """Geocoded label anchors near the candidate city.

    Each entry is (label, mercator, is_city_label). Ambiguous names are
    resolved by choosing the geocode candidate nearest the inferred city —
    a map of Orlando means "Conway" is the Orlando neighborhood, not the
    Arkansas city. City-name labels are flagged: their drawn position is
    cartographic, not geographic, so they validate but never fit.
    """
    from .geocoder import geocode

    city_tokens = {token for token in city_query.lower().replace(",", " ").split() if len(token) > 2}
    anchors: list[tuple[OcrLabel, tuple[float, float], bool]] = []
    for label in labels:
        text = label.text.strip()
        if len(text) < 4 or not any(ch.isalpha() for ch in text):
            continue
        try:
            results = geocode(text, limit=5)
        except Exception:
            continue
        if not results:
            continue
        best: tuple[float, tuple[float, float]] | None = None
        for result in results:
            mercator = result.mercator
            if city_center_mercator is None:
                best = (0.0, mercator)
                break
            distance = float(
                np.hypot(mercator[0] - city_center_mercator[0], mercator[1] - city_center_mercator[1])
            )
            if distance <= RESCUE_MAX_CITY_DISTANCE_M and (best is None or distance < best[0]):
                best = (distance, mercator)
        if best is None:
            continue
        label_tokens = {token for token in text.lower().split() if len(token) > 2}
        is_city_label = bool(city_tokens and city_tokens & label_tokens)
        anchors.append((label, best[1], is_city_label))
    return anchors


def _north_up_anchor_fit(
    anchors: list[tuple[OcrLabel, tuple[float, float], bool]],
    width: int,
    height: int,
    city: str,
) -> GeoreferenceResult | None:
    """Least-squares north-up similarity fit from geocoded label anchors.

    Screenshots of coverage maps are north-up; locking rotation to zero
    makes two anchors sufficient and immune to the rotated local optima the
    road search can fall into. City-name labels are excluded (their drawn
    position is arbitrary)."""
    from .georef_transform import mercator_to_lonlat

    fitting = [(label, mercator) for label, mercator, is_city in anchors if not is_city]
    if len(fitting) < 2:
        return None
    pixels = np.array([[label.x, -label.y] for label, _ in fitting])
    mercs = np.array([list(mercator) for _, mercator in fitting])
    spread = float(np.hypot(*(pixels.max(axis=0) - pixels.min(axis=0))))
    if spread < 0.12 * float(np.hypot(width, height)):
        return None
    # Solve merc = origin + pixel * mpp in least squares over both axes.
    pixel_deltas = pixels - pixels.mean(axis=0)
    merc_deltas = mercs - mercs.mean(axis=0)
    denominator = float((pixel_deltas**2).sum())
    if denominator <= 0:
        return None
    mpp = float((pixel_deltas * merc_deltas).sum() / denominator)
    if not RESCUE_MIN_METERS_PER_PIXEL <= mpp <= RESCUE_MAX_METERS_PER_PIXEL:
        return None
    origin = mercs.mean(axis=0) - pixels.mean(axis=0) * mpp
    residuals = np.hypot(*(mercs - (origin + pixels * mpp)).T)
    lon, lat = mercator_to_lonlat(float(origin[0]), float(origin[1]))
    transform = GeoreferenceTransform(
        city=city,
        lon=lon,
        lat=lat,
        origin_x_ratio=0.0,
        origin_y_ratio=0.0,
        meters_per_pixel=mpp,
        rotation_radians=0.0,
        confidence=0.58,
        source="label-anchors:north-up-fit",
    )
    return GeoreferenceResult(
        transform=transform,
        control_points=[],
        residual_median_m=float(np.median(residuals)),
        residual_p90_m=float(np.quantile(residuals, 0.9)),
    )


def _mercator_to_pixel(
    mercator: tuple[float, float],
    width: int,
    height: int,
    transform,
) -> tuple[float, float]:
    import math

    from .georef_transform import lonlat_to_mercator

    origin_x = transform.origin_x_ratio * width
    origin_y = transform.origin_y_ratio * height
    origin_merc = lonlat_to_mercator(transform.lon, transform.lat)
    rx = (mercator[0] - origin_merc[0]) / transform.meters_per_pixel
    ry = (mercator[1] - origin_merc[1]) / transform.meters_per_pixel
    cos_r = math.cos(transform.rotation_radians)
    sin_r = math.sin(transform.rotation_radians)
    px = rx * cos_r + ry * sin_r
    py = -rx * sin_r + ry * cos_r
    return px + origin_x, origin_y - py


def _rescue_fit_is_sane(
    result: GeoreferenceResult,
    anchors: list[tuple[OcrLabel, tuple[float, float]]],
    width: int,
    height: int,
) -> bool:
    transform = result.transform
    rotation_degrees = abs(transform.rotation_radians) * 180.0 / np.pi
    if rotation_degrees > RESCUE_MAX_ROTATION_DEGREES:
        return False
    if not RESCUE_MIN_METERS_PER_PIXEL <= transform.meters_per_pixel <= RESCUE_MAX_METERS_PER_PIXEL:
        return False
    if anchors:
        diagonal = float(np.hypot(width, height))
        errors = []
        for label, mercator, _is_city in anchors:
            px, py = _mercator_to_pixel(mercator, width, height, transform)
            errors.append(float(np.hypot(px - label.x, py - label.y)))
        if float(np.median(errors)) > RESCUE_MAX_LABEL_ERROR_FRACTION * diagonal:
            return False
    return True


def _road_search_fallback(
    rgb: np.ndarray,
    extraction: ExtractionResult,
    labels: tuple[OcrLabel, ...],
    city: str | None,
) -> GeoreferenceResult | None:
    try:
        contexts = resolve_city_contexts(list(labels), city)
    except Exception:
        return None
    if not contexts:
        return None
    candidates = [context.query for context in contexts[:2]]
    height, width = rgb.shape[:2]
    center_mercator = contexts[0].center.mercator
    anchors = _label_anchors(labels, center_mercator, contexts[0].query)

    # A north-up anchor fit beats road matching whenever at least two solid
    # geocoded neighborhood labels exist (three is the standard fit's
    # minimum, which already declined). Rotation is locked to zero — these
    # screenshots are north-up — making the fit immune to the rotated
    # optima the road search can fall into.
    north_up = _north_up_anchor_fit(anchors, width, height, contexts[0].query)
    if north_up is not None and _rescue_fit_is_sane(north_up, anchors, width, height):
        return north_up

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
        rescued = GeoreferenceResult(
            transform=transform,
            control_points=result.control_points,
            residual_median_m=result.residual_median_m,
            residual_p90_m=result.residual_p90_m,
        )
        # Road matching can converge to rotated or mis-scaled optima; a fit
        # that disagrees with where geocoded labels actually sit is wrong.
        if not _rescue_fit_is_sane(rescued, anchors, width, height):
            continue
        return rescued
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
