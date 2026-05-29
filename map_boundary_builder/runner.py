from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from dataclasses import dataclass
from math import hypot
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from shapely.geometry import shape

from .catalog_match import (
    CATALOG_LABEL_HINT_MIN_IOU,
    catalog_style_supported,
    catalog_feature_collection,
    has_active_catalog_area_hint,
    has_stale_catalog_area_hint,
    match_service_area_catalog,
)
from .extract import (
    DEFAULT_SIMPLIFY_PX,
    extract_service_area,
    extraction_scale_factor,
    load_rgb,
    write_mask_png,
    write_overlay_png,
)
from .georeference import (
    CityContext,
    filename_city_contexts,
    georeference_from_city_context,
    georeference_from_label_context,
    georeference_from_labels,
    infer_city_contexts,
    is_credible_context_hint_georeference,
)
from .georef_transform import lonlat_to_mercator
from .geojson import feature_collection, write_geojson
from .image_io import is_svg_image, normalize_image_for_processing
from .ocr import extract_ocr_labels, extract_ocr_labels_from_rgb
from .osm_roads import image_feature_distance
from .runtime_config import RAPIDOCR_MAX_DIMENSION, RAPIDOCR_PURPLE_FILL_MAX_DIMENSION

ProgressCallback = Callable[[dict[str, Any]], None]
MAX_ROAD_CONTEXT_CANDIDATES = 1
CATALOG_EXTRACT_MAX_DIMENSION = max(0, int(os.environ.get("MAP_BOUNDARY_CATALOG_EXTRACT_MAX_DIMENSION", "240")))
CATALOG_RETRY_EXTRACT_MAX_DIMENSION = max(
    0,
    int(os.environ.get("MAP_BOUNDARY_CATALOG_RETRY_EXTRACT_MAX_DIMENSION", "400")),
)
GENERAL_EXTRACT_MAX_DIMENSION = max(0, int(os.environ.get("MAP_BOUNDARY_GENERAL_EXTRACT_MAX_DIMENSION", "1600")))
DEFAULT_CATALOG_MISS_REFINE_MAX_DIMENSION = (
    min(GENERAL_EXTRACT_MAX_DIMENSION, 1400) if GENERAL_EXTRACT_MAX_DIMENSION > 0 else 0
)
CATALOG_MISS_REFINE_MAX_DIMENSION = max(
    0,
    int(
        os.environ.get(
            "MAP_BOUNDARY_CATALOG_MISS_REFINE_MAX_DIMENSION",
            str(DEFAULT_CATALOG_MISS_REFINE_MAX_DIMENSION),
        )
    ),
)
CATALOG_LABEL_HINT_MIN_CONFIDENCE = 85.0
CATALOG_LABEL_HINT_MAX_IMAGE_DIMENSION = 900
CATALOG_LABEL_HINT_SPARSE_LABEL_COUNT = 5
LOW_RES_SHAPE_CATALOG_MAX_IMAGE_DIMENSION = 520
LOW_RES_SHAPE_CATALOG_MIN_IOU = 0.94
LOW_RES_SHAPE_CATALOG_MIN_MARGIN = 0.24
LOW_RES_SHAPE_CATALOG_MIN_AREA_RATIO = 0.92
LOW_RES_SHAPE_CATALOG_MAX_AREA_RATIO = 1.08
LOW_RES_SHAPE_CATALOG_MIN_EXTRACTION_CONFIDENCE = 0.98
ROAD_NETWORK_CONTEXT_FALLBACK_ENV = "MAP_BOUNDARY_ENABLE_ROAD_CONTEXT_FALLBACK"


@dataclass(frozen=True)
class BoundaryBuildOptions:
    simplify_px: float = DEFAULT_SIMPLIFY_PX
    min_confidence: float = 0.55
    min_control_points: int = 3
    preview_max_dimension: int | None = None
    write_mask_artifact: bool = True
    allow_catalog: bool = True
    filename_hint: str | None = None


@dataclass(frozen=True)
class BoundaryBuildResult:
    geojson: dict[str, Any]
    summary: dict[str, Any]
    output_path: Path
    mask_path: Path | None = None
    overlay_path: Path | None = None


def emit_progress(
    progress: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    percent: int,
    status: str = "running",
    details: dict[str, Any] | None = None,
) -> None:
    if progress is None:
        return
    event: dict[str, Any] = {
        "stage": stage,
        "message": message,
        "percent": percent,
        "status": status,
    }
    if details:
        event["details"] = details
    progress(event)


def build_boundary(
    image_path: str | Path,
    city: str | None,
    output_path: str | Path,
    *,
    debug_dir: str | Path | None = None,
    options: BoundaryBuildOptions | None = None,
    progress: ProgressCallback | None = None,
) -> BoundaryBuildResult:
    opts = options or BoundaryBuildOptions()
    image_path = Path(image_path)
    output_path = Path(output_path)
    debug_path = Path(debug_dir) if debug_dir else None
    city_input = city.strip() if isinstance(city, str) and city.strip() else None
    allow_catalog = catalog_matching_enabled(opts)
    filename_hint = opts.filename_hint or image_path.stem
    allow_pre_ocr_catalog = should_try_pre_ocr_catalog(
        city_input=city_input,
        allow_catalog=allow_catalog,
        filename_hint=filename_hint,
    )

    emit_progress(
        progress,
        stage="inspect",
        message="Rasterizing SVG upload" if is_svg_image(image_path) else "Reading image metadata",
        percent=5,
    )
    image_path = normalize_image_for_processing(
        image_path,
        output_dir=debug_path or output_path.parent,
        composite_transparent_rasters=False,
    )
    with Image.open(image_path) as img:
        width, height = img.size

    labels_future: Future[list[Any]] | None = None
    ocr_executor: ThreadPoolExecutor | None = None
    road_feature_future: Future[Any] | None = None
    road_feature_executor: ThreadPoolExecutor | None = None
    georef_resource_future: Future[Any] | None = None
    georef_resource_executor: ThreadPoolExecutor | None = None

    def ensure_georeference_resource_preload() -> None:
        nonlocal georef_resource_future, georef_resource_executor
        if georef_resource_future is not None:
            return
        georef_resource_executor = ThreadPoolExecutor(max_workers=1)
        georef_resource_future = georef_resource_executor.submit(preload_georeference_resources)

    if should_overlap_ocr_with_extraction(
        city_input=city_input,
        allow_catalog=allow_catalog,
        filename_hint=filename_hint,
    ):
        ocr_executor = ThreadPoolExecutor(max_workers=1)
        labels_future = ocr_executor.submit(extract_ocr_labels, str(image_path))
        ensure_georeference_resource_preload()

    try:
        emit_progress(
            progress,
            stage="extract",
            message="Extracting service-area pixels",
            percent=18,
            details={"width": width, "height": height},
        )
        rgb = load_rgb(image_path)
        extraction_max_dimension = (
            CATALOG_EXTRACT_MAX_DIMENSION if allow_pre_ocr_catalog else GENERAL_EXTRACT_MAX_DIMENSION
        )
        used_catalog_scaled_extraction = (
            allow_pre_ocr_catalog and extraction_scale_factor(rgb, extraction_max_dimension) < 1.0
        )
        extraction = extract_service_area(
            image_path,
            simplify_px=opts.simplify_px,
            rgb=rgb,
            max_dimension=extraction_max_dimension,
        )
        emit_progress(
            progress,
            stage="extract",
            message="Pixel polygon extracted",
            percent=36,
            details={
                "style": extraction.style,
                "coverage_ratio": round(extraction.coverage_ratio, 6),
                "contour_count": extraction.contour_count,
                "confidence": extraction.confidence,
            },
        )

        catalog_style_can_match = catalog_style_supported(extraction.style)
        if allow_pre_ocr_catalog and catalog_style_can_match:
            catalog_match = match_service_area_catalog(
                extraction.pixel_geometry,
                style=extraction.style,
                area_hint_texts=[city_input] if city_input is not None else None,
            )
            if catalog_match is not None:
                return finish_catalog_boundary_result(
                    extraction,
                    catalog_match,
                    width=width,
                    height=height,
                    image_path=image_path,
                    city_input=city_input or "Auto",
                    output_path=output_path,
                    debug_path=debug_path,
                    opts=opts,
                    rgb=rgb,
                    progress=progress,
                )
            catalog_match = low_resolution_shape_catalog_match(
                extraction,
                width=width,
                height=height,
                city_input=city_input,
            )
            if catalog_match is not None:
                return finish_catalog_boundary_result(
                    extraction,
                    catalog_match,
                    width=width,
                    height=height,
                    image_path=image_path,
                    city_input=city_input or "Auto",
                    output_path=output_path,
                    debug_path=debug_path,
                    opts=opts,
                    rgb=rgb,
                    progress=progress,
                    georeference_source="catalog-shape-match:low-res-shape",
                )

        if should_retry_pre_ocr_catalog(
            city_input=city_input,
            filename_hint=filename_hint,
            allow_pre_ocr_catalog=allow_pre_ocr_catalog,
            used_catalog_scaled_extraction=used_catalog_scaled_extraction,
            catalog_style_can_match=catalog_style_can_match,
        ):
            emit_progress(
                progress,
                stage="extract",
                message="Retrying known service-area shape",
                percent=37,
                details={"width": width, "height": height},
            )
            retry_extraction = extract_service_area(
                image_path,
                simplify_px=opts.simplify_px,
                rgb=rgb,
                max_dimension=CATALOG_RETRY_EXTRACT_MAX_DIMENSION,
            )
            catalog_match = match_service_area_catalog(
                retry_extraction.pixel_geometry,
                style=retry_extraction.style,
                area_hint_texts=[city_input] if city_input is not None else None,
            )
            if catalog_match is not None:
                return finish_catalog_boundary_result(
                    retry_extraction,
                    catalog_match,
                    width=width,
                    height=height,
                    image_path=image_path,
                    city_input=city_input or "Auto",
                    output_path=output_path,
                    debug_path=debug_path,
                    opts=opts,
                    rgb=rgb,
                    progress=progress,
                    georeference_source="catalog-shape-match:retry",
                )

        if used_catalog_scaled_extraction:
            if labels_future is None:
                ocr_executor = ThreadPoolExecutor(max_workers=1)
                labels_future = submit_ocr_labels_from_rgb(
                    ocr_executor,
                    image_path,
                    rgb,
                    style=extraction.style,
                )
                ensure_georeference_resource_preload()
            emit_progress(
                progress,
                stage="extract",
                message="Refining service-area pixels",
                percent=38,
                details={"width": width, "height": height},
            )
            extraction = extract_service_area(
                image_path,
                simplify_px=opts.simplify_px,
                rgb=rgb,
                max_dimension=CATALOG_MISS_REFINE_MAX_DIMENSION,
            )
            emit_progress(
                progress,
                stage="extract",
                message="Pixel polygon refined",
                percent=40,
                details={
                    "style": extraction.style,
                    "coverage_ratio": round(extraction.coverage_ratio, 6),
                    "contour_count": extraction.contour_count,
                    "confidence": extraction.confidence,
                },
            )
            if allow_pre_ocr_catalog and catalog_style_supported(extraction.style):
                catalog_match = match_service_area_catalog(
                    extraction.pixel_geometry,
                    style=extraction.style,
                    area_hint_texts=[city_input] if city_input is not None else None,
                )
                if catalog_match is not None:
                    return finish_catalog_boundary_result(
                        extraction,
                        catalog_match,
                        width=width,
                        height=height,
                        image_path=image_path,
                        city_input=city_input or "Auto",
                        output_path=output_path,
                        debug_path=debug_path,
                        opts=opts,
                        rgb=rgb,
                        progress=progress,
                    )
                catalog_match = low_resolution_shape_catalog_match(
                    extraction,
                    width=width,
                    height=height,
                    city_input=city_input,
                )
                if catalog_match is not None:
                    return finish_catalog_boundary_result(
                        extraction,
                        catalog_match,
                        width=width,
                        height=height,
                        image_path=image_path,
                        city_input=city_input or "Auto",
                        output_path=output_path,
                        debug_path=debug_path,
                        opts=opts,
                        rgb=rgb,
                        progress=progress,
                        georeference_source="catalog-shape-match:low-res-shape",
                    )

        label_y_max = (
            extraction.pixel_geometry.bounds[3] + max(24.0, height * 0.04)
            if extraction.style == "dark-teal"
            else None
        )
        label_y_min = (
            extraction.pixel_geometry.bounds[1] - max(18.0, height * 0.06)
            if extraction.style == "gray-fill"
            else None
        )
        if labels_future is None:
            ocr_executor = ThreadPoolExecutor(max_workers=1)
            labels_future = submit_ocr_labels_from_rgb(
                ocr_executor,
                image_path,
                rgb,
                style=extraction.style,
            )
            ensure_georeference_resource_preload()
        if should_precompute_road_features(extraction.style, width, height):
            road_feature_executor = ThreadPoolExecutor(max_workers=1)
            road_feature_future = road_feature_executor.submit(image_feature_distance, rgb)
        emit_progress(
            progress,
            stage="ocr",
            message="Reading map labels on server",
            percent=44,
        )
        labels = labels_future.result()
    finally:
        if ocr_executor is not None:
            ocr_executor.shutdown(wait=False, cancel_futures=True)
        if road_feature_executor is not None:
            road_feature_executor.shutdown(wait=False, cancel_futures=True)
        if georef_resource_executor is not None:
            georef_resource_executor.shutdown(wait=False, cancel_futures=False)
    road_feature_distance = ready_future_result(road_feature_future)
    emit_progress(
        progress,
        stage="ocr",
        message="Map labels read",
        percent=47,
        details={
            "label_count": len(labels),
            "top_labels": [label.text for label in labels[:8]],
        },
    )
    if city_input is None and allow_catalog and should_try_label_hinted_catalog(width, height, labels):
        label_hints = high_confidence_label_texts(labels)
        catalog_match = match_service_area_catalog(
            extraction.pixel_geometry,
            style=extraction.style,
            min_iou=CATALOG_LABEL_HINT_MIN_IOU,
            area_hint_texts=label_hints,
        )
        if catalog_match is not None:
            data = catalog_feature_collection(
                extraction,
                catalog_match,
                width=width,
                height=height,
                image_path=image_path,
                city_input="Auto",
            )
            properties = data["features"][0]["properties"]
            properties["georeference_source"] = "catalog-shape-match:ocr-label-hint"
            properties["catalog_label_hints"] = label_hints[:5]
            combined_confidence = properties["combined_confidence"]
            emit_progress(
                progress,
                stage="georeference",
                message="Matched known service-area shape from labels",
                percent=78,
                details={
                    "source": "catalog-shape-match:ocr-label-hint",
                    "catalog_slug": catalog_match.entry.slug,
                    "shape_iou": round(catalog_match.iou, 3),
                    "combined_confidence": combined_confidence,
                    "control_points": 0,
                    "median_residual_m": 0.0,
                    "p90_residual_m": 0.0,
                },
            )
            if combined_confidence < opts.min_confidence:
                raise ValueError(
                    f"Combined confidence {combined_confidence:.2f} is below --min-confidence "
                    f"{opts.min_confidence:.2f}. Provide a clearer map crop or lower the threshold."
                )
            return finish_boundary_result(
                data,
                extraction,
                image_path,
                output_path,
                debug_path,
                opts,
                width,
                height,
                city_input="Auto",
                rgb=rgb,
                progress=progress,
            )
    wait_future_result(georef_resource_future)
    georef = fit_georeference(
        labels,
        image_path,
        extraction.pixel_geometry,
        rgb=rgb,
        city_input=city_input,
        context_hints=filename_city_contexts(filename_hint) if city_input is None else None,
        width=width,
        height=height,
        coverage_ratio=extraction.coverage_ratio,
        min_control_points=opts.min_control_points,
        label_y_min=label_y_min,
        label_y_max=label_y_max,
        road_feature_distance=road_feature_distance,
        progress=progress,
    )
    if georef is None:
        raise ValueError(
            "Could not infer a reliable map location and georeference from OCR/geocoded map labels. "
            "Provide a higher-resolution map crop with readable city or neighborhood labels."
        )

    geo_transform = georef.transform
    data = feature_collection(extraction, width, height, geo_transform, str(image_path), city_input or "Auto")
    geom = shape(data["features"][0]["geometry"])
    combined_confidence = min(extraction.confidence, geo_transform.confidence)
    properties = data["features"][0]["properties"]
    properties["combined_confidence"] = combined_confidence
    properties["geodesic_bbox_lonlat"] = list(geom.bounds)
    properties["georeference_control_points"] = len(georef.control_points)
    properties["georeference_residual_median_m"] = georef.residual_median_m
    properties["georeference_residual_p90_m"] = georef.residual_p90_m
    if city_input is None and should_label_as_regional_area(geom.bounds, properties["city"], georef):
        properties["city"] = "Inferred map area"
    if georef.road_match is not None:
        properties["road_match_score"] = georef.road_match.score
        properties["road_match_base_score"] = georef.road_match.base_score
        properties["road_match_sampled_points"] = georef.road_match.sampled_points
        if georef.road_match_elapsed_s is not None:
            properties["road_match_elapsed_s"] = round(georef.road_match_elapsed_s, 6)
        if georef.road_match.anchor_label is not None:
            properties["road_match_anchor_label"] = georef.road_match.anchor_label.text

    emit_progress(
        progress,
        stage="georeference",
        message="Map transform fitted",
        percent=78,
        details={
            "source": geo_transform.source,
            "combined_confidence": combined_confidence,
            "control_points": len(georef.control_points),
            "median_residual_m": round(georef.residual_median_m, 1),
            "p90_residual_m": round(georef.residual_p90_m, 1),
        },
    )

    if combined_confidence < opts.min_confidence:
        raise ValueError(
            f"Combined confidence {combined_confidence:.2f} is below --min-confidence "
            f"{opts.min_confidence:.2f}. Provide a clearer map crop or lower the threshold."
        )

    return finish_boundary_result(
        data,
        extraction,
        image_path,
        output_path,
        debug_path,
        opts,
        width,
        height,
        city_input=city_input or "Auto",
        rgb=rgb,
        progress=progress,
    )


def should_try_label_hinted_catalog(width: int, height: int, labels: list[Any]) -> bool:
    if not high_confidence_label_texts(labels):
        return False
    if len(labels) <= CATALOG_LABEL_HINT_SPARSE_LABEL_COUNT:
        return True
    return max(width, height) <= CATALOG_LABEL_HINT_MAX_IMAGE_DIMENSION


def should_precompute_road_features(style: str, width: int, height: int) -> bool:
    return style == "bright-blue" and max(width, height) >= 1000


def ready_future_result(future: Future[Any] | None) -> Any | None:
    if future is None or not future.done() or future.cancelled():
        return None
    try:
        return future.result()
    except Exception:
        return None


def wait_future_result(future: Future[Any] | None) -> Any | None:
    if future is None or future.cancelled():
        return None
    try:
        return future.result()
    except Exception:
        return None


def preload_georeference_resources() -> dict[str, int]:
    from .geocoder import load_geocoder_seed
    from .osm_places import load_osm_places_seed
    from .osm_roads import load_road_points_seed

    return {
        "geocoder_seed_entries": len(load_geocoder_seed()),
        "osm_place_seed_entries": len(load_osm_places_seed()),
        "road_seed_entries": len(load_road_points_seed()),
    }


def submit_ocr_labels_from_rgb(
    executor: ThreadPoolExecutor,
    image_path: str | Path,
    rgb,
    *,
    style: str,
) -> Future[list[Any]]:
    rapidocr_max_dimension = rapidocr_max_dimension_for_extraction_style(style)
    if rapidocr_max_dimension is None:
        return executor.submit(extract_ocr_labels_from_rgb, str(image_path), rgb)
    return executor.submit(
        extract_ocr_labels_from_rgb,
        str(image_path),
        rgb,
        rapidocr_max_dimension=rapidocr_max_dimension,
    )


def rapidocr_max_dimension_for_extraction_style(style: str) -> int | None:
    if style != "purple-fill":
        return None
    if RAPIDOCR_MAX_DIMENSION <= 0 or RAPIDOCR_PURPLE_FILL_MAX_DIMENSION <= 0:
        return None
    if RAPIDOCR_PURPLE_FILL_MAX_DIMENSION >= RAPIDOCR_MAX_DIMENSION:
        return None
    return RAPIDOCR_PURPLE_FILL_MAX_DIMENSION


def low_resolution_shape_catalog_match(
    extraction,
    *,
    width: int,
    height: int,
    city_input: str | None,
):
    if max(width, height) > LOW_RES_SHAPE_CATALOG_MAX_IMAGE_DIMENSION:
        return None
    if extraction.confidence < LOW_RES_SHAPE_CATALOG_MIN_EXTRACTION_CONFIDENCE:
        return None
    match = match_service_area_catalog(
        extraction.pixel_geometry,
        style=extraction.style,
        min_iou=LOW_RES_SHAPE_CATALOG_MIN_IOU,
        min_margin=LOW_RES_SHAPE_CATALOG_MIN_MARGIN,
        area_hint_texts=[city_input] if city_input is not None else None,
    )
    if match is None:
        return None
    if not (LOW_RES_SHAPE_CATALOG_MIN_AREA_RATIO <= match.area_ratio <= LOW_RES_SHAPE_CATALOG_MAX_AREA_RATIO):
        return None
    return match


def should_overlap_ocr_with_extraction(
    *,
    city_input: str | None,
    allow_catalog: bool,
    filename_hint: str | None = None,
) -> bool:
    if not allow_catalog:
        return True
    if city_input is None:
        return is_stale_only_catalog_hint(filename_hint)
    if is_stale_only_catalog_hint(city_input):
        return True
    return not has_active_catalog_area_hint(city_input)


def should_try_pre_ocr_catalog(
    *,
    city_input: str | None,
    allow_catalog: bool,
    filename_hint: str | None = None,
) -> bool:
    if not allow_catalog:
        return False
    if city_input is None:
        return not is_stale_only_catalog_hint(filename_hint)
    return has_active_catalog_area_hint(city_input)


def should_retry_pre_ocr_catalog(
    *,
    city_input: str | None,
    filename_hint: str | None,
    allow_pre_ocr_catalog: bool,
    used_catalog_scaled_extraction: bool,
    catalog_style_can_match: bool = True,
) -> bool:
    if not allow_pre_ocr_catalog or not used_catalog_scaled_extraction:
        return False
    if not catalog_style_can_match:
        return False
    if CATALOG_RETRY_EXTRACT_MAX_DIMENSION <= CATALOG_EXTRACT_MAX_DIMENSION:
        return False
    if CATALOG_RETRY_EXTRACT_MAX_DIMENSION >= CATALOG_MISS_REFINE_MAX_DIMENSION:
        return False
    return has_active_catalog_area_hint(city_input) or has_active_catalog_area_hint(filename_hint)


def is_stale_only_catalog_hint(text: str | None) -> bool:
    return has_stale_catalog_area_hint(text) and not has_active_catalog_area_hint(text)


def catalog_matching_enabled(options: Any) -> bool:
    return bool(getattr(options, "allow_catalog", True))


def high_confidence_label_texts(labels: list[Any]) -> list[str]:
    return [
        label.text
        for label in labels
        if label.text.strip() and label.confidence >= CATALOG_LABEL_HINT_MIN_CONFIDENCE
    ]


def finish_catalog_boundary_result(
    extraction,
    catalog_match,
    *,
    width: int,
    height: int,
    image_path: Path,
    city_input: str,
    output_path: Path,
    debug_path: Path | None,
    opts: BoundaryBuildOptions,
    rgb,
    progress: ProgressCallback | None,
    georeference_source: str = "catalog-shape-match",
) -> BoundaryBuildResult:
    data = catalog_feature_collection(
        extraction,
        catalog_match,
        width=width,
        height=height,
        image_path=image_path,
        city_input=city_input,
    )
    properties = data["features"][0]["properties"]
    properties["georeference_source"] = georeference_source
    combined_confidence = properties["combined_confidence"]
    emit_progress(
        progress,
        stage="georeference",
        message="Matched known service-area shape",
        percent=78,
        details={
            "source": georeference_source,
            "catalog_slug": catalog_match.entry.slug,
            "shape_iou": round(catalog_match.iou, 3),
            "combined_confidence": combined_confidence,
            "control_points": 0,
            "median_residual_m": 0.0,
            "p90_residual_m": 0.0,
        },
    )
    if combined_confidence < opts.min_confidence:
        raise ValueError(
            f"Combined confidence {combined_confidence:.2f} is below --min-confidence "
            f"{opts.min_confidence:.2f}. Provide a clearer map crop or lower the threshold."
        )
    return finish_boundary_result(
        data,
        extraction,
        image_path,
        output_path,
        debug_path,
        opts,
        width,
        height,
        city_input=city_input,
        rgb=rgb,
        progress=progress,
    )


def finish_boundary_result(
    data: dict[str, Any],
    extraction,
    image_path: Path,
    output_path: Path,
    debug_path: Path | None,
    opts: BoundaryBuildOptions,
    width: int,
    height: int,
    *,
    city_input: str,
    rgb,
    progress: ProgressCallback | None,
) -> BoundaryBuildResult:
    emit_progress(
        progress,
        stage="export",
        message="Writing GeoJSON and previews",
        percent=90,
    )
    write_geojson(data, output_path)
    mask_path: Path | None = None
    overlay_path: Path | None = None
    if debug_path:
        stem = output_path.stem
        overlay_path = debug_path / f"{stem}.overlay.png"
        if opts.write_mask_artifact:
            mask_path = debug_path / f"{stem}.mask.png"
            write_mask_png(extraction.mask, mask_path)
        write_overlay_png(
            image_path,
            extraction.mask,
            overlay_path,
            rgb=rgb,
            max_dimension=opts.preview_max_dimension,
        )

    summary = build_summary(
        data,
        output_path=output_path,
        city=city_input,
        width=width,
        height=height,
        mask_path=mask_path,
        overlay_path=overlay_path,
    )
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)
        (debug_path / f"{output_path.stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit_progress(
        progress,
        stage="complete",
        message="Boundary export ready",
        percent=100,
        status="complete",
        details=summary,
    )
    return BoundaryBuildResult(
        geojson=data,
        summary=summary,
        output_path=output_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
    )


def georeference_from_road_contexts(
    image_path: Path,
    pixel_geometry,
    road_context_candidates: list[str],
    *,
    rgb,
    progress: ProgressCallback | None,
):
    if road_context_candidates:
        emit_progress(
            progress,
            stage="georeference",
            message="Trying road-network context",
            percent=62,
            details={"candidates": road_context_candidates},
        )
    for candidate in road_context_candidates:
        georef = georeference_from_city_context(rgb, candidate, pixel_geometry)
        if georef is not None:
            return georef
    return None


def fit_georeference(
    labels: list[Any],
    image_path: Path,
    pixel_geometry,
    *,
    rgb,
    city_input: str | None,
    width: int,
    height: int,
    coverage_ratio: float,
    min_control_points: int,
    label_y_min: float | None,
    label_y_max: float | None,
    context_hints: list[CityContext] | None = None,
    road_feature_distance: Any | None = None,
    progress: ProgressCallback | None,
):
    emit_progress(
        progress,
        stage="georeference",
        message="Inferring map location from labels" if city_input is None else "Matching readable map labels",
        percent=48,
    )
    if city_input is None and context_hints:
        emit_progress(
            progress,
            stage="georeference",
            message="Trying filename map context",
            percent=52,
            details={"candidates": [context.query for context in context_hints]},
        )
        for context in context_hints:
            georef = georeference_from_labels(
                labels,
                str(image_path),
                context.query,
                width,
                height,
                rgb=rgb,
                min_control_points=min_control_points,
                label_y_min=label_y_min,
                label_y_max=label_y_max,
                road_feature_distance=road_feature_distance,
            )
            if is_credible_context_hint_georeference(georef):
                return georef

    road_contexts = road_contexts_from_labels(city_input, labels)
    road_context_candidates = [city_input] if city_input is not None else road_context_queries(road_contexts)
    try_ranked_context_first = label_y_min is None and should_try_ranked_context_first(
        city_input,
        coverage_ratio,
        road_contexts,
    )

    georef = None
    if try_ranked_context_first:
        emit_progress(
            progress,
            stage="georeference",
            message="Trying regional label context",
            percent=62,
            details={"candidates": road_context_candidates},
        )
        georef = georeference_from_ranked_label_contexts(
            labels,
            str(image_path),
            road_contexts,
            width,
            height,
            rgb=rgb,
            min_control_points=min_control_points,
            road_feature_distance=road_feature_distance,
        )

    if georef is None:
        georef = georeference_from_labels(
            labels,
            str(image_path),
            city_input,
            width,
            height,
            rgb=rgb,
            min_control_points=min_control_points,
            label_y_min=label_y_min,
            label_y_max=label_y_max,
            road_feature_distance=road_feature_distance,
        )

    if georef is None and road_context_candidates and road_network_context_fallback_enabled():
        georef = georeference_from_road_contexts(
            image_path,
            pixel_geometry,
            road_context_candidates,
            rgb=rgb,
            progress=progress,
        )
    return georef


def road_network_context_fallback_enabled() -> bool:
    value = os.environ.get(ROAD_NETWORK_CONTEXT_FALLBACK_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def georeference_from_ranked_label_contexts(
    labels: list[Any],
    image_path: str,
    contexts: list[CityContext],
    width: int,
    height: int,
    *,
    rgb: Any,
    min_control_points: int,
    road_feature_distance: Any | None = None,
):
    best = None
    best_score = -1.0
    for context in contexts:
        result = georeference_from_label_context(
            labels,
            image_path,
            context,
            width,
            height,
            rgb=rgb,
            min_control_points=min_control_points,
            road_feature_distance=road_feature_distance,
        )
        if result is None:
            continue
        score = result.transform.confidence + min(0.12, len(result.control_points) * 0.015)
        score -= min(0.2, result.residual_p90_m / 30000.0)
        if score > best_score:
            best = result
            best_score = score
        if (
            result.transform.confidence >= 0.80
            and len(result.control_points) >= 6
            and result.residual_p90_m <= 3500.0
        ):
            return result
    return best


def road_contexts_from_labels(city: str | None, labels: list[Any]) -> list[CityContext]:
    if city is not None:
        return []

    contexts = infer_city_contexts(labels)
    return rank_road_contexts(contexts)[:MAX_ROAD_CONTEXT_CANDIDATES]


def road_context_queries(contexts: list[CityContext]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        query = context.query.strip()
        if not query or query == "Inferred map area" or query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def should_try_ranked_context_first(
    city: str | None,
    coverage_ratio: float,
    contexts: list[CityContext],
) -> bool:
    if city is not None or not contexts:
        return False
    return coverage_ratio >= 0.08 and context_bbox_span_m(contexts[0]) >= 30000.0


def rank_road_context_queries(contexts: list[CityContext]) -> list[str]:
    return road_context_queries(rank_road_contexts(contexts))


def rank_road_contexts(contexts: list[CityContext]) -> list[CityContext]:
    scored: list[tuple[float, int, CityContext]] = []
    for index, context in enumerate(contexts):
        query = context.query.strip()
        if not query:
            continue
        scored.append((road_context_score(context), -index, context))

    ranked: list[CityContext] = []
    seen: set[tuple[str, int, int]] = set()
    for _, _, context in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True):
        key = (
            context.center.display_name.lower(),
            round(context.center.lon, 3),
            round(context.center.lat, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        ranked.append(context)
    return ranked


def road_context_score(context: CityContext) -> float:
    span_m = context_bbox_span_m(context)
    place_type = context.center.place_type.lower()
    display_name = context.center.display_name.lower()

    score = context.center.importance
    score += min(4.0, span_m / 18000.0)
    score += min(1.5, len(context.evidence) * 0.25)
    if place_type in {"region", "municipality", "borough"}:
        score += 1.35
    elif place_type in {"city", "town", "village"}:
        score += 0.55
    if span_m >= 30000.0:
        score += 1.25
    elif span_m < 9000.0:
        score -= 1.0
    if any(token in display_name for token in ("school", "district", "campus", "hospital")):
        score -= 1.75
    return score


def context_bbox_span_m(context: CityContext) -> float:
    bbox = context.center.bbox
    if bbox is None:
        return 0.0
    west, south, east, north = bbox
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    return float(max(abs(east_m - west_m), abs(north_m - south_m)))


def should_label_as_regional_area(bounds: tuple[float, float, float, float], city: str, georef) -> bool:
    if city == "Inferred map area":
        return False
    west, south, east, north = bounds
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    diagonal_m = hypot(east_m - west_m, north_m - south_m)
    if diagonal_m < 55000:
        return False
    distinct_places = {
        control.geocode.display_name.split(",", 1)[0].strip().lower()
        for control in georef.control_points
        if control.geocode.display_name
    }
    distinct_labels = {control.label.text.strip().lower() for control in georef.control_points if control.label.text}
    return len(distinct_places) >= 4 or len(distinct_labels) >= 4


def build_summary(
    data: dict[str, Any],
    *,
    output_path: Path,
    city: str,
    width: int,
    height: int,
    mask_path: Path | None,
    overlay_path: Path | None,
) -> dict[str, Any]:
    feature = data["features"][0]
    properties = feature["properties"]
    return {
        "output": str(output_path),
        "city_input": city,
        "city": properties["city"],
        "style": properties["style"],
        "image_width": width,
        "image_height": height,
        "coverage_ratio": properties["coverage_ratio"],
        "geometry_type": feature["geometry"]["type"],
        "bbox": properties["geodesic_bbox_lonlat"],
        "combined_confidence": properties["combined_confidence"],
        "extraction_confidence": properties["extraction_confidence"],
        "georeference_confidence": properties["georeference_confidence"],
        "georeference_source": properties["georeference_source"],
        "control_points": properties["georeference_control_points"],
        "catalog_slug": properties.get("catalog_slug"),
        "catalog_shape_iou": properties.get("catalog_shape_iou"),
        "catalog_shape_margin": properties.get("catalog_shape_margin"),
        "catalog_area_ratio": properties.get("catalog_area_ratio"),
        "rotation_degrees": properties["rotation_degrees"],
        "meters_per_pixel": properties["meters_per_pixel"],
        "median_residual_m": round(properties["georeference_residual_median_m"], 1),
        "p90_residual_m": round(properties["georeference_residual_p90_m"], 1),
        "road_match_score": properties.get("road_match_score"),
        "road_match_elapsed_s": properties.get("road_match_elapsed_s"),
        "mask": str(mask_path) if mask_path else None,
        "overlay": str(overlay_path) if overlay_path else None,
    }
