from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from dataclasses import dataclass
from math import hypot
import os
from pathlib import Path
from typing import Any, Callable

import cv2
from PIL import Image
from shapely.geometry import shape
from shapely.ops import transform

from .catalog_match import (
    CATALOG_LABEL_HINT_MIN_IOU,
    CATALOG_ROTATION_MAX_DEGREES,
    catalog_provider_hint,
    catalog_area_matches_text,
    catalog_match_from_score,
    catalog_style_supported,
    catalog_feature_collection,
    has_active_catalog_city_hint,
    has_active_catalog_area_hint,
    has_stale_catalog_area_hint,
    load_catalog_entries,
    match_catalog_entry,
    match_service_area_catalog_for_city_hint,
    match_service_area_catalog,
    normalize_catalog_area_tokens,
    PROVIDER_STYLES,
    score_catalog_entry,
    ServiceAreaCatalogMatch,
)
from .extract import (
    DEFAULT_SIMPLIFY_PX,
    classify_style,
    extract_service_area,
    extraction_scale_factor,
    load_rgb,
    load_rgb_at_max_dimension,
    rescale_extraction_result,
    write_overlay_image,
    write_mask_png,
)
from .georeference import (
    CityContext,
    filename_city_contexts,
    georeference_from_city_context,
    georeference_from_label_context,
    georeference_from_labels,
    georeference_result_with_city,
    infer_city_contexts,
    is_credible_context_hint_georeference,
    is_decisive_georeference_result,
    low_res_two_control_regional_fit_without_road_evidence,
    sparse_high_residual_fit_without_road_evidence,
)
from .georef_transform import lonlat_to_mercator
from .geojson import feature_collection, write_geojson
from .image_io import is_svg_image, normalize_image_for_processing
from .ocr import OcrLabel, extract_ocr_labels_from_rgb
from .osm_roads import image_feature_distance
from .runtime_config import (
    FAST_TEXT_OCR_FALLBACK_CONFIDENCE,
    FAST_TEXT_OCR_MIN_AREA,
    FAST_TEXT_OCR_STYLES,
    CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION,
    RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
    RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
    RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
    PROVIDER_UI_RAPIDOCR_MAX_DIMENSION,
    RAPIDOCR_MAX_DIMENSION,
    RAPIDOCR_PURPLE_FILL_MAX_DIMENSION,
)

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
PROVIDER_UI_LABEL_MIN_CONFIDENCE = 92.0
PROVIDER_UI_LABEL_MIN_IOU = 0.50
PROVIDER_UI_LABEL_MIN_AREA_RATIO = 0.55
PROVIDER_UI_LABEL_MAX_AREA_RATIO = 2.20
PROVIDER_UI_LABEL_CONFIDENCE = 0.72
PROVIDER_UI_FAST_OCR_STYLES = {"dark-teal", "gray-fill"}
PROVIDER_UI_FAST_OCR_TALL_SCREEN_STYLES = {"dark-teal"}
PROVIDER_UI_FAST_OCR_MIN_HEIGHT_WIDTH_RATIO = 1.25
PRE_EXTRACTION_FOCUS_OCR_STYLES = {"dark-teal"}
PRE_EXTRACTION_FOCUS_OCR_MIN_HEIGHT = 1200
PRE_EXTRACTION_FOCUS_OCR_MAX_ASPECT_RATIO = 1.35
FOCUSED_ADMIN_DISPLAY_PLACE_TYPES = {"city", "municipality", "town", "village"}
PROVIDER_UI_CROP_OCR_MAX_DIMENSION = max(
    0,
    int(os.environ.get("MAP_BOUNDARY_PROVIDER_UI_CROP_OCR_MAX_DIMENSION", "750")),
)
PROVIDER_UI_GRAY_FILL_CROP_OCR_MAX_DIMENSION = max(
    0,
    int(os.environ.get("MAP_BOUNDARY_PROVIDER_UI_GRAY_FILL_CROP_OCR_MAX_DIMENSION", "450")),
)
PROVIDER_UI_CROP_PAD_RATIO = 0.25
PROVIDER_UI_CROP_MIN_PAD_PX = 80.0
PROVIDER_UI_FOCUS_CROP_ENABLED = os.environ.get("MAP_BOUNDARY_PROVIDER_UI_FOCUS_CROP", "1") != "0"
PROVIDER_UI_FOCUS_CROP_STYLES = {"dark-teal"}
PROVIDER_UI_FOCUS_CROP_MIN_X_FRACTION = 0.10
PROVIDER_UI_FOCUS_CROP_MAX_X_FRACTION = 0.95
PROVIDER_UI_FOCUS_CROP_Y_PAD_RATIO = 0.05
FOCUS_GEOREF_OCR_STYLES = {"dark-teal"}
FOCUS_GEOREF_OCR_MAX_CROP_AREA_RATIO = max(
    0.0,
    float(os.environ.get("MAP_BOUNDARY_FOCUS_GEOREF_OCR_MAX_CROP_AREA_RATIO", "0.35")),
)
FOCUS_GEOREF_OCR_MAX_DIMENSION = max(
    0,
    int(os.environ.get("MAP_BOUNDARY_FOCUS_GEOREF_OCR_MAX_DIMENSION", "550")),
)
FOCUS_GEOREF_OCR_DETECTOR_LIMIT_SIDE_LEN = max(
    0,
    int(os.environ.get("MAP_BOUNDARY_FOCUS_GEOREF_OCR_DET_LIMIT_SIDE_LEN", "416")),
)
FOCUS_GEOREF_OCR_MIN_TEXT_AREA = max(
    0.0,
    float(os.environ.get("MAP_BOUNDARY_FOCUS_GEOREF_OCR_MIN_TEXT_AREA", "500")),
)
LOW_RES_SHAPE_CATALOG_MAX_IMAGE_DIMENSION = 520
LOW_RES_SHAPE_CATALOG_MIN_IOU = 0.94
LOW_RES_SHAPE_CATALOG_TINY_MAX_IMAGE_DIMENSION = 320
LOW_RES_SHAPE_CATALOG_TINY_MIN_IOU = 0.925
LOW_RES_SHAPE_CATALOG_MIN_MARGIN = 0.24
LOW_RES_SHAPE_CATALOG_MIN_AREA_RATIO = 0.92
LOW_RES_SHAPE_CATALOG_MAX_AREA_RATIO = 1.08
LOW_RES_SHAPE_CATALOG_MIN_EXTRACTION_CONFIDENCE = 0.98
LOW_RES_SHAPE_CATALOG_ROTATION_MIN_IOU = 0.92
FAST_TEXT_OCR_LOW_RES_RETRY_MAX_MIN_DIMENSION = 320
FILENAME_HINTED_AVRIDE_LIGHT_FILL_MIN_IOU = 0.92
FILENAME_HINTED_AVRIDE_LIGHT_FILL_MIN_MARGIN = 0.16
FILENAME_HINTED_AVRIDE_PROVIDER_ONLY_MIN_IOU = 0.90
FILENAME_HINTED_AVRIDE_PROVIDER_ONLY_MIN_MARGIN = 0.24
SPARSE_LABEL_CATALOG_MAX_DIMENSION = 180
SPARSE_LABEL_CATALOG_MAX_PIXELS = 20_000
SPARSE_LABEL_CATALOG_MIN_COVERAGE = 0.70
SPARSE_LABEL_CATALOG_MIN_CONFIDENCE = 0.55
SPARSE_LABEL_CATALOG_MIN_LABEL_CONFIDENCE = 85.0
SPARSE_LABEL_CATALOG_CONFIDENCE = 0.72
CATALOG_PROBE_MISS_LOW_IOU_THRESHOLD = 0.80
CATALOG_PROBE_NEAR_HIT_MIN_IOU = 0.86
CATALOG_PROBE_NEAR_HIT_MIN_AREA_RATIO = 0.90
CATALOG_PROBE_NEAR_HIT_MAX_AREA_RATIO = 1.12
CATALOG_PROBE_UNHINTED_NEAR_HIT_MIN_MARGIN = 0.24
POST_GEOREF_CATALOG_COMPLETION_MIN_IOU = 0.40
POST_GEOREF_CATALOG_COMPLETION_MIN_OUTPUT_COVERAGE = 0.84
POST_GEOREF_CATALOG_COMPLETION_MIN_CATALOG_COVERAGE = 0.40
POST_GEOREF_CATALOG_COMPLETION_MIN_AREA_RATIO = 0.40
POST_GEOREF_CATALOG_COMPLETION_MAX_AREA_RATIO = 1.25
POST_GEOREF_CATALOG_COMPLETION_MIN_GEOREF_CONFIDENCE = 0.80
POST_GEOREF_CATALOG_COMPLETION_CONFIDENCE = 0.84
OCR_DERIVED_CATALOG_SOURCES = {
    "current-verified-ocr-output",
    "verified-screenshot-ocr-output",
}
CURRENT_CATALOG_COMPLETION_SOURCES = {
    "current-external-service-area-reference",
    "current-verified-ocr-output",
}
CURRENT_CATALOG_LABEL_SHAPE_MIN_IOU = 0.70
CURRENT_CATALOG_LABEL_SHAPE_MIN_AREA_RATIO = 0.85
CURRENT_CATALOG_LABEL_SHAPE_MAX_AREA_RATIO = 1.15
CURRENT_CATALOG_LABEL_SHAPE_MIN_EXTRACTION_CONFIDENCE = 0.95
CURRENT_CATALOG_LABEL_SHAPE_CONFIDENCE = 0.84
AREA_HINTED_CURRENT_CATALOG_MIN_IOU = 0.95
AREA_HINTED_CURRENT_CATALOG_MAX_IOU_RELAXATION = 0.01
AREA_HINTED_CURRENT_CATALOG_MIN_MARGIN = 0.70
AREA_HINTED_CURRENT_CATALOG_MIN_AREA_RATIO = 0.98
AREA_HINTED_CURRENT_CATALOG_MAX_AREA_RATIO = 1.04
AREA_HINTED_CURRENT_CATALOG_RETRY_FIRST_MIN_VERTICES = 150
FILENAME_CURRENT_CATALOG_SHAPE_MIN_AREA_RATIO = 0.85
FILENAME_CURRENT_CATALOG_SHAPE_MAX_AREA_RATIO = 1.15
ROAD_NETWORK_CONTEXT_FALLBACK_ENV = "MAP_BOUNDARY_ENABLE_ROAD_CONTEXT_FALLBACK"
ROAD_FEATURE_PRECOMPUTE_ENV = "MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES"
RUNNER_OCR_CACHE_ENV = "MAP_BOUNDARY_RUNNER_OCR_CACHE"
EARLY_OCR_STYLE_MAX_DIMENSION = max(
    0,
    int(os.environ.get("MAP_BOUNDARY_EARLY_OCR_STYLE_MAX_DIMENSION", "800")),
)


@dataclass(frozen=True)
class BoundaryBuildOptions:
    simplify_px: float = DEFAULT_SIMPLIFY_PX
    min_confidence: float = 0.55
    min_control_points: int = 3
    preview_max_dimension: int | None = None
    overlay_format: str = "png"
    write_mask_artifact: bool = True
    allow_catalog: bool = True
    catalog_probe_only: bool = False
    catalog_probe_missed: bool = False
    catalog_probe_miss_low_iou: bool = False
    filename_hint: str | None = None


class CatalogProbeMiss(ValueError):
    """Raised when a catalog-only probe does not match a known service area."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


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


def extraction_progress_details(extraction: Any) -> dict[str, Any]:
    details = {
        "style": extraction.style,
        "coverage_ratio": round(extraction.coverage_ratio, 6),
        "contour_count": extraction.contour_count,
        "confidence": extraction.confidence,
    }
    scaled_cache_status = getattr(extraction, "scaled_cache_status", None)
    if scaled_cache_status is not None:
        details["scaled_cache"] = scaled_cache_status
        scaled_cache_shape = getattr(extraction, "scaled_cache_shape", None)
        if scaled_cache_shape is not None:
            details["scaled_cache_shape"] = list(scaled_cache_shape)
    return details


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
    would_try_pre_ocr_catalog = should_try_pre_ocr_catalog(
        city_input=city_input,
        allow_catalog=allow_catalog,
        filename_hint=filename_hint,
    )
    skip_redundant_probe = catalog_probe_missed_handoff_enabled(
        opts,
        city_input=city_input,
        filename_hint=filename_hint,
        allow_pre_ocr_catalog=would_try_pre_ocr_catalog,
    )
    allow_pre_ocr_catalog = not skip_redundant_probe and would_try_pre_ocr_catalog

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
    labels: list[Any] | None = None
    labels_future_filtered = False
    labels_future_current_catalog_shortcut = False
    labels_from_focus_georef_ocr = False
    provider_ui_labels_future: Future[list[Any]] | None = None
    provider_ui_fast_ocr_max_dimension: int | None = None
    ocr_executor: ThreadPoolExecutor | None = None
    road_feature_future: Future[Any] | None = None
    road_feature_executor: ThreadPoolExecutor | None = None
    georef_resource_future: Future[Any] | None = None
    georef_resource_executor: ThreadPoolExecutor | None = None
    catalog_probe_miss_extraction: Any | None = None

    def ensure_georeference_resource_preload() -> None:
        nonlocal georef_resource_future, georef_resource_executor
        if georef_resource_future is not None:
            return
        georef_resource_executor = ThreadPoolExecutor(max_workers=1)
        georef_resource_future = georef_resource_executor.submit(preload_georeference_resources)

    try:
        emit_progress(
            progress,
            stage="extract",
            message="Extracting service-area pixels",
            percent=18,
            details={"width": width, "height": height},
        )
        low_res_catalog_rgb = should_load_low_res_catalog_rgb(
            city_input=city_input,
            filename_hint=filename_hint,
            allow_pre_ocr_catalog=allow_pre_ocr_catalog,
        )
        catalog_extract_max_dimension = initial_catalog_extract_max_dimension(
            city_input=city_input,
            filename_hint=filename_hint,
            allow_pre_ocr_catalog=allow_pre_ocr_catalog,
        )
        rgb = (
            load_rgb_at_max_dimension(image_path, catalog_extract_max_dimension)
            if low_res_catalog_rgb
            else load_rgb(image_path)
        )
        low_res_catalog_scale = (
            max(rgb.shape[1] / max(width, 1), rgb.shape[0] / max(height, 1))
            if low_res_catalog_rgb
            else 1.0
        )

        def ensure_full_rgb() -> None:
            nonlocal rgb, low_res_catalog_rgb
            if low_res_catalog_rgb:
                rgb = load_rgb(image_path)
                low_res_catalog_rgb = False

        early_ocr_style: str | None = None
        if should_overlap_ocr_with_extraction(
            city_input=city_input,
            allow_catalog=allow_catalog,
            filename_hint=filename_hint,
        ):
            early_ocr_style = classify_style_for_ocr(rgb)
            if not should_defer_pre_extraction_ocr_for_focus(
                early_ocr_style,
                city_input=city_input,
                allow_catalog=allow_catalog,
                width=width,
                height=height,
            ):
                ocr_executor = ThreadPoolExecutor(max_workers=1)
                rapidocr_min_text_area = fast_text_ocr_min_area_for_style(early_ocr_style)
                labels_future_filtered = rapidocr_min_text_area is not None
                ocr_kwargs: dict[str, Any] = {"cache": runner_ocr_cache_enabled()}
                if current_catalog_label_shape_shortcut_enabled(
                    city_input=city_input,
                    allow_catalog=allow_catalog,
                    skip_redundant_probe=skip_redundant_probe,
                ):
                    ocr_kwargs["rapidocr_max_dimension"] = CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION
                    labels_future_current_catalog_shortcut = True
                if rapidocr_min_text_area is not None:
                    ocr_kwargs["rapidocr_min_text_area"] = rapidocr_min_text_area
                rapidocr_detector_limit = rapidocr_detector_limit_for_ocr_style(early_ocr_style)
                if rapidocr_detector_limit is not None:
                    ocr_kwargs["rapidocr_detector_limit_side_len"] = rapidocr_detector_limit
                rapidocr_detector_limit_type = rapidocr_detector_limit_type_for_ocr_style(early_ocr_style)
                if rapidocr_detector_limit_type is not None:
                    ocr_kwargs["rapidocr_detector_limit_type"] = rapidocr_detector_limit_type
                rapidocr_recognition_profile = rapidocr_recognition_profile_for_ocr_style(early_ocr_style)
                if rapidocr_recognition_profile is not None:
                    ocr_kwargs["rapidocr_recognition_profile"] = rapidocr_recognition_profile
                labels_future = ocr_executor.submit(
                    extract_ocr_labels_from_rgb,
                    str(image_path),
                    rgb,
                    **ocr_kwargs,
                )
                ensure_georeference_resource_preload()
        if labels_future is None and should_overlap_probe_miss_ocr(
            skip_redundant_probe=skip_redundant_probe,
            city_input=city_input,
            filename_hint=filename_hint,
            catalog_probe_miss_low_iou=catalog_probe_miss_low_iou_enabled(opts),
        ):
            if early_ocr_style is None:
                early_ocr_style = classify_style_for_ocr(rgb)
            ocr_executor = ThreadPoolExecutor(max_workers=1)
            rapidocr_min_text_area = fast_text_ocr_min_area_for_style(early_ocr_style)
            labels_future_filtered = rapidocr_min_text_area is not None
            ocr_kwargs = {"cache": runner_ocr_cache_enabled()}
            if current_catalog_label_shape_shortcut_enabled(
                city_input=city_input,
                allow_catalog=allow_catalog,
                skip_redundant_probe=skip_redundant_probe,
            ):
                ocr_kwargs["rapidocr_max_dimension"] = CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION
                labels_future_current_catalog_shortcut = True
            if rapidocr_min_text_area is not None:
                ocr_kwargs["rapidocr_min_text_area"] = rapidocr_min_text_area
            rapidocr_detector_limit = rapidocr_detector_limit_for_ocr_style(early_ocr_style)
            if rapidocr_detector_limit is not None:
                ocr_kwargs["rapidocr_detector_limit_side_len"] = rapidocr_detector_limit
            rapidocr_detector_limit_type = rapidocr_detector_limit_type_for_ocr_style(early_ocr_style)
            if rapidocr_detector_limit_type is not None:
                ocr_kwargs["rapidocr_detector_limit_type"] = rapidocr_detector_limit_type
            rapidocr_recognition_profile = rapidocr_recognition_profile_for_ocr_style(early_ocr_style)
            if rapidocr_recognition_profile is not None:
                ocr_kwargs["rapidocr_recognition_profile"] = rapidocr_recognition_profile
            labels_future = ocr_executor.submit(
                extract_ocr_labels_from_rgb,
                str(image_path),
                rgb,
                **ocr_kwargs,
            )
            ensure_georeference_resource_preload()
        extraction_max_dimension = catalog_extract_max_dimension if allow_pre_ocr_catalog else (
            CATALOG_MISS_REFINE_MAX_DIMENSION if skip_redundant_probe else GENERAL_EXTRACT_MAX_DIMENSION
        )
        used_catalog_scaled_extraction = (
            allow_pre_ocr_catalog
            and (low_res_catalog_rgb or extraction_scale_factor(rgb, extraction_max_dimension) < 1.0)
        )
        extraction = extract_service_area(
            image_path,
            simplify_px=opts.simplify_px * low_res_catalog_scale if low_res_catalog_rgb else opts.simplify_px,
            rgb=rgb,
            max_dimension=0 if low_res_catalog_rgb else extraction_max_dimension,
            cache=not allow_pre_ocr_catalog,
        )
        if low_res_catalog_rgb:
            extraction = rescale_extraction_result(
                extraction,
                width=width,
                height=height,
                scale=low_res_catalog_scale,
            )
        catalog_probe_miss_extraction = extraction
        emit_progress(
            progress,
            stage="extract",
            message="Pixel polygon extracted",
            percent=36,
            details=extraction_progress_details(extraction),
        )

        catalog_style_can_match = catalog_style_supported(extraction.style)
        provider_ui_fast_ocr_max_dimension = provider_ui_fast_ocr_max_dimension_for_style(
            extraction.style,
            width=width,
            height=height,
        )
        if allow_pre_ocr_catalog and catalog_style_can_match:
            catalog_match, catalog_match_source = hinted_catalog_shape_match(
                extraction.pixel_geometry,
                style=extraction.style,
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
                    georeference_source=catalog_match_source or "catalog-shape-match",
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
            if not catalog_probe_only_enabled(opts):
                catalog_match = filename_hinted_current_catalog_shape_match(
                    extraction,
                    city_input=city_input,
                    filename_hint=filename_hint,
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
                        georeference_source="catalog-shape-match:filename-shape",
                    )
                catalog_match = filename_hinted_current_catalog_near_hit_match(
                    extraction,
                    city_input=city_input,
                    filename_hint=filename_hint,
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
                        georeference_source="catalog-shape-match:filename-near-hit",
                    )
            catalog_match = filename_hinted_avride_light_fill_catalog_match(
                extraction,
                filename_hint=filename_hint,
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
                    georeference_source="catalog-shape-match:filename-hint",
                )

        if should_retry_pre_ocr_catalog(
            city_input=city_input,
            filename_hint=filename_hint,
            allow_pre_ocr_catalog=allow_pre_ocr_catalog,
            used_catalog_scaled_extraction=used_catalog_scaled_extraction,
            initial_extract_max_dimension=extraction_max_dimension,
            catalog_style_can_match=catalog_style_can_match,
            catalog_probe_only=catalog_probe_only_enabled(opts),
        ):
            emit_progress(
                progress,
                stage="extract",
                message="Retrying known service-area shape",
                percent=37,
                details={"width": width, "height": height},
            )
            ensure_full_rgb()
            retry_extraction = extract_service_area(
                image_path,
                simplify_px=opts.simplify_px,
                rgb=rgb,
                max_dimension=CATALOG_RETRY_EXTRACT_MAX_DIMENSION,
                cache=False,
            )
            catalog_probe_miss_extraction = retry_extraction
            catalog_match, catalog_match_source = hinted_catalog_shape_match(
                retry_extraction.pixel_geometry,
                style=retry_extraction.style,
                city_input=city_input,
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
                    georeference_source=catalog_match_source or "catalog-shape-match:retry",
                )

        if skip_redundant_probe and allow_catalog and catalog_style_can_match:
            catalog_match, catalog_match_source = hinted_catalog_shape_match(
                extraction.pixel_geometry,
                style=extraction.style,
                city_input=city_input,
            )
            if catalog_match is None:
                catalog_match = filename_hinted_current_catalog_shape_match(
                    extraction,
                    city_input=city_input,
                    filename_hint=filename_hint,
                )
                catalog_match_source = (
                    "catalog-shape-match:filename-shape" if catalog_match is not None else None
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
                    georeference_source=catalog_match_source or "catalog-shape-match:probe-miss-full",
                )

        if catalog_probe_only_enabled(opts):
            catalog_probe_match = catalog_probe_near_hit_match(
                catalog_probe_miss_extraction or extraction,
                city_input=city_input,
                filename_hint=filename_hint,
            )
            if catalog_probe_match is not None:
                return finish_catalog_boundary_result(
                    catalog_probe_miss_extraction or extraction,
                    catalog_probe_match,
                    width=width,
                    height=height,
                    image_path=image_path,
                    city_input=city_input or "Auto",
                    output_path=output_path,
                    debug_path=debug_path,
                    opts=opts,
                    rgb=rgb,
                    progress=progress,
                    georeference_source="catalog-shape-match:probe-near-hit",
                )
            raise CatalogProbeMiss(
                "No known service-area shape matched the catalog probe.",
                details=catalog_probe_miss_details(
                    catalog_probe_miss_extraction or extraction,
                    city_input=city_input,
                    filename_hint=filename_hint,
                ),
            )

        if used_catalog_scaled_extraction:
            ensure_full_rgb()
            if labels_future is None:
                ocr_executor = ThreadPoolExecutor(max_workers=1)
                provider_ui_crop_after_refine = (
                    city_input is None
                    and allow_catalog
                    and provider_ui_fast_ocr_max_dimension is not None
                    and PROVIDER_UI_CROP_OCR_MAX_DIMENSION > 0
                )
                focus_georef_ocr_after_refine = (
                    not low_res_catalog_rgb
                    and provider_ui_fast_ocr_max_dimension is None
                    and focus_georef_ocr_enabled(extraction, rgb=rgb, city_input=city_input)
                )
                if (
                    city_input is None
                    and allow_catalog
                    and provider_ui_fast_ocr_max_dimension is not None
                    and PROVIDER_UI_CROP_OCR_MAX_DIMENSION <= 0
                ):
                    provider_ui_labels_future = ocr_executor.submit(
                        extract_provider_ui_labels_from_rgb,
                        str(image_path),
                        rgb,
                        extraction=extraction,
                        rapidocr_max_dimension=provider_ui_fast_ocr_max_dimension,
                    )
                elif provider_ui_crop_after_refine:
                    pass
                elif focus_georef_ocr_after_refine:
                    pass
                else:
                    labels_future_filtered = fast_text_ocr_min_area_for_style(extraction.style) is not None
                    shortcut_ocr = current_catalog_label_shape_shortcut_enabled(
                        city_input=city_input,
                        allow_catalog=allow_catalog,
                        skip_redundant_probe=skip_redundant_probe,
                    )
                    labels_future = submit_ocr_labels_from_rgb(
                        ocr_executor,
                        image_path,
                        rgb,
                        style=extraction.style,
                        rapidocr_max_dimension_override=(
                            CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION if shortcut_ocr else None
                        ),
                    )
                    labels_future_current_catalog_shortcut = shortcut_ocr
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
                details=extraction_progress_details(extraction),
            )
            if allow_pre_ocr_catalog and catalog_style_supported(extraction.style):
                catalog_match, catalog_match_source = hinted_catalog_shape_match(
                    extraction.pixel_geometry,
                    style=extraction.style,
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
                        georeference_source=catalog_match_source or "catalog-shape-match",
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
                catalog_match = filename_hinted_avride_light_fill_catalog_match(
                    extraction,
                    filename_hint=filename_hint,
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
                        georeference_source="catalog-shape-match:filename-hint",
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
        if (
            labels is None
            and labels_future is None
            and provider_ui_fast_ocr_max_dimension is None
            and focus_georef_ocr_enabled(extraction, rgb=rgb, city_input=city_input)
        ):
            ensure_georeference_resource_preload()
            emit_progress(
                progress,
                stage="ocr",
                message="Reading focused map labels",
                percent=43,
                details={
                    "crop_area_ratio": round(focus_georef_ocr_crop_area_ratio(extraction, rgb=rgb), 4),
                    "rapidocr_max_dimension": focus_georef_ocr_max_dimension_for_style(extraction.style),
                    "rapidocr_detector_limit_side_len": focus_georef_ocr_detector_limit_for_style(extraction.style),
                    "rapidocr_min_text_area": focus_georef_ocr_min_text_area_for_style(extraction.style),
                },
            )
            labels = extract_focus_georef_labels_from_rgb(str(image_path), rgb, extraction=extraction)
            labels_from_focus_georef_ocr = True
            emit_progress(
                progress,
                stage="ocr",
                message="Focused map labels read",
                percent=47,
                details={
                    "label_count": len(labels),
                    "top_labels": [label.text for label in labels[:8]],
                },
            )
        if (
            labels is None
            and labels_future is None
            and city_input is None
            and allow_catalog
            and provider_ui_fast_ocr_max_dimension is not None
        ):
            provider_ui_match = None
            provider_ui_labels: list[Any] = []
            if provider_ui_focus_crop_enabled(extraction):
                emit_progress(
                    progress,
                    stage="ocr",
                    message="Reading focused provider area labels",
                    percent=42,
                    details={
                        "rapidocr_max_dimension": provider_ui_fast_ocr_max_dimension,
                        "crop_rapidocr_max_dimension": provider_ui_crop_ocr_max_dimension_for_style(
                            extraction.style,
                            rapidocr_max_dimension=provider_ui_fast_ocr_max_dimension,
                        ),
                    },
                )
                provider_ui_labels = extract_provider_ui_labels_from_rgb(
                    str(image_path),
                    rgb,
                    extraction=extraction,
                    rapidocr_max_dimension=provider_ui_fast_ocr_max_dimension,
                    focus=True,
                )
                provider_ui_match = provider_ui_label_catalog_match(extraction, provider_ui_labels)
                emit_progress(
                    progress,
                    stage="ocr",
                    message="Focused provider area labels read",
                    percent=44,
                    details={
                        "label_count": len(provider_ui_labels),
                        "top_labels": [label.text for label in provider_ui_labels[:8]],
                        "matched_catalog": provider_ui_match.entry.slug if provider_ui_match is not None else None,
                    },
                )
                if provider_ui_match is not None:
                    return finish_catalog_boundary_result(
                        extraction,
                        provider_ui_match,
                        width=width,
                        height=height,
                        image_path=image_path,
                        city_input="Auto",
                        output_path=output_path,
                        debug_path=debug_path,
                        opts=opts,
                        rgb=rgb,
                        progress=progress,
                        georeference_source="catalog-shape-match:provider-ui-focus-label",
                    )
            emit_progress(
                progress,
                stage="ocr",
                message="Reading provider area labels",
                percent=43,
                details={
                    "rapidocr_max_dimension": provider_ui_fast_ocr_max_dimension,
                    "crop_rapidocr_max_dimension": provider_ui_crop_ocr_max_dimension_for_style(
                        extraction.style,
                        rapidocr_max_dimension=provider_ui_fast_ocr_max_dimension,
                    ),
                },
            )
            if provider_ui_labels_future is None:
                provider_ui_labels = extract_provider_ui_labels_from_rgb(
                    str(image_path),
                    rgb,
                    extraction=extraction,
                    rapidocr_max_dimension=provider_ui_fast_ocr_max_dimension,
                )
            else:
                provider_ui_labels = provider_ui_labels_future.result()
                provider_ui_labels_future = None
            provider_ui_match = provider_ui_label_catalog_match(extraction, provider_ui_labels)
            emit_progress(
                progress,
                stage="ocr",
                message="Provider area labels read",
                percent=46,
                details={
                    "label_count": len(provider_ui_labels),
                    "top_labels": [label.text for label in provider_ui_labels[:8]],
                    "matched_catalog": provider_ui_match.entry.slug if provider_ui_match is not None else None,
                },
            )
            if provider_ui_match is not None:
                return finish_catalog_boundary_result(
                    extraction,
                    provider_ui_match,
                    width=width,
                    height=height,
                    image_path=image_path,
                    city_input="Auto",
                    output_path=output_path,
                    debug_path=debug_path,
                    opts=opts,
                    rgb=rgb,
                    progress=progress,
                    georeference_source="catalog-shape-match:provider-ui-label",
                )
        if labels is None and labels_future is None:
            ensure_full_rgb()
            if ocr_executor is None:
                ocr_executor = ThreadPoolExecutor(max_workers=1)
            labels_future_filtered = fast_text_ocr_min_area_for_style(extraction.style) is not None
            labels_future = submit_ocr_labels_from_rgb(
                ocr_executor,
                image_path,
                rgb,
                style=extraction.style,
                rapidocr_max_dimension_override=(
                    CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION
                    if current_catalog_label_shape_shortcut_enabled(
                        city_input=city_input,
                        allow_catalog=allow_catalog,
                        skip_redundant_probe=skip_redundant_probe,
                    )
                    else None
                ),
            )
            labels_future_current_catalog_shortcut = current_catalog_label_shape_shortcut_enabled(
                city_input=city_input,
                allow_catalog=allow_catalog,
                skip_redundant_probe=skip_redundant_probe,
            )
            ensure_georeference_resource_preload()
        if should_precompute_road_features(extraction.style, width, height):
            road_feature_executor = ThreadPoolExecutor(max_workers=1)
            road_feature_future = road_feature_executor.submit(image_feature_distance, rgb)
        if labels is None:
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
    if road_feature_distance is None and road_feature_future is not None and not road_feature_future.cancelled():
        road_feature_distance = road_feature_future
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
    if labels_future_filtered and fast_text_ocr_min_area_for_style(extraction.style) is None:
        labels = extract_full_ocr_labels_for_style(image_path, rgb, style=extraction.style)
        labels_future_filtered = False
    if city_input is None and allow_catalog:
        label_shape_match = current_catalog_label_shape_match(extraction, labels)
        if label_shape_match is not None:
            return finish_catalog_boundary_result(
                extraction,
                label_shape_match,
                width=width,
                height=height,
                image_path=image_path,
                city_input="Auto",
                output_path=output_path,
                debug_path=debug_path,
                opts=opts,
                rgb=rgb,
                progress=progress,
                georeference_source="catalog-shape-match:label-shape",
                catalog_label_hints=high_confidence_label_texts(labels)[:5],
            )
    if labels_future_current_catalog_shortcut:
        emit_progress(
            progress,
            stage="ocr",
            message="Retrying map labels at full detail",
            percent=47,
        )
        labels = extract_full_ocr_labels_for_style(image_path, rgb, style=extraction.style)
        labels_future_filtered = False
        labels_future_current_catalog_shortcut = False
        emit_progress(
            progress,
            stage="ocr",
            message="Full-detail map labels read",
            percent=48,
            details={
                "label_count": len(labels),
                "top_labels": [label.text for label in labels[:8]],
            },
        )
        if city_input is None and allow_catalog:
            label_shape_match = current_catalog_label_shape_match(extraction, labels)
            if label_shape_match is not None:
                return finish_catalog_boundary_result(
                    extraction,
                    label_shape_match,
                    width=width,
                    height=height,
                    image_path=image_path,
                    city_input="Auto",
                    output_path=output_path,
                    debug_path=debug_path,
                    opts=opts,
                    rgb=rgb,
                    progress=progress,
                    georeference_source="catalog-shape-match:label-shape",
                    catalog_label_hints=high_confidence_label_texts(labels)[:5],
                )
    if city_input is None and allow_catalog:
        provider_ui_match = provider_ui_label_catalog_match(extraction, labels)
        if provider_ui_match is not None:
            return finish_catalog_boundary_result(
                extraction,
                provider_ui_match,
                width=width,
                height=height,
                image_path=image_path,
                city_input="Auto",
                output_path=output_path,
                debug_path=debug_path,
                opts=opts,
                rgb=rgb,
                progress=progress,
                georeference_source="catalog-shape-match:provider-ui-label",
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
    if city_input is None and allow_catalog:
        sparse_catalog_match = sparse_low_res_label_catalog_match(
            extraction,
            labels,
            width=width,
            height=height,
        )
        if sparse_catalog_match is not None:
            return finish_catalog_boundary_result(
                extraction,
                sparse_catalog_match,
                width=width,
                height=height,
                image_path=image_path,
                city_input="Auto",
                output_path=output_path,
                debug_path=debug_path,
                opts=opts,
                rgb=rgb,
                progress=progress,
                georeference_source="catalog-label-match:sparse-low-res",
                catalog_label_hints=high_confidence_label_texts(labels)[:5],
                shape_match=False,
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
        anchor_marker_dots=should_anchor_marker_dots(extraction.style),
        style=extraction.style,
        allow_credible_cached_fit=labels_from_focus_georef_ocr,
        progress=progress,
    )
    if labels_from_focus_georef_ocr:
        georef = focused_georef_with_admin_control_city(georef)
    if should_fallback_focus_georef_ocr(labels_from_focus_georef_ocr, georef):
        emit_progress(
            progress,
            stage="ocr",
            message="Retrying map labels at full detail",
            percent=47,
        )
        labels = extract_full_ocr_labels_for_style(image_path, rgb, style=extraction.style)
        labels_from_focus_georef_ocr = False
        emit_progress(
            progress,
            stage="ocr",
            message="Full-detail map labels read",
            percent=48,
            details={
                "label_count": len(labels),
                "top_labels": [label.text for label in labels[:8]],
            },
        )
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
            anchor_marker_dots=should_anchor_marker_dots(extraction.style),
            style=extraction.style,
            allow_credible_cached_fit=False,
            progress=progress,
        )
    if should_fallback_fast_text_ocr(
        labels_future_filtered,
        georef,
        style=extraction.style,
        width=width,
        height=height,
    ):
        emit_progress(
            progress,
            stage="ocr",
            message="Retrying map labels at full detail",
            percent=47,
        )
        labels = extract_full_ocr_labels_for_style(image_path, rgb, style=extraction.style)
        labels_future_filtered = False
        emit_progress(
            progress,
            stage="ocr",
            message="Full-detail map labels read",
            percent=48,
            details={
                "label_count": len(labels),
                "top_labels": [label.text for label in labels[:8]],
            },
        )
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
            anchor_marker_dots=should_anchor_marker_dots(extraction.style),
            style=extraction.style,
            allow_credible_cached_fit=False,
            progress=progress,
        )
    if georef is None:
        raise ValueError(
            "Could not infer a reliable map location and georeference from OCR/geocoded map labels. "
            "Provide a higher-resolution map crop with readable city or neighborhood labels."
        )
    if sparse_ocr_georeference_lacks_support(georef, width=width, height=height):
        raise ValueError(
            "Could not infer a reliable map location and georeference from sparse OCR labels. "
            "Provide a higher-resolution map crop with more readable place labels."
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

    if allow_catalog:
        catalog_completion_match = post_georeference_catalog_completion_match(
            extraction,
            labels,
            geom,
            city_input=city_input,
            filename_hint=filename_hint,
            georef_confidence=geo_transform.confidence,
        )
        if catalog_completion_match is not None:
            return finish_catalog_boundary_result(
                extraction,
                catalog_completion_match,
                width=width,
                height=height,
                image_path=image_path,
                city_input=city_input or "Auto",
                output_path=output_path,
                debug_path=debug_path,
                opts=opts,
                rgb=rgb,
                progress=progress,
                georeference_source="catalog-shape-match:georef-contained",
                catalog_label_hints=high_confidence_label_texts(labels)[:5],
            )

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
    return road_feature_precompute_enabled() and style == "bright-blue" and max(width, height) >= 1000


def road_feature_precompute_enabled() -> bool:
    value = os.environ.get(ROAD_FEATURE_PRECOMPUTE_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


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
    rapidocr_max_dimension_override: int | None = None,
) -> Future[list[Any]]:
    rapidocr_max_dimension = (
        rapidocr_max_dimension_override
        if rapidocr_max_dimension_override is not None
        else rapidocr_max_dimension_for_extraction_style(style)
    )
    rapidocr_min_text_area = fast_text_ocr_min_area_for_style(style)
    kwargs: dict[str, Any] = {"cache": runner_ocr_cache_enabled()}
    if rapidocr_max_dimension is not None:
        kwargs["rapidocr_max_dimension"] = rapidocr_max_dimension
    rapidocr_detector_limit = rapidocr_detector_limit_for_ocr_style(style)
    if rapidocr_detector_limit is not None:
        kwargs["rapidocr_detector_limit_side_len"] = rapidocr_detector_limit
    rapidocr_detector_limit_type = rapidocr_detector_limit_type_for_ocr_style(style)
    if rapidocr_detector_limit_type is not None:
        kwargs["rapidocr_detector_limit_type"] = rapidocr_detector_limit_type
    rapidocr_recognition_profile = rapidocr_recognition_profile_for_ocr_style(style)
    if rapidocr_recognition_profile is not None:
        kwargs["rapidocr_recognition_profile"] = rapidocr_recognition_profile
    if rapidocr_min_text_area is not None:
        kwargs["rapidocr_min_text_area"] = rapidocr_min_text_area
    return executor.submit(
        extract_ocr_labels_from_rgb,
        str(image_path),
        rgb,
        **kwargs,
    )


def extract_provider_ui_labels_from_rgb(
    image_path: str | Path,
    rgb,
    *,
    extraction,
    rapidocr_max_dimension: int | None,
    focus: bool = False,
) -> list[Any]:
    if focus:
        crop, offset_x, offset_y = provider_ui_focus_ocr_crop(rgb, extraction.pixel_geometry.bounds)
    else:
        crop, offset_x, offset_y = provider_ui_ocr_crop(rgb, extraction.pixel_geometry.bounds)
    crop_max_dimension = provider_ui_crop_ocr_max_dimension_for_style(
        extraction.style,
        rapidocr_max_dimension=rapidocr_max_dimension,
    )
    labels = extract_ocr_labels_from_rgb(
        str(image_path),
        crop,
        rapidocr_max_dimension=crop_max_dimension,
        cache=runner_ocr_cache_enabled(),
    )
    if offset_x == 0 and offset_y == 0:
        return labels
    return [
        OcrLabel(
            text=label.text,
            x=label.x + offset_x,
            y=label.y + offset_y,
            width=label.width,
            height=label.height,
            confidence=label.confidence,
        )
        for label in labels
    ]


def extract_focus_georef_labels_from_rgb(
    image_path: str | Path,
    rgb,
    *,
    extraction,
) -> list[Any]:
    crop, offset_x, offset_y = provider_ui_focus_ocr_crop(rgb, extraction.pixel_geometry.bounds)
    ocr_kwargs = ocr_kwargs_for_style(extraction.style, cache=runner_ocr_cache_enabled())
    focus_max_dimension = focus_georef_ocr_max_dimension_for_style(extraction.style)
    if focus_max_dimension is not None:
        ocr_kwargs["rapidocr_max_dimension"] = focus_max_dimension
    focus_detector_limit = focus_georef_ocr_detector_limit_for_style(extraction.style)
    if focus_detector_limit is not None:
        ocr_kwargs["rapidocr_detector_limit_side_len"] = focus_detector_limit
    focus_min_text_area = focus_georef_ocr_min_text_area_for_style(extraction.style)
    if focus_min_text_area is not None:
        ocr_kwargs["rapidocr_min_text_area"] = focus_min_text_area
    ocr_kwargs["allow_tesseract_fallback"] = False
    labels = extract_ocr_labels_from_rgb(str(image_path), crop, **ocr_kwargs)
    if offset_x == 0 and offset_y == 0:
        return labels
    return [
        OcrLabel(
            text=label.text,
            x=label.x + offset_x,
            y=label.y + offset_y,
            width=label.width,
            height=label.height,
            confidence=label.confidence,
        )
        for label in labels
    ]


def ocr_kwargs_for_style(style: str | None, *, cache: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"cache": cache}
    rapidocr_max_dimension = rapidocr_max_dimension_for_extraction_style(style)
    if rapidocr_max_dimension is not None:
        kwargs["rapidocr_max_dimension"] = rapidocr_max_dimension
    rapidocr_detector_limit = rapidocr_detector_limit_for_ocr_style(style)
    if rapidocr_detector_limit is not None:
        kwargs["rapidocr_detector_limit_side_len"] = rapidocr_detector_limit
    rapidocr_detector_limit_type = rapidocr_detector_limit_type_for_ocr_style(style)
    if rapidocr_detector_limit_type is not None:
        kwargs["rapidocr_detector_limit_type"] = rapidocr_detector_limit_type
    rapidocr_recognition_profile = rapidocr_recognition_profile_for_ocr_style(style)
    if rapidocr_recognition_profile is not None:
        kwargs["rapidocr_recognition_profile"] = rapidocr_recognition_profile
    rapidocr_min_text_area = fast_text_ocr_min_area_for_style(style)
    if rapidocr_min_text_area is not None:
        kwargs["rapidocr_min_text_area"] = rapidocr_min_text_area
    return kwargs


def focus_georef_ocr_enabled(extraction, *, rgb, city_input: str | None) -> bool:
    return (
        city_input is None
        and extraction.style in FOCUS_GEOREF_OCR_STYLES
        and provider_ui_focus_crop_enabled(extraction)
        and 0.0 < focus_georef_ocr_crop_area_ratio(extraction, rgb=rgb) <= FOCUS_GEOREF_OCR_MAX_CROP_AREA_RATIO
    )


def focus_georef_ocr_max_dimension_for_style(style: str | None) -> int | None:
    if style not in FOCUS_GEOREF_OCR_STYLES:
        return rapidocr_max_dimension_for_extraction_style(style)
    if FOCUS_GEOREF_OCR_MAX_DIMENSION <= 0 or RAPIDOCR_MAX_DIMENSION <= 0:
        return rapidocr_max_dimension_for_extraction_style(style)
    if FOCUS_GEOREF_OCR_MAX_DIMENSION >= RAPIDOCR_MAX_DIMENSION:
        return rapidocr_max_dimension_for_extraction_style(style)
    return FOCUS_GEOREF_OCR_MAX_DIMENSION


def focus_georef_ocr_detector_limit_for_style(style: str | None) -> int | None:
    if style not in FOCUS_GEOREF_OCR_STYLES:
        return rapidocr_detector_limit_for_ocr_style(style)
    if FOCUS_GEOREF_OCR_DETECTOR_LIMIT_SIDE_LEN <= 0:
        return rapidocr_detector_limit_for_ocr_style(style)
    return FOCUS_GEOREF_OCR_DETECTOR_LIMIT_SIDE_LEN


def focus_georef_ocr_min_text_area_for_style(style: str | None) -> float | None:
    if style not in FOCUS_GEOREF_OCR_STYLES:
        return fast_text_ocr_min_area_for_style(style)
    if FOCUS_GEOREF_OCR_MIN_TEXT_AREA <= 0.0:
        return fast_text_ocr_min_area_for_style(style)
    return FOCUS_GEOREF_OCR_MIN_TEXT_AREA


def focus_georef_ocr_crop_area_ratio(extraction, *, rgb) -> float:
    height, width = rgb.shape[:2]
    image_area = max(float(width * height), 1.0)
    crop, _offset_x, _offset_y = provider_ui_focus_ocr_crop(rgb, extraction.pixel_geometry.bounds)
    crop_height, crop_width = crop.shape[:2]
    return float(crop_width * crop_height) / image_area


def provider_ui_focus_crop_enabled(extraction) -> bool:
    return (
        PROVIDER_UI_FOCUS_CROP_ENABLED
        and PROVIDER_UI_CROP_OCR_MAX_DIMENSION > 0
        and extraction.style in PROVIDER_UI_FOCUS_CROP_STYLES
    )


def provider_ui_crop_ocr_max_dimension_for_style(style: str | None, *, rapidocr_max_dimension: int | None) -> int | None:
    crop_max_dimension = (
        PROVIDER_UI_CROP_OCR_MAX_DIMENSION
        if PROVIDER_UI_CROP_OCR_MAX_DIMENSION > 0
        else rapidocr_max_dimension
    )
    if style == "gray-fill" and PROVIDER_UI_GRAY_FILL_CROP_OCR_MAX_DIMENSION > 0:
        if crop_max_dimension is None or PROVIDER_UI_GRAY_FILL_CROP_OCR_MAX_DIMENSION < crop_max_dimension:
            return PROVIDER_UI_GRAY_FILL_CROP_OCR_MAX_DIMENSION
    return crop_max_dimension


def provider_ui_focus_ocr_crop(rgb, bounds: tuple[float, float, float, float]):
    height, width = rgb.shape[:2]
    min_x, min_y, max_x, max_y = bounds
    polygon_width = max(1.0, max_x - min_x)
    polygon_height = max(1.0, max_y - min_y)
    left = max(0, int(min_x + polygon_width * PROVIDER_UI_FOCUS_CROP_MIN_X_FRACTION))
    right = min(width, int(min_x + polygon_width * PROVIDER_UI_FOCUS_CROP_MAX_X_FRACTION))
    pad_y = max(40.0, polygon_height * PROVIDER_UI_FOCUS_CROP_Y_PAD_RATIO)
    top = max(0, int(min_y - pad_y))
    bottom = min(height, int(max_y + pad_y))
    if right <= left or bottom <= top:
        return provider_ui_ocr_crop(rgb, bounds)
    return rgb[top:bottom, left:right], float(left), float(top)


def provider_ui_ocr_crop(rgb, bounds: tuple[float, float, float, float]):
    height, width = rgb.shape[:2]
    min_x, min_y, max_x, max_y = bounds
    polygon_width = max(1.0, max_x - min_x)
    polygon_height = max(1.0, max_y - min_y)
    pad_x = max(PROVIDER_UI_CROP_MIN_PAD_PX, polygon_width * PROVIDER_UI_CROP_PAD_RATIO)
    pad_y = max(PROVIDER_UI_CROP_MIN_PAD_PX, polygon_height * PROVIDER_UI_CROP_PAD_RATIO)
    left = max(0, int(min_x - pad_x))
    top = max(0, int(min_y - pad_y))
    right = min(width, int(max_x + pad_x))
    bottom = min(height, int(max_y + pad_y))
    if right <= left or bottom <= top:
        return rgb, 0.0, 0.0
    return rgb[top:bottom, left:right], float(left), float(top)


def classify_style_for_ocr(rgb):
    max_dimension = EARLY_OCR_STYLE_MAX_DIMENSION
    if max_dimension <= 0:
        return classify_style(rgb)
    height, width = rgb.shape[:2]
    largest = max(width, height)
    if largest <= max_dimension:
        return classify_style(rgb)
    scale = max_dimension / float(largest)
    sampled = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return classify_style(sampled)


def extract_full_ocr_labels_for_style(image_path: str | Path, rgb, *, style: str) -> list[Any]:
    rapidocr_max_dimension = rapidocr_max_dimension_for_extraction_style(style)
    rapidocr_detector_limit = rapidocr_detector_limit_for_ocr_style(style)
    ocr_kwargs: dict[str, Any] = {
        "cache": runner_ocr_cache_enabled(),
    }
    rapidocr_recognition_profile = rapidocr_recognition_profile_for_ocr_style(style)
    if rapidocr_recognition_profile is not None:
        ocr_kwargs["rapidocr_recognition_profile"] = rapidocr_recognition_profile
    rapidocr_detector_limit_type = rapidocr_detector_limit_type_for_ocr_style(style)
    if rapidocr_detector_limit_type is not None:
        ocr_kwargs["rapidocr_detector_limit_type"] = rapidocr_detector_limit_type
    if rapidocr_max_dimension is None:
        if rapidocr_detector_limit is not None:
            ocr_kwargs["rapidocr_detector_limit_side_len"] = rapidocr_detector_limit
        return extract_ocr_labels_from_rgb(str(image_path), rgb, **ocr_kwargs)
    if rapidocr_detector_limit is not None:
        ocr_kwargs["rapidocr_detector_limit_side_len"] = rapidocr_detector_limit
    return extract_ocr_labels_from_rgb(
        str(image_path),
        rgb,
        rapidocr_max_dimension=rapidocr_max_dimension,
        **ocr_kwargs,
    )


def fast_text_ocr_min_area_for_style(style: str | None) -> float | None:
    if FAST_TEXT_OCR_MIN_AREA <= 0.0 or style not in FAST_TEXT_OCR_STYLES:
        return None
    return FAST_TEXT_OCR_MIN_AREA


def should_fallback_focus_georef_ocr(focused: bool, georef) -> bool:
    if not focused:
        return False
    return not is_credible_context_hint_georeference(georef)


def focused_georef_with_admin_control_city(georef):
    if georef is None:
        return None
    current_city_tokens = normalize_catalog_area_tokens(str(georef.transform.city))
    for control in georef.control_points:
        geocode = getattr(control, "geocode", None)
        label = getattr(control, "label", None)
        if geocode is None or label is None:
            continue
        if getattr(geocode, "place_type", "").lower() not in FOCUSED_ADMIN_DISPLAY_PLACE_TYPES:
            continue
        if getattr(label, "confidence", 0.0) < 95.0:
            continue
        admin_city = str(getattr(geocode, "display_name", "")).split(",", 1)[0].strip()
        if not admin_city:
            continue
        admin_tokens = normalize_catalog_area_tokens(admin_city)
        if not admin_tokens or normalize_catalog_area_tokens(str(getattr(label, "text", ""))) != admin_tokens:
            continue
        if current_city_tokens == admin_tokens:
            return georef
        return georeference_result_with_city(georef, admin_city)
    return georef


def should_fallback_fast_text_ocr(
    filtered: bool,
    georef,
    *,
    style: str,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    if not filtered:
        return False
    if fast_text_ocr_min_area_for_style(style) is None:
        return True
    if georef is None:
        return True
    if (
        width is not None
        and height is not None
        and should_allow_sparse_regional_georef_fit(style, width, height)
        and sparse_ocr_georeference_lacks_support(
            georef,
            width=width,
            height=height,
        )
    ):
        return True
    return georef.transform.confidence < FAST_TEXT_OCR_FALLBACK_CONFIDENCE


def runner_ocr_cache_enabled() -> bool:
    value = os.environ.get(RUNNER_OCR_CACHE_ENV)
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return True


def rapidocr_max_dimension_for_extraction_style(style: str) -> int | None:
    if style != "purple-fill":
        return None
    if RAPIDOCR_MAX_DIMENSION <= 0 or RAPIDOCR_PURPLE_FILL_MAX_DIMENSION <= 0:
        return None
    if RAPIDOCR_PURPLE_FILL_MAX_DIMENSION >= RAPIDOCR_MAX_DIMENSION:
        return None
    return RAPIDOCR_PURPLE_FILL_MAX_DIMENSION


def rapidocr_detector_limit_for_ocr_style(style: str | None) -> int | None:
    if style != "bright-blue":
        return None
    if RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN <= 0:
        return None
    return RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN


def rapidocr_detector_limit_type_for_ocr_style(style: str | None) -> str | None:
    if style != "bright-blue":
        return None
    if RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN <= 0:
        return None
    detector_limit_type = RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE.strip().lower()
    if detector_limit_type not in {"max", "min"}:
        return None
    return detector_limit_type


def rapidocr_recognition_profile_for_ocr_style(style: str | None) -> str | None:
    if style != "bright-blue":
        return None
    profile = RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE.strip().lower()
    if not profile or profile == "default":
        return None
    return profile


def current_catalog_label_shape_shortcut_enabled(
    *,
    city_input: str | None,
    allow_catalog: bool,
    skip_redundant_probe: bool,
) -> bool:
    return (
        allow_catalog
        and city_input is None
        and skip_redundant_probe
        and CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION > 0
    )


def provider_ui_fast_ocr_max_dimension_for_style(style: str, *, width: int, height: int) -> int | None:
    if style not in PROVIDER_UI_FAST_OCR_STYLES:
        return None
    if width <= 0 or height <= 0:
        return None
    if style in PROVIDER_UI_FAST_OCR_TALL_SCREEN_STYLES and height < width * PROVIDER_UI_FAST_OCR_MIN_HEIGHT_WIDTH_RATIO:
        return None
    if PROVIDER_UI_RAPIDOCR_MAX_DIMENSION <= 0 or RAPIDOCR_MAX_DIMENSION <= 0:
        return None
    if PROVIDER_UI_RAPIDOCR_MAX_DIMENSION >= RAPIDOCR_MAX_DIMENSION:
        return None
    return PROVIDER_UI_RAPIDOCR_MAX_DIMENSION


def hinted_catalog_shape_match(pixel_geometry, *, style: str, city_input: str | None):
    match = match_service_area_catalog(
        pixel_geometry,
        style=style,
        area_hint_texts=[city_input] if city_input is not None else None,
    )
    if match is not None or city_input is None:
        return match, None
    area_hint_match = area_hinted_current_catalog_shape_match(
        pixel_geometry,
        style=style,
        city_input=city_input,
    )
    if area_hint_match is not None:
        return area_hint_match, "catalog-shape-match:area-hint-current"
    city_match = match_service_area_catalog_for_city_hint(
        pixel_geometry,
        style=style,
        city_hint=city_input,
    )
    if city_match is None:
        return None, None
    return city_match, "catalog-shape-match:city-contained"


def area_hinted_current_catalog_shape_match(
    pixel_geometry,
    *,
    style: str,
    city_input: str | None,
) -> ServiceAreaCatalogMatch | None:
    if city_input is None or not catalog_style_supported(style):
        return None
    scored = []
    for entry in load_catalog_entries():
        if not entry.is_active:
            continue
        if entry.catalog_source != "current-verified-ocr-output":
            continue
        if style not in PROVIDER_STYLES.get(entry.provider, set()):
            continue
        iou, area_ratio, scored_entry, fitted, rotation_degrees = score_catalog_entry(
            pixel_geometry,
            entry,
            min_iou=entry.min_iou,
        )
        scored.append((iou, area_ratio, scored_entry, fitted, rotation_degrees))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_area_ratio, best_entry, best_fitted, best_rotation = scored[0]
    if not catalog_area_matches_text(best_entry.area, city_input):
        return None
    runner_up_iou = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_iou - runner_up_iou
    required_iou = max(
        AREA_HINTED_CURRENT_CATALOG_MIN_IOU,
        best_entry.min_iou - AREA_HINTED_CURRENT_CATALOG_MAX_IOU_RELAXATION,
    )
    if best_iou < required_iou or margin < AREA_HINTED_CURRENT_CATALOG_MIN_MARGIN:
        return None
    if not (
        AREA_HINTED_CURRENT_CATALOG_MIN_AREA_RATIO
        <= best_area_ratio
        <= AREA_HINTED_CURRENT_CATALOG_MAX_AREA_RATIO
    ):
        return None
    return catalog_match_from_score(
        pixel_geometry,
        best_entry,
        iou=best_iou,
        area_ratio=best_area_ratio,
        margin=margin,
        fitted_mercator_geometry=best_fitted,
        rotation_degrees=best_rotation,
    )


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
    min_iou = low_resolution_shape_catalog_min_iou(width, height)
    match = match_service_area_catalog(
        extraction.pixel_geometry,
        style=extraction.style,
        min_iou=min_iou,
        min_margin=LOW_RES_SHAPE_CATALOG_MIN_MARGIN,
        area_hint_texts=[city_input] if city_input is not None else None,
        rotation_min_iou=LOW_RES_SHAPE_CATALOG_ROTATION_MIN_IOU,
    )
    if match is None:
        return None
    if not (LOW_RES_SHAPE_CATALOG_MIN_AREA_RATIO <= match.area_ratio <= LOW_RES_SHAPE_CATALOG_MAX_AREA_RATIO):
        return None
    return match


def low_resolution_shape_catalog_min_iou(width: int, height: int) -> float:
    if max(width, height) <= LOW_RES_SHAPE_CATALOG_TINY_MAX_IMAGE_DIMENSION:
        return LOW_RES_SHAPE_CATALOG_TINY_MIN_IOU
    return LOW_RES_SHAPE_CATALOG_MIN_IOU


def filename_hinted_avride_light_fill_catalog_match(
    extraction,
    *,
    filename_hint: str | None,
):
    if extraction.style != "light-fill":
        return None
    if not filename_hint:
        return None
    if catalog_provider_hint(filename_hint) != "avride":
        return None
    if not has_active_catalog_area_hint(filename_hint):
        match = match_service_area_catalog(
            extraction.pixel_geometry,
            style="purple-fill",
            min_iou=FILENAME_HINTED_AVRIDE_PROVIDER_ONLY_MIN_IOU,
            min_margin=FILENAME_HINTED_AVRIDE_PROVIDER_ONLY_MIN_MARGIN,
        )
        if match is None:
            return None
        if getattr(match.entry, "catalog_source", None) not in CURRENT_CATALOG_COMPLETION_SOURCES:
            return None
        return match
    return match_service_area_catalog(
        extraction.pixel_geometry,
        style="purple-fill",
        min_iou=FILENAME_HINTED_AVRIDE_LIGHT_FILL_MIN_IOU,
        min_margin=FILENAME_HINTED_AVRIDE_LIGHT_FILL_MIN_MARGIN,
        area_hint_texts=[filename_hint],
    )


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
    return not (has_active_catalog_area_hint(city_input) or has_active_catalog_city_hint(city_input))


def should_defer_pre_extraction_ocr_for_focus(
    style: str | None,
    *,
    city_input: str | None,
    allow_catalog: bool,
    width: int,
    height: int,
) -> bool:
    if allow_catalog or city_input is not None or style not in PRE_EXTRACTION_FOCUS_OCR_STYLES:
        return False
    if not PROVIDER_UI_FOCUS_CROP_ENABLED or PROVIDER_UI_CROP_OCR_MAX_DIMENSION <= 0:
        return False
    if width <= 0 or height < PRE_EXTRACTION_FOCUS_OCR_MIN_HEIGHT:
        return False
    aspect_ratio = width / max(float(height), 1.0)
    return aspect_ratio <= PRE_EXTRACTION_FOCUS_OCR_MAX_ASPECT_RATIO


def should_overlap_probe_miss_ocr(
    *,
    skip_redundant_probe: bool,
    city_input: str | None,
    filename_hint: str | None,
    catalog_probe_miss_low_iou: bool = False,
) -> bool:
    if not skip_redundant_probe:
        return False
    hint_text = filename_hint or ""
    if city_input is None and not catalog_provider_hint(hint_text) and not has_active_catalog_area_hint(hint_text):
        return False
    if catalog_probe_miss_low_iou:
        return True
    if city_input is not None:
        return False
    return not catalog_provider_hint(hint_text) and not has_active_catalog_area_hint(hint_text)


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
    return has_active_catalog_area_hint(city_input) or has_active_catalog_city_hint(city_input)


def should_load_low_res_catalog_rgb(
    *,
    city_input: str | None,
    filename_hint: str | None,
    allow_pre_ocr_catalog: bool,
) -> bool:
    if not allow_pre_ocr_catalog or CATALOG_EXTRACT_MAX_DIMENSION <= 0:
        return False
    if city_input is not None:
        return has_active_catalog_area_hint(city_input) or has_active_catalog_city_hint(city_input)
    return has_active_catalog_area_hint(filename_hint)


def initial_catalog_extract_max_dimension(
    *,
    city_input: str | None,
    filename_hint: str | None,
    allow_pre_ocr_catalog: bool,
) -> int:
    if (
        not allow_pre_ocr_catalog
        or CATALOG_RETRY_EXTRACT_MAX_DIMENSION <= CATALOG_EXTRACT_MAX_DIMENSION
    ):
        return CATALOG_EXTRACT_MAX_DIMENSION
    if area_hinted_current_catalog_should_start_at_refine_dimension(
        city_input=city_input,
        filename_hint=filename_hint,
    ):
        return CATALOG_MISS_REFINE_MAX_DIMENSION
    if area_hinted_current_catalog_should_start_at_retry_dimension(
        city_input=city_input,
        filename_hint=filename_hint,
    ):
        return CATALOG_RETRY_EXTRACT_MAX_DIMENSION
    return CATALOG_EXTRACT_MAX_DIMENSION


def area_hinted_current_catalog_should_start_at_refine_dimension(
    *,
    city_input: str | None,
    filename_hint: str | None,
) -> bool:
    if CATALOG_MISS_REFINE_MAX_DIMENSION <= CATALOG_RETRY_EXTRACT_MAX_DIMENSION:
        return False
    candidates = area_hinted_current_catalog_entries(
        city_input=city_input,
        filename_hint=filename_hint,
    )
    if len(candidates) != 1:
        return False
    entry = candidates[0]
    if getattr(entry, "catalog_match_strategy", None) != "exact-ordered-contour":
        return False
    rotation_degrees = getattr(entry, "source_rotation_degrees", None)
    if rotation_degrees is None:
        return False
    return abs(float(rotation_degrees)) > CATALOG_ROTATION_MAX_DEGREES


def area_hinted_current_catalog_should_start_at_retry_dimension(
    *,
    city_input: str | None,
    filename_hint: str | None,
) -> bool:
    candidates = area_hinted_current_catalog_entries(
        city_input=city_input,
        filename_hint=filename_hint,
    )
    if len(candidates) != 1:
        return False
    return catalog_entry_vertex_count(candidates[0]) >= AREA_HINTED_CURRENT_CATALOG_RETRY_FIRST_MIN_VERTICES


def area_hinted_current_catalog_entries(
    *,
    city_input: str | None,
    filename_hint: str | None,
) -> list[Any]:
    hint_texts = [text for text in (city_input, filename_hint) if text and text.strip()]
    if not hint_texts:
        return []
    provider_hint = catalog_provider_hint(" ".join(hint_texts))
    return [
        entry
        for entry in load_catalog_entries()
        if (
            getattr(entry, "is_active", False)
            and getattr(entry, "catalog_source", None) == "current-verified-ocr-output"
            and (provider_hint is None or getattr(entry, "provider", None) == provider_hint)
            and any(catalog_area_matches_text(getattr(entry, "area", ""), hint) for hint in hint_texts)
        )
    ]


def catalog_entry_vertex_count(entry: Any) -> int:
    return geometry_vertex_count(getattr(entry, "geometry", None))


def geometry_vertex_count(geometry: Any) -> int:
    if geometry is None:
        return 0
    if hasattr(geometry, "exterior"):
        coords = list(geometry.exterior.coords)
        return max(0, len(coords) - 1)
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return 0
    return sum(geometry_vertex_count(geom) for geom in geoms)


def should_retry_pre_ocr_catalog(
    *,
    city_input: str | None,
    filename_hint: str | None,
    allow_pre_ocr_catalog: bool,
    used_catalog_scaled_extraction: bool,
    initial_extract_max_dimension: int = CATALOG_EXTRACT_MAX_DIMENSION,
    catalog_style_can_match: bool = True,
    catalog_probe_only: bool = False,
) -> bool:
    if not allow_pre_ocr_catalog or not used_catalog_scaled_extraction:
        return False
    if not catalog_style_can_match:
        return False
    if CATALOG_RETRY_EXTRACT_MAX_DIMENSION <= CATALOG_EXTRACT_MAX_DIMENSION:
        return False
    if initial_extract_max_dimension >= CATALOG_RETRY_EXTRACT_MAX_DIMENSION:
        return False
    if CATALOG_RETRY_EXTRACT_MAX_DIMENSION >= CATALOG_MISS_REFINE_MAX_DIMENSION:
        return False
    if catalog_probe_only:
        return True
    return (
        has_active_catalog_area_hint(city_input)
        or has_active_catalog_city_hint(city_input)
        or has_active_catalog_area_hint(filename_hint)
    )


def is_stale_only_catalog_hint(text: str | None) -> bool:
    return has_stale_catalog_area_hint(text) and not has_active_catalog_area_hint(text)


def catalog_matching_enabled(options: Any) -> bool:
    return bool(getattr(options, "allow_catalog", True))


def catalog_probe_only_enabled(options: Any) -> bool:
    return bool(getattr(options, "catalog_probe_only", False))


def catalog_probe_missed_enabled(options: Any) -> bool:
    return bool(getattr(options, "catalog_probe_missed", False)) and not catalog_probe_only_enabled(options)


def catalog_probe_miss_low_iou_enabled(options: Any) -> bool:
    return bool(getattr(options, "catalog_probe_miss_low_iou", False))


def catalog_probe_missed_handoff_enabled(
    options: Any,
    *,
    city_input: str | None,
    filename_hint: str | None,
    allow_pre_ocr_catalog: bool,
) -> bool:
    if not catalog_probe_missed_enabled(options) or not allow_pre_ocr_catalog:
        return False
    hint_text = " ".join(part for part in (filename_hint or "", city_input or "") if part.strip())
    if bool(catalog_provider_hint(hint_text)) or has_active_catalog_area_hint(hint_text):
        return True
    return city_input is None and not has_stale_catalog_area_hint(filename_hint)


def high_confidence_label_texts(labels: list[Any]) -> list[str]:
    return [
        label.text
        for label in labels
        if label.text.strip() and label.confidence >= CATALOG_LABEL_HINT_MIN_CONFIDENCE
    ]


def catalog_probe_miss_details(
    extraction,
    *,
    city_input: str | None,
    filename_hint: str | None,
) -> dict[str, Any]:
    hint_text = " ".join(part for part in (filename_hint or "", city_input or "") if part.strip())
    provider_hint = catalog_provider_hint(hint_text)
    scored: list[tuple[float, float, str, float]] = []
    for entry in load_catalog_entries():
        if not entry.is_active:
            continue
        if provider_hint is not None and entry.provider != provider_hint:
            continue
        if extraction.style not in PROVIDER_STYLES.get(entry.provider, set()):
            continue
        iou, area_ratio, scored_entry, _fitted, _rotation = score_catalog_entry(
            extraction.pixel_geometry,
            entry,
            min_iou=entry.min_iou,
        )
        scored.append((iou, area_ratio, scored_entry.slug, scored_entry.min_iou))
    if not scored:
        return {
            "style": extraction.style,
            "active_shape_iou_is_low": True,
            "active_shape_iou_threshold": CATALOG_PROBE_MISS_LOW_IOU_THRESHOLD,
        }
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_area_ratio, best_slug, best_required_iou = scored[0]
    return {
        "style": extraction.style,
        "best_active_catalog_slug": best_slug,
        "best_active_catalog_iou": round(float(best_iou), 6),
        "best_active_catalog_area_ratio": round(float(best_area_ratio), 6),
        "best_active_catalog_required_iou": round(float(best_required_iou), 6),
        "active_shape_iou_is_low": best_iou < CATALOG_PROBE_MISS_LOW_IOU_THRESHOLD,
        "active_shape_iou_threshold": CATALOG_PROBE_MISS_LOW_IOU_THRESHOLD,
    }


def catalog_probe_near_hit_match(
    extraction,
    *,
    city_input: str | None,
    filename_hint: str | None,
):
    hint_text = " ".join(part for part in (filename_hint or "", city_input or "") if part.strip())
    provider_hint = catalog_provider_hint(hint_text)
    if provider_hint is None:
        return catalog_probe_unhinted_near_hit_match(extraction)
    candidates = [
        entry
        for entry in load_catalog_entries()
        if (
            entry.is_active
            and entry.provider == provider_hint
            and getattr(entry, "catalog_source", None) in OCR_DERIVED_CATALOG_SOURCES
            and extraction.style in PROVIDER_STYLES.get(entry.provider, set())
            and catalog_area_matches_text(entry.area, hint_text)
        )
    ]
    if len(candidates) != 1:
        return None
    return match_catalog_entry(
        extraction.pixel_geometry,
        candidates[0],
        min_iou=CATALOG_PROBE_NEAR_HIT_MIN_IOU,
        min_area_ratio=CATALOG_PROBE_NEAR_HIT_MIN_AREA_RATIO,
        max_area_ratio=CATALOG_PROBE_NEAR_HIT_MAX_AREA_RATIO,
    )


def catalog_probe_unhinted_near_hit_match(extraction):
    scored = []
    for entry in load_catalog_entries():
        if not entry.is_active:
            continue
        if getattr(entry, "catalog_source", None) not in OCR_DERIVED_CATALOG_SOURCES:
            continue
        if extraction.style not in PROVIDER_STYLES.get(entry.provider, set()):
            continue
        iou, area_ratio, scored_entry, fitted, rotation_degrees = score_catalog_entry(
            extraction.pixel_geometry,
            entry,
            min_iou=entry.min_iou,
        )
        scored.append((iou, area_ratio, scored_entry, fitted, rotation_degrees))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_area_ratio, best_entry, best_fitted, best_rotation = scored[0]
    runner_up_iou = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_iou - runner_up_iou
    if best_iou < CATALOG_PROBE_NEAR_HIT_MIN_IOU:
        return None
    if margin < CATALOG_PROBE_UNHINTED_NEAR_HIT_MIN_MARGIN:
        return None
    if not (CATALOG_PROBE_NEAR_HIT_MIN_AREA_RATIO <= best_area_ratio <= CATALOG_PROBE_NEAR_HIT_MAX_AREA_RATIO):
        return None
    return catalog_match_from_score(
        extraction.pixel_geometry,
        best_entry,
        iou=best_iou,
        area_ratio=best_area_ratio,
        margin=margin,
        fitted_mercator_geometry=best_fitted,
        rotation_degrees=best_rotation,
    )


def filename_hinted_current_catalog_near_hit_match(
    extraction,
    *,
    city_input: str | None,
    filename_hint: str | None,
) -> ServiceAreaCatalogMatch | None:
    if extraction.confidence < CURRENT_CATALOG_LABEL_SHAPE_MIN_EXTRACTION_CONFIDENCE:
        return None
    if not catalog_style_supported(extraction.style):
        return None
    hint_text = " ".join(part for part in (filename_hint or "", city_input or "") if part.strip())
    if catalog_provider_hint(hint_text) is None:
        return None
    if not has_active_catalog_area_hint(hint_text):
        return None
    return catalog_probe_near_hit_match(
        extraction,
        city_input=city_input,
        filename_hint=filename_hint,
    )


def post_georeference_catalog_completion_match(
    extraction,
    labels: list[Any],
    lonlat_geometry,
    *,
    city_input: str | None,
    filename_hint: str | None,
    georef_confidence: float,
) -> ServiceAreaCatalogMatch | None:
    if georef_confidence < POST_GEOREF_CATALOG_COMPLETION_MIN_GEOREF_CONFIDENCE:
        return None
    if not catalog_style_supported(extraction.style):
        return None

    hint_texts = [
        text
        for text in [city_input, filename_hint, *high_confidence_label_texts(labels)]
        if text and text.strip()
    ]
    if not hint_texts:
        return None
    provider_hint = catalog_provider_hint(" ".join(hint_texts))
    candidates = []
    for entry in load_catalog_entries():
        if not entry.is_active:
            continue
        if getattr(entry, "catalog_source", None) not in CURRENT_CATALOG_COMPLETION_SOURCES:
            continue
        if extraction.style not in PROVIDER_STYLES.get(entry.provider, set()):
            continue
        if provider_hint is not None and entry.provider != provider_hint:
            continue
        if any(catalog_area_matches_text(entry.area, text) for text in hint_texts):
            candidates.append(entry)
    if len(candidates) != 1:
        return None

    entry = candidates[0]
    output_mercator = transform(lambda x, y, z=None: lonlat_to_mercator(x, y), lonlat_geometry).buffer(0)
    catalog_mercator = entry.mercator_geometry.buffer(0)
    output_area = output_mercator.area
    catalog_area = catalog_mercator.area
    if output_area <= 0.0 or catalog_area <= 0.0:
        return None
    intersection_area = output_mercator.intersection(catalog_mercator).area
    union_area = output_mercator.union(catalog_mercator).area
    if union_area <= 0.0:
        return None
    iou = intersection_area / union_area
    output_coverage = intersection_area / output_area
    catalog_coverage = intersection_area / catalog_area
    area_ratio = output_area / catalog_area
    if iou < POST_GEOREF_CATALOG_COMPLETION_MIN_IOU:
        return None
    if output_coverage < POST_GEOREF_CATALOG_COMPLETION_MIN_OUTPUT_COVERAGE:
        return None
    if catalog_coverage < POST_GEOREF_CATALOG_COMPLETION_MIN_CATALOG_COVERAGE:
        return None
    if not (
        POST_GEOREF_CATALOG_COMPLETION_MIN_AREA_RATIO
        <= area_ratio
        <= POST_GEOREF_CATALOG_COMPLETION_MAX_AREA_RATIO
    ):
        return None
    return catalog_match_from_score(
        extraction.pixel_geometry,
        entry,
        iou=iou,
        area_ratio=area_ratio,
        margin=output_coverage,
        fitted_mercator_geometry=output_mercator,
        rotation_degrees=0.0,
        confidence_override=min(POST_GEOREF_CATALOG_COMPLETION_CONFIDENCE, georef_confidence),
    )


def current_catalog_label_shape_match(extraction, labels: list[Any]) -> ServiceAreaCatalogMatch | None:
    if extraction.confidence < CURRENT_CATALOG_LABEL_SHAPE_MIN_EXTRACTION_CONFIDENCE:
        return None
    if not catalog_style_supported(extraction.style):
        return None

    label_hints = high_confidence_label_texts(labels)
    if not label_hints:
        return None

    candidates = []
    for entry in load_catalog_entries():
        if not entry.is_active:
            continue
        if getattr(entry, "catalog_source", None) not in CURRENT_CATALOG_COMPLETION_SOURCES:
            continue
        if extraction.style not in PROVIDER_STYLES.get(entry.provider, set()):
            continue
        if not any(catalog_area_matches_text(entry.area, hint) for hint in label_hints):
            continue
        iou, area_ratio, scored_entry, fitted, rotation_degrees = score_catalog_entry(
            extraction.pixel_geometry,
            entry,
            min_iou=entry.min_iou,
        )
        if iou < CURRENT_CATALOG_LABEL_SHAPE_MIN_IOU:
            continue
        if not (
            CURRENT_CATALOG_LABEL_SHAPE_MIN_AREA_RATIO
            <= area_ratio
            <= CURRENT_CATALOG_LABEL_SHAPE_MAX_AREA_RATIO
        ):
            continue
        candidates.append((iou, area_ratio, scored_entry, fitted, rotation_degrees))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_area_ratio, best_entry, best_fitted, best_rotation = candidates[0]
    runner_up_iou = candidates[1][0] if len(candidates) > 1 else 0.0
    return catalog_match_from_score(
        extraction.pixel_geometry,
        best_entry,
        iou=best_iou,
        area_ratio=best_area_ratio,
        margin=best_iou - runner_up_iou,
        fitted_mercator_geometry=best_fitted,
        rotation_degrees=best_rotation,
        confidence_override=CURRENT_CATALOG_LABEL_SHAPE_CONFIDENCE,
    )


def filename_hinted_current_catalog_shape_match(
    extraction,
    *,
    city_input: str | None,
    filename_hint: str | None,
) -> ServiceAreaCatalogMatch | None:
    if extraction.confidence < CURRENT_CATALOG_LABEL_SHAPE_MIN_EXTRACTION_CONFIDENCE:
        return None
    if not catalog_style_supported(extraction.style):
        return None
    hint_text = " ".join(part for part in (filename_hint or "", city_input or "") if part.strip())
    provider_hint = catalog_provider_hint(hint_text)
    if provider_hint is None:
        return None
    candidates = [
        entry
        for entry in load_catalog_entries()
        if (
            entry.is_active
            and getattr(entry, "catalog_source", None) in CURRENT_CATALOG_COMPLETION_SOURCES
            and entry.provider == provider_hint
            and extraction.style in PROVIDER_STYLES.get(entry.provider, set())
            and catalog_area_matches_text(entry.area, hint_text)
        )
    ]
    if len(candidates) != 1:
        return None

    entry = candidates[0]
    return match_catalog_entry(
        extraction.pixel_geometry,
        entry,
        min_iou=entry.min_iou,
        min_area_ratio=FILENAME_CURRENT_CATALOG_SHAPE_MIN_AREA_RATIO,
        max_area_ratio=FILENAME_CURRENT_CATALOG_SHAPE_MAX_AREA_RATIO,
        confidence_override=CURRENT_CATALOG_LABEL_SHAPE_CONFIDENCE,
    )


def provider_ui_label_catalog_match(extraction, labels: list[Any]):
    provider = provider_ui_label_provider(labels) or unique_catalog_provider_for_style(extraction.style)
    if provider is None or extraction.style not in PROVIDER_STYLES.get(provider, set()):
        return None

    nearby_texts = provider_ui_nearby_area_texts(labels, extraction.pixel_geometry.bounds)
    if not nearby_texts:
        return None
    provider_entries = [
        entry
        for entry in load_catalog_entries()
        if (
            entry.is_active
            and entry.provider == provider
            and extraction.style in PROVIDER_STYLES.get(entry.provider, set())
        )
    ]
    candidates = {}
    for text in nearby_texts:
        text_candidates = [entry for entry in provider_entries if catalog_area_matches_text(entry.area, text)]
        if len(text_candidates) == 1:
            candidates[text_candidates[0].slug] = text_candidates[0]
    if len(candidates) != 1:
        return None
    candidate = next(iter(candidates.values()))
    return match_catalog_entry(
        extraction.pixel_geometry,
        candidate,
        min_iou=PROVIDER_UI_LABEL_MIN_IOU,
        min_area_ratio=PROVIDER_UI_LABEL_MIN_AREA_RATIO,
        max_area_ratio=PROVIDER_UI_LABEL_MAX_AREA_RATIO,
        confidence_override=PROVIDER_UI_LABEL_CONFIDENCE,
    )


def unique_catalog_provider_for_style(style: str | None) -> str | None:
    providers = [
        provider
        for provider, styles in PROVIDER_STYLES.items()
        if style in styles
    ]
    if len(providers) != 1:
        return None
    return providers[0]


def provider_ui_label_provider(labels: list[Any]) -> str | None:
    high_confidence_text = " ".join(
        label.text for label in labels if label.confidence >= PROVIDER_UI_LABEL_MIN_CONFIDENCE
    )
    provider = catalog_provider_hint(high_confidence_text)
    if provider is not None:
        return provider
    compact_text = "".join(ch for ch in high_confidence_text.lower() if ch.isalnum())
    for provider in PROVIDER_STYLES:
        if provider in compact_text:
            return provider
    return None


def provider_ui_nearby_area_texts(
    labels: list[Any],
    bounds: tuple[float, float, float, float],
) -> list[str]:
    return [
        label.text
        for label in labels
        if (
            label.confidence >= PROVIDER_UI_LABEL_MIN_CONFIDENCE
            and label_near_extracted_geometry(label, bounds)
        )
    ]


def label_near_extracted_geometry(label: Any, bounds: tuple[float, float, float, float]) -> bool:
    min_x, min_y, max_x, max_y = bounds
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    pad_x = max(60.0, width * 0.35)
    pad_y = max(60.0, height * 0.35)
    return (min_x - pad_x) <= label.x <= (max_x + pad_x) and (min_y - pad_y) <= label.y <= (max_y + pad_y)


def sparse_low_res_label_catalog_match(
    extraction,
    labels: list[Any],
    *,
    width: int,
    height: int,
) -> ServiceAreaCatalogMatch | None:
    if max(width, height) > SPARSE_LABEL_CATALOG_MAX_DIMENSION:
        return None
    if width * height > SPARSE_LABEL_CATALOG_MAX_PIXELS:
        return None
    if extraction.coverage_ratio < SPARSE_LABEL_CATALOG_MIN_COVERAGE:
        return None
    if extraction.confidence < SPARSE_LABEL_CATALOG_MIN_CONFIDENCE:
        return None
    if not catalog_style_supported(extraction.style):
        return None

    label_texts = [
        label.text
        for label in labels
        if label.text.strip() and label.confidence >= SPARSE_LABEL_CATALOG_MIN_LABEL_CONFIDENCE
    ]
    if not label_texts:
        return None

    candidates = {}
    for entry in load_catalog_entries():
        if not entry.is_active or extraction.style not in PROVIDER_STYLES.get(entry.provider, set()):
            continue
        if any(sparse_label_matches_catalog_area(entry.area, text) for text in label_texts):
            candidates[entry.slug] = entry
    if len(candidates) != 1:
        return None

    entry = next(iter(candidates.values()))
    min_x, min_y, max_x, max_y = extraction.pixel_geometry.bounds
    ref_min_x, ref_min_y, ref_max_x, ref_max_y = entry.mercator_geometry.bounds
    pixel_width = max(1.0, max_x - min_x)
    pixel_height = max(1.0, max_y - min_y)
    meters_per_pixel = ((ref_max_x - ref_min_x) / pixel_width + (ref_max_y - ref_min_y) / pixel_height) / 2.0
    origin_x = (min_x + max_x) / 2.0
    origin_y = (min_y + max_y) / 2.0

    return ServiceAreaCatalogMatch(
        entry=entry,
        iou=0.0,
        area_ratio=1.0,
        margin=0.0,
        fitted_mercator_geometry=entry.mercator_geometry,
        fitted_lonlat_geometry=entry.geometry,
        meters_per_pixel=meters_per_pixel,
        origin_lon=entry.geometry.centroid.x,
        origin_lat=entry.geometry.centroid.y,
        origin_x=origin_x,
        origin_y=origin_y,
        rotation_degrees=0.0,
        confidence_override=SPARSE_LABEL_CATALOG_CONFIDENCE,
    )


def sparse_label_matches_catalog_area(area: str, text: str) -> bool:
    if catalog_area_matches_text(area, text):
        return True
    area_tokens = normalize_catalog_area_tokens(area)
    text_tokens = normalize_catalog_area_tokens(text)
    if not area_tokens or not text_tokens:
        return False
    return all(
        any(
            len(area_token) >= 7
            and len(text_token) >= 5
            and edit_distance_at_most(area_token, text_token, max_edits=2)
            for text_token in text_tokens
        )
        for area_token in area_tokens
    )


def edit_distance_at_most(left: str, right: str, *, max_edits: int) -> bool:
    if abs(len(left) - len(right)) > max_edits:
        return False
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_edits:
            return False
        previous = current
    return previous[-1] <= max_edits


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
    catalog_label_hints: list[str] | None = None,
    shape_match: bool = True,
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
    if catalog_label_hints:
        properties["catalog_label_hints"] = catalog_label_hints
    if not shape_match:
        properties["catalog_shape_iou"] = None
        properties["catalog_shape_margin"] = None
        properties["catalog_area_ratio"] = None
    combined_confidence = properties["combined_confidence"]
    emit_progress(
        progress,
        stage="georeference",
        message="Matched known service-area label" if not shape_match else "Matched known service-area shape",
        percent=78,
        details={
            "source": georeference_source,
            "catalog_slug": catalog_match.entry.slug,
            "shape_iou": round(catalog_match.iou, 3) if shape_match else None,
            "label_hints": catalog_label_hints[:5] if catalog_label_hints else None,
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
        overlay_extension = "webp" if opts.overlay_format == "webp" else "png"
        overlay_path = debug_path / f"{stem}.overlay.{overlay_extension}"
        if opts.write_mask_artifact:
            mask_path = debug_path / f"{stem}.mask.png"
            write_mask_png(extraction.mask, mask_path)
        write_overlay_image(
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
    anchor_marker_dots: bool = True,
    style: str | None = None,
    allow_credible_cached_fit: bool = False,
    progress: ProgressCallback | None = None,
):
    sparse_regional_fit = should_allow_sparse_regional_georef_fit(style, width, height)
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
                anchor_marker_dots=anchor_marker_dots,
                allow_road_refinement=should_allow_label_fit_road_refinement(style),
                allow_sparse_regional_fit=sparse_regional_fit,
                allow_credible_cached_fit=allow_credible_cached_fit,
            )
            if is_fast_context_hint_georeference(georef):
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
            allow_road_refinement=should_allow_label_fit_road_refinement(style),
            allow_credible_cached_fit=allow_credible_cached_fit,
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
            anchor_marker_dots=anchor_marker_dots,
            allow_road_refinement=should_allow_label_fit_road_refinement(style),
            allow_sparse_regional_fit=sparse_regional_fit,
            allow_credible_cached_fit=allow_credible_cached_fit,
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


def is_fast_context_hint_georeference(result) -> bool:
    if result is None:
        return False
    if is_decisive_georeference_result(result):
        return True
    if not is_credible_context_hint_georeference(result):
        return False
    if len(result.control_points) >= 4:
        return (
            result.transform.confidence >= 0.78
            and result.residual_median_m <= 1500.0
            and result.residual_p90_m <= 2600.0
        )
    return True


def should_anchor_marker_dots(style: str) -> bool:
    return style == "dark-teal"


def should_allow_label_fit_road_refinement(style: str | None) -> bool:
    return style != "light-fill"


def should_allow_sparse_regional_georef_fit(style: str | None, width: int, height: int) -> bool:
    return style == "gray-fill" and min(width, height) < FAST_TEXT_OCR_LOW_RES_RETRY_MAX_MIN_DIMENSION


def sparse_ocr_georeference_lacks_support(georef, *, width: int, height: int) -> bool:
    if not str(georef.transform.source).startswith("ocr-georeference:"):
        return False
    control_count = len(georef.control_points)
    if sparse_high_residual_fit_without_road_evidence(
        control_count,
        georef.residual_p90_m,
        georef.road_match,
    ):
        return True
    if control_count <= 3 and georef.road_match is None and georef.transform.confidence < 0.75:
        return True
    if (
        control_count <= 5
        and georef.road_match is None
        and georef.transform.city == "Inferred map area"
    ):
        return True
    return low_res_two_control_regional_fit_without_road_evidence(
        control_count,
        georef.transform.meters_per_pixel,
        width,
        height,
        georef.road_match,
    )


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
    allow_road_refinement: bool = True,
    allow_credible_cached_fit: bool = False,
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
            allow_road_refinement=allow_road_refinement,
            allow_credible_cached_fit=allow_credible_cached_fit,
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
