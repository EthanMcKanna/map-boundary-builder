from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .pipeline_version import runtime_dependency_signature
from .runtime_config import (
    FAST_TEXT_OCR_RESCUE_MIN_AREA,
    FAST_TEXT_OCR_RESCUE_MIN_ASPECT,
    ONNXRUNTIME_ALLOW_SPINNING,
    ONNXRUNTIME_ENABLE_CPU_MEM_ARENA,
    RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS,
    RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
    RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
    RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION,
    RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
    RAPIDOCR_CLS_BATCH_NUM,
    RAPIDOCR_DARK_TEAL_REC_BATCH_NUM,
    RAPIDOCR_DET_LIMIT_SIDE_LEN,
    RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION,
    RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN,
    RAPIDOCR_MAX_DIMENSION,
    RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION,
    RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
    RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION,
    RAPIDOCR_WARM_SAMPLE_MAX_DIMENSION,
    RAPIDOCR_REC_BATCH_NUM,
    TESSERACT_FALLBACK_MIN_USEFUL_LABELS,
    rapidocr_english_ppocrv5_asset_paths,
    rapidocr_english_ppocrv5_asset_signature,
    rapidocr_warm_detector_limits,
)


@dataclass(frozen=True)
class OcrLabel:
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float


_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
OCR_CACHE_DIR = _CACHE_ROOT / "ocr-labels"
OCR_CACHE_VERSION = "ocr-labels-v8"
OCR_VISUAL_CACHE_QUANTIZATION_MASK = 0xFC
OCR_COARSE_VISUAL_CACHE_QUANTIZATION_MASK = 0xF8
OCR_BORDER_COLOR_TOLERANCE = 6
OCR_BORDER_ROW_MATCH_RATIO = 0.995
OCR_MEMORY_CACHE_MAX = 128
OCR_DISK_CACHE_ENABLED = os.environ.get("MAP_BOUNDARY_OCR_DISK_CACHE", "").lower() in {
    "1",
    "true",
    "yes",
}
OCR_CACHE_DEPENDENCY_PACKAGES = (
    "onnxruntime",
    "opencv-python",
    "opencv-python-headless",
    "pillow",
    "rapidocr-onnxruntime",
)
RAPIDOCR_RECOGNITION_PROFILE_DEFAULT = "default"
RAPIDOCR_RECOGNITION_PROFILE_EN_PPOCRV5 = "en-ppocrv5"
RAPIDOCR_DETECTOR_LIMIT_TYPE_DEFAULT = "default"
RAPIDOCR_PROFILE_DURATION_KEYS = (
    "input_s",
    "load_img_s",
    "preprocess_s",
    "det_elapsed_s",
    "crop_s",
    "rec_elapsed_s",
    "final_s",
    "profiled_total_s",
    "classifier_total_s",
    "total_s",
)
RAPIDOCR_PROFILE_CONFIDENCE_COUNT_KEYS = (
    "label_confidence_lt_50_count",
    "label_confidence_lt_70_count",
    "label_confidence_lt_80_count",
    "label_confidence_lt_90_count",
)
RAPIDOCR_PROFILE_COUNT_KEYS = (
    "raw_box_count",
    "selected_box_count",
    "result_count",
    "label_count",
    "useful_label_count",
    "header_region_filter_used",
    "footer_region_filter_used",
    *RAPIDOCR_PROFILE_CONFIDENCE_COUNT_KEYS,
)
RAPIDOCR_PROFILE_CONFIDENCE_DISTRIBUTION_KEYS = (
    "label_confidence_min",
    "label_confidence_p25",
    "label_confidence_p50",
    "label_confidence_p75",
    "label_confidence_p90",
    "label_confidence_max",
)
RAPIDOCR_PROFILE_CONTEXT_KEYS = (
    "input_kind",
    "input_shape",
    "scale_x",
    "scale_y",
    "detector_limit",
    "detector_limit_type",
    "recognition_profile",
    "rec_batch_num",
    "min_text_area",
    "classifier_retry",
    "header_region_filter",
)
RAPIDOCR_HEADER_REGION_FILTER_MIN_BOXES = 24
RAPIDOCR_HEADER_REGION_FILTER_MIN_HEADER_BOXES = 8
RAPIDOCR_HEADER_REGION_FILTER_MIN_SELECTED_BOXES = 10
RAPIDOCR_HEADER_REGION_FILTER_MIN_REMOVED_BOXES = 5
RAPIDOCR_HEADER_REGION_FILTER_MIN_DIMENSION = 600
RAPIDOCR_HEADER_REGION_FILTER_Y_RATIO = 0.12
RAPIDOCR_HEADER_REGION_FILTER_MIN_Y = 90.0
RAPIDOCR_HEADER_REGION_FILTER_MAX_Y = 180.0
RAPIDOCR_HEADER_REGION_FILTER_LEFT_TITLE_X_RATIO = 0.44
RAPIDOCR_HEADER_REGION_FILTER_WIDE_TEXT_RATIO = 0.12
RAPIDOCR_HEADER_REGION_FILTER_FOOTER_Y_RATIO = 0.93
RAPIDOCR_HEADER_REGION_FILTER_MIN_FOOTER_REMOVED_BOXES = 2
RAPIDOCR_HEADER_REGION_FILTER_MIN_FOOTER_SELECTED_BOXES = 10
_OCR_MEMORY_CACHE: OrderedDict[str, tuple[OcrLabel, ...]] = OrderedDict()
_OCR_MEMORY_CACHE_LOCK = threading.RLock()
_RAPIDOCR_SESSION_OPTIONS_PATCHED = False
_RAPIDOCR_PROFILE_LOCK = threading.RLock()
_RAPIDOCR_PROFILE_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "rapidocr_profile_events",
    default=None,
)


@contextmanager
def collect_rapidocr_profiles():
    events: list[dict[str, Any]] = []
    token = _RAPIDOCR_PROFILE_EVENTS.set(events)
    try:
        yield events
    finally:
        _RAPIDOCR_PROFILE_EVENTS.reset(token)


def submit_with_rapidocr_profile_context(executor: ThreadPoolExecutor, fn, *args, **kwargs):
    context = copy_context()
    return executor.submit(context.run, fn, *args, **kwargs)


def rapidocr_profile_enabled() -> bool:
    return _RAPIDOCR_PROFILE_EVENTS.get() is not None


def record_rapidocr_profile(profile: dict[str, Any]) -> None:
    events = _RAPIDOCR_PROFILE_EVENTS.get()
    if events is None:
        return
    with _RAPIDOCR_PROFILE_LOCK:
        events.append(sanitize_rapidocr_profile(profile))


def summarize_rapidocr_profile_events(events: list[dict[str, Any]] | None) -> dict[str, Any]:
    calls = events or []
    summary: dict[str, Any] = {"calls": len(calls)}
    for key in RAPIDOCR_PROFILE_DURATION_KEYS:
        total = sum_rapidocr_profile_number(call.get(key) for call in calls)
        if total is not None:
            summary[key] = round(total, 6)
    for key in RAPIDOCR_PROFILE_COUNT_KEYS:
        total = sum_rapidocr_profile_int(call.get(key) for call in calls)
        if total is not None:
            summary[key] = total
    copy_single_rapidocr_profile_values(
        summary,
        calls,
        (*RAPIDOCR_PROFILE_CONTEXT_KEYS, *RAPIDOCR_PROFILE_CONFIDENCE_DISTRIBUTION_KEYS),
    )
    summary["calls_detail"] = calls
    return summary


def summarize_rapidocr_profile_summaries(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not profiles:
        return None
    summary: dict[str, Any] = {
        "fixtures": len(profiles),
        "calls": sum_rapidocr_profile_int(profile.get("calls") for profile in profiles) or 0,
    }
    for key in RAPIDOCR_PROFILE_DURATION_KEYS:
        total = sum_rapidocr_profile_number(profile.get(key) for profile in profiles)
        if total is not None:
            summary[key] = round(total, 6)
    for key in RAPIDOCR_PROFILE_COUNT_KEYS:
        total = sum_rapidocr_profile_int(profile.get(key) for profile in profiles)
        if total is not None:
            summary[key] = total
    copy_single_rapidocr_profile_values(
        summary,
        profiles,
        (*RAPIDOCR_PROFILE_CONTEXT_KEYS, *RAPIDOCR_PROFILE_CONFIDENCE_DISTRIBUTION_KEYS),
    )
    return summary


def copy_single_rapidocr_profile_values(
    summary: dict[str, Any],
    profiles: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> None:
    if len(profiles) != 1:
        return
    profile = profiles[0]
    for key in keys:
        if key in profile:
            summary[key] = profile[key]


def sum_rapidocr_profile_number(values) -> float | None:
    total = 0.0
    seen = False
    for value in values:
        if value is None:
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration < 0.0:
            continue
        total += duration
        seen = True
    return total if seen else None


def sum_rapidocr_profile_int(values) -> int | None:
    total = 0
    seen = False
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        total += max(0, parsed)
        seen = True
    return total if seen else None


def sanitize_rapidocr_profile(profile: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in profile.items():
        if isinstance(value, bool) or value is None:
            sanitized[key] = value
        elif isinstance(value, int):
            sanitized[key] = value
        elif isinstance(value, float):
            sanitized[key] = round(max(0.0, value), 6) if key.endswith("_s") else value
        elif isinstance(value, str):
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            sanitized[key] = list(value)
    return sanitized


def rapidocr_box_area_profile(prefix: str, boxes: list[np.ndarray]) -> dict[str, int | float]:
    areas = [rapidocr_box_area(box) for box in boxes]
    if not areas:
        return {
            f"{prefix}_box_area_min": None,
            f"{prefix}_box_area_p25": None,
            f"{prefix}_box_area_p50": None,
            f"{prefix}_box_area_p75": None,
            f"{prefix}_box_area_p90": None,
            f"{prefix}_box_area_max": None,
            f"{prefix}_box_area_lt_500_count": 0,
            f"{prefix}_box_area_lt_900_count": 0,
            f"{prefix}_box_area_lt_1300_count": 0,
            f"{prefix}_box_area_lt_1500_count": 0,
        }
    return {
        f"{prefix}_box_area_min": round(min(areas), 3),
        f"{prefix}_box_area_p25": round(percentile_linear(areas, 25), 3),
        f"{prefix}_box_area_p50": round(percentile_linear(areas, 50), 3),
        f"{prefix}_box_area_p75": round(percentile_linear(areas, 75), 3),
        f"{prefix}_box_area_p90": round(percentile_linear(areas, 90), 3),
        f"{prefix}_box_area_max": round(max(areas), 3),
        f"{prefix}_box_area_lt_500_count": count_values_below(areas, 500.0),
        f"{prefix}_box_area_lt_900_count": count_values_below(areas, 900.0),
        f"{prefix}_box_area_lt_1300_count": count_values_below(areas, 1300.0),
        f"{prefix}_box_area_lt_1500_count": count_values_below(areas, 1500.0),
    }


def ocr_label_confidence_profile(prefix: str, labels: list[OcrLabel]) -> dict[str, int | float | None]:
    confidences = [max(0.0, min(100.0, float(label.confidence))) for label in labels]
    if not confidences:
        return {
            f"{prefix}_confidence_min": None,
            f"{prefix}_confidence_p25": None,
            f"{prefix}_confidence_p50": None,
            f"{prefix}_confidence_p75": None,
            f"{prefix}_confidence_p90": None,
            f"{prefix}_confidence_max": None,
            f"{prefix}_confidence_lt_50_count": 0,
            f"{prefix}_confidence_lt_70_count": 0,
            f"{prefix}_confidence_lt_80_count": 0,
            f"{prefix}_confidence_lt_90_count": 0,
        }
    return {
        f"{prefix}_confidence_min": round(min(confidences), 3),
        f"{prefix}_confidence_p25": round(percentile_linear(confidences, 25), 3),
        f"{prefix}_confidence_p50": round(percentile_linear(confidences, 50), 3),
        f"{prefix}_confidence_p75": round(percentile_linear(confidences, 75), 3),
        f"{prefix}_confidence_p90": round(percentile_linear(confidences, 90), 3),
        f"{prefix}_confidence_max": round(max(confidences), 3),
        f"{prefix}_confidence_lt_50_count": count_values_below(confidences, 50.0),
        f"{prefix}_confidence_lt_70_count": count_values_below(confidences, 70.0),
        f"{prefix}_confidence_lt_80_count": count_values_below(confidences, 80.0),
        f"{prefix}_confidence_lt_90_count": count_values_below(confidences, 90.0),
    }


def percentile_linear(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def count_values_below(values: list[float], threshold: float) -> int:
    return sum(value < threshold for value in values)


def extract_ocr_labels(
    image_path: str | Path,
    *,
    prepared_bgr: np.ndarray | None = None,
    composited_alpha: bool = False,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
    allow_tesseract_fallback: bool = True,
    cache: bool = True,
) -> list[OcrLabel]:
    use_tesseract: bool | None = None

    def tesseract_is_available() -> bool:
        nonlocal use_tesseract
        if not allow_tesseract_fallback:
            return False
        if use_tesseract is None:
            use_tesseract = tesseract_available()
        return use_tesseract

    label_shape = ocr_label_target_shape(image_path, prepared_bgr)
    prepared_bgr_has_distinct_shape: bool | None = None

    def prepared_bgr_shape_differs_from_source() -> bool:
        nonlocal prepared_bgr_has_distinct_shape
        if prepared_bgr is None:
            return False
        if prepared_bgr_has_distinct_shape is None:
            source_shape = source_image_shape(Path(image_path))
            prepared_bgr_has_distinct_shape = source_shape is None or prepared_bgr.shape[:2] != source_shape
        return prepared_bgr_has_distinct_shape

    use_raw_cache_key = cache and prepared_bgr is None
    cache_key = (
        ocr_cache_key(
            image_path,
            use_tesseract=tesseract_is_available(),
            rapidocr_max_dimension=rapidocr_max_dimension,
            rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
            rapidocr_detector_limit_type=rapidocr_detector_limit_type,
            rapidocr_recognition_profile=rapidocr_recognition_profile,
            rapidocr_rec_batch_num=rapidocr_rec_batch_num,
            rapidocr_min_text_area=rapidocr_min_text_area,
        )
        if use_raw_cache_key
        else None
    )
    if cache_key is not None:
        cached = read_ocr_cache(cache_key)
        if cached is not None:
            return list(cached)

    prepared_composited_alpha = composited_alpha
    visual_cache_key: str | None = None
    near_visual_cache_key: str | None = None
    coarse_visual_cache_key: str | None = None
    canonical_visual_cache_key: str | None = None
    canonical_origin = (0.0, 0.0)
    canonical_cache_checked = False
    if cache:
        if prepared_bgr is not None:
            prepared_bgr = np.ascontiguousarray(prepared_bgr)
        else:
            prepared_bgr, prepared_composited_alpha = load_rapidocr_bgr(image_path)
        visual_cache_key = ocr_visual_cache_key(
            prepared_bgr,
            use_tesseract=tesseract_is_available(),
            rapidocr_max_dimension=rapidocr_max_dimension,
            rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
            rapidocr_detector_limit_type=rapidocr_detector_limit_type,
            rapidocr_recognition_profile=rapidocr_recognition_profile,
            rapidocr_rec_batch_num=rapidocr_rec_batch_num,
            rapidocr_min_text_area=rapidocr_min_text_area,
        )
        if visual_cache_key is not None and visual_cache_key != cache_key:
            cached = read_ocr_cache(visual_cache_key)
            if cached is not None:
                if cache_key is not None:
                    write_ocr_cache(cache_key, list(cached))
                return list(cached)
        canonical_bgr, canonical_origin = canonical_ocr_bgr(prepared_bgr)
        canonical_visual_cache_key = ocr_canonical_visual_cache_key(
            canonical_bgr,
            use_tesseract=tesseract_is_available(),
            rapidocr_max_dimension=rapidocr_max_dimension,
            rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
            rapidocr_detector_limit_type=rapidocr_detector_limit_type,
            rapidocr_recognition_profile=rapidocr_recognition_profile,
            rapidocr_rec_batch_num=rapidocr_rec_batch_num,
            rapidocr_min_text_area=rapidocr_min_text_area,
        )
        canonical_trimmed = canonical_ocr_bgr_trimmed(prepared_bgr, canonical_bgr, canonical_origin)
        if (
            canonical_trimmed
            and canonical_visual_cache_key is not None
            and canonical_visual_cache_key not in {cache_key, visual_cache_key}
        ):
            canonical_cache_checked = True
            cached = read_ocr_cache(canonical_visual_cache_key)
            if cached is not None:
                labels = shift_ocr_labels(cached, canonical_origin[0], canonical_origin[1])
                if cache_key is not None:
                    write_ocr_cache(cache_key, labels)
                if visual_cache_key is not None:
                    write_ocr_cache(visual_cache_key, labels)
                return labels
        near_visual_cache_key = ocr_near_visual_cache_key(
            prepared_bgr,
            use_tesseract=tesseract_is_available(),
            rapidocr_max_dimension=rapidocr_max_dimension,
            rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
            rapidocr_detector_limit_type=rapidocr_detector_limit_type,
            rapidocr_recognition_profile=rapidocr_recognition_profile,
            rapidocr_rec_batch_num=rapidocr_rec_batch_num,
            rapidocr_min_text_area=rapidocr_min_text_area,
        )
        if near_visual_cache_key is not None and near_visual_cache_key not in {cache_key, visual_cache_key}:
            cached = read_ocr_cache(near_visual_cache_key)
            if cached is not None:
                labels = list(cached)
                if cache_key is not None:
                    write_ocr_cache(cache_key, labels)
                if visual_cache_key is not None:
                    write_ocr_cache(visual_cache_key, labels)
                return labels
        coarse_visual_cache_key = ocr_coarse_visual_cache_key(
            prepared_bgr,
            use_tesseract=tesseract_is_available(),
            rapidocr_max_dimension=rapidocr_max_dimension,
            rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
            rapidocr_detector_limit_type=rapidocr_detector_limit_type,
            rapidocr_recognition_profile=rapidocr_recognition_profile,
            rapidocr_rec_batch_num=rapidocr_rec_batch_num,
            rapidocr_min_text_area=rapidocr_min_text_area,
        )
        if coarse_visual_cache_key is not None and coarse_visual_cache_key not in {
            cache_key,
            visual_cache_key,
            near_visual_cache_key,
        }:
            cached = read_ocr_cache(coarse_visual_cache_key)
            if cached is not None:
                labels = list(cached)
                if cache_key is not None:
                    write_ocr_cache(cache_key, labels)
                if visual_cache_key is not None:
                    write_ocr_cache(visual_cache_key, labels)
                if near_visual_cache_key is not None:
                    write_ocr_cache(near_visual_cache_key, labels)
                return labels
        if (
            not canonical_cache_checked
            and canonical_visual_cache_key is not None
            and canonical_visual_cache_key not in {
                cache_key,
                visual_cache_key,
                near_visual_cache_key,
                coarse_visual_cache_key,
            }
        ):
            cached = read_ocr_cache(canonical_visual_cache_key)
            if cached is not None:
                labels = shift_ocr_labels(cached, canonical_origin[0], canonical_origin[1])
                if cache_key is not None:
                    write_ocr_cache(cache_key, labels)
                if visual_cache_key is not None:
                    write_ocr_cache(visual_cache_key, labels)
                if near_visual_cache_key is not None:
                    write_ocr_cache(near_visual_cache_key, labels)
                if coarse_visual_cache_key is not None:
                    write_ocr_cache(coarse_visual_cache_key, labels)
                return labels

    rapidocr_kwargs: dict[str, int | float] = {}
    if rapidocr_max_dimension is not None:
        rapidocr_kwargs["rapidocr_max_dimension"] = rapidocr_max_dimension
    if rapidocr_min_text_area is not None:
        rapidocr_kwargs["rapidocr_min_text_area"] = rapidocr_min_text_area
    rapid_words: list[OcrLabel] = run_rapidocr_words(
        image_path,
        prepared_bgr=prepared_bgr,
        composited_alpha=prepared_composited_alpha,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        **rapidocr_kwargs,
    )
    words: list[OcrLabel] = list(rapid_words)
    used_tesseract_fallback = False
    if count_useful_labels(words) < TESSERACT_FALLBACK_MIN_USEFUL_LABELS and tesseract_is_available():
        if prepared_bgr_shape_differs_from_source():
            words = run_tesseract_array(prepared_bgr)
        else:
            words = run_tesseract_words(image_path)
        words = [word for word in words if is_useful_text(word.text)]
        if len(words) < 80:
            preprocessed_words = (
                run_preprocessed_tesseract_bgr(prepared_bgr)
                if prepared_bgr_shape_differs_from_source()
                else run_preprocessed_tesseract_words(image_path)
            )
            words.extend(word for word in preprocessed_words if is_useful_text(word.text))
        used_tesseract_fallback = True
    if used_tesseract_fallback and count_useful_labels(words) < TESSERACT_FALLBACK_MIN_USEFUL_LABELS:
        words.extend(rapid_words)
    elif used_tesseract_fallback:
        words.extend(
            word for word in rapid_words if word.confidence >= 80.0 and is_useful_text(word.text)
        )
    words = dedupe_labels(words)
    labels = list(words)
    labels.extend(group_line_labels(words))
    labels.extend(group_stacked_labels(words))
    labels = dedupe_labels(labels)
    labels = filter_ocr_labels_to_image_bounds(labels, label_shape)
    if cache_key is not None:
        write_ocr_cache(cache_key, labels)
    if visual_cache_key is not None and visual_cache_key != cache_key:
        write_ocr_cache(visual_cache_key, labels)
    if near_visual_cache_key is not None and near_visual_cache_key not in {cache_key, visual_cache_key}:
        write_ocr_cache(near_visual_cache_key, labels)
    if coarse_visual_cache_key is not None and coarse_visual_cache_key not in {
        cache_key,
        visual_cache_key,
        near_visual_cache_key,
    }:
        write_ocr_cache(coarse_visual_cache_key, labels)
    if canonical_visual_cache_key is not None and canonical_visual_cache_key not in {
        cache_key,
        visual_cache_key,
        near_visual_cache_key,
        coarse_visual_cache_key,
    }:
        write_ocr_cache(
            canonical_visual_cache_key,
            shift_ocr_labels(labels, -canonical_origin[0], -canonical_origin[1]),
        )
    return labels


def extract_ocr_labels_from_rgb(
    image_path: str | Path,
    rgb: np.ndarray,
    *,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
    allow_tesseract_fallback: bool = True,
    cache: bool = True,
) -> list[OcrLabel]:
    return extract_ocr_labels(
        image_path,
        prepared_bgr=rgb_to_bgr(rgb),
        rapidocr_max_dimension=rapidocr_max_dimension,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        rapidocr_min_text_area=rapidocr_min_text_area,
        allow_tesseract_fallback=allow_tesseract_fallback,
        cache=cache,
    )


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray | None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ocr_label_target_shape(
    image_path: str | Path,
    prepared_bgr: np.ndarray | None,
) -> tuple[float, float] | None:
    if prepared_bgr is not None:
        height, width = prepared_bgr.shape[:2]
        return float(height), float(width)
    shape = source_image_shape(Path(image_path))
    if shape is None:
        return None
    height, width = shape
    return float(height), float(width)


def ocr_cache_key(
    image_path: str | Path,
    *,
    use_tesseract: bool,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
) -> str | None:
    try:
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    except OSError:
        return None
    return ocr_cache_key_for_digest(
        "raw-sha256",
        digest,
        use_tesseract=use_tesseract,
        rapidocr_max_dimension=rapidocr_max_dimension,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        rapidocr_min_text_area=rapidocr_min_text_area,
    )


def ocr_visual_cache_key(
    bgr: np.ndarray | None,
    *,
    use_tesseract: bool,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
) -> str | None:
    if bgr is None:
        return None
    digest = hashlib.sha256()
    digest.update(b"bgr")
    digest.update(str(tuple(bgr.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(bgr).data)
    return ocr_cache_key_for_digest(
        "visual-bgr-sha256",
        digest.hexdigest(),
        use_tesseract=use_tesseract,
        rapidocr_max_dimension=rapidocr_max_dimension,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        rapidocr_min_text_area=rapidocr_min_text_area,
    )


def ocr_near_visual_cache_key(
    bgr: np.ndarray | None,
    *,
    use_tesseract: bool,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
) -> str | None:
    return ocr_quantized_visual_cache_key(
        bgr,
        use_tesseract=use_tesseract,
        rapidocr_max_dimension=rapidocr_max_dimension,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        rapidocr_min_text_area=rapidocr_min_text_area,
        mask=OCR_VISUAL_CACHE_QUANTIZATION_MASK,
        digest_kind="visual-bgr6-sha256",
        digest_tag=b"bgr-quantized",
    )


def ocr_coarse_visual_cache_key(
    bgr: np.ndarray | None,
    *,
    use_tesseract: bool,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
) -> str | None:
    return ocr_quantized_visual_cache_key(
        bgr,
        use_tesseract=use_tesseract,
        rapidocr_max_dimension=rapidocr_max_dimension,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        rapidocr_min_text_area=rapidocr_min_text_area,
        mask=OCR_COARSE_VISUAL_CACHE_QUANTIZATION_MASK,
        digest_kind="visual-bgr5-sha256",
        digest_tag=b"bgr-coarse-quantized",
    )


def ocr_quantized_visual_cache_key(
    bgr: np.ndarray | None,
    *,
    use_tesseract: bool,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
    mask: int,
    digest_kind: str,
    digest_tag: bytes,
) -> str | None:
    if bgr is None:
        return None
    quantized = np.bitwise_and(np.ascontiguousarray(bgr), mask)
    digest = hashlib.sha256()
    digest.update(digest_tag)
    digest.update(str(tuple(quantized.shape)).encode("ascii"))
    digest.update(quantized.data)
    return ocr_cache_key_for_digest(
        digest_kind,
        digest.hexdigest(),
        use_tesseract=use_tesseract,
        rapidocr_max_dimension=rapidocr_max_dimension,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        rapidocr_min_text_area=rapidocr_min_text_area,
    )


def ocr_canonical_visual_cache_key(
    bgr: np.ndarray | None,
    *,
    use_tesseract: bool,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
) -> str | None:
    if bgr is None:
        return None
    digest = hashlib.sha256()
    digest.update(b"bgr-canonical-content")
    digest.update(str(tuple(bgr.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(bgr).data)
    return ocr_cache_key_for_digest(
        "visual-canonical-bgr-sha256",
        digest.hexdigest(),
        use_tesseract=use_tesseract,
        rapidocr_max_dimension=rapidocr_max_dimension,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
        rapidocr_detector_limit_type=rapidocr_detector_limit_type,
        rapidocr_recognition_profile=rapidocr_recognition_profile,
        rapidocr_rec_batch_num=rapidocr_rec_batch_num,
        rapidocr_min_text_area=rapidocr_min_text_area,
    )


def canonical_ocr_bgr(bgr: np.ndarray | None) -> tuple[np.ndarray | None, tuple[float, float]]:
    if bgr is None or bgr.ndim != 3 or bgr.shape[0] < 3 or bgr.shape[1] < 3:
        return bgr, (0.0, 0.0)
    contiguous = np.ascontiguousarray(bgr)
    border_color = canonical_border_color(contiguous)

    height, width = contiguous.shape[:2]
    top = leading_matching_border_rows(contiguous, border_color, reverse=False)
    bottom_trim = leading_matching_border_rows(contiguous, border_color, reverse=True)
    left = leading_matching_border_cols(contiguous, border_color, reverse=False)
    right_trim = leading_matching_border_cols(contiguous, border_color, reverse=True)
    bottom = height - bottom_trim
    right = width - right_trim
    if top >= bottom or left >= right:
        return contiguous, (0.0, 0.0)
    if top == 0 and left == 0 and bottom == height and right == width:
        return contiguous, (0.0, 0.0)
    return np.ascontiguousarray(contiguous[top:bottom, left:right]), (float(left), float(top))


def canonical_ocr_bgr_trimmed(
    original_bgr: np.ndarray | None,
    canonical_bgr: np.ndarray | None,
    origin: tuple[float, float],
) -> bool:
    if original_bgr is None or canonical_bgr is None:
        return False
    return origin != (0.0, 0.0) or original_bgr.shape[:2] != canonical_bgr.shape[:2]


def canonical_border_color(bgr: np.ndarray) -> np.ndarray:
    border_samples = np.concatenate(
        (
            bgr[0, :, :],
            bgr[-1, :, :],
            bgr[:, 0, :],
            bgr[:, -1, :],
        ),
        axis=0,
    )
    return np.median(border_samples.astype(np.int16), axis=0)


def leading_matching_border_rows(bgr: np.ndarray, border_color: np.ndarray, *, reverse: bool) -> int:
    height = bgr.shape[0]
    count = 0
    indexes = range(height - 1, -1, -1) if reverse else range(height)
    for index in indexes:
        if not border_pixels_match(bgr[index, :, :], border_color):
            break
        count += 1
    return count


def leading_matching_border_cols(bgr: np.ndarray, border_color: np.ndarray, *, reverse: bool) -> int:
    width = bgr.shape[1]
    count = 0
    indexes = range(width - 1, -1, -1) if reverse else range(width)
    for index in indexes:
        if not border_pixels_match(bgr[:, index, :], border_color):
            break
        count += 1
    return count


def border_pixels_match(pixels: np.ndarray, border_color: np.ndarray) -> bool:
    delta = np.max(np.abs(pixels.astype(np.int16) - border_color), axis=1)
    return bool(np.mean(delta <= OCR_BORDER_COLOR_TOLERANCE) >= OCR_BORDER_ROW_MATCH_RATIO)


def shift_ocr_labels(labels: tuple[OcrLabel, ...] | list[OcrLabel], dx: float, dy: float) -> list[OcrLabel]:
    if dx == 0.0 and dy == 0.0:
        return list(labels)
    return [
        OcrLabel(
            text=label.text,
            x=label.x + dx,
            y=label.y + dy,
            width=label.width,
            height=label.height,
            confidence=label.confidence,
        )
        for label in labels
    ]


def effective_rapidocr_max_dimension(rapidocr_max_dimension: int | None = None) -> int:
    if rapidocr_max_dimension is None:
        return RAPIDOCR_MAX_DIMENSION
    return max(0, int(rapidocr_max_dimension))


def effective_rapidocr_detector_limit_override(rapidocr_detector_limit_side_len: int | None = None) -> int:
    if rapidocr_detector_limit_side_len is None:
        return 0
    return max(0, int(rapidocr_detector_limit_side_len))


def normalized_rapidocr_detector_limit_type(rapidocr_detector_limit_type: str | None = None) -> str:
    normalized = (rapidocr_detector_limit_type or "").strip().lower()
    if normalized in {"max", "min"}:
        return normalized
    return RAPIDOCR_DETECTOR_LIMIT_TYPE_DEFAULT


def normalized_rapidocr_recognition_profile(rapidocr_recognition_profile: str | None = None) -> str:
    normalized = (rapidocr_recognition_profile or "").strip().lower()
    if normalized in {"", "default", "ppocrv4", "ch-ppocrv4"}:
        return RAPIDOCR_RECOGNITION_PROFILE_DEFAULT
    if normalized in {"en-ppocrv5", "ppocrv5-en", "v5-en"}:
        return RAPIDOCR_RECOGNITION_PROFILE_EN_PPOCRV5
    return RAPIDOCR_RECOGNITION_PROFILE_DEFAULT


def effective_rapidocr_rec_batch_num(rapidocr_rec_batch_num: int | None = None) -> int:
    if rapidocr_rec_batch_num is None:
        return RAPIDOCR_REC_BATCH_NUM
    return max(1, int(rapidocr_rec_batch_num))


def rapidocr_recognition_profile_asset_signature(rapidocr_recognition_profile: str) -> str:
    if rapidocr_recognition_profile != RAPIDOCR_RECOGNITION_PROFILE_EN_PPOCRV5:
        return ""
    return rapidocr_english_ppocrv5_asset_signature()


def ocr_cache_key_for_digest(
    digest_kind: str,
    digest: str,
    *,
    use_tesseract: bool,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
) -> str:
    engine = "tesseract" if use_tesseract else "rapidocr"
    effective_max_dimension = effective_rapidocr_max_dimension(rapidocr_max_dimension)
    effective_detector_limit_type = normalized_rapidocr_detector_limit_type(rapidocr_detector_limit_type)
    effective_recognition_profile = normalized_rapidocr_recognition_profile(rapidocr_recognition_profile)
    effective_rec_batch_num = effective_rapidocr_rec_batch_num(rapidocr_rec_batch_num)
    recognition_asset_signature = rapidocr_recognition_profile_asset_signature(effective_recognition_profile)
    return hashlib.sha256(
        (
            f"{OCR_CACHE_VERSION}:{engine}:rapidocr-max-dim={effective_max_dimension}:"
            f"rapidocr-det-limit-override={effective_rapidocr_detector_limit_override(rapidocr_detector_limit_side_len)}:"
            f"rapidocr-det-limit-type-override={effective_detector_limit_type}:"
            f"rapidocr-recognition-profile={effective_recognition_profile}:"
            f"rapidocr-recognition-asset={recognition_asset_signature}:"
            f"rapidocr-det-limit={RAPIDOCR_DET_LIMIT_SIDE_LEN}:"
            f"rapidocr-large-det-limit={RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN}:"
            f"rapidocr-bright-blue-det-limit={RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN}:"
            f"rapidocr-bright-blue-det-limit-type={RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE}:"
            f"rapidocr-large-det-min={RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION}:"
            f"rapidocr-native-array-min={RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION}:"
            f"rapidocr-cls-batch={RAPIDOCR_CLS_BATCH_NUM}:"
            f"rapidocr-rec-batch={effective_rec_batch_num}:"
            f"rapidocr-cls-retry-min={RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS}:"
            f"fast-text-rescue-area={round(float(FAST_TEXT_OCR_RESCUE_MIN_AREA), 4)}:"
            f"fast-text-rescue-aspect={round(float(FAST_TEXT_OCR_RESCUE_MIN_ASPECT), 4)}:"
            f"rapidocr-min-text-area={round(float(rapidocr_min_text_area or 0.0), 4)}:"
            f"tesseract-fallback-min={TESSERACT_FALLBACK_MIN_USEFUL_LABELS}:"
            f"deps={ocr_cache_dependency_signature()}:"
            f"{digest_kind}:{digest}"
        ).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def ocr_cache_dependency_signature() -> str:
    return runtime_dependency_signature(OCR_CACHE_DEPENDENCY_PACKAGES)


def read_ocr_cache(cache_key: str) -> tuple[OcrLabel, ...] | None:
    with _OCR_MEMORY_CACHE_LOCK:
        cached = _OCR_MEMORY_CACHE.get(cache_key)
        if cached is not None:
            _OCR_MEMORY_CACHE.move_to_end(cache_key)
            return cached
    if not OCR_DISK_CACHE_ENABLED:
        return None
    cache_path = OCR_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        labels = tuple(OcrLabel(**item) for item in data if isinstance(item, dict))
    except Exception:
        return None
    remember_ocr_memory_cache(cache_key, labels)
    return labels


def write_ocr_cache(cache_key: str, labels: list[OcrLabel]) -> None:
    cached = tuple(labels)
    remember_ocr_memory_cache(cache_key, cached)
    if not OCR_DISK_CACHE_ENABLED:
        return
    cache_path = OCR_CACHE_DIR / f"{cache_key}.json"
    tmp_path = ocr_cache_tmp_path(cache_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([label.__dict__ for label in cached], separators=(",", ":"))
        tmp_path.write_text(payload)
        tmp_path.replace(cache_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        return


def remember_ocr_memory_cache(cache_key: str, labels: tuple[OcrLabel, ...]) -> None:
    with _OCR_MEMORY_CACHE_LOCK:
        _OCR_MEMORY_CACHE[cache_key] = labels
        _OCR_MEMORY_CACHE.move_to_end(cache_key)
        while len(_OCR_MEMORY_CACHE) > OCR_MEMORY_CACHE_MAX:
            _OCR_MEMORY_CACHE.popitem(last=False)


def ocr_cache_tmp_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def run_tesseract_words(image_path: str | Path) -> list[OcrLabel]:
    command = ["tesseract", str(image_path), "stdout", "--psm", "11", "tsv"]
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    return parse_tesseract_tsv(completed.stdout)


def run_preprocessed_tesseract_words(image_path: str | Path) -> list[OcrLabel]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    return run_preprocessed_tesseract_bgr(bgr)


def run_preprocessed_tesseract_bgr(bgr: np.ndarray) -> list[OcrLabel]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    service_fill = (
        ((hue >= 92) & (hue <= 116) & (saturation >= 55) & (value >= 85))
        | ((hue >= 75) & (hue <= 105) & (saturation >= 35) & (value >= 45) & (value <= 210))
    )

    neutralized = rgb.copy()
    neutralized[service_fill] = (245, 245, 245)
    neutral_gray = cv2.cvtColor(neutralized, cv2.COLOR_RGB2GRAY)
    neutral_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(neutral_gray)
    blur = cv2.GaussianBlur(neutral_gray, (0, 0), 3)
    high_pass = cv2.addWeighted(neutral_gray, 1.8, blur, -0.8, 0)
    dark_ink = ((neutral_gray < 175).astype(np.uint8) * 255)
    light_map_text = (((value > 95) & (saturation < 150)).astype(np.uint8) * 255)
    variants = [
        (gray, 1.0, 1.0),
        (cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray), 1.0, 1.0),
        ((((value > 145) & (saturation < 120)).astype(np.uint8) * 255), 1.0, 1.0),
        (cv2.resize(light_map_text, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), 3.0, 3.0),
        (cv2.resize(neutral_clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0, 2.0),
        (cv2.resize(high_pass, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0, 2.0),
        (cv2.resize(dark_ink, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), 2.0, 2.0),
    ]
    words: list[OcrLabel] = []
    max_workers = max(1, min(4, len(variants), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_tesseract_array, variant, scale_x=scale_x, scale_y=scale_y)
            for variant, scale_x, scale_y in variants
        ]
        for future in futures:
            words.extend(future.result())
    return words


def run_rapidocr_words(
    image_path: str | Path,
    *,
    prepared_bgr: np.ndarray | None = None,
    composited_alpha: bool = False,
    rapidocr_max_dimension: int | None = None,
    rapidocr_detector_limit_side_len: int | None = None,
    rapidocr_detector_limit_type: str | None = None,
    rapidocr_recognition_profile: str | None = None,
    rapidocr_rec_batch_num: int | None = None,
    rapidocr_min_text_area: float | None = None,
) -> list[OcrLabel]:
    profile_enabled = rapidocr_profile_enabled()
    profile_started = time.perf_counter()
    input_started = time.perf_counter()
    ocr_input, scale_x, scale_y = rapidocr_input_array(
        image_path,
        prepared_bgr=prepared_bgr,
        composited_alpha=composited_alpha,
        rapidocr_max_dimension=rapidocr_max_dimension,
    )
    input_elapsed = time.perf_counter() - input_started
    label_shape = rapidocr_label_target_shape(ocr_input, scale_x=scale_x, scale_y=scale_y)
    detector_limit = rapidocr_detector_limit_for_input(
        ocr_input,
        rapidocr_detector_limit_side_len=rapidocr_detector_limit_side_len,
    )
    min_text_area = max(0.0, float(rapidocr_min_text_area or 0.0))
    recognition_profile = normalized_rapidocr_recognition_profile(rapidocr_recognition_profile)
    detector_limit_type = normalized_rapidocr_detector_limit_type(rapidocr_detector_limit_type)
    rec_batch_num = effective_rapidocr_rec_batch_num(rapidocr_rec_batch_num)
    header_region_filter = should_filter_rapidocr_header_region(
        min_text_area=min_text_area,
        recognition_profile=recognition_profile,
        detector_limit_type=detector_limit_type,
        rec_batch_num=rec_batch_num,
    )
    profile: dict[str, Any] | None = None
    if profile_enabled:
        profile = {
            "input_s": input_elapsed,
            "input_kind": "array" if isinstance(ocr_input, np.ndarray) else "path",
            "input_shape": list(ocr_input.shape[:2]) if isinstance(ocr_input, np.ndarray) else None,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "detector_limit": detector_limit,
            "detector_limit_type": detector_limit_type,
            "recognition_profile": recognition_profile,
            "rec_batch_num": rec_batch_num,
            "min_text_area": min_text_area,
            "classifier_retry": False,
            "header_region_filter": header_region_filter,
            "header_region_filter_used": 0,
            "footer_region_filter_used": 0,
        }
    try:
        engine = rapidocr_engine(detector_limit, recognition_profile, detector_limit_type, rec_batch_num)
        if profile_enabled:
            result, engine_profile = run_rapidocr_profiled_items(
                engine,
                ocr_input,
                min_text_area=min_text_area,
                header_region_filter=header_region_filter,
            )
            if profile is not None:
                profile.update(engine_profile)
        elif min_text_area > 0.0 or header_region_filter:
            result = run_rapidocr_filtered_items(
                engine,
                ocr_input,
                min_text_area=min_text_area,
                header_region_filter=header_region_filter,
            )
        else:
            result, _elapsed = engine(ocr_input, use_cls=False)
        labels = filter_ocr_labels_to_image_bounds(
            scale_rapidocr_labels(rapidocr_items_to_labels(result), scale_x, scale_y),
            label_shape,
        )
        if not should_retry_rapidocr_with_classifier(labels):
            if profile is not None:
                profile["total_s"] = time.perf_counter() - profile_started
                profile["label_count"] = len(labels)
                profile["useful_label_count"] = count_useful_labels(labels)
                profile.update(ocr_label_confidence_profile("label", labels))
                record_rapidocr_profile(profile)
            return labels
        classifier_started = time.perf_counter()
        result, _elapsed = rapidocr_classifier_engine(
            detector_limit,
            recognition_profile,
            detector_limit_type,
            rec_batch_num,
        )(ocr_input, use_cls=True)
        if profile is not None:
            profile["classifier_retry"] = True
            profile["classifier_total_s"] = time.perf_counter() - classifier_started
    except Exception:
        if profile is not None:
            profile["total_s"] = time.perf_counter() - profile_started
            profile["error"] = "rapidocr_failed"
            record_rapidocr_profile(profile)
        return []
    labels = scale_rapidocr_labels(rapidocr_items_to_labels(result), scale_x, scale_y)
    labels = filter_ocr_labels_to_image_bounds(labels, label_shape)
    if profile is not None:
        profile["total_s"] = time.perf_counter() - profile_started
        profile["label_count"] = len(labels)
        profile["useful_label_count"] = count_useful_labels(labels)
        profile.update(ocr_label_confidence_profile("label", labels))
        record_rapidocr_profile(profile)
    return labels


def rapidocr_label_target_shape(
    ocr_input: Path | np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float] | None:
    if isinstance(ocr_input, np.ndarray):
        height, width = ocr_input.shape[:2]
        return height / max(scale_y, 1e-9), width / max(scale_x, 1e-9)
    shape = source_image_shape(Path(ocr_input))
    if shape is None:
        return None
    height, width = shape
    return float(height), float(width)


def filter_ocr_labels_to_image_bounds(
    labels: list[OcrLabel],
    image_shape: tuple[float, float] | None,
) -> list[OcrLabel]:
    if image_shape is None:
        return labels
    height, width = image_shape
    if height <= 0 or width <= 0:
        return labels
    margin = max(2.0, min(width, height) * 0.025)
    max_label_width = width + 2.0 * margin
    max_label_height = height + 2.0 * margin
    return [
        label
        for label in labels
        if -margin <= label.x <= width + margin
        and -margin <= label.y <= height + margin
        and 0.0 < label.width <= max_label_width
        and 0.0 < label.height <= max_label_height
    ]


def run_rapidocr_filtered_items(
    engine,
    ocr_input: Path | np.ndarray,
    *,
    min_text_area: float,
    header_region_filter: bool = False,
):
    img = engine.load_img(ocr_input)
    raw_h, raw_w = img.shape[:2]
    img, ratio_h, ratio_w = engine.preprocess(img)
    op_record = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
    img, op_record = engine.maybe_add_letterbox(img, op_record)
    dt_boxes, _det_elapsed = engine.auto_text_det(img)
    if dt_boxes is None:
        return None
    selected, _header_filter_used, _footer_filter_used = select_rapidocr_boxes(
        list(dt_boxes),
        img.shape[:2],
        min_text_area=min_text_area,
        header_region_filter=header_region_filter,
    )
    if not selected:
        return None
    crop_images = engine.get_crop_img_list(img, selected)
    rec_res, _rec_elapsed = engine.text_rec(crop_images, False)
    origin_boxes = engine._get_origin_points(selected, op_record, raw_h, raw_w)
    result, _elapsed = engine.get_final_res(origin_boxes, None, rec_res, 0.0, 0.0, 0.0)
    return result


def run_rapidocr_profiled_items(
    engine,
    ocr_input: Path | np.ndarray,
    *,
    min_text_area: float,
    header_region_filter: bool = False,
) -> tuple[Any, dict[str, Any]]:
    profile: dict[str, Any] = {}
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    img = engine.load_img(ocr_input)
    profile["load_img_s"] = time.perf_counter() - load_started
    raw_h, raw_w = img.shape[:2]
    preprocess_started = time.perf_counter()
    img, ratio_h, ratio_w = engine.preprocess(img)
    op_record = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
    img, op_record = engine.maybe_add_letterbox(img, op_record)
    profile["preprocess_s"] = time.perf_counter() - preprocess_started
    dt_boxes, det_elapsed = engine.auto_text_det(img)
    profile["det_elapsed_s"] = float(det_elapsed or 0.0)
    if dt_boxes is None:
        profile.update(
            {
                "raw_box_count": 0,
                "selected_box_count": 0,
                "result_count": 0,
                "profiled_total_s": time.perf_counter() - total_started,
            }
        )
        return None, profile

    raw_boxes = list(dt_boxes)
    selected, header_filter_used, footer_filter_used = select_rapidocr_boxes(
        raw_boxes,
        img.shape[:2],
        min_text_area=min_text_area,
        header_region_filter=header_region_filter,
    )
    profile["raw_box_count"] = len(raw_boxes)
    profile["selected_box_count"] = len(selected)
    profile["header_region_filter_used"] = int(header_filter_used)
    profile["footer_region_filter_used"] = int(footer_filter_used)
    profile.update(rapidocr_box_area_profile("raw", raw_boxes))
    profile.update(rapidocr_box_area_profile("selected", selected))
    if not selected:
        profile.update(
            {
                "result_count": 0,
                "profiled_total_s": time.perf_counter() - total_started,
            }
        )
        return None, profile

    crop_started = time.perf_counter()
    crop_images = engine.get_crop_img_list(img, selected)
    profile["crop_s"] = time.perf_counter() - crop_started
    rec_res, rec_elapsed = engine.text_rec(crop_images, False)
    profile["rec_elapsed_s"] = float(rec_elapsed or 0.0)
    final_started = time.perf_counter()
    origin_boxes = engine._get_origin_points(selected, op_record, raw_h, raw_w)
    result, _elapsed = engine.get_final_res(origin_boxes, None, rec_res, 0.0, 0.0, 0.0)
    profile["final_s"] = time.perf_counter() - final_started
    profile["result_count"] = len(result) if isinstance(result, list) else 0
    profile["profiled_total_s"] = time.perf_counter() - total_started
    return result, profile


def should_filter_rapidocr_header_region(
    *,
    min_text_area: float,
    recognition_profile: str,
    detector_limit_type: str,
    rec_batch_num: int,
) -> bool:
    if min_text_area > 0.0:
        return False
    if recognition_profile != RAPIDOCR_RECOGNITION_PROFILE_DEFAULT:
        return False
    if detector_limit_type != RAPIDOCR_DETECTOR_LIMIT_TYPE_DEFAULT:
        return False
    if RAPIDOCR_DARK_TEAL_REC_BATCH_NUM <= 0:
        return False
    if RAPIDOCR_DARK_TEAL_REC_BATCH_NUM == RAPIDOCR_REC_BATCH_NUM:
        return False
    return rec_batch_num == RAPIDOCR_DARK_TEAL_REC_BATCH_NUM


def select_rapidocr_boxes(
    raw_boxes: list[np.ndarray],
    image_shape: tuple[int, int],
    *,
    min_text_area: float,
    header_region_filter: bool,
) -> tuple[list[np.ndarray], bool, bool]:
    selected = [
        box
        for box in raw_boxes
        if min_text_area <= 0.0 or rapidocr_box_area(box) >= min_text_area or rapidocr_rescue_text_box(box)
    ]
    if not header_region_filter:
        return selected, False, False
    header_selected = select_rapidocr_header_region_boxes(selected, image_shape)
    if header_selected is None:
        return selected, False, False
    footer_selected = select_rapidocr_extreme_footer_boxes(header_selected, image_shape)
    if footer_selected is None:
        return header_selected, True, False
    return footer_selected, True, True


def select_rapidocr_header_region_boxes(
    boxes: list[np.ndarray],
    image_shape: tuple[int, int],
) -> list[np.ndarray] | None:
    height, width = image_shape
    if min(width, height) < RAPIDOCR_HEADER_REGION_FILTER_MIN_DIMENSION:
        return None
    if len(boxes) < RAPIDOCR_HEADER_REGION_FILTER_MIN_BOXES:
        return None
    header_cutoff_y = min(
        RAPIDOCR_HEADER_REGION_FILTER_MAX_Y,
        max(RAPIDOCR_HEADER_REGION_FILTER_MIN_Y, height * RAPIDOCR_HEADER_REGION_FILTER_Y_RATIO),
    )
    header_boxes = [box for box in boxes if rapidocr_box_center_y(box) < header_cutoff_y]
    if len(header_boxes) < RAPIDOCR_HEADER_REGION_FILTER_MIN_HEADER_BOXES:
        return None
    left_title_x = width * RAPIDOCR_HEADER_REGION_FILTER_LEFT_TITLE_X_RATIO
    wide_header_width = width * RAPIDOCR_HEADER_REGION_FILTER_WIDE_TEXT_RATIO
    selected = [
        box
        for box in boxes
        if (
            rapidocr_box_center_y(box) >= header_cutoff_y
            or rapidocr_box_center_x(box) <= left_title_x
            or rapidocr_box_width(box) >= wide_header_width
        )
    ]
    if len(selected) < RAPIDOCR_HEADER_REGION_FILTER_MIN_SELECTED_BOXES:
        return None
    if len(boxes) - len(selected) < RAPIDOCR_HEADER_REGION_FILTER_MIN_REMOVED_BOXES:
        return None
    return selected


def select_rapidocr_extreme_footer_boxes(
    boxes: list[np.ndarray],
    image_shape: tuple[int, int],
) -> list[np.ndarray] | None:
    height, _width = image_shape
    if height <= 0:
        return None
    footer_cutoff_y = height * RAPIDOCR_HEADER_REGION_FILTER_FOOTER_Y_RATIO
    selected = [box for box in boxes if rapidocr_box_center_y(box) <= footer_cutoff_y]
    if len(selected) < RAPIDOCR_HEADER_REGION_FILTER_MIN_FOOTER_SELECTED_BOXES:
        return None
    if len(boxes) - len(selected) < RAPIDOCR_HEADER_REGION_FILTER_MIN_FOOTER_REMOVED_BOXES:
        return None
    return selected


def rapidocr_box_area(box: np.ndarray) -> float:
    return rapidocr_box_width(box) * rapidocr_box_height(box)


def rapidocr_box_aspect(box: np.ndarray) -> float:
    return rapidocr_box_width(box) / max(rapidocr_box_height(box), 1.0)


def rapidocr_box_width(box: np.ndarray) -> float:
    return max(float(np.linalg.norm(box[0] - box[1])), float(np.linalg.norm(box[2] - box[3])))


def rapidocr_box_height(box: np.ndarray) -> float:
    return max(float(np.linalg.norm(box[0] - box[3])), float(np.linalg.norm(box[1] - box[2])))


def rapidocr_box_center_x(box: np.ndarray) -> float:
    return float(sum(float(point[0]) for point in box) / 4.0)


def rapidocr_box_center_y(box: np.ndarray) -> float:
    return float(sum(float(point[1]) for point in box) / 4.0)


def rapidocr_rescue_text_box(box: np.ndarray) -> bool:
    if FAST_TEXT_OCR_RESCUE_MIN_AREA <= 0.0 or FAST_TEXT_OCR_RESCUE_MIN_ASPECT <= 0.0:
        return False
    return (
        rapidocr_box_area(box) >= FAST_TEXT_OCR_RESCUE_MIN_AREA
        and rapidocr_box_aspect(box) >= FAST_TEXT_OCR_RESCUE_MIN_ASPECT
    )


def rapidocr_detector_limit_for_input(
    ocr_input: Path | np.ndarray,
    *,
    rapidocr_detector_limit_side_len: int | None = None,
) -> int:
    detector_override = effective_rapidocr_detector_limit_override(rapidocr_detector_limit_side_len)
    if detector_override > 0:
        return detector_override
    if (
        RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN <= 0
        or RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION <= 0
        or not isinstance(ocr_input, np.ndarray)
    ):
        return RAPIDOCR_DET_LIMIT_SIDE_LEN
    height, width = ocr_input.shape[:2]
    if max(width, height) >= RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION:
        return RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN
    return RAPIDOCR_DET_LIMIT_SIDE_LEN


def warm_rapidocr_runtime() -> bool:
    warmed = _warm_rapidocr_runtime_cached()
    if not warmed:
        _warm_rapidocr_runtime_cached.cache_clear()
    return warmed


@lru_cache(maxsize=1)
def _warm_rapidocr_runtime_cached() -> bool:
    try:
        samples: dict[int, np.ndarray] = {}
        for (
            detector_limit,
            recognition_profile,
            detector_limit_type,
            rec_batch_num,
            warm_side,
        ) in rapidocr_warm_engine_sample_plan():
            engine = rapidocr_engine(detector_limit, recognition_profile, detector_limit_type, rec_batch_num)
            sample = samples.get(warm_side)
            if sample is None:
                sample = rapidocr_warm_sample_for_side(warm_side)
                samples[warm_side] = sample
            engine(sample, use_cls=False)
    except Exception:
        return False
    return True


warm_rapidocr_runtime.cache_clear = _warm_rapidocr_runtime_cached.cache_clear  # type: ignore[attr-defined]


def rapidocr_warm_engine_keys() -> list[tuple[int, str, str, int]]:
    keys: list[tuple[int, str, str, int]] = []
    generic_detector_limits = rapidocr_warm_detector_limits()
    for detector_limit in generic_detector_limits:
        key = (
            detector_limit,
            RAPIDOCR_RECOGNITION_PROFILE_DEFAULT,
            RAPIDOCR_DETECTOR_LIMIT_TYPE_DEFAULT,
            RAPIDOCR_REC_BATCH_NUM,
        )
        if detector_limit > 0 and key not in keys:
            keys.append(key)
    bright_blue_profile = normalized_rapidocr_recognition_profile(RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE)
    bright_blue_detector_type = normalized_rapidocr_detector_limit_type(RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE)
    if RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN > 0 and (
        bright_blue_profile != RAPIDOCR_RECOGNITION_PROFILE_DEFAULT
        or bright_blue_detector_type != RAPIDOCR_DETECTOR_LIMIT_TYPE_DEFAULT
        or RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN not in generic_detector_limits
    ):
        key = (
            RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            bright_blue_profile,
            bright_blue_detector_type,
            RAPIDOCR_REC_BATCH_NUM,
        )
        if key not in keys:
            keys.append(key)
    if RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN > 0:
        key = (
            RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            bright_blue_profile,
            bright_blue_detector_type,
            RAPIDOCR_REC_BATCH_NUM,
        )
        if key not in keys:
            keys.append(key)
    if RAPIDOCR_DARK_TEAL_REC_BATCH_NUM > 0 and RAPIDOCR_DARK_TEAL_REC_BATCH_NUM != RAPIDOCR_REC_BATCH_NUM:
        for detector_limit in generic_detector_limits:
            key = (
                detector_limit,
                RAPIDOCR_RECOGNITION_PROFILE_DEFAULT,
                RAPIDOCR_DETECTOR_LIMIT_TYPE_DEFAULT,
                RAPIDOCR_DARK_TEAL_REC_BATCH_NUM,
            )
            if detector_limit > 0 and key not in keys:
                keys.append(key)
    return keys


def rapidocr_warm_engine_sample_plan() -> list[tuple[int, str, str, int, int]]:
    generic_side = rapidocr_generic_warm_sample_side()
    bright_blue_warm_side = rapidocr_bright_blue_large_warm_sample_side()
    bright_blue_key = rapidocr_bright_blue_warm_key()
    svg_bright_blue_warm_side = rapidocr_svg_bright_blue_large_warm_sample_side()
    svg_bright_blue_key = rapidocr_svg_bright_blue_warm_key()
    large_warm_keys: dict[tuple[int, str, str, int], int] = {}
    if bright_blue_warm_side > 0:
        large_warm_keys[bright_blue_key] = bright_blue_warm_side
    if svg_bright_blue_warm_side > 0:
        large_warm_keys[svg_bright_blue_key] = max(
            svg_bright_blue_warm_side,
            large_warm_keys.get(svg_bright_blue_key, 0),
        )
    plan: list[tuple[int, str, str, int, int]] = []
    for key in rapidocr_warm_engine_keys():
        if key in large_warm_keys:
            continue
        plan.append((*key, generic_side))
    for key, warm_side in large_warm_keys.items():
        plan.append((*key, warm_side))
    return plan


def rapidocr_warm_sample() -> np.ndarray:
    return rapidocr_warm_sample_for_side(rapidocr_generic_warm_sample_side())


def rapidocr_generic_warm_sample_side() -> int:
    warm_side = RAPIDOCR_WARM_SAMPLE_MAX_DIMENSION
    if warm_side <= 0:
        warm_side = effective_rapidocr_max_dimension()
    if warm_side <= 0:
        warm_side = 1600
    return bounded_rapidocr_warm_sample_side(warm_side)


def rapidocr_bright_blue_large_warm_sample_side() -> int:
    if RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION <= 0:
        return 0
    if RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN <= 0:
        return 0
    warm_side = bounded_rapidocr_warm_sample_side(RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION)
    if warm_side <= rapidocr_generic_warm_sample_side():
        return 0
    return warm_side


def rapidocr_svg_bright_blue_large_warm_sample_side() -> int:
    if RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION <= 0:
        return 0
    if RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN <= 0:
        return 0
    warm_side = bounded_rapidocr_warm_sample_side(RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION)
    if warm_side <= rapidocr_generic_warm_sample_side():
        return 0
    return warm_side


def rapidocr_bright_blue_warm_key() -> tuple[int, str, str, int]:
    return (
        RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
        normalized_rapidocr_recognition_profile(RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE),
        normalized_rapidocr_detector_limit_type(RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE),
        RAPIDOCR_REC_BATCH_NUM,
    )


def rapidocr_svg_bright_blue_warm_key() -> tuple[int, str, str, int]:
    return (
        RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
        normalized_rapidocr_recognition_profile(RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE),
        normalized_rapidocr_detector_limit_type(RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE),
        RAPIDOCR_REC_BATCH_NUM,
    )


def bounded_rapidocr_warm_sample_side(warm_side: int) -> int:
    return max(384, min(1600, int(warm_side)))


def rapidocr_warm_sample_for_side(warm_side: int) -> np.ndarray:
    warm_side = bounded_rapidocr_warm_sample_side(warm_side)
    sample = np.full((warm_side, warm_side, 3), 255, dtype=np.uint8)
    font_scale = max(1.0, warm_side / 400.0)
    thickness = max(2, round(warm_side / 200))
    origin_x = max(18, round(warm_side * 0.1))
    for text, y_ratio in (("Miami", 0.28), ("Downtown", 0.52), ("Houston", 0.76)):
        cv2.putText(
            sample,
            text,
            (origin_x, round(warm_side * y_ratio)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return sample


def should_retry_rapidocr_with_classifier(labels: list[OcrLabel]) -> bool:
    return count_useful_labels(labels) < RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS


def scale_rapidocr_labels(labels: list[OcrLabel], scale_x: float, scale_y: float) -> list[OcrLabel]:
    if scale_x == 1.0 and scale_y == 1.0:
        return labels
    return [
        OcrLabel(
            text=label.text,
            x=label.x / scale_x,
            y=label.y / scale_y,
            width=label.width / scale_x,
            height=label.height / scale_y,
            confidence=label.confidence,
        )
        for label in labels
    ]


def rapidocr_input_array(
    image_path: str | Path,
    *,
    prepared_bgr: np.ndarray | None = None,
    composited_alpha: bool = False,
    rapidocr_max_dimension: int | None = None,
) -> tuple[Path | np.ndarray, float, float]:
    source_path = Path(image_path)
    max_ocr_dimension = effective_rapidocr_max_dimension(rapidocr_max_dimension)
    if max_ocr_dimension <= 0:
        return source_path, 1.0, 1.0
    prepared_shape: tuple[int, int] | None = None
    if prepared_bgr is None:
        bgr, composited_alpha = load_rapidocr_bgr(source_path)
    else:
        bgr = prepared_bgr
        prepared_shape = bgr.shape[:2]
    if bgr is None:
        return source_path, 1.0, 1.0
    height, width = bgr.shape[:2]
    max_dimension = max(width, height)
    if max_dimension <= max_ocr_dimension:
        if source_path.suffix.lower() == ".webp":
            return bgr, 1.0, 1.0
        if (
            RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION > 0
            and max_dimension >= RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION
        ):
            return bgr, 1.0, 1.0
        if composited_alpha:
            return bgr, 1.0, 1.0
        if prepared_shape is not None and prepared_shape != source_image_shape(source_path):
            return bgr, 1.0, 1.0
        return source_path, 1.0, 1.0
    scale = max_ocr_dimension / float(max_dimension)
    resized = cv2.resize(
        bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale, scale


def source_image_shape(image_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return None
    return height, width


def load_rapidocr_bgr(image_path: str | Path) -> tuple[np.ndarray | None, bool]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, False
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), False
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        bgr = image[:, :, :3]
        if np.all(alpha == 255):
            return np.ascontiguousarray(bgr), False
        from .extract import load_rgb

        rgb = load_rgb(image_path)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), True
    return image, False


def rapidocr_input_image(
    image_path: str | Path,
    *,
    rapidocr_max_dimension: int | None = None,
) -> tuple[Path, float, float]:
    source_path = Path(image_path)
    max_ocr_dimension = effective_rapidocr_max_dimension(rapidocr_max_dimension)
    if max_ocr_dimension <= 0:
        return source_path, 1.0, 1.0
    bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return source_path, 1.0, 1.0
    height, width = bgr.shape[:2]
    max_dimension = max(width, height)
    if max_dimension <= max_ocr_dimension:
        return source_path, 1.0, 1.0
    scale = max_ocr_dimension / float(max_dimension)
    resized = cv2.resize(
        bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    if not cv2.imwrite(tmp_path, resized):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return source_path, 1.0, 1.0
    return Path(tmp_path), scale, scale


@lru_cache(maxsize=8)
def rapidocr_engine(
    det_limit_side_len: int | None = None,
    recognition_profile: str | None = None,
    detector_limit_type: str | None = None,
    rec_batch_num: int | None = None,
):
    configure_rapidocr_onnxruntime_session_options()
    return rapidocr_engine_without_classifier(
        **rapidocr_engine_kwargs(det_limit_side_len, recognition_profile, detector_limit_type, rec_batch_num)
    )


@lru_cache(maxsize=8)
def rapidocr_classifier_engine(
    det_limit_side_len: int | None = None,
    recognition_profile: str | None = None,
    detector_limit_type: str | None = None,
    rec_batch_num: int | None = None,
):
    configure_rapidocr_onnxruntime_session_options()
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(
        **rapidocr_engine_kwargs(det_limit_side_len, recognition_profile, detector_limit_type, rec_batch_num)
    )


def rapidocr_engine_kwargs(
    det_limit_side_len: int | None = None,
    recognition_profile: str | None = None,
    detector_limit_type: str | None = None,
    rec_batch_num: int | None = None,
) -> dict[str, object]:
    kwargs = {
        "cls_batch_num": RAPIDOCR_CLS_BATCH_NUM,
        "rec_batch_num": effective_rapidocr_rec_batch_num(rec_batch_num),
    }
    det_limit = RAPIDOCR_DET_LIMIT_SIDE_LEN if det_limit_side_len is None else max(0, int(det_limit_side_len))
    if det_limit > 0:
        kwargs["det_limit_side_len"] = det_limit
    normalized_detector_limit_type = normalized_rapidocr_detector_limit_type(detector_limit_type)
    if normalized_detector_limit_type != RAPIDOCR_DETECTOR_LIMIT_TYPE_DEFAULT:
        kwargs["det_limit_type"] = normalized_detector_limit_type
    kwargs.update(rapidocr_recognition_profile_kwargs(recognition_profile))
    return kwargs


@lru_cache(maxsize=4)
def rapidocr_recognition_profile_kwargs(recognition_profile: str | None = None) -> dict[str, object]:
    profile = normalized_rapidocr_recognition_profile(recognition_profile)
    if profile != RAPIDOCR_RECOGNITION_PROFILE_EN_PPOCRV5:
        return {}
    asset_paths = rapidocr_english_ppocrv5_asset_paths()
    if asset_paths is None:
        return {}
    rec_model_path, rec_keys_path = asset_paths
    if not all(path.is_file() for path in (rec_model_path, rec_keys_path)):
        return {}
    return {
        "rec_model_path": str(rec_model_path),
        "rec_keys_path": str(rec_keys_path),
        "rec_img_shape": [3, 48, 320],
    }


def configure_rapidocr_onnxruntime_session_options() -> None:
    global _RAPIDOCR_SESSION_OPTIONS_PATCHED
    if _RAPIDOCR_SESSION_OPTIONS_PATCHED:
        return
    try:
        import onnxruntime as ort
        from rapidocr_onnxruntime.utils import OrtInferSession
    except Exception:
        return

    def init_sess_opts(config):
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = ONNXRUNTIME_ENABLE_CPU_MEM_ARENA
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count() or 1
        intra_op_num_threads = config.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = config.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        allow_spinning = "1" if ONNXRUNTIME_ALLOW_SPINNING else "0"
        sess_opt.add_session_config_entry("session.intra_op.allow_spinning", allow_spinning)
        sess_opt.add_session_config_entry("session.inter_op.allow_spinning", allow_spinning)
        return sess_opt

    OrtInferSession._init_sess_opts = staticmethod(init_sess_opts)
    _RAPIDOCR_SESSION_OPTIONS_PATCHED = True


def rapidocr_engine_without_classifier(**kwargs):
    from rapidocr_onnxruntime.cal_rec_boxes import CalRecBoxes
    from rapidocr_onnxruntime.ch_ppocr_det import TextDetector
    from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
    from rapidocr_onnxruntime.main import DEFAULT_CFG_PATH, RapidOCR
    from rapidocr_onnxruntime.utils import LoadImage, UpdateParameters, read_yaml, update_model_path

    class RapidOCRWithoutClassifier(RapidOCR):
        def __init__(self, **engine_kwargs):
            config = update_model_path(read_yaml(DEFAULT_CFG_PATH))
            if engine_kwargs:
                config = UpdateParameters()(config, **engine_kwargs)

            global_config = config["Global"]
            self.print_verbose = global_config["print_verbose"]
            self.text_score = global_config["text_score"]
            self.min_height = global_config["min_height"]
            self.width_height_ratio = global_config["width_height_ratio"]
            self.use_det = global_config["use_det"]
            self.text_det = TextDetector(config["Det"])
            # The map pipeline first runs RapidOCR with use_cls=False. Avoid
            # initializing the classifier ONNX session unless the sparse-label
            # fallback asks for it through rapidocr_classifier_engine().
            self.use_cls = False
            self.use_rec = global_config["use_rec"]
            self.text_rec = TextRecognizer(config["Rec"])
            self.load_img = LoadImage()
            self.max_side_len = global_config["max_side_len"]
            self.min_side_len = global_config["min_side_len"]
            self.cal_rec_boxes = CalRecBoxes()

    return RapidOCRWithoutClassifier(**kwargs)


def rapidocr_items_to_labels(items: object) -> list[OcrLabel]:
    if not isinstance(items, list):
        return []

    labels: list[OcrLabel] = []
    for item in items:
        try:
            box, raw_text, raw_score = item
            points = [(float(point[0]), float(point[1])) for point in box]
            text = clean_text(str(raw_text))
            score = float(raw_score)
        except Exception:
            continue
        if score < 0.35 or not is_useful_text(text):
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue
        labels.append(
            OcrLabel(
                text=text,
                x=(left + right) / 2.0,
                y=(top + bottom) / 2.0,
                width=width,
                height=height,
                confidence=max(0.0, min(100.0, score * 100.0)),
            )
        )
    return labels


def run_tesseract_array(image: np.ndarray, *, scale_x: float = 1.0, scale_y: float = 1.0) -> list[OcrLabel]:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(tmp_path, image)
        command = ["tesseract", tmp_path, "stdout", "--psm", "11", "tsv"]
        try:
            completed = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            return []
        if completed.returncode != 0:
            return []
        words = parse_tesseract_tsv(completed.stdout)
        if scale_x == 1.0 and scale_y == 1.0:
            return words
        return [
            OcrLabel(
                text=word.text,
                x=word.x / scale_x,
                y=word.y / scale_y,
                width=word.width / scale_x,
                height=word.height / scale_y,
                confidence=word.confidence,
            )
            for word in words
        ]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def parse_tesseract_tsv(tsv: str) -> list[OcrLabel]:
    rows = csv.DictReader(tsv.splitlines(), delimiter="\t")
    labels: list[OcrLabel] = []
    for row in rows:
        if row.get("level") != "5":
            continue
        text = clean_text(row.get("text", ""))
        if not text:
            continue
        try:
            confidence = float(row.get("conf", "-1"))
            if confidence < 20:
                continue
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
        except Exception:
            continue
        labels.append(
            OcrLabel(
                text=text,
                x=left + width / 2,
                y=top + height / 2,
                width=width,
                height=height,
                confidence=confidence,
            )
        )
    return labels


def group_line_labels(words: list[OcrLabel]) -> list[OcrLabel]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda word: (word.y, word.x))
    median_height = sorted(word.height for word in sorted_words)[len(sorted_words) // 2]
    y_threshold = max(10.0, median_height * 0.8)
    rows: list[list[OcrLabel]] = []
    for word in sorted_words:
        for row in rows:
            if abs(row[0].y - word.y) <= y_threshold:
                row.append(word)
                break
        else:
            rows.append([word])

    labels: list[OcrLabel] = []
    for row in rows:
        row = sorted(row, key=lambda word: word.x)
        current: list[OcrLabel] = []
        for word in row:
            if not current:
                current = [word]
                continue
            previous = current[-1]
            gap = word.x - previous.x - previous.width / 2 - word.width / 2
            if gap <= max(80.0, median_height * 3.5):
                current.append(word)
            else:
                labels.extend(join_windows(current))
                current = [word]
        labels.extend(join_windows(current))
    return labels


def join_windows(words: list[OcrLabel]) -> list[OcrLabel]:
    if len(words) < 2:
        return []
    labels: list[OcrLabel] = []
    for size in (2, 3):
        if len(words) < size:
            continue
        for start in range(0, len(words) - size + 1):
            chunk = words[start : start + size]
            text = clean_text(" ".join(word.text for word in chunk))
            if not is_useful_text(text):
                continue
            left = min(word.x - word.width / 2 for word in chunk)
            right = max(word.x + word.width / 2 for word in chunk)
            top = min(word.y - word.height / 2 for word in chunk)
            bottom = max(word.y + word.height / 2 for word in chunk)
            labels.append(
                OcrLabel(
                    text=text,
                    x=(left + right) / 2,
                    y=(top + bottom) / 2,
                    width=right - left,
                    height=bottom - top,
                    confidence=sum(word.confidence for word in chunk) / len(chunk),
                )
            )
    return labels


def group_stacked_labels(words: list[OcrLabel]) -> list[OcrLabel]:
    if not words:
        return []
    heights = sorted(max(1.0, word.height) for word in words)
    median_height = heights[len(heights) // 2]
    labels: list[OcrLabel] = []
    for top in words:
        for bottom in words:
            if bottom.y <= top.y:
                continue
            if abs(top.x - bottom.x) > max(90.0, top.width, bottom.width):
                continue
            vertical_gap = bottom.y - top.y - top.height / 2 - bottom.height / 2
            vertical_threshold = min(
                48.0,
                max(14.0, median_height * 3.0, (top.height + bottom.height) * 1.8),
            )
            if vertical_gap < -8.0 or vertical_gap > vertical_threshold:
                continue
            text = clean_text(f"{top.text} {bottom.text}")
            if not is_useful_text(text):
                continue
            left = min(top.x - top.width / 2, bottom.x - bottom.width / 2)
            right = max(top.x + top.width / 2, bottom.x + bottom.width / 2)
            upper = min(top.y - top.height / 2, bottom.y - bottom.height / 2)
            lower = max(top.y + top.height / 2, bottom.y + bottom.height / 2)
            labels.append(
                OcrLabel(
                    text=text,
                    x=(left + right) / 2,
                    y=(upper + lower) / 2,
                    width=right - left,
                    height=lower - upper,
                    confidence=(top.confidence + bottom.confidence) / 2,
                )
            )
    return labels


def dedupe_labels(labels: list[OcrLabel]) -> list[OcrLabel]:
    best: dict[tuple[str, int, int], OcrLabel] = {}
    for label in labels:
        key = (label.text.lower(), round(label.x / 20), round(label.y / 20))
        old = best.get(key)
        if old is None or label.confidence > old.confidence:
            best[key] = label
    return sorted(best.values(), key=lambda label: label.confidence, reverse=True)


def count_useful_labels(labels: list[OcrLabel]) -> int:
    return sum(1 for label in labels if is_useful_text(label.text))


def clean_text(text: str) -> str:
    text = text.strip().replace("|", "I")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"[^A-Za-z0-9 &'/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -/'")
    return text


def is_useful_text(text: str) -> bool:
    if len(text) < 3:
        return False
    if text.isdigit():
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= max(3, len(text.replace(" ", "")) // 2)
