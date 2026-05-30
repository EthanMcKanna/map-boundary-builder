from __future__ import annotations

import os
from typing import Any


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return default


RAPIDOCR_MAX_DIMENSION = env_int("MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION", 1600)
RAPIDOCR_PURPLE_FILL_MAX_DIMENSION = env_int("MAP_BOUNDARY_RAPIDOCR_PURPLE_FILL_MAX_DIMENSION", 800)
PROVIDER_UI_RAPIDOCR_MAX_DIMENSION = env_int("MAP_BOUNDARY_PROVIDER_UI_RAPIDOCR_MAX_DIMENSION", 1200)
RAPIDOCR_DET_LIMIT_SIDE_LEN = env_int("MAP_BOUNDARY_RAPIDOCR_DET_LIMIT_SIDE_LEN", 608)
RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN = env_int(
    "MAP_BOUNDARY_RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN",
    608,
)
RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION",
    1000,
)
RAPIDOCR_CLS_BATCH_NUM = env_int("MAP_BOUNDARY_RAPIDOCR_CLS_BATCH_NUM", 24, minimum=1)
RAPIDOCR_REC_BATCH_NUM = env_int("MAP_BOUNDARY_RAPIDOCR_REC_BATCH_NUM", 12, minimum=1)
RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS = env_int(
    "MAP_BOUNDARY_RAPIDOCR_CLS_RETRY_MIN_LABELS",
    2,
)
TESSERACT_FALLBACK_MIN_USEFUL_LABELS = env_int(
    "MAP_BOUNDARY_TESSERACT_FALLBACK_MIN_USEFUL_LABELS",
    3,
)
RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION",
    1000,
)
ONNXRUNTIME_ENABLE_CPU_MEM_ARENA = env_bool(
    "MAP_BOUNDARY_ONNXRUNTIME_ENABLE_CPU_MEM_ARENA",
    True,
)
ONNXRUNTIME_ALLOW_SPINNING = env_bool(
    "MAP_BOUNDARY_ONNXRUNTIME_ALLOW_SPINNING",
    True,
)
OPENVINO_RECOGNIZER_ENABLED = env_bool("MAP_BOUNDARY_OPENVINO_RECOGNIZER", True)
OPENVINO_RECOGNIZER_MIN_CROPS = env_int("MAP_BOUNDARY_OPENVINO_RECOGNIZER_MIN_CROPS", 14)
FAST_TEXT_OCR_STYLES = frozenset({"bright-blue", "gray-fill"})
FAST_TEXT_OCR_MIN_AREA = env_float("MAP_BOUNDARY_FAST_TEXT_OCR_MIN_AREA", 800.0)
FAST_TEXT_OCR_FALLBACK_CONFIDENCE = env_float(
    "MAP_BOUNDARY_FAST_TEXT_OCR_FALLBACK_CONFIDENCE",
    0.70,
)


def rapidocr_warm_detector_limit() -> int:
    if RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN > 0:
        return RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN
    return RAPIDOCR_DET_LIMIT_SIDE_LEN


def rapidocr_warm_detector_limits() -> list[int]:
    limits: list[int] = []
    for limit in (rapidocr_warm_detector_limit(), RAPIDOCR_DET_LIMIT_SIDE_LEN):
        if limit > 0 and limit not in limits:
            limits.append(limit)
    return limits


def ocr_runtime_config() -> dict[str, Any]:
    return {
        "rapidocr_max_dimension": RAPIDOCR_MAX_DIMENSION,
        "rapidocr_purple_fill_max_dimension": RAPIDOCR_PURPLE_FILL_MAX_DIMENSION,
        "provider_ui_rapidocr_max_dimension": PROVIDER_UI_RAPIDOCR_MAX_DIMENSION,
        "rapidocr_detector_limit_side_len": RAPIDOCR_DET_LIMIT_SIDE_LEN,
        "rapidocr_large_image_detector_limit_side_len": RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN,
        "rapidocr_large_image_detector_limit_min_dimension": RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION,
        "rapidocr_cls_batch_num": RAPIDOCR_CLS_BATCH_NUM,
        "rapidocr_rec_batch_num": RAPIDOCR_REC_BATCH_NUM,
        "rapidocr_classifier_retry_min_labels": RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS,
        "tesseract_fallback_min_useful_labels": TESSERACT_FALLBACK_MIN_USEFUL_LABELS,
        "rapidocr_warm_detector_limit": rapidocr_warm_detector_limit(),
        "rapidocr_warm_detector_limits": rapidocr_warm_detector_limits(),
        "rapidocr_native_array_min_dimension": RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION,
        "onnxruntime_enable_cpu_mem_arena": ONNXRUNTIME_ENABLE_CPU_MEM_ARENA,
        "onnxruntime_allow_spinning": ONNXRUNTIME_ALLOW_SPINNING,
        "openvino_recognizer_enabled": OPENVINO_RECOGNIZER_ENABLED,
        "openvino_recognizer_min_crops": OPENVINO_RECOGNIZER_MIN_CROPS,
        "fast_text_ocr_styles": sorted(FAST_TEXT_OCR_STYLES),
        "fast_text_ocr_min_area": FAST_TEXT_OCR_MIN_AREA,
        "fast_text_ocr_fallback_confidence": FAST_TEXT_OCR_FALLBACK_CONFIDENCE,
    }
