from __future__ import annotations

import os
from typing import Any


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


RAPIDOCR_MAX_DIMENSION = env_int("MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION", 1600)
RAPIDOCR_DET_LIMIT_SIDE_LEN = env_int("MAP_BOUNDARY_RAPIDOCR_DET_LIMIT_SIDE_LEN", 608)
RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN = env_int(
    "MAP_BOUNDARY_RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN",
    640,
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
    0,
)


def rapidocr_warm_detector_limit() -> int:
    if RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN > 0:
        return RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN
    return RAPIDOCR_DET_LIMIT_SIDE_LEN


def ocr_runtime_config() -> dict[str, Any]:
    return {
        "rapidocr_max_dimension": RAPIDOCR_MAX_DIMENSION,
        "rapidocr_detector_limit_side_len": RAPIDOCR_DET_LIMIT_SIDE_LEN,
        "rapidocr_large_image_detector_limit_side_len": RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN,
        "rapidocr_large_image_detector_limit_min_dimension": RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION,
        "rapidocr_cls_batch_num": RAPIDOCR_CLS_BATCH_NUM,
        "rapidocr_rec_batch_num": RAPIDOCR_REC_BATCH_NUM,
        "rapidocr_classifier_retry_min_labels": RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS,
        "tesseract_fallback_min_useful_labels": TESSERACT_FALLBACK_MIN_USEFUL_LABELS,
        "rapidocr_warm_detector_limit": rapidocr_warm_detector_limit(),
        "rapidocr_native_array_min_dimension": RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION,
    }
