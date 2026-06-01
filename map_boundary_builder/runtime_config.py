from __future__ import annotations

import importlib.resources as importlib_resources
import os
from pathlib import Path
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


def env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value in choices:
        return value
    return default


RAPIDOCR_MAX_DIMENSION = env_int("MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION", 1600)
RAPIDOCR_PURPLE_FILL_MAX_DIMENSION = env_int("MAP_BOUNDARY_RAPIDOCR_PURPLE_FILL_MAX_DIMENSION", 800)
PROVIDER_UI_RAPIDOCR_MAX_DIMENSION = env_int("MAP_BOUNDARY_PROVIDER_UI_RAPIDOCR_MAX_DIMENSION", 1200)
CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION",
    875,
)
RAPIDOCR_DET_LIMIT_SIDE_LEN = env_int("MAP_BOUNDARY_RAPIDOCR_DET_LIMIT_SIDE_LEN", 608)
RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN = env_int(
    "MAP_BOUNDARY_RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN",
    608,
)
RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN = env_int(
    "MAP_BOUNDARY_RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN",
    608,
)
RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE = env_choice(
    "MAP_BOUNDARY_RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE",
    "max",
    {"max", "min"},
)
RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE = (
    os.environ.get("MAP_BOUNDARY_RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE", "en-ppocrv5").strip().lower()
    or "default"
)
RAPIDOCR_EN_PPOCRV5_REC_MODEL_PATH = os.environ.get(
    "MAP_BOUNDARY_RAPIDOCR_EN_PPOCRV5_REC_MODEL_PATH",
    "",
).strip()
RAPIDOCR_EN_PPOCRV5_REC_KEYS_PATH = os.environ.get(
    "MAP_BOUNDARY_RAPIDOCR_EN_PPOCRV5_REC_KEYS_PATH",
    "",
).strip()
RAPIDOCR_EN_PPOCRV5_REC_MODEL_NAME = "en_PP-OCRv5_rec_mobile.onnx"
RAPIDOCR_EN_PPOCRV5_REC_KEYS_NAME = "ppocrv5_en_dict.txt"
RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION",
    1000,
)
RAPIDOCR_CLS_BATCH_NUM = env_int("MAP_BOUNDARY_RAPIDOCR_CLS_BATCH_NUM", 24, minimum=1)
RAPIDOCR_REC_BATCH_NUM = env_int("MAP_BOUNDARY_RAPIDOCR_REC_BATCH_NUM", 24, minimum=1)
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
FAST_TEXT_OCR_STYLES = frozenset({"bright-blue", "gray-fill", "light-fill"})
FAST_TEXT_OCR_MIN_AREA = env_float("MAP_BOUNDARY_FAST_TEXT_OCR_MIN_AREA", 1500.0)
FAST_TEXT_OCR_RESCUE_MIN_AREA = env_float(
    "MAP_BOUNDARY_FAST_TEXT_OCR_RESCUE_MIN_AREA",
    900.0,
)
FAST_TEXT_OCR_RESCUE_MIN_ASPECT = env_float(
    "MAP_BOUNDARY_FAST_TEXT_OCR_RESCUE_MIN_ASPECT",
    2.8,
)
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
    for limit in (
        rapidocr_warm_detector_limit(),
        RAPIDOCR_DET_LIMIT_SIDE_LEN,
        RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
    ):
        if limit > 0 and limit not in limits:
            limits.append(limit)
    return limits


def _rapidocr_english_ppocrv5_asset_paths(package: str, model_dir: str) -> tuple[Path, Path] | None:
    try:
        models_dir = importlib_resources.files(package).joinpath(model_dir)
    except Exception:
        return None
    return models_dir / RAPIDOCR_EN_PPOCRV5_REC_MODEL_NAME, models_dir / RAPIDOCR_EN_PPOCRV5_REC_KEYS_NAME


def _existing_rapidocr_english_ppocrv5_asset_paths(package: str, model_dir: str) -> tuple[Path, Path] | None:
    asset_paths = _rapidocr_english_ppocrv5_asset_paths(package, model_dir)
    if asset_paths is not None and all(path.is_file() for path in asset_paths):
        return asset_paths
    return None


def rapidocr_english_ppocrv5_asset_paths() -> tuple[Path, Path] | None:
    if RAPIDOCR_EN_PPOCRV5_REC_MODEL_PATH and RAPIDOCR_EN_PPOCRV5_REC_KEYS_PATH:
        return Path(RAPIDOCR_EN_PPOCRV5_REC_MODEL_PATH), Path(RAPIDOCR_EN_PPOCRV5_REC_KEYS_PATH)
    bundled_paths = _existing_rapidocr_english_ppocrv5_asset_paths("map_boundary_builder", "ocr_models")
    if bundled_paths is not None:
        return bundled_paths
    rapidocr_paths = _rapidocr_english_ppocrv5_asset_paths("rapidocr", "models")
    if rapidocr_paths is None:
        return None
    if all(path.is_file() for path in rapidocr_paths):
        return rapidocr_paths
    return rapidocr_paths


def rapidocr_english_ppocrv5_assets_available() -> bool:
    asset_paths = rapidocr_english_ppocrv5_asset_paths()
    return asset_paths is not None and all(path.is_file() for path in asset_paths)


def rapidocr_bright_blue_recognition_assets_available() -> bool:
    profile = RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE.strip().lower()
    if profile in {"", "default", "ppocrv4", "ch-ppocrv4"}:
        return True
    if profile in {"en-ppocrv5", "ppocrv5-en", "v5-en"}:
        return rapidocr_english_ppocrv5_assets_available()
    return False


def rapidocr_bright_blue_effective_recognition_profile() -> str:
    profile = RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE.strip().lower()
    if profile in {"en-ppocrv5", "ppocrv5-en", "v5-en"} and rapidocr_english_ppocrv5_assets_available():
        return "en-ppocrv5"
    return "default"


def ocr_runtime_config() -> dict[str, Any]:
    return {
        "rapidocr_max_dimension": RAPIDOCR_MAX_DIMENSION,
        "rapidocr_purple_fill_max_dimension": RAPIDOCR_PURPLE_FILL_MAX_DIMENSION,
        "provider_ui_rapidocr_max_dimension": PROVIDER_UI_RAPIDOCR_MAX_DIMENSION,
        "current_catalog_label_ocr_max_dimension": CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION,
        "rapidocr_detector_limit_side_len": RAPIDOCR_DET_LIMIT_SIDE_LEN,
        "rapidocr_large_image_detector_limit_side_len": RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN,
        "rapidocr_bright_blue_detector_limit_side_len": RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
        "rapidocr_bright_blue_detector_limit_type": RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
        "rapidocr_bright_blue_recognition_profile": RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
        "rapidocr_bright_blue_recognition_assets_available": (
            rapidocr_bright_blue_recognition_assets_available()
        ),
        "rapidocr_bright_blue_effective_recognition_profile": (
            rapidocr_bright_blue_effective_recognition_profile()
        ),
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
        "fast_text_ocr_styles": sorted(FAST_TEXT_OCR_STYLES),
        "fast_text_ocr_min_area": FAST_TEXT_OCR_MIN_AREA,
        "fast_text_ocr_rescue_min_area": FAST_TEXT_OCR_RESCUE_MIN_AREA,
        "fast_text_ocr_rescue_min_aspect": FAST_TEXT_OCR_RESCUE_MIN_ASPECT,
        "fast_text_ocr_fallback_confidence": FAST_TEXT_OCR_FALLBACK_CONFIDENCE,
    }
