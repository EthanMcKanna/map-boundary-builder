from __future__ import annotations

import importlib.resources as importlib_resources
import os
from pathlib import Path
from typing import Any

from .pipeline_version import PIPELINE_VERSION_ENV


GENERATION_ENV_DEFAULTS = {
    "MAP_BOUNDARY_BLOCK_NETWORK": "",
    "MAP_BOUNDARY_CACHE_DIR": ".cache/map-boundary-builder",
    "MAP_BOUNDARY_CATALOG_EXTRACT_MAX_DIMENSION": "240",
    "MAP_BOUNDARY_CATALOG_MISS_REFINE_MAX_DIMENSION": "",
    "MAP_BOUNDARY_CATALOG_RETRY_EXTRACT_MAX_DIMENSION": "400",
    "MAP_BOUNDARY_EARLY_OCR_STYLE_MAX_DIMENSION": "800",
    "MAP_BOUNDARY_ENABLE_ROAD_CONTEXT_FALLBACK": "",
    "MAP_BOUNDARY_EXTRACT_MAX_DIMENSION": "0",
    "MAP_BOUNDARY_EXTRACTION_CACHE": "1",
    "MAP_BOUNDARY_EXTRACTION_DISK_CACHE": "",
    "MAP_BOUNDARY_EXTRACTION_TRIMMED_CACHE_MAX_PIXELS": "3000000",
    "MAP_BOUNDARY_EXTRACTION_UNTRIMMED_CACHE_MAX_PIXELS": "1000000",
    "MAP_BOUNDARY_FOCUS_GEOREF_OCR_DET_LIMIT_SIDE_LEN": "416",
    "MAP_BOUNDARY_FOCUS_GEOREF_OCR_MAX_CROP_AREA_RATIO": "0.35",
    "MAP_BOUNDARY_FOCUS_GEOREF_OCR_MAX_DIMENSION": "550",
    "MAP_BOUNDARY_FOCUS_GEOREF_OCR_MIN_TEXT_AREA": "500",
    "MAP_BOUNDARY_GENERAL_EXTRACT_MAX_DIMENSION": "1600",
    "MAP_BOUNDARY_GEOCODE_BATCH_SIZE": "12",
    "MAP_BOUNDARY_GEOCODE_LABEL_LOOKAHEAD": "3",
    "MAP_BOUNDARY_GEOCODE_WORKERS": "6",
    "MAP_BOUNDARY_RAPIDOCR_GRAY_FILL_MAX_DIMENSION": "800",
    "MAP_BOUNDARY_GRAY_FILL_ROUTE_UI_OCR_MAX_DIMENSION": "1000",
    "MAP_BOUNDARY_LIGHT_FILL_ROUTE_UI_OCR_MAX_DIMENSION": "1000",
    "MAP_BOUNDARY_NOMINATIM_TIMEOUT_SECONDS": "4.0",
    "MAP_BOUNDARY_OCR_DISK_CACHE": "",
    "MAP_BOUNDARY_PLACE_BEFORE_LIVE_TIMEOUT_SECONDS": "1.0",
    "MAP_BOUNDARY_PLACE_FAST_PATH_TIMEOUT_SECONDS": "0.08",
    "MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES": "1",
    "MAP_BOUNDARY_PROVIDER_UI_CROP_OCR_MAX_DIMENSION": "750",
    "MAP_BOUNDARY_PROVIDER_UI_FOCUS_CROP": "1",
    "MAP_BOUNDARY_PROVIDER_UI_GRAY_FILL_CROP_OCR_MAX_DIMENSION": "450",
    "MAP_BOUNDARY_ROAD_MATCH_MAX_POINTS": "4000",
    "MAP_BOUNDARY_ROAD_REFINE_CACHE_MAX_PIXELS": "1000000",
    "MAP_BOUNDARY_ROAD_REFINE_COARSE_FEATURE_SCALE": "4",
    "MAP_BOUNDARY_ROAD_REFINE_FINE_FEATURE_SCALE": "2",
    "MAP_BOUNDARY_ROAD_REFINE_FULL_FALLBACK_MIN_SCORE": "0.60",
    "MAP_BOUNDARY_RUNNER_OCR_CACHE": "1",
    "MAP_BOUNDARY_SCALED_EXTRACTION_CACHE_MAX_PIXELS": "3000000",
    "MAP_BOUNDARY_SCALED_EXTRACTION_MEMORY_CACHE_MAX": "24",
    PIPELINE_VERSION_ENV: "",
}


def generation_env_config() -> dict[str, str]:
    return {
        name: os.environ.get(name, default)
        for name, default in sorted(GENERATION_ENV_DEFAULTS.items())
    }


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
RAPIDOCR_GRAY_FILL_MAX_DIMENSION = env_int("MAP_BOUNDARY_RAPIDOCR_GRAY_FILL_MAX_DIMENSION", 800)
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
    256,
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
RAPIDOCR_BRIGHT_BLUE_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_BRIGHT_BLUE_MAX_DIMENSION",
    1400,
)
RAPIDOCR_BRIGHT_BLUE_FULL_DETAIL_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_BRIGHT_BLUE_FULL_DETAIL_MAX_DIMENSION",
    1500,
)
RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION",
    RAPIDOCR_BRIGHT_BLUE_MAX_DIMENSION,
)
RAPIDOCR_SVG_BRIGHT_BLUE_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_SVG_BRIGHT_BLUE_MAX_DIMENSION",
    1600,
)
RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN = env_int(
    "MAP_BOUNDARY_RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN",
    208,
)
RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION",
    RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION,
)
SVG_RASTER_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_SVG_RASTER_MAX_DIMENSION",
    RAPIDOCR_SVG_BRIGHT_BLUE_MAX_DIMENSION,
)
RAPIDOCR_DARK_TEAL_WIDE_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_DARK_TEAL_WIDE_MAX_DIMENSION",
    1400,
)
RAPIDOCR_DARK_TEAL_WIDE_MAX_HEIGHT_WIDTH_RATIO = env_float(
    "MAP_BOUNDARY_RAPIDOCR_DARK_TEAL_WIDE_MAX_HEIGHT_WIDTH_RATIO",
    1.25,
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
RAPIDOCR_WARM_SAMPLE_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_RAPIDOCR_WARM_SAMPLE_MAX_DIMENSION",
    608,
)
RAPIDOCR_CLS_BATCH_NUM = env_int("MAP_BOUNDARY_RAPIDOCR_CLS_BATCH_NUM", 24, minimum=1)
RAPIDOCR_REC_BATCH_NUM = env_int("MAP_BOUNDARY_RAPIDOCR_REC_BATCH_NUM", 12, minimum=1)
RAPIDOCR_DARK_TEAL_REC_BATCH_NUM = env_int(
    "MAP_BOUNDARY_RAPIDOCR_DARK_TEAL_REC_BATCH_NUM",
    16,
    minimum=0,
)
LIGHT_FILL_ROUTE_UI_OCR_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_LIGHT_FILL_ROUTE_UI_OCR_MAX_DIMENSION",
    1000,
)
GRAY_FILL_ROUTE_UI_OCR_MAX_DIMENSION = env_int(
    "MAP_BOUNDARY_GRAY_FILL_ROUTE_UI_OCR_MAX_DIMENSION",
    1000,
)
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
    False,
)
ONNXRUNTIME_ALLOW_SPINNING = env_bool(
    "MAP_BOUNDARY_ONNXRUNTIME_ALLOW_SPINNING",
    True,
)
FAST_TEXT_OCR_STYLES = frozenset({"bright-blue", "gray-fill", "light-fill"})
FAST_TEXT_OCR_MIN_AREA = env_float("MAP_BOUNDARY_FAST_TEXT_OCR_MIN_AREA", 1500.0)
BRIGHT_BLUE_FAST_TEXT_OCR_MIN_AREA = env_float(
    "MAP_BOUNDARY_BRIGHT_BLUE_FAST_TEXT_OCR_MIN_AREA",
    2300.0,
)
SVG_BRIGHT_BLUE_FAST_TEXT_OCR_MIN_AREA = env_float(
    "MAP_BOUNDARY_SVG_BRIGHT_BLUE_FAST_TEXT_OCR_MIN_AREA",
    300.0,
)
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
    ):
        if limit > 0 and limit not in limits:
            limits.append(limit)
    return limits


def rapidocr_warm_engine_keys_config() -> list[list[int | str]]:
    keys: list[list[int | str]] = []
    generic_detector_limits = rapidocr_warm_detector_limits()
    for detector_limit in generic_detector_limits:
        key: list[int | str] = [detector_limit, "default", "default", RAPIDOCR_REC_BATCH_NUM]
        if detector_limit > 0 and key not in keys:
            keys.append(key)

    bright_blue_profile = normalized_rapidocr_recognition_profile(
        RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE
    )
    bright_blue_detector_type = normalized_rapidocr_detector_limit_type(
        RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE
    )
    if RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN > 0 and (
        bright_blue_profile != "default"
        or bright_blue_detector_type != "default"
        or RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN not in generic_detector_limits
    ):
        key = [
            RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            bright_blue_profile,
            bright_blue_detector_type,
            RAPIDOCR_REC_BATCH_NUM,
        ]
        if key not in keys:
            keys.append(key)
    if RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN > 0:
        key = [
            RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            bright_blue_profile,
            bright_blue_detector_type,
            RAPIDOCR_REC_BATCH_NUM,
        ]
        if key not in keys:
            keys.append(key)
    if RAPIDOCR_DARK_TEAL_REC_BATCH_NUM > 0 and RAPIDOCR_DARK_TEAL_REC_BATCH_NUM != RAPIDOCR_REC_BATCH_NUM:
        for detector_limit in generic_detector_limits:
            key = [
                detector_limit,
                "default",
                "default",
                RAPIDOCR_DARK_TEAL_REC_BATCH_NUM,
            ]
            if detector_limit > 0 and key not in keys:
                keys.append(key)
    return keys


def rapidocr_generic_warm_sample_side_config() -> int:
    warm_side = RAPIDOCR_WARM_SAMPLE_MAX_DIMENSION
    if warm_side <= 0:
        warm_side = RAPIDOCR_MAX_DIMENSION
    if warm_side <= 0:
        warm_side = 1600
    return bounded_rapidocr_warm_sample_side(warm_side)


def rapidocr_bright_blue_large_warm_sample_side_config() -> int:
    if RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION <= 0:
        return 0
    if RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN <= 0:
        return 0
    warm_side = bounded_rapidocr_warm_sample_side(RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION)
    if warm_side <= rapidocr_generic_warm_sample_side_config():
        return 0
    return warm_side


def rapidocr_svg_bright_blue_large_warm_sample_side_config() -> int:
    if RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION <= 0:
        return 0
    if RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN <= 0:
        return 0
    warm_side = bounded_rapidocr_warm_sample_side(RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION)
    if warm_side <= rapidocr_generic_warm_sample_side_config():
        return 0
    return warm_side


def rapidocr_bright_blue_warm_key_config() -> list[int | str]:
    return [
        RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
        normalized_rapidocr_recognition_profile(RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE),
        normalized_rapidocr_detector_limit_type(RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE),
        RAPIDOCR_REC_BATCH_NUM,
    ]


def rapidocr_svg_bright_blue_warm_key_config() -> list[int | str]:
    return [
        RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
        normalized_rapidocr_recognition_profile(RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE),
        normalized_rapidocr_detector_limit_type(RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE),
        RAPIDOCR_REC_BATCH_NUM,
    ]


def rapidocr_warm_engine_sample_plan_config() -> list[list[int | str]]:
    generic_side = rapidocr_generic_warm_sample_side_config()
    bright_blue_warm_side = rapidocr_bright_blue_large_warm_sample_side_config()
    bright_blue_key = rapidocr_bright_blue_warm_key_config()
    svg_bright_blue_warm_side = rapidocr_svg_bright_blue_large_warm_sample_side_config()
    svg_bright_blue_key = rapidocr_svg_bright_blue_warm_key_config()
    large_warm_keys: dict[tuple[int | str, ...], int] = {}
    if bright_blue_warm_side > 0:
        large_warm_keys[tuple(bright_blue_key)] = bright_blue_warm_side
    if svg_bright_blue_warm_side > 0:
        key = tuple(svg_bright_blue_key)
        large_warm_keys[key] = max(svg_bright_blue_warm_side, large_warm_keys.get(key, 0))
    plan: list[list[int | str]] = []
    for key in rapidocr_warm_engine_keys_config():
        if tuple(key) in large_warm_keys:
            continue
        plan.append([*key, generic_side])
    for key, warm_side in large_warm_keys.items():
        plan.append([*key, warm_side])
    return plan


def bounded_rapidocr_warm_sample_side(warm_side: int) -> int:
    return max(384, min(1600, int(warm_side)))


def normalized_rapidocr_detector_limit_type(value: str | None = None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"max", "min"}:
        return normalized
    return "default"


def normalized_rapidocr_recognition_profile(value: str | None = None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"", "default", "ppocrv4", "ch-ppocrv4"}:
        return "default"
    if normalized in {"en-ppocrv5", "ppocrv5-en", "v5-en"}:
        return "en-ppocrv5"
    return "default"


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
        "rapidocr_gray_fill_max_dimension": RAPIDOCR_GRAY_FILL_MAX_DIMENSION,
        "provider_ui_rapidocr_max_dimension": PROVIDER_UI_RAPIDOCR_MAX_DIMENSION,
        "current_catalog_label_ocr_max_dimension": CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION,
        "rapidocr_detector_limit_side_len": RAPIDOCR_DET_LIMIT_SIDE_LEN,
        "rapidocr_large_image_detector_limit_side_len": RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN,
        "rapidocr_bright_blue_detector_limit_side_len": RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
        "rapidocr_bright_blue_detector_limit_type": RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
        "rapidocr_bright_blue_recognition_profile": RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
        "rapidocr_bright_blue_max_dimension": RAPIDOCR_BRIGHT_BLUE_MAX_DIMENSION,
        "rapidocr_bright_blue_full_detail_max_dimension": RAPIDOCR_BRIGHT_BLUE_FULL_DETAIL_MAX_DIMENSION,
        "rapidocr_bright_blue_warm_sample_max_dimension": RAPIDOCR_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION,
        "rapidocr_svg_bright_blue_max_dimension": RAPIDOCR_SVG_BRIGHT_BLUE_MAX_DIMENSION,
        "rapidocr_svg_bright_blue_detector_limit_side_len": RAPIDOCR_SVG_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
        "rapidocr_svg_bright_blue_warm_sample_max_dimension": (
            RAPIDOCR_SVG_BRIGHT_BLUE_WARM_SAMPLE_MAX_DIMENSION
        ),
        "rapidocr_dark_teal_wide_max_dimension": RAPIDOCR_DARK_TEAL_WIDE_MAX_DIMENSION,
        "rapidocr_dark_teal_wide_max_height_width_ratio": RAPIDOCR_DARK_TEAL_WIDE_MAX_HEIGHT_WIDTH_RATIO,
        "rapidocr_bright_blue_recognition_assets_available": (
            rapidocr_bright_blue_recognition_assets_available()
        ),
        "rapidocr_bright_blue_effective_recognition_profile": (
            rapidocr_bright_blue_effective_recognition_profile()
        ),
        "rapidocr_large_image_detector_limit_min_dimension": RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION,
        "rapidocr_cls_batch_num": RAPIDOCR_CLS_BATCH_NUM,
        "rapidocr_rec_batch_num": RAPIDOCR_REC_BATCH_NUM,
        "rapidocr_dark_teal_rec_batch_num": RAPIDOCR_DARK_TEAL_REC_BATCH_NUM,
        "light_fill_route_ui_ocr_max_dimension": LIGHT_FILL_ROUTE_UI_OCR_MAX_DIMENSION,
        "gray_fill_route_ui_ocr_max_dimension": GRAY_FILL_ROUTE_UI_OCR_MAX_DIMENSION,
        "rapidocr_classifier_retry_min_labels": RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS,
        "tesseract_fallback_min_useful_labels": TESSERACT_FALLBACK_MIN_USEFUL_LABELS,
        "rapidocr_warm_detector_limit": rapidocr_warm_detector_limit(),
        "rapidocr_warm_detector_limits": rapidocr_warm_detector_limits(),
        "rapidocr_warm_engine_keys": rapidocr_warm_engine_keys_config(),
        "rapidocr_warm_engine_sample_plan": rapidocr_warm_engine_sample_plan_config(),
        "rapidocr_warm_sample_max_dimension": RAPIDOCR_WARM_SAMPLE_MAX_DIMENSION,
        "rapidocr_native_array_min_dimension": RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION,
        "onnxruntime_enable_cpu_mem_arena": ONNXRUNTIME_ENABLE_CPU_MEM_ARENA,
        "onnxruntime_allow_spinning": ONNXRUNTIME_ALLOW_SPINNING,
        "fast_text_ocr_styles": sorted(FAST_TEXT_OCR_STYLES),
        "fast_text_ocr_min_area": FAST_TEXT_OCR_MIN_AREA,
        "bright_blue_fast_text_ocr_min_area": BRIGHT_BLUE_FAST_TEXT_OCR_MIN_AREA,
        "svg_bright_blue_fast_text_ocr_min_area": SVG_BRIGHT_BLUE_FAST_TEXT_OCR_MIN_AREA,
        "fast_text_ocr_rescue_min_area": FAST_TEXT_OCR_RESCUE_MIN_AREA,
        "fast_text_ocr_rescue_min_aspect": FAST_TEXT_OCR_RESCUE_MIN_ASPECT,
        "fast_text_ocr_fallback_confidence": FAST_TEXT_OCR_FALLBACK_CONFIDENCE,
    }
