"""Optional ONNX mask-model extraction helpers.

The production extractor remains deterministic by default. These helpers define
the narrow contract a trained segmentation model must satisfy before it can
participate in the existing mask-to-geometry pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

from .extract import DEFAULT_SIMPLIFY_PX, ExtractionResult, extraction_confidence, load_rgb, mask_to_geometry


class InferenceSessionLike(Protocol):
    def get_inputs(self) -> list[Any]: ...

    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class ModelExtractionConfig:
    input_width: int = 512
    input_height: int = 512
    threshold: float = 0.5
    simplify_px: float = DEFAULT_SIMPLIFY_PX
    style: str = "model-mask"
    output_activation: Literal["probability", "logits"] = "probability"
    input_channels: int = 3

    def __post_init__(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("model input dimensions must be positive")
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.output_activation not in {"probability", "logits"}:
            raise ValueError("output_activation must be 'probability' or 'logits'")
        if self.input_channels not in {3, 5}:
            raise ValueError("input_channels must be 3 or 5")


def extract_service_area_with_model(
    image_path: str | Path,
    model_path: str | Path,
    *,
    config: ModelExtractionConfig | None = None,
    hints: Any = None,
) -> ExtractionResult:
    session = load_onnx_session(model_path)
    rgb = load_rgb(image_path)
    return extract_service_area_from_rgb_with_session(rgb, session, config=config, hints=hints)


@lru_cache(maxsize=2)
def load_onnx_session(model_path: str | Path) -> InferenceSessionLike:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for model-backed extraction") from exc
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def extract_service_area_from_rgb_with_session(
    rgb: np.ndarray,
    session: InferenceSessionLike,
    *,
    config: ModelExtractionConfig | None = None,
    hints: Any = None,
) -> ExtractionResult:
    cfg = config or ModelExtractionConfig()
    probabilities = predict_mask_probabilities(rgb, session, config=cfg, hints=hints)
    mask = probabilities >= cfg.threshold
    pixel_geometry, contour_count = mask_to_geometry(mask, cfg.simplify_px)
    uncertainty_fraction = float(((probabilities >= 0.40) & (probabilities <= 0.60)).mean())
    confidence = min(
        extraction_confidence(mask, cfg.style, contour_count),
        max(0.0, 1.0 - uncertainty_fraction),
    )
    return ExtractionResult(
        mask=mask,
        style=cfg.style,
        pixel_geometry=pixel_geometry,
        coverage_ratio=float(mask.mean()),
        contour_count=contour_count,
        confidence=confidence,
        diagnostics={
            "model_input_shape": [cfg.input_height, cfg.input_width],
            "model_threshold": cfg.threshold,
            "model_input_channels": cfg.input_channels,
            "model_guidance": guidance_diagnostics(hints),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "probability_mean": float(probabilities.mean()),
            "uncertainty_fraction": uncertainty_fraction,
        },
    )


def predict_mask_probabilities(
    rgb: np.ndarray,
    session: InferenceSessionLike,
    *,
    config: ModelExtractionConfig | None = None,
    hints: Any = None,
) -> np.ndarray:
    cfg = config or ModelExtractionConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (height, width, 3), got {rgb.shape}")
    source_height, source_width = rgb.shape[:2]
    input_tensor = preprocess_rgb_for_model(rgb, config=cfg, hints=hints)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    if not outputs:
        raise ValueError("model did not return any outputs")
    probability = normalize_model_output(outputs[0], output_activation=cfg.output_activation)
    resized = cv2.resize(
        probability.astype(np.float32),
        (source_width, source_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0)


def preprocess_rgb_for_model(
    rgb: np.ndarray,
    *,
    config: ModelExtractionConfig | None = None,
    hints: Any = None,
) -> np.ndarray:
    cfg = config or ModelExtractionConfig()
    resized = cv2.resize(rgb, (cfg.input_width, cfg.input_height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    channels = np.transpose(normalized, (2, 0, 1))
    if cfg.input_channels == 5:
        guidance = guidance_channels(rgb, cfg.input_width, cfg.input_height, hints=hints)
        channels = np.concatenate([channels, guidance], axis=0)
    return channels[np.newaxis, :, :, :]


def guidance_channels(
    rgb: np.ndarray,
    width: int,
    height: int,
    *,
    hints: Any = None,
) -> np.ndarray:
    seed_map = np.zeros((height, width), dtype=np.float32)
    target_map = np.zeros((height, width), dtype=np.float32)
    seed_point = hint_value(hints, "seed_point")
    if seed_point is not None:
        source_h, source_w = rgb.shape[:2]
        source_x = float(np.clip(float(seed_point[0]), 0.0, max(0, source_w - 1)))
        source_y = float(np.clip(float(seed_point[1]), 0.0, max(0, source_h - 1)))
        x = source_x * width / max(1, source_w)
        y = source_y * height / max(1, source_h)
        yy, xx = np.mgrid[0:height, 0:width]
        sigma = max(2.0, min(width, height) * 0.035)
        seed_map = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    target_rgb = hint_value(hints, "target_rgb")
    if target_rgb is not None:
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
        target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 1, 3)
        distance = np.linalg.norm(resized - target, axis=2) / np.sqrt(3.0 * 255.0**2)
        target_map = (1.0 - np.clip(distance, 0.0, 1.0)).astype(np.float32)
    return np.stack([seed_map, target_map], axis=0)


def hint_value(hints: Any, name: str) -> Any:
    if hints is None:
        return None
    if isinstance(hints, dict):
        return hints.get(name)
    return getattr(hints, name, None)


def guidance_diagnostics(hints: Any) -> dict[str, bool]:
    return {
        "seed_point": hint_value(hints, "seed_point") is not None,
        "target_rgb": hint_value(hints, "target_rgb") is not None,
    }


def normalize_model_output(
    output: np.ndarray,
    *,
    output_activation: Literal["probability", "logits"] = "probability",
) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got output shape {arr.shape}")
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"expected single-channel model output, got shape {output.shape}")
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mask probabilities, got shape {output.shape}")
    arr = arr.astype(np.float32, copy=False)
    if output_activation == "logits":
        arr = np.clip(arr, -60.0, 60.0)
        arr = 1.0 / (1.0 + np.exp(-arr))
    elif output_activation == "probability":
        if arr.min() < 0.0 or arr.max() > 1.0:
            raise ValueError("probability model output must be between 0 and 1")
    else:
        raise ValueError("output_activation must be 'probability' or 'logits'")
    return arr
