"""Optional ONNX mask-model extraction helpers.

The production extractor remains deterministic by default. These helpers define
the narrow contract a trained segmentation model must satisfy before it can
participate in the existing mask-to-geometry pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("model input dimensions must be positive")
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.output_activation not in {"probability", "logits"}:
            raise ValueError("output_activation must be 'probability' or 'logits'")


def extract_service_area_with_model(
    image_path: str | Path,
    model_path: str | Path,
    *,
    config: ModelExtractionConfig | None = None,
) -> ExtractionResult:
    session = load_onnx_session(model_path)
    rgb = load_rgb(image_path)
    return extract_service_area_from_rgb_with_session(rgb, session, config=config)


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
) -> ExtractionResult:
    cfg = config or ModelExtractionConfig()
    probabilities = predict_mask_probabilities(rgb, session, config=cfg)
    mask = probabilities >= cfg.threshold
    pixel_geometry, contour_count = mask_to_geometry(mask, cfg.simplify_px)
    confidence = extraction_confidence(mask, cfg.style, contour_count)
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
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
        },
    )


def predict_mask_probabilities(
    rgb: np.ndarray,
    session: InferenceSessionLike,
    *,
    config: ModelExtractionConfig | None = None,
) -> np.ndarray:
    cfg = config or ModelExtractionConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (height, width, 3), got {rgb.shape}")
    source_height, source_width = rgb.shape[:2]
    input_tensor = preprocess_rgb_for_model(rgb, config=cfg)
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
) -> np.ndarray:
    cfg = config or ModelExtractionConfig()
    resized = cv2.resize(rgb, (cfg.input_width, cfg.input_height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, :, :, :]


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
