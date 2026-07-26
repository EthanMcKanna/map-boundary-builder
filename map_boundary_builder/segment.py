"""Model-first service-area segmentation.

One packaged ONNX model produces the mask for every input style. The LAB
auto-fill clusterer remains as the single deterministic fallback, used only
when the model output is degenerate. There are no style branches and no
referee re-runs.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from .extract import (
    DEFAULT_SIMPLIFY_PX,
    ExtractionHints,
    ExtractionResult,
    auto_fill_extraction_result,
    extraction_confidence,
    fill_binary_holes,
    keep_main_components,
    mask_to_geometry,
    resolve_extraction_hints,
)
from .model_extract import (
    ModelExtractionConfig,
    load_onnx_session,
    predict_mask_probabilities,
)

MODEL_STYLE = "model-mask"
DEFAULT_MODEL_FILENAME = "boundary_v1.onnx"
DEFAULT_MODEL_INPUT_SIZE = 384
DEFAULT_MODEL_THRESHOLD = 0.5

# The model result is discarded in favor of the auto-fill fallback when it
# looks degenerate: almost nothing selected, almost everything selected, a
# large uncertain band, or a mask that hugs the image border.
DEGENERATE_MIN_COVERAGE = 0.005
DEGENERATE_MAX_COVERAGE = 0.90
DEGENERATE_MAX_UNCERTAINTY = 0.35
DEGENERATE_MAX_BORDER_FRACTION = 0.60


def default_model_path() -> Path:
    return Path(str(resources.files("map_boundary_builder").joinpath("models", DEFAULT_MODEL_FILENAME)))


def load_model_config(
    model_path: str | Path,
    *,
    simplify_px: float = DEFAULT_SIMPLIFY_PX,
    threshold: float | None = None,
) -> ModelExtractionConfig:
    """Build the inference config from the model's JSON sidecar, if present."""
    input_size = DEFAULT_MODEL_INPUT_SIZE
    sidecar_threshold = DEFAULT_MODEL_THRESHOLD
    activation = "logits"
    sidecar = Path(str(model_path) + ".json")
    if sidecar.is_file():
        try:
            production = json.loads(sidecar.read_text(encoding="utf-8")).get("production", {})
        except (OSError, json.JSONDecodeError):
            production = {}
        input_size = int(production.get("input_width", input_size))
        sidecar_threshold = float(production.get("threshold", sidecar_threshold))
        activation = str(production.get("output_activation", activation))
    return ModelExtractionConfig(
        input_width=input_size,
        input_height=input_size,
        threshold=float(threshold) if threshold is not None else sidecar_threshold,
        simplify_px=simplify_px,
        style=MODEL_STYLE,
        output_activation=activation,  # type: ignore[arg-type]
        input_channels=3,
    )


def segment_image(
    rgb: np.ndarray,
    *,
    model_path: str | Path | None = None,
    simplify_px: float = DEFAULT_SIMPLIFY_PX,
    threshold: float | None = None,
    hints: ExtractionHints | dict[str, object] | None = None,
) -> ExtractionResult:
    """Extract the service-area mask, model first, auto-fill fallback.

    Raises ValueError when neither producer yields a usable boundary.
    """
    extraction_hints = resolve_extraction_hints(hints)
    model_result = _model_segmentation(
        rgb,
        model_path=model_path,
        simplify_px=simplify_px,
        threshold=threshold,
        hints=extraction_hints,
    )
    if model_result is not None:
        return model_result
    fallback = auto_fill_extraction_result(rgb, simplify_px=simplify_px, hints=extraction_hints)
    if fallback is not None:
        diagnostics = dict(fallback.diagnostics or {})
        diagnostics["segmentation_engine"] = "auto-fill-fallback"
        return ExtractionResult(
            mask=fallback.mask,
            style=fallback.style,
            pixel_geometry=fallback.pixel_geometry,
            coverage_ratio=fallback.coverage_ratio,
            contour_count=fallback.contour_count,
            confidence=fallback.confidence,
            extraction_profile=fallback.extraction_profile,
            diagnostics=diagnostics,
        )
    raise ValueError("No service-area polygon could be extracted")


def _model_segmentation(
    rgb: np.ndarray,
    *,
    model_path: str | Path | None,
    simplify_px: float,
    threshold: float | None,
    hints: ExtractionHints,
) -> ExtractionResult | None:
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.is_file():
        return None
    config = load_model_config(path, simplify_px=simplify_px, threshold=threshold)
    try:
        session = load_onnx_session(str(path))
        probabilities = predict_mask_probabilities(rgb, session, config=config)
    except Exception:
        return None
    raw_mask = probabilities >= config.threshold
    uncertainty_fraction = float(((probabilities >= 0.40) & (probabilities <= 0.60)).mean())
    diagnostics = {
        "segmentation_engine": "model",
        "model_path": path.name,
        "model_threshold": config.threshold,
        "model_input_size": config.input_width,
        "uncertainty_fraction": uncertainty_fraction,
        "probability_mean": float(probabilities.mean()),
    }
    # Degeneracy is judged on the raw model output: zone selection could trim
    # an everything-is-boundary prediction into something plausible-looking,
    # but such an output is untrustworthy and belongs to the fallback.
    degenerate = _degenerate_reason(raw_mask, uncertainty_fraction)
    if degenerate is not None:
        diagnostics["degenerate_reason"] = degenerate
        return None
    mask = select_primary_zone(raw_mask, rgb, probabilities, hints)
    mask = select_hinted_components(mask, rgb, hints)
    mask = keep_main_components(mask, max_components=3)
    mask = fill_binary_holes(mask)
    try:
        pixel_geometry, contour_count = mask_to_geometry(mask, simplify_px)
    except ValueError:
        return None
    confidence = min(
        extraction_confidence(mask, MODEL_STYLE, contour_count),
        max(0.0, 1.0 - uncertainty_fraction),
    )
    return ExtractionResult(
        mask=mask,
        style=MODEL_STYLE,
        pixel_geometry=pixel_geometry,
        coverage_ratio=float(mask.mean()),
        contour_count=contour_count,
        confidence=round(confidence, 3),
        diagnostics=diagnostics,
    )


def _degenerate_reason(mask: np.ndarray, uncertainty_fraction: float) -> str | None:
    coverage = float(mask.mean())
    if coverage < DEGENERATE_MIN_COVERAGE:
        return "coverage_too_low"
    if coverage > DEGENERATE_MAX_COVERAGE:
        return "coverage_too_high"
    if uncertainty_fraction > DEGENERATE_MAX_UNCERTAINTY:
        return "uncertain_probabilities"
    if _border_fraction(mask) > DEGENERATE_MAX_BORDER_FRACTION:
        return "mask_hugs_border"
    return None


def _border_fraction(mask: np.ndarray) -> float:
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.size == 0:
        return 0.0
    return float(border.mean())


def select_hinted_components(
    mask: np.ndarray,
    rgb: np.ndarray,
    hints: ExtractionHints,
) -> np.ndarray:
    """Reduce a multi-component mask using optional seed/color guidance.

    Replaces the old five-channel model input: guidance now selects among
    predicted components instead of steering the network.
    """
    if hints.seed_point is None and hints.target_rgb is None:
        return mask
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if count <= 2:
        return mask
    if hints.seed_point is not None:
        x = int(np.clip(round(hints.seed_point[0]), 0, mask.shape[1] - 1))
        y = int(np.clip(round(hints.seed_point[1]), 0, mask.shape[0] - 1))
        label = int(labels[y, x])
        if label == 0:
            label = _nearest_component_label(labels, count, x, y)
        return labels == label
    target = np.asarray(hints.target_rgb, dtype=np.float64)
    best_label = 0
    best_distance = np.inf
    for label in range(1, count):
        component = labels == label
        mean_color = rgb[component].reshape(-1, 3).mean(axis=0)
        distance = float(np.linalg.norm(mean_color - target))
        if distance < best_distance:
            best_distance = distance
            best_label = label
    if best_label == 0:
        return mask
    return labels == best_label


ZONE_MERGE_MAX_AB_DISTANCE = 30.0
ZONE_MERGE_MAX_GRAY_CHROMA = 12.0
ZONE_KMEANS_MAX_SAMPLES = 50_000
ZONE_MIN_COVERAGE_OF_MASK = 0.02
ZONE_MIN_OPENING_RETENTION = 0.5


def select_primary_zone(
    mask: np.ndarray,
    rgb: np.ndarray,
    probabilities: np.ndarray,
    hints: ExtractionHints,
) -> np.ndarray:
    """Reduce a multi-zone mask to its primary color-consistent zone.

    A screenshot can contain several differently-colored highlighted regions
    with different meanings (the target service area plus unrelated shaded
    districts), and they may touch, so connected components cannot separate
    them. The mask is clustered by LAB color into zones; the winning zone is
    the one containing the seed / matching the target color when hints are
    given, otherwise the most salient by model confidence, size, centrality,
    and border avoidance — a highlighted target is usually centered while
    background shading runs off the crop edges.
    """
    if not mask.any():
        return mask
    zones = _color_zones(mask, rgb)
    if len(zones) <= 1:
        return mask
    if hints.seed_point is not None:
        x = int(np.clip(round(hints.seed_point[0]), 0, mask.shape[1] - 1))
        y = int(np.clip(round(hints.seed_point[1]), 0, mask.shape[0] - 1))
        containing = [zone for zone in zones if zone[y, x]]
        if containing:
            return containing[0]
    if hints.target_rgb is not None:
        target = np.asarray(hints.target_rgb, dtype=np.float64)
        return min(zones, key=lambda zone: float(np.linalg.norm(rgb[zone].reshape(-1, 3).mean(axis=0) - target)))
    return max(zones, key=lambda zone: _zone_salience(zone, probabilities))


def _color_zones(mask: np.ndarray, rgb: np.ndarray) -> list[np.ndarray]:
    """Split the mask into distinct-color zones, conservatively.

    Clustering runs on the chromatic (a, b) LAB plane only: a translucent
    overlay's lightness varies with the basemap underneath, but its hue
    direction is stable, while genuinely different zones (green vs orange)
    sit far apart chromatically. The split is accepted only when every
    candidate zone is spatially coherent — texture speckle inside one
    translucent fill interleaves and dissolves under morphological opening,
    whereas real zones survive it.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ab = lab[:, :, 1:] - 128.0
    ys, xs = np.nonzero(mask)
    samples = ab[ys, xs]
    if len(samples) > ZONE_KMEANS_MAX_SAMPLES:
        step = len(samples) // ZONE_KMEANS_MAX_SAMPLES + 1
        fit_samples = np.ascontiguousarray(samples[::step])
    else:
        fit_samples = samples
    if len(fit_samples) < 8:
        return [mask]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _compactness, _labels, centers = cv2.kmeans(
        fit_samples, min(3, len(fit_samples)), None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    # Merge centers that are chromatically the same overlay color; low-chroma
    # (grayish) centers have no meaningful hue and always merge together.
    merged: list[np.ndarray] = []
    for center in centers.astype(np.float64):
        for index, existing in enumerate(merged):
            close = float(np.linalg.norm(center - existing)) <= ZONE_MERGE_MAX_AB_DISTANCE
            both_gray = (
                float(np.linalg.norm(center)) <= ZONE_MERGE_MAX_GRAY_CHROMA
                and float(np.linalg.norm(existing)) <= ZONE_MERGE_MAX_GRAY_CHROMA
            )
            if close or both_gray:
                merged[index] = (existing + center) / 2.0
                break
        else:
            merged.append(center)
    if len(merged) <= 1:
        return [mask]

    centers_array = np.stack(merged)
    assignments = np.argmin(
        np.linalg.norm(samples[:, None, :].astype(np.float64) - centers_array[None, :, :], axis=2),
        axis=1,
    )
    zones = []
    minimum = max(64, int(ZONE_MIN_COVERAGE_OF_MASK * len(samples)))
    kernel = np.ones((5, 5), np.uint8)
    for index in range(len(merged)):
        selected = assignments == index
        if int(selected.sum()) < minimum:
            continue
        zone = np.zeros_like(mask)
        zone[ys[selected], xs[selected]] = True
        opened = cv2.morphologyEx(zone.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        retention = float(opened.sum()) / max(1.0, float(zone.sum()))
        if retention < ZONE_MIN_OPENING_RETENTION:
            # Interleaved speckle, not a coherent zone: refuse to split.
            return [mask]
        zone = cv2.morphologyEx(zone.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        zones.append(zone & mask)
    if len(zones) <= 1:
        return [mask]
    return zones


def _zone_salience(zone: np.ndarray, probabilities: np.ndarray) -> float:
    area = int(zone.sum())
    if area == 0:
        return 0.0
    height, width = zone.shape
    ys, xs = np.nonzero(zone)
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    center_norm = float(np.hypot(center_x, center_y))
    border = float(
        zone[0, :].sum() + zone[-1, :].sum() + zone[:, 0].sum() + zone[:, -1].sum()
    ) / max(1.0, 2.0 * (height + width))
    centroid_distance = float(np.hypot(xs.mean() - center_x, ys.mean() - center_y)) / max(1.0, center_norm)
    mean_probability = float(probabilities[zone].mean())
    return (
        mean_probability
        * float(np.sqrt(area))
        * (1.0 - min(0.8, 3.0 * border))
        * (1.0 - 0.5 * centroid_distance)
    )


def _nearest_component_label(labels: np.ndarray, count: int, x: int, y: int) -> int:
    best_label = 1
    best_distance = np.inf
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if ys.size == 0:
            continue
        distance = float(np.min((xs - x) ** 2 + (ys - y) ** 2))
        if distance < best_distance:
            best_distance = distance
            best_label = label
    return best_label
