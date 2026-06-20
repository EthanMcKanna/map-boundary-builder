"""Standalone mask and geometry evaluation helpers."""

from __future__ import annotations

from math import hypot
from typing import Any

import cv2
import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

MaskLike = np.ndarray | list[Any] | tuple[Any, ...]

__all__ = [
    "area_ratio",
    "binary_mask_confusion_counts",
    "boundary_iou",
    "boundary_mask",
    "centroid_distance_px",
    "confusion_counts",
    "dice",
    "geometry_validity_summary",
    "iou",
    "precision",
    "recall",
]


def _as_binary_mask(mask: MaskLike, *, name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _paired_masks(
    predicted_mask: MaskLike,
    reference_mask: MaskLike,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = _as_binary_mask(predicted_mask, name="predicted_mask")
    reference = _as_binary_mask(reference_mask, name="reference_mask")
    if predicted.shape != reference.shape:
        raise ValueError(
            "predicted_mask and reference_mask must have the same shape, "
            f"got {predicted.shape} and {reference.shape}"
        )
    return predicted, reference


def _metric_from_counts(
    numerator: int | float,
    denominator: int | float,
    *,
    empty_value: float,
) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator / denominator)


def confusion_counts(predicted_mask: MaskLike, reference_mask: MaskLike) -> dict[str, int]:
    """Return binary segmentation confusion counts.

    Counts use the common segmentation convention where ``predicted_mask`` is
    compared against ``reference_mask``. Any non-zero value is treated as true.
    """

    predicted, reference = _paired_masks(predicted_mask, reference_mask)
    true_positive = int(np.logical_and(predicted, reference).sum())
    false_positive = int(np.logical_and(predicted, ~reference).sum())
    false_negative = int(np.logical_and(~predicted, reference).sum())
    true_negative = int(np.logical_and(~predicted, ~reference).sum())
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }


def binary_mask_confusion_counts(predicted_mask: MaskLike, reference_mask: MaskLike) -> dict[str, int]:
    """Alias for :func:`confusion_counts` with an explicit mask-oriented name."""

    return confusion_counts(predicted_mask, reference_mask)


def iou(predicted_mask: MaskLike, reference_mask: MaskLike) -> float:
    """Return intersection-over-union for two binary masks.

    Two empty masks score ``1.0`` because they agree exactly.
    """

    counts = confusion_counts(predicted_mask, reference_mask)
    denominator = counts["tp"] + counts["fp"] + counts["fn"]
    return _metric_from_counts(counts["tp"], denominator, empty_value=1.0)


def dice(predicted_mask: MaskLike, reference_mask: MaskLike) -> float:
    """Return the Dice coefficient for two binary masks."""

    counts = confusion_counts(predicted_mask, reference_mask)
    numerator = 2 * counts["tp"]
    denominator = numerator + counts["fp"] + counts["fn"]
    return _metric_from_counts(numerator, denominator, empty_value=1.0)


def precision(predicted_mask: MaskLike, reference_mask: MaskLike) -> float:
    """Return mask precision, treating two empty masks as perfect agreement."""

    counts = confusion_counts(predicted_mask, reference_mask)
    denominator = counts["tp"] + counts["fp"]
    empty_value = 1.0 if counts["fn"] == 0 else 0.0
    return _metric_from_counts(counts["tp"], denominator, empty_value=empty_value)


def recall(predicted_mask: MaskLike, reference_mask: MaskLike) -> float:
    """Return mask recall, treating two empty masks as perfect agreement."""

    counts = confusion_counts(predicted_mask, reference_mask)
    denominator = counts["tp"] + counts["fn"]
    empty_value = 1.0 if counts["fp"] == 0 else 0.0
    return _metric_from_counts(counts["tp"], denominator, empty_value=empty_value)


def area_ratio(predicted_mask: MaskLike, reference_mask: MaskLike) -> float:
    """Return predicted positive area divided by reference positive area."""

    predicted, reference = _paired_masks(predicted_mask, reference_mask)
    predicted_area = int(predicted.sum())
    reference_area = int(reference.sum())
    if reference_area == 0:
        return 1.0 if predicted_area == 0 else float("inf")
    return float(predicted_area / reference_area)


def centroid_distance_px(predicted_mask: MaskLike, reference_mask: MaskLike) -> float:
    """Return Euclidean distance between positive-pixel centroids in pixels."""

    predicted, reference = _paired_masks(predicted_mask, reference_mask)
    predicted_count = int(predicted.sum())
    reference_count = int(reference.sum())
    if predicted_count == 0 and reference_count == 0:
        return 0.0
    if predicted_count == 0 or reference_count == 0:
        return float("inf")

    predicted_yx = np.argwhere(predicted).mean(axis=0)
    reference_yx = np.argwhere(reference).mean(axis=0)
    return float(hypot(*(predicted_yx - reference_yx)))


def boundary_mask(mask: MaskLike) -> np.ndarray:
    """Return a one-pixel interior boundary mask for a binary mask."""

    binary = _as_binary_mask(mask, name="mask")
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)

    mask_u8 = binary.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        mask_u8,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.logical_and(binary, eroded == 0)


def _dilate_boundary(boundary: np.ndarray, tolerance_px: int) -> np.ndarray:
    if tolerance_px <= 0 or not boundary.any():
        return boundary
    size = (tolerance_px * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dilated = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def boundary_iou(
    predicted_mask: MaskLike,
    reference_mask: MaskLike,
    *,
    tolerance_px: int = 1,
) -> float:
    """Return IoU between boundary bands with optional pixel tolerance."""

    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")

    predicted, reference = _paired_masks(predicted_mask, reference_mask)
    predicted_boundary = _dilate_boundary(boundary_mask(predicted), tolerance_px)
    reference_boundary = _dilate_boundary(boundary_mask(reference), tolerance_px)
    return iou(predicted_boundary, reference_boundary)


def geometry_validity_summary(geometry: BaseGeometry | None) -> dict[str, Any]:
    """Return a compact validity summary for a Shapely geometry."""

    if geometry is None:
        return {
            "is_present": False,
            "geometry_type": None,
            "is_empty": True,
            "is_valid": False,
            "validity_reason": "missing geometry",
            "area": 0.0,
            "length": 0.0,
            "bounds": None,
        }
    if not isinstance(geometry, BaseGeometry):
        raise TypeError(f"geometry must be a Shapely geometry or None, got {type(geometry)!r}")

    bounds = None if geometry.is_empty else tuple(float(value) for value in geometry.bounds)
    return {
        "is_present": True,
        "geometry_type": geometry.geom_type,
        "is_empty": bool(geometry.is_empty),
        "is_valid": bool(geometry.is_valid),
        "validity_reason": explain_validity(geometry),
        "area": float(geometry.area),
        "length": float(geometry.length),
        "bounds": bounds,
    }
