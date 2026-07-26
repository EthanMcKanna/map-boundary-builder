from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# This path is intentionally narrow. It handles dark, nearly monochrome maps
# whose service area is communicated by both a brighter translucent fill and a
# crisp light outline. Color-cluster extraction cannot distinguish that fill
# from the gray basemap, while the closed outline provides strong geometry.
GRAY_OUTLINE_MIN_DARK_FRACTION = 0.80
GRAY_OUTLINE_MIN_LOW_SATURATION_FRACTION = 0.75
GRAY_OUTLINE_MAX_SATURATION = 60
GRAY_OUTLINE_MIN_THRESHOLD = 180
GRAY_OUTLINE_MAX_THRESHOLD = 220
GRAY_OUTLINE_THRESHOLD_PERCENTILE = 99.4
GRAY_OUTLINE_THRESHOLD_OFFSET = 8.0
GRAY_OUTLINE_ENSEMBLE_STEP = 20
GRAY_OUTLINE_MIN_ENSEMBLE_CANDIDATES = 3
GRAY_OUTLINE_MIN_ENSEMBLE_IOU = 0.985
GRAY_OUTLINE_MIN_COMPONENT_COVERAGE = 0.0015
GRAY_OUTLINE_MAX_COMPONENT_COVERAGE = 0.04
GRAY_OUTLINE_MIN_SPAN_RATIO = 0.25
GRAY_OUTLINE_MAX_BBOX_DENSITY = 0.08
GRAY_OUTLINE_MIN_ENCLOSED_COVERAGE = 0.03
GRAY_OUTLINE_MAX_ENCLOSED_COVERAGE = 0.75
GRAY_OUTLINE_MIN_ENCLOSED_TO_STROKE_RATIO = 8.0
GRAY_OUTLINE_MIN_DOMINANT_HOLE_RATIO = 0.90
GRAY_OUTLINE_MIN_INTERIOR_EXTERIOR_MEDIAN_DELTA = 12.0
GRAY_OUTLINE_LOW_FREQUENCY_KERNEL_RATIO = 0.025
GRAY_OUTLINE_LOW_FREQUENCY_KERNEL_MIN = 7
GRAY_OUTLINE_LOW_FREQUENCY_KERNEL_MAX = 61
GRAY_OUTLINE_MIN_LOW_FREQUENCY_IOU = 0.95


@dataclass(frozen=True)
class GrayOutlineMask:
    mask: np.ndarray
    diagnostics: dict[str, object]


def detect_gray_outline_mask(rgb: np.ndarray) -> GrayOutlineMask | None:
    """Recover a gray service fill from its closed light outline.

    The detector is fail-closed: the frame must be dark and nearly monochrome,
    a sparse light component must form one dominant large enclosure, and that
    enclosure must be materially brighter than a local exterior ring. Those
    independent signals reject road lattices, route shields, title cards, and
    map-frame chrome without relying on a city name or catalog geometry.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 64:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    dark_fraction = float((value < 100).mean())
    low_saturation_fraction = float((saturation <= GRAY_OUTLINE_MAX_SATURATION).mean())
    if (
        dark_fraction < GRAY_OUTLINE_MIN_DARK_FRACTION
        or low_saturation_fraction < GRAY_OUTLINE_MIN_LOW_SATURATION_FRACTION
    ):
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    initial_threshold = int(
        round(
            np.clip(
                np.percentile(gray, GRAY_OUTLINE_THRESHOLD_PERCENTILE)
                + GRAY_OUTLINE_THRESHOLD_OFFSET,
                GRAY_OUTLINE_MIN_THRESHOLD,
                GRAY_OUTLINE_MAX_THRESHOLD,
            )
        )
    )
    thresholds: list[int] = []
    for threshold in (
        initial_threshold,
        initial_threshold - GRAY_OUTLINE_ENSEMBLE_STEP,
        initial_threshold - 2 * GRAY_OUTLINE_ENSEMBLE_STEP,
    ):
        threshold = max(GRAY_OUTLINE_MIN_THRESHOLD, int(threshold))
        if threshold not in thresholds:
            thresholds.append(threshold)

    candidates: list[GrayOutlineMask] = []
    for threshold in thresholds:
        light = (gray >= threshold) & (saturation <= GRAY_OUTLINE_MAX_SATURATION)
        candidate = best_gray_outline_component(
            light,
            gray,
            threshold=threshold,
            dark_fraction=dark_fraction,
            low_saturation_fraction=low_saturation_fraction,
        )
        if candidate is not None:
            candidates.append(candidate)
    if len(candidates) < GRAY_OUTLINE_MIN_ENSEMBLE_CANDIDATES:
        return None

    pairwise_ious = [
        binary_iou(first.mask, second.mask)
        for index, first in enumerate(candidates)
        for second in candidates[:index]
    ]
    minimum_iou = min(pairwise_ious, default=1.0)
    if minimum_iou < GRAY_OUTLINE_MIN_ENSEMBLE_IOU:
        return None

    # The highest threshold gives the sharpest outer stroke edge. Lower
    # thresholds are consensus witnesses only and are never exported.
    selected = candidates[0]
    return GrayOutlineMask(
        mask=selected.mask,
        diagnostics={
            **selected.diagnostics,
            "ensemble_thresholds": [
                int(candidate.diagnostics["threshold"]) for candidate in candidates
            ],
            "ensemble_minimum_iou": round(minimum_iou, 6),
        },
    )


def best_gray_outline_component(
    light: np.ndarray,
    gray: np.ndarray,
    *,
    threshold: int,
    dark_fraction: float,
    low_saturation_fraction: float,
) -> GrayOutlineMask | None:
    height, width = light.shape
    pixel_count = float(height * width)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        light.astype(np.uint8),
        8,
    )
    if component_count <= 1:
        return None

    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    best: tuple[float, GrayOutlineMask] | None = None
    for label in order:
        left, top, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        coverage = area / pixel_count
        if coverage < GRAY_OUTLINE_MIN_COMPONENT_COVERAGE:
            break
        if coverage > GRAY_OUTLINE_MAX_COMPONENT_COVERAGE:
            continue
        if (
            component_width < width * GRAY_OUTLINE_MIN_SPAN_RATIO
            or component_height < height * GRAY_OUTLINE_MIN_SPAN_RATIO
        ):
            continue
        if left == 0 or top == 0 or left + component_width == width or top + component_height == height:
            continue
        bbox_density = area / float(component_width * component_height)
        if bbox_density > GRAY_OUTLINE_MAX_BBOX_DENSITY:
            continue

        component = labels == label
        holes = enclosed_regions(component)
        hole_count, hole_labels, hole_stats, _hole_centroids = cv2.connectedComponentsWithStats(
            holes.astype(np.uint8),
            8,
        )
        if hole_count <= 1:
            continue
        hole_areas = hole_stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        largest_hole_label = int(np.argmax(hole_areas) + 1)
        largest_hole_area = float(hole_areas[largest_hole_label - 1])
        total_hole_area = float(hole_areas.sum())
        dominant_hole_ratio = largest_hole_area / max(total_hole_area, 1.0)
        enclosed_coverage = largest_hole_area / pixel_count
        if not (
            GRAY_OUTLINE_MIN_ENCLOSED_COVERAGE
            <= enclosed_coverage
            <= GRAY_OUTLINE_MAX_ENCLOSED_COVERAGE
        ):
            continue
        if largest_hole_area < area * GRAY_OUTLINE_MIN_ENCLOSED_TO_STROKE_RATIO:
            continue
        if dominant_hole_ratio < GRAY_OUTLINE_MIN_DOMINANT_HOLE_RATIO:
            continue

        interior = hole_labels == largest_hole_label
        # Retain only stroke pixels adjacent to the dominant enclosure. This
        # reaches the outline's outer edge while dropping any distant text or
        # road branch that happens to touch the same connected component.
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        adjacent_stroke = component & (
            cv2.dilate(interior.astype(np.uint8), stroke_kernel) > 0
        )
        filled = interior | adjacent_stroke
        contrast = gray_fill_contrast(gray, interior, filled)
        if contrast is None:
            continue
        median_delta, mean_delta = contrast
        if median_delta < GRAY_OUTLINE_MIN_INTERIOR_EXTERIOR_MEDIAN_DELTA:
            continue
        semantic = low_frequency_fill_agreement(gray, filled)
        if semantic is None:
            continue
        low_frequency_iou, otsu_threshold, low_frequency_coverage = semantic
        if low_frequency_iou < GRAY_OUTLINE_MIN_LOW_FREQUENCY_IOU:
            continue

        score = (
            enclosed_coverage
            * dominant_hole_ratio
            * min(2.0, median_delta / GRAY_OUTLINE_MIN_INTERIOR_EXTERIOR_MEDIAN_DELTA)
            * low_frequency_iou
        )
        result = GrayOutlineMask(
            mask=filled,
            diagnostics={
                "method": "luminance-outline",
                "verified_source_native": True,
                "threshold": threshold,
                "score": round(float(score), 6),
                "dark_fraction": round(dark_fraction, 6),
                "low_saturation_fraction": round(low_saturation_fraction, 6),
                "stroke_coverage": round(coverage, 6),
                "enclosed_coverage": round(enclosed_coverage, 6),
                "dominant_hole_ratio": round(dominant_hole_ratio, 6),
                "interior_exterior_median_delta": round(median_delta, 3),
                "interior_exterior_mean_delta": round(mean_delta, 3),
                "low_frequency_fill_iou": round(low_frequency_iou, 6),
                "low_frequency_otsu_threshold": round(otsu_threshold, 3),
                "low_frequency_fill_coverage": round(low_frequency_coverage, 6),
            },
        )
        if best is None or score > best[0]:
            best = (score, result)
    return None if best is None else best[1]


def enclosed_regions(component: np.ndarray) -> np.ndarray:
    padded = np.pad(component.astype(bool), 1, mode="constant", constant_values=False)
    background = (~padded).astype(np.uint8) * 255
    cv2.floodFill(background, None, (0, 0), 0)
    return background[1:-1, 1:-1] > 0


def gray_fill_contrast(
    gray: np.ndarray,
    interior: np.ndarray,
    filled: np.ndarray,
) -> tuple[float, float] | None:
    min_dimension = min(gray.shape)
    interior_size = max(3, round(min_dimension * 0.01)) | 1
    exterior_size = max(5, round(min_dimension * 0.04)) | 1
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (interior_size, interior_size),
    )
    exterior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (exterior_size, exterior_size),
    )
    interior_probe = cv2.erode(interior.astype(np.uint8), interior_kernel) > 0
    exterior_probe = (
        cv2.dilate(filled.astype(np.uint8), exterior_kernel) > 0
    ) & ~filled
    if not interior_probe.any() or not exterior_probe.any():
        return None
    interior_values = gray[interior_probe].astype(np.float32)
    exterior_values = gray[exterior_probe].astype(np.float32)
    return (
        float(np.median(interior_values) - np.median(exterior_values)),
        float(interior_values.mean() - exterior_values.mean()),
    )


def low_frequency_fill_agreement(
    gray: np.ndarray,
    outlined_fill: np.ndarray,
) -> tuple[float, float, float] | None:
    """Validate the outline against a separately inferred smooth gray fill.

    A median blur removes thin roads, text, and the outline itself. Otsu then
    separates the broad translucent fill from the dark basemap. The dominant
    non-border bright component must agree with the outline geometry, giving
    the detector an independent semantic signal instead of accepting any large
    closed white road loop.
    """

    height, width = gray.shape
    kernel_size = round(min(height, width) * GRAY_OUTLINE_LOW_FREQUENCY_KERNEL_RATIO)
    kernel_size = max(
        GRAY_OUTLINE_LOW_FREQUENCY_KERNEL_MIN,
        min(GRAY_OUTLINE_LOW_FREQUENCY_KERNEL_MAX, kernel_size),
    )
    kernel_size |= 1
    smoothed = cv2.medianBlur(gray, kernel_size)
    otsu_threshold, thresholded = cv2.threshold(
        smoothed,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (thresholded > 0).astype(np.uint8),
        8,
    )
    best_label: int | None = None
    best_area = 0
    for label in range(1, count):
        left, top, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        coverage = area / float(height * width)
        if not (
            GRAY_OUTLINE_MIN_ENCLOSED_COVERAGE
            <= coverage
            <= GRAY_OUTLINE_MAX_ENCLOSED_COVERAGE
        ):
            continue
        if left == 0 or top == 0 or left + component_width == width or top + component_height == height:
            continue
        if area > best_area:
            best_area = area
            best_label = label
    if best_label is None:
        return None
    low_frequency_fill = labels == best_label
    intersection = int(np.logical_and(low_frequency_fill, outlined_fill).sum())
    union = int(np.logical_or(low_frequency_fill, outlined_fill).sum())
    if union == 0:
        return None
    return (
        intersection / union,
        float(otsu_threshold),
        float(low_frequency_fill.mean()),
    )


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 1.0
