"""Standalone mask and geometry evaluation helpers."""

from __future__ import annotations

from math import acos, degrees, hypot
from typing import Any

import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

MaskLike = np.ndarray | list[Any] | tuple[Any, ...]

__all__ = [
    "area_ratio",
    "binary_mask_confusion_counts",
    "boundary_iou",
    "boundary_f1",
    "boundary_distance_summary_px",
    "boundary_distance_percentile_px",
    "boundary_mask",
    "centroid_distance_px",
    "confusion_counts",
    "dice",
    "geometry_validity_summary",
    "geometry_corner_f1",
    "geometry_complexity_comparison",
    "geometry_complexity_summary",
    "geometry_straight_run_rms_px",
    "geometry_topology_signature",
    "iou",
    "precision",
    "recall",
    "corner_f1",
    "rasterize_geometry_mask",
    "topology_signature",
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


def boundary_f1(
    predicted_mask: MaskLike,
    reference_mask: MaskLike,
    *,
    tolerance_px: float = 1.0,
) -> dict[str, float]:
    """Return symmetric boundary precision, recall, and F1 at a pixel radius.

    Unlike an IoU of dilated boundary bands, this metric scores each original
    boundary pixel exactly once. A predicted boundary pixel is precise when it
    lies within ``tolerance_px`` of any reference boundary pixel; recall is the
    converse. Distances use OpenCV's exact Euclidean distance transform.
    """

    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")
    predicted, reference = _paired_masks(predicted_mask, reference_mask)
    predicted_boundary = boundary_mask(predicted)
    reference_boundary = boundary_mask(reference)
    predicted_count = int(predicted_boundary.sum())
    reference_count = int(reference_boundary.sum())
    if predicted_count == 0 and reference_count == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if predicted_count == 0 or reference_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    distance_to_reference = _distance_to_boundary(reference_boundary)
    distance_to_prediction = _distance_to_boundary(predicted_boundary)
    precision = float((distance_to_reference[predicted_boundary] <= tolerance_px).mean())
    recall = float((distance_to_prediction[reference_boundary] <= tolerance_px).mean())
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _distance_to_boundary(boundary: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(
        (~boundary).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )


def _symmetric_boundary_distances_px(
    predicted_mask: MaskLike,
    reference_mask: MaskLike,
) -> np.ndarray:
    predicted, reference = _paired_masks(predicted_mask, reference_mask)
    predicted_boundary = boundary_mask(predicted)
    reference_boundary = boundary_mask(reference)
    if not predicted_boundary.any() and not reference_boundary.any():
        return np.zeros(0, dtype=np.float64)
    if not predicted_boundary.any() or not reference_boundary.any():
        return np.asarray([float("inf")], dtype=np.float64)
    distance_to_reference = _distance_to_boundary(reference_boundary)
    distance_to_prediction = _distance_to_boundary(predicted_boundary)
    return np.concatenate(
        [distance_to_reference[predicted_boundary], distance_to_prediction[reference_boundary]]
    ).astype(np.float64, copy=False)


def boundary_distance_percentile_px(
    predicted_mask: MaskLike,
    reference_mask: MaskLike,
    *,
    percentile: float = 95.0,
) -> float:
    """Return a symmetric percentile surface distance in source pixels."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    distances = _symmetric_boundary_distances_px(predicted_mask, reference_mask)
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, percentile))


def boundary_distance_summary_px(
    predicted_mask: MaskLike,
    reference_mask: MaskLike,
) -> dict[str, float]:
    """Return mean, p95, p99, and maximum symmetric boundary distance."""

    distances = _symmetric_boundary_distances_px(predicted_mask, reference_mask)
    if distances.size == 0:
        return {"mean_px": 0.0, "p95_px": 0.0, "p99_px": 0.0, "max_px": 0.0}
    return {
        "mean_px": float(np.mean(distances)),
        "p95_px": float(np.percentile(distances, 95.0)),
        "p99_px": float(np.percentile(distances, 99.0)),
        "max_px": float(np.max(distances)),
    }


def corner_f1(
    predicted_mask: MaskLike,
    reference_mask: MaskLike,
    *,
    tolerance_px: float = 2.0,
) -> dict[str, float | int]:
    """Score sharp contour vertices and their angles with one-to-one matches."""
    predicted, reference = _paired_masks(predicted_mask, reference_mask)
    return _match_corner_observations(
        _mask_corner_observations(predicted),
        _mask_corner_observations(reference),
        tolerance_px=tolerance_px,
    )


def geometry_corner_f1(
    predicted_geometry: BaseGeometry,
    reference_geometry: BaseGeometry,
    *,
    tolerance_px: float = 2.0,
    angle_threshold_degrees: float = 150.0,
) -> dict[str, float | int]:
    """Score meaningful vector corners and their angles one-to-one."""
    return _match_corner_observations(
        _geometry_corner_observations(predicted_geometry, angle_threshold_degrees),
        _geometry_corner_observations(reference_geometry, angle_threshold_degrees),
        tolerance_px=tolerance_px,
    )


def topology_signature(mask: MaskLike) -> dict[str, int]:
    binary = _as_binary_mask(mask, name="mask")
    count, _labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    _contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0 if hierarchy is None else sum(1 for row in hierarchy[0] if int(row[3]) >= 0)
    return {"components": max(0, int(count) - 1), "holes": int(holes)}


def geometry_topology_signature(geometry: BaseGeometry) -> dict[str, int]:
    """Return vector topology without raster-connectivity artifacts.

    A single diagonally exterior-connected raster pixel can appear as a hole
    to contour tracing even when the source polygon has none. Promotion gates
    therefore use the actual polygonal contract; the raster signature remains
    useful as a rendering diagnostic.
    """

    if geometry is None or geometry.is_empty:
        return {"components": 0, "holes": 0}
    if geometry.geom_type == "Polygon":
        return {"components": 1, "holes": len(geometry.interiors)}
    if geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
        return {
            "components": len(polygons),
            "holes": sum(len(polygon.interiors) for polygon in polygons),
        }
    if hasattr(geometry, "geoms"):
        signatures = [
            geometry_topology_signature(part)
            for part in geometry.geoms
            if part.geom_type in {"Polygon", "MultiPolygon"}
        ]
        return {
            "components": sum(item["components"] for item in signatures),
            "holes": sum(item["holes"] for item in signatures),
        }
    return {"components": 0, "holes": 0}


def _match_corner_observations(
    predicted: list[tuple[float, float, float]],
    reference: list[tuple[float, float, float]],
    *,
    tolerance_px: float,
) -> dict[str, float | int]:
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")
    if not predicted and not reference:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "matched_count": 0,
            "mean_angular_error_degrees": 0.0,
            "p95_angular_error_degrees": 0.0,
            "max_angular_error_degrees": 0.0,
        }
    if not predicted or not reference:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_count": 0,
            "mean_angular_error_degrees": float("inf"),
            "p95_angular_error_degrees": float("inf"),
            "max_angular_error_degrees": float("inf"),
        }

    predicted_xy = np.asarray([item[:2] for item in predicted], dtype=np.float64)
    reference_xy = np.asarray([item[:2] for item in reference], dtype=np.float64)
    distances = np.linalg.norm(predicted_xy[:, None, :] - reference_xy[None, :, :], axis=2)
    adjacency = {
        predicted_index: sorted(
            (
                (float(distances[predicted_index, reference_index]), reference_index)
                for reference_index in range(len(reference))
                if distances[predicted_index, reference_index] <= tolerance_px
            ),
            key=lambda item: item[0],
        )
        for predicted_index in range(len(predicted))
    }
    reference_match: dict[int, int] = {}

    def augment(predicted_index: int, visited_references: set[int]) -> bool:
        for _distance, reference_index in adjacency[predicted_index]:
            if reference_index in visited_references:
                continue
            visited_references.add(reference_index)
            displaced = reference_match.get(reference_index)
            if displaced is None or augment(displaced, visited_references):
                reference_match[reference_index] = predicted_index
                return True
        return False

    # Constrained observations get first choice. Augmenting paths retain maximum
    # cardinality where a nearest-only greedy match can undercount close corners.
    match_order = sorted(
        range(len(predicted)),
        key=lambda index: (
            len(adjacency[index]),
            adjacency[index][0][0] if adjacency[index] else float("inf"),
        ),
    )
    for predicted_index in match_order:
        augment(predicted_index, set())

    angular_errors = [
        abs(predicted[predicted_index][2] - reference[reference_index][2])
        for reference_index, predicted_index in reference_match.items()
    ]
    matched_count = len(reference_match)
    precision_value = float(matched_count / len(predicted))
    recall_value = float(matched_count / len(reference))
    f1_value = 2.0 * precision_value * recall_value / max(1e-12, precision_value + recall_value)
    if angular_errors:
        mean_angular_error = float(np.mean(angular_errors))
        p95_angular_error = float(np.percentile(angular_errors, 95.0))
        max_angular_error = float(max(angular_errors))
    else:
        mean_angular_error = p95_angular_error = max_angular_error = float("inf")
    return {
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "matched_count": matched_count,
        "mean_angular_error_degrees": mean_angular_error,
        "p95_angular_error_degrees": p95_angular_error,
        "max_angular_error_degrees": max_angular_error,
    }


def _mask_corner_observations(mask: np.ndarray) -> list[tuple[float, float, float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    corners: list[tuple[float, float, float]] = []
    for contour in contours:
        if len(contour) < 8:
            continue
        approximation = cv2.approxPolyDP(contour, epsilon=1.25, closed=True)
        points = approximation[:, 0, :].astype(np.float32)
        for index, point in enumerate(points):
            before = points[index - 1] - point
            after = points[(index + 1) % len(points)] - point
            denominator = max(1e-6, float(np.linalg.norm(before) * np.linalg.norm(after)))
            angle = float(np.degrees(np.arccos(np.clip(np.dot(before, after) / denominator, -1.0, 1.0))))
            if angle < 155.0:
                corners.append((float(point[0]), float(point[1]), angle))
    return corners


def _geometry_corner_observations(
    geometry: BaseGeometry,
    angle_threshold_degrees: float,
) -> list[tuple[float, float, float]]:
    if geometry.is_empty:
        return []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", ()))
    corners: list[tuple[float, float, float]] = []
    for polygon in polygons:
        if polygon.geom_type != "Polygon":
            continue
        for ring in (polygon.exterior, *polygon.interiors):
            simplified = ring.simplify(0.75, preserve_topology=False)
            points = list(simplified.coords)
            if len(points) > 1 and points[0] == points[-1]:
                points.pop()
            if len(points) < 3:
                continue
            for index, point in enumerate(points):
                before = np.asarray(points[index - 1], dtype=np.float64) - point
                after = np.asarray(points[(index + 1) % len(points)], dtype=np.float64) - point
                denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
                if denominator < 1e-6:
                    continue
                angle = degrees(acos(float(np.clip(np.dot(before, after) / denominator, -1.0, 1.0))))
                if angle < angle_threshold_degrees:
                    corners.append((float(point[0]), float(point[1]), angle))
    return corners


def _geometry_rings(geometry: BaseGeometry) -> list[list[tuple[float, float]]]:
    if geometry.is_empty:
        return []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", ()))
    rings: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        if polygon.geom_type != "Polygon":
            continue
        for ring in (polygon.exterior, *polygon.interiors):
            points = [(float(x), float(y)) for x, y, *_rest in ring.coords]
            if len(points) > 1 and points[0] == points[-1]:
                points.pop()
            if len(points) >= 3:
                rings.append(points)
    return rings


def _ring_vertex_count(geometry: BaseGeometry) -> int:
    return sum(len(points) for points in _geometry_rings(geometry))


def _cyclic_slice(points: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        return points[start : end + 1]
    return np.concatenate([points[start:], points[: end + 1]], axis=0)


def geometry_straight_run_rms_px(
    geometry: BaseGeometry,
    *,
    anchor_tolerance_px: float = 2.0,
    min_run_length_px: float = 8.0,
) -> float:
    """Measure high-frequency wobble along simplified straight boundary runs.

    Douglas-Peucker anchor segments define the intended runs. Original vertices
    between each pair of anchors are measured against that segment, so exact
    linework scores zero while stair steps and rough curves remain visible. The
    result is a length-weighted RMS in source pixels.
    """

    if anchor_tolerance_px <= 0:
        raise ValueError("anchor_tolerance_px must be positive")
    if min_run_length_px < 0:
        raise ValueError("min_run_length_px must be non-negative")
    squared_error_sum = 0.0
    weight_sum = 0.0
    for points_list in _geometry_rings(geometry):
        points = np.asarray(points_list, dtype=np.float64)
        closed = np.vstack([points, points[0]])
        # Shapely's simplifier retains source vertices, which gives stable
        # indices back into the original ring without a rasterization step.
        simplified = LineString(closed).simplify(anchor_tolerance_px, preserve_topology=False)
        anchors = np.asarray(simplified.coords, dtype=np.float64)
        if len(anchors) > 1 and np.allclose(anchors[0], anchors[-1]):
            anchors = anchors[:-1]
        if len(anchors) < 3:
            continue
        anchor_indices: list[int] = []
        for anchor in anchors:
            distances = np.linalg.norm(points - anchor, axis=1)
            anchor_indices.append(int(np.argmin(distances)))
        # Remove duplicate anchors while preserving cyclic order.
        anchor_indices = list(dict.fromkeys(anchor_indices))
        if len(anchor_indices) < 3:
            continue
        anchor_indices.sort()
        for anchor_index, start in enumerate(anchor_indices):
            end = anchor_indices[(anchor_index + 1) % len(anchor_indices)]
            run = _cyclic_slice(points, start, end)
            if len(run) < 2:
                continue
            vector = run[-1] - run[0]
            length = float(np.linalg.norm(vector))
            if length < min_run_length_px:
                continue
            offsets = run - run[0]
            distances = np.abs((vector[0] * offsets[:, 1]) - (vector[1] * offsets[:, 0])) / length
            # Give each source segment approximately length-proportional weight
            # instead of letting densely tessellated regions dominate.
            if len(run) == 2:
                weights = np.asarray([length / 2.0, length / 2.0])
            else:
                segment_lengths = np.linalg.norm(np.diff(run, axis=0), axis=1)
                weights = np.empty(len(run), dtype=np.float64)
                weights[0] = segment_lengths[0] / 2.0
                weights[-1] = segment_lengths[-1] / 2.0
                weights[1:-1] = (segment_lengths[:-1] + segment_lengths[1:]) / 2.0
            squared_error_sum += float(np.sum(weights * np.square(distances)))
            weight_sum += float(np.sum(weights))
    if weight_sum == 0.0:
        return 0.0
    return float(np.sqrt(squared_error_sum / weight_sum))


def geometry_complexity_summary(
    geometry: BaseGeometry,
    *,
    max_deviation_px: float = 0.35,
) -> dict[str, float | int]:
    """Summarize vector complexity at a fixed source-pixel error budget."""

    if max_deviation_px <= 0:
        raise ValueError("max_deviation_px must be positive")
    raw_vertices = _ring_vertex_count(geometry)
    simplified = geometry.simplify(max_deviation_px, preserve_topology=True)
    simplified_vertices = _ring_vertex_count(simplified)
    perimeter = float(geometry.length)
    return {
        "vertex_count": raw_vertices,
        "simplified_vertex_count": simplified_vertices,
        "canonical_vertex_count": simplified_vertices,
        "excess_vertex_ratio": float(raw_vertices / max(1, simplified_vertices)),
        "vertices_per_100px": float(100.0 * raw_vertices / perimeter) if perimeter > 0 else 0.0,
        "straight_run_rms_px": geometry_straight_run_rms_px(geometry),
    }


def geometry_complexity_comparison(
    predicted_geometry: BaseGeometry,
    reference_geometry: BaseGeometry,
    *,
    max_deviation_px: float = 0.35,
) -> dict[str, float | int]:
    """Compare roughness against the reference geometry at one pixel budget.

    Absolute self-roughness remains useful for diagnostics, but it has a
    reference-dependent floor: even exact authored geometry can contain nearby
    legitimate vertices that simplify into a slightly non-straight run. The
    promotion metrics therefore gate only excess straight-run RMS and the ratio
    of canonical (simplified) predicted vertices to canonical reference
    vertices. An exact prediction scores ``0.0`` and ``1.0`` respectively.
    """

    predicted = geometry_complexity_summary(
        predicted_geometry,
        max_deviation_px=max_deviation_px,
    )
    reference = geometry_complexity_summary(
        reference_geometry,
        max_deviation_px=max_deviation_px,
    )
    predicted_canonical = int(predicted["canonical_vertex_count"])
    reference_canonical = int(reference["canonical_vertex_count"])
    if reference_canonical:
        canonical_ratio = float(predicted_canonical / reference_canonical)
    else:
        canonical_ratio = 1.0 if predicted_canonical == 0 else float("inf")
    predicted_rms = float(predicted["straight_run_rms_px"])
    reference_rms = float(reference["straight_run_rms_px"])
    return {
        "predicted_straight_run_rms_px": predicted_rms,
        "reference_straight_run_rms_px": reference_rms,
        "straight_run_rms_excess_px": max(0.0, predicted_rms - reference_rms),
        "predicted_excess_vertex_ratio": float(predicted["excess_vertex_ratio"]),
        "reference_excess_vertex_ratio": float(reference["excess_vertex_ratio"]),
        "predicted_canonical_vertex_count": predicted_canonical,
        "reference_canonical_vertex_count": reference_canonical,
        "canonical_complexity_ratio": canonical_ratio,
    }


def rasterize_geometry_mask(
    geometry: BaseGeometry,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize polygonal source-pixel geometry with the production contract.

    Synthetic truth must use the same rounding, clipping, and OpenCV fill
    convention as runtime geometry. Keeping this dependency-light helper in the
    evaluation layer lets the synthetic generator import it without importing
    the extraction/runtime graph or creating a circular dependency.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if geometry is None or geometry.is_empty:
        return np.zeros((height, width), dtype=bool)
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
    else:
        raise ValueError(f"geometry must be Polygon or MultiPolygon, got {geometry.geom_type}")

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        exterior = _raster_ring(polygon.exterior.coords, width=width, height=height)
        if len(exterior) >= 3:
            cv2.fillPoly(mask, [exterior], 1)
        for interior in polygon.interiors:
            hole = _raster_ring(interior.coords, width=width, height=height)
            if len(hole) >= 3:
                cv2.fillPoly(mask, [hole], 0)
    return mask.astype(bool)


def _raster_ring(coordinates: Any, *, width: int, height: int) -> np.ndarray:
    points = np.asarray(list(coordinates), dtype=np.float64)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if not len(points):
        return np.empty((0, 2), dtype=np.int32)
    points[:, 0] = np.clip(np.rint(points[:, 0]), 0, width - 1)
    points[:, 1] = np.clip(np.rint(points[:, 1]), 0, height - 1)
    return points.astype(np.int32)


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
