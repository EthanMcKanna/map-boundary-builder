"""Robust sharp-vector reconstruction for dense rectilinear rings.

The EdgeGraph localizer deliberately emits a dense source-pixel contour.  That
is the right representation for edge localization, but it is a poor final
representation for maps whose true boundary is made from straight orthogonal
runs: small independent localization errors become visibly wavy curves.

This module fits such contours in a deliberately fail-closed way.  It infers a
pair of dominant orthogonal axes, segments the cyclic contour with a two-state
model, robustly fits axis-constrained lines, and intersects neighboring lines.
The candidate is returned only when it remains a simple polygon and is tightly
supported by the dense input in both directions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon


@dataclass(frozen=True)
class RectilinearFitConfig:
    """Conservative acceptance and fitting parameters in source pixels."""

    resample_step_px: float = 1.25
    maximum_resampled_points: int = 16_384
    axis_window_px: float = 13.0
    label_window_px: float = 3.0
    smoothing_radius_px: float = 1.75
    axis_alignment_tolerance_degrees: float = 13.0
    minimum_axis_concentration: float = 0.58
    minimum_aligned_fraction: float = 0.72
    minimum_secondary_axis_fraction: float = 0.025
    label_switch_penalty: float = 0.40
    minimum_run_progress_px: float = 2.0
    minimum_run_efficiency: float = 0.52
    maximum_runs: int = 96
    maximum_line_residual_p95_px: float = 3.25
    maximum_line_residual_median_px: float = 1.65
    support_sample_step_px: float = 2.0
    maximum_support_p95_px: float = 3.5
    maximum_support_p99_px: float = 5.5
    maximum_support_distance_px: float = 12.0
    minimum_area_ratio: float = 0.90
    maximum_area_ratio: float = 1.10
    maximum_symmetric_difference_fraction: float = 0.10
    maximum_complexity_ratio: float = 0.40
    minimum_polygon_area_px: float = 16.0
    maximum_tiny_loop_area_px: float = 256.0
    maximum_tiny_loop_area_fraction: float = 0.001
    maximum_tiny_loop_deviation_px: float = 12.0

    def __post_init__(self) -> None:
        if self.resample_step_px <= 0.0:
            raise ValueError("resample_step_px must be positive")
        if self.maximum_resampled_points < 16:
            raise ValueError("maximum_resampled_points must be at least 16")
        if self.axis_window_px <= 0.0 or self.label_window_px <= 0.0:
            raise ValueError("axis and label windows must be positive")
        if self.smoothing_radius_px < 0.0:
            raise ValueError("smoothing_radius_px must be non-negative")
        if not 0.0 < self.axis_alignment_tolerance_degrees < 45.0:
            raise ValueError("axis alignment tolerance must be in (0, 45)")
        if not 0.0 <= self.minimum_axis_concentration <= 1.0:
            raise ValueError("minimum_axis_concentration must be in [0, 1]")
        if not 0.0 <= self.minimum_aligned_fraction <= 1.0:
            raise ValueError("minimum_aligned_fraction must be in [0, 1]")
        if not 0.0 <= self.minimum_secondary_axis_fraction <= 0.5:
            raise ValueError("minimum_secondary_axis_fraction must be in [0, 0.5]")
        if self.label_switch_penalty < 0.0:
            raise ValueError("label_switch_penalty must be non-negative")
        if self.minimum_run_progress_px <= 0.0:
            raise ValueError("minimum_run_progress_px must be positive")
        if not 0.0 <= self.minimum_run_efficiency <= 1.0:
            raise ValueError("minimum_run_efficiency must be in [0, 1]")
        if self.maximum_runs < 4:
            raise ValueError("maximum_runs must be at least four")
        if not 0.0 < self.minimum_area_ratio <= 1.0:
            raise ValueError("minimum_area_ratio must be in (0, 1]")
        if self.maximum_area_ratio < 1.0:
            raise ValueError("maximum_area_ratio must be at least one")
        if not 0.0 < self.maximum_complexity_ratio <= 1.0:
            raise ValueError("maximum_complexity_ratio must be in (0, 1]")
        if self.maximum_tiny_loop_area_px <= 0.0:
            raise ValueError("maximum_tiny_loop_area_px must be positive")
        if not 0.0 < self.maximum_tiny_loop_area_fraction < 0.01:
            raise ValueError("maximum_tiny_loop_area_fraction must be in (0, 0.01)")
        if self.maximum_tiny_loop_deviation_px <= 0.0:
            raise ValueError("maximum_tiny_loop_deviation_px must be positive")


@dataclass(frozen=True)
class RectilinearFitDiagnostics:
    accepted: bool
    reason: str
    input_point_count: int = 0
    resampled_point_count: int = 0
    axis_angle_degrees: float = 0.0
    axis_concentration: float = 0.0
    aligned_fraction: float = 0.0
    secondary_axis_fraction: float = 0.0
    run_count: int = 0
    vertex_count: int = 0
    line_residual_median_px: float = math.inf
    line_residual_p95_px: float = math.inf
    dense_to_candidate_p95_px: float = math.inf
    candidate_to_dense_p95_px: float = math.inf
    support_p95_px: float = math.inf
    support_p99_px: float = math.inf
    support_maximum_px: float = math.inf
    area_ratio: float = 0.0
    symmetric_difference_fraction: float = math.inf
    complexity_ratio: float = math.inf
    source_repaired: bool = False
    discarded_tiny_loop_area_px: float = 0.0
    discarded_tiny_loop_area_fraction: float = 0.0
    tiny_loop_maximum_deviation_px: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RectilinearFitResult:
    candidate: Polygon | None
    diagnostics: RectilinearFitDiagnostics

    @property
    def polygon(self) -> Polygon | None:
        """Alias for callers that use geometry terminology."""

        return self.candidate


@dataclass(frozen=True)
class _Run:
    label: int
    indices: np.ndarray
    line_coordinate: float
    residuals: np.ndarray
    net_progress: float
    path_progress: float


@dataclass(frozen=True)
class _TinyLoopRepair:
    source: Polygon
    ring: np.ndarray
    discarded_area_px: float
    discarded_area_fraction: float
    maximum_deviation_px: float


def fit_rectilinear_ring(
    points: np.ndarray | list[tuple[float, float]],
    *,
    config: RectilinearFitConfig | None = None,
) -> RectilinearFitResult:
    """Fit a sharp orthogonal polygon to a dense closed ring.

    ``candidate`` is ``None`` whenever the source is not convincingly
    rectilinear or any geometric/support gate fails.  Invalid rings fail
    closed except for one narrowly defined localizer artifact: negligible
    touching lobes may be removed when every lobe passes independent topology,
    absolute-area, aggregate relative-area, and maximum-deviation gates.
    General ``buffer(0)`` repair is deliberately forbidden because it can
    silently change topology.
    """

    cfg = config or RectilinearFitConfig()
    dense = _normalize_ring(points)
    if dense is None:
        return _reject("invalid-input")
    input_count = len(dense)
    observed_dense = dense
    source = Polygon(dense)
    repair: _TinyLoopRepair | None = None
    if not source.is_empty and not source.is_valid:
        repair = _repair_tiny_touching_loops(dense, source=source, config=cfg)
        if repair is not None:
            dense = repair.ring
            source = repair.source
    repair_common = {
        "source_repaired": repair is not None,
        "discarded_tiny_loop_area_px": (
            repair.discarded_area_px if repair is not None else 0.0
        ),
        "discarded_tiny_loop_area_fraction": (
            repair.discarded_area_fraction if repair is not None else 0.0
        ),
        "tiny_loop_maximum_deviation_px": (
            repair.maximum_deviation_px if repair is not None else 0.0
        ),
    }
    if (
        source.is_empty
        or not source.is_valid
        or source.area < cfg.minimum_polygon_area_px
    ):
        return _reject(
            "invalid-source-geometry",
            input_point_count=input_count,
            **repair_common,
        )

    resampled, sample_step = _resample_ring(dense, cfg)
    count = len(resampled)
    if count < 12:
        return _reject(
            "insufficient-perimeter",
            input_point_count=input_count,
            resampled_point_count=count,
            **repair_common,
        )

    smoothed = _cyclic_median_smooth(
        resampled,
        radius=max(0, int(round(cfg.smoothing_radius_px / sample_step))),
    )
    axis = _infer_axes(smoothed, sample_step=sample_step, config=cfg)
    common = {
        "input_point_count": input_count,
        "resampled_point_count": count,
        "axis_angle_degrees": axis[2],
        "axis_concentration": axis[3],
        "aligned_fraction": axis[4],
        "secondary_axis_fraction": axis[5],
        **repair_common,
    }
    if axis[3] < cfg.minimum_axis_concentration:
        return _reject("weak-axis-concentration", **common)
    if axis[4] < cfg.minimum_aligned_fraction:
        return _reject("insufficient-axis-alignment", **common)
    if axis[5] < cfg.minimum_secondary_axis_fraction:
        return _reject("degenerate-axis-coverage", **common)
    first_axis, second_axis = axis[:2]

    transformed = np.stack(
        [resampled @ first_axis, resampled @ second_axis], axis=1
    )
    transformed_smoothed = np.stack(
        [smoothed @ first_axis, smoothed @ second_axis], axis=1
    )
    labels = _segment_axis_runs(
        transformed_smoothed,
        sample_step=sample_step,
        config=cfg,
    )
    labels = _remove_weak_runs(labels, transformed, config=cfg)
    spans = _cyclic_run_spans(labels)
    if len(spans) < 4 or len(spans) % 2:
        return _reject("invalid-run-cycle", run_count=len(spans), **common)
    if len(spans) > cfg.maximum_runs:
        return _reject("excessive-run-complexity", run_count=len(spans), **common)

    runs = [_fit_constrained_run(transformed, labels, span) for span in spans]
    residuals = np.concatenate([run.residuals for run in runs])
    residual_median = float(np.median(residuals))
    residual_p95 = float(np.quantile(residuals, 0.95))
    fit_common = {
        **common,
        "run_count": len(runs),
        "line_residual_median_px": residual_median,
        "line_residual_p95_px": residual_p95,
    }
    if residual_median > cfg.maximum_line_residual_median_px:
        return _reject("large-median-line-residual", **fit_common)
    if residual_p95 > cfg.maximum_line_residual_p95_px:
        return _reject("large-line-residual", **fit_common)

    vertices_uv = _intersect_run_lines(runs)
    vertices_xy = (
        vertices_uv[:, 0, None] * first_axis[None, :]
        + vertices_uv[:, 1, None] * second_axis[None, :]
    )
    vertices_xy = _remove_duplicate_vertices(vertices_xy)
    vertex_count = len(vertices_xy)
    if vertex_count < 4 or vertex_count % 2:
        return _reject("invalid-candidate-cycle", vertex_count=vertex_count, **fit_common)
    candidate = Polygon(vertices_xy)
    if candidate.is_empty or not candidate.is_valid or candidate.area < cfg.minimum_polygon_area_px:
        return _reject("invalid-candidate-geometry", vertex_count=vertex_count, **fit_common)

    # Preserve winding so callers can substitute the ring without another
    # orientation normalization pass.
    if _signed_area(observed_dense) * _signed_area(vertices_xy) < 0.0:
        vertices_xy = vertices_xy[::-1].copy()
        candidate = Polygon(vertices_xy)

    area_ratio = float(candidate.area / source.area)
    difference_fraction = float(source.symmetric_difference(candidate).area / source.area)
    # Complexity is measured against the source-native representation used by
    # the fitter.  A four-corner rectangle passed sparsely and the same ring
    # sampled at every pixel should make the same acceptance decision.
    complexity_ratio = float(vertex_count / max(1, count))
    geometry_common = {
        **fit_common,
        "vertex_count": vertex_count,
        "area_ratio": area_ratio,
        "symmetric_difference_fraction": difference_fraction,
        "complexity_ratio": complexity_ratio,
    }
    if not cfg.minimum_area_ratio <= area_ratio <= cfg.maximum_area_ratio:
        return _reject("area-ratio-gate", **geometry_common)
    if difference_fraction > cfg.maximum_symmetric_difference_fraction:
        return _reject("overlap-gate", **geometry_common)
    if complexity_ratio > cfg.maximum_complexity_ratio:
        return _reject("complexity-gate", **geometry_common)

    candidate_samples = _resample_polygon_boundary(
        vertices_xy,
        step=cfg.support_sample_step_px,
        maximum_points=cfg.maximum_resampled_points,
    )
    dense_to_candidate = _point_to_ring_distances(resampled, vertices_xy)
    candidate_to_dense = _point_to_ring_distances(candidate_samples, dense)
    if repair is not None:
        # The discarded lobe is excluded from line/run inference, but it is
        # still held against the final candidate.  This prevents a nominally
        # small-area yet spatially long excursion from bypassing the existing
        # support-distance gates.
        dense_to_candidate = np.concatenate(
            [
                dense_to_candidate,
                _point_to_ring_distances(observed_dense, vertices_xy),
            ]
        )
        candidate_to_dense = np.maximum(
            candidate_to_dense,
            _point_to_ring_distances(candidate_samples, observed_dense),
        )
    support = np.concatenate([dense_to_candidate, candidate_to_dense])
    dense_p95 = float(np.quantile(dense_to_candidate, 0.95))
    candidate_p95 = float(np.quantile(candidate_to_dense, 0.95))
    support_p95 = float(np.quantile(support, 0.95))
    support_p99 = float(np.quantile(support, 0.99))
    support_maximum = float(np.max(support, initial=0.0))
    accepted_common = {
        **geometry_common,
        "dense_to_candidate_p95_px": dense_p95,
        "candidate_to_dense_p95_px": candidate_p95,
        "support_p95_px": support_p95,
        "support_p99_px": support_p99,
        "support_maximum_px": support_maximum,
    }
    if support_p95 > cfg.maximum_support_p95_px:
        return _reject("support-p95-gate", **accepted_common)
    if support_p99 > cfg.maximum_support_p99_px:
        return _reject("support-p99-gate", **accepted_common)
    if support_maximum > cfg.maximum_support_distance_px:
        return _reject("support-maximum-gate", **accepted_common)

    diagnostics = RectilinearFitDiagnostics(
        accepted=True,
        reason="accepted-tiny-loop-repair" if repair is not None else "accepted",
        **accepted_common,
    )
    return RectilinearFitResult(candidate=candidate, diagnostics=diagnostics)


def _reject(reason: str, **values: Any) -> RectilinearFitResult:
    diagnostics = RectilinearFitDiagnostics(accepted=False, reason=reason, **values)
    return RectilinearFitResult(candidate=None, diagnostics=diagnostics)


def _normalize_ring(
    points: np.ndarray | list[tuple[float, float]],
) -> np.ndarray | None:
    try:
        ring = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(ring) < 4 or not np.isfinite(ring).all():
        return None
    keep = np.ones(len(ring), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(ring, axis=0), axis=1) > 1e-7
    ring = ring[keep]
    if len(ring) > 1 and float(np.linalg.norm(ring[-1] - ring[0])) <= 1e-7:
        ring = ring[:-1]
    if len(ring) < 4:
        return None
    return ring


def _repair_tiny_touching_loops(
    ring: np.ndarray,
    *,
    source: Polygon,
    config: RectilinearFitConfig,
) -> _TinyLoopRepair | None:
    """Remove provably negligible touching lobes from an invalid ring.

    ``make_valid`` is used only to inspect the linework.  Repair is accepted
    when it produces one dominant simple, hole-free polygon plus one or more
    simple, hole-free components that touch it at self-intersections.  Every
    discarded component must be absolutely tiny, their aggregate area must be
    negligible relative to the observed linework, and the original ring must
    remain spatially close to the dominant boundary.  Bow ties, holes,
    disconnected components, and material topology changes therefore remain
    invalid.
    """

    try:
        repaired = make_valid(source)
    except GEOSException:
        return None
    if not isinstance(repaired, MultiPolygon) or len(repaired.geoms) < 2:
        return None
    parts = sorted(repaired.geoms, key=lambda part: part.area, reverse=True)
    dominant, artifacts = parts[0], parts[1:]
    if (
        dominant.is_empty
        or not dominant.is_valid
        or len(dominant.interiors) != 0
        or dominant.area < config.minimum_polygon_area_px
        or any(
            artifact.is_empty
            or not artifact.is_valid
            or len(artifact.interiors) != 0
            or not dominant.touches(artifact)
            for artifact in artifacts
        )
    ):
        return None

    discarded_area = float(sum(artifact.area for artifact in artifacts))
    combined_area = float(dominant.area + discarded_area)
    if combined_area <= 0.0:
        return None
    discarded_fraction = discarded_area / combined_area
    if (
        any(
            artifact.area > config.maximum_tiny_loop_area_px
            for artifact in artifacts
        )
        or discarded_fraction > config.maximum_tiny_loop_area_fraction
    ):
        return None

    dominant_ring = np.asarray(dominant.exterior.coords[:-1], dtype=np.float64)
    if len(dominant_ring) < 4:
        return None
    if _signed_area(ring) * _signed_area(dominant_ring) < 0.0:
        dominant_ring = dominant_ring[::-1].copy()
    deviation = np.concatenate(
        [
            _point_to_ring_distances(ring, dominant_ring),
            _point_to_ring_distances(dominant_ring, ring),
        ]
    )
    maximum_deviation = float(np.max(deviation, initial=0.0))
    if maximum_deviation > config.maximum_tiny_loop_deviation_px:
        return None
    return _TinyLoopRepair(
        source=dominant,
        ring=dominant_ring,
        discarded_area_px=discarded_area,
        discarded_area_fraction=discarded_fraction,
        maximum_deviation_px=maximum_deviation,
    )


def _resample_ring(
    ring: np.ndarray,
    config: RectilinearFitConfig,
) -> tuple[np.ndarray, float]:
    vectors = np.roll(ring, -1, axis=0) - ring
    lengths = np.linalg.norm(vectors, axis=1)
    perimeter = float(lengths.sum())
    step = max(
        config.resample_step_px,
        perimeter / float(config.maximum_resampled_points),
    )
    count = max(4, min(config.maximum_resampled_points, int(math.ceil(perimeter / step))))
    step = perimeter / count
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    distances = np.arange(count, dtype=np.float64) * step
    indices = np.searchsorted(cumulative[1:], distances, side="right")
    fractions = (distances - cumulative[indices]) / np.maximum(lengths[indices], 1e-9)
    samples = ring[indices] + vectors[indices] * fractions[:, None]
    return samples, step


def _cyclic_median_smooth(points: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0:
        return points.copy()
    radius = min(radius, max(1, (len(points) - 1) // 4))
    neighbors = [np.roll(points, shift, axis=0) for shift in range(-radius, radius + 1)]
    return np.median(np.stack(neighbors, axis=0), axis=0)


def _infer_axes(
    points: np.ndarray,
    *,
    sample_step: float,
    config: RectilinearFitConfig,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    radius = max(1, int(round(config.axis_window_px / (2.0 * sample_step))))
    tangents = np.roll(points, -radius, axis=0) - np.roll(points, radius, axis=0)
    lengths = np.linalg.norm(tangents, axis=1)
    valid = lengths > sample_step * 0.35
    if not valid.any():
        zero = np.asarray([1.0, 0.0], dtype=np.float64)
        return zero, zero[::-1], 0.0, 0.0, 0.0, 0.0
    angles = np.arctan2(tangents[valid, 1], tangents[valid, 0])
    weights = lengths[valid]
    fourth = np.sum(weights * np.exp(4j * angles))
    total = float(weights.sum())
    concentration = float(abs(fourth) / max(total, 1e-9))
    axis_angle = 0.25 * float(np.angle(fourth))
    first = np.asarray([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float64)
    second = np.asarray([-first[1], first[0]], dtype=np.float64)
    unit = tangents[valid] / weights[:, None]
    first_projection = np.abs(unit @ first)
    second_projection = np.abs(unit @ second)
    best = np.maximum(first_projection, second_projection)
    tolerance_cosine = math.cos(math.radians(config.axis_alignment_tolerance_degrees))
    aligned = float(np.sum(weights[best >= tolerance_cosine]) / max(total, 1e-9))
    first_weight = float(np.sum(weights[first_projection >= second_projection]))
    second_weight = total - first_weight
    secondary_fraction = min(first_weight, second_weight) / max(total, 1e-9)
    return (
        first,
        second,
        math.degrees(axis_angle),
        concentration,
        aligned,
        secondary_fraction,
    )


def _segment_axis_runs(
    points_uv: np.ndarray,
    *,
    sample_step: float,
    config: RectilinearFitConfig,
) -> np.ndarray:
    radius = max(1, int(round(config.label_window_px / (2.0 * sample_step))))
    tangent = np.roll(points_uv, -radius, axis=0) - np.roll(points_uv, radius, axis=0)
    projection = np.abs(tangent)
    total = np.maximum(projection.sum(axis=1), sample_step * 0.1)
    # A mild confidence amplification makes a clearly axial short step worth
    # two state transitions while ambiguous corner samples remain cheap.
    emission = np.stack([projection[:, 1], projection[:, 0]], axis=1) / total[:, None]
    return _cyclic_two_state_viterbi(emission, config.label_switch_penalty)


def _cyclic_two_state_viterbi(emission: np.ndarray, switch_penalty: float) -> np.ndarray:
    count = len(emission)
    best_cost = math.inf
    best_labels: np.ndarray | None = None
    for start in (0, 1):
        previous = np.full(2, math.inf, dtype=np.float64)
        previous[start] = emission[0, start]
        back = np.zeros((count, 2), dtype=np.int8)
        for index in range(1, count):
            stay = previous
            switch = previous[::-1] + switch_penalty
            choose_switch = switch < stay
            current = emission[index] + np.where(choose_switch, switch, stay)
            back[index] = np.where(choose_switch, 1 - np.arange(2), np.arange(2))
            previous = current
        closure = previous + switch_penalty * (np.arange(2) != start)
        end = int(np.argmin(closure))
        cost = float(closure[end])
        if cost >= best_cost:
            continue
        labels = np.empty(count, dtype=np.int8)
        labels[-1] = end
        for index in range(count - 1, 0, -1):
            labels[index - 1] = back[index, labels[index]]
        best_cost = cost
        best_labels = labels
    assert best_labels is not None
    return best_labels


def _remove_weak_runs(
    labels: np.ndarray,
    points_uv: np.ndarray,
    *,
    config: RectilinearFitConfig,
) -> np.ndarray:
    labels = labels.copy()
    # Each merge removes at least one run, so this is strictly bounded.
    for _iteration in range(min(len(labels), config.maximum_runs * 4)):
        spans = _cyclic_run_spans(labels)
        if len(spans) <= 4:
            break
        weakest: tuple[float, np.ndarray, int] | None = None
        for label, indices in spans:
            edge_end = (int(indices[-1]) + 1) % len(points_uv)
            displacement = points_uv[edge_end] - points_uv[int(indices[0])]
            net = abs(float(displacement[label]))
            edges = np.roll(points_uv, -1, axis=0)[indices] - points_uv[indices]
            path = float(np.sum(np.abs(edges[:, label])))
            efficiency = net / max(path, 1e-9)
            if net >= config.minimum_run_progress_px and efficiency >= config.minimum_run_efficiency:
                continue
            weakness = net + config.minimum_run_progress_px * efficiency
            if weakest is None or weakness < weakest[0]:
                weakest = (weakness, indices, label)
        if weakest is None:
            break
        _weakness, indices, label = weakest
        labels[indices] = 1 - label
    return labels


def _cyclic_run_spans(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    count = len(labels)
    transitions = np.flatnonzero(labels != np.roll(labels, 1))
    if len(transitions) == 0:
        return [(int(labels[0]), np.arange(count, dtype=np.int32))]
    start = int(transitions[0])
    rotated = np.roll(labels, -start)
    boundaries = np.flatnonzero(rotated != np.roll(rotated, 1))[1:]
    boundaries = np.concatenate([[0], boundaries, [count]])
    runs: list[tuple[int, np.ndarray]] = []
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        indices = (np.arange(low, high, dtype=np.int32) + start) % count
        runs.append((int(labels[indices[0]]), indices))
    return runs


def _fit_constrained_run(
    points_uv: np.ndarray,
    labels: np.ndarray,
    span: tuple[int, np.ndarray],
) -> _Run:
    label, indices = span
    point_indices = np.concatenate([indices, [((int(indices[-1]) + 1) % len(points_uv))]])
    run_points = points_uv[point_indices]
    cross_axis = 1 - label
    cross = run_points[:, cross_axis]
    center = _robust_location(cross)
    residuals = np.abs(cross - center)
    displacement = run_points[-1] - run_points[0]
    edges = np.diff(run_points, axis=0)
    net = abs(float(displacement[label]))
    path = float(np.sum(np.abs(edges[:, label])))
    return _Run(
        label=label,
        indices=indices,
        line_coordinate=center,
        residuals=residuals,
        net_progress=net,
        path_progress=path,
    )


def _robust_location(values: np.ndarray) -> float:
    median = float(np.median(values))
    deviation = np.abs(values - median)
    mad = float(np.median(deviation))
    scale = max(0.35, 1.4826 * mad)
    cutoff = max(1.25, 2.75 * scale)
    inliers = values[deviation <= cutoff]
    if len(inliers) < max(2, len(values) // 3):
        return median
    # The inlier mean reduces zero-mean localization jitter while the initial
    # MAD gate prevents labels or map crossings from pulling the fitted line.
    return float(np.mean(inliers))


def _intersect_run_lines(runs: list[_Run]) -> np.ndarray:
    vertices: list[tuple[float, float]] = []
    for index, current in enumerate(runs):
        following = runs[(index + 1) % len(runs)]
        if current.label == 0:
            vertices.append((following.line_coordinate, current.line_coordinate))
        else:
            vertices.append((current.line_coordinate, following.line_coordinate))
    return np.asarray(vertices, dtype=np.float64)


def _remove_duplicate_vertices(vertices: np.ndarray) -> np.ndarray:
    if not len(vertices):
        return vertices
    keep = np.ones(len(vertices), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(vertices, axis=0), axis=1) > 1e-5
    result = vertices[keep]
    if len(result) > 1 and float(np.linalg.norm(result[-1] - result[0])) <= 1e-5:
        result = result[:-1]
    return result


def _resample_polygon_boundary(
    vertices: np.ndarray,
    *,
    step: float,
    maximum_points: int,
) -> np.ndarray:
    vectors = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(vectors, axis=1)
    perimeter = float(lengths.sum())
    count = max(4, min(maximum_points, int(math.ceil(perimeter / max(step, 1e-6)))))
    distances = np.arange(count, dtype=np.float64) * (perimeter / count)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    indices = np.searchsorted(cumulative[1:], distances, side="right")
    fractions = (distances - cumulative[indices]) / np.maximum(lengths[indices], 1e-9)
    return vertices[indices] + vectors[indices] * fractions[:, None]


def _point_to_ring_distances(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    starts = ring
    vectors = np.roll(ring, -1, axis=0) - ring
    squared_lengths = np.sum(vectors * vectors, axis=1)
    output = np.empty(len(points), dtype=np.float64)
    # Bound peak memory for long source-native rings.
    for low in range(0, len(points), 256):
        query = points[low : low + 256]
        relative = query[:, None, :] - starts[None, :, :]
        fraction = np.sum(relative * vectors[None, :, :], axis=2) / np.maximum(
            squared_lengths[None, :], 1e-12
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        nearest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
        distances = np.linalg.norm(query[:, None, :] - nearest, axis=2)
        output[low : low + len(query)] = np.min(distances, axis=1)
    return output


def _signed_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]))
