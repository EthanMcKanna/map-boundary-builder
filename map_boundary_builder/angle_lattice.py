"""Fail-closed angle-lattice reconstruction for sharp polygonal boundaries.

The learned EdgeGraph localizer is deliberately free to place every vertex at
subpixel precision.  For authored service-area maps, however, long neighboring
segments commonly share a small direction vocabulary.  Independent endpoint
noise then appears as rough curves even when the raster mask is accurate.

This module learns a global orientation phase modulo a candidate spacing,
snaps only sufficiently long segments, reconstructs corners from adjacent line
intersections, and refuses candidates whose vertices move too far.  Raster,
topology, and source-image support gates remain the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


@dataclass(frozen=True)
class AngleLatticeResult:
    geometry: Polygon | MultiPolygon | None
    spacing_degrees: float
    phase_degrees: float
    p90_residual_degrees: float
    mean_residual_degrees: float
    maximum_vertex_displacement_px: float
    snapped_segment_count: int
    inferred_connector_count: int
    reason: str


def snap_geometry_to_angle_lattice(
    geometry: Polygon | MultiPolygon,
    *,
    spacing_degrees: float,
    simplification_tolerance_px: float = 0.75,
    minimum_segment_length_px: float = 8.0,
    maximum_segment_residual_degrees: float = 5.0,
    maximum_vertex_displacement_px: float = 2.0,
) -> AngleLatticeResult:
    """Return a bounded lattice candidate or a diagnostic rejection.

    Short segments normally retain their observed direction.  The one
    exception is a short connector bracketed by parallel long segments: when
    it is already close to perpendicular, the two long runs determine the
    connector direction more reliably than two noisy raster endpoints.
    """

    if spacing_degrees <= 0.0 or spacing_degrees > 90.0:
        raise ValueError("spacing_degrees must be in (0, 90]")
    if simplification_tolerance_px < 0.0:
        raise ValueError("simplification_tolerance_px must be non-negative")
    if minimum_segment_length_px <= 0.0:
        raise ValueError("minimum_segment_length_px must be positive")
    if maximum_segment_residual_degrees <= 0.0:
        raise ValueError("maximum_segment_residual_degrees must be positive")
    if maximum_vertex_displacement_px <= 0.0:
        raise ValueError("maximum_vertex_displacement_px must be positive")
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return _rejected(spacing_degrees, "unsupported-geometry")
    if geometry.is_empty or not geometry.is_valid:
        return _rejected(spacing_degrees, "invalid-geometry")

    simplified = geometry.simplify(
        simplification_tolerance_px,
        preserve_topology=True,
    )
    if (
        not isinstance(simplified, (Polygon, MultiPolygon))
        or simplified.is_empty
        or not simplified.is_valid
    ):
        return _rejected(spacing_degrees, "invalid-simplified-geometry")
    polygons = (
        [simplified]
        if isinstance(simplified, Polygon)
        else list(simplified.geoms)
    )
    rings: list[np.ndarray] = []
    for polygon in polygons:
        rings.append(_ring_points(polygon.exterior.coords))
        rings.extend(_ring_points(interior.coords) for interior in polygon.interiors)

    angles: list[float] = []
    lengths: list[float] = []
    for points in rings:
        vectors = np.roll(points, -1, axis=0) - points
        segment_lengths = np.linalg.norm(vectors, axis=1)
        long = segment_lengths >= minimum_segment_length_px
        angles.extend(
            (
                np.degrees(
                    np.arctan2(vectors[long, 1], vectors[long, 0])
                )
                % 180.0
            ).tolist()
        )
        lengths.extend(segment_lengths[long].tolist())
    if len(angles) < 3:
        return _rejected(spacing_degrees, "insufficient-long-segments")

    angle_values = np.asarray(angles, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    phase, residuals = _best_phase(
        angle_values,
        length_values,
        spacing_degrees,
    )
    p90_residual = _weighted_quantile(
        residuals,
        length_values,
        0.90,
    )
    mean_residual = float(
        np.sum(residuals * length_values)
        / max(1e-9, float(length_values.sum()))
    )

    reconstructed: list[np.ndarray] = []
    displacements: list[float] = []
    snapped_segment_count = 0
    inferred_connector_count = 0
    for points in rings:
        vectors = np.roll(points, -1, axis=0) - points
        segment_lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.maximum(segment_lengths[:, None], 1e-9)
        segment_angles = np.degrees(
            np.arctan2(directions[:, 1], directions[:, 0])
        )
        snapped_angles = phase + np.round(
            (segment_angles - phase) / spacing_degrees
        ) * spacing_degrees
        segment_residuals = np.abs(
            _angle_delta(segment_angles, snapped_angles)
        )
        snap = (
            (segment_lengths >= minimum_segment_length_px)
            & (
                segment_residuals
                <= maximum_segment_residual_degrees
            )
        )
        snapped_segment_count += int(np.count_nonzero(snap))

        count = len(points)
        for index in range(count):
            previous = (index - 1) % count
            following = (index + 1) % count
            if snap[index]:
                continue
            if not snap[previous] or not snap[following]:
                continue
            previous_angle = snapped_angles[previous]
            following_angle = snapped_angles[following]
            parallel = (
                abs(_angle_delta(previous_angle, following_angle)) <= 2.0
            )
            perpendicular = (
                abs(
                    abs(
                        _angle_delta(
                            segment_angles[index],
                            previous_angle,
                        )
                    )
                    - 90.0
                )
                <= 12.0
            )
            if parallel and perpendicular:
                snapped_angles[index] = previous_angle + 90.0
                snap[index] = True
                inferred_connector_count += 1

        models: list[tuple[np.ndarray, np.ndarray]] = []
        for index, (point, vector) in enumerate(
            zip(points, vectors, strict=True)
        ):
            center = point + vector * 0.5
            if snap[index]:
                radians = math.radians(float(snapped_angles[index]))
                direction = np.asarray(
                    [math.cos(radians), math.sin(radians)],
                    dtype=np.float64,
                )
            else:
                direction = directions[index]
            models.append((center, direction))

        output: list[np.ndarray] = []
        for index, point in enumerate(points):
            joined = _line_intersection(
                *models[index - 1],
                *models[index],
            )
            if joined is None:
                joined = point
            displacement = float(np.linalg.norm(joined - point))
            displacements.append(displacement)
            if displacement > maximum_vertex_displacement_px:
                return AngleLatticeResult(
                    geometry=None,
                    spacing_degrees=spacing_degrees,
                    phase_degrees=phase,
                    p90_residual_degrees=p90_residual,
                    mean_residual_degrees=mean_residual,
                    maximum_vertex_displacement_px=max(displacements),
                    snapped_segment_count=snapped_segment_count,
                    inferred_connector_count=inferred_connector_count,
                    reason="vertex-displacement-gate",
                )
            output.append(joined)
        reconstructed.append(np.asarray(output, dtype=np.float64))

    rebuilt: list[Polygon] = []
    cursor = 0
    for polygon in polygons:
        exterior = reconstructed[cursor]
        cursor += 1
        holes: list[np.ndarray] = []
        for _interior in polygon.interiors:
            holes.append(reconstructed[cursor])
            cursor += 1
        rebuilt.append(Polygon(exterior, holes))
    candidate: Polygon | MultiPolygon = (
        rebuilt[0]
        if len(rebuilt) == 1
        else MultiPolygon(rebuilt)
    )
    if candidate.is_empty or not candidate.is_valid:
        return AngleLatticeResult(
            geometry=None,
            spacing_degrees=spacing_degrees,
            phase_degrees=phase,
            p90_residual_degrees=p90_residual,
            mean_residual_degrees=mean_residual,
            maximum_vertex_displacement_px=max(
                displacements,
                default=0.0,
            ),
            snapped_segment_count=snapped_segment_count,
            inferred_connector_count=inferred_connector_count,
            reason="invalid-candidate",
        )
    return AngleLatticeResult(
        geometry=candidate,
        spacing_degrees=spacing_degrees,
        phase_degrees=phase,
        p90_residual_degrees=p90_residual,
        mean_residual_degrees=mean_residual,
        maximum_vertex_displacement_px=max(displacements, default=0.0),
        snapped_segment_count=snapped_segment_count,
        inferred_connector_count=inferred_connector_count,
        reason="candidate",
    )


def _rejected(spacing_degrees: float, reason: str) -> AngleLatticeResult:
    return AngleLatticeResult(
        geometry=None,
        spacing_degrees=spacing_degrees,
        phase_degrees=0.0,
        p90_residual_degrees=math.inf,
        mean_residual_degrees=math.inf,
        maximum_vertex_displacement_px=math.inf,
        snapped_segment_count=0,
        inferred_connector_count=0,
        reason=reason,
    )


def _ring_points(coordinates: object) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return points


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    index = int(
        np.searchsorted(
            cumulative,
            quantile * float(ordered_weights.sum()),
        )
    )
    return float(ordered_values[min(len(ordered_values) - 1, index)])


def _angle_delta(angle: np.ndarray | float, lattice: np.ndarray | float):
    return (angle - lattice + 90.0) % 180.0 - 90.0


def _best_phase(
    angles: np.ndarray,
    lengths: np.ndarray,
    spacing_degrees: float,
) -> tuple[float, np.ndarray]:
    # A one-hundredth-degree grid is deterministic and negligible compared
    # with the raster/line-fitting work that precedes this post-process.
    count = max(1, int(round(spacing_degrees * 100.0)))
    phases = np.linspace(
        0.0,
        spacing_degrees,
        count + 1,
        endpoint=False,
    )
    best_loss = math.inf
    best_phase = 0.0
    best_residual = np.full_like(angles, math.inf)
    for phase in phases:
        snapped = phase + np.round(
            (angles - phase) / spacing_degrees
        ) * spacing_degrees
        residual = np.abs(_angle_delta(angles, snapped))
        loss = float(
            np.sum(lengths * np.minimum(residual, 8.0) ** 2)
            / max(1e-9, float(lengths.sum()))
        )
        if loss < best_loss:
            best_loss = loss
            best_phase = float(phase)
            best_residual = residual
    return best_phase, best_residual


def _line_intersection(
    first_center: np.ndarray,
    first_direction: np.ndarray,
    second_center: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray | None:
    matrix = np.stack([first_direction, -second_direction], axis=1)
    if abs(float(np.linalg.det(matrix))) < 1e-5:
        return None
    scale = np.linalg.solve(
        matrix,
        second_center - first_center,
    )
    return first_center + first_direction * float(scale[0])


__all__ = [
    "AngleLatticeResult",
    "snap_geometry_to_angle_lattice",
]
