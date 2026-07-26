import math

import numpy as np
from shapely.geometry import Polygon

from map_boundary_builder.rectilinear_fit import fit_rectilinear_ring


def _dense_ring(vertices: np.ndarray, *, step: float = 1.0) -> np.ndarray:
    output: list[np.ndarray] = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        count = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / step)))
        output.extend(start + (end - start) * (index / count) for index in range(count))
    return np.asarray(output, dtype=np.float64)


def _noisy_rectangle() -> tuple[np.ndarray, Polygon]:
    vertices = np.asarray(
        [(10.0, 10.0), (210.0, 10.0), (210.0, 130.0), (10.0, 130.0)]
    )
    rng = np.random.default_rng(404)
    noisy: list[np.ndarray] = []
    for edge_index, (start, end) in enumerate(
        zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    ):
        length = float(np.linalg.norm(end - start))
        count = int(math.ceil(length))
        direction = (end - start) / length
        normal = np.asarray([-direction[1], direction[0]])
        for index in range(count):
            fraction = index / count
            # Real localizer error is predominantly normal to the boundary.
            # Fade it at exact corners so the test ring remains simple.
            envelope = min(1.0, 10.0 * fraction, 10.0 * (1.0 - fraction))
            jitter = envelope * (
                1.7 * math.sin(index * 0.45 + edge_index)
                + float(rng.normal(0.0, 0.22))
            )
            noisy.append(start + (end - start) * fraction + normal * jitter)
    return np.asarray(noisy), Polygon(vertices)


def test_exact_rectangle_becomes_four_sharp_corners() -> None:
    expected = Polygon([(20, 15), (180, 15), (180, 95), (20, 95)])

    result = fit_rectilinear_ring(_dense_ring(np.asarray(expected.exterior.coords[:-1])))

    assert result.candidate is not None
    assert result.diagnostics.accepted
    assert result.diagnostics.reason == "accepted"
    assert result.diagnostics.run_count == 4
    assert result.diagnostics.vertex_count == 4
    assert result.candidate.hausdorff_distance(expected) < 0.05
    assert result.diagnostics.support_p95_px < 0.05


def test_noisy_rectangle_fits_robust_constrained_lines() -> None:
    noisy, expected = _noisy_rectangle()
    assert Polygon(noisy).is_valid

    result = fit_rectilinear_ring(noisy)

    assert result.candidate is not None, result.diagnostics
    assert result.diagnostics.axis_concentration > 0.75
    assert result.diagnostics.aligned_fraction > 0.85
    assert result.diagnostics.vertex_count == 4
    assert result.candidate.hausdorff_distance(expected) < 0.35
    assert result.diagnostics.line_residual_p95_px < 2.0


def test_short_parallel_step_is_preserved() -> None:
    # The three-pixel vertical connectors are genuine topology, not rounded
    # corner noise, and must survive the cyclic denoising penalty.
    vertices = np.asarray(
        [
            (0.0, 0.0),
            (120.0, 0.0),
            (120.0, 50.0),
            (72.0, 50.0),
            (72.0, 53.0),
            (45.0, 53.0),
            (45.0, 50.0),
            (0.0, 50.0),
        ]
    )
    expected = Polygon(vertices)

    result = fit_rectilinear_ring(_dense_ring(vertices, step=0.8))

    assert result.candidate is not None, result.diagnostics
    assert result.diagnostics.run_count == 8
    assert result.diagnostics.vertex_count == 8
    assert result.candidate.hausdorff_distance(expected) < 0.15
    top_levels = sorted({round(y, 1) for _x, y in result.candidate.exterior.coords})
    assert top_levels == [0.0, 50.0, 53.0]


def test_rotated_rectangle_infers_rotated_orthogonal_axes() -> None:
    vertices = np.asarray([(0.0, 0.0), (150.0, 0.0), (150.0, 65.0), (0.0, 65.0)])
    angle = math.radians(27.0)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rotated = vertices @ rotation.T + np.asarray([80.0, 45.0])
    expected = Polygon(rotated)

    result = fit_rectilinear_ring(_dense_ring(rotated))

    assert result.candidate is not None, result.diagnostics
    assert result.diagnostics.vertex_count == 4
    assert abs(result.diagnostics.axis_angle_degrees - 27.0) < 0.5
    assert result.candidate.hausdorff_distance(expected) < 0.08


def test_non_orthogonal_angular_ring_fails_closed() -> None:
    angular = np.asarray(
        [(0.0, 0.0), (100.0, 0.0), (128.0, 38.0), (74.0, 94.0), (0.0, 70.0)]
    )

    result = fit_rectilinear_ring(_dense_ring(angular))

    assert result.candidate is None
    assert not result.diagnostics.accepted
    assert result.diagnostics.reason in {
        "weak-axis-concentration",
        "insufficient-axis-alignment",
        "overlap-gate",
        "support-p95-gate",
    }


def test_radial_ring_fails_closed() -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, 480, endpoint=False)
    circle = np.stack(
        [160.0 + 90.0 * np.cos(angles), 120.0 + 90.0 * np.sin(angles)], axis=1
    )

    result = fit_rectilinear_ring(circle)

    assert result.candidate is None
    assert result.diagnostics.reason == "weak-axis-concentration"
    assert result.diagnostics.axis_concentration < 0.05


def test_self_intersecting_source_is_rejected_without_repair() -> None:
    bow_tie = np.asarray([(0.0, 0.0), (100.0, 80.0), (0.0, 80.0), (100.0, 0.0)])

    result = fit_rectilinear_ring(bow_tie)

    assert result.candidate is None
    assert result.diagnostics.reason == "invalid-source-geometry"
    assert result.diagnostics.source_repaired is False


def _rectangle_with_touching_loop(
    *,
    width: float = 800.0,
    height: float = 500.0,
    half_width: float = 40.0,
    half_height: float = 3.0,
) -> np.ndarray:
    anchor = 280.0 if width >= 600.0 else 80.0
    vertices = np.asarray(
        [
            (0.0, 0.0),
            (anchor, 0.0),
            (anchor + half_width, half_height),
            (anchor + 2.0 * half_width, 0.0),
            (anchor + half_width, -half_height),
            (anchor + 2.0, 2.0),
            (width, 0.0),
            (width, height),
            (0.0, height),
        ]
    )
    return _dense_ring(vertices)


def _rectangle_with_two_touching_loops(*, height: float = 500.0) -> np.ndarray:
    vertices = np.asarray(
        [
            (0.0, 0.0),
            (180.0, 0.0),
            (220.0, 3.0),
            (260.0, 0.0),
            (220.0, -3.0),
            (182.0, 2.0),
            (480.0, 0.0),
            (520.0, 3.0),
            (560.0, 0.0),
            (520.0, -3.0),
            (482.0, 2.0),
            (1000.0, 0.0),
            (1000.0, height),
            (0.0, height),
        ]
    )
    return _dense_ring(vertices)


def test_one_tiny_touching_loop_is_repaired_before_rectilinear_fit() -> None:
    result = fit_rectilinear_ring(_rectangle_with_touching_loop())

    assert result.candidate is not None, result.diagnostics
    assert result.diagnostics.accepted
    assert result.diagnostics.reason == "accepted-tiny-loop-repair"
    assert result.diagnostics.source_repaired
    assert 180.0 < result.diagnostics.discarded_tiny_loop_area_px < 200.0
    assert result.diagnostics.discarded_tiny_loop_area_fraction < 0.001
    assert result.diagnostics.tiny_loop_maximum_deviation_px < 6.0
    assert result.diagnostics.support_maximum_px < 4.0
    assert result.diagnostics.vertex_count == 4


def test_multiple_tiny_touching_loops_are_repaired_as_one_bounded_artifact() -> None:
    result = fit_rectilinear_ring(_rectangle_with_two_touching_loops())

    assert result.candidate is not None, result.diagnostics
    assert result.diagnostics.accepted
    assert result.diagnostics.reason == "accepted-tiny-loop-repair"
    assert result.diagnostics.source_repaired
    assert 370.0 < result.diagnostics.discarded_tiny_loop_area_px < 380.0
    assert result.diagnostics.discarded_tiny_loop_area_fraction < 0.001
    assert result.diagnostics.tiny_loop_maximum_deviation_px < 6.0
    assert result.diagnostics.support_maximum_px < 5.0
    assert result.diagnostics.vertex_count == 4


def test_multiple_tiny_loops_must_be_negligible_in_aggregate() -> None:
    result = fit_rectilinear_ring(
        _rectangle_with_two_touching_loops(height=200.0)
    )

    assert result.candidate is None
    assert result.diagnostics.reason == "invalid-source-geometry"
    assert result.diagnostics.source_repaired is False


def test_small_absolute_loop_with_material_area_fraction_is_rejected() -> None:
    result = fit_rectilinear_ring(
        _rectangle_with_touching_loop(
            width=200.0,
            height=100.0,
            half_width=10.0,
        )
    )

    assert result.candidate is None
    assert result.diagnostics.reason == "invalid-source-geometry"
    assert result.diagnostics.source_repaired is False


def test_low_area_loop_with_large_spatial_excursion_is_rejected() -> None:
    result = fit_rectilinear_ring(
        _rectangle_with_touching_loop(half_width=3.0, half_height=20.0)
    )

    assert result.candidate is None
    assert result.diagnostics.reason == "invalid-source-geometry"
    assert result.diagnostics.source_repaired is False


def test_material_touching_loop_is_rejected_even_on_large_polygon() -> None:
    result = fit_rectilinear_ring(
        _rectangle_with_touching_loop(half_width=60.0, half_height=3.0)
    )

    assert result.candidate is None
    assert result.diagnostics.reason == "invalid-source-geometry"
    assert result.diagnostics.source_repaired is False


def test_hole_producing_self_intersection_is_not_treated_as_a_tiny_loop() -> None:
    vertices = np.asarray(
        [
            (0.0, 0.0),
            (280.0, 0.0),
            (300.0, 10.0),
            (280.0, 10.0),
            (300.0, 0.0),
            (600.0, 0.0),
            (600.0, 400.0),
            (0.0, 400.0),
        ]
    )

    result = fit_rectilinear_ring(_dense_ring(vertices))

    assert result.candidate is None
    assert result.diagnostics.reason == "invalid-source-geometry"
    assert result.diagnostics.source_repaired is False
