import math

import numpy as np
from shapely.geometry import GeometryCollection, Polygon

from map_boundary_builder.evaluation import (
    area_ratio,
    binary_mask_confusion_counts,
    boundary_distance_summary_px,
    boundary_f1,
    boundary_iou,
    boundary_distance_percentile_px,
    boundary_mask,
    centroid_distance_px,
    dice,
    geometry_validity_summary,
    geometry_corner_f1,
    geometry_complexity_comparison,
    geometry_complexity_summary,
    geometry_straight_run_rms_px,
    geometry_topology_signature,
    iou,
    precision,
    recall,
    corner_f1,
    rasterize_geometry_mask,
    topology_signature,
)


def test_geometry_corner_f1_scores_source_vector_vertices() -> None:
    reference = Polygon([(4, 4), (24, 4), (24, 20), (4, 20)])
    close = Polygon([(5, 4), (25, 4), (25, 20), (5, 20)])
    far = Polygon([(12, 4), (32, 4), (32, 20), (12, 20)])

    assert geometry_corner_f1(close, reference, tolerance_px=2.0)["f1"] == 1.0
    assert geometry_corner_f1(far, reference, tolerance_px=2.0)["f1"] < 1.0


def test_binary_mask_confusion_counts_treat_nonzero_values_as_true() -> None:
    reference = np.array(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 0],
        ]
    )
    predicted = np.array(
        [
            [0, 1, 0],
            [3, 0, 2],
            [0, 0, 0],
        ]
    )

    assert binary_mask_confusion_counts(predicted, reference) == {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "tn": 5,
    }


def test_mask_overlap_metrics_are_computed_from_confusion_counts() -> None:
    reference = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    )
    predicted = np.array(
        [
            [1, 0, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )

    assert iou(predicted, reference) == 0.5
    assert dice(predicted, reference) == 2 / 3
    assert precision(predicted, reference) == 2 / 3
    assert recall(predicted, reference) == 2 / 3
    assert area_ratio(predicted, reference) == 1.0


def test_empty_masks_have_stable_metric_values() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    nonempty = empty.copy()
    nonempty[1, 1] = True

    assert iou(empty, empty) == 1.0
    assert dice(empty, empty) == 1.0
    assert precision(empty, empty) == 1.0
    assert recall(empty, empty) == 1.0
    assert area_ratio(empty, empty) == 1.0
    assert centroid_distance_px(empty, empty) == 0.0

    assert iou(nonempty, empty) == 0.0
    assert dice(nonempty, empty) == 0.0
    assert precision(nonempty, empty) == 0.0
    assert recall(nonempty, empty) == 0.0
    assert math.isinf(area_ratio(nonempty, empty))
    assert math.isinf(centroid_distance_px(nonempty, empty))


def test_centroid_distance_uses_pixel_coordinates() -> None:
    reference = np.zeros((8, 8), dtype=bool)
    predicted = np.zeros((8, 8), dtype=bool)
    reference[1:3, 1:3] = True
    predicted[4:6, 5:7] = True

    assert centroid_distance_px(predicted, reference) == 5.0


def test_boundary_mask_extracts_one_pixel_interior_edge() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 1:6] = True

    boundary = boundary_mask(mask)

    assert boundary.dtype == bool
    assert int(boundary.sum()) == 12
    assert not bool(boundary[3, 3])
    assert bool(boundary[2, 1])
    assert bool(boundary[4, 5])


def test_boundary_mask_marks_image_border_as_boundary() -> None:
    mask = np.ones((4, 4), dtype=bool)

    boundary = boundary_mask(mask)

    assert int(boundary.sum()) == 12
    assert not boundary[1:3, 1:3].any()


def test_boundary_iou_honors_tolerance_pixels() -> None:
    reference = np.zeros((12, 12), dtype=bool)
    predicted = np.zeros((12, 12), dtype=bool)
    reference[3:9, 3:9] = True
    predicted[3:9, 4:10] = True

    assert boundary_iou(predicted, reference, tolerance_px=0) < 1.0
    assert boundary_iou(predicted, reference, tolerance_px=1) > boundary_iou(
        predicted,
        reference,
        tolerance_px=0,
    )
    assert boundary_iou(reference, reference, tolerance_px=2) == 1.0


def test_boundary_iou_rejects_negative_tolerance() -> None:
    mask = np.zeros((3, 3), dtype=bool)

    try:
        boundary_iou(mask, mask, tolerance_px=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative boundary tolerance to raise ValueError")


def test_boundary_distance_reports_source_pixel_offset() -> None:
    reference = np.zeros((64, 64), dtype=bool)
    predicted = np.zeros((64, 64), dtype=bool)
    reference[12:52, 12:52] = True
    predicted[12:52, 14:54] = True

    assert boundary_distance_percentile_px(reference, reference) == 0.0
    assert 1.9 <= boundary_distance_percentile_px(predicted, reference) <= 2.1


def test_boundary_f1_scores_original_edges_at_one_pixel() -> None:
    reference = np.zeros((32, 32), dtype=bool)
    predicted = np.zeros((32, 32), dtype=bool)
    reference[8:24, 8:24] = True
    predicted[8:24, 9:25] = True

    assert boundary_f1(predicted, reference, tolerance_px=0)["f1"] == 0.5
    assert boundary_f1(predicted, reference, tolerance_px=1) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_boundary_distance_summary_includes_tail_and_maximum() -> None:
    reference = np.zeros((32, 32), dtype=bool)
    predicted = np.zeros((32, 32), dtype=bool)
    reference[8:24, 8:24] = True
    predicted[8:24, 9:25] = True

    summary = boundary_distance_summary_px(predicted, reference)

    assert summary["mean_px"] == 0.5
    assert summary["p95_px"] == 1.0
    assert summary["p99_px"] == 1.0
    assert summary["max_px"] == 1.0


def test_geometry_corner_metric_reports_angular_error() -> None:
    reference = Polygon([(4, 4), (24, 4), (24, 20), (4, 20)])
    slanted = Polygon([(4, 4), (24, 5), (23, 20), (4, 20)])

    score = geometry_corner_f1(slanted, reference, tolerance_px=2.0)

    assert score["matched_count"] == 4
    assert score["f1"] == 1.0
    assert 1.0 < score["mean_angular_error_degrees"] < 3.0
    assert score["max_angular_error_degrees"] > score["mean_angular_error_degrees"]


def test_geometry_straightness_and_complexity_expose_wobble() -> None:
    exact = Polygon([(4, 4), (24, 4), (24, 20), (4, 20)])
    rough = Polygon(
        [(4, 4), (8, 5), (12, 3), (16, 5), (20, 3), (24, 4), (24, 20), (4, 20)]
    )

    assert geometry_straight_run_rms_px(exact) == 0.0
    assert geometry_straight_run_rms_px(rough) > 0.35
    assert geometry_complexity_summary(exact)["vertex_count"] == 4
    assert geometry_complexity_summary(rough)["vertex_count"] == 8


def test_geometry_complexity_is_normalized_to_reference_floor() -> None:
    reference = Polygon(
        [(4, 4), (8, 5), (12, 3), (16, 5), (20, 3), (24, 4), (24, 20), (4, 20)]
    )
    rougher = Polygon(
        [
            (4, 4),
            (7, 6),
            (10, 2),
            (13, 6),
            (16, 2),
            (19, 6),
            (22, 2),
            (24, 4),
            (24, 20),
            (4, 20),
        ]
    )

    oracle = geometry_complexity_comparison(reference, reference)
    degraded = geometry_complexity_comparison(rougher, reference)

    assert oracle["predicted_straight_run_rms_px"] > 0.35
    assert oracle["straight_run_rms_excess_px"] == 0.0
    assert oracle["canonical_complexity_ratio"] == 1.0
    assert degraded["straight_run_rms_excess_px"] > 0.0
    assert degraded["canonical_complexity_ratio"] > 1.0


def test_evaluation_rasterizer_matches_production_rounding_and_holes() -> None:
    from map_boundary_builder.edgegraph import rasterize_geometry_mask as production_rasterize

    polygon = Polygon(
        [(-1.2, 2.4), (22.6, 1.5), (25.8, 20.6), (1.4, 22.7)],
        [[(7.4, 7.6), (15.5, 7.4), (15.6, 15.7), (7.5, 15.4)]],
    )

    expected = production_rasterize(polygon, width=24, height=24)
    actual = rasterize_geometry_mask(polygon, width=24, height=24)

    assert np.array_equal(actual, expected)
    assert actual.dtype == bool
    assert actual[:, 0].any()
    assert not actual[10, 10]


def test_corner_f1_and_topology_capture_vector_fidelity() -> None:
    reference = np.zeros((80, 80), dtype=bool)
    reference[10:70, 10:70] = True
    reference[30:50, 30:50] = False
    shifted = np.zeros_like(reference)
    shifted[11:71, 11:71] = True
    shifted[31:51, 31:51] = False

    score = corner_f1(shifted, reference, tolerance_px=2.0)

    assert score["f1"] == 1.0
    assert topology_signature(reference) == {"components": 1, "holes": 1}
    assert topology_signature(shifted) == topology_signature(reference)


def test_geometry_topology_signature_counts_polygon_components_and_holes() -> None:
    polygon = Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        [[(4, 4), (8, 4), (8, 8), (4, 8)]],
    )
    second = Polygon([(30, 0), (35, 0), (35, 5), (30, 5)])

    assert geometry_topology_signature(polygon) == {"components": 1, "holes": 1}
    assert geometry_topology_signature(polygon.union(second)) == {
        "components": 2,
        "holes": 1,
    }


def test_shape_mismatch_raises_clear_value_error() -> None:
    try:
        iou(np.zeros((2, 3)), np.zeros((3, 2)))
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched masks to raise ValueError")


def test_geometry_validity_summary_reports_valid_polygon() -> None:
    polygon = Polygon([(0, 0), (2, 0), (2, 3), (0, 3)])

    summary = geometry_validity_summary(polygon)

    assert summary["is_present"] is True
    assert summary["geometry_type"] == "Polygon"
    assert summary["is_empty"] is False
    assert summary["is_valid"] is True
    assert summary["validity_reason"] == "Valid Geometry"
    assert summary["area"] == 6.0
    assert summary["bounds"] == (0.0, 0.0, 2.0, 3.0)


def test_geometry_validity_summary_reports_invalid_and_empty_geometries() -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    empty = GeometryCollection()

    invalid_summary = geometry_validity_summary(bowtie)
    empty_summary = geometry_validity_summary(empty)
    missing_summary = geometry_validity_summary(None)

    assert invalid_summary["is_valid"] is False
    assert "Self-intersection" in invalid_summary["validity_reason"]
    assert empty_summary["is_present"] is True
    assert empty_summary["is_empty"] is True
    assert empty_summary["bounds"] is None
    assert missing_summary["is_present"] is False
    assert missing_summary["validity_reason"] == "missing geometry"
