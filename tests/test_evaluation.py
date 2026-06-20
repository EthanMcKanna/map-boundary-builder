import math

import numpy as np
from shapely.geometry import GeometryCollection, Polygon

from map_boundary_builder.evaluation import (
    area_ratio,
    binary_mask_confusion_counts,
    boundary_iou,
    boundary_mask,
    centroid_distance_px,
    dice,
    geometry_validity_summary,
    iou,
    precision,
    recall,
)


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
