from math import cos, radians, sin

import numpy as np
from shapely.affinity import rotate
from shapely.geometry import shape
from shapely.ops import transform

from map_boundary_builder.catalog_match import (
    CATALOG_LABEL_HINT_MIN_IOU,
    catalog_feature_collection,
    load_catalog_entries,
    match_service_area_catalog,
)
from map_boundary_builder.extract import ExtractionResult


def test_catalog_shape_match_accepts_current_high_confidence_shape() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "phoenix-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)

    match = match_service_area_catalog(pixel_geometry, style="bright-blue")

    assert match is not None
    assert match.entry.slug == "phoenix-waymo"
    assert match.iou > 0.99


def test_catalog_shape_match_rejects_wrong_style() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "phoenix-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)

    assert match_service_area_catalog(pixel_geometry, style="gray-fill") is None


def test_ocr_derived_catalog_entry_preserves_original_confidence_cap() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "miami-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)

    match = match_service_area_catalog(pixel_geometry, style="bright-blue")

    assert match is not None
    assert match.entry.slug == "miami-waymo"
    assert match.confidence == entry.max_confidence
    assert match.confidence == 0.864


def test_catalog_shape_match_accepts_near_miss_after_small_rotation() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "houston-tesla")
    unrotated_reference = rotate(entry.mercator_geometry, 1.4, origin="centroid", use_radians=False)
    pixel_geometry = mercator_geometry_to_pixel(unrotated_reference)

    match = match_service_area_catalog(pixel_geometry, style="gray-fill")

    assert match is not None
    assert match.entry.slug == "houston-tesla"
    assert match.iou >= 0.97
    assert abs(match.rotation_degrees) > 0.0
    assert match.confidence == 0.853


def test_catalog_feature_collection_has_summary_properties() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "phoenix-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)
    match = match_service_area_catalog(pixel_geometry, style="bright-blue")
    assert match is not None
    extraction = ExtractionResult(
        mask=np.zeros((100, 100), dtype=np.uint8),
        style="bright-blue",
        pixel_geometry=pixel_geometry,
        coverage_ratio=0.25,
        contour_count=1,
        confidence=0.98,
    )

    data = catalog_feature_collection(
        extraction,
        match,
        width=1000,
        height=1000,
        image_path="input.png",
        city_input="Auto",
    )
    properties = data["features"][0]["properties"]

    assert properties["city"] == "Phoenix"
    assert properties["georeference_source"] == "catalog-shape-match"
    assert properties["catalog_slug"] == "phoenix-waymo"
    assert properties["combined_confidence"] == 0.98
    assert data["features"][0]["geometry"]["type"] == "Polygon"


def test_current_verified_catalog_entry_outputs_exact_geometry_after_match() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "miami-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)
    match = match_service_area_catalog(pixel_geometry, style="bright-blue")
    assert match is not None
    extraction = ExtractionResult(
        mask=np.zeros((100, 100), dtype=np.uint8),
        style="bright-blue",
        pixel_geometry=pixel_geometry,
        coverage_ratio=0.25,
        contour_count=1,
        confidence=0.98,
    )

    data = catalog_feature_collection(
        extraction,
        match,
        width=1000,
        height=1000,
        image_path="input.png",
        city_input="Auto",
    )

    output_geometry = shape(data["features"][0]["geometry"])
    assert output_geometry.equals_exact(entry.geometry, tolerance=1e-7)


def test_current_verified_catalog_entry_can_declare_tight_min_shape_iou() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "bay-area-waymo")
    near_miss_reference = rotate(entry.mercator_geometry, 1.0, origin="centroid", use_radians=False)
    pixel_geometry = mercator_geometry_to_pixel(near_miss_reference)

    match = match_service_area_catalog(pixel_geometry, style="bright-blue")

    assert entry.min_iou == 0.965
    assert match is not None
    assert match.entry.slug == "bay-area-waymo"
    assert 0.965 <= match.iou < 0.97
    assert match.confidence == 0.877


def test_current_verified_catalog_entry_uses_exact_ordered_contour_fit() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "las-vegas-zoox")
    pixel_geometry = mercator_geometry_to_similarity_pixel(entry.mercator_geometry, rotation_degrees=10.0)

    match = match_service_area_catalog(pixel_geometry, style="light-fill")

    assert match is not None
    assert match.entry.slug == "las-vegas-zoox"
    assert match.iou >= 0.985
    assert abs(match.rotation_degrees) > 2.0
    assert match.confidence == 0.767


def test_label_hint_accepts_sparse_low_resolution_shape_match() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "nashville-waymo")
    simplified_reference = entry.mercator_geometry.simplify(500, preserve_topology=True)
    pixel_geometry = mercator_geometry_to_pixel(simplified_reference)

    strict_match = match_service_area_catalog(pixel_geometry, style="bright-blue")
    hinted_match = match_service_area_catalog(
        pixel_geometry,
        style="bright-blue",
        min_iou=CATALOG_LABEL_HINT_MIN_IOU,
        area_hint_texts=["Nashville"],
    )

    assert strict_match is None
    assert hinted_match is not None
    assert hinted_match.entry.slug == "nashville-waymo"
    assert CATALOG_LABEL_HINT_MIN_IOU <= hinted_match.iou < 0.97


def test_label_hint_rejects_wrong_area_match() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "nashville-waymo")
    simplified_reference = entry.mercator_geometry.simplify(500, preserve_topology=True)
    pixel_geometry = mercator_geometry_to_pixel(simplified_reference)

    match = match_service_area_catalog(
        pixel_geometry,
        style="bright-blue",
        min_iou=CATALOG_LABEL_HINT_MIN_IOU,
        area_hint_texts=["Phoenix"],
    )

    assert match is None


def mercator_geometry_to_pixel(geometry):
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max_x - min_x
    height = max_y - min_y

    def to_pixel(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        return (x - min_x) / width * 1000.0, (max_y - y) / height * 1000.0

    return transform(to_pixel, geometry)


def mercator_geometry_to_similarity_pixel(geometry, *, rotation_degrees: float):
    center_x, center_y = geometry.centroid.coords[0]
    angle = radians(rotation_degrees)
    cos_r = cos(angle)
    sin_r = sin(angle)

    def to_pixel(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        dx = (x - center_x) / 1000.0
        dy = (y - center_y) / 1000.0
        pixel_x = dx * cos_r + dy * sin_r + 500.0
        pixel_y = -(-dx * sin_r + dy * cos_r) + 500.0
        return pixel_x, pixel_y

    return transform(to_pixel, geometry)
