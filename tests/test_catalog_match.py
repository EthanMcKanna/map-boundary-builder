import numpy as np
from shapely.ops import transform

from map_boundary_builder.catalog_match import (
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


def mercator_geometry_to_pixel(geometry):
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max_x - min_x
    height = max_y - min_y

    def to_pixel(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        return (x - min_x) / width * 1000.0, (max_y - y) / height * 1000.0

    return transform(to_pixel, geometry)
