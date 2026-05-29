import json
from math import cos, radians, sin
from pathlib import Path

import numpy as np
from shapely.affinity import rotate
from shapely.geometry import shape
from shapely.ops import transform

from map_boundary_builder.catalog_match import (
    CATALOG_LABEL_HINT_MIN_IOU,
    catalog_area_matches_text,
    catalog_feature_collection,
    has_active_catalog_area_hint,
    has_stale_catalog_area_hint,
    load_catalog_entries,
    match_service_area_catalog,
)
from map_boundary_builder.extract import ExtractionResult
from map_boundary_builder.runner import low_resolution_shape_catalog_match


KNOWN_CURRENT_CHANGED_CATALOG_SLUGS = {
    "bay-area-waymo",
    "bay-area-zoox",
    "houston-tesla",
    "houston-waymo",
    "miami-waymo",
}

KNOWN_STALE_DERIVED_CHANGED_CATALOG_SLUGS = {
    "bay-area-tesla",
}

KNOWN_CURRENT_EXTERNAL_CATALOG_SLUGS = {
    "atlanta-waymo",
    "austin-waymo",
    "bay-area-waymo",
    "bay-area-zoox",
    "houston-tesla",
    "houston-waymo",
    "miami-waymo",
}

KNOWN_CURRENT_EXTERNAL_CHANGED_SLUGS = {
    "bay-area-zoox",
    "bay-area-waymo",
    "houston-tesla",
    "houston-waymo",
    "miami-waymo",
}

KNOWN_CHANGED_REFERENCE_MISMATCH_SLUGS = {
    "bay-area-tesla",
    "bay-area-waymo",
    "bay-area-zoox",
    "houston-tesla",
    "houston-waymo",
    "miami-waymo",
}

STYLE_BY_PROVIDER = {
    "tesla": "gray-fill",
    "waymo": "bright-blue",
    "zoox": "dark-teal",
}


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


def test_catalog_shape_match_respects_area_hints() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "phoenix-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)

    match = match_service_area_catalog(pixel_geometry, style="bright-blue", area_hint_texts=["Phoenix"])
    wrong_city_match = match_service_area_catalog(pixel_geometry, style="bright-blue", area_hint_texts=["Dallas"])

    assert match is not None
    assert match.entry.slug == "phoenix-waymo"
    assert wrong_city_match is None


def test_catalog_area_aliases_understand_bay_area_text() -> None:
    assert catalog_area_matches_text("Bay Area", "San Francisco")
    assert catalog_area_matches_text("Bay Area", "SF")


def test_catalog_area_hints_distinguish_active_and_stale_markets() -> None:
    assert has_active_catalog_area_hint("Waymo Phoenix")
    assert not has_stale_catalog_area_hint("Waymo Phoenix")
    assert has_active_catalog_area_hint("Waymo Miami")
    assert has_active_catalog_area_hint("Waymo Houston")
    assert has_active_catalog_area_hint("Waymo Bay Area")
    assert has_active_catalog_area_hint("Houston")
    assert has_active_catalog_area_hint("Bay Area")
    assert has_active_catalog_area_hint("Tesla Houston")
    assert has_active_catalog_area_hint("Zoox San Francisco")
    assert not has_active_catalog_area_hint("Tesla Bay Area")
    assert not has_stale_catalog_area_hint("Waymo Miami")
    assert not has_stale_catalog_area_hint("Waymo Houston")
    assert not has_stale_catalog_area_hint("Waymo Bay Area")
    assert not has_stale_catalog_area_hint("Houston")
    assert has_stale_catalog_area_hint("Bay Area")
    assert not has_stale_catalog_area_hint("Tesla Houston")
    assert not has_stale_catalog_area_hint("Zoox San Francisco")
    assert has_stale_catalog_area_hint("Tesla Bay Area")


def test_ocr_derived_catalog_entry_preserves_original_confidence_cap() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "los-angeles-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)

    match = match_service_area_catalog(pixel_geometry, style="bright-blue")

    assert match is not None
    assert match.entry.slug == "los-angeles-waymo"
    assert match.confidence == entry.max_confidence
    assert match.confidence == 0.859


def test_catalog_shape_match_accepts_near_miss_after_small_rotation() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "los-angeles-waymo")
    unrotated_reference = rotate(entry.mercator_geometry, 1.4, origin="centroid", use_radians=False)
    pixel_geometry = mercator_geometry_to_pixel(unrotated_reference)

    match = match_service_area_catalog(pixel_geometry, style="bright-blue")

    assert match is not None
    assert match.entry.slug == "los-angeles-waymo"
    assert match.iou >= 0.97
    assert abs(match.rotation_degrees) > 0.0
    assert match.confidence == 0.859


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
    entry = next(item for item in load_catalog_entries() if item.slug == "los-angeles-waymo")
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


def test_reference_catalog_entry_outputs_exact_geometry_after_match() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "phoenix-waymo")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)
    match = match_service_area_catalog(pixel_geometry, style="bright-blue")
    assert match is not None
    extraction = ExtractionResult(
        mask=np.zeros((100, 100), dtype=np.uint8),
        style="bright-blue",
        pixel_geometry=pixel_geometry.simplify(500, preserve_topology=True),
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
        city_input="input.png",
    )

    output_geometry = shape(data["features"][0]["geometry"])
    assert output_geometry.equals_exact(entry.geometry, tolerance=1e-7)


def test_current_external_catalog_entry_can_keep_audit_threshold() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "bay-area-waymo")

    assert entry.is_active
    assert entry.status == "active"
    assert entry.min_iou == 0.965
    assert entry.stale_reason is None


def test_known_changed_catalog_entries_are_current_and_matched() -> None:
    entries = {item.slug: item for item in load_catalog_entries()}

    assert KNOWN_CURRENT_CHANGED_CATALOG_SLUGS <= set(entries)
    for slug in KNOWN_CURRENT_CHANGED_CATALOG_SLUGS:
        entry = entries[slug]
        pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)
        match = match_service_area_catalog(pixel_geometry, style=STYLE_BY_PROVIDER[entry.provider])

        assert entry.is_active
        assert entry.status == "active"
        assert entry.stale_reason is None
        assert match is not None
        assert match.entry.slug == slug


def test_stale_ocr_derived_changed_catalog_entries_are_not_matched() -> None:
    entries = {item.slug: item for item in load_catalog_entries()}

    assert KNOWN_STALE_DERIVED_CHANGED_CATALOG_SLUGS <= set(entries)
    for slug in KNOWN_STALE_DERIVED_CHANGED_CATALOG_SLUGS:
        entry = entries[slug]
        pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)
        match = match_service_area_catalog(pixel_geometry, style=STYLE_BY_PROVIDER[entry.provider])

        assert not entry.is_active
        assert entry.status == "stale"
        assert entry.stale_reason is not None
        assert match is None


def test_changed_reference_mismatch_catalog_entries_use_current_external_references() -> None:
    fixture_config = json.loads(
        Path("benchmarks/service-area-fixtures.json").read_text()
    )["fixtures"]
    changed_slugs = {
        slug
        for slug, metadata in fixture_config.items()
        if metadata.get("status") == "reference_mismatch"
        and "changed" in metadata.get("note", "").lower()
    }
    entries = {item.slug: item for item in load_catalog_entries()}

    assert changed_slugs == KNOWN_CHANGED_REFERENCE_MISMATCH_SLUGS
    for slug in KNOWN_CURRENT_EXTERNAL_CHANGED_SLUGS:
        entry = entries[slug]
        assert entry.is_active
        assert entry.status == "active"
        assert entry.stale_reason is None
        assert entry.min_iou == 0.965


def test_reference_only_waymo_catalog_entries_are_active() -> None:
    entries = {item.slug: item for item in load_catalog_entries()}

    for slug in ("atlanta-waymo", "austin-waymo"):
        entry = entries[slug]
        pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry)
        match = match_service_area_catalog(pixel_geometry, style="bright-blue")

        assert entry.is_active
        assert entry.status == "active"
        assert entry.stale_reason is None
        assert entry.min_iou == 0.965
        assert match is not None
        assert match.entry.slug == slug


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


def test_low_resolution_shape_catalog_match_accepts_high_margin_sparse_shape() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "nashville-waymo")
    simplified_reference = entry.mercator_geometry.simplify(500, preserve_topology=True)
    pixel_geometry = mercator_geometry_to_pixel(simplified_reference)
    extraction = ExtractionResult(
        mask=np.zeros((236, 420), dtype=np.uint8),
        style="bright-blue",
        pixel_geometry=pixel_geometry,
        coverage_ratio=0.22,
        contour_count=1,
        confidence=1.0,
    )

    match = low_resolution_shape_catalog_match(extraction, width=420, height=236, city_input=None)
    large_match = low_resolution_shape_catalog_match(extraction, width=1000, height=700, city_input=None)

    assert match is not None
    assert match.entry.slug == "nashville-waymo"
    assert match.iou >= 0.945
    assert large_match is None


def test_low_resolution_shape_catalog_match_accepts_downsampled_sparse_shape() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "nashville-waymo")
    simplified_reference = entry.mercator_geometry.simplify(580, preserve_topology=True)
    pixel_geometry = mercator_geometry_to_pixel(simplified_reference)
    extraction = ExtractionResult(
        mask=np.zeros((202, 360), dtype=np.uint8),
        style="bright-blue",
        pixel_geometry=pixel_geometry,
        coverage_ratio=0.22,
        contour_count=1,
        confidence=1.0,
    )

    match = low_resolution_shape_catalog_match(extraction, width=360, height=202, city_input=None)

    assert match is not None
    assert match.entry.slug == "nashville-waymo"
    assert 0.94 <= match.iou < 0.945
    assert match.margin >= 0.24


def test_label_hint_accepts_single_edit_ocr_area_typo() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "nashville-waymo")
    simplified_reference = entry.mercator_geometry.simplify(500, preserve_topology=True)
    pixel_geometry = mercator_geometry_to_pixel(simplified_reference)

    hinted_match = match_service_area_catalog(
        pixel_geometry,
        style="bright-blue",
        min_iou=CATALOG_LABEL_HINT_MIN_IOU,
        area_hint_texts=["Naslville"],
    )

    assert hinted_match is not None
    assert hinted_match.entry.slug == "nashville-waymo"


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
