from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

import map_boundary_builder.runner as runner
from map_boundary_builder.extract import ExtractionResult
from map_boundary_builder.georef_transform import GeoreferenceTransform
from map_boundary_builder.georeference import GeoreferenceResult
from map_boundary_builder.runner import build_boundary, build_summary


def base_feature_collection(properties: dict) -> dict:
    merged = {
        "city": "Phoenix",
        "style": "bright-blue",
        "coverage_ratio": 0.237119,
        "geodesic_bbox_lonlat": [-112.1166355, 33.2312436, -111.8164536, 33.6877976],
        "combined_confidence": 0.984,
        "extraction_confidence": 1.0,
        "georeference_confidence": 0.984,
        "georeference_source": "catalog-shape-match",
        "georeference_control_points": 0,
        "rotation_degrees": 0.0,
        "meters_per_pixel": 28.6,
        "georeference_residual_median_m": 0.0,
        "georeference_residual_p90_m": 0.0,
    }
    merged.update(properties)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": merged,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-112.1, 33.2],
                            [-111.8, 33.2],
                            [-111.8, 33.6],
                            [-112.1, 33.2],
                        ]
                    ],
                },
            }
        ],
    }


def test_summary_exposes_catalog_match_metadata() -> None:
    data = base_feature_collection(
        {
            "catalog_slug": "phoenix-waymo",
            "catalog_shape_iou": 0.984044,
            "catalog_shape_margin": 0.42,
            "catalog_area_ratio": 1.01,
        }
    )

    summary = build_summary(
        data,
        output_path=Path("boundary.geojson"),
        city="Phoenix",
        width=2400,
        height=2400,
        mask_path=None,
        overlay_path=None,
    )

    assert summary["catalog_slug"] == "phoenix-waymo"
    assert summary["catalog_shape_iou"] == 0.984044
    assert summary["catalog_shape_margin"] == 0.42
    assert summary["catalog_area_ratio"] == 1.01


def test_summary_marks_non_catalog_outputs_with_null_catalog_metadata() -> None:
    data = base_feature_collection({"georeference_source": "ocr-georeference:nominatim-label-fit"})

    summary = build_summary(
        data,
        output_path=Path("boundary.geojson"),
        city="Auto",
        width=2400,
        height=2400,
        mask_path=None,
        overlay_path=None,
    )

    assert summary["catalog_slug"] is None
    assert summary["catalog_shape_iou"] is None
    assert summary["catalog_shape_margin"] is None
    assert summary["catalog_area_ratio"] is None


def test_catalog_miss_refines_at_general_processing_cap(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "unknown-waymo.png"
    Image.new("RGB", (2000, 1000), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((1000, 2000, 3), 245, dtype=np.uint8)
    mask = np.zeros((1000, 2000), dtype=bool)
    mask[200:700, 500:1500] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(500, 200), (1500, 200), (1500, 700), (500, 700)]),
        coverage_ratio=0.25,
        contour_count=1,
        confidence=1.0,
    )
    max_dimensions: list[int] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        return extraction

    ocr_rgb_shapes: list[tuple[int, ...]] = []

    def fake_extract_ocr_labels_from_rgb(_path, prepared_rgb):
        ocr_rgb_shapes.append(tuple(prepared_rgb.shape))
        return []

    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Testville",
            lon=-80.0,
            lat=25.0,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=20.0,
            rotation_radians=0.0,
            confidence=0.9,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[],
        residual_median_m=0.0,
        residual_p90_m=0.0,
    )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: georef)

    build_boundary(image_path, None, output_path)

    assert max_dimensions[:2] == [
        runner.CATALOG_EXTRACT_MAX_DIMENSION,
        runner.CATALOG_MISS_REFINE_MAX_DIMENSION,
    ]
    assert runner.CATALOG_MISS_REFINE_MAX_DIMENSION == runner.GENERAL_EXTRACT_MAX_DIMENSION
    assert ocr_rgb_shapes == [(1000, 2000, 3)]


def test_active_catalog_hint_gets_intermediate_retry_before_ocr(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Tesla Bay Area.png"
    Image.new("RGB", (2000, 1000), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((1000, 2000, 3), 245, dtype=np.uint8)
    mask = np.zeros((1000, 2000), dtype=bool)
    mask[200:700, 500:1500] = True
    coarse_extraction = ExtractionResult(
        mask=mask,
        style="gray-fill",
        pixel_geometry=Polygon([(500, 200), (1500, 200), (1500, 700), (500, 700)]),
        coverage_ratio=0.25,
        contour_count=1,
        confidence=1.0,
    )
    retry_extraction = ExtractionResult(
        mask=mask,
        style="gray-fill",
        pixel_geometry=Polygon([(510, 210), (1490, 210), (1490, 690), (510, 690)]),
        coverage_ratio=0.24,
        contour_count=1,
        confidence=1.0,
    )
    max_dimensions: list[int] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        if max_dimension == runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION:
            return retry_extraction
        return coarse_extraction

    def fake_match_service_area_catalog(pixel_geometry, *_args, **_kwargs):
        if pixel_geometry is retry_extraction.pixel_geometry:
            return SimpleNamespace(
                entry=SimpleNamespace(slug="bay-area-tesla"),
                iou=0.969651,
                confidence=0.99,
                margin=0.3,
                area_ratio=1.0,
            )
        return None

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("OCR should not start before the hinted catalog retry succeeds")

    def fake_finish_catalog_boundary_result(
        extraction,
        catalog_match,
        *,
        output_path,
        georeference_source="catalog-shape-match",
        **_kwargs,
    ):
        return runner.BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": georeference_source,
                "style": extraction.style,
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", fake_match_service_area_catalog)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels", unexpected_ocr)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(image_path, "Tesla Bay Area", output_path)

    assert max_dimensions == [
        runner.CATALOG_EXTRACT_MAX_DIMENSION,
        runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION,
    ]
    assert result.summary["catalog_slug"] == "bay-area-tesla"
    assert result.summary["georeference_source"] == "catalog-shape-match:retry"


def test_purple_fill_catalog_miss_uses_smaller_ocr_dimension(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "avride dallas.png"
    Image.new("RGB", (1400, 933), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((933, 1400, 3), 245, dtype=np.uint8)
    mask = np.zeros((933, 1400), dtype=bool)
    mask[150:835, 470:995] = True
    extraction = ExtractionResult(
        mask=mask,
        style="purple-fill",
        pixel_geometry=Polygon([(470, 150), (995, 150), (995, 835), (470, 835)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    ocr_kwargs: list[dict] = []

    def fake_extract_service_area(*_args, **_kwargs):
        return extraction

    def fake_extract_ocr_labels_from_rgb(_path, _prepared_rgb, **kwargs):
        ocr_kwargs.append(kwargs)
        return []

    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Dallas",
            lon=-96.8,
            lat=32.8,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=13.5,
            rotation_radians=0.0,
            confidence=0.9,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[],
        residual_median_m=0.0,
        residual_p90_m=0.0,
    )

    monkeypatch.setattr(runner, "RAPIDOCR_MAX_DIMENSION", 1600)
    monkeypatch.setattr(runner, "RAPIDOCR_PURPLE_FILL_MAX_DIMENSION", 1000)
    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: georef)

    build_boundary(image_path, None, output_path)

    assert ocr_kwargs == [{"rapidocr_max_dimension": 1000}]
