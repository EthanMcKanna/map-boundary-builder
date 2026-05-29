from pathlib import Path

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
    monkeypatch.setattr(runner, "extract_ocr_labels", lambda _path: [])
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: georef)

    build_boundary(image_path, None, output_path)

    assert max_dimensions[:2] == [
        runner.CATALOG_EXTRACT_MAX_DIMENSION,
        runner.CATALOG_MISS_REFINE_MAX_DIMENSION,
    ]
    assert runner.CATALOG_MISS_REFINE_MAX_DIMENSION == runner.GENERAL_EXTRACT_MAX_DIMENSION
