from pathlib import Path

from map_boundary_builder.runner import build_summary


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
