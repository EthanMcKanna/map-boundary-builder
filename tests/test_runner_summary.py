from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from shapely.affinity import rotate
from shapely.geometry import Polygon, mapping
from shapely.ops import transform

import map_boundary_builder.runner as runner
from map_boundary_builder.catalog_match import load_catalog_entries
from map_boundary_builder.extract import ExtractionResult
from map_boundary_builder.georef_transform import GeoreferenceTransform
from map_boundary_builder.georeference import GeoreferenceResult
from map_boundary_builder.ocr import OcrLabel
from map_boundary_builder.runner import BoundaryBuildResult, build_boundary, build_summary


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


def mercator_geometry_to_pixel(geometry):
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max_x - min_x
    height = max_y - min_y

    def to_pixel(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        return (x - min_x) / width * 1000.0, (max_y - y) / height * 1000.0

    return transform(to_pixel, geometry)


def test_extraction_progress_details_include_scaled_cache_telemetry() -> None:
    mask = np.ones((2, 3), dtype=bool)
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (3, 0), (3, 2), (0, 0)]),
        coverage_ratio=float(mask.mean()),
        contour_count=1,
        confidence=0.98,
        scaled_cache_status="hit",
        scaled_cache_shape=(400, 400),
    )

    assert runner.extraction_progress_details(extraction) == {
        "style": "bright-blue",
        "coverage_ratio": 1.0,
        "contour_count": 1,
        "confidence": 0.98,
        "scaled_cache": "hit",
        "scaled_cache_shape": [400, 400],
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


def test_provider_ui_label_catalog_match_uses_nearby_selected_area() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "las-vegas-zoox")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry.simplify(6000, preserve_topology=True))
    extraction = ExtractionResult(
        mask=np.zeros((1600, 800), dtype=np.uint8),
        style="dark-teal",
        pixel_geometry=pixel_geometry,
        coverage_ratio=0.15,
        contour_count=1,
        confidence=1.0,
    )
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    labels = [
        OcrLabel(
            text="You can ride with Zoox within the highlighted area on the map",
            x=360,
            y=1450,
            width=600,
            height=32,
            confidence=96,
        ),
        OcrLabel(
            text="Las Vegas",
            x=(min_x + max_x) / 2,
            y=(min_y + max_y) / 2,
            width=150,
            height=32,
            confidence=99,
        ),
        OcrLabel(
            text="Las Vegas San Francisco",
            x=(min_x + max_x) / 2,
            y=(min_y + max_y) / 2 + 24,
            width=260,
            height=32,
            confidence=98,
        ),
        OcrLabel(
            text="San Francisco",
            x=(min_x + max_x) / 2,
            y=max_y + 600,
            width=180,
            height=32,
            confidence=99,
        ),
    ]

    match = runner.provider_ui_label_catalog_match(extraction, labels)

    assert match is not None
    assert match.entry.slug == "las-vegas-zoox"
    assert match.iou < 0.94
    assert match.confidence >= 0.55


def test_provider_ui_label_provider_accepts_glued_ocr_provider_text() -> None:
    labels = [
        OcrLabel(
            text="Youcanridewith Zooxwithinthehighlighted areaonthemap",
            x=360,
            y=1450,
            width=600,
            height=32,
            confidence=96,
        )
    ]

    assert runner.provider_ui_label_provider(labels) == "zoox"


def test_provider_ui_label_catalog_match_infers_unique_provider_from_style() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "las-vegas-zoox")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry.simplify(6000, preserve_topology=True))
    extraction = ExtractionResult(
        mask=np.zeros((1600, 800), dtype=np.uint8),
        style="dark-teal",
        pixel_geometry=pixel_geometry,
        coverage_ratio=0.15,
        contour_count=1,
        confidence=1.0,
    )
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    labels = [
        OcrLabel(
            text="Las Vegas",
            x=(min_x + max_x) / 2,
            y=(min_y + max_y) / 2,
            width=150,
            height=32,
            confidence=99,
        )
    ]

    match = runner.provider_ui_label_catalog_match(extraction, labels)

    assert runner.unique_catalog_provider_for_style("dark-teal") == "zoox"
    assert match is not None
    assert match.entry.slug == "las-vegas-zoox"


def test_unique_catalog_provider_for_style_rejects_ambiguous_style(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "PROVIDER_STYLES",
        {
            "alpha": {"shared-style"},
            "beta": {"shared-style"},
            "gamma": {"single-style"},
        },
    )

    assert runner.unique_catalog_provider_for_style("shared-style") is None
    assert runner.unique_catalog_provider_for_style("single-style") == "gamma"


def test_provider_ui_ocr_crop_uses_tight_geometry_padding() -> None:
    rgb = np.zeros((1000, 500, 3), dtype=np.uint8)

    crop, offset_x, offset_y = runner.provider_ui_ocr_crop(rgb, (100.0, 200.0, 400.0, 600.0))

    assert crop.shape[:2] == (600, 460)
    assert offset_x == 20.0
    assert offset_y == 100.0


def test_provider_ui_focus_ocr_crop_targets_geometry_interior() -> None:
    rgb = np.zeros((1000, 500, 3), dtype=np.uint8)

    crop, offset_x, offset_y = runner.provider_ui_focus_ocr_crop(rgb, (100.0, 200.0, 400.0, 600.0))

    assert crop.shape[:2] == (480, 255)
    assert offset_x == 130.0
    assert offset_y == 160.0


def test_provider_ui_fast_ocr_allows_gray_fill_non_tall_screens() -> None:
    assert runner.provider_ui_fast_ocr_max_dimension_for_style("gray-fill", width=1200, height=1014) == 1200
    assert runner.provider_ui_fast_ocr_max_dimension_for_style("dark-teal", width=1200, height=1014) is None


def test_style_aware_rapidocr_max_dimension_caps_large_map_ocr_without_tall_dark_teal() -> None:
    assert runner.rapidocr_max_dimension_for_ocr_style("bright-blue", width=2400, height=2400) == 1400
    assert (
        runner.rapidocr_max_dimension_for_ocr_style(
            "bright-blue",
            width=3840,
            height=2055,
            source_is_svg=True,
        )
        == 1600
    )
    assert runner.rapidocr_max_dimension_for_ocr_style("dark-teal", width=1280, height=1012) == 1400
    assert runner.rapidocr_max_dimension_for_ocr_style("dark-teal", width=734, height=1596) is None
    assert runner.rapidocr_max_dimension_for_ocr_style("purple-fill", width=1400, height=933) == 800
    assert runner.rapidocr_rec_batch_num_for_ocr_style("bright-blue") is None
    assert runner.rapidocr_rec_batch_num_for_ocr_style("dark-teal") == 16


def test_focus_georef_ocr_requires_small_dark_teal_crop() -> None:
    rgb = np.zeros((700, 1000, 3), dtype=np.uint8)
    small_dark_teal = ExtractionResult(
        mask=np.zeros((700, 1000), dtype=bool),
        style="dark-teal",
        pixel_geometry=Polygon([(200, 120), (500, 120), (500, 620), (200, 620)]),
        coverage_ratio=0.2,
        contour_count=1,
        confidence=1.0,
    )
    large_dark_teal = ExtractionResult(
        mask=np.zeros((700, 1000), dtype=bool),
        style="dark-teal",
        pixel_geometry=Polygon([(0, 0), (1000, 0), (1000, 700), (0, 700)]),
        coverage_ratio=0.95,
        contour_count=1,
        confidence=1.0,
    )
    gray_fill = ExtractionResult(
        mask=np.zeros((700, 1000), dtype=bool),
        style="gray-fill",
        pixel_geometry=small_dark_teal.pixel_geometry,
        coverage_ratio=0.2,
        contour_count=1,
        confidence=1.0,
    )

    assert runner.focus_georef_ocr_enabled(small_dark_teal, rgb=rgb, city_input=None)
    assert not runner.focus_georef_ocr_enabled(small_dark_teal, rgb=rgb, city_input="Ann Arbor")
    assert not runner.focus_georef_ocr_enabled(large_dark_teal, rgb=rgb, city_input=None)
    assert not runner.focus_georef_ocr_enabled(gray_fill, rgb=rgb, city_input=None)
    assert runner.focus_georef_ocr_max_dimension_for_style("dark-teal") == 550
    assert runner.focus_georef_ocr_max_dimension_for_style("gray-fill") is None
    assert runner.focus_georef_ocr_detector_limit_for_style("dark-teal") == 416
    assert runner.focus_georef_ocr_detector_limit_for_style("gray-fill") is None
    assert runner.focus_georef_ocr_min_text_area_for_style("dark-teal") == 500.0
    assert runner.focus_georef_ocr_min_text_area_for_style("gray-fill") == 1500.0


def test_focused_sparse_fast_fail_requires_no_catalog_tall_provider_ui() -> None:
    labels = [OcrLabel("Las Vegas", x=10, y=10, width=80, height=20, confidence=95)]

    assert runner.should_fast_fail_focused_sparse_ocr(
        True,
        None,
        labels,
        style="dark-teal",
        allow_catalog=False,
        width=734,
        height=1596,
        provider_ui_fast_ocr_max_dimension=1200,
    )
    assert not runner.should_fast_fail_focused_sparse_ocr(
        True,
        object(),
        labels,
        style="dark-teal",
        allow_catalog=False,
        width=734,
        height=1596,
        provider_ui_fast_ocr_max_dimension=1200,
    )
    assert not runner.should_fast_fail_focused_sparse_ocr(
        True,
        None,
        labels,
        style="dark-teal",
        allow_catalog=True,
        width=734,
        height=1596,
        provider_ui_fast_ocr_max_dimension=1200,
    )
    assert not runner.should_fast_fail_focused_sparse_ocr(
        True,
        None,
        labels,
        style="dark-teal",
        allow_catalog=False,
        width=1696,
        height=1365,
        provider_ui_fast_ocr_max_dimension=None,
    )
    assert not runner.should_fast_fail_focused_sparse_ocr(
        True,
        None,
        [OcrLabel(str(index), x=10, y=10, width=80, height=20, confidence=95) for index in range(15)],
        style="dark-teal",
        allow_catalog=False,
        width=734,
        height=1596,
        provider_ui_fast_ocr_max_dimension=1200,
    )


def test_provider_ui_crop_ocr_max_dimension_uses_gray_fill_cap() -> None:
    assert runner.provider_ui_crop_ocr_max_dimension_for_style("gray-fill", rapidocr_max_dimension=1200) == 450
    assert runner.provider_ui_crop_ocr_max_dimension_for_style("dark-teal", rapidocr_max_dimension=1200) == 750


def test_focus_georef_ocr_uses_focused_max_dimension(monkeypatch) -> None:
    rgb = np.zeros((700, 1000, 3), dtype=np.uint8)
    extraction = ExtractionResult(
        mask=np.zeros((700, 1000), dtype=bool),
        style="dark-teal",
        pixel_geometry=Polygon([(200, 120), (500, 120), (500, 620), (200, 620)]),
        coverage_ratio=0.2,
        contour_count=1,
        confidence=1.0,
    )
    captured: dict[str, object] = {}

    def fake_extract_ocr_labels_from_rgb(_path, prepared_rgb, **kwargs):
        captured["shape"] = prepared_rgb.shape[:2]
        captured["kwargs"] = kwargs
        return [OcrLabel("Ann Arbor", x=10, y=20, width=80, height=20, confidence=95)]

    monkeypatch.setattr(runner, "FOCUS_GEOREF_OCR_MAX_DIMENSION", 550)
    monkeypatch.setattr(runner, "FOCUS_GEOREF_OCR_DETECTOR_LIMIT_SIDE_LEN", 416)
    monkeypatch.setattr(runner, "FOCUS_GEOREF_OCR_MIN_TEXT_AREA", 500.0)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)

    labels = runner.extract_focus_georef_labels_from_rgb("map.png", rgb, extraction=extraction)

    assert labels[0].x == 240.0
    assert labels[0].y == 100.0
    assert captured == {
        "shape": (580, 255),
        "kwargs": {
            "allow_tesseract_fallback": False,
            "cache": True,
            "rapidocr_detector_limit_side_len": 416,
            "rapidocr_max_dimension": 550,
            "rapidocr_min_text_area": 500.0,
            "rapidocr_rec_batch_num": 16,
        },
    }


def test_provider_ui_label_catalog_match_rejects_only_ambiguous_area_text() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "las-vegas-zoox")
    pixel_geometry = mercator_geometry_to_pixel(entry.mercator_geometry.simplify(6000, preserve_topology=True))
    extraction = ExtractionResult(
        mask=np.zeros((1600, 800), dtype=np.uint8),
        style="dark-teal",
        pixel_geometry=pixel_geometry,
        coverage_ratio=0.15,
        contour_count=1,
        confidence=1.0,
    )
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    labels = [
        OcrLabel(
            text="Youcanridewith Zooxwithinthehighlighted areaonthemap",
            x=360,
            y=1450,
            width=600,
            height=32,
            confidence=96,
        ),
        OcrLabel(
            text="Las Vegas San Francisco",
            x=(min_x + max_x) / 2,
            y=(min_y + max_y) / 2,
            width=260,
            height=32,
            confidence=99,
        ),
    ]

    assert runner.provider_ui_label_catalog_match(extraction, labels) is None


def test_sparse_low_res_label_catalog_match_recovers_tiny_ocr_typo() -> None:
    extraction = ExtractionResult(
        mask=np.ones((60, 130), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (129, 0), (129, 59), (0, 59)]),
        coverage_ratio=0.79,
        contour_count=1,
        confidence=1.0,
    )
    labels = [OcrLabel("Naslvillk", x=80, y=30, width=33, height=7, confidence=88.9)]

    match = runner.sparse_low_res_label_catalog_match(extraction, labels, width=130, height=60)

    assert match is not None
    assert match.entry.slug == "nashville-waymo"
    assert match.confidence == runner.SPARSE_LABEL_CATALOG_CONFIDENCE


def test_sparse_low_res_label_catalog_match_rejects_larger_map_crops() -> None:
    extraction = ExtractionResult(
        mask=np.ones((236, 420), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (419, 0), (419, 235), (0, 235)]),
        coverage_ratio=0.79,
        contour_count=1,
        confidence=1.0,
    )
    labels = [OcrLabel("Nashville", x=250, y=99, width=37, height=10, confidence=98.1)]

    assert runner.sparse_low_res_label_catalog_match(extraction, labels, width=420, height=236) is None


def test_low_resolution_shape_catalog_match_retries_near_rotated_shape() -> None:
    entry = next(item for item in load_catalog_entries() if item.slug == "nashville-waymo")
    pixel_geometry = mercator_geometry_to_pixel(
        entry.mercator_geometry.simplify(500, preserve_topology=True)
    )
    rotated_geometry = rotate(pixel_geometry, 1.5, origin="centroid", use_radians=False)
    extraction = ExtractionResult(
        mask=np.ones((236, 420), dtype=bool),
        style="bright-blue",
        pixel_geometry=rotated_geometry,
        coverage_ratio=0.2,
        contour_count=1,
        confidence=1.0,
    )

    match = runner.low_resolution_shape_catalog_match(
        extraction,
        width=420,
        height=236,
        city_input=None,
    )

    assert match is not None
    assert match.entry.slug == "nashville-waymo"
    assert match.iou >= runner.LOW_RES_SHAPE_CATALOG_MIN_IOU
    assert match.rotation_degrees != 0.0


def test_runner_ocr_cache_defaults_on_without_disk_cache(monkeypatch) -> None:
    monkeypatch.delenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", raising=False)
    monkeypatch.delenv("MAP_BOUNDARY_OCR_DISK_CACHE", raising=False)

    assert runner.runner_ocr_cache_enabled() is True


def test_runner_ocr_cache_can_be_disabled_or_enabled_by_override(monkeypatch) -> None:
    monkeypatch.delenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", raising=False)
    monkeypatch.setenv("MAP_BOUNDARY_OCR_DISK_CACHE", "1")
    assert runner.runner_ocr_cache_enabled() is True

    monkeypatch.setenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", "0")
    assert runner.runner_ocr_cache_enabled() is False

    monkeypatch.setenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", "true")
    monkeypatch.delenv("MAP_BOUNDARY_OCR_DISK_CACHE", raising=False)
    assert runner.runner_ocr_cache_enabled() is True


def test_road_feature_precompute_default_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES", raising=False)
    assert runner.road_feature_precompute_enabled() is True
    assert runner.should_precompute_road_features("bright-blue", 1000, 800) is True
    assert runner.should_precompute_road_features("gray-fill", 1000, 800) is False
    assert runner.should_precompute_road_features("bright-blue", 999, 800) is False

    monkeypatch.setenv("MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES", "0")
    assert runner.road_feature_precompute_enabled() is False
    assert runner.should_precompute_road_features("bright-blue", 1000, 800) is False

    monkeypatch.setenv("MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES", "true")
    assert runner.road_feature_precompute_enabled() is True
    assert runner.should_precompute_road_features("bright-blue", 1000, 800) is True


def test_classify_style_for_ocr_uses_bounded_sample(monkeypatch) -> None:
    rgb = np.zeros((1600, 3200, 3), dtype=np.uint8)
    calls: list[tuple[int, ...]] = []

    def fake_classify_style(sampled_rgb):
        calls.append(tuple(sampled_rgb.shape))
        return "bright-blue"

    monkeypatch.setattr(runner, "EARLY_OCR_STYLE_MAX_DIMENSION", 800)
    monkeypatch.setattr(runner, "classify_style", fake_classify_style)

    assert runner.classify_style_for_ocr(rgb) == "bright-blue"
    assert calls == [(400, 800, 3)]


def test_classify_style_for_ocr_keeps_small_images_unscaled(monkeypatch) -> None:
    rgb = np.zeros((400, 600, 3), dtype=np.uint8)
    calls: list[tuple[int, ...]] = []

    def fake_classify_style(sampled_rgb):
        calls.append(tuple(sampled_rgb.shape))
        return "gray-fill"

    monkeypatch.setattr(runner, "EARLY_OCR_STYLE_MAX_DIMENSION", 800)
    monkeypatch.setattr(runner, "classify_style", fake_classify_style)

    assert runner.classify_style_for_ocr(rgb) == "gray-fill"
    assert calls == [(400, 600, 3)]


def test_fast_text_ocr_filter_only_applies_to_safe_styles(monkeypatch) -> None:
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_MIN_AREA", 1300.0)

    assert runner.fast_text_ocr_min_area_for_style("bright-blue") == 1300.0
    assert runner.fast_text_ocr_min_area_for_style("bright-blue", source_is_svg=True) is None
    assert runner.fast_text_ocr_min_area_for_style("gray-fill") == 1300.0
    assert runner.fast_text_ocr_min_area_for_style("dark-teal") is None
    assert runner.fast_text_ocr_min_area_for_style("light-fill") == 1300.0
    assert runner.fast_text_ocr_min_area_for_style(None) is None


def test_fast_text_ocr_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_MIN_AREA", 0.0)

    assert runner.fast_text_ocr_min_area_for_style("bright-blue") is None


def test_fast_text_ocr_fallback_guard(monkeypatch) -> None:
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_MIN_AREA", 1200.0)
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_FALLBACK_CONFIDENCE", 0.80)
    low_confidence = SimpleNamespace(transform=SimpleNamespace(confidence=0.79))
    high_confidence = SimpleNamespace(transform=SimpleNamespace(confidence=0.80))
    sparse_high_confidence = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="San Francisco Bay Area",
            lon=-122.45,
            lat=38.01,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=418.0,
            rotation_radians=-0.22,
            confidence=0.82,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[object(), object()],
        residual_median_m=0.0,
        residual_p90_m=0.0,
    )

    assert runner.should_fallback_fast_text_ocr(False, None, style="bright-blue") is False
    assert runner.should_fallback_fast_text_ocr(True, None, style="bright-blue") is True
    assert runner.should_fallback_fast_text_ocr(True, low_confidence, style="bright-blue") is True
    assert runner.should_fallback_fast_text_ocr(True, high_confidence, style="bright-blue") is False
    assert (
        runner.should_fallback_fast_text_ocr(
            True,
            sparse_high_confidence,
            style="gray-fill",
            width=278,
            height=280,
        )
        is True
    )
    assert (
        runner.should_fallback_fast_text_ocr(
            True,
            sparse_high_confidence,
            style="gray-fill",
            width=556,
            height=560,
        )
        is False
    )
    assert runner.should_fallback_fast_text_ocr(True, high_confidence, style="dark-teal") is True


def test_light_fill_label_fits_skip_road_refinement() -> None:
    assert runner.should_allow_label_fit_road_refinement("bright-blue") is True
    assert runner.should_allow_label_fit_road_refinement("gray-fill") is True
    assert runner.should_allow_label_fit_road_refinement(None) is True
    assert runner.should_allow_label_fit_road_refinement("light-fill") is False


def test_sparse_ocr_georeference_rejects_low_res_regional_two_point_fit() -> None:
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="San Francisco Bay Area",
            lon=-122.45,
            lat=38.01,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=418.0,
            rotation_radians=-0.22,
            confidence=0.825,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[object(), object()],
        residual_median_m=0.0,
        residual_p90_m=0.0,
    )

    assert runner.sparse_ocr_georeference_lacks_support(georef, width=278, height=280) is True
    assert runner.sparse_ocr_georeference_lacks_support(georef, width=556, height=560) is False


def test_sparse_ocr_georeference_rejects_high_p90_without_road_evidence() -> None:
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Las Vegas",
            lon=-115.32,
            lat=36.19,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=80.8,
            rotation_radians=-0.17,
            confidence=0.758,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[object(), object(), object(), object()],
        residual_median_m=296.0,
        residual_p90_m=3999.0,
    )

    assert runner.sparse_ocr_georeference_lacks_support(georef, width=311, height=292) is True

    road_supported = GeoreferenceResult(
        transform=georef.transform,
        control_points=georef.control_points,
        residual_median_m=georef.residual_median_m,
        residual_p90_m=georef.residual_p90_m,
        road_match=object(),
    )
    assert runner.sparse_ocr_georeference_lacks_support(road_supported, width=311, height=292) is False


def test_sparse_ocr_georeference_rejects_three_point_drift_without_road_evidence() -> None:
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Las Vegas",
            lon=-115.32,
            lat=36.19,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=58.2,
            rotation_radians=0.05,
            confidence=0.574,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[object(), object(), object()],
        residual_median_m=1996.7,
        residual_p90_m=2024.0,
    )

    assert runner.sparse_ocr_georeference_lacks_support(georef, width=734, height=1596) is True

    road_supported = GeoreferenceResult(
        transform=georef.transform,
        control_points=georef.control_points,
        residual_median_m=georef.residual_median_m,
        residual_p90_m=georef.residual_p90_m,
        road_match=object(),
    )
    assert runner.sparse_ocr_georeference_lacks_support(road_supported, width=734, height=1596) is False


def test_sparse_ocr_georeference_rejects_low_confidence_three_point_fit() -> None:
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Las Vegas",
            lon=-115.16,
            lat=36.09,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=13.9,
            rotation_radians=0.19,
            confidence=0.72,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[object(), object(), object()],
        residual_median_m=419.7,
        residual_p90_m=594.6,
    )
    assert runner.sparse_ocr_georeference_lacks_support(georef, width=1206, height=2622) is True

    supported = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city=georef.transform.city,
            lon=georef.transform.lon,
            lat=georef.transform.lat,
            origin_x_ratio=georef.transform.origin_x_ratio,
            origin_y_ratio=georef.transform.origin_y_ratio,
            meters_per_pixel=georef.transform.meters_per_pixel,
            rotation_radians=georef.transform.rotation_radians,
            confidence=0.752,
            source=georef.transform.source,
        ),
        control_points=georef.control_points,
        residual_median_m=georef.residual_median_m,
        residual_p90_m=georef.residual_p90_m,
    )
    assert runner.sparse_ocr_georeference_lacks_support(supported, width=1206, height=2622) is False


def test_sparse_ocr_georeference_rejects_small_inferred_area_without_road_evidence() -> None:
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Inferred map area",
            lon=-115.18,
            lat=36.11,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=11.8,
            rotation_radians=-0.0002,
            confidence=0.873,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[object(), object(), object(), object(), object()],
        residual_median_m=6.3,
        residual_p90_m=487.8,
    )
    assert runner.sparse_ocr_georeference_lacks_support(georef, width=1206, height=2622) is True

    named_city = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Las Vegas",
            lon=georef.transform.lon,
            lat=georef.transform.lat,
            origin_x_ratio=georef.transform.origin_x_ratio,
            origin_y_ratio=georef.transform.origin_y_ratio,
            meters_per_pixel=georef.transform.meters_per_pixel,
            rotation_radians=georef.transform.rotation_radians,
            confidence=georef.transform.confidence,
            source=georef.transform.source,
        ),
        control_points=georef.control_points,
        residual_median_m=georef.residual_median_m,
        residual_p90_m=georef.residual_p90_m,
    )
    assert runner.sparse_ocr_georeference_lacks_support(named_city, width=1206, height=2622) is False

    road_supported = GeoreferenceResult(
        transform=georef.transform,
        control_points=georef.control_points,
        residual_median_m=georef.residual_median_m,
        residual_p90_m=georef.residual_p90_m,
        road_match=object(),
    )
    assert runner.sparse_ocr_georeference_lacks_support(road_supported, width=1206, height=2622) is False


def test_build_boundary_fails_closed_for_sparse_unsupported_georeference(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "tiny-bay-area.png"
    output_path = tmp_path / "boundary.geojson"
    Image.new("RGB", (278, 280), (245, 245, 245)).save(image_path)
    rgb = np.full((280, 278, 3), 245, dtype=np.uint8)
    mask = np.zeros((280, 278), dtype=bool)
    mask[70:210, 80:220] = True
    extraction = ExtractionResult(
        mask=mask,
        style="gray-fill",
        pixel_geometry=Polygon([(80, 70), (220, 70), (220, 210), (80, 210)]),
        coverage_ratio=0.25,
        contour_count=1,
        confidence=1.0,
    )
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="San Francisco Bay Area",
            lon=-122.45,
            lat=38.01,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=418.0,
            rotation_radians=-0.22,
            confidence=0.825,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[object(), object()],
        residual_median_m=0.0,
        residual_p90_m=0.0,
    )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: georef)

    with pytest.raises(ValueError, match="sparse OCR labels"):
        build_boundary(
            image_path,
            None,
            output_path,
            options=runner.BoundaryBuildOptions(allow_catalog=False),
        )


def test_bright_blue_ocr_uses_style_specific_detector_limit(monkeypatch) -> None:
    monkeypatch.setattr(runner, "RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN", 512)
    monkeypatch.setattr(runner, "RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE", "max")

    assert runner.rapidocr_detector_limit_for_ocr_style("bright-blue") == 512
    assert runner.rapidocr_detector_limit_type_for_ocr_style("bright-blue") == "max"
    assert runner.rapidocr_detector_limit_for_ocr_style("gray-fill") is None
    assert runner.rapidocr_detector_limit_type_for_ocr_style("gray-fill") is None
    assert runner.rapidocr_detector_limit_for_ocr_style(None) is None
    assert runner.rapidocr_detector_limit_type_for_ocr_style(None) is None


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


def test_summary_exposes_road_refinement_elapsed_time() -> None:
    data = base_feature_collection(
        {
            "georeference_source": "ocr-georeference:nominatim-label-fit+osm-road-refine",
            "road_match_score": 0.706233,
            "road_match_elapsed_s": 0.312345,
        }
    )

    summary = build_summary(
        data,
        output_path=Path("boundary.geojson"),
        city="Auto",
        width=2400,
        height=2400,
        mask_path=None,
        overlay_path=None,
    )

    assert summary["road_match_score"] == 0.706233
    assert summary["road_match_elapsed_s"] == 0.312345


def test_catalog_miss_refines_at_bounded_processing_cap(tmp_path, monkeypatch) -> None:
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
    cache_flags: list[bool] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        cache_flags.append(_kwargs.get("cache", True))
        return extraction

    ocr_rgb_shapes: list[tuple[int, ...]] = []
    ocr_kwargs: list[dict] = []

    def fake_extract_ocr_labels_from_rgb(_path, prepared_rgb, **kwargs):
        ocr_rgb_shapes.append(tuple(prepared_rgb.shape))
        ocr_kwargs.append(kwargs)
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
        control_points=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
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
    assert cache_flags[:2] == [False, True]
    assert runner.CATALOG_MISS_REFINE_MAX_DIMENSION == runner.DEFAULT_CATALOG_MISS_REFINE_MAX_DIMENSION
    assert runner.CATALOG_MISS_REFINE_MAX_DIMENSION < runner.GENERAL_EXTRACT_MAX_DIMENSION
    assert ocr_rgb_shapes == [(1000, 2000, 3)]
    assert ocr_kwargs == [
        {
            "rapidocr_max_dimension": runner.RAPIDOCR_BRIGHT_BLUE_MAX_DIMENSION,
            "rapidocr_min_text_area": runner.FAST_TEXT_OCR_MIN_AREA,
            "rapidocr_detector_limit_side_len": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            "rapidocr_detector_limit_type": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
            "rapidocr_recognition_profile": runner.RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
            "cache": True,
        }
    ]


def test_svg_bright_blue_path_uses_vector_ocr_profile(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "unknown-map.png"
    image_path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800">
<rect width="1200" height="800" fill="#ffffff"/>
<path d="M250 120h500v500H250z" fill="#0877ee"/>
</svg>
""",
        encoding="utf-8",
    )
    raster_path = tmp_path / "unknown-map.raster.png"
    Image.new("RGB", (1200, 800), (245, 245, 245)).save(raster_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((800, 1200, 3), 245, dtype=np.uint8)
    mask = np.zeros((800, 1200), dtype=bool)
    mask[160:640, 300:900] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(300, 160), (900, 160), (900, 640), (300, 640)]),
        coverage_ratio=0.3,
        contour_count=1,
        confidence=1.0,
    )
    ocr_kwargs: list[dict] = []

    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Austin",
            lon=-97.74,
            lat=30.27,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=20.0,
            rotation_radians=0.0,
            confidence=0.91,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
        residual_median_m=0.0,
        residual_p90_m=0.0,
    )

    def fake_normalize_image_for_processing(path, **_kwargs):
        assert Path(path) == image_path
        return raster_path

    def fake_extract_ocr_labels_from_rgb(_path, _prepared_rgb, **kwargs):
        ocr_kwargs.append(kwargs)
        return [
            OcrLabel("Austin", 600, 400, 80, 20, 98.0),
            OcrLabel("South Lamar", 540, 500, 100, 20, 96.0),
            OcrLabel("Lady Bird Lake", 660, 420, 120, 20, 96.0),
        ]

    monkeypatch.setattr(runner, "normalize_image_for_processing", fake_normalize_image_for_processing)
    monkeypatch.setattr(runner, "classify_style_for_ocr", lambda _rgb: "bright-blue")
    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "preload_georeference_resources", lambda: {})
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: georef)

    build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(allow_catalog=False, write_mask_artifact=False),
    )

    assert ocr_kwargs == [
        {
            "rapidocr_max_dimension": runner.RAPIDOCR_SVG_BRIGHT_BLUE_MAX_DIMENSION,
            "rapidocr_detector_limit_side_len": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            "rapidocr_detector_limit_type": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
            "rapidocr_recognition_profile": runner.RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
            "cache": True,
        }
    ]


def test_dark_teal_catalog_miss_uses_focused_georef_ocr(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "unknown-dark-teal.png"
    Image.new("RGB", (1000, 700), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((700, 1000, 3), 245, dtype=np.uint8)
    extraction = ExtractionResult(
        mask=np.zeros((700, 1000), dtype=bool),
        style="dark-teal",
        pixel_geometry=Polygon([(200, 120), (500, 120), (500, 620), (200, 620)]),
        coverage_ratio=0.2,
        contour_count=1,
        confidence=1.0,
    )
    ocr_rgb_shapes: list[tuple[int, ...]] = []
    labels = [
        OcrLabel("Ann Arbor", 565, 340, 100, 24, 96.0),
        OcrLabel("Michigan Union", 520, 460, 130, 24, 96.0),
        OcrLabel("Amtrak Station", 500, 240, 130, 24, 96.0),
    ]
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Ann Arbor",
            lon=-83.74,
            lat=42.28,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=4.0,
            rotation_radians=0.0,
            confidence=0.82,
            source="test-georef",
        ),
        control_points=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
        residual_median_m=500.0,
        residual_p90_m=800.0,
    )

    def fake_extract_ocr_labels_from_rgb(_path, prepared_rgb, **_kwargs):
        ocr_rgb_shapes.append(tuple(prepared_rgb.shape))
        return labels

    def fake_fit_georeference(seen_labels, *_args, **_kwargs):
        assert [label.text for label in seen_labels] == [label.text for label in labels]
        return georef

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "fit_georeference", fake_fit_georeference)

    result = build_boundary(image_path, None, output_path)

    assert result.summary["city"] == "Ann Arbor"
    assert ocr_rgb_shapes == [(580, 255, 3)]


def test_focused_georef_ocr_falls_back_to_full_detail(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "unknown-dark-teal.png"
    Image.new("RGB", (1000, 700), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((700, 1000, 3), 245, dtype=np.uint8)
    extraction = ExtractionResult(
        mask=np.zeros((700, 1000), dtype=bool),
        style="dark-teal",
        pixel_geometry=Polygon([(200, 120), (500, 120), (500, 620), (200, 620)]),
        coverage_ratio=0.2,
        contour_count=1,
        confidence=1.0,
    )
    focus_labels = [OcrLabel("Hands On", 430, 340, 80, 24, 96.0)]
    full_labels = [OcrLabel("Ann Arbor", 565, 340, 100, 24, 96.0)]
    ocr_rgb_shapes: list[tuple[int, ...]] = []
    fit_label_batches: list[list[OcrLabel]] = []
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Ann Arbor",
            lon=-83.74,
            lat=42.28,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=4.0,
            rotation_radians=0.0,
            confidence=0.82,
            source="test-georef",
        ),
        control_points=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
        residual_median_m=500.0,
        residual_p90_m=800.0,
    )

    def fake_extract_ocr_labels_from_rgb(_path, prepared_rgb, **_kwargs):
        shape = tuple(prepared_rgb.shape)
        ocr_rgb_shapes.append(shape)
        return focus_labels if shape == (580, 255, 3) else full_labels

    def fake_fit_georeference(seen_labels, *_args, **_kwargs):
        fit_label_batches.append(seen_labels)
        return None if [label.text for label in seen_labels] == ["Hands On"] else georef

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "fit_georeference", fake_fit_georeference)

    result = build_boundary(image_path, None, output_path)

    assert result.summary["city"] == "Ann Arbor"
    assert ocr_rgb_shapes == [(580, 255, 3), (700, 1000, 3)]
    assert [[label.text for label in batch] for batch in fit_label_batches] == [["Hands On"], ["Ann Arbor"]]


def test_catalog_probe_miss_label_shape_shortcut_uses_one_low_detail_ocr(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "uploaded-map.png"
    Image.new("RGB", (2000, 1000), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((1000, 2000, 3), 245, dtype=np.uint8)
    extraction = ExtractionResult(
        mask=np.ones((40, 40), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(500, 200), (1500, 200), (1500, 700), (500, 700)]),
        coverage_ratio=0.25,
        contour_count=1,
        confidence=1.0,
    )
    ocr_kwargs: list[dict] = []
    labels = [OcrLabel("Houston", 10, 10, 80, 20, 96.0)]
    match = SimpleNamespace(entry=SimpleNamespace(slug="houston-waymo"))

    def fake_extract_ocr_labels_from_rgb(_path, _prepared_rgb, **kwargs):
        ocr_kwargs.append(kwargs)
        return labels

    def fake_current_catalog_label_shape_match(_extraction, seen_labels):
        assert seen_labels is labels
        return match

    def unexpected_georef(*_args, **_kwargs):
        raise AssertionError("label-shape catalog shortcut should return before full georeference")

    def fake_finish_catalog_boundary_result(_extraction, catalog_match, *, output_path, **kwargs):
        return BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "current_catalog_label_shape_match", fake_current_catalog_label_shape_match)
    monkeypatch.setattr(runner, "fit_georeference", unexpected_georef)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(
            catalog_probe_missed=True,
            catalog_probe_miss_low_iou=True,
            filename_hint="uploaded-map.webp",
            write_mask_artifact=False,
        ),
    )

    assert result.summary["catalog_slug"] == "houston-waymo"
    assert result.summary["georeference_source"] == "catalog-shape-match:label-shape"
    assert ocr_kwargs == [
        {
            "rapidocr_max_dimension": runner.CURRENT_CATALOG_LABEL_OCR_MAX_DIMENSION,
            "rapidocr_min_text_area": runner.FAST_TEXT_OCR_MIN_AREA,
            "rapidocr_detector_limit_side_len": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            "rapidocr_detector_limit_type": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
            "rapidocr_recognition_profile": runner.RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
            "cache": True,
        }
    ]


def test_catalog_probe_miss_filename_shape_shortcut_avoids_ocr(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Houston.catalog-handoff.webp"
    Image.new("RGB", (1600, 1600), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((1600, 1600, 3), 245, dtype=np.uint8)
    extraction = ExtractionResult(
        mask=np.ones((40, 40), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(400, 300), (1300, 300), (1300, 1200), (400, 1200)]),
        coverage_ratio=0.32,
        contour_count=1,
        confidence=1.0,
    )
    match = SimpleNamespace(entry=SimpleNamespace(slug="houston-waymo"))
    shortcut_calls: list[dict[str, object]] = []

    def fake_filename_hinted_current_catalog_shape_match(seen_extraction, *, city_input, filename_hint):
        shortcut_calls.append(
            {
                "extraction": seen_extraction,
                "city_input": city_input,
                "filename_hint": filename_hint,
            }
        )
        return match

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("filename-shape handoff shortcut should avoid OCR")

    def fake_finish_catalog_boundary_result(_extraction, catalog_match, *, output_path, **kwargs):
        return BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "filename_hinted_current_catalog_shape_match",
        fake_filename_hinted_current_catalog_shape_match,
    )
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(
            catalog_probe_missed=True,
            filename_hint="Waymo Houston.catalog-handoff.webp",
            write_mask_artifact=False,
        ),
    )

    assert result.summary["catalog_slug"] == "houston-waymo"
    assert result.summary["georeference_source"] == "catalog-shape-match:filename-shape"
    assert len(shortcut_calls) == 1
    assert shortcut_calls[0]["extraction"] is extraction
    assert shortcut_calls[0]["city_input"] is None
    assert shortcut_calls[0]["filename_hint"] == "Waymo Houston.catalog-handoff.webp"


def test_filename_shape_shortcut_runs_before_ocr_for_direct_hinted_upload(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Houston.png"
    Image.new("RGB", (1600, 1600), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((1600, 1600, 3), 245, dtype=np.uint8)
    extraction = ExtractionResult(
        mask=np.ones((40, 40), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(400, 300), (1300, 300), (1300, 1200), (400, 1200)]),
        coverage_ratio=0.32,
        contour_count=1,
        confidence=1.0,
    )
    match = SimpleNamespace(entry=SimpleNamespace(slug="houston-waymo"))
    shortcut_calls: list[dict[str, object]] = []

    def fake_filename_hinted_current_catalog_shape_match(seen_extraction, *, city_input, filename_hint):
        shortcut_calls.append(
            {
                "extraction": seen_extraction,
                "city_input": city_input,
                "filename_hint": filename_hint,
            }
        )
        return match

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("direct provider/area filename-shape shortcut should avoid OCR")

    def fake_finish_catalog_boundary_result(_extraction, catalog_match, *, output_path, **kwargs):
        return BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "filename_hinted_current_catalog_shape_match",
        fake_filename_hinted_current_catalog_shape_match,
    )
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(
            filename_hint="Waymo Houston.png",
            write_mask_artifact=False,
        ),
    )

    assert result.summary["catalog_slug"] == "houston-waymo"
    assert result.summary["georeference_source"] == "catalog-shape-match:filename-shape"
    assert len(shortcut_calls) == 1
    assert shortcut_calls[0]["extraction"].style == extraction.style
    assert shortcut_calls[0]["city_input"] is None
    assert shortcut_calls[0]["filename_hint"] == "Waymo Houston.png"


def test_direct_provider_area_near_hit_returns_before_refine_and_ocr(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Bay Area.png"
    Image.new("RGB", (2400, 2400), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    low_rgb = np.full((400, 400, 3), 245, dtype=np.uint8)
    low_mask = np.zeros((400, 400), dtype=bool)
    low_mask[80:320, 70:330] = True
    extraction = ExtractionResult(
        mask=low_mask,
        style="bright-blue",
        pixel_geometry=Polygon([(70, 80), (330, 80), (330, 320), (70, 320)]),
        coverage_ratio=float(low_mask.mean()),
        contour_count=1,
        confidence=1.0,
    )
    match = SimpleNamespace(entry=SimpleNamespace(slug="bay-area-waymo"))
    extract_dimensions: list[int] = []
    near_hit_calls: list[dict[str, object]] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        extract_dimensions.append(max_dimension)
        return extraction

    def fake_near_hit(seen_extraction, *, city_input, filename_hint):
        near_hit_calls.append(
            {
                "extraction": seen_extraction,
                "city_input": city_input,
                "filename_hint": filename_hint,
            }
        )
        return match

    def unexpected_full_rgb(_path):
        raise AssertionError("direct provider/area near-hit should not decode full RGB")

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("direct provider/area near-hit should avoid OCR")

    def fake_finish_catalog_boundary_result(_extraction, catalog_match, *, output_path, **kwargs):
        return BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb_at_max_dimension", lambda _path, _max_dimension: low_rgb)
    monkeypatch.setattr(runner, "load_rgb", unexpected_full_rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "filename_hinted_current_catalog_shape_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "filename_hinted_current_catalog_near_hit_match", fake_near_hit)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(
            filename_hint="Waymo Bay Area.png",
            write_mask_artifact=False,
        ),
    )

    assert result.summary["catalog_slug"] == "bay-area-waymo"
    assert result.summary["georeference_source"] == "catalog-shape-match:filename-near-hit"
    assert extract_dimensions == [0]
    assert len(near_hit_calls) == 1
    assert near_hit_calls[0]["city_input"] is None
    assert near_hit_calls[0]["filename_hint"] == "Waymo Bay Area.png"


def test_filename_shape_shortcut_requires_provider_hint(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((40, 40), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(400, 300), (1300, 300), (1300, 1200), (400, 1200)]),
        coverage_ratio=0.32,
        contour_count=1,
        confidence=1.0,
    )

    def unexpected_catalog_scan():
        raise AssertionError("area-only filenames must not use the filename-shape shortcut")

    monkeypatch.setattr(runner, "load_catalog_entries", unexpected_catalog_scan)

    assert (
        runner.filename_hinted_current_catalog_shape_match(
            extraction,
            city_input=None,
            filename_hint="Houston.catalog-handoff.webp",
        )
        is None
    )


def test_filename_shape_shortcut_requires_strong_shape_evidence(monkeypatch) -> None:
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Houston",
        slug="houston-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        mercator_geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        max_confidence=0.88,
        use_exact_geometry=True,
    )
    loose_extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (10, 20), (10, 10), (0, 10)]),
        coverage_ratio=0.75,
        contour_count=1,
        confidence=1.0,
    )

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])

    assert (
        runner.filename_hinted_current_catalog_shape_match(
            loose_extraction,
            city_input=None,
            filename_hint="Waymo Houston.png",
        )
        is None
    )


def test_filename_shape_shortcut_accepts_exact_provider_area_shape(monkeypatch) -> None:
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Houston",
        slug="houston-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        mercator_geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        max_confidence=0.88,
        use_exact_geometry=True,
    )
    exact_extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=1.0,
    )

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])

    match = runner.filename_hinted_current_catalog_shape_match(
        exact_extraction,
        city_input=None,
        filename_hint="Waymo Houston.png",
    )

    assert match is not None
    assert match.entry.slug == "houston-waymo"
    assert match.iou == pytest.approx(1.0)
    assert match.confidence == pytest.approx(runner.CURRENT_CATALOG_LABEL_SHAPE_CONFIDENCE)


def test_filename_near_hit_shortcut_requires_provider_and_active_area(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((40, 40), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(400, 300), (1300, 300), (1300, 1200), (400, 1200)]),
        coverage_ratio=0.32,
        contour_count=1,
        confidence=1.0,
    )
    match = SimpleNamespace(entry=SimpleNamespace(slug="bay-area-waymo"))
    calls: list[tuple[str | None, str | None]] = []

    def fake_catalog_probe_near_hit_match(_extraction, *, city_input, filename_hint):
        calls.append((city_input, filename_hint))
        return match

    monkeypatch.setattr(runner, "catalog_probe_near_hit_match", fake_catalog_probe_near_hit_match)

    assert (
        runner.filename_hinted_current_catalog_near_hit_match(
            extraction,
            city_input=None,
            filename_hint="Bay Area.png",
        )
        is None
    )
    assert (
        runner.filename_hinted_current_catalog_near_hit_match(
            extraction,
            city_input=None,
            filename_hint="Waymo map.png",
        )
        is None
    )

    accepted = runner.filename_hinted_current_catalog_near_hit_match(
        extraction,
        city_input=None,
        filename_hint="Waymo Bay Area.png",
    )

    assert accepted is match
    assert calls == [(None, "Waymo Bay Area.png")]


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
    cache_flags: list[bool] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        cache_flags.append(_kwargs.get("cache", True))
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
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(image_path, "Tesla Bay Area", output_path)

    assert max_dimensions == [0, runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION]
    assert cache_flags == [False, False]
    assert result.summary["catalog_slug"] == "bay-area-tesla"
    assert result.summary["georeference_source"] == "catalog-shape-match:retry"


def test_active_catalog_hit_uses_low_res_rgb_before_full_decode(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Houston.png"
    Image.new("RGB", (2400, 1200), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    low_rgb = np.full((120, 240, 3), 245, dtype=np.uint8)
    low_mask = np.zeros((120, 240), dtype=bool)
    low_mask[20:90, 50:190] = True
    extraction = ExtractionResult(
        mask=low_mask,
        style="bright-blue",
        pixel_geometry=Polygon([(50, 20), (190, 20), (190, 90), (50, 90)]),
        coverage_ratio=float(low_mask.mean()),
        contour_count=1,
        confidence=1.0,
    )
    loader_dimensions: list[int] = []
    extract_dimensions: list[int] = []

    def fake_load_rgb_at_max_dimension(_path, max_dimension):
        loader_dimensions.append(max_dimension)
        return low_rgb

    def unexpected_full_rgb(_path):
        raise AssertionError("active catalog hits should not decode the full image before returning")

    def fake_extract_service_area(*_args, max_dimension=None, rgb=None, **_kwargs):
        extract_dimensions.append(max_dimension)
        assert rgb is low_rgb
        return extraction

    def fake_match_service_area_catalog(*_args, **_kwargs):
        return SimpleNamespace(
            entry=SimpleNamespace(slug="houston-waymo"),
            iou=0.979470,
            confidence=0.88,
            margin=0.36,
            area_ratio=1.0,
        )

    def fake_finish_catalog_boundary_result(extraction, catalog_match, *, output_path, **_kwargs):
        assert extraction.mask.shape == (1200, 2400)
        return runner.BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": _kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb_at_max_dimension", fake_load_rgb_at_max_dimension)
    monkeypatch.setattr(runner, "load_rgb", unexpected_full_rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", fake_match_service_area_catalog)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(image_path, "Houston", output_path)

    assert loader_dimensions == [runner.CATALOG_EXTRACT_MAX_DIMENSION]
    assert extract_dimensions == [0]
    assert result.summary["catalog_slug"] == "houston-waymo"


def test_catalog_probe_only_miss_stops_before_ocr_and_full_refine(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Houston probe.jpg"
    Image.new("RGB", (1400, 933), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((933, 1400, 3), 245, dtype=np.uint8)
    mask = np.zeros((933, 1400), dtype=bool)
    mask[150:835, 470:995] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(470, 150), (995, 150), (995, 835), (470, 835)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    max_dimensions: list[int] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        return extraction

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("catalog probes must not fall through to OCR")

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)

    entry = SimpleNamespace(is_active=True, provider="waymo", slug="houston-waymo", min_iou=0.965)

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])
    monkeypatch.setattr(
        runner,
        "score_catalog_entry",
        lambda *_args, **_kwargs: (0.61, 1.0, entry, Polygon(), 0.0),
    )

    with pytest.raises(runner.CatalogProbeMiss) as exc_info:
        build_boundary(
            image_path,
            None,
            output_path,
            options=runner.BoundaryBuildOptions(
                catalog_probe_only=True,
                filename_hint="Waymo Houston probe.jpg",
                write_mask_artifact=False,
            ),
        )

    assert max_dimensions == [0, runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION]
    assert exc_info.value.details["best_active_catalog_slug"] == "houston-waymo"
    assert exc_info.value.details["best_active_catalog_iou"] == 0.61
    assert exc_info.value.details["active_shape_iou_is_low"] is True


def test_catalog_probe_only_accepts_hinted_verified_near_hit(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Bay Area probe.webp"
    Image.new("RGB", (520, 520), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((520, 520, 3), 245, dtype=np.uint8)
    mask = np.zeros((520, 520), dtype=bool)
    mask[120:420, 80:430] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(80, 120), (430, 120), (430, 420), (80, 420)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Bay Area",
        slug="bay-area-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
    )

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("near-hit catalog probes must not fall through to OCR")

    def fake_match_catalog_entry(_pixel_geometry, candidate, *, min_iou, min_area_ratio, max_area_ratio):
        assert candidate is entry
        assert min_iou == runner.CATALOG_PROBE_NEAR_HIT_MIN_IOU
        assert min_area_ratio == runner.CATALOG_PROBE_NEAR_HIT_MIN_AREA_RATIO
        assert max_area_ratio == runner.CATALOG_PROBE_NEAR_HIT_MAX_AREA_RATIO
        return SimpleNamespace(entry=entry, iou=0.881, margin=0.881, area_ratio=1.03, confidence=0.877)

    def fake_finish_catalog_boundary_result(_extraction, catalog_match, *, output_path, **kwargs):
        return runner.BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])
    monkeypatch.setattr(runner, "match_catalog_entry", fake_match_catalog_entry)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(
            catalog_probe_only=True,
            filename_hint="Waymo Bay Area probe.webp",
            write_mask_artifact=False,
        ),
    )

    assert result.summary["catalog_slug"] == "bay-area-waymo"
    assert result.summary["georeference_source"] == "catalog-shape-match:probe-near-hit"


def test_catalog_probe_near_hit_accepts_unique_unhinted_verified_source(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=1.0,
    )
    bay = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Bay Area",
        slug="bay-area-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
    )
    miami = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Miami",
        slug="miami-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
    )

    def fake_score(_pixel_geometry, entry, *, min_iou):
        if entry is bay:
            return 0.881, 1.03, entry, Polygon(), 0.0
        return 0.22, 0.8, entry, Polygon(), 0.0

    def fake_catalog_match_from_score(_pixel_geometry, entry, **kwargs):
        assert entry is bay
        assert kwargs["margin"] > runner.CATALOG_PROBE_UNHINTED_NEAR_HIT_MIN_MARGIN
        return SimpleNamespace(entry=entry, iou=kwargs["iou"], margin=kwargs["margin"], area_ratio=kwargs["area_ratio"])

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [bay, miami])
    monkeypatch.setattr(runner, "score_catalog_entry", fake_score)
    monkeypatch.setattr(runner, "catalog_match_from_score", fake_catalog_match_from_score)

    match = runner.catalog_probe_near_hit_match(
        extraction,
        city_input=None,
        filename_hint="uploaded-map.webp",
    )

    assert match is not None
    assert match.entry.slug == "bay-area-waymo"


def test_catalog_probe_near_hit_rejects_unhinted_low_margin(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=1.0,
    )
    bay = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Bay Area",
        slug="bay-area-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
    )
    miami = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Miami",
        slug="miami-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
    )

    def fake_score(_pixel_geometry, entry, *, min_iou):
        if entry is bay:
            return 0.881, 1.03, entry, Polygon(), 0.0
        return 0.72, 1.01, entry, Polygon(), 0.0

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [bay, miami])
    monkeypatch.setattr(runner, "score_catalog_entry", fake_score)

    assert (
        runner.catalog_probe_near_hit_match(
            extraction,
            city_input=None,
            filename_hint="uploaded-map.webp",
        )
        is None
    )


def test_post_georeference_catalog_completion_accepts_visible_current_subset(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=0.95,
    )
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Houston",
        slug="houston-waymo",
        catalog_source="current-verified-ocr-output",
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        mercator_geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        max_confidence=0.88,
        use_exact_geometry=True,
    )
    visible_subset = Polygon([(1, 1), (7, 1), (7, 9), (1, 9)])

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])
    monkeypatch.setattr(runner, "lonlat_to_mercator", lambda lon, lat: (lon, lat))

    match = runner.post_georeference_catalog_completion_match(
        extraction,
        [OcrLabel("Houston", 10, 10, 80, 20, 96.0)],
        visible_subset,
        city_input=None,
        filename_hint="uploaded-map.webp",
        georef_confidence=0.91,
    )

    assert match is not None
    assert match.entry.slug == "houston-waymo"
    assert match.iou == pytest.approx(0.48)
    assert match.area_ratio == pytest.approx(0.48)
    assert match.margin == pytest.approx(1.0)
    assert match.confidence == pytest.approx(runner.POST_GEOREF_CATALOG_COMPLETION_CONFIDENCE)


def test_post_georeference_catalog_completion_rejects_tiny_subset(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=0.95,
    )
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Miami",
        slug="miami-waymo",
        catalog_source="current-verified-ocr-output",
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        mercator_geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        max_confidence=0.897,
        use_exact_geometry=True,
    )
    tiny_visible_subset = Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])
    monkeypatch.setattr(runner, "lonlat_to_mercator", lambda lon, lat: (lon, lat))

    assert (
        runner.post_georeference_catalog_completion_match(
            extraction,
            [OcrLabel("Miami", 10, 10, 80, 20, 96.0)],
            tiny_visible_subset,
            city_input=None,
            filename_hint="uploaded-map.webp",
            georef_confidence=0.91,
        )
        is None
    )


def test_current_catalog_label_shape_match_accepts_current_area_label(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=1.0,
    )
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Houston",
        slug="houston-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        mercator_geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        max_confidence=0.88,
        use_exact_geometry=True,
    )

    def fake_score(_pixel_geometry, candidate, *, min_iou):
        assert candidate is entry
        return 0.88, 1.05, candidate, Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 0.0

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])
    monkeypatch.setattr(runner, "score_catalog_entry", fake_score)

    match = runner.current_catalog_label_shape_match(
        extraction,
        [OcrLabel("Downtown Houston", 10, 10, 80, 20, 96.0)],
    )

    assert match is not None
    assert match.entry.slug == "houston-waymo"
    assert match.iou == pytest.approx(0.88)
    assert match.area_ratio == pytest.approx(1.05)
    assert match.confidence == pytest.approx(runner.CURRENT_CATALOG_LABEL_SHAPE_CONFIDENCE)


def test_current_catalog_label_shape_match_rejects_weak_shape(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="bright-blue",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=1.0,
    )
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        area="Miami",
        slug="miami-waymo",
        min_iou=0.965,
        catalog_source="current-verified-ocr-output",
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        mercator_geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        max_confidence=0.897,
        use_exact_geometry=True,
    )

    def fake_score(_pixel_geometry, candidate, *, min_iou):
        assert candidate is entry
        return 0.49, 1.0, candidate, Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 0.0

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])
    monkeypatch.setattr(runner, "score_catalog_entry", fake_score)

    assert (
        runner.current_catalog_label_shape_match(
            extraction,
            [OcrLabel("Miami Beach", 10, 10, 80, 20, 96.0)],
        )
        is None
    )


def test_area_hinted_current_catalog_match_accepts_high_margin_verified_source(monkeypatch) -> None:
    pixel_geometry = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    bay_area = SimpleNamespace(
        slug="bay-area-waymo",
        is_active=True,
        catalog_source="current-verified-ocr-output",
        provider="waymo",
        area="Bay Area",
        min_iou=0.965,
        mercator_geometry=Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]),
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        max_confidence=0.877,
        use_exact_geometry=True,
    )
    phoenix = SimpleNamespace(
        slug="phoenix-waymo",
        is_active=True,
        catalog_source="current-verified-ocr-output",
        provider="waymo",
        area="Phoenix",
        min_iou=0.97,
        mercator_geometry=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        max_confidence=0.9,
        use_exact_geometry=True,
    )

    def fake_score(_pixel_geometry, entry, *, min_iou):
        assert min_iou == entry.min_iou
        iou = 0.959 if entry.slug == "bay-area-waymo" else 0.2
        return iou, 1.0, entry, entry.mercator_geometry, 0.0

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [bay_area, phoenix])
    monkeypatch.setattr(runner, "score_catalog_entry", fake_score)

    match = runner.area_hinted_current_catalog_shape_match(
        pixel_geometry,
        style="bright-blue",
        city_input="Bay Area",
    )

    assert match is not None
    assert match.entry.slug == "bay-area-waymo"
    assert match.iou == pytest.approx(0.959)
    assert match.margin == pytest.approx(0.759)


def test_area_hinted_current_catalog_match_rejects_when_best_shape_is_different_area(monkeypatch) -> None:
    pixel_geometry = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    bay_area = SimpleNamespace(
        slug="bay-area-waymo",
        is_active=True,
        catalog_source="current-verified-ocr-output",
        provider="waymo",
        area="Bay Area",
        min_iou=0.965,
        mercator_geometry=Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]),
        geometry=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        max_confidence=0.877,
        use_exact_geometry=True,
    )
    phoenix = SimpleNamespace(
        slug="phoenix-waymo",
        is_active=True,
        catalog_source="current-verified-ocr-output",
        provider="waymo",
        area="Phoenix",
        min_iou=0.97,
        mercator_geometry=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        max_confidence=0.9,
        use_exact_geometry=True,
    )

    def fake_score(_pixel_geometry, entry, *, min_iou):
        iou = 0.959 if entry.slug == "bay-area-waymo" else 0.97
        return iou, 1.0, entry, entry.mercator_geometry, 0.0

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [bay_area, phoenix])
    monkeypatch.setattr(runner, "score_catalog_entry", fake_score)

    match = runner.area_hinted_current_catalog_shape_match(
        pixel_geometry,
        style="bright-blue",
        city_input="Bay Area",
    )

    assert match is None


def test_complex_area_hinted_current_catalog_starts_at_retry_dimension(monkeypatch) -> None:
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        catalog_source="current-verified-ocr-output",
        area="Bay Area",
        geometry=Polygon([(index, index % 7) for index in range(151)]),
    )

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])

    assert (
        runner.initial_catalog_extract_max_dimension(
            city_input="Bay Area",
            filename_hint="Waymo Bay Area.png",
            allow_pre_ocr_catalog=True,
        )
        == runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION
    )


def test_rotated_exact_contour_catalog_hint_starts_at_refine_dimension() -> None:
    assert (
        runner.initial_catalog_extract_max_dimension(
            city_input=None,
            filename_hint="Zoox Las Vegas.png",
            allow_pre_ocr_catalog=True,
        )
        == runner.CATALOG_MISS_REFINE_MAX_DIMENSION
    )


def test_smaller_area_hinted_current_catalog_keeps_tiny_probe(monkeypatch) -> None:
    entry = SimpleNamespace(
        is_active=True,
        catalog_source="current-verified-ocr-output",
        area="Miami",
        geometry=Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
    )

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])

    assert (
        runner.initial_catalog_extract_max_dimension(
            city_input="Miami",
            filename_hint="Waymo Miami.png",
            allow_pre_ocr_catalog=True,
        )
        == runner.CATALOG_EXTRACT_MAX_DIMENSION
    )


def test_area_hinted_current_catalog_respects_provider_hint(monkeypatch) -> None:
    entry = SimpleNamespace(
        is_active=True,
        provider="waymo",
        catalog_source="current-verified-ocr-output",
        area="Bay Area",
        geometry=Polygon([(index, index % 7) for index in range(151)]),
    )

    monkeypatch.setattr(runner, "load_catalog_entries", lambda: [entry])

    assert (
        runner.initial_catalog_extract_max_dimension(
            city_input="Tesla Bay Area",
            filename_hint="Tesla Bay Area.png",
            allow_pre_ocr_catalog=True,
        )
        == runner.CATALOG_EXTRACT_MAX_DIMENSION
    )


def test_pre_ocr_catalog_retry_skips_duplicate_retry_dimension() -> None:
    assert (
        runner.should_retry_pre_ocr_catalog(
            city_input="Bay Area",
            filename_hint="Waymo Bay Area.png",
            allow_pre_ocr_catalog=True,
            used_catalog_scaled_extraction=True,
            initial_extract_max_dimension=runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION,
        )
        is False
    )


def test_catalog_probe_only_retries_without_area_hint_before_miss(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "h-waymo probe.jpg"
    Image.new("RGB", (520, 520), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((520, 520, 3), 245, dtype=np.uint8)
    mask = np.zeros((520, 520), dtype=bool)
    mask[120:420, 80:430] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(80, 120), (430, 120), (430, 420), (80, 420)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    max_dimensions: list[int] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        return extraction

    def fake_match_service_area_catalog(*_args, **_kwargs):
        if max_dimensions[-1] == runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION:
            return SimpleNamespace(
                entry=SimpleNamespace(slug="houston-waymo"),
                iou=0.968,
                confidence=0.88,
                margin=0.35,
                area_ratio=1.01,
            )
        return None

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("catalog probes must not fall through to OCR")

    def fake_finish_catalog_boundary_result(_extraction, catalog_match, *, output_path, **_kwargs):
        return runner.BoundaryBuildResult(
            geojson={},
            summary={
                "catalog_slug": catalog_match.entry.slug,
                "georeference_source": _kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", fake_match_service_area_catalog)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(
            catalog_probe_only=True,
            filename_hint="h-waymo probe.jpg",
            write_mask_artifact=False,
        ),
    )

    assert max_dimensions == [
        runner.CATALOG_EXTRACT_MAX_DIMENSION,
        runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION,
    ]
    assert result.summary["catalog_slug"] == "houston-waymo"
    assert result.summary["georeference_source"] == "catalog-shape-match:retry"


def test_catalog_probe_missed_skips_low_res_probes_but_keeps_full_catalog_match(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Bay Area.png"
    Image.new("RGB", (2400, 2400), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((2400, 2400, 3), 245, dtype=np.uint8)
    mask = np.zeros((2400, 2400), dtype=bool)
    mask[400:2000, 500:1900] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(500, 400), (1900, 400), (1900, 2000), (500, 2000)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    max_dimensions: list[int] = []
    cache_flags: list[bool] = []
    match_calls: list[dict] = []

    def fake_extract_service_area(*_args, max_dimension=None, cache=True, **_kwargs):
        max_dimensions.append(max_dimension)
        cache_flags.append(cache)
        return extraction

    def fake_match_service_area_catalog(pixel_geometry, *, style, area_hint_texts=None, **_kwargs):
        match_calls.append({"style": style, "area_hint_texts": area_hint_texts})
        return SimpleNamespace(entry=SimpleNamespace(slug="bay-area-waymo"), iou=0.98, margin=0.3, area_ratio=1.0)

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("full catalog match after a probe miss must still avoid OCR")

    def fake_finish_catalog_boundary_result(_extraction, _match, **kwargs):
        return BoundaryBuildResult(
            geojson={
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(extraction.pixel_geometry)}],
            },
            summary={
                "catalog_slug": "bay-area-waymo",
                "georeference_source": kwargs["georeference_source"],
            },
            output_path=output_path,
        )

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", fake_match_service_area_catalog)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(
        image_path,
        "Bay Area",
        output_path,
        options=runner.BoundaryBuildOptions(
            catalog_probe_missed=True,
            filename_hint="Waymo Bay Area.png",
            write_mask_artifact=False,
        ),
    )

    assert max_dimensions == [runner.CATALOG_MISS_REFINE_MAX_DIMENSION]
    assert cache_flags == [True]
    assert match_calls == [{"style": "bright-blue", "area_hint_texts": ["Bay Area"]}]
    assert result.summary["catalog_slug"] == "bay-area-waymo"
    assert result.summary["georeference_source"] == "catalog-shape-match:probe-miss-full"


def test_catalog_probe_missed_skips_low_res_probes_for_generic_requests(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "unlabeled-map.png"
    Image.new("RGB", (2400, 2400), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((2400, 2400, 3), 245, dtype=np.uint8)
    mask = np.zeros((2400, 2400), dtype=bool)
    mask[400:2000, 500:1900] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(500, 400), (1900, 400), (1900, 2000), (500, 2000)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    max_dimensions: list[int] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        return extraction

    def stop_at_ocr(*_args, **_kwargs):
        raise RuntimeError("stop after generic extraction")

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", stop_at_ocr)

    with pytest.raises(RuntimeError, match="stop after generic extraction"):
        build_boundary(
            image_path,
            None,
            output_path,
            options=runner.BoundaryBuildOptions(
                catalog_probe_missed=True,
                filename_hint="unlabeled-map.png",
                write_mask_artifact=False,
            ),
        )

    assert max_dimensions == [
        runner.CATALOG_MISS_REFINE_MAX_DIMENSION,
    ]


def test_catalog_probe_low_iou_miss_allows_early_ocr_for_provider_hints() -> None:
    assert (
        runner.should_overlap_probe_miss_ocr(
            skip_redundant_probe=True,
            city_input=None,
            filename_hint="uploaded-map.png",
            catalog_probe_miss_low_iou=True,
        )
        is False
    )
    assert (
        runner.should_overlap_probe_miss_ocr(
            skip_redundant_probe=True,
            city_input=None,
            filename_hint="Waymo Houston.png",
        )
        is False
    )
    assert (
        runner.should_overlap_probe_miss_ocr(
            skip_redundant_probe=True,
            city_input=None,
            filename_hint="Waymo Houston.png",
            catalog_probe_miss_low_iou=True,
        )
        is True
    )


def test_no_catalog_dark_teal_defers_pre_extraction_ocr_only_for_large_near_square_focus(monkeypatch) -> None:
    monkeypatch.setattr(runner, "PROVIDER_UI_FOCUS_CROP_ENABLED", True)
    monkeypatch.setattr(runner, "PROVIDER_UI_CROP_OCR_MAX_DIMENSION", 750)

    assert (
        runner.should_defer_pre_extraction_ocr_for_focus(
            "dark-teal",
            city_input=None,
            allow_catalog=False,
            width=1696,
            height=1365,
        )
        is True
    )
    assert (
        runner.should_defer_pre_extraction_ocr_for_focus(
            "dark-teal",
            city_input=None,
            allow_catalog=False,
            width=2880,
            height=1620,
        )
        is False
    )
    assert (
        runner.should_defer_pre_extraction_ocr_for_focus(
            "dark-teal",
            city_input=None,
            allow_catalog=False,
            width=1280,
            height=1012,
        )
        is False
    )
    assert (
        runner.should_defer_pre_extraction_ocr_for_focus(
            "bright-blue",
            city_input=None,
            allow_catalog=False,
            width=1696,
            height=1365,
        )
        is False
    )


def test_no_catalog_tall_dark_teal_focus_can_fail_sparse_without_full_retry(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "tall-zoox-like.png"
    output_path = tmp_path / "boundary.geojson"
    Image.new("RGB", (734, 1596), (18, 65, 70)).save(image_path)
    rgb = np.full((1596, 734, 3), 40, dtype=np.uint8)
    extraction = ExtractionResult(
        mask=np.zeros((1596, 734), dtype=bool),
        style="dark-teal",
        pixel_geometry=Polygon([(80, 420), (680, 420), (680, 900), (80, 900)]),
        coverage_ratio=0.18,
        contour_count=1,
        confidence=1.0,
    )
    focus_labels = [
        OcrLabel("Las Vegas", x=130, y=500, width=90, height=24, confidence=96),
        OcrLabel("Paradise", x=160, y=560, width=80, height=22, confidence=94),
    ]

    monkeypatch.setattr(runner, "classify_style_for_ocr", lambda _rgb: "dark-teal")
    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "extract_focus_georef_labels_from_rgb", lambda *_args, **_kwargs: focus_labels)
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "extract_full_ocr_labels_for_style",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full OCR should not run")),
    )

    with pytest.raises(ValueError, match="sparse OCR labels"):
        build_boundary(
            image_path,
            None,
            output_path,
            options=runner.BoundaryBuildOptions(allow_catalog=False, write_mask_artifact=False),
        )


def test_focused_georef_display_city_prefers_exact_admin_control() -> None:
    georef = GeoreferenceResult(
        transform=GeoreferenceTransform(
            city="Yost Ice Arena",
            lon=-83.74,
            lat=42.28,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=4.0,
            rotation_radians=0.0,
            confidence=0.85,
            source="ocr-georeference:nominatim-label-fit",
        ),
        control_points=[
            SimpleNamespace(
                label=OcrLabel("Yost Ice Arena", 900, 900, 120, 22, 93.5),
                geocode=SimpleNamespace(
                    display_name="Yost Ice Arena, Ann Arbor, Michigan",
                    place_type="leisure",
                ),
            ),
            SimpleNamespace(
                label=OcrLabel("Ann Arbor", 800, 436, 95, 22, 99.5),
                geocode=SimpleNamespace(
                    display_name="Ann Arbor, Washtenaw County, Michigan, United States",
                    place_type="city",
                ),
            ),
        ],
        residual_median_m=500.0,
        residual_p90_m=700.0,
    )

    updated = runner.focused_georef_with_admin_control_city(georef)

    assert updated.transform.city == "Ann Arbor"
    assert updated.transform.confidence == georef.transform.confidence
    assert updated.control_points == georef.control_points


def test_avride_light_fill_filename_hint_uses_catalog_before_ocr(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "uber-avride-operating-map-dallas.png"
    Image.new("RGB", (680, 551), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((551, 680, 3), 245, dtype=np.uint8)
    mask = np.zeros((551, 680), dtype=bool)
    mask[120:450, 60:630] = True
    extraction = ExtractionResult(
        mask=mask,
        style="light-fill",
        pixel_geometry=Polygon([(60, 120), (630, 120), (630, 450), (60, 450)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    match_calls: list[dict] = []

    def fake_match_service_area_catalog(pixel_geometry, *, style, min_iou=None, min_margin=None, area_hint_texts=None):
        match_calls.append(
            {
                "style": style,
                "min_iou": min_iou,
                "min_margin": min_margin,
                "area_hint_texts": area_hint_texts,
            }
        )
        if style == "purple-fill":
            return SimpleNamespace(
                entry=SimpleNamespace(slug="dallas-avride"),
                iou=0.926,
                confidence=0.922,
                margin=0.3,
                area_ratio=0.98,
            )
        return None

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("filename-hinted Avride catalog match should avoid OCR")

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
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "match_service_area_catalog", fake_match_service_area_catalog)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", unexpected_ocr)
    monkeypatch.setattr(runner, "finish_catalog_boundary_result", fake_finish_catalog_boundary_result)

    result = build_boundary(image_path, None, output_path)

    assert result.summary["catalog_slug"] == "dallas-avride"
    assert result.summary["georeference_source"] == "catalog-shape-match:filename-hint"
    assert match_calls[-1] == {
        "style": "purple-fill",
        "min_iou": runner.FILENAME_HINTED_AVRIDE_LIGHT_FILL_MIN_IOU,
        "min_margin": runner.FILENAME_HINTED_AVRIDE_LIGHT_FILL_MIN_MARGIN,
        "area_hint_texts": ["uber-avride-operating-map-dallas"],
    }


def test_avride_light_fill_provider_hint_can_use_current_catalog_without_area_hint(monkeypatch) -> None:
    extraction = ExtractionResult(
        mask=np.ones((20, 20), dtype=bool),
        style="light-fill",
        pixel_geometry=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        coverage_ratio=1.0,
        contour_count=1,
        confidence=1.0,
    )
    match_calls: list[dict] = []

    def fake_match_service_area_catalog(pixel_geometry, *, style, min_iou=None, min_margin=None, area_hint_texts=None):
        match_calls.append(
            {
                "style": style,
                "min_iou": min_iou,
                "min_margin": min_margin,
                "area_hint_texts": area_hint_texts,
            }
        )
        return SimpleNamespace(
            entry=SimpleNamespace(slug="dallas-avride", catalog_source="current-verified-ocr-output"),
            iou=0.927,
            confidence=0.922,
            margin=0.927,
            area_ratio=0.98,
        )

    monkeypatch.setattr(runner, "has_active_catalog_area_hint", lambda _hint: False)
    monkeypatch.setattr(runner, "match_service_area_catalog", fake_match_service_area_catalog)

    match = runner.filename_hinted_avride_light_fill_catalog_match(
        extraction,
        filename_hint="neutral-avride-upload.webp",
    )

    assert match is not None
    assert match.entry.slug == "dallas-avride"
    assert match_calls == [
        {
            "style": "purple-fill",
            "min_iou": runner.FILENAME_HINTED_AVRIDE_PROVIDER_ONLY_MIN_IOU,
            "min_margin": runner.FILENAME_HINTED_AVRIDE_PROVIDER_ONLY_MIN_MARGIN,
            "area_hint_texts": None,
        }
    ]


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
    monkeypatch.setattr(runner, "RAPIDOCR_PURPLE_FILL_MAX_DIMENSION", 800)
    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: georef)

    build_boundary(image_path, None, output_path)

    assert ocr_kwargs == [{"rapidocr_max_dimension": 800, "cache": True}]


def test_unsupported_style_catalog_miss_skips_catalog_retry(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "waymo dallas.png"
    Image.new("RGB", (1400, 933), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((933, 1400, 3), 245, dtype=np.uint8)
    mask = np.zeros((933, 1400), dtype=bool)
    mask[150:835, 470:995] = True
    extraction = ExtractionResult(
        mask=mask,
        style="orange-fill",
        pixel_geometry=Polygon([(470, 150), (995, 150), (995, 835), (470, 835)]),
        coverage_ratio=0.27,
        contour_count=1,
        confidence=1.0,
    )
    max_dimensions: list[int] = []

    def fake_extract_service_area(*_args, max_dimension=None, **_kwargs):
        max_dimensions.append(max_dimension)
        return extraction

    def unexpected_catalog(*_args, **_kwargs):
        raise AssertionError("unsupported styles cannot match the current catalog provider styles")

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

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", fake_extract_service_area)
    monkeypatch.setattr(runner, "match_service_area_catalog", unexpected_catalog)
    monkeypatch.setattr(runner, "low_resolution_shape_catalog_match", unexpected_catalog)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "fit_georeference", lambda *_args, **_kwargs: georef)

    build_boundary(image_path, None, output_path)

    assert max_dimensions == [0, runner.CATALOG_MISS_REFINE_MAX_DIMENSION]


def test_no_catalog_path_preloads_georeference_resources_before_fit(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "unknown-map.png"
    Image.new("RGB", (1200, 800), (245, 245, 245)).save(image_path)
    output_path = tmp_path / "boundary.geojson"
    rgb = np.full((800, 1200, 3), 245, dtype=np.uint8)
    mask = np.zeros((800, 1200), dtype=bool)
    mask[160:640, 300:900] = True
    extraction = ExtractionResult(
        mask=mask,
        style="bright-blue",
        pixel_geometry=Polygon([(300, 160), (900, 160), (900, 640), (300, 640)]),
        coverage_ratio=0.3,
        contour_count=1,
        confidence=1.0,
    )
    order: list[str] = []

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

    def fake_preload_georeference_resources():
        order.append("preload")
        return {"geocoder_seed_entries": 1, "osm_place_seed_entries": 1, "road_seed_entries": 1}

    def fake_fit_georeference(*_args, **kwargs):
        order.append("fit")
        assert "preload" in order
        assert kwargs["anchor_marker_dots"] is False
        return georef

    ocr_rgb_shapes: list[tuple[int, ...]] = []
    ocr_kwargs: list[dict] = []

    def fake_extract_ocr_labels_from_rgb(_path, prepared_rgb, **kwargs):
        ocr_rgb_shapes.append(tuple(prepared_rgb.shape))
        ocr_kwargs.append(kwargs)
        return []

    monkeypatch.setattr(runner, "load_rgb", lambda _path: rgb)
    monkeypatch.setattr(runner, "extract_service_area", lambda *_args, **_kwargs: extraction)
    monkeypatch.setattr(runner, "extract_ocr_labels_from_rgb", fake_extract_ocr_labels_from_rgb)
    monkeypatch.setattr(runner, "preload_georeference_resources", fake_preload_georeference_resources)
    monkeypatch.setattr(runner, "fit_georeference", fake_fit_georeference)

    build_boundary(
        image_path,
        None,
        output_path,
        options=runner.BoundaryBuildOptions(allow_catalog=False, write_mask_artifact=False),
    )

    assert order == ["preload", "fit"]
    assert ocr_rgb_shapes == [(800, 1200, 3)]
    assert ocr_kwargs == [
        {
            "rapidocr_max_dimension": runner.RAPIDOCR_BRIGHT_BLUE_MAX_DIMENSION,
            "rapidocr_min_text_area": runner.FAST_TEXT_OCR_MIN_AREA,
            "rapidocr_detector_limit_side_len": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_SIDE_LEN,
            "rapidocr_detector_limit_type": runner.RAPIDOCR_BRIGHT_BLUE_DET_LIMIT_TYPE,
            "rapidocr_recognition_profile": runner.RAPIDOCR_BRIGHT_BLUE_RECOGNITION_PROFILE,
            "cache": True,
        }
    ]
