from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
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


def test_runner_ocr_cache_defaults_off_without_disk_cache(monkeypatch) -> None:
    monkeypatch.delenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", raising=False)
    monkeypatch.delenv("MAP_BOUNDARY_OCR_DISK_CACHE", raising=False)

    assert runner.runner_ocr_cache_enabled() is False


def test_runner_ocr_cache_can_be_enabled_for_disk_or_override(monkeypatch) -> None:
    monkeypatch.delenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", raising=False)
    monkeypatch.setenv("MAP_BOUNDARY_OCR_DISK_CACHE", "1")
    assert runner.runner_ocr_cache_enabled() is True

    monkeypatch.setenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", "0")
    assert runner.runner_ocr_cache_enabled() is False

    monkeypatch.setenv("MAP_BOUNDARY_RUNNER_OCR_CACHE", "true")
    monkeypatch.delenv("MAP_BOUNDARY_OCR_DISK_CACHE", raising=False)
    assert runner.runner_ocr_cache_enabled() is True


def test_fast_text_ocr_filter_only_applies_to_safe_styles(monkeypatch) -> None:
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_MIN_AREA", 1200.0)

    assert runner.fast_text_ocr_min_area_for_style("bright-blue") == 1200.0
    assert runner.fast_text_ocr_min_area_for_style("gray-fill") == 1200.0
    assert runner.fast_text_ocr_min_area_for_style("dark-teal") is None
    assert runner.fast_text_ocr_min_area_for_style("light-fill") is None
    assert runner.fast_text_ocr_min_area_for_style(None) is None


def test_fast_text_ocr_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_MIN_AREA", 0.0)

    assert runner.fast_text_ocr_min_area_for_style("bright-blue") is None


def test_fast_text_ocr_fallback_guard(monkeypatch) -> None:
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_MIN_AREA", 1200.0)
    monkeypatch.setattr(runner, "FAST_TEXT_OCR_FALLBACK_CONFIDENCE", 0.80)
    low_confidence = SimpleNamespace(transform=SimpleNamespace(confidence=0.79))
    high_confidence = SimpleNamespace(transform=SimpleNamespace(confidence=0.80))

    assert runner.should_fallback_fast_text_ocr(False, None, style="bright-blue") is False
    assert runner.should_fallback_fast_text_ocr(True, None, style="bright-blue") is True
    assert runner.should_fallback_fast_text_ocr(True, low_confidence, style="bright-blue") is True
    assert runner.should_fallback_fast_text_ocr(True, high_confidence, style="bright-blue") is False
    assert runner.should_fallback_fast_text_ocr(True, high_confidence, style="dark-teal") is True


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
    assert cache_flags[:2] == [False, True]
    assert runner.CATALOG_MISS_REFINE_MAX_DIMENSION == runner.DEFAULT_CATALOG_MISS_REFINE_MAX_DIMENSION
    assert runner.CATALOG_MISS_REFINE_MAX_DIMENSION < runner.GENERAL_EXTRACT_MAX_DIMENSION
    assert ocr_rgb_shapes == [(1000, 2000, 3)]
    assert ocr_kwargs == [{"rapidocr_min_text_area": 800.0, "cache": False}]


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

    assert max_dimensions == [
        runner.CATALOG_EXTRACT_MAX_DIMENSION,
        runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION,
    ]
    assert cache_flags == [False, False]
    assert result.summary["catalog_slug"] == "bay-area-tesla"
    assert result.summary["georeference_source"] == "catalog-shape-match:retry"


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

    with pytest.raises(runner.CatalogProbeMiss):
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

    assert max_dimensions == [
        runner.CATALOG_EXTRACT_MAX_DIMENSION,
        runner.CATALOG_RETRY_EXTRACT_MAX_DIMENSION,
    ]


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

    assert ocr_kwargs == [{"rapidocr_max_dimension": 800, "cache": False}]


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

    assert max_dimensions == [
        runner.CATALOG_EXTRACT_MAX_DIMENSION,
        runner.CATALOG_MISS_REFINE_MAX_DIMENSION,
    ]


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

    def fake_fit_georeference(*_args, **_kwargs):
        order.append("fit")
        assert "preload" in order
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
    assert ocr_kwargs == [{"rapidocr_min_text_area": 800.0, "cache": False}]
