from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from map_boundary_builder import pipeline as pipeline_module
from map_boundary_builder.extract import ExtractionResult
from map_boundary_builder.georef_transform import GeoreferenceTransform
from map_boundary_builder.georeference import GeoreferenceResult
from map_boundary_builder.ocr import OcrLabel
from map_boundary_builder.pipeline import (
    PipelineOptions,
    complete_with_city,
    run_pipeline,
)


@pytest.fixture
def image_path(tmp_path):
    rgb = np.full((120, 160, 3), 245, dtype=np.uint8)
    rgb[30:90, 40:120] = (60, 120, 235)
    path = tmp_path / "sample.png"
    Image.fromarray(rgb).save(path)
    return path


def fake_extraction() -> ExtractionResult:
    mask = np.zeros((120, 160), dtype=bool)
    mask[30:90, 40:120] = True
    return ExtractionResult(
        mask=mask,
        style="model-mask",
        pixel_geometry=Polygon([(40, 30), (120, 30), (120, 90), (40, 90)]),
        coverage_ratio=float(mask.mean()),
        contour_count=1,
        confidence=0.9,
        diagnostics={"segmentation_engine": "model"},
    )


def fake_labels() -> list[OcrLabel]:
    return [
        OcrLabel(text="Main St", x=50.0, y=40.0, width=30.0, height=8.0, confidence=0.95),
        OcrLabel(text="Oak Ave", x=80.0, y=70.0, width=30.0, height=8.0, confidence=0.90),
    ]


def fake_georeference() -> GeoreferenceResult:
    transform = GeoreferenceTransform(
        city="Austin, TX",
        lon=-97.74,
        lat=30.27,
        origin_x_ratio=0.5,
        origin_y_ratio=0.5,
        meters_per_pixel=25.0,
        rotation_radians=0.0,
        confidence=0.8,
        source="ocr-georeference:nominatim-label-fit",
    )
    return GeoreferenceResult(
        transform=transform,
        control_points=[],
        residual_median_m=500.0,
        residual_p90_m=1200.0,
    )


@pytest.fixture
def patched_stages(monkeypatch):
    monkeypatch.setattr(pipeline_module, "segment_image", lambda rgb, **kwargs: fake_extraction())
    monkeypatch.setattr(
        pipeline_module,
        "extract_ocr_labels_from_rgb",
        lambda path, rgb, **kwargs: fake_labels(),
    )
    return monkeypatch


def test_complete_run(image_path, tmp_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: fake_georeference(),
    )
    output = tmp_path / "out" / "boundary.geojson"
    debug = tmp_path / "debug"
    result = run_pipeline(image_path, output_path=output, debug_dir=debug)
    assert result.status == "complete"
    assert result.reason is None
    assert result.geojson is not None
    assert result.geojson["type"] == "FeatureCollection"
    assert result.summary["city"] == "Austin, TX"
    assert result.summary["georeference"]["control_points"] == 0
    assert result.summary["combined_confidence"] == pytest.approx(0.72)
    assert output.is_file()
    assert (debug / "mask.png").is_file()
    assert (debug / "overlay.png").is_file()
    assert result.cacheable


def test_needs_city_when_no_georeference(image_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: None,
    )
    patched_stages.setattr(
        pipeline_module,
        "resolve_city_contexts",
        lambda labels, city: [],
    )
    result = run_pipeline(image_path)
    assert result.status == "needs_city"
    assert result.reason == "no_city_context"
    assert result.needs_city is not None
    assert result.needs_city.ocr_label_count == 2
    assert "Main St" in result.needs_city.sample_labels
    assert result.extraction is not None
    assert result.geojson is None
    assert result.summary["needs_city"]["reason"] == "no_city_context"


def test_failed_when_city_supplied_but_unfittable(image_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: None,
    )
    result = run_pipeline(image_path, city="Austin, TX")
    assert result.status == "failed"
    assert result.reason == "georeference_failed_with_city"
    assert result.needs_city is None
    assert result.cacheable


def test_failed_when_no_boundary(image_path, monkeypatch):
    def raise_extraction(rgb, **kwargs):
        raise ValueError("No service-area polygon could be extracted")

    monkeypatch.setattr(pipeline_module, "segment_image", raise_extraction)
    monkeypatch.setattr(
        pipeline_module,
        "extract_ocr_labels_from_rgb",
        lambda path, rgb, **kwargs: fake_labels(),
    )
    result = run_pipeline(image_path)
    assert result.status == "failed"
    assert result.reason == "no_boundary_found"
    assert result.summary["ocr_label_count"] == 2
    assert result.cacheable


def test_failed_invalid_image(tmp_path):
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"not an image")
    result = run_pipeline(bogus)
    assert result.status == "failed"
    assert result.reason == "invalid_image"


def test_transient_georeference_error_not_cacheable(image_path, patched_stages):
    def explode(*args, **kwargs):
        raise TimeoutError("nominatim timed out")

    patched_stages.setattr(pipeline_module, "georeference_from_labels", explode)
    result = run_pipeline(image_path)
    assert result.status == "failed"
    assert result.reason == "georeference_error"
    assert not result.cacheable


def test_low_confidence_gate(image_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: fake_georeference(),
    )
    result = run_pipeline(image_path, options=PipelineOptions(min_confidence=0.95))
    assert result.status == "failed"
    assert result.reason == "low_confidence"


def test_complete_with_city_reuses_extraction(image_path, tmp_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: None,
    )
    first = run_pipeline(image_path)
    assert first.status == "needs_city"

    calls = {"segment": 0}

    def count_segment(rgb, **kwargs):
        calls["segment"] += 1
        return fake_extraction()

    patched_stages.setattr(pipeline_module, "segment_image", count_segment)
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: fake_georeference(),
    )
    output = tmp_path / "retry.geojson"
    second = complete_with_city(image_path, first, "Austin, TX", output_path=output)
    assert second.status == "complete"
    assert calls["segment"] == 0
    assert output.is_file()
    assert second.summary["city_input"] == "Austin, TX"


def test_road_search_fallback_rescues_label_fit_failure(image_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: None,
    )
    calls: list[str] = []

    def fake_road_search(rgb, candidate, geometry):
        calls.append(candidate)
        return fake_georeference()

    patched_stages.setattr(pipeline_module, "georeference_from_city_context", fake_road_search)
    result = run_pipeline(image_path, city="Tampa, FL")
    assert result.status == "complete"
    assert calls == ["Tampa, FL"]


def test_road_search_fallback_uses_inferred_contexts(image_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: None,
    )

    class FakeCenter:
        mercator = (-9175000.0, 3225000.0)

    class FakeContext:
        def __init__(self, query):
            self.query = query
            self.center = FakeCenter()

    patched_stages.setattr(
        pipeline_module,
        "resolve_city_contexts",
        lambda labels, city: [FakeContext("Tampa"), FakeContext("Hillsborough County")],
    )
    patched_stages.setattr(pipeline_module, "_label_anchors", lambda labels, center, query="": [])
    patched_stages.setattr(pipeline_module, "_rescue_fit_is_sane", lambda result, anchors, w, h: True)
    calls: list[str] = []

    def fake_road_search(rgb, candidate, geometry):
        calls.append(candidate)
        return fake_georeference() if candidate == "Hillsborough County" else None

    patched_stages.setattr(pipeline_module, "georeference_from_city_context", fake_road_search)
    result = run_pipeline(image_path)
    assert result.status == "complete"
    assert calls == ["Tampa", "Hillsborough County"]


def test_progress_callback_stages(image_path, patched_stages):
    patched_stages.setattr(
        pipeline_module,
        "georeference_from_labels",
        lambda *args, **kwargs: fake_georeference(),
    )
    stages: list[str] = []
    run_pipeline(image_path, progress=lambda stage, percent, detail: stages.append(stage))
    assert stages == ["load", "extract", "georeference", "export", "complete"]
