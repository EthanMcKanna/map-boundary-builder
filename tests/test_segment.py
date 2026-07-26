from __future__ import annotations

import json

import numpy as np
import pytest

from map_boundary_builder.extract import ExtractionHints
from map_boundary_builder import segment as segment_module
from map_boundary_builder.segment import (
    DEFAULT_MODEL_THRESHOLD,
    _degenerate_reason,
    load_model_config,
    segment_image,
    select_hinted_components,
)


def synthetic_fill_image(width: int = 320, height: int = 240) -> np.ndarray:
    """Map-like fixture: light basemap with road lines and a blue overlay fill.

    The auto-fill fallback gates on interior texture (real map fills keep
    roads visible through the overlay), so the fixture draws a road grid
    under a translucent-looking fill rather than flat rectangles.
    """
    rgb = np.full((height, width, 3), 245, dtype=np.uint8)
    for x in range(0, width, 24):
        rgb[:, x : x + 2] = 210
    for y in range(0, height, 24):
        rgb[y : y + 2, :] = 210
    fill = rgb[60:180, 80:240].astype(np.float64)
    overlay = np.array([60.0, 120.0, 235.0])
    rgb[60:180, 80:240] = (0.35 * fill + 0.65 * overlay).astype(np.uint8)
    return rgb


def test_segment_image_falls_back_without_model(tmp_path):
    rgb = synthetic_fill_image()
    result = segment_image(rgb, model_path=tmp_path / "missing.onnx")
    assert result.mask.shape == rgb.shape[:2]
    assert result.diagnostics["segmentation_engine"] == "auto-fill-fallback"
    assert 0.05 < result.coverage_ratio < 0.5
    assert result.mask[120, 160]


def test_segment_image_raises_when_nothing_extractable(tmp_path):
    flat = np.full((64, 64, 3), 250, dtype=np.uint8)
    with pytest.raises(ValueError):
        segment_image(flat, model_path=tmp_path / "missing.onnx")


def test_segment_image_prefers_model_result(tmp_path, monkeypatch):
    rgb = synthetic_fill_image()
    probabilities = np.zeros(rgb.shape[:2], dtype=np.float32)
    probabilities[60:180, 80:240] = 0.95
    model_path = tmp_path / "boundary_v1.onnx"
    model_path.write_bytes(b"stub")
    monkeypatch.setattr(segment_module, "load_onnx_session", lambda path: object())
    monkeypatch.setattr(
        segment_module,
        "predict_mask_probabilities",
        lambda rgb, session, config: probabilities,
    )
    result = segment_image(rgb, model_path=model_path)
    assert result.diagnostics["segmentation_engine"] == "model"
    assert result.style == "model-mask"
    assert result.confidence > 0.5
    assert result.mask[120, 160]
    assert not result.mask[10, 10]


def test_segment_image_degenerate_model_falls_back(tmp_path, monkeypatch):
    rgb = synthetic_fill_image()
    # Near-total coverage is degenerate; the auto-fill fallback should win.
    probabilities = np.full(rgb.shape[:2], 0.99, dtype=np.float32)
    model_path = tmp_path / "boundary_v1.onnx"
    model_path.write_bytes(b"stub")
    monkeypatch.setattr(segment_module, "load_onnx_session", lambda path: object())
    monkeypatch.setattr(
        segment_module,
        "predict_mask_probabilities",
        lambda rgb, session, config: probabilities,
    )
    result = segment_image(rgb, model_path=model_path)
    assert result.diagnostics["segmentation_engine"] == "auto-fill-fallback"


def test_load_model_config_reads_sidecar(tmp_path):
    model_path = tmp_path / "boundary_v1.onnx"
    model_path.write_bytes(b"stub")
    sidecar = {
        "production": {
            "threshold": 0.42,
            "output_activation": "logits",
            "input_width": 320,
            "input_height": 320,
        }
    }
    (tmp_path / "boundary_v1.onnx.json").write_text(json.dumps(sidecar), encoding="utf-8")
    config = load_model_config(model_path)
    assert config.threshold == 0.42
    assert config.input_width == 320
    assert config.output_activation == "logits"


def test_load_model_config_defaults_without_sidecar(tmp_path):
    config = load_model_config(tmp_path / "boundary_v1.onnx")
    assert config.threshold == DEFAULT_MODEL_THRESHOLD
    assert config.input_channels == 3


def test_load_model_config_threshold_override(tmp_path):
    config = load_model_config(tmp_path / "boundary_v1.onnx", threshold=0.61)
    assert config.threshold == 0.61


def test_degenerate_reasons():
    empty = np.zeros((50, 50), dtype=bool)
    assert _degenerate_reason(empty, 0.0) == "coverage_too_low"
    full = np.ones((50, 50), dtype=bool)
    assert _degenerate_reason(full, 0.0) == "coverage_too_high"
    ok = np.zeros((50, 50), dtype=bool)
    ok[10:40, 10:40] = True
    assert _degenerate_reason(ok, 0.0) is None
    assert _degenerate_reason(ok, 0.5) == "uncertain_probabilities"
    border = np.zeros((50, 50), dtype=bool)
    border[0:40, :] = True
    assert _degenerate_reason(border, 0.0) == "mask_hugs_border"


def test_select_hinted_components_seed():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[60:90, 60:90] = True
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    hints = ExtractionHints(seed_point=(70.0, 70.0))
    selected = select_hinted_components(mask, rgb, hints)
    assert selected[75, 75]
    assert not selected[20, 20]


def test_select_hinted_components_seed_off_component_snaps_to_nearest():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[60:90, 60:90] = True
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    hints = ExtractionHints(seed_point=(55.0, 55.0))
    selected = select_hinted_components(mask, rgb, hints)
    assert selected[75, 75]
    assert not selected[20, 20]


def test_select_hinted_components_target_color():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[60:90, 60:90] = True
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[10:30, 10:30] = (200, 40, 40)
    rgb[60:90, 60:90] = (40, 40, 200)
    hints = ExtractionHints(target_rgb=(210, 50, 50))
    selected = select_hinted_components(mask, rgb, hints)
    assert selected[20, 20]
    assert not selected[75, 75]


def test_select_hinted_components_single_component_untouched():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    hints = ExtractionHints(seed_point=(70.0, 70.0))
    selected = select_hinted_components(mask, rgb, hints)
    assert np.array_equal(selected, mask)
