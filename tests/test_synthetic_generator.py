import json
from pathlib import Path

import numpy as np
from PIL import Image

from map_boundary_builder.synthetic import (
    DEFAULT_OVERLAY_STYLES,
    SyntheticSceneConfig,
    generate_synthetic_dataset,
    generate_synthetic_sample,
)


def test_generate_synthetic_sample_writes_required_artifacts(tmp_path: Path) -> None:
    result = generate_synthetic_sample(
        tmp_path,
        SyntheticSceneConfig(
            provider="Waymo",
            service_area="Phoenix",
            variant="outline",
            width=320,
            height=220,
            seed=42,
            overlay_style=DEFAULT_OVERLAY_STYLES[-1],
            touch_border=True,
            include_ui_chrome=True,
            include_hole=True,
        ),
    )

    result.sample.validate_required_artifacts(tmp_path)
    assert result.sample.image_size == (320, 220)
    assert result.sample.properties["renderer"] == "procedural-pillow"
    assert result.mask_area_px > 0

    screenshot = Image.open(tmp_path / result.sample.artifacts.screenshot)
    overlay = Image.open(tmp_path / result.sample.artifacts.overlay)
    mask = Image.open(tmp_path / result.sample.artifacts.mask)

    assert screenshot.size == (320, 220)
    assert overlay.size == (320, 220)
    assert mask.size == (320, 220)
    assert set(np.unique(np.asarray(mask))).issubset({0, 255})

    geojson = json.loads((tmp_path / result.sample.artifacts.geojson).read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"
    assert geojson["metadata"]["pixel_geometry"]["type"] == "Polygon"
    assert geojson["metadata"]["image_width"] == 320


def test_generate_synthetic_sample_is_deterministic(tmp_path: Path) -> None:
    config = SyntheticSceneConfig(width=240, height=180, seed=7, variant="deterministic")
    first = generate_synthetic_sample(tmp_path / "first", config)
    second = generate_synthetic_sample(tmp_path / "second", config)

    first_mask = (tmp_path / "first" / first.sample.artifacts.mask).read_bytes()
    second_mask = (tmp_path / "second" / second.sample.artifacts.mask).read_bytes()
    first_geojson = (tmp_path / "first" / first.sample.artifacts.geojson).read_text(encoding="utf-8")
    second_geojson = (tmp_path / "second" / second.sample.artifacts.geojson).read_text(encoding="utf-8")

    assert first.sample.content_hash == second.sample.content_hash
    assert first.sample.sample_id == second.sample.sample_id
    assert first_mask == second_mask
    assert first_geojson == second_geojson


def test_generate_synthetic_dataset_writes_manifest(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=5, seed=100, width=200, height=140)

    manifest.validate_required_artifacts(tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert saved["properties"]["count"] == 5
    assert len(saved["samples"]) == 5
    assert len({sample.sample_id for sample in manifest.samples}) == 5
