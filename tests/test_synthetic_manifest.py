from pathlib import Path

import pytest

from map_boundary_builder.synthetic import (
    OverlayStyleMetadata,
    SyntheticArtifactPaths,
    SyntheticDatasetManifest,
    SyntheticSampleMetadata,
    deterministic_content_hash,
    deterministic_sample_id,
)


def make_sample() -> SyntheticSampleMetadata:
    return SyntheticSampleMetadata.create(
        provider="Waymo",
        service_area="Phoenix",
        variant="default",
        image_size=(1200, 800),
        overlay_style=OverlayStyleMetadata(
            name="waymo-blue",
            fill_color="#2d7ff9",
            fill_opacity=0.42,
            stroke_color="#123456",
            stroke_width_px=2.5,
        ),
        artifacts=SyntheticArtifactPaths(
            screenshot="samples/phoenix.png",
            overlay="samples/phoenix-overlay.png",
            mask="samples/phoenix-mask.png",
            geojson="samples/phoenix.geojson",
        ),
        base_map="openfreemap",
        seed=17,
        generator_version="test-generator",
        properties={"catalog_slug": "phoenix-waymo"},
    )


def test_deterministic_sample_ids_and_hashes_are_stable() -> None:
    first = make_sample()
    second = make_sample()

    assert first.sample_id == second.sample_id
    assert first.content_hash == second.content_hash
    assert first.recompute_content_hash() == first.content_hash
    assert first.sample_id.startswith("synthetic-waymo-phoenix-default-")
    assert deterministic_sample_id("Waymo", "Phoenix", "default") == deterministic_sample_id(
        "Waymo", "Phoenix", "default"
    )
    assert deterministic_content_hash({"b": 2, "a": 1}) == deterministic_content_hash({"a": 1, "b": 2})


def test_sample_metadata_round_trips_through_json(tmp_path: Path) -> None:
    sample = make_sample()
    manifest_path = tmp_path / "sample.json"

    sample.write_json(manifest_path)
    loaded = SyntheticSampleMetadata.read_json(manifest_path)

    assert loaded.to_dict() == sample.to_dict()
    assert SyntheticSampleMetadata.from_json(sample.to_json()).to_dict() == sample.to_dict()


def test_dataset_manifest_round_trips_and_validates_artifacts(tmp_path: Path) -> None:
    sample = make_sample()
    sample_root = tmp_path / "samples"
    sample_root.mkdir()
    for name in ("phoenix.png", "phoenix-overlay.png", "phoenix-mask.png", "phoenix.geojson"):
        (sample_root / name).write_text("artifact", encoding="utf-8")

    manifest = SyntheticDatasetManifest(
        name="unit-test-synthetic-boundaries",
        version="2026-06-20",
        samples=[sample],
    )
    manifest_path = tmp_path / "manifest.json"

    manifest.validate_required_artifacts(tmp_path)
    manifest.write_json(manifest_path)
    loaded = SyntheticDatasetManifest.read_json(manifest_path)

    assert loaded.to_dict() == manifest.to_dict()


def test_required_artifact_validation_reports_missing_paths(tmp_path: Path) -> None:
    sample = make_sample()
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "phoenix.png").write_text("artifact", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as context:
        sample.validate_required_artifacts(tmp_path)

    message = str(context.value)
    assert "phoenix-overlay.png" in message
    assert "phoenix-mask.png" in message
    assert "phoenix.geojson" in message
