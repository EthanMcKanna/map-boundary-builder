from pathlib import Path

from map_boundary_builder.pipeline_version import (
    get_pipeline_version,
    pipeline_version_dependency_versions,
    pipeline_version_sources,
)


def test_pipeline_version_is_stable_hash() -> None:
    first = get_pipeline_version()
    second = get_pipeline_version()

    assert first == second
    assert first.startswith("pipeline-")
    assert len(first) == len("pipeline-") + 16


def test_pipeline_version_tracks_runtime_dependency_versions() -> None:
    versions = dict(pipeline_version_dependency_versions())

    assert versions["onnxruntime"]
    assert "opencv-python" in versions
    assert versions["opencv-python-headless"]
    assert versions["cv2"]
    assert versions["rapidocr-onnxruntime"]


def test_pipeline_version_tracks_api_handler_when_present() -> None:
    sources = dict(pipeline_version_sources())

    if Path("api/index.py").exists():
        assert "api/index.py" in sources


def test_pipeline_version_tracks_runtime_config() -> None:
    sources = dict(pipeline_version_sources())

    assert "runtime_config.py" in sources
