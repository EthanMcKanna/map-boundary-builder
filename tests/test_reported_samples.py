from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import shape

import map_boundary_builder.geocoder as geocoder
import map_boundary_builder.osm_places as osm_places
from map_boundary_builder.runner import BoundaryBuildOptions, build_boundary


FIXTURES = Path(__file__).parent / "fixtures" / "reported_samples"


@pytest.fixture
def offline_map_services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAP_BOUNDARY_BLOCK_NETWORK", "1")
    monkeypatch.setattr(geocoder, "CACHE_DIR", tmp_path / "geocoder")
    monkeypatch.setattr(geocoder, "PHOTON_CACHE_DIR", tmp_path / "geocoder-photon")
    monkeypatch.setattr(osm_places, "CACHE_DIR", tmp_path / "overpass-places")
    geocoder._geocode_cached.cache_clear()
    osm_places.load_place_points.cache_clear()
    osm_places.load_overpass_places.cache_clear()
    yield
    geocoder._geocode_cached.cache_clear()
    osm_places.load_place_points.cache_clear()
    osm_places.load_overpass_places.cache_clear()


def test_issue_7_satellite_regional_map_completes_offline(
    tmp_path: Path,
    offline_map_services,
) -> None:
    result = build_boundary(
        FIXTURES / "issue-7-input.png",
        None,
        tmp_path / "issue-7.geojson",
        options=BoundaryBuildOptions(filename_hint="image.png"),
    )

    summary = result.summary
    assert summary["style"] == "auto-fill"
    assert summary["coverage_ratio"] == pytest.approx(0.152524, abs=0.003)
    assert summary["control_points"] >= 3
    assert summary["combined_confidence"] >= 0.82
    assert summary["georeference_source"] == "ocr-georeference:regional-admin-label-fit"
    assert summary["bbox"] == pytest.approx(
        [-96.1007927, 30.5650361, -95.9522852, 30.6625853],
        abs=0.003,
    )
    assert abs(summary["rotation_degrees"]) <= 1.0
    assert summary["p90_residual_m"] <= 100.0
    assert shape(result.geojson["features"][0]["geometry"]).is_valid


def test_issue_13_miami_context_card_completes_offline(
    tmp_path: Path,
    offline_map_services,
) -> None:
    result = build_boundary(
        FIXTURES / "issue-13-input.jpeg",
        None,
        tmp_path / "issue-13.geojson",
        options=BoundaryBuildOptions(filename_hint="IMG_3899.jpeg"),
    )

    summary = result.summary
    assert summary["city"] == "Miami"
    assert summary["style"] == "light-fill"
    assert summary["coverage_ratio"] == pytest.approx(0.302475, abs=0.003)
    assert summary["control_points"] >= 3
    assert summary["combined_confidence"] >= 0.86
    assert summary["georeference_source"] == "ocr-georeference:nominatim-label-fit"
    assert summary["bbox"] == pytest.approx(
        [-80.3712696, 25.7327833, -80.2402685, 25.8099164],
        abs=0.003,
    )
    assert shape(result.geojson["features"][0]["geometry"]).is_valid
