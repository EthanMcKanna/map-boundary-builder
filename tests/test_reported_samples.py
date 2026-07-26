from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import shape

import map_boundary_builder.geocoder as geocoder
import map_boundary_builder.osm_places as osm_places
from map_boundary_builder.pipeline import run_pipeline


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


def geojson_bbox(geojson: dict) -> list[float]:
    geometry = shape(geojson["features"][0]["geometry"])
    return list(geometry.bounds)


def test_issue_7_satellite_regional_map_completes_offline(
    tmp_path: Path,
    offline_map_services,
) -> None:
    result = run_pipeline(
        FIXTURES / "issue-7-input.png",
        output_path=tmp_path / "issue-7.geojson",
    )

    assert result.status == "complete"
    extraction = result.summary["extraction"]
    # v3 (trained with admin-shading and textured-basemap scenes) recovers the
    # originally user-validated extraction of the green Reinvestment Zone.
    assert extraction["coverage_ratio"] == pytest.approx(0.156, abs=0.05)
    georeference = result.summary["georeference"]
    assert georeference["control_points"] >= 3
    assert geojson_bbox(result.geojson) == pytest.approx(
        [-96.1008, 30.5650, -95.9523, 30.6626],
        abs=0.02,
    )
    assert shape(result.geojson["features"][0]["geometry"]).is_valid


def test_issue_13_miami_context_card_completes_offline(
    tmp_path: Path,
    offline_map_services,
) -> None:
    result = run_pipeline(
        FIXTURES / "issue-13-input.jpeg",
        output_path=tmp_path / "issue-13.geojson",
    )

    assert result.status == "complete"
    assert "Miami" in (result.summary["georeference"]["city"] or "")
    extraction = result.summary["extraction"]
    # The model keeps the Hialeah triangle the old light-fill extractor
    # clipped, so coverage sits slightly above the pre-revamp expectation.
    assert extraction["coverage_ratio"] == pytest.approx(0.36, abs=0.05)
    assert result.summary["georeference"]["control_points"] >= 3
    assert geojson_bbox(result.geojson) == pytest.approx(
        [-80.372, 25.732, -80.255, 25.810],
        abs=0.02,
    )
    assert shape(result.geojson["features"][0]["geometry"]).is_valid
