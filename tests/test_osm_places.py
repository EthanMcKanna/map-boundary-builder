import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import map_boundary_builder.osm_places as osm_places


class OsmPlacesSeedTests(unittest.TestCase):
    def tearDown(self) -> None:
        osm_places.load_overpass_places.cache_clear()
        osm_places.load_place_points.cache_clear()

    def test_overpass_places_seed_serves_without_network(self) -> None:
        bbox = (-81.5, 28.3, -81.2, 28.7)
        key = osm_places.overpass_places_cache_file(bbox).stem
        seed = {
            "version": 1,
            "overpass_places": {
                key: {
                    "elements": [
                        {
                            "type": "node",
                            "id": 1,
                            "lat": 28.5421,
                            "lon": -81.379,
                            "tags": {"name": "Orlando", "place": "city"},
                        }
                    ]
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(osm_places, "CACHE_DIR", Path(tmpdir) / "overpass-places"),
                patch.object(osm_places, "_OSM_PLACES_SEED", seed),
                patch.object(osm_places, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                osm_places.load_overpass_places.cache_clear()
                osm_places.load_place_points.cache_clear()
                places = osm_places.load_place_points(bbox)

        self.assertEqual(len(places), 1)
        self.assertEqual(places[0].name, "Orlando")
        self.assertEqual(places[0].place_type, "city")

    def test_bundled_los_angeles_places_seed_serves_without_network(self) -> None:
        bbox = (-119.322708895, 33.72594655139415, -118.05137880500003, 34.39134623761558)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(osm_places, "CACHE_DIR", Path(tmpdir) / "overpass-places"),
                patch.object(osm_places, "_OSM_PLACES_SEED", None),
                patch.object(osm_places, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                osm_places.load_overpass_places.cache_clear()
                osm_places.load_place_points.cache_clear()
                places = osm_places.load_place_points(bbox)

        place_names = {place.name for place in places}
        self.assertIn("Los Angeles", place_names)
        self.assertIn("Hollywood", place_names)

    def test_bundled_bay_area_regional_places_seed_serves_without_network(self) -> None:
        bbox = (-122.6639845, 37.1319772, -121.6341266, 37.9903379)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(osm_places, "CACHE_DIR", Path(tmpdir) / "overpass-places"),
                patch.object(osm_places, "_OSM_PLACES_SEED", None),
                patch.object(osm_places, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                osm_places.load_overpass_places.cache_clear()
                osm_places.load_place_points.cache_clear()
                places = osm_places.load_place_points(bbox)

        place_names = {place.name for place in places}
        self.assertIn("Menlo Park", place_names)
        self.assertIn("Sunnyvale", place_names)


if __name__ == "__main__":
    unittest.main()
