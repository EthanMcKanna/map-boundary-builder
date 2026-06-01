import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import map_boundary_builder.geocoder as geocoder


class GeocoderSeedTests(unittest.TestCase):
    def tearDown(self) -> None:
        geocoder._geocode_cached.cache_clear()

    def test_nominatim_seed_serves_without_network(self) -> None:
        key = geocoder.cache_file("Orlando", 3, "us").stem
        seed = {
            "version": 1,
            "nominatim": {
                key: [
                    {
                        "lon": "-81.3790",
                        "lat": "28.5421",
                        "display_name": "Orlando, Orange County, Florida, United States",
                        "boundingbox": ["28.35", "28.65", "-81.55", "-81.20"],
                        "importance": 0.7,
                        "addresstype": "city",
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "_GEOCODER_SEED", seed),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("Orlando", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].display_name, "Orlando, Orange County, Florida, United States")
        self.assertEqual(results[0].bbox, (-81.55, 28.35, -81.2, 28.65))

    def test_empty_nominatim_seed_can_fall_through_to_photon_seed(self) -> None:
        nominatim_key = geocoder.cache_file("Ped Pflugerville", 3, "us").stem
        photon_key = geocoder.photon_cache_file("Ped Pflugerville", 3, "us").stem
        seed = {
            "version": 1,
            "nominatim": {nominatim_key: []},
            "photon": {
                photon_key: {
                    "features": [
                        {
                            "geometry": {"coordinates": [-97.62, 30.44]},
                            "properties": {
                                "name": "Pflugerville",
                                "city": "Pflugerville",
                                "state": "Texas",
                                "country": "United States",
                                "countrycode": "US",
                                "osm_value": "city",
                                "extent": [-97.75, 30.55, -97.50, 30.30],
                            },
                        }
                    ]
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", seed),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("Ped Pflugerville", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].display_name, "Pflugerville, Texas, United States")
        self.assertEqual(results[0].bbox, (-97.75, 30.3, -97.5, 30.55))

    def test_block_network_env_returns_empty_uncached_miss_without_urlopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", {"version": 1, "nominatim": {}, "photon": {}}),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
                patch.dict("os.environ", {"MAP_BOUNDARY_BLOCK_NETWORK": "1"}),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("Definitely Missing Place", limit=1)

        self.assertEqual(results, [])

    def test_bundled_los_angeles_seed_serves_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("Playa Vista, Los Angeles", limit=1)

        self.assertEqual(len(results), 1)
        self.assertIn("Playa Vista", results[0].display_name)

    def test_bundled_nashville_seed_serves_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("North Nashville, Nashville", limit=1)

        self.assertEqual(len(results), 1)
        self.assertIn("North Nashville", results[0].display_name)

    def test_bundled_miami_seed_serves_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("Miami", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].display_name, "Miami, Miami-Dade County, Florida, United States")
        self.assertEqual(results[0].bbox, (-80.31976, 25.7090517, -80.139157, 25.8557827))

    def test_bundled_las_vegas_seed_serves_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("Las Vegas", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].display_name, "Las Vegas, Clark County, Nevada, United States")
        self.assertEqual(results[0].bbox, (-115.406575, 36.129554, -115.062066, 36.401481))

    def test_bundled_miami_label_seeds_serve_without_network(self) -> None:
        queries = [
            "Coral Gables, Miami",
            "Coral Gables, Florida",
            "Coral Gables",
            "Downtown Miami, Miami",
            "Downtown Miami, Florida",
            "Downtown Miami",
            "Downtown Brickell, Miami",
            "Downtown Brickell, Florida",
            "Downtown Brickell",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                for query in queries:
                    with self.subTest(query=query):
                        results = geocoder.geocode(query, limit=3)
                        self.assertTrue(results)
                        self.assertTrue(any("Miami" in result.display_name for result in results))
                        for result in results:
                            self.assertGreaterEqual(result.lon, -80.35)
                            self.assertLessEqual(result.lon, -80.10)
                            self.assertGreaterEqual(result.lat, 25.60)
                            self.assertLessEqual(result.lat, 25.90)

    def test_bundled_north_miami_beach_seed_serves_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                results = geocoder.geocode("North Miami Beach", limit=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].display_name,
            "North Miami Beach, Miami-Dade County, Florida, 33162, United States",
        )
        self.assertEqual(
            results[0].bbox,
            (-80.2085683, 25.9004315, -80.1308922, 25.9571715),
        )

    def test_bundled_miami_expansion_label_seeds_serve_without_network(self) -> None:
        expected_names = {
            "West Miami": "West Miami",
            "Miami Beach": "Miami Beach",
            "Little Havana": "Little Havana",
            "South Miami": "South Miami",
            "Miami Shores": "Miami Shores",
            "Brickell": "Brickell",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                for query, expected_name in expected_names.items():
                    with self.subTest(query=query):
                        results = geocoder.geocode(query, limit=2)
                        self.assertTrue(results)
                        self.assertEqual(results[0].display_name.split(",", 1)[0], expected_name)
                        self.assertIn("Miami", results[0].display_name)

    def test_bundled_miami_noise_misses_avoid_network(self) -> None:
        queries = [
            "North Miami Beach, Miami",
            "West Miami Coral Gables, Miami",
            "West Miami Coral Gables",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                for query in queries:
                    with self.subTest(query=query):
                        results = geocoder.geocode(query, limit=3)
                        self.assertEqual(results, [])

    def test_bundled_ann_arbor_label_seeds_serve_without_network(self) -> None:
        expected_names = {
            "Ann Arbor": "Ann Arbor",
            "Amtrak Station, Ann Arbor": "Ann Arbor Amtrak Station",
            "Michigan Union, Ann Arbor": "Michigan Union",
            "Nickols Arcade, Ann Arbor": "Nickels Arcade",
            "Ross Schoolof Business, Ann Arbor": "Ross School of Business Building",
            "Ann Arbor Farmer Market, Ann Arbor": "Ann Arbor Farmers Market",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                for query, expected_name in expected_names.items():
                    with self.subTest(query=query):
                        results = geocoder.geocode(query, limit=3)
                        self.assertTrue(results)
                        self.assertEqual(results[0].display_name.split(",", 1)[0], expected_name)
                        self.assertIn("Ann Arbor", results[0].display_name)

    def test_bundled_ann_arbor_noise_misses_avoid_network(self) -> None:
        queries = [
            "Whuon",
            "Hands",
            "Amtrak Station",
            "Farmer Market",
            "Uof Mmuseumof Art, Ann Arbor",
            "May Mobility Ovia, Ann Arbor",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(geocoder, "CACHE_DIR", Path(tmpdir) / "geocoder"),
                patch.object(geocoder, "PHOTON_CACHE_DIR", Path(tmpdir) / "photon"),
                patch.object(geocoder, "_GEOCODER_SEED", None),
                patch.object(geocoder, "urlopen", side_effect=AssertionError("network should not run")),
            ):
                geocoder._geocode_cached.cache_clear()
                for query in queries:
                    with self.subTest(query=query):
                        results = geocoder.geocode(query, limit=3)
                        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
