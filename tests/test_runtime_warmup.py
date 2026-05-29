import unittest
from unittest.mock import Mock, patch

from map_boundary_builder.runtime_warmup import prewarm_generation_runtime, warm_extraction_runtime


class RuntimeWarmupTests(unittest.TestCase):
    def test_warm_extraction_runtime_exercises_bright_blue_extraction(self) -> None:
        profile = warm_extraction_runtime()

        self.assertEqual(profile["style"], "bright-blue")
        self.assertEqual(profile["contour_count"], 1)
        self.assertGreater(profile["coverage_ratio"], 0.2)
        self.assertGreater(profile["confidence"], 0.9)

    def test_prewarm_generation_runtime_reports_extraction_warmup(self) -> None:
        with (
            patch("map_boundary_builder.catalog_match.load_catalog_entries", return_value=[object(), object()]),
            patch("map_boundary_builder.geocoder.load_geocoder_seed", return_value={"San Francisco": object()}),
            patch("map_boundary_builder.osm_places.load_osm_places_seed", return_value={"sf": object()}),
            patch("map_boundary_builder.osm_roads.load_road_points_seed", return_value={"sf": object()}),
            patch(
                "map_boundary_builder.runtime_warmup.warm_extraction_runtime",
                return_value={"style": "bright-blue", "contour_count": 1},
            ) as warm_extraction,
            patch("map_boundary_builder.ocr.warm_rapidocr_runtime", Mock(return_value=True)) as warm_ocr,
        ):
            profile = prewarm_generation_runtime()

        self.assertEqual(profile["status"], "ok")
        self.assertEqual(profile["catalog_entries"], 2)
        self.assertEqual(profile["geocoder_seed_entries"], 1)
        self.assertTrue(profile["extraction_warmed"])
        self.assertEqual(profile["extraction_style"], "bright-blue")
        self.assertEqual(profile["extraction_contour_count"], 1)
        self.assertIn("extraction_s", profile)
        self.assertTrue(profile["rapidocr_inference_warmed"])
        warm_extraction.assert_called_once_with()
        warm_ocr.assert_called_once_with()

    def test_prewarm_generation_runtime_marks_rapidocr_warm_failure_unhealthy(self) -> None:
        with (
            patch("map_boundary_builder.catalog_match.load_catalog_entries", return_value=[]),
            patch("map_boundary_builder.geocoder.load_geocoder_seed", return_value={}),
            patch("map_boundary_builder.osm_places.load_osm_places_seed", return_value={}),
            patch("map_boundary_builder.osm_roads.load_road_points_seed", return_value={}),
            patch(
                "map_boundary_builder.runtime_warmup.warm_extraction_runtime",
                return_value={"style": "bright-blue", "contour_count": 1},
            ),
            patch("map_boundary_builder.ocr.warm_rapidocr_runtime", Mock(return_value=False)),
            patch("map_boundary_builder.ocr.rapidocr_runtime_warm_error", Mock(return_value="RuntimeError: cold install")),
        ):
            profile = prewarm_generation_runtime()

        self.assertEqual(profile["status"], "error")
        self.assertFalse(profile["rapidocr_inference_warmed"])
        self.assertEqual(profile["error"], "RuntimeError: cold install")
