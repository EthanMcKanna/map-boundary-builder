import unittest
from unittest.mock import Mock, patch

import numpy as np

from map_boundary_builder.runtime_warmup import prewarm_generation_runtime, warm_segmentation_runtime


class RuntimeWarmupTests(unittest.TestCase):
    def test_warm_segmentation_runtime_extracts_the_fill(self) -> None:
        profile = warm_segmentation_runtime()

        self.assertIn(profile["engine"], {"model", "auto-fill-fallback"})
        self.assertEqual(profile["contour_count"], 1)
        self.assertGreater(profile["coverage_ratio"], 0.1)
        self.assertGreater(profile["confidence"], 0.5)

    def test_prewarm_generation_runtime_reports_segmentation_warmup(self) -> None:
        with (
            patch("map_boundary_builder.geocoder.load_geocoder_seed", return_value={"San Francisco": object()}),
            patch("map_boundary_builder.osm_places.load_osm_places_seed", return_value={"sf": object()}),
            patch("map_boundary_builder.osm_roads.load_road_points_seed", return_value={"sf": np.zeros((2, 2))}),
            patch("map_boundary_builder.osm_roads.seeded_road_points_source_digest", return_value="digest") as warm_roads,
            patch(
                "map_boundary_builder.runtime_warmup.warm_segmentation_runtime",
                return_value={"engine": "model", "contour_count": 1},
            ) as warm_segmentation,
            patch("map_boundary_builder.ocr.warm_rapidocr_runtime", Mock(return_value=True)) as warm_ocr,
        ):
            profile = prewarm_generation_runtime()

        self.assertEqual(profile["status"], "ok")
        self.assertEqual(profile["geocoder_seed_entries"], 1)
        self.assertEqual(profile["road_seed_entries"], 1)
        self.assertEqual(profile["road_seed_digest_entries"], 1)
        self.assertTrue(profile["segmentation_warmed"])
        self.assertEqual(profile["segmentation_engine"], "model")
        self.assertEqual(profile["segmentation_contour_count"], 1)
        self.assertIn("segmentation_s", profile)
        self.assertTrue(profile["rapidocr_inference_warmed"])
        warm_roads.assert_called_once_with("sf")
        warm_segmentation.assert_called_once_with()
        warm_ocr.assert_called_once_with()
