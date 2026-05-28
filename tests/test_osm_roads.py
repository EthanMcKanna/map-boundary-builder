import unittest

import numpy as np

from map_boundary_builder.georef_transform import GeoreferenceTransform, mercator_to_lonlat
from map_boundary_builder.osm_roads import (
    RoadMatchResult,
    load_road_points_seed,
    read_road_refine_cache,
    score_georeference_transform,
    score_transform_batch,
    seed_road_points,
    write_road_refine_cache,
)


class RoadScoringTests(unittest.TestCase):
    def test_batch_scoring_matches_scalar_scoring(self) -> None:
        feature_distance = np.zeros((80, 90), dtype=np.float32)
        feature_distance[20:25, 30:55] = 2.5
        feature_distance[40:55, 10:28] = 8.0
        road_points = np.array(
            [
                [120.0, -90.0],
                [180.0, -120.0],
                [260.0, -155.0],
                [360.0, -210.0],
                [520.0, -260.0],
            ],
            dtype=float,
        )
        params = [
            (6.0, 0.0, 0.0, 0.0),
            (7.5, 0.05, 10.0, -20.0),
            (9.0, -0.08, -40.0, 15.0),
        ]

        batch_scores = score_transform_batch(road_points, feature_distance, params)

        for (scale, rotation, tx, ty), (batch_score, batch_count) in zip(params, batch_scores):
            lon, lat = mercator_to_lonlat(tx, ty)
            transform = GeoreferenceTransform(
                city="Test",
                lon=lon,
                lat=lat,
                origin_x_ratio=0.0,
                origin_y_ratio=0.0,
                meters_per_pixel=scale,
                rotation_radians=rotation,
                confidence=0.8,
                source="test",
            )
            scalar_score, scalar_count = score_georeference_transform(road_points, feature_distance, transform)
            self.assertEqual(batch_count, scalar_count)
            self.assertAlmostEqual(batch_score, scalar_score, places=7)

    def test_road_refine_cache_round_trips_result(self) -> None:
        transform = GeoreferenceTransform(
            city="Miami",
            lon=-80.2,
            lat=25.8,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=18.0,
            rotation_radians=0.01,
            confidence=0.91,
            source="ocr-georeference:nominatim-label-fit+osm-road-refine",
        )
        result = RoadMatchResult(transform=transform, score=0.7, sampled_points=1234, base_score=0.42)

        write_road_refine_cache("unit-test-road-cache", result)

        self.assertEqual(read_road_refine_cache("unit-test-road-cache"), result)

    def test_road_point_seed_contains_refinement_contexts(self) -> None:
        seed = load_road_points_seed()

        self.assertIn("4d5722451b742341f86a6928", seed)
        self.assertIn("c7c13d1754292efb8db6bb0f", seed)
        self.assertGreater(len(seed_road_points("4d5722451b742341f86a6928")), 5000)
        self.assertEqual(seed_road_points("missing-road-seed"), None)


if __name__ == "__main__":
    unittest.main()
