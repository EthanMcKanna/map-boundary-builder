import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import map_boundary_builder.osm_roads as osm_roads
from map_boundary_builder.georef_transform import GeoreferenceTransform, lonlat_to_mercator, mercator_to_lonlat
from map_boundary_builder.osm_roads import (
    ROAD_REFINE_MEMORY_CACHE_MAX,
    RoadMatchResult,
    _ROAD_REFINE_MEMORY_CACHE,
    feature_score_image,
    image_feature_distance,
    load_road_points_seed,
    read_road_refine_cache,
    refine_transform_with_osm_roads,
    road_refine_cache_key,
    road_points_source_digest,
    score_georeference_transform,
    search_near_transform,
    score_transform_batch_arrays_on_score_image,
    score_transform_batch,
    score_transform_batch_on_score_image,
    seed_road_points,
    write_road_refine_cache,
)


def unit_road_match_result() -> RoadMatchResult:
    return RoadMatchResult(
        transform=GeoreferenceTransform(
            city="Miami",
            lon=-80.2,
            lat=25.8,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=18.0,
            rotation_radians=0.01,
            confidence=0.91,
            source="ocr-georeference:nominatim-label-fit+osm-road-refine",
        ),
        score=0.7,
        sampled_points=1234,
        base_score=0.42,
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
        score_image_batch_scores = score_transform_batch_on_score_image(
            road_points,
            feature_score_image(feature_distance),
            params,
        )

        self.assertEqual(batch_scores, score_image_batch_scores)

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

    def test_search_near_transform_matches_exhaustive_batch_scores(self) -> None:
        rng = np.random.default_rng(42)
        feature_scores = rng.random((64, 72), dtype=np.float32)
        road_points = np.column_stack(
            (np.linspace(-160.0, 180.0, 80), np.linspace(-110.0, 130.0, 80))
        ).astype(np.float32)
        base_tx, base_ty = 8.0, -12.0
        lon, lat = mercator_to_lonlat(base_tx, base_ty)
        base = GeoreferenceTransform(
            city="Test",
            lon=lon,
            lat=lat,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=5.0,
            rotation_radians=0.02,
            confidence=0.8,
            source="test",
        )
        scale_multipliers = np.array([0.95, 1.0, 1.05])
        rotation_offsets = np.deg2rad(np.array([-0.5, 0.0, 0.5]))
        offset_meters = np.array([-18.0, 0.0, 18.0])

        result = search_near_transform(
            road_points,
            np.zeros_like(feature_scores, dtype=np.float32),
            base,
            base_tx,
            base_ty,
            feature_scores=feature_scores,
            scale_multipliers=scale_multipliers,
            rotation_offsets=rotation_offsets,
            offset_meters=offset_meters,
            min_count=1,
        )
        self.assertIsNotNone(result)
        assert result is not None

        params = np.asarray(
            [
                (
                    base.meters_per_pixel * scale_multiplier,
                    base.rotation_radians + rotation_offset,
                    base_tx + offset_x,
                    base_ty + offset_y,
                )
                for scale_multiplier in scale_multipliers
                for rotation_offset in rotation_offsets
                for offset_x in offset_meters
                for offset_y in offset_meters
            ],
            dtype=np.float32,
        )
        scores, counts = score_transform_batch_arrays_on_score_image(road_points, feature_scores, params)
        valid_scores = np.where(counts >= 1, scores, -np.inf)
        best_index = int(np.argmax(valid_scores))
        expected_scale, expected_rotation, expected_tx, expected_ty = params[best_index]
        actual_score, actual_count, actual = result
        actual_tx, actual_ty = lonlat_to_mercator(actual.lon, actual.lat)

        self.assertAlmostEqual(actual_score, float(scores[best_index]), places=7)
        self.assertEqual(actual_count, int(counts[best_index]))
        self.assertAlmostEqual(actual.meters_per_pixel, float(expected_scale), places=7)
        self.assertAlmostEqual(actual.rotation_radians, float(expected_rotation), places=7)
        self.assertAlmostEqual(actual_tx, float(expected_tx), places=5)
        self.assertAlmostEqual(actual_ty, float(expected_ty), places=5)

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

    def test_road_refine_memory_cache_evicts_oldest_entries(self) -> None:
        result = unit_road_match_result()

        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.object(osm_roads, "ROAD_REFINE_CACHE_DIR", Path(cache_dir)):
                _ROAD_REFINE_MEMORY_CACHE.clear()
                try:
                    for index in range(ROAD_REFINE_MEMORY_CACHE_MAX + 1):
                        write_road_refine_cache(f"key-{index}", result)

                    self.assertNotIn("key-0", _ROAD_REFINE_MEMORY_CACHE)
                    self.assertIn(f"key-{ROAD_REFINE_MEMORY_CACHE_MAX}", _ROAD_REFINE_MEMORY_CACHE)
                    self.assertEqual(len(_ROAD_REFINE_MEMORY_CACHE), ROAD_REFINE_MEMORY_CACHE_MAX)
                finally:
                    _ROAD_REFINE_MEMORY_CACHE.clear()

    def test_road_refine_memory_cache_refreshes_recent_reads(self) -> None:
        result = unit_road_match_result()

        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.object(osm_roads, "ROAD_REFINE_CACHE_DIR", Path(cache_dir)):
                _ROAD_REFINE_MEMORY_CACHE.clear()
                try:
                    for index in range(ROAD_REFINE_MEMORY_CACHE_MAX):
                        write_road_refine_cache(f"key-{index}", result)
                    self.assertEqual(read_road_refine_cache("key-0"), result)
                    write_road_refine_cache("new-key", result)

                    self.assertIn("key-0", _ROAD_REFINE_MEMORY_CACHE)
                    self.assertNotIn("key-1", _ROAD_REFINE_MEMORY_CACHE)
                    self.assertIn("new-key", _ROAD_REFINE_MEMORY_CACHE)
                finally:
                    _ROAD_REFINE_MEMORY_CACHE.clear()

    def test_road_refine_cache_key_ignores_non_feature_pixel_noise(self) -> None:
        rgb = np.full((80, 80, 3), 245, dtype=np.uint8)
        rgb[:, 40:42] = (190, 190, 190)
        noisy = rgb.copy()
        noisy[0, 0] = (250, 245, 245)
        transform = GeoreferenceTransform(
            city="Phoenix",
            lon=-112.0,
            lat=33.4,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=25.0,
            rotation_radians=0.0,
            confidence=0.91,
            source="ocr-georeference:nominatim-label-fit",
        )
        center = type("Center", (), {"bbox": (-112.2, 33.2, -111.8, 33.7)})()

        clean_key = road_refine_cache_key(
            image_feature_distance(rgb),
            center,
            transform,
            lock_scale=False,
            road_source_digest="roads-a",
        )
        noisy_key = road_refine_cache_key(
            image_feature_distance(noisy),
            center,
            transform,
            lock_scale=False,
            road_source_digest="roads-a",
        )

        self.assertEqual(clean_key, noisy_key)

    def test_road_refine_cache_key_depends_on_road_source_digest(self) -> None:
        feature_distance = np.zeros((20, 20), dtype=np.float32)
        transform = GeoreferenceTransform(
            city="Phoenix",
            lon=-112.0,
            lat=33.4,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=25.0,
            rotation_radians=0.0,
            confidence=0.91,
            source="ocr-georeference:nominatim-label-fit",
        )
        center = type("Center", (), {"bbox": (-112.2, 33.2, -111.8, 33.7)})()

        key_a = road_refine_cache_key(
            feature_distance,
            center,
            transform,
            lock_scale=False,
            road_source_digest="roads-a",
        )
        key_b = road_refine_cache_key(
            feature_distance,
            center,
            transform,
            lock_scale=False,
            road_source_digest="roads-b",
        )

        self.assertNotEqual(key_a, key_b)

    def test_road_points_source_digest_tracks_overpass_cache_bytes(self) -> None:
        bbox = (-10.0, -10.0, -9.9, -9.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(osm_roads, "CACHE_DIR", Path(tmpdir)),
                patch.object(osm_roads, "seed_road_points", return_value=None),
            ):
                cache_path = osm_roads.overpass_cache_file(bbox)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text('{"elements":[]}\n')
                first = road_points_source_digest(bbox)
                cache_path.write_text('{"elements":[{"id":1}]}\n')
                second = road_points_source_digest(bbox)

        self.assertNotEqual(first, second)

    def test_load_road_points_cache_depends_on_road_source_digest(self) -> None:
        bbox = (-10.0, -10.0, -9.9, -9.9)
        short_road = (
            '{"elements":[{"geometry":[{"lon":-10.0,"lat":-10.0},{"lon":-9.999,"lat":-10.0}]}]}\n'
        )
        long_road = (
            '{"elements":[{"geometry":[{"lon":-10.0,"lat":-10.0},{"lon":-9.98,"lat":-10.0}]}]}\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(osm_roads, "CACHE_DIR", Path(tmpdir)),
                patch.object(osm_roads, "seed_road_points", return_value=None),
            ):
                osm_roads.load_road_points.cache_clear()
                osm_roads.load_overpass_roads.cache_clear()
                try:
                    cache_path = osm_roads.overpass_cache_file(bbox)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(short_road)
                    first = osm_roads.load_road_points(bbox)
                    cache_path.write_text(long_road)
                    second = osm_roads.load_road_points(bbox)
                finally:
                    osm_roads.load_road_points.cache_clear()
                    osm_roads.load_overpass_roads.cache_clear()

        self.assertGreater(len(second), len(first))

    def test_road_refinement_cache_reuses_same_feature_field(self) -> None:
        rgb = np.full((80, 80, 3), 245, dtype=np.uint8)
        rgb[:, 40:42] = (190, 190, 190)
        noisy = rgb.copy()
        noisy[0, 0] = (250, 245, 245)
        self.assertTrue(np.array_equal(image_feature_distance(rgb), image_feature_distance(noisy)))
        initial = GeoreferenceTransform(
            city="Phoenix",
            lon=-112.0,
            lat=33.4,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=25.0,
            rotation_radians=0.0,
            confidence=0.84,
            source="ocr-georeference:nominatim-label-fit",
        )
        refined = GeoreferenceTransform(
            city="Phoenix",
            lon=-112.01,
            lat=33.41,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=24.5,
            rotation_radians=0.01,
            confidence=0.84,
            source="ocr-georeference:nominatim-label-fit",
        )
        center = type("Center", (), {"bbox": (-112.2, 33.2, -111.8, 33.7)})()
        road_points = np.column_stack(
            (np.linspace(-10000.0, 10000.0, 1200), np.linspace(-5000.0, 5000.0, 1200))
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as cache_dir:
            osm_roads._ROAD_REFINE_MEMORY_CACHE.clear()
            try:
                with (
                    patch.object(osm_roads, "ROAD_REFINE_CACHE_DIR", Path(cache_dir)),
                    patch.object(osm_roads, "road_points_source_digest", return_value="roads-a"),
                    patch.object(osm_roads, "load_road_points", return_value=road_points) as load_points,
                    patch.object(
                        osm_roads,
                        "score_georeference_transform_on_score_image",
                        return_value=(0.5, 1200),
                    ) as score,
                    patch.object(osm_roads, "search_near_transform", return_value=(0.6, 1200, refined)) as search,
                ):
                    first = refine_transform_with_osm_roads(rgb, center, initial)
                    first_counts = (load_points.call_count, score.call_count, search.call_count)
                    second = refine_transform_with_osm_roads(noisy, center, initial)

                self.assertEqual(first, second)
                self.assertGreater(first_counts[0], 0)
                self.assertGreater(first_counts[1], 0)
                self.assertGreater(first_counts[2], 0)
                self.assertEqual((load_points.call_count, score.call_count, search.call_count), first_counts)
                self.assertEqual(len(list(Path(cache_dir).glob("*.json"))), 1)
            finally:
                osm_roads._ROAD_REFINE_MEMORY_CACHE.clear()

    def test_road_refinement_cache_misses_when_road_source_changes(self) -> None:
        rgb = np.full((80, 80, 3), 245, dtype=np.uint8)
        rgb[:, 40:42] = (190, 190, 190)
        initial = GeoreferenceTransform(
            city="Phoenix",
            lon=-112.0,
            lat=33.4,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=25.0,
            rotation_radians=0.0,
            confidence=0.84,
            source="ocr-georeference:nominatim-label-fit",
        )
        refined = GeoreferenceTransform(
            city="Phoenix",
            lon=-112.01,
            lat=33.41,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=24.5,
            rotation_radians=0.01,
            confidence=0.84,
            source="ocr-georeference:nominatim-label-fit",
        )
        center = type("Center", (), {"bbox": (-112.2, 33.2, -111.8, 33.7)})()
        road_points = np.column_stack(
            (np.linspace(-10000.0, 10000.0, 1200), np.linspace(-5000.0, 5000.0, 1200))
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as cache_dir:
            osm_roads._ROAD_REFINE_MEMORY_CACHE.clear()
            try:
                with (
                    patch.object(osm_roads, "ROAD_REFINE_CACHE_DIR", Path(cache_dir)),
                    patch.object(
                        osm_roads,
                        "road_points_source_digest",
                        side_effect=["roads-a", "roads-a", "roads-b", "roads-b"],
                    ),
                    patch.object(osm_roads, "load_road_points", return_value=road_points) as load_points,
                    patch.object(
                        osm_roads,
                        "score_georeference_transform_on_score_image",
                        return_value=(0.5, 1200),
                    ),
                    patch.object(osm_roads, "search_near_transform", return_value=(0.6, 1200, refined)),
                ):
                    first = refine_transform_with_osm_roads(rgb, center, initial)
                    second = refine_transform_with_osm_roads(rgb, center, initial)

                self.assertEqual(first, second)
                self.assertEqual(load_points.call_count, 2)
                self.assertEqual(len(list(Path(cache_dir).glob("*.json"))), 2)
            finally:
                osm_roads._ROAD_REFINE_MEMORY_CACHE.clear()

    def test_road_refinement_accepts_precomputed_feature_distance(self) -> None:
        rgb = np.full((40, 40, 3), 245, dtype=np.uint8)
        feature_distance = image_feature_distance(rgb)
        initial = GeoreferenceTransform(
            city="Phoenix",
            lon=-112.0,
            lat=33.4,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=25.0,
            rotation_radians=0.0,
            confidence=0.84,
            source="ocr-georeference:nominatim-label-fit",
        )
        center = type("Center", (), {"bbox": (-112.2, 33.2, -111.8, 33.7)})()

        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch.object(osm_roads, "ROAD_REFINE_CACHE_DIR", Path(cache_dir)),
                patch.object(osm_roads, "image_feature_distance", side_effect=AssertionError("feature should be reused")),
                patch.object(osm_roads, "load_road_points", return_value=np.empty((0, 2))),
            ):
                self.assertIsNone(
                    refine_transform_with_osm_roads(
                        rgb,
                        center,
                        initial,
                        feature_distance=feature_distance,
                    )
                )

    def test_road_point_seed_contains_refinement_contexts(self) -> None:
        seed = load_road_points_seed()

        self.assertIn("4d5722451b742341f86a6928", seed)
        self.assertIn("c7c13d1754292efb8db6bb0f", seed)
        self.assertIn("93e43ee6c074669e9df90297", seed)
        self.assertGreater(len(seed_road_points("4d5722451b742341f86a6928")), 5000)
        self.assertGreater(len(seed_road_points("93e43ee6c074669e9df90297")), 5000)
        self.assertEqual(seed_road_points("missing-road-seed"), None)

    def test_block_network_env_returns_empty_uncached_overpass_miss_without_urlopen(self) -> None:
        bbox = (-10.0, -10.0, -9.9, -9.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(osm_roads, "CACHE_DIR", Path(tmpdir) / "overpass"),
                patch.object(osm_roads, "urlopen", side_effect=AssertionError("network should not run")),
                patch.dict("os.environ", {"MAP_BOUNDARY_BLOCK_NETWORK": "1"}),
            ):
                osm_roads.load_overpass_roads.cache_clear()
                payload = osm_roads.load_overpass_roads(bbox)

        self.assertEqual(payload, {"elements": []})


if __name__ == "__main__":
    unittest.main()
