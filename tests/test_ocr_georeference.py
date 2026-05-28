import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

import map_boundary_builder.ocr as ocr_module
from map_boundary_builder.georeference import (
    CityContext,
    GeoreferenceResult,
    LabelGeocodeCandidate,
    candidate_place_labels,
    direct_city_contexts_from_labels,
    geocode_many,
    geocode_contexts,
    georeference_from_labels,
    has_reliable_candidate_cluster,
    infer_city_contexts,
    is_noisy_regional_control_query,
    is_noisy_poi_query,
    is_reliable_single_token_context,
    place_query_text,
    place_tokens,
    single_tokens_supported_by_fuller_labels,
)
from map_boundary_builder.geocoder import GeocodeResult
from map_boundary_builder.georef_transform import GeoreferenceTransform
from map_boundary_builder.ocr import (
    OcrLabel,
    extract_ocr_labels,
    group_stacked_labels,
    ocr_cache_key,
    rapidocr_input_image,
    rapidocr_items_to_labels,
    read_ocr_cache,
    write_ocr_cache,
)
from map_boundary_builder.runner import fit_georeference, rank_road_context_queries


class OcrGroupingTests(unittest.TestCase):
    def test_stacked_labels_require_nearby_rows(self) -> None:
        labels = [
            OcrLabel("Dallas", x=342, y=293.5, width=70, height=19, confidence=96),
            OcrLabel("DEEP", x=399, y=385, width=32, height=8, confidence=94),
        ]

        grouped = group_stacked_labels(labels)

        self.assertNotIn("Dallas DEEP", {label.text for label in grouped})

    def test_stacked_labels_keep_tight_multiline_places(self) -> None:
        labels = [
            OcrLabel("NORTH", x=257, y=149.5, width=46, height=9, confidence=94),
            OcrLabel("OAKLAWN", x=257, y=163, width=68, height=8, confidence=90),
        ]

        grouped = group_stacked_labels(labels)

        self.assertIn("NORTH OAKLAWN", {label.text for label in grouped})

    def test_rapidocr_items_are_converted_to_ocr_labels(self) -> None:
        labels = rapidocr_items_to_labels(
            [
                (
                    [[10, 20], [110, 24], [108, 54], [12, 50]],
                    "Miami Beach",
                    0.96,
                ),
                (
                    [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "12",
                    0.99,
                ),
            ]
        )

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].text, "Miami Beach")
        self.assertAlmostEqual(labels[0].x, 60.0)
        self.assertAlmostEqual(labels[0].y, 37.0)
        self.assertAlmostEqual(labels[0].confidence, 96.0)

    def test_ocr_label_cache_round_trips_labels(self) -> None:
        label = OcrLabel("Miami Beach", x=60, y=37, width=100, height=34, confidence=96)

        write_ocr_cache("unit-test-ocr-cache", [label])

        self.assertEqual(read_ocr_cache("unit-test-ocr-cache"), (label,))

    def test_rapidocr_input_image_downscales_when_configured(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_MAX_DIMENSION", 10):
                ocr_path, scale_x, scale_y = rapidocr_input_image(image_path)

            try:
                self.assertNotEqual(ocr_path, image_path)
                with Image.open(ocr_path) as resized:
                    self.assertEqual(resized.size, (10, 5))
                self.assertAlmostEqual(scale_x, 0.5)
                self.assertAlmostEqual(scale_y, 0.5)
            finally:
                ocr_path.unlink(missing_ok=True)

    def test_ocr_cache_key_depends_on_rapidocr_detector_limit(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_DET_LIMIT_SIDE_LEN", 640):
                key_640 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_DET_LIMIT_SIDE_LEN", 736):
                key_736 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_640, key_736)

    def test_ocr_cache_key_depends_on_rapidocr_classifier_batch(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_CLS_BATCH_NUM", 6):
                key_6 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_CLS_BATCH_NUM", 24):
                key_24 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_6, key_24)

    def test_extract_ocr_labels_does_not_rerun_rapidocr_without_tesseract(self) -> None:
        rapid_label = OcrLabel("Bay Area CA", x=10, y=10, width=80, height=20, confidence=96)

        with (
            patch.object(ocr_module, "ocr_cache_key", return_value=None),
            patch.object(ocr_module, "tesseract_available", return_value=False),
            patch.object(ocr_module, "run_rapidocr_words", return_value=[rapid_label]) as rapidocr,
        ):
            labels = extract_ocr_labels("unused.png")

        self.assertEqual(rapidocr.call_count, 1)
        self.assertIn("Bay Area CA", {label.text for label in labels})

    def test_extract_ocr_labels_reuses_rapidocr_words_after_tesseract_fallback(self) -> None:
        rapid_label = OcrLabel("Bay Area CA", x=10, y=10, width=80, height=20, confidence=96)

        with (
            patch.object(ocr_module, "ocr_cache_key", return_value=None),
            patch.object(ocr_module, "tesseract_available", return_value=True),
            patch.object(ocr_module, "run_rapidocr_words", return_value=[rapid_label]) as rapidocr,
            patch.object(ocr_module, "run_tesseract_words", return_value=[]),
            patch.object(ocr_module, "run_preprocessed_tesseract_words", return_value=[]),
        ):
            labels = extract_ocr_labels("unused.png")

        self.assertEqual(rapidocr.call_count, 1)
        self.assertIn("Bay Area CA", {label.text for label in labels})


class PlaceCandidateTests(unittest.TestCase):
    def test_geocode_many_preserves_request_order_and_dedupes(self) -> None:
        calls: list[tuple[str, int, str]] = []

        def fake_geocode(query: str, *, limit: int = 3, country_codes: str = "us"):
            calls.append((query, limit, country_codes))
            return [
                GeocodeResult(
                    label=query,
                    lon=-80.0,
                    lat=25.0,
                    display_name=f"{query}, Florida, United States",
                    bbox=(-80.1, 24.9, -79.9, 25.1),
                    importance=0.5,
                    place_type="city",
                )
            ][:limit]

        with patch("map_boundary_builder.georeference.geocode", side_effect=fake_geocode):
            results = geocode_many([("Miami", 2), ("Orlando", 1), ("Miami", 2)])

        self.assertEqual([items[0].label for items in results], ["Miami", "Orlando", "Miami"])
        self.assertEqual(set(calls), {("Miami", 2, "us"), ("Orlando", 1, "us")})

    def test_geocode_contexts_skip_synthetic_inferred_area(self) -> None:
        center = GeocodeResult(
            label="Inferred map area",
            lon=-80.2,
            lat=25.8,
            display_name="Inferred map area",
            bbox=(-80.4, 25.6, -80.0, 26.0),
            importance=0.5,
            place_type="region",
        )

        self.assertEqual(geocode_contexts("Inferred map area", center), [])

    def test_concise_high_confidence_labels_are_preserved(self) -> None:
        noisy_labels = [
            OcrLabel(f"NOISY LABEL {index}", x=index, y=index, width=90, height=20, confidence=70)
            for index in range(160)
        ]
        dallas = OcrLabel("Dallas", x=342, y=293.5, width=70, height=19, confidence=96)

        candidates = candidate_place_labels([*noisy_labels, dallas])

        self.assertIn(dallas, candidates)

    def test_place_query_text_removes_noise_and_repairs_aliases(self) -> None:
        self.assertEqual(place_query_text("edwood City acy"), "Redwood City")
        self.assertEqual(place_query_text("San Jos"), "San Jose")
        self.assertEqual(place_query_text("VILLOWBROOK"), "Willowbrook")
        self.assertEqual(place_query_text("C-ARVERDALE"), "Carverdale")

    def test_fuller_labels_identify_single_token_fragments(self) -> None:
        labels = [
            OcrLabel("Camellia Gardens", x=100, y=100, width=160, height=24, confidence=96),
            OcrLabel("Gardens", x=120, y=132, width=80, height=20, confidence=94),
            OcrLabel("Orlando", x=250, y=250, width=120, height=30, confidence=98),
        ]

        self.assertEqual(single_tokens_supported_by_fuller_labels(labels), {"camellia", "gardens"})

    def test_tiny_single_token_places_are_not_direct_contexts(self) -> None:
        francisco = GeocodeResult(
            label="Francisco",
            lon=-87.445,
            lat=38.334,
            display_name="Francisco, Gibson County, Indiana, United States",
            bbox=(-87.4578, 38.3279, -87.4407, 38.3380),
            importance=0.30,
            place_type="village",
        )
        dallas = GeocodeResult(
            label="Dallas",
            lon=-96.7970,
            lat=32.7767,
            display_name="Dallas, Dallas County, Texas, United States",
            bbox=(-97.0000, 32.6000, -96.5500, 33.0500),
            importance=0.72,
            place_type="city",
        )

        self.assertFalse(is_reliable_single_token_context(francisco))
        self.assertTrue(is_reliable_single_token_context(dallas))

    def test_multiword_label_tokens_can_promote_repeated_city_context(self) -> None:
        labels = [
            OcrLabel("Orlando Lake", x=120, y=100, width=260, height=120, confidence=96),
            OcrLabel("Parramore Orlando", x=180, y=160, width=280, height=110, confidence=94),
            OcrLabel("Downtown Orlando", x=240, y=220, width=300, height=100, confidence=95),
            OcrLabel("Orlando", x=260, y=245, width=120, height=80, confidence=96),
            OcrLabel("Lake Heights", x=300, y=280, width=180, height=70, confidence=94),
        ]
        calls: list[str] = []

        def fake_geocode(query: str, *, limit: int = 3, country_codes: str = "us"):
            calls.append(query)
            if query == "Orlando":
                return [
                    GeocodeResult(
                        label=query,
                        lon=-81.3792,
                        lat=28.5383,
                        display_name="Orlando, Orange County, Florida, United States",
                        bbox=(-81.51, 28.35, -81.22, 28.65),
                        importance=0.72,
                        place_type="city",
                    )
                ]
            return []

        with (
            patch("map_boundary_builder.georeference.geocode_cached_only", side_effect=fake_geocode),
            patch("map_boundary_builder.georeference.geocode", side_effect=fake_geocode),
        ):
            contexts = direct_city_contexts_from_labels(labels)

        self.assertEqual(contexts[0].query, "Orlando")
        self.assertIn("Orlando", calls)

    def test_broad_region_label_can_be_direct_context(self) -> None:
        labels = [
            OcrLabel("Bay Area CA", x=114, y=54, width=155, height=30, confidence=98),
            OcrLabel("Redwood City", x=248, y=301, width=127, height=28, confidence=96),
            OcrLabel("San Jose", x=396, y=375, width=115, height=31, confidence=95),
        ]
        calls: list[str] = []

        def fake_geocode(query: str, *, limit: int = 3, country_codes: str = "us"):
            calls.append(query)
            if query == "Bay Area":
                return [
                    GeocodeResult(
                        label=query,
                        lon=-122.35,
                        lat=37.78,
                        display_name="San Francisco Bay Area, San Francisco, California, United States",
                        bbox=(-123.35, 36.78, -121.35, 38.78),
                        importance=0.63,
                        place_type="region",
                    )
                ]
            if query == "San Jose":
                return [
                    GeocodeResult(
                        label=query,
                        lon=-121.8863,
                        lat=37.3382,
                        display_name="San Jose, Santa Clara County, California, United States",
                        bbox=(-122.04, 37.12, -121.58, 37.47),
                        importance=0.68,
                        place_type="city",
                    )
                ]
            return []

        with (
            patch("map_boundary_builder.georeference.geocode_cached_only", side_effect=fake_geocode),
            patch("map_boundary_builder.georeference.geocode", side_effect=fake_geocode),
        ):
            contexts = direct_city_contexts_from_labels(labels)

        self.assertEqual(contexts[0].query, "San Francisco Bay Area")
        self.assertEqual(contexts[0].evidence, ("Bay Area CA",))
        self.assertEqual(calls, ["Bay Area"])

    def test_broad_region_control_filter_skips_merged_ocr_labels(self) -> None:
        bay_area = GeocodeResult(
            label="Bay Area",
            lon=-122.35,
            lat=37.78,
            display_name="San Francisco Bay Area, San Francisco, California, United States",
            bbox=(-123.35, 36.78, -121.35, 38.78),
            importance=0.63,
            place_type="region",
        )
        city_tokens = place_tokens("San Francisco Bay Area")

        self.assertTrue(
            is_noisy_regional_control_query(place_tokens("Francisco Daly City"), city_tokens, bay_area)
        )
        self.assertTrue(
            is_noisy_regional_control_query(place_tokens("Oakland Daly City"), city_tokens, bay_area)
        )
        self.assertTrue(
            is_noisy_regional_control_query(place_tokens("Bay Area Oakland"), city_tokens, bay_area)
        )
        self.assertFalse(is_noisy_regional_control_query(place_tokens("Redwood City"), city_tokens, bay_area))
        self.assertFalse(is_noisy_regional_control_query(place_tokens("San Jose"), city_tokens, bay_area))

    def test_poi_descriptor_filter_keeps_city_labels(self) -> None:
        self.assertTrue(is_noisy_poi_query(place_tokens("Recreation Center")))
        self.assertTrue(is_noisy_poi_query(place_tokens("Airport International")))
        self.assertTrue(is_noisy_poi_query(place_tokens("Tempe Campus")))
        self.assertTrue(is_noisy_poi_query(place_tokens("Scottsdale Quarter")))
        self.assertFalse(is_noisy_poi_query(place_tokens("Redwood City")))
        self.assertFalse(is_noisy_poi_query(place_tokens("San Jose")))
        self.assertFalse(is_noisy_poi_query(place_tokens("Downtown Phoenix")))

    def test_direct_city_context_expands_when_labels_span_adjacent_places(self) -> None:
        labels = [
            OcrLabel("Miami", x=1300, y=1500, width=160, height=50, confidence=98),
            OcrLabel("Miami Gardens", x=1060, y=385, width=120, height=50, confidence=96),
            OcrLabel("South Miami", x=800, y=1890, width=160, height=28, confidence=96),
            OcrLabel("Coral Gables", x=930, y=1630, width=170, height=28, confidence=96),
            OcrLabel("Coconut Grove", x=1080, y=1710, width=150, height=28, confidence=96),
        ]

        def fake_geocode(query: str, *, limit: int = 3, country_codes: str = "us"):
            results = {
                "Miami": [
                    GeocodeResult(
                        label=query,
                        lon=-80.1936,
                        lat=25.7742,
                        display_name="Miami, Miami-Dade County, Florida, United States",
                        bbox=(-80.31976, 25.7090517, -80.139157, 25.8557827),
                        importance=0.73,
                        place_type="city",
                    )
                ],
                "Miami Gardens": [
                    GeocodeResult(
                        label=query,
                        lon=-80.2456,
                        lat=25.9420,
                        display_name="Miami Gardens, Miami-Dade County, Florida, United States",
                        bbox=(-80.31, 25.89, -80.18, 25.98),
                        importance=0.55,
                        place_type="city",
                    )
                ],
                "South Miami": [
                    GeocodeResult(
                        label=query,
                        lon=-80.2934,
                        lat=25.7076,
                        display_name="South Miami, Miami-Dade County, Florida, United States",
                        bbox=(-80.32, 25.69, -80.27, 25.73),
                        importance=0.55,
                        place_type="city",
                    )
                ],
                "Coral Gables": [
                    GeocodeResult(
                        label=query,
                        lon=-80.2585,
                        lat=25.7331,
                        display_name="Coral Gables, Miami-Dade County, Florida, United States",
                        bbox=(-80.32, 25.69, -80.22, 25.77),
                        importance=0.55,
                        place_type="city",
                    )
                ],
                "Coconut Grove": [
                    GeocodeResult(
                        label=query,
                        lon=-80.2570,
                        lat=25.7126,
                        display_name="Coconut Grove, Miami, Miami-Dade County, Florida, United States",
                        bbox=(-80.28, 25.69, -80.23, 25.74),
                        importance=0.45,
                        place_type="neighbourhood",
                    )
                ],
            }
            return results.get(query, [])[:limit]

        with (
            patch("map_boundary_builder.georeference.geocode_cached_only", side_effect=fake_geocode),
            patch("map_boundary_builder.georeference.geocode", side_effect=fake_geocode),
        ):
            contexts = infer_city_contexts(labels)

        self.assertGreater(contexts[0].center.bbox[3], 25.94)
        self.assertTrue(any(context.center.display_name.startswith("Miami,") for context in contexts[1:]))

    def test_early_context_cluster_requires_regional_breadth(self) -> None:
        def candidate(name: str, lon: float, lat: float, x: float, y: float) -> LabelGeocodeCandidate:
            return LabelGeocodeCandidate(
                label=OcrLabel(name, x=x, y=y, width=90, height=20, confidence=94),
                geocode=GeocodeResult(
                    label=name,
                    lon=lon,
                    lat=lat,
                    display_name=f"{name}, California, United States",
                    bbox=(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01),
                    importance=0.5,
                    place_type="city",
                ),
            )

        local_cluster = [
            candidate("Atherton", -122.2058, 37.4538, 10, 10),
            candidate("Menlo Park", -122.1780, 37.4520, 20, 20),
            candidate("Redwood City", -122.2325, 37.4863, 30, 30),
            candidate("Foster City", -122.2689, 37.5600, 40, 40),
            candidate("San Mateo", -122.3253, 37.5630, 50, 50),
            candidate("Belmont", -122.2942, 37.5165, 60, 60),
            candidate("Burlingame", -122.3473, 37.5781, 70, 70),
            candidate("San Bruno", -122.4111, 37.6305, 80, 80),
        ]
        regional_cluster = [
            *local_cluster,
            candidate("San Francisco", -122.4194, 37.7749, 90, 90),
            candidate("Palo Alto", -122.1598, 37.4443, 100, 100),
        ]

        self.assertFalse(has_reliable_candidate_cluster(local_cluster))
        self.assertTrue(has_reliable_candidate_cluster(regional_cluster))


class RoadContextRankingTests(unittest.TestCase):
    def test_broad_regional_context_beats_small_city_label(self) -> None:
        from map_boundary_builder.georeference import CityContext

        sunnyvale = CityContext(
            query="Sunnyvale",
            center=GeocodeResult(
                label="Sunnyvale",
                lon=-122.0363,
                lat=37.3688,
                display_name="Sunnyvale, Santa Clara County, California, United States",
                bbox=(-122.0652, 37.3302, -121.9825, 37.4637),
                importance=0.55,
                place_type="city",
            ),
            inferred=True,
            evidence=("Sunnyvale",),
        )
        santa_clara_region = CityContext(
            query="Santa Clara",
            center=GeocodeResult(
                label="Santa Clara",
                lon=-122.1483,
                lat=37.3509,
                display_name="Santa Clara",
                bbox=(-122.3846, 37.0939, -121.9119, 37.6070),
                importance=0.5,
                place_type="region",
            ),
            inferred=True,
            evidence=("Menlo Park", "Mountain View", "Redwood City", "Sunnyvale"),
        )
        school_district = CityContext(
            query="Mountain View Los Altos Union High School District",
            center=GeocodeResult(
                label="Mountain Altos",
                lon=-122.0648,
                lat=37.3610,
                display_name="Mountain View Los Altos Union High School District",
                bbox=(-122.0652, 37.3608, -122.0647, 37.3612),
                importance=0.35,
                place_type="educational_institution",
            ),
            inferred=True,
            evidence=("Mountain View",),
        )

        ranked = rank_road_context_queries([sunnyvale, school_district, santa_clara_region])

        self.assertEqual("Santa Clara", ranked[0])
        self.assertEqual("Mountain View Los Altos Union High School District", ranked[-1])


class GeoreferenceFallbackTests(unittest.TestCase):
    def test_ranked_context_failure_falls_back_to_label_fit(self) -> None:
        labels = [OcrLabel("Nashville", x=1141, y=454, width=162, height=44, confidence=98)]
        fallback_result = object()

        with (
            patch("map_boundary_builder.runner.road_contexts_from_labels", return_value=[object()]),
            patch("map_boundary_builder.runner.road_context_queries", return_value=[]),
            patch("map_boundary_builder.runner.should_try_ranked_context_first", return_value=True),
            patch("map_boundary_builder.runner.georeference_from_ranked_label_contexts", return_value=None),
            patch("map_boundary_builder.runner.georeference_from_labels", return_value=fallback_result) as label_fit,
        ):
            result = fit_georeference(
                labels,
                Path("input.png"),
                pixel_geometry=object(),
                rgb=None,
                city_input=None,
                width=1920,
                height=1080,
                coverage_ratio=0.22,
                min_control_points=3,
                label_y_min=None,
                label_y_max=None,
                progress=None,
            )

        self.assertIs(result, fallback_result)
        label_fit.assert_called_once()

    def test_specific_city_fit_keeps_city_after_synthetic_context_failure(self) -> None:
        labels = [OcrLabel("Nashville", x=1141, y=454, width=162, height=44, confidence=98)]
        synthetic_context = CityContext(
            query="Inferred map area",
            center=GeocodeResult(
                label="Inferred map area",
                lon=-77.1,
                lat=39.0,
                display_name="Inferred map area",
                bbox=(-77.4, 38.7, -76.8, 39.3),
                importance=0.5,
                place_type="region",
            ),
            inferred=True,
        )
        nashville_context = CityContext(
            query="Nashville",
            center=GeocodeResult(
                label="Nashville",
                lon=-86.7816,
                lat=36.1627,
                display_name="Nashville, Davidson County, Tennessee, United States",
                bbox=(-87.05, 35.96, -86.51, 36.4),
                importance=0.7,
                place_type="city",
            ),
            inferred=True,
        )
        nashville_result = GeoreferenceResult(
            transform=GeoreferenceTransform(
                city="Nashville",
                lon=-86.94,
                lat=36.25,
                origin_x_ratio=0.0,
                origin_y_ratio=0.0,
                meters_per_pixel=20.8,
                rotation_radians=0.0,
                confidence=0.76,
                source="test",
            ),
            control_points=[],
            residual_median_m=1000.0,
            residual_p90_m=1100.0,
        )

        with (
            patch("map_boundary_builder.georeference.anchor_labels_to_marker_dots", return_value=labels),
            patch("map_boundary_builder.georeference.resolve_city_contexts", return_value=[synthetic_context, nashville_context]),
            patch("map_boundary_builder.georeference.georeference_from_label_context", side_effect=[None, nashville_result]),
        ):
            result = georeference_from_labels(labels, "input.png", None, width=1920, height=1080)

        self.assertEqual(result.transform.city, "Nashville")


if __name__ == "__main__":
    unittest.main()
