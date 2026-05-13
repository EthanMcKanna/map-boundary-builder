import unittest
from unittest.mock import patch

from map_boundary_builder.georeference import candidate_place_labels, infer_city_contexts, is_reliable_single_token_context, place_query_text
from map_boundary_builder.geocoder import GeocodeResult
from map_boundary_builder.ocr import OcrLabel, group_stacked_labels, rapidocr_items_to_labels
from map_boundary_builder.runner import rank_road_context_queries


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


class PlaceCandidateTests(unittest.TestCase):
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

        with patch("map_boundary_builder.georeference.geocode", side_effect=fake_geocode):
            contexts = infer_city_contexts(labels)

        self.assertGreater(contexts[0].center.bbox[3], 25.94)
        self.assertTrue(any(context.center.display_name.startswith("Miami,") for context in contexts[1:]))


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


if __name__ == "__main__":
    unittest.main()
