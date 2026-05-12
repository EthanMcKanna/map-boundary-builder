import unittest

from map_boundary_builder.georeference import candidate_place_labels
from map_boundary_builder.ocr import OcrLabel, group_stacked_labels


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


class PlaceCandidateTests(unittest.TestCase):
    def test_concise_high_confidence_labels_are_preserved(self) -> None:
        noisy_labels = [
            OcrLabel(f"NOISY LABEL {index}", x=index, y=index, width=90, height=20, confidence=70)
            for index in range(160)
        ]
        dallas = OcrLabel("Dallas", x=342, y=293.5, width=70, height=19, confidence=96)

        candidates = candidate_place_labels([*noisy_labels, dallas])

        self.assertIn(dallas, candidates)


if __name__ == "__main__":
    unittest.main()
