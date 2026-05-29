import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, PngImagePlugin

import map_boundary_builder.ocr as ocr_module
from map_boundary_builder.georeference import (
    CityContext,
    ControlPoint,
    GeoreferenceResult,
    LabelGeocodeCandidate,
    build_control_points,
    candidate_place_labels,
    direct_city_contexts_from_labels,
    detect_label_marker_dots,
    filename_city_contexts,
    filename_context_queries,
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
    residual_median_p90,
    should_try_road_refinement,
    single_tokens_supported_by_fuller_labels,
)
from map_boundary_builder.geocoder import GeocodeResult
from map_boundary_builder.georef_transform import GeoreferenceTransform
from map_boundary_builder.ocr import (
    OCR_MEMORY_CACHE_MAX,
    OcrLabel,
    _OCR_MEMORY_CACHE,
    extract_ocr_labels,
    group_stacked_labels,
    load_rapidocr_bgr,
    ocr_cache_key,
    ocr_coarse_visual_cache_key,
    ocr_cache_dependency_signature,
    ocr_near_visual_cache_key,
    ocr_visual_cache_key,
    rapidocr_detector_limit_for_input,
    rapidocr_input_array,
    rapidocr_input_image,
    rapidocr_items_to_labels,
    read_ocr_cache,
    rgb_to_bgr,
    warm_rapidocr_runtime,
    write_ocr_cache,
)
from map_boundary_builder.runner import fit_georeference, rank_road_context_queries


class FakeRapidOcrEngine:
    def __init__(self, responses: dict[bool | None, list]) -> None:
        self.responses = responses
        self.use_cls_calls: list[bool | None] = []

    def __call__(self, _image, *, use_cls=None):
        self.use_cls_calls.append(use_cls)
        return self.responses.get(use_cls, []), 0.0


def unit_ocr_box(x: float = 0.0) -> list[list[float]]:
    return [[x, 0.0], [x + 80.0, 0.0], [x + 80.0, 20.0], [x, 20.0]]


class OcrGroupingTests(unittest.TestCase):
    def test_residual_median_p90_matches_numpy_linear_percentile(self) -> None:
        for values in ([4.0], [8.0, 2.0], [12.0, 2.0, 7.0], [0.0, 100.0, 300.0, 900.0, 1400.0]):
            median, p90 = residual_median_p90(values)

            self.assertAlmostEqual(median, float(np.median(values)))
            self.assertAlmostEqual(p90, float(np.percentile(values, 90)))

    def test_residual_median_p90_handles_empty_numpy_arrays(self) -> None:
        self.assertEqual(residual_median_p90(np.array([]), empty=float("inf")), (float("inf"), float("inf")))

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

    def test_ocr_memory_cache_evicts_oldest_entries(self) -> None:
        label = OcrLabel("Dallas", x=60, y=37, width=100, height=34, confidence=96)

        with TemporaryDirectory() as workdir:
            with patch.object(ocr_module, "OCR_CACHE_DIR", Path(workdir)):
                _OCR_MEMORY_CACHE.clear()
                try:
                    for index in range(OCR_MEMORY_CACHE_MAX + 1):
                        write_ocr_cache(f"key-{index}", [label])

                    self.assertNotIn("key-0", _OCR_MEMORY_CACHE)
                    self.assertIn(f"key-{OCR_MEMORY_CACHE_MAX}", _OCR_MEMORY_CACHE)
                    self.assertEqual(len(_OCR_MEMORY_CACHE), OCR_MEMORY_CACHE_MAX)
                finally:
                    _OCR_MEMORY_CACHE.clear()

    def test_ocr_memory_cache_refreshes_recent_reads(self) -> None:
        label = OcrLabel("Dallas", x=60, y=37, width=100, height=34, confidence=96)

        with TemporaryDirectory() as workdir:
            with patch.object(ocr_module, "OCR_CACHE_DIR", Path(workdir)):
                _OCR_MEMORY_CACHE.clear()
                try:
                    for index in range(OCR_MEMORY_CACHE_MAX):
                        write_ocr_cache(f"key-{index}", [label])
                    self.assertEqual(read_ocr_cache("key-0"), (label,))
                    write_ocr_cache("new-key", [label])

                    self.assertIn("key-0", _OCR_MEMORY_CACHE)
                    self.assertNotIn("key-1", _OCR_MEMORY_CACHE)
                    self.assertIn("new-key", _OCR_MEMORY_CACHE)
                finally:
                    _OCR_MEMORY_CACHE.clear()

    def test_ocr_visual_cache_key_ignores_png_metadata(self) -> None:
        with TemporaryDirectory() as workdir:
            first = Path(workdir) / "first.png"
            second = Path(workdir) / "second.png"
            image = Image.new("RGB", (20, 10), (12, 34, 56))
            image.save(first)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("cache_bust", "same pixels")
            image.save(second, pnginfo=metadata)

            first_bgr, _ = load_rapidocr_bgr(first)
            second_bgr, _ = load_rapidocr_bgr(second)

            self.assertNotEqual(ocr_cache_key(first, use_tesseract=False), ocr_cache_key(second, use_tesseract=False))
            self.assertEqual(
                ocr_visual_cache_key(first_bgr, use_tesseract=False),
                ocr_visual_cache_key(second_bgr, use_tesseract=False),
            )

    def test_ocr_near_visual_cache_key_tolerates_low_bit_pixel_noise(self) -> None:
        with TemporaryDirectory() as workdir:
            first = Path(workdir) / "first.png"
            second = Path(workdir) / "second.png"
            Image.new("RGB", (20, 10), (12, 32, 56)).save(first)
            Image.new("RGB", (20, 10), (13, 35, 59)).save(second)
            first_bgr, _ = load_rapidocr_bgr(first)
            second_bgr, _ = load_rapidocr_bgr(second)

            self.assertNotEqual(
                ocr_visual_cache_key(first_bgr, use_tesseract=False),
                ocr_visual_cache_key(second_bgr, use_tesseract=False),
            )
            self.assertEqual(
                ocr_near_visual_cache_key(first_bgr, use_tesseract=False),
                ocr_near_visual_cache_key(second_bgr, use_tesseract=False),
            )

    def test_ocr_coarse_visual_cache_key_tolerates_quantization_boundary_noise(self) -> None:
        with TemporaryDirectory() as workdir:
            first = Path(workdir) / "first.png"
            second = Path(workdir) / "second.png"
            Image.new("RGB", (20, 10), (12, 36, 56)).save(first)
            Image.new("RGB", (20, 10), (8, 32, 56)).save(second)
            first_bgr, _ = load_rapidocr_bgr(first)
            second_bgr, _ = load_rapidocr_bgr(second)

            self.assertNotEqual(
                ocr_visual_cache_key(first_bgr, use_tesseract=False),
                ocr_visual_cache_key(second_bgr, use_tesseract=False),
            )
            self.assertNotEqual(
                ocr_near_visual_cache_key(first_bgr, use_tesseract=False),
                ocr_near_visual_cache_key(second_bgr, use_tesseract=False),
            )
            self.assertEqual(
                ocr_coarse_visual_cache_key(first_bgr, use_tesseract=False),
                ocr_coarse_visual_cache_key(second_bgr, use_tesseract=False),
            )

    def test_rgb_to_bgr_converts_rgb_channels_for_prepared_ocr_input(self) -> None:
        rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

        bgr = rgb_to_bgr(rgb)

        assert bgr is not None
        self.assertTrue(bgr.flags.c_contiguous)
        self.assertEqual(bgr.tolist(), [[[3, 2, 1], [6, 5, 4]]])

    def test_prepared_ocr_bgr_avoids_second_image_decode_on_visual_cache_hit(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "image.png"
            Image.new("RGB", (20, 10), (12, 34, 56)).save(image_path)
            prepared_bgr = rgb_to_bgr(np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8))
            visual_key = ocr_visual_cache_key(prepared_bgr, use_tesseract=False)
            label = OcrLabel("Dallas", x=60, y=37, width=100, height=34, confidence=96)

            assert prepared_bgr is not None
            assert visual_key is not None
            write_ocr_cache(visual_key, [label])
            with patch.object(ocr_module, "tesseract_available", return_value=False), patch.object(
                ocr_module,
                "load_rapidocr_bgr",
                side_effect=AssertionError("prepared OCR input should avoid re-decoding the image"),
            ), patch.object(
                ocr_module,
                "run_rapidocr_words",
                side_effect=AssertionError("visual cache hit should avoid OCR"),
            ):
                labels = extract_ocr_labels(image_path, prepared_bgr=prepared_bgr)

            self.assertEqual(labels, [label])

    def test_ocr_visual_cache_hit_backfills_raw_key(self) -> None:
        with TemporaryDirectory() as workdir:
            first = Path(workdir) / "first.png"
            second = Path(workdir) / "second.png"
            image = Image.new("RGB", (20, 10), (12, 34, 56))
            image.save(first)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("cache_bust", "same pixels")
            image.save(second, pnginfo=metadata)
            first_bgr, _ = load_rapidocr_bgr(first)
            visual_key = ocr_visual_cache_key(first_bgr, use_tesseract=False)
            raw_second_key = ocr_cache_key(second, use_tesseract=False)
            label = OcrLabel("Miami Beach", x=60, y=37, width=100, height=34, confidence=96)

            assert visual_key is not None
            assert raw_second_key is not None
            write_ocr_cache(visual_key, [label])
            with patch.object(ocr_module, "tesseract_available", return_value=False), patch.object(
                ocr_module,
                "run_rapidocr_words",
                side_effect=AssertionError("visual cache hit should avoid OCR"),
            ):
                labels = extract_ocr_labels(second)

            self.assertEqual(labels, [label])
            self.assertEqual(read_ocr_cache(raw_second_key), (label,))

    def test_ocr_near_visual_cache_hit_backfills_raw_and_exact_visual_keys(self) -> None:
        with TemporaryDirectory() as workdir:
            first = Path(workdir) / "first.png"
            second = Path(workdir) / "second.png"
            Image.new("RGB", (20, 10), (12, 32, 56)).save(first)
            Image.new("RGB", (20, 10), (13, 35, 59)).save(second)
            first_bgr, _ = load_rapidocr_bgr(first)
            second_bgr, _ = load_rapidocr_bgr(second)
            near_visual_key = ocr_near_visual_cache_key(first_bgr, use_tesseract=False)
            exact_second_key = ocr_visual_cache_key(second_bgr, use_tesseract=False)
            raw_second_key = ocr_cache_key(second, use_tesseract=False)
            label = OcrLabel("Miami Beach", x=60, y=37, width=100, height=34, confidence=96)

            assert near_visual_key is not None
            assert exact_second_key is not None
            assert raw_second_key is not None
            write_ocr_cache(near_visual_key, [label])
            with patch.object(ocr_module, "tesseract_available", return_value=False), patch.object(
                ocr_module,
                "run_rapidocr_words",
                side_effect=AssertionError("near visual cache hit should avoid OCR"),
            ):
                labels = extract_ocr_labels(second)

            self.assertEqual(labels, [label])
            self.assertEqual(read_ocr_cache(raw_second_key), (label,))
            self.assertEqual(read_ocr_cache(exact_second_key), (label,))

    def test_ocr_coarse_visual_cache_hit_backfills_raw_and_visual_keys(self) -> None:
        with TemporaryDirectory() as workdir:
            first = Path(workdir) / "first.png"
            second = Path(workdir) / "second.png"
            Image.new("RGB", (20, 10), (12, 36, 56)).save(first)
            Image.new("RGB", (20, 10), (8, 32, 56)).save(second)
            first_bgr, _ = load_rapidocr_bgr(first)
            second_bgr, _ = load_rapidocr_bgr(second)
            coarse_visual_key = ocr_coarse_visual_cache_key(first_bgr, use_tesseract=False)
            near_second_key = ocr_near_visual_cache_key(second_bgr, use_tesseract=False)
            exact_second_key = ocr_visual_cache_key(second_bgr, use_tesseract=False)
            raw_second_key = ocr_cache_key(second, use_tesseract=False)
            label = OcrLabel("Miami Beach", x=60, y=37, width=100, height=34, confidence=96)

            assert coarse_visual_key is not None
            assert near_second_key is not None
            assert exact_second_key is not None
            assert raw_second_key is not None
            write_ocr_cache(coarse_visual_key, [label])
            with patch.object(ocr_module, "tesseract_available", return_value=False), patch.object(
                ocr_module,
                "run_rapidocr_words",
                side_effect=AssertionError("coarse visual cache hit should avoid OCR"),
            ):
                labels = extract_ocr_labels(second)

            self.assertEqual(labels, [label])
            self.assertEqual(read_ocr_cache(raw_second_key), (label,))
            self.assertEqual(read_ocr_cache(exact_second_key), (label,))
            self.assertEqual(read_ocr_cache(near_second_key), (label,))

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

    def test_rapidocr_input_array_downscales_without_temp_file(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_MAX_DIMENSION", 10):
                ocr_input, scale_x, scale_y = rapidocr_input_array(image_path)

            self.assertIsInstance(ocr_input, np.ndarray)
            self.assertEqual(ocr_input.shape[:2], (5, 10))
            self.assertAlmostEqual(scale_x, 0.5)
            self.assertAlmostEqual(scale_y, 0.5)

    def test_rapidocr_input_array_can_return_loaded_array_for_native_moderate_images(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with (
                patch.object(ocr_module, "RAPIDOCR_MAX_DIMENSION", 40),
                patch.object(ocr_module, "RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION", 10),
            ):
                ocr_input, scale_x, scale_y = rapidocr_input_array(image_path)

            self.assertIsInstance(ocr_input, np.ndarray)
            self.assertEqual(scale_x, 1.0)
            self.assertEqual(scale_y, 1.0)

    def test_rapidocr_input_array_composites_transparent_png(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            image = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
            image.putpixel((1, 0), (255, 128, 0, 255))
            image.putpixel((2, 0), (10, 20, 30, 128))
            image.save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_MAX_DIMENSION", 10):
                ocr_input, scale_x, scale_y = rapidocr_input_array(image_path)

            self.assertIsInstance(ocr_input, np.ndarray)
            self.assertEqual(tuple(ocr_input[0, 0]), (255, 255, 255))
            self.assertEqual(tuple(ocr_input[0, 1]), (0, 128, 255))
            self.assertEqual(tuple(ocr_input[0, 2]), (142, 137, 132))
            self.assertEqual(scale_x, 1.0)
            self.assertEqual(scale_y, 1.0)

    def test_ocr_cache_key_depends_on_rapidocr_detector_limit(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_DET_LIMIT_SIDE_LEN", 640):
                key_640 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_DET_LIMIT_SIDE_LEN", 736):
                key_736 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_640, key_736)

    def test_ocr_cache_key_depends_on_large_rapidocr_detector_limit(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN", 608):
                key_608 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN", 640):
                key_640 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_608, key_640)

    def test_ocr_cache_key_depends_on_native_rapidocr_array_threshold(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION", 0):
                key_0 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION", 1000):
                key_1000 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_0, key_1000)

    def test_ocr_cache_key_depends_on_runtime_dependency_signature(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            ocr_cache_dependency_signature.cache_clear()
            try:
                with patch.object(ocr_module, "version", return_value="1.0"):
                    key_1 = ocr_cache_key(image_path, use_tesseract=False)
                ocr_cache_dependency_signature.cache_clear()
                with patch.object(ocr_module, "version", return_value="2.0"):
                    key_2 = ocr_cache_key(image_path, use_tesseract=False)
            finally:
                ocr_cache_dependency_signature.cache_clear()

        self.assertNotEqual(key_1, key_2)

    def test_ocr_cache_key_depends_on_rapidocr_classifier_batch(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_CLS_BATCH_NUM", 6):
                key_6 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_CLS_BATCH_NUM", 24):
                key_24 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_6, key_24)

    def test_ocr_cache_key_depends_on_rapidocr_recognition_batch(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_REC_BATCH_NUM", 6):
                key_6 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_REC_BATCH_NUM", 24):
                key_24 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_6, key_24)

    def test_ocr_cache_key_depends_on_rapidocr_classifier_retry_threshold(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS", 1):
                key_1 = ocr_cache_key(image_path, use_tesseract=False)
            with patch.object(ocr_module, "RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS", 3):
                key_3 = ocr_cache_key(image_path, use_tesseract=False)

        self.assertNotEqual(key_1, key_3)

    def test_ocr_cache_key_depends_on_tesseract_fallback_threshold(self) -> None:
        with TemporaryDirectory() as workdir:
            image_path = Path(workdir) / "input.png"
            Image.new("RGB", (20, 10), (255, 255, 255)).save(image_path)

            with patch.object(ocr_module, "TESSERACT_FALLBACK_MIN_USEFUL_LABELS", 3):
                key_3 = ocr_cache_key(image_path, use_tesseract=True)
            with patch.object(ocr_module, "TESSERACT_FALLBACK_MIN_USEFUL_LABELS", 12):
                key_12 = ocr_cache_key(image_path, use_tesseract=True)

        self.assertNotEqual(key_3, key_12)

    def test_rapidocr_skips_classifier_when_fast_pass_has_labels(self) -> None:
        engine = FakeRapidOcrEngine(
            {
                False: [
                    [unit_ocr_box(), "Orlando", 0.98],
                    [unit_ocr_box(x=100.0), "Southchase", 0.97],
                ],
                True: [[unit_ocr_box(), "Should Not Run", 0.98]],
            }
        )

        with (
            patch.object(ocr_module, "rapidocr_input_array", return_value=("image", 1.0, 1.0)),
            patch.object(ocr_module, "rapidocr_engine", return_value=engine),
            patch.object(ocr_module, "RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS", 2),
        ):
            labels = ocr_module.run_rapidocr_words("unused.png")

        self.assertEqual(engine.use_cls_calls, [False])
        self.assertEqual([label.text for label in labels], ["Orlando", "Southchase"])

    def test_rapidocr_uses_large_detector_limit_for_large_arrays(self) -> None:
        engine = FakeRapidOcrEngine(
            {
                False: [
                    [unit_ocr_box(), "Orlando", 0.98],
                    [unit_ocr_box(x=100.0), "Southchase", 0.97],
                ]
            }
        )
        image = np.zeros((1200, 1600, 3), dtype=np.uint8)

        with (
            patch.object(ocr_module, "rapidocr_input_array", return_value=(image, 1.0, 1.0)),
            patch.object(ocr_module, "rapidocr_engine", return_value=engine) as rapidocr,
            patch.object(ocr_module, "RAPIDOCR_DET_LIMIT_SIDE_LEN", 608),
            patch.object(ocr_module, "RAPIDOCR_LARGE_IMAGE_DET_LIMIT_SIDE_LEN", 640),
            patch.object(ocr_module, "RAPIDOCR_LARGE_IMAGE_DET_LIMIT_MIN_DIMENSION", 1000),
        ):
            labels = ocr_module.run_rapidocr_words("unused.png")

        rapidocr.assert_called_once_with(640)
        self.assertEqual([label.text for label in labels], ["Orlando", "Southchase"])

    def test_rapidocr_keeps_base_detector_limit_for_small_inputs(self) -> None:
        self.assertEqual(
            rapidocr_detector_limit_for_input(np.zeros((400, 800, 3), dtype=np.uint8)),
            ocr_module.RAPIDOCR_DET_LIMIT_SIDE_LEN,
        )

    def test_configure_rapidocr_session_options_applies_runtime_defaults(self) -> None:
        class FakeSessionOptions:
            def __init__(self) -> None:
                self.log_severity_level = None
                self.enable_cpu_mem_arena = None
                self.graph_optimization_level = None
                self.intra_op_num_threads = None
                self.inter_op_num_threads = None
                self.entries = {}

            def add_session_config_entry(self, key, value) -> None:
                self.entries[key] = value

        class FakeOrtInferSession:
            pass

        fake_ort = SimpleNamespace(
            SessionOptions=FakeSessionOptions,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        )
        fake_utils = SimpleNamespace(OrtInferSession=FakeOrtInferSession)
        was_patched = ocr_module._RAPIDOCR_SESSION_OPTIONS_PATCHED
        try:
            ocr_module._RAPIDOCR_SESSION_OPTIONS_PATCHED = False
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "onnxruntime": fake_ort,
                        "rapidocr_onnxruntime.utils": fake_utils,
                    },
                ),
                patch.object(ocr_module.os, "cpu_count", return_value=8),
                patch.object(ocr_module, "ONNXRUNTIME_ENABLE_CPU_MEM_ARENA", True),
                patch.object(ocr_module, "ONNXRUNTIME_ALLOW_SPINNING", False),
            ):
                ocr_module.configure_rapidocr_onnxruntime_session_options()
                opts = FakeOrtInferSession._init_sess_opts(
                    {"intra_op_num_threads": 2, "inter_op_num_threads": 3}
                )
        finally:
            ocr_module._RAPIDOCR_SESSION_OPTIONS_PATCHED = was_patched

        self.assertTrue(opts.enable_cpu_mem_arena)
        self.assertEqual(opts.intra_op_num_threads, 2)
        self.assertEqual(opts.inter_op_num_threads, 3)
        self.assertEqual(
            opts.entries,
            {
                "session.intra_op.allow_spinning": "0",
                "session.inter_op.allow_spinning": "0",
            },
        )

    def test_rapidocr_retries_classifier_when_fast_pass_is_sparse(self) -> None:
        fast_engine = FakeRapidOcrEngine({False: []})
        classifier_engine = FakeRapidOcrEngine(
            {True: [[unit_ocr_box(), "Southchase", 0.99]]}
        )

        with (
            patch.object(ocr_module, "rapidocr_input_array", return_value=("image", 1.0, 1.0)),
            patch.object(ocr_module, "rapidocr_engine", return_value=fast_engine),
            patch.object(ocr_module, "rapidocr_classifier_engine", return_value=classifier_engine),
            patch.object(ocr_module, "RAPIDOCR_CLASSIFIER_RETRY_MIN_LABELS", 2),
        ):
            labels = ocr_module.run_rapidocr_words("unused.png")

        self.assertEqual(fast_engine.use_cls_calls, [False])
        self.assertEqual(classifier_engine.use_cls_calls, [True])
        self.assertEqual([label.text for label in labels], ["Southchase"])

    def test_warm_rapidocr_runtime_runs_synthetic_fast_pass_for_warm_limits(self) -> None:
        engine = FakeRapidOcrEngine({False: [[unit_ocr_box(), "Miami", 0.98]]})
        warm_rapidocr_runtime.cache_clear()
        try:
            with patch.object(ocr_module, "rapidocr_engine", return_value=engine) as rapidocr:
                self.assertTrue(warm_rapidocr_runtime())
                self.assertTrue(warm_rapidocr_runtime())

            expected_limits = ocr_module.rapidocr_warm_detector_limits()
            self.assertEqual(
                [call.args[0] for call in rapidocr.call_args_list],
                expected_limits,
            )
            self.assertEqual(engine.use_cls_calls, [False] * len(expected_limits))
        finally:
            warm_rapidocr_runtime.cache_clear()

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
            patch.object(ocr_module, "TESSERACT_FALLBACK_MIN_USEFUL_LABELS", 3),
            patch.object(ocr_module, "tesseract_available", return_value=True),
            patch.object(ocr_module, "run_rapidocr_words", return_value=[rapid_label]) as rapidocr,
            patch.object(ocr_module, "run_tesseract_words", return_value=[]),
            patch.object(ocr_module, "run_preprocessed_tesseract_words", return_value=[]),
        ):
            labels = extract_ocr_labels("unused.png")

        self.assertEqual(rapidocr.call_count, 1)
        self.assertIn("Bay Area CA", {label.text for label in labels})

    def test_extract_ocr_labels_preserves_high_confidence_rapidocr_after_noisy_tesseract(self) -> None:
        rapid_label = OcrLabel("Nashville", x=250, y=99, width=37, height=10, confidence=98)
        noisy_tesseract_labels = [
            OcrLabel("Mar", x=156, y=98, width=9, height=3, confidence=50),
            OcrLabel("fee", x=221, y=116, width=12, height=3, confidence=45),
            OcrLabel("Bur", x=164, y=144, width=22, height=5, confidence=34),
        ]

        with (
            patch.object(ocr_module, "ocr_cache_key", return_value=None),
            patch.object(ocr_module, "TESSERACT_FALLBACK_MIN_USEFUL_LABELS", 3),
            patch.object(ocr_module, "tesseract_available", return_value=True),
            patch.object(ocr_module, "run_rapidocr_words", return_value=[rapid_label]) as rapidocr,
            patch.object(ocr_module, "run_tesseract_words", return_value=noisy_tesseract_labels),
            patch.object(ocr_module, "run_preprocessed_tesseract_words", return_value=[]),
        ):
            labels = extract_ocr_labels("unused.png")

        self.assertEqual(rapidocr.call_count, 1)
        self.assertIn("Nashville", {label.text for label in labels})

    def test_extract_ocr_labels_skips_tesseract_when_rapidocr_has_enough_labels(self) -> None:
        rapid_labels = [
            OcrLabel("University Park", x=10, y=10, width=80, height=20, confidence=96),
            OcrLabel("Highland Park", x=110, y=10, width=80, height=20, confidence=96),
            OcrLabel("Dallas", x=210, y=10, width=80, height=20, confidence=96),
        ]

        with (
            patch.object(ocr_module, "ocr_cache_key", return_value=None),
            patch.object(ocr_module, "TESSERACT_FALLBACK_MIN_USEFUL_LABELS", 3),
            patch.object(ocr_module, "tesseract_available", return_value=True),
            patch.object(ocr_module, "run_rapidocr_words", return_value=rapid_labels),
            patch.object(ocr_module, "run_tesseract_words") as tesseract,
            patch.object(ocr_module, "run_preprocessed_tesseract_words") as preprocessed,
        ):
            labels = extract_ocr_labels("unused.png")

        tesseract.assert_not_called()
        preprocessed.assert_not_called()
        self.assertIn("University Park", {label.text for label in labels})


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
        self.assertEqual(place_query_text("Ersey Village"), "Jersey Village")
        self.assertEqual(place_query_text("rsey Village"), "Jersey Village")
        self.assertEqual(place_query_text("HUNTRIDG"), "Huntridge")
        self.assertEqual(place_query_text("ILLOWBROOK"), "Willowbrook")
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

    def test_strong_standalone_fragment_promotes_cached_city_context(self) -> None:
        labels = [
            OcrLabel("UPTOWN-KNOX Dallas", x=409, y=269, width=208, height=76, confidence=97),
            OcrLabel("Dallas HARWOOD", x=299.5, y=316, width=163, height=68, confidence=99),
            OcrLabel("Maplelawn", x=129.5, y=103, width=99, height=20, confidence=99),
            OcrLabel("Scyener", x=577.5, y=452.5, width=77, height=25, confidence=99),
            OcrLabel("Dallas", x=343, y=294.5, width=76, height=25, confidence=99),
        ]
        calls: list[tuple[str, str]] = []

        def fake_cached(query: str, *, limit: int = 3, country_codes: str = "us"):
            calls.append(("cached", query))
            if query == "Dallas":
                return [
                    GeocodeResult(
                        label=query,
                        lon=-96.7970,
                        lat=32.7767,
                        display_name="Dallas, Dallas County, Texas, United States",
                        bbox=(-97.0000, 32.6000, -96.5500, 33.0500),
                        importance=0.72,
                        place_type="city",
                    )
                ]
            return []

        def fake_live(query: str, *, limit: int = 3, country_codes: str = "us"):
            calls.append(("live", query))
            return []

        with (
            patch("map_boundary_builder.georeference.geocode_cached_only", side_effect=fake_cached),
            patch("map_boundary_builder.georeference.geocode", side_effect=fake_live),
        ):
            contexts = direct_city_contexts_from_labels(labels)

        self.assertEqual(contexts[0].query, "Dallas")
        self.assertIn(("cached", "Dallas"), calls)
        self.assertFalse([query for provider, query in calls if provider == "live"])

    def test_clean_city_label_survives_noisy_road_context_labels(self) -> None:
        labels = [
            OcrLabel("W-Flamingo R Cameron St", x=170, y=88, width=245, height=100, confidence=97),
            OcrLabel("Lindell-Rd CHARLESTON", x=165, y=136, width=210, height=90, confidence=98),
            OcrLabel("WRussell Rd Patrick Ln", x=150, y=182, width=190, height=82, confidence=97),
            OcrLabel("Badura Ave", x=140, y=220, width=115, height=55, confidence=99),
            OcrLabel("FFALO", x=215, y=255, width=95, height=26, confidence=99),
            OcrLabel("Las Vegas", x=387, y=267, width=102, height=23, confidence=95),
        ]
        calls: list[str] = []

        def fake_geocode(query: str, *, limit: int = 3, country_codes: str = "us"):
            calls.append(query)
            if query == "Las Vegas":
                return [
                    GeocodeResult(
                        label=query,
                        lon=-115.1484,
                        lat=36.1674,
                        display_name="Las Vegas, Clark County, Nevada, United States",
                        bbox=(-115.406575, 36.129554, -115.062066, 36.401481),
                        importance=0.724,
                        place_type="city",
                    )
                ]
            return []

        with (
            patch("map_boundary_builder.georeference.geocode_cached_only", side_effect=fake_geocode),
            patch("map_boundary_builder.georeference.geocode", side_effect=fake_geocode),
        ):
            contexts = direct_city_contexts_from_labels(labels)

        self.assertEqual(contexts[0].query, "Las Vegas")
        self.assertIn("Las Vegas", calls)

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
    def test_ready_place_controls_skip_live_geocoded_control_lookup(self) -> None:
        center = GeocodeResult(
            label="Las Vegas",
            lon=-115.1484,
            lat=36.1674,
            display_name="Las Vegas, Clark County, Nevada, United States",
            bbox=(-115.406575, 36.129554, -115.062066, 36.401481),
            importance=0.724,
            place_type="city",
        )
        labels = [
            OcrLabel("Las Vegas", x=390, y=267, width=102, height=23, confidence=99),
            OcrLabel("Huntridge", x=250, y=180, width=92, height=24, confidence=98),
        ]
        place_controls = [
            ControlPoint(
                label=OcrLabel(name, x=100 + index * 20, y=100 + index * 16, width=80, height=20, confidence=95),
                geocode=GeocodeResult(
                    label=name,
                    lon=-115.20 + index * 0.01,
                    lat=36.10 + index * 0.01,
                    display_name=f"{name}, suburb",
                    bbox=None,
                    importance=0.5,
                ),
            )
            for index, name in enumerate(("Huntridge", "Charleston Heights", "Angel Park Lindell"))
        ]
        geocode_network_modes: list[bool] = []

        def fake_geocoded_controls(*args, allow_network: bool = True, **kwargs):
            geocode_network_modes.append(allow_network)
            if allow_network:
                raise AssertionError("live geocoding should not run when place controls are ready")
            return []

        with (
            patch("map_boundary_builder.georeference.build_osm_place_control_points", return_value=place_controls),
            patch("map_boundary_builder.georeference.build_geocoded_control_points", side_effect=fake_geocoded_controls),
        ):
            controls = build_control_points(labels, "Las Vegas", center)

        self.assertEqual(controls, place_controls)
        self.assertEqual(geocode_network_modes, [False])

    def test_tight_five_control_label_fit_skips_road_refinement(self) -> None:
        context = CityContext(
            query="Dallas",
            center=GeocodeResult(
                label="Dallas",
                lon=-96.797,
                lat=32.776,
                display_name="Dallas, Dallas County, Texas, United States",
                bbox=(-97.0, 32.6, -96.5, 33.0),
                importance=0.72,
                place_type="city",
            ),
            inferred=True,
        )

        self.assertFalse(
            should_try_road_refinement(
                context,
                meters_per_pixel=10.68,
                inlier_count=5,
                residual_median_m=126.0,
                residual_p90_m=374.7,
                spread=397085.0,
                width=2400,
                height=2400,
            )
        )

    def test_sparse_label_fit_can_still_try_road_refinement(self) -> None:
        context = CityContext(
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

        with patch("map_boundary_builder.georeference.has_local_road_points", return_value=True):
            self.assertTrue(
                should_try_road_refinement(
                    context,
                    meters_per_pixel=10.79,
                    inlier_count=3,
                    residual_median_m=385.6,
                    residual_p90_m=425.5,
                    spread=507150.0,
                    width=2400,
                    height=2400,
                )
            )

    def test_sparse_good_label_fit_skips_live_road_refinement_without_local_roads(self) -> None:
        context = CityContext(
            query="Las Vegas",
            center=GeocodeResult(
                label="Las Vegas",
                lon=-115.1484,
                lat=36.1674,
                display_name="Las Vegas, Clark County, Nevada, United States",
                bbox=(-115.406575, 36.129554, -115.062066, 36.401481),
                importance=0.724,
                place_type="city",
            ),
            inferred=True,
        )

        with patch("map_boundary_builder.georeference.has_local_road_points", return_value=False):
            self.assertFalse(
                should_try_road_refinement(
                    context,
                    meters_per_pixel=40.4,
                    inlier_count=3,
                    residual_median_m=0.0,
                    residual_p90_m=0.0,
                    spread=65000.0,
                    width=393,
                    height=523,
                )
            )

        with patch("map_boundary_builder.georeference.has_local_road_points", return_value=True):
            self.assertTrue(
                should_try_road_refinement(
                    context,
                    meters_per_pixel=40.4,
                    inlier_count=3,
                    residual_median_m=0.0,
                    residual_p90_m=0.0,
                    spread=65000.0,
                    width=393,
                    height=523,
                )
            )

    def test_sparse_reasonable_label_fit_skips_live_road_refinement_without_local_roads(self) -> None:
        context = CityContext(
            query="Dallas",
            center=GeocodeResult(
                label="Dallas",
                lon=-96.797,
                lat=32.776,
                display_name="Dallas, Dallas County, Texas, United States",
                bbox=(-97.0, 32.6, -96.5, 33.0),
                importance=0.72,
                place_type="city",
            ),
            inferred=True,
        )

        with patch("map_boundary_builder.georeference.has_local_road_points", return_value=False):
            self.assertFalse(
                should_try_road_refinement(
                    context,
                    meters_per_pixel=16.0,
                    inlier_count=3,
                    residual_median_m=1080.0,
                    residual_p90_m=1350.0,
                    spread=76000.0,
                    width=680,
                    height=551,
                )
            )

        with patch("map_boundary_builder.georeference.has_local_road_points", return_value=True):
            self.assertTrue(
                should_try_road_refinement(
                    context,
                    meters_per_pixel=16.0,
                    inlier_count=3,
                    residual_median_m=1080.0,
                    residual_p90_m=1350.0,
                    spread=76000.0,
                    width=680,
                    height=551,
                )
            )

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

    def test_filename_context_queries_extract_city_without_provider_noise(self) -> None:
        queries = filename_context_queries("Avride Dallas df72214 small variant.png")

        self.assertIn("Dallas", queries)
        self.assertNotIn("Avride Dallas", queries)
        self.assertNotIn("Small Variant", queries)
        self.assertNotIn("Dallas Png", queries)
        self.assertNotIn("Variant Png", queries)

    def test_filename_city_contexts_use_cached_city_and_bay_area_hints(self) -> None:
        dallas_contexts = filename_city_contexts("Avride Dallas df72214 small variant.png")
        bay_area_contexts = filename_city_contexts("Waymo Bay Area screenshot.png")

        self.assertTrue(dallas_contexts)
        self.assertEqual(dallas_contexts[0].query, "Dallas")
        self.assertTrue(bay_area_contexts)
        self.assertEqual(bay_area_contexts[0].query, "San Francisco Bay Area")

    def test_context_hint_fast_path_skips_expensive_context_inference(self) -> None:
        labels = [OcrLabel("Belmont", x=10, y=10, width=80, height=24, confidence=96)]
        hinted_result = object()

        with (
            patch("map_boundary_builder.runner.georeference_from_labels", return_value=hinted_result) as label_fit,
            patch("map_boundary_builder.runner.is_credible_context_hint_georeference", return_value=True),
            patch("map_boundary_builder.runner.road_contexts_from_labels") as road_contexts,
        ):
            result = fit_georeference(
                labels,
                Path("input.png"),
                pixel_geometry=object(),
                rgb=None,
                city_input=None,
                context_hints=[SimpleNamespace(query="Dallas")],
                width=680,
                height=551,
                coverage_ratio=0.27,
                min_control_points=3,
                label_y_min=None,
                label_y_max=None,
                progress=None,
            )

        self.assertIs(result, hinted_result)
        label_fit.assert_called_once()
        road_contexts.assert_not_called()

    def test_context_hint_failure_falls_back_to_normal_context_inference(self) -> None:
        labels = [OcrLabel("Nashville", x=1141, y=454, width=162, height=44, confidence=98)]
        weak_result = object()
        fallback_result = object()

        with (
            patch(
                "map_boundary_builder.runner.georeference_from_labels",
                side_effect=[weak_result, fallback_result],
            ) as label_fit,
            patch("map_boundary_builder.runner.is_credible_context_hint_georeference", return_value=False),
            patch("map_boundary_builder.runner.road_contexts_from_labels", return_value=[]),
        ):
            result = fit_georeference(
                labels,
                Path("input.png"),
                pixel_geometry=object(),
                rgb=None,
                city_input=None,
                context_hints=[SimpleNamespace(query="Dallas")],
                width=1920,
                height=1080,
                coverage_ratio=0.22,
                min_control_points=3,
                label_y_min=None,
                label_y_max=None,
                progress=None,
            )

        self.assertIs(result, fallback_result)
        self.assertEqual(label_fit.call_count, 2)

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

    def test_detect_label_marker_dots_reuses_supplied_rgb(self) -> None:
        rgb = np.full((20, 20, 3), 255, dtype=np.uint8)

        with patch("map_boundary_builder.extract.load_rgb", side_effect=AssertionError("should not reload")):
            markers = detect_label_marker_dots("missing.png", rgb=rgb)

        self.assertEqual(markers, [])


if __name__ == "__main__":
    unittest.main()
