import unittest
from http import HTTPStatus
from io import BytesIO
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import api.index as api_index
from map_boundary_builder.pipeline import PipelineResult


def png_bytes(*, compress_level: int) -> bytes:
    image = Image.new("RGB", (24, 24), (60, 120, 235))
    for x in range(24):
        image.putpixel((x, x), (255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def complete_result() -> PipelineResult:
    return PipelineResult(
        status="complete",
        reason=None,
        summary={"status": "complete", "city": "Austin, TX"},
        geojson={"type": "FeatureCollection", "features": []},
    )


def needs_city_result() -> PipelineResult:
    return PipelineResult(
        status="needs_city",
        reason="no_city_context",
        summary={
            "status": "needs_city",
            "needs_city": {"reason": "no_city_context", "ocr_label_count": 0, "sample_labels": []},
        },
    )


def failed_result(*, cacheable: bool) -> PipelineResult:
    return PipelineResult(
        status="failed",
        reason="no_boundary_found" if cacheable else "georeference_error",
        summary={"status": "failed", "message": "boom"},
        cacheable=cacheable,
    )


class ApiCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self._cache_dir_patch = patch.object(api_index, "RUN_RESULT_CACHE_DIR", Path(self._temp.name))
        self._cache_dir_patch.start()
        api_index._RUN_RESULT_MEMORY_CACHE.clear()

    def tearDown(self) -> None:
        self._cache_dir_patch.stop()
        self._temp.cleanup()
        api_index._RUN_RESULT_MEMORY_CACHE.clear()

    def post_run(self, image_bytes: bytes, fields: dict[str, str], result: PipelineResult, calls: list):
        request = api_index.handler.__new__(api_index.handler)
        request.parse_upload_request = lambda: (fields, {"image": ("upload.png", image_bytes)}, "multipart")
        captured: dict[str, object] = {}

        def send_json(payload, *, status=HTTPStatus.OK):
            captured["payload"] = payload
            captured["status"] = status

        request.send_json = send_json

        def fake_run_pipeline(image_path, **kwargs):
            calls.append(kwargs.get("city"))
            if isinstance(result, Exception):
                raise result
            return result

        with patch("map_boundary_builder.pipeline.run_pipeline", fake_run_pipeline):
            request.handle_create_run()
        return captured

    def test_complete_result_is_cached_by_raw_bytes(self) -> None:
        calls: list = []
        image = png_bytes(compress_level=1)
        first = self.post_run(image, {}, complete_result(), calls)
        self.assertEqual(first["status"], HTTPStatus.CREATED)
        second = self.post_run(image, {}, complete_result(), calls)
        self.assertEqual(second["status"], HTTPStatus.OK)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second["payload"]["status"], "complete")
        self.assertEqual(second["payload"]["profile"]["cache_hit"], "raw")

    def test_reencoded_image_hits_visual_cache(self) -> None:
        calls: list = []
        original = png_bytes(compress_level=1)
        reencoded = png_bytes(compress_level=9)
        self.assertNotEqual(original, reencoded)
        self.post_run(original, {}, complete_result(), calls)
        second = self.post_run(reencoded, {}, complete_result(), calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second["payload"]["profile"]["cache_hit"], "visual")

    def test_needs_city_is_cached_and_keyed_by_city(self) -> None:
        calls: list = []
        image = png_bytes(compress_level=1)
        first = self.post_run(image, {}, needs_city_result(), calls)
        self.assertEqual(first["status"], HTTPStatus.OK)
        self.assertEqual(first["payload"]["status"], "needs_city")

        cached = self.post_run(image, {}, needs_city_result(), calls)
        self.assertEqual(cached["payload"]["profile"]["cache_hit"], "raw")
        self.assertEqual(len(calls), 1)

        retry = self.post_run(image, {"city": "Austin, TX"}, complete_result(), calls)
        self.assertEqual(len(calls), 2)
        self.assertEqual(retry["payload"]["status"], "complete")
        self.assertEqual(calls[-1], "Austin, TX")

    def test_deterministic_failure_is_cached_as_422(self) -> None:
        calls: list = []
        image = png_bytes(compress_level=1)
        first = self.post_run(image, {}, failed_result(cacheable=True), calls)
        self.assertEqual(first["status"], HTTPStatus.UNPROCESSABLE_ENTITY)
        second = self.post_run(image, {}, failed_result(cacheable=True), calls)
        self.assertEqual(second["status"], HTTPStatus.UNPROCESSABLE_ENTITY)
        self.assertEqual(len(calls), 1)

    def test_transient_failure_is_never_cached(self) -> None:
        calls: list = []
        image = png_bytes(compress_level=1)
        first = self.post_run(image, {}, failed_result(cacheable=False), calls)
        self.assertEqual(first["status"], HTTPStatus.UNPROCESSABLE_ENTITY)
        second = self.post_run(image, {}, failed_result(cacheable=False), calls)
        self.assertEqual(len(calls), 2)

    def test_pipeline_exception_returns_503_and_is_not_cached(self) -> None:
        calls: list = []
        image = png_bytes(compress_level=1)
        first = self.post_run(image, {}, RuntimeError("onnx session exploded"), calls)
        self.assertEqual(first["status"], HTTPStatus.SERVICE_UNAVAILABLE)
        second = self.post_run(image, {}, complete_result(), calls)
        self.assertEqual(second["status"], HTTPStatus.CREATED)
        self.assertEqual(len(calls), 2)

    def test_options_change_cache_identity(self) -> None:
        calls: list = []
        image = png_bytes(compress_level=1)
        self.post_run(image, {}, complete_result(), calls)
        self.post_run(image, {"simplify_px": "2.0"}, complete_result(), calls)
        self.assertEqual(len(calls), 2)

    def test_response_status_mapping(self) -> None:
        self.assertEqual(api_index.response_status("complete", cached=False), HTTPStatus.CREATED)
        self.assertEqual(api_index.response_status("complete", cached=True), HTTPStatus.OK)
        self.assertEqual(api_index.response_status("needs_city", cached=False), HTTPStatus.OK)
        self.assertEqual(api_index.response_status("needs_city", cached=True), HTTPStatus.OK)
        self.assertEqual(api_index.response_status("failed", cached=True), HTTPStatus.UNPROCESSABLE_ENTITY)
