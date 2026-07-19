import os
import unittest
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import map_boundary_builder.web as web


class LocalWebHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        web.RUNS.clear()

    def tearDown(self) -> None:
        web.RUNS.clear()

    def test_background_run_preserves_original_filename_hint(self) -> None:
        request = web.BoundaryWebHandler.__new__(web.BoundaryWebHandler)
        request.parse_upload_request = lambda: (
            {"source_was_svg": "1"},
            {"image": ("Waymo Bay Area.png", b"image-bytes")},
        )
        captured_response: dict[str, object] = {}
        captured_threads: list[object] = []

        def send_json(payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            captured_response["payload"] = payload
            captured_response["status"] = status

        class FakeThread:
            def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool) -> None:
                self.target = target
                self.args = args
                self.daemon = daemon
                captured_threads.append(self)

            def start(self) -> None:
                return None

        request.send_json = send_json

        with TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"MAP_BOUNDARY_WEB_OUT": temp_dir}),
                patch("map_boundary_builder.web.secrets.token_hex", return_value="run-id"),
                patch("map_boundary_builder.web.get_pipeline_version", return_value="pipeline-test"),
                patch("map_boundary_builder.web.threading.Thread", FakeThread),
            ):
                request.handle_create_run()

        self.assertEqual(captured_response["status"], HTTPStatus.CREATED)
        self.assertEqual(captured_response["payload"], {"id": "run-id", "status_url": "/api/runs/run-id"})
        self.assertEqual(len(captured_threads), 1)
        state, options = captured_threads[0].args
        self.assertEqual(state.profile["pipeline_version"], "pipeline-test")
        self.assertEqual(state.original_filename, "Waymo Bay Area.png")
        self.assertEqual(options.filename_hint, "Waymo Bay Area.png")
        self.assertTrue(options.source_was_svg)

    def test_background_run_treats_auto_city_as_no_hint(self) -> None:
        request = web.BoundaryWebHandler.__new__(web.BoundaryWebHandler)
        request.parse_upload_request = lambda: (
            {"city": "Auto", "include_overlay": "0"},
            {"image": ("Waymo Dallas.png", b"image-bytes")},
        )
        captured_response: dict[str, object] = {}
        captured_threads: list[object] = []

        def send_json(payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            captured_response["payload"] = payload
            captured_response["status"] = status

        class FakeThread:
            def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool) -> None:
                self.args = args
                captured_threads.append(self)

            def start(self) -> None:
                return None

        request.send_json = send_json
        with TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"MAP_BOUNDARY_WEB_OUT": temp_dir}),
                patch("map_boundary_builder.web.secrets.token_hex", return_value="run-id"),
                patch("map_boundary_builder.web.threading.Thread", FakeThread),
            ):
                request.handle_create_run()

        self.assertEqual(captured_response["status"], HTTPStatus.CREATED)
        self.assertEqual(captured_response["payload"], {"id": "run-id", "status_url": "/api/runs/run-id"})
        state, _options = captured_threads[0].args
        self.assertIsNone(state.city)

    def test_city_hint_for_request_normalizes_auto_placeholders(self) -> None:
        self.assertIsNone(web.city_hint_for_request({}))
        self.assertIsNone(web.city_hint_for_request({"city": ""}))
        self.assertIsNone(web.city_hint_for_request({"city": "Auto"}))
        self.assertIsNone(web.city_hint_for_request({"city": "Auto-detect"}))
        self.assertIsNone(web.city_hint_for_request({"city": "automatic"}))
        self.assertEqual(web.city_hint_for_request({"city": " Dallas "}), "Dallas")

    def test_background_failure_records_terminal_failed_event(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state = web.RunState(
                run_id="run-id",
                city=None,
                original_filename="blank.png",
                run_dir=Path(temp_dir),
                image_path=Path(temp_dir) / "blank.png",
                output_path=Path(temp_dir) / "boundary.geojson",
                debug_dir=Path(temp_dir) / "debug",
                profile={"pipeline_version": "pipeline-test"},
            )
            state.image_path.write_bytes(b"not-an-image")

            def fail_build(*args, **kwargs):
                kwargs["progress"](
                    {
                        "stage": "extract",
                        "message": "Extracting service-area pixels",
                        "percent": 18,
                        "status": "running",
                        "details": {"width": 160, "height": 160},
                    }
                )
                raise ValueError("No service-area polygon could be extracted from the image.")

            with patch("map_boundary_builder.web.build_boundary", side_effect=fail_build):
                web.run_worker(state, web.BoundaryBuildOptions())

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.percent, 100)
        self.assertEqual(state.error, "No service-area polygon could be extracted from the image.")
        self.assertEqual(state.events[-1]["stage"], "failed")
        self.assertEqual(state.events[-1]["status"], "failed")
        self.assertEqual(state.events[-1]["message"], "Generation failed")
        self.assertEqual(
            state.events[-1]["details"],
            {"error": "No service-area polygon could be extracted from the image."},
        )
        self.assertIn("build_stage_elapsed_s", state.profile)

    def test_background_run_ignores_legacy_catalog_fields(self) -> None:
        request = web.BoundaryWebHandler.__new__(web.BoundaryWebHandler)
        request.parse_upload_request = lambda: (
            {
                "allow_catalog": "1",
                "catalog_probe_only": "1",
                "catalog_probe_missed": "1",
                "catalog_probe_miss_low_iou": "1",
                "fast_catalog_handoff": "1",
            },
            {"image": ("New service area.png", b"image-bytes")},
        )
        captured_response: dict[str, object] = {}
        captured_threads: list[object] = []

        def send_json(payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            captured_response["payload"] = payload
            captured_response["status"] = status

        class FakeThread:
            def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool) -> None:
                self.args = args
                captured_threads.append(self)

            def start(self) -> None:
                return None

        request.send_json = send_json
        with TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"MAP_BOUNDARY_WEB_OUT": temp_dir}),
                patch("map_boundary_builder.web.secrets.token_hex", return_value="run-id"),
                patch("map_boundary_builder.web.threading.Thread", FakeThread),
            ):
                request.handle_create_run()

        self.assertEqual(captured_response["status"], HTTPStatus.CREATED)
        _state, options = captured_threads[0].args
        self.assertFalse(options.allow_catalog)
        self.assertFalse(options.catalog_probe_only)
        self.assertFalse(options.catalog_probe_missed)
        self.assertFalse(options.catalog_probe_miss_low_iou)


if __name__ == "__main__":
    unittest.main()
