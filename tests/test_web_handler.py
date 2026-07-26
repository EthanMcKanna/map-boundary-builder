import os
import unittest
from http import HTTPStatus
from tempfile import TemporaryDirectory
from unittest.mock import patch

import map_boundary_builder.web as web
from map_boundary_builder.pipeline import PipelineOptions


class LocalWebHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        web.RUNS.clear()

    def tearDown(self) -> None:
        web.RUNS.clear()

    def create_run(self, fields: dict[str, str], filename: str = "Waymo Dallas.png"):
        request = web.BoundaryWebHandler.__new__(web.BoundaryWebHandler)
        request.parse_upload_request = lambda: (fields, {"image": (filename, b"image-bytes")})
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
        return captured_response, captured_threads

    def test_create_run_builds_pipeline_options(self) -> None:
        response, threads = self.create_run(
            {
                "city": "Dallas",
                "simplify_px": "4.5",
                "min_confidence": "0.7",
                "min_control_points": "5",
                "seed_x": "12",
                "seed_y": "34",
                "target_color": "#7c3aed",
            }
        )
        self.assertEqual(response["status"], HTTPStatus.CREATED)
        self.assertEqual(response["payload"], {"id": "run-id", "status_url": "/api/runs/run-id"})
        self.assertEqual(len(threads), 1)
        state, options = threads[0].args
        self.assertEqual(state.city, "Dallas")
        self.assertEqual(state.original_filename, "Waymo Dallas.png")
        self.assertEqual(state.profile["pipeline_version"], "pipeline-test")
        self.assertIsInstance(options, PipelineOptions)
        self.assertEqual(options.simplify_px, 4.5)
        self.assertEqual(options.min_confidence, 0.7)
        self.assertEqual(options.min_control_points, 5)
        self.assertEqual(options.seed_point, (12.0, 34.0))
        self.assertEqual(options.target_rgb, (124, 58, 237))

    def test_create_run_treats_auto_city_as_no_hint(self) -> None:
        _response, threads = self.create_run({"city": "Auto"})
        state, _options = threads[0].args
        self.assertIsNone(state.city)

    def test_needs_city_event_is_terminal_and_stores_summary(self) -> None:
        _response, threads = self.create_run({})
        state, _options = threads[0].args
        summary = {"status": "needs_city", "needs_city": {"reason": "no_city_context"}}
        web.record_event(
            state,
            {
                "stage": "needs_city",
                "message": "City required",
                "percent": 100,
                "status": "needs_city",
                "details": summary,
            },
        )
        self.assertEqual(state.status, "needs_city")
        self.assertEqual(state.summary, summary)
        self.assertIn("needs_city", web.TERMINAL_STATUSES)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["status"], "needs_city")

    def test_failure_event_records_error(self) -> None:
        _response, threads = self.create_run({})
        state, _options = threads[0].args
        web.record_event(
            state,
            {
                "stage": "failed",
                "message": "Generation failed",
                "percent": 100,
                "status": "failed",
                "details": {"error": "No service-area polygon could be extracted"},
            },
        )
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.error, "No service-area polygon could be extracted")
