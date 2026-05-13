from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .extract import DEFAULT_SIMPLIFY_PX
from .github_reports import FailureReport, GithubReportError, create_failure_issue
from .image_io import safe_image_extension
from .ocr import parse_client_ocr_labels
from .runner import BoundaryBuildOptions, build_boundary

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
TERMINAL_STATUSES = {"complete", "error"}
RUNS: dict[str, "RunState"] = {}
RUNS_LOCK = threading.Lock()


@dataclass
class RunState:
    run_id: str
    city: str | None
    original_filename: str
    run_dir: Path
    image_path: Path
    output_path: Path
    debug_dir: Path
    ocr_labels: list[Any] | None = None
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    percent: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    error: str | None = None
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "id": self.run_id,
                "city": self.summary["city"] if self.summary else self.city or "Auto",
                "filename": self.original_filename,
                "status": self.status,
                "percent": self.percent,
                "created_at": self.created_at,
                "summary": self.summary,
                "error": self.error,
                "events": self.events[-20:],
                "artifacts": artifact_urls(self) if self.status == "complete" else {},
            }


class RequestError(ValueError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class BoundaryWebHandler(BaseHTTPRequestHandler):
    server_version = "MapBoundaryWeb/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_asset("index.html")
                return
            if parsed.path.startswith("/static/"):
                self.send_asset(unquote(parsed.path.removeprefix("/static/")))
                return
            if parsed.path.startswith("/api/runs/"):
                self.handle_run_get(parsed.path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except RequestError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)
        except BrokenPipeError:
            return

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                self.handle_create_run()
                return
            if parsed.path == "/api/reports":
                self.handle_create_report()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except RequestError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[map-boundary-web] {self.address_string()} - {fmt % args}")

    def handle_run_get(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            raise RequestError(HTTPStatus.NOT_FOUND, "Run not found")
        run_id = parts[2]
        state = get_run(run_id)
        if state is None:
            raise RequestError(HTTPStatus.NOT_FOUND, "Run not found")

        if len(parts) == 3:
            self.send_json(state.snapshot())
            return
        if len(parts) == 4 and parts[3] == "events":
            self.stream_events(state)
            return
        if len(parts) == 5 and parts[3] == "artifact":
            self.send_artifact(state, parts[4])
            return
        raise RequestError(HTTPStatus.NOT_FOUND, "Run not found")

    def handle_create_run(self) -> None:
        fields, files = self.parse_multipart()
        city = fields.get("city", "").strip() or None
        upload = files.get("image")
        if upload is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Image upload is required.")

        original_filename, image_bytes = upload
        if not image_bytes:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Uploaded image is empty.")

        run_id = secrets.token_hex(6)
        output_root = Path(os.environ.get("MAP_BOUNDARY_WEB_OUT", "out/web-runs"))
        run_dir = output_root / run_id
        debug_dir = run_dir / "debug"
        run_dir.mkdir(parents=True, exist_ok=True)
        ext = safe_extension(original_filename)
        image_path = run_dir / f"input{ext}"
        image_path.write_bytes(image_bytes)
        output_path = run_dir / "boundary.geojson"

        state = RunState(
            run_id=run_id,
            city=city,
            original_filename=Path(original_filename).name or "uploaded-image",
            run_dir=run_dir,
            image_path=image_path,
            output_path=output_path,
            debug_dir=debug_dir,
            ocr_labels=parse_client_ocr_labels(fields.get("ocr_labels")),
        )
        with RUNS_LOCK:
            RUNS[run_id] = state

        options = BoundaryBuildOptions(
            simplify_px=float_field(fields, "simplify_px", DEFAULT_SIMPLIFY_PX, 0.0, 10.0),
            min_confidence=float_field(fields, "min_confidence", 0.55, 0.0, 1.0),
            min_control_points=int_field(fields, "min_control_points", 3, 0, 12),
        )
        record_event(
            state,
            {
                "stage": "queued",
                "message": "Run queued",
                "percent": 1,
                "status": "queued",
            },
        )
        thread = threading.Thread(target=run_worker, args=(state, options), daemon=True)
        thread.start()
        self.send_json({"id": run_id, "status_url": f"/api/runs/{run_id}"}, status=HTTPStatus.CREATED)

    def handle_create_report(self) -> None:
        fields, files = self.parse_multipart()
        upload = files.get("image")
        if upload is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Image upload is required.")
        original_filename, image_bytes = upload
        try:
            events = json.loads(fields.get("events", "[]") or "[]")
        except json.JSONDecodeError:
            events = []
        try:
            settings = json.loads(fields.get("settings", "{}") or "{}")
        except json.JSONDecodeError:
            settings = {}
        try:
            summary = json.loads(fields.get("summary", "{}") or "{}")
        except json.JSONDecodeError:
            summary = {}
        try:
            result = create_failure_issue(
                FailureReport(
                    filename=original_filename,
                    image_bytes=image_bytes,
                    error=fields.get("error", "").strip() or "Generation failed without an error message.",
                    issue_type=fields.get("issue_type", "").strip() or "Generation issue",
                    generation_status=fields.get("generation_status", "").strip() or "unknown",
                    user_note=fields.get("user_note", "").strip() or None,
                    run_id=fields.get("run_id", "").strip() or None,
                    events=events if isinstance(events, list) else [],
                    user_agent=fields.get("user_agent", "").strip() or self.headers.get("User-Agent"),
                    page_url=fields.get("page_url", "").strip() or None,
                    settings=settings if isinstance(settings, dict) else {},
                    summary=summary if isinstance(summary, dict) else {},
                )
            )
        except GithubReportError as exc:
            raise RequestError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
        self.send_json(result, status=HTTPStatus.CREATED)

    def parse_multipart(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        content_type = self.headers.get("Content-Type", "")
        match = re.search(r'boundary="?([^";]+)"?', content_type)
        if "multipart/form-data" not in content_type or match is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data.")
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Request body is empty.")
        if length > MAX_UPLOAD_BYTES:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Upload is larger than 50 MB.")

        body = self.rfile.read(length)
        boundary = ("--" + match.group(1)).encode()
        fields: dict[str, str] = {}
        files: dict[str, tuple[str, bytes]] = {}
        for raw_part in body.split(boundary):
            part = raw_part
            if part.startswith(b"\r\n"):
                part = part[2:]
            if part.endswith(b"\r\n"):
                part = part[:-2]
            if not part or part == b"--":
                continue
            if part.endswith(b"--"):
                part = part[:-2].rstrip(b"\r\n")
            header_blob, separator, content = part.partition(b"\r\n\r\n")
            if not separator:
                continue
            headers = header_blob.decode("utf-8", "replace").split("\r\n")
            disposition = next(
                (header for header in headers if header.lower().startswith("content-disposition:")),
                "",
            )
            name_match = re.search(r'name="([^"]+)"', disposition)
            if name_match is None:
                continue
            field_name = name_match.group(1)
            filename_match = re.search(r'filename="([^"]*)"', disposition)
            if filename_match is not None:
                files[field_name] = (filename_match.group(1), content)
            else:
                fields[field_name] = content.decode("utf-8", "replace").strip()
        return fields, files

    def stream_events(self, state: RunState) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        index = 0
        while True:
            with state.condition:
                if index >= len(state.events) and state.status not in TERMINAL_STATUSES:
                    state.condition.wait(timeout=15)
                new_events = state.events[index:]
                index = len(state.events)
                terminal = state.status in TERMINAL_STATUSES

            if not new_events:
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
            for event in new_events:
                self.wfile.write(sse_event(event))
                self.wfile.flush()
            if terminal and index >= len(state.events):
                self.close_connection = True
                return

    def send_artifact(self, state: RunState, name: str) -> None:
        artifact_path = artifact_file(state, name)
        if artifact_path is None or not artifact_path.exists():
            raise RequestError(HTTPStatus.NOT_FOUND, "Artifact not available.")
        mime = mimetypes.guess_type(str(artifact_path))[0] or "application/octet-stream"
        if artifact_path.suffix == ".geojson":
            mime = "application/geo+json"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(artifact_path.stat().st_size))
        if name in {"geojson", "summary"}:
            self.send_header("Content-Disposition", f'attachment; filename="{artifact_path.name}"')
        self.end_headers()
        with artifact_path.open("rb") as handle:
            self.wfile.write(handle.read())

    def send_asset(self, name: str) -> None:
        if "/" in name or "\\" in name or name.startswith("."):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        asset = resources.files("map_boundary_builder").joinpath("web_assets", name)
        if not asset.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = asset.read_bytes()
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if name.endswith(".js"):
            mime = "text/javascript; charset=utf-8"
        elif name.endswith(".css"):
            mime = "text/css; charset=utf-8"
        elif name.endswith(".html"):
            mime = "text/html; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_worker(state: RunState, options: BoundaryBuildOptions) -> None:
    try:
        result = build_boundary(
            state.image_path,
            state.city,
            state.output_path,
            debug_dir=state.debug_dir,
            options=options,
            progress=lambda event: record_event(state, event),
            ocr_labels=state.ocr_labels,
        )
        with state.condition:
            state.summary = result.summary
            state.condition.notify_all()
    except Exception as exc:
        record_event(
            state,
            {
                "stage": "error",
                "message": str(exc),
                "percent": state.percent,
                "status": "error",
            },
        )


def record_event(state: RunState, event: dict[str, Any]) -> None:
    enriched = {
        "timestamp": time.time(),
        **event,
    }
    with state.condition:
        state.events.append(enriched)
        state.status = str(enriched.get("status", state.status))
        state.percent = int(enriched.get("percent", state.percent))
        if state.status == "complete":
            state.summary = enriched.get("details") if isinstance(enriched.get("details"), dict) else state.summary
        elif state.status == "error":
            state.error = str(enriched.get("message", "Run failed."))
        state.condition.notify_all()


def get_run(run_id: str) -> RunState | None:
    with RUNS_LOCK:
        return RUNS.get(run_id)


def artifact_urls(state: RunState) -> dict[str, str]:
    base = f"/api/runs/{state.run_id}/artifact"
    urls = {
        "input": f"{base}/input",
        "geojson": f"{base}/geojson",
        "summary": f"{base}/summary",
    }
    if artifact_file(state, "mask") and artifact_file(state, "mask").exists():
        urls["mask"] = f"{base}/mask"
    if artifact_file(state, "overlay") and artifact_file(state, "overlay").exists():
        urls["overlay"] = f"{base}/overlay"
    return urls


def artifact_file(state: RunState, name: str) -> Path | None:
    if name == "input":
        return state.image_path
    if name == "geojson":
        return state.output_path
    if name == "mask":
        return state.debug_dir / "boundary.mask.png"
    if name == "overlay":
        return state.debug_dir / "boundary.overlay.png"
    if name == "summary":
        return state.debug_dir / "boundary.summary.json"
    return None


def sse_event(event: dict[str, Any]) -> bytes:
    return f"event: update\ndata: {json.dumps(event)}\n\n".encode("utf-8")


def safe_extension(filename: str) -> str:
    return safe_image_extension(filename)


def float_field(fields: dict[str, str], name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(fields.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def int_field(fields: dict[str, str], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(fields.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map-boundary-web",
        description="Run the interactive Map Boundary Builder web app.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", default="out/web-runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ["MAP_BOUNDARY_WEB_OUT"] = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), BoundaryWebHandler)
    print(f"Map Boundary Builder running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Map Boundary Builder.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
