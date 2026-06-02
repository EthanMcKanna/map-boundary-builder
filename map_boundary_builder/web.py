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
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .asset_response import web_asset_response
from .extract import DEFAULT_SIMPLIFY_PX
from .github_reports import FailureReport, GithubReportError, create_failure_issue
from .image_io import safe_image_extension
from .pipeline_version import get_pipeline_version, pipeline_version_dependency_versions
from .runner import BoundaryBuildOptions, CatalogProbeMiss, build_boundary
from .runtime_warmup import prewarm_generation_runtime, should_prewarm_generation_runtime
from .upload_payload import UploadPayloadError, json_upload_body_limit, parse_json_upload_body

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
TERMINAL_STATUSES = {"complete", "error", "failed"}
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
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    percent: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
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
                "profile": self.profile,
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
            if parsed.path == "/api/health":
                payload: dict[str, Any] = {
                    "ok": True,
                    "runtime": "local-python",
                    "pipeline_version": get_pipeline_version(),
                    "runtime_dependencies": dict(pipeline_version_dependency_versions()),
                }
                warm = first_query_value(parse_qs(parsed.query), "warm")
                if should_prewarm_generation_runtime(warm):
                    payload["warm"] = prewarm_generation_runtime()
                self.send_json(payload)
                return
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
        fields, files = self.parse_upload_request()
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
        catalog_probe_only = bool_field(fields, "catalog_probe_only", default=False)
        fast_catalog_handoff = bool_field(fields, "fast_catalog_handoff", default=False)
        allow_catalog = allow_catalog_for_request(fields)
        if catalog_probe_only or fast_catalog_handoff:
            events: list[dict[str, Any]] = []
            profile: dict[str, Any] = {
                "pipeline_version": get_pipeline_version(),
                "upload_bytes": len(image_bytes),
            }

            def progress(event: dict[str, Any]) -> None:
                events.append({"timestamp": time.time(), **event})

            options = BoundaryBuildOptions(
                simplify_px=float_field(fields, "simplify_px", DEFAULT_SIMPLIFY_PX, 0.0, 10.0),
                min_confidence=float_field(fields, "min_confidence", 0.55, 0.0, 1.0),
                min_control_points=int_field(fields, "min_control_points", 3, 0, 12),
                allow_catalog=allow_catalog,
                catalog_probe_only=catalog_probe_only,
                catalog_probe_missed=fast_catalog_handoff or bool_field(fields, "catalog_probe_missed", default=False),
                write_mask_artifact=False,
                filename_hint=original_filename,
                source_was_svg=bool_field(fields, "source_was_svg", default=False),
            )
            try:
                build_started = time.perf_counter()
                result = build_boundary(image_path, city, output_path, debug_dir=None, options=options, progress=progress)
            except CatalogProbeMiss as exc:
                events = terminal_events(
                    events,
                    stage="catalog_miss",
                    message="Catalog probe missed",
                    status="catalog_miss",
                    details=exc.details,
                )
                profile["build_boundary_s"] = elapsed_seconds(build_started)
                profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(events)
                profile["total_before_send_s"] = profile["build_boundary_s"]
                self.send_json(
                    {
                        "id": run_id,
                        "filename": Path(original_filename).name or "uploaded-image",
                        "status": "catalog_miss",
                        "percent": 100,
                        "error": str(exc),
                        "catalog_probe_miss": exc.details,
                        "events": events[-20:],
                        "profile": profile,
                    },
                    status=HTTPStatus.OK,
                )
                return
            profile["build_boundary_s"] = elapsed_seconds(build_started)
            profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(events)
            profile["total_before_send_s"] = profile["build_boundary_s"]
            self.send_json(
                {
                    "id": run_id,
                    "city": result.summary["city"],
                    "filename": Path(original_filename).name or "uploaded-image",
                    "status": "complete",
                    "percent": 100,
                    "summary": result.summary,
                    "events": events[-20:],
                    "profile": profile,
                    "artifacts": {"geojson_inline": result.geojson},
                },
                status=HTTPStatus.CREATED,
            )
            return

        state = RunState(
            run_id=run_id,
            city=city,
            original_filename=Path(original_filename).name or "uploaded-image",
            run_dir=run_dir,
            image_path=image_path,
            output_path=output_path,
            debug_dir=debug_dir,
            profile={
                "pipeline_version": get_pipeline_version(),
                "upload_bytes": len(image_bytes),
            },
        )
        with RUNS_LOCK:
            RUNS[run_id] = state

        options = BoundaryBuildOptions(
            simplify_px=float_field(fields, "simplify_px", DEFAULT_SIMPLIFY_PX, 0.0, 10.0),
            min_confidence=float_field(fields, "min_confidence", 0.55, 0.0, 1.0),
            min_control_points=int_field(fields, "min_control_points", 3, 0, 12),
            allow_catalog=allow_catalog,
            catalog_probe_missed=bool_field(fields, "catalog_probe_missed", default=False),
            catalog_probe_miss_low_iou=bool_field(fields, "catalog_probe_miss_low_iou", default=False),
            filename_hint=original_filename,
            source_was_svg=bool_field(fields, "source_was_svg", default=False),
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
        fields, files = self.parse_upload_request()
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
            profile = json.loads(fields.get("profile", "{}") or "{}")
        except json.JSONDecodeError:
            profile = {}
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
                    profile=profile if isinstance(profile, dict) else {},
                )
            )
        except GithubReportError as exc:
            raise RequestError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
        self.send_json(result, status=HTTPStatus.CREATED)

    def parse_upload_request(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        content_type = self.headers.get("Content-Type", "").lower()
        if "multipart/form-data" in content_type:
            return self.parse_multipart()
        if "application/json" in content_type:
            return self.parse_json_upload()
        raise RequestError(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data or application/json.")

    def parse_json_upload(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Request body is empty.")
        if length > json_upload_body_limit(MAX_UPLOAD_BYTES):
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Upload is larger than 50 MB.")
        try:
            return parse_json_upload_body(self.rfile.read(length), max_upload_bytes=MAX_UPLOAD_BYTES)
        except UploadPayloadError as exc:
            raise RequestError(exc.status, str(exc)) from exc

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
        try:
            data, mime = web_asset_response(name)
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
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
        build_started = time.perf_counter()
        result = build_boundary(
            state.image_path,
            state.city,
            state.output_path,
            debug_dir=state.debug_dir,
            options=options,
            progress=lambda event: record_event(state, event),
        )
        with state.condition:
            state.summary = result.summary
            profile = dict(state.profile or {})
            profile["build_boundary_s"] = elapsed_seconds(build_started)
            profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(state.events)
            profile["total_before_send_s"] = profile["build_boundary_s"]
            state.profile = profile
            state.condition.notify_all()
    except Exception as exc:
        with state.condition:
            profile = dict(state.profile or {})
            if "build_started" in locals():
                profile["build_boundary_s"] = elapsed_seconds(build_started)
                profile["total_before_send_s"] = profile["build_boundary_s"]
            state.profile = profile
        record_event(
            state,
            {
                "stage": "failed",
                "message": "Generation failed",
                "percent": 100,
                "status": "failed",
                "details": {"error": str(exc)},
            },
        )
        with state.condition:
            profile = dict(state.profile or {})
            if "build_started" in locals():
                profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(state.events)
            state.profile = profile
            state.condition.notify_all()


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started), 6)


def event_stage_elapsed_seconds(events: list[dict[str, Any]]) -> dict[str, float]:
    timestamped: list[tuple[str, float]] = []
    for event in events:
        stage = event.get("stage")
        timestamp = event.get("timestamp")
        if isinstance(stage, str) and isinstance(timestamp, (int, float)):
            timestamped.append((stage, float(timestamp)))

    totals: dict[str, float] = {}
    for (stage, timestamp), (_, next_timestamp) in zip(timestamped, timestamped[1:]):
        totals[stage] = totals.get(stage, 0.0) + max(0.0, next_timestamp - timestamp)
    return {stage: round(total, 6) for stage, total in totals.items()}


def terminal_events(
    events: list[dict[str, Any]],
    *,
    stage: str,
    message: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if events and events[-1].get("stage") == stage and events[-1].get("status") == status:
        return events[-20:]
    event: dict[str, Any] = {
        "timestamp": time.time(),
        "stage": stage,
        "message": message,
        "percent": 100,
        "status": status,
    }
    if details:
        event["details"] = details
    return [*events[-19:], event]


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
        elif state.status in {"error", "failed"}:
            details = enriched.get("details")
            error = details.get("error") if isinstance(details, dict) else None
            state.error = str(error or enriched.get("message", "Run failed."))
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


def first_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if not values:
        return None
    return values[0]


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


def bool_field(fields: dict[str, str], name: str, *, default: bool) -> bool:
    value = fields.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def allow_catalog_for_request(fields: dict[str, str]) -> bool:
    if bool_field(fields, "no_catalog", default=False):
        return False
    return bool_field(fields, "allow_catalog", default=True)


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
