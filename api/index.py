from __future__ import annotations

import base64
from collections import OrderedDict
import gzip
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import time
from io import BytesIO
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

os.environ.setdefault("MAP_BOUNDARY_CACHE_DIR", "/tmp/map-boundary-builder-cache")

from map_boundary_builder.asset_response import web_asset_response
from map_boundary_builder.pipeline_version import get_pipeline_version, pipeline_version_dependency_versions
from map_boundary_builder.runtime_warmup import (
    prewarm_generation_runtime,
    should_prewarm_generation_runtime,
)
from map_boundary_builder.runtime_config import ocr_runtime_config
from map_boundary_builder.upload_payload import (
    UploadPayloadError,
    json_upload_body_limit,
    parse_json_upload_body,
)

DEFAULT_SIMPLIFY_PX = 6.0
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_INLINE_OVERLAY_BYTES = 1_800_000
INLINE_OVERLAY_OPTIMIZE_BYTES = 64_000
INLINE_OVERLAY_MAX_DIMENSION = 1200
CRON_WARM_PATH = "/api/cron/warm-generation-v2"
LEGACY_CRON_WARM_PATH = "/api/cron/warm-generation"
CRON_WARM_PATHS = frozenset({CRON_WARM_PATH, LEGACY_CRON_WARM_PATH})
RUN_RESULT_CACHE_VERSION = "run-result-v7"
RUN_RESULT_CACHE_DIR = Path(os.environ["MAP_BOUNDARY_CACHE_DIR"]) / "run-results"
RUN_RESULT_MEMORY_CACHE_MAX = 64
RUN_RESULT_MEMORY_CACHE_MAX_BYTES = 512_000
RUN_RESULT_RUNTIME_ENV_DEFAULTS = {
    "MAP_BOUNDARY_BLOCK_NETWORK": "",
    "MAP_BOUNDARY_CATALOG_EXTRACT_MAX_DIMENSION": "240",
    "MAP_BOUNDARY_CATALOG_MISS_REFINE_MAX_DIMENSION": "",
    "MAP_BOUNDARY_CATALOG_RETRY_EXTRACT_MAX_DIMENSION": "400",
    "MAP_BOUNDARY_EARLY_OCR_STYLE_MAX_DIMENSION": "800",
    "MAP_BOUNDARY_ENABLE_ROAD_CONTEXT_FALLBACK": "",
    "MAP_BOUNDARY_EXTRACT_MAX_DIMENSION": "0",
    "MAP_BOUNDARY_EXTRACTION_DISK_CACHE": "",
    "MAP_BOUNDARY_EXTRACTION_UNTRIMMED_CACHE_MAX_PIXELS": "1000000",
    "MAP_BOUNDARY_GENERAL_EXTRACT_MAX_DIMENSION": "1600",
    "MAP_BOUNDARY_GEOCODE_BATCH_SIZE": "12",
    "MAP_BOUNDARY_GEOCODE_LABEL_LOOKAHEAD": "3",
    "MAP_BOUNDARY_GEOCODE_WORKERS": "6",
    "MAP_BOUNDARY_NOMINATIM_TIMEOUT_SECONDS": "4.0",
    "MAP_BOUNDARY_OCR_DISK_CACHE": "",
    "MAP_BOUNDARY_PLACE_BEFORE_LIVE_TIMEOUT_SECONDS": "1.0",
    "MAP_BOUNDARY_PLACE_FAST_PATH_TIMEOUT_SECONDS": "0.08",
    "MAP_BOUNDARY_PROVIDER_UI_CROP_OCR_MAX_DIMENSION": "750",
    "MAP_BOUNDARY_PROVIDER_UI_FOCUS_CROP": "1",
    "MAP_BOUNDARY_PROVIDER_UI_GRAY_FILL_CROP_OCR_MAX_DIMENSION": "450",
    "MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES": "1",
    "MAP_BOUNDARY_ROAD_MATCH_MAX_POINTS": "4000",
    "MAP_BOUNDARY_ROAD_REFINE_CACHE_MAX_PIXELS": "1000000",
    "MAP_BOUNDARY_ROAD_REFINE_COARSE_FEATURE_SCALE": "4",
    "MAP_BOUNDARY_ROAD_REFINE_FINE_FEATURE_SCALE": "2",
    "MAP_BOUNDARY_ROAD_REFINE_FULL_FALLBACK_MIN_SCORE": "0.60",
    "MAP_BOUNDARY_RUNNER_OCR_CACHE": "",
}
_RUN_RESULT_MEMORY_CACHE: OrderedDict[str, str] = OrderedDict()
_RUN_RESULT_MEMORY_CACHE_LOCK = threading.RLock()
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_NON_VISUAL_CHUNKS = {b"tEXt", b"zTXt", b"iTXt", b"tIME"}
JPEG_SIGNATURE = b"\xff\xd8"
JPEG_COMMENT_MARKER = 0xFE
JPEG_APP1_MARKER = 0xE1
JPEG_START_OF_SCAN_MARKER = 0xDA
JPEG_END_OF_IMAGE = b"\xff\xd9"
JPEG_EXIF_PREFIX = b"Exif\x00\x00"
JPEG_XMP_PREFIX = b"http://ns.adobe.com/xap/1.0/\x00"
WEBP_RIFF_SIGNATURE = b"RIFF"
WEBP_SIGNATURE = b"WEBP"
WEBP_NON_VISUAL_CHUNKS = {b"EXIF", b"XMP "}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".avif",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".svg",
    ".svgz",
}
FILENAME_HINT_CACHE_NOISE_TOKENS = {
    "app",
    "avif",
    "after",
    "boundary",
    "boundaries",
    "bmp",
    "bust",
    "cache",
    "capture",
    "copy",
    "coverage",
    "current",
    "final",
    "frame",
    "geojson",
    "gif",
    "hint",
    "image",
    "img",
    "jpeg",
    "jpg",
    "latency",
    "map",
    "maps",
    "neutral",
    "operating",
    "pipeline",
    "png",
    "prod",
    "production",
    "proof",
    "repeat",
    "roadskip",
    "run",
    "screenshot",
    "service",
    "small",
    "snap",
    "tif",
    "tiff",
    "ui",
    "upload",
    "uploaded",
    "variant",
    "version",
    "web",
    "webp",
}
FILENAME_HINT_CACHE_ALLOWED_PHRASES = {
    ("bay", "area"),
    ("las", "vegas"),
    ("los", "angeles"),
    ("san", "antonio"),
    ("san", "francisco"),
}


class RequestError(ValueError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class handler(BaseHTTPRequestHandler):
    server_version = "MapBoundaryVercel/0.1"

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(health_response_status(health_payload()))
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
            self.end_headers()
            return
        if parsed.path == "/" or parsed.path.startswith("/static/"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in CRON_WARM_PATHS:
                payload, status = cron_warm_generation_payload(
                    authorization_header=self.headers.get("Authorization")
                )
                self.send_json(payload, status=status)
                return
            if parsed.path == "/api/health":
                query = parse_qs(parsed.query)
                payload = health_payload(warm=first_query_value(query, "warm"))
                self.send_json(payload, status=health_response_status(payload))
                return
            if parsed.path == "/":
                self.send_asset("index.html")
                return
            if parsed.path.startswith("/static/"):
                self.send_asset(unquote(parsed.path.removeprefix("/static/")))
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
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_create_run(self) -> None:
        request_started = time.perf_counter()
        fields, files, upload_encoding = self.parse_upload_request()
        profile: dict[str, Any] = {
            "pipeline_version": get_pipeline_version(),
            "parse_upload_s": elapsed_seconds(request_started),
            "upload_encoding": upload_encoding,
        }
        city = fields.get("city", "").strip() or None
        upload = files.get("image")
        if upload is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Image upload is required.")
        original_filename, image_bytes = upload
        if not image_bytes:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Uploaded image is empty.")
        profile["upload_bytes"] = len(image_bytes)

        events: list[dict[str, Any]] = [
            {"stage": "queued", "message": "Run queued", "percent": 1, "status": "queued"}
        ]

        def progress(event: dict[str, Any]) -> None:
            events.append({"timestamp": time.time(), **event})

        normalized_cache_lookup = bool_field(fields, "normalized_cache_lookup", default=False)
        catalog_probe_only = bool_field(fields, "catalog_probe_only", default=False)
        include_overlay = include_overlay_for_request(fields, catalog_probe_only=catalog_probe_only)
        catalog_probe_missed = bool_field(fields, "catalog_probe_missed", default=False)
        allow_catalog = allow_catalog_for_request(fields)
        options = SimpleNamespace(
            simplify_px=float_field(fields, "simplify_px", DEFAULT_SIMPLIFY_PX, 0.0, 10.0),
            min_confidence=float_field(fields, "min_confidence", 0.55, 0.0, 1.0),
            min_control_points=int_field(fields, "min_control_points", 3, 0, 12),
            include_overlay=include_overlay,
            preview_max_dimension=INLINE_OVERLAY_MAX_DIMENSION if include_overlay else None,
            overlay_format="webp" if include_overlay else "png",
            write_mask_artifact=False,
            allow_catalog=allow_catalog,
            catalog_probe_only=catalog_probe_only,
            catalog_probe_missed=catalog_probe_missed,
            catalog_probe_miss_low_iou=bool_field(fields, "catalog_probe_miss_low_iou", default=False),
            filename_hint=original_filename,
        )
        run_id = f"{int(time.time())}-{os.urandom(4).hex()}"
        raw_cache_started = time.perf_counter()
        raw_cache_key = raw_run_result_cache_key(image_bytes, city, options)
        cached = read_run_result_cache(raw_cache_key)
        profile["raw_cache_lookup_s"] = elapsed_seconds(raw_cache_started)
        if cached is not None:
            profile["cache_hit"] = "raw"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = cached_run_payload(cached, run_id, original_filename, events, profile=profile)
            self.send_json(
                payload,
                status=cached_run_response_status(payload),
            )
            return

        png_visual_cache_started = time.perf_counter()
        png_visual_cache_key = png_visual_run_result_cache_key(image_bytes, city, options)
        if png_visual_cache_key is not None:
            cached = read_run_result_cache(png_visual_cache_key)
        profile["png_visual_cache_lookup_s"] = elapsed_seconds(png_visual_cache_started)
        if png_visual_cache_key is not None and cached is not None:
            raw_cache_write_started = time.perf_counter()
            write_run_result_cache(raw_cache_key, cached)
            profile["raw_cache_write_s"] = elapsed_seconds(raw_cache_write_started)
            profile["cache_hit"] = "png-visual"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = cached_run_payload(cached, run_id, original_filename, events, profile=profile)
            self.send_json(
                payload,
                status=cached_run_response_status(payload),
            )
            return

        jpeg_commentless_cache_started = time.perf_counter()
        jpeg_commentless_cache_key = jpeg_commentless_run_result_cache_key(image_bytes, city, options)
        if jpeg_commentless_cache_key is not None:
            cached = read_run_result_cache(jpeg_commentless_cache_key)
        profile["jpeg_commentless_cache_lookup_s"] = elapsed_seconds(jpeg_commentless_cache_started)
        if jpeg_commentless_cache_key is not None and cached is not None:
            raw_cache_write_started = time.perf_counter()
            write_run_result_cache(raw_cache_key, cached)
            profile["raw_cache_write_s"] = elapsed_seconds(raw_cache_write_started)
            profile["cache_hit"] = "jpeg-commentless"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = cached_run_payload(cached, run_id, original_filename, events, profile=profile)
            self.send_json(
                payload,
                status=cached_run_response_status(payload),
            )
            return

        jpeg_visual_cache_started = time.perf_counter()
        jpeg_visual_cache_key = jpeg_visual_run_result_cache_key(image_bytes, city, options)
        if jpeg_visual_cache_key is not None:
            cached = read_run_result_cache(jpeg_visual_cache_key)
        profile["jpeg_visual_cache_lookup_s"] = elapsed_seconds(jpeg_visual_cache_started)
        if jpeg_visual_cache_key is not None and cached is not None:
            raw_cache_write_started = time.perf_counter()
            write_run_result_cache(raw_cache_key, cached)
            profile["raw_cache_write_s"] = elapsed_seconds(raw_cache_write_started)
            profile["cache_hit"] = "jpeg-visual"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = cached_run_payload(cached, run_id, original_filename, events, profile=profile)
            self.send_json(
                payload,
                status=cached_run_response_status(payload),
            )
            return

        webp_visual_cache_started = time.perf_counter()
        webp_visual_cache_key = webp_visual_run_result_cache_key(image_bytes, city, options)
        if webp_visual_cache_key is not None:
            cached = read_run_result_cache(webp_visual_cache_key)
        profile["webp_visual_cache_lookup_s"] = elapsed_seconds(webp_visual_cache_started)
        if webp_visual_cache_key is not None and cached is not None:
            raw_cache_write_started = time.perf_counter()
            write_run_result_cache(raw_cache_key, cached)
            profile["raw_cache_write_s"] = elapsed_seconds(raw_cache_write_started)
            profile["cache_hit"] = "webp-visual"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = cached_run_payload(cached, run_id, original_filename, events, profile=profile)
            self.send_json(
                payload,
                status=cached_run_response_status(payload),
            )
            return

        cache_key: str | None = None
        profile["normalized_cache_lookup_enabled"] = normalized_cache_lookup
        if normalized_cache_lookup:
            normalized_cache_started = time.perf_counter()
            cache_key = run_result_cache_key(image_bytes, city, options)
            cached = read_run_result_cache(cache_key)
            profile["normalized_cache_lookup_s"] = elapsed_seconds(normalized_cache_started)
            if cached is not None:
                raw_cache_write_started = time.perf_counter()
                write_run_result_cache(raw_cache_key, cached)
                profile["raw_cache_write_s"] = elapsed_seconds(raw_cache_write_started)
                profile["cache_hit"] = "normalized"
                profile["total_before_send_s"] = elapsed_seconds(request_started)
                payload = cached_run_payload(cached, run_id, original_filename, events, profile=profile)
                self.send_json(
                    payload,
                    status=cached_run_response_status(payload),
                )
                return
        else:
            profile["normalized_cache_lookup_s"] = 0.0

        run_dir = Path(tempfile.gettempdir()) / "map-boundary-builder" / run_id
        debug_dir = run_dir / "debug" if options.include_overlay else None
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / f"input{safe_extension(original_filename)}"
        output_path = run_dir / "boundary.geojson"
        write_upload_started = time.perf_counter()
        image_path.write_bytes(image_bytes)
        profile["write_upload_s"] = elapsed_seconds(write_upload_started)

        build_started = time.perf_counter()
        try:
            from map_boundary_builder.runner import CatalogProbeMiss, build_boundary
        except Exception as exc:
            events = generation_failure_events(events, exc)
            profile["build_boundary_s"] = elapsed_seconds(build_started)
            profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(events)
            profile["cache_hit"] = "miss"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = generation_error_payload(exc, run_id, original_filename, events, profile)
            self.send_json(
                payload,
                status=generation_error_status(exc),
            )
            return

        try:
            result = build_boundary(
                image_path,
                city,
                output_path,
                debug_dir=debug_dir,
                options=options,
                progress=progress,
            )
            profile["build_boundary_s"] = elapsed_seconds(build_started)
            profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(events)
        except CatalogProbeMiss as exc:
            events = terminal_run_events(
                events,
                stage="catalog_miss",
                message="Catalog probe missed",
                status="catalog_miss",
                details=exc.details,
            )
            profile["build_boundary_s"] = elapsed_seconds(build_started)
            profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(events)
            profile["cache_hit"] = "miss"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = {
                "id": run_id,
                "filename": Path(original_filename).name or "uploaded-image",
                "status": "catalog_miss",
                "percent": 100,
                "error": str(exc),
                "catalog_probe_miss": exc.details,
                "events": events[-20:],
                "profile": profile,
            }
            if cache_key is not None:
                write_run_result_cache(cache_key, payload)
            if png_visual_cache_key is not None:
                write_run_result_cache(png_visual_cache_key, payload)
            if jpeg_commentless_cache_key is not None:
                write_run_result_cache(jpeg_commentless_cache_key, payload)
            if jpeg_visual_cache_key is not None:
                write_run_result_cache(jpeg_visual_cache_key, payload)
            if webp_visual_cache_key is not None:
                write_run_result_cache(webp_visual_cache_key, payload)
            write_run_result_cache(raw_cache_key, payload)
            self.send_json(payload, status=HTTPStatus.OK)
            return
        except Exception as exc:
            events = generation_failure_events(events, exc)
            profile["build_boundary_s"] = elapsed_seconds(build_started)
            profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(events)
            profile["cache_hit"] = "miss"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = generation_error_payload(exc, run_id, original_filename, events, profile)
            if generation_error_status(exc) == HTTPStatus.UNPROCESSABLE_ENTITY:
                if cache_key is not None:
                    write_run_result_cache(cache_key, payload)
                if png_visual_cache_key is not None:
                    write_run_result_cache(png_visual_cache_key, payload)
                if jpeg_commentless_cache_key is not None:
                    write_run_result_cache(jpeg_commentless_cache_key, payload)
                if jpeg_visual_cache_key is not None:
                    write_run_result_cache(jpeg_visual_cache_key, payload)
                if webp_visual_cache_key is not None:
                    write_run_result_cache(webp_visual_cache_key, payload)
                write_run_result_cache(raw_cache_key, payload)
            self.send_json(
                payload,
                status=generation_error_status(exc),
            )
            return
        artifacts_started = time.perf_counter()
        artifacts = {
            "geojson_inline": result.geojson,
        }
        if options.include_overlay:
            artifacts["overlay_data_url"] = inline_overlay(result.overlay_path)
        profile["build_artifacts_s"] = elapsed_seconds(artifacts_started)
        payload = {
            "id": run_id,
            "city": result.summary["city"],
            "filename": Path(original_filename).name or "uploaded-image",
            "status": "complete",
            "percent": 100,
            "summary": result.summary,
            "events": events[-20:],
            "artifacts": artifacts,
        }
        cache_write_started = time.perf_counter()
        if cache_key is not None:
            write_run_result_cache(cache_key, payload)
        if png_visual_cache_key is not None:
            write_run_result_cache(png_visual_cache_key, payload)
        if jpeg_commentless_cache_key is not None:
            write_run_result_cache(jpeg_commentless_cache_key, payload)
        if jpeg_visual_cache_key is not None:
            write_run_result_cache(jpeg_visual_cache_key, payload)
        if webp_visual_cache_key is not None:
            write_run_result_cache(webp_visual_cache_key, payload)
        write_run_result_cache(raw_cache_key, payload)
        profile["cache_write_s"] = elapsed_seconds(cache_write_started)
        profile["cache_hit"] = "miss"
        profile["total_before_send_s"] = elapsed_seconds(request_started)
        payload["profile"] = profile
        self.send_json(payload, status=HTTPStatus.CREATED)

    def handle_create_report(self) -> None:
        fields, files, _upload_encoding = self.parse_upload_request()
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
        from map_boundary_builder.github_reports import FailureReport, GithubReportError, create_failure_issue

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

    def parse_upload_request(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]], str]:
        content_type = self.headers.get("Content-Type", "").lower()
        if "multipart/form-data" in content_type:
            fields, files = self.parse_multipart()
            return fields, files, "multipart"
        if "application/json" in content_type:
            fields, files = self.parse_json_upload()
            return fields, files, "json-base64"
        raise RequestError(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data or application/json.")

    def parse_json_upload(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Request body is empty.")
        if length > json_upload_body_limit(MAX_UPLOAD_BYTES):
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Upload is larger than {limit_mb} MB.")
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
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Upload is larger than {limit_mb} MB.")

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

    def send_asset(self, name: str) -> None:
        try:
            data, mime = web_asset_response(name)
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data, extra_headers = json_response_body(
            payload,
            accept_encoding=self.headers.get("Accept-Encoding", ""),
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        for name, value in extra_headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def inline_overlay(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    data = path.read_bytes()
    mime = "image/webp" if path.suffix.lower() == ".webp" else "image/png"
    if mime == "image/png" and len(data) > INLINE_OVERLAY_OPTIMIZE_BYTES:
        optimized = optimized_overlay_bytes(path, original_size=len(data))
        if optimized is not None:
            mime, data = optimized
    if len(data) > MAX_INLINE_OVERLAY_BYTES:
        try:
            from PIL import Image

            jpeg_path = path.with_suffix(".jpg")
            with Image.open(path) as image:
                image.convert("RGB").save(jpeg_path, format="JPEG", quality=82, optimize=True)
            data = jpeg_path.read_bytes()
            mime = "image/jpeg"
        except Exception:
            return None
    if len(data) > MAX_INLINE_OVERLAY_BYTES:
        return None
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def optimized_overlay_bytes(path: Path, *, original_size: int) -> tuple[str, bytes] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            max_dimension = max(rgb.size)
            if max_dimension > INLINE_OVERLAY_MAX_DIMENSION:
                scale = max_dimension / INLINE_OVERLAY_MAX_DIMENSION
                size = (
                    max(1, int(round(rgb.width / scale))),
                    max(1, int(round(rgb.height / scale))),
                )
                rgb = rgb.resize(size, Image.Resampling.LANCZOS)
            webp = BytesIO()
            rgb.save(webp, format="WEBP", quality=90, method=4)
            webp_data = webp.getvalue()
            if webp_data and len(webp_data) < original_size:
                return "image/webp", webp_data
    except Exception:
        return None
    return None


def safe_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return ext if ext in SUPPORTED_IMAGE_EXTENSIONS else ".png"


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


def include_overlay_for_request(fields: dict[str, str], *, catalog_probe_only: bool) -> bool:
    return bool_field(fields, "include_overlay", default=not catalog_probe_only)


def allow_catalog_for_request(fields: dict[str, str]) -> bool:
    if bool_field(fields, "no_catalog", default=False):
        return False
    return bool_field(fields, "allow_catalog", default=True)


def json_response_body(payload: dict[str, Any], *, accept_encoding: str = "") -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(data) < 1024 or "gzip" not in accept_encoding.lower():
        return data, {}
    return gzip.compress(data, compresslevel=3), {
        "Content-Encoding": "gzip",
        "Vary": "Accept-Encoding",
    }


def first_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if not values:
        return None
    return values[0]


def health_payload(*, warm: str | None = None) -> dict[str, Any]:
    runtime_dependencies = dict(pipeline_version_dependency_versions())
    tmp_writable = os.access(tempfile.gettempdir(), os.W_OK)
    payload: dict[str, Any] = {
        "ok": runtime_health_ok(runtime_dependencies, tmp_writable=tmp_writable),
        "runtime": "vercel-python",
        "tesseract": shutil.which("tesseract"),
        "tmp_writable": tmp_writable,
        "pipeline_version": get_pipeline_version(),
        "runtime_dependencies": runtime_dependencies,
        "ocr": ocr_runtime_config(),
        "generation_env": generation_runtime_env_config(),
    }
    if should_prewarm_generation_runtime(warm):
        warm_payload = prewarm_generation_runtime()
        payload["warm"] = warm_payload
        if not warm_generation_ok(warm_payload):
            payload["ok"] = False
    return payload


def cron_warm_generation_payload(*, authorization_header: str | None) -> tuple[dict[str, Any], HTTPStatus]:
    if not authorized_cron_request(authorization_header):
        return {"ok": False, "error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED
    warm_payload = prewarm_generation_runtime()
    ok = warm_generation_ok(warm_payload)
    return {
        "ok": ok,
        "pipeline_version": get_pipeline_version(),
        "warm": warm_payload,
    }, HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE


def runtime_health_ok(runtime_dependencies: dict[str, str], *, tmp_writable: bool) -> bool:
    if not tmp_writable:
        return False
    for dependency in ("numpy", "onnxruntime", "pillow", "rapidocr-onnxruntime", "shapely", "cv2"):
        if runtime_dependencies.get(dependency) in {None, "", "missing", "unknown"}:
            return False
    return True


def warm_generation_ok(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "ok"


def health_response_status(payload: dict[str, Any]) -> HTTPStatus:
    return HTTPStatus.OK if payload.get("ok") is True else HTTPStatus.SERVICE_UNAVAILABLE


def authorized_cron_request(authorization_header: str | None) -> bool:
    secret = os.environ.get("CRON_SECRET", "")
    if not secret:
        return False
    return hmac.compare_digest(authorization_header or "", f"Bearer {secret}")


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


def generation_error_status(exc: Exception) -> HTTPStatus:
    if isinstance(exc, ValueError):
        return HTTPStatus.UNPROCESSABLE_ENTITY
    return HTTPStatus.INTERNAL_SERVER_ERROR


def generation_failure_events(events: list[dict[str, Any]], exc: Exception) -> list[dict[str, Any]]:
    return terminal_run_events(
        events,
        stage="failed",
        message="Generation failed",
        status="failed",
        details={"error": str(exc)},
    )


def terminal_run_events(
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


def generation_error_payload(
    exc: Exception,
    run_id: str,
    original_filename: str,
    events: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    events = generation_failure_events(events, exc)
    return {
        "id": run_id,
        "filename": Path(original_filename).name or "uploaded-image",
        "status": "failed",
        "percent": 100,
        "error": str(exc),
        "events": events[-20:],
        "profile": profile,
    }


def run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str:
    return run_result_cache_key_for_hash("image_pixel_sha256", normalized_image_sha256(image_bytes), city, options)


def raw_run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str:
    return run_result_cache_key_for_hash("image_raw_sha256", hashlib.sha256(image_bytes).hexdigest(), city, options)


def png_visual_run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str | None:
    visual_hash = png_visual_sha256(image_bytes)
    if visual_hash is None:
        return None
    return run_result_cache_key_for_hash("png_visual_sha256", visual_hash, city, options)


def jpeg_commentless_run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str | None:
    visual_hash = jpeg_commentless_sha256(image_bytes)
    if visual_hash is None:
        return None
    return run_result_cache_key_for_hash("jpeg_commentless_sha256", visual_hash, city, options)


def jpeg_visual_run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str | None:
    visual_hash = jpeg_visual_sha256(image_bytes)
    if visual_hash is None:
        return None
    return run_result_cache_key_for_hash("jpeg_visual_sha256", visual_hash, city, options)


def webp_visual_run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str | None:
    visual_hash = webp_visual_sha256(image_bytes)
    if visual_hash is None:
        return None
    return run_result_cache_key_for_hash("webp_visual_sha256", visual_hash, city, options)


def run_result_cache_key_for_hash(
    image_hash_name: str,
    image_hash: str,
    city: str | None,
    options: Any,
) -> str:
    parts = {
        "version": RUN_RESULT_CACHE_VERSION,
        "pipeline_version": get_pipeline_version(),
        "runtime_config": run_result_runtime_config(),
        image_hash_name: image_hash,
        "city": city or "",
        "simplify_px": round(float(options.simplify_px), 4),
        "min_confidence": round(float(options.min_confidence), 4),
        "min_control_points": int(options.min_control_points),
        "include_overlay": bool(getattr(options, "include_overlay", True)),
        "preview_max_dimension": getattr(options, "preview_max_dimension", None) or "",
        "overlay_format": getattr(options, "overlay_format", "png"),
        "write_mask_artifact": bool(getattr(options, "write_mask_artifact", True)),
        "allow_catalog": bool(getattr(options, "allow_catalog", True)),
        "catalog_probe_only": bool(getattr(options, "catalog_probe_only", False)),
        "catalog_probe_missed": bool(getattr(options, "catalog_probe_missed", False)),
        "catalog_probe_miss_low_iou": bool(getattr(options, "catalog_probe_miss_low_iou", False)),
        "filename_hint": filename_hint_cache_value(getattr(options, "filename_hint", None)),
    }
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_result_runtime_config() -> dict[str, Any]:
    return {
        "ocr": ocr_runtime_config(),
        "generation_env": generation_runtime_env_config(),
    }


def generation_runtime_env_config() -> dict[str, str]:
    return {
        name: os.environ.get(name, default)
        for name, default in sorted(RUN_RESULT_RUNTIME_ENV_DEFAULTS.items())
    }


def filename_hint_cache_value(filename_hint: object) -> str:
    if not filename_hint:
        return ""
    filename = Path(str(filename_hint)).name
    path = Path(filename)
    extension = path.suffix.lower().lstrip(".")
    raw_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", path.stem.lower())
        if len(token) >= 2 and not any(char.isdigit() for char in token)
    ]
    protected_indexes = filename_hint_cache_phrase_indexes(raw_tokens)
    tokens: list[str] = []
    seen: set[str] = set()
    for index, token in enumerate(raw_tokens):
        if index not in protected_indexes and token in FILENAME_HINT_CACHE_NOISE_TOKENS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    token_part = " ".join(tokens)
    if extension:
        return f"{extension}:{token_part}"
    return token_part


def filename_hint_cache_phrase_indexes(tokens: list[str]) -> set[int]:
    protected: set[int] = set()
    for phrase in FILENAME_HINT_CACHE_ALLOWED_PHRASES:
        size = len(phrase)
        for index in range(0, max(0, len(tokens) - size + 1)):
            if tuple(tokens[index : index + size]) == phrase:
                protected.update(range(index, index + size))
    return protected


def normalized_image_sha256(image_bytes: bytes) -> str:
    try:
        from PIL import Image, ImageOps

        with Image.open(BytesIO(image_bytes)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGBA")
            digest = hashlib.sha256()
            digest.update(str(normalized.size).encode("ascii"))
            digest.update(normalized.mode.encode("ascii"))
            digest.update(normalized.tobytes())
            return digest.hexdigest()
    except Exception:
        return hashlib.sha256(image_bytes).hexdigest()


def png_visual_sha256(image_bytes: bytes) -> str | None:
    if not image_bytes.startswith(PNG_SIGNATURE):
        return None
    digest = hashlib.sha256()
    digest.update(b"png-visual-v1")
    digest.update(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    seen_iend = False
    while offset + 12 <= len(image_bytes):
        chunk_length = int.from_bytes(image_bytes[offset : offset + 4], "big")
        chunk_type = image_bytes[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + chunk_length
        crc_end = data_end + 4
        if crc_end > len(image_bytes):
            return None
        if chunk_type not in PNG_NON_VISUAL_CHUNKS:
            digest.update(chunk_type)
            digest.update(chunk_length.to_bytes(4, "big"))
            digest.update(image_bytes[data_start:data_end])
        offset = crc_end
        if chunk_type == b"IEND":
            seen_iend = True
            break
    if not seen_iend:
        return None
    return digest.hexdigest()


def jpeg_commentless_sha256(image_bytes: bytes) -> str | None:
    if not image_bytes.startswith(JPEG_SIGNATURE):
        return None
    digest = hashlib.sha256()
    digest.update(b"jpeg-commentless-v1")
    digest.update(JPEG_SIGNATURE)
    offset = len(JPEG_SIGNATURE)
    while offset < len(image_bytes):
        if image_bytes[offset] != 0xFF:
            return None
        while offset < len(image_bytes) and image_bytes[offset] == 0xFF:
            offset += 1
        if offset >= len(image_bytes):
            return None
        marker = image_bytes[offset]
        offset += 1
        if marker == 0x00:
            return None
        marker_bytes = bytes((0xFF, marker))
        if jpeg_marker_has_no_payload(marker):
            digest.update(marker_bytes)
            if marker_bytes == JPEG_END_OF_IMAGE:
                return digest.hexdigest()
            continue
        if offset + 2 > len(image_bytes):
            return None
        segment_length = int.from_bytes(image_bytes[offset : offset + 2], "big")
        if segment_length < 2:
            return None
        segment_end = offset + segment_length
        if segment_end > len(image_bytes):
            return None
        if marker != JPEG_COMMENT_MARKER:
            digest.update(marker_bytes)
            digest.update(image_bytes[offset : offset + 2])
            digest.update(image_bytes[offset + 2 : segment_end])
        offset = segment_end
        if marker == JPEG_START_OF_SCAN_MARKER:
            scan_bytes = image_bytes[offset:]
            if JPEG_END_OF_IMAGE not in scan_bytes:
                return None
            digest.update(scan_bytes)
            return digest.hexdigest()
    return None


def jpeg_marker_has_no_payload(marker: int) -> bool:
    return marker == 0x01 or marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7


def jpeg_visual_sha256(image_bytes: bytes) -> str | None:
    if not image_bytes.startswith(JPEG_SIGNATURE):
        return None
    digest = hashlib.sha256()
    digest.update(b"jpeg-visual-v1")
    digest.update(JPEG_SIGNATURE)
    offset = len(JPEG_SIGNATURE)
    while offset < len(image_bytes):
        if image_bytes[offset] != 0xFF:
            return None
        while offset < len(image_bytes) and image_bytes[offset] == 0xFF:
            offset += 1
        if offset >= len(image_bytes):
            return None
        marker = image_bytes[offset]
        offset += 1
        if marker == 0x00:
            return None
        marker_bytes = bytes((0xFF, marker))
        if jpeg_marker_has_no_payload(marker):
            digest.update(marker_bytes)
            if marker_bytes == JPEG_END_OF_IMAGE:
                return digest.hexdigest()
            continue
        if offset + 2 > len(image_bytes):
            return None
        segment_length = int.from_bytes(image_bytes[offset : offset + 2], "big")
        if segment_length < 2:
            return None
        segment_end = offset + segment_length
        if segment_end > len(image_bytes):
            return None
        payload = image_bytes[offset + 2 : segment_end]
        if not jpeg_segment_is_non_visual(marker, payload):
            digest.update(marker_bytes)
            digest.update(image_bytes[offset : offset + 2])
            digest.update(payload)
        offset = segment_end
        if marker == JPEG_START_OF_SCAN_MARKER:
            scan_bytes = image_bytes[offset:]
            if JPEG_END_OF_IMAGE not in scan_bytes:
                return None
            digest.update(scan_bytes)
            return digest.hexdigest()
    return None


def jpeg_segment_is_non_visual(marker: int, payload: bytes) -> bool:
    if marker == JPEG_COMMENT_MARKER:
        return True
    if marker != JPEG_APP1_MARKER:
        return False
    return payload.startswith(JPEG_EXIF_PREFIX) or payload.startswith(JPEG_XMP_PREFIX)


def webp_visual_sha256(image_bytes: bytes) -> str | None:
    if (
        len(image_bytes) < 12
        or image_bytes[:4] != WEBP_RIFF_SIGNATURE
        or image_bytes[8:12] != WEBP_SIGNATURE
    ):
        return None
    riff_size = int.from_bytes(image_bytes[4:8], "little")
    if riff_size + 8 > len(image_bytes):
        return None
    digest = hashlib.sha256()
    digest.update(b"webp-visual-v1")
    digest.update(WEBP_SIGNATURE)
    offset = 12
    riff_end = 8 + riff_size
    while offset + 8 <= riff_end:
        chunk_type = image_bytes[offset : offset + 4]
        chunk_length = int.from_bytes(image_bytes[offset + 4 : offset + 8], "little")
        data_start = offset + 8
        data_end = data_start + chunk_length
        padded_end = data_end + (chunk_length % 2)
        if padded_end > len(image_bytes) or padded_end > riff_end:
            return None
        if chunk_type not in WEBP_NON_VISUAL_CHUNKS:
            digest.update(chunk_type)
            digest.update(chunk_length.to_bytes(4, "little"))
            digest.update(image_bytes[data_start:data_end])
        offset = padded_end
    if offset != riff_end:
        return None
    return digest.hexdigest()


def read_run_result_cache(cache_key: str) -> dict[str, Any] | None:
    with _RUN_RESULT_MEMORY_CACHE_LOCK:
        cached_json = _RUN_RESULT_MEMORY_CACHE.get(cache_key)
        if cached_json is not None:
            _RUN_RESULT_MEMORY_CACHE.move_to_end(cache_key)
    if cached_json is not None:
        try:
            return json.loads(cached_json)
        except Exception:
            with _RUN_RESULT_MEMORY_CACHE_LOCK:
                _RUN_RESULT_MEMORY_CACHE.pop(cache_key, None)
            return None
    cache_path = RUN_RESULT_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        encoded = cache_path.read_text()
        payload = json.loads(encoded)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    remember_run_result_cache(cache_key, payload, encoded=encoded)
    return payload


def write_run_result_cache(cache_key: str, payload: dict[str, Any]) -> None:
    if payload.get("status") == "failed":
        cached = {
            "status": "failed",
            "error": payload.get("error"),
        }
    elif payload.get("status") == "catalog_miss":
        cached = {
            "status": "catalog_miss",
            "error": payload.get("error"),
        }
        if isinstance(payload.get("catalog_probe_miss"), dict):
            cached["catalog_probe_miss"] = payload.get("catalog_probe_miss")
    else:
        cached = {
            "city": payload.get("city"),
            "summary": payload.get("summary"),
            "artifacts": payload.get("artifacts"),
        }
    encoded = remember_run_result_cache(cache_key, cached)
    cache_path = RUN_RESULT_CACHE_DIR / f"{cache_key}.json"
    tmp_path = run_result_cache_tmp_path(cache_path)
    try:
        RUN_RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(encoded)
        tmp_path.replace(cache_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        return


def remember_run_result_cache(cache_key: str, payload: dict[str, Any], *, encoded: str | None = None) -> str:
    if encoded is None:
        encoded = json.dumps(payload, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > RUN_RESULT_MEMORY_CACHE_MAX_BYTES:
        with _RUN_RESULT_MEMORY_CACHE_LOCK:
            _RUN_RESULT_MEMORY_CACHE.pop(cache_key, None)
        return encoded
    with _RUN_RESULT_MEMORY_CACHE_LOCK:
        _RUN_RESULT_MEMORY_CACHE[cache_key] = encoded
        _RUN_RESULT_MEMORY_CACHE.move_to_end(cache_key)
        while len(_RUN_RESULT_MEMORY_CACHE) > RUN_RESULT_MEMORY_CACHE_MAX:
            _RUN_RESULT_MEMORY_CACHE.popitem(last=False)
    return encoded


def run_result_cache_tmp_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def cached_run_payload(
    cached: dict[str, Any],
    run_id: str,
    original_filename: str,
    events: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(cached))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    raw_status = payload.get("status")
    status = raw_status if raw_status in {"catalog_miss", "failed"} else "complete"
    event_details = cached_run_event_details(status, payload=payload, summary=summary)
    event_message = {
        "catalog_miss": "Catalog miss ready from cache",
        "failed": "Generation failure ready from cache",
    }.get(status, "Boundary export ready from cache")
    payload.update(
        {
            "id": run_id,
            "city": payload.get("city") or summary.get("city"),
            "filename": Path(original_filename).name or "uploaded-image",
            "status": status,
            "percent": 100,
            "cached": True,
            "events": [
                *events,
                {
                    "timestamp": time.time(),
                    "stage": status,
                    "message": event_message,
                    "percent": 100,
                    "status": status,
                    "details": event_details,
                },
            ],
        }
    )
    if profile is not None:
        payload["profile"] = profile
    return payload


def cached_run_event_details(
    status: str,
    *,
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    if status == "catalog_miss" and isinstance(payload.get("catalog_probe_miss"), dict):
        return payload["catalog_probe_miss"]
    if status == "failed":
        error = payload.get("error")
        if isinstance(error, str) and error:
            return {"error": error}
    return summary


def cached_run_response_status(payload: dict[str, Any]) -> HTTPStatus:
    if payload.get("status") == "failed":
        return HTTPStatus.UNPROCESSABLE_ENTITY
    if payload.get("status") == "catalog_miss":
        return HTTPStatus.OK
    return HTTPStatus.CREATED
