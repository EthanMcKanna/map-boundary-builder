from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .extract import DEFAULT_SIMPLIFY_PX
from .runner import BoundaryBuildOptions, build_boundary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map-boundary-builder",
        description="Extract a scaled GeoJSON service-area polygon from a map screenshot.",
    )
    parser.add_argument("--image", help="Input service-map screenshot.")
    parser.add_argument("--city", help="Optional city override. Omit to infer from map labels.")
    parser.add_argument("--output", "-o", help="Output GeoJSON path.")
    parser.add_argument("--debug-dir", help="Optional directory for mask and overlay PNGs.")
    parser.add_argument("--simplify-px", type=float, default=DEFAULT_SIMPLIFY_PX, help="Pixel simplification tolerance.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Fail below this combined confidence.")
    parser.add_argument("--min-control-points", type=int, default=3, help="Minimum OCR/geocoder control points for georeferencing.")
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        help="Bypass bundled service-area catalog matching and force OCR/georeference inference.",
    )
    parser.add_argument(
        "--catalog-probe-missed",
        action="store_true",
        help="Skip the low-resolution catalog probe after a prior probe miss and run the full handoff path.",
    )
    parser.add_argument(
        "--catalog-probe-miss-low-iou",
        action="store_true",
        help="Treat the prior catalog probe miss as far from active catalog shapes and overlap OCR with extraction.",
    )
    parser.add_argument("--print-summary", action="store_true", help="Print a compact JSON summary.")
    parser.add_argument(
        "--profile-events",
        action="store_true",
        help="Include progress events and per-stage elapsed seconds in the printed summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for required_arg in ("image", "output"):
        if getattr(args, required_arg) is None:
            parser.error(f"--{required_arg.replace('_', '-')} is required")

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Input image does not exist: {image_path}")

    events: list[dict[str, Any]] = []
    started = time.perf_counter()

    def progress(event: dict[str, Any]) -> None:
        events.append({"elapsed_s": round(time.perf_counter() - started, 6), **event})

    try:
        result = build_boundary(
            image_path,
            args.city,
            args.output,
            debug_dir=args.debug_dir,
            options=BoundaryBuildOptions(
                simplify_px=args.simplify_px,
                min_confidence=args.min_confidence,
                min_control_points=args.min_control_points,
                allow_catalog=not args.no_catalog,
                catalog_probe_missed=args.catalog_probe_missed,
                catalog_probe_miss_low_iou=args.catalog_probe_miss_low_iou,
                filename_hint=image_path.name,
            ),
            progress=progress if args.profile_events else None,
        )

        if args.print_summary:
            summary = dict(result.summary)
            if args.profile_events:
                summary["event_profile"] = {
                    "total_elapsed_s": round(time.perf_counter() - started, 6),
                    "stage_elapsed_s": stage_elapsed_seconds(events),
                    "events": events,
                }
            print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        if args.print_summary:
            summary: dict[str, Any] = {
                "status": "failed",
                "error": str(exc),
            }
            if args.profile_events:
                summary["event_profile"] = {
                    "total_elapsed_s": round(time.perf_counter() - started, 6),
                    "stage_elapsed_s": stage_elapsed_seconds(events),
                    "events": events,
                }
            print(json.dumps(summary, indent=2))
        print(f"map-boundary-builder: error: {exc}", file=sys.stderr)
        return 1


def stage_elapsed_seconds(events: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for current, following in zip(events, events[1:]):
        stage = current.get("stage")
        elapsed = current.get("elapsed_s")
        next_elapsed = following.get("elapsed_s")
        if not isinstance(stage, str) or not isinstance(elapsed, (int, float)):
            continue
        if not isinstance(next_elapsed, (int, float)):
            continue
        totals[stage] = totals.get(stage, 0.0) + max(0.0, float(next_elapsed) - float(elapsed))
    return {stage: round(total, 6) for stage, total in totals.items()}


if __name__ == "__main__":
    raise SystemExit(main())
