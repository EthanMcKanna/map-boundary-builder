"""Real-screenshot benchmark for the unified pipeline.

Runs ``run_pipeline`` over the real-screenshot manifest and scores each case
against its durable expectations: final status, inferred city, output
bounding box, control-point support, and combined confidence. Legacy
expectation fields tied to the old runner internals (OCR engine call
budgets, source strings, catalog behavior) are ignored.

Usage:
    map-boundary-real-benchmark \
        --manifest benchmarks/real-screenshot-stress.json \
        --image-root benchmarks/real-screenshots \
        --out out/real-benchmark.json \
        --baseline benchmarks/baselines/real-hard-gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .pipeline import PipelineOptions, PipelineResult, run_pipeline

# Old-runner statuses map onto the pipeline's coarser set. "missing" is kept
# for images that no longer exist on disk.
DURABLE_STATUSES = {"complete", "needs_city", "failed", "missing"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the unified pipeline against real screenshots.")
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/real-screenshot-stress.json"))
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("benchmarks/real-screenshots"),
        help="Directory searched for manifest images when their recorded absolute paths are missing.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output report path.")
    parser.add_argument("--baseline", type=Path, default=None, help="Prior stress/report JSON to diff statuses against.")
    parser.add_argument("--only", action="append", default=None, help="Run only this slug. May be repeated.")
    parser.add_argument("--model-path", type=Path, default=None, help="Override the packaged segmentation model.")
    parser.add_argument("--use-expected-city", action="store_true", help="Pass each case's expected city as the city argument (measures the assisted path).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.only:
        wanted = set(args.only)
        cases = [case for case in cases if case["slug"] in wanted]
    baseline_statuses = load_baseline_statuses(args.baseline) if args.baseline else {}

    options = PipelineOptions(model_path=args.model_path)
    rows: list[dict[str, Any]] = []
    result_cache: dict[tuple[str, str | None], PipelineResult] = {}
    for case in cases:
        rows.append(
            score_case(
                case,
                image_root=args.image_root,
                options=options,
                use_expected_city=args.use_expected_city,
                result_cache=result_cache,
            )
        )

    report = build_report(rows, baseline_statuses)
    rendered = json.dumps(report["summary"], indent=2, sort_keys=True)
    print(rendered)
    for row in rows:
        marker = "ok" if row["passed"] else "!!"
        print(f"{marker} {row['slug']}: {row['status']} city={row.get('city')} {' '.join(row['failures'])}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["summary"]["failed_cases"] == 0 else 1


def resolve_image_path(recorded: str, image_root: Path) -> Path | None:
    path = Path(recorded)
    if path.is_file():
        return path
    fallback = image_root / path.name
    if fallback.is_file():
        return fallback
    return None


def score_case(
    case: dict[str, Any],
    *,
    image_root: Path,
    options: PipelineOptions,
    use_expected_city: bool,
    result_cache: dict[tuple[str, str | None], PipelineResult],
) -> dict[str, Any]:
    expect = case.get("expect", {})
    slug = case["slug"]
    image_path = resolve_image_path(case["image"], image_root)
    if image_path is None:
        return {
            "slug": slug,
            "status": "missing",
            "passed": expect.get("status") == "missing",
            "failures": [] if expect.get("status") == "missing" else ["image file missing"],
        }

    city = expect.get("city_equals") if use_expected_city else None
    cache_key = (str(image_path), city)
    started = time.monotonic()
    if cache_key in result_cache:
        result = result_cache[cache_key]
        elapsed = 0.0
    else:
        result = run_pipeline(image_path, city=city, options=options)
        elapsed = time.monotonic() - started
        result_cache[cache_key] = result

    failures: list[str] = []
    expected_status = normalize_expected_status(expect.get("status"))
    if expected_status and result.status != expected_status:
        failures.append(f"status {result.status} != expected {expected_status}")

    row: dict[str, Any] = {
        "slug": slug,
        "status": result.status,
        "reason": result.reason,
        "elapsed_s": round(elapsed, 3),
        "engine": (result.summary.get("extraction") or {}).get("engine"),
        "ocr_label_count": result.summary.get("ocr_label_count"),
    }

    if result.status == "complete":
        georef = result.summary.get("georeference", {})
        row["city"] = georef.get("city")
        row["control_points"] = georef.get("control_points")
        row["combined_confidence"] = result.summary.get("combined_confidence")

        expected_city = expect.get("city_equals")
        if expected_city and not city_matches(georef.get("city"), expected_city):
            failures.append(f"city {georef.get('city')!r} != expected {expected_city!r}")
        min_controls = expect.get("min_control_points")
        if min_controls is not None and (row["control_points"] or 0) < min_controls:
            failures.append(f"control points {row['control_points']} < {min_controls}")
        min_confidence = expect.get("min_combined_confidence")
        if min_confidence is not None and (row["combined_confidence"] or 0.0) < min_confidence:
            failures.append(f"combined confidence {row['combined_confidence']} < {min_confidence}")
        bbox_error = bbox_max_corner_error_m(result.geojson, expect.get("bbox_approx"))
        if bbox_error is not None:
            row["bbox_max_corner_error_m"] = round(bbox_error, 1)
            max_error = expect.get("max_bbox_error_m")
            if max_error is not None and bbox_error > max_error:
                failures.append(f"bbox max corner error {bbox_error:.0f}m above {max_error}m")

    row["failures"] = failures
    row["passed"] = not failures
    return row


def normalize_expected_status(expected: str | None) -> str | None:
    if expected is None:
        return None
    if expected in DURABLE_STATUSES:
        return expected
    # Every old failure flavor collapses to "failed".
    return "failed"


def city_matches(actual: str | None, expected: str) -> bool:
    if not actual:
        return False
    return expected.strip().lower() in actual.strip().lower()


def geojson_bbox(geojson: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not geojson:
        return None
    lons: list[float] = []
    lats: list[float] = []

    def walk(coords: Any) -> None:
        if isinstance(coords, (list, tuple)):
            if len(coords) == 2 and all(isinstance(v, (int, float)) for v in coords):
                lons.append(float(coords[0]))
                lats.append(float(coords[1]))
            else:
                for item in coords:
                    walk(item)

    for feature in geojson.get("features", []):
        walk(feature.get("geometry", {}).get("coordinates", []))
    if not lons:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def bbox_max_corner_error_m(geojson: dict[str, Any] | None, expected_bbox: list[float] | None) -> float | None:
    if expected_bbox is None:
        return None
    actual = geojson_bbox(geojson)
    if actual is None:
        return None
    ax_min, ay_min, ax_max, ay_max = actual
    ex_min, ey_min, ex_max, ey_max = expected_bbox
    mid_lat = (ey_min + ey_max) / 2.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))
    meters_per_deg_lat = 110_540.0
    corners = [
        (ax_min - ex_min, ay_min - ey_min),
        (ax_min - ex_min, ay_max - ey_max),
        (ax_max - ex_max, ay_min - ey_min),
        (ax_max - ex_max, ay_max - ey_max),
    ]
    return max(
        math.hypot(d_lon * meters_per_deg_lon, d_lat * meters_per_deg_lat)
        for d_lon, d_lat in corners
    )


def load_baseline_statuses(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or data.get("cases") or []
    statuses: dict[str, str] = {}
    for row in rows:
        slug = row.get("slug")
        status = row.get("status")
        if slug and status:
            statuses[slug] = status
    return statuses


def build_report(rows: list[dict[str, Any]], baseline_statuses: dict[str, str]) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    for row in rows:
        baseline = baseline_statuses.get(row["slug"])
        if baseline is not None:
            row["baseline_status"] = baseline
            row["transition"] = f"{baseline}->{row['status']}"
            transitions[row["transition"]] += 1
    summary = {
        "total_cases": len(rows),
        "passed_cases": sum(1 for row in rows if row["passed"]),
        "failed_cases": sum(1 for row in rows if not row["passed"]),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "status_transitions": dict(transitions),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    raise SystemExit(main())
