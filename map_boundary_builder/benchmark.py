from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform

from .extract import extract_service_area
from .georef_transform import lonlat_to_mercator
from .network_policy import NETWORK_BLOCK_ENV

DEFAULT_POLYGON_DIR = Path("/Users/ethanmckanna/GitHub/av-coverage-checker/data/service-areas/polygons")
DEFAULT_IMAGE_DIR = Path("/Users/ethanmckanna/Downloads/service area images")
DEFAULT_OUT_DIR = Path("out/service-area-benchmark")
DEFAULT_FIXTURE_CONFIG = Path("benchmarks/service-area-fixtures.json")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
AREA_ALIASES = {
    "bay area": "bay-area",
    "san francisco": "bay-area",
    "sf": "bay-area",
    "las vegas": "las-vegas",
    "los angeles": "los-angeles",
    "san antonio": "san-antonio",
}

@dataclass(frozen=True)
class BenchmarkFixture:
    slug: str
    provider: str
    area: str
    image_path: Path
    reference_path: Path
    status: str = "active"
    note: str | None = None


@dataclass(frozen=True)
class BenchmarkScore:
    slug: str
    image: str
    mode: str
    passed: bool
    iou: float | None
    area_ratio: float | None
    centroid_distance_m: float | None
    vertices: int | None
    style: str | None
    duration_s: float | None = None
    georeference_source: str | None = None
    combined_confidence: float | None = None
    catalog_slug: str | None = None
    stage_elapsed_s: dict[str, float] | None = None
    error: str | None = None
    status: str = "active"
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "image": self.image,
            "mode": self.mode,
            "passed": self.passed,
            "iou": round(self.iou, 6) if self.iou is not None else None,
            "area_ratio": round(self.area_ratio, 6) if self.area_ratio is not None else None,
            "centroid_distance_m": round(self.centroid_distance_m, 1) if self.centroid_distance_m is not None else None,
            "vertices": self.vertices,
            "style": self.style,
            "duration_s": round(self.duration_s, 3) if self.duration_s is not None else None,
            "georeference_source": self.georeference_source,
            "combined_confidence": round(self.combined_confidence, 6) if self.combined_confidence is not None else None,
            "catalog_slug": self.catalog_slug,
            "stage_elapsed_s": self.stage_elapsed_s,
            "error": self.error,
            "status": self.status,
            "note": self.note,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map-boundary-benchmark",
        description="Benchmark service-area screenshot extraction against reference service-area polygons.",
    )
    parser.add_argument("--polygon-dir", type=Path, default=DEFAULT_POLYGON_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--fixture-config",
        type=Path,
        default=DEFAULT_FIXTURE_CONFIG,
        help="Optional JSON config for fixture status overrides and notes.",
    )
    parser.add_argument(
        "--mode",
        choices=("extraction", "full"),
        default="extraction",
        help="extraction scores the detected pixel shape after reference-bounds fitting; full scores the exported GeoJSON.",
    )
    parser.add_argument("--min-iou", type=float, default=0.78)
    parser.add_argument("--mean-iou", type=float, default=0.90)
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Per-image timeout for --mode full.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only fixtures whose slug or image name contains this value. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--city-overrides",
        action="store_true",
        help="For --mode full, pass the reference area name as --city. The default tests image-only inference.",
    )
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        help="For --mode full, bypass catalog matching so OCR/georeference inference remains benchmarked.",
    )
    parser.add_argument(
        "--execution",
        choices=("subprocess", "in-process"),
        default="subprocess",
        help=(
            "For --mode full, subprocess preserves the historical cold-ish CLI gate; "
            "in-process measures warm production-instance generation without interpreter startup."
        ),
    )
    parser.add_argument(
        "--no-debug-artifacts",
        action="store_true",
        help="For --mode full, skip mask/overlay debug artifacts to mirror the production web API path.",
    )
    parser.add_argument(
        "--smoke-skipped",
        action="store_true",
        help=(
            "For --mode full, run non-active fixtures without scoring their stale references. "
            "Useful for reference_mismatch service-area drift checks."
        ),
    )
    parser.add_argument(
        "--require-smoked-catalog-miss",
        action="store_true",
        help=(
            "With --smoke-skipped, fail smoke-checked fixtures that return a catalog_slug. "
            "Use with targeted --only filters when those drifted screenshots must stay on OCR/georeference."
        ),
    )
    parser.add_argument(
        "--block-network",
        action="store_true",
        help="Block live geocoder/Overpass fallbacks during full benchmark generation.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Optional prior benchmark report; fail if active fixture IoU regresses against it.",
    )
    parser.add_argument(
        "--max-iou-drop",
        type=float,
        default=0.0,
        help="Maximum allowed per-fixture IoU drop when --baseline-report is provided.",
    )
    parser.add_argument(
        "--max-mean-iou-drop",
        type=float,
        default=0.0,
        help="Maximum allowed average IoU drop when --baseline-report is provided.",
    )
    parser.add_argument(
        "--max-duration-increase-ratio",
        type=float,
        default=None,
        help="Optional maximum per-fixture duration increase ratio against --baseline-report.",
    )
    parser.add_argument(
        "--max-duration-increase-s",
        type=float,
        default=0.0,
        help="Allowed absolute per-fixture duration increase before ratio checks fail.",
    )
    parser.add_argument(
        "--max-total-duration-increase-ratio",
        type=float,
        default=None,
        help="Optional maximum total-duration increase ratio against --baseline-report.",
    )
    parser.add_argument(
        "--max-total-duration-increase-s",
        type=float,
        default=0.0,
        help="Allowed absolute total-duration increase before ratio checks fail.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=None,
        help="Optional absolute maximum active-fixture duration budget.",
    )
    parser.add_argument(
        "--max-total-duration-s",
        type=float,
        default=None,
        help="Optional absolute maximum total active-fixture duration budget.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON report instead of the compact table.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        polygon_dir=args.polygon_dir,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        mode=args.mode,
        min_iou=args.min_iou,
        mean_iou=args.mean_iou,
        timeout_seconds=args.timeout_seconds,
        city_overrides=args.city_overrides,
        no_catalog=args.no_catalog,
        only_filters=args.only,
        fixture_config=args.fixture_config,
        execution=args.execution,
        debug_artifacts=not args.no_debug_artifacts,
        smoke_skipped=args.smoke_skipped,
        require_smoked_catalog_miss=args.require_smoked_catalog_miss,
        block_network=args.block_network,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f"{args.mode}-report.json"
    if args.baseline_report is not None:
        baseline_report = json.loads(args.baseline_report.read_text())
        regression_check = compare_report_regressions(
            report,
            baseline_report,
            baseline_path=args.baseline_report,
            max_iou_drop=args.max_iou_drop,
            max_mean_iou_drop=args.max_mean_iou_drop,
            max_duration_increase_ratio=args.max_duration_increase_ratio,
            max_duration_increase_s=args.max_duration_increase_s,
            max_total_duration_increase_ratio=args.max_total_duration_increase_ratio,
            max_total_duration_increase_s=args.max_total_duration_increase_s,
        )
        report["regression_check"] = regression_check
        report["summary"]["regression_check_passed"] = regression_check["passed"]
        report["summary"]["passed"] = bool(report["summary"]["passed"] and regression_check["passed"])
    if args.max_duration_s is not None or args.max_total_duration_s is not None:
        latency_budget_check = check_report_latency_budgets(
            report,
            max_duration_s=args.max_duration_s,
            max_total_duration_s=args.max_total_duration_s,
        )
        report["latency_budget_check"] = latency_budget_check
        report["summary"]["latency_budget_check_passed"] = latency_budget_check["passed"]
        report["summary"]["passed"] = bool(report["summary"]["passed"] and latency_budget_check["passed"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report, report_path)
    return 0 if report["summary"]["passed"] else 1


def run_benchmark(
    *,
    polygon_dir: Path,
    image_dir: Path,
    out_dir: Path,
    mode: str,
    min_iou: float,
    mean_iou: float,
    timeout_seconds: int,
    city_overrides: bool,
    only_filters: list[str],
    fixture_config: Path,
    no_catalog: bool = False,
    execution: str = "subprocess",
    debug_artifacts: bool = True,
    smoke_skipped: bool = False,
    require_smoked_catalog_miss: bool = False,
    block_network: bool = False,
) -> dict[str, Any]:
    config = load_fixture_config(fixture_config)
    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, config)
    filters = normalize_only_filters(only_filters)
    if filters:
        before_count = len(fixtures)
        fixtures = [fixture for fixture in fixtures if fixture_matches_filters(fixture, filters)]
        inventory["filtered_from"] = before_count
        inventory["only_filters"] = filters
    scores: list[BenchmarkScore] = []
    with benchmark_network_policy(block_network):
        for fixture in fixtures:
            if fixture.status != "active":
                if mode == "full" and smoke_skipped:
                    score = score_full_fixture(
                        fixture,
                        out_dir=out_dir,
                        min_iou=min_iou,
                        timeout_seconds=timeout_seconds,
                        city_overrides=city_overrides,
                        no_catalog=no_catalog,
                        execution=execution,
                        debug_artifacts=debug_artifacts,
                        score_reference=False,
                    )
                    if require_smoked_catalog_miss and score.catalog_slug:
                        score = replace(
                            score,
                            passed=False,
                            error=(
                                f"smoke-checked skipped fixture returned catalog_slug={score.catalog_slug}; "
                                "expected OCR/georeference catalog miss"
                            ),
                        )
                    scores.append(score)
                else:
                    scores.append(skipped_fixture_score(fixture, mode=mode))
                continue
            if mode == "full":
                scores.append(
                    score_full_fixture(
                        fixture,
                        out_dir=out_dir,
                        min_iou=min_iou,
                        timeout_seconds=timeout_seconds,
                        city_overrides=city_overrides,
                        no_catalog=no_catalog,
                        execution=execution,
                        debug_artifacts=debug_artifacts,
                        score_reference=True,
                    )
                )
            else:
                scores.append(score_extraction_fixture(fixture, min_iou=min_iou))

    scored = [score for score in scores if score.status == "active"]
    skipped = [score for score in scores if score.status != "active"]
    smoke_validated = [score for score in skipped if score.duration_s is not None or score.error is not None]
    ious = [score.iou for score in scored if score.iou is not None]
    durations = [score.duration_s for score in scored if score.duration_s is not None]
    smoke_durations = [score.duration_s for score in smoke_validated if score.duration_s is not None]
    average_iou = float(mean(ious)) if ious else 0.0
    min_seen_iou = float(min(ious)) if ious else 0.0
    passed_count = sum(score.passed for score in scored)
    failed_count = len(scored) - passed_count
    smoke_failed_count = sum(not score.passed for score in smoke_validated)
    active_passed = bool(scored) and failed_count == 0 and average_iou >= mean_iou
    smoke_only_passed = not scored and bool(smoke_validated) and smoke_failed_count == 0
    passed = (active_passed or smoke_only_passed) and smoke_failed_count == 0
    return {
        "mode": mode,
        "thresholds": {
            "min_iou": min_iou,
            "mean_iou": mean_iou,
            "no_catalog": no_catalog,
            "execution": execution,
            "debug_artifacts": debug_artifacts,
            "smoke_skipped": smoke_skipped,
            "require_smoked_catalog_miss": require_smoked_catalog_miss,
            "block_network": block_network,
        },
        "summary": {
            "passed": passed,
            "fixtures": len(scores),
            "scored_fixtures": len(scored),
            "skipped_fixtures": len(skipped),
            "skipped_by_status": summarize_statuses(skipped),
            "smoked_skipped_fixtures": len(smoke_validated),
            "failed_smoked_skipped_fixtures": smoke_failed_count,
            "smoked_skipped_duration_s": round(sum(smoke_durations), 3),
            "passed_fixtures": passed_count,
            "failed_fixtures": failed_count,
            "average_iou": round(average_iou, 6),
            "min_iou": round(min_seen_iou, 6),
            "total_duration_s": round(sum(durations), 3),
            "average_duration_s": round(float(mean(durations)), 3) if durations else None,
            "max_duration_s": round(max(durations), 3) if durations else None,
        },
        "inventory": inventory,
        "scores": [score.as_dict() for score in sorted(scores, key=score_sort_key)],
    }


def load_fixture_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "fixtures": {}}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Fixture config must be a JSON object: {path}")
    fixtures = data.setdefault("fixtures", {})
    if not isinstance(fixtures, dict):
        raise ValueError(f"Fixture config 'fixtures' must be an object: {path}")
    data["path"] = str(path)
    return data


def discover_fixtures(
    polygon_dir: Path,
    image_dir: Path,
    config: dict[str, Any],
) -> tuple[list[BenchmarkFixture], dict[str, Any]]:
    references = {path.stem: path for path in sorted(polygon_dir.glob("*.json"))}
    images = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    fixtures: list[BenchmarkFixture] = []
    missing_references: list[str] = []
    fixture_config = config.get("fixtures", {})
    for image_path in images:
        provider, area_slug, area_name = parse_image_name(image_path)
        slug = f"{area_slug}-{provider}" if provider and area_slug else ""
        reference_path = references.get(slug)
        if reference_path is None:
            missing_references.append(image_path.name)
            continue
        override = fixture_config.get(slug, {})
        if not isinstance(override, dict):
            raise ValueError(f"Fixture override for {slug} must be an object")
        fixtures.append(
            BenchmarkFixture(
                slug=slug,
                provider=provider,
                area=area_name,
                image_path=image_path,
                reference_path=reference_path,
                status=str(override.get("status", "active")),
                note=str(override["note"]) if override.get("note") else None,
            )
        )
    covered = {fixture.slug for fixture in fixtures}
    return fixtures, {
        "polygon_dir": str(polygon_dir),
        "image_dir": str(image_dir),
        "fixture_config": str(config.get("path", "")),
        "matched_images": len(fixtures),
        "missing_reference_images": missing_references,
        "references_without_images": sorted(set(references) - covered),
    }


def parse_image_name(path: Path) -> tuple[str, str, str]:
    text = normalized_words(path.stem)
    provider = ""
    for candidate in ("tesla", "waymo", "zoox"):
        if candidate in text.split():
            provider = candidate
            text = " ".join(word for word in text.split() if word != candidate)
            break
    area_name = " ".join(text.split())
    area_slug = AREA_ALIASES.get(area_name, slugify(area_name))
    return provider, area_slug, area_name.title()


def normalize_only_filters(raw_filters: list[str]) -> list[str]:
    filters: list[str] = []
    for raw in raw_filters:
        filters.extend(value.strip().lower() for value in raw.split(",") if value.strip())
    return filters


def fixture_matches_filters(fixture: BenchmarkFixture, filters: list[str]) -> bool:
    haystack = f"{fixture.slug} {fixture.image_path.name}".lower()
    normalized_haystack = normalized_words(haystack)
    return any(value in haystack or normalized_words(value) in normalized_haystack for value in filters)


@contextmanager
def benchmark_network_policy(block_network: bool):
    if not block_network:
        yield
        return
    previous = os.environ.get(NETWORK_BLOCK_ENV)
    os.environ[NETWORK_BLOCK_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(NETWORK_BLOCK_ENV, None)
        else:
            os.environ[NETWORK_BLOCK_ENV] = previous


def score_extraction_fixture(fixture: BenchmarkFixture, *, min_iou: float) -> BenchmarkScore:
    started = time.perf_counter()
    try:
        extraction = extract_service_area(fixture.image_path)
        reference = project_geometry(load_reference_geometry(fixture.reference_path))
        fitted = fit_pixel_geometry_to_reference_bounds(extraction.pixel_geometry, reference)
        metrics = compare_geometries(fitted, reference)
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="extraction",
            passed=metrics["iou"] >= min_iou,
            iou=metrics["iou"],
            area_ratio=metrics["area_ratio"],
            centroid_distance_m=metrics["centroid_distance_m"],
            vertices=count_vertices(extraction.pixel_geometry),
            style=extraction.style,
            duration_s=time.perf_counter() - started,
            status=fixture.status,
            note=fixture.note,
        )
    except Exception as exc:
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="extraction",
            passed=False,
            iou=None,
            area_ratio=None,
            centroid_distance_m=None,
            vertices=None,
            style=None,
            duration_s=time.perf_counter() - started,
            error=str(exc),
            status=fixture.status,
            note=fixture.note,
        )


def score_full_fixture(
    fixture: BenchmarkFixture,
    *,
    out_dir: Path,
    min_iou: float,
    timeout_seconds: int,
    city_overrides: bool,
    no_catalog: bool,
    execution: str,
    debug_artifacts: bool,
    score_reference: bool = True,
) -> BenchmarkScore:
    output_path = out_dir / "full-outputs" / f"{fixture.slug}.geojson"
    debug_dir = out_dir / "full-debug" / fixture.slug if debug_artifacts else None
    if execution == "in-process":
        return score_full_fixture_in_process(
            fixture,
            output_path=output_path,
            debug_dir=debug_dir,
            min_iou=min_iou,
            city_overrides=city_overrides,
            no_catalog=no_catalog,
            debug_artifacts=debug_artifacts,
            score_reference=score_reference,
        )
    if execution != "subprocess":
        raise ValueError(f"Unsupported full benchmark execution mode: {execution}")
    command = [
        sys.executable,
        "-m",
        "map_boundary_builder.cli",
        "--image",
        str(fixture.image_path),
        "--output",
        str(output_path),
        "--print-summary",
        "--profile-events",
    ]
    if debug_dir is not None:
        command.extend(["--debug-dir", str(debug_dir)])
    if city_overrides:
        command.extend(["--city", fixture.area])
    if no_catalog:
        command.append("--no-catalog")
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return failed_full_score(fixture, f"timed out after {timeout_seconds}s", duration_s=time.perf_counter() - started)
    duration_s = time.perf_counter() - started
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return failed_full_score(fixture, error, duration_s=duration_s)

    try:
        output = json.loads(output_path.read_text())
        summary = json.loads(completed.stdout)
        event_profile = summary.get("event_profile") if isinstance(summary, dict) else None
        stage_elapsed_s = event_profile.get("stage_elapsed_s") if isinstance(event_profile, dict) else None
        properties = output["features"][0].get("properties", {})
        output_geometry = shape(output["features"][0]["geometry"])
        metrics = score_output_geometry(output_geometry, fixture.reference_path, min_iou) if score_reference else None
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True if metrics is None else metrics["passed"],
            iou=None if metrics is None else metrics["iou"],
            area_ratio=None if metrics is None else metrics["area_ratio"],
            centroid_distance_m=None if metrics is None else metrics["centroid_distance_m"],
            vertices=count_vertices(output_geometry),
            style=summary.get("style"),
            duration_s=duration_s,
            georeference_source=summary.get("georeference_source"),
            combined_confidence=summary.get("combined_confidence"),
            catalog_slug=summary.get("catalog_slug") or properties.get("catalog_slug"),
            stage_elapsed_s=stage_elapsed_s if isinstance(stage_elapsed_s, dict) else None,
            status=fixture.status,
            note=fixture.note,
        )
    except Exception as exc:
        return failed_full_score(fixture, str(exc), duration_s=duration_s)


def score_full_fixture_in_process(
    fixture: BenchmarkFixture,
    *,
    output_path: Path,
    debug_dir: Path | None,
    min_iou: float,
    city_overrides: bool,
    no_catalog: bool,
    debug_artifacts: bool,
    score_reference: bool = True,
) -> BenchmarkScore:
    from .cli import stage_elapsed_seconds
    from .runner import BoundaryBuildOptions, build_boundary

    events: list[dict[str, Any]] = []
    started = time.perf_counter()

    def progress(event: dict[str, Any]) -> None:
        events.append({"elapsed_s": round(time.perf_counter() - started, 6), **event})

    try:
        result = build_boundary(
            fixture.image_path,
            fixture.area if city_overrides else None,
            output_path,
            debug_dir=debug_dir,
            options=BoundaryBuildOptions(
                allow_catalog=not no_catalog,
                write_mask_artifact=debug_artifacts,
            ),
            progress=progress,
        )
        duration_s = time.perf_counter() - started
        output_geometry = shape(result.geojson["features"][0]["geometry"])
        metrics = score_output_geometry(output_geometry, fixture.reference_path, min_iou) if score_reference else None
        properties = result.geojson["features"][0].get("properties", {})
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True if metrics is None else metrics["passed"],
            iou=None if metrics is None else metrics["iou"],
            area_ratio=None if metrics is None else metrics["area_ratio"],
            centroid_distance_m=None if metrics is None else metrics["centroid_distance_m"],
            vertices=count_vertices(output_geometry),
            style=result.summary.get("style"),
            duration_s=duration_s,
            georeference_source=result.summary.get("georeference_source"),
            combined_confidence=result.summary.get("combined_confidence"),
            catalog_slug=result.summary.get("catalog_slug") or properties.get("catalog_slug"),
            stage_elapsed_s=stage_elapsed_seconds(events),
            status=fixture.status,
            note=fixture.note,
        )
    except Exception as exc:
        return failed_full_score(fixture, str(exc), duration_s=time.perf_counter() - started)


def score_output_geometry(output_geometry, reference_path: Path, min_iou: float) -> dict[str, Any]:
    predicted = project_geometry(output_geometry)
    reference = project_geometry(load_reference_geometry(reference_path))
    metrics = compare_geometries(predicted, reference)
    return {
        "passed": metrics["iou"] >= min_iou,
        **metrics,
    }


def failed_full_score(fixture: BenchmarkFixture, error: str, *, duration_s: float | None = None) -> BenchmarkScore:
    return BenchmarkScore(
        slug=fixture.slug,
        image=fixture.image_path.name,
        mode="full",
        passed=False,
        iou=None,
        area_ratio=None,
        centroid_distance_m=None,
        vertices=None,
        style=None,
        duration_s=duration_s,
        error=error,
        status=fixture.status,
        note=fixture.note,
    )


def skipped_fixture_score(fixture: BenchmarkFixture, *, mode: str) -> BenchmarkScore:
    return BenchmarkScore(
        slug=fixture.slug,
        image=fixture.image_path.name,
        mode=mode,
        passed=False,
        iou=None,
        area_ratio=None,
        centroid_distance_m=None,
        vertices=None,
        style=None,
        duration_s=None,
        status=fixture.status,
        note=fixture.note,
    )


def load_reference_geometry(path: Path) -> Polygon | MultiPolygon:
    data = json.loads(path.read_text())
    if data.get("type"):
        return shape(data["features"][0]["geometry"] if data["type"] == "FeatureCollection" else data)
    coordinates = data["coordinates"]
    if coordinates[0] != coordinates[-1]:
        coordinates = [*coordinates, coordinates[0]]
    return Polygon(coordinates)


def fit_pixel_geometry_to_reference_bounds(
    pixel_geometry: Polygon | MultiPolygon,
    reference_mercator: Polygon | MultiPolygon,
) -> Polygon | MultiPolygon:
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    ref_min_x, ref_min_y, ref_max_x, ref_max_y = reference_mercator.bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError("extracted pixel geometry has empty bounds")

    scale_x = (ref_max_x - ref_min_x) / width
    scale_y = (ref_max_y - ref_min_y) / height

    def fit(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        return ref_min_x + (x - min_x) * scale_x, ref_max_y - (y - min_y) * scale_y

    return transform(fit, pixel_geometry)


def project_geometry(geometry: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    return transform(lonlat_to_mercator, geometry)


def compare_geometries(predicted: Polygon | MultiPolygon, reference: Polygon | MultiPolygon) -> dict[str, float]:
    predicted = predicted.buffer(0)
    reference = reference.buffer(0)
    intersection = predicted.intersection(reference).area
    union = predicted.union(reference).area
    iou = intersection / union if union else 0.0
    area_ratio = predicted.area / reference.area if reference.area else 0.0
    centroid_distance_m = predicted.centroid.distance(reference.centroid)
    return {
        "iou": float(iou),
        "area_ratio": float(area_ratio),
        "centroid_distance_m": float(centroid_distance_m),
    }


def count_vertices(geometry: Polygon | MultiPolygon) -> int:
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    return sum(len(poly.exterior.coords) + sum(len(ring.coords) for ring in poly.interiors) for poly in polygons)


def normalized_words(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def slugify(text: str) -> str:
    return "-".join(normalized_words(text).split())


def score_sort_key(score: BenchmarkScore) -> tuple[int, bool, float]:
    return (0 if score.status == "active" else 1, score.passed, score.iou or -1.0)


def summarize_statuses(scores: list[BenchmarkScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in scores:
        counts[score.status] = counts.get(score.status, 0) + 1
    return dict(sorted(counts.items()))


def compare_report_regressions(
    report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    baseline_path: Path | None = None,
    max_iou_drop: float = 0.0,
    max_mean_iou_drop: float = 0.0,
    max_duration_increase_ratio: float | None = None,
    max_duration_increase_s: float = 0.0,
    max_total_duration_increase_ratio: float | None = None,
    max_total_duration_increase_s: float = 0.0,
) -> dict[str, Any]:
    candidate_scores = active_iou_scores_by_slug(report)
    baseline_scores = active_iou_scores_by_slug(baseline_report)
    issues: list[dict[str, Any]] = []
    tolerance = max(0.0, float(max_iou_drop))
    duration_tolerance = (
        None
        if max_duration_increase_ratio is None
        else max(0.0, float(max_duration_increase_ratio))
    )
    duration_tolerance_s = max(0.0, float(max_duration_increase_s))
    for slug, baseline_score in sorted(baseline_scores.items()):
        candidate_score = candidate_scores.get(slug)
        if candidate_score is None:
            issues.append(
                {
                    "slug": slug,
                    "kind": "missing_candidate_score",
                    "baseline_iou": baseline_score["iou"],
                }
            )
            continue
        drop = float(baseline_score["iou"]) - float(candidate_score["iou"])
        if drop > tolerance:
            issues.append(
                {
                    "slug": slug,
                    "kind": "iou_drop",
                    "baseline_iou": baseline_score["iou"],
                    "candidate_iou": candidate_score["iou"],
                    "drop": round(drop, 6),
                }
            )
        if duration_tolerance is not None:
            baseline_duration = parse_report_duration(baseline_score.get("duration_s"))
            candidate_duration = parse_report_duration(candidate_score.get("duration_s"))
            if baseline_duration is not None and candidate_duration is not None and baseline_duration > 0:
                increase_s = candidate_duration - baseline_duration
                increase_ratio = candidate_duration / baseline_duration - 1.0
                if increase_s > duration_tolerance_s and increase_ratio > duration_tolerance:
                    issues.append(
                        {
                            "slug": slug,
                            "kind": "duration_increase",
                            "baseline_duration_s": round(baseline_duration, 6),
                            "candidate_duration_s": round(candidate_duration, 6),
                            "increase_s": round(increase_s, 6),
                            "increase_ratio": round(increase_ratio, 6),
                        }
                    )

    baseline_mean = float(baseline_report.get("summary", {}).get("average_iou", 0.0))
    candidate_mean = float(report.get("summary", {}).get("average_iou", 0.0))
    mean_drop = baseline_mean - candidate_mean
    mean_tolerance = max(0.0, float(max_mean_iou_drop))
    if mean_drop > mean_tolerance:
        issues.append(
            {
                "kind": "average_iou_drop",
                "baseline_average_iou": round(baseline_mean, 6),
                "candidate_average_iou": round(candidate_mean, 6),
                "drop": round(mean_drop, 6),
            }
        )
    total_duration_tolerance = (
        None
        if max_total_duration_increase_ratio is None
        else max(0.0, float(max_total_duration_increase_ratio))
    )
    total_duration_tolerance_s = max(0.0, float(max_total_duration_increase_s))
    if total_duration_tolerance is not None:
        baseline_total = parse_report_duration(baseline_report.get("summary", {}).get("total_duration_s"))
        candidate_total = parse_report_duration(report.get("summary", {}).get("total_duration_s"))
        if baseline_total is not None and candidate_total is not None and baseline_total > 0:
            total_increase_s = candidate_total - baseline_total
            total_increase_ratio = candidate_total / baseline_total - 1.0
            if total_increase_s > total_duration_tolerance_s and total_increase_ratio > total_duration_tolerance:
                issues.append(
                    {
                        "kind": "total_duration_increase",
                        "baseline_total_duration_s": round(baseline_total, 6),
                        "candidate_total_duration_s": round(candidate_total, 6),
                        "increase_s": round(total_increase_s, 6),
                        "increase_ratio": round(total_increase_ratio, 6),
                    }
                )

    return {
        "passed": not issues,
        "baseline_report": str(baseline_path) if baseline_path is not None else None,
        "max_iou_drop": tolerance,
        "max_mean_iou_drop": mean_tolerance,
        "max_duration_increase_ratio": duration_tolerance,
        "max_duration_increase_s": duration_tolerance_s,
        "max_total_duration_increase_ratio": total_duration_tolerance,
        "max_total_duration_increase_s": total_duration_tolerance_s,
        "compared_fixtures": len(baseline_scores),
        "issues": issues,
    }


def check_report_latency_budgets(
    report: dict[str, Any],
    *,
    max_duration_s: float | None = None,
    max_total_duration_s: float | None = None,
) -> dict[str, Any]:
    duration_budget = None if max_duration_s is None else max(0.0, float(max_duration_s))
    total_budget = None if max_total_duration_s is None else max(0.0, float(max_total_duration_s))
    issues: list[dict[str, Any]] = []
    if duration_budget is not None:
        for row in report.get("scores", []):
            if not isinstance(row, dict) or row.get("status") != "active":
                continue
            duration = parse_report_duration(row.get("duration_s"))
            if duration is not None and duration > duration_budget:
                slug = row.get("slug")
                issues.append(
                    {
                        "slug": slug if isinstance(slug, str) else "",
                        "kind": "duration_budget_exceeded",
                        "duration_s": round(duration, 6),
                        "max_duration_s": duration_budget,
                        "excess_s": round(duration - duration_budget, 6),
                    }
                )
    if total_budget is not None:
        total_duration = parse_report_duration(report.get("summary", {}).get("total_duration_s"))
        if total_duration is not None and total_duration > total_budget:
            issues.append(
                {
                    "kind": "total_duration_budget_exceeded",
                    "total_duration_s": round(total_duration, 6),
                    "max_total_duration_s": total_budget,
                    "excess_s": round(total_duration - total_budget, 6),
                }
            )
    return {
        "passed": not issues,
        "max_duration_s": duration_budget,
        "max_total_duration_s": total_budget,
        "issues": issues,
    }


def active_iou_scores_by_slug(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for row in report.get("scores", []):
        if not isinstance(row, dict):
            continue
        if row.get("status") != "active" or row.get("iou") is None:
            continue
        slug = row.get("slug")
        if isinstance(slug, str) and slug:
            scores[slug] = row
    return scores


def parse_report_duration(value: Any) -> float | None:
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration >= 0.0 else None


def print_table(report: dict[str, Any], report_path: Path) -> None:
    summary = report["summary"]
    if summary["passed"]:
        status = "PASS"
    elif summary["scored_fixtures"] == 0 and summary["skipped_fixtures"] > 0:
        status = "SKIP"
    else:
        status = "FAIL"
    skipped_text = f"{summary['skipped_fixtures']} skipped"
    skipped_by_status = summary.get("skipped_by_status", {})
    if skipped_by_status:
        skipped_reasons = ", ".join(f"{reason}={count}" for reason, count in skipped_by_status.items())
        skipped_text = f"{skipped_text} ({skipped_reasons})"
    if summary.get("smoked_skipped_fixtures"):
        skipped_text = (
            f"{skipped_text}, {summary['smoked_skipped_fixtures']} smoke-checked, "
            f"{summary.get('failed_smoked_skipped_fixtures', 0)} smoke failed, "
            f"smoke total {format_duration(summary.get('smoked_skipped_duration_s'))}"
        )
    print(
        f"{status} {report['mode']} benchmark: "
        f"{summary['passed_fixtures']}/{summary['scored_fixtures']} scored fixtures, "
        f"{skipped_text}, "
        f"avg IoU {summary['average_iou']:.3f}, min IoU {summary['min_iou']:.3f}, "
        f"total {format_duration(summary.get('total_duration_s'))}"
    )
    print(f"report: {report_path}")
    print("")
    print(f"{'status':6s} {'iou':>6s} {'time':>7s} {'area':>6s} {'verts':>6s} {'style':12s} {'source':38s} slug")
    for row in report["scores"]:
        row_status = "PASS" if row["passed"] else "FAIL"
        if row["status"] != "active":
            row_status = "SMOKE" if row.get("duration_s") is not None and row["passed"] else "SKIP"
            if row.get("error"):
                row_status = "FAIL"
        iou = f"{row['iou']:.3f}" if row["iou"] is not None else ("err" if row.get("error") else "-")
        area = f"{row['area_ratio']:.2f}" if row["area_ratio"] is not None else "-"
        duration = format_duration(row.get("duration_s"))
        vertices = str(row["vertices"]) if row["vertices"] is not None else "-"
        style = row["style"] or "-"
        source = (row.get("georeference_source") or "-")[:38]
        print(
            f"{row_status:6s} {iou:>6s} {duration:>7s} {area:>6s} "
            f"{vertices:>6s} {style:12s} {source:38s} {row['slug']}"
        )
        if row["error"]:
            print(f"       error: {row['error']}")
        if row["note"]:
            print(f"       note: {row['note']}")
    references_without_images = report["inventory"]["references_without_images"]
    if references_without_images:
        print("")
        print("references without screenshots: " + ", ".join(references_without_images))
    regression_check = report.get("regression_check")
    if regression_check:
        print("")
        regression_status = "PASS" if regression_check["passed"] else "FAIL"
        print(
            f"{regression_status} regression check: "
            f"{len(regression_check['issues'])} issues against {regression_check['baseline_report']}"
        )
        for issue in regression_check["issues"][:12]:
            if issue["kind"] == "iou_drop":
                print(
                    f"       {issue['slug']}: IoU {issue['baseline_iou']:.6f} -> "
                    f"{issue['candidate_iou']:.6f} (drop {issue['drop']:.6f})"
                )
            elif issue["kind"] == "average_iou_drop":
                print(
                    f"       average IoU {issue['baseline_average_iou']:.6f} -> "
                    f"{issue['candidate_average_iou']:.6f} (drop {issue['drop']:.6f})"
                )
            elif issue["kind"] == "duration_increase":
                print(
                    f"       {issue['slug']}: duration {issue['baseline_duration_s']:.3f}s -> "
                    f"{issue['candidate_duration_s']:.3f}s "
                    f"(+{issue['increase_s']:.3f}s, ratio {issue['increase_ratio']:.3f})"
                )
            elif issue["kind"] == "total_duration_increase":
                print(
                    f"       total duration {issue['baseline_total_duration_s']:.3f}s -> "
                    f"{issue['candidate_total_duration_s']:.3f}s "
                    f"(+{issue['increase_s']:.3f}s, ratio {issue['increase_ratio']:.3f})"
                )
            else:
                print(f"       {issue['slug']}: missing candidate score")
    latency_budget_check = report.get("latency_budget_check")
    if latency_budget_check:
        print("")
        latency_status = "PASS" if latency_budget_check["passed"] else "FAIL"
        print(
            f"{latency_status} latency budget: "
            f"{len(latency_budget_check['issues'])} issues"
        )
        for issue in latency_budget_check["issues"][:12]:
            if issue["kind"] == "duration_budget_exceeded":
                print(
                    f"       {issue['slug']}: duration {issue['duration_s']:.3f}s "
                    f"> budget {issue['max_duration_s']:.3f}s "
                    f"(+{issue['excess_s']:.3f}s)"
                )
            elif issue["kind"] == "total_duration_budget_exceeded":
                print(
                    f"       total duration {issue['total_duration_s']:.3f}s "
                    f"> budget {issue['max_total_duration_s']:.3f}s "
                    f"(+{issue['excess_s']:.3f}s)"
                )


def format_duration(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    raise SystemExit(main())
