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
from statistics import mean, median
from typing import Any

from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform

from .extract import extract_service_area
from .georef_transform import lonlat_to_mercator
from .network_policy import NETWORK_BLOCK_ENV
from .pipeline_version import PIPELINE_VERSION_ENV, get_pipeline_version
from .runtime_config import ocr_runtime_config

DEFAULT_POLYGON_DIR = Path("/Users/ethanmckanna/GitHub/av-coverage-checker/data/service-areas/polygons")
DEFAULT_IMAGE_DIR = Path("/Users/ethanmckanna/Downloads/service area images")
DEFAULT_OUT_DIR = Path("out/service-area-benchmark")
DEFAULT_FIXTURE_CONFIG = Path("benchmarks/service-area-fixtures.json")
DEFAULT_SCORED_CATALOG_EVIDENCE_MIN_IOU = 0.70
DEFAULT_SCORED_CATALOG_EVIDENCE_MIN_AREA_RATIO = 0.85
DEFAULT_SCORED_CATALOG_EVIDENCE_MAX_AREA_RATIO = 1.15
BENCHMARK_GENERATION_ENV_DEFAULTS = {
    "MAP_BOUNDARY_BLOCK_NETWORK": "",
    "MAP_BOUNDARY_CACHE_DIR": ".cache/map-boundary-builder",
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
    "MAP_BOUNDARY_RUNNER_OCR_CACHE": "1",
    "MAP_BOUNDARY_SCALED_EXTRACTION_CACHE_MAX_PIXELS": "3000000",
    "MAP_BOUNDARY_SCALED_EXTRACTION_MEMORY_CACHE_MAX": "24",
    PIPELINE_VERSION_ENV: "",
}

IMAGE_SUFFIXES = {".avif", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AREA_ALIASES = {
    "bay area": "bay-area",
    "san francisco": "bay-area",
    "sf": "bay-area",
    "las vegas": "las-vegas",
    "los angeles": "los-angeles",
    "san antonio": "san-antonio",
}
PROVIDERS = ("avride", "tesla", "waymo", "zoox")

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
    catalog_shape_iou: float | None = None
    catalog_area_ratio: float | None = None
    road_match_score: float | None = None
    road_match_elapsed_s: float | None = None
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
            "duration_s": round(self.duration_s, 6) if self.duration_s is not None else None,
            "georeference_source": self.georeference_source,
            "combined_confidence": round(self.combined_confidence, 6) if self.combined_confidence is not None else None,
            "catalog_slug": self.catalog_slug,
            "catalog_shape_iou": round(self.catalog_shape_iou, 6) if self.catalog_shape_iou is not None else None,
            "catalog_area_ratio": round(self.catalog_area_ratio, 6) if self.catalog_area_ratio is not None else None,
            "road_match_score": round(self.road_match_score, 6) if self.road_match_score is not None else None,
            "road_match_elapsed_s": (
                round(self.road_match_elapsed_s, 6) if self.road_match_elapsed_s is not None else None
            ),
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
        "--catalog-probe-missed",
        action="store_true",
        help="For --mode full, exercise the production handoff path after a low-resolution catalog probe miss.",
    )
    parser.add_argument(
        "--catalog-probe-miss-low-iou",
        action="store_true",
        help="For --mode full, mark the prior catalog probe miss as far from active catalog shapes.",
    )
    parser.add_argument(
        "--neutral-filename-hint",
        action="store_true",
        help=(
            "For --mode full, replace provider/market filename hints with a neutral "
            "upload name so image-only generalization can be measured."
        ),
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
        "--score-skipped-catalog-references",
        action="store_true",
        help=(
            "For --mode full, score non-active fixtures against matching current catalog geometry "
            "when the saved reference is stale."
        ),
    )
    parser.add_argument(
        "--require-scored-catalog-evidence",
        action="store_true",
        help=(
            "Fail catalog outputs whose recorded image-to-catalog shape evidence is weak. "
            "This is enabled automatically by --score-skipped-catalog-references so exact "
            "catalog geometry cannot score as a tautological 1.0 without a strong "
            "source-image match."
        ),
    )
    parser.add_argument(
        "--min-scored-catalog-shape-iou",
        type=float,
        default=DEFAULT_SCORED_CATALOG_EVIDENCE_MIN_IOU,
        help="Minimum catalog_shape_iou required by --require-scored-catalog-evidence.",
    )
    parser.add_argument(
        "--min-scored-catalog-area-ratio",
        type=float,
        default=DEFAULT_SCORED_CATALOG_EVIDENCE_MIN_AREA_RATIO,
        help="Minimum catalog_area_ratio required by --require-scored-catalog-evidence.",
    )
    parser.add_argument(
        "--max-scored-catalog-area-ratio",
        type=float,
        default=DEFAULT_SCORED_CATALOG_EVIDENCE_MAX_AREA_RATIO,
        help="Maximum catalog_area_ratio required by --require-scored-catalog-evidence.",
    )
    parser.add_argument(
        "--require-smoked-catalog-miss",
        action="store_true",
        help=(
            "Implies --smoke-skipped and fails smoke-checked fixtures that return a catalog_slug. "
            "Use with targeted --only filters when those drifted screenshots must stay on OCR/georeference."
        ),
    )
    parser.add_argument(
        "--block-network",
        action="store_true",
        help="Block live geocoder/Overpass fallbacks during full benchmark generation.",
    )
    parser.add_argument(
        "--repeat-profile-runs",
        type=int,
        default=0,
        help=(
            "For --mode full with --execution in-process, rerun each evaluated fixture this many "
            "additional times and record warm-instance latency samples without changing score gates."
        ),
    )
    parser.add_argument(
        "--repeat-profile-warmups",
        type=int,
        default=0,
        help="Number of repeat-profile samples per fixture to exclude from aggregate repeat statistics.",
    )
    parser.add_argument(
        "--max-repeat-profile-duration-s",
        type=float,
        default=None,
        help="Optional maximum analyzed repeat-profile sample duration budget.",
    )
    parser.add_argument(
        "--max-repeat-profile-median-duration-s",
        type=float,
        default=None,
        help="Optional maximum analyzed repeat-profile median duration budget.",
    )
    parser.add_argument(
        "--max-repeat-profile-stage-duration-s",
        action="append",
        default=[],
        metavar="STAGE=SECONDS",
        help=(
            "Optional maximum analyzed repeat-profile per-stage duration budget. "
            "Repeat or comma-separate entries such as ocr=0.8,extract=0.3."
        ),
    )
    parser.add_argument(
        "--min-repeat-profile-pass-ratio",
        type=float,
        default=None,
        help="Optional minimum pass ratio across analyzed repeat-profile samples.",
    )
    parser.add_argument(
        "--min-repeat-profile-subsecond-ratio",
        type=float,
        default=None,
        help="Optional minimum ratio of analyzed repeat-profile samples under one second.",
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
        "--max-evaluated-duration-increase-ratio",
        type=float,
        default=None,
        help=(
            "Optional maximum active plus smoke-checked duration increase ratio "
            "against --baseline-report."
        ),
    )
    parser.add_argument(
        "--max-evaluated-duration-increase-s",
        type=float,
        default=0.0,
        help="Allowed absolute active plus smoke-checked duration increase before ratio checks fail.",
    )
    parser.add_argument(
        "--max-evaluated-stage-duration-increase-ratio",
        type=float,
        default=None,
        help=(
            "Optional maximum per-stage active plus smoke-checked duration increase "
            "ratio against --baseline-report."
        ),
    )
    parser.add_argument(
        "--max-evaluated-stage-duration-increase-s",
        type=float,
        default=0.0,
        help="Allowed absolute per-stage active plus smoke-checked duration increase before ratio checks fail.",
    )
    parser.add_argument(
        "--max-evaluated-stage-duration-s",
        action="append",
        default=[],
        metavar="STAGE=SECONDS",
        help=(
            "Optional absolute evaluated stage-duration budget. "
            "Repeat or comma-separate entries such as ocr=4.0,extract=2.0."
        ),
    )
    parser.add_argument(
        "--max-evaluated-road-match-increase-ratio",
        type=float,
        default=None,
        help=(
            "Optional maximum active plus smoke-checked road-match elapsed increase "
            "ratio against --baseline-report."
        ),
    )
    parser.add_argument(
        "--max-evaluated-road-match-increase-s",
        type=float,
        default=0.0,
        help="Allowed absolute active plus smoke-checked road-match elapsed increase before ratio checks fail.",
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
    parser.add_argument(
        "--max-evaluated-duration-s",
        type=float,
        default=None,
        help="Optional absolute maximum active plus smoke-checked fixture duration budget.",
    )
    parser.add_argument(
        "--max-evaluated-road-match-s",
        type=float,
        default=None,
        help="Optional absolute maximum active plus smoke-checked road-match elapsed budget.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON report instead of the compact table.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        max_evaluated_stage_duration_s = parse_stage_duration_budgets(
            args.max_evaluated_stage_duration_s
        )
        max_repeat_profile_stage_duration_s = parse_stage_duration_budgets(
            args.max_repeat_profile_stage_duration_s
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.repeat_profile_runs < 0:
        parser.error("--repeat-profile-runs must be non-negative")
    if args.repeat_profile_warmups < 0:
        parser.error("--repeat-profile-warmups must be non-negative")
    if args.repeat_profile_runs and args.mode != "full":
        parser.error("--repeat-profile-runs requires --mode full")
    if args.repeat_profile_runs and args.execution != "in-process":
        parser.error("--repeat-profile-runs requires --execution in-process")
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
        catalog_probe_missed=args.catalog_probe_missed,
        catalog_probe_miss_low_iou=args.catalog_probe_miss_low_iou,
        neutral_filename_hint=args.neutral_filename_hint,
        only_filters=args.only,
        fixture_config=args.fixture_config,
        execution=args.execution,
        debug_artifacts=not args.no_debug_artifacts,
        smoke_skipped=args.smoke_skipped,
        score_skipped_catalog_references=args.score_skipped_catalog_references,
        require_scored_catalog_evidence=(
            args.require_scored_catalog_evidence or args.score_skipped_catalog_references
        ),
        min_scored_catalog_shape_iou=args.min_scored_catalog_shape_iou,
        min_scored_catalog_area_ratio=args.min_scored_catalog_area_ratio,
        max_scored_catalog_area_ratio=args.max_scored_catalog_area_ratio,
        require_smoked_catalog_miss=args.require_smoked_catalog_miss,
        block_network=args.block_network,
        repeat_profile_runs=args.repeat_profile_runs,
        repeat_profile_warmups=args.repeat_profile_warmups,
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
            max_evaluated_duration_increase_ratio=args.max_evaluated_duration_increase_ratio,
            max_evaluated_duration_increase_s=args.max_evaluated_duration_increase_s,
            max_evaluated_stage_duration_increase_ratio=(
                args.max_evaluated_stage_duration_increase_ratio
            ),
            max_evaluated_stage_duration_increase_s=args.max_evaluated_stage_duration_increase_s,
            max_evaluated_road_match_increase_ratio=args.max_evaluated_road_match_increase_ratio,
            max_evaluated_road_match_increase_s=args.max_evaluated_road_match_increase_s,
        )
        report["regression_check"] = regression_check
        report["summary"]["regression_check_passed"] = regression_check["passed"]
        report["summary"]["passed"] = bool(report["summary"]["passed"] and regression_check["passed"])
    if (
        args.max_duration_s is not None
        or args.max_total_duration_s is not None
        or args.max_evaluated_duration_s is not None
        or bool(max_evaluated_stage_duration_s)
        or args.max_evaluated_road_match_s is not None
        or args.max_repeat_profile_duration_s is not None
        or args.max_repeat_profile_median_duration_s is not None
        or bool(max_repeat_profile_stage_duration_s)
        or args.min_repeat_profile_pass_ratio is not None
        or args.min_repeat_profile_subsecond_ratio is not None
    ):
        latency_budget_check = check_report_latency_budgets(
            report,
            max_duration_s=args.max_duration_s,
            max_total_duration_s=args.max_total_duration_s,
            max_evaluated_duration_s=args.max_evaluated_duration_s,
            max_evaluated_stage_duration_s=max_evaluated_stage_duration_s,
            max_evaluated_road_match_s=args.max_evaluated_road_match_s,
            max_repeat_profile_duration_s=args.max_repeat_profile_duration_s,
            max_repeat_profile_median_duration_s=args.max_repeat_profile_median_duration_s,
            max_repeat_profile_stage_duration_s=max_repeat_profile_stage_duration_s,
            min_repeat_profile_pass_ratio=args.min_repeat_profile_pass_ratio,
            min_repeat_profile_subsecond_ratio=args.min_repeat_profile_subsecond_ratio,
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
    catalog_probe_missed: bool = False,
    catalog_probe_miss_low_iou: bool = False,
    neutral_filename_hint: bool = False,
    execution: str = "subprocess",
    debug_artifacts: bool = True,
    smoke_skipped: bool = False,
    score_skipped_catalog_references: bool = False,
    require_scored_catalog_evidence: bool = False,
    min_scored_catalog_shape_iou: float = DEFAULT_SCORED_CATALOG_EVIDENCE_MIN_IOU,
    min_scored_catalog_area_ratio: float = DEFAULT_SCORED_CATALOG_EVIDENCE_MIN_AREA_RATIO,
    max_scored_catalog_area_ratio: float = DEFAULT_SCORED_CATALOG_EVIDENCE_MAX_AREA_RATIO,
    require_smoked_catalog_miss: bool = False,
    block_network: bool = False,
    repeat_profile_runs: int = 0,
    repeat_profile_warmups: int = 0,
) -> dict[str, Any]:
    if repeat_profile_runs < 0:
        raise ValueError("repeat_profile_runs must be non-negative")
    if repeat_profile_warmups < 0:
        raise ValueError("repeat_profile_warmups must be non-negative")
    if repeat_profile_runs and mode != "full":
        raise ValueError("repeat_profile_runs requires mode='full'")
    if repeat_profile_runs and execution != "in-process":
        raise ValueError("repeat_profile_runs requires execution='in-process'")
    require_scored_catalog_evidence = bool(
        require_scored_catalog_evidence or score_skipped_catalog_references
    )
    smoke_skipped = bool(smoke_skipped or require_smoked_catalog_miss)
    config = load_fixture_config(fixture_config)
    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, config)
    filters = normalize_only_filters(only_filters)
    if filters:
        before_count = len(fixtures)
        fixtures = [fixture for fixture in fixtures if fixture_matches_filters(fixture, filters)]
        inventory["filtered_from"] = before_count
        inventory["only_filters"] = filters
    scores: list[BenchmarkScore] = []
    repeat_targets: list[dict[str, Any]] = []
    with benchmark_network_policy(block_network):
        for fixture in fixtures:
            if fixture.status != "active":
                catalog_reference = (
                    catalog_reference_geometry_for_fixture(fixture)
                    if mode == "full" and score_skipped_catalog_references
                    else None
                )
                if catalog_reference is not None:
                    score_fixture = replace(
                        fixture,
                        status="active",
                        note=fixture_catalog_reference_note(fixture),
                    )
                    score = score_full_fixture(
                        score_fixture,
                        out_dir=out_dir,
                        min_iou=min_iou,
                        timeout_seconds=timeout_seconds,
                        city_overrides=city_overrides,
                        no_catalog=no_catalog,
                        catalog_probe_missed=catalog_probe_missed,
                        catalog_probe_miss_low_iou=catalog_probe_miss_low_iou,
                        neutral_filename_hint=neutral_filename_hint,
                        execution=execution,
                        debug_artifacts=debug_artifacts,
                        score_reference=True,
                        reference_geometry=catalog_reference,
                    )
                    if require_scored_catalog_evidence:
                        score = require_current_catalog_evidence(
                            score,
                            min_shape_iou=min_scored_catalog_shape_iou,
                            min_area_ratio=min_scored_catalog_area_ratio,
                            max_area_ratio=max_scored_catalog_area_ratio,
                        )
                    scores.append(score)
                    repeat_targets.append(
                        {
                            "fixture": score_fixture,
                            "score_reference": True,
                            "reference_geometry": catalog_reference,
                        }
                    )
                elif mode == "full" and smoke_skipped:
                    score = score_full_fixture(
                        fixture,
                        out_dir=out_dir,
                        min_iou=min_iou,
                        timeout_seconds=timeout_seconds,
                        city_overrides=city_overrides,
                        no_catalog=no_catalog,
                        catalog_probe_missed=catalog_probe_missed,
                        catalog_probe_miss_low_iou=catalog_probe_miss_low_iou,
                        neutral_filename_hint=neutral_filename_hint,
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
                    repeat_targets.append(
                        {
                            "fixture": fixture,
                            "score_reference": False,
                            "reference_geometry": None,
                        }
                    )
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
                        catalog_probe_missed=catalog_probe_missed,
                        catalog_probe_miss_low_iou=catalog_probe_miss_low_iou,
                        neutral_filename_hint=neutral_filename_hint,
                        execution=execution,
                        debug_artifacts=debug_artifacts,
                        score_reference=True,
                    )
                )
                repeat_targets.append(
                    {
                        "fixture": fixture,
                        "score_reference": True,
                        "reference_geometry": None,
                    }
                )
            else:
                scores.append(score_extraction_fixture(fixture, min_iou=min_iou))
        repeat_profile = (
            build_repeat_profile(
                repeat_targets,
                runs_per_fixture=repeat_profile_runs,
                warmup_runs_per_fixture=repeat_profile_warmups,
                out_dir=out_dir,
                min_iou=min_iou,
                timeout_seconds=timeout_seconds,
                city_overrides=city_overrides,
                no_catalog=no_catalog,
                catalog_probe_missed=catalog_probe_missed,
                catalog_probe_miss_low_iou=catalog_probe_miss_low_iou,
                neutral_filename_hint=neutral_filename_hint,
                execution=execution,
                debug_artifacts=debug_artifacts,
            )
            if repeat_profile_runs
            else None
        )
        runtime_config = benchmark_runtime_config()

    scored = [score for score in scores if score.status == "active"]
    skipped = [score for score in scores if score.status != "active"]
    smoke_validated = [score for score in skipped if score.duration_s is not None or score.error is not None]
    ious = [score.iou for score in scored if score.iou is not None]
    durations = [score.duration_s for score in scored if score.duration_s is not None]
    smoke_durations = [score.duration_s for score in smoke_validated if score.duration_s is not None]
    active_total_duration = sum(durations)
    smoke_total_duration = sum(smoke_durations)
    active_stage_duration = summarize_stage_durations(scored)
    smoke_stage_duration = summarize_stage_durations(smoke_validated)
    evaluated_stage_duration = combine_stage_durations(active_stage_duration, smoke_stage_duration)
    active_road_match_elapsed = summarize_road_match_elapsed(scored)
    smoke_road_match_elapsed = summarize_road_match_elapsed(smoke_validated)
    average_iou = float(mean(ious)) if ious else 0.0
    min_seen_iou = float(min(ious)) if ious else 0.0
    passed_count = sum(score.passed for score in scored)
    failed_count = len(scored) - passed_count
    smoke_failed_count = sum(not score.passed for score in smoke_validated)
    active_passed = bool(scored) and failed_count == 0 and average_iou >= mean_iou
    smoke_only_passed = not scored and bool(smoke_validated) and smoke_failed_count == 0
    passed = (active_passed or smoke_only_passed) and smoke_failed_count == 0
    report = {
        "mode": mode,
        "thresholds": {
            "min_iou": min_iou,
            "mean_iou": mean_iou,
            "no_catalog": no_catalog,
            "catalog_probe_missed": catalog_probe_missed,
            "catalog_probe_miss_low_iou": catalog_probe_miss_low_iou,
            "neutral_filename_hint": neutral_filename_hint,
            "execution": execution,
            "debug_artifacts": debug_artifacts,
            "smoke_skipped": smoke_skipped,
            "score_skipped_catalog_references": score_skipped_catalog_references,
            "require_scored_catalog_evidence": require_scored_catalog_evidence,
            "min_scored_catalog_shape_iou": max(0.0, float(min_scored_catalog_shape_iou)),
            "min_scored_catalog_area_ratio": max(0.0, float(min_scored_catalog_area_ratio)),
            "max_scored_catalog_area_ratio": max(0.0, float(max_scored_catalog_area_ratio)),
            "require_smoked_catalog_miss": require_smoked_catalog_miss,
            "block_network": block_network,
            "repeat_profile_runs": repeat_profile_runs,
            "repeat_profile_warmups": repeat_profile_warmups,
        },
        "runtime_config": runtime_config,
        "summary": {
            "passed": passed,
            "fixtures": len(scores),
            "scored_fixtures": len(scored),
            "skipped_fixtures": len(skipped),
            "skipped_by_status": summarize_statuses(skipped),
            "smoked_skipped_fixtures": len(smoke_validated),
            "failed_smoked_skipped_fixtures": smoke_failed_count,
            "smoked_skipped_duration_s": round(smoke_total_duration, 6),
            "passed_fixtures": passed_count,
            "failed_fixtures": failed_count,
            "average_iou": round(average_iou, 6),
            "min_iou": round(min_seen_iou, 6),
            "total_duration_s": round(active_total_duration, 6),
            "active_total_duration_s": round(active_total_duration, 6),
            "evaluated_duration_s": round(active_total_duration + smoke_total_duration, 6),
            "active_stage_duration_s": active_stage_duration,
            "smoked_skipped_stage_duration_s": smoke_stage_duration,
            "evaluated_stage_duration_s": evaluated_stage_duration,
            "active_road_match_elapsed_s": active_road_match_elapsed,
            "smoked_skipped_road_match_elapsed_s": smoke_road_match_elapsed,
            "evaluated_road_match_elapsed_s": round(active_road_match_elapsed + smoke_road_match_elapsed, 6),
            "average_duration_s": round(float(mean(durations)), 6) if durations else None,
            "max_duration_s": round(max(durations), 6) if durations else None,
        },
        "inventory": inventory,
        "scores": [score.as_dict() for score in sorted(scores, key=score_sort_key)],
    }
    if repeat_profile is not None:
        report["repeat_profile"] = repeat_profile
    return report


def build_repeat_profile(
    targets: list[dict[str, Any]],
    *,
    runs_per_fixture: int,
    warmup_runs_per_fixture: int,
    out_dir: Path,
    min_iou: float,
    timeout_seconds: int,
    city_overrides: bool,
    no_catalog: bool,
    catalog_probe_missed: bool,
    catalog_probe_miss_low_iou: bool,
    neutral_filename_hint: bool,
    execution: str,
    debug_artifacts: bool,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for target in targets:
        fixture = target["fixture"]
        for repeat_index in range(1, runs_per_fixture + 1):
            score = score_full_fixture(
                fixture,
                out_dir=out_dir,
                min_iou=min_iou,
                timeout_seconds=timeout_seconds,
                city_overrides=city_overrides,
                no_catalog=no_catalog,
                catalog_probe_missed=catalog_probe_missed,
                catalog_probe_miss_low_iou=catalog_probe_miss_low_iou,
                neutral_filename_hint=neutral_filename_hint,
                execution=execution,
                debug_artifacts=debug_artifacts,
                score_reference=bool(target["score_reference"]),
                reference_geometry=target.get("reference_geometry"),
            ).as_dict()
            score = {
                "repeat_index": repeat_index,
                "warmup": repeat_index <= warmup_runs_per_fixture,
                **score,
            }
            samples.append(score)
    return summarize_repeat_profile_samples(
        samples,
        runs_per_fixture=runs_per_fixture,
        warmup_runs_per_fixture=warmup_runs_per_fixture,
    )


def summarize_repeat_profile_samples(
    samples: list[dict[str, Any]],
    *,
    runs_per_fixture: int,
    warmup_runs_per_fixture: int,
) -> dict[str, Any]:
    fixture_samples: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        slug = str(sample.get("slug") or "")
        if not slug:
            continue
        fixture_samples.setdefault(slug, []).append(sample)
    fixture_summaries = {
        slug: summarize_repeat_profile_sample_group(slug_samples)
        for slug, slug_samples in sorted(fixture_samples.items())
    }
    analyzed_samples = repeat_profile_analyzed_samples(samples)
    subsecond_fixture_count = sum(
        1
        for fixture_summary in fixture_summaries.values()
        if parse_report_duration(fixture_summary.get("min_duration_s")) is not None
        and float(fixture_summary["min_duration_s"]) < 1.0
    )
    return {
        "runs_per_fixture": runs_per_fixture,
        "warmup_runs_per_fixture": warmup_runs_per_fixture,
        "summary": {
            "fixtures": len(fixture_summaries),
            "samples": len(samples),
            "analyzed_samples": len(analyzed_samples),
            "passed_samples": count_passed_samples(analyzed_samples),
            "failed_samples": len(analyzed_samples) - count_passed_samples(analyzed_samples),
            "subsecond_samples": count_subsecond_samples(analyzed_samples),
            "subsecond_fixture_min_duration_count": subsecond_fixture_count,
            **repeat_profile_duration_stats(analyzed_samples),
            **repeat_profile_iou_stats(analyzed_samples),
            "stage_duration_s": repeat_profile_stage_duration_stats(analyzed_samples),
        },
        "fixtures": fixture_summaries,
        "samples": samples,
    }


def summarize_repeat_profile_sample_group(samples: list[dict[str, Any]]) -> dict[str, Any]:
    analyzed_samples = repeat_profile_analyzed_samples(samples)
    return {
        "samples": len(samples),
        "analyzed_samples": len(analyzed_samples),
        "passed_samples": count_passed_samples(analyzed_samples),
        "failed_samples": len(analyzed_samples) - count_passed_samples(analyzed_samples),
        "subsecond_samples": count_subsecond_samples(analyzed_samples),
        **repeat_profile_duration_stats(analyzed_samples),
        **repeat_profile_iou_stats(analyzed_samples),
        "stage_duration_s": repeat_profile_stage_duration_stats(analyzed_samples),
    }


def repeat_profile_analyzed_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in samples if not sample.get("warmup")]


def count_passed_samples(samples: list[dict[str, Any]]) -> int:
    return sum(sample.get("passed") is True for sample in samples)


def count_subsecond_samples(samples: list[dict[str, Any]]) -> int:
    return sum(
        duration is not None and duration < 1.0
        for duration in (parse_report_duration(sample.get("duration_s")) for sample in samples)
    )


def repeat_profile_duration_stats(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    durations = [
        duration
        for duration in (parse_report_duration(sample.get("duration_s")) for sample in samples)
        if duration is not None
    ]
    return duration_distribution_stats(durations)


def repeat_profile_iou_stats(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    ious = [
        iou
        for iou in (parse_report_duration(sample.get("iou")) for sample in samples)
        if iou is not None
    ]
    if not ious:
        return {
            "min_iou": None,
            "average_iou": None,
        }
    return {
        "min_iou": round(min(ious), 6),
        "average_iou": round(float(mean(ious)), 6),
    }


def repeat_profile_stage_duration_stats(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stage_durations: dict[str, list[float]] = {}
    for sample in samples:
        raw_stage_durations = sample.get("stage_elapsed_s")
        if not isinstance(raw_stage_durations, dict):
            continue
        for stage, duration in raw_stage_durations.items():
            if not isinstance(stage, str) or not stage:
                continue
            parsed_duration = parse_report_duration(duration)
            if parsed_duration is None:
                continue
            stage_durations.setdefault(stage, []).append(parsed_duration)
    return {
        stage: {
            "samples": len(durations),
            **duration_distribution_stats(durations),
        }
        for stage, durations in sorted(stage_durations.items())
    }


def duration_distribution_stats(durations: list[float]) -> dict[str, float | None]:
    if not durations:
        return {
            "min_duration_s": None,
            "median_duration_s": None,
            "average_duration_s": None,
            "max_duration_s": None,
        }
    return {
        "min_duration_s": round(min(durations), 6),
        "median_duration_s": round(float(median(durations)), 6),
        "average_duration_s": round(float(mean(durations)), 6),
        "max_duration_s": round(max(durations), 6),
    }


def benchmark_runtime_config() -> dict[str, Any]:
    return {
        "pipeline_version": get_pipeline_version(),
        "ocr": ocr_runtime_config(),
        "generation_env": benchmark_generation_env_config(),
    }


def benchmark_generation_env_config() -> dict[str, str]:
    return {
        name: os.environ.get(name, default)
        for name, default in sorted(BENCHMARK_GENERATION_ENV_DEFAULTS.items())
    }


def load_fixture_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "changed_areas": {}, "fixtures": {}}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Fixture config must be a JSON object: {path}")
    changed_areas = data.setdefault("changed_areas", {})
    if not isinstance(changed_areas, dict):
        raise ValueError(f"Fixture config 'changed_areas' must be an object: {path}")
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
    configured_image_overrides: dict[str, str] = {}
    missing_configured_images: dict[str, str] = {}
    changed_area_config = config.get("changed_areas", {})
    fixture_config = config.get("fixtures", {})
    for image_path in images:
        provider, area_slug, area_name = parse_image_name(image_path)
        slug = f"{area_slug}-{provider}" if provider and area_slug else ""
        reference_path = references.get(slug)
        if reference_path is None:
            missing_references.append(image_path.name)
            continue
        area_override = changed_area_config.get(area_slug, {})
        if not isinstance(area_override, dict):
            raise ValueError(f"Changed-area override for {area_slug} must be an object")
        override = fixture_config.get(slug, {})
        if not isinstance(override, dict):
            raise ValueError(f"Fixture override for {slug} must be an object")
        merged_override = {**area_override, **override}
        configured_image_path = configured_fixture_image_path(
            image_dir=image_dir,
            default_image_path=image_path,
            fixture_slug=slug,
            override=merged_override,
            configured_image_overrides=configured_image_overrides,
            missing_configured_images=missing_configured_images,
        )
        fixtures.append(
            BenchmarkFixture(
                slug=slug,
                provider=provider,
                area=area_name,
                image_path=configured_image_path,
                reference_path=reference_path,
                status=str(merged_override.get("status", "active")),
                note=str(merged_override["note"]) if merged_override.get("note") else None,
            )
        )
    covered = {fixture.slug for fixture in fixtures}
    return fixtures, {
        "polygon_dir": str(polygon_dir),
        "image_dir": str(image_dir),
        "fixture_config": str(config.get("path", "")),
        "matched_images": len(fixtures),
        "missing_reference_images": missing_references,
        "configured_image_overrides": configured_image_overrides,
        "missing_configured_images": missing_configured_images,
        "references_without_images": sorted(set(references) - covered),
    }


def configured_fixture_image_path(
    *,
    image_dir: Path,
    default_image_path: Path,
    fixture_slug: str,
    override: dict[str, Any],
    configured_image_overrides: dict[str, str],
    missing_configured_images: dict[str, str],
) -> Path:
    configured = override.get("current_image")
    if not isinstance(configured, str) or not configured.strip():
        return default_image_path
    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        configured_path = image_dir / configured_path
    configured_path = configured_path.resolve()
    if configured_path.exists() and configured_path.suffix.lower() in IMAGE_SUFFIXES:
        configured_image_overrides[fixture_slug] = str(configured_path)
        return configured_path
    missing_configured_images[fixture_slug] = str(configured_path)
    return default_image_path


def parse_image_name(path: Path) -> tuple[str, str, str]:
    text = normalized_words(path.stem)
    provider = ""
    for candidate in PROVIDERS:
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


def parse_stage_duration_budgets(raw_budgets: list[str]) -> dict[str, float]:
    budgets: dict[str, float] = {}
    for raw in raw_budgets:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"Stage budget must use STAGE=SECONDS: {entry}")
            stage, raw_value = entry.split("=", 1)
            stage = stage.strip().lower()
            if not stage:
                raise ValueError(f"Stage budget is missing a stage name: {entry}")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Stage budget seconds must be numeric: {entry}") from exc
            budgets[stage] = max(0.0, value)
    return dict(sorted(budgets.items()))


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
    catalog_probe_missed: bool = False,
    catalog_probe_miss_low_iou: bool = False,
    neutral_filename_hint: bool = False,
    score_reference: bool = True,
    reference_geometry: Polygon | MultiPolygon | None = None,
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
            catalog_probe_missed=catalog_probe_missed,
            catalog_probe_miss_low_iou=catalog_probe_miss_low_iou,
            neutral_filename_hint=neutral_filename_hint,
            debug_artifacts=debug_artifacts,
            score_reference=score_reference,
            reference_geometry=reference_geometry,
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
    if catalog_probe_missed:
        command.append("--catalog-probe-missed")
    if catalog_probe_miss_low_iou:
        command.append("--catalog-probe-miss-low-iou")
    if neutral_filename_hint:
        command.extend(["--filename-hint", neutral_fixture_filename_hint(fixture)])
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return failed_full_score(fixture, f"timed out after {timeout_seconds}s", duration_s=time.perf_counter() - started)
    duration_s = time.perf_counter() - started
    if completed.returncode != 0:
        summary = parse_cli_summary(completed.stdout)
        error = (
            summary.get("error")
            if isinstance(summary.get("error"), str)
            else completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        )
        event_profile = summary.get("event_profile")
        stage_elapsed_s = event_profile.get("stage_elapsed_s") if isinstance(event_profile, dict) else None
        return failed_full_score(
            fixture,
            error,
            duration_s=duration_s,
            stage_elapsed_s=stage_elapsed_s if isinstance(stage_elapsed_s, dict) else None,
        )

    try:
        output = json.loads(output_path.read_text())
        summary = parse_cli_summary(completed.stdout)
        event_profile = summary.get("event_profile") if isinstance(summary, dict) else None
        stage_elapsed_s = event_profile.get("stage_elapsed_s") if isinstance(event_profile, dict) else None
        properties = output["features"][0].get("properties", {})
        output_geometry = shape(output["features"][0]["geometry"])
        metrics = (
            score_output_geometry(
                output_geometry,
                reference_path=fixture.reference_path,
                min_iou=min_iou,
                reference_geometry=reference_geometry,
            )
            if score_reference
            else None
        )
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
            catalog_shape_iou=first_float(summary.get("catalog_shape_iou"), properties.get("catalog_shape_iou")),
            catalog_area_ratio=first_float(summary.get("catalog_area_ratio"), properties.get("catalog_area_ratio")),
            road_match_score=first_float(summary.get("road_match_score"), properties.get("road_match_score")),
            road_match_elapsed_s=first_float(
                summary.get("road_match_elapsed_s"),
                properties.get("road_match_elapsed_s"),
            ),
            stage_elapsed_s=stage_elapsed_s if isinstance(stage_elapsed_s, dict) else None,
            status=fixture.status,
            note=fixture.note,
        )
    except Exception as exc:
        return failed_full_score(fixture, str(exc), duration_s=duration_s)


def parse_cli_summary(stdout: str) -> dict[str, Any]:
    try:
        summary = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return summary if isinstance(summary, dict) else {}


def score_full_fixture_in_process(
    fixture: BenchmarkFixture,
    *,
    output_path: Path,
    debug_dir: Path | None,
    min_iou: float,
    city_overrides: bool,
    no_catalog: bool,
    debug_artifacts: bool,
    catalog_probe_missed: bool = False,
    catalog_probe_miss_low_iou: bool = False,
    neutral_filename_hint: bool = False,
    score_reference: bool = True,
    reference_geometry: Polygon | MultiPolygon | None = None,
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
                catalog_probe_missed=catalog_probe_missed,
                catalog_probe_miss_low_iou=catalog_probe_miss_low_iou,
                filename_hint=fixture_filename_hint(fixture, neutral=neutral_filename_hint),
                write_mask_artifact=debug_artifacts,
            ),
            progress=progress,
        )
        duration_s = time.perf_counter() - started
        output_geometry = shape(result.geojson["features"][0]["geometry"])
        metrics = (
            score_output_geometry(
                output_geometry,
                reference_path=fixture.reference_path,
                min_iou=min_iou,
                reference_geometry=reference_geometry,
            )
            if score_reference
            else None
        )
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
            catalog_shape_iou=first_float(
                result.summary.get("catalog_shape_iou"),
                properties.get("catalog_shape_iou"),
            ),
            catalog_area_ratio=first_float(
                result.summary.get("catalog_area_ratio"),
                properties.get("catalog_area_ratio"),
            ),
            road_match_score=first_float(
                result.summary.get("road_match_score"),
                properties.get("road_match_score"),
            ),
            road_match_elapsed_s=first_float(
                result.summary.get("road_match_elapsed_s"),
                properties.get("road_match_elapsed_s"),
            ),
            stage_elapsed_s=stage_elapsed_seconds(events),
            status=fixture.status,
            note=fixture.note,
        )
    except Exception as exc:
        return failed_full_score(fixture, str(exc), duration_s=time.perf_counter() - started)


def fixture_filename_hint(fixture: BenchmarkFixture, *, neutral: bool) -> str:
    return neutral_fixture_filename_hint(fixture) if neutral else fixture.image_path.name


def neutral_fixture_filename_hint(fixture: BenchmarkFixture) -> str:
    suffix = fixture.image_path.suffix.lower() or ".png"
    return f"uploaded-map{suffix}"


def score_output_geometry(
    output_geometry,
    reference_path: Path | None = None,
    min_iou: float | None = None,
    *,
    reference_geometry: Polygon | MultiPolygon | None = None,
) -> dict[str, Any]:
    if min_iou is None:
        raise ValueError("min_iou is required")
    predicted = project_geometry(output_geometry)
    if reference_geometry is None:
        if reference_path is None:
            raise ValueError("reference_path or reference_geometry is required")
        reference_geometry = load_reference_geometry(reference_path)
    reference = project_geometry(reference_geometry)
    metrics = compare_geometries(predicted, reference)
    return {
        "passed": metrics["iou"] >= min_iou,
        **metrics,
    }


def catalog_reference_geometry_for_fixture(fixture: BenchmarkFixture) -> Polygon | MultiPolygon | None:
    from .catalog_match import load_catalog_entries

    for entry in load_catalog_entries():
        if entry.slug == fixture.slug and entry.status == "active":
            return entry.geometry
    return None


def fixture_catalog_reference_note(fixture: BenchmarkFixture) -> str:
    base_note = fixture.note.rstrip(".") if fixture.note else "Saved benchmark reference is stale"
    return f"{base_note}. Scored against current catalog geometry instead of the stale saved reference."


def require_current_catalog_evidence(
    score: BenchmarkScore,
    *,
    min_shape_iou: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> BenchmarkScore:
    if not score.passed or not score.catalog_slug:
        return score
    min_shape_iou = max(0.0, float(min_shape_iou))
    min_area_ratio = max(0.0, float(min_area_ratio))
    max_area_ratio = max(0.0, float(max_area_ratio))
    if max_area_ratio < min_area_ratio:
        min_area_ratio, max_area_ratio = max_area_ratio, min_area_ratio

    evidence_issues: list[str] = []
    if score.catalog_shape_iou is None:
        evidence_issues.append("missing catalog_shape_iou")
    elif score.catalog_shape_iou < min_shape_iou:
        evidence_issues.append(
            f"catalog_shape_iou {score.catalog_shape_iou:.6f} < required {min_shape_iou:.6f}"
        )
    if score.catalog_area_ratio is None:
        evidence_issues.append("missing catalog_area_ratio")
    elif not (min_area_ratio <= score.catalog_area_ratio <= max_area_ratio):
        evidence_issues.append(
            f"catalog_area_ratio {score.catalog_area_ratio:.6f} outside "
            f"{min_area_ratio:.6f}-{max_area_ratio:.6f}"
        )

    if not evidence_issues:
        return score
    return replace(
        score,
        passed=False,
        error=(
            f"catalog_slug={score.catalog_slug} matched current catalog reference, "
            "but source-image catalog evidence is weak: "
            + "; ".join(evidence_issues)
        ),
    )


def first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def failed_full_score(
    fixture: BenchmarkFixture,
    error: str,
    *,
    duration_s: float | None = None,
    stage_elapsed_s: dict[str, float] | None = None,
) -> BenchmarkScore:
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
        stage_elapsed_s=stage_elapsed_s,
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


def summarize_stage_durations(scores: list[BenchmarkScore]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for score in scores:
        if not score.stage_elapsed_s:
            continue
        for stage, duration in score.stage_elapsed_s.items():
            if not isinstance(stage, str) or not stage:
                continue
            parsed_duration = parse_report_duration(duration)
            if parsed_duration is None:
                continue
            totals[stage] = totals.get(stage, 0.0) + parsed_duration
    return {stage: round(total, 6) for stage, total in sorted(totals.items())}


def summarize_road_match_elapsed(scores: list[BenchmarkScore]) -> float:
    total = 0.0
    for score in scores:
        parsed_duration = parse_report_duration(score.road_match_elapsed_s)
        if parsed_duration is None:
            continue
        total += parsed_duration
    return round(total, 6)


def combine_stage_durations(*summaries: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for summary in summaries:
        for stage, duration in summary.items():
            parsed_duration = parse_report_duration(duration)
            if parsed_duration is None:
                continue
            totals[stage] = totals.get(stage, 0.0) + parsed_duration
    return {stage: round(total, 6) for stage, total in sorted(totals.items())}


def report_summary_stage_durations(report: dict[str, Any], key: str) -> dict[str, float]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    raw_stage_durations = summary.get(key)
    if not isinstance(raw_stage_durations, dict):
        return {}
    stage_durations: dict[str, float] = {}
    for stage, duration in raw_stage_durations.items():
        if not isinstance(stage, str) or not stage:
            continue
        parsed_duration = parse_report_duration(duration)
        if parsed_duration is None:
            continue
        stage_durations[stage] = round(parsed_duration, 6)
    return dict(sorted(stage_durations.items()))


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
    max_evaluated_duration_increase_ratio: float | None = None,
    max_evaluated_duration_increase_s: float = 0.0,
    max_evaluated_stage_duration_increase_ratio: float | None = None,
    max_evaluated_stage_duration_increase_s: float = 0.0,
    max_evaluated_road_match_increase_ratio: float | None = None,
    max_evaluated_road_match_increase_s: float = 0.0,
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
    compared_iou_pairs: list[tuple[float, float]] = []
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
        baseline_iou = float(baseline_score["iou"])
        candidate_iou = float(candidate_score["iou"])
        compared_iou_pairs.append((baseline_iou, candidate_iou))
        drop = baseline_iou - candidate_iou
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

    baseline_mean = (
        mean(pair[0] for pair in compared_iou_pairs) if compared_iou_pairs else 0.0
    )
    candidate_mean = (
        mean(pair[1] for pair in compared_iou_pairs) if compared_iou_pairs else 0.0
    )
    mean_drop = baseline_mean - candidate_mean
    mean_tolerance = max(0.0, float(max_mean_iou_drop))
    if mean_drop > mean_tolerance:
        issues.append(
            {
                "kind": "average_iou_drop",
                "baseline_average_iou": round(baseline_mean, 6),
                "candidate_average_iou": round(candidate_mean, 6),
                "drop": round(mean_drop, 6),
                "average_iou_scope": "compared_fixtures",
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
    evaluated_duration_tolerance = (
        None
        if max_evaluated_duration_increase_ratio is None
        else max(0.0, float(max_evaluated_duration_increase_ratio))
    )
    evaluated_duration_tolerance_s = max(0.0, float(max_evaluated_duration_increase_s))
    if evaluated_duration_tolerance is not None:
        baseline_evaluated = report_evaluated_duration(baseline_report)
        candidate_evaluated = report_evaluated_duration(report)
        if baseline_evaluated is not None and candidate_evaluated is not None and baseline_evaluated > 0:
            evaluated_increase_s = candidate_evaluated - baseline_evaluated
            evaluated_increase_ratio = candidate_evaluated / baseline_evaluated - 1.0
            if (
                evaluated_increase_s > evaluated_duration_tolerance_s
                and evaluated_increase_ratio > evaluated_duration_tolerance
            ):
                issues.append(
                    {
                        "kind": "evaluated_duration_increase",
                        "baseline_evaluated_duration_s": round(baseline_evaluated, 6),
                        "candidate_evaluated_duration_s": round(candidate_evaluated, 6),
                        "increase_s": round(evaluated_increase_s, 6),
                        "increase_ratio": round(evaluated_increase_ratio, 6),
                    }
                )
    evaluated_stage_duration_tolerance = (
        None
        if max_evaluated_stage_duration_increase_ratio is None
        else max(0.0, float(max_evaluated_stage_duration_increase_ratio))
    )
    evaluated_stage_duration_tolerance_s = max(
        0.0,
        float(max_evaluated_stage_duration_increase_s),
    )
    compared_evaluated_stage_durations = 0
    if evaluated_stage_duration_tolerance is not None:
        baseline_stage_durations = report_summary_stage_durations(
            baseline_report,
            "evaluated_stage_duration_s",
        )
        candidate_stage_durations = report_summary_stage_durations(
            report,
            "evaluated_stage_duration_s",
        )
        for stage, baseline_stage_duration in baseline_stage_durations.items():
            candidate_stage_duration = candidate_stage_durations.get(stage)
            if candidate_stage_duration is None or baseline_stage_duration <= 0:
                continue
            compared_evaluated_stage_durations += 1
            stage_increase_s = candidate_stage_duration - baseline_stage_duration
            stage_increase_ratio = candidate_stage_duration / baseline_stage_duration - 1.0
            if (
                stage_increase_s > evaluated_stage_duration_tolerance_s
                and stage_increase_ratio > evaluated_stage_duration_tolerance
            ):
                issues.append(
                    {
                        "stage": stage,
                        "kind": "evaluated_stage_duration_increase",
                        "baseline_stage_duration_s": round(baseline_stage_duration, 6),
                        "candidate_stage_duration_s": round(candidate_stage_duration, 6),
                        "increase_s": round(stage_increase_s, 6),
                        "increase_ratio": round(stage_increase_ratio, 6),
                    }
                )

    evaluated_road_match_tolerance = (
        None
        if max_evaluated_road_match_increase_ratio is None
        else max(0.0, float(max_evaluated_road_match_increase_ratio))
    )
    evaluated_road_match_tolerance_s = max(0.0, float(max_evaluated_road_match_increase_s))
    if evaluated_road_match_tolerance is not None:
        baseline_road_match = report_evaluated_road_match_elapsed(baseline_report)
        candidate_road_match = report_evaluated_road_match_elapsed(report)
        if baseline_road_match is not None and candidate_road_match is not None and baseline_road_match > 0:
            road_match_increase_s = candidate_road_match - baseline_road_match
            road_match_increase_ratio = candidate_road_match / baseline_road_match - 1.0
            if (
                road_match_increase_s > evaluated_road_match_tolerance_s
                and road_match_increase_ratio > evaluated_road_match_tolerance
            ):
                issues.append(
                    {
                        "kind": "evaluated_road_match_elapsed_increase",
                        "baseline_evaluated_road_match_elapsed_s": round(baseline_road_match, 6),
                        "candidate_evaluated_road_match_elapsed_s": round(candidate_road_match, 6),
                        "increase_s": round(road_match_increase_s, 6),
                        "increase_ratio": round(road_match_increase_ratio, 6),
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
        "max_evaluated_duration_increase_ratio": evaluated_duration_tolerance,
        "max_evaluated_duration_increase_s": evaluated_duration_tolerance_s,
        "max_evaluated_stage_duration_increase_ratio": evaluated_stage_duration_tolerance,
        "max_evaluated_stage_duration_increase_s": evaluated_stage_duration_tolerance_s,
        "max_evaluated_road_match_increase_ratio": evaluated_road_match_tolerance,
        "max_evaluated_road_match_increase_s": evaluated_road_match_tolerance_s,
        "compared_fixtures": len(baseline_scores),
        "compared_iou_fixtures": len(compared_iou_pairs),
        "compared_evaluated_stage_durations": compared_evaluated_stage_durations,
        "baseline_average_iou": round(baseline_mean, 6),
        "candidate_average_iou": round(candidate_mean, 6),
        "average_iou_scope": "compared_fixtures",
        "issues": issues,
    }


def check_report_latency_budgets(
    report: dict[str, Any],
    *,
    max_duration_s: float | None = None,
    max_total_duration_s: float | None = None,
    max_evaluated_duration_s: float | None = None,
    max_evaluated_stage_duration_s: dict[str, float] | None = None,
    max_evaluated_road_match_s: float | None = None,
    max_repeat_profile_duration_s: float | None = None,
    max_repeat_profile_median_duration_s: float | None = None,
    max_repeat_profile_stage_duration_s: dict[str, float] | None = None,
    min_repeat_profile_pass_ratio: float | None = None,
    min_repeat_profile_subsecond_ratio: float | None = None,
) -> dict[str, Any]:
    duration_budget = None if max_duration_s is None else max(0.0, float(max_duration_s))
    total_budget = None if max_total_duration_s is None else max(0.0, float(max_total_duration_s))
    evaluated_budget = (
        None if max_evaluated_duration_s is None else max(0.0, float(max_evaluated_duration_s))
    )
    evaluated_stage_budgets = {
        stage: max(0.0, float(duration))
        for stage, duration in (max_evaluated_stage_duration_s or {}).items()
        if stage
    }
    evaluated_road_match_budget = (
        None
        if max_evaluated_road_match_s is None
        else max(0.0, float(max_evaluated_road_match_s))
    )
    repeat_profile_duration_budget = (
        None
        if max_repeat_profile_duration_s is None
        else max(0.0, float(max_repeat_profile_duration_s))
    )
    repeat_profile_median_duration_budget = (
        None
        if max_repeat_profile_median_duration_s is None
        else max(0.0, float(max_repeat_profile_median_duration_s))
    )
    repeat_profile_stage_budgets = {
        stage: max(0.0, float(duration))
        for stage, duration in (max_repeat_profile_stage_duration_s or {}).items()
        if stage
    }
    repeat_profile_pass_ratio_budget = (
        None
        if min_repeat_profile_pass_ratio is None
        else min(1.0, max(0.0, float(min_repeat_profile_pass_ratio)))
    )
    repeat_profile_subsecond_ratio_budget = (
        None
        if min_repeat_profile_subsecond_ratio is None
        else min(1.0, max(0.0, float(min_repeat_profile_subsecond_ratio)))
    )
    active_total_duration = parse_report_duration(report.get("summary", {}).get("total_duration_s"))
    smoke_total_duration = parse_report_duration(report.get("summary", {}).get("smoked_skipped_duration_s"))
    evaluated_duration = report_evaluated_duration(report)
    evaluated_stage_durations = report_summary_stage_durations(report, "evaluated_stage_duration_s")
    evaluated_road_match = report_evaluated_road_match_elapsed(report)
    repeat_profile_summary = report_repeat_profile_summary(report)
    repeat_profile_analyzed_samples = repeat_profile_analyzed_sample_count(repeat_profile_summary)
    repeat_profile_max_duration = report_repeat_profile_duration(repeat_profile_summary, "max_duration_s")
    repeat_profile_median_duration = report_repeat_profile_duration(
        repeat_profile_summary,
        "median_duration_s",
    )
    repeat_profile_stage_durations = report_repeat_profile_stage_durations(repeat_profile_summary)
    repeat_profile_pass_ratio = report_repeat_profile_sample_ratio(
        repeat_profile_summary,
        "passed_samples",
    )
    repeat_profile_subsecond_ratio = report_repeat_profile_sample_ratio(
        repeat_profile_summary,
        "subsecond_samples",
    )
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
        if active_total_duration is not None and active_total_duration > total_budget:
            issues.append(
                {
                    "kind": "total_duration_budget_exceeded",
                    "total_duration_s": round(active_total_duration, 6),
                    "max_total_duration_s": total_budget,
                    "excess_s": round(active_total_duration - total_budget, 6),
                }
            )
    if evaluated_budget is not None:
        if evaluated_duration is not None and evaluated_duration > evaluated_budget:
            issues.append(
                {
                    "kind": "evaluated_duration_budget_exceeded",
                    "evaluated_duration_s": round(evaluated_duration, 6),
                    "max_evaluated_duration_s": evaluated_budget,
                    "excess_s": round(evaluated_duration - evaluated_budget, 6),
                }
            )
    for stage, stage_budget in sorted(evaluated_stage_budgets.items()):
        stage_duration = evaluated_stage_durations.get(stage)
        if stage_duration is None:
            issues.append(
                {
                    "stage": stage,
                    "kind": "evaluated_stage_duration_missing",
                    "max_evaluated_stage_duration_s": stage_budget,
                }
            )
            continue
        if stage_duration > stage_budget:
            issues.append(
                {
                    "stage": stage,
                    "kind": "evaluated_stage_duration_budget_exceeded",
                    "evaluated_stage_duration_s": round(stage_duration, 6),
                    "max_evaluated_stage_duration_s": stage_budget,
                    "excess_s": round(stage_duration - stage_budget, 6),
                }
            )
    if evaluated_road_match_budget is not None:
        if evaluated_road_match is not None and evaluated_road_match > evaluated_road_match_budget:
            issues.append(
                {
                    "kind": "evaluated_road_match_budget_exceeded",
                    "evaluated_road_match_elapsed_s": round(evaluated_road_match, 6),
                    "max_evaluated_road_match_s": evaluated_road_match_budget,
                    "excess_s": round(evaluated_road_match - evaluated_road_match_budget, 6),
                }
            )
    repeat_profile_budget_requested = any(
        budget is not None
        for budget in (
            repeat_profile_duration_budget,
            repeat_profile_median_duration_budget,
            repeat_profile_stage_budgets if repeat_profile_stage_budgets else None,
            repeat_profile_pass_ratio_budget,
            repeat_profile_subsecond_ratio_budget,
        )
    )
    if repeat_profile_budget_requested and repeat_profile_summary is None:
        issues.append({"kind": "repeat_profile_missing"})
    elif repeat_profile_budget_requested and repeat_profile_analyzed_samples <= 0:
        issues.append({"kind": "repeat_profile_analyzed_samples_missing"})
    has_repeat_profile_samples = repeat_profile_summary is not None and repeat_profile_analyzed_samples > 0
    if repeat_profile_duration_budget is not None and has_repeat_profile_samples:
        if repeat_profile_max_duration is None:
            issues.append(
                {
                    "kind": "repeat_profile_duration_missing",
                    "max_repeat_profile_duration_s": repeat_profile_duration_budget,
                }
            )
        elif repeat_profile_max_duration > repeat_profile_duration_budget:
            issues.append(
                {
                    "kind": "repeat_profile_duration_budget_exceeded",
                    "repeat_profile_max_duration_s": round(repeat_profile_max_duration, 6),
                    "max_repeat_profile_duration_s": repeat_profile_duration_budget,
                    "excess_s": round(repeat_profile_max_duration - repeat_profile_duration_budget, 6),
                }
            )
    if repeat_profile_median_duration_budget is not None and has_repeat_profile_samples:
        if repeat_profile_median_duration is None:
            issues.append(
                {
                    "kind": "repeat_profile_median_duration_missing",
                    "max_repeat_profile_median_duration_s": repeat_profile_median_duration_budget,
                }
            )
        elif repeat_profile_median_duration > repeat_profile_median_duration_budget:
            issues.append(
                {
                    "kind": "repeat_profile_median_duration_budget_exceeded",
                    "repeat_profile_median_duration_s": round(repeat_profile_median_duration, 6),
                    "max_repeat_profile_median_duration_s": repeat_profile_median_duration_budget,
                    "excess_s": round(
                        repeat_profile_median_duration - repeat_profile_median_duration_budget,
                        6,
                    ),
                }
            )
    for stage, stage_budget in sorted(repeat_profile_stage_budgets.items()):
        if not has_repeat_profile_samples:
            continue
        stage_duration = repeat_profile_stage_durations.get(stage)
        if stage_duration is None:
            issues.append(
                {
                    "stage": stage,
                    "kind": "repeat_profile_stage_duration_missing",
                    "max_repeat_profile_stage_duration_s": stage_budget,
                }
            )
            continue
        if stage_duration > stage_budget:
            issues.append(
                {
                    "stage": stage,
                    "kind": "repeat_profile_stage_duration_budget_exceeded",
                    "repeat_profile_stage_duration_s": round(stage_duration, 6),
                    "max_repeat_profile_stage_duration_s": stage_budget,
                    "excess_s": round(stage_duration - stage_budget, 6),
                }
            )
    if repeat_profile_pass_ratio_budget is not None and has_repeat_profile_samples:
        if repeat_profile_pass_ratio is None:
            issues.append(
                {
                    "kind": "repeat_profile_pass_ratio_missing",
                    "min_repeat_profile_pass_ratio": repeat_profile_pass_ratio_budget,
                }
            )
        elif repeat_profile_pass_ratio < repeat_profile_pass_ratio_budget:
            issues.append(
                {
                    "kind": "repeat_profile_pass_ratio_below_min",
                    "repeat_profile_pass_ratio": round(repeat_profile_pass_ratio, 6),
                    "min_repeat_profile_pass_ratio": repeat_profile_pass_ratio_budget,
                    "shortfall": round(repeat_profile_pass_ratio_budget - repeat_profile_pass_ratio, 6),
                }
            )
    if repeat_profile_subsecond_ratio_budget is not None and has_repeat_profile_samples:
        if repeat_profile_subsecond_ratio is None:
            issues.append(
                {
                    "kind": "repeat_profile_subsecond_ratio_missing",
                    "min_repeat_profile_subsecond_ratio": repeat_profile_subsecond_ratio_budget,
                }
            )
        elif repeat_profile_subsecond_ratio < repeat_profile_subsecond_ratio_budget:
            issues.append(
                {
                    "kind": "repeat_profile_subsecond_ratio_below_min",
                    "repeat_profile_subsecond_ratio": round(repeat_profile_subsecond_ratio, 6),
                    "min_repeat_profile_subsecond_ratio": repeat_profile_subsecond_ratio_budget,
                    "shortfall": round(
                        repeat_profile_subsecond_ratio_budget - repeat_profile_subsecond_ratio,
                        6,
                    ),
                }
            )
    return {
        "passed": not issues,
        "max_duration_s": duration_budget,
        "max_total_duration_s": total_budget,
        "max_evaluated_duration_s": evaluated_budget,
        "max_evaluated_stage_duration_s": evaluated_stage_budgets,
        "max_evaluated_road_match_s": evaluated_road_match_budget,
        "max_repeat_profile_duration_s": repeat_profile_duration_budget,
        "max_repeat_profile_median_duration_s": repeat_profile_median_duration_budget,
        "max_repeat_profile_stage_duration_s": repeat_profile_stage_budgets,
        "min_repeat_profile_pass_ratio": repeat_profile_pass_ratio_budget,
        "min_repeat_profile_subsecond_ratio": repeat_profile_subsecond_ratio_budget,
        "active_total_duration_s": round(active_total_duration, 6) if active_total_duration is not None else None,
        "smoked_skipped_duration_s": round(smoke_total_duration, 6) if smoke_total_duration is not None else None,
        "evaluated_duration_s": round(evaluated_duration, 6) if evaluated_duration is not None else None,
        "evaluated_stage_duration_s": evaluated_stage_durations,
        "evaluated_road_match_elapsed_s": (
            round(evaluated_road_match, 6) if evaluated_road_match is not None else None
        ),
        "repeat_profile_analyzed_samples": repeat_profile_analyzed_samples,
        "repeat_profile_max_duration_s": (
            round(repeat_profile_max_duration, 6)
            if repeat_profile_max_duration is not None
            else None
        ),
        "repeat_profile_median_duration_s": (
            round(repeat_profile_median_duration, 6)
            if repeat_profile_median_duration is not None
            else None
        ),
        "repeat_profile_stage_duration_s": repeat_profile_stage_durations,
        "repeat_profile_pass_ratio": (
            round(repeat_profile_pass_ratio, 6)
            if repeat_profile_pass_ratio is not None
            else None
        ),
        "repeat_profile_subsecond_ratio": (
            round(repeat_profile_subsecond_ratio, 6)
            if repeat_profile_subsecond_ratio is not None
            else None
        ),
        "issues": issues,
    }


def report_repeat_profile_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    repeat_profile = report.get("repeat_profile")
    if not isinstance(repeat_profile, dict):
        return None
    summary = repeat_profile.get("summary")
    return summary if isinstance(summary, dict) else None


def repeat_profile_analyzed_sample_count(summary: dict[str, Any] | None) -> int:
    if summary is None:
        return 0
    value = summary.get("analyzed_samples")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def report_repeat_profile_duration(summary: dict[str, Any] | None, key: str) -> float | None:
    if summary is None:
        return None
    return parse_report_duration(summary.get(key))


def report_repeat_profile_stage_durations(summary: dict[str, Any] | None) -> dict[str, float]:
    if summary is None:
        return {}
    raw_stage_durations = summary.get("stage_duration_s")
    if not isinstance(raw_stage_durations, dict):
        return {}
    stage_durations: dict[str, float] = {}
    for stage, stats in raw_stage_durations.items():
        if not isinstance(stage, str) or not stage or not isinstance(stats, dict):
            continue
        duration = parse_report_duration(stats.get("max_duration_s"))
        if duration is None:
            continue
        stage_durations[stage] = round(duration, 6)
    return dict(sorted(stage_durations.items()))


def report_repeat_profile_sample_ratio(summary: dict[str, Any] | None, key: str) -> float | None:
    analyzed_samples = repeat_profile_analyzed_sample_count(summary)
    if summary is None or analyzed_samples <= 0:
        return None
    value = summary.get(key)
    try:
        count = max(0, int(value))
    except (TypeError, ValueError):
        return None
    return min(1.0, count / analyzed_samples)


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


def report_evaluated_duration(report: dict[str, Any]) -> float | None:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return None
    evaluated_duration = parse_report_duration(summary.get("evaluated_duration_s"))
    if evaluated_duration is not None:
        return evaluated_duration
    active_total_duration = parse_report_duration(summary.get("total_duration_s"))
    smoke_total_duration = parse_report_duration(summary.get("smoked_skipped_duration_s"))
    if active_total_duration is None and smoke_total_duration is None:
        return None
    return (active_total_duration or 0.0) + (smoke_total_duration or 0.0)


def report_evaluated_road_match_elapsed(report: dict[str, Any]) -> float | None:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return None
    evaluated_elapsed = parse_report_duration(summary.get("evaluated_road_match_elapsed_s"))
    if evaluated_elapsed is not None:
        return evaluated_elapsed
    active_elapsed = parse_report_duration(summary.get("active_road_match_elapsed_s"))
    smoke_elapsed = parse_report_duration(summary.get("smoked_skipped_road_match_elapsed_s"))
    if active_elapsed is None and smoke_elapsed is None:
        return None
    return (active_elapsed or 0.0) + (smoke_elapsed or 0.0)


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
    duration_text = f"active total {format_duration(summary.get('total_duration_s'))}"
    if summary.get("smoked_skipped_fixtures"):
        duration_text = (
            f"{duration_text}, evaluated total {format_duration(summary.get('evaluated_duration_s'))}"
        )
    print(
        f"{status} {report['mode']} benchmark: "
        f"{summary['passed_fixtures']}/{summary['scored_fixtures']} scored fixtures, "
        f"{skipped_text}, "
        f"avg IoU {summary['average_iou']:.3f}, min IoU {summary['min_iou']:.3f}, "
        f"{duration_text}"
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
                label = (
                    "compared-fixture average IoU"
                    if issue.get("average_iou_scope") == "compared_fixtures"
                    else "average IoU"
                )
                print(
                    f"       {label} {issue['baseline_average_iou']:.6f} -> "
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
            elif issue["kind"] == "evaluated_duration_increase":
                print(
                    f"       evaluated duration {issue['baseline_evaluated_duration_s']:.3f}s -> "
                    f"{issue['candidate_evaluated_duration_s']:.3f}s "
                    f"(+{issue['increase_s']:.3f}s, ratio {issue['increase_ratio']:.3f})"
                )
            elif issue["kind"] == "evaluated_stage_duration_increase":
                print(
                    f"       {issue['stage']}: evaluated stage duration "
                    f"{issue['baseline_stage_duration_s']:.3f}s -> "
                    f"{issue['candidate_stage_duration_s']:.3f}s "
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
            elif issue["kind"] == "evaluated_duration_budget_exceeded":
                print(
                    f"       evaluated duration {issue['evaluated_duration_s']:.3f}s "
                    f"> budget {issue['max_evaluated_duration_s']:.3f}s "
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
