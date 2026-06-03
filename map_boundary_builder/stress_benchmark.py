from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from .cli import stage_elapsed_seconds
from .extract import EXTRACTION_CACHE_ENV
from .georef_transform import lonlat_to_mercator
from .ocr import (
    collect_rapidocr_profiles,
    summarize_rapidocr_profile_events,
    summarize_rapidocr_profile_summaries,
)
from .pipeline_version import get_pipeline_version, pipeline_version_dependency_versions
from .runner import BoundaryBuildOptions, RUNNER_OCR_CACHE_ENV, build_boundary
from .runtime_warmup import prewarm_generation_runtime
from .runtime_config import generation_env_config, ocr_runtime_config


DEFAULT_MANIFEST = Path("benchmarks/real-screenshot-stress.json")
DEFAULT_OUT_DIR = Path("out/real-screenshot-stress")
GENERIC_FILENAME_HINT = "upload.png"
REAL_SCREENSHOT_HARD_GATE_PRESET_NAME = "real-screenshot-hard-gate"
REAL_SCREENSHOT_HARD_GATE_PRESET_VERSION = 11
FOCUSED_REAL_SCREENSHOT_GATE_PRESET_NAME = "focused-real-screenshot-gate"
FOCUSED_REAL_SCREENSHOT_GATE_PRESET_VERSION = 10
OCR_ENGINE_STAGE_MAX_KEYS = ("input_s", "det_elapsed_s", "rec_elapsed_s", "total_s")
BASELINE_REPEAT_OCR_STAGE_DELTA_DISPLAY = (
    ("input_s", "input_p95"),
    ("det_elapsed_s", "det_p95"),
    ("rec_elapsed_s", "rec_p95"),
    ("total_s", "total_p95"),
)
BASELINE_PRIMARY_OCR_STAGE_DELTA_DISPLAY = (
    ("input_s", "input"),
    ("det_elapsed_s", "det"),
    ("rec_elapsed_s", "rec"),
)
OCR_ENGINE_PRIMARY_DOMINANT_STAGE_FIELDS = (
    ("input", "input_s"),
    ("det", "det_elapsed_s"),
    ("rec", "rec_elapsed_s"),
)
OCR_ENGINE_REPEAT_P95_DOMINANT_STAGE_FIELDS = (
    ("input", "p95_input_s"),
    ("det", "p95_det_elapsed_s"),
    ("rec", "p95_rec_elapsed_s"),
)
PIPELINE_STAGE_DISPLAY_ORDER = ("extract", "ocr", "georeference")
OCR_ENGINE_DETAIL_CONTEXT_KEYS = (
    "input_kind",
    "input_shape",
    "detector_limit",
    "detector_limit_type",
    "recognition_profile",
    "rec_batch_num",
    "min_text_area",
    "classifier_retry",
    "header_region_filter",
)
OCR_ENGINE_STAGE_METRIC_ALIASES = {
    "total_elapsed_s": "total_s",
}
PREWARM_STAGE_DURATION_KEYS = ("catalog_s", "seed_s", "extraction_s", "rapidocr_s", "total_s")
PREWARM_STAGE_METRIC_ALIASES = {
    "total_elapsed_s": "total_s",
}
OCR_ENGINE_BOX_AREA_KEYS = (
    "raw_box_area_min",
    "raw_box_area_p25",
    "raw_box_area_p50",
    "raw_box_area_p75",
    "raw_box_area_p90",
    "raw_box_area_max",
    "raw_box_area_lt_500_count",
    "raw_box_area_lt_900_count",
    "raw_box_area_lt_1300_count",
    "raw_box_area_lt_1500_count",
    "selected_box_area_min",
    "selected_box_area_p25",
    "selected_box_area_p50",
    "selected_box_area_p75",
    "selected_box_area_p90",
    "selected_box_area_max",
    "selected_box_area_lt_500_count",
    "selected_box_area_lt_900_count",
    "selected_box_area_lt_1300_count",
    "selected_box_area_lt_1500_count",
)
OCR_ENGINE_CONFIDENCE_KEYS = (
    "label_confidence_min",
    "label_confidence_p25",
    "label_confidence_p50",
    "label_confidence_p75",
    "label_confidence_p90",
    "label_confidence_max",
    "label_confidence_lt_50_count",
    "label_confidence_lt_70_count",
    "label_confidence_lt_80_count",
    "label_confidence_lt_90_count",
)
OCR_ENGINE_COUNT_METRIC_KEYS = (
    "raw_box_count",
    "selected_box_count",
    "result_count",
    "label_count",
    "useful_label_count",
    *tuple(key for key in OCR_ENGINE_BOX_AREA_KEYS if key.endswith("_count")),
    *tuple(key for key in OCR_ENGINE_CONFIDENCE_KEYS if key.endswith("_count")),
)
OCR_ENGINE_COUNT_DISPLAY_KEYS = (
    "raw_box_count",
    "selected_box_count",
    "result_count",
    "label_count",
    "label_confidence_lt_50_count",
    "label_confidence_lt_70_count",
    "label_confidence_lt_80_count",
    "label_confidence_lt_90_count",
)
OCR_ENGINE_COUNT_DISPLAY_LABELS = {
    "label_confidence_lt_50_count": "conf_lt50",
    "label_confidence_lt_70_count": "conf_lt70",
    "label_confidence_lt_80_count": "conf_lt80",
    "label_confidence_lt_90_count": "conf_lt90",
}
REAL_SCREENSHOT_HARD_GATE_REPEAT_COUNT_BUDGET = (
    "raw_box_count=50,"
    "selected_box_count=30,"
    "result_count=29,"
    "label_count=29,"
    "label_confidence_lt_90_count=3"
)
REAL_SCREENSHOT_HARD_GATE_BASELINE_REGRESSION_S = 0.25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map-boundary-stress",
        description="Run real screenshot stress probes through the map-boundary-builder CLI.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Stress manifest JSON path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for GeoJSON outputs and summary.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only a manifest slug. May be repeated.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Per-case subprocess timeout.",
    )
    parser.add_argument(
        "--write-debug",
        action="store_true",
        help="Write per-case debug masks, overlays, and runner summaries.",
    )
    parser.add_argument(
        "--fail-on-unexpected",
        action="store_true",
        help="Exit non-zero when any non-missing case violates manifest expectations.",
    )
    parser.add_argument(
        "--fail-on-repeat-signature-drift",
        action="store_true",
        help="Exit non-zero when analyzed repeat-profile samples produce different output signatures.",
    )
    parser.add_argument(
        "--compare-baseline-report",
        default=None,
        help="Compare this run against an earlier stress-summary.json and save behavior/latency deltas.",
    )
    parser.add_argument(
        "--fail-on-baseline-signature-drift",
        action="store_true",
        help="Exit non-zero when --compare-baseline-report finds output signature changes.",
    )
    parser.add_argument(
        "--fail-on-baseline-config-drift",
        action="store_true",
        help="Exit non-zero when --compare-baseline-report has incompatible timing/cache settings.",
    )
    parser.add_argument(
        "--fail-on-baseline-coverage-gap",
        action="store_true",
        help="Exit non-zero when --compare-baseline-report did not compare every selected row.",
    )
    parser.add_argument(
        "--fail-on-baseline-expectation-regression",
        action="store_true",
        help="Exit non-zero when --compare-baseline-report shows rows that passed baseline expectations but fail candidate expectations.",
    )
    parser.add_argument(
        "--max-baseline-total-regression-s",
        type=float,
        default=None,
        help=(
            "Exit non-zero when any compared primary row is more than this many "
            "seconds slower than --compare-baseline-report."
        ),
    )
    parser.add_argument(
        "--max-baseline-repeat-p95-regression-s",
        type=float,
        default=None,
        help=(
            "Exit non-zero when analyzed repeat-profile p95 is more than this many "
            "seconds slower than --compare-baseline-report."
        ),
    )
    parser.add_argument(
        "--max-baseline-ocr-total-regression-s",
        type=float,
        default=None,
        help=(
            "Exit non-zero when any compared primary row has OCR engine total "
            "runtime more than this many seconds slower than --compare-baseline-report."
        ),
    )
    parser.add_argument(
        "--max-baseline-repeat-ocr-total-p95-regression-s",
        type=float,
        default=None,
        help=(
            "Exit non-zero when analyzed repeat-profile OCR engine total p95 is "
            "more than this many seconds slower than --compare-baseline-report."
        ),
    )
    parser.add_argument(
        "--real-screenshot-hard-gate",
        action="store_true",
        help=(
            "Apply the current full real-screenshot production-warm hard gate preset: "
            "in-process execution, OCR profiling, prewarm, repeat signatures, latency/OCR "
            "budgets, and manifest OCR contract coverage budgets."
        ),
    )
    parser.add_argument(
        "--focused-real-screenshot-gate",
        action="store_true",
        help=(
            "Apply the real-screenshot production-warm gate settings to selected rows. "
            "Unlike --real-screenshot-hard-gate, this allows --only and skips full-manifest "
            "OCR contract coverage budgets."
        ),
    )
    parser.add_argument(
        "--profile-ocr-engine",
        action="store_true",
        help="Ask the CLI to include RapidOCR detector/recognizer timing details.",
    )
    parser.add_argument(
        "--disable-ocr-cache",
        action="store_true",
        help="Disable the runner OCR cache so repeat profiles keep paying fresh OCR cost.",
    )
    parser.add_argument(
        "--disable-extraction-cache",
        action="store_true",
        help="Disable extraction memory/disk cache reads and writes for fresh-image timing probes.",
    )
    parser.add_argument(
        "--execution",
        choices=("subprocess", "in-process"),
        default="subprocess",
        help=(
            "subprocess preserves the historical CLI stress gate; in-process measures warm "
            "production-instance generation without interpreter startup."
        ),
    )
    parser.add_argument(
        "--prewarm-runtime",
        action="store_true",
        help=(
            "Run the same generation prewarm used by /api/health?warm=ocr before primary "
            "stress rows. Useful for production-warm latency gates."
        ),
    )
    parser.add_argument(
        "--repeat-profile-runs",
        type=int,
        default=0,
        help="Rerun each selected stress case this many additional times and summarize latency variance.",
    )
    parser.add_argument(
        "--repeat-profile-warmups",
        type=int,
        default=0,
        help="Number of repeat-profile samples per stress case to exclude from aggregate repeat statistics.",
    )
    parser.add_argument(
        "--max-total-elapsed-s",
        type=float,
        default=None,
        help=(
            "Fail when any primary row or analyzed repeat-profile sample exceeds this total elapsed budget."
        ),
    )
    parser.add_argument(
        "--max-repeat-profile-p95-duration-s",
        type=float,
        default=None,
        help="Fail when analyzed repeat-profile total p95 exceeds this duration budget.",
    )
    parser.add_argument(
        "--max-prewarm-runtime-s",
        type=float,
        default=None,
        help="Fail when generation runtime prewarm total_s is missing or exceeds this duration budget.",
    )
    parser.add_argument(
        "--max-prewarm-stage-s",
        action="append",
        default=[],
        metavar="METRIC=SECONDS",
        help=(
            "Fail when a generation prewarm stage metric is missing or exceeds this duration budget. "
            "Repeat or comma-separate entries such as rapidocr_s=1.2,extraction_s=0.05,total_s=1.5."
        ),
    )
    parser.add_argument(
        "--max-repeat-ocr-engine-p95-duration-s",
        action="append",
        default=[],
        metavar="METRIC=SECONDS",
        help=(
            "Fail when a profiled repeat OCR engine metric p95 exceeds this budget. "
            "Repeat or comma-separate entries such as det_elapsed_s=0.3,rec_elapsed_s=0.6,total_s=0.9."
        ),
    )
    parser.add_argument(
        "--max-ocr-engine-duration-s",
        action="append",
        default=[],
        metavar="METRIC=SECONDS",
        help=(
            "Fail when a profiled primary OCR engine metric exceeds this budget. "
            "Repeat or comma-separate entries such as det_elapsed_s=0.3,rec_elapsed_s=0.6,total_s=0.9."
        ),
    )
    parser.add_argument(
        "--max-repeat-ocr-engine-p95-count",
        action="append",
        default=[],
        metavar="METRIC=COUNT",
        help=(
            "Fail when a profiled repeat OCR engine count metric p95 exceeds this budget. "
            "Repeat or comma-separate entries such as selected_box_count=30,raw_box_count=50."
        ),
    )
    parser.add_argument(
        "--max-repeat-ocr-engine-max-count",
        action="append",
        default=[],
        metavar="METRIC=COUNT",
        help=(
            "Fail when a profiled repeat OCR engine count metric max exceeds this budget. "
            "Repeat or comma-separate entries such as selected_box_count=30,raw_box_count=50."
        ),
    )
    parser.add_argument(
        "--max-ocr-engine-count",
        action="append",
        default=[],
        metavar="METRIC=COUNT",
        help=(
            "Fail when a profiled primary OCR engine count metric exceeds this budget. "
            "Repeat or comma-separate entries such as selected_box_count=30,raw_box_count=50."
        ),
    )
    parser.add_argument(
        "--min-ocr-call-contract-rows",
        type=int,
        default=None,
        help="Fail when fewer selected manifest rows define max_ocr_engine_calls.",
    )
    parser.add_argument(
        "--min-ocr-count-contract-rows",
        type=int,
        default=None,
        help="Fail when fewer selected manifest rows define at least one valid max_ocr_engine_counts metric.",
    )
    parser.add_argument(
        "--max-positive-ocr-call-only-rows",
        type=int,
        default=None,
        help=(
            "Fail when more selected rows with max_ocr_engine_calls > 0 lack valid "
            "max_ocr_engine_counts metrics."
        ),
    )
    parser.add_argument(
        "--fail-on-invalid-ocr-count-contracts",
        action="store_true",
        help="Fail when selected manifest rows define invalid max_ocr_engine_counts metrics or values.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_real_screenshot_hard_gate_preset(args, parser)
    if args.repeat_profile_runs < 0:
        parser.error("--repeat-profile-runs must be non-negative")
    if args.repeat_profile_warmups < 0:
        parser.error("--repeat-profile-warmups must be non-negative")
    if args.fail_on_repeat_signature_drift and args.repeat_profile_runs == 0:
        parser.error("--fail-on-repeat-signature-drift requires --repeat-profile-runs")
    if args.fail_on_baseline_signature_drift and not args.compare_baseline_report:
        parser.error("--fail-on-baseline-signature-drift requires --compare-baseline-report")
    if args.fail_on_baseline_config_drift and not args.compare_baseline_report:
        parser.error("--fail-on-baseline-config-drift requires --compare-baseline-report")
    if args.fail_on_baseline_coverage_gap and not args.compare_baseline_report:
        parser.error("--fail-on-baseline-coverage-gap requires --compare-baseline-report")
    if args.fail_on_baseline_expectation_regression and not args.compare_baseline_report:
        parser.error("--fail-on-baseline-expectation-regression requires --compare-baseline-report")
    if args.max_baseline_total_regression_s is not None and not args.compare_baseline_report:
        parser.error("--max-baseline-total-regression-s requires --compare-baseline-report")
    if args.max_baseline_repeat_p95_regression_s is not None and not args.compare_baseline_report:
        parser.error("--max-baseline-repeat-p95-regression-s requires --compare-baseline-report")
    if args.max_baseline_ocr_total_regression_s is not None and not args.compare_baseline_report:
        parser.error("--max-baseline-ocr-total-regression-s requires --compare-baseline-report")
    if (
        args.max_baseline_repeat_ocr_total_p95_regression_s is not None
        and not args.compare_baseline_report
    ):
        parser.error("--max-baseline-repeat-ocr-total-p95-regression-s requires --compare-baseline-report")
    if args.max_total_elapsed_s is not None and args.max_total_elapsed_s <= 0.0:
        parser.error("--max-total-elapsed-s must be positive")
    if args.max_repeat_profile_p95_duration_s is not None and args.max_repeat_profile_p95_duration_s <= 0.0:
        parser.error("--max-repeat-profile-p95-duration-s must be positive")
    if args.max_prewarm_runtime_s is not None and args.max_prewarm_runtime_s <= 0.0:
        parser.error("--max-prewarm-runtime-s must be positive")
    if args.max_baseline_total_regression_s is not None and args.max_baseline_total_regression_s < 0.0:
        parser.error("--max-baseline-total-regression-s must be non-negative")
    if (
        args.max_baseline_repeat_p95_regression_s is not None
        and args.max_baseline_repeat_p95_regression_s < 0.0
    ):
        parser.error("--max-baseline-repeat-p95-regression-s must be non-negative")
    if (
        args.max_baseline_ocr_total_regression_s is not None
        and args.max_baseline_ocr_total_regression_s < 0.0
    ):
        parser.error("--max-baseline-ocr-total-regression-s must be non-negative")
    if (
        args.max_baseline_repeat_ocr_total_p95_regression_s is not None
        and args.max_baseline_repeat_ocr_total_p95_regression_s < 0.0
    ):
        parser.error("--max-baseline-repeat-ocr-total-p95-regression-s must be non-negative")
    if args.min_ocr_call_contract_rows is not None and args.min_ocr_call_contract_rows < 0:
        parser.error("--min-ocr-call-contract-rows must be non-negative")
    if args.min_ocr_count_contract_rows is not None and args.min_ocr_count_contract_rows < 0:
        parser.error("--min-ocr-count-contract-rows must be non-negative")
    if args.max_positive_ocr_call_only_rows is not None and args.max_positive_ocr_call_only_rows < 0:
        parser.error("--max-positive-ocr-call-only-rows must be non-negative")
    try:
        max_prewarm_stage_s = parse_prewarm_stage_duration_budgets(args.max_prewarm_stage_s)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        max_repeat_ocr_engine_p95_duration_s = parse_metric_duration_budgets(
            args.max_repeat_ocr_engine_p95_duration_s
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        max_ocr_engine_duration_s = parse_metric_duration_budgets(args.max_ocr_engine_duration_s)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        max_repeat_ocr_engine_p95_count = parse_metric_count_budgets(
            args.max_repeat_ocr_engine_p95_count
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        max_repeat_ocr_engine_max_count = parse_metric_count_budgets(
            args.max_repeat_ocr_engine_max_count
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        max_ocr_engine_count = parse_metric_count_budgets(args.max_ocr_engine_count)
    except ValueError as exc:
        parser.error(str(exc))
    report = run_stress_benchmark(
        Path(args.manifest),
        Path(args.out_dir),
        only_slugs=args.only,
        timeout_seconds=args.timeout_seconds,
        write_debug=args.write_debug,
        profile_ocr_engine=args.profile_ocr_engine,
        runner_ocr_cache=not args.disable_ocr_cache,
        extraction_cache=not args.disable_extraction_cache,
        execution=args.execution,
        prewarm_runtime=args.prewarm_runtime,
        repeat_profile_runs=args.repeat_profile_runs,
        repeat_profile_warmups=args.repeat_profile_warmups,
        max_total_elapsed_s=args.max_total_elapsed_s,
        max_repeat_profile_p95_duration_s=args.max_repeat_profile_p95_duration_s,
        max_prewarm_runtime_s=args.max_prewarm_runtime_s,
        max_prewarm_stage_s=max_prewarm_stage_s,
        max_ocr_engine_duration_s=max_ocr_engine_duration_s,
        max_ocr_engine_count=max_ocr_engine_count,
        max_repeat_ocr_engine_p95_duration_s=max_repeat_ocr_engine_p95_duration_s,
        max_repeat_ocr_engine_p95_count=max_repeat_ocr_engine_p95_count,
        max_repeat_ocr_engine_max_count=max_repeat_ocr_engine_max_count,
        min_ocr_call_contract_rows=args.min_ocr_call_contract_rows,
        min_ocr_count_contract_rows=args.min_ocr_count_contract_rows,
        max_positive_ocr_call_only_rows=args.max_positive_ocr_call_only_rows,
        fail_on_invalid_ocr_count_contracts=args.fail_on_invalid_ocr_count_contracts,
        compare_baseline_report=Path(args.compare_baseline_report) if args.compare_baseline_report else None,
        max_baseline_total_regression_s=args.max_baseline_total_regression_s,
        max_baseline_repeat_p95_regression_s=args.max_baseline_repeat_p95_regression_s,
        max_baseline_ocr_total_regression_s=args.max_baseline_ocr_total_regression_s,
        max_baseline_repeat_ocr_total_p95_regression_s=(
            args.max_baseline_repeat_ocr_total_p95_regression_s
        ),
        preset=stress_preset_from_args(args),
    )
    print_stress_table(report)
    latency_budget = report.get("latency_budget")
    if isinstance(latency_budget, dict) and latency_budget.get("passed") is False:
        return 1
    manifest_contract_budget = report.get("manifest_contract_budget")
    if isinstance(manifest_contract_budget, dict) and manifest_contract_budget.get("passed") is False:
        return 1
    if args.prewarm_runtime and not prewarm_runtime_ok(report.get("prewarm")):
        return 1
    if args.fail_on_repeat_signature_drift and repeat_profile_signature_drift_cases(report):
        return 1
    if args.fail_on_baseline_signature_drift and baseline_comparison_signature_drift_cases(report):
        return 1
    if args.fail_on_baseline_config_drift and baseline_comparison_configuration_changes(report):
        return 1
    if args.fail_on_baseline_coverage_gap and baseline_comparison_coverage_gaps(report):
        return 1
    if (
        args.fail_on_baseline_expectation_regression
        and baseline_comparison_expectation_regressions(report)
    ):
        return 1
    if baseline_comparison_regression_budget_failed(report):
        return 1
    if args.fail_on_unexpected and (
        report["summary"]["unexpected"] or repeat_profile_unexpected_sample_count(report)
    ):
        return 1
    return 0


def apply_real_screenshot_hard_gate_preset(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.real_screenshot_hard_gate and args.focused_real_screenshot_gate:
        parser.error("--real-screenshot-hard-gate and --focused-real-screenshot-gate cannot be combined")
    if not args.real_screenshot_hard_gate and not args.focused_real_screenshot_gate:
        return
    if args.real_screenshot_hard_gate and args.only:
        parser.error("--real-screenshot-hard-gate targets the full manifest and cannot be combined with --only")
    if args.focused_real_screenshot_gate and not args.only:
        parser.error("--focused-real-screenshot-gate requires at least one --only slug")
    args.execution = "in-process"
    args.profile_ocr_engine = True
    args.prewarm_runtime = True
    args.disable_ocr_cache = True
    args.disable_extraction_cache = True
    args.fail_on_unexpected = True
    args.fail_on_repeat_signature_drift = True
    if args.compare_baseline_report:
        args.fail_on_baseline_signature_drift = True
        args.fail_on_baseline_config_drift = True
        args.fail_on_baseline_coverage_gap = True
        args.fail_on_baseline_expectation_regression = True
        if args.max_baseline_total_regression_s is None:
            args.max_baseline_total_regression_s = REAL_SCREENSHOT_HARD_GATE_BASELINE_REGRESSION_S
        if args.max_baseline_repeat_p95_regression_s is None:
            args.max_baseline_repeat_p95_regression_s = REAL_SCREENSHOT_HARD_GATE_BASELINE_REGRESSION_S
        if args.max_baseline_ocr_total_regression_s is None:
            args.max_baseline_ocr_total_regression_s = REAL_SCREENSHOT_HARD_GATE_BASELINE_REGRESSION_S
        if args.max_baseline_repeat_ocr_total_p95_regression_s is None:
            args.max_baseline_repeat_ocr_total_p95_regression_s = (
                REAL_SCREENSHOT_HARD_GATE_BASELINE_REGRESSION_S
            )
    if args.repeat_profile_runs == 0:
        args.repeat_profile_runs = 3
    if args.repeat_profile_warmups == 0:
        args.repeat_profile_warmups = 1
    if args.max_total_elapsed_s is None:
        args.max_total_elapsed_s = 1.0
    if args.max_repeat_profile_p95_duration_s is None:
        args.max_repeat_profile_p95_duration_s = 0.8
    if args.max_prewarm_runtime_s is None:
        args.max_prewarm_runtime_s = 2.0
    args.max_prewarm_stage_s = prepend_default_budget_entries(
        ["rapidocr_s=1.8,total_s=2.0"],
        args.max_prewarm_stage_s,
    )
    args.max_repeat_ocr_engine_p95_duration_s = prepend_default_budget_entries(
        ["total_s=0.7"],
        args.max_repeat_ocr_engine_p95_duration_s,
    )
    args.max_repeat_ocr_engine_p95_count = prepend_default_budget_entries(
        [REAL_SCREENSHOT_HARD_GATE_REPEAT_COUNT_BUDGET],
        args.max_repeat_ocr_engine_p95_count,
    )
    args.max_repeat_ocr_engine_max_count = prepend_default_budget_entries(
        [REAL_SCREENSHOT_HARD_GATE_REPEAT_COUNT_BUDGET],
        args.max_repeat_ocr_engine_max_count,
    )
    args.fail_on_invalid_ocr_count_contracts = True
    if args.focused_real_screenshot_gate:
        return
    if args.min_ocr_call_contract_rows is None:
        args.min_ocr_call_contract_rows = 49
    if args.min_ocr_count_contract_rows is None:
        args.min_ocr_count_contract_rows = 38
    if args.max_positive_ocr_call_only_rows is None:
        args.max_positive_ocr_call_only_rows = 0


def stress_preset_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.real_screenshot_hard_gate:
        return {
            "name": REAL_SCREENSHOT_HARD_GATE_PRESET_NAME,
            "version": REAL_SCREENSHOT_HARD_GATE_PRESET_VERSION,
        }
    if args.focused_real_screenshot_gate:
        return {
            "name": FOCUSED_REAL_SCREENSHOT_GATE_PRESET_NAME,
            "version": FOCUSED_REAL_SCREENSHOT_GATE_PRESET_VERSION,
            "only": list(args.only),
        }
    return None


def prepend_default_budget_entries(defaults: list[str], values: list[str]) -> list[str]:
    return [*defaults, *values]


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if isinstance(manifest, list):
        return {"version": 1, "cases": manifest}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError("Stress manifest must be a JSON object with a cases list.")
    return manifest


def run_stress_benchmark(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    only_slugs: list[str] | None = None,
    timeout_seconds: float = 30.0,
    write_debug: bool = False,
    profile_ocr_engine: bool = False,
    runner_ocr_cache: bool = True,
    extraction_cache: bool = True,
    execution: str = "subprocess",
    prewarm_runtime: bool = False,
    repeat_profile_runs: int = 0,
    repeat_profile_warmups: int = 0,
    max_total_elapsed_s: float | None = None,
    max_repeat_profile_p95_duration_s: float | None = None,
    max_prewarm_runtime_s: float | None = None,
    max_prewarm_stage_s: dict[str, float] | None = None,
    max_ocr_engine_duration_s: dict[str, float] | None = None,
    max_ocr_engine_count: dict[str, float] | None = None,
    max_repeat_ocr_engine_p95_duration_s: dict[str, float] | None = None,
    max_repeat_ocr_engine_p95_count: dict[str, float] | None = None,
    max_repeat_ocr_engine_max_count: dict[str, float] | None = None,
    min_ocr_call_contract_rows: int | None = None,
    min_ocr_count_contract_rows: int | None = None,
    max_positive_ocr_call_only_rows: int | None = None,
    fail_on_invalid_ocr_count_contracts: bool = False,
    compare_baseline_report: Path | None = None,
    max_baseline_total_regression_s: float | None = None,
    max_baseline_repeat_p95_regression_s: float | None = None,
    max_baseline_ocr_total_regression_s: float | None = None,
    max_baseline_repeat_ocr_total_p95_regression_s: float | None = None,
    preset: dict[str, Any] | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    if repeat_profile_runs < 0:
        raise ValueError("repeat_profile_runs must be non-negative")
    if repeat_profile_warmups < 0:
        raise ValueError("repeat_profile_warmups must be non-negative")
    if max_total_elapsed_s is not None and max_total_elapsed_s <= 0.0:
        raise ValueError("max_total_elapsed_s must be positive")
    if max_repeat_profile_p95_duration_s is not None and max_repeat_profile_p95_duration_s <= 0.0:
        raise ValueError("max_repeat_profile_p95_duration_s must be positive")
    if max_prewarm_runtime_s is not None and max_prewarm_runtime_s <= 0.0:
        raise ValueError("max_prewarm_runtime_s must be positive")
    if min_ocr_call_contract_rows is not None and min_ocr_call_contract_rows < 0:
        raise ValueError("min_ocr_call_contract_rows must be non-negative")
    if min_ocr_count_contract_rows is not None and min_ocr_count_contract_rows < 0:
        raise ValueError("min_ocr_count_contract_rows must be non-negative")
    if max_positive_ocr_call_only_rows is not None and max_positive_ocr_call_only_rows < 0:
        raise ValueError("max_positive_ocr_call_only_rows must be non-negative")
    if max_baseline_total_regression_s is not None and max_baseline_total_regression_s < 0.0:
        raise ValueError("max_baseline_total_regression_s must be non-negative")
    if max_baseline_repeat_p95_regression_s is not None and max_baseline_repeat_p95_regression_s < 0.0:
        raise ValueError("max_baseline_repeat_p95_regression_s must be non-negative")
    if max_baseline_ocr_total_regression_s is not None and max_baseline_ocr_total_regression_s < 0.0:
        raise ValueError("max_baseline_ocr_total_regression_s must be non-negative")
    if (
        max_baseline_repeat_ocr_total_p95_regression_s is not None
        and max_baseline_repeat_ocr_total_p95_regression_s < 0.0
    ):
        raise ValueError("max_baseline_repeat_ocr_total_p95_regression_s must be non-negative")
    ocr_engine_budgets = dict(max_ocr_engine_duration_s or {})
    ocr_engine_count_budgets = dict(max_ocr_engine_count or {})
    ocr_engine_p95_budgets = dict(max_repeat_ocr_engine_p95_duration_s or {})
    ocr_engine_p95_count_budgets = dict(max_repeat_ocr_engine_p95_count or {})
    ocr_engine_max_count_budgets = dict(max_repeat_ocr_engine_max_count or {})
    prewarm_stage_budgets = dict(max_prewarm_stage_s or {})
    if execution not in {"subprocess", "in-process"}:
        raise ValueError(f"Unsupported stress execution mode: {execution}")
    manifest = load_manifest(manifest_path)
    cases = select_cases(manifest["cases"], only_slugs or [])
    out_dir.mkdir(parents=True, exist_ok=True)
    prewarm = (
        run_generation_prewarm(
            runner_ocr_cache=runner_ocr_cache,
            extraction_cache=extraction_cache,
        )
        if prewarm_runtime
        else None
    )

    rows = [
        run_stress_case(
            case,
            out_dir,
            timeout_seconds=timeout_seconds,
            write_debug=write_debug,
            profile_ocr_engine=profile_ocr_engine,
            runner_ocr_cache=runner_ocr_cache,
            extraction_cache=extraction_cache,
            execution=execution,
            python_executable=python_executable,
        )
        for case in cases
    ]
    repeat_profile = (
        build_repeat_profile(
            cases,
            out_dir=out_dir,
            runs_per_case=repeat_profile_runs,
            warmup_runs_per_case=repeat_profile_warmups,
            timeout_seconds=timeout_seconds,
            write_debug=write_debug,
            profile_ocr_engine=profile_ocr_engine,
            runner_ocr_cache=runner_ocr_cache,
            extraction_cache=extraction_cache,
            execution=execution,
            python_executable=python_executable,
        )
        if repeat_profile_runs
        else None
    )
    runtime_config = stress_runtime_config(
        runner_ocr_cache=runner_ocr_cache,
        extraction_cache=extraction_cache,
    )
    manifest_contracts = summarize_manifest_contracts(cases)
    report = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "profile_ocr_engine": profile_ocr_engine,
        "runner_ocr_cache": runner_ocr_cache,
        "extraction_cache": extraction_cache,
        "execution": execution,
        "prewarm_runtime": prewarm_runtime,
        "repeat_profile_runs": repeat_profile_runs,
        "repeat_profile_warmups": repeat_profile_warmups,
        "runtime_config": runtime_config,
        "manifest_contracts": manifest_contracts,
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    if preset is not None:
        report["preset"] = preset
    if prewarm is not None:
        report["prewarm"] = prewarm
    if repeat_profile is not None:
        report["repeat_profile"] = repeat_profile
    if (
        max_total_elapsed_s is not None
        or max_repeat_profile_p95_duration_s is not None
        or max_prewarm_runtime_s is not None
        or prewarm_stage_budgets
        or ocr_engine_budgets
        or ocr_engine_count_budgets
        or ocr_engine_p95_budgets
        or ocr_engine_p95_count_budgets
        or ocr_engine_max_count_budgets
    ):
        report["latency_budget"] = build_latency_budget_summary(
            rows,
            repeat_profile,
            prewarm=prewarm,
            max_total_elapsed_s=max_total_elapsed_s,
            max_repeat_profile_p95_duration_s=max_repeat_profile_p95_duration_s,
            max_prewarm_runtime_s=max_prewarm_runtime_s,
            max_prewarm_stage_s=prewarm_stage_budgets,
            max_ocr_engine_duration_s=ocr_engine_budgets,
            max_ocr_engine_count=ocr_engine_count_budgets,
            max_repeat_ocr_engine_p95_duration_s=ocr_engine_p95_budgets,
            max_repeat_ocr_engine_p95_count=ocr_engine_p95_count_budgets,
            max_repeat_ocr_engine_max_count=ocr_engine_max_count_budgets,
        )
    if (
        min_ocr_call_contract_rows is not None
        or min_ocr_count_contract_rows is not None
        or max_positive_ocr_call_only_rows is not None
        or fail_on_invalid_ocr_count_contracts
    ):
        report["manifest_contract_budget"] = build_manifest_contract_budget_summary(
            manifest_contracts,
            min_ocr_call_contract_rows=min_ocr_call_contract_rows,
            min_ocr_count_contract_rows=min_ocr_count_contract_rows,
            max_positive_ocr_call_only_rows=max_positive_ocr_call_only_rows,
            fail_on_invalid_ocr_count_contracts=fail_on_invalid_ocr_count_contracts,
        )
    if compare_baseline_report is not None:
        report["baseline_comparison"] = compare_stress_reports(
            load_stress_report(compare_baseline_report),
            report,
            baseline_report_path=compare_baseline_report,
            max_total_elapsed_regression_s=max_baseline_total_regression_s,
            max_repeat_p95_regression_s=max_baseline_repeat_p95_regression_s,
            max_ocr_engine_total_regression_s=max_baseline_ocr_total_regression_s,
            max_repeat_ocr_engine_total_p95_regression_s=(
                max_baseline_repeat_ocr_total_p95_regression_s
            ),
        )
    (out_dir / "stress-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_stress_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Baseline stress report must be a JSON object.")
    return payload


def compare_stress_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    baseline_report_path: Path | None = None,
    max_total_elapsed_regression_s: float | None = None,
    max_repeat_p95_regression_s: float | None = None,
    max_ocr_engine_total_regression_s: float | None = None,
    max_repeat_ocr_engine_total_p95_regression_s: float | None = None,
) -> dict[str, Any]:
    baseline_rows = stress_report_rows_by_slug(baseline_report)
    candidate_rows = stress_report_rows_by_slug(candidate_report)
    candidate_order = [slug for slug in stress_report_slug_order(candidate_report) if slug in candidate_rows]
    compared_slugs = [slug for slug in candidate_order if slug in baseline_rows]
    candidate_scope_slugs = stress_report_candidate_scope_slugs(candidate_report)
    candidate_scope_set = set(candidate_scope_slugs) if candidate_scope_slugs is not None else None
    missing_in_baseline = sorted(slug for slug in candidate_rows if slug not in baseline_rows)
    if candidate_scope_set is None:
        missing_in_candidate = sorted(slug for slug in baseline_rows if slug not in candidate_rows)
        baseline_rows_outside_candidate_scope: list[str] = []
    else:
        missing_in_candidate = sorted(
            slug for slug in candidate_scope_slugs if slug in baseline_rows and slug not in candidate_rows
        )
        baseline_rows_outside_candidate_scope = sorted(
            slug for slug in baseline_rows if slug not in candidate_scope_set
        )
    signature_changes: list[dict[str, Any]] = []
    expectation_changes: list[dict[str, Any]] = []
    latency_deltas: list[dict[str, Any]] = []
    expectation_compared_rows = 0
    baseline_expectation_passed_rows = 0
    candidate_expectation_passed_rows = 0

    for slug in compared_slugs:
        baseline_row = baseline_rows[slug]
        candidate_row = candidate_rows[slug]
        expectation_change = stress_row_expectation_change(slug, baseline_row, candidate_row)
        if expectation_change is not None:
            expectation_compared_rows += 1
            if expectation_change["baseline_expectation_passed"]:
                baseline_expectation_passed_rows += 1
            if expectation_change["candidate_expectation_passed"]:
                candidate_expectation_passed_rows += 1
            if (
                expectation_change["baseline_expectation_passed"]
                != expectation_change["candidate_expectation_passed"]
            ):
                expectation_changes.append(expectation_change)
        baseline_signature = repeat_profile_output_signature(baseline_row)
        candidate_signature = repeat_profile_output_signature(candidate_row)
        if baseline_signature != candidate_signature:
            signature_changes.append(
                {
                    "slug": slug,
                    "changed_fields": stress_signature_changed_fields(
                        baseline_signature,
                        candidate_signature,
                    ),
                    "baseline": baseline_signature,
                    "candidate": candidate_signature,
                }
            )
        latency_delta = stress_row_latency_delta(slug, baseline_row, candidate_row)
        if latency_delta is not None:
            latency_deltas.append(latency_delta)

    total_deltas = [
        float(delta["total_elapsed_delta_s"])
        for delta in latency_deltas
        if isinstance(delta.get("total_elapsed_delta_s"), (int, float))
    ]
    comparison: dict[str, Any] = {
        "baseline_report": str(baseline_report_path) if baseline_report_path is not None else None,
        "compared_rows": len(compared_slugs),
        "missing_in_baseline": missing_in_baseline,
        "missing_in_candidate": missing_in_candidate,
        "configuration_changes": stress_report_configuration_changes(
            baseline_report,
            candidate_report,
        ),
        "expectation_compared_rows": expectation_compared_rows,
        "baseline_expectation_passed_rows": baseline_expectation_passed_rows,
        "candidate_expectation_passed_rows": candidate_expectation_passed_rows,
        "expectation_passed_delta": candidate_expectation_passed_rows
        - baseline_expectation_passed_rows,
        "expectation_change_count": len(expectation_changes),
        "expectation_changes": expectation_changes,
        "signature_change_count": len(signature_changes),
        "signature_changed_field_counts": signature_changed_field_counts(signature_changes),
        "signature_changes": signature_changes,
        "latency_deltas": latency_deltas,
        "largest_total_regressions": ranked_latency_deltas(latency_deltas, reverse=True),
        "largest_total_improvements": ranked_latency_deltas(latency_deltas, reverse=False),
        "largest_ocr_engine_total_regressions": ranked_ocr_engine_total_deltas(
            latency_deltas, reverse=True
        ),
        "largest_ocr_engine_total_improvements": ranked_ocr_engine_total_deltas(
            latency_deltas, reverse=False
        ),
        "largest_ocr_overlap_hidden_regressions": ranked_ocr_overlap_hidden_deltas(
            latency_deltas, reverse=True
        ),
        "largest_ocr_overlap_hidden_improvements": ranked_ocr_overlap_hidden_deltas(
            latency_deltas, reverse=False
        ),
    }
    if candidate_scope_slugs is not None:
        comparison["candidate_scope"] = {
            "kind": "only",
            "slugs": candidate_scope_slugs,
            "baseline_rows_outside_candidate_scope_count": len(baseline_rows_outside_candidate_scope),
            "baseline_rows_outside_candidate_scope": baseline_rows_outside_candidate_scope,
        }
    if total_deltas:
        comparison.update(
            {
                "median_total_elapsed_delta_s": round(float(median(total_deltas)), 6),
                "average_total_elapsed_delta_s": round(float(mean(total_deltas)), 6),
                "max_total_elapsed_delta_s": round(max(total_deltas), 6),
                "min_total_elapsed_delta_s": round(min(total_deltas), 6),
            }
        )
    repeat_delta = stress_repeat_profile_delta(baseline_report, candidate_report)
    if repeat_delta is not None:
        comparison["repeat_profile_delta"] = repeat_delta
    repeat_case_coverage = stress_repeat_profile_case_coverage(
        baseline_report,
        candidate_report,
        compared_slugs,
    )
    if repeat_case_coverage is not None:
        comparison["repeat_profile_case_expected_rows"] = repeat_case_coverage["expected_rows"]
        comparison["repeat_profile_case_compared_rows"] = repeat_case_coverage["compared_rows"]
        comparison["repeat_profile_case_missing_in_baseline"] = repeat_case_coverage[
            "missing_in_baseline"
        ]
        comparison["repeat_profile_case_missing_in_candidate"] = repeat_case_coverage[
            "missing_in_candidate"
        ]
        comparison["repeat_profile_case_underanalyzed_in_baseline"] = repeat_case_coverage[
            "underanalyzed_in_baseline"
        ]
        comparison["repeat_profile_case_underanalyzed_in_candidate"] = repeat_case_coverage[
            "underanalyzed_in_candidate"
        ]
    repeat_case_deltas = stress_repeat_profile_case_deltas(
        baseline_report,
        candidate_report,
        slugs=compared_slugs,
    )
    if repeat_case_deltas:
        comparison["repeat_profile_case_deltas"] = repeat_case_deltas
        comparison["largest_repeat_profile_case_p95_regressions"] = ranked_repeat_profile_case_deltas(
            repeat_case_deltas,
            reverse=True,
        )
        comparison["largest_repeat_profile_case_p95_improvements"] = ranked_repeat_profile_case_deltas(
            repeat_case_deltas,
            reverse=False,
        )
        comparison["largest_repeat_profile_case_ocr_p95_regressions"] = (
            ranked_repeat_profile_case_ocr_deltas(
                repeat_case_deltas,
                reverse=True,
            )
        )
        comparison["largest_repeat_profile_case_ocr_p95_improvements"] = (
            ranked_repeat_profile_case_ocr_deltas(
                repeat_case_deltas,
                reverse=False,
            )
        )
    if (
        max_total_elapsed_regression_s is not None
        or max_repeat_p95_regression_s is not None
        or max_ocr_engine_total_regression_s is not None
        or max_repeat_ocr_engine_total_p95_regression_s is not None
    ):
        comparison["regression_budget"] = baseline_regression_budget(
            comparison,
            max_total_elapsed_regression_s=max_total_elapsed_regression_s,
            max_repeat_p95_regression_s=max_repeat_p95_regression_s,
            max_ocr_engine_total_regression_s=max_ocr_engine_total_regression_s,
            max_repeat_ocr_engine_total_p95_regression_s=(
                max_repeat_ocr_engine_total_p95_regression_s
            ),
        )
    return comparison


def baseline_regression_budget(
    comparison: dict[str, Any],
    *,
    max_total_elapsed_regression_s: float | None = None,
    max_repeat_p95_regression_s: float | None = None,
    max_ocr_engine_total_regression_s: float | None = None,
    max_repeat_ocr_engine_total_p95_regression_s: float | None = None,
) -> dict[str, Any]:
    budget: dict[str, Any] = {"violations": []}
    violations: list[dict[str, Any]] = []
    skipped_primary_ocr_zero_call_rows: list[str] = []
    latency_deltas = comparison.get("latency_deltas")
    if max_total_elapsed_regression_s is not None:
        max_total = round(float(max_total_elapsed_regression_s), 6)
        budget["max_total_elapsed_regression_s"] = max_total
        if isinstance(latency_deltas, list):
            for delta in latency_deltas:
                if not isinstance(delta, dict):
                    continue
                total_delta = parse_signed_float(delta.get("total_elapsed_delta_s"))
                if total_delta is None or total_delta <= max_total_elapsed_regression_s:
                    continue
                violation = {
                    "kind": "primary_total_regression_exceeded",
                    "slug": delta.get("slug"),
                    "total_elapsed_delta_s": round(total_delta, 6),
                    "max_total_elapsed_regression_s": max_total,
                }
                for key in ("baseline_total_elapsed_s", "candidate_total_elapsed_s"):
                    value = parse_nonnegative_float(delta.get(key))
                    if value is not None:
                        violation[key] = round(value, 6)
                violations.append(violation)
    if max_ocr_engine_total_regression_s is not None:
        max_ocr_total = round(float(max_ocr_engine_total_regression_s), 6)
        budget["max_ocr_engine_total_regression_s"] = max_ocr_total
        if isinstance(latency_deltas, list):
            for delta in latency_deltas:
                if not isinstance(delta, dict):
                    continue
                ocr_total_delta = parse_signed_float(delta.get("ocr_engine_total_delta_s"))
                if ocr_total_delta is None:
                    baseline_calls = parse_nonnegative_count_metric(
                        delta.get("baseline_ocr_engine_calls")
                    )
                    candidate_calls = parse_nonnegative_count_metric(
                        delta.get("candidate_ocr_engine_calls")
                    )
                    if baseline_calls == 0.0 and candidate_calls == 0.0:
                        slug = delta.get("slug")
                        if isinstance(slug, str) and slug:
                            skipped_primary_ocr_zero_call_rows.append(slug)
                        continue
                    violation = {
                        "kind": "primary_ocr_total_delta_missing",
                        "slug": delta.get("slug"),
                        "max_ocr_engine_total_regression_s": max_ocr_total,
                    }
                    if baseline_calls is not None:
                        violation["baseline_ocr_engine_calls"] = round(baseline_calls, 6)
                    if candidate_calls is not None:
                        violation["candidate_ocr_engine_calls"] = round(candidate_calls, 6)
                    violations.append(violation)
                    continue
                if ocr_total_delta <= max_ocr_engine_total_regression_s:
                    continue
                violation = {
                    "kind": "primary_ocr_total_regression_exceeded",
                    "slug": delta.get("slug"),
                    "ocr_engine_total_delta_s": round(ocr_total_delta, 6),
                    "max_ocr_engine_total_regression_s": max_ocr_total,
                }
                for key in ("baseline_ocr_engine_total_s", "candidate_ocr_engine_total_s"):
                    value = parse_nonnegative_float(delta.get(key))
                    if value is not None:
                        violation[key] = round(value, 6)
                violations.append(violation)
        if skipped_primary_ocr_zero_call_rows:
            budget["skipped_primary_ocr_zero_call_row_count"] = len(
                skipped_primary_ocr_zero_call_rows
            )
            budget["skipped_primary_ocr_zero_call_rows"] = skipped_primary_ocr_zero_call_rows
    if max_repeat_p95_regression_s is not None:
        max_repeat = round(float(max_repeat_p95_regression_s), 6)
        budget["max_repeat_p95_regression_s"] = max_repeat
        repeat_delta = comparison.get("repeat_profile_delta")
        repeat_p95_delta = (
            repeat_delta_metric_value(
                repeat_delta,
                ("duration_s", "p95_total_elapsed_s"),
                "delta_s",
            )
            if isinstance(repeat_delta, dict)
            else None
        )
        if repeat_p95_delta is None:
            violations.append(
                {
                    "kind": "repeat_profile_p95_delta_missing",
                    "max_repeat_p95_regression_s": max_repeat,
                }
            )
        elif repeat_p95_delta > max_repeat_p95_regression_s:
            repeat_p95 = repeat_delta_metric_value(
                repeat_delta,
                ("duration_s", "p95_total_elapsed_s"),
                "candidate",
            )
            baseline_p95 = repeat_delta_metric_value(
                repeat_delta,
                ("duration_s", "p95_total_elapsed_s"),
                "baseline",
            )
            violation = {
                "kind": "repeat_profile_p95_regression_exceeded",
                "delta_s": round(repeat_p95_delta, 6),
                "max_repeat_p95_regression_s": max_repeat,
            }
            if baseline_p95 is not None:
                violation["baseline_p95_total_elapsed_s"] = round(baseline_p95, 6)
            if repeat_p95 is not None:
                violation["candidate_p95_total_elapsed_s"] = round(repeat_p95, 6)
            violations.append(violation)
        case_deltas = comparison.get("repeat_profile_case_deltas")
        if isinstance(case_deltas, list):
            for case_delta in case_deltas:
                if not isinstance(case_delta, dict):
                    continue
                p95_delta = repeat_case_delta_metric_value(
                    case_delta,
                    "p95_total_elapsed_s",
                    "delta_s",
                )
                if p95_delta is None or p95_delta <= max_repeat_p95_regression_s:
                    continue
                violation = {
                    "kind": "repeat_profile_case_p95_regression_exceeded",
                    "slug": case_delta.get("slug"),
                    "delta_s": round(p95_delta, 6),
                    "max_repeat_p95_regression_s": max_repeat,
                }
                baseline_case_p95 = repeat_case_delta_metric_value(
                    case_delta,
                    "p95_total_elapsed_s",
                    "baseline",
                )
                candidate_case_p95 = repeat_case_delta_metric_value(
                    case_delta,
                    "p95_total_elapsed_s",
                    "candidate",
                )
                if baseline_case_p95 is not None:
                    violation["baseline_p95_total_elapsed_s"] = round(baseline_case_p95, 6)
                if candidate_case_p95 is not None:
                    violation["candidate_p95_total_elapsed_s"] = round(candidate_case_p95, 6)
                violations.append(violation)
    if max_repeat_ocr_engine_total_p95_regression_s is not None:
        max_repeat_ocr_total = round(float(max_repeat_ocr_engine_total_p95_regression_s), 6)
        budget["max_repeat_ocr_engine_total_p95_regression_s"] = max_repeat_ocr_total
        repeat_delta = comparison.get("repeat_profile_delta")
        repeat_ocr_p95_delta = (
            repeat_delta_metric_value(
                repeat_delta,
                ("ocr_engine_stage_duration_s", "total_s", "p95_duration_s"),
                "delta_s",
            )
            if isinstance(repeat_delta, dict)
            else None
        )
        if repeat_ocr_p95_delta is None:
            violations.append(
                {
                    "kind": "repeat_ocr_total_p95_delta_missing",
                    "max_repeat_ocr_engine_total_p95_regression_s": max_repeat_ocr_total,
                }
            )
        elif repeat_ocr_p95_delta > max_repeat_ocr_engine_total_p95_regression_s:
            repeat_ocr_p95 = repeat_delta_metric_value(
                repeat_delta,
                ("ocr_engine_stage_duration_s", "total_s", "p95_duration_s"),
                "candidate",
            )
            baseline_ocr_p95 = repeat_delta_metric_value(
                repeat_delta,
                ("ocr_engine_stage_duration_s", "total_s", "p95_duration_s"),
                "baseline",
            )
            violation = {
                "kind": "repeat_ocr_total_p95_regression_exceeded",
                "delta_s": round(repeat_ocr_p95_delta, 6),
                "max_repeat_ocr_engine_total_p95_regression_s": max_repeat_ocr_total,
            }
            if baseline_ocr_p95 is not None:
                violation["baseline_ocr_engine_total_p95_duration_s"] = round(baseline_ocr_p95, 6)
            if repeat_ocr_p95 is not None:
                violation["candidate_ocr_engine_total_p95_duration_s"] = round(repeat_ocr_p95, 6)
            violations.append(violation)
        case_deltas = comparison.get("repeat_profile_case_deltas")
        if isinstance(case_deltas, list):
            for case_delta in case_deltas:
                if not isinstance(case_delta, dict):
                    continue
                p95_delta = repeat_case_delta_ocr_stage_value(
                    case_delta,
                    "total_s",
                    "p95_duration_s",
                    "delta_s",
                )
                if (
                    p95_delta is None
                    or p95_delta <= max_repeat_ocr_engine_total_p95_regression_s
                ):
                    continue
                violation = {
                    "kind": "repeat_ocr_case_total_p95_regression_exceeded",
                    "slug": case_delta.get("slug"),
                    "delta_s": round(p95_delta, 6),
                    "max_repeat_ocr_engine_total_p95_regression_s": max_repeat_ocr_total,
                }
                baseline_case_p95 = repeat_case_delta_ocr_stage_value(
                    case_delta,
                    "total_s",
                    "p95_duration_s",
                    "baseline",
                )
                candidate_case_p95 = repeat_case_delta_ocr_stage_value(
                    case_delta,
                    "total_s",
                    "p95_duration_s",
                    "candidate",
                )
                if baseline_case_p95 is not None:
                    violation["baseline_ocr_engine_total_p95_duration_s"] = round(
                        baseline_case_p95,
                        6,
                    )
                if candidate_case_p95 is not None:
                    violation["candidate_ocr_engine_total_p95_duration_s"] = round(
                        candidate_case_p95,
                        6,
                    )
                violations.append(violation)
    budget["violations"] = violations
    budget["passed"] = not violations
    return budget


def stress_repeat_profile_delta(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any] | None:
    baseline_summary = stress_report_repeat_profile_summary(baseline_report)
    candidate_summary = stress_report_repeat_profile_summary(candidate_report)
    if baseline_summary is None or candidate_summary is None:
        return None
    delta: dict[str, Any] = {}
    count_fields: dict[str, dict[str, float]] = {}
    for key in (
        "analyzed_samples",
        "expectation_passed_samples",
        "unexpected_samples",
        "subsecond_samples",
        "ocr_full_detail_retry_samples",
    ):
        metric_delta = stress_scalar_delta(baseline_summary, candidate_summary, key, delta_key="delta")
        if metric_delta is not None:
            count_fields[key] = metric_delta
    if count_fields:
        delta["sample_counts"] = count_fields

    duration_fields: dict[str, dict[str, float]] = {}
    for key in ("median_total_elapsed_s", "p95_total_elapsed_s", "max_total_elapsed_s"):
        metric_delta = stress_scalar_delta(baseline_summary, candidate_summary, key, delta_key="delta_s")
        if metric_delta is not None:
            duration_fields[key] = metric_delta
    if duration_fields:
        delta["duration_s"] = duration_fields

    stage_fields = stress_stage_duration_deltas(baseline_summary, candidate_summary)
    if stage_fields:
        delta["stage_duration_s"] = stage_fields

    ocr_stage_fields = stress_nested_metric_deltas(
        baseline_summary,
        candidate_summary,
        section_key="ocr_engine_stage_duration_s",
        metric_keys=OCR_ENGINE_STAGE_MAX_KEYS,
        stat_keys=("p95_duration_s", "max_duration_s"),
        delta_key="delta_s",
    )
    if ocr_stage_fields:
        delta["ocr_engine_stage_duration_s"] = ocr_stage_fields

    ocr_count_fields = stress_nested_metric_deltas(
        baseline_summary,
        candidate_summary,
        section_key="ocr_engine_count_metric",
        metric_keys=OCR_ENGINE_COUNT_DISPLAY_KEYS,
        stat_keys=("p95_count", "max_count"),
        delta_key="delta_count",
    )
    if ocr_count_fields:
        delta["ocr_engine_count_metric"] = ocr_count_fields
    hidden_overlap_delta = stress_metric_group_delta(
        baseline_summary,
        candidate_summary,
        section_key="ocr_overlap_hidden_s",
        stat_keys=("p95_duration_s", "max_duration_s", "total_s"),
        delta_key="delta_s",
    )
    if hidden_overlap_delta:
        delta["ocr_overlap_hidden_s"] = hidden_overlap_delta
    return delta or None


def stress_repeat_profile_case_deltas(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    slugs: list[str] | None = None,
) -> list[dict[str, Any]]:
    baseline_cases = stress_report_repeat_profile_cases(baseline_report)
    candidate_cases = stress_report_repeat_profile_cases(candidate_report)
    if baseline_cases is None or candidate_cases is None:
        return []
    deltas: list[dict[str, Any]] = []
    compared_slugs = (
        [slug for slug in slugs if slug in baseline_cases and slug in candidate_cases]
        if slugs is not None
        else sorted(set(baseline_cases) & set(candidate_cases))
    )
    for slug in compared_slugs:
        baseline_case = baseline_cases[slug]
        candidate_case = candidate_cases[slug]
        delta: dict[str, Any] = {"slug": slug}
        duration_fields: dict[str, dict[str, float]] = {}
        for key in ("median_total_elapsed_s", "p95_total_elapsed_s", "max_total_elapsed_s"):
            metric_delta = stress_scalar_delta(
                baseline_case,
                candidate_case,
                key,
                delta_key="delta_s",
            )
            if metric_delta is not None:
                duration_fields[key] = metric_delta
        if duration_fields:
            delta["duration_s"] = duration_fields
        stage_fields = stress_stage_duration_deltas(baseline_case, candidate_case)
        if stage_fields:
            delta["stage_duration_s"] = stage_fields
        ocr_stage_fields = stress_nested_metric_deltas(
            baseline_case,
            candidate_case,
            section_key="ocr_engine_stage_duration_s",
            metric_keys=OCR_ENGINE_STAGE_MAX_KEYS,
            stat_keys=("p95_duration_s", "max_duration_s"),
            delta_key="delta_s",
        )
        if ocr_stage_fields:
            delta["ocr_engine_stage_duration_s"] = ocr_stage_fields
        if len(delta) > 1:
            deltas.append(delta)
    return deltas


def stress_repeat_profile_case_coverage(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    compared_slugs: list[str],
) -> dict[str, Any] | None:
    baseline_cases = stress_report_repeat_profile_cases(baseline_report)
    candidate_cases = stress_report_repeat_profile_cases(candidate_report)
    if baseline_cases is None and candidate_cases is None:
        return None
    baseline_case_slugs = set(baseline_cases or {})
    candidate_case_slugs = set(candidate_cases or {})
    missing_in_baseline = sorted(slug for slug in compared_slugs if slug not in baseline_case_slugs)
    missing_in_candidate = sorted(slug for slug in compared_slugs if slug not in candidate_case_slugs)
    compared_case_slugs = [
        slug
        for slug in compared_slugs
        if slug in baseline_case_slugs and slug in candidate_case_slugs
    ]
    return {
        "expected_rows": len(compared_slugs),
        "compared_rows": len(compared_case_slugs),
        "missing_in_baseline": missing_in_baseline,
        "missing_in_candidate": missing_in_candidate,
        "underanalyzed_in_baseline": stress_repeat_profile_underanalyzed_cases(
            baseline_report,
            baseline_cases,
            compared_slugs,
        ),
        "underanalyzed_in_candidate": stress_repeat_profile_underanalyzed_cases(
            candidate_report,
            candidate_cases,
            compared_slugs,
        ),
    }


def stress_repeat_profile_underanalyzed_cases(
    report: dict[str, Any],
    cases: dict[str, dict[str, Any]] | None,
    compared_slugs: list[str],
) -> list[dict[str, Any]]:
    if cases is None:
        return []
    expected_analyzed = stress_report_expected_analyzed_repeat_samples_per_case(report)
    if expected_analyzed is None or expected_analyzed <= 0:
        return []
    underanalyzed: list[dict[str, Any]] = []
    for slug in compared_slugs:
        case_summary = cases.get(slug)
        if not isinstance(case_summary, dict):
            continue
        analyzed_samples = parse_nonnegative_int(case_summary.get("analyzed_samples"))
        if analyzed_samples is not None and analyzed_samples >= expected_analyzed:
            continue
        gap: dict[str, Any] = {
            "slug": slug,
            "expected_analyzed_samples": expected_analyzed,
        }
        if analyzed_samples is not None:
            gap["analyzed_samples"] = analyzed_samples
        underanalyzed.append(gap)
    return underanalyzed


def stress_report_expected_analyzed_repeat_samples_per_case(report: dict[str, Any]) -> int | None:
    repeat_profile = report.get("repeat_profile")
    repeat_profile_runs = None
    repeat_profile_warmups = None
    if isinstance(repeat_profile, dict):
        repeat_profile_runs = parse_nonnegative_int(repeat_profile.get("runs_per_case"))
        repeat_profile_warmups = parse_nonnegative_int(
            repeat_profile.get("warmup_runs_per_case")
        )
    if repeat_profile_runs is None:
        repeat_profile_runs = parse_nonnegative_int(report.get("repeat_profile_runs"))
    if repeat_profile_warmups is None:
        repeat_profile_warmups = parse_nonnegative_int(report.get("repeat_profile_warmups")) or 0
    if repeat_profile_runs is None:
        return None
    return max(0, repeat_profile_runs - repeat_profile_warmups)


def stress_report_repeat_profile_cases(report: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    repeat_profile = report.get("repeat_profile")
    if not isinstance(repeat_profile, dict):
        return None
    samples = repeat_profile.get("samples")
    if isinstance(samples, list):
        sample_dicts = [sample for sample in samples if isinstance(sample, dict)]
        if sample_dicts:
            rebuilt = summarize_repeat_profile_samples(
                sample_dicts,
                runs_per_case=parse_nonnegative_int(repeat_profile.get("runs_per_case")) or 0,
                warmup_runs_per_case=parse_nonnegative_int(
                    repeat_profile.get("warmup_runs_per_case")
                )
                or 0,
            )
            rebuilt_cases = rebuilt.get("cases")
            if isinstance(rebuilt_cases, dict) and rebuilt_cases:
                return {
                    slug: summary
                    for slug, summary in rebuilt_cases.items()
                    if isinstance(slug, str) and slug and isinstance(summary, dict)
                }
    cases = repeat_profile.get("cases")
    if not isinstance(cases, dict):
        return None
    case_summaries: dict[str, dict[str, Any]] = {}
    for raw_slug, raw_summary in cases.items():
        if not isinstance(raw_slug, str) or not raw_slug or not isinstance(raw_summary, dict):
            continue
        case_summaries[raw_slug] = raw_summary
    return case_summaries


def ranked_repeat_profile_case_deltas(
    case_deltas: list[dict[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = [
        delta
        for delta in case_deltas
        if repeat_case_delta_metric_value(delta, "p95_total_elapsed_s", "delta_s") is not None
    ]
    ranked.sort(
        key=lambda delta: (
            float(
                repeat_case_delta_metric_value(
                    delta,
                    "p95_total_elapsed_s",
                    "delta_s",
                )
                or 0.0
            ),
            str(delta.get("slug") or ""),
        ),
        reverse=reverse,
    )
    return ranked[: max(0, limit)]


def ranked_repeat_profile_case_ocr_deltas(
    case_deltas: list[dict[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = [
        delta
        for delta in case_deltas
        if repeat_case_delta_ocr_stage_value(
            delta,
            "total_s",
            "p95_duration_s",
            "delta_s",
        )
        is not None
    ]
    ranked.sort(
        key=lambda delta: (
            float(
                repeat_case_delta_ocr_stage_value(
                    delta,
                    "total_s",
                    "p95_duration_s",
                    "delta_s",
                )
                or 0.0
            ),
            str(delta.get("slug") or ""),
        ),
        reverse=reverse,
    )
    return ranked[: max(0, limit)]


def repeat_case_delta_metric_value(
    case_delta: dict[str, Any],
    metric: str,
    value_key: str,
) -> float | None:
    duration_s = case_delta.get("duration_s")
    if not isinstance(duration_s, dict):
        return None
    metric_delta = duration_s.get(metric)
    if not isinstance(metric_delta, dict):
        return None
    return parse_signed_float(metric_delta.get(value_key))


def repeat_case_delta_ocr_stage_value(
    case_delta: dict[str, Any],
    metric: str,
    stat: str,
    value_key: str,
) -> float | None:
    stage_duration = case_delta.get("ocr_engine_stage_duration_s")
    if not isinstance(stage_duration, dict):
        return None
    metric_delta = stage_duration.get(metric)
    if not isinstance(metric_delta, dict):
        return None
    stat_delta = metric_delta.get(stat)
    if not isinstance(stat_delta, dict):
        return None
    return parse_signed_float(stat_delta.get(value_key))


def repeat_case_delta_stage_value(
    case_delta: dict[str, Any],
    stage: str,
    stat: str,
    value_key: str,
) -> float | None:
    stage_duration = case_delta.get("stage_duration_s")
    if not isinstance(stage_duration, dict):
        return None
    metric_delta = stage_duration.get(stage)
    if not isinstance(metric_delta, dict):
        return None
    stat_delta = metric_delta.get(stat)
    if not isinstance(stat_delta, dict):
        return None
    return parse_signed_float(stat_delta.get(value_key))


def stress_report_repeat_profile_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    repeat_profile = report.get("repeat_profile")
    if not isinstance(repeat_profile, dict):
        return None
    summary = repeat_profile.get("summary")
    if not isinstance(summary, dict):
        return None
    samples = repeat_profile.get("samples")
    if not isinstance(samples, list):
        return summary
    analyzed_samples = repeat_profile_analyzed_samples(
        [sample for sample in samples if isinstance(sample, dict)]
    )
    enriched: dict[str, Any] | None = None
    if not isinstance(summary.get("stage_duration_s"), dict):
        stage_stats = repeat_profile_stage_duration_stats(analyzed_samples)
        if stage_stats:
            enriched = dict(summary)
            enriched["stage_duration_s"] = stage_stats
    if not isinstance(summary.get("ocr_overlap_hidden_s"), dict):
        hidden_stats = repeat_profile_ocr_overlap_hidden_stats(analyzed_samples)
        if hidden_stats is not None:
            if enriched is None:
                enriched = dict(summary)
            enriched["ocr_overlap_hidden_s"] = hidden_stats
    return enriched or summary


def stress_report_candidate_scope_slugs(report: dict[str, Any]) -> list[str] | None:
    preset = report.get("preset")
    if not isinstance(preset, dict):
        return None
    only_slugs = preset.get("only")
    if not isinstance(only_slugs, list):
        return None
    slugs: list[str] = []
    seen: set[str] = set()
    for value in only_slugs:
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        slugs.append(value)
    return slugs


def stress_report_configuration_changes(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = (
        "execution",
        "profile_ocr_engine",
        "runner_ocr_cache",
        "extraction_cache",
        "prewarm_runtime",
        "repeat_profile_runs",
        "repeat_profile_warmups",
    )
    changes: list[dict[str, Any]] = []
    for field in fields:
        baseline_value = baseline_report.get(field)
        candidate_value = candidate_report.get(field)
        if baseline_value == candidate_value:
            continue
        changes.append(
            {
                "field": field,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
        )
    baseline_preset = stress_report_preset_label(baseline_report)
    candidate_preset = stress_report_preset_label(candidate_report)
    if baseline_preset != candidate_preset:
        changes.append(
            {
                "field": "preset",
                "baseline": baseline_preset,
                "candidate": candidate_preset,
            }
        )
    return changes


def stress_report_preset_label(report: dict[str, Any]) -> str | None:
    preset = report.get("preset")
    if not isinstance(preset, dict):
        return None
    name = preset.get("name")
    if not isinstance(name, str) or not name:
        return None
    version = preset.get("version")
    version_text = f"@v{version}" if isinstance(version, int) else ""
    only = preset.get("only")
    only_text = ""
    if isinstance(only, list):
        only_count = sum(1 for slug in only if isinstance(slug, str) and slug)
        only_text = f":only{only_count}" if only_count else ""
    return f"{name}{version_text}{only_text}"


def stress_signature_changed_fields(
    baseline_signature: dict[str, Any],
    candidate_signature: dict[str, Any],
) -> list[str]:
    fields = sorted(set(baseline_signature) | set(candidate_signature))
    return [
        field
        for field in fields
        if baseline_signature.get(field) != candidate_signature.get(field)
    ]


def signature_changed_field_counts(signature_changes: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for change in signature_changes:
        fields = change.get("changed_fields")
        if not isinstance(fields, list):
            continue
        for field in fields:
            if isinstance(field, str) and field:
                counts[field] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def stress_scalar_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    key: str,
    *,
    delta_key: str,
) -> dict[str, float] | None:
    baseline_value = parse_nonnegative_float(baseline.get(key))
    candidate_value = parse_nonnegative_float(candidate.get(key))
    if baseline_value is None or candidate_value is None:
        return None
    return {
        "baseline": round(baseline_value, 6),
        "candidate": round(candidate_value, 6),
        delta_key: round(candidate_value - baseline_value, 6),
    }


def stress_nested_metric_deltas(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    section_key: str,
    metric_keys: tuple[str, ...],
    stat_keys: tuple[str, ...],
    delta_key: str,
) -> dict[str, dict[str, dict[str, float]]]:
    baseline_section = baseline_summary.get(section_key)
    candidate_section = candidate_summary.get(section_key)
    if not isinstance(baseline_section, dict) or not isinstance(candidate_section, dict):
        return {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for metric in metric_keys:
        baseline_metric = baseline_section.get(metric)
        candidate_metric = candidate_section.get(metric)
        if not isinstance(baseline_metric, dict) or not isinstance(candidate_metric, dict):
            continue
        stat_deltas: dict[str, dict[str, float]] = {}
        for stat in stat_keys:
            metric_delta = stress_scalar_delta(baseline_metric, candidate_metric, stat, delta_key=delta_key)
            if metric_delta is not None:
                stat_deltas[stat] = metric_delta
        if stat_deltas:
            deltas[metric] = stat_deltas
    return deltas


def stress_stage_duration_deltas(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    metric_keys = stress_stage_duration_metric_keys(baseline_summary, candidate_summary)
    if not metric_keys:
        return {}
    return stress_nested_metric_deltas(
        baseline_summary,
        candidate_summary,
        section_key="stage_duration_s",
        metric_keys=metric_keys,
        stat_keys=("p95_duration_s", "max_duration_s"),
        delta_key="delta_s",
    )


def stress_stage_duration_metric_keys(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> tuple[str, ...]:
    baseline_stage = baseline_summary.get("stage_duration_s")
    candidate_stage = candidate_summary.get("stage_duration_s")
    if not isinstance(baseline_stage, dict) or not isinstance(candidate_stage, dict):
        return ()
    shared = {
        stage
        for stage in set(baseline_stage) & set(candidate_stage)
        if isinstance(stage, str) and stage
    }
    return tuple(sorted(shared, key=pipeline_stage_sort_key))


def pipeline_stage_sort_key(stage: str) -> tuple[int, str]:
    try:
        return (PIPELINE_STAGE_DISPLAY_ORDER.index(stage), stage)
    except ValueError:
        return (len(PIPELINE_STAGE_DISPLAY_ORDER), stage)


def stress_metric_group_delta(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    section_key: str,
    stat_keys: tuple[str, ...],
    delta_key: str,
) -> dict[str, dict[str, float]]:
    baseline_section = baseline_summary.get(section_key)
    candidate_section = candidate_summary.get(section_key)
    if not isinstance(baseline_section, dict) or not isinstance(candidate_section, dict):
        return {}
    deltas: dict[str, dict[str, float]] = {}
    for stat in stat_keys:
        metric_delta = stress_scalar_delta(baseline_section, candidate_section, stat, delta_key=delta_key)
        if metric_delta is not None:
            deltas[stat] = metric_delta
    return deltas


def stress_report_rows_by_slug(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        return {}
    rows_by_slug: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = row.get("slug")
        if isinstance(slug, str) and slug:
            rows_by_slug[slug] = row
    return rows_by_slug


def stress_report_slug_order(report: dict[str, Any]) -> list[str]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        return []
    slugs: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = row.get("slug")
        if isinstance(slug, str) and slug:
            slugs.append(slug)
    return slugs


def stress_row_latency_delta(
    slug: str,
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, Any] | None:
    baseline_total = parse_nonnegative_float(baseline_row.get("total_elapsed_s"))
    candidate_total = parse_nonnegative_float(candidate_row.get("total_elapsed_s"))
    if baseline_total is None or candidate_total is None:
        return None
    delta: dict[str, Any] = {
        "slug": slug,
        "baseline_total_elapsed_s": round(baseline_total, 6),
        "candidate_total_elapsed_s": round(candidate_total, 6),
        "total_elapsed_delta_s": round(candidate_total - baseline_total, 6),
    }
    stage_deltas = stress_row_stage_deltas(baseline_row, candidate_row)
    if stage_deltas:
        delta["stage_delta_s"] = stage_deltas
    baseline_ocr_calls = stress_row_ocr_engine_calls(baseline_row)
    candidate_ocr_calls = stress_row_ocr_engine_calls(candidate_row)
    if baseline_ocr_calls is not None:
        delta["baseline_ocr_engine_calls"] = round(baseline_ocr_calls, 6)
    if candidate_ocr_calls is not None:
        delta["candidate_ocr_engine_calls"] = round(candidate_ocr_calls, 6)
    ocr_total_values = stress_row_ocr_engine_metric_values(baseline_row, candidate_row, "total_s")
    if ocr_total_values is not None:
        baseline_ocr_total, candidate_ocr_total = ocr_total_values
        delta["baseline_ocr_engine_total_s"] = round(baseline_ocr_total, 6)
        delta["candidate_ocr_engine_total_s"] = round(candidate_ocr_total, 6)
        delta["ocr_engine_total_delta_s"] = round(candidate_ocr_total - baseline_ocr_total, 6)
    ocr_stage_deltas = stress_row_ocr_engine_stage_deltas(baseline_row, candidate_row)
    if ocr_stage_deltas:
        delta["ocr_engine_stage_delta_s"] = ocr_stage_deltas
    ocr_hidden_values = stress_row_ocr_overlap_hidden_values(baseline_row, candidate_row)
    if ocr_hidden_values is not None:
        baseline_hidden, candidate_hidden = ocr_hidden_values
        delta["baseline_ocr_overlap_hidden_s"] = round(baseline_hidden, 6)
        delta["candidate_ocr_overlap_hidden_s"] = round(candidate_hidden, 6)
        delta["ocr_overlap_hidden_delta_s"] = round(candidate_hidden - baseline_hidden, 6)
    return delta


def stress_row_expectation_change(
    slug: str,
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, Any] | None:
    baseline_passed = baseline_row.get("expectation_passed")
    candidate_passed = candidate_row.get("expectation_passed")
    if not isinstance(baseline_passed, bool) or not isinstance(candidate_passed, bool):
        return None
    change: dict[str, Any] = {
        "slug": slug,
        "baseline_expectation_passed": baseline_passed,
        "candidate_expectation_passed": candidate_passed,
    }
    for prefix, row in (("baseline", baseline_row), ("candidate", candidate_row)):
        observed_status = row.get("observed_status")
        if isinstance(observed_status, str) and observed_status:
            change[f"{prefix}_observed_status"] = observed_status
        expected_status = row.get("expected_status")
        if isinstance(expected_status, str) and expected_status:
            change[f"{prefix}_expected_status"] = expected_status
        issues = row.get("expectation_issues")
        if isinstance(issues, list):
            text_issues = [issue for issue in issues if isinstance(issue, str) and issue]
            if text_issues:
                change[f"{prefix}_expectation_issues"] = text_issues
    return change


def stress_row_stage_deltas(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, float]:
    baseline_stages = baseline_row.get("stages")
    candidate_stages = candidate_row.get("stages")
    if not isinstance(baseline_stages, dict) or not isinstance(candidate_stages, dict):
        return {}
    deltas: dict[str, float] = {}
    for stage in sorted(set(baseline_stages) & set(candidate_stages)):
        baseline_stage = parse_nonnegative_float(baseline_stages.get(stage))
        candidate_stage = parse_nonnegative_float(candidate_stages.get(stage))
        if baseline_stage is None or candidate_stage is None:
            continue
        deltas[str(stage)] = round(candidate_stage - baseline_stage, 6)
    return deltas


def stress_row_ocr_engine_stage_deltas(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in OCR_ENGINE_STAGE_MAX_KEYS:
        delta = stress_row_ocr_engine_metric_delta(baseline_row, candidate_row, key)
        if delta is not None:
            deltas[key] = delta
    return deltas


def stress_row_ocr_engine_calls(row: dict[str, Any]) -> float | None:
    profile = row.get("ocr_engine_profile")
    if not isinstance(profile, dict):
        return None
    return parse_nonnegative_count_metric(profile.get("calls"))


def stress_row_ocr_engine_metric_delta(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    metric: str,
) -> float | None:
    values = stress_row_ocr_engine_metric_values(baseline_row, candidate_row, metric)
    if values is None:
        return None
    baseline_value, candidate_value = values
    return round(candidate_value - baseline_value, 6)


def stress_row_ocr_engine_metric_values(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    metric: str,
) -> tuple[float, float] | None:
    baseline_profile = baseline_row.get("ocr_engine_profile")
    candidate_profile = candidate_row.get("ocr_engine_profile")
    if not isinstance(baseline_profile, dict) or not isinstance(candidate_profile, dict):
        return None
    baseline_value = parse_nonnegative_float(baseline_profile.get(metric))
    candidate_value = parse_nonnegative_float(candidate_profile.get(metric))
    if baseline_value is None or candidate_value is None:
        return None
    return baseline_value, candidate_value


def stress_row_ocr_overlap_hidden_values(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> tuple[float, float] | None:
    baseline_hidden = stress_row_ocr_overlap_hidden_value(baseline_row)
    candidate_hidden = stress_row_ocr_overlap_hidden_value(candidate_row)
    if baseline_hidden is None or candidate_hidden is None:
        return None
    return baseline_hidden, candidate_hidden


def stress_row_ocr_overlap_hidden_value(row: dict[str, Any]) -> float | None:
    hidden_s = parse_nonnegative_float(row.get("ocr_overlap_hidden_s"))
    if hidden_s is not None:
        return hidden_s
    return ocr_overlap_hidden_seconds(row)


def ranked_latency_deltas(
    latency_deltas: list[dict[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = [
        delta
        for delta in latency_deltas
        if isinstance(delta.get("total_elapsed_delta_s"), (int, float))
    ]
    ranked.sort(
        key=lambda delta: (
            float(delta["total_elapsed_delta_s"]),
            str(delta.get("slug") or ""),
        ),
        reverse=reverse,
    )
    return ranked[: max(0, limit)]


def ranked_ocr_engine_total_deltas(
    latency_deltas: list[dict[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = [
        delta
        for delta in latency_deltas
        if isinstance(delta.get("ocr_engine_total_delta_s"), (int, float))
    ]
    ranked.sort(
        key=lambda delta: (
            float(delta["ocr_engine_total_delta_s"]),
            str(delta.get("slug") or ""),
        ),
        reverse=reverse,
    )
    return ranked[: max(0, limit)]


def ranked_ocr_overlap_hidden_deltas(
    latency_deltas: list[dict[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = [
        delta
        for delta in latency_deltas
        if isinstance(delta.get("ocr_overlap_hidden_delta_s"), (int, float))
    ]
    ranked.sort(
        key=lambda delta: (
            float(delta["ocr_overlap_hidden_delta_s"]),
            str(delta.get("slug") or ""),
        ),
        reverse=reverse,
    )
    return ranked[: max(0, limit)]


def run_generation_prewarm(*, runner_ocr_cache: bool, extraction_cache: bool) -> dict[str, Any]:
    with temporary_cache_env(runner_ocr_cache=runner_ocr_cache, extraction_cache=extraction_cache):
        return prewarm_generation_runtime()


def prewarm_runtime_ok(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "ok"


def stress_runtime_config(*, runner_ocr_cache: bool, extraction_cache: bool) -> dict[str, Any]:
    with temporary_cache_env(runner_ocr_cache=runner_ocr_cache, extraction_cache=extraction_cache):
        return {
            "pipeline_version": get_pipeline_version(),
            "runtime_dependencies": dict(pipeline_version_dependency_versions()),
            "ocr": ocr_runtime_config(),
            "generation_env": generation_env_config(),
        }


def select_cases(cases: list[dict[str, Any]], only_slugs: list[str]) -> list[dict[str, Any]]:
    if not only_slugs:
        return cases
    wanted = set(only_slugs)
    selected = [case for case in cases if case.get("slug") in wanted]
    missing = sorted(wanted - {case.get("slug") for case in selected})
    if missing:
        raise ValueError(f"Unknown stress slug(s): {', '.join(missing)}")
    return selected


def summarize_manifest_contracts(cases: list[dict[str, Any]]) -> dict[str, Any]:
    call_contract_slugs: list[str] = []
    missing_call_contract_slugs: list[str] = []
    positive_call_contract_slugs: list[str] = []
    zero_call_contract_slugs: list[str] = []
    count_contract_slugs: list[str] = []
    positive_call_rows_without_count_contract: list[str] = []
    invalid_count_contract_rows: list[dict[str, Any]] = []
    metric_counts: dict[str, int] = {}

    for case in cases:
        slug = str(case.get("slug"))
        expect = case.get("expect") if isinstance(case.get("expect"), dict) else {}
        max_ocr_calls = parse_nonnegative_count_metric(expect.get("max_ocr_engine_calls"))
        if max_ocr_calls is None:
            missing_call_contract_slugs.append(slug)
        else:
            call_contract_slugs.append(slug)
            if max_ocr_calls == 0:
                zero_call_contract_slugs.append(slug)
            elif max_ocr_calls > 0:
                positive_call_contract_slugs.append(slug)

        count_metrics: set[str] = set()
        raw_count_budgets = expect.get("max_ocr_engine_counts")
        if isinstance(raw_count_budgets, dict):
            for raw_metric, raw_budget in raw_count_budgets.items():
                if not isinstance(raw_metric, str):
                    invalid_count_contract_rows.append({"slug": slug, "metric": raw_metric})
                    continue
                try:
                    metric = normalize_ocr_engine_count_metric(raw_metric)
                except ValueError:
                    invalid_count_contract_rows.append({"slug": slug, "metric": raw_metric})
                    continue
                if parse_nonnegative_count_metric(raw_budget) is None:
                    invalid_count_contract_rows.append({"slug": slug, "metric": raw_metric})
                    continue
                count_metrics.add(metric)
            if count_metrics:
                count_contract_slugs.append(slug)
                for metric in count_metrics:
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1

        if max_ocr_calls is not None and max_ocr_calls > 0 and not count_metrics:
            positive_call_rows_without_count_contract.append(slug)

    return {
        "total_cases": len(cases),
        "ocr_call_contract_rows": len(call_contract_slugs),
        "ocr_call_contract_missing_rows": missing_call_contract_slugs,
        "ocr_positive_call_contract_rows": len(positive_call_contract_slugs),
        "ocr_zero_call_contract_rows": len(zero_call_contract_slugs),
        "ocr_count_contract_rows": len(count_contract_slugs),
        "ocr_count_contract_slugs": count_contract_slugs,
        "ocr_positive_call_rows_without_count_contract": positive_call_rows_without_count_contract,
        "ocr_count_contract_metric_counts": dict(sorted(metric_counts.items())),
        "invalid_ocr_count_contract_rows": invalid_count_contract_rows,
    }


def build_manifest_contract_budget_summary(
    manifest_contracts: dict[str, Any],
    *,
    min_ocr_call_contract_rows: int | None = None,
    min_ocr_count_contract_rows: int | None = None,
    max_positive_ocr_call_only_rows: int | None = None,
    fail_on_invalid_ocr_count_contracts: bool = False,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    ocr_call_contract_rows = manifest_contract_count(manifest_contracts, "ocr_call_contract_rows")
    ocr_count_contract_rows = manifest_contract_count(manifest_contracts, "ocr_count_contract_rows")
    positive_call_only_rows = manifest_contract_list(
        manifest_contracts,
        "ocr_positive_call_rows_without_count_contract",
    )
    invalid_count_contract_rows = manifest_contract_list(
        manifest_contracts,
        "invalid_ocr_count_contract_rows",
    )

    if min_ocr_call_contract_rows is not None:
        minimum = int(min_ocr_call_contract_rows)
        if ocr_call_contract_rows < minimum:
            violations.append(
                {
                    "kind": "ocr_call_contract_rows_below_min",
                    "ocr_call_contract_rows": ocr_call_contract_rows,
                    "min_ocr_call_contract_rows": minimum,
                    "missing_rows": manifest_contract_list(
                        manifest_contracts,
                        "ocr_call_contract_missing_rows",
                    ),
                }
            )
    if min_ocr_count_contract_rows is not None:
        minimum = int(min_ocr_count_contract_rows)
        if ocr_count_contract_rows < minimum:
            violations.append(
                {
                    "kind": "ocr_count_contract_rows_below_min",
                    "ocr_count_contract_rows": ocr_count_contract_rows,
                    "min_ocr_count_contract_rows": minimum,
                }
            )
    if max_positive_ocr_call_only_rows is not None:
        maximum = int(max_positive_ocr_call_only_rows)
        if len(positive_call_only_rows) > maximum:
            violations.append(
                {
                    "kind": "positive_ocr_call_only_rows_above_max",
                    "positive_ocr_call_only_rows": len(positive_call_only_rows),
                    "max_positive_ocr_call_only_rows": maximum,
                    "rows": positive_call_only_rows,
                }
            )
    if fail_on_invalid_ocr_count_contracts and invalid_count_contract_rows:
        violations.append(
            {
                "kind": "invalid_ocr_count_contracts",
                "invalid_ocr_count_contract_rows": invalid_count_contract_rows,
            }
        )

    summary: dict[str, Any] = {
        "passed": not violations,
        "violations": violations,
    }
    if min_ocr_call_contract_rows is not None:
        summary["min_ocr_call_contract_rows"] = int(min_ocr_call_contract_rows)
    if min_ocr_count_contract_rows is not None:
        summary["min_ocr_count_contract_rows"] = int(min_ocr_count_contract_rows)
    if max_positive_ocr_call_only_rows is not None:
        summary["max_positive_ocr_call_only_rows"] = int(max_positive_ocr_call_only_rows)
    if fail_on_invalid_ocr_count_contracts:
        summary["fail_on_invalid_ocr_count_contracts"] = True
    return summary


def manifest_contract_count(manifest_contracts: dict[str, Any], key: str) -> int:
    value = manifest_contracts.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def manifest_contract_list(manifest_contracts: dict[str, Any], key: str) -> list[Any]:
    value = manifest_contracts.get(key)
    return [item for item in value] if isinstance(value, list) else []


def run_stress_case(
    case: dict[str, Any],
    out_dir: Path,
    *,
    timeout_seconds: float,
    write_debug: bool,
    profile_ocr_engine: bool = False,
    runner_ocr_cache: bool = True,
    extraction_cache: bool = True,
    execution: str = "subprocess",
    python_executable: str,
) -> dict[str, Any]:
    slug = require_string(case, "slug")
    image = Path(require_string(case, "image"))
    expect = case.get("expect") if isinstance(case.get("expect"), dict) else {}
    expected_status = str(expect.get("status", "complete"))

    if not image.exists():
        row = base_row(case, expected_status=expected_status, observed_status="missing")
        row["expectation_issues"] = [] if expected_status == "missing" else [f"expected {expected_status}, got missing"]
        row["expectation_passed"] = not row["expectation_issues"]
        return row

    if execution == "in-process":
        return run_stress_case_in_process(
            case,
            image,
            out_dir,
            expected_status=expected_status,
            timeout_seconds=timeout_seconds,
            write_debug=write_debug,
            profile_ocr_engine=profile_ocr_engine,
            runner_ocr_cache=runner_ocr_cache,
            extraction_cache=extraction_cache,
        )
    if execution != "subprocess":
        raise ValueError(f"Unsupported stress execution mode: {execution}")

    output_path = out_dir / f"{slug}.geojson"
    command = build_cli_command(
        case,
        image,
        output_path,
        out_dir,
        write_debug,
        profile_ocr_engine,
        python_executable,
    )
    started = time.perf_counter()
    try:
        run_kwargs: dict[str, Any] = {
            "text": True,
            "capture_output": True,
            "timeout": timeout_seconds,
            "check": False,
        }
        subprocess_env = subprocess_cache_env(
            runner_ocr_cache=runner_ocr_cache,
            extraction_cache=extraction_cache,
        )
        if subprocess_env is not None:
            run_kwargs["env"] = subprocess_env
        completed = subprocess.run(command, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        wall_s = round(time.perf_counter() - started, 6)
        row = base_row(case, expected_status=expected_status, observed_status="timeout")
        row.update(
            {
                "returncode": None,
                "wall_s": wall_s,
                "error": f"Timed out after {timeout_seconds:g}s",
                "stdout": truncate_text(exc.stdout),
                "stderr": truncate_text(exc.stderr),
                "command": command,
            }
        )
        row["expectation_issues"] = check_expectations(row, expect)
        row["expectation_passed"] = not row["expectation_issues"]
        return row

    wall_s = round(time.perf_counter() - started, 6)
    summary, parse_error = parse_summary(completed.stdout)
    row = row_from_process(
        case,
        command=command,
        completed=completed,
        wall_s=wall_s,
        summary=summary,
        parse_error=parse_error,
        expected_status=expected_status,
    )
    row["runner_ocr_cache"] = runner_ocr_cache
    row["extraction_cache"] = extraction_cache
    attach_geojson_geometry_summary(row, output_path)
    row["expectation_issues"] = check_expectations(row, expect)
    row["expectation_passed"] = not row["expectation_issues"]
    return row


def run_stress_case_in_process(
    case: dict[str, Any],
    image: Path,
    out_dir: Path,
    *,
    expected_status: str,
    timeout_seconds: float,
    write_debug: bool,
    profile_ocr_engine: bool,
    runner_ocr_cache: bool,
    extraction_cache: bool,
) -> dict[str, Any]:
    slug = require_string(case, "slug")
    expect = case.get("expect") if isinstance(case.get("expect"), dict) else {}
    output_path = out_dir / f"{slug}.geojson"
    debug_dir = out_dir / f"{slug}-debug" if write_debug else None
    city = case.get("city")
    city_input = city.strip() if isinstance(city, str) and city.strip() else None
    filename_hint = case.get("filename_hint")
    if not isinstance(filename_hint, str):
        filename_hint = GENERIC_FILENAME_HINT
    source_was_svg = bool(case.get("source_was_svg", False))
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    ocr_engine_events: list[dict[str, Any]] | None = None

    def progress(event: dict[str, Any]) -> None:
        events.append({"elapsed_s": round(time.perf_counter() - started, 6), **event})

    def event_profile() -> dict[str, Any]:
        return {
            "total_elapsed_s": round(time.perf_counter() - started, 6),
            "stage_elapsed_s": stage_elapsed_seconds(events),
            "events": events,
        }

    def run_build_boundary():
        return build_boundary(
            image,
            city_input,
            output_path,
            debug_dir=debug_dir,
            options=BoundaryBuildOptions(
                allow_catalog=not case.get("no_catalog", True),
                filename_hint=filename_hint,
                source_was_svg=source_was_svg,
            ),
            progress=progress,
        )

    command = build_in_process_command_summary(
        case,
        image,
        output_path,
        debug_dir,
        runner_ocr_cache=runner_ocr_cache,
        extraction_cache=extraction_cache,
    )

    def run_build_boundary_with_cache_settings():
        with temporary_cache_env(runner_ocr_cache=runner_ocr_cache, extraction_cache=extraction_cache):
            return run_build_boundary()

    try:
        if profile_ocr_engine:
            with collect_rapidocr_profiles() as collected:
                ocr_engine_events = collected
                result = run_build_boundary_with_cache_settings()
        else:
            result = run_build_boundary_with_cache_settings()
        wall_s = round(time.perf_counter() - started, 6)
        summary = dict(result.summary)
        summary["pipeline_version"] = get_pipeline_version()
        summary["event_profile"] = event_profile()
        if profile_ocr_engine:
            summary["ocr_engine_profile"] = summarize_rapidocr_profile_events(ocr_engine_events)
        completed = subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    except Exception as exc:
        wall_s = round(time.perf_counter() - started, 6)
        summary = {
            "status": "failed",
            "error": str(exc),
            "pipeline_version": get_pipeline_version(),
            "event_profile": event_profile(),
        }
        if profile_ocr_engine:
            summary["ocr_engine_profile"] = summarize_rapidocr_profile_events(ocr_engine_events)
        completed = subprocess.CompletedProcess(command, 1, stdout="", stderr=f"map-boundary-builder: error: {exc}")

    row = row_from_process(
        case,
        command=command,
        completed=completed,
        wall_s=wall_s,
        summary=summary,
        parse_error=None,
        expected_status=expected_status,
    )
    row["execution"] = "in-process"
    row["timeout_seconds"] = timeout_seconds
    row["runner_ocr_cache"] = runner_ocr_cache
    row["extraction_cache"] = extraction_cache
    attach_geojson_geometry_summary(row, output_path)
    row["expectation_issues"] = check_expectations(row, expect)
    row["expectation_passed"] = not row["expectation_issues"]
    return row


def build_in_process_command_summary(
    case: dict[str, Any],
    image: Path,
    output_path: Path,
    debug_dir: Path | None,
    *,
    runner_ocr_cache: bool,
    extraction_cache: bool,
) -> list[str]:
    command = [
        "in-process",
        "--image",
        str(image),
        "--output",
        str(output_path),
    ]
    if case.get("no_catalog", True):
        command.append("--no-catalog")
    city = case.get("city")
    if isinstance(city, str) and city.strip():
        command.extend(["--city", city.strip()])
    filename_hint = case.get("filename_hint")
    if not isinstance(filename_hint, str):
        filename_hint = GENERIC_FILENAME_HINT
    command.extend(["--filename-hint", filename_hint])
    if case.get("source_was_svg", False):
        command.append("--source-was-svg")
    if debug_dir is not None:
        command.extend(["--debug-dir", str(debug_dir)])
    if not runner_ocr_cache:
        command.append("--disable-ocr-cache")
    if not extraction_cache:
        command.append("--disable-extraction-cache")
    return command


def subprocess_cache_env(*, runner_ocr_cache: bool, extraction_cache: bool) -> dict[str, str] | None:
    if runner_ocr_cache and extraction_cache:
        return None
    env = dict(os.environ)
    if not runner_ocr_cache:
        env[RUNNER_OCR_CACHE_ENV] = "0"
    if not extraction_cache:
        env[EXTRACTION_CACHE_ENV] = "0"
    return env


@contextmanager
def temporary_cache_env(*, runner_ocr_cache: bool, extraction_cache: bool):
    if runner_ocr_cache and extraction_cache:
        yield
        return
    previous_runner_ocr = os.environ.get(RUNNER_OCR_CACHE_ENV)
    previous_extraction = os.environ.get(EXTRACTION_CACHE_ENV)
    if not runner_ocr_cache:
        os.environ[RUNNER_OCR_CACHE_ENV] = "0"
    if not extraction_cache:
        os.environ[EXTRACTION_CACHE_ENV] = "0"
    try:
        yield
    finally:
        if previous_runner_ocr is None:
            os.environ.pop(RUNNER_OCR_CACHE_ENV, None)
        else:
            os.environ[RUNNER_OCR_CACHE_ENV] = previous_runner_ocr
        if previous_extraction is None:
            os.environ.pop(EXTRACTION_CACHE_ENV, None)
        else:
            os.environ[EXTRACTION_CACHE_ENV] = previous_extraction


def build_cli_command(
    case: dict[str, Any],
    image: Path,
    output_path: Path,
    out_dir: Path,
    write_debug: bool,
    profile_ocr_engine: bool,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "map_boundary_builder.cli",
        "--image",
        str(image),
        "--output",
        str(output_path),
        "--print-summary",
        "--profile-events",
    ]
    if case.get("no_catalog", True):
        command.append("--no-catalog")
    city = case.get("city")
    if isinstance(city, str) and city.strip():
        command.extend(["--city", city.strip()])
    filename_hint = case.get("filename_hint")
    if not isinstance(filename_hint, str):
        filename_hint = GENERIC_FILENAME_HINT
    command.extend(["--filename-hint", filename_hint])
    if case.get("source_was_svg", False):
        command.append("--source-was-svg")
    if write_debug:
        command.extend(["--debug-dir", str(out_dir / f"{case['slug']}-debug")])
    if profile_ocr_engine:
        command.append("--profile-ocr-engine")
    return command


def row_from_process(
    case: dict[str, Any],
    *,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    wall_s: float,
    summary: dict[str, Any],
    parse_error: str | None,
    expected_status: str,
) -> dict[str, Any]:
    observed_status = observed_status_from_process(completed.returncode, summary)
    event_profile = summary.get("event_profile") if isinstance(summary.get("event_profile"), dict) else {}
    events = event_profile_events(event_profile)
    extraction_details = latest_event_details(events, stage="extract", required_key="style")
    extract_start_details = latest_event_details(events, stage="extract", required_key="width")
    complete_details = latest_event_details(events, stage="complete")
    ocr_details, ocr_label_event = latest_ocr_label_details(events)
    ocr_label_events = ocr_label_event_summaries(events)
    route_ui_reject_details = latest_message_details(
        events,
        stage="georeference",
        message="Rejecting ride-route UI",
    )
    non_map_ui_reject_details = latest_message_details(
        events,
        stage="georeference",
        message="Rejecting non-map app UI",
    )
    thematic_map_reject_details = latest_message_details(
        events,
        stage="georeference",
        message="Rejecting thematic map",
    )
    raw_ocr_engine_profile = summary.get("ocr_engine_profile")
    ocr_engine_profile = raw_ocr_engine_profile if isinstance(raw_ocr_engine_profile, dict) else None
    row = base_row(case, expected_status=expected_status, observed_status=observed_status)
    row.update(
        {
            "returncode": completed.returncode,
            "wall_s": wall_s,
            "pipeline_version": summary.get("pipeline_version"),
            "city": summary.get("city"),
            "style": summary.get("style") or extraction_details.get("style") or complete_details.get("style"),
            "source": summary.get("georeference_source"),
            "catalog_slug": summary.get("catalog_slug"),
            "catalog_shape_iou": summary.get("catalog_shape_iou"),
            "catalog_shape_margin": summary.get("catalog_shape_margin"),
            "catalog_area_ratio": summary.get("catalog_area_ratio"),
            "combined_confidence": summary.get("combined_confidence"),
            "georeference_confidence": summary.get("georeference_confidence"),
            "control_points": summary.get("control_points"),
            "road_match_score": summary.get("road_match_score"),
            "road_match_base_score": summary.get("road_match_base_score"),
            "road_match_sampled_points": summary.get("road_match_sampled_points"),
            "road_match_elapsed_s": summary.get("road_match_elapsed_s"),
            "bbox": summary.get("bbox"),
            "geojson_geometry_hash": None,
            "geojson_coordinate_count": None,
            "image_width": first_present_number(
                summary.get("image_width"),
                complete_details.get("image_width"),
                extract_start_details.get("width"),
            ),
            "image_height": first_present_number(
                summary.get("image_height"),
                complete_details.get("image_height"),
                extract_start_details.get("height"),
            ),
            "coverage_ratio": first_present_number(
                summary.get("coverage_ratio"),
                extraction_details.get("coverage_ratio"),
                complete_details.get("coverage_ratio"),
            ),
            "contour_count": first_present_number(extraction_details.get("contour_count")),
            "ocr_label_count": first_present_number(ocr_details.get("label_count")),
            "ocr_top_labels": (
                ocr_details.get("top_labels") if isinstance(ocr_details.get("top_labels"), list) else None
            ),
            "ocr_label_event": ocr_label_event,
            "ocr_label_events": ocr_label_events,
            "route_ui_categories": (
                route_ui_reject_details.get("route_ui_categories")
                if isinstance(route_ui_reject_details.get("route_ui_categories"), list)
                else None
            ),
            "route_metric_labels": (
                route_ui_reject_details.get("route_metric_labels")
                if isinstance(route_ui_reject_details.get("route_metric_labels"), list)
                else None
            ),
            "non_map_ui_categories": (
                non_map_ui_reject_details.get("non_map_ui_categories")
                if isinstance(non_map_ui_reject_details.get("non_map_ui_categories"), list)
                else None
            ),
            "non_map_ui_labels": (
                non_map_ui_reject_details.get("non_map_ui_labels")
                if isinstance(non_map_ui_reject_details.get("non_map_ui_labels"), list)
                else None
            ),
            "thematic_map_labels": (
                thematic_map_reject_details.get("thematic_map_labels")
                if isinstance(thematic_map_reject_details.get("thematic_map_labels"), list)
                else None
            ),
            "ocr_full_detail_retry": any(
                event.get("message") == "Full-detail map labels read" for event in ocr_label_events
            ),
            "ocr_engine_profile": ocr_engine_profile,
            "total_elapsed_s": event_profile.get("total_elapsed_s"),
            "stages": (
                event_profile.get("stage_elapsed_s")
                if isinstance(event_profile.get("stage_elapsed_s"), dict)
                else {}
            ),
            "georeference_events": stage_event_segments(
                event_profile,
                stage="georeference",
            ),
            "error": summary.get("error"),
            "stdout_json_error": parse_error,
            "stderr": truncate_text(completed.stderr),
            "command": command,
        }
    )
    ocr_hidden_s = ocr_overlap_hidden_seconds(row)
    if ocr_hidden_s is not None:
        row["ocr_overlap_hidden_s"] = ocr_hidden_s
    return row


def attach_geojson_geometry_summary(row: dict[str, Any], output_path: Path) -> None:
    row.update(geojson_geometry_summary(output_path))


def geojson_geometry_summary(output_path: Path) -> dict[str, Any]:
    empty = {"geojson_geometry_hash": None, "geojson_coordinate_count": None}
    if not output_path.exists():
        return empty
    try:
        data = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return empty
    geometries = geojson_geometries(data)
    if not geometries:
        return empty
    normalized = [normalize_geojson_geometry(geometry) for geometry in geometries]
    normalized = [geometry for geometry in normalized if geometry is not None]
    if not normalized:
        return empty
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return {
        "geojson_geometry_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
        "geojson_coordinate_count": sum(geojson_coordinate_count(geometry) for geometry in normalized),
        **geojson_road_match_summary(data),
    }


def geojson_road_match_summary(value: Any) -> dict[str, Any]:
    properties = first_geojson_feature_properties(value)
    if properties is None:
        return {}
    return {
        key: properties.get(key)
        for key in (
            "road_match_score",
            "road_match_base_score",
            "road_match_sampled_points",
            "road_match_elapsed_s",
        )
        if key in properties
    }


def first_geojson_feature_properties(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("type") == "Feature":
        properties = value.get("properties")
        return properties if isinstance(properties, dict) else None
    if value.get("type") != "FeatureCollection":
        return None
    features = value.get("features")
    if not isinstance(features, list):
        return None
    for feature in features:
        properties = first_geojson_feature_properties(feature)
        if properties is not None:
            return properties
    return None


def geojson_geometries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    value_type = value.get("type")
    if value_type == "FeatureCollection":
        features = value.get("features")
        if not isinstance(features, list):
            return []
        geometries: list[dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            geometries.extend(geojson_geometries(feature))
        return geometries
    if value_type == "Feature":
        geometry = value.get("geometry")
        return [geometry] if isinstance(geometry, dict) else []
    if value_type == "GeometryCollection":
        geometries = value.get("geometries")
        if not isinstance(geometries, list):
            return []
        return [geometry for geometry in geometries if isinstance(geometry, dict)]
    return [value] if isinstance(value.get("coordinates"), list) else []


def normalize_geojson_geometry(geometry: dict[str, Any]) -> dict[str, Any] | None:
    geometry_type = geometry.get("type")
    if not isinstance(geometry_type, str):
        return None
    if geometry_type == "GeometryCollection":
        geometries = geometry.get("geometries")
        if not isinstance(geometries, list):
            return None
        normalized_geometries = [
            normalized
            for item in geometries
            if isinstance(item, dict)
            for normalized in [normalize_geojson_geometry(item)]
            if normalized is not None
        ]
        return {"type": geometry_type, "geometries": normalized_geometries}
    coordinates = normalize_geojson_coordinates(geometry.get("coordinates"))
    if coordinates is None:
        return None
    return {"type": geometry_type, "coordinates": coordinates}


def normalize_geojson_coordinates(value: Any) -> Any:
    if is_geojson_position(value):
        return [round(float(coordinate), 6) for coordinate in value]
    if isinstance(value, list):
        normalized = [normalize_geojson_coordinates(item) for item in value]
        if any(item is None for item in normalized):
            return None
        return normalized
    return None


def is_geojson_position(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False
    return all(isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool) for coordinate in value)


def geojson_coordinate_count(geometry: dict[str, Any]) -> int:
    if geometry.get("type") == "GeometryCollection":
        geometries = geometry.get("geometries")
        if not isinstance(geometries, list):
            return 0
        return sum(geojson_coordinate_count(item) for item in geometries if isinstance(item, dict))
    return geojson_coordinate_count_from_coordinates(geometry.get("coordinates"))


def geojson_coordinate_count_from_coordinates(value: Any) -> int:
    if is_geojson_position(value):
        return 1
    if isinstance(value, list):
        return sum(geojson_coordinate_count_from_coordinates(item) for item in value)
    return 0


def event_profile_events(event_profile: dict[str, Any]) -> list[dict[str, Any]]:
    events = event_profile.get("events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def stage_event_segments(
    event_profile: dict[str, Any],
    *,
    stage: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    events = event_profile_events(event_profile)
    total_elapsed_s = parse_nonnegative_float(event_profile.get("total_elapsed_s"))
    segments: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("stage") != stage:
            continue
        started_s = parse_nonnegative_float(event.get("elapsed_s"))
        next_elapsed_s = stage_next_event_elapsed_s(
            events,
            index,
            total_elapsed_s=total_elapsed_s,
        )
        segment: dict[str, Any] = {}
        message = event.get("message")
        if isinstance(message, str) and message:
            segment["message"] = message
        if started_s is not None:
            segment["at_s"] = round(started_s, 6)
        if started_s is not None and next_elapsed_s is not None:
            segment["elapsed_s"] = round(max(0.0, next_elapsed_s - started_s), 6)
        details = compact_stage_event_details(event.get("details"))
        if details:
            segment["details"] = details
        if segment:
            segments.append(segment)
    return segments[: max(0, limit)]


def stage_next_event_elapsed_s(
    events: list[dict[str, Any]],
    index: int,
    *,
    total_elapsed_s: float | None,
) -> float | None:
    for next_event in events[index + 1 :]:
        next_elapsed_s = parse_nonnegative_float(next_event.get("elapsed_s"))
        if next_elapsed_s is not None:
            return next_elapsed_s
    return total_elapsed_s


def compact_stage_event_details(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    details: dict[str, Any] = {}
    for key in (
        "candidates",
        "label_count",
        "top_labels",
        "route_ui_categories",
        "non_map_ui_categories",
        "thematic_map_labels",
    ):
        compacted = compact_stage_event_detail_value(value.get(key))
        if compacted is not None:
            details[key] = compacted
    return details


def compact_stage_event_detail_value(value: Any) -> Any:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, list):
        return None
    compacted: list[Any] = []
    for item in value[:5]:
        if isinstance(item, (str, int, float)) and not isinstance(item, bool):
            compacted.append(item)
    return compacted if compacted else None


def latest_event_details(
    events: list[dict[str, Any]],
    *,
    stage: str | None = None,
    required_key: str | None = None,
) -> dict[str, Any]:
    for event in reversed(events):
        if stage is not None and event.get("stage") != stage:
            continue
        details = event.get("details")
        if not isinstance(details, dict):
            continue
        if required_key is not None and required_key not in details:
            continue
        return details
    return {}


def latest_message_details(
    events: list[dict[str, Any]],
    *,
    stage: str | None = None,
    message: str,
) -> dict[str, Any]:
    for event in reversed(events):
        if stage is not None and event.get("stage") != stage:
            continue
        if event.get("message") != message:
            continue
        details = event.get("details")
        if isinstance(details, dict):
            return details
    return {}


def latest_ocr_label_details(events: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    for event in reversed(events):
        if event.get("stage") != "ocr":
            continue
        details = event.get("details")
        if not isinstance(details, dict) or "label_count" not in details:
            continue
        message = event.get("message")
        return details, message if isinstance(message, str) else None
    return {}, None


def ocr_label_event_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for event in events:
        if event.get("stage") != "ocr":
            continue
        details = event.get("details")
        if not isinstance(details, dict) or "label_count" not in details:
            continue
        message = event.get("message")
        summary: dict[str, Any] = {
            "message": message if isinstance(message, str) else None,
            "label_count": details.get("label_count"),
        }
        top_labels = details.get("top_labels")
        if isinstance(top_labels, list):
            summary["top_labels"] = top_labels
        summaries.append(summary)
    return summaries


def first_present_number(*values: Any) -> int | float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def base_row(case: dict[str, Any], *, expected_status: str, observed_status: str) -> dict[str, Any]:
    filename_hint = case.get("filename_hint")
    if not isinstance(filename_hint, str):
        filename_hint = GENERIC_FILENAME_HINT
    return {
        "slug": case.get("slug"),
        "image": case.get("image"),
        "no_catalog": bool(case.get("no_catalog", True)),
        "filename_hint": filename_hint,
        "source_was_svg": bool(case.get("source_was_svg", False)),
        "expected_status": expected_status,
        "observed_status": observed_status,
        "status": observed_status,
    }


def observed_status_from_process(returncode: int, summary: dict[str, Any]) -> str:
    if summary.get("status") == "failed":
        return "failed"
    if returncode == 0:
        return "complete"
    return "failed"


def check_expectations(row: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected_status = str(expect.get("status", "complete"))
    if row.get("observed_status") != expected_status:
        issues.append(f"expected {expected_status}, got {row.get('observed_status')}")
        return issues

    if expected_status != "complete":
        expected_error = expect.get("error_contains")
        error = row.get("error") or ""
        if isinstance(expected_error, str) and expected_error not in error:
            issues.append(f"error did not contain {expected_error!r}")
        append_min_ocr_labels_expectation_issue(row, expect, issues)
        append_ocr_top_labels_expectation_issues(row, expect, issues)
        append_route_ui_expectation_issues(row, expect, issues)
        append_non_map_ui_expectation_issues(row, expect, issues)
        append_thematic_map_expectation_issues(row, expect, issues)
        append_total_elapsed_expectation_issue(row, expect, issues)
        append_max_ocr_engine_calls_expectation_issue(row, expect, issues)
        append_max_ocr_engine_counts_expectation_issues(row, expect, issues)
        return issues

    source_equals = expect.get("source_equals")
    source_prefix = expect.get("source_prefix")
    source = row.get("source") or ""
    if isinstance(source_equals, str) and str(source) != source_equals:
        issues.append(f"source {source!r} did not equal {source_equals!r}")
    elif isinstance(source_prefix, str) and not str(source).startswith(source_prefix):
        issues.append(f"source {source!r} did not start with {source_prefix!r}")

    city_equals = expect.get("city_equals")
    if isinstance(city_equals, str) and row.get("city") != city_equals:
        issues.append(f"city {row.get('city')!r} did not equal {city_equals!r}")

    catalog_slug_equals = expect.get("catalog_slug_equals")
    if isinstance(catalog_slug_equals, str) and row.get("catalog_slug") != catalog_slug_equals:
        issues.append(f"catalog_slug {row.get('catalog_slug')!r} did not equal {catalog_slug_equals!r}")

    min_control_points = expect.get("min_control_points")
    if isinstance(min_control_points, int):
        control_points = row.get("control_points")
        if not isinstance(control_points, int) or control_points < min_control_points:
            issues.append(f"control_points {control_points!r} below {min_control_points}")

    append_min_ocr_labels_expectation_issue(row, expect, issues)
    append_ocr_top_labels_expectation_issues(row, expect, issues)
    append_max_ocr_engine_calls_expectation_issue(row, expect, issues)
    append_max_ocr_engine_counts_expectation_issues(row, expect, issues)

    append_min_confidence_expectation_issue(
        row,
        expect,
        row_key="combined_confidence",
        expect_key="min_combined_confidence",
        issues=issues,
    )
    append_min_confidence_expectation_issue(
        row,
        expect,
        row_key="georeference_confidence",
        expect_key="min_georeference_confidence",
        issues=issues,
    )

    append_total_elapsed_expectation_issue(row, expect, issues)

    expected_bbox = expect.get("bbox_approx")
    max_bbox_error_m = expect.get("max_bbox_error_m")
    if isinstance(expected_bbox, list) and isinstance(max_bbox_error_m, (int, float)):
        bbox_error_m = bbox_max_corner_error_m(row.get("bbox"), expected_bbox)
        if bbox_error_m is None:
            issues.append("bbox was missing or invalid")
        elif bbox_error_m > float(max_bbox_error_m):
            issues.append(f"bbox max corner error {bbox_error_m:.1f}m above {float(max_bbox_error_m):g}m")
    return issues


def append_min_ocr_labels_expectation_issue(row: dict[str, Any], expect: dict[str, Any], issues: list[str]) -> None:
    min_ocr_labels = expect.get("min_ocr_labels")
    if isinstance(min_ocr_labels, int):
        ocr_label_count = row.get("ocr_label_count")
        if not isinstance(ocr_label_count, int) or ocr_label_count < min_ocr_labels:
            issues.append(f"ocr_label_count {ocr_label_count!r} below {min_ocr_labels}")


def append_ocr_top_labels_expectation_issues(row: dict[str, Any], expect: dict[str, Any], issues: list[str]) -> None:
    expected_snippets = expect.get("ocr_top_labels_contain")
    if not isinstance(expected_snippets, list):
        return
    top_labels = row.get("ocr_top_labels")
    if not isinstance(top_labels, list):
        issues.append("ocr_top_labels missing")
        return
    normalized_labels = [str(label).lower() for label in top_labels if isinstance(label, str)]
    missing = [
        snippet
        for snippet in expected_snippets
        if isinstance(snippet, str) and not any(snippet.lower() in label for label in normalized_labels)
    ]
    if missing:
        issues.append(f"ocr_top_labels missing snippets {missing!r}")


def append_route_ui_expectation_issues(row: dict[str, Any], expect: dict[str, Any], issues: list[str]) -> None:
    expected_categories = expect.get("route_ui_categories_include")
    if isinstance(expected_categories, list):
        categories = row.get("route_ui_categories")
        if not isinstance(categories, list):
            issues.append("route_ui_categories missing")
        else:
            missing = [
                category
                for category in expected_categories
                if isinstance(category, str) and category not in categories
            ]
            if missing:
                issues.append(f"route_ui_categories missing {missing!r}")
    min_route_metric_labels = parse_nonnegative_count_metric(expect.get("min_route_metric_labels"))
    if min_route_metric_labels is None:
        return
    metric_labels = row.get("route_metric_labels")
    metric_count = len(metric_labels) if isinstance(metric_labels, list) else None
    if metric_count is None or metric_count < min_route_metric_labels:
        issues.append(f"route_metric_labels count {metric_count!r} below {min_route_metric_labels:g}")


def append_non_map_ui_expectation_issues(row: dict[str, Any], expect: dict[str, Any], issues: list[str]) -> None:
    expected_categories = expect.get("non_map_ui_categories_include")
    if isinstance(expected_categories, list):
        categories = row.get("non_map_ui_categories")
        if not isinstance(categories, list):
            issues.append("non_map_ui_categories missing")
        else:
            missing = [
                category
                for category in expected_categories
                if isinstance(category, str) and category not in categories
            ]
            if missing:
                issues.append(f"non_map_ui_categories missing {missing!r}")
    min_non_map_ui_labels = parse_nonnegative_count_metric(expect.get("min_non_map_ui_labels"))
    if min_non_map_ui_labels is None:
        return
    labels = row.get("non_map_ui_labels")
    label_count = len(labels) if isinstance(labels, list) else None
    if label_count is None or label_count < min_non_map_ui_labels:
        issues.append(f"non_map_ui_labels count {label_count!r} below {min_non_map_ui_labels:g}")


def append_thematic_map_expectation_issues(row: dict[str, Any], expect: dict[str, Any], issues: list[str]) -> None:
    min_thematic_map_labels = parse_nonnegative_count_metric(expect.get("min_thematic_map_labels"))
    labels = row.get("thematic_map_labels")
    label_count = len(labels) if isinstance(labels, list) else None
    if min_thematic_map_labels is not None and (
        label_count is None or label_count < min_thematic_map_labels
    ):
        issues.append(f"thematic_map_labels count {label_count!r} below {min_thematic_map_labels:g}")

    expected_snippets = expect.get("thematic_map_labels_contain")
    if not isinstance(expected_snippets, list):
        return
    if not isinstance(labels, list):
        issues.append("thematic_map_labels missing")
        return
    normalized_labels = [str(label).lower() for label in labels if isinstance(label, str)]
    missing = [
        snippet
        for snippet in expected_snippets
        if isinstance(snippet, str) and not any(snippet.lower() in label for label in normalized_labels)
    ]
    if missing:
        issues.append(f"thematic_map_labels missing snippets {missing!r}")


def append_max_ocr_engine_calls_expectation_issue(
    row: dict[str, Any], expect: dict[str, Any], issues: list[str]
) -> None:
    max_ocr_engine_calls = parse_nonnegative_count_metric(expect.get("max_ocr_engine_calls"))
    if max_ocr_engine_calls is None:
        return
    ocr_engine_profile = row.get("ocr_engine_profile")
    raw_calls = ocr_engine_profile.get("calls") if isinstance(ocr_engine_profile, dict) else None
    calls = parse_nonnegative_count_metric(raw_calls)
    if calls is None or calls > max_ocr_engine_calls:
        call_text = repr(raw_calls) if calls is None else f"{calls:g}"
        issues.append(f"ocr_engine_profile.calls {call_text} above {max_ocr_engine_calls:g}")


def append_max_ocr_engine_counts_expectation_issues(
    row: dict[str, Any], expect: dict[str, Any], issues: list[str]
) -> None:
    raw_budgets = expect.get("max_ocr_engine_counts")
    if not isinstance(raw_budgets, dict):
        return
    for raw_metric, raw_budget in sorted(raw_budgets.items()):
        if not isinstance(raw_metric, str):
            issues.append(f"ocr_engine_profile metric {raw_metric!r} is invalid")
            continue
        try:
            metric = normalize_ocr_engine_count_metric(raw_metric)
        except ValueError:
            issues.append(f"ocr_engine_profile metric {raw_metric!r} is invalid")
            continue
        budget = parse_nonnegative_count_metric(raw_budget)
        if budget is None:
            issues.append(f"ocr_engine_profile.{metric} budget {raw_budget!r} is invalid")
            continue
        count = row_ocr_engine_metric(row, metric)
        if count is None or count > budget:
            count_text = "missing" if count is None else f"{count:g}"
            issues.append(f"ocr_engine_profile.{metric} {count_text} above {budget:g}")


def append_total_elapsed_expectation_issue(row: dict[str, Any], expect: dict[str, Any], issues: list[str]) -> None:
    max_total_elapsed_s = expect.get("max_total_elapsed_s")
    if isinstance(max_total_elapsed_s, (int, float)):
        total_elapsed_s = row.get("total_elapsed_s")
        if not isinstance(total_elapsed_s, (int, float)) or total_elapsed_s > float(max_total_elapsed_s):
            issues.append(f"total_elapsed_s {total_elapsed_s!r} above {max_total_elapsed_s}")


def append_min_confidence_expectation_issue(
    row: dict[str, Any],
    expect: dict[str, Any],
    *,
    row_key: str,
    expect_key: str,
    issues: list[str],
) -> None:
    minimum = expect.get(expect_key)
    if not isinstance(minimum, (int, float)):
        return
    confidence = row.get(row_key)
    if not isinstance(confidence, (int, float)) or float(confidence) < float(minimum):
        issues.append(f"{row_key} {confidence!r} below {minimum}")


def bbox_max_corner_error_m(observed: object, expected: object) -> float | None:
    observed_bbox = parse_bbox(observed)
    expected_bbox = parse_bbox(expected)
    if observed_bbox is None or expected_bbox is None:
        return None
    observed_corners = bbox_corners(observed_bbox)
    expected_corners = bbox_corners(expected_bbox)
    max_error_m = 0.0
    for observed_corner, expected_corner in zip(observed_corners, expected_corners):
        observed_x, observed_y = lonlat_to_mercator(*observed_corner)
        expected_x, expected_y = lonlat_to_mercator(*expected_corner)
        corner_error_m = ((observed_x - expected_x) ** 2 + (observed_y - expected_y) ** 2) ** 0.5
        max_error_m = max(max_error_m, corner_error_m)
    return max_error_m


def parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        min_lon, min_lat, max_lon, max_lat = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if min_lon > max_lon or min_lat > max_lat:
        return None
    return min_lon, min_lat, max_lon, max_lat


def bbox_corners(bbox: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unexpected = [row["slug"] for row in rows if not row.get("expectation_passed")]
    statuses: dict[str, int] = {}
    sources: dict[str, int] = {}
    ocr_label_event_counts: dict[str, int] = {}
    ocr_full_detail_retry_rows: list[str] = []
    ocr_engine_profiles: list[dict[str, Any]] = []
    stage_totals: dict[str, float] = {}
    stage_max_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = str(row.get("observed_status"))
        statuses[status] = statuses.get(status, 0) + 1
        source = row.get("source")
        if isinstance(source, str) and source:
            sources[source] = sources.get(source, 0) + 1
        ocr_label_events = row.get("ocr_label_events")
        if isinstance(ocr_label_events, list):
            for event in ocr_label_events:
                if not isinstance(event, dict):
                    continue
                message = event.get("message")
                if isinstance(message, str) and message:
                    ocr_label_event_counts[message] = ocr_label_event_counts.get(message, 0) + 1
        if row.get("ocr_full_detail_retry"):
            ocr_full_detail_retry_rows.append(str(row.get("slug")))
        ocr_engine_profile = row.get("ocr_engine_profile")
        if isinstance(ocr_engine_profile, dict):
            ocr_engine_profiles.append(ocr_engine_profile)
        stages = row.get("stages")
        if isinstance(stages, dict):
            for stage, elapsed_s in stages.items():
                if not isinstance(stage, str) or not isinstance(elapsed_s, (int, float)):
                    continue
                stage_totals[stage] = stage_totals.get(stage, 0.0) + float(elapsed_s)
                prior = stage_max_rows.get(stage)
                if prior is None or float(elapsed_s) > float(prior["elapsed_s"]):
                    stage_max_rows[stage] = {
                        "slug": row.get("slug"),
                        "elapsed_s": round(float(elapsed_s), 6),
                    }
    elapsed_values = [
        row.get("total_elapsed_s")
        for row in rows
        if isinstance(row.get("total_elapsed_s"), (int, float))
    ]
    return {
        "total": len(rows),
        "expectation_passed": len(rows) - len(unexpected),
        "unexpected": unexpected,
        "statuses": dict(sorted(statuses.items())),
        "sources": dict(sorted(sources.items())),
        "ocr_label_event_counts": dict(sorted(ocr_label_event_counts.items())),
        "ocr_full_detail_retry_count": len(ocr_full_detail_retry_rows),
        "ocr_full_detail_retry_rows": ocr_full_detail_retry_rows,
        "ocr_engine_profile": summarize_rapidocr_profile_summaries(ocr_engine_profiles),
        "ocr_engine_stage_max_rows": ocr_engine_stage_max_rows(rows),
        "ocr_engine_slowest_cases": primary_ocr_engine_slowest_cases(rows),
        "slowest_cases": primary_slowest_cases_from_rows(rows),
        "ocr_overlap_hidden_s": summarize_ocr_overlap_hidden(rows),
        "max_total_elapsed_s": round(max(elapsed_values), 6) if elapsed_values else None,
        "stage_duration_s": {stage: round(elapsed_s, 6) for stage, elapsed_s in sorted(stage_totals.items())},
        "stage_max_rows": dict(sorted(stage_max_rows.items())),
    }


def summarize_ocr_overlap_hidden(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    measured: list[tuple[float, str]] = []
    for row in rows:
        hidden_s = row.get("ocr_overlap_hidden_s")
        if hidden_s is None:
            hidden_s = ocr_overlap_hidden_seconds(row)
        hidden = parse_nonnegative_float(hidden_s)
        if hidden is None:
            continue
        measured.append((hidden, str(row.get("slug") or "")))
    if not measured:
        return None
    total_s = sum(value for value, _slug in measured)
    max_s, max_slug = max(measured, key=lambda item: (item[0], item[1]))
    return {
        "rows": len(measured),
        "total_s": round(total_s, 6),
        "max_s": round(max_s, 6),
        "max_slug": max_slug or None,
    }


def ocr_engine_stage_max_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    maxima: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = row.get("slug")
        ocr_engine_profile = row.get("ocr_engine_profile")
        if not isinstance(ocr_engine_profile, dict):
            continue
        calls = ocr_engine_profile.get("calls_detail")
        call_profiles = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
        if not call_profiles:
            call_profiles = [ocr_engine_profile]
        for call in call_profiles:
            for key in OCR_ENGINE_STAGE_MAX_KEYS:
                elapsed_s = parse_nonnegative_float(call.get(key))
                if elapsed_s is None:
                    continue
                prior = maxima.get(key)
                if prior is not None and elapsed_s <= float(prior["elapsed_s"]):
                    continue
                maxima[key] = ocr_engine_stage_max_row(slug, elapsed_s, call)
    return dict(sorted(maxima.items()))


def ocr_engine_stage_max_row(slug: Any, elapsed_s: float, call: dict[str, Any]) -> dict[str, Any]:
    row = {
        "slug": slug,
        "elapsed_s": round(elapsed_s, 6),
    }
    for key in (
        *OCR_ENGINE_DETAIL_CONTEXT_KEYS,
        "raw_box_count",
        "selected_box_count",
        "result_count",
        "label_count",
        "useful_label_count",
        *OCR_ENGINE_BOX_AREA_KEYS,
        *OCR_ENGINE_CONFIDENCE_KEYS,
    ):
        value = call.get(key)
        if value is not None:
            row[key] = value
    return row


def primary_slowest_cases_from_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        total_elapsed_s = parse_nonnegative_float(row.get("total_elapsed_s"))
        if total_elapsed_s is None:
            continue
        ranked.append((total_elapsed_s, str(row.get("slug") or ""), row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        primary_slowest_case_summary(row, total_elapsed_s)
        for total_elapsed_s, _slug, row in ranked[: max(0, limit)]
    ]


def primary_slowest_case_summary(
    row: dict[str, Any],
    total_elapsed_s: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "slug": row.get("slug"),
        "total_elapsed_s": round(total_elapsed_s, 6),
        "observed_status": row.get("observed_status"),
        "expectation_passed": row.get("expectation_passed"),
    }
    top_stage = slowest_stage_summary(row)
    if top_stage is not None:
        result["top_stage"] = top_stage
        if top_stage.get("stage") == "georeference":
            georeference_events = row.get("georeference_events")
            if isinstance(georeference_events, list) and georeference_events:
                result["georeference_events"] = [
                    event for event in georeference_events if isinstance(event, dict)
                ][:5]
    ocr_engine = slowest_sample_ocr_engine_summary(row)
    if ocr_engine:
        result["ocr_engine"] = ocr_engine
    return result


def primary_ocr_engine_slowest_cases(
    rows: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        ocr_engine_profile = row.get("ocr_engine_profile")
        if not isinstance(ocr_engine_profile, dict):
            continue
        total_s = parse_nonnegative_float(ocr_engine_profile.get("total_s"))
        if total_s is None:
            continue
        ranked.append((total_s, str(row.get("slug") or ""), row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        primary_ocr_engine_slowest_case_summary(row, total_s)
        for total_s, _slug, row in ranked[: max(0, limit)]
    ]


def primary_ocr_engine_slowest_case_summary(
    row: dict[str, Any],
    total_s: float,
) -> dict[str, Any]:
    ocr_engine_profile = row.get("ocr_engine_profile")
    if not isinstance(ocr_engine_profile, dict):
        return {}
    result: dict[str, Any] = {
        "slug": row.get("slug"),
        "total_s": round(total_s, 6),
    }
    for key in ("input_s", "rec_elapsed_s", "det_elapsed_s"):
        elapsed_s = parse_nonnegative_float(ocr_engine_profile.get(key))
        if elapsed_s is not None:
            result[key] = round(elapsed_s, 6)
    hidden_s = row.get("ocr_overlap_hidden_s")
    if hidden_s is None:
        hidden_s = ocr_overlap_hidden_seconds(row)
    hidden = parse_nonnegative_float(hidden_s)
    if hidden is not None:
        result["overlap_hidden_s"] = round(hidden, 6)
    add_ocr_engine_dominant_stage_summary(
        result,
        OCR_ENGINE_PRIMARY_DOMINANT_STAGE_FIELDS,
    )
    calls = ocr_engine_profile.get("calls")
    if isinstance(calls, int) and not isinstance(calls, bool):
        result["calls"] = calls
    detail_profile = slowest_ocr_engine_detail_profile(ocr_engine_profile) or ocr_engine_profile
    for key in OCR_ENGINE_DETAIL_CONTEXT_KEYS:
        value = detail_profile.get(key)
        if value is not None:
            result[key] = value
    for key in ("raw_box_count", "selected_box_count", "result_count", "label_count", "useful_label_count"):
        value = ocr_engine_profile.get(key)
        if value is None:
            value = detail_profile.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    for key in (*OCR_ENGINE_BOX_AREA_KEYS, *OCR_ENGINE_CONFIDENCE_KEYS):
        value = ocr_engine_profile.get(key)
        if value is None:
            value = detail_profile.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
        elif isinstance(value, float):
            result[key] = round(value, 6)
    return result


def build_repeat_profile(
    cases: list[dict[str, Any]],
    *,
    out_dir: Path,
    runs_per_case: int,
    warmup_runs_per_case: int,
    timeout_seconds: float,
    write_debug: bool,
    profile_ocr_engine: bool,
    runner_ocr_cache: bool,
    extraction_cache: bool,
    execution: str,
    python_executable: str,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    repeat_base_dir = out_dir / "repeat-profile"
    for case in cases:
        for repeat_index in range(1, runs_per_case + 1):
            sample_out_dir = repeat_base_dir / f"run-{repeat_index}"
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            row = run_stress_case(
                case,
                sample_out_dir,
                timeout_seconds=timeout_seconds,
                write_debug=write_debug,
                profile_ocr_engine=profile_ocr_engine,
                runner_ocr_cache=runner_ocr_cache,
                extraction_cache=extraction_cache,
                execution=execution,
                python_executable=python_executable,
            )
            samples.append(
                {
                    "repeat_index": repeat_index,
                    "warmup": repeat_index <= warmup_runs_per_case,
                    **row,
                }
            )
    return summarize_repeat_profile_samples(
        samples,
        runs_per_case=runs_per_case,
        warmup_runs_per_case=warmup_runs_per_case,
    )


def summarize_repeat_profile_samples(
    samples: list[dict[str, Any]],
    *,
    runs_per_case: int,
    warmup_runs_per_case: int,
) -> dict[str, Any]:
    case_samples: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        slug = sample.get("slug")
        if isinstance(slug, str) and slug:
            case_samples.setdefault(slug, []).append(sample)
    case_summaries = {
        slug: summarize_repeat_profile_sample_group(slug_samples)
        for slug, slug_samples in sorted(case_samples.items())
    }
    unstable_signature_cases = [
        slug
        for slug, case_summary in case_summaries.items()
        if not case_summary.get("signature_stability", {}).get("stable", True)
    ]
    analyzed_samples = repeat_profile_analyzed_samples(samples)
    subsecond_case_count = sum(
        1
        for case_summary in case_summaries.values()
        if parse_nonnegative_float(case_summary.get("min_total_elapsed_s")) is not None
        and float(case_summary["min_total_elapsed_s"]) < 1.0
    )
    return {
        "runs_per_case": runs_per_case,
        "warmup_runs_per_case": warmup_runs_per_case,
        "summary": {
            "cases": len(case_summaries),
            "samples": len(samples),
            "analyzed_samples": len(analyzed_samples),
            "expectation_passed_samples": count_repeat_expectation_passed_samples(analyzed_samples),
            "unexpected_samples": len(analyzed_samples)
            - count_repeat_expectation_passed_samples(analyzed_samples),
            "subsecond_samples": count_repeat_subsecond_samples(analyzed_samples),
            "ocr_full_detail_retry_samples": count_repeat_full_detail_retry_samples(analyzed_samples),
            "subsecond_case_min_total_count": subsecond_case_count,
            "stable_signature_cases": len(case_summaries) - len(unstable_signature_cases),
            "unstable_signature_cases": unstable_signature_cases,
            "ocr_full_detail_retry_cases": repeat_profile_ocr_full_detail_retry_cases(case_summaries),
            **repeat_profile_total_elapsed_stats(analyzed_samples),
            "stage_duration_s": repeat_profile_stage_duration_stats(analyzed_samples),
            "slowest_samples": repeat_profile_slowest_samples(analyzed_samples),
            "slowest_cases": repeat_profile_slowest_cases(case_summaries),
            "ocr_engine_profile": summarize_repeat_profile_ocr_engine(analyzed_samples),
            "ocr_engine_stage_duration_s": repeat_profile_ocr_engine_stage_duration_stats(analyzed_samples),
            "ocr_engine_count_metric": repeat_profile_ocr_engine_count_stats(analyzed_samples),
            "ocr_overlap_hidden_s": repeat_profile_ocr_overlap_hidden_stats(analyzed_samples),
            "ocr_engine_stage_max_rows": ocr_engine_stage_max_rows(analyzed_samples),
            "ocr_engine_slowest_cases": repeat_profile_ocr_engine_slowest_cases(case_summaries),
        },
        "cases": case_summaries,
        "samples": samples,
    }


def summarize_repeat_profile_sample_group(samples: list[dict[str, Any]]) -> dict[str, Any]:
    analyzed_samples = repeat_profile_analyzed_samples(samples)
    return {
        "samples": len(samples),
        "analyzed_samples": len(analyzed_samples),
        "expectation_passed_samples": count_repeat_expectation_passed_samples(analyzed_samples),
        "unexpected_samples": len(analyzed_samples)
        - count_repeat_expectation_passed_samples(analyzed_samples),
        "subsecond_samples": count_repeat_subsecond_samples(analyzed_samples),
        "ocr_full_detail_retry_samples": count_repeat_full_detail_retry_samples(analyzed_samples),
        "signature_stability": repeat_profile_signature_stability(analyzed_samples),
        **repeat_profile_total_elapsed_stats(analyzed_samples),
        "stage_duration_s": repeat_profile_stage_duration_stats(analyzed_samples),
        "ocr_engine_profile": summarize_repeat_profile_ocr_engine(analyzed_samples),
        "ocr_engine_stage_duration_s": repeat_profile_ocr_engine_stage_duration_stats(analyzed_samples),
        "ocr_engine_count_metric": repeat_profile_ocr_engine_count_stats(analyzed_samples),
        "ocr_overlap_hidden_s": repeat_profile_ocr_overlap_hidden_stats(analyzed_samples),
        "ocr_engine_stage_max_rows": ocr_engine_stage_max_rows(analyzed_samples),
    }


def repeat_profile_analyzed_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in samples if not sample.get("warmup")]


def count_repeat_full_detail_retry_samples(samples: list[dict[str, Any]]) -> int:
    return sum(1 for sample in samples if sample.get("ocr_full_detail_retry") is True)


def repeat_profile_signature_stability(samples: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = [repeat_profile_output_signature(sample) for sample in samples]
    counts = Counter(json.dumps(signature, sort_keys=True) for signature in signatures)
    return {
        "samples": len(signatures),
        "stable": len(counts) <= 1,
        "unique_signatures": len(counts),
        "signatures": [
            {"count": count, **json.loads(encoded)}
            for encoded, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def repeat_profile_output_signature(sample: dict[str, Any]) -> dict[str, Any]:
    top_labels = sample.get("ocr_top_labels")
    return {
        "observed_status": sample.get("observed_status"),
        "city": sample.get("city"),
        "source": sample.get("source"),
        "control_points": sample.get("control_points"),
        "bbox": repeat_profile_bbox_signature(sample.get("bbox")),
        "geojson_geometry_hash": sample.get("geojson_geometry_hash"),
        "geojson_coordinate_count": sample.get("geojson_coordinate_count"),
        "combined_confidence": repeat_profile_confidence_signature(sample.get("combined_confidence")),
        "georeference_confidence": repeat_profile_confidence_signature(sample.get("georeference_confidence")),
        "road_match_score": repeat_profile_confidence_signature(sample.get("road_match_score")),
        "road_match_base_score": repeat_profile_confidence_signature(sample.get("road_match_base_score")),
        "road_match_sampled_points": repeat_profile_int_signature(sample.get("road_match_sampled_points")),
        "ocr_label_count": sample.get("ocr_label_count"),
        "ocr_label_event": sample.get("ocr_label_event"),
        "ocr_full_detail_retry": sample.get("ocr_full_detail_retry"),
        "ocr_top_labels": top_labels if isinstance(top_labels, list) else None,
        "route_ui_categories": repeat_profile_list_signature(sample.get("route_ui_categories")),
        "route_metric_labels": repeat_profile_list_signature(sample.get("route_metric_labels")),
        "non_map_ui_categories": repeat_profile_list_signature(sample.get("non_map_ui_categories")),
        "non_map_ui_labels": repeat_profile_list_signature(sample.get("non_map_ui_labels")),
        "thematic_map_labels": repeat_profile_list_signature(sample.get("thematic_map_labels")),
        "error": sample.get("error"),
    }


def repeat_profile_list_signature(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        items.append(item)
    return items


def repeat_profile_bbox_signature(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    bbox: list[float] = []
    for coordinate in value:
        try:
            number = float(coordinate)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        bbox.append(round(number, 6))
    return bbox


def repeat_profile_confidence_signature(value: Any) -> float | None:
    parsed = parse_nonnegative_float(value)
    if parsed is None:
        return None
    return round(parsed, 6)


def repeat_profile_int_signature(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def repeat_profile_slowest_samples(
    samples: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for sample in samples:
        total_elapsed_s = parse_nonnegative_float(sample.get("total_elapsed_s"))
        if total_elapsed_s is None:
            continue
        ranked.append((total_elapsed_s, sample))
    ranked.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("slug") or ""),
            int(item[1].get("repeat_index") or 0),
        )
    )
    return [
        repeat_profile_slowest_sample_summary(sample, total_elapsed_s)
        for total_elapsed_s, sample in ranked[: max(0, limit)]
    ]


def repeat_profile_slowest_cases(
    case_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    for slug, summary in case_summaries.items():
        if not isinstance(summary, dict):
            continue
        p95_total = parse_nonnegative_float(summary.get("p95_total_elapsed_s"))
        max_total = parse_nonnegative_float(summary.get("max_total_elapsed_s"))
        if p95_total is None:
            continue
        ranked.append((p95_total, max_total or p95_total, slug, summary))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        repeat_profile_slowest_case_summary(slug, summary, p95_total, max_total)
        for p95_total, max_total, slug, summary in ranked[: max(0, limit)]
    ]


def repeat_profile_slowest_case_summary(
    slug: str,
    summary: dict[str, Any],
    p95_total: float,
    max_total: float,
) -> dict[str, Any]:
    result = {
        "slug": slug,
        "p95_total_elapsed_s": round(p95_total, 6),
        "max_total_elapsed_s": round(max_total, 6),
    }
    for key in ("samples", "analyzed_samples", "expectation_passed_samples", "unexpected_samples"):
        value = summary.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    top_stage = repeat_profile_case_top_stage_summary(summary)
    if top_stage is not None:
        result["top_stage"] = top_stage
    return result


def repeat_profile_case_top_stage_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    stage_stats = summary.get("stage_duration_s")
    if not isinstance(stage_stats, dict):
        return None
    ranked: list[tuple[float, float, str]] = []
    for stage, raw_stats in stage_stats.items():
        if not isinstance(stage, str) or not stage or not isinstance(raw_stats, dict):
            continue
        p95_duration_s = parse_nonnegative_float(raw_stats.get("p95_duration_s"))
        if p95_duration_s is None:
            continue
        max_duration_s = parse_nonnegative_float(raw_stats.get("max_duration_s"))
        ranked.append((p95_duration_s, max_duration_s or p95_duration_s, stage))
    if not ranked:
        return None
    p95_duration_s, max_duration_s, stage = max(ranked, key=lambda item: (item[0], item[1], item[2]))
    return {
        "stage": stage,
        "p95_duration_s": round(p95_duration_s, 6),
        "max_duration_s": round(max_duration_s, 6),
    }


def repeat_profile_ocr_full_detail_retry_cases(
    case_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, str, dict[str, Any]]] = []
    for slug, summary in case_summaries.items():
        if not isinstance(summary, dict):
            continue
        retry_samples = summary.get("ocr_full_detail_retry_samples")
        if not isinstance(retry_samples, int) or isinstance(retry_samples, bool) or retry_samples <= 0:
            continue
        analyzed_samples = summary.get("analyzed_samples")
        analyzed_count = analyzed_samples if isinstance(analyzed_samples, int) else 0
        ranked.append((retry_samples, analyzed_count, slug, summary))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        repeat_profile_ocr_full_detail_retry_case_summary(slug, summary, retry_samples)
        for retry_samples, _analyzed_count, slug, summary in ranked[: max(0, limit)]
    ]


def repeat_profile_ocr_full_detail_retry_case_summary(
    slug: str,
    summary: dict[str, Any],
    retry_samples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "slug": slug,
        "ocr_full_detail_retry_samples": retry_samples,
    }
    for key in ("analyzed_samples", "unexpected_samples"):
        value = summary.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    return result


def repeat_profile_ocr_engine_slowest_cases(
    case_summaries: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    for slug, summary in case_summaries.items():
        if not isinstance(summary, dict):
            continue
        total_stats = repeat_profile_case_ocr_engine_metric(summary, "total_s")
        if total_stats is None:
            continue
        p95_total = parse_nonnegative_float(total_stats.get("p95_duration_s"))
        max_total = parse_nonnegative_float(total_stats.get("max_duration_s"))
        if p95_total is None:
            continue
        ranked.append((p95_total, max_total or p95_total, slug, summary))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        repeat_profile_ocr_engine_slowest_case_summary(slug, summary, p95_total, max_total)
        for p95_total, max_total, slug, summary in ranked[: max(0, limit)]
    ]


def repeat_profile_ocr_engine_slowest_case_summary(
    slug: str,
    summary: dict[str, Any],
    p95_total: float,
    max_total: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "slug": slug,
        "p95_total_s": round(p95_total, 6),
        "max_total_s": round(max_total, 6),
    }
    input_stats = repeat_profile_case_ocr_engine_metric(summary, "input_s")
    if input_stats is not None:
        input_p95 = parse_nonnegative_float(input_stats.get("p95_duration_s"))
        if input_p95 is not None:
            result["p95_input_s"] = round(input_p95, 6)
    rec_stats = repeat_profile_case_ocr_engine_metric(summary, "rec_elapsed_s")
    if rec_stats is not None:
        rec_p95 = parse_nonnegative_float(rec_stats.get("p95_duration_s"))
        if rec_p95 is not None:
            result["p95_rec_elapsed_s"] = round(rec_p95, 6)
    det_stats = repeat_profile_case_ocr_engine_metric(summary, "det_elapsed_s")
    if det_stats is not None:
        det_p95 = parse_nonnegative_float(det_stats.get("p95_duration_s"))
        if det_p95 is not None:
            result["p95_det_elapsed_s"] = round(det_p95, 6)
    add_ocr_engine_dominant_stage_summary(
        result,
        OCR_ENGINE_REPEAT_P95_DOMINANT_STAGE_FIELDS,
        prefix="p95_",
    )
    count_stats = repeat_profile_case_ocr_engine_count_metric(summary, "selected_box_count")
    if count_stats is not None:
        selected_p95 = parse_nonnegative_float(count_stats.get("p95_count"))
        selected_max = parse_nonnegative_float(count_stats.get("max_count"))
        if selected_p95 is not None:
            result["p95_selected_box_count"] = round(selected_p95, 6)
        if selected_max is not None:
            result["max_selected_box_count"] = round(selected_max, 6)
    small_area_stats = repeat_profile_case_ocr_engine_count_metric(
        summary, "selected_box_area_lt_1300_count"
    )
    if small_area_stats is not None:
        small_area_p95 = parse_nonnegative_float(small_area_stats.get("p95_count"))
        small_area_max = parse_nonnegative_float(small_area_stats.get("max_count"))
        if small_area_p95 is not None:
            result["p95_selected_box_area_lt_1300_count"] = round(small_area_p95, 6)
        if small_area_max is not None:
            result["max_selected_box_area_lt_1300_count"] = round(small_area_max, 6)
    result.update(repeat_profile_ocr_engine_context(summary))
    return result


def repeat_profile_ocr_engine_context(summary: dict[str, Any]) -> dict[str, Any]:
    stage_max_rows = summary.get("ocr_engine_stage_max_rows")
    if not isinstance(stage_max_rows, dict):
        return {}
    row = stage_max_rows.get("total_s")
    if not isinstance(row, dict):
        row = next((value for value in stage_max_rows.values() if isinstance(value, dict)), None)
    if not isinstance(row, dict):
        return {}
    context: dict[str, Any] = {}
    for key in OCR_ENGINE_DETAIL_CONTEXT_KEYS:
        value = row.get(key)
        if value is not None:
            context[key] = value
    return context


def repeat_profile_case_ocr_engine_metric(summary: dict[str, Any], metric: str) -> dict[str, Any] | None:
    stage_stats = summary.get("ocr_engine_stage_duration_s")
    if not isinstance(stage_stats, dict):
        return None
    metric_stats = stage_stats.get(metric)
    return metric_stats if isinstance(metric_stats, dict) else None


def repeat_profile_case_ocr_engine_count_metric(summary: dict[str, Any], metric: str) -> dict[str, Any] | None:
    count_stats = summary.get("ocr_engine_count_metric")
    if not isinstance(count_stats, dict):
        return None
    metric_stats = count_stats.get(metric)
    return metric_stats if isinstance(metric_stats, dict) else None


def add_ocr_engine_dominant_stage_summary(
    result: dict[str, Any],
    fields: tuple[tuple[str, str], ...],
    *,
    prefix: str = "",
) -> None:
    dominant = ocr_engine_dominant_stage(result, fields)
    if dominant is None:
        return
    stage, elapsed_s = dominant
    result[f"{prefix}dominant_stage"] = stage
    result[f"{prefix}dominant_stage_s"] = round(elapsed_s, 6)


def ocr_engine_dominant_stage(
    values: dict[str, Any],
    fields: tuple[tuple[str, str], ...],
) -> tuple[str, float] | None:
    candidates: list[tuple[str, float]] = []
    for label, key in fields:
        elapsed_s = parse_nonnegative_float(values.get(key))
        if elapsed_s is not None and elapsed_s > 0.0:
            candidates.append((label, elapsed_s))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def repeat_profile_slowest_sample_summary(
    sample: dict[str, Any],
    total_elapsed_s: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "slug": sample.get("slug"),
        "repeat_index": sample.get("repeat_index"),
        "total_elapsed_s": round(total_elapsed_s, 6),
        "observed_status": sample.get("observed_status"),
        "expectation_passed": sample.get("expectation_passed"),
    }
    top_stage = slowest_stage_summary(sample)
    if top_stage is not None:
        summary["top_stage"] = top_stage
        if top_stage.get("stage") == "georeference":
            georeference_events = sample.get("georeference_events")
            if isinstance(georeference_events, list) and georeference_events:
                summary["georeference_events"] = [
                    event for event in georeference_events if isinstance(event, dict)
                ][:5]
    ocr_label_count = sample.get("ocr_label_count")
    if isinstance(ocr_label_count, (int, float)) and not isinstance(ocr_label_count, bool):
        summary["ocr_label_count"] = int(ocr_label_count)
    ocr_label_event = sample.get("ocr_label_event")
    if isinstance(ocr_label_event, str) and ocr_label_event:
        summary["ocr_label_event"] = ocr_label_event
    ocr_top_labels = sample.get("ocr_top_labels")
    if isinstance(ocr_top_labels, list):
        summary["ocr_top_labels"] = ocr_top_labels[:5]
    ocr_engine = slowest_sample_ocr_engine_summary(sample)
    if ocr_engine:
        summary["ocr_engine"] = ocr_engine
    return summary


def slowest_stage_summary(sample: dict[str, Any]) -> dict[str, Any] | None:
    stages = sample.get("stages")
    if not isinstance(stages, dict):
        return None
    best_name: str | None = None
    best_elapsed: float | None = None
    for stage, raw_elapsed_s in stages.items():
        if not isinstance(stage, str) or not stage:
            continue
        elapsed_s = parse_nonnegative_float(raw_elapsed_s)
        if elapsed_s is None:
            continue
        if best_elapsed is None or elapsed_s > best_elapsed:
            best_name = stage
            best_elapsed = elapsed_s
    if best_name is None or best_elapsed is None:
        return None
    return {"stage": best_name, "elapsed_s": round(best_elapsed, 6)}


def slowest_sample_ocr_engine_summary(sample: dict[str, Any]) -> dict[str, Any]:
    ocr_engine_profile = sample.get("ocr_engine_profile")
    if not isinstance(ocr_engine_profile, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in OCR_ENGINE_STAGE_MAX_KEYS:
        elapsed_s = parse_nonnegative_float(ocr_engine_profile.get(key))
        if elapsed_s is not None:
            summary[key] = round(elapsed_s, 6)
    hidden_s = sample.get("ocr_overlap_hidden_s")
    if hidden_s is None:
        hidden_s = ocr_overlap_hidden_seconds(sample)
    hidden = parse_nonnegative_float(hidden_s)
    if hidden is not None:
        summary["overlap_hidden_s"] = round(hidden, 6)
    calls = ocr_engine_profile.get("calls")
    if isinstance(calls, int):
        summary["calls"] = calls
    for key in ("raw_box_count", "selected_box_count", "result_count", "label_count", "useful_label_count"):
        value = ocr_engine_profile.get(key)
        if isinstance(value, int):
            summary[key] = value
    detail_profile = slowest_ocr_engine_detail_profile(ocr_engine_profile)
    context_profile = detail_profile or ocr_engine_profile
    for key in OCR_ENGINE_DETAIL_CONTEXT_KEYS:
        value = context_profile.get(key)
        if value is not None:
            summary[key] = value
    for key in (*OCR_ENGINE_BOX_AREA_KEYS, *OCR_ENGINE_CONFIDENCE_KEYS):
        value = ocr_engine_profile.get(key)
        if value is None and detail_profile is not None:
            value = detail_profile.get(key)
        if isinstance(value, int):
            summary[key] = value
        elif isinstance(value, float):
            summary[key] = round(value, 6)
    return summary


def slowest_ocr_engine_detail_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    calls = profile.get("calls_detail")
    call_profiles = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
    if not call_profiles:
        return None
    return max(
        call_profiles,
        key=lambda call: parse_nonnegative_float(call.get("total_s")) or 0.0,
    )


def repeat_profile_signature_drift_cases(report: dict[str, Any]) -> list[str]:
    repeat_profile = report.get("repeat_profile")
    if not isinstance(repeat_profile, dict):
        return []
    summary = repeat_profile.get("summary")
    if not isinstance(summary, dict):
        return []
    unstable_cases = summary.get("unstable_signature_cases")
    if not isinstance(unstable_cases, list):
        return []
    return [str(slug) for slug in unstable_cases]


def baseline_comparison_signature_drift_cases(report: dict[str, Any]) -> list[str]:
    comparison = report.get("baseline_comparison")
    if not isinstance(comparison, dict):
        return []
    changes = comparison.get("signature_changes")
    if not isinstance(changes, list):
        return []
    slugs: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        slug = change.get("slug")
        if isinstance(slug, str) and slug:
            slugs.append(slug)
    return slugs


def baseline_comparison_configuration_changes(report: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = report.get("baseline_comparison")
    if not isinstance(comparison, dict):
        return []
    changes = comparison.get("configuration_changes")
    if not isinstance(changes, list):
        return []
    return [change for change in changes if isinstance(change, dict)]


def baseline_comparison_expectation_regressions(report: dict[str, Any]) -> list[str]:
    comparison = report.get("baseline_comparison")
    if not isinstance(comparison, dict):
        return []
    changes = comparison.get("expectation_changes")
    if not isinstance(changes, list):
        return []
    slugs: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        if change.get("baseline_expectation_passed") is not True:
            continue
        if change.get("candidate_expectation_passed") is not False:
            continue
        slug = change.get("slug")
        if isinstance(slug, str) and slug:
            slugs.append(slug)
    return slugs


def baseline_comparison_coverage_gaps(report: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = report.get("baseline_comparison")
    if not isinstance(comparison, dict):
        return []
    gaps: list[dict[str, Any]] = []
    missing_in_baseline = comparison.get("missing_in_baseline")
    if isinstance(missing_in_baseline, list):
        slugs = [str(slug) for slug in missing_in_baseline if isinstance(slug, (str, int))]
        if slugs:
            gaps.append({"kind": "missing_in_baseline", "slugs": slugs})
    missing_in_candidate = comparison.get("missing_in_candidate")
    if isinstance(missing_in_candidate, list):
        slugs = [str(slug) for slug in missing_in_candidate if isinstance(slug, (str, int))]
        if slugs:
            gaps.append({"kind": "missing_in_candidate", "slugs": slugs})
    repeat_case_missing_in_baseline = comparison.get("repeat_profile_case_missing_in_baseline")
    if isinstance(repeat_case_missing_in_baseline, list):
        slugs = [
            str(slug)
            for slug in repeat_case_missing_in_baseline
            if isinstance(slug, (str, int))
        ]
        if slugs:
            gaps.append({"kind": "repeat_profile_case_missing_in_baseline", "slugs": slugs})
    repeat_case_missing_in_candidate = comparison.get("repeat_profile_case_missing_in_candidate")
    if isinstance(repeat_case_missing_in_candidate, list):
        slugs = [
            str(slug)
            for slug in repeat_case_missing_in_candidate
            if isinstance(slug, (str, int))
        ]
        if slugs:
            gaps.append({"kind": "repeat_profile_case_missing_in_candidate", "slugs": slugs})
    repeat_case_underanalyzed_in_baseline = comparison.get(
        "repeat_profile_case_underanalyzed_in_baseline"
    )
    if isinstance(repeat_case_underanalyzed_in_baseline, list):
        slugs = repeat_profile_case_coverage_gap_slugs(repeat_case_underanalyzed_in_baseline)
        if slugs:
            gaps.append({"kind": "repeat_profile_case_underanalyzed_in_baseline", "slugs": slugs})
    repeat_case_underanalyzed_in_candidate = comparison.get(
        "repeat_profile_case_underanalyzed_in_candidate"
    )
    if isinstance(repeat_case_underanalyzed_in_candidate, list):
        slugs = repeat_profile_case_coverage_gap_slugs(repeat_case_underanalyzed_in_candidate)
        if slugs:
            gaps.append({"kind": "repeat_profile_case_underanalyzed_in_candidate", "slugs": slugs})
    compared_rows = comparison.get("compared_rows")
    if isinstance(compared_rows, int) and compared_rows <= 0:
        gaps.append({"kind": "no_compared_rows"})
    return gaps


def repeat_profile_case_coverage_gap_slugs(gaps: list[Any]) -> list[str]:
    slugs: list[str] = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        slug = gap.get("slug")
        if isinstance(slug, (str, int)):
            slugs.append(str(slug))
    return slugs


def baseline_comparison_regression_budget_failed(report: dict[str, Any]) -> bool:
    comparison = report.get("baseline_comparison")
    if not isinstance(comparison, dict):
        return False
    regression_budget = comparison.get("regression_budget")
    return isinstance(regression_budget, dict) and regression_budget.get("passed") is False


def repeat_profile_unexpected_sample_count(report: dict[str, Any]) -> int:
    repeat_profile = report.get("repeat_profile")
    if not isinstance(repeat_profile, dict):
        return 0
    summary = repeat_profile.get("summary")
    if isinstance(summary, dict):
        unexpected_samples = summary.get("unexpected_samples")
        if isinstance(unexpected_samples, int) and unexpected_samples > 0:
            return unexpected_samples
    return sum(
        unexpected_samples
        for unexpected_samples in repeat_profile_case_unexpected_sample_counts(repeat_profile)
    )


def repeat_profile_unexpected_cases(report: dict[str, Any]) -> list[str]:
    repeat_profile = report.get("repeat_profile")
    if not isinstance(repeat_profile, dict):
        return []
    cases = repeat_profile.get("cases")
    if not isinstance(cases, dict):
        return []
    return [
        str(slug)
        for slug, case_summary in sorted(cases.items())
        if isinstance(case_summary, dict)
        and isinstance(case_summary.get("unexpected_samples"), int)
        and case_summary["unexpected_samples"] > 0
    ]


def repeat_profile_case_unexpected_sample_counts(repeat_profile: dict[str, Any]) -> list[int]:
    cases = repeat_profile.get("cases")
    if not isinstance(cases, dict):
        return []
    return [
        case_summary["unexpected_samples"]
        for case_summary in cases.values()
        if isinstance(case_summary, dict)
        and isinstance(case_summary.get("unexpected_samples"), int)
        and case_summary["unexpected_samples"] > 0
    ]


def build_latency_budget_summary(
    rows: list[dict[str, Any]],
    repeat_profile: dict[str, Any] | None,
    *,
    prewarm: dict[str, Any] | None = None,
    max_total_elapsed_s: float | None = None,
    max_repeat_profile_p95_duration_s: float | None = None,
    max_prewarm_runtime_s: float | None = None,
    max_prewarm_stage_s: dict[str, float] | None = None,
    max_ocr_engine_duration_s: dict[str, float] | None = None,
    max_ocr_engine_count: dict[str, float] | None = None,
    max_repeat_ocr_engine_p95_duration_s: dict[str, float] | None = None,
    max_repeat_ocr_engine_p95_count: dict[str, float] | None = None,
    max_repeat_ocr_engine_max_count: dict[str, float] | None = None,
) -> dict[str, Any]:
    repeat_samples: list[dict[str, Any]] = []
    if isinstance(repeat_profile, dict):
        samples = repeat_profile.get("samples")
        if isinstance(samples, list):
            repeat_samples = repeat_profile_analyzed_samples(
                [sample for sample in samples if isinstance(sample, dict)]
            )
    primary_violations: list[dict[str, Any]] = []
    repeat_violations: list[dict[str, Any]] = []
    if max_total_elapsed_s is not None:
        primary_violations = latency_budget_violations(rows, max_total_elapsed_s=max_total_elapsed_s)
        repeat_violations = latency_budget_violations(repeat_samples, max_total_elapsed_s=max_total_elapsed_s)
    ocr_engine_violations = ocr_engine_duration_budget_violations(
        rows,
        max_ocr_engine_duration_s or {},
    )
    ocr_engine_count_violations = ocr_engine_count_budget_violations(
        rows,
        max_ocr_engine_count or {},
    )
    repeat_ocr_engine_p95_violations = repeat_ocr_engine_p95_budget_violations(
        repeat_profile,
        max_repeat_ocr_engine_p95_duration_s or {},
    )
    repeat_ocr_engine_count_p95_violations = repeat_ocr_engine_count_p95_budget_violations(
        repeat_profile,
        max_repeat_ocr_engine_p95_count or {},
    )
    repeat_ocr_engine_count_max_violations = repeat_ocr_engine_count_max_budget_violations(
        repeat_profile,
        max_repeat_ocr_engine_max_count or {},
    )
    repeat_p95_violations = repeat_profile_p95_budget_violations(
        repeat_profile,
        max_repeat_profile_p95_duration_s=max_repeat_profile_p95_duration_s,
    )
    prewarm_violations = prewarm_runtime_budget_violations(
        prewarm,
        max_prewarm_runtime_s=max_prewarm_runtime_s,
    )
    prewarm_stage_violations = prewarm_stage_budget_violations(
        prewarm,
        max_prewarm_stage_s or {},
    )
    summary = {
        "passed": (
            not primary_violations
            and not repeat_violations
            and not repeat_p95_violations
            and not prewarm_violations
            and not prewarm_stage_violations
            and not ocr_engine_violations
            and not ocr_engine_count_violations
            and not repeat_ocr_engine_p95_violations
            and not repeat_ocr_engine_count_p95_violations
            and not repeat_ocr_engine_count_max_violations
        ),
        "primary_violations": primary_violations,
        "repeat_violations": repeat_violations,
    }
    if max_total_elapsed_s is not None:
        summary["max_total_elapsed_s"] = round(float(max_total_elapsed_s), 6)
    if max_repeat_profile_p95_duration_s is not None:
        summary["max_repeat_profile_p95_duration_s"] = round(float(max_repeat_profile_p95_duration_s), 6)
        summary["repeat_p95_violations"] = repeat_p95_violations
    if max_prewarm_runtime_s is not None:
        summary["max_prewarm_runtime_s"] = round(float(max_prewarm_runtime_s), 6)
        summary["prewarm_violations"] = prewarm_violations
    if max_prewarm_stage_s:
        summary["max_prewarm_stage_s"] = {
            metric: round(float(seconds), 6)
            for metric, seconds in sorted(max_prewarm_stage_s.items())
        }
        summary["prewarm_stage_violations"] = prewarm_stage_violations
    if max_ocr_engine_duration_s:
        summary["max_ocr_engine_duration_s"] = {
            metric: round(float(seconds), 6)
            for metric, seconds in sorted(max_ocr_engine_duration_s.items())
        }
        summary["ocr_engine_violations"] = ocr_engine_violations
    if max_ocr_engine_count:
        summary["max_ocr_engine_count"] = {
            metric: round(float(count), 6)
            for metric, count in sorted(max_ocr_engine_count.items())
        }
        summary["ocr_engine_count_violations"] = ocr_engine_count_violations
    if max_repeat_ocr_engine_p95_duration_s:
        summary["max_repeat_ocr_engine_p95_duration_s"] = {
            metric: round(float(seconds), 6)
            for metric, seconds in sorted(max_repeat_ocr_engine_p95_duration_s.items())
        }
        summary["repeat_ocr_engine_p95_violations"] = repeat_ocr_engine_p95_violations
    if max_repeat_ocr_engine_p95_count:
        summary["max_repeat_ocr_engine_p95_count"] = {
            metric: round(float(count), 6)
            for metric, count in sorted(max_repeat_ocr_engine_p95_count.items())
        }
        summary["repeat_ocr_engine_count_p95_violations"] = repeat_ocr_engine_count_p95_violations
    if max_repeat_ocr_engine_max_count:
        summary["max_repeat_ocr_engine_max_count"] = {
            metric: round(float(count), 6)
            for metric, count in sorted(max_repeat_ocr_engine_max_count.items())
        }
        summary["repeat_ocr_engine_count_max_violations"] = repeat_ocr_engine_count_max_violations
    return summary


def prewarm_runtime_budget_violations(
    prewarm: dict[str, Any] | None,
    *,
    max_prewarm_runtime_s: float | None,
) -> list[dict[str, Any]]:
    if max_prewarm_runtime_s is None:
        return []
    budget = float(max_prewarm_runtime_s)
    total_s = prewarm.get("total_s") if isinstance(prewarm, dict) else None
    parsed = parse_nonnegative_float(total_s)
    if parsed is None:
        return [
            {
                "kind": "prewarm_runtime_missing",
                "max_prewarm_runtime_s": round(budget, 6),
            }
        ]
    if parsed <= budget:
        return []
    return [
        {
            "kind": "prewarm_runtime_budget_exceeded",
            "prewarm_total_s": round(parsed, 6),
            "max_prewarm_runtime_s": round(budget, 6),
            "excess_s": round(parsed - budget, 6),
        }
    ]


def prewarm_stage_budget_violations(
    prewarm: dict[str, Any] | None,
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for metric, budget in sorted(budgets.items()):
        stage_s = prewarm.get(metric) if isinstance(prewarm, dict) else None
        parsed = parse_nonnegative_float(stage_s)
        if parsed is None:
            violations.append(
                {
                    "kind": "prewarm_stage_missing",
                    "metric": metric,
                    "max_prewarm_stage_s": round(float(budget), 6),
                }
            )
            continue
        if parsed > budget:
            violations.append(
                {
                    "kind": "prewarm_stage_budget_exceeded",
                    "metric": metric,
                    "duration_s": round(parsed, 6),
                    "max_prewarm_stage_s": round(float(budget), 6),
                    "excess_s": round(parsed - budget, 6),
                }
            )
    return violations


def repeat_profile_p95_budget_violations(
    repeat_profile: dict[str, Any] | None,
    *,
    max_repeat_profile_p95_duration_s: float | None,
) -> list[dict[str, Any]]:
    if max_repeat_profile_p95_duration_s is None:
        return []
    budget = float(max_repeat_profile_p95_duration_s)
    if not isinstance(repeat_profile, dict):
        return [
            {
                "kind": "repeat_profile_p95_missing",
                "max_repeat_profile_p95_duration_s": round(budget, 6),
            }
        ]
    summary = repeat_profile.get("summary")
    p95_duration = summary.get("p95_total_elapsed_s") if isinstance(summary, dict) else None
    parsed = parse_nonnegative_float(p95_duration)
    if parsed is None:
        return [
            {
                "kind": "repeat_profile_p95_missing",
                "max_repeat_profile_p95_duration_s": round(budget, 6),
            }
        ]
    if parsed <= budget:
        return []
    return [
        {
            "kind": "repeat_profile_p95_budget_exceeded",
            "p95_total_elapsed_s": round(parsed, 6),
            "max_repeat_profile_p95_duration_s": round(budget, 6),
            "excess_s": round(parsed - budget, 6),
        }
    ]


def parse_metric_duration_budgets(raw_budgets: list[str]) -> dict[str, float]:
    budgets: dict[str, float] = {}
    for raw in raw_budgets:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"OCR engine budget must use METRIC=SECONDS: {entry}")
            raw_metric, raw_value = (part.strip() for part in entry.split("=", 1))
            if not raw_metric:
                raise ValueError(f"OCR engine budget is missing a metric name: {entry}")
            metric = normalize_ocr_engine_duration_metric(raw_metric)
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"OCR engine budget seconds must be numeric: {entry}") from exc
            if value <= 0.0:
                raise ValueError(f"OCR engine budget seconds must be positive: {entry}")
            budgets[metric] = value
    return dict(sorted(budgets.items()))


def parse_prewarm_stage_duration_budgets(raw_budgets: list[str]) -> dict[str, float]:
    budgets: dict[str, float] = {}
    for raw in raw_budgets:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"Prewarm stage budget must use METRIC=SECONDS: {entry}")
            raw_metric, raw_value = (part.strip() for part in entry.split("=", 1))
            if not raw_metric:
                raise ValueError(f"Prewarm stage budget is missing a metric name: {entry}")
            metric = normalize_prewarm_stage_duration_metric(raw_metric)
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Prewarm stage budget seconds must be numeric: {entry}") from exc
            if value <= 0.0:
                raise ValueError(f"Prewarm stage budget seconds must be positive: {entry}")
            budgets[metric] = value
    return dict(sorted(budgets.items()))


def parse_metric_count_budgets(raw_budgets: list[str]) -> dict[str, float]:
    budgets: dict[str, float] = {}
    for raw in raw_budgets:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"OCR engine count budget must use METRIC=COUNT: {entry}")
            raw_metric, raw_value = (part.strip() for part in entry.split("=", 1))
            if not raw_metric:
                raise ValueError(f"OCR engine count budget is missing a metric name: {entry}")
            metric = normalize_ocr_engine_count_metric(raw_metric)
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"OCR engine count budget must be numeric: {entry}") from exc
            if value < 0.0:
                raise ValueError(f"OCR engine count budget must be non-negative: {entry}")
            budgets[metric] = value
    return dict(sorted(budgets.items()))


def normalize_ocr_engine_duration_metric(metric: str) -> str:
    normalized = OCR_ENGINE_STAGE_METRIC_ALIASES.get(metric, metric)
    if normalized not in OCR_ENGINE_STAGE_MAX_KEYS:
        expected = ", ".join(OCR_ENGINE_STAGE_MAX_KEYS)
        raise ValueError(f"Unknown OCR engine duration metric {metric!r}; expected one of: {expected}")
    return normalized


def normalize_prewarm_stage_duration_metric(metric: str) -> str:
    normalized = PREWARM_STAGE_METRIC_ALIASES.get(metric.strip(), metric.strip())
    if normalized not in PREWARM_STAGE_DURATION_KEYS:
        expected = ", ".join(PREWARM_STAGE_DURATION_KEYS)
        raise ValueError(f"Unknown prewarm stage metric {metric!r}; expected one of: {expected}")
    return normalized


def normalize_ocr_engine_count_metric(metric: str) -> str:
    normalized = metric.strip()
    if normalized not in OCR_ENGINE_COUNT_METRIC_KEYS:
        expected = ", ".join(OCR_ENGINE_COUNT_METRIC_KEYS)
        raise ValueError(f"Unknown OCR engine count metric {metric!r}; expected one of: {expected}")
    return normalized


def repeat_ocr_engine_p95_budget_violations(
    repeat_profile: dict[str, Any] | None,
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    if not budgets:
        return []
    if not isinstance(repeat_profile, dict):
        return [
            {
                "kind": "repeat_ocr_engine_p95_missing",
                "metric": metric,
                "max_repeat_ocr_engine_p95_duration_s": round(float(budget), 6),
            }
            for metric, budget in sorted(budgets.items())
        ]
    if repeat_profile_has_only_zero_ocr_engine_calls(repeat_profile):
        return []
    summary = repeat_profile.get("summary")
    stage_stats = summary.get("ocr_engine_stage_duration_s") if isinstance(summary, dict) else None
    if not isinstance(stage_stats, dict):
        stage_stats = {}
    violations: list[dict[str, Any]] = []
    for metric, budget in sorted(budgets.items()):
        stats = stage_stats.get(metric)
        p95_duration_s = stats.get("p95_duration_s") if isinstance(stats, dict) else None
        parsed = parse_nonnegative_float(p95_duration_s)
        if parsed is None:
            violations.append(
                {
                    "kind": "repeat_ocr_engine_p95_missing",
                    "metric": metric,
                    "max_repeat_ocr_engine_p95_duration_s": round(float(budget), 6),
                }
            )
            continue
        if parsed > budget:
            violations.append(
                {
                    "kind": "repeat_ocr_engine_p95_budget_exceeded",
                    "metric": metric,
                    "p95_duration_s": round(parsed, 6),
                    "max_repeat_ocr_engine_p95_duration_s": round(float(budget), 6),
                    "excess_s": round(parsed - budget, 6),
                }
            )
    return violations


def ocr_engine_duration_budget_violations(
    rows: list[dict[str, Any]],
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        for metric, budget in sorted(budgets.items()):
            duration_s = row_ocr_engine_metric(row, metric)
            if duration_s is None:
                if row_has_no_ocr_engine_calls(row):
                    continue
                violations.append(
                    ocr_engine_budget_missing_violation(
                        row,
                        metric=metric,
                        kind="ocr_engine_duration_missing",
                        budget_key="max_ocr_engine_duration_s",
                        budget=budget,
                    )
                )
                continue
            if duration_s > budget:
                violation = ocr_engine_budget_base_violation(
                    row,
                    kind="ocr_engine_duration_budget_exceeded",
                    metric=metric,
                )
                violation.update(
                    {
                        "duration_s": round(duration_s, 6),
                        "max_ocr_engine_duration_s": round(float(budget), 6),
                        "excess_s": round(duration_s - budget, 6),
                    }
                )
                violations.append(violation)
    return violations


def ocr_engine_count_budget_violations(
    rows: list[dict[str, Any]],
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        for metric, budget in sorted(budgets.items()):
            count = row_ocr_engine_metric(row, metric)
            if count is None:
                if row_has_no_ocr_engine_calls(row):
                    continue
                violations.append(
                    ocr_engine_budget_missing_violation(
                        row,
                        metric=metric,
                        kind="ocr_engine_count_missing",
                        budget_key="max_ocr_engine_count",
                        budget=budget,
                    )
                )
                continue
            if count > budget:
                violation = ocr_engine_budget_base_violation(
                    row,
                    kind="ocr_engine_count_budget_exceeded",
                    metric=metric,
                )
                violation.update(
                    {
                        "count": round(count, 6),
                        "max_ocr_engine_count": round(float(budget), 6),
                        "excess_count": round(count - budget, 6),
                    }
                )
                violations.append(violation)
    return violations


def row_ocr_engine_metric(row: dict[str, Any], metric: str) -> float | None:
    profile = row.get("ocr_engine_profile")
    if not isinstance(profile, dict):
        return None
    parsed = parse_nonnegative_float(profile.get(metric))
    if parsed is not None:
        return parsed
    detail_profile = slowest_ocr_engine_detail_profile(profile)
    if detail_profile is None:
        return None
    return parse_nonnegative_float(detail_profile.get(metric))


def row_has_no_ocr_engine_calls(row: dict[str, Any]) -> bool:
    profile = row.get("ocr_engine_profile")
    if not isinstance(profile, dict):
        return False
    calls = parse_nonnegative_float(profile.get("calls"))
    return calls == 0.0


def repeat_profile_has_only_zero_ocr_engine_calls(repeat_profile: dict[str, Any]) -> bool:
    samples = repeat_profile.get("samples")
    if not isinstance(samples, list):
        return False
    analyzed_samples = repeat_profile_analyzed_samples([sample for sample in samples if isinstance(sample, dict)])
    return bool(analyzed_samples) and all(row_has_no_ocr_engine_calls(sample) for sample in analyzed_samples)


def ocr_engine_budget_missing_violation(
    row: dict[str, Any],
    *,
    metric: str,
    kind: str,
    budget_key: str,
    budget: float,
) -> dict[str, Any]:
    violation = ocr_engine_budget_base_violation(row, kind=kind, metric=metric)
    violation[budget_key] = round(float(budget), 6)
    return violation


def ocr_engine_budget_base_violation(
    row: dict[str, Any],
    *,
    kind: str,
    metric: str,
) -> dict[str, Any]:
    violation: dict[str, Any] = {
        "kind": kind,
        "slug": row.get("slug"),
        "metric": metric,
    }
    observed_status = row.get("observed_status")
    if observed_status is not None:
        violation["observed_status"] = observed_status
    return violation


def repeat_ocr_engine_count_p95_budget_violations(
    repeat_profile: dict[str, Any] | None,
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    if not budgets:
        return []
    if not isinstance(repeat_profile, dict):
        return [
            {
                "kind": "repeat_ocr_engine_count_p95_missing",
                "metric": metric,
                "max_repeat_ocr_engine_p95_count": round(float(budget), 6),
            }
            for metric, budget in sorted(budgets.items())
        ]
    if repeat_profile_has_only_zero_ocr_engine_calls(repeat_profile):
        return []
    summary = repeat_profile.get("summary")
    count_stats = summary.get("ocr_engine_count_metric") if isinstance(summary, dict) else None
    if not isinstance(count_stats, dict):
        count_stats = {}
    violations: list[dict[str, Any]] = []
    for metric, budget in sorted(budgets.items()):
        stats = count_stats.get(metric)
        p95_count = stats.get("p95_count") if isinstance(stats, dict) else None
        parsed = parse_nonnegative_float(p95_count)
        if parsed is None:
            violations.append(
                {
                    "kind": "repeat_ocr_engine_count_p95_missing",
                    "metric": metric,
                    "max_repeat_ocr_engine_p95_count": round(float(budget), 6),
                }
            )
            continue
        if parsed > budget:
            violations.append(
                {
                    "kind": "repeat_ocr_engine_count_p95_budget_exceeded",
                    "metric": metric,
                    "p95_count": round(parsed, 6),
                    "max_repeat_ocr_engine_p95_count": round(float(budget), 6),
                    "excess_count": round(parsed - budget, 6),
                }
            )
    return violations


def repeat_ocr_engine_count_max_budget_violations(
    repeat_profile: dict[str, Any] | None,
    budgets: dict[str, float],
) -> list[dict[str, Any]]:
    if not budgets:
        return []
    if not isinstance(repeat_profile, dict):
        return [
            {
                "kind": "repeat_ocr_engine_count_max_missing",
                "metric": metric,
                "max_repeat_ocr_engine_max_count": round(float(budget), 6),
            }
            for metric, budget in sorted(budgets.items())
        ]
    if repeat_profile_has_only_zero_ocr_engine_calls(repeat_profile):
        return []
    summary = repeat_profile.get("summary")
    count_stats = summary.get("ocr_engine_count_metric") if isinstance(summary, dict) else None
    if not isinstance(count_stats, dict):
        count_stats = {}
    violations: list[dict[str, Any]] = []
    for metric, budget in sorted(budgets.items()):
        stats = count_stats.get(metric)
        max_count = stats.get("max_count") if isinstance(stats, dict) else None
        parsed = parse_nonnegative_float(max_count)
        if parsed is None:
            violations.append(
                {
                    "kind": "repeat_ocr_engine_count_max_missing",
                    "metric": metric,
                    "max_repeat_ocr_engine_max_count": round(float(budget), 6),
                }
            )
            continue
        if parsed > budget:
            violations.append(
                {
                    "kind": "repeat_ocr_engine_count_max_budget_exceeded",
                    "metric": metric,
                    "max_count": round(parsed, 6),
                    "max_repeat_ocr_engine_max_count": round(float(budget), 6),
                    "excess_count": round(parsed - budget, 6),
                }
            )
    return violations


def latency_budget_violations(
    rows: list[dict[str, Any]],
    *,
    max_total_elapsed_s: float,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        total_elapsed_s = parse_nonnegative_float(row.get("total_elapsed_s"))
        if total_elapsed_s is None or total_elapsed_s <= max_total_elapsed_s:
            continue
        violation: dict[str, Any] = {
            "slug": row.get("slug"),
            "total_elapsed_s": round(total_elapsed_s, 6),
            "over_by_s": round(total_elapsed_s - max_total_elapsed_s, 6),
        }
        repeat_index = row.get("repeat_index")
        if isinstance(repeat_index, int):
            violation["repeat_index"] = repeat_index
        observed_status = row.get("observed_status")
        if observed_status is not None:
            violation["observed_status"] = observed_status
        violations.append(violation)
    return violations


def count_repeat_expectation_passed_samples(samples: list[dict[str, Any]]) -> int:
    return sum(sample.get("expectation_passed") is True for sample in samples)


def count_repeat_subsecond_samples(samples: list[dict[str, Any]]) -> int:
    return sum(
        elapsed_s is not None and elapsed_s < 1.0
        for elapsed_s in (parse_nonnegative_float(sample.get("total_elapsed_s")) for sample in samples)
    )


def repeat_profile_total_elapsed_stats(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    durations = [
        elapsed_s
        for elapsed_s in (parse_nonnegative_float(sample.get("total_elapsed_s")) for sample in samples)
        if elapsed_s is not None
    ]
    if not durations:
        return {
            "min_total_elapsed_s": None,
            "median_total_elapsed_s": None,
            "average_total_elapsed_s": None,
            "p90_total_elapsed_s": None,
            "p95_total_elapsed_s": None,
            "max_total_elapsed_s": None,
        }
    return {
        "min_total_elapsed_s": round(min(durations), 6),
        "median_total_elapsed_s": round(float(median(durations)), 6),
        "average_total_elapsed_s": round(float(mean(durations)), 6),
        "p90_total_elapsed_s": round(percentile_linear(durations, 90), 6),
        "p95_total_elapsed_s": round(percentile_linear(durations, 95), 6),
        "max_total_elapsed_s": round(max(durations), 6),
    }


def repeat_profile_stage_duration_stats(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stage_durations: dict[str, list[float]] = {}
    for sample in samples:
        stages = sample.get("stages")
        if not isinstance(stages, dict):
            continue
        for stage, elapsed_s in stages.items():
            if not isinstance(stage, str) or not stage:
                continue
            parsed = parse_nonnegative_float(elapsed_s)
            if parsed is None:
                continue
            stage_durations.setdefault(stage, []).append(parsed)
    return {
        stage: {
            "samples": len(durations),
            **repeat_profile_stage_duration_distribution(durations),
        }
        for stage, durations in sorted(stage_durations.items())
    }


def repeat_profile_stage_duration_distribution(durations: list[float]) -> dict[str, float | None]:
    if not durations:
        return {
            "min_duration_s": None,
            "median_duration_s": None,
            "average_duration_s": None,
            "p90_duration_s": None,
            "p95_duration_s": None,
            "max_duration_s": None,
        }
    return {
        "min_duration_s": round(min(durations), 6),
        "median_duration_s": round(float(median(durations)), 6),
        "average_duration_s": round(float(mean(durations)), 6),
        "p90_duration_s": round(percentile_linear(durations, 90), 6),
        "p95_duration_s": round(percentile_linear(durations, 95), 6),
        "max_duration_s": round(max(durations), 6),
    }


def percentile_linear(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def summarize_repeat_profile_ocr_engine(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    profiles = [
        profile
        for profile in (sample.get("ocr_engine_profile") for sample in samples)
        if isinstance(profile, dict)
    ]
    return summarize_rapidocr_profile_summaries(profiles)


def repeat_profile_ocr_engine_stage_duration_stats(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    durations: dict[str, list[float]] = {}
    for sample in samples:
        ocr_engine_profile = sample.get("ocr_engine_profile")
        if not isinstance(ocr_engine_profile, dict):
            continue
        calls = ocr_engine_profile.get("calls_detail")
        call_profiles = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
        if not call_profiles:
            call_profiles = [ocr_engine_profile]
        for call in call_profiles:
            for key in OCR_ENGINE_STAGE_MAX_KEYS:
                elapsed_s = parse_nonnegative_float(call.get(key))
                if elapsed_s is None:
                    continue
                durations.setdefault(key, []).append(elapsed_s)
    return {
        key: {
            "samples": len(values),
            **repeat_profile_stage_duration_distribution(values),
        }
        for key, values in sorted(durations.items())
    }


def repeat_profile_ocr_engine_count_stats(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, list[float]] = {}
    for sample in samples:
        ocr_engine_profile = sample.get("ocr_engine_profile")
        if not isinstance(ocr_engine_profile, dict):
            continue
        calls = ocr_engine_profile.get("calls_detail")
        call_profiles = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
        for call in call_profiles:
            for key in OCR_ENGINE_COUNT_METRIC_KEYS:
                if key in ocr_engine_profile:
                    continue
                value = parse_nonnegative_count_metric(call.get(key))
                if value is None:
                    continue
                counts.setdefault(key, []).append(value)
        for key in OCR_ENGINE_COUNT_METRIC_KEYS:
            value = parse_nonnegative_count_metric(ocr_engine_profile.get(key))
            if value is None:
                continue
            counts.setdefault(key, []).append(value)
    return {
        key: {
            "samples": len(values),
            **repeat_profile_count_distribution(values),
        }
        for key, values in sorted(counts.items())
    }


def repeat_profile_ocr_overlap_hidden_stats(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    durations = [
        hidden_s
        for hidden_s in (stress_row_ocr_overlap_hidden_value(sample) for sample in samples)
        if hidden_s is not None
    ]
    if not durations:
        return None
    return {
        "samples": len(durations),
        "total_s": round(sum(durations), 6),
        **repeat_profile_stage_duration_distribution(durations),
    }


def repeat_profile_count_distribution(counts: list[float]) -> dict[str, float | None]:
    if not counts:
        return {
            "min_count": None,
            "median_count": None,
            "average_count": None,
            "p90_count": None,
            "p95_count": None,
            "max_count": None,
        }
    return {
        "min_count": round(min(counts), 6),
        "median_count": round(float(median(counts)), 6),
        "average_count": round(float(mean(counts)), 6),
        "p90_count": round(percentile_linear(counts, 90), 6),
        "p95_count": round(percentile_linear(counts, 95), 6),
        "max_count": round(max(counts), 6),
    }


def parse_nonnegative_count_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return parse_nonnegative_float(value)


def parse_nonnegative_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0.0 else None


def ocr_overlap_hidden_seconds(row: dict[str, Any]) -> float | None:
    profile = row.get("ocr_engine_profile")
    stages = row.get("stages")
    if not isinstance(profile, dict) or not isinstance(stages, dict):
        return None
    ocr_total_s = parse_nonnegative_float(profile.get("total_s"))
    stage_ocr_s = parse_nonnegative_float(stages.get("ocr"))
    if ocr_total_s is None or stage_ocr_s is None:
        return None
    hidden_s = ocr_total_s - stage_ocr_s
    if hidden_s <= 0.0:
        return None
    return round(hidden_s, 6)


def primary_slowest_cases_include_input_kind(cases: list[dict[str, Any]]) -> bool:
    for case in cases:
        ocr_engine = case.get("ocr_engine")
        if isinstance(ocr_engine, dict) and isinstance(ocr_engine.get("input_kind"), str):
            return True
    return False


def primary_slowest_cases_include_overlap_hidden(cases: list[dict[str, Any]]) -> bool:
    for case in cases:
        ocr_engine = case.get("ocr_engine")
        if isinstance(ocr_engine, dict) and parse_nonnegative_float(ocr_engine.get("overlap_hidden_s")) is not None:
            return True
    return False


def primary_ocr_engine_slowest_cases_include_input_kind(cases: list[dict[str, Any]]) -> bool:
    return any(isinstance(case.get("input_kind"), str) for case in cases)


def primary_ocr_engine_slowest_cases_include_overlap_hidden(cases: list[dict[str, Any]]) -> bool:
    return any(parse_nonnegative_float(case.get("overlap_hidden_s")) is not None for case in cases)


def repeat_profile_slowest_samples_include_input_kind(samples: list[dict[str, Any]]) -> bool:
    for sample in samples:
        ocr_engine = sample.get("ocr_engine")
        if isinstance(ocr_engine, dict) and isinstance(ocr_engine.get("input_kind"), str):
            return True
    return False


def repeat_profile_slowest_cases_include_top_stage(cases: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(case, dict) and isinstance(case.get("top_stage"), dict)
        for case in cases
    )


def rebuilt_repeat_profile_slowest_cases(repeat_profile: dict[str, Any]) -> list[dict[str, Any]]:
    repeat_cases = stress_report_repeat_profile_cases({"repeat_profile": repeat_profile})
    if repeat_cases is None:
        return []
    return repeat_profile_slowest_cases(repeat_cases)


def repeat_profile_ocr_engine_slowest_cases_include_detail_context(cases: list[dict[str, Any]]) -> bool:
    return any(
        any(case.get(key) is not None for key in OCR_ENGINE_DETAIL_CONTEXT_KEYS)
        for case in cases
        if isinstance(case, dict)
    )


def repeat_profile_ocr_engine_slowest_cases_include_recent_context(cases: list[dict[str, Any]]) -> bool:
    recent_keys = ("rec_batch_num", "classifier_retry", "header_region_filter")
    return any(
        any(case.get(key) is not None for key in recent_keys)
        for case in cases
        if isinstance(case, dict)
    )


def rebuilt_repeat_profile_ocr_engine_slowest_cases(repeat_profile: dict[str, Any]) -> list[dict[str, Any]]:
    repeat_samples = repeat_profile.get("samples")
    if not isinstance(repeat_samples, list):
        return []
    runs_per_case = repeat_profile.get("runs_per_case")
    warmup_runs_per_case = repeat_profile.get("warmup_runs_per_case")
    rebuilt = summarize_repeat_profile_samples(
        [sample for sample in repeat_samples if isinstance(sample, dict)],
        runs_per_case=(
            runs_per_case
            if isinstance(runs_per_case, int) and not isinstance(runs_per_case, bool)
            else 0
        ),
        warmup_runs_per_case=(
            warmup_runs_per_case
            if isinstance(warmup_runs_per_case, int) and not isinstance(warmup_runs_per_case, bool)
            else 0
        ),
    )
    rebuilt_summary = rebuilt.get("summary")
    if not isinstance(rebuilt_summary, dict):
        return []
    rebuilt_cases = rebuilt_summary.get("ocr_engine_slowest_cases")
    return rebuilt_cases if isinstance(rebuilt_cases, list) else []


def should_use_rebuilt_repeat_ocr_slowest_cases(
    current_cases: list[dict[str, Any]],
    rebuilt_cases: list[dict[str, Any]],
) -> bool:
    if not repeat_profile_ocr_engine_slowest_cases_include_detail_context(rebuilt_cases):
        return False
    if not repeat_profile_ocr_engine_slowest_cases_include_detail_context(current_cases):
        return True
    return (
        not repeat_profile_ocr_engine_slowest_cases_include_recent_context(current_cases)
        and repeat_profile_ocr_engine_slowest_cases_include_recent_context(rebuilt_cases)
    )


def ocr_engine_stage_max_rows_include_input_kind(max_rows: Any) -> bool:
    if not isinstance(max_rows, dict):
        return False
    return any(
        isinstance(row, dict) and isinstance(row.get("input_kind"), str)
        for row in max_rows.values()
    )


def print_stress_table(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "stress summary: "
        f"{summary['expectation_passed']}/{summary['total']} expected, "
        f"statuses={summary['statuses']}, "
        f"max_total_elapsed_s={summary['max_total_elapsed_s']}"
    )
    manifest_contracts = report.get("manifest_contracts")
    if isinstance(manifest_contracts, dict):
        total_cases = manifest_contracts.get("total_cases")
        call_contract_rows = manifest_contracts.get("ocr_call_contract_rows")
        positive_call_rows = manifest_contracts.get("ocr_positive_call_contract_rows")
        count_contract_rows = manifest_contracts.get("ocr_count_contract_rows")
        positive_call_rows_without_count_contract = manifest_contracts.get(
            "ocr_positive_call_rows_without_count_contract"
        )
        count_contract_missing = (
            len(positive_call_rows_without_count_contract)
            if isinstance(positive_call_rows_without_count_contract, list)
            else 0
        )
        if all(
            isinstance(value, int)
            for value in (
                total_cases,
                call_contract_rows,
                positive_call_rows,
                count_contract_rows,
            )
        ):
            print(
                "manifest OCR contracts: "
                f"calls={call_contract_rows}/{total_cases}, "
                f"count-capped={count_contract_rows}/{positive_call_rows} positive-call rows, "
                f"positive-call-only={count_contract_missing}"
            )
    manifest_contract_budget = report.get("manifest_contract_budget")
    if isinstance(manifest_contract_budget, dict):
        if manifest_contract_budget.get("passed"):
            print("manifest contract budget: passed")
        else:
            violations = manifest_contract_budget.get("violations")
            violation_count = len(violations) if isinstance(violations, list) else 0
            print(f"manifest contract budget: failed violations={violation_count}")
            if isinstance(violations, list):
                for issue in violations[:6]:
                    if not isinstance(issue, dict):
                        continue
                    if issue.get("kind") == "ocr_call_contract_rows_below_min":
                        print(
                            "   - "
                            f"ocr call contracts {issue['ocr_call_contract_rows']} "
                            f"< minimum {issue['min_ocr_call_contract_rows']}"
                        )
                    elif issue.get("kind") == "ocr_count_contract_rows_below_min":
                        print(
                            "   - "
                            f"ocr count contracts {issue['ocr_count_contract_rows']} "
                            f"< minimum {issue['min_ocr_count_contract_rows']}"
                        )
                    elif issue.get("kind") == "positive_ocr_call_only_rows_above_max":
                        print(
                            "   - "
                            f"positive-call-only rows {issue['positive_ocr_call_only_rows']} "
                            f"> maximum {issue['max_positive_ocr_call_only_rows']}"
                        )
                    elif issue.get("kind") == "invalid_ocr_count_contracts":
                        invalid_rows = issue.get("invalid_ocr_count_contract_rows")
                        invalid_count = len(invalid_rows) if isinstance(invalid_rows, list) else 0
                        print(f"   - invalid ocr count contracts: {invalid_count}")
    baseline_comparison = report.get("baseline_comparison")
    if isinstance(baseline_comparison, dict):
        signature_changes = baseline_comparison.get("signature_changes")
        signature_change_count = (
            len(signature_changes) if isinstance(signature_changes, list) else 0
        )
        missing_in_baseline = baseline_comparison.get("missing_in_baseline")
        missing_in_candidate = baseline_comparison.get("missing_in_candidate")
        missing_baseline_count = len(missing_in_baseline) if isinstance(missing_in_baseline, list) else 0
        missing_candidate_count = len(missing_in_candidate) if isinstance(missing_in_candidate, list) else 0
        candidate_scope = baseline_comparison.get("candidate_scope")
        baseline_out_of_scope_count = (
            candidate_scope.get("baseline_rows_outside_candidate_scope_count")
            if isinstance(candidate_scope, dict)
            else None
        )
        baseline_out_of_scope_text = (
            f", baseline_out_of_scope={baseline_out_of_scope_count}"
            if isinstance(baseline_out_of_scope_count, int) and baseline_out_of_scope_count
            else ""
        )
        median_delta = baseline_comparison.get("median_total_elapsed_delta_s")
        median_delta_text = (
            f", median_delta={float(median_delta):+.3f}s"
            if isinstance(median_delta, (int, float))
            else ""
        )
        repeat_case_coverage_text = baseline_repeat_case_coverage_text(baseline_comparison)
        signature_field_counts_text = baseline_signature_field_counts_text(baseline_comparison)
        print(
            "baseline comparison: "
            f"compared={baseline_comparison.get('compared_rows', 0)}, "
            f"signature_changes={signature_change_count}, "
            f"missing_baseline={missing_baseline_count}, "
            f"missing_candidate={missing_candidate_count}"
            f"{baseline_out_of_scope_text}"
            f"{repeat_case_coverage_text}"
            f"{median_delta_text}"
            f"{signature_field_counts_text}"
        )
        config_changes_text = baseline_configuration_changes_text(baseline_comparison)
        if config_changes_text:
            print(f"baseline config changes: {config_changes_text}")
        expectation_delta_text = baseline_expectation_delta_text(baseline_comparison)
        if expectation_delta_text:
            print(f"baseline expectation delta: {expectation_delta_text}")
        primary_delta_text = baseline_primary_delta_text(baseline_comparison)
        if primary_delta_text:
            print(f"baseline primary delta: {primary_delta_text}")
        primary_ocr_delta_text = baseline_primary_ocr_delta_text(baseline_comparison)
        if primary_ocr_delta_text:
            print(f"baseline primary OCR delta: {primary_ocr_delta_text}")
        primary_hidden_delta_text = baseline_primary_ocr_overlap_hidden_delta_text(baseline_comparison)
        if primary_hidden_delta_text:
            print(f"baseline primary hidden OCR delta: {primary_hidden_delta_text}")
        repeat_delta = baseline_comparison.get("repeat_profile_delta")
        repeat_delta_text = baseline_repeat_delta_text(
            repeat_delta if isinstance(repeat_delta, dict) else None
        )
        if repeat_delta_text:
            print(f"baseline repeat delta: {repeat_delta_text}")
        repeat_stage_delta_text = baseline_repeat_stage_delta_text(
            repeat_delta if isinstance(repeat_delta, dict) else None
        )
        if repeat_stage_delta_text:
            print(f"baseline repeat stage delta: {repeat_stage_delta_text}")
        repeat_case_delta_text = baseline_repeat_case_delta_text(baseline_comparison)
        if repeat_case_delta_text:
            print(f"baseline repeat case delta: {repeat_case_delta_text}")
        repeat_ocr_case_delta_text = baseline_repeat_ocr_case_delta_text(baseline_comparison)
        if repeat_ocr_case_delta_text:
            print(f"baseline repeat OCR case delta: {repeat_ocr_case_delta_text}")
        repeat_ocr_stage_delta_text = baseline_repeat_ocr_stage_delta_text(
            repeat_delta if isinstance(repeat_delta, dict) else None
        )
        if repeat_ocr_stage_delta_text:
            print(f"baseline repeat OCR stage delta: {repeat_ocr_stage_delta_text}")
        regression_budget = baseline_comparison.get("regression_budget")
        if isinstance(regression_budget, dict):
            print(baseline_regression_budget_text(regression_budget))
            violations = regression_budget.get("violations")
            if regression_budget.get("passed") is False and isinstance(violations, list):
                for violation in baseline_regression_budget_violation_samples(violations, limit=6):
                    violation_text = baseline_regression_budget_violation_text(violation)
                    if violation_text:
                        print(f"   - {violation_text}")
        if isinstance(signature_changes, list):
            for change in signature_changes[:5]:
                if not isinstance(change, dict):
                    continue
                fields = change.get("changed_fields")
                fields_text = (
                    f" fields={','.join(fields)}"
                    if isinstance(fields, list) and all(isinstance(field, str) for field in fields)
                    else ""
                )
                print(f"   - signature drift: {change.get('slug')}{fields_text}")
        expectation_changes = baseline_comparison.get("expectation_changes")
        if isinstance(expectation_changes, list):
            for change in expectation_changes[:5]:
                if not isinstance(change, dict):
                    continue
                change_text = baseline_expectation_change_text(change)
                if change_text:
                    print(f"   - expectation drift: {change_text}")
    if summary.get("stage_duration_s"):
        stage_total_text = ", ".join(
            f"{stage}={elapsed_s:.3f}s" for stage, elapsed_s in summary["stage_duration_s"].items()
        )
        print(f"stage totals: {stage_total_text}")
    if summary.get("stage_max_rows"):
        stage_max_text = ", ".join(
            f"{stage}={row['elapsed_s']:.3f}s@{row['slug']}" for stage, row in summary["stage_max_rows"].items()
        )
        print(f"stage max: {stage_max_text}")
    rows = report.get("rows")
    ocr_overlap_hidden = summary.get("ocr_overlap_hidden_s")
    if not isinstance(ocr_overlap_hidden, dict) and isinstance(rows, list):
        ocr_overlap_hidden = summarize_ocr_overlap_hidden(rows)
    if isinstance(ocr_overlap_hidden, dict):
        hidden_total = parse_nonnegative_float(ocr_overlap_hidden.get("total_s"))
        hidden_max = parse_nonnegative_float(ocr_overlap_hidden.get("max_s"))
        hidden_rows = ocr_overlap_hidden.get("rows")
        hidden_slug = ocr_overlap_hidden.get("max_slug")
        if hidden_total is not None and hidden_max is not None and isinstance(hidden_rows, int):
            suffix = f"@{hidden_slug}" if isinstance(hidden_slug, str) and hidden_slug else ""
            print(
                "ocr overlap hidden: "
                f"total={hidden_total:.3f}s, max={hidden_max:.3f}s{suffix}, rows={hidden_rows}"
            )
    primary_slowest_cases = summary.get("slowest_cases")
    if not isinstance(primary_slowest_cases, list):
        primary_slowest_cases = primary_slowest_cases_from_rows(rows) if isinstance(rows, list) else []
    elif isinstance(rows, list):
        enriched_primary_slowest_cases = primary_slowest_cases_from_rows(rows)
        if (
            not primary_slowest_cases_include_input_kind(primary_slowest_cases)
            and primary_slowest_cases_include_input_kind(enriched_primary_slowest_cases)
        ) or (
            not primary_slowest_cases_include_overlap_hidden(primary_slowest_cases)
            and primary_slowest_cases_include_overlap_hidden(enriched_primary_slowest_cases)
        ):
            primary_slowest_cases = enriched_primary_slowest_cases
    if primary_slowest_cases:
        primary_slow_case_text = ", ".join(
            primary_slow_case_text_from_summary(case)
            for case in primary_slowest_cases[:5]
            if isinstance(case, dict)
        )
        if primary_slow_case_text:
            print(f"primary slowest cases: {primary_slow_case_text}")
    if summary.get("ocr_full_detail_retry_count"):
        retry_rows = ", ".join(summary.get("ocr_full_detail_retry_rows", []))
        print(f"ocr full-detail retries: {summary['ocr_full_detail_retry_count']} ({retry_rows})")
    if summary.get("ocr_engine_profile"):
        if report.get("profile_ocr_engine"):
            print("note: OCR engine profiling is enabled; case durations include profiling overhead")
        profile = summary["ocr_engine_profile"]
        print(
            "ocr engine: "
            f"calls={profile.get('calls', 0)}, "
            f"det={float(profile.get('det_elapsed_s', 0.0)):.3f}s, "
            f"rec={float(profile.get('rec_elapsed_s', 0.0)):.3f}s"
        )
    if summary.get("ocr_engine_stage_max_rows"):
        max_rows = summary["ocr_engine_stage_max_rows"]
        if (
            not ocr_engine_stage_max_rows_include_input_kind(max_rows)
            and isinstance(rows, list)
        ):
            enriched_max_rows = ocr_engine_stage_max_rows(rows)
            if ocr_engine_stage_max_rows_include_input_kind(enriched_max_rows):
                max_rows = enriched_max_rows
        labels = {
            "input_s": "input",
            "det_elapsed_s": "det",
            "rec_elapsed_s": "rec",
            "total_s": "total",
        }
        max_text = ", ".join(
            ocr_engine_stage_max_row_text(labels.get(key, key), row)
            for key, row in max_rows.items()
            if isinstance(row, dict) and isinstance(row.get("elapsed_s"), (int, float))
        )
        if max_text:
            print(f"ocr engine max: {max_text}")
    primary_ocr_slowest_cases = summary.get("ocr_engine_slowest_cases")
    if not isinstance(primary_ocr_slowest_cases, list):
        primary_ocr_slowest_cases = (
            primary_ocr_engine_slowest_cases(rows) if isinstance(rows, list) else []
        )
    elif isinstance(rows, list):
        enriched_primary_ocr_slowest_cases = primary_ocr_engine_slowest_cases(rows)
        if (
            not primary_ocr_engine_slowest_cases_include_input_kind(primary_ocr_slowest_cases)
            and primary_ocr_engine_slowest_cases_include_input_kind(enriched_primary_ocr_slowest_cases)
        ) or (
            not primary_ocr_engine_slowest_cases_include_overlap_hidden(primary_ocr_slowest_cases)
            and primary_ocr_engine_slowest_cases_include_overlap_hidden(enriched_primary_ocr_slowest_cases)
        ):
            primary_ocr_slowest_cases = enriched_primary_ocr_slowest_cases
    if primary_ocr_slowest_cases:
        primary_ocr_slow_case_text = ", ".join(
            primary_ocr_engine_slow_case_text(case)
            for case in primary_ocr_slowest_cases[:5]
            if isinstance(case, dict)
        )
        if primary_ocr_slow_case_text:
            print(f"primary ocr slowest cases: {primary_ocr_slow_case_text}")
    if report.get("runner_ocr_cache") is False:
        print("note: runner OCR cache is disabled; repeat samples keep paying OCR cost")
    if report.get("extraction_cache") is False:
        print("note: extraction cache is disabled; repeat samples keep paying extraction cost")
    if report.get("prewarm_runtime"):
        prewarm = report.get("prewarm") if isinstance(report.get("prewarm"), dict) else {}
        status = prewarm.get("status", "missing")
        total_s = prewarm.get("total_s")
        total_text = f", total={float(total_s):.3f}s" if isinstance(total_s, (int, float)) else ""
        print(f"note: generation runtime prewarm status={status}{total_text}")
    latency_budget = report.get("latency_budget")
    if isinstance(latency_budget, dict):
        budget = latency_budget.get("max_total_elapsed_s")
        budget_text = f"{budget:.3f}s" if isinstance(budget, (int, float)) else "-"
        if latency_budget.get("passed"):
            suffix = f" total<={budget_text}" if isinstance(budget, (int, float)) else ""
            print(f"latency budget: passed{suffix}")
        else:
            ocr_engine = latency_budget.get("ocr_engine_violations")
            ocr_engine_count = len(ocr_engine) if isinstance(ocr_engine, list) else 0
            ocr_engine_counts = latency_budget.get("ocr_engine_count_violations")
            ocr_engine_count_budget_count = len(ocr_engine_counts) if isinstance(ocr_engine_counts, list) else 0
            repeat_ocr_engine = latency_budget.get("repeat_ocr_engine_p95_violations")
            repeat_ocr_engine_count = len(repeat_ocr_engine) if isinstance(repeat_ocr_engine, list) else 0
            repeat_ocr_engine_counts = latency_budget.get("repeat_ocr_engine_count_p95_violations")
            repeat_ocr_engine_count_budget_count = (
                len(repeat_ocr_engine_counts) if isinstance(repeat_ocr_engine_counts, list) else 0
            )
            repeat_ocr_engine_max_counts = latency_budget.get("repeat_ocr_engine_count_max_violations")
            repeat_ocr_engine_max_count_budget_count = (
                len(repeat_ocr_engine_max_counts) if isinstance(repeat_ocr_engine_max_counts, list) else 0
            )
            repeat_p95 = latency_budget.get("repeat_p95_violations")
            repeat_p95_count = len(repeat_p95) if isinstance(repeat_p95, list) else 0
            prewarm_violations = latency_budget.get("prewarm_violations")
            prewarm_violation_count = (
                len(prewarm_violations) if isinstance(prewarm_violations, list) else 0
            )
            prewarm_stage_violations = latency_budget.get("prewarm_stage_violations")
            prewarm_stage_violation_count = (
                len(prewarm_stage_violations) if isinstance(prewarm_stage_violations, list) else 0
            )
            total_budget_text = f" total<={budget_text}" if isinstance(budget, (int, float)) else ""
            print(
                "latency budget: failed"
                f"{total_budget_text} "
                f"primary={len(latency_budget.get('primary_violations', []))} "
                f"repeat={len(latency_budget.get('repeat_violations', []))} "
                f"repeat_p95={repeat_p95_count} "
                f"prewarm={prewarm_violation_count} "
                f"prewarm_stage={prewarm_stage_violation_count} "
                f"ocr_engine={ocr_engine_count} "
                f"ocr_engine_count={ocr_engine_count_budget_count} "
                f"repeat_ocr_engine_p95={repeat_ocr_engine_count} "
                f"repeat_ocr_engine_count_p95={repeat_ocr_engine_count_budget_count} "
                f"repeat_ocr_engine_count_max={repeat_ocr_engine_max_count_budget_count}"
            )
            if isinstance(prewarm_violations, list):
                for issue in prewarm_violations[:3]:
                    if not isinstance(issue, dict):
                        continue
                    if issue.get("kind") == "prewarm_runtime_budget_exceeded":
                        print(
                            "   - "
                            f"prewarm {float(issue['prewarm_total_s']):.3f}s "
                            f"> budget {float(issue['max_prewarm_runtime_s']):.3f}s"
                        )
                    elif issue.get("kind") == "prewarm_runtime_missing":
                        print("   - prewarm total_s metric missing")
            if isinstance(prewarm_stage_violations, list):
                for issue in prewarm_stage_violations[:5]:
                    if not isinstance(issue, dict):
                        continue
                    metric = issue.get("metric")
                    if issue.get("kind") == "prewarm_stage_budget_exceeded":
                        print(
                            "   - "
                            f"prewarm {metric} {float(issue['duration_s']):.3f}s "
                            f"> budget {float(issue['max_prewarm_stage_s']):.3f}s"
                        )
                    elif issue.get("kind") == "prewarm_stage_missing":
                        print(f"   - prewarm {metric} metric missing")
            if isinstance(repeat_p95, list):
                for issue in repeat_p95[:3]:
                    if not isinstance(issue, dict):
                        continue
                    if issue.get("kind") == "repeat_profile_p95_budget_exceeded":
                        print(
                            "   - "
                            f"repeat profile p95 {float(issue['p95_total_elapsed_s']):.3f}s "
                            f"> budget {float(issue['max_repeat_profile_p95_duration_s']):.3f}s"
                        )
                    elif issue.get("kind") == "repeat_profile_p95_missing":
                        print("   - repeat profile p95 metric missing")
            if isinstance(ocr_engine, list):
                for issue in ocr_engine[:6]:
                    if not isinstance(issue, dict):
                        continue
                    metric = issue.get("metric")
                    slug = issue.get("slug")
                    if issue.get("kind") == "ocr_engine_duration_budget_exceeded":
                        print(
                            "   - "
                            f"{slug}: {metric} primary OCR engine {float(issue['duration_s']):.3f}s "
                            f"> budget {float(issue['max_ocr_engine_duration_s']):.3f}s"
                        )
                    elif issue.get("kind") == "ocr_engine_duration_missing":
                        print(f"   - {slug}: {metric} primary OCR engine metric missing")
            if isinstance(ocr_engine_counts, list):
                for issue in ocr_engine_counts[:6]:
                    if not isinstance(issue, dict):
                        continue
                    metric = issue.get("metric")
                    slug = issue.get("slug")
                    if issue.get("kind") == "ocr_engine_count_budget_exceeded":
                        print(
                            "   - "
                            f"{slug}: {metric} primary OCR engine count {float(issue['count']):.1f} "
                            f"> budget {float(issue['max_ocr_engine_count']):.1f}"
                        )
                    elif issue.get("kind") == "ocr_engine_count_missing":
                        print(f"   - {slug}: {metric} primary OCR engine count metric missing")
            if isinstance(repeat_ocr_engine, list):
                for issue in repeat_ocr_engine[:6]:
                    if not isinstance(issue, dict):
                        continue
                    metric = issue.get("metric")
                    if issue.get("kind") == "repeat_ocr_engine_p95_budget_exceeded":
                        print(
                            "   - "
                            f"{metric}: repeat OCR engine p95 {float(issue['p95_duration_s']):.3f}s "
                            f"> budget {float(issue['max_repeat_ocr_engine_p95_duration_s']):.3f}s"
                        )
                    elif issue.get("kind") == "repeat_ocr_engine_p95_missing":
                        print(f"   - {metric}: repeat OCR engine p95 metric missing")
            if isinstance(repeat_ocr_engine_counts, list):
                for issue in repeat_ocr_engine_counts[:6]:
                    if not isinstance(issue, dict):
                        continue
                    metric = issue.get("metric")
                    if issue.get("kind") == "repeat_ocr_engine_count_p95_budget_exceeded":
                        print(
                            "   - "
                            f"{metric}: repeat OCR engine count p95 {float(issue['p95_count']):.1f} "
                            f"> budget {float(issue['max_repeat_ocr_engine_p95_count']):.1f}"
                        )
                    elif issue.get("kind") == "repeat_ocr_engine_count_p95_missing":
                        print(f"   - {metric}: repeat OCR engine count p95 metric missing")
            if isinstance(repeat_ocr_engine_max_counts, list):
                for issue in repeat_ocr_engine_max_counts[:6]:
                    if not isinstance(issue, dict):
                        continue
                    metric = issue.get("metric")
                    if issue.get("kind") == "repeat_ocr_engine_count_max_budget_exceeded":
                        print(
                            "   - "
                            f"{metric}: repeat OCR engine count max {float(issue['max_count']):.1f} "
                            f"> budget {float(issue['max_repeat_ocr_engine_max_count']):.1f}"
                        )
                    elif issue.get("kind") == "repeat_ocr_engine_count_max_missing":
                        print(f"   - {metric}: repeat OCR engine count max metric missing")
    repeat_profile = report.get("repeat_profile")
    if isinstance(repeat_profile, dict):
        repeat_summary = repeat_profile.get("summary")
        if isinstance(repeat_summary, dict):
            median_total = repeat_summary.get("median_total_elapsed_s")
            p95_total = repeat_summary.get("p95_total_elapsed_s")
            max_total = repeat_summary.get("max_total_elapsed_s")
            median_text = f"{median_total:.3f}s" if isinstance(median_total, (int, float)) else "-"
            p95_text = f"{p95_total:.3f}s" if isinstance(p95_total, (int, float)) else "-"
            max_text = f"{max_total:.3f}s" if isinstance(max_total, (int, float)) else "-"
            print(
                "repeat profile: "
                f"analyzed={repeat_summary.get('analyzed_samples', 0)}, "
                f"expected={repeat_summary.get('expectation_passed_samples', 0)}, "
                f"subsecond={repeat_summary.get('subsecond_samples', 0)}, "
                f"median_total={median_text}, p95_total={p95_text}, max_total={max_text}"
            )
            slowest_samples = repeat_summary.get("slowest_samples")
            repeat_samples = repeat_profile.get("samples")
            if isinstance(slowest_samples, list) and slowest_samples:
                if (
                    not repeat_profile_slowest_samples_include_input_kind(slowest_samples)
                    and isinstance(repeat_samples, list)
                ):
                    enriched_slowest_samples = repeat_profile_slowest_samples(
                        [sample for sample in repeat_samples if isinstance(sample, dict)]
                    )
                    if repeat_profile_slowest_samples_include_input_kind(enriched_slowest_samples):
                        slowest_samples = enriched_slowest_samples
                slow_text = ", ".join(
                    repeat_profile_slow_sample_text(sample)
                    for sample in slowest_samples[:5]
                    if isinstance(sample, dict)
                )
                if slow_text:
                    print(f"repeat slowest: {slow_text}")
            retry_cases = repeat_summary.get("ocr_full_detail_retry_cases")
            if (
                isinstance(retry_cases, list)
                and retry_cases
                and repeat_summary.get("ocr_full_detail_retry_samples")
            ):
                retry_text = ", ".join(
                    repeat_profile_full_detail_retry_case_text(case)
                    for case in retry_cases[:5]
                    if isinstance(case, dict)
                )
                if retry_text:
                    retry_count = int(repeat_summary.get("ocr_full_detail_retry_samples") or 0)
                    print(f"repeat full-detail retries: {retry_count} sample(s): {retry_text}")
            slowest_cases = repeat_summary.get("slowest_cases")
            if isinstance(slowest_cases, list) and slowest_cases:
                if not repeat_profile_slowest_cases_include_top_stage(slowest_cases):
                    rebuilt_slowest_cases = rebuilt_repeat_profile_slowest_cases(repeat_profile)
                    if repeat_profile_slowest_cases_include_top_stage(rebuilt_slowest_cases):
                        slowest_cases = rebuilt_slowest_cases
                slow_case_text = ", ".join(
                    repeat_profile_slow_case_text(case)
                    for case in slowest_cases[:5]
                    if isinstance(case, dict)
                )
                if slow_case_text:
                    print(f"repeat slowest cases: {slow_case_text}")
            ocr_slowest_cases = repeat_summary.get("ocr_engine_slowest_cases")
            if isinstance(ocr_slowest_cases, list) and ocr_slowest_cases:
                rebuilt_ocr_slowest_cases = rebuilt_repeat_profile_ocr_engine_slowest_cases(repeat_profile)
                if should_use_rebuilt_repeat_ocr_slowest_cases(ocr_slowest_cases, rebuilt_ocr_slowest_cases):
                    ocr_slowest_cases = rebuilt_ocr_slowest_cases
                ocr_slow_case_text = ", ".join(
                    repeat_profile_ocr_engine_slow_case_text(case)
                    for case in ocr_slowest_cases[:5]
                    if isinstance(case, dict)
                )
                if ocr_slow_case_text:
                    print(f"repeat ocr slowest cases: {ocr_slow_case_text}")
            unstable_signature_cases = repeat_summary.get("unstable_signature_cases")
            if isinstance(unstable_signature_cases, list) and unstable_signature_cases:
                print(
                    "repeat signature drift: "
                    + ", ".join(str(slug) for slug in unstable_signature_cases)
                )
            unexpected_samples = repeat_profile_unexpected_sample_count(report)
            if unexpected_samples:
                unexpected_cases = repeat_profile_unexpected_cases(report)
                case_text = ", ".join(unexpected_cases)
                suffix = f": {case_text}" if case_text else ""
                print(f"repeat unexpected: {unexpected_samples} sample(s){suffix}")
            ocr_engine_stage_duration = repeat_summary.get("ocr_engine_stage_duration_s")
            if isinstance(ocr_engine_stage_duration, dict) and ocr_engine_stage_duration:
                labels = {
                    "input_s": "input",
                    "det_elapsed_s": "det",
                    "rec_elapsed_s": "rec",
                    "total_s": "total",
                }
                stage_text = ", ".join(
                    f"{labels.get(key, key)}=p95 {stats['p95_duration_s']:.3f}s max {stats['max_duration_s']:.3f}s"
                    for key, stats in ocr_engine_stage_duration.items()
                    if isinstance(stats, dict)
                    and isinstance(stats.get("p95_duration_s"), (int, float))
                    and isinstance(stats.get("max_duration_s"), (int, float))
                )
                if stage_text:
                    print(f"repeat ocr engine: {stage_text}")
            ocr_engine_counts = repeat_summary.get("ocr_engine_count_metric")
            if isinstance(ocr_engine_counts, dict) and ocr_engine_counts:
                count_text = repeat_profile_ocr_engine_count_metric_text(ocr_engine_counts)
                if count_text:
                    print(f"repeat ocr engine counts: {count_text}")
    for row in report["rows"]:
        mark = "ok" if row.get("expectation_passed") else "!!"
        elapsed = row.get("total_elapsed_s")
        elapsed_text = f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else "-"
        print(
            f"{mark} {row['slug']}: {row['observed_status']} "
            f"source={row.get('source') or '-'} controls={row.get('control_points') or '-'} "
            f"total={elapsed_text}"
        )
        for issue in row.get("expectation_issues", []):
            print(f"   - {issue}")


def repeat_profile_slow_sample_text(sample: dict[str, Any]) -> str:
    slug = sample.get("slug") or "-"
    repeat_index = sample.get("repeat_index")
    repeat_text = f"#{repeat_index}" if isinstance(repeat_index, int) else "#?"
    elapsed_s = parse_nonnegative_float(sample.get("total_elapsed_s"))
    elapsed_text = f"{elapsed_s:.3f}s" if elapsed_s is not None else "-"
    top_stage = sample.get("top_stage")
    stage_text = ""
    if isinstance(top_stage, dict):
        stage = top_stage.get("stage")
        stage_elapsed = parse_nonnegative_float(top_stage.get("elapsed_s"))
        if isinstance(stage, str) and stage and stage_elapsed is not None:
            stage_text = f" {stage}={stage_elapsed:.3f}s"
            if stage == "georeference":
                georeference_text = slow_case_georeference_event_text(sample)
                if georeference_text:
                    stage_text = f"{stage_text} {georeference_text}"
    ocr_engine = sample.get("ocr_engine")
    ocr_text = ""
    if isinstance(ocr_engine, dict):
        total_elapsed_s = parse_nonnegative_float(ocr_engine.get("total_s"))
        input_elapsed_s = parse_nonnegative_float(ocr_engine.get("input_s"))
        rec_elapsed_s = parse_nonnegative_float(ocr_engine.get("rec_elapsed_s"))
        hidden_s = parse_nonnegative_float(ocr_engine.get("overlap_hidden_s"))
        parts = []
        if total_elapsed_s is not None:
            parts.append(f"ocr_total={total_elapsed_s:.3f}s")
        if input_elapsed_s is not None:
            parts.append(f"input={input_elapsed_s:.3f}s")
        if rec_elapsed_s is not None:
            parts.append(f"rec={rec_elapsed_s:.3f}s")
        if hidden_s is not None:
            parts.append(f"hidden_ocr={hidden_s:.3f}s")
        input_kind = ocr_engine.get("input_kind")
        if isinstance(input_kind, str) and input_kind:
            parts.append(f"kind={input_kind}")
        selected_area_p50 = parse_nonnegative_float(ocr_engine.get("selected_box_area_p50"))
        selected_lt_1300 = ocr_engine.get("selected_box_area_lt_1300_count")
        if selected_area_p50 is not None:
            parts.append(f"sel_area_p50={selected_area_p50:.0f}")
        if isinstance(selected_lt_1300, int):
            parts.append(f"sel_lt1300={selected_lt_1300}")
        confidence_p50 = parse_nonnegative_float(ocr_engine.get("label_confidence_p50"))
        confidence_lt_80 = ocr_engine.get("label_confidence_lt_80_count")
        if confidence_p50 is not None:
            parts.append(f"conf_p50={confidence_p50:.1f}")
        if isinstance(confidence_lt_80, int):
            parts.append(f"conf_lt80={confidence_lt_80}")
        if parts:
            ocr_text = " " + " ".join(parts)
    return f"{slug}{repeat_text}={elapsed_text}{stage_text}{ocr_text}"


def ocr_engine_stage_max_row_text(stage_label: str, row: dict[str, Any]) -> str:
    slug = row.get("slug") or "-"
    elapsed_s = parse_nonnegative_float(row.get("elapsed_s"))
    elapsed_text = f"{elapsed_s:.3f}s" if elapsed_s is not None else "-"
    parts = [f"{stage_label}={elapsed_text}@{slug}"]
    shape_text = ocr_engine_input_shape_text(row.get("input_shape"))
    if shape_text:
        parts.append(f"shape={shape_text}")
    input_kind = row.get("input_kind")
    if isinstance(input_kind, str) and input_kind:
        parts.append(f"kind={input_kind}")
    detector_limit = row.get("detector_limit")
    if isinstance(detector_limit, int):
        detector_type = row.get("detector_limit_type")
        type_suffix = f"/{detector_type}" if isinstance(detector_type, str) and detector_type else ""
        parts.append(f"det_limit={detector_limit}{type_suffix}")
    recognition_profile = row.get("recognition_profile")
    if isinstance(recognition_profile, str) and recognition_profile:
        parts.append(f"rec={recognition_profile}")
    min_text_area = parse_nonnegative_float(row.get("min_text_area"))
    if min_text_area is not None:
        parts.append(f"min_area={min_text_area:.0f}")
    raw_count = row.get("raw_box_count")
    selected_count = row.get("selected_box_count")
    label_count = row.get("label_count")
    if isinstance(selected_count, int):
        parts.append(f"selected={selected_count}")
    if isinstance(raw_count, int):
        parts.append(f"raw={raw_count}")
    if isinstance(label_count, int):
        parts.append(f"labels={label_count}")
    selected_lt_1300 = row.get("selected_box_area_lt_1300_count")
    if isinstance(selected_lt_1300, int):
        parts.append(f"sel_lt1300={selected_lt_1300}")
    confidence_lt_90 = row.get("label_confidence_lt_90_count")
    if isinstance(confidence_lt_90, int):
        parts.append(f"conf_lt90={confidence_lt_90}")
    return " ".join(parts)


def ocr_engine_input_shape_text(shape: Any) -> str | None:
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    height, width = shape[0], shape[1]
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    if height <= 0 or width <= 0:
        return None
    return f"{width}x{height}"


def ocr_engine_dominant_stage_text(stage: Any, elapsed_s: Any, *, label: str = "dom") -> str | None:
    if not isinstance(stage, str) or not stage:
        return None
    elapsed = parse_nonnegative_float(elapsed_s)
    if elapsed is None:
        return None
    return f"{label}={stage}:{elapsed:.3f}s"


def primary_ocr_engine_slow_case_text(case: dict[str, Any]) -> str:
    slug = case.get("slug") or "-"
    total_s = parse_nonnegative_float(case.get("total_s"))
    input_s = parse_nonnegative_float(case.get("input_s"))
    rec_s = parse_nonnegative_float(case.get("rec_elapsed_s"))
    det_s = parse_nonnegative_float(case.get("det_elapsed_s"))
    hidden_s = parse_nonnegative_float(case.get("overlap_hidden_s"))
    parts = [str(slug)]
    if total_s is not None:
        parts.append(f"ocr={total_s:.3f}s")
    if input_s is not None:
        parts.append(f"input={input_s:.3f}s")
    if rec_s is not None:
        parts.append(f"rec={rec_s:.3f}s")
    if det_s is not None:
        parts.append(f"det={det_s:.3f}s")
    if hidden_s is not None:
        parts.append(f"hidden_ocr={hidden_s:.3f}s")
    dominant_text = ocr_engine_dominant_stage_text(case.get("dominant_stage"), case.get("dominant_stage_s"))
    if dominant_text is None:
        dominant = ocr_engine_dominant_stage(case, OCR_ENGINE_PRIMARY_DOMINANT_STAGE_FIELDS)
        if dominant is not None:
            stage, elapsed_s = dominant
            dominant_text = f"dom={stage}:{elapsed_s:.3f}s"
    if dominant_text:
        parts.append(dominant_text)
    shape_text = ocr_engine_input_shape_text(case.get("input_shape"))
    if shape_text:
        parts.append(f"shape={shape_text}")
    input_kind = case.get("input_kind")
    if isinstance(input_kind, str) and input_kind:
        parts.append(f"kind={input_kind}")
    detector_limit = case.get("detector_limit")
    if isinstance(detector_limit, int) and not isinstance(detector_limit, bool):
        detector_type = case.get("detector_limit_type")
        type_suffix = f"/{detector_type}" if isinstance(detector_type, str) and detector_type else ""
        parts.append(f"det_limit={detector_limit}{type_suffix}")
    recognition_profile = case.get("recognition_profile")
    if isinstance(recognition_profile, str) and recognition_profile:
        parts.append(f"rec_profile={recognition_profile}")
    min_text_area = parse_nonnegative_float(case.get("min_text_area"))
    if min_text_area is not None:
        parts.append(f"min_area={min_text_area:.0f}")
    selected_count = case.get("selected_box_count")
    raw_count = case.get("raw_box_count")
    label_count = case.get("label_count")
    if isinstance(selected_count, int) and not isinstance(selected_count, bool):
        parts.append(f"selected={selected_count}")
    if isinstance(raw_count, int) and not isinstance(raw_count, bool):
        parts.append(f"raw={raw_count}")
    if isinstance(label_count, int) and not isinstance(label_count, bool):
        parts.append(f"labels={label_count}")
    selected_lt_1300 = case.get("selected_box_area_lt_1300_count")
    if isinstance(selected_lt_1300, int) and not isinstance(selected_lt_1300, bool):
        parts.append(f"sel_lt1300={selected_lt_1300}")
    confidence_lt_90 = case.get("label_confidence_lt_90_count")
    if isinstance(confidence_lt_90, int) and not isinstance(confidence_lt_90, bool):
        parts.append(f"conf_lt90={confidence_lt_90}")
    return " ".join(parts)


def primary_slow_case_text_from_summary(case: dict[str, Any]) -> str:
    slug = case.get("slug") or "-"
    total_elapsed_s = parse_nonnegative_float(case.get("total_elapsed_s"))
    parts = [str(slug)]
    if total_elapsed_s is not None:
        parts[0] = f"{slug}={total_elapsed_s:.3f}s"
    top_stage = case.get("top_stage")
    if isinstance(top_stage, dict):
        stage = top_stage.get("stage")
        stage_elapsed_s = parse_nonnegative_float(top_stage.get("elapsed_s"))
        if isinstance(stage, str) and stage and stage_elapsed_s is not None:
            parts.append(f"{stage}={stage_elapsed_s:.3f}s")
            if stage == "georeference":
                georeference_text = slow_case_georeference_event_text(case)
                if georeference_text:
                    parts.append(georeference_text)
    ocr_engine = case.get("ocr_engine")
    if isinstance(ocr_engine, dict):
        total_s = parse_nonnegative_float(ocr_engine.get("total_s"))
        input_s = parse_nonnegative_float(ocr_engine.get("input_s"))
        rec_s = parse_nonnegative_float(ocr_engine.get("rec_elapsed_s"))
        det_s = parse_nonnegative_float(ocr_engine.get("det_elapsed_s"))
        hidden_s = parse_nonnegative_float(ocr_engine.get("overlap_hidden_s"))
        selected_count = ocr_engine.get("selected_box_count")
        selected_lt_1300 = ocr_engine.get("selected_box_area_lt_1300_count")
        confidence_p50 = parse_nonnegative_float(ocr_engine.get("label_confidence_p50"))
        if total_s is not None:
            parts.append(f"ocr_total={total_s:.3f}s")
        if input_s is not None:
            parts.append(f"input={input_s:.3f}s")
        if rec_s is not None:
            parts.append(f"rec={rec_s:.3f}s")
        if det_s is not None:
            parts.append(f"det={det_s:.3f}s")
        if hidden_s is not None:
            parts.append(f"hidden_ocr={hidden_s:.3f}s")
        dominant = ocr_engine_dominant_stage(
            ocr_engine,
            OCR_ENGINE_PRIMARY_DOMINANT_STAGE_FIELDS,
        )
        if dominant is not None:
            stage, elapsed_s = dominant
            parts.append(f"dom={stage}:{elapsed_s:.3f}s")
        input_kind = ocr_engine.get("input_kind")
        if isinstance(input_kind, str) and input_kind:
            parts.append(f"kind={input_kind}")
        if isinstance(selected_count, int) and not isinstance(selected_count, bool):
            parts.append(f"selected={selected_count}")
        if isinstance(selected_lt_1300, int) and not isinstance(selected_lt_1300, bool):
            parts.append(f"sel_lt1300={selected_lt_1300}")
        if confidence_p50 is not None:
            parts.append(f"conf_p50={confidence_p50:.1f}")
    expectation_passed = case.get("expectation_passed")
    if expectation_passed is False:
        parts.append("unexpected")
    return " ".join(parts)


def slow_case_georeference_event_text(case: dict[str, Any]) -> str:
    events = case.get("georeference_events")
    if not isinstance(events, list):
        return ""
    ranked: list[tuple[float, dict[str, Any]]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        elapsed_s = parse_nonnegative_float(event.get("elapsed_s"))
        if elapsed_s is None:
            continue
        ranked.append((elapsed_s, event))
    if not ranked:
        return ""
    elapsed_s, event = max(ranked, key=lambda item: item[0])
    message = event.get("message")
    if not isinstance(message, str) or not message:
        message = "georeference"
    label = re.sub(r"[^A-Za-z0-9]+", "_", message.strip().lower()).strip("_")
    if not label:
        label = "georeference"
    return f"geo_step={label}:{elapsed_s:.3f}s"


def repeat_profile_slow_case_text(case: dict[str, Any]) -> str:
    slug = case.get("slug") or "-"
    p95_total = parse_nonnegative_float(case.get("p95_total_elapsed_s"))
    max_total = parse_nonnegative_float(case.get("max_total_elapsed_s"))
    parts = [str(slug)]
    if p95_total is not None:
        parts.append(f"p95={p95_total:.3f}s")
    if max_total is not None:
        parts.append(f"max={max_total:.3f}s")
    top_stage = case.get("top_stage")
    if isinstance(top_stage, dict):
        stage = top_stage.get("stage")
        p95_duration_s = parse_nonnegative_float(top_stage.get("p95_duration_s"))
        if isinstance(stage, str) and stage and p95_duration_s is not None:
            parts.append(f"{stage}_p95={p95_duration_s:.3f}s")
    unexpected = case.get("unexpected_samples")
    if isinstance(unexpected, int) and unexpected:
        parts.append(f"unexpected={unexpected}")
    return " ".join(parts)


def repeat_profile_full_detail_retry_case_text(case: dict[str, Any]) -> str:
    slug = case.get("slug") or "-"
    retries = case.get("ocr_full_detail_retry_samples")
    analyzed = case.get("analyzed_samples")
    unexpected = case.get("unexpected_samples")
    parts = [str(slug)]
    if isinstance(retries, int) and not isinstance(retries, bool):
        if isinstance(analyzed, int) and not isinstance(analyzed, bool) and analyzed > 0:
            parts.append(f"{retries}/{analyzed}")
        else:
            parts.append(str(retries))
    if isinstance(unexpected, int) and unexpected:
        parts.append(f"unexpected={unexpected}")
    return " ".join(parts)


def repeat_profile_ocr_engine_slow_case_text(case: dict[str, Any]) -> str:
    slug = case.get("slug") or "-"
    p95_total = parse_nonnegative_float(case.get("p95_total_s"))
    max_total = parse_nonnegative_float(case.get("max_total_s"))
    p95_input = parse_nonnegative_float(case.get("p95_input_s"))
    p95_rec = parse_nonnegative_float(case.get("p95_rec_elapsed_s"))
    p95_det = parse_nonnegative_float(case.get("p95_det_elapsed_s"))
    p95_selected = parse_nonnegative_float(case.get("p95_selected_box_count"))
    p95_small_selected = parse_nonnegative_float(case.get("p95_selected_box_area_lt_1300_count"))
    parts = [str(slug)]
    if p95_total is not None:
        parts.append(f"ocr_p95={p95_total:.3f}s")
    if max_total is not None:
        parts.append(f"ocr_max={max_total:.3f}s")
    if p95_input is not None:
        parts.append(f"input_p95={p95_input:.3f}s")
    if p95_rec is not None:
        parts.append(f"rec_p95={p95_rec:.3f}s")
    if p95_det is not None:
        parts.append(f"det_p95={p95_det:.3f}s")
    dominant_text = ocr_engine_dominant_stage_text(
        case.get("p95_dominant_stage"),
        case.get("p95_dominant_stage_s"),
        label="dom_p95",
    )
    if dominant_text is None:
        dominant = ocr_engine_dominant_stage(case, OCR_ENGINE_REPEAT_P95_DOMINANT_STAGE_FIELDS)
        if dominant is not None:
            stage, elapsed_s = dominant
            dominant_text = f"dom_p95={stage}:{elapsed_s:.3f}s"
    if dominant_text:
        parts.append(dominant_text)
    shape_text = ocr_engine_input_shape_text(case.get("input_shape"))
    if shape_text:
        parts.append(f"shape={shape_text}")
    input_kind = case.get("input_kind")
    if isinstance(input_kind, str) and input_kind:
        parts.append(f"kind={input_kind}")
    detector_limit = case.get("detector_limit")
    if isinstance(detector_limit, int) and not isinstance(detector_limit, bool):
        detector_type = case.get("detector_limit_type")
        type_suffix = f"/{detector_type}" if isinstance(detector_type, str) and detector_type else ""
        parts.append(f"det_limit={detector_limit}{type_suffix}")
    recognition_profile = case.get("recognition_profile")
    if isinstance(recognition_profile, str) and recognition_profile:
        parts.append(f"rec_profile={recognition_profile}")
    rec_batch_num = case.get("rec_batch_num")
    if isinstance(rec_batch_num, int) and not isinstance(rec_batch_num, bool):
        parts.append(f"rec_batch={rec_batch_num}")
    min_text_area = parse_nonnegative_float(case.get("min_text_area"))
    if min_text_area is not None:
        parts.append(f"min_area={min_text_area:.0f}")
    if p95_selected is not None:
        parts.append(f"selected_p95={p95_selected:.1f}")
    if p95_small_selected is not None:
        parts.append(f"sel_lt1300_p95={p95_small_selected:.1f}")
    return " ".join(parts)


def repeat_profile_ocr_engine_count_metric_text(ocr_engine_counts: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in OCR_ENGINE_COUNT_DISPLAY_KEYS:
        stats = ocr_engine_counts.get(key)
        if not isinstance(stats, dict):
            continue
        p95_count = stats.get("p95_count")
        max_count = stats.get("max_count")
        if not isinstance(p95_count, (int, float)) or not isinstance(max_count, (int, float)):
            continue
        label = OCR_ENGINE_COUNT_DISPLAY_LABELS.get(key, key)
        parts.append(f"{label}=p95 {p95_count:.1f} max {max_count:.1f}")
    return ", ".join(parts)


def baseline_repeat_delta_text(repeat_delta: dict[str, Any] | None) -> str:
    if not repeat_delta:
        return ""
    metrics = (
        (
            "p95_total",
            ("duration_s", "p95_total_elapsed_s"),
            "delta_s",
            "{:+.3f}s",
        ),
        (
            "max_total",
            ("duration_s", "max_total_elapsed_s"),
            "delta_s",
            "{:+.3f}s",
        ),
        (
            "ocr_total_p95",
            ("ocr_engine_stage_duration_s", "total_s", "p95_duration_s"),
            "delta_s",
            "{:+.3f}s",
        ),
        (
            "hidden_ocr_p95",
            ("ocr_overlap_hidden_s", "p95_duration_s"),
            "delta_s",
            "{:+.3f}s",
        ),
        (
            "selected_box_p95",
            ("ocr_engine_count_metric", "selected_box_count", "p95_count"),
            "delta_count",
            "{:+.1f}",
        ),
        (
            "full_detail_retries",
            ("sample_counts", "ocr_full_detail_retry_samples"),
            "delta",
            "{:+.0f}",
        ),
    )
    parts: list[str] = []
    for label, path, value_key, formatter in metrics:
        value = repeat_delta_metric_value(repeat_delta, path, value_key)
        if value is not None:
            parts.append(f"{label}={formatter.format(value)}")
    return ", ".join(parts)


def baseline_repeat_case_coverage_text(baseline_comparison: dict[str, Any]) -> str:
    missing_baseline = baseline_comparison.get("repeat_profile_case_missing_in_baseline")
    missing_candidate = baseline_comparison.get("repeat_profile_case_missing_in_candidate")
    underanalyzed_baseline = baseline_comparison.get(
        "repeat_profile_case_underanalyzed_in_baseline"
    )
    underanalyzed_candidate = baseline_comparison.get(
        "repeat_profile_case_underanalyzed_in_candidate"
    )
    compared_rows = parse_nonnegative_int(
        baseline_comparison.get("repeat_profile_case_compared_rows")
    )
    expected_rows = parse_nonnegative_int(
        baseline_comparison.get("repeat_profile_case_expected_rows")
    )
    missing_baseline_count = (
        len(missing_baseline) if isinstance(missing_baseline, list) else 0
    )
    missing_candidate_count = (
        len(missing_candidate) if isinstance(missing_candidate, list) else 0
    )
    underanalyzed_baseline_count = (
        len(underanalyzed_baseline) if isinstance(underanalyzed_baseline, list) else 0
    )
    underanalyzed_candidate_count = (
        len(underanalyzed_candidate) if isinstance(underanalyzed_candidate, list) else 0
    )
    if (
        not missing_baseline_count
        and not missing_candidate_count
        and not underanalyzed_baseline_count
        and not underanalyzed_candidate_count
        and (compared_rows is None or expected_rows is None or compared_rows == expected_rows)
    ):
        return ""
    parts: list[str] = []
    if compared_rows is not None and expected_rows is not None:
        parts.append(f"repeat_case_compared={compared_rows}/{expected_rows}")
    if missing_baseline_count:
        parts.append(f"repeat_case_missing_baseline={missing_baseline_count}")
    if missing_candidate_count:
        parts.append(f"repeat_case_missing_candidate={missing_candidate_count}")
    if underanalyzed_baseline_count:
        parts.append(f"repeat_case_underanalyzed_baseline={underanalyzed_baseline_count}")
    if underanalyzed_candidate_count:
        parts.append(f"repeat_case_underanalyzed_candidate={underanalyzed_candidate_count}")
    return ", " + ", ".join(parts) if parts else ""


def baseline_repeat_case_delta_text(baseline_comparison: dict[str, Any]) -> str:
    worst = first_repeat_case_delta(
        baseline_comparison.get("largest_repeat_profile_case_p95_regressions")
    )
    best = first_repeat_case_delta(
        baseline_comparison.get("largest_repeat_profile_case_p95_improvements")
    )
    parts = []
    worst_text = repeat_case_delta_summary_text("worst_case", worst)
    if worst_text:
        parts.append(worst_text)
    best_text = repeat_case_delta_summary_text("best_case", best)
    if best_text:
        parts.append(best_text)
    return ", ".join(parts)


def first_repeat_case_delta(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list) or not value:
        return None
    first = value[0]
    return first if isinstance(first, dict) else None


def repeat_case_delta_summary_text(label: str, case_delta: dict[str, Any] | None) -> str:
    if not case_delta:
        return ""
    slug = case_delta.get("slug")
    delta = repeat_case_delta_metric_value(case_delta, "p95_total_elapsed_s", "delta_s")
    if not isinstance(slug, str) or not slug or delta is None:
        return ""
    parts = [f"{label}={slug} {delta:+.3f}s"]
    baseline = repeat_case_delta_metric_value(case_delta, "p95_total_elapsed_s", "baseline")
    candidate = repeat_case_delta_metric_value(case_delta, "p95_total_elapsed_s", "candidate")
    if baseline is not None and candidate is not None:
        parts.append(f"(baseline={baseline:.3f}s, candidate={candidate:.3f}s)")
    stage_text = repeat_case_delta_top_stage_text(case_delta)
    if stage_text:
        parts.append(stage_text)
    return " ".join(parts)


def repeat_case_delta_top_stage_text(case_delta: dict[str, Any]) -> str:
    stage_duration = case_delta.get("stage_duration_s")
    if not isinstance(stage_duration, dict):
        return ""
    ranked: list[tuple[float, tuple[int, str], float, str, float | None, float | None]] = []
    for stage, stats in stage_duration.items():
        if not isinstance(stage, str) or not stage or not isinstance(stats, dict):
            continue
        p95_delta = repeat_case_delta_stage_value(
            case_delta,
            stage,
            "p95_duration_s",
            "delta_s",
        )
        if p95_delta is None:
            continue
        baseline = repeat_case_delta_stage_value(
            case_delta,
            stage,
            "p95_duration_s",
            "baseline",
        )
        candidate = repeat_case_delta_stage_value(
            case_delta,
            stage,
            "p95_duration_s",
            "candidate",
        )
        ranked.append(
            (
                abs(p95_delta),
                pipeline_stage_sort_key(stage),
                p95_delta,
                stage,
                baseline,
                candidate,
            )
        )
    if not ranked:
        return ""
    _, _, p95_delta, stage, baseline, candidate = sorted(
        ranked,
        key=lambda item: (-item[0], item[1]),
    )[0]
    text = f"{stage}_p95={p95_delta:+.3f}s"
    if baseline is not None and candidate is not None:
        text += f" ({baseline:.3f}->{candidate:.3f}s)"
    return text


def baseline_repeat_ocr_case_delta_text(baseline_comparison: dict[str, Any]) -> str:
    worst = first_repeat_case_delta(
        baseline_comparison.get("largest_repeat_profile_case_ocr_p95_regressions")
    )
    best = first_repeat_case_delta(
        baseline_comparison.get("largest_repeat_profile_case_ocr_p95_improvements")
    )
    parts = []
    worst_text = repeat_case_ocr_delta_summary_text("worst_ocr_case", worst)
    if worst_text:
        parts.append(worst_text)
    best_text = repeat_case_ocr_delta_summary_text("best_ocr_case", best)
    if best_text:
        parts.append(best_text)
    return ", ".join(parts)


def repeat_case_ocr_delta_summary_text(label: str, case_delta: dict[str, Any] | None) -> str:
    if not case_delta:
        return ""
    slug = case_delta.get("slug")
    delta = repeat_case_delta_ocr_stage_value(
        case_delta,
        "total_s",
        "p95_duration_s",
        "delta_s",
    )
    if not isinstance(slug, str) or not slug or delta is None:
        return ""
    parts = [f"{label}={slug} {delta:+.3f}s"]
    baseline = repeat_case_delta_ocr_stage_value(
        case_delta,
        "total_s",
        "p95_duration_s",
        "baseline",
    )
    candidate = repeat_case_delta_ocr_stage_value(
        case_delta,
        "total_s",
        "p95_duration_s",
        "candidate",
    )
    if baseline is not None and candidate is not None:
        parts.append(f"(baseline={baseline:.3f}s, candidate={candidate:.3f}s)")
    return " ".join(parts)


def baseline_configuration_changes_text(
    baseline_comparison: dict[str, Any],
    *,
    limit: int = 8,
) -> str:
    changes = baseline_comparison.get("configuration_changes")
    if not isinstance(changes, list) or not changes:
        return ""
    parts: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        field = change.get("field")
        if not isinstance(field, str) or not field:
            continue
        baseline_value = config_change_value_text(change.get("baseline"))
        candidate_value = config_change_value_text(change.get("candidate"))
        parts.append(f"{field}={baseline_value}->{candidate_value}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def config_change_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    if isinstance(value, str):
        return value or "empty"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def baseline_expectation_delta_text(baseline_comparison: dict[str, Any]) -> str:
    compared = parse_nonnegative_int(baseline_comparison.get("expectation_compared_rows"))
    baseline_passed = parse_nonnegative_int(
        baseline_comparison.get("baseline_expectation_passed_rows")
    )
    candidate_passed = parse_nonnegative_int(
        baseline_comparison.get("candidate_expectation_passed_rows")
    )
    delta = parse_signed_int(baseline_comparison.get("expectation_passed_delta"))
    change_count = parse_nonnegative_int(baseline_comparison.get("expectation_change_count"))
    if (
        compared is None
        or compared == 0
        or baseline_passed is None
        or candidate_passed is None
        or delta is None
    ):
        return ""
    if delta == 0 and (change_count is None or change_count == 0):
        return ""
    parts = [
        f"baseline={baseline_passed}/{compared}",
        f"candidate={candidate_passed}/{compared}",
        f"delta={delta:+d}",
    ]
    if change_count is not None:
        parts.append(f"changes={change_count}")
    return ", ".join(parts)


def baseline_expectation_change_text(change: dict[str, Any]) -> str:
    slug = change.get("slug")
    if not isinstance(slug, str) or not slug:
        return ""
    baseline_passed = change.get("baseline_expectation_passed")
    candidate_passed = change.get("candidate_expectation_passed")
    if not isinstance(baseline_passed, bool) or not isinstance(candidate_passed, bool):
        return ""
    parts = [
        slug,
        f"baseline={expectation_passed_text(baseline_passed)}",
        f"candidate={expectation_passed_text(candidate_passed)}",
    ]
    candidate_issues = change.get("candidate_expectation_issues")
    if isinstance(candidate_issues, list):
        issues = [issue for issue in candidate_issues if isinstance(issue, str) and issue]
        if issues:
            parts.append(f"candidate_issues={len(issues)}")
            parts.append(f"first={issues[0]}")
    return " ".join(parts)


def expectation_passed_text(value: bool) -> str:
    return "pass" if value else "fail"


def baseline_repeat_ocr_stage_delta_text(repeat_delta: dict[str, Any] | None) -> str:
    if not repeat_delta:
        return ""
    parts: list[str] = []
    for key, label in BASELINE_REPEAT_OCR_STAGE_DELTA_DISPLAY:
        value = repeat_delta_metric_value(
            repeat_delta,
            ("ocr_engine_stage_duration_s", key, "p95_duration_s"),
            "delta_s",
        )
        if value is not None:
            parts.append(f"{label}={value:+.3f}s")
    return ", ".join(parts)


def baseline_repeat_stage_delta_text(repeat_delta: dict[str, Any] | None) -> str:
    if not repeat_delta:
        return ""
    stage_duration = repeat_delta.get("stage_duration_s")
    if not isinstance(stage_duration, dict):
        return ""
    parts: list[str] = []
    for stage in sorted(
        (stage for stage in stage_duration if isinstance(stage, str) and stage),
        key=pipeline_stage_sort_key,
    ):
        value = repeat_delta_metric_value(
            repeat_delta,
            ("stage_duration_s", stage, "p95_duration_s"),
            "delta_s",
        )
        if value is not None:
            parts.append(f"{stage}_p95={value:+.3f}s")
    return ", ".join(parts)


def baseline_primary_delta_text(baseline_comparison: dict[str, Any]) -> str:
    parts: list[str] = []
    worst = baseline_primary_delta_item_text(
        baseline_comparison.get("largest_total_regressions")
    )
    if worst:
        parts.append(f"worst_total={worst}")
    best = baseline_primary_delta_item_text(
        baseline_comparison.get("largest_total_improvements")
    )
    if best:
        parts.append(f"best_total={best}")
    return ", ".join(parts)


def baseline_primary_ocr_delta_text(baseline_comparison: dict[str, Any]) -> str:
    parts: list[str] = []
    worst = baseline_primary_ocr_delta_item_text(
        baseline_comparison.get("largest_ocr_engine_total_regressions")
    )
    if worst:
        parts.append(f"worst_ocr={worst}")
    best = baseline_primary_ocr_delta_item_text(
        baseline_comparison.get("largest_ocr_engine_total_improvements")
    )
    if best:
        parts.append(f"best_ocr={best}")
    return ", ".join(parts)


def baseline_primary_ocr_overlap_hidden_delta_text(baseline_comparison: dict[str, Any]) -> str:
    parts: list[str] = []
    worst = baseline_primary_ocr_overlap_hidden_delta_item_text(
        baseline_comparison.get("largest_ocr_overlap_hidden_regressions")
    )
    if worst:
        parts.append(f"worst_hidden={worst}")
    best = baseline_primary_ocr_overlap_hidden_delta_item_text(
        baseline_comparison.get("largest_ocr_overlap_hidden_improvements")
    )
    if best:
        parts.append(f"best_hidden={best}")
    return ", ".join(parts)


def baseline_primary_delta_item_text(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    item = items[0]
    if not isinstance(item, dict):
        return ""
    slug = item.get("slug")
    delta = parse_signed_float(item.get("total_elapsed_delta_s"))
    if not isinstance(slug, str) or not slug or delta is None:
        return ""
    ocr_delta_text = baseline_primary_ocr_stage_delta_text(item)
    return f"{slug} {delta:+.3f}s{ocr_delta_text}"


def baseline_primary_ocr_delta_item_text(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    item = items[0]
    if not isinstance(item, dict):
        return ""
    slug = item.get("slug")
    delta = parse_signed_float(item.get("ocr_engine_total_delta_s"))
    stage_deltas = item.get("ocr_engine_stage_delta_s")
    if delta is None and isinstance(stage_deltas, dict):
        delta = parse_signed_float(stage_deltas.get("total_s"))
    if not isinstance(slug, str) or not slug or delta is None:
        return ""
    stage_delta_text = baseline_primary_ocr_stage_delta_text(item, include_total=False)
    return f"{slug} {delta:+.3f}s{stage_delta_text}"


def baseline_primary_ocr_overlap_hidden_delta_item_text(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    item = items[0]
    if not isinstance(item, dict):
        return ""
    slug = item.get("slug")
    delta = parse_signed_float(item.get("ocr_overlap_hidden_delta_s"))
    if not isinstance(slug, str) or not slug or delta is None:
        return ""
    baseline_hidden = parse_nonnegative_float(item.get("baseline_ocr_overlap_hidden_s"))
    candidate_hidden = parse_nonnegative_float(item.get("candidate_ocr_overlap_hidden_s"))
    hidden_text = (
        f" (baseline={baseline_hidden:.3f}s, candidate={candidate_hidden:.3f}s)"
        if baseline_hidden is not None and candidate_hidden is not None
        else ""
    )
    return f"{slug} {delta:+.3f}s{hidden_text}"


def baseline_primary_ocr_stage_delta_text(
    item: dict[str, Any],
    *,
    include_total: bool = True,
) -> str:
    parts: list[str] = []
    total_delta = parse_signed_float(item.get("ocr_engine_total_delta_s"))
    stage_deltas = item.get("ocr_engine_stage_delta_s")
    if total_delta is None and isinstance(stage_deltas, dict):
        total_delta = parse_signed_float(stage_deltas.get("total_s"))
    if include_total and total_delta is not None:
        parts.append(f"ocr_total={total_delta:+.3f}s")
    if isinstance(stage_deltas, dict):
        for key, label in BASELINE_PRIMARY_OCR_STAGE_DELTA_DISPLAY:
            value = parse_signed_float(stage_deltas.get(key))
            if value is not None:
                parts.append(f"{label}={value:+.3f}s")
    if not parts:
        return ""
    return f" ({', '.join(parts)})"


def baseline_signature_field_counts_text(baseline_comparison: dict[str, Any], *, limit: int = 4) -> str:
    counts = baseline_comparison.get("signature_changed_field_counts")
    if not isinstance(counts, dict) or not counts:
        changes = baseline_comparison.get("signature_changes")
        if isinstance(changes, list):
            counts = signature_changed_field_counts(
                [change for change in changes if isinstance(change, dict)]
            )
    if not isinstance(counts, dict) or not counts:
        return ""
    parts: list[str] = []
    for field, count in counts.items():
        if not isinstance(field, str) or not isinstance(count, int) or count <= 0:
            continue
        parts.append(f"{field}:{count}")
        if len(parts) >= limit:
            break
    if not parts:
        return ""
    return ", signature_fields=" + ",".join(parts)


def baseline_regression_budget_text(regression_budget: dict[str, Any]) -> str:
    limits: list[str] = []
    max_total = parse_nonnegative_float(regression_budget.get("max_total_elapsed_regression_s"))
    if max_total is not None:
        limits.append(f"primary<={max_total:.3f}s")
    max_ocr_total = parse_nonnegative_float(regression_budget.get("max_ocr_engine_total_regression_s"))
    if max_ocr_total is not None:
        limits.append(f"primary_ocr<={max_ocr_total:.3f}s")
    max_repeat = parse_nonnegative_float(regression_budget.get("max_repeat_p95_regression_s"))
    if max_repeat is not None:
        limits.append(f"repeat_p95<={max_repeat:.3f}s")
    max_repeat_ocr_total = parse_nonnegative_float(
        regression_budget.get("max_repeat_ocr_engine_total_p95_regression_s")
    )
    if max_repeat_ocr_total is not None:
        limits.append(f"repeat_ocr_p95<={max_repeat_ocr_total:.3f}s")
    limit_text = f" {' '.join(limits)}" if limits else ""
    skip_text = baseline_regression_budget_skip_text(regression_budget)
    if regression_budget.get("passed") is False:
        violations = regression_budget.get("violations")
        violation_count = len(violations) if isinstance(violations, list) else 0
        kind_text = (
            baseline_regression_budget_violation_count_text(violations)
            if isinstance(violations, list)
            else ""
        )
        return (
            f"baseline regression budget: failed{limit_text}{skip_text} "
            f"violations={violation_count}{kind_text}"
        )
    return f"baseline regression budget: passed{limit_text}{skip_text}"


def baseline_regression_budget_skip_text(regression_budget: dict[str, Any]) -> str:
    skipped_count = parse_nonnegative_count_metric(
        regression_budget.get("skipped_primary_ocr_zero_call_row_count")
    )
    if skipped_count is None:
        skipped_rows = regression_budget.get("skipped_primary_ocr_zero_call_rows")
        if not isinstance(skipped_rows, list):
            return ""
        skipped_count = float(
            sum(1 for slug in skipped_rows if isinstance(slug, str) and slug)
        )
    if skipped_count <= 0.0:
        return ""
    return f" skipped_primary_ocr_zero_call={skipped_count:g}"


def baseline_regression_budget_violation_count_text(violations: list[Any]) -> str:
    counts: dict[str, int] = {}
    for violation in violations:
        if not isinstance(violation, dict):
            continue
        kind = violation.get("kind")
        if not isinstance(kind, str) or not kind:
            continue
        label = baseline_regression_budget_violation_kind_label(kind)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return " by_kind=" + ",".join(f"{label}:{count}" for label, count in counts.items())


def baseline_regression_budget_violation_kind_label(kind: str) -> str:
    labels = {
        "primary_total_regression_exceeded": "primary",
        "primary_ocr_total_regression_exceeded": "primary_ocr",
        "primary_ocr_total_delta_missing": "primary_ocr_missing",
        "repeat_profile_p95_regression_exceeded": "repeat_p95",
        "repeat_profile_p95_delta_missing": "repeat_p95_missing",
        "repeat_profile_case_p95_regression_exceeded": "repeat_case_p95",
        "repeat_ocr_total_p95_regression_exceeded": "repeat_ocr_p95",
        "repeat_ocr_total_p95_delta_missing": "repeat_ocr_p95_missing",
        "repeat_ocr_case_total_p95_regression_exceeded": "repeat_ocr_case_p95",
    }
    return labels.get(kind, kind)


def baseline_regression_budget_violation_samples(violations: list[Any], *, limit: int = 5) -> list[Any]:
    if limit <= 0:
        return []
    selected_indices: list[int] = []
    selected_index_set: set[int] = set()
    seen_kinds: set[str] = set()
    for index, violation in enumerate(violations):
        if not isinstance(violation, dict):
            continue
        kind = violation.get("kind")
        if not isinstance(kind, str) or not kind or kind in seen_kinds:
            continue
        selected_indices.append(index)
        selected_index_set.add(index)
        seen_kinds.add(kind)
        if len(selected_indices) >= limit:
            return [violations[selected_index] for selected_index in selected_indices]
    for index, _violation in enumerate(violations):
        if index in selected_index_set:
            continue
        selected_indices.append(index)
        if len(selected_indices) >= limit:
            break
    return [violations[selected_index] for selected_index in selected_indices]


def baseline_regression_budget_violation_text(violation: Any) -> str:
    if not isinstance(violation, dict):
        return ""
    kind = violation.get("kind")
    if kind == "primary_total_regression_exceeded":
        slug = violation.get("slug")
        delta = parse_signed_float(violation.get("total_elapsed_delta_s"))
        budget = parse_nonnegative_float(violation.get("max_total_elapsed_regression_s"))
        if not isinstance(slug, str) or delta is None or budget is None:
            return ""
        return f"primary {slug} {delta:+.3f}s > budget {budget:.3f}s"
    if kind == "primary_ocr_total_regression_exceeded":
        slug = violation.get("slug")
        delta = parse_signed_float(violation.get("ocr_engine_total_delta_s"))
        budget = parse_nonnegative_float(violation.get("max_ocr_engine_total_regression_s"))
        if not isinstance(slug, str) or delta is None or budget is None:
            return ""
        return f"primary ocr {slug} {delta:+.3f}s > budget {budget:.3f}s"
    if kind == "primary_ocr_total_delta_missing":
        slug = violation.get("slug")
        budget = parse_nonnegative_float(violation.get("max_ocr_engine_total_regression_s"))
        slug_text = f" {slug}" if isinstance(slug, str) and slug else ""
        budget_text = f" budget {budget:.3f}s" if budget is not None else ""
        return f"primary ocr{slug_text} delta missing{budget_text}"
    if kind == "repeat_profile_p95_regression_exceeded":
        delta = parse_signed_float(violation.get("delta_s"))
        budget = parse_nonnegative_float(violation.get("max_repeat_p95_regression_s"))
        if delta is None or budget is None:
            return ""
        return f"repeat p95 {delta:+.3f}s > budget {budget:.3f}s"
    if kind == "repeat_profile_p95_delta_missing":
        budget = parse_nonnegative_float(violation.get("max_repeat_p95_regression_s"))
        budget_text = f" budget {budget:.3f}s" if budget is not None else ""
        return f"repeat p95 delta missing{budget_text}"
    if kind == "repeat_profile_case_p95_regression_exceeded":
        slug = violation.get("slug")
        delta = parse_signed_float(violation.get("delta_s"))
        budget = parse_nonnegative_float(violation.get("max_repeat_p95_regression_s"))
        if not isinstance(slug, str) or delta is None or budget is None:
            return ""
        return f"repeat case p95 {slug} {delta:+.3f}s > budget {budget:.3f}s"
    if kind == "repeat_ocr_total_p95_regression_exceeded":
        delta = parse_signed_float(violation.get("delta_s"))
        budget = parse_nonnegative_float(violation.get("max_repeat_ocr_engine_total_p95_regression_s"))
        if delta is None or budget is None:
            return ""
        return f"repeat ocr p95 {delta:+.3f}s > budget {budget:.3f}s"
    if kind == "repeat_ocr_total_p95_delta_missing":
        budget = parse_nonnegative_float(violation.get("max_repeat_ocr_engine_total_p95_regression_s"))
        budget_text = f" budget {budget:.3f}s" if budget is not None else ""
        return f"repeat ocr p95 delta missing{budget_text}"
    if kind == "repeat_ocr_case_total_p95_regression_exceeded":
        slug = violation.get("slug")
        delta = parse_signed_float(violation.get("delta_s"))
        budget = parse_nonnegative_float(
            violation.get("max_repeat_ocr_engine_total_p95_regression_s")
        )
        if not isinstance(slug, str) or delta is None or budget is None:
            return ""
        return f"repeat ocr case p95 {slug} {delta:+.3f}s > budget {budget:.3f}s"
    return ""


def repeat_delta_metric_value(
    repeat_delta: dict[str, Any],
    path: tuple[str, ...],
    value_key: str,
) -> float | None:
    current: Any = repeat_delta
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if not isinstance(current, dict):
        return None
    return parse_signed_float(current.get(value_key))


def parse_signed_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def parse_signed_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_summary(stdout: str | bytes | None) -> tuple[dict[str, Any], str | None]:
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if not stdout:
        return {}, "empty stdout"
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON summary: {exc}"
    if not isinstance(value, dict):
        return {}, "summary JSON was not an object"
    return value, None


def truncate_text(value: str | bytes | None, limit: int = 800) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Stress case is missing string field {key!r}.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
