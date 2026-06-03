from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
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
OCR_ENGINE_STAGE_MAX_KEYS = ("det_elapsed_s", "rec_elapsed_s", "total_s")
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
        "--real-screenshot-hard-gate",
        action="store_true",
        help=(
            "Apply the current full real-screenshot production-warm hard gate preset: "
            "in-process execution, OCR profiling, prewarm, repeat signatures, latency/OCR "
            "budgets, and manifest OCR contract coverage budgets."
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
    if args.max_total_elapsed_s is not None and args.max_total_elapsed_s <= 0.0:
        parser.error("--max-total-elapsed-s must be positive")
    if args.max_repeat_profile_p95_duration_s is not None and args.max_repeat_profile_p95_duration_s <= 0.0:
        parser.error("--max-repeat-profile-p95-duration-s must be positive")
    if args.max_prewarm_runtime_s is not None and args.max_prewarm_runtime_s <= 0.0:
        parser.error("--max-prewarm-runtime-s must be positive")
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
    if args.fail_on_unexpected and (
        report["summary"]["unexpected"] or repeat_profile_unexpected_sample_count(report)
    ):
        return 1
    return 0


def apply_real_screenshot_hard_gate_preset(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.real_screenshot_hard_gate:
        return
    if args.only:
        parser.error("--real-screenshot-hard-gate targets the full manifest and cannot be combined with --only")
    args.execution = "in-process"
    args.profile_ocr_engine = True
    args.prewarm_runtime = True
    args.disable_ocr_cache = True
    args.disable_extraction_cache = True
    args.fail_on_unexpected = True
    args.fail_on_repeat_signature_drift = True
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
    if args.min_ocr_call_contract_rows is None:
        args.min_ocr_call_contract_rows = 49
    if args.min_ocr_count_contract_rows is None:
        args.min_ocr_count_contract_rows = 12
    if args.max_positive_ocr_call_only_rows is None:
        args.max_positive_ocr_call_only_rows = 26
    args.fail_on_invalid_ocr_count_contracts = True


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
    (out_dir / "stress-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


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
            "error": summary.get("error"),
            "stdout_json_error": parse_error,
            "stderr": truncate_text(completed.stderr),
            "command": command,
        }
    )
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
        "max_total_elapsed_s": round(max(elapsed_values), 6) if elapsed_values else None,
        "stage_duration_s": {stage: round(elapsed_s, 6) for stage, elapsed_s in sorted(stage_totals.items())},
        "stage_max_rows": dict(sorted(stage_max_rows.items())),
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
        "input_shape",
        "detector_limit",
        "detector_limit_type",
        "recognition_profile",
        "min_text_area",
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
            "subsecond_case_min_total_count": subsecond_case_count,
            "stable_signature_cases": len(case_summaries) - len(unstable_signature_cases),
            "unstable_signature_cases": unstable_signature_cases,
            **repeat_profile_total_elapsed_stats(analyzed_samples),
            "stage_duration_s": repeat_profile_stage_duration_stats(analyzed_samples),
            "slowest_samples": repeat_profile_slowest_samples(analyzed_samples),
            "ocr_engine_profile": summarize_repeat_profile_ocr_engine(analyzed_samples),
            "ocr_engine_stage_duration_s": repeat_profile_ocr_engine_stage_duration_stats(analyzed_samples),
            "ocr_engine_count_metric": repeat_profile_ocr_engine_count_stats(analyzed_samples),
            "ocr_engine_stage_max_rows": ocr_engine_stage_max_rows(analyzed_samples),
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
        "signature_stability": repeat_profile_signature_stability(analyzed_samples),
        **repeat_profile_total_elapsed_stats(analyzed_samples),
        "stage_duration_s": repeat_profile_stage_duration_stats(analyzed_samples),
        "ocr_engine_profile": summarize_repeat_profile_ocr_engine(analyzed_samples),
        "ocr_engine_stage_duration_s": repeat_profile_ocr_engine_stage_duration_stats(analyzed_samples),
        "ocr_engine_count_metric": repeat_profile_ocr_engine_count_stats(analyzed_samples),
        "ocr_engine_stage_max_rows": ocr_engine_stage_max_rows(analyzed_samples),
    }


def repeat_profile_analyzed_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in samples if not sample.get("warmup")]


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
        "error": sample.get("error"),
    }


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
    calls = ocr_engine_profile.get("calls")
    if isinstance(calls, int):
        summary["calls"] = calls
    for key in ("raw_box_count", "selected_box_count", "result_count", "label_count", "useful_label_count"):
        value = ocr_engine_profile.get(key)
        if isinstance(value, int):
            summary[key] = value
    detail_profile = slowest_ocr_engine_detail_profile(ocr_engine_profile)
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
        labels = {
            "det_elapsed_s": "det",
            "rec_elapsed_s": "rec",
            "total_s": "total",
        }
        max_text = ", ".join(
            f"{labels.get(key, key)}={row['elapsed_s']:.3f}s@{row['slug']}"
            for key, row in max_rows.items()
            if isinstance(row, dict) and isinstance(row.get("elapsed_s"), (int, float))
        )
        if max_text:
            print(f"ocr engine max: {max_text}")
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
            if isinstance(slowest_samples, list) and slowest_samples:
                slow_text = ", ".join(
                    repeat_profile_slow_sample_text(sample)
                    for sample in slowest_samples[:5]
                    if isinstance(sample, dict)
                )
                if slow_text:
                    print(f"repeat slowest: {slow_text}")
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
    ocr_engine = sample.get("ocr_engine")
    ocr_text = ""
    if isinstance(ocr_engine, dict):
        rec_elapsed_s = parse_nonnegative_float(ocr_engine.get("rec_elapsed_s"))
        total_elapsed_s = parse_nonnegative_float(ocr_engine.get("total_s"))
        parts = []
        if rec_elapsed_s is not None:
            parts.append(f"rec={rec_elapsed_s:.3f}s")
        if total_elapsed_s is not None:
            parts.append(f"ocr_total={total_elapsed_s:.3f}s")
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
