"""The one promotion gate for the unified pipeline.

Compares a fresh synthetic-benchmark report and a fresh real-screenshot
benchmark report against the committed pre-revamp baselines. Fails closed.

Usage:
    python tools/check_promotion.py \
        --synthetic-report out/synthetic-boundary-v1.json \
        --real-report out/real-benchmark-boundary-v1.json \
        --model map_boundary_builder/models/boundary_v1.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLDS = {
    # The synthetic bars are the committed v12 baseline numbers.
    "synthetic_mean_iou": 0.934,
    "synthetic_p05_iou": 0.746,
    "synthetic_max_failures": 0,
    # Real manifest: nothing that completed before may hard-fail now, and at
    # most this many old completes may convert to needs_city (the accepted
    # trade for deleting the style-tuned OCR crop heuristics).
    "real_max_complete_to_failed": 0,
    "real_max_complete_to_needs_city": 5,
    "real_min_correct_rejections": 8,
    "model_max_bytes": 15 * 1024 * 1024,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the boundary_v1 promotion gates.")
    parser.add_argument("--synthetic-report", type=Path, required=True)
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("map_boundary_builder/models/boundary_v1.onnx"))
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    synthetic = json.loads(args.synthetic_report.read_text(encoding="utf-8"))["summary"]
    real = json.loads(args.real_report.read_text(encoding="utf-8"))
    transitions = real["summary"].get("status_transitions", {})
    correct_rejections = sum(
        1 for row in real["rows"] if row.get("baseline_status") == "failed" and row["status"] == "failed"
    )
    model_bytes = args.model.stat().st_size if args.model.is_file() else 0

    gates = {
        "synthetic_mean_iou": synthetic["mean_iou"] >= THRESHOLDS["synthetic_mean_iou"],
        "synthetic_p05_iou": synthetic["p05_iou"] >= THRESHOLDS["synthetic_p05_iou"],
        "synthetic_failures": synthetic["failure_count"] <= THRESHOLDS["synthetic_max_failures"],
        "real_no_complete_to_failed": transitions.get("complete->failed", 0)
        <= THRESHOLDS["real_max_complete_to_failed"],
        "real_bounded_needs_city": transitions.get("complete->needs_city", 0)
        <= THRESHOLDS["real_max_complete_to_needs_city"],
        "real_correct_rejections": correct_rejections >= THRESHOLDS["real_min_correct_rejections"],
        "model_present": model_bytes > 0,
        "model_size_budget": 0 < model_bytes <= THRESHOLDS["model_max_bytes"],
    }
    report = {
        "passed": all(gates.values()),
        "gates": gates,
        "thresholds": THRESHOLDS,
        "synthetic": {key: synthetic.get(key) for key in ("mean_iou", "p05_iou", "failure_count", "sample_count")},
        "real_transitions": transitions,
        "real_correct_rejections": correct_rejections,
        "model_bytes": model_bytes,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
