from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check generalized v12 promotion gates.")
    parser.add_argument("--synthetic-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--baseline-real-report", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    synthetic = json.loads(args.synthetic_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    real = json.loads(args.real_report.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_real_report.read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    sharp_rows = [
        row
        for row in synthetic["rows"]
        if row["status"] == "scored" and observable_sharp(samples[row["sample_id"]])
    ]
    sharp = profile_summary(sharp_rows)
    real_summary = real["summary"]
    baseline_summary = baseline["summary"]
    real_vertices = [score["vertices"] for score in real["scores"] if score.get("status") == "active"]
    baseline_vertices = [score["vertices"] for score in baseline["scores"] if score.get("status") == "active"]
    external_data = Path(str(args.selector) + ".data")
    artifact_bytes = args.selector.stat().st_size + (external_data.stat().st_size if external_data.exists() else 0)

    gates = {
        "synthetic_no_failures": synthetic["summary"]["failure_count"] == 0,
        "synthetic_robust_mean_iou": synthetic["summary"]["mean_iou"] >= 0.94,
        "synthetic_robust_mean_boundary_iou_2px": synthetic["summary"]["mean_boundary_iou_2px"] >= 0.60,
        "synthetic_warm_p95_s": synthetic["summary"]["p95_duration_s"] <= 0.15,
        "sharp_sample_count": sharp["sample_count"] >= 16,
        "sharp_mean_iou": sharp["mean_iou"] >= 0.985,
        "sharp_p05_iou": sharp["p05_iou"] >= 0.965,
        "sharp_mean_boundary_iou_2px": sharp["mean_boundary_iou_2px"] >= 0.70,
        "sharp_p95_boundary_distance_px": sharp["p95_boundary_distance_px"] <= 7.0,
        "sharp_topology": sharp["topology_error_count"] == 0,
        "real_all_active_pass": real_summary["failed_fixtures"] == 0 and real_summary["scored_fixtures"] >= 9,
        "real_mean_iou": real_summary["average_iou"] >= 0.96,
        "real_min_iou": real_summary["min_iou"] >= 0.90,
        "real_no_v11_regression": real_summary["average_iou"] > baseline_summary["average_iou"],
        "real_dense_geometry": mean(real_vertices) >= 4.0 * mean(baseline_vertices),
        "real_max_duration_s": real_summary["max_duration_s"] <= 0.65,
        "artifact_budget": artifact_bytes < 20 * 1024 * 1024,
    }
    report = {
        "passed": all(gates.values()),
        "gates": gates,
        "synthetic": synthetic["summary"],
        "observable_sharp": sharp,
        "real": real_summary,
        "baseline_real": baseline_summary,
        "mean_real_vertices": round(mean(real_vertices), 3),
        "mean_baseline_vertices": round(mean(baseline_vertices), 3),
        "artifact_bytes": artifact_bytes,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def observable_sharp(sample: dict) -> bool:
    properties = sample["properties"]
    return (
        properties["shape_family"] in {"rectilinear", "angular"}
        and float(sample["overlay_style"]["fill_opacity"]) >= 0.30
        and not properties["labels_on_top"]
        and not properties["include_distractor"]
    )


def profile_summary(rows: list[dict]) -> dict[str, float | int]:
    metrics = [row["metrics"] for row in rows]
    return {
        "sample_count": len(rows),
        "mean_iou": round(mean(item["iou"] for item in metrics), 6),
        "p05_iou": round(float(np.quantile([item["iou"] for item in metrics], 0.05)), 6),
        "mean_boundary_iou_2px": round(mean(item["boundary_iou_2px"] for item in metrics), 6),
        "p95_boundary_distance_px": round(
            float(np.quantile([item["boundary_distance_p95_px"] for item in metrics], 0.95)), 6
        ),
        "topology_error_count": sum(not item["topology_matches"] for item in metrics),
    }


if __name__ == "__main__":
    raise SystemExit(main())
