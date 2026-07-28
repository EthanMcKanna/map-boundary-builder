"""Segmentation probes for model-candidate sweeps.

Runs the segmentation stage (no OCR, no network) on a handful of real
screenshots whose coverage bands past regressions violated. Use it to
sweep exported epoch candidates before spending an hour on the full
landing battery:

    python tools/probe_suite.py --model scratchpad/v4_e032.onnx
    python tools/probe_suite.py --model a.onnx --model b.onnx  # compare

Exit code 0 when every available case passes for at least one model.
Cases whose image is missing on this machine are skipped with a note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from map_boundary_builder.extract import load_rgb  # noqa: E402
from map_boundary_builder.segment import segment_image  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run segmentation probes against model candidates.")
    parser.add_argument(
        "--model",
        action="append",
        type=Path,
        default=None,
        help="ONNX candidate to probe (repeatable). Defaults to the installed package model.",
    )
    parser.add_argument("--cases", type=Path, default=REPO_ROOT / "benchmarks" / "probe-cases.json")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    return parser


def resolve_image(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def run_probes(model: Path | None, cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        image_path = resolve_image(case["image"])
        record = {"name": case["name"], "model": str(model) if model else "installed"}
        if not image_path.is_file():
            record["status"] = "skipped"
            record["detail"] = f"missing image: {image_path}"
            results.append(record)
            continue
        try:
            rgb = load_rgb(image_path)
            extraction = segment_image(rgb, model_path=model)
        except Exception as error:  # noqa: BLE001 - a probe crash is a result, not an abort
            record["status"] = "error"
            record["detail"] = f"{type(error).__name__}: {error}"
            results.append(record)
            continue
        coverage = float(extraction.coverage_ratio)
        components = int(extraction.contour_count)
        ok = (
            case["coverage_min"] <= coverage <= case["coverage_max"]
            and components <= case["max_components"]
        )
        record.update(
            {
                "status": "pass" if ok else "fail",
                "coverage": round(coverage, 4),
                "components": components,
                "coverage_band": [case["coverage_min"], case["coverage_max"]],
                "max_components": case["max_components"],
                "engine": (extraction.diagnostics or {}).get("segmentation_engine"),
            }
        )
        results.append(record)
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = json.loads(args.cases.read_text())["cases"]
    models: list[Path | None] = list(args.model) if args.model else [None]
    all_results = []
    exit_code = 0
    for model in models:
        results = run_probes(model, cases)
        all_results.extend(results)
        failed = [r for r in results if r["status"] in ("fail", "error")]
        if failed:
            exit_code = 1
        if not args.json:
            label = str(model) if model else "installed model"
            print(f"== {label} ==")
            for r in results:
                if r["status"] in ("pass", "fail"):
                    band = r["coverage_band"]
                    print(
                        f"  {r['status'].upper():4} {r['name']:20} coverage={r['coverage']:.3f} "
                        f"[{band[0]:.2f},{band[1]:.2f}] components={r['components']}/{r['max_components']} "
                        f"engine={r['engine']}"
                    )
                else:
                    print(f"  {r['status'].upper():4} {r['name']:20} {r['detail']}")
            passed = sum(1 for r in results if r["status"] == "pass")
            checked = sum(1 for r in results if r["status"] in ("pass", "fail"))
            print(f"  -> {passed}/{checked} probes passed")
    if args.json:
        print(json.dumps(all_results, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
