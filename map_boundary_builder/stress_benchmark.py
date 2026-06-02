from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("benchmarks/real-screenshot-stress.json")
DEFAULT_OUT_DIR = Path("out/real-screenshot-stress")
GENERIC_FILENAME_HINT = "upload.png"


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_stress_benchmark(
        Path(args.manifest),
        Path(args.out_dir),
        only_slugs=args.only,
        timeout_seconds=args.timeout_seconds,
        write_debug=args.write_debug,
    )
    print_stress_table(report)
    if args.fail_on_unexpected and report["summary"]["unexpected"]:
        return 1
    return 0


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
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    cases = select_cases(manifest["cases"], only_slugs or [])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        run_stress_case(
            case,
            out_dir,
            timeout_seconds=timeout_seconds,
            write_debug=write_debug,
            python_executable=python_executable,
        )
        for case in cases
    ]
    report = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    (out_dir / "stress-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def select_cases(cases: list[dict[str, Any]], only_slugs: list[str]) -> list[dict[str, Any]]:
    if not only_slugs:
        return cases
    wanted = set(only_slugs)
    selected = [case for case in cases if case.get("slug") in wanted]
    missing = sorted(wanted - {case.get("slug") for case in selected})
    if missing:
        raise ValueError(f"Unknown stress slug(s): {', '.join(missing)}")
    return selected


def run_stress_case(
    case: dict[str, Any],
    out_dir: Path,
    *,
    timeout_seconds: float,
    write_debug: bool,
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

    output_path = out_dir / f"{slug}.geojson"
    command = build_cli_command(case, image, output_path, out_dir, write_debug, python_executable)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
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
    row["expectation_issues"] = check_expectations(row, expect)
    row["expectation_passed"] = not row["expectation_issues"]
    return row


def build_cli_command(
    case: dict[str, Any],
    image: Path,
    output_path: Path,
    out_dir: Path,
    write_debug: bool,
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
    if write_debug:
        command.extend(["--debug-dir", str(out_dir / f"{case['slug']}-debug")])
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
    row = base_row(case, expected_status=expected_status, observed_status=observed_status)
    row.update(
        {
            "returncode": completed.returncode,
            "wall_s": wall_s,
            "city": summary.get("city"),
            "style": summary.get("style"),
            "source": summary.get("georeference_source"),
            "catalog_slug": summary.get("catalog_slug"),
            "combined_confidence": summary.get("combined_confidence"),
            "georeference_confidence": summary.get("georeference_confidence"),
            "control_points": summary.get("control_points"),
            "bbox": summary.get("bbox"),
            "total_elapsed_s": event_profile.get("total_elapsed_s"),
            "stages": event_profile.get("stage_elapsed_s") if isinstance(event_profile.get("stage_elapsed_s"), dict) else {},
            "error": summary.get("error"),
            "stdout_json_error": parse_error,
            "stderr": truncate_text(completed.stderr),
            "command": command,
        }
    )
    return row


def base_row(case: dict[str, Any], *, expected_status: str, observed_status: str) -> dict[str, Any]:
    return {
        "slug": case.get("slug"),
        "image": case.get("image"),
        "expected_status": expected_status,
        "observed_status": observed_status,
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
        return issues

    source_prefix = expect.get("source_prefix")
    source = row.get("source") or ""
    if isinstance(source_prefix, str) and not str(source).startswith(source_prefix):
        issues.append(f"source {source!r} did not start with {source_prefix!r}")

    min_control_points = expect.get("min_control_points")
    if isinstance(min_control_points, int):
        control_points = row.get("control_points")
        if not isinstance(control_points, int) or control_points < min_control_points:
            issues.append(f"control_points {control_points!r} below {min_control_points}")

    max_total_elapsed_s = expect.get("max_total_elapsed_s")
    if isinstance(max_total_elapsed_s, (int, float)):
        total_elapsed_s = row.get("total_elapsed_s")
        if not isinstance(total_elapsed_s, (int, float)) or total_elapsed_s > float(max_total_elapsed_s):
            issues.append(f"total_elapsed_s {total_elapsed_s!r} above {max_total_elapsed_s}")
    return issues


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unexpected = [row["slug"] for row in rows if not row.get("expectation_passed")]
    statuses: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in rows:
        status = str(row.get("observed_status"))
        statuses[status] = statuses.get(status, 0) + 1
        source = row.get("source")
        if isinstance(source, str) and source:
            sources[source] = sources.get(source, 0) + 1
    elapsed_values = [row.get("total_elapsed_s") for row in rows if isinstance(row.get("total_elapsed_s"), (int, float))]
    return {
        "total": len(rows),
        "expectation_passed": len(rows) - len(unexpected),
        "unexpected": unexpected,
        "statuses": dict(sorted(statuses.items())),
        "sources": dict(sorted(sources.items())),
        "max_total_elapsed_s": round(max(elapsed_values), 6) if elapsed_values else None,
    }


def print_stress_table(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "stress summary: "
        f"{summary['expectation_passed']}/{summary['total']} expected, "
        f"statuses={summary['statuses']}, "
        f"max_total_elapsed_s={summary['max_total_elapsed_s']}"
    )
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
