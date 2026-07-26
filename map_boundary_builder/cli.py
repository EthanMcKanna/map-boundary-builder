from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import DEFAULT_SIMPLIFY_PX
from .pipeline import PipelineOptions, PipelineResult, complete_with_city, run_pipeline
from .pipeline_version import get_pipeline_version

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NEEDS_CITY = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map-boundary-builder",
        description="Extract a georeferenced GeoJSON service-area polygon from a map screenshot.",
    )
    parser.add_argument("--image", help="Input service-map screenshot.")
    parser.add_argument("--city", help="Optional city override. Omit to infer from map labels.")
    parser.add_argument("--output", "-o", help="Output GeoJSON path.")
    parser.add_argument("--debug-dir", help="Optional directory for mask and overlay PNGs.")
    parser.add_argument("--simplify-px", type=float, default=DEFAULT_SIMPLIFY_PX, help="Pixel simplification tolerance.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Fail below this combined confidence.")
    parser.add_argument("--min-control-points", type=int, default=3, help="Minimum OCR/geocoder control points for georeferencing.")
    parser.add_argument("--seed-x", type=float, help="Optional target-region seed x coordinate in source pixels.")
    parser.add_argument("--seed-y", type=float, help="Optional target-region seed y coordinate in source pixels.")
    parser.add_argument("--target-color", help="Optional target overlay color as #RRGGBB.")
    parser.add_argument("--model-path", help="Override the packaged segmentation model.")
    parser.add_argument("--model-threshold", type=float, help="Override the calibrated model threshold.")
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Never prompt interactively; exit with status 3 when a city is needed.",
    )
    parser.add_argument("--print-summary", action="store_true", help="Print a compact JSON summary.")
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

    options = build_options(parser, args)
    result = run_pipeline(
        image_path,
        city=args.city,
        output_path=args.output,
        debug_dir=args.debug_dir,
        options=options,
    )

    if result.status == "needs_city":
        result = maybe_prompt_for_city(args, image_path, result, options)

    if args.print_summary:
        summary = dict(result.summary)
        summary["pipeline_version"] = get_pipeline_version()
        print(json.dumps(summary, indent=2))

    if result.status == "complete":
        return EXIT_OK
    if result.status == "needs_city":
        print(
            "map-boundary-builder: the city could not be inferred from map labels. "
            'Re-run with --city "City, ST".',
            file=sys.stderr,
        )
        if result.needs_city is not None and result.needs_city.sample_labels:
            labels = ", ".join(result.needs_city.sample_labels[:5])
            print(f"map-boundary-builder: labels read from the map: {labels}", file=sys.stderr)
        return EXIT_NEEDS_CITY
    message = result.summary.get("message") or result.reason or "extraction failed"
    print(f"map-boundary-builder: error: {message}", file=sys.stderr)
    return EXIT_FAILED


def build_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> PipelineOptions:
    if (args.seed_x is None) != (args.seed_y is None):
        parser.error("--seed-x and --seed-y must be provided together")
    seed_point = (args.seed_x, args.seed_y) if args.seed_x is not None else None
    target_rgb = None
    target_color = (args.target_color or "").strip().lstrip("#")
    if target_color:
        if len(target_color) != 6:
            parser.error("--target-color must be a six-digit hexadecimal color")
        try:
            target_rgb = tuple(int(target_color[index : index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            parser.error("--target-color must be a six-digit hexadecimal color")
    return PipelineOptions(
        simplify_px=args.simplify_px,
        min_confidence=args.min_confidence,
        min_control_points=args.min_control_points,
        seed_point=seed_point,
        target_rgb=target_rgb,
        model_path=Path(args.model_path) if args.model_path else None,
        model_threshold=args.model_threshold,
    )


def maybe_prompt_for_city(
    args: argparse.Namespace,
    image_path: Path,
    result: PipelineResult,
    options: PipelineOptions,
) -> PipelineResult:
    if args.no_input or not sys.stdin.isatty():
        return result
    detail = result.needs_city
    if detail is not None and detail.sample_labels:
        labels = ", ".join(detail.sample_labels[:5])
        print(f"Boundary extracted, but the city could not be determined. Labels read: {labels}")
    else:
        print("Boundary extracted, but the city could not be determined.")
    try:
        city = input('Enter the city (e.g. "Austin, TX"), or press Enter to abort: ').strip()
    except EOFError:
        return result
    if not city:
        return result
    return complete_with_city(
        image_path,
        result,
        city,
        output_path=args.output,
        debug_dir=args.debug_dir,
        options=options,
    )


if __name__ == "__main__":
    raise SystemExit(main())
