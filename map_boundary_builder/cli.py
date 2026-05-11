from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import DEFAULT_SIMPLIFY_PX
from .runner import BoundaryBuildOptions, build_boundary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map-boundary-builder",
        description="Extract a scaled GeoJSON service-area polygon from a map screenshot.",
    )
    parser.add_argument("--image", help="Input service-map screenshot.")
    parser.add_argument("--city", help="Optional city override. Omit to infer from map labels.")
    parser.add_argument("--output", "-o", help="Output GeoJSON path.")
    parser.add_argument("--debug-dir", help="Optional directory for mask and overlay PNGs.")
    parser.add_argument("--simplify-px", type=float, default=DEFAULT_SIMPLIFY_PX, help="Pixel simplification tolerance.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Fail below this combined confidence.")
    parser.add_argument("--min-control-points", type=int, default=3, help="Minimum OCR/geocoder control points for georeferencing.")
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

    try:
        result = build_boundary(
            image_path,
            args.city,
            args.output,
            debug_dir=args.debug_dir,
            options=BoundaryBuildOptions(
                simplify_px=args.simplify_px,
                min_confidence=args.min_confidence,
                min_control_points=args.min_control_points,
            ),
        )

        if args.print_summary:
            print(json.dumps(result.summary, indent=2))
        return 0
    except Exception as exc:
        print(f"map-boundary-builder: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
