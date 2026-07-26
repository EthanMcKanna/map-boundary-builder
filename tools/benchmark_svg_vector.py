"""Benchmark the exact v20 SVG lane against analytic vector fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from shapely.geometry import Polygon

from map_boundary_builder.evaluation import geometry_corner_f1
from map_boundary_builder.svg_vector import extract_svg_service_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score v20's vector-native SVG path extraction.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observed-svg", type=Path, default=None)
    return parser


def analytic_fixtures():
    yield (
        "viewbox-rectangle",
        b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="100 200 80 40">
        <style>.service { fill: #07f; }</style><path class="service" d="M110 205H170V235H110Z"/></svg>''',
        160,
        80,
        Polygon([(20, 10), (140, 10), (140, 70), (20, 70)]),
    )
    yield (
        "transformed-concave",
        b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <g transform="translate(5 7) scale(2)"><path fill="#0077ff"
        d="M5 5H35V15H25V35H5Z"/></g></svg>''',
        100,
        100,
        Polygon([(15, 17), (75, 17), (75, 37), (55, 37), (55, 77), (15, 77)]),
    )
    yield (
        "evenodd-hole",
        b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path fill="#07f" fill-rule="evenodd"
        d="M10 10H90V90H10Z M35 35H65V65H35Z"/></svg>''',
        100,
        100,
        Polygon(
            [(10, 10), (90, 10), (90, 90), (10, 90)],
            [[(35, 35), (65, 35), (65, 65), (35, 65)]],
        ),
    )


def topology_signature(geometry) -> dict[str, int]:
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    return {
        "components": len(polygons),
        "holes": sum(len(polygon.interiors) for polygon in polygons),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = []
    for name, document, width, height, expected in analytic_fixtures():
        started = time.perf_counter()
        try:
            vector = extract_svg_service_path(document, width=width, height=height)
            corners = geometry_corner_f1(vector.geometry, expected, tolerance_px=0.01)
            deviation = float(vector.geometry.hausdorff_distance(expected))
            rows.append(
                {
                    "name": name,
                    "status": "scored",
                    "duration_s": round(time.perf_counter() - started, 6),
                    "corner_f1": round(float(corners["f1"]), 6),
                    "path_deviation_px": round(deviation, 9),
                    "topology_matches": topology_signature(vector.geometry) == topology_signature(expected),
                    "line_segment_count": vector.line_segment_count,
                    "curve_segment_count": vector.curve_segment_count,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": name,
                    "status": "failed",
                    "duration_s": round(time.perf_counter() - started, 6),
                    "error": str(exc),
                }
            )

    observed = None
    if args.observed_svg is not None:
        from map_boundary_builder.image_io import read_svg_bytes
        from map_boundary_builder.runner import SVG_RASTER_MAX_DIMENSION
        from map_boundary_builder.image_io import rasterize_svg_bytes_to_png
        from PIL import Image
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="edgegraph-svg-observed-") as directory:
            raster = Path(directory) / "observed.png"
            svg_bytes = read_svg_bytes(args.observed_svg)
            source_sha256 = hashlib.sha256(svg_bytes).hexdigest()
            rasterize_svg_bytes_to_png(
                svg_bytes,
                raster,
                max_dimension=SVG_RASTER_MAX_DIMENSION,
                source_path=args.observed_svg,
            )
            with Image.open(raster) as image:
                width, height = image.size
            started = time.perf_counter()
            vector = extract_svg_service_path(svg_bytes, width=width, height=height)
            observed = {
                "path": str(args.observed_svg),
                "source_sha256": source_sha256,
                "width": width,
                "height": height,
                "duration_s": round(time.perf_counter() - started, 6),
                "geometry_type": vector.geometry.geom_type,
                "valid": bool(vector.geometry.is_valid),
                "is_empty": bool(vector.geometry.is_empty),
                "area": round(float(vector.geometry.area), 6),
                "vertices": len(vector.geometry.exterior.coords) if vector.geometry.geom_type == "Polygon" else None,
                "line_segment_count": vector.line_segment_count,
                "curve_segment_count": vector.curve_segment_count,
                "flatten_tolerance_px": vector.flatten_tolerance_px,
            }

    scored = [row for row in rows if row["status"] == "scored"]
    summary = {
        "fixture_count": len(rows),
        "scored_count": len(scored),
        "failure_count": len(rows) - len(scored),
        "topology_error_count": sum(not row["topology_matches"] for row in scored),
        "min_corner_f1": min((row["corner_f1"] for row in scored), default=0.0),
        "max_path_deviation_px": max((row["path_deviation_px"] for row in scored), default=float("inf")),
        "max_duration_s": max((row["duration_s"] for row in scored), default=float("inf")),
    }
    report = {"summary": summary, "rows": rows, "observed_svg": observed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
