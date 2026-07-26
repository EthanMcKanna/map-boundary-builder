"""Generate the locked, observable exact-edge promotion suite for v20."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from map_boundary_builder.synthetic import (
    GENERATOR_VERSION,
    SyntheticDatasetManifest,
    SyntheticSceneConfig,
    generate_synthetic_sample,
)
from map_boundary_builder.synthetic.generator import randomized_overlay_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v20's balanced exact-edge promotion suite.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=202000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 96 or args.count % 4:
        raise SystemExit("--count must be a multiple of four and at least 96")
    families = ("rectilinear", "angular", "road-following", "radial")
    dimensions = ((640, 640), (960, 640), (768, 960), (1024, 768))
    samples = []
    for index in range(args.count):
        width, height = dimensions[(index // 4) % len(dimensions)]
        family = families[index % len(families)]
        style = randomized_overlay_style(args.seed + index, index=index)
        # The edge suite is intentionally observable. Selection ambiguity,
        # distractors, and abstention have their own gate; mixing them here
        # would make a missing target look like geometric edge error.
        style = replace(
            style,
            fill_opacity=0.82,
            fill_enabled=True,
            stroke_width_px=(0.0, 1.0, 2.0, 3.0)[(index // 4) % 4],
            dashed=False,
            labels_on_top=False,
            circular_viewport=False,
            pattern=None,
            stroke_join="miter",
        )
        config = SyntheticSceneConfig(
            provider="synthetic-v20-gold",
            service_area=f"edge-city-{index % 8}",
            variant=f"{family}-{index:04d}",
            width=width,
            height=height,
            seed=args.seed + index,
            overlay_style=style,
            touch_border=index % 19 == 7,
            include_ui_chrome=index % 11 == 5,
            include_hole=index % 13 == 4,
            jpeg_quality=90 if index % 9 == 2 else None,
            labels_on_top=False,
            circular_viewport=False,
            complex_boundary=index % 3 == 0,
            large_service_area=index % 5 == 1,
            include_distractor=False,
            shape_family=family,
        )
        samples.append(generate_synthetic_sample(args.out, config).sample)
    manifest = SyntheticDatasetManifest(
        name="generalized-v20-exact-edge-promotion",
        version=GENERATOR_VERSION,
        samples=samples,
        properties={
            "generator": GENERATOR_VERSION,
            "suite": "observable-exact-edge",
            "seed": args.seed,
            "count": args.count,
            "shape_families": list(families),
            "automatic_selection": False,
            "selection_scope": "observable geometry only",
        },
    )
    manifest.write_json(args.out / "manifest.json")
    print(args.out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
