# Generalized v12 BoundaryField

## Architecture

Generalized v12 separates semantic target selection from geometric boundary
recovery:

1. A five-channel global selector consumes RGB, seed guidance, and learned
   target-color similarity at 320x320. Its output is a probability field,
   not the final polygon.
2. Automatic mode obtains a deterministic candidate and reruns the selector
   with seed and color guidance. This prevents confident-looking selection of
   the wrong overlay while keeping explicit user guidance single-pass.
3. Source-resolution cleanup keeps the guided component, removes tiny speckle,
   and preserves material holes without corner-rounding morphology.
4. Polygonization retains holes, uses dense contours, protects predicted
   corners, and limits simplification to 0.5-1.0 source pixels. The v11 global
   six-pixel simplification is not used.

A sparse 293k-parameter native refiner remains in the training tooling. It was
not enabled in production: controlled ablations improved some boundary tails
but regressed mean overlap and latency. Promotion selects the simplest stage
that clears the complete gate, rather than shipping an unhelpful neural layer.

The runtime artifacts are:

- `map_boundary_builder/models/boundaryfield_v12_selector.onnx` plus its
  external-data shard.

## Training data

The v12 generator balances four shape families: rectilinear, road-following,
angular, and radial. It includes holes, edge-touching regions, distractors,
translucency, outline-only targets, patterns, JPEG capture effects, and miter,
bevel, and round stroke joins. Training and holdout datasets use disjoint seeds.

The refiner is trained on native-resolution boundary crops with deliberately
degraded coarse masks. Its multi-task objective combines boundary-weighted
region loss, Dice, signed-distance regression, edge alignment, signed-distance
classification, and corner detection.

## Promotion gates

Promotion is checked by `tools/check_v12_promotion.py`. It separates exact-edge
fixtures from deliberately unobservable adversarial scenes. The latter remain
in the robustness corpus but cannot define a two-pixel accuracy target when no
boundary signal exists in the screenshot.

Promotion requires all of the following:

- no extraction failures;
- full adversarial mean IoU at least `0.94` and mean boundary IoU at least `0.60`;
- on observable rectilinear/angular fixtures: mean IoU at least `0.985`, p05
  IoU at least `0.965`, mean boundary IoU at least `0.70`, double-tail p95
  boundary distance no more than `7px`, and zero topology mismatches;
- all nine active real references pass, with mean IoU at least `0.96`, minimum
  IoU at least `0.90`, and mean IoU above v11;
- real polygons retain at least four times v11's mean vertex density;
- p95 warm selector inference no more than `0.15s` and real automatic-mode
  maximum no more than `0.65s` on the validation machine;
- the deployed selector artifact is below `20MB`.

The synthetic promotion command is:

```bash
.venv/bin/python -m map_boundary_builder.synthetic_benchmark \
  --dataset-dir out/synthetic-v12-boundaryfield-holdout \
  --model-path map_boundary_builder/models/boundaryfield_v12_selector.onnx \
  --boundaryfield-selector-only \
  --model-input-size 320 \
  --model-input-channels 5 \
  --model-threshold 0.45 \
  --guided \
  --min-iou 0 \
  --mean-iou 0 \
  --out out/boundaryfield-v12-promotion-full-report.json
```

The generated report, manifest, real report, and v11 baseline are then passed
to `tools/check_v12_promotion.py`; the checker owns the thresholds above.

Synthetic results are necessary but not sufficient. Production promotion also
requires local API/UI verification, the existing real-image outcomes, a preview
deployment, production alias verification, and a live upload through
`/api/runs` using `extractor=generalized_v12_boundaryfield`.
