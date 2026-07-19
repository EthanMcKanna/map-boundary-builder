# Generalized v11

## Core problem

Extract the intended service-area boundary from maps whose palette, basemap,
overlay treatment, crop, and file encoding were not known in advance.

## Release scope

- A distinct `generalized_v11` model in the web extractor selector.
- Five-channel inference: RGB, optional seed-point heatmap, and optional target-color similarity.
- Continuous synthetic palette, opacity, outline, pattern, and capture-effect randomization.
- Automatic extraction remains available when guidance is omitted.
- Existing deterministic and experimental-v10 paths remain unchanged and available as fallbacks.
- Model uncertainty, guidance usage, variant, and artifact are emitted in diagnostics.
- Raw pixel masks are evaluated before georeferencing or geometric fitting.

## Explicitly deferred

- A browser-rendered MapLibre tile corpus. The synthetic manifest remains renderer-agnostic so this can be added without changing training or evaluation consumers.
- Native PDF, GeoTIFF, KML, and GeoJSON import. SVG already has a vector-first inspection path; new formats require their own security and fidelity fixtures.
- Automatic semantic selection among several equally plausible thematic regions. Seed and color guidance make that ambiguity explicit.

## Promotion gates

Generalized v11 must not replace Default until it passes:

1. Guided synthetic holdout: zero extraction failures, mean mask IoU at least 0.95, fifth-percentile IoU at least 0.90, and minimum IoU at least 0.25. The minimum is deliberately separated because distractor scenes include semantically ambiguous regions.
2. Real screenshot gate: no regression on the focused real fixtures.
3. Negative gate: no increase in accepted non-map or no-boundary inputs.
4. Guided tests: seed and target-color channels measurably improve ambiguous cases.
5. Runtime gate: inference and memory stay within the hosted API budget.

## Training

```bash
.venv/bin/python tools/train_synthetic_model.py \
  --dataset-dir out/synthetic-v11-train \
  --output map_boundary_builder/models/synthetic_boundary_v11.onnx \
  --checkpoint-dir map_boundary_builder/models/synthetic_boundary_v11.checkpoints \
  --input-channels 5 \
  --arch resunet
```

The packaged ONNX graph must be exported directly to its final filename so an
external-data shard, when produced, keeps the filename recorded by the graph.

## Packaged model evidence

The packaged model was trained on 2,048 training and 256 validation samples at
256x256 resolution with five input channels. The 2,304-sample corpus includes
768 multi-overlay distractor scenes, 908 patterned overlays, 345 outline-only
targets, 576 JPEG captures, and 544 circular viewports.

- Final training validation IoU: `0.96020`.
- Independent guided 256-row score at threshold `0.45`: zero failures, mean IoU
  `0.960477`, fifth-percentile IoU `0.916221`, minimum IoU `0.302798`.
- Independent automatic 256-row score: zero failures, mean IoU `0.942206`,
  fifth-percentile IoU `0.828161`.
- Focused real-screenshot gate: `3/3` expected outcomes, including rejection of
  the climate-vulnerability thematic-map negative.
- Direct generalized-v11 end-to-end checks completed for Ann Arbor and Las
  Vegas; both produced OCR georeferences, while the thematic-map negative
  failed closed.
