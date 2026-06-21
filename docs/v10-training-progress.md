# v10 Training Progress

This note preserves the interrupted v10 training context from June 20, 2026.

Branch: `synthetic-v10-real-style`

Command:

```bash
.venv/bin/python tools/train_synthetic_model.py \
  --dataset-dir out/synthetic-model-v10-train \
  --output map_boundary_builder/models/synthetic_boundary_v10.onnx \
  --count 4096 \
  --validation-count 384 \
  --epochs 18 \
  --batch-size 8 \
  --image-size 256 \
  --render-width 640 \
  --render-height 640 \
  --base-channels 40 \
  --device mps
```

Observed before battery-preservation handoff:

- Model: `resunet`
- Parameters: `6105881`
- Train samples: `4096`
- Validation samples: `384`
- Image size: `256`
- Epoch 1: loss `0.16052`, validation IoU `0.81020`, lr `0.0007939`
- Epoch 2: loss `0.05101`, validation IoU `0.94198`, lr `0.0007759`
- Epoch 3: loss `0.03190`, validation IoU `0.94964`, lr `0.0007464`
- Epoch 4: loss `0.01870`, validation IoU `0.95808`, lr `0.0007064`
- Epoch 5: loss `0.01539`, validation IoU `0.98609`, lr `0.0006571`
- Epoch 6: loss `0.01207`, validation IoU `0.98968`, lr `0.0006000`
- Epoch 7: loss `0.00902`, validation IoU `0.99164`, lr `0.0005368`
- Epoch 8: loss `0.00949`, validation IoU `0.98530`, lr `0.0004695`
- Epoch 9: loss `0.00667`, validation IoU `0.99175`, lr `0.0004000`
- Epoch 10: loss `0.00547`, validation IoU `0.99356`, lr `0.0003305`

The original in-flight process did not have checkpoint support. The trainer now writes per-epoch checkpoints to `--checkpoint-dir` or `<output-stem>.checkpoints` for future runs.

## Final selected artifact

The packaged v10 artifact is exported from `synthetic_boundary_v10.checkpoints/epoch-011.pt`:

```bash
.venv/bin/python tools/train_synthetic_model.py \
  --export-checkpoint map_boundary_builder/models/synthetic_boundary_v10.checkpoints/epoch-011.pt \
  --output map_boundary_builder/models/synthetic_boundary_v10.onnx \
  --base-channels 40
```

Epoch 11 was selected from the independent holdout, not from the highest synthetic validation checkpoint:

- Epoch 11 validation IoU: `0.99395`
- Epoch 12 validation IoU: `0.99458`, rejected because independent holdout min IoU fell to `0.592481`
- Epoch 13 validation IoU: `0.99470`, rejected because independent holdout min IoU fell to `0.582813`

Independent hard holdout: `out/synthetic-v10-holdout-large-4901`, `160` samples, seed `4901`, `960x960` renders.

| Extractor | Threshold | Failures | Mean IoU | Min IoU | Mean boundary IoU 2px |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic | n/a | `22` | `0.684097` | `0.000000` | `0.404320` |
| v2 model | `0.25` | `2` | `0.637911` | `0.005068` | `0.338336` |
| v10 epoch 11 | `0.45` | `0` | `0.988416` | `0.938007` | `0.770845` |

Paired against deterministic on the same holdout, v10 won `151/160` rows. The `9` remaining losses were all clean `waymo-solid-blue` or `waymo-cyan-blue` cases where deterministic behaved like an exact color-threshold oracle and beat v10 by at most `0.007910` IoU. The model still scored every deterministic failure.

Higher-resolution calibration was tested and rejected:

- `512px` fine-tuning was too slow for practical iteration on this Mac.
- `384px` fine-tuning improved boundary IoU but hurt whole-mask reliability, with min IoU around `0.85`.

Real-image sanity check on `/Users/ethanmckanna/Downloads/houston waymo.png` with packaged v10:

- deterministic: style `bright-blue`, coverage `0.273711`, contours `1`, confidence `1.0`
- v10 experimental: style `auto-fill`, coverage `0.273897`, contours `1`, confidence `0.9`, threshold `0.45`
