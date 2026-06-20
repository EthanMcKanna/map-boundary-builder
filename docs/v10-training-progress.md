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
