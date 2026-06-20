from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image

from map_boundary_builder.synthetic import generate_synthetic_dataset, SyntheticDatasetManifest

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny synthetic boundary segmentation model.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/synthetic-model-train"))
    parser.add_argument("--output", type=Path, default=Path("map_boundary_builder/models/synthetic_boundary_v1.onnx"))
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = generate_synthetic_dataset(
        args.dataset_dir,
        count=args.count,
        seed=args.seed,
        width=320,
        height=220,
    )
    dataset = SyntheticBoundaryDataset(args.dataset_dir, manifest, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TinyBoundaryNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        losses: list[float] = []
        for images, masks in loader:
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1} loss={sum(losses) / max(1, len(losses)):.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros((1, 3, args.image_size, args.image_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["image"],
        output_names=["mask_logits"],
        opset_version=17,
    )
    print(f"wrote {args.output}")
    return 0


class SyntheticBoundaryDataset:
    def __init__(self, root: Path, manifest: SyntheticDatasetManifest, image_size: int):
        self.root = root
        self.samples = list(manifest.samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self.root / sample.artifacts.screenshot).convert("RGB")
        mask = Image.open(self.root / sample.artifacts.mask).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        image_tensor = np.transpose(image_arr, (2, 0, 1))
        return image_tensor, mask_arr[np.newaxis, :, :]


class TinyBoundaryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.mid = block(64, 96)
        self.dec3 = block(96 + 64, 64)
        self.dec2 = block(64 + 32, 32)
        self.dec1 = block(32 + 16, 16)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        mid = self.mid(F.max_pool2d(e3, 2))
        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def block(in_channels: int, out_channels: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def dice_loss(logits, masks):
    import torch

    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return (1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))).mean()


if __name__ == "__main__":
    raise SystemExit(main())
