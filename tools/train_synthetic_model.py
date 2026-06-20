from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from map_boundary_builder.synthetic import generate_synthetic_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a synthetic boundary segmentation model.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/synthetic-model-train"))
    parser.add_argument("--output", type=Path, default=Path("map_boundary_builder/models/synthetic_boundary_v2.onnx"))
    parser.add_argument("--count", type=int, default=1536)
    parser.add_argument("--validation-count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--arch", choices=("tiny", "resunet"), default="resunet")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    manifest = generate_synthetic_dataset(
        args.dataset_dir,
        count=args.count + args.validation_count,
        seed=args.seed,
        width=320,
        height=220,
    )
    samples = list(manifest.samples)
    validation_samples = samples[: args.validation_count]
    training_samples = samples[args.validation_count :]
    train_dataset = SyntheticBoundaryDataset(args.dataset_dir, training_samples, args.image_size, augment=True)
    validation_dataset = SyntheticBoundaryDataset(args.dataset_dir, validation_samples, args.image_size, augment=False)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.arch, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    print(
        "training",
        f"arch={args.arch}",
        f"params={sum(param.numel() for param in model.parameters())}",
        f"device={device}",
        f"train={len(train_dataset)}",
        f"validation={len(validation_dataset)}",
        f"image_size={args.image_size}",
        flush=True,
    )
    model.train()
    for epoch in range(args.epochs):
        losses: list[float] = []
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_iou = evaluate_iou(model, validation_loader, device=device)
        print(
            f"epoch={epoch + 1} "
            f"loss={sum(losses) / max(1, len(losses)):.5f} "
            f"validation_iou={validation_iou:.5f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    example = torch.zeros((1, 3, args.image_size, args.image_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["image"],
        output_names=["mask_logits"],
        opset_version=18,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


class SyntheticBoundaryDataset:
    def __init__(
        self,
        root: Path,
        samples: Sequence,
        image_size: int,
        *,
        augment: bool = False,
    ):
        self.root = root
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment

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
        if self.augment:
            if random.random() < 0.5:
                image_arr = image_arr[:, ::-1, :]
                mask_arr = mask_arr[:, ::-1]
            if random.random() < 0.5:
                image_arr = image_arr[::-1, :, :]
                mask_arr = mask_arr[::-1, :]
            image_arr = color_jitter(image_arr)
        image_tensor = np.transpose(image_arr, (2, 0, 1))
        return np.ascontiguousarray(image_tensor), np.ascontiguousarray(mask_arr[np.newaxis, :, :])


def color_jitter(image_arr: np.ndarray) -> np.ndarray:
    brightness = random.uniform(0.88, 1.12)
    contrast = random.uniform(0.90, 1.12)
    mean = image_arr.mean(axis=(0, 1), keepdims=True)
    jittered = (image_arr - mean) * contrast + mean
    jittered = jittered * brightness
    return np.clip(jittered, 0.0, 1.0).astype(np.float32)


def build_model(arch: str, *, base_channels: int) -> nn.Module:
    if arch == "tiny":
        return TinyBoundaryNet()
    if arch == "resunet":
        return ResidualBoundaryNet(base_channels=base_channels)
    raise ValueError(f"unknown architecture: {arch}")


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


class ResidualBoundaryNet(nn.Module):
    def __init__(self, *, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6
        c5 = base_channels * 8
        self.stem = ResidualBlock(3, c1)
        self.enc2 = DownsampleBlock(c1, c2)
        self.enc3 = DownsampleBlock(c2, c3)
        self.enc4 = DownsampleBlock(c3, c4)
        self.mid = DownsampleBlock(c4, c5)
        self.dec4 = UpsampleBlock(c5, c4, c4)
        self.dec3 = UpsampleBlock(c4, c3, c3)
        self.dec2 = UpsampleBlock(c3, c2, c2)
        self.dec1 = UpsampleBlock(c2, c1, c1)
        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        mid = self.mid(e4)
        d4 = self.dec4(mid, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.proj(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(F.max_pool2d(x, 2))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


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


@torch.no_grad()
def evaluate_iou(model: nn.Module, loader, *, device: torch.device) -> float:
    model.eval()
    scores: list[float] = []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        predictions = torch.sigmoid(logits) >= 0.5
        targets = masks >= 0.5
        intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
        union = (predictions | targets).sum(dim=(1, 2, 3)).float()
        scores.extend(((intersection + 1.0) / (union + 1.0)).detach().cpu().tolist())
    model.train()
    return float(sum(scores) / max(1, len(scores)))


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


if __name__ == "__main__":
    raise SystemExit(main())
