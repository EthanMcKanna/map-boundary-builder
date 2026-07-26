from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from map_boundary_builder.synthetic import SyntheticDatasetManifest, generate_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the native-resolution BoundaryField refiner.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/synthetic-v12-boundaryfield-train"))
    parser.add_argument("--output", type=Path, default=Path("map_boundary_builder/models/boundaryfield_v12_refiner.onnx"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--validation-count", type=int, default=384)
    parser.add_argument("--training-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--observable-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=640)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0012)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--coarse-model", type=Path, default=None)
    parser.add_argument("--coarse-input-size", type=int, default=256)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--export-checkpoint", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    if args.export_checkpoint is not None:
        export_checkpoint(args)
        return 0

    manifest_path = args.dataset_dir / "manifest.json"
    if args.reuse_dataset and manifest_path.exists():
        manifest = SyntheticDatasetManifest.read_json(manifest_path)
    else:
        manifest = generate_synthetic_dataset(
            args.dataset_dir,
            count=args.count + args.validation_count,
            seed=args.seed,
            width=args.render_width,
            height=args.render_height,
        )
    samples = list(manifest.samples)
    validation_samples = samples[: args.validation_count]
    training_samples = samples[args.validation_count :]
    if args.observable_only:
        validation_samples = [sample for sample in validation_samples if observable_boundary_sample(sample)]
        training_samples = [sample for sample in training_samples if observable_boundary_sample(sample)]
    if args.validation_limit > 0:
        validation_samples = validation_samples[: args.validation_limit]
    if args.training_limit > 0:
        training_samples = training_samples[: args.training_limit]
    train_dataset = BoundaryPatchDataset(
        args.dataset_dir,
        training_samples,
        args.patch_size,
        augment=True,
        coarse_model=args.coarse_model,
        coarse_input_size=args.coarse_input_size,
    )
    validation_dataset = BoundaryPatchDataset(
        args.dataset_dir,
        validation_samples,
        args.patch_size,
        augment=False,
        coarse_model=args.coarse_model,
        coarse_input_size=args.coarse_input_size,
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    model = BoundaryRefinerNet(input_channels=7, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    checkpoint_dir = args.checkpoint_dir or args.output.parent / f"{args.output.stem}.checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_boundary_iou = -1.0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_boundary_iou = float(checkpoint.get("best_boundary_iou", -1.0))

    print(
        "training-boundaryfield",
        f"params={sum(parameter.numel() for parameter in model.parameters())}",
        f"device={device}",
        f"train={len(train_dataset)}",
        f"validation={len(validation_dataset)}",
        f"patch={args.patch_size}",
        flush=True,
    )
    if start_epoch == 0 and args.resume_checkpoint is None:
        initial_metrics = evaluate(model, validation_loader, device=device)
        best_boundary_iou = initial_metrics["boundary_iou_2px"]
        initial_checkpoint = {
            "epoch": 0,
            "base_channels": args.base_channels,
            "patch_size": args.patch_size,
            "best_boundary_iou": best_boundary_iou,
            "metrics": initial_metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(initial_checkpoint, checkpoint_dir / "epoch-000.pt")
        torch.save(initial_checkpoint, checkpoint_dir / "best.pt")
        print(
            f"epoch=0 iou={initial_metrics['iou']:.5f} "
            f"boundary_iou_2px={initial_metrics['boundary_iou_2px']:.5f} "
            f"corner_f1={initial_metrics['corner_f1']:.5f}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses: list[float] = []
        for images, masks, signed_distances, corners in loader:
            images = images.to(device)
            masks = masks.to(device)
            signed_distances = signed_distances.to(device)
            corners = corners.to(device)
            fields = model(images)
            loss = boundaryfield_loss(fields, masks, signed_distances, corners)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation_loader, device=device)
        scheduler.step()
        is_best = metrics["boundary_iou_2px"] >= best_boundary_iou
        if is_best:
            best_boundary_iou = metrics["boundary_iou_2px"]
        print(
            f"epoch={epoch + 1} loss={sum(losses) / max(1, len(losses)):.5f} "
            f"iou={metrics['iou']:.5f} boundary_iou_2px={metrics['boundary_iou_2px']:.5f} "
            f"corner_f1={metrics['corner_f1']:.5f} lr={scheduler.get_last_lr()[0]:.7f}",
            flush=True,
        )
        checkpoint_path = checkpoint_dir / f"epoch-{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "base_channels": args.base_channels,
                "patch_size": args.patch_size,
                "best_boundary_iou": best_boundary_iou,
                "metrics": metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint_path,
        )
        (checkpoint_dir / "latest.pt").write_bytes(checkpoint_path.read_bytes())
        if is_best:
            (checkpoint_dir / "best.pt").write_bytes(checkpoint_path.read_bytes())

    best_path = checkpoint_dir / "best.pt"
    checkpoint = torch.load(best_path, map_location="cpu")
    model = BoundaryRefinerNet(input_channels=7, base_channels=int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    export_model(model, args.output, patch_size=int(checkpoint["patch_size"]))
    return 0


def observable_boundary_sample(sample) -> bool:
    """Select fixtures whose source-resolution edge is actually visible."""
    return (
        float(sample.overlay_style.fill_opacity) >= 0.30
        and not bool(sample.properties.get("labels_on_top"))
        and not bool(sample.properties.get("include_distractor"))
    )


class BoundaryPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        samples: Sequence,
        patch_size: int,
        *,
        augment: bool,
        coarse_model: Path | None = None,
        coarse_input_size: int = 256,
    ) -> None:
        self.root = root
        self.samples = list(samples)
        self.patch_size = patch_size
        self.augment = augment
        self.coarse_model = coarse_model
        self.coarse_input_size = coarse_input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(self.root / sample.artifacts.screenshot) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(self.root / sample.artifacts.mask) as image:
            mask = np.asarray(image.convert("L")) > 0
        if self.augment and random.random() < 0.5:
            rgb = rgb[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if self.augment and random.random() < 0.5:
            rgb = rgb[::-1].copy()
            mask = mask[::-1].copy()

        rgb_float = rgb.astype(np.float32) / 255.0
        if self.augment:
            rgb_float = color_jitter(rgb_float)
        inference_rgb = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
        if self.coarse_model is not None:
            from map_boundary_builder.model_extract import (
                ModelExtractionConfig,
                guidance_channels,
                load_onnx_session,
                predict_mask_probabilities,
            )

            hints = sample_training_hints(inference_rgb, mask, sample, augment=self.augment)
            coarse = predict_mask_probabilities(
                inference_rgb,
                load_onnx_session(str(self.coarse_model)),
                config=ModelExtractionConfig(
                    input_width=self.coarse_input_size,
                    input_height=self.coarse_input_size,
                    threshold=0.45,
                    output_activation="logits",
                    input_channels=5,
                ),
                hints=hints,
            )
            guide = guidance_channels(inference_rgb, mask.shape[1], mask.shape[0], hints=hints)
        else:
            coarse = simulated_coarse_probability(mask, augment=self.augment)
            guide = training_guidance(rgb_float, mask, sample, augment=self.augment)
        boundary = binary_boundary(mask) | binary_boundary(coarse >= 0.45)
        ys, xs = np.where(boundary)
        if not len(xs):
            raise ValueError(f"training mask has no boundary: {sample.sample_id}")
        pick = random.randrange(len(xs)) if self.augment else (index * 104729) % len(xs)
        center_x = int(xs[pick])
        center_y = int(ys[pick])
        y0 = center_y - self.patch_size // 2
        x0 = center_x - self.patch_size // 2

        gray = cv2.cvtColor((rgb_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        edge /= max(1e-6, float(np.quantile(edge, 0.99)))
        channels = np.concatenate(
            [
                np.transpose(rgb_float, (2, 0, 1)),
                coarse[np.newaxis],
                guide,
                np.clip(edge, 0.0, 1.0)[np.newaxis],
            ],
            axis=0,
        )
        image_patch = crop_with_padding(channels, y0, x0, self.patch_size, mode="reflect")
        mask_patch = crop_with_padding(mask[np.newaxis].astype(np.float32), y0, x0, self.patch_size, mode="edge")
        sdf_patch = signed_distance(mask_patch[0] > 0.5, clip_px=24.0)[np.newaxis]
        corner_patch = corner_target(mask_patch[0] > 0.5)[np.newaxis]
        return (
            np.ascontiguousarray(image_patch, dtype=np.float32),
            np.ascontiguousarray(mask_patch, dtype=np.float32),
            np.ascontiguousarray(sdf_patch, dtype=np.float32),
            np.ascontiguousarray(corner_patch, dtype=np.float32),
        )


def simulated_coarse_probability(mask: np.ndarray, *, augment: bool) -> np.ndarray:
    height, width = mask.shape
    low_size = random.choice((128, 160, 192, 224, 256)) if augment else 256
    low = cv2.resize(mask.astype(np.float32), (low_size, low_size), interpolation=cv2.INTER_AREA)
    if augment:
        shift_x = random.randint(-1, 1)
        shift_y = random.randint(-1, 1)
        low = np.roll(low, (shift_y, shift_x), axis=(0, 1))
        if random.random() < 0.5:
            kernel = np.ones((3, 3), np.uint8)
            low = (
                cv2.dilate((low > 0.45).astype(np.uint8), kernel, iterations=1)
                if random.random() < 0.5
                else cv2.erode((low > 0.45).astype(np.uint8), kernel, iterations=1)
            ).astype(np.float32)
    coarse = cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR)
    if augment:
        coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=random.uniform(0.4, 2.0))
        coarse += np.random.normal(0.0, random.uniform(0.0, 0.035), coarse.shape).astype(np.float32)
    return np.clip(coarse, 0.0, 1.0).astype(np.float32)


def training_guidance(rgb: np.ndarray, mask: np.ndarray, sample, *, augment: bool) -> np.ndarray:
    height, width = mask.shape
    seed_map = np.zeros((height, width), dtype=np.float32)
    target_map = np.zeros((height, width), dtype=np.float32)
    ys, xs = np.where(mask)
    if len(xs) and (not augment or random.random() < 0.65):
        pick = random.randrange(len(xs)) if augment else len(xs) // 2
        yy, xx = np.mgrid[0:height, 0:width]
        sigma = max(2.0, min(width, height) * 0.035)
        seed_map = np.exp(-((xx - xs[pick]) ** 2 + (yy - ys[pick]) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    if len(xs) and (not augment or random.random() < 0.78):
        target = observed_overlay_color(rgb, mask, sample)
        distance = np.linalg.norm(rgb - target.reshape(1, 1, 3), axis=2) / math.sqrt(3.0)
        target_map = (1.0 - np.clip(distance, 0.0, 1.0)).astype(np.float32)
    return np.stack([seed_map, target_map], axis=0)


def sample_training_hints(rgb: np.ndarray, mask: np.ndarray, sample, *, augment: bool) -> dict[str, object] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    hints: dict[str, object] = {}
    if not augment or random.random() < 0.65:
        pick = random.randrange(len(xs)) if augment else len(xs) // 2
        hints["seed_point"] = (float(xs[pick]), float(ys[pick]))
    if not augment or random.random() < 0.78:
        target = observed_overlay_color(rgb.astype(np.float32) / 255.0, mask, sample)
        hints["target_rgb"] = tuple(int(round(value * 255.0)) for value in target)
    return hints or None


def observed_overlay_color(rgb: np.ndarray, mask: np.ndarray, sample) -> np.ndarray:
    style = sample.overlay_style
    if float(style.fill_opacity) > 0.0 or not style.stroke_color:
        return np.median(rgb[mask], axis=0).astype(np.float32)
    binary = mask.astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    ring = cv2.dilate(binary, kernel, iterations=1) != cv2.erode(binary, kernel, iterations=1)
    candidates = rgb[ring]
    color = style.stroke_color.lstrip("#")
    declared = np.asarray([int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
    if len(candidates) == 0:
        return declared
    distances = np.linalg.norm(candidates - declared, axis=1)
    keep = max(8, len(candidates) // 5)
    return np.median(candidates[np.argpartition(distances, keep - 1)[:keep]], axis=0).astype(np.float32)


def color_jitter(rgb: np.ndarray) -> np.ndarray:
    brightness = random.uniform(0.82, 1.18)
    contrast = random.uniform(0.82, 1.18)
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    result = (rgb - mean) * contrast + mean
    result *= brightness
    if random.random() < 0.3:
        gray = result.mean(axis=2, keepdims=True)
        result = gray + (result - gray) * random.uniform(0.45, 1.45)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def crop_with_padding(values: np.ndarray, y0: int, x0: int, size: int, *, mode: str) -> np.ndarray:
    height, width = values.shape[-2:]
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y0 + size - height)
    pad_right = max(0, x0 + size - width)
    padded = np.pad(values, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    start_y = y0 + pad_top
    start_x = x0 + pad_left
    return padded[:, start_y : start_y + size, start_x : start_x + size]


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return (binary > 0) & (eroded == 0)


def signed_distance(mask: np.ndarray, *, clip_px: float) -> np.ndarray:
    binary = mask.astype(np.uint8)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    return np.clip((inside - outside) / clip_px, -1.0, 1.0).astype(np.float32)


def corner_target(mask: np.ndarray) -> np.ndarray:
    target = np.zeros(mask.shape, dtype=np.float32)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 8:
            continue
        approximation = cv2.approxPolyDP(contour, epsilon=1.25, closed=True)
        points = approximation[:, 0, :]
        for index, point in enumerate(points):
            before = points[index - 1].astype(np.float32) - point
            after = points[(index + 1) % len(points)].astype(np.float32) - point
            denominator = max(1e-6, float(np.linalg.norm(before) * np.linalg.norm(after)))
            angle = math.degrees(math.acos(float(np.clip(np.dot(before, after) / denominator, -1.0, 1.0))))
            if angle < 155.0:
                cv2.circle(target, (int(point[0]), int(point[1])), 3, 1.0, thickness=-1)
    return cv2.GaussianBlur(target, (0, 0), sigmaX=1.25)


class BoundaryRefinerNet(nn.Module):
    def __init__(self, *, input_channels: int = 7, base_channels: int = 24) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 3, base_channels * 4
        self.stem = nn.Sequential(nn.Conv2d(input_channels, c1, 3, padding=1, bias=False), norm(c1), nn.SiLU())
        self.enc1 = DSResidual(c1)
        self.down2 = Down(c1, c2)
        self.enc2 = DSResidual(c2)
        self.down3 = Down(c2, c3)
        self.enc3 = DSResidual(c3)
        self.down4 = Down(c3, c4)
        self.mid = nn.Sequential(DSResidual(c4), DSResidual(c4))
        self.up3 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up1 = Up(c2, c1, c1)
        self.head_body = DSResidual(c1)
        self.field_head = nn.Conv2d(c1, 3, kernel_size=1)
        nn.init.zeros_(self.field_head.weight)
        nn.init.zeros_(self.field_head.bias)

    def forward(self, values):
        e1 = self.enc1(self.stem(values))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        mid = self.mid(self.down4(e3))
        decoded = self.up1(self.up2(self.up3(mid, e3), e2), e1)
        residuals = self.field_head(self.head_body(decoded))
        coarse = torch.clamp(values[:, 3:4], 0.001, 0.999)
        coarse_logit = torch.log(coarse) - torch.log1p(-coarse)
        uncertainty = 4.0 * coarse * (1.0 - coarse)
        # The 256px selector can be several source pixels off after upsampling.
        # Zero initialization keeps epoch zero identical, while this bounded
        # residual has enough range to move the native-resolution zero level set.
        mask_logit = coarse_logit + 4.0 * uncertainty * torch.tanh(residuals[:, 0:1])
        signed_distance = torch.clamp(
            coarse_logit / 8.0 + 0.55 * uncertainty * torch.tanh(residuals[:, 1:2]),
            -1.5,
            1.5,
        )
        return torch.cat([mask_logit, signed_distance, residuals[:, 2:3]], dim=1)


class DSResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            norm(channels),
        )

    def forward(self, values):
        return F.silu(values + self.block(values))


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, values):
        return self.block(values)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            norm(out_channels),
            nn.SiLU(),
            DSResidual(out_channels),
        )

    def forward(self, values, skip):
        values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([values, skip], dim=1))


def norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 6 if channels % 6 == 0 else 4
    return nn.GroupNorm(groups, channels)


def boundaryfield_loss(fields, masks, signed_distances, corners):
    mask_logits = fields[:, 0:1]
    sdf_logits = fields[:, 1:2]
    predicted_sdf = torch.clamp(sdf_logits, -1.0, 1.0)
    corner_logits = fields[:, 2:3]
    probabilities = torch.sigmoid(mask_logits)
    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probabilities.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1.0)).mean()
    band_weight = 1.0 + 5.0 * (signed_distances.abs() < 0.35).float()
    region = (F.binary_cross_entropy_with_logits(mask_logits, masks, reduction="none") * band_weight).mean()
    sdf = (F.smooth_l1_loss(predicted_sdf, signed_distances, reduction="none") * band_weight).mean()
    boundary = F.l1_loss(edge_map(probabilities), edge_map(masks))
    sdf_region = (F.binary_cross_entropy_with_logits(sdf_logits * 6.0, masks, reduction="none") * band_weight).mean()
    corner_targets = (corners >= 0.1).float()
    corner = F.binary_cross_entropy_with_logits(
        corner_logits,
        corner_targets,
        pos_weight=torch.tensor(60.0, device=corner_logits.device),
    )
    # Promotion is determined by the native contour. Auxiliary fields remain
    # available for diagnostics, but must not dilute zero-level-set learning.
    return 0.58 * region + 0.16 * dice + 0.26 * boundary + 0.0 * (sdf + sdf_region + corner)


def edge_map(values):
    minimum = -F.max_pool2d(-values, 3, stride=1, padding=1)
    maximum = F.max_pool2d(values, 3, stride=1, padding=1)
    return maximum - minimum


@torch.no_grad()
def evaluate(model: nn.Module, loader, *, device: torch.device) -> dict[str, float]:
    model.eval()
    intersections = unions = 0
    boundary_intersections = boundary_unions = 0
    corner_tp = corner_fp = corner_fn = 0
    kernel = torch.ones((1, 1, 5, 5), device=device)
    for images, masks, _signed_distances, corners in loader:
        fields = model(images.to(device))
        predictions = torch.sigmoid(fields[:, 0:1]) >= 0.5
        targets = masks.to(device) >= 0.5
        intersections += int((predictions & targets).sum())
        unions += int((predictions | targets).sum())
        predicted_boundary = edge_map(predictions.float()) > 0
        target_boundary = edge_map(targets.float()) > 0
        predicted_band = F.conv2d(predicted_boundary.float(), kernel, padding=2) > 0
        target_band = F.conv2d(target_boundary.float(), kernel, padding=2) > 0
        boundary_intersections += int((predicted_band & target_band).sum())
        boundary_unions += int((predicted_band | target_band).sum())
        predicted_corners = torch.sigmoid(fields[:, 2:3]) >= 0.45
        target_corners = corners.to(device) >= 0.1
        dilated_predicted = F.conv2d(predicted_corners.float(), kernel, padding=2) > 0
        dilated_target = F.conv2d(target_corners.float(), kernel, padding=2) > 0
        corner_tp += int((predicted_corners & dilated_target).sum())
        corner_fp += int((predicted_corners & ~dilated_target).sum())
        corner_fn += int((target_corners & ~dilated_predicted).sum())
    precision = corner_tp / max(1, corner_tp + corner_fp)
    recall = corner_tp / max(1, corner_tp + corner_fn)
    return {
        "iou": intersections / max(1, unions),
        "boundary_iou_2px": boundary_intersections / max(1, boundary_unions),
        "corner_f1": 2.0 * precision * recall / max(1e-9, precision + recall),
    }


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def export_checkpoint(args) -> None:
    checkpoint = torch.load(args.export_checkpoint, map_location="cpu")
    model = BoundaryRefinerNet(input_channels=7, base_channels=int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    export_model(model, args.output, patch_size=int(checkpoint["patch_size"]))


def export_model(model: nn.Module, output: Path, *, patch_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros((1, 7, patch_size, patch_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["boundary_patch"],
        output_names=["boundary_fields"],
        dynamic_axes={"boundary_patch": {0: "batch"}, "boundary_fields": {0: "batch"}},
        opset_version=18,
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
