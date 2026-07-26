"""Train the source-resolution contour localizer used by v20 EdgeGraph.

The global selector is deliberately not retrained here.  It supplies semantic
component/topology candidates; this model learns only the one-dimensional
question the old architecture discarded: where, along a source-pixel normal
profile, is the observable boundary?
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from map_boundary_builder.synthetic import SyntheticDatasetManifest


INPUT_CHANNEL_NAMES = (
    "rgb_red",
    "rgb_green",
    "rgb_blue",
    "luminance",
    "scharr_magnitude_p99",
    "normal_projected_scharr_p99",
    "target_rgb_similarity",
    "source_sample_valid",
    "normalized_native_offset",
    "inside_direction_sign",
)
INPUT_CHANNEL_COUNT = len(INPUT_CHANNEL_NAMES)
SOURCE_VALID_CHANNEL_INDEX = INPUT_CHANNEL_NAMES.index("source_sample_valid")
ONNX_INPUT_NAME = "edge_strip_vector_v3"
VECTOR_TARGET_VERSION = "geojson-pixel-ray-segment-v1"
PROFILE_DILATIONS = (1, 2, 4)
PROFILE_RECEPTIVE_FIELD_BINS = 1 + 4 * sum(PROFILE_DILATIONS)
PROFILE_CENTER_SHIFT_MAX_ABS_PX = 6.0
INSIDE_DIRECTION_REFERENCE_VERSION = "unshifted-contour-center-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train generalized v20 EdgeGraph's contour-strip model.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("out/synthetic-v20-edgegraph-train"))
    parser.add_argument("--output", type=Path, default=Path("map_boundary_builder/models/edgegraph_v20_refiner.onnx"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--training-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--strip-radius", type=int, default=12)
    parser.add_argument("--strip-step", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--coarse-model",
        type=Path,
        default=None,
        help="Use frozen selector predictions instead of simulated coarse masks.",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--export-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--conservative-offset-head",
        action="store_true",
        help="Explicitly zero the learned offset head at export for diagnostics only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.export_checkpoint is not None:
        checkpoint = torch.load(args.export_checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(checkpoint)
        model = EdgeStripNet(channels=checkpoint_hidden_channels(checkpoint))
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.conservative_offset_head:
            stabilize_offset_head(model)
        export_model(
            model,
            args.output,
            chunk_length=int(checkpoint["chunk_length"]),
            strip_radius=int(checkpoint["strip_radius"]),
            strip_step=float(checkpoint.get("strip_step", 0.5)),
        )
        write_model_metadata(
            args.output,
            checkpoint,
            conservative_offset_head=bool(args.conservative_offset_head),
        )
        return 0

    if args.epochs < 1:
        raise ValueError("epochs must be at least one so an untrained checkpoint cannot be exported")
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = SyntheticDatasetManifest.read_json(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    samples = list(manifest.samples)
    if not 0 < args.validation_limit < len(samples):
        raise ValueError(
            f"validation-limit must leave training samples; got {args.validation_limit} for {len(samples)} samples"
        )
    validation_samples = samples[: args.validation_limit]
    training_samples = samples[args.validation_limit :]
    if args.training_limit > 0:
        training_samples = training_samples[: args.training_limit]
    device = select_device(args.device)
    training = EdgeStripDataset(
        args.dataset_dir,
        training_samples,
        chunk_length=args.chunk_length,
        strip_radius=args.strip_radius,
        strip_step=args.strip_step,
        coarse_model=args.coarse_model,
        augment=True,
    )
    validation = EdgeStripDataset(
        args.dataset_dir,
        validation_samples,
        chunk_length=args.chunk_length,
        strip_radius=args.strip_radius,
        strip_step=args.strip_step,
        coarse_model=args.coarse_model,
        augment=False,
    )
    train_loader = torch.utils.data.DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    model = EdgeStripNet(channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    checkpoint_dir = args.checkpoint_dir or args.output.parent / f"{args.output.stem}.checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_score = -float("inf")
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        validate_checkpoint_contract(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint.get("best_score", -1.0))

    print(
        "training-edgegraph",
        f"params={sum(parameter.numel() for parameter in model.parameters())}",
        f"device={device}",
        f"train={len(training)}",
        f"validation={len(validation)}",
        f"tensor={INPUT_CHANNEL_COUNT}x{args.chunk_length}x{round(args.strip_radius * 2 / args.strip_step) + 1}",
        flush=True,
    )
    if start_epoch == 0:
        metrics = evaluate(model, validation_loader, device=device, offset_step=args.strip_step)
        initial_score = checkpoint_score(metrics)
        checkpoint = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            hidden_channels=args.channels,
            chunk_length=args.chunk_length,
            strip_radius=args.strip_radius,
            strip_step=args.strip_step,
            best_score=-float("inf"),
            metrics=metrics,
            dataset_version=manifest.version,
            dataset_manifest_sha256=manifest_sha256,
            dataset_sample_count=len(samples),
        )
        checkpoint["initial_score"] = initial_score
        torch.save(checkpoint, checkpoint_dir / "epoch-000.pt")
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        print(
            f"epoch=0 offset_mae={metrics['offset_mae_px']:.4f}px "
            f"within1={metrics['within_1px']:.5f} corner_f1={metrics['corner_f1']:.5f} "
            f"reliability_acc={metrics['reliability_accuracy']:.5f}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses: list[float] = []
        for features, offset_index, corners, reliability in train_loader:
            features = features.to(device)
            offset_index = offset_index.to(device)
            corners = corners.to(device)
            reliability = reliability.to(device)
            outputs = model(features)
            loss = edgegraph_loss(outputs, offset_index, corners, reliability)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation_loader, device=device, offset_step=args.strip_step)
        scheduler.step()
        score = checkpoint_score(metrics)
        is_best = score >= best_score
        if is_best:
            best_score = score
        checkpoint = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            hidden_channels=args.channels,
            chunk_length=args.chunk_length,
            strip_radius=args.strip_radius,
            strip_step=args.strip_step,
            best_score=best_score,
            metrics=metrics,
            dataset_version=manifest.version,
            dataset_manifest_sha256=manifest_sha256,
            dataset_sample_count=len(samples),
        )
        checkpoint_path = checkpoint_dir / f"epoch-{epoch + 1:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        print(
            f"epoch={epoch + 1} loss={sum(losses) / max(1, len(losses)):.5f} "
            f"offset_mae={metrics['offset_mae_px']:.4f}px within1={metrics['within_1px']:.5f} "
            f"corner_f1={metrics['corner_f1']:.5f} reliability_acc={metrics['reliability_accuracy']:.5f} "
            f"lr={scheduler.get_last_lr()[0]:.7f}",
            flush=True,
        )

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    validate_checkpoint_contract(checkpoint)
    model = EdgeStripNet(channels=checkpoint_hidden_channels(checkpoint))
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.conservative_offset_head:
        stabilize_offset_head(model)
    export_model(
        model,
        args.output,
        chunk_length=int(checkpoint["chunk_length"]),
        strip_radius=int(checkpoint["strip_radius"]),
        strip_step=float(checkpoint.get("strip_step", 0.5)),
    )
    write_model_metadata(
        args.output,
        checkpoint,
        conservative_offset_head=bool(args.conservative_offset_head),
    )
    return 0


def checkpoint_hidden_channels(checkpoint: Mapping[str, Any]) -> int:
    value = checkpoint.get("hidden_channels", checkpoint.get("channels"))
    if not isinstance(value, (int, float)):
        raise ValueError("EdgeGraph checkpoint is missing hidden channel width")
    return int(value)


def validate_checkpoint_contract(checkpoint: Mapping[str, Any]) -> None:
    input_channels = int(checkpoint.get("input_channels", 0))
    input_channel_names = checkpoint.get("input_channel_names")
    target_version = checkpoint.get("vector_target_version")
    center_shift_max_abs_px = checkpoint.get("profile_center_shift_max_abs_px")
    inside_direction_reference = checkpoint.get("inside_direction_reference_version")
    if (
        input_channels != INPUT_CHANNEL_COUNT
        or input_channel_names != list(INPUT_CHANNEL_NAMES)
        or target_version != VECTOR_TARGET_VERSION
        or center_shift_max_abs_px != PROFILE_CENTER_SHIFT_MAX_ABS_PX
        or inside_direction_reference != INSIDE_DIRECTION_REFERENCE_VERSION
    ):
        raise ValueError(
            "checkpoint predates the production 10-channel vector-v3 contract; "
            "retrain it instead of exporting incompatible weights"
        )


def checkpoint_score(metrics: Mapping[str, float]) -> float:
    """Prefer subpixel tail quality over a forgiving one-pixel average."""

    return float(
        metrics["within_0_5px"]
        + 0.25 * metrics["within_1px"]
        + 0.10 * metrics["corner_f1"]
        + 0.05 * metrics["reliability_balanced_accuracy"]
        - 0.15 * metrics["offset_p95_px"]
        - 0.05 * metrics["offset_mae_px"]
    )


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    hidden_channels: int,
    chunk_length: int,
    strip_radius: int,
    strip_step: float,
    best_score: float,
    metrics: Mapping[str, float],
    dataset_version: str,
    dataset_manifest_sha256: str,
    dataset_sample_count: int,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "channels": int(hidden_channels),
        "hidden_channels": int(hidden_channels),
        "input_channels": INPUT_CHANNEL_COUNT,
        "input_channel_names": list(INPUT_CHANNEL_NAMES),
        "profile_center_shift_max_abs_px": PROFILE_CENTER_SHIFT_MAX_ABS_PX,
        "inside_direction_reference_version": INSIDE_DIRECTION_REFERENCE_VERSION,
        "vector_target_version": VECTOR_TARGET_VERSION,
        "chunk_length": int(chunk_length),
        "strip_radius": int(strip_radius),
        "strip_step": float(strip_step),
        "best_score": float(best_score),
        "dataset_version": str(dataset_version),
        "dataset_manifest_sha256": str(dataset_manifest_sha256),
        "dataset_sample_count": int(dataset_sample_count),
        "metrics": dict(metrics),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }


def stabilize_offset_head(model: "EdgeStripNet") -> None:
    """Explicit diagnostic ablation; normal v20 exports preserve learned offsets."""
    nn.init.zeros_(model.offset_head.weight)
    nn.init.zeros_(model.offset_head.bias)


def write_model_metadata(output: Path, checkpoint: dict, *, conservative_offset_head: bool) -> None:
    metadata = {
        "architecture": "generalized-v20-edgegraph-stripnet-vector-v3",
        "channels": INPUT_CHANNEL_COUNT,
        "input_channels": list(INPUT_CHANNEL_NAMES),
        "onnx_input_name": ONNX_INPUT_NAME,
        "hidden_channels": checkpoint_hidden_channels(checkpoint),
        "chunk_length": int(checkpoint["chunk_length"]),
        "strip_radius_px": int(checkpoint["strip_radius"]),
        "offset_step_px": float(checkpoint.get("strip_step", 0.5)),
        "conservative_offset_head": conservative_offset_head,
        "learned_offset_logits_preserved": not conservative_offset_head,
        "offset_head": "unbounded_learned_logits",
        "profile_dilations": list(PROFILE_DILATIONS),
        "profile_receptive_field_bins": PROFILE_RECEPTIVE_FIELD_BINS,
        "profile_receptive_field_px": (
            (PROFILE_RECEPTIVE_FIELD_BINS - 1)
            * float(checkpoint.get("strip_step", 0.5))
        ),
        "training_dataset": {
            "version": checkpoint.get("dataset_version"),
            "manifest_sha256": checkpoint.get("dataset_manifest_sha256"),
            "sample_count": checkpoint.get("dataset_sample_count"),
        },
        "training_augmentation": {
            "profile_center_shift_distribution": "uniform",
            "profile_center_shift_max_abs_px": checkpoint.get(
                "profile_center_shift_max_abs_px"
            ),
            "profile_center_shift_scope": "global_per_sample",
            "inside_direction_reference": checkpoint.get(
                "inside_direction_reference_version"
            ),
        },
        "coarse_probability_input": False,
        "target_similarity": "clip(1 - rgb_l2_distance / sqrt(3), 0, 1)",
        "valid_channel": "bilinear_sample_coordinate_inside_source_bounds",
        "normalized_native_offset": "strip_offset_px / strip_radius_px",
        "inside_direction_sign": (
            "repeated sign inferred from unshifted contour-center coarse profiles"
        ),
        "supervision": {
            "source": "GeoJSON metadata.pixel_geometry",
            "target_version": VECTOR_TARGET_VERSION,
            "offsets": "continuous vectorized normal-ray segment intersections",
            "corners": "exact vector vertices with interior angle <= 155 degrees",
        },
        "training_epoch": int(checkpoint.get("epoch", 0)),
        "loss_weights": {
            "offset_distribution": 0.50,
            "subpixel_offset": 0.25,
            "corner": 0.12,
            "reliability": 0.08,
            "supervised_smoothness": 0.05,
        },
        "validation": checkpoint["metrics"],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class VectorBoundary:
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    corners: np.ndarray

    def __post_init__(self) -> None:
        starts = np.asarray(self.segment_starts)
        ends = np.asarray(self.segment_ends)
        corners = np.asarray(self.corners)
        if starts.ndim != 2 or starts.shape[1:] != (2,) or starts.shape != ends.shape:
            raise ValueError("vector boundary segments must have matching [segment, 2] endpoints")
        if not len(starts):
            raise ValueError("vector boundary must contain at least one segment")
        if corners.ndim != 2 or corners.shape[1:] != (2,):
            raise ValueError("vector boundary corners must have shape [corner, 2]")
        if not np.isfinite(starts).all() or not np.isfinite(ends).all() or not np.isfinite(corners).all():
            raise ValueError("vector boundary contains non-finite coordinates")


def load_vector_boundary(path: Path) -> VectorBoundary:
    return vector_boundary_from_geojson(json.loads(path.read_text(encoding="utf-8")))


def vector_boundary_from_geojson(payload: Mapping[str, Any]) -> VectorBoundary:
    metadata = payload.get("metadata")
    pixel_geometry = metadata.get("pixel_geometry") if isinstance(metadata, Mapping) else None
    if not isinstance(pixel_geometry, Mapping):
        raise ValueError("GeoJSON is missing metadata.pixel_geometry source-pixel supervision")
    rings = _pixel_geometry_rings(pixel_geometry)
    segment_starts: list[np.ndarray] = []
    segment_ends: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    for raw_ring in rings:
        ring = _normalized_vector_ring(raw_ring)
        following = np.roll(ring, -1, axis=0)
        segment_length = np.linalg.norm(following - ring, axis=1)
        keep = segment_length > 1e-8
        segment_starts.append(ring[keep])
        segment_ends.append(following[keep])

        before = np.roll(ring, 1, axis=0) - ring
        after = np.roll(ring, -1, axis=0) - ring
        denominator = np.maximum(
            np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1),
            1e-12,
        )
        angle = np.degrees(
            np.arccos(np.clip(np.sum(before * after, axis=1) / denominator, -1.0, 1.0))
        )
        corners.append(ring[angle <= 155.0])
    corner_array = np.concatenate(corners, axis=0) if any(len(item) for item in corners) else np.empty((0, 2))
    return VectorBoundary(
        segment_starts=np.ascontiguousarray(np.concatenate(segment_starts, axis=0), dtype=np.float64),
        segment_ends=np.ascontiguousarray(np.concatenate(segment_ends, axis=0), dtype=np.float64),
        corners=np.ascontiguousarray(corner_array, dtype=np.float64),
    )


def _pixel_geometry_rings(geometry: Mapping[str, Any]) -> list[Sequence[Sequence[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, Sequence):
        return list(coordinates)
    if geometry_type == "MultiPolygon" and isinstance(coordinates, Sequence):
        return [ring for polygon in coordinates for ring in polygon]
    raise ValueError(f"unsupported pixel geometry type: {geometry_type!r}")


def _normalized_vector_ring(coordinates: Sequence[Sequence[float]]) -> np.ndarray:
    ring = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    if len(ring) >= 2 and np.linalg.norm(ring[0] - ring[-1]) <= 1e-8:
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError("vector boundary ring requires at least three distinct vertices")
    previous = np.roll(ring, 1, axis=0)
    ring = ring[np.linalg.norm(ring - previous, axis=1) > 1e-8]
    if len(ring) < 3:
        raise ValueError("vector boundary ring collapsed after duplicate removal")
    return ring


def transform_vector_boundary(
    boundary: VectorBoundary,
    *,
    width: int,
    height: int,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> VectorBoundary:
    def transform(values: np.ndarray) -> np.ndarray:
        result = np.array(values, dtype=np.float64, copy=True)
        if flip_horizontal:
            result[:, 0] = float(width - 1) - result[:, 0]
        if flip_vertical:
            result[:, 1] = float(height - 1) - result[:, 1]
        return np.ascontiguousarray(result)

    return VectorBoundary(
        segment_starts=transform(boundary.segment_starts),
        segment_ends=transform(boundary.segment_ends),
        corners=transform(boundary.corners),
    )


def _cross_2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def vector_boundary_targets(
    boundary: VectorBoundary,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    radius: float,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect every signed normal with every exact vector segment in one array operation."""

    points64 = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normals64 = np.asarray(normals, dtype=np.float64).reshape(-1, 2)
    segments = boundary.segment_ends - boundary.segment_starts
    delta = boundary.segment_starts[None, :, :] - points64[:, None, :]
    denominator = _cross_2d(normals64[:, None, :], segments[None, :, :])
    nonparallel = np.abs(denominator) > 1e-10
    safe_denominator = np.where(nonparallel, denominator, 1.0)
    ray_offset = _cross_2d(delta, segments[None, :, :]) / safe_denominator
    segment_fraction = _cross_2d(delta, normals64[:, None, :]) / safe_denominator
    intersects = (
        nonparallel
        & (segment_fraction >= -1e-7)
        & (segment_fraction <= 1.0 + 1e-7)
        & (np.abs(ray_offset) <= float(radius) + 1e-7)
    )
    distance = np.where(intersects, np.abs(ray_offset), np.inf)
    best_segment = np.argmin(distance, axis=1)
    row = np.arange(len(points64))
    best_distance = distance[row, best_segment]
    reliable = np.isfinite(best_distance)
    offsets = ray_offset[row, best_segment]
    if (width is None) != (height is None):
        raise ValueError("vector target bounds require both width and height")
    if width is not None and height is not None:
        intersection = points64 + normals64 * offsets[:, None]
        reliable &= (
            (intersection[:, 0] >= 0.0)
            & (intersection[:, 0] <= float(width - 1))
            & (intersection[:, 1] >= 0.0)
            & (intersection[:, 1] <= float(height - 1))
        )
    offsets = np.where(reliable, offsets, 0.0)
    return (
        np.ascontiguousarray(offsets, dtype=np.float32),
        np.ascontiguousarray(reliable, dtype=np.float32),
    )


def vector_corner_targets(
    boundary: VectorBoundary,
    boundary_points: np.ndarray,
    *,
    sigma_px: float = 2.0,
) -> np.ndarray:
    if not len(boundary.corners):
        return np.zeros(len(boundary_points), dtype=np.float32)
    points = np.asarray(boundary_points, dtype=np.float64).reshape(-1, 2)
    squared_distance = np.sum((points[:, None, :] - boundary.corners[None, :, :]) ** 2, axis=2)
    nearest = squared_distance.min(axis=1)
    return np.asarray(np.exp(-nearest / (2.0 * sigma_px**2)), dtype=np.float32)


def estimate_target_rgb(rgb: np.ndarray, truth: np.ndarray) -> np.ndarray:
    interior = cv2.erode(truth.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1) > 0
    if int(interior.sum()) < 64:
        interior = np.asarray(truth, dtype=bool)
    if not interior.any():
        raise ValueError("cannot estimate target color from an empty truth geometry")
    return np.asarray(np.median(rgb[interior], axis=0), dtype=np.float32)


class EdgeStripDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        samples: Sequence,
        *,
        chunk_length: int,
        strip_radius: int,
        strip_step: float,
        coarse_model: Path | None,
        augment: bool,
    ) -> None:
        self.root = root
        self.samples = list(samples)
        self.chunk_length = int(chunk_length)
        self.strip_radius = int(strip_radius)
        self.strip_step = float(strip_step)
        self.coarse_model = coarse_model
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(self.root / sample.artifacts.screenshot) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(self.root / sample.artifacts.mask) as image:
            truth = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
        boundary = load_vector_boundary(self.root / sample.artifacts.geojson)
        flip_horizontal = self.augment and random.random() < 0.5
        flip_vertical = self.augment and random.random() < 0.5
        if flip_horizontal:
            rgb = rgb[:, ::-1].copy()
            truth = truth[:, ::-1].copy()
        if flip_vertical:
            rgb = rgb[::-1].copy()
            truth = truth[::-1].copy()
        boundary = transform_vector_boundary(
            boundary,
            width=rgb.shape[1],
            height=rgb.shape[0],
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )
        target_rgb = estimate_target_rgb(rgb, truth)
        if self.coarse_model is None:
            coarse = simulated_coarse_probability(truth, augment=self.augment)
        else:
            coarse = frozen_selector_probability(rgb, truth, self.coarse_model)
        contour = choose_coarse_contour(coarse >= 0.45, index=index, augment=self.augment)
        points = resample_closed_contour(contour, step_px=2.0)
        if len(points) < 8:
            raise ValueError(f"coarse contour too short for {sample.sample_id}")
        tangents, normals = contour_frames(points)
        start = random.randrange(len(points)) if self.augment else (index * 104729) % len(points)
        take = (start + np.arange(self.chunk_length)) % len(points)
        points = points[take]
        tangents = tangents[take]
        normals = normals[take]
        # The sign channel describes the coarse contour's orientation, not the
        # augmented profile center.  Keep it anchored to the same unshifted
        # contour centers used by production; otherwise large negative shifts
        # can put both probes outside the selected region and reverse the
        # global fallback sign.
        inside_direction_reference_points = points.copy()
        points, _profile_center_shift_px = augment_profile_centers(
            points,
            normals,
            augment=self.augment,
        )
        features = strip_features(
            rgb,
            coarse,
            points,
            tangents,
            normals,
            radius=self.strip_radius,
            step=self.strip_step,
            target_rgb=target_rgb,
            inside_direction_reference_points=inside_direction_reference_points,
        )
        target_offsets, reliability = vector_boundary_targets(
            boundary,
            points,
            normals,
            radius=self.strip_radius,
            width=rgb.shape[1],
            height=rgb.shape[0],
        )
        corners = vector_corner_targets(
            boundary,
            points + normals * target_offsets[:, None],
        )
        target_index = (target_offsets + float(self.strip_radius)) / self.strip_step
        return (
            np.ascontiguousarray(features, dtype=np.float32),
            np.ascontiguousarray(target_index, dtype=np.float32),
            np.ascontiguousarray(corners, dtype=np.float32),
            np.ascontiguousarray(reliability, dtype=np.float32),
        )


def simulated_coarse_probability(mask: np.ndarray, *, augment: bool) -> np.ndarray:
    height, width = mask.shape
    if augment:
        short_side = (
            random.choice((72, 80, 96, 112, 128))
            if random.random() < 0.20
            else random.choice((160, 192, 224, 256, 320, 320))
        )
    else:
        short_side = 128
    scale = short_side / float(min(height, width))
    low_width = max(16, round(width * scale))
    low_height = max(16, round(height * scale))
    low = cv2.resize(mask.astype(np.float32), (low_width, low_height), interpolation=cv2.INTER_AREA)
    if augment:
        if random.random() < 0.35:
            low = np.roll(low, (random.randint(-1, 1), random.randint(-1, 1)), axis=(0, 1))
        if random.random() < 0.22:
            kernel = np.ones((3, 3), np.uint8)
            binary = (low >= random.uniform(0.35, 0.60)).astype(np.uint8)
            low = (
                cv2.dilate(binary, kernel, iterations=1)
                if random.random() < 0.5
                else cv2.erode(binary, kernel, iterations=1)
            ).astype(np.float32)
    coarse = cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR)
    sigma = random.uniform(0.25, 1.8) if augment else 0.7
    coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=sigma)
    if augment:
        coarse += np.random.normal(0.0, random.uniform(0.0, 0.025), coarse.shape).astype(np.float32)
    return np.clip(coarse, 0.0, 1.0).astype(np.float32)


def frozen_selector_probability(rgb: np.ndarray, truth: np.ndarray, model_path: Path) -> np.ndarray:
    from map_boundary_builder.model_extract import (
        ModelExtractionConfig,
        predict_mask_probabilities,
    )

    ys, xs = np.where(truth)
    hints: dict[str, object] = {}
    if len(xs):
        center_x = float(np.median(xs))
        center_y = float(np.median(ys))
        closest = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
        hints["seed_point"] = (float(xs[closest]), float(ys[closest]))
        target = estimate_target_rgb(rgb, truth)
        hints["target_rgb"] = tuple(int(round(value)) for value in target)
    session = _cached_selector_session(str(model_path.resolve()))
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) != 4:
        raise ValueError(f"frozen selector must accept NCHW input, got {input_shape}")
    input_channels = int(input_shape[1]) if isinstance(input_shape[1], int) else 5
    input_height = int(input_shape[2]) if isinstance(input_shape[2], int) else 320
    input_width = int(input_shape[3]) if isinstance(input_shape[3], int) else 320
    return predict_mask_probabilities(
        rgb,
        session,
        config=ModelExtractionConfig(
            input_width=input_width,
            input_height=input_height,
            threshold=0.45,
            output_activation="logits",
            input_channels=input_channels,
        ),
        hints=hints,
    )


@lru_cache(maxsize=4)
def _cached_selector_session(model_path: str):
    from map_boundary_builder.model_extract import load_onnx_session

    return load_onnx_session(model_path)


def choose_coarse_contour(mask: np.ndarray, *, index: int, augment: bool) -> np.ndarray:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        raise ValueError("coarse mask has no contour")
    eligible = [
        contour[:, 0, :].astype(np.float32)
        for contour in contours
        if len(contour) >= 8 and cv2.contourArea(contour) >= 8.0
    ]
    if not eligible:
        raise ValueError("coarse mask has no trainable contour")
    if augment and len(eligible) > 1 and random.random() < 0.40:
        return random.choice(eligible)
    if not augment and len(eligible) > 1:
        return eligible[index % len(eligible)]
    return max(eligible, key=lambda contour: cv2.arcLength(contour[:, np.newaxis, :], True))


def resample_closed_contour(contour: np.ndarray, *, step_px: float) -> np.ndarray:
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return points
    closed = np.concatenate([points, points[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    perimeter = float(cumulative[-1])
    count = max(8, int(math.ceil(perimeter / max(0.5, step_px))))
    targets = np.linspace(0.0, perimeter, count, endpoint=False, dtype=np.float32)
    segments = np.minimum(np.searchsorted(cumulative, targets, side="right") - 1, len(points) - 1)
    segment_lengths = np.maximum(lengths[segments], 1e-6)
    fractions = (targets - cumulative[segments]) / segment_lengths
    return closed[segments] + (closed[segments + 1] - closed[segments]) * fractions[:, None]


def contour_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.roll(points, -2, axis=0) - np.roll(points, 2, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return tangent.astype(np.float32), normal.astype(np.float32)


def augment_profile_centers(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    augment: bool,
    max_abs_px: float = PROFILE_CENTER_SHIFT_MAX_ABS_PX,
) -> tuple[np.ndarray, float]:
    """Apply one shared normal offset to every profile in a training sample."""

    if max_abs_px < 0.0:
        raise ValueError("profile center shift bound must be non-negative")
    shift_px = random.uniform(-float(max_abs_px), float(max_abs_px)) if augment else 0.0
    shifted = np.asarray(points, dtype=np.float32) + np.asarray(normals, dtype=np.float32) * shift_px
    return np.ascontiguousarray(shifted, dtype=np.float32), float(shift_px)


def remap_profile(values: np.ndarray, points: np.ndarray, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    map_x, map_y = profile_coordinates(points, normals, offsets)
    return cv2.remap(
        values,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def profile_coordinates(
    points: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    map_x = points[:, 0:1] + normals[:, 0:1] * offsets[None, :]
    map_y = points[:, 1:2] + normals[:, 1:2] * offsets[None, :]
    return np.asarray(map_x, dtype=np.float32), np.asarray(map_y, dtype=np.float32)


def valid_profile(
    points: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    map_x, map_y = profile_coordinates(points, normals, offsets)
    return np.asarray(
        (map_x >= 0.0)
        & (map_x <= float(width - 1))
        & (map_y >= 0.0)
        & (map_y <= float(height - 1)),
        dtype=np.float32,
    )


def inside_direction_sign_from_coarse_profiles(
    coarse: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Infer the selected-region direction without exposing raw probabilities."""

    probabilities = np.asarray(coarse, dtype=np.float32)
    native_offsets = np.asarray(offsets, dtype=np.float32).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(native_offsets):
        raise ValueError("coarse profiles must have shape [sample, offset]")
    center = int(np.argmin(np.abs(native_offsets)))
    step = max(1e-6, float(np.median(np.diff(native_offsets))))
    probe = max(1, int(round(3.0 / step)))
    negative = probabilities[:, max(0, center - probe)]
    positive = probabilities[:, min(len(native_offsets) - 1, center + probe)]
    delta = positive - negative
    global_sign = 1.0 if float(np.mean(delta)) > 0.0 else -1.0
    signs = np.where(delta > 0.0, 1.0, -1.0)
    signs = np.where(np.abs(delta) >= 0.025, signs, global_sign)
    return np.ascontiguousarray(signs, dtype=np.float32)


def strip_features(
    rgb: np.ndarray,
    coarse: np.ndarray,
    points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    *,
    radius: int,
    step: float,
    target_rgb: np.ndarray,
    inside_direction_reference_points: np.ndarray | None = None,
) -> np.ndarray:
    offsets = np.arange(-radius, radius + (step * 0.5), step, dtype=np.float32)
    rgb_float = rgb.astype(np.float32) / 255.0
    sampled_rgb = remap_profile(rgb_float, points, normals, offsets)
    sign_reference_points = (
        points
        if inside_direction_reference_points is None
        else np.asarray(inside_direction_reference_points, dtype=np.float32)
    )
    if sign_reference_points.shape != np.asarray(points).shape:
        raise ValueError(
            "inside-direction reference points must match sampled profile centers"
        )
    sampled_coarse_for_sign = remap_profile(
        coarse.astype(np.float32),
        sign_reference_points,
        normals,
        offsets,
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    scale = max(1e-6, float(np.quantile(np.hypot(grad_x, grad_y), 0.99)))
    grad_x /= scale
    grad_y /= scale
    magnitude = np.clip(np.hypot(grad_x, grad_y), 0.0, 1.0)
    sampled_luminance = remap_profile(gray, points, normals, offsets)
    sampled_magnitude = remap_profile(magnitude, points, normals, offsets)
    sampled_gx = remap_profile(grad_x, points, normals, offsets)
    sampled_gy = remap_profile(grad_y, points, normals, offsets)
    normal_gradient = sampled_gx * normals[:, 0:1] + sampled_gy * normals[:, 1:2]
    target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    target_similarity = np.clip(
        1.0 - np.linalg.norm(sampled_rgb - target, axis=2) / math.sqrt(3.0),
        0.0,
        1.0,
    )
    source_valid = valid_profile(
        points,
        normals,
        offsets,
        width=rgb.shape[1],
        height=rgb.shape[0],
    )
    normalized_offset = np.broadcast_to(
        offsets[None, :] / float(radius),
        sampled_coarse_for_sign.shape,
    )
    inside_direction_sign = np.broadcast_to(
        inside_direction_sign_from_coarse_profiles(
            sampled_coarse_for_sign,
            offsets,
        )[:, None],
        sampled_coarse_for_sign.shape,
    )
    features = np.concatenate(
        [
            sampled_rgb.transpose(2, 0, 1),
            sampled_luminance[np.newaxis],
            sampled_magnitude[np.newaxis],
            np.clip(normal_gradient, -1.0, 1.0)[np.newaxis],
            target_similarity[np.newaxis],
            source_valid[np.newaxis],
            normalized_offset[np.newaxis],
            inside_direction_sign[np.newaxis],
        ],
        axis=0,
    )
    return features.astype(np.float32)


class EdgeStripNet(nn.Module):
    def __init__(self, *, channels: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNEL_COUNT, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        # A rendered 3 px miter stroke commonly spans 3.5--3.9 px after
        # rasterization/obliquity.  Three undilated kernel-5 blocks see only a
        # 3 px radius and can therefore mistake one stroke side for the vector
        # center.  Dilations 1/2/4 cover a 7 px radius with identical parameter
        # count and nearly identical FLOPs.
        self.profile = nn.Sequential(
            *(ProfileResidual(channels, dilation=dilation) for dilation in PROFILE_DILATIONS)
        )
        self.offset_head = nn.Conv2d(channels, 1, 1)
        self.summary_project = nn.Conv1d(channels * 2, channels, 1, bias=False)
        self.sequence = nn.Sequential(
            SequenceResidual(channels, dilation=1),
            SequenceResidual(channels, dilation=2),
            SequenceResidual(channels, dilation=4),
            SequenceResidual(channels, dilation=8),
        )
        self.context_project = nn.Conv1d(channels, channels, 1, bias=False)
        self.corner_head = nn.Conv1d(channels, 1, 1)
        self.reliability_head = nn.Conv1d(channels, 1, 1)

    def forward(self, values):
        features = self.profile(self.stem(values))
        valid = values[
            :, SOURCE_VALID_CHANNEL_INDEX : SOURCE_VALID_CHANNEL_INDEX + 1
        ].clamp(0.0, 1.0)
        valid_count = valid.sum(dim=-1).clamp_min(1.0)
        profile_mean = (features * valid).sum(dim=-1) / valid_count
        profile_maximum = (features + (valid - 1.0) * 1.0e4).amax(dim=-1)
        sequence = self.summary_project(torch.cat([profile_mean, profile_maximum], dim=1))
        sequence = self.sequence(sequence)
        contextual = F.silu(features + self.context_project(sequence).unsqueeze(-1))
        offset_logits = self.offset_head(contextual).squeeze(1)
        offset_logits = offset_logits + (valid.squeeze(1) - 1.0) * 1.0e4
        corner_logits = self.corner_head(sequence).squeeze(1)
        reliability_logits = self.reliability_head(sequence).squeeze(1)
        return offset_logits, corner_logits, reliability_logits


class ProfileResidual(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("profile dilation must be positive")
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                (1, 5),
                padding=(0, 2 * dilation),
                dilation=(1, dilation),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, values):
        return F.silu(values + self.block(values))


class SequenceResidual(nn.Module):
    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, values):
        return F.silu(values + self.block(values))


def edgegraph_loss(outputs, offset_index, corners, reliability):
    offset_logits, corner_logits, reliability_logits = outputs
    bins = torch.arange(offset_logits.shape[-1], device=offset_logits.device, dtype=offset_logits.dtype)
    target_distribution = torch.exp(-0.5 * ((bins[None, None, :] - offset_index[:, :, None]) / 0.55) ** 2)
    target_distribution /= target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    visible_weight = reliability.clamp(0.0, 1.0)
    distribution_per_sample = -(target_distribution * F.log_softmax(offset_logits, dim=-1)).sum(dim=-1)
    distribution = _weighted_mean(distribution_per_sample, visible_weight)
    predicted_index = (torch.softmax(offset_logits, dim=-1) * bins[None, None, :]).sum(dim=-1)
    subpixel = _weighted_mean(
        F.smooth_l1_loss(predicted_index, offset_index, reduction="none"),
        visible_weight,
    )
    corner_targets = (corners >= 0.20).float()
    corner_bce = F.binary_cross_entropy_with_logits(corner_logits, corner_targets, reduction="none")
    corner_probability = torch.sigmoid(corner_logits)
    focal_weight = torch.where(corner_targets > 0, (1.0 - corner_probability) ** 2, corner_probability**2)
    corner = _weighted_mean(
        corner_bce * focal_weight * torch.where(corner_targets > 0, 14.0, 1.0),
        visible_weight,
    )
    reliability_loss = _balanced_binary_cross_entropy(reliability_logits, reliability)
    predicted_second_difference = predicted_index[:, 2:] - 2.0 * predicted_index[:, 1:-1] + predicted_index[:, :-2]
    target_second_difference = offset_index[:, 2:] - 2.0 * offset_index[:, 1:-1] + offset_index[:, :-2]
    smoothness_weight = (
        visible_weight[:, :-2]
        * visible_weight[:, 1:-1]
        * visible_weight[:, 2:]
        * (1.0 - corners[:, 1:-1]).clamp(0.0, 1.0)
    )
    smoothness = _weighted_mean(
        F.smooth_l1_loss(
            predicted_second_difference,
            target_second_difference,
            reduction="none",
        ),
        smoothness_weight,
    )
    return (
        0.50 * distribution
        + 0.25 * subpixel
        + 0.12 * corner
        + 0.08 * reliability_loss
        + 0.05 * smoothness
    )


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _balanced_binary_cross_entropy(logits, targets):
    targets = targets.clamp(0.0, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets.sum()
    negative = (1.0 - targets).sum()
    total = targets.numel()
    positive_weight = float(total) / (2.0 * positive.clamp_min(1.0))
    negative_weight = float(total) / (2.0 * negative.clamp_min(1.0))
    weights = targets * positive_weight + (1.0 - targets) * negative_weight
    return (loss * weights).mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    offset_step: float = 0.5,
) -> dict[str, float]:
    model.eval()
    absolute_errors: list[np.ndarray] = []
    corner_tp = corner_fp = corner_fn = 0
    reliability_correct = reliability_total = 0
    reliability_tp = reliability_tn = reliability_fp = reliability_fn = 0
    for features, offset_index, corners, reliability in loader:
        features = features.to(device)
        offset_index = offset_index.to(device)
        corners = corners.to(device)
        reliability = reliability.to(device)
        offset_logits, corner_logits, reliability_logits = model(features)
        bins = torch.arange(offset_logits.shape[-1], device=device, dtype=offset_logits.dtype)
        predicted_index = (torch.softmax(offset_logits, dim=-1) * bins[None, None, :]).sum(dim=-1)
        error = torch.abs(predicted_index - offset_index) * float(offset_step)
        visible = reliability >= 0.5
        absolute_errors.append(error[visible].detach().cpu().numpy())
        predicted_corner = (torch.sigmoid(corner_logits) >= 0.45) & visible
        target_corner = (corners >= 0.20) & visible
        dilated_predicted = F.max_pool1d(predicted_corner.float(), 5, stride=1, padding=2) > 0
        dilated_target = F.max_pool1d(target_corner.float(), 5, stride=1, padding=2) > 0
        corner_tp += int((predicted_corner & dilated_target).sum())
        corner_fp += int((predicted_corner & ~dilated_target).sum())
        corner_fn += int((target_corner & ~dilated_predicted).sum())
        predicted_reliability = torch.sigmoid(reliability_logits) >= 0.5
        reliability_correct += int((predicted_reliability == visible).sum())
        reliability_total += int(visible.numel())
        reliability_tp += int((predicted_reliability & visible).sum())
        reliability_tn += int((~predicted_reliability & ~visible).sum())
        reliability_fp += int((predicted_reliability & ~visible).sum())
        reliability_fn += int((~predicted_reliability & visible).sum())
    errors = np.concatenate(absolute_errors) if absolute_errors else np.asarray([float("inf")])
    precision = corner_tp / max(1, corner_tp + corner_fp)
    recall = corner_tp / max(1, corner_tp + corner_fn)
    reliability_tpr = reliability_tp / max(1, reliability_tp + reliability_fn)
    reliability_tnr = reliability_tn / max(1, reliability_tn + reliability_fp)
    return {
        "offset_mae_px": float(np.mean(errors)),
        "offset_p95_px": float(np.quantile(errors, 0.95)),
        "offset_p99_px": float(np.quantile(errors, 0.99)),
        "within_0_25px": float((errors <= 0.25).mean()),
        "within_0_5px": float((errors <= 0.5).mean()),
        "within_1px": float((errors <= 1.0).mean()),
        "corner_precision": precision,
        "corner_recall": recall,
        "corner_f1": 2.0 * precision * recall / max(1e-9, precision + recall),
        "reliability_accuracy": reliability_correct / max(1, reliability_total),
        "reliability_balanced_accuracy": 0.5 * (reliability_tpr + reliability_tnr),
    }


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def export_model(
    model: nn.Module,
    output: Path,
    *,
    chunk_length: int,
    strip_radius: int,
    strip_step: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    strip_width = round(strip_radius * 2 / strip_step) + 1
    example = torch.zeros((1, INPUT_CHANNEL_COUNT, chunk_length, strip_width), dtype=torch.float32)
    example[:, SOURCE_VALID_CHANNEL_INDEX] = 1.0
    torch.onnx.export(
        model,
        example,
        output,
        input_names=[ONNX_INPUT_NAME],
        output_names=["offset_logits", "corner_logits", "reliability_logits"],
        dynamic_axes={
            ONNX_INPUT_NAME: {0: "batch", 2: "contour_length"},
            "offset_logits": {0: "batch", 1: "contour_length"},
            "corner_logits": {0: "batch", 1: "contour_length"},
            "reliability_logits": {0: "batch", 1: "contour_length"},
        },
        opset_version=18,
        dynamo=False,
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
