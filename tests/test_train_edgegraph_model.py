import json
import warnings
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from tools.train_edgegraph_model import (
    EdgeStripDataset,
    EdgeStripNet,
    INSIDE_DIRECTION_REFERENCE_VERSION,
    INPUT_CHANNEL_NAMES,
    ONNX_INPUT_NAME,
    PROFILE_DILATIONS,
    PROFILE_CENTER_SHIFT_MAX_ABS_PX,
    PROFILE_RECEPTIVE_FIELD_BINS,
    SOURCE_VALID_CHANNEL_INDEX,
    augment_profile_centers,
    build_parser,
    export_model,
    strip_features,
    vector_boundary_from_geojson,
    vector_boundary_targets,
    vector_corner_targets,
    write_model_metadata,
)


def _rectangle_geojson(
    *,
    left: float = 10.25,
    top: float = 10.5,
    right: float = 30.75,
    bottom: float = 25.5,
) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "metadata": {
            "pixel_geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [left, top],
                        [right, top],
                        [right, bottom],
                        [left, bottom],
                        [left, top],
                    ]
                ],
            }
        },
    }


def test_vector_targets_use_continuous_geojson_intersections_and_exact_corners() -> None:
    boundary = vector_boundary_from_geojson(_rectangle_geojson())
    points = np.asarray([[12.0, 8.0], [20.0, 8.0], [29.0, 8.0]], dtype=np.float32)
    normals = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (len(points), 1))

    offsets, reliability = vector_boundary_targets(boundary, points, normals, radius=8.0)
    corners = vector_corner_targets(boundary, points + normals * offsets[:, None])

    assert np.allclose(offsets, 2.5, atol=1e-6)
    assert np.array_equal(reliability, np.ones(3, dtype=np.float32))
    assert corners[0] > corners[1]
    assert corners[2] > corners[1]
    assert np.allclose(
        np.sort(boundary.corners, axis=0),
        np.sort(np.asarray([[10.25, 10.5], [30.75, 10.5], [30.75, 25.5], [10.25, 25.5]]), axis=0),
    )


def test_strip_features_use_vector_v3_without_raw_coarse_probability() -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :16] = (20, 100, 220)
    rgb[:, 16:] = (230, 230, 230)
    coarse = np.zeros((24, 32), dtype=np.float32)
    coarse[:, :16] = 1.0
    points = np.asarray([[1.0, 6.0], [15.5, 12.0], [30.0, 18.0]], dtype=np.float32)
    tangents = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (len(points), 1))
    normals = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(points), 1))

    features = strip_features(
        rgb,
        coarse,
        points,
        tangents,
        normals,
        radius=4,
        step=1.0,
        target_rgb=np.asarray([20, 100, 220], dtype=np.float32),
    )

    assert features.shape == (10, 3, 9)
    assert np.isfinite(features).all()
    assert 0.0 <= float(features[6].min()) <= float(features[6].max()) <= 1.0
    assert float(features[6, 1, 0]) > float(features[6, 1, -1])
    assert set(np.unique(features[7])) == {0.0, 1.0}
    assert features[7, 0, 0] == 0.0
    assert features[7, 1].all()
    assert np.allclose(features[8, 0], np.linspace(-1.0, 1.0, 9))
    assert set(np.unique(features[9])) <= {-1.0, 1.0}
    assert np.all(features[9] == features[9, :, :1])
    # No v3 channel reproduces the raw selector profile's soft spatial values.
    assert not any(np.allclose(channel, coarse[6:9, :9]) for channel in features)


def test_profile_center_augmentation_applies_one_global_normal_shift(monkeypatch) -> None:
    points = np.asarray([[2.0, 3.0], [8.0, 5.0]], dtype=np.float32)
    normals = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    monkeypatch.setattr(
        "tools.train_edgegraph_model.random.uniform",
        lambda minimum, maximum: maximum,
    )

    shifted, shift_px = augment_profile_centers(points, normals, augment=True)

    assert shift_px == PROFILE_CENTER_SHIFT_MAX_ABS_PX == 6.0
    assert np.allclose(shifted, points + normals * 6.0)
    unchanged, validation_shift = augment_profile_centers(points, normals, augment=False)
    assert validation_shift == 0.0
    assert np.array_equal(unchanged, points)


def test_shifted_profiles_keep_inside_sign_anchored_to_coarse_contour() -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, 16:] = (20, 100, 220)
    coarse = np.zeros((24, 32), dtype=np.float32)
    coarse[:, 16:] = 1.0
    contour_points = np.asarray(
        [[15.5, 6.0], [15.5, 12.0], [15.5, 18.0]],
        dtype=np.float32,
    )
    tangents = np.tile(
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        (len(contour_points), 1),
    )
    normals = np.tile(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        (len(contour_points), 1),
    )
    shifted_points = contour_points - normals * 6.0

    shifted_features = strip_features(
        rgb,
        coarse,
        shifted_points,
        tangents,
        normals,
        radius=4,
        step=1.0,
        target_rgb=np.asarray([20, 100, 220], dtype=np.float32),
    )
    anchored_features = strip_features(
        rgb,
        coarse,
        shifted_points,
        tangents,
        normals,
        radius=4,
        step=1.0,
        target_rgb=np.asarray([20, 100, 220], dtype=np.float32),
        inside_direction_reference_points=contour_points,
    )

    # At the shifted center both +/-3 px probes are outside, so the historical
    # tie fallback points the wrong way.  The reference-anchored channel keeps
    # the +normal direction that production infers at the coarse contour.
    assert np.all(shifted_features[9] == -1.0)
    assert np.all(anchored_features[9] == 1.0)
    assert anchored_features.shape == shifted_features.shape == (10, 3, 9)


def test_edge_strip_net_has_unbounded_learned_logits_and_masks_invalid_bins() -> None:
    model = EdgeStripNet(channels=8).eval()
    torch.nn.init.zeros_(model.offset_head.weight)
    torch.nn.init.constant_(model.offset_head.bias, 7.0)
    values = torch.zeros((1, 10, 12, 49), dtype=torch.float32)
    values[:, SOURCE_VALID_CHANNEL_INDEX] = 1.0
    values[:, SOURCE_VALID_CHANNEL_INDEX, :, :4] = 0.0

    offset_logits, corner_logits, reliability_logits = model(values)

    assert not hasattr(model, "selector_center_prior")
    assert model.stem[0].in_channels == 10
    assert offset_logits.shape == (1, 12, 49)
    assert torch.allclose(offset_logits[:, :, 4:], torch.full_like(offset_logits[:, :, 4:], 7.0))
    assert float(offset_logits[:, :, :4].max().detach()) < -9_000.0
    assert corner_logits.shape == reliability_logits.shape == (1, 12)


def test_profile_tower_covers_both_sides_of_maximum_centered_stroke() -> None:
    model = EdgeStripNet(channels=8)
    observed = tuple(layer.block[0].dilation[1] for layer in model.profile)

    assert observed == PROFILE_DILATIONS == (1, 2, 4)
    assert PROFILE_RECEPTIVE_FIELD_BINS == 29
    assert (PROFILE_RECEPTIVE_FIELD_BINS - 1) * 0.5 == 14.0


def test_dataset_sample_uses_ten_channels_and_vector_supervision(tmp_path) -> None:
    rgb = np.full((64, 64, 3), 230, dtype=np.uint8)
    truth = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(truth, (15, 12), (49, 51), 255, thickness=-1)
    rgb[truth > 0] = (20, 100, 220)
    Image.fromarray(rgb).save(tmp_path / "image.png")
    Image.fromarray(truth).save(tmp_path / "mask.png")
    (tmp_path / "boundary.geojson").write_text(
        json.dumps(_rectangle_geojson(left=15.25, top=12.5, right=48.75, bottom=50.5)),
        encoding="utf-8",
    )
    sample = SimpleNamespace(
        artifacts=SimpleNamespace(
            screenshot="image.png",
            mask="mask.png",
            geojson="boundary.geojson",
        )
    )
    dataset = EdgeStripDataset(
        tmp_path,
        [sample],
        chunk_length=32,
        strip_radius=12,
        strip_step=0.5,
        coarse_model=None,
        augment=False,
    )

    features, offset_index, corners, reliability = dataset[0]

    assert features.shape == (10, 32, 49)
    assert offset_index.shape == corners.shape == reliability.shape == (32,)
    assert np.isfinite(features).all()
    assert float(reliability.mean()) > 0.95
    assert np.max(np.abs((offset_index - 24.0) * 0.5)) < 3.0


def test_cli_and_metadata_preserve_learned_offsets_by_default(tmp_path) -> None:
    args = build_parser().parse_args([])
    assert args.dataset_dir.as_posix() == "out/synthetic-v20-edgegraph-train"
    assert args.conservative_offset_head is False

    output = tmp_path / "edgegraph.onnx"
    checkpoint = {
        "epoch": 2,
        "channels": 24,
        "hidden_channels": 24,
        "input_channels": 10,
        "input_channel_names": list(INPUT_CHANNEL_NAMES),
        "profile_center_shift_max_abs_px": 6.0,
        "inside_direction_reference_version": INSIDE_DIRECTION_REFERENCE_VERSION,
        "vector_target_version": "geojson-pixel-ray-segment-v1",
        "chunk_length": 256,
        "strip_radius": 12,
        "strip_step": 0.5,
        "metrics": {"offset_p99_px": 0.7},
    }
    write_model_metadata(output, checkpoint, conservative_offset_head=False)
    metadata = json.loads(output.with_suffix(".onnx.json").read_text(encoding="utf-8"))

    assert metadata["architecture"] == "generalized-v20-edgegraph-stripnet-vector-v3"
    assert metadata["channels"] == 10
    assert metadata["input_channels"] == list(INPUT_CHANNEL_NAMES)
    assert metadata["onnx_input_name"] == ONNX_INPUT_NAME
    assert metadata["coarse_probability_input"] is False
    assert metadata["training_augmentation"] == {
        "profile_center_shift_distribution": "uniform",
        "profile_center_shift_max_abs_px": 6.0,
        "profile_center_shift_scope": "global_per_sample",
        "inside_direction_reference": "unshifted-contour-center-v1",
    }
    assert metadata["learned_offset_logits_preserved"] is True
    assert metadata["offset_head"] == "unbounded_learned_logits"
    assert metadata["profile_dilations"] == [1, 2, 4]
    assert metadata["profile_receptive_field_bins"] == 29
    assert metadata["profile_receptive_field_px"] == 14.0
    assert metadata["supervision"]["target_version"] == "geojson-pixel-ray-segment-v1"


def test_onnx_export_accepts_ten_channels_and_dynamic_contour_length(tmp_path) -> None:
    output = tmp_path / "edgegraph.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        export_model(
            EdgeStripNet(channels=8),
            output,
            chunk_length=32,
            strip_radius=12,
            strip_step=0.5,
        )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])

    assert session.get_inputs()[0].name == ONNX_INPUT_NAME
    assert session.get_inputs()[0].shape == ["batch", 10, "contour_length", 49]
    for length in (17, 73):
        values = np.zeros((1, 10, length, 49), dtype=np.float32)
        values[:, SOURCE_VALID_CHANNEL_INDEX] = 1.0
        values[:, SOURCE_VALID_CHANNEL_INDEX, :, :2] = 0.0
        offset, corner, reliability = session.run(None, {ONNX_INPUT_NAME: values})

        assert offset.shape == (1, length, 49)
        assert corner.shape == reliability.shape == (1, length)
        assert float(offset[:, :, :2].max()) < -9_000.0
        assert np.isfinite(offset).all()
