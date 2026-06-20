import numpy as np

from map_boundary_builder.model_extract import (
    ModelExtractionConfig,
    extract_service_area_from_rgb_with_session,
    normalize_model_output,
    predict_mask_probabilities,
    preprocess_rgb_for_model,
)


class FakeInput:
    name = "image"


class FakeSession:
    def __init__(self, output: np.ndarray):
        self.output = output
        self.last_feed = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _output_names, input_feed):
        self.last_feed = input_feed
        return [self.output]


def test_preprocess_rgb_for_model_returns_nchw_float_tensor() -> None:
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255

    tensor = preprocess_rgb_for_model(
        rgb,
        config=ModelExtractionConfig(input_width=16, input_height=12),
    )

    assert tensor.shape == (1, 3, 12, 16)
    assert tensor.dtype == np.float32
    assert float(tensor[:, 0].max()) == 1.0


def test_normalize_model_output_accepts_common_single_channel_shapes() -> None:
    two_dimensional = np.ones((8, 9), dtype=np.float32) * 0.25
    nchw = two_dimensional[np.newaxis, np.newaxis, :, :]
    nhwc = two_dimensional[np.newaxis, :, :, np.newaxis]

    np.testing.assert_allclose(normalize_model_output(two_dimensional), two_dimensional)
    np.testing.assert_allclose(normalize_model_output(nchw), two_dimensional)
    np.testing.assert_allclose(normalize_model_output(nhwc), two_dimensional)


def test_normalize_model_output_applies_sigmoid_to_logits() -> None:
    logits = np.array([[-20.0, 0.0, 20.0]], dtype=np.float32)

    probabilities = normalize_model_output(logits, output_activation="logits")

    assert probabilities[0, 0] < 0.001
    assert probabilities[0, 1] == 0.5
    assert probabilities[0, 2] > 0.999


def test_normalize_model_output_rejects_out_of_range_probabilities() -> None:
    probabilities = np.array([[-0.1, 0.5, 1.1]], dtype=np.float32)

    try:
        normalize_model_output(probabilities, output_activation="probability")
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("Expected out-of-range probabilities to raise ValueError")


def test_predict_mask_probabilities_resizes_to_source_shape() -> None:
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    output = np.zeros((1, 1, 10, 10), dtype=np.float32)
    output[:, :, 2:8, 3:9] = 0.9
    session = FakeSession(output)

    probabilities = predict_mask_probabilities(
        rgb,
        session,
        config=ModelExtractionConfig(input_width=10, input_height=10),
    )

    assert probabilities.shape == (40, 60)
    assert session.last_feed["image"].shape == (1, 3, 10, 10)
    assert probabilities.max() > 0.8


def test_extract_service_area_from_rgb_with_session_returns_extraction_result() -> None:
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    output = np.zeros((1, 1, 40, 50), dtype=np.float32)
    output[:, :, 8:32, 10:38] = 0.95
    session = FakeSession(output)

    result = extract_service_area_from_rgb_with_session(
        rgb,
        session,
        config=ModelExtractionConfig(input_width=50, input_height=40, threshold=0.5, simplify_px=1.0),
    )

    assert result.style == "model-mask"
    assert result.mask.shape == (80, 100)
    assert result.coverage_ratio > 0.1
    assert result.contour_count == 1
    assert result.diagnostics["model_threshold"] == 0.5
