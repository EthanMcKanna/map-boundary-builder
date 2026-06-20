"""Synthetic dataset manifest helpers."""

from .generator import (
    DEFAULT_OVERLAY_STYLES,
    GENERATOR_VERSION,
    SyntheticOverlayStyle,
    SyntheticRenderResult,
    SyntheticSceneConfig,
    generate_synthetic_dataset,
    generate_synthetic_sample,
)
from .manifest import (
    OverlayStyleMetadata,
    SyntheticArtifactPaths,
    SyntheticDatasetManifest,
    SyntheticSampleMetadata,
    deterministic_content_hash,
    deterministic_sample_id,
)

__all__ = [
    "DEFAULT_OVERLAY_STYLES",
    "GENERATOR_VERSION",
    "OverlayStyleMetadata",
    "SyntheticArtifactPaths",
    "SyntheticDatasetManifest",
    "SyntheticOverlayStyle",
    "SyntheticRenderResult",
    "SyntheticSampleMetadata",
    "SyntheticSceneConfig",
    "deterministic_content_hash",
    "deterministic_sample_id",
    "generate_synthetic_dataset",
    "generate_synthetic_sample",
]
