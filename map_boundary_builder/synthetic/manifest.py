"""Small JSON manifest primitives for synthetic boundary samples."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


JsonObject = dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def deterministic_content_hash(value: Any) -> str:
    """Return a stable sha256 digest for JSON-serializable content."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _slugify(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return slug.strip("-")


def deterministic_sample_id(*parts: object, prefix: str = "synthetic", digest_chars: int = 12) -> str:
    """Build a readable, deterministic sample id from stable input parts."""

    text_parts = [str(part) for part in parts if part is not None and str(part) != ""]
    slug_parts = [_slugify(part) for part in text_parts[:3]]
    slug = "-".join(part for part in slug_parts if part) or "sample"
    slug = slug[:48].rstrip("-")
    digest = deterministic_content_hash(text_parts)[:digest_chars]
    return f"{_slugify(prefix) or 'synthetic'}-{slug}-{digest}"


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"missing required object: {key}")
    return item


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"missing required string: {key}")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise ValueError(f"expected string for: {key}")
    return item


def _required_float(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)):
        raise ValueError(f"missing required number: {key}")
    return float(item)


def _json_object(value: Mapping[str, Any] | None) -> JsonObject:
    return dict(value or {})


@dataclass
class OverlayStyleMetadata:
    name: str
    fill_color: str
    fill_opacity: float
    stroke_color: str | None = None
    stroke_width_px: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("overlay style name is required")
        if not self.fill_color:
            raise ValueError("overlay fill_color is required")
        if not 0.0 <= float(self.fill_opacity) <= 1.0:
            raise ValueError("overlay fill_opacity must be between 0 and 1")
        if self.stroke_width_px is not None and float(self.stroke_width_px) < 0:
            raise ValueError("overlay stroke_width_px must be non-negative")

    def to_dict(self) -> JsonObject:
        data: JsonObject = {
            "name": self.name,
            "fill_color": self.fill_color,
            "fill_opacity": float(self.fill_opacity),
        }
        if self.stroke_color is not None:
            data["stroke_color"] = self.stroke_color
        if self.stroke_width_px is not None:
            data["stroke_width_px"] = float(self.stroke_width_px)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OverlayStyleMetadata:
        return cls(
            name=_required_string(data, "name"),
            fill_color=_required_string(data, "fill_color"),
            fill_opacity=_required_float(data, "fill_opacity"),
            stroke_color=_optional_string(data, "stroke_color"),
            stroke_width_px=None if data.get("stroke_width_px") is None else float(data["stroke_width_px"]),
        )


@dataclass
class SyntheticArtifactPaths:
    screenshot: str
    overlay: str
    mask: str
    geojson: str
    metadata: str | None = None
    extras: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, path in self.required_artifacts().items():
            if not path:
                raise ValueError(f"artifact path is required: {name}")

    def required_artifacts(self) -> dict[str, str]:
        required = {
            "screenshot": self.screenshot,
            "overlay": self.overlay,
            "mask": self.mask,
            "geojson": self.geojson,
        }
        if self.metadata is not None:
            required["metadata"] = self.metadata
        return required

    def to_dict(self) -> JsonObject:
        data: JsonObject = {
            "screenshot": self.screenshot,
            "overlay": self.overlay,
            "mask": self.mask,
            "geojson": self.geojson,
        }
        if self.metadata is not None:
            data["metadata"] = self.metadata
        if self.extras:
            data["extras"] = dict(sorted(self.extras.items()))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SyntheticArtifactPaths:
        extras = data.get("extras") or {}
        if not isinstance(extras, Mapping):
            raise ValueError("artifact extras must be an object")
        return cls(
            screenshot=_required_string(data, "screenshot"),
            overlay=_required_string(data, "overlay"),
            mask=_required_string(data, "mask"),
            geojson=_required_string(data, "geojson"),
            metadata=_optional_string(data, "metadata"),
            extras={str(key): str(value) for key, value in extras.items()},
        )

    def missing_required_artifacts(self, base_dir: str | Path = ".") -> list[str]:
        root = Path(base_dir)
        missing: list[str] = []
        for path in self.required_artifacts().values():
            artifact_path = Path(path)
            resolved = artifact_path if artifact_path.is_absolute() else root / artifact_path
            if not resolved.exists():
                missing.append(path)
        return missing

    def validate_required_artifacts(self, base_dir: str | Path = ".") -> None:
        missing = self.missing_required_artifacts(base_dir)
        if missing:
            raise FileNotFoundError("missing required synthetic artifacts: " + ", ".join(missing))


@dataclass
class SyntheticSampleMetadata:
    sample_id: str
    content_hash: str
    provider: str
    service_area: str
    variant: str
    image_size: tuple[int, int]
    overlay_style: OverlayStyleMetadata
    artifacts: SyntheticArtifactPaths
    base_map: str | None = None
    seed: int | str | None = None
    generator_version: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if not self.content_hash:
            raise ValueError("content_hash is required")
        if len(self.image_size) != 2 or self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be a positive (width, height) pair")

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        service_area: str,
        variant: str,
        image_size: tuple[int, int],
        overlay_style: OverlayStyleMetadata,
        artifacts: SyntheticArtifactPaths,
        base_map: str | None = None,
        seed: int | str | None = None,
        generator_version: str | None = None,
        properties: Mapping[str, Any] | None = None,
    ) -> SyntheticSampleMetadata:
        content = cls._content_payload(
            provider=provider,
            service_area=service_area,
            variant=variant,
            image_size=image_size,
            overlay_style=overlay_style,
            artifacts=artifacts,
            base_map=base_map,
            seed=seed,
            generator_version=generator_version,
            properties=properties,
        )
        content_hash = deterministic_content_hash(content)
        sample_id = deterministic_sample_id(provider, service_area, variant, seed, content_hash[:16])
        return cls(
            sample_id=sample_id,
            content_hash=content_hash,
            provider=provider,
            service_area=service_area,
            variant=variant,
            image_size=image_size,
            overlay_style=overlay_style,
            artifacts=artifacts,
            base_map=base_map,
            seed=seed,
            generator_version=generator_version,
            properties=_json_object(properties),
        )

    @staticmethod
    def _content_payload(
        *,
        provider: str,
        service_area: str,
        variant: str,
        image_size: tuple[int, int],
        overlay_style: OverlayStyleMetadata,
        artifacts: SyntheticArtifactPaths,
        base_map: str | None,
        seed: int | str | None,
        generator_version: str | None,
        properties: Mapping[str, Any] | None,
    ) -> JsonObject:
        data: JsonObject = {
            "provider": provider,
            "service_area": service_area,
            "variant": variant,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "overlay_style": overlay_style.to_dict(),
            "artifacts": artifacts.to_dict(),
            "properties": _json_object(properties),
        }
        if base_map is not None:
            data["base_map"] = base_map
        if seed is not None:
            data["seed"] = seed
        if generator_version is not None:
            data["generator_version"] = generator_version
        return data

    def content_payload(self) -> JsonObject:
        return self._content_payload(
            provider=self.provider,
            service_area=self.service_area,
            variant=self.variant,
            image_size=self.image_size,
            overlay_style=self.overlay_style,
            artifacts=self.artifacts,
            base_map=self.base_map,
            seed=self.seed,
            generator_version=self.generator_version,
            properties=self.properties,
        )

    def recompute_content_hash(self) -> str:
        return deterministic_content_hash(self.content_payload())

    def to_dict(self) -> JsonObject:
        data = self.content_payload()
        return {"sample_id": self.sample_id, "content_hash": self.content_hash, **data}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SyntheticSampleMetadata:
        image_size = data.get("image_size")
        if not isinstance(image_size, Sequence) or isinstance(image_size, (str, bytes)) or len(image_size) != 2:
            raise ValueError("image_size must be a two-item sequence")
        return cls(
            sample_id=_required_string(data, "sample_id"),
            content_hash=_required_string(data, "content_hash"),
            provider=_required_string(data, "provider"),
            service_area=_required_string(data, "service_area"),
            variant=_required_string(data, "variant"),
            image_size=(int(image_size[0]), int(image_size[1])),
            overlay_style=OverlayStyleMetadata.from_dict(_required_mapping(data, "overlay_style")),
            artifacts=SyntheticArtifactPaths.from_dict(_required_mapping(data, "artifacts")),
            base_map=_optional_string(data, "base_map"),
            seed=data.get("seed"),
            generator_version=_optional_string(data, "generator_version"),
            properties=_json_object(data.get("properties") if isinstance(data.get("properties"), Mapping) else None),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes) -> SyntheticSampleMetadata:
        return cls.from_dict(json.loads(data))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> SyntheticSampleMetadata:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def validate_required_artifacts(self, base_dir: str | Path = ".") -> None:
        self.artifacts.validate_required_artifacts(base_dir)


@dataclass
class SyntheticDatasetManifest:
    name: str
    version: str
    samples: Sequence[SyntheticSampleMetadata]
    properties: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "version": self.version,
            "samples": [sample.to_dict() for sample in self.samples],
            "properties": _json_object(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SyntheticDatasetManifest:
        samples = data.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise ValueError("samples must be a sequence")
        return cls(
            name=_required_string(data, "name"),
            version=_required_string(data, "version"),
            samples=[SyntheticSampleMetadata.from_dict(sample) for sample in samples],
            properties=_json_object(data.get("properties") if isinstance(data.get("properties"), Mapping) else None),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes) -> SyntheticDatasetManifest:
        return cls.from_dict(json.loads(data))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> SyntheticDatasetManifest:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def validate_required_artifacts(self, base_dir: str | Path = ".") -> None:
        missing_by_sample: list[str] = []
        for sample in self.samples:
            missing = sample.artifacts.missing_required_artifacts(base_dir)
            if missing:
                missing_by_sample.append(f"{sample.sample_id}: {', '.join(missing)}")
        if missing_by_sample:
            raise FileNotFoundError("missing required synthetic artifacts: " + "; ".join(missing_by_sample))
