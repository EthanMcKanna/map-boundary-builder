from __future__ import annotations

import gzip
from pathlib import Path
import re
from xml.etree import ElementTree

from PIL import Image


RASTER_IMAGE_EXTENSIONS = {".avif", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SVG_IMAGE_EXTENSIONS = {".svg", ".svgz"}
SUPPORTED_IMAGE_EXTENSIONS = RASTER_IMAGE_EXTENSIONS | SVG_IMAGE_EXTENSIONS


def safe_image_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return ext
    return ".png"


def normalize_image_for_processing(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    composite_transparent_rasters: bool = True,
    svg_max_dimension: int | None = None,
) -> Path:
    image_path = Path(path)
    target_dir = Path(output_dir) if output_dir is not None else image_path.parent
    if is_svg_image(image_path):
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{image_path.stem}.raster.png"
        rasterize_svg_to_png(image_path, target_path, max_dimension=svg_max_dimension or 0)
        return target_path

    if composite_transparent_rasters and raster_has_transparency(image_path):
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{image_path.stem}.opaque.png"
        composite_raster_to_opaque(image_path, target_path)
        return target_path

    return image_path


def is_svg_image(path: str | Path) -> bool:
    image_path = Path(path)
    if image_path.suffix.lower() in SVG_IMAGE_EXTENSIONS:
        return True
    try:
        head = image_path.read_bytes()[:512].lstrip().lower()
    except OSError:
        return False
    return (head.startswith(b"<?xml") and b"<svg" in head[:256]) or head.startswith(b"<svg")


def raster_has_transparency(path: str | Path) -> bool:
    image_path = Path(path)
    if image_path.suffix.lower() not in RASTER_IMAGE_EXTENSIONS:
        return False
    try:
        with Image.open(image_path) as image:
            if image.mode in {"RGBA", "LA"}:
                alpha = image.getchannel("A")
                return alpha.getextrema()[0] < 255
            if image.mode == "P" and "transparency" in image.info:
                return True
    except Exception:
        return False
    return False


def composite_raster_to_opaque(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path).convert("RGBA") as image:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        background.convert("RGB").save(target_path)


def rasterize_svg_to_png(source_path: Path, target_path: Path, *, max_dimension: int = 0) -> None:
    try:
        import cairosvg
    except Exception as exc:  # pragma: no cover - depends on optional native libs.
        raise ValueError(
            "SVG uploads need to be rasterized before extraction. Use the web app upload flow "
            "or install CairoSVG support, then try again."
        ) from exc

    try:
        svg_bytes = read_svg_bytes(source_path)
        kwargs: dict[str, object] = {
            "bytestring": svg_bytes,
            "write_to": str(target_path),
        }
        output_size = capped_svg_output_size(svg_bytes, max_dimension=max_dimension)
        if output_size is not None:
            output_width, output_height = output_size
            kwargs["output_width"] = output_width
            kwargs["output_height"] = output_height
        cairosvg.svg2png(**kwargs)
    except Exception as exc:
        raise ValueError(
            f"Could not rasterize SVG image {source_path.name}. Export it as PNG, JPG, WebP, or TIFF "
            "and try again."
        ) from exc


def read_svg_bytes(source_path: Path) -> bytes:
    data = source_path.read_bytes()
    if source_path.suffix.lower() == ".svgz":
        return gzip.decompress(data)
    return data


def capped_svg_output_size(svg_bytes: bytes, *, max_dimension: int) -> tuple[int, int] | None:
    max_dimension = max(0, int(max_dimension))
    if max_dimension <= 0:
        return None
    intrinsic_size = svg_intrinsic_size(svg_bytes)
    if intrinsic_size is None:
        return None
    width, height = intrinsic_size
    largest = max(width, height)
    if largest <= max_dimension:
        return None
    scale = max_dimension / largest
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def svg_intrinsic_size(svg_bytes: bytes) -> tuple[float, float] | None:
    try:
        root = ElementTree.fromstring(svg_bytes)
    except ElementTree.ParseError:
        return None
    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if view_box:
        parts = re.split(r"[\s,]+", view_box.strip())
        if len(parts) == 4:
            try:
                width = float(parts[2])
                height = float(parts[3])
            except ValueError:
                width = height = 0.0
            if width > 0.0 and height > 0.0:
                return (width, height)
    width = svg_dimension(root.attrib.get("width"))
    height = svg_dimension(root.attrib.get("height"))
    if width is None or height is None:
        return None
    return (width, height)


def svg_dimension(value: str | None) -> float | None:
    if not value:
        return None
    match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", value)
    if match is None:
        return None
    try:
        parsed = float(match.group(1))
    except ValueError:
        return None
    return parsed if parsed > 0.0 else None
