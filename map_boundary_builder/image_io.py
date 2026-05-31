from __future__ import annotations

from pathlib import Path

from PIL import Image


RASTER_IMAGE_EXTENSIONS = {".avif", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}
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
) -> Path:
    image_path = Path(path)
    target_dir = Path(output_dir) if output_dir is not None else image_path.parent
    if is_svg_image(image_path):
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{image_path.stem}.raster.png"
        rasterize_svg_to_png(image_path, target_path)
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


def rasterize_svg_to_png(source_path: Path, target_path: Path) -> None:
    try:
        import cairosvg
    except Exception as exc:  # pragma: no cover - depends on optional native libs.
        raise ValueError(
            "SVG uploads need to be rasterized before extraction. Use the web app upload flow "
            "or install CairoSVG support, then try again."
        ) from exc

    try:
        if source_path.suffix.lower() == ".svgz":
            cairosvg.svg2png(url=str(source_path), write_to=str(target_path))
        else:
            cairosvg.svg2png(bytestring=source_path.read_bytes(), write_to=str(target_path))
    except Exception as exc:
        raise ValueError(
            f"Could not rasterize SVG image {source_path.name}. Export it as PNG, JPG, WebP, or TIFF "
            "and try again."
        ) from exc
