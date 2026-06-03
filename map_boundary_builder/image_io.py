from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import gzip
import importlib
from io import BytesIO
from pathlib import Path
import re
import time
from typing import Any, Callable
from xml.etree import ElementTree

from PIL import Image


RASTER_IMAGE_EXTENSIONS = {".avif", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SVG_IMAGE_EXTENSIONS = {".svg", ".svgz"}
SUPPORTED_IMAGE_EXTENSIONS = RASTER_IMAGE_EXTENSIONS | SVG_IMAGE_EXTENSIONS
SVG_RASTERIZER_PROBE = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 3">
<rect width="4" height="3" fill="#fff"/>
<rect x="1" y="1" width="2" height="1" fill="#0087ff"/>
</svg>"""


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
    svg_bytes = read_svg_bytes(source_path)
    rasterize_svg_bytes_to_png(
        svg_bytes,
        target_path,
        max_dimension=max_dimension,
        source_path=source_path,
    )


def rasterize_svg_bytes_to_png(
    svg_bytes: bytes,
    target_path: Path,
    *,
    max_dimension: int = 0,
    source_path: Path | None = None,
) -> None:
    output_size = capped_svg_output_size(svg_bytes, max_dimension=max_dimension)
    rasterizers: tuple[tuple[str, Callable[..., None]], ...] = (
        ("resvg-py", rasterize_svg_with_resvg),
        ("CairoSVG", rasterize_svg_with_cairosvg),
    )
    errors: list[str] = []
    for rasterizer_name, rasterizer in rasterizers:
        try:
            rasterizer(svg_bytes, target_path, output_size=output_size, source_path=source_path)
            return
        except Exception as exc:
            errors.append(f"{rasterizer_name}: {exception_summary(exc)}")
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
    detail = "; ".join(errors)
    source_name = source_path.name if source_path is not None else "SVG"
    raise ValueError(
        f"Could not rasterize SVG image {source_name}. Export it as PNG, JPG, WebP, or TIFF "
        f"and try again. Rasterizer errors: {detail}"
    )


def rasterize_svg_with_cairosvg(
    svg_bytes: bytes,
    target_path: Path,
    *,
    output_size: tuple[int, int] | None = None,
    source_path: Path | None = None,
) -> None:
    try:
        cairosvg = importlib.import_module("cairosvg")
    except Exception as exc:  # pragma: no cover - depends on optional native libs.
        raise ValueError(
            "CairoSVG support is unavailable."
        ) from exc

    kwargs: dict[str, object] = {
        "bytestring": svg_bytes,
        "write_to": str(target_path),
    }
    if output_size is not None:
        output_width, output_height = output_size
        kwargs["output_width"] = output_width
        kwargs["output_height"] = output_height
    cairosvg.svg2png(**kwargs)


def rasterize_svg_with_resvg(
    svg_bytes: bytes,
    target_path: Path,
    *,
    output_size: tuple[int, int] | None = None,
    source_path: Path | None = None,
) -> None:
    try:
        resvg_py = importlib.import_module("resvg_py")
    except Exception as exc:  # pragma: no cover - depends on optional binary wheel availability.
        raise ValueError("resvg-py support is unavailable.") from exc

    kwargs: dict[str, object] = {
        "svg_string": svg_bytes.decode("utf-8", "replace"),
    }
    if output_size is not None:
        output_width, output_height = output_size
        kwargs["width"] = output_width
        kwargs["height"] = output_height
    if source_path is not None:
        kwargs["resources_dir"] = str(source_path.parent)
    png_bytes = resvg_py.svg_to_bytes(**kwargs)
    target_path.write_bytes(png_bytes)


def svg_rasterizer_diagnostics() -> dict[str, Any]:
    diagnostics = deepcopy(_svg_rasterizer_diagnostics_cached())
    if diagnostics.get("ok") is not True:
        _svg_rasterizer_diagnostics_cached.cache_clear()
    return diagnostics


@lru_cache(maxsize=1)
def _svg_rasterizer_diagnostics_cached() -> dict[str, Any]:
    diagnostics = {
        "ok": False,
        "preferred": None,
        "cairosvg": probe_svg_rasterizer("CairoSVG", rasterize_svg_with_cairosvg),
        "resvg_py": probe_svg_rasterizer("resvg-py", rasterize_svg_with_resvg),
    }
    if diagnostics["resvg_py"]["ok"]:
        diagnostics["ok"] = True
        diagnostics["preferred"] = "resvg-py"
    elif diagnostics["cairosvg"]["ok"]:
        diagnostics["ok"] = True
        diagnostics["preferred"] = "cairosvg"
    return diagnostics


svg_rasterizer_diagnostics.cache_clear = (  # type: ignore[attr-defined]
    _svg_rasterizer_diagnostics_cached.cache_clear
)


def probe_svg_rasterizer(name: str, rasterizer: Callable[..., None]) -> dict[str, Any]:
    started = time.perf_counter()
    target = BytesIO()
    try:
        if name == "CairoSVG":
            cairosvg = importlib.import_module("cairosvg")
            cairosvg.svg2png(bytestring=SVG_RASTERIZER_PROBE, write_to=target)
            size = len(target.getvalue())
        else:
            resvg_py = importlib.import_module("resvg_py")
            png_bytes = resvg_py.svg_to_bytes(svg_string=SVG_RASTERIZER_PROBE.decode("utf-8"))
            size = len(png_bytes)
        return {
            "ok": size > 0,
            "elapsed_s": round(max(0.0, time.perf_counter() - started), 6),
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_s": round(max(0.0, time.perf_counter() - started), 6),
            "error": exception_summary(exc),
        }


def read_svg_bytes(source_path: Path) -> bytes:
    data = source_path.read_bytes()
    if source_path.suffix.lower() == ".svgz":
        return gzip.decompress(data)
    return data


def exception_summary(exc: Exception, *, max_length: int = 240) -> str:
    summary = f"{exc.__class__.__name__}: {exc}"
    summary = " ".join(summary.split())
    if len(summary) > max_length:
        return f"{summary[: max_length - 1]}..."
    return summary


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
