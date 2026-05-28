from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class OcrLabel:
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float


_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
OCR_CACHE_DIR = _CACHE_ROOT / "ocr-labels"
OCR_CACHE_VERSION = "ocr-labels-v2"
RAPIDOCR_MAX_DIMENSION = max(0, int(os.environ.get("MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION", "2000")))
RAPIDOCR_DET_LIMIT_SIDE_LEN = max(0, int(os.environ.get("MAP_BOUNDARY_RAPIDOCR_DET_LIMIT_SIDE_LEN", "640")))
_OCR_MEMORY_CACHE: dict[str, tuple[OcrLabel, ...]] = {}


def extract_ocr_labels(image_path: str | Path) -> list[OcrLabel]:
    use_tesseract = tesseract_available()
    cache_key = ocr_cache_key(image_path, use_tesseract=use_tesseract)
    if cache_key is not None:
        cached = read_ocr_cache(cache_key)
        if cached is not None:
            return list(cached)

    rapid_words: list[OcrLabel] = run_rapidocr_words(image_path)
    words: list[OcrLabel] = list(rapid_words)
    used_tesseract_fallback = False
    if count_useful_labels(words) < 12 and use_tesseract:
        words = run_tesseract_words(image_path)
        words = [word for word in words if is_useful_text(word.text)]
        if len(words) < 80:
            words.extend(word for word in run_preprocessed_tesseract_words(image_path) if is_useful_text(word.text))
        used_tesseract_fallback = True
    if used_tesseract_fallback and count_useful_labels(words) < 12:
        words.extend(rapid_words)
    words = dedupe_labels(words)
    labels = list(words)
    labels.extend(group_line_labels(words))
    labels.extend(group_stacked_labels(words))
    labels = dedupe_labels(labels)
    if cache_key is not None:
        write_ocr_cache(cache_key, labels)
    return labels


def ocr_cache_key(image_path: str | Path, *, use_tesseract: bool) -> str | None:
    try:
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    except OSError:
        return None
    engine = "tesseract" if use_tesseract else "rapidocr"
    return hashlib.sha256(
        (
            f"{OCR_CACHE_VERSION}:{engine}:rapidocr-max-dim={RAPIDOCR_MAX_DIMENSION}:"
            f"rapidocr-det-limit={RAPIDOCR_DET_LIMIT_SIDE_LEN}:{digest}"
        ).encode("utf-8")
    ).hexdigest()


def read_ocr_cache(cache_key: str) -> tuple[OcrLabel, ...] | None:
    cached = _OCR_MEMORY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cache_path = OCR_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        labels = tuple(OcrLabel(**item) for item in data if isinstance(item, dict))
    except Exception:
        return None
    _OCR_MEMORY_CACHE[cache_key] = labels
    return labels


def write_ocr_cache(cache_key: str, labels: list[OcrLabel]) -> None:
    cached = tuple(labels)
    _OCR_MEMORY_CACHE[cache_key] = cached
    cache_path = OCR_CACHE_DIR / f"{cache_key}.json"
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([label.__dict__ for label in cached], separators=(",", ":"))
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(cache_path)
    except OSError:
        return


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def run_tesseract_words(image_path: str | Path) -> list[OcrLabel]:
    command = ["tesseract", str(image_path), "stdout", "--psm", "11", "tsv"]
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    return parse_tesseract_tsv(completed.stdout)


def run_preprocessed_tesseract_words(image_path: str | Path) -> list[OcrLabel]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    service_fill = (
        ((hue >= 92) & (hue <= 116) & (saturation >= 55) & (value >= 85))
        | ((hue >= 75) & (hue <= 105) & (saturation >= 35) & (value >= 45) & (value <= 210))
    )

    neutralized = rgb.copy()
    neutralized[service_fill] = (245, 245, 245)
    neutral_gray = cv2.cvtColor(neutralized, cv2.COLOR_RGB2GRAY)
    neutral_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(neutral_gray)
    blur = cv2.GaussianBlur(neutral_gray, (0, 0), 3)
    high_pass = cv2.addWeighted(neutral_gray, 1.8, blur, -0.8, 0)
    dark_ink = ((neutral_gray < 175).astype(np.uint8) * 255)
    light_map_text = (((value > 95) & (saturation < 150)).astype(np.uint8) * 255)
    variants = [
        (gray, 1.0, 1.0),
        (cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray), 1.0, 1.0),
        ((((value > 145) & (saturation < 120)).astype(np.uint8) * 255), 1.0, 1.0),
        (cv2.resize(light_map_text, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), 3.0, 3.0),
        (cv2.resize(neutral_clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0, 2.0),
        (cv2.resize(high_pass, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0, 2.0),
        (cv2.resize(dark_ink, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), 2.0, 2.0),
    ]
    words: list[OcrLabel] = []
    max_workers = max(1, min(4, len(variants), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_tesseract_array, variant, scale_x=scale_x, scale_y=scale_y)
            for variant, scale_x, scale_y in variants
        ]
        for future in futures:
            words.extend(future.result())
    return words


def run_rapidocr_words(image_path: str | Path) -> list[OcrLabel]:
    ocr_path, scale_x, scale_y = rapidocr_input_image(image_path)
    try:
        engine = rapidocr_engine()
        result, _elapsed = engine(str(ocr_path))
    except Exception:
        return []
    finally:
        if ocr_path != Path(image_path):
            try:
                ocr_path.unlink()
            except OSError:
                pass
    labels = rapidocr_items_to_labels(result)
    if scale_x == 1.0 and scale_y == 1.0:
        return labels
    return [
        OcrLabel(
            text=label.text,
            x=label.x / scale_x,
            y=label.y / scale_y,
            width=label.width / scale_x,
            height=label.height / scale_y,
            confidence=label.confidence,
        )
        for label in labels
    ]


def rapidocr_input_image(image_path: str | Path) -> tuple[Path, float, float]:
    source_path = Path(image_path)
    if RAPIDOCR_MAX_DIMENSION <= 0:
        return source_path, 1.0, 1.0
    bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return source_path, 1.0, 1.0
    height, width = bgr.shape[:2]
    max_dimension = max(width, height)
    if max_dimension <= RAPIDOCR_MAX_DIMENSION:
        return source_path, 1.0, 1.0
    scale = RAPIDOCR_MAX_DIMENSION / float(max_dimension)
    resized = cv2.resize(
        bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    if not cv2.imwrite(tmp_path, resized):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return source_path, 1.0, 1.0
    return Path(tmp_path), scale, scale


@lru_cache(maxsize=1)
def rapidocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    kwargs = {"det_limit_side_len": RAPIDOCR_DET_LIMIT_SIDE_LEN} if RAPIDOCR_DET_LIMIT_SIDE_LEN > 0 else {}
    return RapidOCR(**kwargs)


def rapidocr_items_to_labels(items: object) -> list[OcrLabel]:
    if not isinstance(items, list):
        return []

    labels: list[OcrLabel] = []
    for item in items:
        try:
            box, raw_text, raw_score = item
            points = [(float(point[0]), float(point[1])) for point in box]
            text = clean_text(str(raw_text))
            score = float(raw_score)
        except Exception:
            continue
        if score < 0.35 or not is_useful_text(text):
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue
        labels.append(
            OcrLabel(
                text=text,
                x=(left + right) / 2.0,
                y=(top + bottom) / 2.0,
                width=width,
                height=height,
                confidence=max(0.0, min(100.0, score * 100.0)),
            )
        )
    return labels


def run_tesseract_array(image: np.ndarray, *, scale_x: float = 1.0, scale_y: float = 1.0) -> list[OcrLabel]:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(tmp_path, image)
        command = ["tesseract", tmp_path, "stdout", "--psm", "11", "tsv"]
        try:
            completed = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            return []
        if completed.returncode != 0:
            return []
        words = parse_tesseract_tsv(completed.stdout)
        if scale_x == 1.0 and scale_y == 1.0:
            return words
        return [
            OcrLabel(
                text=word.text,
                x=word.x / scale_x,
                y=word.y / scale_y,
                width=word.width / scale_x,
                height=word.height / scale_y,
                confidence=word.confidence,
            )
            for word in words
        ]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def parse_tesseract_tsv(tsv: str) -> list[OcrLabel]:
    rows = csv.DictReader(tsv.splitlines(), delimiter="\t")
    labels: list[OcrLabel] = []
    for row in rows:
        if row.get("level") != "5":
            continue
        text = clean_text(row.get("text", ""))
        if not text:
            continue
        try:
            confidence = float(row.get("conf", "-1"))
            if confidence < 20:
                continue
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
        except Exception:
            continue
        labels.append(
            OcrLabel(
                text=text,
                x=left + width / 2,
                y=top + height / 2,
                width=width,
                height=height,
                confidence=confidence,
            )
        )
    return labels


def group_line_labels(words: list[OcrLabel]) -> list[OcrLabel]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda word: (word.y, word.x))
    median_height = sorted(word.height for word in sorted_words)[len(sorted_words) // 2]
    y_threshold = max(10.0, median_height * 0.8)
    rows: list[list[OcrLabel]] = []
    for word in sorted_words:
        for row in rows:
            if abs(row[0].y - word.y) <= y_threshold:
                row.append(word)
                break
        else:
            rows.append([word])

    labels: list[OcrLabel] = []
    for row in rows:
        row = sorted(row, key=lambda word: word.x)
        current: list[OcrLabel] = []
        for word in row:
            if not current:
                current = [word]
                continue
            previous = current[-1]
            gap = word.x - previous.x - previous.width / 2 - word.width / 2
            if gap <= max(80.0, median_height * 3.5):
                current.append(word)
            else:
                labels.extend(join_windows(current))
                current = [word]
        labels.extend(join_windows(current))
    return labels


def join_windows(words: list[OcrLabel]) -> list[OcrLabel]:
    if len(words) < 2:
        return []
    labels: list[OcrLabel] = []
    for size in (2, 3):
        if len(words) < size:
            continue
        for start in range(0, len(words) - size + 1):
            chunk = words[start : start + size]
            text = clean_text(" ".join(word.text for word in chunk))
            if not is_useful_text(text):
                continue
            left = min(word.x - word.width / 2 for word in chunk)
            right = max(word.x + word.width / 2 for word in chunk)
            top = min(word.y - word.height / 2 for word in chunk)
            bottom = max(word.y + word.height / 2 for word in chunk)
            labels.append(
                OcrLabel(
                    text=text,
                    x=(left + right) / 2,
                    y=(top + bottom) / 2,
                    width=right - left,
                    height=bottom - top,
                    confidence=sum(word.confidence for word in chunk) / len(chunk),
                )
            )
    return labels


def group_stacked_labels(words: list[OcrLabel]) -> list[OcrLabel]:
    if not words:
        return []
    heights = sorted(max(1.0, word.height) for word in words)
    median_height = heights[len(heights) // 2]
    labels: list[OcrLabel] = []
    for top in words:
        for bottom in words:
            if bottom.y <= top.y:
                continue
            if abs(top.x - bottom.x) > max(90.0, top.width, bottom.width):
                continue
            vertical_gap = bottom.y - top.y - top.height / 2 - bottom.height / 2
            vertical_threshold = min(
                48.0,
                max(14.0, median_height * 3.0, (top.height + bottom.height) * 1.8),
            )
            if vertical_gap < -8.0 or vertical_gap > vertical_threshold:
                continue
            text = clean_text(f"{top.text} {bottom.text}")
            if not is_useful_text(text):
                continue
            left = min(top.x - top.width / 2, bottom.x - bottom.width / 2)
            right = max(top.x + top.width / 2, bottom.x + bottom.width / 2)
            upper = min(top.y - top.height / 2, bottom.y - bottom.height / 2)
            lower = max(top.y + top.height / 2, bottom.y + bottom.height / 2)
            labels.append(
                OcrLabel(
                    text=text,
                    x=(left + right) / 2,
                    y=(upper + lower) / 2,
                    width=right - left,
                    height=lower - upper,
                    confidence=(top.confidence + bottom.confidence) / 2,
                )
            )
    return labels


def dedupe_labels(labels: list[OcrLabel]) -> list[OcrLabel]:
    best: dict[tuple[str, int, int], OcrLabel] = {}
    for label in labels:
        key = (label.text.lower(), round(label.x / 20), round(label.y / 20))
        old = best.get(key)
        if old is None or label.confidence > old.confidence:
            best[key] = label
    return sorted(best.values(), key=lambda label: label.confidence, reverse=True)


def count_useful_labels(labels: list[OcrLabel]) -> int:
    return sum(1 for label in labels if is_useful_text(label.text))


def clean_text(text: str) -> str:
    text = text.strip().replace("|", "I")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"[^A-Za-z0-9 &'/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -/'")
    return text


def is_useful_text(text: str) -> bool:
    if len(text) < 3:
        return False
    if text.isdigit():
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= max(3, len(text.replace(" ", "")) // 2)
