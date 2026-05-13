from __future__ import annotations

from dataclasses import dataclass
import csv
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


def extract_ocr_labels(image_path: str | Path) -> list[OcrLabel]:
    if not tesseract_available():
        return []
    words = run_tesseract_words(image_path)
    words = [word for word in words if is_useful_text(word.text)]
    if len(words) < 80:
        words.extend(word for word in run_preprocessed_tesseract_words(image_path) if is_useful_text(word.text))
    words = dedupe_labels(words)
    labels = list(words)
    labels.extend(group_line_labels(words))
    labels.extend(group_stacked_labels(words))
    return dedupe_labels(labels)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def parse_client_ocr_labels(raw: str | None) -> list[OcrLabel] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, list):
        return None

    words: list[OcrLabel] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = clean_text(str(item.get("text", "")))
        if not is_useful_text(text):
            continue
        try:
            width = float(item.get("width", 0))
            height = float(item.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            words.append(
                OcrLabel(
                    text=text,
                    x=float(item["x"]),
                    y=float(item["y"]),
                    width=width,
                    height=height,
                    confidence=float(item.get("confidence", 50)),
                )
            )
        except Exception:
            continue
    words = dedupe_labels(words)
    if not words:
        return None
    labels = list(words)
    labels.extend(group_line_labels(words))
    labels.extend(group_stacked_labels(words))
    return dedupe_labels(labels)


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
    for variant, scale_x, scale_y in variants:
        words.extend(run_tesseract_array(variant, scale_x=scale_x, scale_y=scale_y))
    return words


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
