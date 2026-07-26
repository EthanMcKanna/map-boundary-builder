from __future__ import annotations

"""Vector-native extraction for service-area paths embedded in SVG maps.

The raster models are deliberately not involved here.  SVG already contains the
authoritative boundary, so converting it to a mask and asking a segmentation
model to rediscover that boundary can only lose information.  This module keeps
line segments exact and adaptively flattens only curves and arcs.
"""

from dataclasses import dataclass
from math import acos, atan2, ceil, cos, hypot, isfinite, pi, radians, sin, sqrt, tan
import re
from typing import Iterable
from xml.etree import ElementTree

import cv2
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union

from .extract import ExtractionResult


DEFAULT_VECTOR_FLATTEN_TOLERANCE_PX = 0.15
DEFAULT_SERVICE_FILL = "#07f"
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")

Point2D = tuple[float, float]
# SVG affine matrix: x' = a*x + c*y + e; y' = b*x + d*y + f.
Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


class SvgVectorError(ValueError):
    """The SVG does not contain one unambiguous vector service-area path."""


@dataclass(frozen=True)
class SvgVectorPath:
    geometry: Polygon | MultiPolygon
    mask: np.ndarray
    width: int
    height: int
    flatten_tolerance_px: float
    line_segment_count: int
    curve_segment_count: int


@dataclass
class _PathStats:
    lines: int = 0
    curves: int = 0


@dataclass(frozen=True)
class _Candidate:
    geometry: Polygon | MultiPolygon
    stats: _PathStats


def extract_svg_service_path(
    svg_bytes: bytes,
    *,
    width: int,
    height: int,
    fill: str = DEFAULT_SERVICE_FILL,
    flatten_tolerance_px: float = DEFAULT_VECTOR_FLATTEN_TOLERANCE_PX,
) -> SvgVectorPath:
    """Extract one filled SVG path directly in output-pixel coordinates.

    ``width`` and ``height`` must be the dimensions of the raster produced from
    the SVG.  The viewBox mapping (including preserveAspectRatio) and every
    ancestor/path transform are included in the resulting geometry.
    """

    if width <= 0 or height <= 0:
        raise SvgVectorError("SVG output dimensions must be positive")
    if not 0.0 < flatten_tolerance_px <= 1.0:
        raise SvgVectorError("SVG vector flattening tolerance must be in (0, 1] pixels")
    try:
        root = ElementTree.fromstring(svg_bytes)
    except ElementTree.ParseError as exc:
        raise SvgVectorError("SVG XML could not be parsed") from exc
    if _local_name(root.tag) != "svg":
        raise SvgVectorError("Document root is not an SVG element")

    viewbox = _parse_viewbox(root)
    viewport_transform = _viewbox_transform(
        viewbox,
        width=width,
        height=height,
        preserve_aspect_ratio=root.attrib.get("preserveAspectRatio", "xMidYMid meet"),
    )
    class_styles = _class_style_rules(root)
    target_fill = normalize_svg_color(fill)
    candidates: list[_Candidate] = []

    def visit(
        element: ElementTree.Element,
        parent_transform: Matrix,
        inherited: dict[str, str],
        hidden: bool,
    ) -> None:
        element_name = _local_name(element.tag)
        properties = _computed_properties(element, inherited, class_styles)
        element_hidden = (
            hidden
            or element_name in {"clippath", "defs", "marker", "mask", "pattern", "symbol"}
            or properties.get("display") == "none"
            or properties.get("visibility") == "hidden"
        )
        opacity = _finite_float(properties.get("opacity"), default=1.0)
        element_hidden = element_hidden or opacity <= 0.0
        transform = multiply_matrix(parent_transform, parse_transform(element.attrib.get("transform")))
        if (
            not element_hidden
            and element_name == "path"
            and normalize_svg_color(properties.get("fill", "black")) == target_fill
        ):
            path_data = element.attrib.get("d", "")
            if path_data.strip():
                try:
                    rings, stats = flatten_svg_path(
                        path_data,
                        transform=transform,
                        tolerance_px=flatten_tolerance_px,
                    )
                    geometry = rings_to_filled_geometry(
                        rings,
                        fill_rule=properties.get("fill-rule", "nonzero"),
                    )
                except SvgVectorError:
                    # Exporters commonly leave zero-area marker paths in the
                    # same CSS class.  They are not service-area candidates.
                    geometry = None
                if geometry is not None and geometry.area > 1e-8:
                    candidates.append(_Candidate(geometry=geometry, stats=stats))
        for child in element:
            if _local_name(child.tag) != "style":
                visit(child, transform, properties, element_hidden)

    # The viewport mapping owns the SVG root coordinate system.  A transform on
    # the root itself is still honored by visit.
    visit(root, viewport_transform, {}, False)
    if len(candidates) != 1:
        raise SvgVectorError(
            f"Expected one visible {fill} service-area path, found {len(candidates)}"
        )

    candidate = candidates[0]
    geometry = _polygonal_only(candidate.geometry.intersection(box(0.0, 0.0, float(width), float(height))))
    if geometry is None or geometry.is_empty:
        raise SvgVectorError("SVG service-area path did not form a valid filled polygon")
    mask = geometry_to_mask(geometry, width=width, height=height)
    if not mask.any():
        raise SvgVectorError("SVG service-area path is outside the output viewport")
    return SvgVectorPath(
        geometry=geometry,
        mask=mask,
        width=width,
        height=height,
        flatten_tolerance_px=flatten_tolerance_px,
        line_segment_count=candidate.stats.lines,
        curve_segment_count=candidate.stats.curves,
    )


def svg_service_path_extraction(
    svg_bytes: bytes,
    *,
    width: int,
    height: int,
    fill: str = DEFAULT_SERVICE_FILL,
    flatten_tolerance_px: float = DEFAULT_VECTOR_FLATTEN_TOLERANCE_PX,
) -> ExtractionResult:
    vector = extract_svg_service_path(
        svg_bytes,
        width=width,
        height=height,
        fill=fill,
        flatten_tolerance_px=flatten_tolerance_px,
    )
    polygon_count = 1 if isinstance(vector.geometry, Polygon) else len(vector.geometry.geoms)
    return ExtractionResult(
        mask=vector.mask,
        style="bright-blue",
        pixel_geometry=vector.geometry,
        coverage_ratio=float(vector.geometry.area / float(width * height)),
        contour_count=polygon_count,
        confidence=1.0,
        diagnostics={
            "extractor": "svg-vector-v20",
            "svg_vector_path": True,
            "flatten_tolerance_px": vector.flatten_tolerance_px,
            "line_segment_count": vector.line_segment_count,
            "curve_segment_count": vector.curve_segment_count,
        },
    )


def flatten_svg_path(
    path_data: str,
    *,
    transform: Matrix = IDENTITY,
    tolerance_px: float = DEFAULT_VECTOR_FLATTEN_TOLERANCE_PX,
) -> tuple[list[list[Point2D]], _PathStats]:
    tokens = _PATH_TOKEN_RE.findall(path_data)
    if not tokens:
        raise SvgVectorError("SVG path data is empty")
    residue = _PATH_TOKEN_RE.sub("", path_data)
    if re.sub(r"[\s,]+", "", residue):
        raise SvgVectorError("SVG path contains unsupported syntax")

    index = 0
    command: str | None = None
    previous_command: str | None = None
    current: Point2D = (0.0, 0.0)
    subpath_start: Point2D = current
    cubic_control: Point2D | None = None
    quadratic_control: Point2D | None = None
    rings: list[list[Point2D]] = []
    active: list[Point2D] | None = None
    stats = _PathStats()

    def has_number() -> bool:
        return index < len(tokens) and not _is_command(tokens[index])

    def values(count: int) -> list[float]:
        nonlocal index
        if index + count > len(tokens) or any(_is_command(token) for token in tokens[index : index + count]):
            raise SvgVectorError("SVG path command has too few parameters")
        result = [float(token) for token in tokens[index : index + count]]
        if not all(isfinite(value) for value in result):
            raise SvgVectorError("SVG path contains a non-finite coordinate")
        index += count
        return result

    def relative(point: Point2D, is_relative: bool) -> Point2D:
        return (point[0] + current[0], point[1] + current[1]) if is_relative else point

    def begin(point: Point2D) -> None:
        nonlocal active, subpath_start
        if active is not None and len(active) >= 3:
            _close_ring(active)
            rings.append(active)
        active = [apply_matrix(transform, point)]
        subpath_start = point

    def append_source(point: Point2D) -> None:
        nonlocal active
        if active is None:
            begin(current)
        _append_distinct(active, apply_matrix(transform, point))

    def append_curve(points: Iterable[Point2D]) -> None:
        nonlocal active
        if active is None:
            begin(current)
        for point in points:
            _append_distinct(active, point)

    while index < len(tokens):
        if _is_command(tokens[index]):
            command = tokens[index]
            index += 1
        elif command is None:
            raise SvgVectorError("SVG path must begin with a command")
        assert command is not None
        absolute = command.isupper()
        op = command.upper()

        if op == "Z":
            if active is not None:
                _close_ring(active)
                if len(active) >= 4:
                    rings.append(active)
                active = None
            current = subpath_start
            cubic_control = quadratic_control = None
            previous_command = command
            command = None
            continue

        if op == "M":
            x, y = values(2)
            current = relative((x, y), not absolute)
            begin(current)
            cubic_control = quadratic_control = None
            previous_command = command
            # Every coordinate pair after moveto is an implicit lineto.
            command = "L" if absolute else "l"
            while has_number():
                x, y = values(2)
                current = relative((x, y), not absolute)
                append_source(current)
                stats.lines += 1
                previous_command = command
            continue

        parameter_count = {"L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}.get(op)
        if parameter_count is None:
            raise SvgVectorError(f"Unsupported SVG path command: {command}")
        if not has_number():
            raise SvgVectorError(f"SVG path command {command} has no parameters")

        while has_number():
            args = values(parameter_count)
            start = current
            if op == "L":
                current = relative((args[0], args[1]), not absolute)
                append_source(current)
                stats.lines += 1
            elif op == "H":
                current = ((current[0] + args[0]) if not absolute else args[0], current[1])
                append_source(current)
                stats.lines += 1
            elif op == "V":
                current = (current[0], (current[1] + args[0]) if not absolute else args[0])
                append_source(current)
                stats.lines += 1
            elif op == "C":
                control1 = relative((args[0], args[1]), not absolute)
                control2 = relative((args[2], args[3]), not absolute)
                current = relative((args[4], args[5]), not absolute)
                append_curve(
                    flatten_cubic(
                        apply_matrix(transform, start),
                        apply_matrix(transform, control1),
                        apply_matrix(transform, control2),
                        apply_matrix(transform, current),
                        tolerance_px,
                    )
                )
                cubic_control = control2
                quadratic_control = None
                stats.curves += 1
            elif op == "S":
                control1 = (
                    (2.0 * start[0] - cubic_control[0], 2.0 * start[1] - cubic_control[1])
                    if previous_command is not None and previous_command.upper() in {"C", "S"} and cubic_control is not None
                    else start
                )
                control2 = relative((args[0], args[1]), not absolute)
                current = relative((args[2], args[3]), not absolute)
                append_curve(
                    flatten_cubic(
                        apply_matrix(transform, start),
                        apply_matrix(transform, control1),
                        apply_matrix(transform, control2),
                        apply_matrix(transform, current),
                        tolerance_px,
                    )
                )
                cubic_control = control2
                quadratic_control = None
                stats.curves += 1
            elif op == "Q":
                control = relative((args[0], args[1]), not absolute)
                current = relative((args[2], args[3]), not absolute)
                append_curve(
                    flatten_quadratic(
                        apply_matrix(transform, start),
                        apply_matrix(transform, control),
                        apply_matrix(transform, current),
                        tolerance_px,
                    )
                )
                quadratic_control = control
                cubic_control = None
                stats.curves += 1
            elif op == "T":
                control = (
                    (2.0 * start[0] - quadratic_control[0], 2.0 * start[1] - quadratic_control[1])
                    if previous_command is not None and previous_command.upper() in {"Q", "T"} and quadratic_control is not None
                    else start
                )
                current = relative((args[0], args[1]), not absolute)
                append_curve(
                    flatten_quadratic(
                        apply_matrix(transform, start),
                        apply_matrix(transform, control),
                        apply_matrix(transform, current),
                        tolerance_px,
                    )
                )
                quadratic_control = control
                cubic_control = None
                stats.curves += 1
            elif op == "A":
                rx, ry, rotation, large_arc, sweep, x, y = args
                if large_arc not in {0.0, 1.0} or sweep not in {0.0, 1.0}:
                    raise SvgVectorError("SVG arc flags must be 0 or 1")
                current = relative((x, y), not absolute)
                append_curve(
                    flatten_arc(
                        start,
                        current,
                        rx=rx,
                        ry=ry,
                        x_axis_rotation=rotation,
                        large_arc=bool(large_arc),
                        sweep=bool(sweep),
                        transform=transform,
                        tolerance_px=tolerance_px,
                    )
                )
                cubic_control = quadratic_control = None
                stats.curves += 1

            if op not in {"C", "S", "Q", "T"}:
                cubic_control = quadratic_control = None
            previous_command = command

    if active is not None and len(active) >= 3:
        # SVG fill semantics implicitly close open subpaths.
        _close_ring(active)
        if len(active) >= 4:
            rings.append(active)
    if not rings:
        raise SvgVectorError("SVG path does not contain a filled subpath")
    return rings, stats


def flatten_quadratic(
    start: Point2D,
    control: Point2D,
    end: Point2D,
    tolerance_px: float,
) -> list[Point2D]:
    result: list[Point2D] = []

    def recurse(a: Point2D, b: Point2D, c: Point2D, depth: int) -> None:
        if depth >= 24 or _distance_to_line(b, a, c) <= tolerance_px:
            result.append(c)
            return
        ab = _midpoint(a, b)
        bc = _midpoint(b, c)
        recurse(a, ab, _midpoint(ab, bc), depth + 1)
        recurse(_midpoint(ab, bc), bc, c, depth + 1)

    recurse(start, control, end, 0)
    return result


def flatten_cubic(
    start: Point2D,
    control1: Point2D,
    control2: Point2D,
    end: Point2D,
    tolerance_px: float,
) -> list[Point2D]:
    result: list[Point2D] = []

    def recurse(a: Point2D, b: Point2D, c: Point2D, d: Point2D, depth: int) -> None:
        flatness = max(_distance_to_line(b, a, d), _distance_to_line(c, a, d))
        if depth >= 24 or flatness <= tolerance_px:
            result.append(d)
            return
        ab = _midpoint(a, b)
        bc = _midpoint(b, c)
        cd = _midpoint(c, d)
        abc = _midpoint(ab, bc)
        bcd = _midpoint(bc, cd)
        middle = _midpoint(abc, bcd)
        recurse(a, ab, abc, middle, depth + 1)
        recurse(middle, bcd, cd, d, depth + 1)

    recurse(start, control1, control2, end, 0)
    return result


def flatten_arc(
    start: Point2D,
    end: Point2D,
    *,
    rx: float,
    ry: float,
    x_axis_rotation: float,
    large_arc: bool,
    sweep: bool,
    transform: Matrix,
    tolerance_px: float,
) -> list[Point2D]:
    rx, ry = abs(rx), abs(ry)
    transformed_end = apply_matrix(transform, end)
    if rx <= 1e-12 or ry <= 1e-12 or _points_equal(start, end):
        return [transformed_end]

    phi = radians(x_axis_rotation % 360.0)
    cos_phi, sin_phi = cos(phi), sin(phi)
    dx2 = (start[0] - end[0]) / 2.0
    dy2 = (start[1] - end[1]) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2
    scale = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if scale > 1.0:
        scale = sqrt(scale)
        rx *= scale
        ry *= scale

    numerator = max(0.0, rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p)
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coefficient = 0.0 if denominator <= 1e-24 else sqrt(numerator / denominator)
    if large_arc == sweep:
        coefficient = -coefficient
    cxp = coefficient * (rx * y1p / ry)
    cyp = coefficient * (-ry * x1p / rx)
    cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta1 = atan2(uy, ux)
    delta = _signed_vector_angle((ux, uy), (vx, vy))
    if not sweep and delta > 0.0:
        delta -= 2.0 * pi
    elif sweep and delta < 0.0:
        delta += 2.0 * pi

    def point_at(theta: float) -> Point2D:
        local = (
            cx + cos_phi * rx * cos(theta) - sin_phi * ry * sin(theta),
            cy + sin_phi * rx * cos(theta) + cos_phi * ry * sin(theta),
        )
        return apply_matrix(transform, local)

    result: list[Point2D] = []

    def recurse(t0: float, p0: Point2D, t1: float, p1: Point2D, depth: int) -> None:
        tm = (t0 + t1) / 2.0
        pm = point_at(tm)
        # Quarter probes keep highly eccentric/transformed arcs from looking
        # deceptively flat at the single midpoint.
        q1 = point_at((t0 + tm) / 2.0)
        q3 = point_at((tm + t1) / 2.0)
        error = max(
            _distance_to_line(pm, p0, p1),
            _distance_to_line(q1, p0, p1),
            _distance_to_line(q3, p0, p1),
        )
        if depth >= 24 or error <= tolerance_px:
            result.append(p1)
            return
        recurse(t0, p0, tm, pm, depth + 1)
        recurse(tm, pm, t1, p1, depth + 1)

    segment_count = max(1, int(ceil(abs(delta) / (pi / 2.0))))
    previous_t = theta1
    previous_point = apply_matrix(transform, start)
    for segment in range(segment_count):
        next_t = theta1 + delta * ((segment + 1) / segment_count)
        next_point = transformed_end if segment + 1 == segment_count else point_at(next_t)
        recurse(previous_t, previous_point, next_t, next_point, 0)
        previous_t, previous_point = next_t, next_point
    return result


def rings_to_filled_geometry(
    rings: list[list[Point2D]],
    *,
    fill_rule: str = "nonzero",
) -> Polygon | MultiPolygon | None:
    linework = unary_union([LineString(ring) for ring in rings if len(ring) >= 4])
    faces = list(polygonize(linework))
    if not faces:
        return None
    evenodd = fill_rule.strip().lower() == "evenodd"
    selected = []
    for face in faces:
        probe = face.representative_point()
        winding = sum(_ring_winding_number(ring, probe.x, probe.y) for ring in rings)
        if (abs(winding) % 2 == 1) if evenodd else (winding != 0):
            selected.append(face)
    if not selected:
        return None
    return _polygonal_only(unary_union(selected))


def geometry_to_mask(
    geometry: Polygon | MultiPolygon,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = [geometry] if isinstance(geometry, Polygon) else sorted(geometry.geoms, key=lambda item: item.area, reverse=True)
    for polygon in polygons:
        exterior = _cv_contour(polygon.exterior.coords, width=width, height=height)
        if len(exterior) >= 3:
            cv2.fillPoly(mask, [exterior], 1)
        for interior in polygon.interiors:
            hole = _cv_contour(interior.coords, width=width, height=height)
            if len(hole) >= 3:
                cv2.fillPoly(mask, [hole], 0)
    return mask.astype(bool)


def normalize_svg_color(value: str | None) -> str:
    if value is None:
        return ""
    normalized = re.sub(r"\s*!important\s*$", "", value.strip().lower())
    if re.fullmatch(r"#[0-9a-f]{3}", normalized):
        return "#" + "".join(character * 2 for character in normalized[1:])
    if re.fullmatch(r"#[0-9a-f]{6}", normalized):
        return normalized
    rgb = re.fullmatch(r"rgb\(\s*(\d+)\s*[, ]\s*(\d+)\s*[, ]\s*(\d+)\s*\)", normalized)
    if rgb is not None:
        channels = [min(255, int(channel)) for channel in rgb.groups()]
        return "#" + "".join(f"{channel:02x}" for channel in channels)
    return normalized


def parse_transform(value: str | None) -> Matrix:
    if not value or not value.strip():
        return IDENTITY
    matrix = IDENTITY
    matches = list(_TRANSFORM_RE.finditer(value))
    residue = _TRANSFORM_RE.sub("", value)
    if re.sub(r"[\s,]+", "", residue):
        raise SvgVectorError("SVG contains an invalid transform")
    for match in matches:
        name = match.group(1).lower()
        args = [float(number) for number in _NUMBER_RE.findall(match.group(2))]
        if not all(isfinite(number) for number in args):
            raise SvgVectorError("SVG transform contains a non-finite value")
        if name == "matrix" and len(args) == 6:
            operation: Matrix = tuple(args)  # type: ignore[assignment]
        elif name == "translate" and len(args) in {1, 2}:
            operation = (1.0, 0.0, 0.0, 1.0, args[0], args[1] if len(args) == 2 else 0.0)
        elif name == "scale" and len(args) in {1, 2}:
            operation = (args[0], 0.0, 0.0, args[-1], 0.0, 0.0)
        elif name == "rotate" and len(args) in {1, 3}:
            angle = radians(args[0])
            rotation: Matrix = (cos(angle), sin(angle), -sin(angle), cos(angle), 0.0, 0.0)
            if len(args) == 3:
                cx, cy = args[1], args[2]
                operation = multiply_matrix(
                    multiply_matrix((1.0, 0.0, 0.0, 1.0, cx, cy), rotation),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                operation = rotation
        elif name == "skewx" and len(args) == 1:
            operation = (1.0, 0.0, tan(radians(args[0])), 1.0, 0.0, 0.0)
        elif name == "skewy" and len(args) == 1:
            operation = (1.0, tan(radians(args[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            raise SvgVectorError(f"Unsupported or malformed SVG transform: {match.group(0)}")
        matrix = multiply_matrix(matrix, operation)
    return matrix


def multiply_matrix(left: Matrix, right: Matrix) -> Matrix:
    la, lb, lc, ld, le, lf = left
    ra, rb, rc, rd, re_, rf = right
    return (
        la * ra + lc * rb,
        lb * ra + ld * rb,
        la * rc + lc * rd,
        lb * rc + ld * rd,
        la * re_ + lc * rf + le,
        lb * re_ + ld * rf + lf,
    )


def apply_matrix(matrix: Matrix, point: Point2D) -> Point2D:
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def _parse_viewbox(root: ElementTree.Element) -> tuple[float, float, float, float]:
    value = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if not value:
        raise SvgVectorError("SVG vector extraction requires a viewBox")
    parts = re.split(r"[\s,]+", value.strip())
    if len(parts) != 4:
        raise SvgVectorError("SVG viewBox must contain four numbers")
    try:
        min_x, min_y, width, height = (float(part) for part in parts)
    except ValueError as exc:
        raise SvgVectorError("SVG viewBox contains an invalid number") from exc
    if not all(isfinite(part) for part in (min_x, min_y, width, height)) or width <= 0.0 or height <= 0.0:
        raise SvgVectorError("SVG viewBox dimensions must be finite and positive")
    return (min_x, min_y, width, height)


def _viewbox_transform(
    viewbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    preserve_aspect_ratio: str,
) -> Matrix:
    min_x, min_y, box_width, box_height = viewbox
    sx, sy = width / box_width, height / box_height
    setting = " ".join(preserve_aspect_ratio.strip().split())
    if setting.lower().startswith("defer "):
        setting = setting[6:].lstrip()
    parts = setting.split()
    align = parts[0] if parts else "xMidYMid"
    if align.lower() == "none":
        return (sx, 0.0, 0.0, sy, -min_x * sx, -min_y * sy)
    meet_or_slice = parts[1].lower() if len(parts) > 1 else "meet"
    scale = max(sx, sy) if meet_or_slice == "slice" else min(sx, sy)
    rendered_width, rendered_height = box_width * scale, box_height * scale
    x_align = 0.0 if "xMin" in align else (width - rendered_width if "xMax" in align else (width - rendered_width) / 2.0)
    y_align = 0.0 if "YMin" in align else (height - rendered_height if "YMax" in align else (height - rendered_height) / 2.0)
    return (scale, 0.0, 0.0, scale, x_align - min_x * scale, y_align - min_y * scale)


def _class_style_rules(root: ElementTree.Element) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for element in root.iter():
        if _local_name(element.tag) != "style" or not element.text:
            continue
        css = re.sub(r"/\*.*?\*/", "", element.text, flags=re.DOTALL)
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css, flags=re.DOTALL):
            declarations = _style_declarations(body)
            for selector in selectors.split(","):
                # This intentionally supports the exported-map selectors we can
                # resolve without a browser cascade: .class and path.class.
                match = re.fullmatch(r"\s*(?:path)?\.([A-Za-z_][\w-]*)\s*", selector)
                if match is not None:
                    result.setdefault(match.group(1), {}).update(declarations)
    return result


def _computed_properties(
    element: ElementTree.Element,
    inherited: dict[str, str],
    class_styles: dict[str, dict[str, str]],
) -> dict[str, str]:
    properties = {
        name: inherited[name]
        for name in ("fill", "fill-rule", "visibility", "opacity")
        if name in inherited
    }
    for name in ("fill", "fill-rule", "display", "visibility", "opacity"):
        if name in element.attrib:
            properties[name] = element.attrib[name].strip()
    for class_name in element.attrib.get("class", "").split():
        properties.update(class_styles.get(class_name, {}))
    properties.update(_style_declarations(element.attrib.get("style", "")))
    return properties


def _style_declarations(style: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in style.split(";"):
        if ":" not in declaration:
            continue
        name, value = declaration.split(":", 1)
        normalized_name = name.strip().lower()
        if normalized_name in {"fill", "fill-rule", "display", "visibility", "opacity"}:
            result[normalized_name] = re.sub(r"\s*!important\s*$", "", value.strip())
    return result


def _polygonal_only(geometry: object) -> Polygon | MultiPolygon | None:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return geometry
    geoms = getattr(geometry, "geoms", ())
    polygons = [item for item in geoms if isinstance(item, Polygon)]
    if not polygons:
        return None
    merged = unary_union(polygons)
    return merged if isinstance(merged, (Polygon, MultiPolygon)) else None


def _ring_winding_number(ring: list[Point2D], x: float, y: float) -> int:
    winding = 0
    for first, second in zip(ring, ring[1:]):
        if first[1] <= y < second[1] and _is_left(first, second, (x, y)) > 0.0:
            winding += 1
        elif second[1] <= y < first[1] and _is_left(first, second, (x, y)) < 0.0:
            winding -= 1
    return winding


def _is_left(first: Point2D, second: Point2D, point: Point2D) -> float:
    return (second[0] - first[0]) * (point[1] - first[1]) - (point[0] - first[0]) * (second[1] - first[1])


def _cv_contour(coords: Iterable[Point2D], *, width: int, height: int) -> np.ndarray:
    points = np.asarray(list(coords), dtype=np.float64)
    points[:, 0] = np.clip(np.rint(points[:, 0]), -1, width)
    points[:, 1] = np.clip(np.rint(points[:, 1]), -1, height)
    return points.astype(np.int32).reshape((-1, 1, 2))


def _distance_to_line(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length <= 1e-15:
        return hypot(point[0] - start[0], point[1] - start[1])
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / length


def _signed_vector_angle(first: Point2D, second: Point2D) -> float:
    dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    angle = acos(dot)
    return -angle if first[0] * second[1] - first[1] * second[0] < 0.0 else angle


def _midpoint(first: Point2D, second: Point2D) -> Point2D:
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def _append_distinct(points: list[Point2D], point: Point2D) -> None:
    if not points or not _points_equal(points[-1], point):
        points.append(point)


def _close_ring(points: list[Point2D]) -> None:
    if points and not _points_equal(points[0], points[-1]):
        points.append(points[0])


def _points_equal(first: Point2D, second: Point2D) -> bool:
    return abs(first[0] - second[0]) <= 1e-9 and abs(first[1] - second[1]) <= 1e-9


def _is_command(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _finite_float(value: str | None, *, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except ValueError:
        return default
    return parsed if isfinite(parsed) else default
