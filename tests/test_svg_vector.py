from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon

from map_boundary_builder.svg_vector import (
    extract_svg_service_path,
    flatten_cubic,
    flatten_svg_path,
    svg_service_path_extraction,
)


def test_vector_rectangle_preserves_exact_line_corners_and_viewbox_offset() -> None:
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="100 200 80 40">
<style>.service { fill: #07f; }</style>
<path class="service" d="M110 205 H170 V235 H110 Z"/>
</svg>"""

    result = svg_service_path_extraction(svg, width=160, height=80)

    expected = Polygon([(20, 10), (140, 10), (140, 70), (20, 70)])
    assert result.pixel_geometry.equals(expected)
    assert set(result.pixel_geometry.exterior.coords[:-1]) == set(expected.exterior.coords[:-1])
    assert len(result.pixel_geometry.exterior.coords) == 5
    assert result.pixel_geometry.area == 7200.0
    assert result.confidence == 1.0
    assert result.diagnostics == {
        "extractor": "svg-vector-v20",
        "svg_vector_path": True,
        "flatten_tolerance_px": 0.15,
        "line_segment_count": 3,
        "curve_segment_count": 0,
    }


def test_vector_path_honors_nested_transforms_and_path_transform_order() -> None:
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<style>path.service { fill: rgb(0, 119, 255); }</style>
<g transform="translate(10 20)">
  <path class="service" transform="scale(2)" d="M1 2h10v5H1z"/>
</g>
</svg>"""

    vector = extract_svg_service_path(svg, width=100, height=100)

    assert vector.geometry.bounds == (12.0, 24.0, 32.0, 34.0)
    assert vector.geometry.area == 200.0
    assert len(vector.geometry.exterior.coords) == 5


def test_non_rendered_defs_path_is_ignored_and_group_fill_is_inherited() -> None:
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<defs><path fill="#07f" d="M1 1H99V99H1Z"/></defs>
<g fill="#0077ff"><path d="M20 25H80V75H20Z"/></g>
</svg>"""

    vector = extract_svg_service_path(svg, width=100, height=100)

    assert vector.geometry.bounds == (20.0, 25.0, 80.0, 75.0)
    assert vector.geometry.area == 3000.0


def test_path_parser_supports_every_svg_path_segment_family() -> None:
    rings, stats = flatten_svg_path(
        "M10 50 L20 40 h10 v-5 C35 20 45 20 50 35 "
        "S65 50 70 35 Q75 20 80 35 T90 35 A10 10 0 0 1 90 55 L10 55 z",
        tolerance_px=0.15,
    )

    assert stats.lines == 4
    assert stats.curves == 5
    assert len(rings) == 1
    assert rings[0][0] == rings[0][-1]
    assert len(rings[0]) > 20
    assert LineString(rings[0]).is_simple


def test_cubic_flattening_stays_within_requested_pixel_error() -> None:
    start = (0.0, 0.0)
    first = (0.0, 80.0)
    second = (100.0, -60.0)
    end = (120.0, 20.0)
    flattened = [start, *flatten_cubic(start, first, second, end, 0.15)]
    line = LineString(flattened)

    errors = []
    for index in range(2001):
        t = index / 2000.0
        u = 1.0 - t
        point = (
            u**3 * start[0] + 3 * u * u * t * first[0] + 3 * u * t * t * second[0] + t**3 * end[0],
            u**3 * start[1] + 3 * u * u * t * first[1] + 3 * u * t * t * second[1] + t**3 * end[1],
        )
        errors.append(line.distance(Point(point)))

    assert max(errors) <= 0.15


def test_evenodd_fill_preserves_hole_and_mask_topology() -> None:
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<path fill="#0077ff" fill-rule="evenodd"
      d="M10 10H90V90H10Z M35 35H65V65H35Z"/>
</svg>"""

    vector = extract_svg_service_path(svg, width=100, height=100)

    assert len(vector.geometry.interiors) == 1
    assert vector.geometry.area == 5500.0
    assert vector.mask[20, 20]
    assert not vector.mask[50, 50]


def test_curve_vertices_are_adaptive_while_hard_line_corner_is_unchanged() -> None:
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<path style="fill:#07f" d="M10 80 L10 20 C35 0 65 0 90 20 L90 80 Z"/>
</svg>"""

    vector = extract_svg_service_path(svg, width=100, height=100, flatten_tolerance_px=0.15)
    vertices = list(vector.geometry.exterior.coords)

    assert (10.0, 20.0) in vertices
    assert (90.0, 20.0) in vertices
    assert (90.0, 80.0) in vertices
    assert len(vertices) > 8
    assert vector.curve_segment_count == 1
    assert vector.line_segment_count == 2
