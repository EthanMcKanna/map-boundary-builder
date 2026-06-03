import api.index as api_index
import map_boundary_builder.web as web
from map_boundary_builder import request_options


def test_api_and_local_web_share_request_option_helpers() -> None:
    for name in (
        "allow_catalog_for_request",
        "bool_field",
        "city_hint_for_request",
        "float_field",
        "int_field",
    ):
        assert getattr(api_index, name) is getattr(request_options, name)
        assert getattr(web, name) is getattr(request_options, name)
    assert api_index.include_overlay_for_request is request_options.include_overlay_for_request


def test_shared_request_option_parsing_matches_api_contract() -> None:
    assert request_options.city_hint_for_request({"city": " Auto-detect "}) is None
    assert request_options.city_hint_for_request({"city": "Dallas"}) == "Dallas"
    assert request_options.bool_field({}, "include_overlay", default=True) is True
    assert request_options.bool_field({"include_overlay": "off"}, "include_overlay", default=True) is False
    assert request_options.float_field({"min_confidence": "2"}, "min_confidence", 0.55, 0, 1) == 1
    assert request_options.float_field({"min_confidence": "bad"}, "min_confidence", 0.55, 0, 1) == 0.55
    assert request_options.int_field({"min_control_points": "1.0"}, "min_control_points", 3, 0, 12) == 3
    assert request_options.allow_catalog_for_request({"no_catalog": "1", "allow_catalog": "1"}) is False
    assert request_options.include_overlay_for_request({}, catalog_probe_only=True) is False
