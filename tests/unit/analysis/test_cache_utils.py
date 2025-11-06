from wjp_analyser.analysis.cache_utils import build_cache_key, filter_cached_report


def test_build_cache_key_deterministic():
    key_a = build_cache_key("filehash", {"b": 2, "a": 1})
    key_b = build_cache_key("filehash", {"a": 1, "b": 2})
    assert key_a == key_b


def test_filter_cached_report_applies_group_selection():
    cached = {
        "components": [
            {"id": 1, "group": "outer", "area": 10.0, "perimeter": 20.0},
            {"id": 2, "group": "inner", "area": 5.0, "perimeter": 12.0},
        ],
        "metrics": {},
        "layers": {"OUTER": 1, "INNER": 1},
    }

    filtered = filter_cached_report(cached, selected_groups=["outer"])
    assert len(filtered["components"]) == 1
    assert filtered["components"][0]["id"] == 1
    assert filtered["metrics"]["object_count"] == 1
    assert filtered["metrics"]["total_area"] == 10.0
    assert cached["components"][0]["id"] == 1  # original untouched
