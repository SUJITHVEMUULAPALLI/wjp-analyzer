from wjpanalyser.app.services.grouping import area_key, shape_hash, cluster


def test_area_key_rounding():
    assert area_key(12.3456) == 12.35
    assert area_key(12.344) == 12.34


def test_shape_hash_deterministic():
    pts = [(0, 0), (1, 0), (1, 1)]
    h1 = shape_hash(pts)
    h2 = shape_hash(list(pts))
    assert h1 == h2


def test_cluster_groups_similar():
    parts = [
        {"id": "a", "area": 10.001, "hash": "x", "points": []},
        {"id": "b", "area": 10.002, "hash": "x", "points": []},
        {"id": "c", "area": 20.0, "hash": "y", "points": []},
    ]
    bins = cluster(parts)
    sizes = sorted(len(b) for b in bins)
    assert sizes == [1, 2]

