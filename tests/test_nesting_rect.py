from wjpanalyser.app.services.nesting.pack_rect import pack_rectangles


def test_rects_pack_within_sheet():
    parts = [
        {"id": "p1", "bbox": (0, 0, 50, 20)},
        {"id": "p2", "bbox": (0, 0, 30, 30)},
        {"id": "p3", "bbox": (0, 0, 10, 60)},
    ]
    placements = pack_rectangles(parts, 200, 200, gap=2.0)
    for p in placements:
        assert 0 <= p["x"] <= 200
        assert 0 <= p["y"] <= 200

