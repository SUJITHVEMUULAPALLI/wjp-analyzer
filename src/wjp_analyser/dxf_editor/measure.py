import math


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def bbox_of_entity(e):
    try:
        return e.bbox()
    except Exception:
        if e.dxftype() == "LINE":
            x1, y1 = e.dxf.start
            x2, y2 = e.dxf.end
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if e.dxftype() == "CIRCLE":
            cx, cy = e.dxf.center
            r = e.dxf.radius
            return (cx - r, cy - r, cx + r, cy + r)
        if e.dxftype() == "LWPOLYLINE":
            pts = list(e.get_points())
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
    return (0, 0, 0, 0)


def bbox_size(b):
    x1, y1, x2, y2 = b
    return (x2 - x1, y2 - y1)


def polyline_length(e):
    if e.dxftype() != "LWPOLYLINE":
        return 0.0
    pts = list(e.get_points())
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        total += distance((x1, y1), (x2, y2))
    return total





