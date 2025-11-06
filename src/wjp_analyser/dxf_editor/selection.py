import math
from typing import Iterable, Optional


def _xy(obj):
    try:
        if hasattr(obj, "x") and hasattr(obj, "y"):
            return float(obj.x), float(obj.y)
    except Exception:
        pass
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1])
    t = tuple(obj)
    return float(t[0]), float(t[1])


def is_point_near_line(px: float, py: float, x1: float, y1: float, x2: float, y2: float, tol: float = 2.0) -> bool:
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1) <= tol
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2) <= tol
    b = c1 / c2
    bx, by = x1 + b * vx, y1 + b * vy
    return math.hypot(px - bx, py - by) <= tol


def pick_entity(entities: Iterable, click_x: float, click_y: float, tol: float = 2.0) -> Optional[object]:
    for e in entities:
        et = e.dxftype()
        if et == "LINE":
            sx, sy = _xy(e.dxf.start)
            ex, ey = _xy(e.dxf.end)
            if is_point_near_line(click_x, click_y, sx, sy, ex, ey, tol):
                return e
        elif et == "CIRCLE":
            cx, cy = _xy(e.dxf.center)
            if abs(math.hypot(click_x - cx, click_y - cy) - float(e.dxf.radius)) <= tol:
                return e
        elif et == "LWPOLYLINE":
            pts = list(e.get_points())
            if len(pts) >= 2:
                for p1, p2 in zip(pts, pts[1:]):
                    x1, y1 = float(p1[0]), float(p1[1])
                    x2, y2 = float(p2[0]), float(p2[1])
                    if is_point_near_line(click_x, click_y, x1, y1, x2, y2, tol):
                        return e
    return None





