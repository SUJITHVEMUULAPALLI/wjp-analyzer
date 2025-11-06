import math

def is_point_near_line(px, py, x1, y1, x2, y2, tol=2.0):
    # Distance from point to line segment
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx*wx + vy*wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1) <= tol
    c2 = vx*vx + vy*vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2) <= tol
    b = c1 / c2
    bx, by = x1 + b*vx, y1 + b*vy
    return math.hypot(px - bx, py - by) <= tol

def pick_entity(entities, click_x, click_y, tol=2.0):
    # Naive first-hit; could be improved by z-order / nearest dist
    for e in entities:
        if e.dxftype() == "LINE":
            if is_point_near_line(click_x, click_y,
                                  e.dxf.start[0], e.dxf.start[1],
                                  e.dxf.end[0], e.dxf.end[1], tol):
                return e
        elif e.dxftype() == "CIRCLE":
            cx, cy = e.dxf.center
            if abs(math.hypot(click_x-cx, click_y-cy) - e.dxf.radius) <= tol:
                return e
        elif e.dxftype() == "LWPOLYLINE":
            pts = list(e.get_points())
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                if is_point_near_line(click_x, click_y, x1, y1, x2, y2, tol):
                    return e
    return None
