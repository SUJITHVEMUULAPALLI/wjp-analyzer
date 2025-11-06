def add_line(msp, x1, y1, x2, y2, layer="0"):
    return msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer})


def add_circle(msp, cx, cy, r, layer="0"):
    return msp.add_circle((cx, cy), r, dxfattribs={"layer": layer})


def add_rect(msp, x, y, w, h, layer="0"):
    p1 = (x, y)
    p2 = (x + w, y)
    p3 = (x + w, y + h)
    p4 = (x, y + h)
    pl = msp.add_lwpolyline([p1, p2, p3, p4, p1], dxfattribs={"layer": layer})
    return pl


def add_polyline(msp, points, closed=False, layer="0"):
    pts = list(points)
    if closed and (len(pts) == 0 or pts[0] != pts[-1]):
        pts.append(pts[0])
    return msp.add_lwpolyline(pts, dxfattribs={"layer": layer})





