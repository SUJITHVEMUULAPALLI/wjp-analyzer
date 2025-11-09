import math
from typing import Tuple


def _rot(point: Tuple[float, float], angle_rad: float) -> Tuple[float, float]:
    x, y = point
    return (
        x * math.cos(angle_rad) - y * math.sin(angle_rad),
        x * math.sin(angle_rad) + y * math.cos(angle_rad),
    )


def translate(entity, dx: float, dy: float) -> None:
    if entity.dxftype() == "LINE":
        entity.dxf.start = (entity.dxf.start[0] + dx, entity.dxf.start[1] + dy)
        entity.dxf.end = (entity.dxf.end[0] + dx, entity.dxf.end[1] + dy)
    elif entity.dxftype() == "CIRCLE":
        entity.dxf.center = (entity.dxf.center[0] + dx, entity.dxf.center[1] + dy)
    elif entity.dxftype() == "ARC":
        entity.dxf.center = (entity.dxf.center[0] + dx, entity.dxf.center[1] + dy)
    elif entity.dxftype() == "LWPOLYLINE":
        new_pts = []
        for p in entity.get_points():
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                # Preserve extra fields like bulge, start_width, end_width if present
                x, y = p[0] + dx, p[1] + dy
                if len(p) == 2:
                    new_pts.append((x, y))
                else:
                    new_pts.append((x, y, *p[2:]))
        entity.set_points(new_pts)
    elif entity.dxftype() == "POLYLINE":
        # Translate all vertices
        for vertex in entity.vertices:
            vertex.dxf.location = (vertex.dxf.location.x + dx, vertex.dxf.location.y + dy)
    elif entity.dxftype() == "SPLINE":
        # Translate control points
        try:
            for cp in entity.control_points:
                cp[0] += dx
                cp[1] += dy
        except Exception:
            # Alternative: translate fit points
            try:
                for fp in entity.fit_points:
                    fp[0] += dx
                    fp[1] += dy
            except Exception:
                pass


def scale(entity, factor: float) -> None:
    if entity.dxftype() == "LINE":
        entity.dxf.start = (entity.dxf.start[0] * factor, entity.dxf.start[1] * factor)
        entity.dxf.end = (entity.dxf.end[0] * factor, entity.dxf.end[1] * factor)
    elif entity.dxftype() == "CIRCLE":
        entity.dxf.center = (entity.dxf.center[0] * factor, entity.dxf.center[1] * factor)
        entity.dxf.radius *= factor
    elif entity.dxftype() == "ARC":
        entity.dxf.center = (entity.dxf.center[0] * factor, entity.dxf.center[1] * factor)
        entity.dxf.radius *= factor
    elif entity.dxftype() == "LWPOLYLINE":
        new_pts = []
        for p in entity.get_points():
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0] * factor, p[1] * factor
                if len(p) == 2:
                    new_pts.append((x, y))
                else:
                    new_pts.append((x, y, *p[2:]))
        entity.set_points(new_pts)
    elif entity.dxftype() == "POLYLINE":
        # Scale all vertices
        for vertex in entity.vertices:
            vertex.dxf.location = (vertex.dxf.location.x * factor, vertex.dxf.location.y * factor)
    elif entity.dxftype() == "SPLINE":
        # Scale control points
        try:
            for cp in entity.control_points:
                cp[0] *= factor
                cp[1] *= factor
        except Exception:
            # Alternative: scale fit points
            try:
                for fp in entity.fit_points:
                    fp[0] *= factor
                    fp[1] *= factor
            except Exception:
                pass


def rotate(entity, angle_deg: float) -> None:
    angle = math.radians(angle_deg)
    if entity.dxftype() == "LINE":
        entity.dxf.start = _rot(entity.dxf.start, angle)
        entity.dxf.end = _rot(entity.dxf.end, angle)
    elif entity.dxftype() == "CIRCLE":
        entity.dxf.center = _rot(entity.dxf.center, angle)
    elif entity.dxftype() == "ARC":
        entity.dxf.center = _rot(entity.dxf.center, angle)
        # Rotate start and end angles
        entity.dxf.start_angle += angle_deg
        entity.dxf.end_angle += angle_deg
    elif entity.dxftype() == "LWPOLYLINE":
        new_pts = []
        for p in entity.get_points():
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = _rot((p[0], p[1]), angle)
                if len(p) == 2:
                    new_pts.append((x, y))
                else:
                    new_pts.append((x, y, *p[2:]))
        entity.set_points(new_pts)
    elif entity.dxftype() == "POLYLINE":
        # Rotate all vertices
        for vertex in entity.vertices:
            loc = vertex.dxf.location
            vertex.dxf.location = _rot((loc.x, loc.y), angle)
    elif entity.dxftype() == "SPLINE":
        # Rotate control points
        try:
            for cp in entity.control_points:
                x, y = _rot((cp[0], cp[1]), angle)
                cp[0] = x
                cp[1] = y
        except Exception:
            # Alternative: rotate fit points
            try:
                for fp in entity.fit_points:
                    x, y = _rot((fp[0], fp[1]), angle)
                    fp[0] = x
                    fp[1] = y
            except Exception:
                pass


