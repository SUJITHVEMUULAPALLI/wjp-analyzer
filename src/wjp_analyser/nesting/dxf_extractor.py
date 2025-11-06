"""
DXF extractor: tessellate mixed entities (LINE/LWPOLYLINE/POLYLINE/ARC/CIRCLE/SPLINE)
into polygons suitable for nesting. Uses ezdxf + shapely to polygonize.
"""

from __future__ import annotations

from typing import List, Dict, Any

import ezdxf  # type: ignore
from shapely.geometry import LineString, Polygon  # type: ignore
from shapely.ops import unary_union, polygonize  # type: ignore


def _arc_points(cx: float, cy: float, r: float, start_deg: float, end_deg: float, segments: int = 96) -> List[tuple[float, float]]:
    import math
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    if a1 < a0:
        a1 += 2 * math.pi
    step = max(1, int(segments * abs(a1 - a0) / (2 * math.pi)))
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / step), cy + r * math.sin(a0 + (a1 - a0) * i / step)) for i in range(step + 1)]


def _add_poly(polys: List[Dict[str, Any]], pts: List[tuple[float, float]], handle: str, layer: str) -> None:
    if len(pts) < 3:
        return
    try:
        poly = Polygon(pts)
        if poly.is_empty or not poly.is_valid or poly.area <= 0:
            return
        polys.append({
            "id": handle,
            "layer": layer,
            "area": float(poly.area),
            "bbox": tuple(poly.bounds),
            "poly": poly,
            "points": pts,
        })
    except Exception:
        return


def extract_polygons(dxf_path: str) -> List[Dict[str, Any]]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    lines: List[LineString] = []
    polygons: List[Dict[str, Any]] = []

    for e in msp:
        try:
            et = e.dxftype()
            h = e.dxf.handle if hasattr(e, "dxf") and hasattr(e.dxf, "handle") else "unknown"
            layer = e.dxf.layer if hasattr(e, "dxf") and hasattr(e.dxf, "layer") else "0"
            if et == "LWPOLYLINE":
                pts = [(v[0], v[1]) for v in e.get_points("xy")]
                if e.closed:
                    _add_poly(polygons, pts, h, layer)
                elif len(pts) >= 2:
                    lines.append(LineString(pts))
            elif et == "POLYLINE":
                pts = [(v[0], v[1]) for v in e.get_points("xy")]
                if getattr(e, "closed", False) and len(pts) >= 3:
                    _add_poly(polygons, pts, h, layer)
                elif len(pts) >= 2:
                    lines.append(LineString(pts))
            elif et == "LINE":
                p1 = (e.dxf.start.x, e.dxf.start.y)
                p2 = (e.dxf.end.x, e.dxf.end.y)
                lines.append(LineString([p1, p2]))
            elif et == "ARC":
                pts = _arc_points(e.dxf.center.x, e.dxf.center.y, abs(e.dxf.radius), e.dxf.start_angle, e.dxf.end_angle, segments=96)
                lines.append(LineString(pts))
            elif et == "CIRCLE":
                pts = _arc_points(e.dxf.center.x, e.dxf.center.y, abs(e.dxf.radius), 0.0, 360.0, segments=160)
                lines.append(LineString(pts))
            elif et == "SPLINE":
                try:
                    tool = e.construction_tool()
                    pts = [(p[0], p[1]) for p in tool.approximate(120)]
                    if len(pts) >= 2:
                        if pts[0] == pts[-1] and len(pts) >= 3:
                            _add_poly(polygons, pts[:-1], h, layer)
                        else:
                            lines.append(LineString(pts))
                except Exception:
                    pass
        except Exception:
            continue

    # Stitch linework into polygons
    try:
        if lines:
            merged = unary_union(lines)
            for poly in polygonize(merged):
                ext = list(poly.exterior.coords)[:-1]
                _add_poly(polygons, ext, "stitched", "0")
    except Exception:
        pass

    return polygons

























