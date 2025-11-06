import ezdxf
import math
from typing import List, Tuple, Dict, Any
from shapely.geometry import LineString, Point
from shapely import geometry as geom

def _spline_to_coords(e, tol: float) -> list[tuple[float, float]]:
    try:
        # flattening gives iterable of Points
        pts = [(p[0], p[1]) for p in e.flattening(distance=tol)]
        # ensure no duplicates in a row
        out = []
        for x,y in pts:
            if not out or (x,y)!=out[-1]:
                out.append((x,y))
        return out
    except Exception:
        return []

def _arc_to_coords(center, radius, start_angle, end_angle, tol_deg=5.0):
    # approximate arc by small segments
    if end_angle < start_angle:
        end_angle += 360.0
    steps = max(3, int(abs(end_angle-start_angle)/tol_deg))
    coords = []
    for i in range(steps+1):
        ang = math.radians(start_angle + (end_angle-start_angle)*i/steps)
        x = center[0] + radius*math.cos(ang)
        y = center[1] + radius*math.sin(ang)
        coords.append((x,y))
    return coords

def load_dxf_as_lines(dxf_path: str, spline_tol: float=0.1) -> list[LineString]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    lines: list[LineString] = []
    for e in msp:
        t = e.dxftype()
        if t == "LINE":
            lines.append(LineString([(e.dxf.start.x, e.dxf.start.y),(e.dxf.end.x, e.dxf.end.y)]))
        elif t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points('xy')]
            if e.closed and pts and pts[0]!=pts[-1]:
                pts.append(pts[0])
            if len(pts)>=2:
                lines.append(LineString(pts))
        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if e.is_closed and pts and pts[0]!=pts[-1]:
                pts.append(pts[0])
            if len(pts)>=2:
                lines.append(LineString(pts))
        elif t == "CIRCLE":
            c = (e.dxf.center.x, e.dxf.center.y)
            r = e.dxf.radius
            coords = _arc_to_coords(c, r, 0, 360, tol_deg=5)
            lines.append(LineString(coords))
        elif t == "ARC":
            c = (e.dxf.center.x, e.dxf.center.y)
            r = e.dxf.radius
            coords = _arc_to_coords(c, r, e.dxf.start_angle, e.dxf.end_angle, tol_deg=5)
            lines.append(LineString(coords))
        elif t == "SPLINE":
            coords = _spline_to_coords(e, tol=spline_tol)
            if len(coords)>=2:
                lines.append(LineString(coords))
        else:
            # ignore TEXT, DIMENSION, etc.
            continue
    return lines

def load_dxf_lines_with_layers(dxf_path: str, spline_tol: float = 0.1) -> tuple[list[LineString], list[str]]:
    """Load DXF as line strings along with their layer names.

    Returns a tuple: (lines, layers) where each entry in layers corresponds
    to the same-index line in lines.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    lines: list[LineString] = []
    layers: list[str] = []
    for e in msp:
        t = e.dxftype()
        layer = str(getattr(e.dxf, "layer", ""))
        if t == "LINE":
            lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
            layers.append(layer)
        elif t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points('xy')]
            if e.closed and pts and pts[0] != pts[-1]:
                pts.append(pts[0])
            if len(pts) >= 2:
                lines.append(LineString(pts))
                layers.append(layer)
        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if e.is_closed and pts and pts[0] != pts[-1]:
                pts.append(pts[0])
            if len(pts) >= 2:
                lines.append(LineString(pts))
                layers.append(layer)
        elif t == "CIRCLE":
            c = (e.dxf.center.x, e.dxf.center.y)
            r = e.dxf.radius
            coords = _arc_to_coords(c, r, 0, 360, tol_deg=5)
            lines.append(LineString(coords))
            layers.append(layer)
        elif t == "ARC":
            c = (e.dxf.center.x, e.dxf.center.y)
            r = e.dxf.radius
            coords = _arc_to_coords(c, r, e.dxf.start_angle, e.dxf.end_angle, tol_deg=5)
            lines.append(LineString(coords))
            layers.append(layer)
        elif t == "SPLINE":
            coords = _spline_to_coords(e, tol=spline_tol)
            if len(coords) >= 2:
                lines.append(LineString(coords))
                layers.append(layer)
        else:
            continue
    return lines, layers

def load_dxf(dxf_path: str) -> List[Any]:
    """
    Load DXF file and return raw entities for enhanced geometry cleaning.
    
    Args:
        dxf_path: Path to DXF file
        
    Returns:
        List of DXF entities
    """
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        entities = list(msp)
        return entities
    except Exception as e:
        print(f"Error loading DXF file {dxf_path}: {e}")
        return []
