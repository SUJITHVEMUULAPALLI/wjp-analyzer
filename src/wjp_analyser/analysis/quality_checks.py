from __future__ import annotations

from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from typing import List, Dict

def check_open_contours(merged_lines: list[LineString]) -> list[dict]:
    issues = []
    for i, ln in enumerate(merged_lines):
        if ln.is_ring:
            continue
        # if not ring and length > tiny, flag
        if ln.length > 1e-6:
            length = float(ln.length)
            issues.append(
                {
                    "type": "open_contour",
                    "id": i,
                    "length": length,
                    "message": f"Open contour {i} length {length:.2f} mm",
                }
            )
    return issues


# Re-define with clean ASCII message to avoid encoding issues on some consoles
def check_acute_vertices(polys: list[Polygon], limit_deg: float = 30.0) -> list[dict]:
    """
    Cheap acute angle finder on exterior ring only.
    """
    import math
    issues: list[dict] = []
    for idx, p in enumerate(polys):
        coords = list(p.exterior.coords)
        n = len(coords) - 1
        for k in range(1, n - 1):
            ax, ay = coords[k - 1]
            bx, by = coords[k]
            cx, cy = coords[k + 1]
            v1 = (ax - bx, ay - by)
            v2 = (cx - bx, cy - by)
            def dot(u, v):
                return u[0] * v[0] + u[1] * v[1]
            def norm(u):
                return math.hypot(u[0], u[1])
            d = dot(v1, v2) / (norm(v1) * norm(v2) + 1e-9)
            d = max(-1.0, min(1.0, d))
            ang = math.degrees(math.acos(d))
            if ang < limit_deg:
                angle = round(ang, 1)
                issues.append(
                    {
                        "type": "acute_angle",
                        "polygon": idx,
                        "vertex_index": k,
                        "angle_deg": angle,
                        "message": f"Acute angle {angle:.1f} deg at vertex {k} in polygon {idx}",
                    }
                )
    return issues

def check_min_spacing(polys: list[Polygon], limit_mm: float=3.0) -> list[dict]:
    """
    Very rough: buffer each polygon inward/outward by limit/2 and check overlaps.
    """
    issues = []
    if len(polys) < 2:
        return issues
    # Create expanded solids
    expanded = [p.buffer(limit_mm / 2.0) for p in polys]
    try:
        from shapely.strtree import STRtree
        tree = STRtree(expanded)
        ia, ib = tree.query_bulk(expanded, predicate="intersects")
        for k in range(len(ia)):
            i = int(ia[k])
            j = int(ib[k])
            if j <= i:
                continue
            issues.append(
                {
                    "type": "min_spacing",
                    "a": i,
                    "b": j,
                    "limit_mm": float(limit_mm),
                    "message": f"Spacing between {i} and {j} is below {limit_mm} mm",
                }
            )
    except Exception:
        # Fallback: O(n^2)
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if expanded[i].intersects(expanded[j]):
                    issues.append(
                        {
                            "type": "min_spacing",
                            "a": i,
                            "b": j,
                            "limit_mm": float(limit_mm),
                            "message": f"Spacing between {i} and {j} is below {limit_mm} mm",
                        }
                    )
    return issues
