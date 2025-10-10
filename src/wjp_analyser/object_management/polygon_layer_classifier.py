"""
DXF polyline classifier → layered DXF exporter.

This utility reads LWPOLYLINE entities from an input DXF, classifies them
into OUTER, HOLE, COMPLEX, DECOR buckets using simple geometric heuristics,
creates layers, and writes a new DXF with entities assigned to those layers.

Kept separate from core LayerManager APIs to avoid coupling and to allow
standalone CLI usage for quick workflows.
"""

from __future__ import annotations

import ezdxf
from shapely.geometry import Polygon, LinearRing
from typing import Dict, List, Tuple


def _safe_polygon(points: List[Tuple[float, float]]) -> Polygon:
    """Return a valid Polygon from points; falls back to convex hull if invalid."""
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def classify_polylines(polylines: List[List[Tuple[float, float]]]) -> Dict[str, List[List[Tuple[float, float]]]]:
    """
    Classify polylines into OUTER, HOLE, COMPLEX, DECOR based on area, containment, and node count.
    Returns a mapping of layer name → list of point lists.
    """
    layers: Dict[str, List[List[Tuple[float, float]]]] = {
        "OUTER": [],
        "HOLE": [],
        "COMPLEX": [],
        "DECOR": [],
    }

    if not polylines:
        return layers

    # Sort by absolute area, largest first
    polys_with_area = [(pts, abs(_safe_polygon(pts).area)) for pts in polylines]
    polys_with_area.sort(key=lambda x: x[1], reverse=True)

    outer_boundary_pts = polys_with_area[0][0]
    outer_polygon = _safe_polygon(outer_boundary_pts)

    for pts, _ in polys_with_area:
        # Complexity threshold: many vertices → COMPLEX
        if len(pts) > 200:
            layers["COMPLEX"].append(pts)
            continue

        # Assign the largest as OUTER
        if pts is outer_boundary_pts:
            layers["OUTER"].append(pts)
            continue

        poly = _safe_polygon(pts)
        # Holes: contained by outer boundary
        if outer_polygon.contains(poly):
            layers["HOLE"].append(pts)
            continue

        # Otherwise, decorative/inner features
        layers["DECOR"].append(pts)

    return layers


def create_layers(input_dxf: str, output_dxf: str) -> None:
    """Load DXF → classify LWPOLYLINEs → save with new layers.

    Uses a fresh DXF document for output to avoid API differences across ezdxf versions.
    """
    source_doc = ezdxf.readfile(input_dxf)
    source_msp = source_doc.modelspace()

    polylines: List[List[Tuple[float, float]]] = []
    for e in source_msp.query("LWPOLYLINE"):
        points = [(v[0], v[1]) for v in e.get_points()]
        polylines.append(points)

    layer_groups = classify_polylines(polylines)

    # Create a new DXF for output to avoid clearing modelspace (not available in some versions)
    out_doc = ezdxf.new(setup=True)
    out_msp = out_doc.modelspace()

    # Ensure layers exist
    for lname in layer_groups.keys():
        if lname not in out_doc.layers:
            out_doc.layers.new(name=lname)

    # Add entities on their respective layers
    for lname, plist in layer_groups.items():
        for pts in plist:
            out_msp.add_lwpolyline(pts, dxfattribs={"layer": lname})

    out_doc.saveas(output_dxf)


__all__ = [
    "classify_polylines",
    "create_layers",
]


