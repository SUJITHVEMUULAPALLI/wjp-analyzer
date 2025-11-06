"""
DXF Editor Preview Utilities
============================

Helper functions for preview rendering with normalization, layer classification,
and color scheme management.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import math

# Default color scheme
COLOR_SCHEME = {
    "OUTER": {
        "fill": "#0066CC",      # Blue
        "edge": "#004499",      # Darker blue
        "alpha": 0.7
    },
    "INNER": {
        "fill": "#666666",      # Gray
        "edge": "#444444",      # Darker gray
        "alpha": 0.6
    },
    "HOLE": {
        "fill": "#666666",      # Same as inner
        "edge": "#444444",
        "alpha": 0.6
    },
    "COMPLEX": {
        "fill": "#666666",      # Same as inner
        "edge": "#444444",
        "alpha": 0.6
    },
    "DECOR": {
        "fill": "#666666",      # Same as inner
        "edge": "#444444",
        "alpha": 0.6
    },
    "WARNING": {
        "background": "#FFEBEE",  # Light red
        "border": "#F44336",      # Red
        "text": "#C62828"         # Dark red
    },
    "SELECTED": {
        "fill": "#FF0000",        # Red
        "edge": "#CC0000",
        "alpha": 0.8
    }
}


def normalize_to_origin(points_list: List[List[Tuple[float, float]]]) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float]]:
    """
    Normalize all points so minimum x,y is at (0,0).
    
    Args:
        points_list: List of point lists (each sublist is a polygon/entity)
        
    Returns:
        Tuple of (normalized_points, offset_xy) where offset is (min_x, min_y)
    """
    if not points_list:
        return points_list, (0.0, 0.0)
    
    # Collect all points to find bounding box
    all_points = []
    for pts in points_list:
        if pts:
            all_points.extend(pts)
    
    if not all_points:
        return points_list, (0.0, 0.0)
    
    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    
    # Normalize each point list
    normalized = []
    for pts in points_list:
        normalized.append([(x - min_x, y - min_y) for x, y in pts])
    
    return normalized, (min_x, min_y)


def classify_polygon_layers(polygons: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Classify polygons into layer types (OUTER, INNER, HOLE, etc.).
    Uses the same logic as dxf_analyzer._classify_polylines.
    
    Args:
        polygons: List of polygon dicts with 'points' and optionally 'area'
        
    Returns:
        Dict mapping layer name to list of classified polygons
    """
    from shapely.geometry import Polygon
    
    if not polygons:
        return {"OUTER": [], "INNER": [], "HOLE": [], "COMPLEX": [], "DECOR": []}
    
    # Convert to (points, polygon, area) tuples
    valid_items = []
    for idx, item in enumerate(polygons):
        pts = item.get("points", [])
        if len(pts) < 3:
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid or poly.area == 0:
                continue
            area = abs(poly.area)
            valid_items.append((idx, pts, poly, area, item))
        except Exception:
            continue
    
    if not valid_items:
        return {"OUTER": [], "INNER": [], "HOLE": [], "COMPLEX": [], "DECOR": []}
    
    # Sort by area - largest is outer boundary
    valid_items.sort(key=lambda x: x[3], reverse=True)
    outer_boundary = valid_items[0][2] if valid_items else None
    
    # Classify each polygon
    classified = {
        "OUTER": [],
        "INNER": [],
        "HOLE": [],
        "COMPLEX": [],
        "DECOR": []
    }
    
    for idx, pts, poly, area, item in valid_items:
        vertex_count = max(0, len(pts) - 1)
        
        # Determine layer type
        if poly.area < 0:
            layer_name = "INNER"
        elif vertex_count > 200:
            layer_name = "COMPLEX"
        elif outer_boundary and poly.equals(outer_boundary):
            layer_name = "OUTER"
        elif outer_boundary and outer_boundary.contains(poly):
            layer_name = "HOLE"
        else:
            layer_name = "DECOR"
        
        # Add metadata to item
        classified_item = dict(item)
        classified_item["layer"] = layer_name
        classified_item["area"] = area
        classified_item["vertex_count"] = vertex_count
        classified[layer_name].append(classified_item)
    
    return classified


def get_layer_color(layer_name: str, is_selected: bool = False) -> Dict[str, Any]:
    """
    Get color scheme for a layer type.
    
    Args:
        layer_name: Layer type (OUTER, INNER, HOLE, etc.)
        is_selected: Whether entity is selected
        
    Returns:
        Color dict with fill, edge, alpha
    """
    if is_selected:
        return COLOR_SCHEME["SELECTED"]
    
    # Map layer names to color scheme
    if layer_name in COLOR_SCHEME:
        return COLOR_SCHEME[layer_name]
    
    # Default to INNER for unknown layers
    return COLOR_SCHEME["INNER"]


def convert_hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color to RGB tuple (0-1 range for matplotlib).
    
    Args:
        hex_color: Hex color string like "#0066CC"
        
    Returns:
        RGB tuple (r, g, b) with values 0-1
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


