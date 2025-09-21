from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon
from typing import List

def merge_and_polygonize(lines: list[LineString]) -> tuple[list[LineString], list[Polygon]]:
    """
    Returns merged lines and polygonized closed polygons.
    """
    if not lines:
        return [], []
    merged = linemerge(unary_union(lines))
    # merged can be LineString or MultiLineString
    if hasattr(merged, "geoms"):
        merged_lines = list(merged.geoms)
    else:
        merged_lines = [merged]
    polys = list(polygonize(merged_lines))
    return merged_lines, polys
