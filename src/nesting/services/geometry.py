from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

def padded(poly, pad_mm):
    return poly.buffer(pad_mm, join_style=2)

def can_place(frame_poly, placed_union, candidate_poly):
    return frame_poly.contains(candidate_poly) and placed_union.disjoint(candidate_poly)

def rotate(poly, deg):
    return affinity.rotate(poly, deg, origin='centroid')

def translate(poly, x, y):
    return affinity.translate(poly, xoff=x, yoff=y)
