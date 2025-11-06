from shapely.geometry import Polygon
from typing import List, Dict

def compute_lengths(polys: list[Polygon], classes: dict[int,str]) -> dict:
    length_outer = 0.0
    length_internal = 0.0
    pierces = 0
    for i, p in enumerate(polys):
        # Count one pierce per exterior loop
        pierces += 1
        perim = p.exterior.length
        if classes.get(i) == "outer":
            length_outer += perim
        else:
            length_internal += perim
        # Holes: add length and each hole typically implies a pierce
        for ring in p.interiors:
            length_internal += ring.length
            pierces += 1
    return {
        "length_outer_mm": length_outer,
        "length_internal_mm": length_internal,
        "pierces": pierces,
    }
