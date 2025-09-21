from shapely.geometry import Polygon
from typing import List, Dict

def compute_lengths(polys: list[Polygon], classes: dict[int,str]) -> dict:
    length_outer = 0.0
    length_internal = 0.0
    for i,p in enumerate(polys):
        perim = p.exterior.length
        if classes.get(i) == "outer":
            length_outer += perim
        else:
            length_internal += perim
        for ring in p.interiors:
            # holes count as internal length to cut
            length_internal += ring.length
    pierces = len(polys)  # rough proxy
    return {"length_outer_mm": length_outer, "length_internal_mm": length_internal, "pierces": pierces}
