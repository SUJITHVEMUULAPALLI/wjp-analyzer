from shapely.geometry import Polygon
from typing import List, Dict

def containment_depth(polys: list[Polygon]) -> dict[int,int]:
    """
    Compute nesting depth for each polygon index (0 = outermost).
    """
    depths = {i:0 for i in range(len(polys))}
    for i, pi in enumerate(polys):
        for j, pj in enumerate(polys):
            if i==j: 
                continue
            if pj.contains(pi):
                depths[i] += 1
    return depths
