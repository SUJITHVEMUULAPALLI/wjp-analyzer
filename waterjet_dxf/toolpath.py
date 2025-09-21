from shapely.geometry import Polygon
from typing import List, Dict

def plan_order(polys: list[Polygon], classes: dict[int,str]) -> list[int]:
    """
    Internal (small->large) first, outer last.
    """
    internals = [(i,p) for i,p in enumerate(polys) if classes.get(i)!="outer"]
    outers = [(i,p) for i,p in enumerate(polys) if classes.get(i)=="outer"]
    internals.sort(key=lambda t: t[1].area)  # small first
    order = [i for i,_ in internals] + [i for i,_ in outers]
    return order

def kerf_preview(polys: list[Polygon], kerf_mm: float=1.1) -> list[Polygon]:
    # half-kerf offset for preview (not true tool compensation)
    half = kerf_mm/2.0
    return [p.buffer(-half) for p in polys]
