from shapely.geometry import Polygon
from typing import List, Dict

LAYER_HINTS = {
    "OUTER": "outer",
    "PROFILE": "outer",
    "BOUNDARY": "outer",
    "INNER": "inner",
    "CUTOUT": "inner",
    "HOLE": "inner",
    "INLAY": "inlay",
    "GROOVE": "inlay",
    "POCKET": "inlay",
}

def classify_by_depth(depths: dict[int,int]) -> dict[int,str]:
    """
    Even depth -> outer, odd -> inner. You can override by layer later.
    """
    classes = {}
    for idx, d in depths.items():
        classes[idx] = "outer" if d%2==0 else "inner"
    return classes
