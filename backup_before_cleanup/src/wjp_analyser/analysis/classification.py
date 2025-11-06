from shapely.geometry import Polygon
from shapely.strtree import STRtree
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

def classify_by_depth_and_layers(
    polys: list[Polygon],
    depths: dict[int, int],
    line_geoms: list,
    line_layers: list[str],
) -> dict[int, str]:
    """Depth-based classes overridden by layer-name hints.

    For each polygon, we intersect its exterior with source DXF linework and
    tally layer names that match LAYER_HINTS keys (substring, case-insensitive).
    The majority hinted class overrides the depth-based class.
    """
    base = classify_by_depth(depths)
    if not polys or not line_geoms or not line_layers or len(line_geoms) != len(line_layers):
        return base

    # Spatial index to find lines near each polygon boundary
    tree = STRtree(line_geoms)

    def hint_for_layer(layer: str) -> str | None:
        up = (layer or "").upper()
        for key, val in LAYER_HINTS.items():
            if key in up:
                return val
        return None

    classes = dict(base)
    for i, p in enumerate(polys):
        boundary = p.exterior
        votes: Dict[str, int] = {}
        if hasattr(tree, "query_bulk"):
            ia, ib = tree.query_bulk([boundary], predicate="intersects")
            for k in range(len(ib)):
                j = int(ib[k])
                hint = hint_for_layer(line_layers[j])
                if hint:
                    votes[hint] = votes.get(hint, 0) + 1
        else:
            # Shapely 1.x fallback: id mapping with query
            id_to_idx = {id(g): idx for idx, g in enumerate(line_geoms)}
            candidates = tree.query(boundary)
            if not isinstance(candidates, (list, tuple)):
                candidates = list(candidates)
            for g in candidates:
                if hasattr(g, "geom_type"):
                    j = id_to_idx.get(id(g))
                    if j is None:
                        # As a fallback, locate by geometric equality
                        j = next((idx for idx, lg in enumerate(line_geoms) if lg.equals(g)), None)
                else:
                    try:
                        j = int(g)
                    except Exception:
                        j = None
                if j is None:
                    continue
                hint = hint_for_layer(line_layers[j])
                if hint:
                    votes[hint] = votes.get(hint, 0) + 1
        if votes:
            # Choose the class with max votes; break ties by keeping base
            winner = max(votes.items(), key=lambda kv: kv[1])[0]
            classes[i] = winner
    return classes
