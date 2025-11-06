from __future__ import annotations

from shapely.geometry import Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree
from typing import List, Dict

def containment_depth(polys: list[Polygon]) -> dict[int, int]:
    """
    Compute nesting depth for each polygon index (0 = outermost).
    Uses a spatial index to reduce pairwise checks.
    """
    n = len(polys)
    depths = {i: 0 for i in range(n)}
    if n == 0:
        return depths

    tree = STRtree(polys)
    prepared = [prep(p) for p in polys]

    # Prefer bulk query if available (Shapely 2.x); else per-geom (Shapely 1.x)
    if hasattr(tree, "query_bulk"):
        ia, ib = tree.query_bulk(polys)
        for k in range(len(ia)):
            i = int(ia[k])
            j = int(ib[k])
            if i == j:
                continue
            if prepared[j].contains(polys[i]):
                depths[i] += 1
    else:
        # Fallback: use id mapping on query results (identity preserved in 1.x)
        for i, pi in enumerate(polys):
            candidates = tree.query(pi)
            if not isinstance(candidates, (list, tuple)):
                candidates = list(candidates)
            for cand in candidates:
                # Some implementations return indices instead of geometries
                if hasattr(cand, "geom_type"):
                    pj = cand
                else:
                    try:
                        j = int(cand)
                        pj = polys[j]
                    except Exception:
                        continue
                if pj is pi:
                    continue
                if pj.contains(pi):
                    depths[i] += 1
    return depths
