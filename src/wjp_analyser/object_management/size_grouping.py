"""
Size-based Object Grouping & Layering Helpers
--------------------------------------------

Implements a simple, deterministic pipeline for:
 - identify_objects(dxf_path): extract closed contours as objects with basic geometry
 - group_objects(objects, ...): group by bounding-box size within tolerances
 - apply_layer_selection(...): move all objects in a selected group to a named layer

Notes
 - Uses DXFObjectManager for robust geometry extraction (LWPOLYLINE, POLYLINE, CIRCLE, ARC, LINE)
 - Grouping is size-first: canonical dims (W>=H) rounded to the chosen tolerance
 - Layer names derive from group size, e.g., "SQ_200" or "RECT_100x50"
 - Relayering uses entity handles via dxf_analyzer.relayer_entities_by_handle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import os

from .dxf_object_manager import DXFObjectManager
from ..analysis.dxf_analyzer import relayer_entities_by_handle


@dataclass
class ObjectInfo:
    id: str
    handle: Optional[str]
    layer: str
    bbox: Tuple[float, float, float, float]
    width: float
    height: float
    area: float
    perimeter: float
    centroid: Tuple[float, float]


def identify_objects(dxf_path: str) -> Dict[str, dict]:
    """Return object metadata keyed by ID like {'OBJ-001': {...}}.

    Metadata keys:
      group: None (to be set later)
      bbox: [width_mm, height_mm]
      area: mm^2
      perimeter: mm
      centroid: [x_mm, y_mm]
      layer: original DXF layer
      handle: entity handle (string) if available
    """
    mgr = DXFObjectManager()
    objs = mgr.load_dxf_objects(dxf_path)
    result: Dict[str, dict] = {}
    seq = 1
    for o in objs:
        # Only treat closed shapes as "objects" for grouping/layering
        if not o.geometry.is_closed and o.object_type.name not in ("CIRCLE",):
            continue
        oid = f"OBJ-{seq:03d}"
        seq += 1
        minx, miny, maxx, maxy = o.geometry.bounding_box
        width = float(maxx - minx)
        height = float(maxy - miny)
        handle = None
        try:
            handle = str(getattr(o.entity.dxf, "handle", None)) if hasattr(o, "entity") else None
        except Exception:
            handle = None
        result[oid] = {
            "group": None,
            "bbox": [round(width, 3), round(height, 3)],
            "area": round(float(o.geometry.area), 3),
            "perimeter": round(float(o.geometry.perimeter), 3),
            "centroid": [round(float(o.geometry.centroid[0]), 3), round(float(o.geometry.centroid[1]), 3)],
            "layer": str(o.metadata.layer_name or "0"),
            "handle": handle,
        }
    return result


def _canonical_dims(w: float, h: float) -> Tuple[float, float]:
    return (w, h) if w >= h else (h, w)


def _round_to_tol(x: float, tol: float) -> float:
    if tol <= 0:
        return x
    return round(round(x / tol) * tol, 3)


def group_objects(objects: Dict[str, dict], tol_mm: float = 1.0, tol_pct: Optional[float] = None,
                  use_area: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
    """Group objects by size with absolute or percent tolerance.

    Returns:
      groups: mapping 'G1' -> [obj_ids]
      group_meta: mapping 'G1' -> {'width_mm','height_mm','area_mm2','count','layer_name'}
    """
    items = []
    for oid, meta in objects.items():
        w, h = _canonical_dims(float(meta.get("bbox", [0, 0])[0]), float(meta.get("bbox", [0, 0])[1]))
        area = float(meta.get("area", 0.0))
        items.append((oid, w, h, area))

    groups: Dict[str, List[str]] = {}
    meta_out: Dict[str, dict] = {}
    gid = 1

    def within(a: float, b: float) -> bool:
        if tol_pct and tol_pct > 0:
            return abs(a - b) <= tol_pct * max(b, 1e-9)
        return abs(a - b) <= tol_mm

    buckets: Dict[Tuple[float, float], List[str]] = {}
    if use_area:
        # Group by rounded area when use_area=True
        tmp: Dict[float, List[str]] = {}
        for oid, w, h, area in items:
            key = _round_to_tol(area, tol_mm)
            tmp.setdefault(key, []).append(oid)
        for key, ids in tmp.items():
            gname = f"G{gid}"
            gid += 1
            groups[gname] = ids
            meta_out[gname] = {"width_mm": None, "height_mm": None, "area_mm2": key, "count": len(ids)}
        return groups, meta_out

    # Group by size (W,H)
    for oid, w, h, area in items:
        # try quantized key first for deterministic grouping
        qkey = (_round_to_tol(w, tol_mm), _round_to_tol(h, tol_mm))
        buckets.setdefault(qkey, []).append(oid)

    for (qw, qh), ids in buckets.items():
        # Validate within tolerance against first
        first = ids[0]
        fw, fh = _canonical_dims(*objects[first]["bbox"])
        ok_ids = []
        for oid in ids:
            w, h = _canonical_dims(*objects[oid]["bbox"])
            if within(w, fw) and within(h, fh):
                ok_ids.append(oid)
        if ok_ids:
            gname = f"G{gid}"
            gid += 1
            groups[gname] = ok_ids
            meta_out[gname] = {"width_mm": qw, "height_mm": qh, "area_mm2": None, "count": len(ok_ids)}

    return groups, meta_out


def _layer_name_for_group(width_mm: Optional[float], height_mm: Optional[float]) -> str:
    if width_mm is None or height_mm is None:
        return "GROUP"
    w = round(float(width_mm))
    h = round(float(height_mm))
    if abs(w - h) <= 1:  # within ~1 mm treated as square
        return f"SQ_{max(w, h)}"
    return f"RECT_{max(w, h)}x{min(w, h)}"


def apply_layer_selection(dxf_path: str, selected_obj_id: str, objects: Dict[str, dict],
                          groups: Dict[str, List[str]], group_meta: Dict[str, dict],
                          output_path: Optional[str] = None) -> Tuple[str, str, List[str]]:
    """Move all objects in the selected object's group to a new layer.

    Returns: (updated_dxf_path, layer_name, affected_object_ids)
    """
    # Find the group of the selected object
    target_group = None
    for gname, ids in groups.items():
        if selected_obj_id in ids:
            target_group = gname
            break
    if not target_group:
        raise ValueError(f"Object {selected_obj_id} not found in any group")

    # Determine target layer name from group size
    gmeta = group_meta.get(target_group, {})
    lname = _layer_name_for_group(gmeta.get("width_mm"), gmeta.get("height_mm"))

    # Collect handles for group
    handles: List[str] = []
    for oid in groups[target_group]:
        h = objects.get(oid, {}).get("handle")
        if h:
            handles.append(str(h))

    if not handles:
        raise ValueError("Selected group has no resolvable entity handles for relayering.")

    # Compute output path
    if not output_path:
        base, ext = os.path.splitext(dxf_path)
        output_path = f"{base}_{lname}{ext}"

    updated = relayer_entities_by_handle(dxf_path, handles, lname, output_path=output_path)
    return updated, lname, list(groups[target_group])

