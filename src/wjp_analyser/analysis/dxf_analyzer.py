"""
Waterjet DXF Analyzer - Unified Workflow with Caching
===================================================

This module provides comprehensive DXF analysis for waterjet cutting operations,
including geometric analysis, cost estimation, quality assessment, and optimization.
Now integrated with service architecture for path management, authentication, 
report generation, and results caching.

Core Workflow: Upload → Auth → Cache Check → Analysis → Layer Management → 
        Nesting → Toolpath → Costing

The legacy modular pipeline has been simplified into a single cohesive module.
All high-level steps remain accessible through the familiar public API used by
CLI/web tooling (`AnalyzeArgs`, `analyze_dxf`, `load_polys_and_classes`, etc.).

Key Features:
- Geometric analysis of DXF entities (lines, arcs, circles, polylines)
- Shape classification and grouping for efficient processing
- Cutting length and pierce point calculation 
- Material-specific cost estimation
- Quality assessment and validation
- Toolpath generation and optimization

Cache & Optimization:
- File content-based cache keys for deterministic results
- Selection filtering applied to cached results
- Efficient selective updates of cached data
- Cache miss tracking for optimization

Security Considerations:
- Input validation for DXF files
- Path sanitization for file operations  
- Error handling with user-friendly messages
- Logging of analysis operations and cache usage

Performance Optimizations:
- Smart caching of analysis results 
- Efficient geometric algorithms
- Memory management for large files
- Parallel processing where applicable

Author: WJP Analyser Team
Version: 0.1.1 
License: MIT
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Optional, TYPE_CHECKING

# Lazy imports for optional services (may require SQLAlchemy/database)
# Only import when actually needed to avoid import errors if SQLAlchemy unavailable
if TYPE_CHECKING:
    from ..services.path_manager import PathManager
    from ..services.auth_service import AuthService
    from ..services.report_generator import ReportGenerator
    from ..services.logging_service import LoggingService

import ezdxf
from shapely.geometry import Polygon

from .cache_utils import build_cache_key, filter_cached_report
from ..services.cache_service import CacheService

# G-code generation moved to a separate module/UI page. Not used here.


# ---------------------------------------------------------------------------
# Lightweight DXF diagnostics
# ---------------------------------------------------------------------------
def _collect_entity_counts(dxf_path: str) -> dict:
    """Return counts of entity dxftypes present in the DXF.

    Safe utility: if file can't be read, returns empty dict.
    """
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
    except Exception:
        return {}
    counts: dict = {}
    try:
        for e in msp:
            t = e.dxftype()
            counts[t] = counts.get(t, 0) + 1
    except Exception:
        pass
    return counts


# ---------------------------------------------------------------------------
# Mastery checklist helpers
# ---------------------------------------------------------------------------
def _list_layer_names(doc: ezdxf.EzDxf) -> list[str]:
    try:
        return [str(l.dxf.name) for l in doc.layers]
    except Exception:
        try:
            # Fallback for older ezdxf versions
            return [str(name) for name in doc.layers.names()]
        except Exception:
            return []


def _color_linetype_summary(doc: ezdxf.EzDxf) -> dict:
    """Summarize common metadata usage: colors, linetypes, lineweights."""
    res = {"colors": {}, "linetypes": {}, "lineweights": {}}
    try:
        msp = doc.modelspace()
        for e in msp:
            try:
                col = int(e.dxf.color) if hasattr(e.dxf, "color") else None
                if col is not None:
                    res["colors"][col] = res["colors"].get(col, 0) + 1
            except Exception:
                pass
            try:
                lt = str(e.dxf.linetype) if hasattr(e.dxf, "linetype") else None
                if lt:
                    res["linetypes"][lt] = res["linetypes"].get(lt, 0) + 1
            except Exception:
                pass
            try:
                lw = int(e.dxf.lineweight) if hasattr(e.dxf, "lineweight") else None
                if lw is not None:
                    res["lineweights"][lw] = res["lineweights"].get(lw, 0) + 1
            except Exception:
                pass
    except Exception:
        return {"colors": {}, "linetypes": {}, "lineweights": {}}
    return res


def _min_spacing_violations(components: list[dict], min_gap: float = 3.0) -> tuple[int, float]:
    """Return (violation_count, min_distance) between polygon pairs using shapely.

    Counts pairs closer than min_gap (in mm). Returns 0, inf if no components.
    """
    try:
        from shapely.geometry import Polygon as _Poly
    except Exception:
        return 0, float("inf")
    polys: list[_Poly] = []
    for comp in components or []:
        try:
            poly = _Poly(comp.get("points", []))
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception:
            continue
    if len(polys) < 2:
        return 0, float("inf")
    violations = 0
    min_d = float("inf")
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                d = polys[i].distance(polys[j])
            except Exception:
                continue
            if d < min_d:
                min_d = d
            if d < float(min_gap):
                violations += 1
    return violations, (min_d if min_d != float("inf") else 0.0)


def _estimate_min_corner_radius_all(components: list[dict]) -> float:
    """Estimate a minimal corner radius across all polygons using circumcircle radius per vertex.

    This is an approximation to flag potentially too-sharp corners; returns 0.0 if unknown.
    """
    import math
    def tri_radius(ax, ay, bx, by, cx, cy) -> float:
        # Circumcircle radius R = (abc) / (4A), where A is triangle area
        a = math.hypot(ax - bx, ay - by)
        b = math.hypot(bx - cx, by - cy)
        c = math.hypot(cx - ax, cy - ay)
        s = (a + b + c) / 2.0
        area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
        if area_sq <= 1e-18:
            return float("inf")
        area = math.sqrt(area_sq)
        R = (a * b * c) / max(1e-12, 4.0 * area)
        return R
    best = float("inf")
    for comp in components or []:
        pts = comp.get("points", [])
        if len(pts) < 3:
            continue
        # ensure non-duplicated end for iteration
        work = pts[:-1] if pts[0] == pts[-1] else pts
        n = len(work)
        for i in range(n):
            ax, ay = work[(i - 1) % n]
            bx, by = work[i]
            cx, cy = work[(i + 1) % n]
            r = tri_radius(ax, ay, bx, by, cx, cy)
            if r < best:
                best = r
    if best == float("inf"):
        return 0.0
    return float(best)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Polyline = List[Tuple[float, float]]
LayerBuckets = Dict[str, List[Polyline]]

LAYER_NAMES = ["OUTER", "INNER", "COMPLEX", "HOLE", "DECOR"]


# ---------------------------------------------------------------------------
# Dataclass matching the original public API
# ---------------------------------------------------------------------------
@dataclass
class AnalyzeArgs:
    material: str = "Generic Material"
    thickness: float = 10.0
    kerf: float = 1.0
    rate_per_m: float = 800.0
    pierce_cost: float = 5.0
    out: str = "out"
    sheet_width: float = 1000.0
    sheet_height: float = 1000.0
    # Legacy fields retained for CLI compatibility (no-op in simplified workflow)
    use_advanced_toolpath: bool = False
    rapid_speed: float = 10000.0
    cutting_speed: float = 1200.0
    pierce_time: float = 0.5
    optimize_rapids: bool = True
    optimize_direction: bool = True
    entry_strategy: str = "tangent"
    # Quality checks
    quality_tolerance: float = 2.0  # mm threshold for tiny segments
    shaky_threshold: int = 200      # vertex-count threshold for "shaky" polylines
    # Softening (smoothing/simplification) options
    soften_method: str = "none"      # one of: none, simplify, simplify_topo, chaikin, rdp, visvalingam, colinear, decimate, resample
    soften_tolerance: float = 0.2    # for simplify/rdp (mm)
    soften_iterations: int = 1       # for chaikin iterations
    soften_preserve_topology: bool = False  # for simplify (topology-safe)
    soften_visvalingam_area_mm2: float = 0.5  # min triangle area for VW method
    soften_colinear_angle_deg: float = 2.0    # treat angles within this as colinear
    soften_decimate_keep_every: int = 2       # keep every Nth vertex (>=2)
    soften_resample_step_mm: float = 1.0      # resample edge step in mm
    # Measurement-guided softening options
    soften_measure_min_segment_mm: float = 1.0
    soften_measure_max_deviation_mm: float = 0.2
    soften_measure_min_corner_radius_mm: float = 1.0
    soften_measure_snap_grid_mm: float = 0.0
    # Corner fillet options (bulge-based arcs in DXF)
    fillet_radius_mm: float = 0.0
    fillet_min_angle_deg: float = 135.0
    # Scaling options
    scale_mode: str = "auto"  # 'auto' | 'factor' | 'decade_fit'
    scale_factor: float = 1.0  # drawing units to mm when scale_mode='factor'
    # Decade fit options (scale by powers of 10 until target reached)
    scale_decade_base: float = 10.0
    scale_decade_direction: str = "auto"  # 'auto' | 'up' | 'down'
    scale_decade_max_steps: int = 6
    scale_decade_allow_overshoot: bool = True
    scale_decade_exact_fit: bool = False
    # Fit-to-frame normalization (post-units)
    normalize_mode: str = "none"  # 'none' | 'fit'
    target_frame_w_mm: float = 1000.0
    target_frame_h_mm: float = 1000.0
    frame_margin_mm: float = 0.0
    normalize_origin: bool = True
    require_fit_within_frame: bool = True
    frame_quantity: int = 1
    
    # Performance optimization flags (Phase 4)
    streaming_mode: bool = False  # Use streaming parser for large files
    early_simplify_tolerance: float = 0.0  # Early simplification tolerance (0 = disabled)
    min_segment_length: float = 0.0  # Minimum segment length to keep (0 = disabled)
    coordinate_precision: int = 3  # Decimal precision for coordinates (reduces memory)
    use_float32: bool = False  # Use float32 instead of float64 (reduces memory by ~50%)


# ---------------------------------------------------------------------------
# 1. DXF Analysis
# ---------------------------------------------------------------------------
def _extract_polylines(input_dxf: str) -> list[dict]:
    """Extract closed polygonal rings from a DXF, stitching segments when needed.

    Supports LWPOLYLINE (closed), approximates ARCs/CIRCLEs into segments and
    stitches with LINE segments to form closed loops where possible.
    """
    import math

    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()
    items: list[dict] = []

    # 1) Closed LWPOLYLINEs directly as polygons; open LWPOLYLINEs will be added to segment pool
    segs: list[tuple[tuple[float, float], tuple[float, float], str | None, str | None]] = []  # (p1, p2, layer, handle)

    def add_ring(pts: list[tuple[float, float]], layer_name: str | None, handle: str | None):
        if not pts:
            return
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        items.append({"points": pts, "handle": handle, "source_layer": layer_name})

    for entity in msp.query("LWPOLYLINE"):
        try:
            pts = [(float(v[0]), float(v[1])) for v in entity.get_points("xy")]
        except Exception:
            continue
        if not pts:
            continue
        layer_name = None
        handle = None
        try:
            layer_name = str(entity.dxf.layer)
        except Exception:
            pass
        try:
            handle = str(entity.dxf.handle)
        except Exception:
            pass
        if bool(entity.closed):
            add_ring(pts, layer_name, handle)
        else:
            # add segments to pool for potential stitching
            for a, b in zip(pts, pts[1:]):
                segs.append((a, b, layer_name, handle))

    # Classic POLYLINE (2D/3D) support
    for pl in msp.query("POLYLINE"):
        pts = []
        try:
            pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in pl.vertices()]
        except Exception:
            try:
                pts = [(float(v.dxf.x), float(v.dxf.y)) for v in pl.vertices()]
            except Exception:
                pts = []
        if not pts:
            continue
        layer_name = None
        handle = None
        try:
            layer_name = str(pl.dxf.layer)
        except Exception:
            pass
        try:
            handle = str(pl.dxf.handle)
        except Exception:
            pass
        is_closed = False
        try:
            is_closed = bool(pl.is_closed)
        except Exception:
            try:
                is_closed = bool(int(pl.dxf.flags) & 1)
            except Exception:
                is_closed = False
        if is_closed:
            add_ring(pts, layer_name, handle)
        else:
            for a, b in zip(pts, pts[1:]):
                segs.append((a, b, layer_name, handle))

    # 2) Add LINE segments
    for ln in msp.query("LINE"):
        try:
            p1 = (float(ln.dxf.start.x), float(ln.dxf.start.y))
            p2 = (float(ln.dxf.end.x), float(ln.dxf.end.y))
        except Exception:
            continue
        layer_name = None
        handle = None
        try:
            layer_name = str(ln.dxf.layer)
        except Exception:
            pass
        try:
            handle = str(ln.dxf.handle)
        except Exception:
            pass
        segs.append((p1, p2, layer_name, handle))

    # 3) Approximate ARCs and CIRCLEs into polyline segments and add to pool
    def arc_points(center, radius, start_deg, end_deg, max_seg_len=2.0):
        # Normalize angles
        s = math.radians(float(start_deg))
        e = math.radians(float(end_deg))
        # Ensure positive sweep
        while e < s:
            e += 2 * math.pi
        sweep = e - s
        n = max(8, int(max(1.0, (radius * sweep) / max(0.1, max_seg_len))))
        pts = []
        for i in range(n + 1):
            a = s + sweep * (i / n)
            pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
        return pts

    for arc in msp.query("ARC"):
        try:
            cx, cy = float(arc.dxf.center.x), float(arc.dxf.center.y)
            r = float(arc.dxf.radius)
            sdeg = float(arc.dxf.start_angle)
            edeg = float(arc.dxf.end_angle)
        except Exception:
            continue
        pts = arc_points((cx, cy), r, sdeg, edeg)
        layer_name = None
        handle = None
        try:
            layer_name = str(arc.dxf.layer)
        except Exception:
            pass
        try:
            handle = str(arc.dxf.handle)
        except Exception:
            pass
        for a, b in zip(pts, pts[1:]):
            segs.append((a, b, layer_name, handle))

    for cir in msp.query("CIRCLE"):
        try:
            cx, cy = float(cir.dxf.center.x), float(cir.dxf.center.y)
            r = float(cir.dxf.radius)
        except Exception:
            continue
        pts = arc_points((cx, cy), r, 0.0, 360.0)
        layer_name = None
        handle = None
        try:
            layer_name = str(cir.dxf.layer)
        except Exception:
            pass
        try:
            handle = str(cir.dxf.handle)
        except Exception:
            pass
        add_ring(pts, layer_name, handle)

    # SPLINE → approximate into line segments
    for spl in msp.query("SPLINE"):
        pts = []
        try:
            tool = spl.construction_tool()
            vtx = tool.approximate(128)
            pts = [(float(p.x), float(p.y)) for p in vtx]
        except Exception:
            try:
                pts = [(float(p.x), float(p.y)) for p in spl.approximate(segments=64)]
            except Exception:
                try:
                    pts = [(float(p.x), float(p.y)) for p in spl.approximate(64)]
                except Exception:
                    pts = []
        if len(pts) >= 2:
            layer_name = None
            handle = None
            try:
                layer_name = str(spl.dxf.layer)
            except Exception:
                pass
            try:
                handle = str(spl.dxf.handle)
            except Exception:
                pass
            for a, b in zip(pts, pts[1:]):
                segs.append((a, b, layer_name, handle))

    # ELLIPSE → approximate into line segments
    for el in msp.query("ELLIPSE"):
        pts = []
        try:
            pts = [(float(p.x), float(p.y)) for p in el.flattening(2.0)]
        except Exception:
            try:
                tool = el.construction_tool()
                vtx = tool.approximate(128)
                pts = [(float(p.x), float(p.y)) for p in vtx]
            except Exception:
                pts = []
        if len(pts) >= 2:
            layer_name = None
            handle = None
            try:
                layer_name = str(el.dxf.layer)
            except Exception:
                pass
            try:
                handle = str(el.dxf.handle)
            except Exception:
                pass
            # Close if full ellipse recognized
            is_closed = False
            try:
                is_closed = abs(float(el.dxf.start_param) - float(el.dxf.end_param)) < 1e-9
            except Exception:
                is_closed = False
            if is_closed and pts[0] != pts[-1]:
                pts.append(pts[0])
            if pts and pts[0] == pts[-1]:
                add_ring(pts, layer_name, handle)
            else:
                for a, b in zip(pts, pts[1:]):
                    segs.append((a, b, layer_name, handle))

    # 4) Attempt to stitch remaining segments into closed loops
    if segs:
        TOL = 0.25  # mm endpoint merge tolerance
        def key(pt):
            return (round(pt[0] / TOL), round(pt[1] / TOL))

        unused = [True] * len(segs)
        start_index = 0
        while True:
            # find first unused seg
            try:
                start_index = unused.index(True)
            except ValueError:
                break
            unused[start_index] = False
            a, b, layer_name, handle = segs[start_index]
            ring = [a, b]
            last = b
            iters = 0
            # greedily connect endpoints
            while iters < 10000:
                found = False
                for i in range(len(segs)):
                    if not unused[i]:
                        continue
                    p, q, lyr, h = segs[i]
                    if key(p) == key(last):
                        ring.append(q)
                        last = q
                        layer_name = layer_name or lyr
                        handle = handle or h
                        unused[i] = False
                        found = True
                        break
                    if key(q) == key(last):
                        ring.append(p)
                        last = p
                        layer_name = layer_name or lyr
                        handle = handle or h
                        unused[i] = False
                        found = True
                        break
                iters += 1
                # closed?
                if key(ring[0]) == key(ring[-1]) and len(ring) >= 4:
                    # ensure exact closure
                    ring[-1] = ring[0]
                    add_ring(ring, layer_name, handle)
                    break
                if not found:
                    # cannot extend further; drop if not closed
                    break

    # 5) Handle block INSERTs by expanding virtual entities and repeating key logic
    # Note: placed after direct entities to avoid duplication; we only add geometry if still empty
    if not items:
        try:
            for ins in msp.query("INSERT"):
                try:
                    for ent in ins.virtual_entities():
                        et = ent.dxftype()
                        layer_name = None
                        handle = None
                        try:
                            layer_name = str(ent.dxf.layer)
                        except Exception:
                            pass
                        try:
                            handle = str(ent.dxf.handle)
                        except Exception:
                            pass
                        if et == "LWPOLYLINE":
                            try:
                                pts = [(float(v[0]), float(v[1])) for v in ent.get_points("xy")]
                            except Exception:
                                pts = []
                            if pts:
                                if pts[0] != pts[-1]:
                                    pts.append(pts[0])
                                items.append({"points": pts, "handle": handle, "source_layer": layer_name})
                        elif et == "LINE":
                            try:
                                p1 = (float(ent.dxf.start.x), float(ent.dxf.start.y))
                                p2 = (float(ent.dxf.end.x), float(ent.dxf.end.y))
                                segs.append((p1, p2, layer_name, handle))
                            except Exception:
                                pass
                        elif et == "ARC":
                            try:
                                cx, cy = float(ent.dxf.center.x), float(ent.dxf.center.y)
                                r = float(ent.dxf.radius)
                                sdeg = float(ent.dxf.start_angle)
                                edeg = float(ent.dxf.end_angle)
                                pts = arc_points((cx, cy), r, sdeg, edeg)
                                for a, b in zip(pts, pts[1:]):
                                    segs.append((a, b, layer_name, handle))
                            except Exception:
                                pass
                        elif et == "CIRCLE":
                            try:
                                cx, cy = float(ent.dxf.center.x), float(ent.dxf.center.y)
                                r = float(ent.dxf.radius)
                                pts = arc_points((cx, cy), r, 0.0, 360.0)
                                if pts[0] != pts[-1]:
                                    pts.append(pts[0])
                                items.append({"points": pts, "handle": handle, "source_layer": layer_name})
                            except Exception:
                                pass
                        elif et == "SPLINE":
                            pts2 = []
                            try:
                                tool = ent.construction_tool(); vtx = tool.approximate(128)
                                pts2 = [(float(p.x), float(p.y)) for p in vtx]
                            except Exception:
                                try:
                                    pts2 = [(float(p.x), float(p.y)) for p in ent.approximate(64)]
                                except Exception:
                                    try:
                                        pts2 = [(float(p.x), float(p.y)) for p in ent.approximate(segments=64)]
                                    except Exception:
                                        pts2 = []
                            for a, b in zip(pts2, pts2[1:]):
                                segs.append((a, b, layer_name, handle))
                        elif et == "ELLIPSE":
                            pts2 = []
                            try:
                                pts2 = [(float(p.x), float(p.y)) for p in ent.flattening(2.0)]
                            except Exception:
                                try:
                                    tool = ent.construction_tool(); vtx = tool.approximate(128)
                                    pts2 = [(float(p.x), float(p.y)) for p in vtx]
                                except Exception:
                                    pts2 = []
                            if pts2 and pts2[0] == pts2[-1]:
                                items.append({"points": pts2, "handle": handle, "source_layer": layer_name})
                            else:
                                for a, b in zip(pts2, pts2[1:]):
                                    segs.append((a, b, layer_name, handle))
                        # other types can be added here if needed
                except Exception:
                    continue
            # after expanding INSERTs, try stitching added segments
            if segs and not items:
                TOL = 0.25
                def key2(pt):
                    return (round(pt[0] / TOL), round(pt[1] / TOL))
                unused = [True] * len(segs)
                while True:
                    try:
                        si = unused.index(True)
                    except ValueError:
                        break
                    unused[si] = False
                    a, b, layer_name, handle = segs[si]
                    ring = [a, b]
                    last = b
                    iters = 0
                    while iters < 10000:
                        found = False
                        for i in range(len(segs)):
                            if not unused[i]:
                                continue
                            p, q, lyr, h = segs[i]
                            if key2(p) == key2(last):
                                ring.append(q); last = q; layer_name = layer_name or lyr; handle = handle or h; unused[i] = False; found = True; break
                            if key2(q) == key2(last):
                                ring.append(p); last = p; layer_name = layer_name or lyr; handle = handle or h; unused[i] = False; found = True; break
                        iters += 1
                        if key2(ring[0]) == key2(ring[-1]) and len(ring) >= 4:
                            ring[-1] = ring[0]
                            items.append({"points": ring, "handle": handle, "source_layer": layer_name})
                            break
                        if not found:
                            break
        except Exception:
            pass

    print(f"Found {len(items)} objects (stitched from entities)")
    return items


# ---------------------------------------------------------------------------
# 2. Layer Manager
# ---------------------------------------------------------------------------
def _classify_polylines(polylines: Iterable) -> tuple[LayerBuckets, list[dict]]:
    buckets: LayerBuckets = {name: [] for name in LAYER_NAMES}
    component_records: list[dict] = []

    valid_items: list[tuple[Polyline, Polygon, float, dict]] = []

    # Support both raw point lists and dicts with metadata
    for item in polylines:
        if isinstance(item, dict):
            pts = item.get("points", [])
            meta = {"handle": item.get("handle"), "source_layer": item.get("source_layer")}
        else:
            pts = item
            meta = {"handle": None, "source_layer": None}
        try:
            poly = Polygon(pts)
        except Exception:
            continue
        if not poly.is_valid or poly.area == 0:
            continue
        valid_items.append((list(pts), poly, abs(poly.area), meta))

    if not valid_items:
        return buckets, component_records

    # largest area considered outer boundary
    valid_items.sort(key=lambda item: item[2], reverse=True)
    outer_boundary = valid_items[0][1]

    for pts, poly, abs_area, meta in valid_items:
        vertex_count = max(0, len(pts) - 1)
        perimeter = float(poly.length)

        if poly.area < 0:
            layer_name = "INNER"
        elif vertex_count > 200:
            layer_name = "COMPLEX"
        elif poly.equals(outer_boundary):
            layer_name = "OUTER"
        elif outer_boundary.contains(poly):
            layer_name = "HOLE"
        else:
            layer_name = "DECOR"

        buckets[layer_name].append(pts)

        component_records.append({
            "id": len(component_records),
            "layer": layer_name,
            "points": pts,
            "area": abs_area,
            "perimeter": perimeter,
            "vertex_count": vertex_count,
            "handle": meta.get("handle"),
            "source_layer": meta.get("source_layer"),
        })

    return buckets, component_records


# ---------------------------------------------------------------------------
# 0. Optional softening/simplification
# ---------------------------------------------------------------------------
def _chaikin_once(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(pts) < 3:
        return pts
    closed = pts[0] == pts[-1]
    work = pts[:-1] if closed else pts[:]
    out: List[Tuple[float, float]] = []
    n = len(work)
    for i in range(n):
        p0 = work[i]
        p1 = work[(i + 1) % n]
        qx = 0.75 * p0[0] + 0.25 * p1[0]
        qy = 0.75 * p0[1] + 0.25 * p1[1]
        rx = 0.25 * p0[0] + 0.75 * p1[0]
        ry = 0.25 * p0[1] + 0.75 * p1[1]
        out.append((qx, qy))
        out.append((rx, ry))
    if closed:
        out.append(out[0])
    return out


def _apply_soften(polys: Iterable[dict], method: str, tol: float, iterations: int) -> List[dict]:
    from shapely.geometry import LineString
    import math

    def _is_closed(points: List[Tuple[float, float]]) -> bool:
        return len(points) >= 2 and points[0] == points[-1]

    def _ensure_closed(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not points:
            return points
        if points[0] != points[-1]:
            return points + [points[0]]
        return points

    def _visvalingam(points: List[Tuple[float, float]], area_thr: float) -> List[Tuple[float, float]]:
        if len(points) <= 3:
            return points
        closed = _is_closed(points)
        work = points[:-1] if closed else points[:]
        n = len(work)
        # Compute initial triangle areas for internal points
        def tri_area(a, b, c) -> float:
            return abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])) * 0.5)
        indices = list(range(n))
        # Use iterative removal by minimal area above threshold
        while True:
            if len(indices) <= 2:
                break
            min_area = None
            min_idx_pos = None
            # skip endpoints for open; for closed treat circular
            for pos in range(len(indices)):
                if not closed and (pos == 0 or pos == len(indices)-1):
                    continue
                i_prev = indices[(pos-1) % len(indices)]
                i = indices[pos]
                i_next = indices[(pos+1) % len(indices)]
                area = tri_area(work[i_prev], work[i], work[i_next])
                if min_area is None or area < min_area:
                    min_area = area
                    min_idx_pos = pos
            if min_area is None or min_area >= area_thr:
                break
            # remove the point at min_idx_pos
            indices.pop(min_idx_pos)
            # Safety to avoid infinite loop
            if len(indices) <= 2:
                break
        result = [work[i] for i in indices]
        return _ensure_closed(result) if closed else result

    def _merge_colinear(points: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
        if len(points) < 3:
            return points
        closed = _is_closed(points)
        work = points[:-1] if closed else points[:]
        thr = math.radians(max(0.0, angle_deg))
        def angle_between(a, b, c) -> float:
            v1x, v1y = b[0]-a[0], b[1]-a[1]
            v2x, v2y = c[0]-b[0], c[1]-b[1]
            d1 = math.hypot(v1x, v1y)
            d2 = math.hypot(v2x, v2y)
            if d1 < 1e-12 or d2 < 1e-12:
                return 0.0
            v1x /= d1; v1y /= d1; v2x /= d2; v2y /= d2
            dot = max(-1.0, min(1.0, v1x*v2x + v1y*v2y))
            return math.acos(dot)  # 0 for colinear straight line
        keep = []
        for i in range(len(work)):
            if not closed and (i == 0 or i == len(work)-1):
                keep.append(work[i])
                continue
            a = work[(i-1) % len(work)]
            b = work[i]
            c = work[(i+1) % len(work)]
            ang = angle_between(a, b, c)
            # if angle close to pi (straight), drop b
            if abs(math.pi - ang) <= thr:
                continue
            keep.append(b)
        if closed:
            keep = _ensure_closed(keep)
        return keep

    def _decimate(points: List[Tuple[float, float]], n_keep_every: int) -> List[Tuple[float, float]]:
        if len(points) <= 3 or n_keep_every <= 1:
            return points
        closed = _is_closed(points)
        work = points[:-1] if closed else points[:]
        out = [work[i] for i in range(0, len(work), n_keep_every)]
        # ensure last vertex is kept for open shapes
        if not closed and out[-1] != work[-1]:
            out.append(work[-1])
        return _ensure_closed(out) if closed else out

    def _resample(points: List[Tuple[float, float]], step_mm: float) -> List[Tuple[float, float]]:
        if len(points) < 2 or step_mm <= 0:
            return points
        closed = _is_closed(points)
        work = points[:-1] if closed else points[:]
        # build cumulative lengths
        segs: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        total = 0.0
        for a, b in zip(work, work[1:]+([work[0]] if closed else [])):
            dx = b[0]-a[0]; dy = b[1]-a[1]
            d = math.hypot(dx, dy)
            segs.append((a, b, d))
            total += d
        if total == 0:
            return points
        out: List[Tuple[float, float]] = []
        dist = 0.0
        si = 0
        a, b, d = segs[0]
        while dist <= total + 1e-9:
            # move along segments to reach current dist
            remaining = dist
            acc = 0.0
            for si in range(len(segs)):
                a, b, d = segs[si]
                if acc + d >= remaining:
                    t = 0.0 if d == 0 else (remaining - acc) / d
                    x = a[0] + t * (b[0] - a[0])
                    y = a[1] + t * (b[1] - a[1])
                    out.append((x, y))
                    break
                acc += d
            dist += step_mm
        # ensure last point closes/open ends maintained
        if closed:
            out = _ensure_closed(out)
        else:
            if out[-1] != work[-1]:
                out.append(work[-1])
        return out
    out: List[dict] = []
    method = (method or "none").lower()
    for item in polys:
        pts = list(item.get("points", []))
        if len(pts) < 3:
            out.append(item)
            continue
        if method == "simplify":
            try:
                ls = LineString(pts)
                simp = ls.simplify(tol, preserve_topology=False)
                new_pts = list(map(tuple, simp.coords))
                if new_pts and new_pts[0] != new_pts[-1]:
                    new_pts.append(new_pts[0])
                item = {**item, "points": new_pts}
            except Exception:
                pass
        elif method == "simplify_topo" or method == "rdp_topo":
            try:
                ls = LineString(pts)
                simp = ls.simplify(tol, preserve_topology=True)
                new_pts = list(map(tuple, simp.coords))
                if new_pts and new_pts[0] != new_pts[-1]:
                    new_pts.append(new_pts[0])
                item = {**item, "points": new_pts}
            except Exception:
                pass
        elif method == "rdp":
            try:
                ls = LineString(pts)
                simp = ls.simplify(tol, preserve_topology=False)
                new_pts = list(map(tuple, simp.coords))
                if new_pts and new_pts[0] != new_pts[-1]:
                    new_pts.append(new_pts[0])
                item = {**item, "points": new_pts}
            except Exception:
                pass
        elif method == "chaikin":
            new_pts = pts[:]
            for _ in range(max(1, int(iterations))):
                new_pts = _chaikin_once(new_pts)
            item = {**item, "points": new_pts}
        elif method == "visvalingam":
            try:
                area_thr = float(getattr(item, "_vw_area_thr", tol))  # tol used as proxy unless provided
            except Exception:
                area_thr = tol
            new_pts = _visvalingam(pts, max(0.0, area_thr))
            item = {**item, "points": new_pts}
        elif method == "measurement":
            # Remove tiny segments, enforce min corner radius, optional grid snap
            min_seg = float(item.get("_min_seg_mm", 1.0)) if isinstance(item, dict) else 1.0
            max_dev = float(item.get("_max_dev_mm", 0.2)) if isinstance(item, dict) else 0.2
            min_corner = float(item.get("_min_corner_r_mm", 1.0)) if isinstance(item, dict) else 1.0
            snap = float(item.get("_snap_grid_mm", 0.0)) if isinstance(item, dict) else 0.0
            # 1) collapse short edges
            collapsed: List[Tuple[float, float]] = []
            acc = pts[:]
            if _is_closed(acc):
                acc = acc[:-1]
                closed2 = True
            else:
                closed2 = False
            i = 0
            while i < len(acc):
                a = acc[i]
                b = acc[(i+1) % len(acc)] if i+1 < len(acc) else (acc[0] if closed2 else None)
                if b is None:
                    collapsed.append(a)
                    break
                d = math.hypot(b[0]-a[0], b[1]-a[1])
                if d < max(1e-9, min_seg):
                    # skip b by merging midpoint
                    mid = ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)
                    collapsed.append(mid)
                    i += 2
                else:
                    collapsed.append(a)
                    i += 1
            if closed2 and (not collapsed or collapsed[0] != collapsed[-1]):
                collapsed.append(collapsed[0])
            # 2) topology-preserving simplification by max deviation
            try:
                ls2 = LineString(collapsed)
                simp2 = ls2.simplify(max_dev, preserve_topology=True)
                collapsed = list(map(tuple, simp2.coords))
            except Exception:
                pass
            # 3) enforce min corner radius by light Chaikin if below threshold
            def _estimate_min_radius_local(p):
                if len(p) < 3:
                    return float("inf")
                work = p[:-1] if _is_closed(p) else p
                best = float("inf")
                for j in range(1, len(work)-1):
                    ax, ay = work[j-1]
                    bx, by = work[j]
                    cx, cy = work[j+1]
                    # same tri radius as above
                    a = math.hypot(ax - bx, ay - by)
                    b = math.hypot(bx - cx, by - cy)
                    c = math.hypot(cx - ax, cy - ay)
                    s = (a + b + c) / 2.0
                    area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
                    if area_sq <= 1e-18:
                        continue
                    area = math.sqrt(area_sq)
                    R = (a * b * c) / max(1e-12, 4.0 * area)
                    best = min(best, R)
                return best if best != float("inf") else 0.0
            r_now = _estimate_min_radius_local(collapsed)
            new_pts = collapsed
            if r_now < min_corner:
                # one pass chaikin to soften corners
                new_pts = _chaikin_once(new_pts)
            # 4) grid snap
            if snap > 0:
                snapped = []
                for x, y in new_pts:
                    sx = round(x / snap) * snap
                    sy = round(y / snap) * snap
                    snapped.append((sx, sy))
                new_pts = snapped
            new_pts = _ensure_closed(new_pts)
            item = {**item, "points": new_pts}
        elif method == "colinear":
            ang_deg = float(item.get("_colinear_angle_deg", 2.0)) if isinstance(item, dict) else 2.0
            new_pts = _merge_colinear(pts, ang_deg)
            item = {**item, "points": new_pts}
        elif method == "decimate":
            keep_every = int(item.get("_decimate_n", 2)) if isinstance(item, dict) else 2
            keep_every = max(2, keep_every)
            new_pts = _decimate(pts, keep_every)
            item = {**item, "points": new_pts}
        elif method == "resample":
            step = float(item.get("_resample_step", tol)) if isinstance(item, dict) else tol
            step = max(1e-6, step)
            new_pts = _resample(pts, step)
            item = {**item, "points": new_pts}
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Corner filleting (DXF LWPOLYLINE bulge-based arcs)
# ---------------------------------------------------------------------------
def _fillet_polyline_points(pts: List[Tuple[float, float]], closed: bool, r: float, min_angle_deg: float) -> List[Tuple[float, float, float]]:
    import math
    if len(pts) < 3 or r <= 0:
        # Return as-is with zero bulge
        return [(x, y, 0.0) for x, y in pts]

    def norm(vx, vy):
        d = math.hypot(vx, vy)
        return (vx / d, vy / d) if d > 1e-12 else (0.0, 0.0)

    def dot(ax, ay, bx, by):
        return ax * bx + ay * by

    def crossz(ax, ay, bx, by):
        return ax * by - ay * bx

    min_angle = math.radians(min_angle_deg)
    n = len(pts)
    out: List[Tuple[float, float, float]] = []

    # For closed, treat indices circularly; for open, leave endpoints unchanged
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        p_prev = pts[prev_i]
        p_curr = pts[i]
        p_next = pts[next_i]

        if (not closed) and (i == 0 or i == n - 1):
            # Keep endpoints
            out.append((p_curr[0], p_curr[1], 0.0))
            continue

        # Directions
        ax, ay = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
        bx, by = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
        a_nx, a_ny = norm(ax, ay)
        b_nx, b_ny = norm(bx, by)
        # Interior angle between incoming and outgoing
        # Angle between -a and b
        cosa = dot(-a_nx, -a_ny, b_nx, b_ny)
        cosa = max(-1.0, min(1.0, cosa))
        alpha = math.acos(cosa)
        if not (alpha < min_angle):
            # Not sharp enough; keep original vertex
            out.append((p_curr[0], p_curr[1], 0.0))
            continue

        # Distance from corner along each leg
        t = r * math.tan(alpha / 2.0)
        len1 = math.hypot(ax, ay)
        len2 = math.hypot(bx, by)
        if t <= 1e-9 or t >= len1 or t >= len2:
            out.append((p_curr[0], p_curr[1], 0.0))
            continue

        # Fillet tangent points
        t1x = p_curr[0] - a_nx * t
        t1y = p_curr[1] - a_ny * t
        t2x = p_curr[0] + b_nx * t
        t2y = p_curr[1] + b_ny * t

        # Bulge: tan(included_angle/4) with sign of orientation
        gamma = math.pi - alpha
        bul = math.tan(gamma / 4.0)
        orient = crossz(a_nx, a_ny, b_nx, b_ny)
        if orient < 0:
            bul = -bul

        # Append T1 with bulge to T2, then T2 (bulge 0)
        out.append((t1x, t1y, bul))
        out.append((t2x, t2y, 0.0))

    return out


def _apply_fillet_to_file(input_dxf: str, output_dxf: str, r: float, min_angle_deg: float) -> str:
    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()
    for e in list(msp.query("LWPOLYLINE")):
        try:
            closed = bool(e.closed)
            pts = [(float(v[0]), float(v[1])) for v in e.get_points("xy")]
            new_xyb = _fillet_polyline_points(pts, closed, r, min_angle_deg)
            # Replace points with bulges
            e.set_points(new_xyb, format="xyb")
            e.closed = closed
        except Exception:
            continue
    doc.saveas(output_dxf)
    return output_dxf


def _units_to_mm(insunits: int) -> float:
    # DXF $INSUNITS codes: 1=in, 2=ft, 3=mi, 4=mm, 5=cm, 6=m, 7=km, 8=us_survey_ft, etc.
    mapping = {
        0: 1.0,      # Unitless (assume mm)
        1: 25.4,     # Inches → mm
        2: 304.8,    # Feet → mm
        3: 1609344,  # Miles → mm
        4: 1.0,      # Millimeters
        5: 10.0,     # Centimeters → mm
        6: 1000.0,   # Meters → mm
        7: 1.0e6,    # Kilometers → mm
        8: 304.8006096,  # US survey foot → mm
        9: 33.3333333333, # yard → mm
        10: 0.0000254,    # Angstrom → mm
        11: 0.001,        # Nanometer → mm
        12: 0.01,         # Micron → mm
        13: 1.0e6,        # Decimeter? (DXF uses 5 for cm; 13=decimeter per some refs) → 100 mm, but safer to 1.0 keep unknowns
    }
    return float(mapping.get(int(insunits or 0), 1.0))


def _scale_polylines(items: list[dict], factor: float) -> list[dict]:
    if not items or abs(factor - 1.0) < 1e-12:
        return items
    scaled: list[dict] = []
    for it in items:
        pts = it.get("points", [])
        new_pts = [(float(x) * factor, float(y) * factor) for x, y in pts]
        scaled.append({**it, "points": new_pts})
    return scaled


def _compute_bbox(items: list[dict]) -> dict:
    mnx = mny = None
    mxx = mxy = None
    count = 0
    for it in items or []:
        pts = it.get("points", [])
        for x, y in pts:
            if mnx is None or x < mnx:
                mnx = x
            if mny is None or y < mny:
                mny = y
            if mxx is None or x > mxx:
                mxx = x
            if mxy is None or y > mxy:
                mxy = y
        if pts:
            count += 1
    if mnx is None:
        return {"minx": 0.0, "miny": 0.0, "maxx": 0.0, "maxy": 0.0, "width": 0.0, "height": 0.0, "count": 0}
    width = mxx - mnx
    height = mxy - mny
    return {"minx": mnx, "miny": mny, "maxx": mxx, "maxy": mxy, "width": width, "height": height, "count": count}


def _scale_translate(items: list[dict], factor: float, dx: float, dy: float) -> list[dict]:
    if not items:
        return items
    out: list[dict] = []
    for it in items:
        pts = it.get("points", [])
        new_pts = [((float(x) * factor) + dx, (float(y) * factor) + dy) for x, y in pts]
        out.append({**it, "points": new_pts})
    return out


def _apply_decade_fit(
    items: list[dict],
    bbox0: dict,
    target_w: float,
    target_h: float,
    margin: float,
    base: float,
    direction: str,
    max_steps: int,
    allow_overshoot: bool,
    exact_fit: bool,
    translate_origin: bool,
) -> tuple[list[dict], dict]:
    import math
    meta: dict = {}
    if bbox0.get("width", 0.0) <= 0 or bbox0.get("height", 0.0) <= 0:
        return items, {"mode": "decade_fit", "skipped": True, "reason": "zero_size"}
    inner_w = max(1e-9, float(target_w) - 2.0 * float(margin))
    inner_h = max(1e-9, float(target_h) - 2.0 * float(margin))
    target_dim = min(inner_w, inner_h)
    max_dim = max(bbox0["width"], bbox0["height"])
    if target_dim <= 0 or max_dim <= 0:
        return items, {"mode": "decade_fit", "skipped": True, "reason": "invalid_target"}
    f_exact = target_dim / max_dim
    if f_exact == 1.0:
        return items, {
            "mode": "decade_fit",
            "base": base,
            "steps": 0,
            "factor_exact": f_exact,
            "factor_decade": 1.0,
            "applied_factor": 1.0,
            "overshoot": False,
            "residual_applied": False,
        }
    # Determine exponent n for base**n
    if direction not in ("auto", "up", "down"):
        direction = "auto"
    n: int
    if direction == "up" or (direction == "auto" and f_exact >= 1.0):
        n = math.ceil(math.log(f_exact, base))
    else:  # down or auto with f_exact < 1
        n = math.floor(math.log(f_exact, base))
    # clamp steps
    if n > max_steps:
        n = max_steps
    if n < -max_steps:
        n = -max_steps
    f_decade = float(base) ** float(n)
    overshoot = (max_dim * f_decade) > target_dim if f_exact >= 1.0 else (max_dim * f_decade) < target_dim
    applied_factor = f_decade
    if not allow_overshoot and overshoot:
        # step back one decade
        if n != 0:
            n = n - 1 if f_exact >= 1.0 else n + 1
            f_decade = float(base) ** float(n)
            applied_factor = f_decade
            overshoot = False
    residual_applied = False
    if exact_fit and f_decade != 0:
        applied_factor = f_decade * (f_exact / f_decade)
        residual_applied = True
    dx = dy = 0.0
    if translate_origin:
        dx = -bbox0["minx"] * applied_factor + margin
        dy = -bbox0["miny"] * applied_factor + margin
    new_items = _scale_translate(items, applied_factor, dx, dy)
    bbox1 = _compute_bbox(new_items)
    meta = {
        "mode": "decade_fit",
        "base": base,
        "steps": n,
        "factor_exact": f_exact,
        "factor_decade": f_decade,
        "applied_factor": applied_factor,
        "overshoot": overshoot,
        "residual_applied": residual_applied,
        "bbox_before": bbox0,
        "bbox_after": bbox1,
    }
    return new_items, meta


def group_similar_objects(
    components: list[dict],
    area_tol: float = 0.01,
    peri_tol: float = 0.01,
    circ_tol: float = 0.05,
    ar_tol: float = 0.02,
) -> tuple[Dict[str, dict], list[dict]]:
    """Group components by geometric similarity using the GroupManager.
    
    Returns both the group dictionary and updated components list.
    """
    from .group_manager import create_group_manager
    
    # Create and initialize the group manager
    manager = create_group_manager(components)
    
    # Get the exported group data
    group_data = manager.export_group_data()
    
    # Update component group assignments
    for comp in components:
        # Find the most specific group this component belongs to
        assigned = False
        for group_name, group_info in group_data['groups'].items():
            if comp.get('id') in group_info.get('metadata', {}).get('component_ids', []):
                comp['group'] = group_name
                comp['layer'] = group_info['layer']
                assigned = True
                break
                
        if not assigned:
            comp['group'] = 'Ungrouped'
            
    return group_data['groups'], components


def generate_intelligent_group_name(group_meta: dict, group_index: int) -> str:
    """Generate descriptive group names based on object properties."""
    avg_area = group_meta.get("avg_area", 0)
    complexity = group_meta.get("complexity", "simple")
    vcount = group_meta.get("vcount", 0)
    circularity = group_meta.get("avg_circ", 0)
    count = group_meta.get("count", 1)
    
    # Size classification
    if avg_area < 100:
        size_class = "Small"
    elif avg_area < 1000:
        size_class = "Medium"
    else:
        size_class = "Large"
    
    # Shape classification based on circularity and vertex count
    if circularity > 0.8:
        shape_type = "Circle"
        size_info = f"{avg_area:.0f}mm²"
    elif vcount <= 4 and circularity < 0.3:
        shape_type = "Rectangle"
        size_info = f"{avg_area:.0f}mm²"
    elif vcount <= 6:
        shape_type = "Polygon"
        size_info = f"{vcount}sides"
    elif complexity == "simple":
        shape_type = "Simple"
        size_info = f"{avg_area:.0f}mm²"
    elif complexity == "moderate":
        shape_type = "Moderate"
        size_info = f"{avg_area:.0f}mm²"
    else:
        shape_type = "Complex"
        size_info = f"{avg_area:.0f}mm²"
    
    # Generate descriptive name
    if count > 1:
        return f"{size_class}_{shape_type}_{size_info}_x{count}"
    else:
        return f"{size_class}_{shape_type}_{size_info}"


def generate_intelligent_object_name(obj: dict, obj_index: int) -> str:
    """Generate descriptive object names based on individual object properties."""
    area = obj.get("area", 0)
    perimeter = obj.get("perimeter", 0)
    vcount = obj.get("vertex_count", 0)
    layer = obj.get("layer", "Unknown")
    
    # Calculate circularity
    circularity = (4.0 * 3.14159 * area) / (perimeter * perimeter + 1e-9) if perimeter > 0 else 0
    
    # Determine complexity
    if vcount <= 4:
        complexity = "Simple"
    elif vcount <= 20:
        complexity = "Moderate"
    else:
        complexity = "Complex"
    
    # Size classification
    if area < 100:
        size_class = "Small"
    elif area < 1000:
        size_class = "Medium"
    else:
        size_class = "Large"
    
    # Shape classification
    if circularity > 0.8:
        shape_type = "Circle"
    elif vcount <= 4 and circularity < 0.3:
        shape_type = "Rectangle"
    elif vcount <= 6:
        shape_type = "Polygon"
    else:
        shape_type = "Complex"
    
    # Generate descriptive name
    return f"Object{obj_index}_{size_class}_{shape_type}_{area:.0f}mm²"


def _extract_points_from_entity(entity) -> list[tuple[float, float]]:
    """Extract points from DXF entity."""
    points = []
    
    if entity.dxftype() == "LWPOLYLINE":
        # Lightweight polyline
        for point in entity.get_points():
            points.append((point[0], point[1]))
    elif entity.dxftype() == "POLYLINE":
        # Heavy polyline
        for vertex in entity.vertices:
            points.append((vertex.dxf.location.x, vertex.dxf.location.y))
    elif entity.dxftype() == "LINE":
        # Line
        points.append((entity.dxf.start.x, entity.dxf.start.y))
        points.append((entity.dxf.end.x, entity.dxf.end.y))
    elif entity.dxftype() == "ARC":
        # Arc - approximate with points
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        
        # Generate points along the arc
        num_points = max(8, int(abs(end_angle - start_angle) * 180 / math.pi))
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((x, y))
    elif entity.dxftype() == "CIRCLE":
        # Circle - approximate with points
        center = entity.dxf.center
        radius = entity.dxf.radius
        
        # Generate points around the circle
        num_points = 32
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((x, y))
    
    return points


def _calculate_area(points: list[tuple[float, float]]) -> float:
    """Calculate area using shoelace formula."""
    if len(points) < 3:
        return 0.0
    
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _calculate_perimeter(points: list[tuple[float, float]]) -> float:
    """Calculate perimeter by summing distances between consecutive points."""
    if len(points) < 2:
        return 0.0
    
    perimeter = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        dx = points[j][0] - points[i][0]
        dy = points[j][1] - points[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)
    return perimeter


def _layers_from_components(components: Iterable[dict]) -> LayerBuckets:
    buckets: LayerBuckets = {name: [] for name in LAYER_NAMES}
    for comp in components:
        buckets.setdefault(comp["layer"], []).append(comp["points"])
    return buckets


# G-code helpers removed from analyzer


# ---------------------------------------------------------------------------
# 3. Nesting (simple bounding-box nesting)
# ---------------------------------------------------------------------------
def _simple_nesting(layer_groups: LayerBuckets, sheet_size: Tuple[float, float]) -> list[tuple[Polyline, Tuple[float, float]]]:
    sheet_w, sheet_h = sheet_size
    x_offset, y_offset = 0.0, 0.0
    spacing = 20.0  # mm gap between objects

    placements: list[tuple[Polyline, Tuple[float, float]]] = []
    for polys in layer_groups.values():
        for poly in polys:
            try:
                bounds = Polygon(poly).bounds
            except Exception:
                continue
            minx, miny, maxx, maxy = bounds
            width, height = maxx - minx, maxy - miny

            if x_offset + width > sheet_w:
                x_offset = 0.0
                y_offset += height + spacing

            if y_offset + height > sheet_h:
                print("WARNING: Out of sheet space during nesting")
                continue

            placements.append((poly, (x_offset, y_offset)))
            x_offset += width + spacing

    return placements


# ---------------------------------------------------------------------------
# 4. Toolpath (cutting order optimization)
# ---------------------------------------------------------------------------
# Toolpath generation moved to gcode workflow module


# ---------------------------------------------------------------------------
# 5. Costing
# ---------------------------------------------------------------------------
# Costing moved to gcode workflow module


# ---------------------------------------------------------------------------
# Helpers for outputs
# ---------------------------------------------------------------------------
# File writers moved to gcode workflow module


# File writers moved to gcode workflow module


# File writers moved to gcode workflow module


# ---------------------------------------------------------------------------
# Geometry Quality Report
# ---------------------------------------------------------------------------
def _quality_report(file_path: str, tolerance: float = 2.0, shaky_threshold: int = 200) -> dict:
    import math
    try:
        doc = ezdxf.readfile(file_path)
    except Exception:
        return {}
    msp = doc.modelspace()

    entity_report = {
        "Total Entities": len(msp),
        "Lines": len(msp.query("LINE")),
        "Arcs": len(msp.query("ARC")),
        "Circles": len(msp.query("CIRCLE")),
        "Polylines": len(msp.query("LWPOLYLINE")),
        "Open Polylines": 0,
        "Shaky Polylines": [],
        "Tiny Segments": 0,
        "Duplicate Candidates": 0,
    }

    seen = set()
    for idx, pl in enumerate(msp.query("LWPOLYLINE")):
        try:
            pts = list(pl.get_points("xy"))
        except Exception:
            continue

        # Open polylines
        if not pl.closed:
            entity_report["Open Polylines"] += 1

        # Shaky (too many vertices)
        if len(pts) > shaky_threshold:
            entity_report["Shaky Polylines"].append({"Index": idx, "Vertices": len(pts)})

        # Tiny segments
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < tolerance:
                entity_report["Tiny Segments"] += 1

        # Duplicate candidates via rounded bbox
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = (round(min(xs), 1), round(max(xs), 1), round(min(ys), 1), round(max(ys), 1))
            if bbox in seen:
                entity_report["Duplicate Candidates"] += 1
            else:
                seen.add(bbox)

    return entity_report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_dxf_objects_only(
    dxf_path: str,
    args: AnalyzeArgs | None = None,
) -> dict:
    """Analyze DXF and return individual objects without grouping."""
    if args is None:
        args = AnalyzeArgs()
    
    # Load and process DXF
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # Extract components
    components = []
    for entity in msp:
        if entity.dxftype() in ["LWPOLYLINE", "POLYLINE", "LINE", "ARC", "CIRCLE"]:
            try:
                points = _extract_points_from_entity(entity)
                if len(points) >= 2:
                    # Calculate properties
                    area = _calculate_area(points)
                    perimeter = _calculate_perimeter(points)
                    vcount = len(points)
                    
                    # Generate intelligent name
                    obj = {
                        "id": len(components) + 1,
                        "points": points,
                        "area": area,
                        "perimeter": perimeter,
                        "vertex_count": vcount,
                        "layer": entity.dxf.layer if hasattr(entity.dxf, 'layer') else "0",
                        "selected": True,  # All objects selected by default
                    }
                    obj["name"] = generate_intelligent_object_name(obj, obj["id"])
                    components.append(obj)
            except Exception:
                continue
    
    # Calculate overall metrics
    total_area = sum(c["area"] for c in components)
    total_perimeter = sum(c["perimeter"] for c in components)
    
    return {
        "file": os.path.basename(dxf_path),
        "components": components,
        "metrics": {
            "total_area": total_area,
            "total_perimeter": total_perimeter,
            "object_count": len(components),
        },
        "groups": {},  # No groups yet
        "layers": {},
        "quality": {},
        "toolpath": {},
        "nesting": {},
        "artifacts": {},
        "selection": {"groups": [], "component_ids": [c["id"] for c in components]},
    }


def analyze_dxf(
    dxf_path: str,
    args: AnalyzeArgs | None = None,
    selected_groups: list[str] | None = None,
    group_layer_overrides: Dict[str, str] | None = None,
    path_manager: Optional["PathManager"] = None,
    auth_service: Optional["AuthService"] = None,
    report_generator: Optional["ReportGenerator"] = None,
    logging_service: Optional["LoggingService"] = None,
) -> dict:
    args = args or AnalyzeArgs()
    
    cache_service = None
    cache_key = None
    cached_payload = None
    try:
        from ..performance.cache_manager import get_cache_manager, compute_file_hash

        cache_dir = os.path.join(args.out, ".cache")
        cache_service = get_cache_manager(cache_dir)
        file_hash = compute_file_hash(dxf_path)
        param_dict = {
            "material": args.material,
            "thickness": args.thickness,
            "kerf": args.kerf,
            "streaming_mode": getattr(args, "streaming_mode", False),
            "early_simplify": getattr(args, "early_simplify_tolerance", 0.0),
        }
        cache_key = build_cache_key(file_hash, param_dict)
        cached_payload = cache_service.get(cache_key)
        if cached_payload and logging_service:
            logging_service.info(
                "Using cached analysis result",
                {
                    "cache_type": "hit",
                    "cache_key": cache_key,
                    "file": os.path.basename(dxf_path),
                },
            )
    except ImportError:
        try:
            import hashlib

            cache_service = CacheService(os.path.join(args.out, ".cache"))
            with open(dxf_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            param_dict = {
                "material": args.material,
                "thickness": args.thickness,
                "kerf": args.kerf,
                "streaming_mode": getattr(args, "streaming_mode", False),
                "early_simplify": getattr(args, "early_simplify_tolerance", 0.0),
            }
            cache_key = build_cache_key(file_hash, param_dict)
            cached_payload = cache_service.get(cache_key)
            if cached_payload and logging_service:
                logging_service.info(
                    "Using cached analysis result",
                    {
                        "cache_type": "hit",
                        "cache_key": cache_key,
                        "file": os.path.basename(dxf_path),
                    },
                )
        except Exception:
            cache_service = None
            cached_payload = None
    except Exception:
        cache_service = None
        cached_payload = None

    if cached_payload:
        return filter_cached_report(
            cached_payload,
            selected_groups=selected_groups,
            group_layer_overrides=group_layer_overrides,
            logging_service=logging_service,
        )
    
    # Initialize logging if service provided
    if logging_service:
        logging_service.info("Starting DXF analysis", {
            "file": os.path.basename(dxf_path),
            "material": args.material,
            "thickness": args.thickness,
            "cache_key": cache_key,
            "cache_type": "miss"
        })
    
    # Use path manager if provided, or create basic output directory
    if path_manager:
        output_dir = path_manager.get_dxf_output_path(os.path.basename(dxf_path))
        args.out = str(output_dir.parent)
        cache_service = CacheService(os.path.join(str(output_dir.parent), ".cache"))
        if logging_service:
            logging_service.debug("Using managed output directory", {
                "output_dir": str(output_dir)
            })
    else:
        os.makedirs(args.out, exist_ok=True)
        if logging_service:
            logging_service.debug("Created basic output directory", {
                "output_dir": args.out
            })

    # Validate authentication if auth service provided
    if auth_service:
        # In real usage, token would come from request context
        # Here we just validate that auth service is working
        auth_service.cleanup_expired_tokens()

    # Determine scale factor (drawing units → mm)
    scale_factor = 1.0
    try:
        doc0 = ezdxf.readfile(dxf_path)
        insunits = int(doc0.header.get("$INSUNITS", 0))
        if getattr(args, "scale_mode", "auto") == "auto":
            scale_factor = _units_to_mm(insunits)
        else:
            scale_factor = float(getattr(args, "scale_factor", 1.0)) or 1.0
    except Exception:
        scale_factor = float(getattr(args, "scale_factor", 1.0)) or 1.0

    # Optional fillet pre-processing (write a temp DXF with arc bulges); radius is in mm → convert to drawing units
    source_path = dxf_path
    if float(getattr(args, "fillet_radius_mm", 0.0)) > 0.0:
        try:
            os.makedirs(args.out, exist_ok=True)
            filleted_path = os.path.join(args.out, "__filleted_input.dxf")
            _apply_fillet_to_file(
                dxf_path,
                filleted_path,
                r=float(args.fillet_radius_mm) / max(scale_factor, 1e-12),
                min_angle_deg=float(getattr(args, "fillet_min_angle_deg", 135.0)),
            )
            source_path = filleted_path
        except Exception:
            source_path = dxf_path

    # Pre-scan entity counts for diagnostics
    entity_counts = _collect_entity_counts(source_path)

    # Use streaming parser for large files or if explicitly requested
    # Check file size using os.path.getsize (more reliable than Path)
    file_size = os.path.getsize(source_path)
    use_streaming = getattr(args, "streaming_mode", False) or file_size > 10 * 1024 * 1024  # 10MB threshold
    
    if use_streaming:
        try:
            from ..performance.streaming_parser import StreamingDXFParser, normalize_entities, parse_with_early_simplification
            from ..performance.memory_optimizer import filter_tiny_segments
            
            # Use streaming parser with early simplification
            parser = StreamingDXFParser()
            early_simplify = getattr(args, "early_simplify_tolerance", 0.1)
            
            if early_simplify > 0:
                # Parse with early simplification
                entities = parse_with_early_simplification(source_path, tolerance=early_simplify)
            else:
                # Parse normally but stream
                entities = list(parser.parse_in_chunks(source_path))
            
            # Normalize entities (explode SPLINE/ELLIPSE)
            entities = normalize_entities(entities)
            
            # Convert to polylines format
            polylines = []
            for entity in entities:
                points = entity.get("points", [])
                if len(points) >= 2:
                    # Filter tiny segments if enabled
                    min_seg = getattr(args, "min_segment_length", 0.01)
                    if min_seg > 0:
                        from ..performance.memory_optimizer import filter_tiny_segments
                        points = filter_tiny_segments(points, epsilon=min_seg)
                    
                    if len(points) >= 2:
                        polylines.append({
                            "points": points,
                            "handle": entity.get("handle"),
                            "source_layer": entity.get("layer", "0"),
                        })
        except ImportError:
            # Fallback to standard parser if performance module not available
            polylines = _extract_polylines(source_path)
        except Exception:
            # Fallback on any error
            polylines = _extract_polylines(source_path)
    else:
        # Standard parser for smaller files
        polylines = _extract_polylines(source_path)
    # Apply scaling to mm (units -> mm)
    polylines = _scale_polylines(polylines, scale_factor)

    # Optional decade fit scaling
    scale_meta = {}
    bbox0 = _compute_bbox(polylines)
    if getattr(args, "scale_mode", "auto") == "decade_fit":
        polylines, dec_meta = _apply_decade_fit(
            polylines,
            bbox0,
            float(getattr(args, "target_frame_w_mm", 1000.0)),
            float(getattr(args, "target_frame_h_mm", 1000.0)),
            float(getattr(args, "frame_margin_mm", 0.0)),
            float(getattr(args, "scale_decade_base", 10.0)),
            str(getattr(args, "scale_decade_direction", "auto")),
            int(getattr(args, "scale_decade_max_steps", 6)),
            bool(getattr(args, "scale_decade_allow_overshoot", True)),
            bool(getattr(args, "scale_decade_exact_fit", False)),
            bool(getattr(args, "normalize_origin", True)),
        )
        bbox0 = _compute_bbox(polylines)
        scale_meta.setdefault("decade", dec_meta)
    if getattr(args, "normalize_mode", "none") == "fit" and bbox0["width"] > 0 and bbox0["height"] > 0:
        inner_w = max(1e-9, float(getattr(args, "target_frame_w_mm", 1000.0)) - 2.0 * float(getattr(args, "frame_margin_mm", 0.0)))
        inner_h = max(1e-9, float(getattr(args, "target_frame_h_mm", 1000.0)) - 2.0 * float(getattr(args, "frame_margin_mm", 0.0)))
        sf = min(inner_w / bbox0["width"], inner_h / bbox0["height"])
        margin = float(getattr(args, "frame_margin_mm", 0.0))
        if bool(getattr(args, "normalize_origin", True)):
            dx = -bbox0["minx"] * sf + margin
            dy = -bbox0["miny"] * sf + margin
        else:
            dx = 0.0
            dy = 0.0
        polylines = _scale_translate(polylines, sf, dx, dy)
        bbox1 = _compute_bbox(polylines)
        scale_meta["normalize"] = {
            "mode": "fit",
            "factor": sf,
            "frame": {
                "w": float(getattr(args, "target_frame_w_mm", 1000.0)),
                "h": float(getattr(args, "target_frame_h_mm", 1000.0)),
                "margin": margin,
            },
            "bbox_before": bbox0,
            "bbox_after": bbox1,
        }
    else:
        scale_meta.setdefault("normalize", {
            "mode": str(getattr(args, "normalize_mode", "none")),
            "factor": 1.0,
            "frame": None,
            "bbox_before": bbox0,
            "bbox_after": bbox0,
        })
    # Enforce must-fit within frame (safety clamp)
    if bool(getattr(args, "require_fit_within_frame", True)):
        norm = scale_meta.get("normalize", {})
        frame = norm.get("frame") or {"w": getattr(args, "target_frame_w_mm", 1000.0), "h": getattr(args, "target_frame_h_mm", 1000.0), "margin": getattr(args, "frame_margin_mm", 0.0)}
        inner_w = max(1e-9, float(frame.get("w", 1000.0)) - 2.0 * float(frame.get("margin", 0.0)))
        inner_h = max(1e-9, float(frame.get("h", 1000.0)) - 2.0 * float(frame.get("margin", 0.0)))
        bbox_now = _compute_bbox(polylines)
        if bbox_now["width"] > inner_w or bbox_now["height"] > inner_h:
            clamp = min(inner_w / max(1e-9, bbox_now["width"]), inner_h / max(1e-9, bbox_now["height"]))
            minx, miny = bbox_now["minx"], bbox_now["miny"]
            margin = float(frame.get("margin", 0.0))
            polylines = _scale_translate(polylines, clamp, dx=margin - minx * clamp, dy=margin - miny * clamp)
            scale_meta["clamp"] = {"applied": True, "factor": clamp, "bbox_after": _compute_bbox(polylines)}
        else:
            scale_meta["clamp"] = {"applied": False}

    # Optional smoothing/simplification
    if getattr(args, "soften_method", "none") and args.soften_method.lower() != "none":
        polylines = _apply_soften(
            polylines,
            method=args.soften_method,
            tol=float(getattr(args, "soften_tolerance", 0.2)),
            iterations=int(getattr(args, "soften_iterations", 1)),
        )
    _, components = _classify_polylines(polylines)
    groups, components = group_similar_objects(components)

    if groups:
        if selected_groups:
            active_groups = [g for g in selected_groups if g in groups]
            if not active_groups:
                active_groups = list(groups.keys())
        else:
            active_groups = list(groups.keys())
        active_components = [comp for comp in components if comp.get("group") in active_groups]
    else:
        active_groups = []
        active_components = components

    # Apply optional group->layer overrides before building layers
    if group_layer_overrides:
        valid_targets = set(LAYER_NAMES)
        for comp in active_components:
            gname = comp.get("group")
            target = group_layer_overrides.get(gname) if gname else None
            if target and target in valid_targets:
                comp["layer"] = target

    # Multiply by frame quantity (tile count)
    qty = max(1, int(getattr(args, "frame_quantity", 1)))
    if qty > 1 and active_components:
        active_components = active_components * qty

    layers_active = _layers_from_components(active_components)

    # Toolpath, nesting, and costing moved to gcode workflow page/module
    placements = []
    metrics = {}

    layered_path = os.path.join(args.out, "layered_output.dxf")
    lengths_path = os.path.join(args.out, "lengths.csv")
    report_path = os.path.join(args.out, "report.json")

    # Diagnostics for feedback when metrics appear empty or geometry is skipped
    unsupported_types = {}
    try:
        processed = {"LWPOLYLINE", "POLYLINE", "LINE", "ARC", "CIRCLE", "SPLINE", "ELLIPSE", "INSERT"}
        for k, v in (entity_counts or {}).items():
            if k not in processed:
                unsupported_types[k] = v
    except Exception:
        unsupported_types = {}

    # Build diagnostics and checklist items
    diagnostics = {
        "extracted_objects": len(polylines or []),
        "components": len(components or []),
        "groups": len(groups or {}),
        "entity_counts": entity_counts,
        "unsupported_entities": unsupported_types,
        "hints": []  # filled below
    }

    if diagnostics["extracted_objects"] == 0 or diagnostics["components"] == 0:
        hints = diagnostics["hints"]
        if (entity_counts or {}).get("SPLINE", 0) > 0:
            hints.append("Convert SPLINE to polylines/arcs or pre-simplify (see tools/dxf_preserve_arcs_cleaner.py)")
        if (entity_counts or {}).get("HATCH", 0) > 0:
            hints.append("HATCH not used for contours; explode or derive boundaries")
        if (entity_counts or {}).get("INSERT", 0) > 0:
            hints.append("Blocks present; ensure virtual entities expand to closed loops")
        if not entity_counts:
            hints.append("DXF read failed or empty modelspace")

    # Quality report (used by checklist and UI)
    try:
        quality = _quality_report(
            dxf_path,
            tolerance=float(getattr(args, "quality_tolerance", 2.0)),
            shaky_threshold=int(getattr(args, "shaky_threshold", 200)),
        )
    except Exception:
        quality = {}

    # DXF file info for checklist
    dxf_version = None
    dxf_supported = True
    insunits_value = None
    layer_names: list[str] = []
    color_meta = {"colors": {}, "linetypes": {}, "lineweights": {}}
    try:
        info_doc = ezdxf.readfile(source_path)
        dxf_version = str(getattr(info_doc, "dxfversion", None))
        try:
            insunits_value = int(info_doc.header.get("$INSUNITS", 0))
        except Exception:
            insunits_value = None
        layer_names = _list_layer_names(info_doc)
        color_meta = _color_linetype_summary(info_doc)
        # AC1009 (R12) and newer considered supported here
        # Compare lexicographically on ACxxxx; if parsing fails, keep True
        try:
            code = int(dxf_version.replace("AC", "")) if dxf_version and dxf_version.startswith("AC") else 1009
            dxf_supported = code >= 1009
        except Exception:
            dxf_supported = True
    except Exception:
        pass

    # Spacing and corner radius checks
    spacing_violations, min_spacing = _min_spacing_violations(components, min_gap=3.0)
    min_corner_radius = _estimate_min_corner_radius_all(components)

    warnings: list[str] = []
    if not entity_counts:
        warnings.append("File appears empty or unreadable")
    if (insunits_value or 0) == 0:
        warnings.append("Units not defined; assumed mm")
    if unsupported_types:
        warnings.append("Unsupported entities present")
    if quality.get("Open Polylines", 0) > 0:
        warnings.append("Open polylines found")
    if quality.get("Duplicate Candidates", 0) > 0:
        warnings.append("Duplicate geometry suspected")
    if spacing_violations > 0:
        warnings.append("Minimum spacing violations (< 3mm)")
    if 0.0 < min_corner_radius < 2.0:
        warnings.append("Minimum corner radius below 2mm")

    checklist = {
        "Entities": entity_counts or {},
        "OpenPolylines": quality.get("Open Polylines", 0),
        "ShakyPolylines": quality.get("Shaky Polylines", []),
        "TinySegments": quality.get("Tiny Segments", 0),
        "Duplicates": quality.get("Duplicate Candidates", 0),
        # Length and pierces now produced by G-code workflow
        "Warnings": warnings,
    }

    report = {
        "file": os.path.basename(dxf_path),
        "material": {"name": args.material, "thickness_mm": args.thickness},
        "kerf_mm": args.kerf,
        "metrics": metrics,
        "scale": {
            "units_mode": getattr(args, "scale_mode", "auto"),
            "units_factor": scale_factor,
            "normalize": scale_meta,
        },
        "layers_present": layer_names,
        "layers": {name: len(polys) for name, polys in layers_active.items()},
        "toolpath": {},
        "nesting": {},
        "artifacts": {
            "report_json": report_path,
            "lengths_csv": lengths_path,
            "layered_dxf": layered_path,
        },
        "selection": {
            "groups": active_groups,
            "component_ids": [comp["id"] for comp in active_components],
        },
        "diagnostics": diagnostics,
        "mastery_checklist": checklist,
        "file_integrity": {"version": dxf_version, "supported": dxf_supported, "open_ok": bool(entity_counts)},
        "metadata_usage": color_meta,
        "geometry_validation": {
            "min_spacing_mm": min_spacing,
            "min_spacing_violations": spacing_violations,
            "min_corner_radius_mm": min_corner_radius,
            "tiny_segment_threshold_mm": float(getattr(args, "quality_tolerance", 2.0)),
        },
    }

    # Attach quality report (already computed above)
    report["quality"] = quality

    # TODO (Codex): Rename all base entities currently labeled as "GroupX" to "ObjectX".
    # These represent individual DXF polylines before similarity grouping.
    # After similarity grouping (via group_similar_objects), use "GroupX" only for clustered similar shapes.
    #
    # Implementation details:
    # 1. When plotting or listing raw detected polylines (before grouping),
    #    label them as Object1, Object2, ... instead of Group1, Group2.
    # 2. After group_similar_objects() runs, its output keys remain as Group1, Group2, ...
    # 3. Update legends, tooltips, and any text in the UI/CSV to follow:
    #       - Pre-grouping stage → "Object"
    #       - Post-grouping stage → "Group"
    # 4. Ensure exported CSV reflects the same distinction.

    groups_export = {}
    for name, meta in groups.items():
        assigned_layer = None
        if group_layer_overrides and name in group_layer_overrides:
            assigned_layer = group_layer_overrides[name]
        groups_export[name] = {
            **meta,
            "selected": name in active_groups,
            "assigned_layer": assigned_layer,
        }

    components_export = []
    for comp in components:
        # compute per-component size
        try:
            xs = [p[0] for p in comp["points"]]
            ys = [p[1] for p in comp["points"]]
            size_w = float(max(xs) - min(xs)) if xs else 0.0
            size_h = float(max(ys) - min(ys)) if ys else 0.0
        except Exception:
            size_w = size_h = 0.0
        comp_entry = {
            "id": comp["id"],
            "layer": comp["layer"],
            "group": comp.get("group", "Ungrouped"),
            "area": round(comp["area"], 2),
            "perimeter": round(comp["perimeter"], 2),
            "vertex_count": comp["vertex_count"],
            "selected": comp["group"] in active_groups,
            "points": [[float(x), float(y)] for x, y in comp["points"]],
            "handle": comp.get("handle"),
            "source_layer": comp.get("source_layer"),
            "size_w_mm": round(size_w, 2),
            "size_h_mm": round(size_h, 2),
        }
        components_export.append(comp_entry)

    report["groups"] = groups_export
    report["components"] = components_export

    # Add size-based grouping for UI
    size_groups: dict = {}
    for comp in components_export:
        key = (round(comp.get("size_w_mm", 0.0), 1), round(comp.get("size_h_mm", 0.0), 1))
        name = f"Size {key[0]}x{key[1]}"
        entry = size_groups.setdefault(name, {"size_w_mm": key[0], "size_h_mm": key[1], "count": 0, "component_ids": []})
        entry["count"] += 1
        entry["component_ids"].append(comp["id"])
    report["size_groups"] = size_groups

    # Report/layered outputs are produced by the G-code workflow module

    # Generate PDF report if report generator provided
    if report_generator:
        try:
            pdf_path = os.path.join(args.out, "analysis_report.pdf")
            report_generator.generate_pdf_report(report, pdf_path)
            report["artifacts"]["pdf_report"] = pdf_path
        except Exception as e:
            print(f"Warning: PDF report generation failed: {e}")
            
    if cache_service and cache_key:
        try:
            cache_service.set(cache_key, report)
            if logging_service:
                logging_service.info(
                    "Cached analysis result",
                    {
                        "file": os.path.basename(dxf_path),
                        "cache_key": cache_key,
                        "cache_type": "miss->store",
                    },
                )
        except Exception:
            if logging_service:
                logging_service.warning(
                    "Failed to cache analysis result",
                    {
                        "file": os.path.basename(dxf_path),
                        "cache_key": cache_key,
                    },
                )

    return report


def relayer_entities_by_handle(
    input_dxf: str,
    handles: Iterable[str],
    target_layer: str,
    output_path: Optional[str] = None,
) -> str:
    """Move entities with given handles to a target layer and save DXF.

    Returns path to the updated DXF file.
    """
    doc = ezdxf.readfile(input_dxf)
    if target_layer not in doc.layers:
        try:
            doc.layers.new(name=target_layer)
        except Exception:
            pass
    changed = 0
    for h in handles:
        if not h:
            continue
        try:
            ent = doc.entitydb.get(str(h))
        except Exception:
            ent = None
        if ent is None:
            continue
        try:
            ent.dxf.layer = target_layer
            changed += 1
        except Exception:
            continue
    if not output_path:
        base, ext = os.path.splitext(input_dxf)
        output_path = f"{base}_relayered{ext}"
    doc.saveas(output_path)
    return output_path


def load_polys_and_classes(dxf_path: str):
    polylines = _extract_polylines(dxf_path)
    _, components = _classify_polylines(polylines)

    polygons: list[Polygon] = []
    classes: dict[int, str] = {}

    for comp in components:
        try:
            poly = Polygon(comp["points"])
            if not poly.is_valid or poly.is_empty:
                continue
        except Exception:
            continue
        idx = len(polygons)
        polygons.append(poly)
        if comp["layer"] in {"HOLE", "INNER"}:
            classes[idx] = "inner"
        else:
            classes[idx] = "outer"

    order = list(range(len(polygons)))
    return polygons, classes, order


# G-code API removed from analyzer; provided in a dedicated module/page


def run_dxf_analyzer(
    input_dxf: str, 
    output_dir: str = "out",
    path_manager: Optional["PathManager"] = None,
    auth_service: Optional["AuthService"] = None,
    report_generator: Optional["ReportGenerator"] = None
) -> dict:
    """Run DXF analysis with optional service integrations."""
    
    # Use provided path manager or default to basic directory
    if path_manager:
        out_path = path_manager.get_dxf_output_path(os.path.basename(input_dxf))
        output_dir = str(out_path.parent)
    
    args = AnalyzeArgs(out=output_dir)
    report = analyze_dxf(
        input_dxf, 
        args,
        path_manager=path_manager,
        auth_service=auth_service,
        report_generator=report_generator
    )
    
    print("✅ Layering complete")
    print(f"✅ Toolpath with {report['metrics']['pierces']} cuts")
    print("💰 Cost Report:", report["metrics"])
    print(f"💾 Saved artifacts under {output_dir}")
    
    if "pdf_report" in report.get("artifacts", {}):
        print(f"📄 PDF Report generated: {report['artifacts']['pdf_report']}")
    
    return report


if __name__ == "__main__":
    default_input = "Tile45_converted.dxf"
    default_output = "out"
    if os.path.exists(default_input):
        run_dxf_analyzer(default_input, default_output)
    else:
        print(f"Input DXF '{default_input}' not found. Provide a path to analyze.")
