#!/usr/bin/env python3
"""
DXF Cleaner (Preserve Arcs)
 - Preserves true ARC/CIRCLE entities
 - Simplifies only straight polylines using Shapely's simplify()
 - Skips simplification for polylines with bulge (arc segments)

Usage:
  python tools/dxf_preserve_arcs_cleaner.py input.dxf -o output_cleaned.dxf \
         --tol 0.05 --min-feature 2.0

Notes:
 - Units are assumed to be mm.
 - Designed for waterjet use where arcs should not be exploded.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

try:
    import ezdxf
except Exception as e:
    print("Missing dependency: ezdxf. Please install requirements.txt.")
    raise

try:
    from shapely.geometry import LineString
except Exception:
    print("Missing dependency: shapely. Please install requirements.txt.")
    raise


def _lwpolyline_has_bulge(ent) -> bool:
    try:
        # ezdxf API: get_points supports attributes; use 'xyb' to include bulge
        for p in ent.get_points("xyb"):
            # p is (x, y, bulge)
            if len(p) >= 3 and abs(p[2]) > 1e-12:
                return True
        return False
    except Exception:
        # Fallback: try entity.has_arc flag when available
        return bool(getattr(ent, "has_arc", False))


def _is_closed_polyline(ent) -> bool:
    # LWPOLYLINE has closed flag
    try:
        return bool(ent.closed)
    except Exception:
        return False


def _bbox_size(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys))


def clean_dxf(input_path: str, output_path: str, tol: float, min_feature: float) -> None:
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # Create new document to avoid mutating original
    out_doc = ezdxf.new()
    out_msp = out_doc.modelspace()

    kept = 0
    simplified = 0
    skipped = 0

    for e in msp:
        t = e.dxftype()

        # Preserve arcs and circles exactly
        if t == "ARC":
            try:
                out_msp.add_arc(
                    center=(e.dxf.center.x, e.dxf.center.y),
                    radius=e.dxf.radius,
                    start_angle=e.dxf.start_angle,
                    end_angle=e.dxf.end_angle,
                    dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None,
                )
                kept += 1
            except Exception:
                skipped += 1
            continue

        if t == "CIRCLE":
            try:
                out_msp.add_circle(
                    center=(e.dxf.center.x, e.dxf.center.y),
                    radius=e.dxf.radius,
                    dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None,
                )
                kept += 1
            except Exception:
                skipped += 1
            continue

        # LWPOLYLINE: simplify only if no bulge (no arc segments)
        if t == "LWPOLYLINE":
            try:
                # Extract points as (x, y)
                pts = [(p[0], p[1]) for p in e.get_points()]
                closed = _is_closed_polyline(e)

                if _lwpolyline_has_bulge(e):
                    # Preserve as-is to keep arcs
                    out_msp.add_lwpolyline(pts, format="xy", close=closed,
                                           dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                    kept += 1
                else:
                    # Skip simplification for very small features (likely holes/tabs)
                    w, h = _bbox_size(pts)
                    if closed and max(w, h) < min_feature:
                        out_msp.add_lwpolyline(pts, format="xy", close=closed,
                                               dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                        kept += 1
                    else:
                        line = LineString(pts)
                        simp = line.simplify(tol, preserve_topology=True)
                        simp_pts = list(simp.coords)
                        if len(simp_pts) >= 2:
                            out_msp.add_lwpolyline(simp_pts, format="xy", close=closed,
                                                   dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                            simplified += 1
                        else:
                            # Fallback keep original
                            out_msp.add_lwpolyline(pts, format="xy", close=closed,
                                                   dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                            kept += 1
            except Exception:
                skipped += 1
            continue

        # POLYLINE (2D): treat as straight segments and simplify
        if t == "POLYLINE":
            try:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()]
                closed = bool(e.is_closed)
                w, h = _bbox_size(pts) if pts else (0.0, 0.0)
                if closed and max(w, h) < min_feature:
                    out_msp.add_lwpolyline(pts, format="xy", close=closed,
                                           dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                    kept += 1
                else:
                    line = LineString(pts)
                    simp = line.simplify(tol, preserve_topology=True)
                    simp_pts = list(simp.coords)
                    if len(simp_pts) >= 2:
                        out_msp.add_lwpolyline(simp_pts, format="xy", close=closed,
                                               dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                        simplified += 1
                    else:
                        out_msp.add_lwpolyline(pts, format="xy", close=closed,
                                               dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                        kept += 1
            except Exception:
                skipped += 1
            continue

        # Preserve lines unchanged
        if t == "LINE":
            try:
                out_msp.add_line(
                    (e.dxf.start.x, e.dxf.start.y),
                    (e.dxf.end.x, e.dxf.end.y),
                    dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None,
                )
                kept += 1
            except Exception:
                skipped += 1
            continue

        # SPLINE and other entities: keep as-is when possible (avoid densification)
        if t == "SPLINE":
            try:
                new_ent = out_msp.add_spline(dxfattribs={"layer": e.dxf.layer} if hasattr(e.dxf, "layer") else None)
                # Copy fit points when available
                try:
                    for fp in e.fit_points:
                        new_ent.append_fit_point((fp.x, fp.y, 0.0))
                except Exception:
                    pass
                kept += 1
            except Exception:
                skipped += 1
            continue

        # Unknown/unsupported: skip quietly
        skipped += 1

    out_doc.saveas(output_path)
    print(f"DXF cleaned. kept={kept}, simplified={simplified}, skipped={skipped}")


def main() -> int:
    ap = argparse.ArgumentParser(description="DXF cleaner that preserves arcs and simplifies straight polylines.")
    ap.add_argument("input", help="Input DXF path")
    ap.add_argument("-o", "--output", help="Output DXF path (default: <input>_cleaned.dxf)")
    ap.add_argument("--tol", type=float, default=0.05, help="Simplify tolerance in mm (default: 0.05)")
    ap.add_argument("--min-feature", type=float, default=2.0, help="Do not simplify closed polylines smaller than this size (mm)")
    args = ap.parse_args()

    input_path = args.input
    output_path = args.output or f"{os.path.splitext(input_path)[0]}_cleaned.dxf"

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return 2

    clean_dxf(input_path, output_path, tol=args.tol, min_feature=args.min_feature)
    return 0


if __name__ == "__main__":
    sys.exit(main())

