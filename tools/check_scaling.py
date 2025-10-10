#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import math

try:
    import ezdxf
except Exception as e:
    print("Missing dependency: ezdxf. Please install requirements.txt.")
    raise


def units_to_mm(insunits: int) -> float:
    mapping = {
        0: 1.0,       # Unitless
        1: 25.4,      # Inches
        2: 304.8,     # Feet
        3: 1609344,   # Miles
        4: 1.0,       # Millimeters
        5: 10.0,      # Centimeters
        6: 1000.0,    # Meters
        7: 1.0e6,     # Kilometers
        8: 304.8006096,  # US survey foot
        9: 914.4 * 3.0 / 3.0,  # Yard (approx)
    }
    return float(mapping.get(int(insunits or 0), 1.0))


def _upd_bbox(x: float, y: float, bb: dict) -> None:
    bb["minx"] = x if bb["minx"] is None else min(bb["minx"], x)
    bb["miny"] = y if bb["miny"] is None else min(bb["miny"], y)
    bb["maxx"] = x if bb["maxx"] is None else max(bb["maxx"], x)
    bb["maxy"] = y if bb["maxy"] is None else max(bb["maxy"], y)


def compute_extents(doc) -> Tuple[float, float, float, float]:
    msp = doc.modelspace()
    bb = {"minx": None, "miny": None, "maxx": None, "maxy": None}
    seen = {}

    def add_points(pts):
        for (x, y) in pts:
            _upd_bbox(float(x), float(y), bb)

    def arc_points(center, radius, start_deg, end_deg, n=64):
        # oversample within arc span
        s = math.radians(float(start_deg))
        e = math.radians(float(end_deg))
        while e < s:
            e += 2 * math.pi
        for i in range(n + 1):
            a = s + (e - s) * (i / n)
            yield (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))

    for e in msp:
        t = e.dxftype()
        seen[t] = seen.get(t, 0) + 1
        try:
            if t == "LINE":
                add_points([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)])
            elif t == "LWPOLYLINE":
                add_points([(p[0], p[1]) for p in e.get_points("xy")])
            elif t == "POLYLINE":
                add_points([(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()])
            elif t == "CIRCLE":
                cx, cy = e.dxf.center.x, e.dxf.center.y
                r = float(e.dxf.radius)
                add_points([(cx - r, cy - r), (cx + r, cy + r)])
            elif t == "ARC":
                cx, cy = e.dxf.center.x, e.dxf.center.y
                r = float(e.dxf.radius)
                add_points(list(arc_points((cx, cy), r, e.dxf.start_angle, e.dxf.end_angle, n=36)))
            elif t == "SPLINE":
                pts = []
                try:
                    pts = [(p.x, p.y) for p in e.approximate(segments=128)]
                except Exception:
                    try:
                        pts = [(p[0], p[1]) for p in e.approximate(128)]
                    except Exception:
                        pass
                if not pts:
                    # fallback to fit or control points
                    try:
                        pts = [(p.x, p.y) for p in getattr(e, 'fit_points', [])]
                    except Exception:
                        pts = []
                    if not pts:
                        try:
                            pts = [(p.x, p.y) for p in getattr(e, 'control_points', [])]
                        except Exception:
                            pts = []
                if not pts:
                    try:
                        tool = e.construction_tool()
                        pts = [(p.x, p.y) for p in tool.approximate(128)]
                    except Exception:
                        pts = []
                add_points(pts)
            elif t == "ELLIPSE":
                try:
                    pts = [(p.x, p.y) for p in e.flattening(2.0)]
                except Exception:
                    pts = [(p.x, p.y) for p in e.construction_tool().approximate(128)]
                add_points(pts)
            elif t == "INSERT":
                # Skip detailed expansion for speed; extents may under-report if only blocks exist
                pass
        except Exception:
            continue

    if bb["minx"] is None:
        # Emit quick inventory for debugging
        print("Entities seen:", seen)
        return (0.0, 0.0, 0.0, 0.0)
    return (float(bb["minx"]), float(bb["miny"]), float(bb["maxx"]), float(bb["maxy"]))


def suggest_fit(minx, miny, maxx, maxy, units_factor: float, frame_w: float, frame_h: float, margin: float) -> dict:
    w = maxx - minx
    h = maxy - miny
    w_mm = w * units_factor
    h_mm = h * units_factor
    inner_w = max(1e-9, float(frame_w) - 2.0 * float(margin))
    inner_h = max(1e-9, float(frame_h) - 2.0 * float(margin))
    if w_mm <= 0 or h_mm <= 0:
        return {
            "width_mm": w_mm,
            "height_mm": h_mm,
            "units_factor": units_factor,
            "fit_factor": None,
        }
    f = min(inner_w / w_mm, inner_h / h_mm)
    return {
        "width_mm": round(w_mm, 3),
        "height_mm": round(h_mm, 3),
        "units_factor": units_factor,
        "fit_factor": round(f, 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check DXF units and extents, and suggest fit-to-frame scaling.")
    ap.add_argument("input", help="Input DXF path")
    ap.add_argument("--sheet", nargs=2, type=float, default=[1000.0, 1000.0], metavar=("W", "H"), help="Sheet size in mm")
    ap.add_argument("--margin", type=float, default=0.0, help="Frame margin in mm")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input not found: {path}")
        return 2

    doc = ezdxf.readfile(str(path))
    try:
        insunits = int(doc.header.get("$INSUNITS", 0))
    except Exception:
        insunits = 0
    factor = units_to_mm(insunits)
    minx, miny, maxx, maxy = compute_extents(doc)
    sugg = suggest_fit(minx, miny, maxx, maxy, factor, args.sheet[0], args.sheet[1], args.margin)

    print("INSUNITS:", insunits)
    print("Units->mm factor:", factor)
    print(f"Raw extents (units): min=({minx:.3f},{miny:.3f}) max=({maxx:.3f},{maxy:.3f}) w={maxx-minx:.3f} h={maxy-miny:.3f}")
    print(f"Extents (mm): w={sugg['width_mm']} h={sugg['height_mm']}")
    print(f"Suggested fit factor for sheet {args.sheet[0]}x{args.sheet[1]} mm (margin {args.margin}): {sugg['fit_factor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
