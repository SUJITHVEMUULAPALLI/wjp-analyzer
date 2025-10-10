#!/usr/bin/env python3
"""
G-code Arc Fitter
 - Converts chains of small G1 XY moves into G2/G3 arcs
 - Uses least-squares circle fit with chordal tolerance
 - Emits I/J center offsets (incremental) in G17 plane

Usage:
  python tools/gcode_arc_fitter.py input.nc -o output_arcfit.nc \
         --tol 0.05 --min-radius 0.5 --min-length 1.0 --split-gt-180

Notes:
 - Assumes absolute XY (G90) and outputs incremental I/J for arcs.
 - Only fits arcs in XY plane (G17). Z changes or G0 rapids break chains.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import List, Tuple, Optional


Word = Tuple[str, float]


def parse_line_words(line: str) -> List[Word]:
    # Strip inline comments ; or ( ... ) while preserving original for pass-through
    s = re.sub(r"\(.*?\)", "", line)
    s = s.split(";")[0]
    words: List[Word] = []
    for m in re.finditer(r"([A-Za-z])(\s*[-+]?\d+(?:\.\d+)?)", s):
        letter = m.group(1).upper()
        try:
            value = float(m.group(2))
        except ValueError:
            continue
        words.append((letter, value))
    return words


def fmt(val: float, decimals: int) -> str:
    return f"{val:.{decimals}f}"


def circle_fit_pratt(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    # Robust enough for small arcs; returns (cx, cy, r)
    n = len(points)
    if n < 3:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    # Shift to centroid
    u = [p[0] - mx for p in points]
    v = [p[1] - my for p in points]
    Suu = sum(ui * ui for ui in u)
    Svv = sum(vi * vi for vi in v)
    Suv = sum(ui * vi for ui, vi in zip(u, v))
    Suuu = sum(ui ** 3 for ui in u)
    Svvv = sum(vi ** 3 for vi in v)
    Suvv = sum(ui * (vi ** 2) for ui, vi in zip(u, v))
    Svuu = sum(vi * (ui ** 2) for ui, vi in zip(u, v))
    det = (Suu * Svv - Suv * Suv)
    if abs(det) < 1e-12:
        return None
    a = (Suuu + Suvv) / 2.0
    b = (Svvv + Svuu) / 2.0
    uc = (a * Svv - b * Suv) / det
    vc = (b * Suu - a * Suv) / det
    cx = mx + uc
    cy = my + vc
    r = math.sqrt(uc * uc + vc * vc + (Suu + Svv) / n)
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)):
        return None
    return cx, cy, r


def unwrap_angles(angles: List[float]) -> List[float]:
    if not angles:
        return angles
    out = [angles[0]]
    base = angles[0]
    for ang in angles[1:]:
        a = ang
        while a - out[-1] > math.pi:
            a -= 2 * math.pi
        while a - out[-1] < -math.pi:
            a += 2 * math.pi
        out.append(a)
    return out


def max_radial_deviation(points: List[Tuple[float, float]], cx: float, cy: float, r: float) -> float:
    dev = 0.0
    for x, y in points:
        dev = max(dev, abs(math.hypot(x - cx, y - cy) - r))
    return dev


def fit_arc_over_chain(points: List[Tuple[float, float]], tol: float, min_radius: float,
                       min_arc_len: float) -> Optional[Tuple[int, Tuple[float, float, float, float, float, int]]]:
    """
    Try to fit an arc starting at points[0] over as many subsequent points as possible.
    Returns (last_index, (cx, cy, r, start_angle, end_angle, dir_sign))
    where dir_sign = +1 for CCW (G3), -1 for CW (G2).
    """
    if len(points) < 3:
        return None
    best_j = None
    best_params = None
    for j in range(2, len(points)):
        sub = points[: j + 1]
        fit = circle_fit_pratt(sub)
        if not fit:
            break
        cx, cy, r = fit
        if r < min_radius:
            break
        dev = max_radial_deviation(sub, cx, cy, r)
        if dev > tol:
            break
        # Direction and angle checks
        angs = [math.atan2(y - cy, x - cx) for (x, y) in sub]
        angs = unwrap_angles(angs)
        # check consistent monotonic direction (allow tiny noise)
        diffs = [angs[k + 1] - angs[k] for k in range(len(angs) - 1)]
        pos = sum(1 for d in diffs if d > 1e-8)
        neg = sum(1 for d in diffs if d < -1e-8)
        if pos > 0 and neg > 0:
            # direction flips -> not a valid single arc
            break
        dir_sign = 1 if pos >= neg else -1
        start_a = angs[0]
        end_a = angs[-1]
        arc_len = r * abs(end_a - start_a)
        if arc_len < min_arc_len:
            # Not worth emitting yet; keep expanding
            best_j = j
            best_params = (cx, cy, r, start_a, end_a, dir_sign)
            continue
        best_j = j
        best_params = (cx, cy, r, start_a, end_a, dir_sign)
    if best_j is None or best_params is None:
        return None
    return best_j, best_params


def split_arc_over_pi(cx: float, cy: float, r: float, a0: float, a1: float, dir_sign: int) -> List[Tuple[float, float]]:
    """
    Return list of intermediate target angles to ensure each sweep <= pi radians.
    Includes only internal split points (not endpoints).
    """
    total = a1 - a0
    if dir_sign < 0 and total > 0:
        # ensure sign coherence with dir
        total -= 2 * math.pi
    if dir_sign > 0 and total < 0:
        total += 2 * math.pi
    steps = int(abs(total) // math.pi)
    if steps <= 1:
        return []
    step = total / (steps + 1)
    return [a0 + step * k for k in range(1, steps + 1)]


def arcfit_gcode(lines: List[str], tol: float, min_radius: float, min_length: float, decimals: int,
                 split_gt_180: bool) -> List[str]:
    out: List[str] = []
    plane = "G17"  # active plane; keep default XY
    abs_mode = True
    x = y = z = 0.0
    have_xy = False
    chain: List[Tuple[float, float]] = []
    chain_raw_lines: List[str] = []

    def flush_chain():
        nonlocal chain, chain_raw_lines, x, y
        if len(chain) < 3:
            out.extend(chain_raw_lines)
        else:
            i = 0
            cur = chain[0]
            out.append(chain_raw_lines[0])  # move to start via original G1
            i += 1
            while i < len(chain):
                sub = chain[i - 1 :]
                fit = fit_arc_over_chain(sub, tol, min_radius, min_length)
                if not fit:
                    # emit the original linear for this point
                    out.append(chain_raw_lines[i])
                    i += 1
                else:
                    last_idx, params = fit
                    cx, cy, r, a0, a1, dir_sign = params
                    start = sub[0]
                    end = sub[last_idx]
                    # Emit one or more arcs, possibly split
                    splits = []
                    if split_gt_180:
                        splits = split_arc_over_pi(cx, cy, r, a0, a1, dir_sign)
                    targets = [a0] + splits + [a1]
                    for t0, t1 in zip(targets[:-1], targets[1:]):
                        ex = cx + r * math.cos(t1)
                        ey = cy + r * math.sin(t1)
                        i_ofs = cx - start[0]
                        j_ofs = cy - start[1]
                        cmd = "G03" if dir_sign > 0 else "G02"
                        out.append(f"{cmd} X{fmt(ex, decimals)} Y{fmt(ey, decimals)} I{fmt(i_ofs, decimals)} J{fmt(j_ofs, decimals)}")
                        start = (ex, ey)
                    i += last_idx
        chain = []
        chain_raw_lines = []

    for raw in lines:
        line = raw.rstrip("\n")
        words = parse_line_words(line)
        g_codes = [int(w[1]) for w in words if w[0] == "G" and w[1].is_integer()]
        modal_g = g_codes[-1] if g_codes else None
        # Handle plane and modes
        for w, v in words:
            if w == "G":
                iv = int(v) if v.is_integer() else None
                if iv == 17:
                    plane = "G17"
                elif iv == 18 or iv == 19:
                    # not supported for arc fitting; flush
                    flush_chain()
                    plane = f"G{iv}"
                elif iv == 90:
                    abs_mode = True
                elif iv == 91:
                    abs_mode = False
        # Non-XY moves or unsupported modes break the chain
        if modal_g in (0, None) or plane != "G17":
            flush_chain()
            out.append(line)
            # update position if G0 contains XY
            if modal_g == 0:
                nx, ny = x, y
                for w, v in words:
                    if w == "X":
                        nx = v if abs_mode else nx + v
                    elif w == "Y":
                        ny = v if abs_mode else ny + v
                    elif w == "Z":
                        z = v if abs_mode else z + v
                x, y = nx, ny
            continue

        # If Z is specified, flush and pass-through
        if any(w == "Z" for w, _ in words):
            flush_chain()
            out.append(line)
            # Update XYZ
            nx, ny, nz = x, y, z
            for w, v in words:
                if w == "X":
                    nx = v if abs_mode else nx + v
                elif w == "Y":
                    ny = v if abs_mode else ny + v
                elif w == "Z":
                    nz = v if abs_mode else nz + v
            x, y, z = nx, ny, nz
            continue

        # Only consider G1 moves with XY for arc fitting
        if modal_g == 1 and any(w in ("X", "Y") for w, _ in words):
            nx, ny = x, y
            for w, v in words:
                if w == "X":
                    nx = v if abs_mode else nx + v
                elif w == "Y":
                    ny = v if abs_mode else ny + v
            if not chain:
                chain.append((x, y))
                chain_raw_lines.append(f"G01 X{fmt(x, decimals)} Y{fmt(y, decimals)}")
            chain.append((nx, ny))
            chain_raw_lines.append(f"G01 X{fmt(nx, decimals)} Y{fmt(ny, decimals)}")
            x, y = nx, ny
            have_xy = True
            continue

        # Any other line: flush and pass through
        flush_chain()
        out.append(line)

    # flush at end
    flush_chain()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert G1 chains to G2/G3 arcs with tolerance.")
    ap.add_argument("input", help="Input G-code file")
    ap.add_argument("-o", "--output", help="Output G-code file (default: <input>_arcfit.nc)")
    ap.add_argument("--tol", type=float, default=0.05, help="Chordal tolerance in mm (default: 0.05)")
    ap.add_argument("--min-radius", type=float, default=0.5, help="Minimum arc radius in mm (default: 0.5)")
    ap.add_argument("--min-length", type=float, default=1.0, help="Minimum arc length in mm (default: 1.0)")
    ap.add_argument("--decimals", type=int, default=3, help="Decimal places for XY/IJ (default: 3)")
    ap.add_argument("--split-gt-180", action="store_true", help="Split arcs that exceed 180 degrees")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or f"{os.path.splitext(in_path)[0]}_arcfit.nc"
    if not os.path.exists(in_path):
        print(f"Input not found: {in_path}")
        return 2
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    out_lines = arcfit_gcode(lines, tol=args.tol, min_radius=args.min_radius,
                             min_length=args.min_length, decimals=args.decimals,
                             split_gt_180=args.split_gt_180)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Arc fitting complete. Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

