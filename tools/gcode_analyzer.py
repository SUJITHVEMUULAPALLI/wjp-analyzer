#!/usr/bin/env python3
"""
G-code Analyzer for Waterjet
 - Reports segment counts, pierces, cut vs travel distances, basic stats
 - Detects jet on/off via M62/M63 (also supports M3/M5 and M8/M9 fallback)

Usage:
  python tools/gcode_analyzer.py "Program (26).nc" -o output\program26_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple


def parse_words(line: str):
    # Remove () comments and ; trailing comments
    s = re.sub(r"\(.*?\)", "", line)
    s = s.split(";")[0]
    out = []
    for m in re.finditer(r"([A-Za-z])(\s*[-+]?\d+(?:\.\d+)?)", s):
        out.append((m.group(1).upper(), float(m.group(2))))
    return out


def analyze_gcode(lines: List[str]) -> Dict:
    abs_mode = True  # G90/G91
    x = y = z = 0.0
    last_xy = (0.0, 0.0)
    jet_on = False
    jet_seen = False

    # Per-object capture while jet is ON
    # Each object stores: start_point (first move start), end_point (last XY), and list of points encountered
    objects: List[Dict] = []
    cur_points: List[Tuple[float, float]] = []
    cur_start_point: Tuple[float, float] | None = None

    counts = {
        "lines": 0,
        "g0": 0,
        "g1": 0,
        "g2": 0,
        "g3": 0,
        "pierces": 0,
        "jet_on_cmds": 0,
        "jet_off_cmds": 0,
    }
    # Distances
    dists = {
        "travel_mm": 0.0,
        "cut_mm": 0.0,
        "g1_len_mm": 0.0,
        "g2g3_len_mm": 0.0,
    }
    g1_lengths = []

    def hypot2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        counts["lines"] += 1
        words = parse_words(line)
        # modal G
        g_code = None
        for w, v in words:
            if w == "G":
                gv = int(v) if float(v).is_integer() else None
                if gv in (0, 1, 2, 3, 17, 18, 19, 90, 91):
                    g_code = gv
                if gv == 90:
                    abs_mode = True
                elif gv == 91:
                    abs_mode = False
        # M codes for jet
        if re.search(r"\bM\s*62\b", line, re.IGNORECASE) or re.search(r"\bM\s*3\b", line, re.IGNORECASE) or re.search(r"\bM\s*8\b", line, re.IGNORECASE):
            if not jet_on:
                counts["pierces"] += 1
            jet_on = True
            jet_seen = True
            counts["jet_on_cmds"] += 1
        if re.search(r"\bM\s*63\b", line, re.IGNORECASE) or re.search(r"\bM\s*5\b", line, re.IGNORECASE) or re.search(r"\bM\s*9\b", line, re.IGNORECASE):
            # Closing current object if any
            if jet_on:
                # End point is the current x,y before turning off
                end_pt = (x, y)
                if cur_points and cur_start_point is not None:
                    objects.append({
                        "start_point": cur_start_point,
                        "end_point": end_pt,
                        "points": cur_points[:],
                    })
                # reset
                cur_points = []
                cur_start_point = None
            jet_on = False
            counts["jet_off_cmds"] += 1

        # Moves
        if g_code in (0, 1, 2, 3):
            nx, ny = x, y
            i_off = j_off = 0.0
            for w, v in words:
                if w == "X":
                    nx = v if abs_mode else nx + v
                elif w == "Y":
                    ny = v if abs_mode else ny + v
                elif w == "I":
                    i_off = v
                elif w == "J":
                    j_off = v
            start = (x, y)
            end = (nx, ny)
            if end != start:
                if g_code == 0:
                    counts["g0"] += 1
                    d = hypot2(start, end)
                    dists["travel_mm"] += d
                elif g_code == 1:
                    counts["g1"] += 1
                    d = hypot2(start, end)
                    dists["g1_len_mm"] += d
                    g1_lengths.append(d)
                    if jet_on or not jet_seen:
                        dists["cut_mm"] += d
                    else:
                        dists["travel_mm"] += d
                    # Capture object points when jet is on
                    if jet_on:
                        if cur_start_point is None:
                            cur_start_point = start
                            if not cur_points:
                                cur_points.append(start)
                        cur_points.append(end)
                elif g_code in (2, 3):
                    if g_code == 2:
                        counts["g2"] += 1
                    else:
                        counts["g3"] += 1
                    # Estimate arc length from I/J center if provided; else approximate chord
                    if i_off != 0.0 or j_off != 0.0:
                        cx = start[0] + i_off
                        cy = start[1] + j_off
                        r = math.hypot(start[0] - cx, start[1] - cy)
                        a0 = math.atan2(start[1] - cy, start[0] - cx)
                        a1 = math.atan2(end[1] - cy, end[0] - cx)
                        da = a1 - a0
                        # normalize to the commanded direction minimal sweep
                        while da <= -math.pi:
                            da += 2 * math.pi
                        while da > math.pi:
                            da -= 2 * math.pi
                        if g_code == 2 and da > 0:
                            da -= 2 * math.pi
                        if g_code == 3 and da < 0:
                            da += 2 * math.pi
                        d = abs(r * da)
                    else:
                        d = hypot2(start, end)
                    dists["g2g3_len_mm"] += d
                    if jet_on or not jet_seen:
                        dists["cut_mm"] += d
                    else:
                        dists["travel_mm"] += d
                    # Capture object points when jet is on (use endpoints; arc midpoints not sampled)
                    if jet_on:
                        if cur_start_point is None:
                            cur_start_point = start
                            if not cur_points:
                                cur_points.append(start)
                        cur_points.append(end)
            x, y = nx, ny

    # If file ends with jet on and no explicit off, close last object
    if jet_on and cur_points and cur_start_point is not None:
        objects.append({
            "start_point": cur_start_point,
            "end_point": (x, y),
            "points": cur_points[:],
        })
        cur_points = []
        cur_start_point = None

    # Link analysis: measure actual travel vs nearest-corner travel to next object
    link_reports: List[Dict] = []
    extra_total = 0.0
    success = 0
    for i in range(1, len(objects)):
        prev_end = tuple(objects[i - 1]["end_point"])  # type: ignore[index]
        cur = objects[i]
        cur_start = tuple(cur.get("start_point", prev_end))  # type: ignore[assignment]
        pts = cur.get("points", [])
        # actual travel = distance(prev_end -> cur_start)
        actual = hypot2(prev_end, cur_start)
        # nearest corner among commanded points
        nearest = min((hypot2(prev_end, p) for p in pts), default=actual)
        is_nearest = actual <= nearest + 1e-3
        extra = max(0.0, actual - nearest)
        extra_total += extra
        if is_nearest:
            success += 1
        link_reports.append({
            "from_object": i - 1,
            "to_object": i,
            "prev_end": [round(prev_end[0], 3), round(prev_end[1], 3)],
            "cur_start": [round(cur_start[0], 3), round(cur_start[1], 3)],
            "actual_travel_mm": round(actual, 3),
            "nearest_corner_travel_mm": round(nearest, 3),
            "extra_travel_mm": round(extra, 3),
            "nearest_corner_start": is_nearest,
        })

    total_len = dists["cut_mm"] + dists["travel_mm"]
    seg_count = counts["g1"] + counts["g2"] + counts["g3"]
    avg_len = (sum(g1_lengths) / len(g1_lengths)) if g1_lengths else 0.0
    g1_sorted = sorted(g1_lengths)
    med_len = (g1_sorted[len(g1_sorted) // 2] if g1_sorted else 0.0)

    return {
        "lines": counts["lines"],
        "segments": {
            "g0": counts["g0"],
            "g1": counts["g1"],
            "g2": counts["g2"],
            "g3": counts["g3"],
            "total_move_segments": seg_count,
        },
        "pierces": counts["pierces"],
        "jet": {"on_cmds": counts["jet_on_cmds"], "off_cmds": counts["jet_off_cmds"]},
        "lengths_mm": {
            "cut": round(dists["cut_mm"], 3),
            "travel": round(dists["travel_mm"], 3),
            "total": round(total_len, 3),
            "g1_total": round(dists["g1_len_mm"], 3),
            "g2g3_total": round(dists["g2g3_len_mm"], 3),
        },
        "ratios": {
            "cut_ratio": round((dists["cut_mm"] / total_len) if total_len > 0 else 0.0, 4),
            "travel_ratio": round((dists["travel_mm"] / total_len) if total_len > 0 else 0.0, 4),
        },
        "g1_stats_mm": {
            "count": len(g1_lengths),
            "avg": round(avg_len, 4),
            "median": round(med_len, 4),
            "min": round(min(g1_lengths) if g1_lengths else 0.0, 4),
            "max": round(max(g1_lengths) if g1_lengths else 0.0, 4),
        },
        "objects": {
            "count": len(objects),
            "link_evaluation": {
                "links": link_reports,
                "links_count": len(link_reports),
                "nearest_start_success": success,
                "nearest_start_rate": round((success / len(link_reports)) if link_reports else 0.0, 4),
                "total_extra_travel_mm": round(extra_total, 3),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze G-code: segments, pierces, cut/travel ratio")
    ap.add_argument("gcode", help="Input .nc/.gcode file path")
    ap.add_argument("-o", "--output", help="Optional JSON output path")
    args = ap.parse_args()
    if not os.path.exists(args.gcode):
        print(f"Input not found: {args.gcode}")
        return 2
    with open(args.gcode, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    report = analyze_gcode(lines)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as h:
            json.dump(report, h, indent=2)
        print(f"Saved analysis: {args.output}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
