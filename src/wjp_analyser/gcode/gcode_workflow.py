"""
G-code Workflow Page/Module
===========================

This module encapsulates the post-analysis workflow for generating
manufacturing artifacts:
- Toolpath ordering
- Cost estimation
- Layered DXF export
- Lengths CSV export
- Report JSON export

The DXF analysis module is intentionally free of these responsibilities.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Optional
import os
import json

from shapely.geometry import Polygon

# We depend on analysis helpers for component layering and nesting
from ..analysis.dxf_analyzer import (
    AnalyzeArgs,
    _layers_from_components,
)

# Import services for consolidated logic
from ..services.costing_service import estimate_cost_from_toolpath
from ..services.layered_dxf_service import write_layered_dxf_from_layer_buckets


Polyline = List[Tuple[float, float]]
LayerBuckets = Dict[str, List[Polyline]]


def generate_toolpath(layer_groups: LayerBuckets) -> List[Polyline]:
    """Order polylines for cutting: holes/inners first, outers last, nearest-neighbor within sets."""
    import math
    from shapely.geometry import Polygon as _Poly

    def center(poly: Polyline) -> tuple[float, float]:
        try:
            p = _Poly(poly)
            x, y = p.centroid.x, p.centroid.y
            return float(x), float(y)
        except Exception:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            return (float(sum(xs) / max(1, len(xs))), float(sum(ys) / max(1, len(ys))))

    def nn_order(polys: list[Polyline], start_xy: tuple[float, float]) -> tuple[list[Polyline], tuple[float, float]]:
        remaining = polys[:]
        ordered: list[Polyline] = []
        cx, cy = start_xy
        while remaining:
            best_i = 0
            best_d = float("inf")
            for i, poly in enumerate(remaining):
                px, py = center(poly)
                d = (px - cx) * (px - cx) + (py - cy) * (py - cy)
                if d < best_d:
                    best_d = d
                    best_i = i
            chosen = remaining.pop(best_i)
            ordered.append(chosen)
            cx, cy = center(chosen)
        return ordered, (cx, cy)

    priority = ["HOLE", "INNER", "DECOR", "COMPLEX", "OUTER"]
    head = (0.0, 0.0)
    result: list[Polyline] = []
    for lname in priority:
        batch = layer_groups.get(lname, [])
        if not batch:
            continue
        seq, head = nn_order(batch, head)
        result.extend(seq)
    return result


# Legacy function kept for backward compatibility, but now uses service
def calculate_cost(toolpath: Iterable[Polyline], rate_per_mtr: float, pierce_cost: float) -> dict:
    """
    Calculate cost from toolpath.
    
    DEPRECATED: Use wjp_analyser.services.costing_service.estimate_cost_from_toolpath() instead.
    This function is kept for backward compatibility.
    """
    result = estimate_cost_from_toolpath(toolpath, rate_per_m=rate_per_mtr, pierce_cost=pierce_cost)
    # Return in old format for compatibility
    return {
        "cutting_length_mm": result["cutting_length_mm"],
        "pierce_count": result["pierce_count"],
        "cutting_cost": result["cutting_cost"],
        "total_cost": result["total_cost"],
    }


# Legacy function kept for backward compatibility, but now uses service
def write_layered_dxf(layers: LayerBuckets, output_path: str) -> str:
    """
    Write layered DXF from layer buckets.
    
    DEPRECATED: Use wjp_analyser.services.layered_dxf_service.write_layered_dxf_from_layer_buckets() instead.
    This function is kept for backward compatibility.
    """
    return write_layered_dxf_from_layer_buckets(layers, output_path)


def write_lengths_csv(layers: LayerBuckets, csv_path: str) -> str:
    import csv
    rows: list[dict] = []
    for layer_name, polygons in layers.items():
        for idx, pts in enumerate(polygons):
            try:
                poly = Polygon(pts)
                perimeter = poly.length
                area = poly.area
            except Exception:
                perimeter = 0.0
                area = 0.0
            rows.append({
                "layer": layer_name,
                "id": idx,
                "perimeter_mm": round(perimeter, 2),
                "area_mm2": round(area, 2),
            })
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "id", "perimeter_mm", "area_mm2"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_report(report: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def run_gcode_workflow(
    components: list[dict],
    args: AnalyzeArgs,
) -> dict:
    """Produce manufacturing outputs from analyzed components.

    Returns a dict with paths and metrics.
    """
    os.makedirs(args.out, exist_ok=True)
    layers = _layers_from_components(components)

    toolpath = generate_toolpath(layers)
    cost = calculate_cost(toolpath, rate_per_mtr=args.rate_per_m, pierce_cost=args.pierce_cost)

    layered_path = os.path.join(args.out, "layered_output.dxf")
    lengths_path = os.path.join(args.out, "lengths.csv")
    report_path = os.path.join(args.out, "report.json")

    write_lengths_csv(layers, lengths_path)
    write_layered_dxf(layers, layered_path)

    report = {
        "metrics": {
            "length_internal_mm": cost["cutting_length_mm"],
            "length_outer_mm": cost["cutting_length_mm"],
            "pierces": cost["pierce_count"],
            "estimated_cutting_cost_inr": cost["total_cost"],
        },
        "toolpath": {"order": list(range(len(toolpath)))},
        "artifacts": {
            "report_json": report_path,
            "lengths_csv": lengths_path,
            "layered_dxf": layered_path,
        },
    }
    write_report(report, report_path)

    return report


