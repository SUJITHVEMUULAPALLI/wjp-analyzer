import os
import uuid
from typing import Dict, Any, Tuple

from shapely.geometry import Polygon

from ..io.dxf_io import load_dxf_lines_with_layers
from ..analysis.geometry_cleaner import merge_and_polygonize
from ..analysis.topology import containment_depth
from ..analysis.classification import classify_by_depth_and_layers
from ..manufacturing.toolpath import plan_order
from ..manufacturing.gcode_generator import write_gcode
from ..manufacturing.cost_calculator import compute_lengths


class DXFProcessingError(RuntimeError):
    """Raised when DXF processing cannot produce usable data."""


def _classify_polygons(dxf_path: str) -> Tuple[list[Polygon], Dict[int, str]]:
    """Load a DXF file and return polygons with classified contour types."""
    lines, layers = load_dxf_lines_with_layers(dxf_path)
    _, polygons = merge_and_polygonize(lines)
    if not polygons:
        raise DXFProcessingError("No closed polygons could be generated from the DXF file.")

    depths = containment_depth(polygons)
    classes = classify_by_depth_and_layers(polygons, depths, lines, layers)
    return polygons, classes


def generate_gcode_from_dxf(
    dxf_path: str,
    output_root: str,
    *,
    feed: float = 1200.0,
    m_on: str = "M62",
    m_off: str = "M63",
    pierce_ms: int = 500,
) -> Dict[str, Any]:
    """Generate G-code for the supplied DXF file and return metadata."""
    polygons, classes = _classify_polygons(dxf_path)
    order = plan_order(polygons, classes)
    if not order:
        order = list(range(len(polygons)))

    gcode_id = f"gcode_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(output_root, gcode_id)
    os.makedirs(output_dir, exist_ok=True)

    gcode_path = os.path.join(output_dir, "program.nc")
    write_gcode(gcode_path, polygons, order, feed=feed, m_on=m_on, m_off=m_off, pierce_ms=pierce_ms)

    with open(gcode_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    metrics = compute_lengths(polygons, classes)
    total_length_mm = metrics.get("length_outer_mm", 0.0) + metrics.get("length_internal_mm", 0.0)
    pierce_count = metrics.get("pierces", len(polygons))

    feed_rate = max(feed, 1.0)
    cutting_minutes = total_length_mm / feed_rate
    pierce_minutes = (pierce_ms * pierce_count) / 60000.0
    estimated_minutes = cutting_minutes + pierce_minutes

    return {
        "gcode_id": gcode_id,
        "gcode_path": gcode_path,
        "line_count": len(lines),
        "gcode_preview": lines,
        "metrics": {
            "length_mm": total_length_mm,
            "pierce_count": pierce_count,
            "polygons": len(polygons),
        },
        "estimated_time_minutes": estimated_minutes,
        "classes": classes,
        "polygons": polygons,
    }


def calculate_costs_from_dxf(
    dxf_path: str,
    *,
    rate_per_m: float = 825.0,
    machine_rate_per_min: float = 20.0,
    pierce_cost: float = 1.5,
    setup_cost: float = 15.0,
    pierce_ms: int = 500,
    feed: float = 1200.0,
) -> Dict[str, Any]:
    """Compute a simple manufacturing cost breakdown from the DXF file."""
    polygons, classes = _classify_polygons(dxf_path)
    metrics = compute_lengths(polygons, classes)

    total_length_mm = metrics.get("length_outer_mm", 0.0) + metrics.get("length_internal_mm", 0.0)
    total_length_m = total_length_mm / 1000.0
    pierce_count = metrics.get("pierces", len(polygons))

    feed_rate = max(feed, 1.0)
    cutting_minutes = total_length_mm / feed_rate
    pierce_minutes = (pierce_ms * pierce_count) / 60000.0
    machine_minutes = cutting_minutes + pierce_minutes

    cutting_cost = total_length_m * rate_per_m
    machine_cost = machine_minutes * machine_rate_per_min
    pierce_total = pierce_count * pierce_cost

    total_cost = cutting_cost + machine_cost + pierce_total + setup_cost

    return {
        "total_cost": round(total_cost, 2),
        "material_cost": 0.0,
        "cutting_cost": round(cutting_cost, 2),
        "machine_cost": round(machine_cost, 2),
        "pierce_cost": round(pierce_total, 2),
        "setup_cost": round(setup_cost, 2),
        "waste_cost": 0.0,
        "metrics": {
            "length_mm": total_length_mm,
            "length_m": total_length_m,
            "pierce_count": pierce_count,
            "cutting_minutes": cutting_minutes,
            "machine_minutes": machine_minutes,
        },
    }














