from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from pathlib import Path
import os

# Import layered DXF service for writing DXF files
from .layered_dxf_service import (
    write_layered_dxf_from_components,
    write_layered_dxf_from_report,
)


def list_components(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    comps = report.get("components", []) if isinstance(report, dict) else []
    result: List[Dict[str, Any]] = []
    for c in comps:
        result.append(
            {
                "id": c.get("id"),
                "layer": c.get("layer", "0"),
                "group": c.get("group"),
                "area": float(c.get("area", 0.0)),
                "perimeter": float(c.get("perimeter", 0.0)),
                "selected": True,
            }
        )
    return result


def export_components_csv(report: Dict[str, Any], output_path: str) -> str:
    """
    Export component-level CSV with columns: ID, Layer, Group, Area_mm2, Perimeter_mm, Selected
    
    Args:
        report: Analysis report containing components
        output_path: Path to save the CSV file
        
    Returns:
        Path to the saved CSV file
    """
    comps = report.get("components", []) if isinstance(report, dict) else []
    rows = []
    for c in comps:
        rows.append({
            "ID": c.get("id", ""),
            "Layer": c.get("layer", "0"),
            "Group": c.get("group", "Ungrouped"),
            "Area_mm2": float(c.get("area", 0.0)),
            "Perimeter_mm": float(c.get("perimeter", 0.0)),
            "Selected": True,  # Default all to selected
        })
    
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    return output_path


def set_layer_bulk(doc_path: str, component_ids: List[str], target_layer: str) -> str:
    """Example layer operation using dxf object manager primitives (left as placeholder)."""
    # Intentionally minimal; real implementation can open DXF and change layers by handle
    return doc_path


def write_layered_dxf_from_analysis(
    report: Dict[str, Any],
    output_path: str,
    selected_only: bool = False,
) -> str:
    """
    Write layered DXF from analysis report.
    
    This is a convenience wrapper that uses the layered_dxf_service.
    
    Args:
        report: Analysis report containing components
        output_path: Path to save the DXF file
        selected_only: If True, only write selected components
        
    Returns:
        Path to the saved DXF file
    """
    return write_layered_dxf_from_report(
        report=report,
        output_path=output_path,
        selected_only=selected_only,
    )


