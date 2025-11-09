"""
DXF Export Utilities

Provides export functionality for DXF, SVG, and JSON formats with metadata.
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import ezdxf

from .dxf_renderer import render_svg
from .dxf_utils import entity_summary


def export_dxf(doc, output_path: str, metadata: Optional[Dict] = None) -> str:
    """
    Export DXF document to file.
    
    Args:
        doc: DXF document
        output_path: Output file path
        metadata: Optional metadata to embed in DXF header
    
    Returns:
        Path to exported file
    """
    # Add metadata to DXF header if provided
    if metadata:
        header = doc.header
        for key, value in metadata.items():
            try:
                # Store as custom variable
                header['$CUSTOM_' + str(key)] = str(value)
            except:
                pass
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    doc.saveas(output_path)
    return output_path


def export_svg(doc, output_path: str, layer_visibility: Optional[Dict[str, bool]] = None,
                include_grid: bool = False, grid_size: float = 10.0) -> str:
    """
    Export DXF as SVG file.
    
    Args:
        doc: DXF document
        output_path: Output file path
        layer_visibility: Layer visibility settings
        include_grid: Whether to include grid overlay
        grid_size: Grid spacing if grid is included
    
    Returns:
        Path to exported file
    """
    svg_text = render_svg(doc, layer_visibility=layer_visibility)
    
    if include_grid:
        from .dxf_viewport import add_grid_overlay
        svg_text = add_grid_overlay(svg_text, grid_size=grid_size)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_text)
    
    return output_path


def export_json(doc, output_path: str, edit_log: Optional[List[Dict]] = None,
                include_entities: bool = True) -> str:
    """
    Export DXF metadata and edit history as JSON.
    
    Args:
        doc: DXF document
        output_path: Output file path
        edit_log: Edit log entries
        include_entities: Whether to include entity data
    
    Returns:
        Path to exported file
    """
    data = {
        "export_date": datetime.now().isoformat(),
        "dxf_version": doc.dxfversion,
        "layers": [layer.dxf.name for layer in doc.layers],
        "edit_log": edit_log or [],
    }
    
    if include_entities:
        data["entities"] = entity_summary(doc)
    
    # Add document properties
    try:
        data["properties"] = {
            "units": str(doc.units),
            "modelspace_entity_count": len(list(doc.modelspace())),
        }
    except:
        pass
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_path


def export_all_formats(doc, base_path: str, layer_visibility: Optional[Dict[str, bool]] = None,
                       edit_log: Optional[List[Dict]] = None, include_grid: bool = False) -> Dict[str, str]:
    """
    Export DXF in all formats (DXF, SVG, JSON).
    
    Args:
        doc: DXF document
        base_path: Base path (without extension)
        layer_visibility: Layer visibility settings
        edit_log: Edit log entries
        include_grid: Whether to include grid in SVG
    
    Returns:
        Dictionary mapping format to file path
    """
    base = base_path.rsplit('.', 1)[0] if '.' in base_path else base_path
    
    results = {}
    
    # Export DXF
    dxf_path = f"{base}.dxf"
    results['dxf'] = export_dxf(doc, dxf_path, metadata={
        "export_date": datetime.now().isoformat(),
        "edit_count": len(edit_log) if edit_log else 0,
    })
    
    # Export SVG
    svg_path = f"{base}.svg"
    results['svg'] = export_svg(doc, svg_path, layer_visibility=layer_visibility, 
                                include_grid=include_grid)
    
    # Export JSON
    json_path = f"{base}_metadata.json"
    results['json'] = export_json(doc, json_path, edit_log=edit_log)
    
    return results

