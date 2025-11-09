"""
Layered DXF Service
===================

First-class service for writing layered DXF files from analysis components.
This service consolidates DXF writing logic that was previously scattered
across UI pages and the gcode workflow module.

This addresses the issue where the analyzer doesn't write layered DXF by
default, forcing workarounds in the UI layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import os


def write_layered_dxf_from_components(
    components: List[Dict[str, Any]],
    output_path: str,
    dxf_version: str = "R2010",
    close_polylines: bool = True,
) -> str:
    """
    Write a layered DXF file from analysis report components.
    
    This is the canonical way to write layered DXF files from analysis results.
    It handles component-to-polyline conversion, layer management, and DXF creation.
    
    Args:
        components: List of component dictionaries from analysis report.
                   Each component must have:
                   - "points": List of [x, y] coordinate pairs
                   - "layer": Layer name (defaults to "0" if missing)
        output_path: Path where the DXF file should be saved
        dxf_version: DXF version (default: "R2010")
        close_polylines: Whether to close polylines (default: True)
        
    Returns:
        Path to the saved DXF file
        
    Raises:
        ImportError: If ezdxf is not available
        ValueError: If components list is empty
        IOError: If output path cannot be written
    """
    import ezdxf
    
    if not components:
        raise ValueError("Components list cannot be empty")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create new DXF document
    doc = ezdxf.new(dxf_version, setup=True)
    msp = doc.modelspace()
    
    # Collect unique layers from components
    layers = set()
    for comp in components:
        layer = comp.get("layer", "0")
        layers.add(layer)
    
    # Create layers in DXF document
    for layer_name in sorted(layers):
        if layer_name not in doc.layers:
            doc.layers.new(name=layer_name)
    
    # Write components as polylines
    written_count = 0
    errors = []
    
    for comp in components:
        pts = comp.get("points", [])
        if not pts or len(pts) < 2:
            continue  # Skip invalid components
            
        layer = comp.get("layer", "0")
        
        # Convert points to list of tuples
        try:
            polyline_points = [(float(p[0]), float(p[1])) for p in pts]
        except (ValueError, TypeError, IndexError) as e:
            errors.append(f"Component {comp.get('id', 'unknown')}: Invalid points format: {e}")
            continue
        
        # Add polyline to DXF
        try:
            if close_polylines:
                # Try to close the polyline
                try:
                    msp.add_lwpolyline(
                        polyline_points,
                        dxfattribs={"layer": layer},
                        close=True
                    )
                    written_count += 1
                except Exception:
                    # If closing fails, try without closing
                    msp.add_lwpolyline(
                        polyline_points,
                        dxfattribs={"layer": layer}
                    )
                    written_count += 1
            else:
                msp.add_lwpolyline(
                    polyline_points,
                    dxfattribs={"layer": layer}
                )
                written_count += 1
        except Exception as e:
            errors.append(f"Component {comp.get('id', 'unknown')}: Failed to add polyline: {e}")
            continue
    
    # Save DXF file
    try:
        doc.saveas(output_path)
    except Exception as e:
        raise IOError(f"Failed to save DXF file to {output_path}: {e}")
    
    # Log warnings if any components failed
    if errors:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Some components failed to write: {len(errors)} errors")
        for error in errors[:5]:  # Log first 5 errors
            logger.warning(error)
    
    return output_path


def write_layered_dxf_from_report(
    report: Dict[str, Any],
    output_path: str,
    selected_only: bool = False,
    dxf_version: str = "R2010",
    close_polylines: bool = True,
) -> str:
    """
    Write a layered DXF file from an analysis report.
    
    Convenience function that extracts components from a report and writes them.
    
    Args:
        report: Analysis report dictionary containing "components" key
        output_path: Path where the DXF file should be saved
        selected_only: If True, only write components where "selected" is True
        dxf_version: DXF version (default: "R2010")
        close_polylines: Whether to close polylines (default: True)
        
    Returns:
        Path to the saved DXF file
    """
    components = report.get("components", [])
    
    if not components:
        raise ValueError("Report contains no components")
    
    # Filter by selection if requested
    if selected_only:
        components = [c for c in components if c.get("selected", True)]
    
    return write_layered_dxf_from_components(
        components=components,
        output_path=output_path,
        dxf_version=dxf_version,
        close_polylines=close_polylines,
    )


def write_layered_dxf_from_layer_buckets(
    layer_buckets: Dict[str, List[List[tuple[float, float]]]],
    output_path: str,
    dxf_version: str = "R2010",
    close_polylines: bool = True,
) -> str:
    """
    Write a layered DXF file from layer buckets (legacy format).
    
    This function is provided for compatibility with the existing gcode_workflow
    module that uses LayerBuckets format.
    
    Args:
        layer_buckets: Dictionary mapping layer names to lists of polylines.
                      Each polyline is a list of (x, y) tuples.
        output_path: Path where the DXF file should be saved
        dxf_version: DXF version (default: "R2010")
        close_polylines: Whether to close polylines (default: True)
        
    Returns:
        Path to the saved DXF file
    """
    import ezdxf
    
    if not layer_buckets:
        raise ValueError("Layer buckets dictionary cannot be empty")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create new DXF document
    doc = ezdxf.new(dxf_version, setup=True)
    msp = doc.modelspace()
    
    # Create layers
    for layer_name in sorted(layer_buckets.keys()):
        if layer_name not in doc.layers:
            doc.layers.new(name=layer_name)
    
    # Write polylines
    written_count = 0
    for layer_name, polygons in layer_buckets.items():
        for pts in polygons:
            if len(pts) < 2:
                continue
            try:
                if close_polylines:
                    msp.add_lwpolyline(pts, dxfattribs={"layer": layer_name}, close=True)
                else:
                    msp.add_lwpolyline(pts, dxfattribs={"layer": layer_name})
                written_count += 1
            except Exception:
                # Skip invalid polylines
                continue
    
    # Save DXF file
    doc.saveas(output_path)
    return output_path








