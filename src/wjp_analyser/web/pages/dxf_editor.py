"""
DXF Editor - Enhanced Streamlit Interface
==========================================

A comprehensive DXF editor with visualization, editing, analysis, and export capabilities.

Features:
- Upload and visualize DXF files
- Layer management (create, rename, recolor, visibility)
- Entity selection and grouping
- Transform operations (translate, scale, rotate)
- Drawing tools (line, circle, rectangle, polyline)
- Measurement and validation tools
- AI-powered analysis and recommendations
- Waterjet-specific cleanup and validation
- Export edited DXF files
"""

import os
from pathlib import Path
import sys
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple

# Ensure 'src' is on sys.path so 'wjp_analyser' is importable when Streamlit runs this page directly
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = str(_THIS_FILE.parents[3])  # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import DXF editor utilities
from wjp_analyser.dxf_editor import (
    load_dxf,
    save_dxf,
    translate,
    scale,
    rotate,
    plot_entities,
    pick_entity,
    ensure_layer,
    rename_layer,
    recolor_layer,
    move_entities_to_layer,
    create_group,
    list_groups,
    add_line,
    add_circle,
    add_rect,
    add_polyline,
    distance,
    check_min_radius,
    kerf_preview_value,
    load_session,
    save_session,
)
from wjp_analyser.dxf_editor.measure import bbox_of_entity, bbox_size, polyline_length
from wjp_analyser.dxf_editor.operation_executor import OperationExecutor
from wjp_analyser.ai.recommendation_engine import Operation, OperationType

# Reload module to avoid stale imports during Streamlit hot-reload
import wjp_analyser.dxf_editor as dxf_mod
dxf_mod = importlib.reload(dxf_mod)

# ============================================================================
# Configuration and Constants
# ============================================================================

OUTPUT_DIR = Path("output") / "dxf_editor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def safe_rerun():
    """Safely trigger a Streamlit rerun."""
    st.rerun()


def initialize_session_state():
    """Initialize session state variables."""
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "selected": [],
            "hidden_layers": [],
            "session_path": "session.json"
        }
    # Initialize analysis results
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None
    if "show_recommendations" not in st.session_state:
        st.session_state["show_recommendations"] = False


def get_layer_names(doc) -> List[str]:
    """Get list of layer names from DXF document."""
    try:
        return list(dxf_mod.get_layers(doc).keys())
    except Exception:
        try:
            return [getattr(e.dxf, "name", getattr(e, "name", "0")) for e in doc.layers]
        except Exception:
            return ["0"]


def render_fallback_preview(doc, save_dxf_func):
    """Render polygonized preview when no supported entities are found."""
    st.markdown("""
    <div style="background-color: #FFEBEE; border-left: 4px solid #F44336; padding: 10px; margin: 10px 0; border-radius: 4px;">
        <p style="color: #C62828; font-weight: 500; margin: 0;">⚠️ <strong>Warning:</strong> No supported entities (LINE, CIRCLE, LWPOLYLINE) found. Showing polygonized preview for reference.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from wjp_analyser.nesting.dxf_extractor import extract_polygons
        from wjp_analyser.dxf_editor.preview_utils import (
            classify_polygon_layers,
            get_layer_color,
            convert_hex_to_rgb
        )
        
        # Save to temp and extract
        temp_dir = Path("output") / "editor_preview"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"preview_src_{int(time.time())}.dxf"
        save_dxf_func(doc, str(temp_path))
        polys = extract_polygons(str(temp_path))
        
        if not polys:
            st.info("No closed polygons detected in this DXF.")
            return
        
        # Classify polygons into layers
        classified = classify_polygon_layers(polys)
        
        # Find bounding box for normalization
        all_points = []
        for p in polys:
            pts = p.get("points", [])
            if len(pts) >= 3:
                all_points.extend(pts)
        
        if not all_points:
            st.info("No valid polygons detected in this DXF.")
            return
        
        min_x = min(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        
        # Create figure with normalized coordinates
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Render polygons with layer-based colors
        layer_order = ["OUTER", "INNER", "HOLE", "COMPLEX", "DECOR"]
        for layer_name in layer_order:
            layer_polys = classified.get(layer_name, [])
            if not layer_polys:
                continue
            
            color_info = get_layer_color(layer_name)
            fill_color = convert_hex_to_rgb(color_info["fill"])
            edge_color = convert_hex_to_rgb(color_info["edge"])
            alpha = color_info["alpha"]
            
            # Render each polygon in this layer with normalization
            for poly_data in layer_polys:
                original_points = poly_data.get("points", [])
                if len(original_points) < 3:
                    continue
                
                # Normalize this polygon's points
                norm_pts = [(x - min_x, y - min_y) for x, y in original_points]
                
                # Ensure polygon is closed for proper rendering
                if norm_pts[0] != norm_pts[-1]:
                    norm_pts.append(norm_pts[0])
                
                xs = [x for x, y in norm_pts]
                ys = [y for x, y in norm_pts]
                
                # Render with appropriate z-order (OUTER first as background)
                z_order = 1 if layer_name == "OUTER" else 2
                ax.fill(xs, ys, color=fill_color, alpha=alpha, 
                       edgecolor=edge_color, linewidth=0.8, zorder=z_order)
        
        # Set up axes
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title("DXF Preview (Normalized to Origin)", fontsize=14, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = []
        for layer_name in ["OUTER", "INNER"]:
            if classified.get(layer_name):
                color_info = get_layer_color(layer_name)
                fill_color = convert_hex_to_rgb(color_info["fill"])
                count = len(classified[layer_name])
                legend_elements.append(
                    Patch(facecolor=fill_color, edgecolor=convert_hex_to_rgb(color_info["edge"]),
                         label=f"{layer_name} ({count})")
                )
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
    except Exception as ex:
        import traceback
        st.error(f"Preview extraction unavailable: {ex}")
        if st.checkbox("Show detailed error", key="show_preview_error"):
            st.code(traceback.format_exc())


def build_warning_markers() -> Dict[str, List[str]]:
    """Build warning markers from AI analysis if available."""
    warning_markers = {}
    
    if not (st.session_state.get("_editor_ai_analysis") and st.session_state.get("_editor_analysis_report")):
        return warning_markers
    
    report = st.session_state.get("_editor_analysis_report")
    
    # Get enabled warnings
    enabled_warnings = {}
    if "editor_recommendations" in st.session_state:
        for w_type, w_data in st.session_state["editor_recommendations"].get("warnings", {}).items():
            if w_data.get("enabled", True):
                enabled_warnings[w_type] = w_data.get("data", {})
    
    # Map component IDs to entity handles via handle attribute
    components = report.get("components", [])
    for comp in components:
        comp_id = comp.get("id")
        handle = comp.get("handle")
        comp_warnings = []
        
        # Check warning conditions
        if comp.get("area", 0) == 0 and "zero_area" in enabled_warnings:
            comp_warnings.append("zero_area")
        area = comp.get("area", 0)
        if 0 < area < 1.0 and "too_many_tiny" in enabled_warnings:
            comp_warnings.append("too_many_tiny")
        
        if comp_warnings and handle:
            warning_markers[handle] = comp_warnings
    
    return warning_markers


# ============================================================================
# UI Components
# ============================================================================

def render_layer_management(doc, layer_names: List[str]):
    """Render layer management UI in sidebar."""
    st.header("Layers")
    
    # Create new layer
    new_layer = st.text_input("Create Layer", value="WJP_NEW", key="new_layer_input")
    if st.button("Add Layer", key="add_layer_btn"):
        try:
            ensure_layer(doc, new_layer, color=7)
            st.success(f"Layer '{new_layer}' created.")
            safe_rerun()
        except BaseException as e:
            # Re-raise Streamlit's internal rerun exceptions immediately
            exception_type = type(e).__name__
            exception_module = type(e).__module__ if hasattr(type(e), '__module__') else ''
            if "Rerun" in exception_type or "streamlit" in exception_module.lower():
                if "Rerun" in exception_type:
                    raise  # Re-raise Streamlit's internal rerun exceptions
            # Only show error for actual failures
            st.error(f"Failed to create layer: {e}")
    
    # Rename layer
    if layer_names:
        layer_pick = st.selectbox("Pick Layer", options=layer_names, key="layer_pick_select")
        new_name = st.text_input("Rename Layer To", value=layer_pick, key="rename_layer_input")
        if st.button("Rename Layer", key="rename_layer_btn"):
            try:
                rename_layer(doc, layer_pick, new_name)
                st.success(f"Renamed layer {layer_pick} → {new_name}")
                safe_rerun()
            except BaseException as e:
                exception_type = type(e).__name__
                if "Rerun" in exception_type:
                    raise
                st.error(f"Failed to rename layer: {e}")
        
        # Recolor layer
        new_color = st.number_input("ACI Color (1-255)", value=7, min_value=1, max_value=255, step=1, key="layer_color_input")
        if st.button("Recolor Layer", key="recolor_layer_btn"):
            try:
                recolor_layer(doc, layer_pick, int(new_color))
                st.success(f"Layer {layer_pick} recolored to {int(new_color)}")
            except Exception as e:
                st.error(f"Failed to recolor layer: {e}")
    
    # Layer visibility
    st.subheader("Layer Visibility")
    vis_multi = st.multiselect(
        "Hidden Layers", 
        options=layer_names if layer_names else ["0"], 
        default=st.session_state["state"]["hidden_layers"],
        key="hidden_layers_select"
    )
    st.session_state["state"]["hidden_layers"] = vis_multi


def render_group_management(doc, entities: List):
    """Render group management UI in sidebar."""
    st.header("Groups")
    
    try:
        existing_groups = list_groups(doc)
        if existing_groups:
            st.write("Existing:", existing_groups)
        else:
            st.caption("No groups yet")
    except Exception as e:
        st.warning(f"Could not list groups: {e}")
    
    grp_name = st.text_input("New Group Name", value="group_1", key="group_name_input")
    if st.button("Group Selection", key="group_selection_btn"):
        try:
            sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
            if sel:
                create_group(doc, grp_name, [e.dxf.handle for e in sel])
                st.success(f"Created group '{grp_name}' with {len(sel)} entities")
            else:
                st.warning("No entities selected. Select entities first.")
        except Exception as e:
            st.error(f"Failed to create group: {e}")


def render_selection_tools(doc, entities: List):
    """Render entity selection tools in sidebar."""
    st.header("Selection")
    
    selected_count = len(st.session_state["state"]["selected"])
    st.caption(f"{selected_count} entity(ies) selected")
    
    if selected_count > 0:
        if st.button("Clear Selection", key="clear_selection_btn"):
            st.session_state["state"]["selected"] = []
            safe_rerun()
    
    # Pick entity by coordinates
    st.subheader("Pick Entity")
    col_x, col_y = st.columns(2)
    with col_x:
        x = st.number_input("X", value=0.0, key="pick_x_input")
    with col_y:
        y = st.number_input("Y", value=0.0, key="pick_y_input")
    tol = st.number_input("Tolerance", value=2.0, min_value=0.1, step=0.1, key="pick_tol_input")
    
    if st.button("Pick @ (X,Y)", key="pick_entity_btn"):
        try:
            picked = pick_entity(entities, x, y, tol)
            if picked:
                handle = picked.dxf.handle
                if handle not in st.session_state["state"]["selected"]:
                    st.session_state["state"]["selected"].append(handle)
                    st.success(f"Selected {picked.dxftype()} on layer {picked.dxf.layer}")
                else:
                    st.info("Entity already selected")
            else:
                st.warning("No entity near that point.")
        except Exception as e:
            st.error(f"Failed to pick entity: {e}")
    
    # Move selected to layer
    st.subheader("Move Selected to Layer")
    move_to = st.text_input("Target Layer", value="OUTER", key="move_to_layer_input")
    if st.button("Move to Layer", key="move_to_layer_btn"):
        try:
            sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
            if sel:
                ensure_layer(doc, move_to, color=7)
                move_entities_to_layer(sel, move_to)
                st.success(f"Moved {len(sel)} entities to layer '{move_to}'")
            else:
                st.warning("No entities selected.")
        except Exception as e:
            st.error(f"Failed to move entities: {e}")


def render_transform_tools(doc, msp, entities: List):
    """Render transform operations UI."""
    st.header("Transform Operations")
    st.caption("Apply transformations to selected entities")
    
    selected_count = len(st.session_state["state"]["selected"])
    if selected_count == 0:
        st.info("Select entities first to apply transformations")
        return
    
    st.write(f"**{selected_count} entity(ies) selected**")
    
    # Translate
    with st.expander("📍 Translate (Move)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            dx = st.number_input("ΔX (mm)", value=0.0, step=1.0, key="translate_dx")
        with col2:
            dy = st.number_input("ΔY (mm)", value=0.0, step=1.0, key="translate_dy")
        if st.button("Apply Translation", key="translate_btn"):
            try:
                sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
                for e in sel:
                    translate(e, dx, dy)
                st.success(f"Translated {len(sel)} entities by ({dx}, {dy})")
                safe_rerun()
            except Exception as e:
                st.error(f"Translation failed: {e}")
    
    # Scale
    with st.expander("🔍 Scale"):
        scale_factor = st.number_input("Scale Factor", value=1.0, min_value=0.01, max_value=100.0, step=0.1, key="scale_factor")
        if st.button("Apply Scaling", key="scale_btn"):
            try:
                sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
                for e in sel:
                    scale(e, scale_factor)
                st.success(f"Scaled {len(sel)} entities by {scale_factor}x")
                safe_rerun()
            except Exception as e:
                st.error(f"Scaling failed: {e}")
    
    # Rotate
    with st.expander("🔄 Rotate"):
        angle = st.number_input("Angle (degrees)", value=0.0, step=15.0, key="rotate_angle")
        if st.button("Apply Rotation", key="rotate_btn"):
            try:
                sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
                for e in sel:
                    rotate(e, angle)
                st.success(f"Rotated {len(sel)} entities by {angle}°")
                safe_rerun()
            except Exception as e:
                st.error(f"Rotation failed: {e}")


def render_drawing_tools(doc, msp):
    """Render drawing tools UI."""
    st.header("Drawing Tools")
    st.caption("Add new entities to the drawing")
    
    # Get current layer
    layer_names = get_layer_names(doc)
    current_layer = st.selectbox("Target Layer", options=layer_names if layer_names else ["0"], key="draw_layer")
    
    # Line
    with st.expander("📏 Line", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x1 = st.number_input("X1", value=0.0, key="line_x1")
        with col2:
            y1 = st.number_input("Y1", value=0.0, key="line_y1")
        with col3:
            x2 = st.number_input("X2", value=100.0, key="line_x2")
        with col4:
            y2 = st.number_input("Y2", value=100.0, key="line_y2")
        if st.button("Add Line", key="add_line_btn"):
            try:
                add_line(msp, x1, y1, x2, y2, layer=current_layer)
                st.success(f"Line added from ({x1}, {y1}) to ({x2}, {y2})")
                safe_rerun()
            except Exception as e:
                st.error(f"Failed to add line: {e}")
    
    # Circle
    with st.expander("⭕ Circle"):
        col1, col2, col3 = st.columns(3)
        with col1:
            cx = st.number_input("Center X", value=0.0, key="circle_cx")
        with col2:
            cy = st.number_input("Center Y", value=0.0, key="circle_cy")
        with col3:
            radius = st.number_input("Radius", value=50.0, min_value=0.1, key="circle_r")
        if st.button("Add Circle", key="add_circle_btn"):
            try:
                add_circle(msp, cx, cy, radius, layer=current_layer)
                st.success(f"Circle added at ({cx}, {cy}) with radius {radius}")
                safe_rerun()
            except Exception as e:
                st.error(f"Failed to add circle: {e}")
    
    # Rectangle
    with st.expander("▭ Rectangle"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rect_x = st.number_input("X", value=0.0, key="rect_x")
        with col2:
            rect_y = st.number_input("Y", value=0.0, key="rect_y")
        with col3:
            rect_w = st.number_input("Width", value=100.0, min_value=0.1, key="rect_w")
        with col4:
            rect_h = st.number_input("Height", value=100.0, min_value=0.1, key="rect_h")
        if st.button("Add Rectangle", key="add_rect_btn"):
            try:
                add_rect(msp, rect_x, rect_y, rect_w, rect_h, layer=current_layer)
                st.success(f"Rectangle added at ({rect_x}, {rect_y}) size {rect_w}x{rect_h}")
                safe_rerun()
            except Exception as e:
                st.error(f"Failed to add rectangle: {e}")
    
    # Polyline
    with st.expander("📐 Polyline"):
        st.caption("Enter points as comma-separated x,y pairs (one per line)")
        polyline_points = st.text_area(
            "Points (x1,y1\\nx2,y2\\n...)", 
            value="0,0\n100,0\n100,100\n0,100",
            key="polyline_points",
            height=100
        )
        closed = st.checkbox("Closed", value=True, key="polyline_closed")
        if st.button("Add Polyline", key="add_polyline_btn"):
            try:
                points = []
                for line in polyline_points.strip().split('\n'):
                    line = line.strip()
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            x = float(parts[0].strip())
                            y = float(parts[1].strip())
                            points.append((x, y))
                if len(points) >= 2:
                    add_polyline(msp, points, closed=closed, layer=current_layer)
                    st.success(f"Polyline added with {len(points)} points")
                    safe_rerun()
                else:
                    st.warning("Need at least 2 points for a polyline")
            except Exception as e:
                st.error(f"Failed to add polyline: {e}")


def render_measurement_tools(entities: List):
    """Render measurement tools UI."""
    st.header("Measurement Tools")
    st.caption("Measure distances and dimensions")
    
    selected_count = len(st.session_state["state"]["selected"])
    
    # Distance between two points
    with st.expander("📏 Distance Between Points", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            p1x = st.number_input("Point 1 X", value=0.0, key="dist_p1x")
        with col2:
            p1y = st.number_input("Point 1 Y", value=0.0, key="dist_p1y")
        with col3:
            p2x = st.number_input("Point 2 X", value=100.0, key="dist_p2x")
        with col4:
            p2y = st.number_input("Point 2 Y", value=100.0, key="dist_p2y")
        if st.button("Calculate Distance", key="calc_dist_btn"):
            try:
                dist = distance((p1x, p1y), (p2x, p2y))
                st.metric("Distance", f"{dist:.2f} mm")
            except Exception as e:
                st.error(f"Distance calculation failed: {e}")
    
    # Selected entity measurements
    if selected_count > 0:
        with st.expander("📊 Selected Entity Measurements"):
            sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
            if sel:
                try:
                    for i, e in enumerate(sel[:5]):  # Limit to first 5
                        st.write(f"**Entity {i+1}:** {e.dxftype()}")
                        bbox = bbox_of_entity(e)
                        if bbox and bbox != (0, 0, 0, 0):
                            size = bbox_size(bbox)
                            st.write(f"  - Bounding Box: {bbox}")
                            st.write(f"  - Size: {size[0]:.2f} × {size[1]:.2f} mm")
                        
                        if e.dxftype() == "LWPOLYLINE":
                            length = polyline_length(e)
                            st.write(f"  - Length: {length:.2f} mm")
                        st.divider()
                    
                    if len(sel) > 5:
                        st.caption(f"... and {len(sel) - 5} more entities")
                except Exception as e:
                    st.error(f"Measurement failed: {e}")


def render_validation_tools(entities: List):
    """Render validation tools UI."""
    st.header("Validation Tools")
    st.caption("Check for waterjet compatibility issues")
    
    selected_count = len(st.session_state["state"]["selected"])
    
    # Min radius check
    with st.expander("🔍 Minimum Radius Check", expanded=True):
        min_radius = st.number_input("Minimum Radius (mm)", value=2.0, min_value=0.1, step=0.1, key="min_radius")
        
        if selected_count > 0:
            check_selected = st.checkbox("Check Selected Entities Only", value=True, key="check_selected")
        else:
            check_selected = False
        
        if st.button("Run Validation", key="validate_btn"):
            try:
                entities_to_check = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]] if check_selected and selected_count > 0 else entities
                
                violations = []
                passed = 0
                for e in entities_to_check:
                    is_valid, info = check_min_radius(e, min_radius)
                    if not is_valid and info.get("type") == "radius":
                        violations.append({
                            "entity": e,
                            "type": e.dxftype(),
                            "radius": info.get("value", 0),
                            "min": info.get("min", min_radius)
                        })
                    elif is_valid:
                        passed += 1
                
                if violations:
                    st.warning(f"⚠️ Found {len(violations)} violation(s)")
                    for v in violations[:10]:  # Show first 10
                        st.write(f"- {v['type']}: radius {v['radius']:.2f} mm < {v['min']:.2f} mm")
                    if len(violations) > 10:
                        st.caption(f"... and {len(violations) - 10} more")
                else:
                    st.success(f"✅ All {passed} entities passed minimum radius check")
            except Exception as e:
                st.error(f"Validation failed: {e}")
    
    # Kerf preview
    with st.expander("⚙️ Kerf Preview"):
        kerf = st.number_input("Kerf Width (mm)", value=1.1, min_value=0.0, step=0.1, key="kerf_value")
        kerf_val = kerf_preview_value(kerf)
        st.info(f"Kerf preview value: {kerf_val:.2f} mm")
        st.caption("This value can be used for offset calculations in cutting operations")


def run_dxf_analysis(doc) -> Optional[Dict[str, Any]]:
    """Run DXF analysis on current document and get recommendations."""
    try:
        # Save document to temporary file
        temp_dir = Path("output") / "dxf_editor" / "temp_analysis"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"analysis_{int(time.time())}.dxf"
        
        # Save current document
        save_dxf(doc, str(temp_path))
        
        # Run analysis
        from wjp_analyser.analysis.dxf_analyzer import analyze_dxf
        
        with st.spinner("Running DXF analysis..."):
            report = analyze_dxf(str(temp_path))
        
        # Get recommendations
        from wjp_analyser.ai.recommendation_engine import analyze_and_recommend
        
        with st.spinner("Generating recommendations..."):
            recommendations = analyze_and_recommend(report)
        
        # Clean up temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        
        return {
            "report": report,
            "recommendations": recommendations,
            "operations": recommendations.get("operations", []),
            "readiness_score": recommendations.get("readiness_score", {}),
            "summary": recommendations.get("summary", {})
        }
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        if st.checkbox("Show detailed error", key="show_analysis_error"):
            st.code(traceback.format_exc())
        return None


def preview_operation(operation_dict: Dict, doc, msp) -> Dict[str, Any]:
    """Preview an operation without applying it."""
    try:
        # Convert dict to Operation object
        op = Operation(
            operation=OperationType(operation_dict["operation"]),
            parameters=operation_dict.get("parameters", {}),
            rationale=operation_dict.get("rationale", ""),
            estimated_impact=operation_dict.get("estimated_impact", {}),
            auto_apply=operation_dict.get("auto_apply", False),
            affected_count=operation_dict.get("affected_count", 0),
            severity=operation_dict.get("severity", "info")
        )
        
        executor = OperationExecutor(doc, msp)
        result = executor.execute(op, preview=True)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def apply_operation(operation_dict: Dict, doc, msp) -> Dict[str, Any]:
    """Apply an operation to the DXF document."""
    try:
        # Convert dict to Operation object
        op = Operation(
            operation=OperationType(operation_dict["operation"]),
            parameters=operation_dict.get("parameters", {}),
            rationale=operation_dict.get("rationale", ""),
            estimated_impact=operation_dict.get("estimated_impact", {}),
            auto_apply=operation_dict.get("auto_apply", False),
            affected_count=operation_dict.get("affected_count", 0),
            severity=operation_dict.get("severity", "info")
        )
        
        executor = OperationExecutor(doc, msp)
        result = executor.execute(op, preview=False)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def apply_all_auto_fixes(operations: List[Dict], doc, msp):
    """Apply all auto-apply operations in batch."""
    if not operations:
        st.info("No auto-apply operations available")
        return
    
    progress_bar = st.progress(0)
    results = []
    
    for i, op in enumerate(operations):
        progress_bar.progress((i + 1) / len(operations))
        result = apply_operation(op, doc, msp)
        results.append(result)
    
    progress_bar.empty()
    
    # Show summary
    success_count = sum(1 for r in results if r.get("success"))
    total_affected = sum(r.get("affected_count", 0) for r in results if r.get("success"))
    
    if success_count == len(operations):
        st.success(f"✅ Successfully applied {success_count} auto-fixes affecting {total_affected} entities")
    else:
        st.warning(f"⚠️ Applied {success_count}/{len(operations)} auto-fixes. {len(operations) - success_count} failed.")
        for i, result in enumerate(results):
            if not result.get("success"):
                st.error(f"Failed: {operations[i].get('operation')} - {result.get('error', 'Unknown error')}")
    
    # Clear analysis results to force re-analysis
    st.session_state["analysis_results"] = None
    safe_rerun()


def render_analysis_and_recommendations(doc, msp, entities: List):
    """Render analysis and recommendations UI section."""
    st.header("🔍 Analysis & Recommendations")
    
    # Run Analysis button
    if st.button("Run Analysis", type="primary", key="run_analysis_btn", use_container_width=True):
        results = run_dxf_analysis(doc)
        if results:
            st.session_state["analysis_results"] = results
            st.success("Analysis completed!")
            safe_rerun()
    
    # Display results if available
    if st.session_state.get("analysis_results"):
        results = st.session_state["analysis_results"]
        
        # Readiness Score
        readiness = results.get("readiness_score", {})
        score = readiness.get("score", 0)
        level = readiness.get("level", "unknown")
        
        # Color coding
        if score >= 80:
            color = "🟢"
            status_color = "green"
        elif score >= 60:
            color = "🟡"
            status_color = "orange"
        elif score >= 40:
            color = "🟠"
            status_color = "orange"
        else:
            color = "🔴"
            status_color = "red"
        
        st.markdown(f"### {color} Readiness: {score}% ({level.upper()})")
        
        # Summary
        summary = results.get("summary", {})
        st.markdown("#### Issues Found")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical = summary.get("critical_count", 0)
            if critical > 0:
                st.metric("Critical", critical, delta=None, delta_color="inverse")
            else:
                st.metric("Critical", 0)
        with col2:
            error = summary.get("error_count", 0)
            if error > 0:
                st.metric("Errors", error, delta=None, delta_color="inverse")
            else:
                st.metric("Errors", 0)
        with col3:
            warning = summary.get("warning_count", 0)
            if warning > 0:
                st.metric("Warnings", warning, delta=None, delta_color="off")
            else:
                st.metric("Warnings", 0)
        with col4:
            info = summary.get("info_count", 0)
            st.metric("Info", info)
        
        # Operations summary
        total_ops = summary.get("total_operations", 0)
        auto_apply = summary.get("auto_apply_count", 0)
        
        if total_ops > 0:
            st.markdown(f"**Total Recommendations**: {total_ops}")
            if auto_apply > 0:
                st.info(f"⚡ {auto_apply} operation(s) can be auto-applied")
        
        # Show recommendations button
        if st.button("View Recommendations", key="view_recommendations_btn", use_container_width=True):
            st.session_state["show_recommendations"] = not st.session_state.get("show_recommendations", False)
            safe_rerun()
        
        # Display recommendations if requested
        if st.session_state.get("show_recommendations"):
            operations = results.get("operations", [])
            if operations:
                st.markdown("---")
                st.markdown("### 📋 Recommendations")
                
                # Group by severity
                critical_ops = [op for op in operations if op.get("severity") == "critical"]
                error_ops = [op for op in operations if op.get("severity") == "error"]
                warning_ops = [op for op in operations if op.get("severity") == "warning"]
                info_ops = [op for op in operations if op.get("severity") == "info"]
                
                # Batch apply auto-fixes button
                auto_apply_ops = [op for op in operations if op.get("auto_apply")]
                if auto_apply_ops:
                    st.markdown("---")
                    if st.button("⚡ Apply All Auto-Fixes", type="primary", key="apply_all_auto_btn", use_container_width=True):
                        apply_all_auto_fixes(auto_apply_ops, doc, msp)
                
                # Critical
                if critical_ops:
                    st.markdown("#### 🔴 Critical")
                    for i, op in enumerate(critical_ops):
                        op_key = f"op_critical_{i}"
                        with st.expander(f"{op.get('operation', 'Unknown')} ({op.get('affected_count', 0)} affected)", expanded=True):
                            st.write(f"**Rationale**: {op.get('rationale', 'No description')}")
                            st.write(f"**Affected**: {op.get('affected_count', 0)} entities")
                            if op.get("auto_apply"):
                                st.markdown('<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.8em;">⚡ Auto-Apply</span>', unsafe_allow_html=True)
                            
                            # Preview and Apply buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👁️ Preview", key=f"preview_{op_key}"):
                                    preview_result = preview_operation(op, doc, msp)
                                    if preview_result.get("success"):
                                        st.success(f"✅ Preview: {preview_result.get('message', 'Operation ready')}")
                                        st.write(f"Will affect: {preview_result.get('affected_count', 0)} entities")
                                    else:
                                        st.error(f"❌ Preview failed: {preview_result.get('error', 'Unknown error')}")
                            with col2:
                                if st.button("✅ Apply", key=f"apply_{op_key}", type="primary"):
                                    apply_result = apply_operation(op, doc, msp)
                                    if apply_result.get("success"):
                                        st.success(f"✅ Applied: {apply_result.get('message', 'Operation completed')}")
                                        # Clear analysis results to force re-analysis
                                        st.session_state["analysis_results"] = None
                                        safe_rerun()
                                    else:
                                        st.error(f"❌ Apply failed: {apply_result.get('error', 'Unknown error')}")
                
                # Errors
                if error_ops:
                    st.markdown("#### ⚠️ Errors")
                    for i, op in enumerate(error_ops):
                        op_key = f"op_error_{i}"
                        with st.expander(f"{op.get('operation', 'Unknown')} ({op.get('affected_count', 0)} affected)"):
                            st.write(f"**Rationale**: {op.get('rationale', 'No description')}")
                            st.write(f"**Affected**: {op.get('affected_count', 0)} entities")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👁️ Preview", key=f"preview_{op_key}"):
                                    preview_result = preview_operation(op, doc, msp)
                                    if preview_result.get("success"):
                                        st.success(f"✅ Preview: {preview_result.get('message', 'Operation ready')}")
                                        st.write(f"Will affect: {preview_result.get('affected_count', 0)} entities")
                                    else:
                                        st.error(f"❌ Preview failed: {preview_result.get('error', 'Unknown error')}")
                            with col2:
                                if st.button("✅ Apply", key=f"apply_{op_key}", type="primary"):
                                    apply_result = apply_operation(op, doc, msp)
                                    if apply_result.get("success"):
                                        st.success(f"✅ Applied: {apply_result.get('message', 'Operation completed')}")
                                        st.session_state["analysis_results"] = None
                                        safe_rerun()
                                    else:
                                        st.error(f"❌ Apply failed: {apply_result.get('error', 'Unknown error')}")
                
                # Warnings
                if warning_ops:
                    st.markdown("#### 🟡 Warnings")
                    for i, op in enumerate(warning_ops):
                        op_key = f"op_warning_{i}"
                        with st.expander(f"{op.get('operation', 'Unknown')} ({op.get('affected_count', 0)} affected)"):
                            st.write(f"**Rationale**: {op.get('rationale', 'No description')}")
                            st.write(f"**Affected**: {op.get('affected_count', 0)} entities")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👁️ Preview", key=f"preview_{op_key}"):
                                    preview_result = preview_operation(op, doc, msp)
                                    if preview_result.get("success"):
                                        st.success(f"✅ Preview: {preview_result.get('message', 'Operation ready')}")
                                        st.write(f"Will affect: {preview_result.get('affected_count', 0)} entities")
                                    else:
                                        st.error(f"❌ Preview failed: {preview_result.get('error', 'Unknown error')}")
                            with col2:
                                if st.button("✅ Apply", key=f"apply_{op_key}"):
                                    apply_result = apply_operation(op, doc, msp)
                                    if apply_result.get("success"):
                                        st.success(f"✅ Applied: {apply_result.get('message', 'Operation completed')}")
                                        st.session_state["analysis_results"] = None
                                        safe_rerun()
                                    else:
                                        st.error(f"❌ Apply failed: {apply_result.get('error', 'Unknown error')}")
                
                # Info
                if info_ops:
                    st.markdown("#### ℹ️ Suggestions")
                    for i, op in enumerate(info_ops):
                        op_key = f"op_info_{i}"
                        with st.expander(f"{op.get('operation', 'Unknown')} ({op.get('affected_count', 0)} affected)"):
                            st.write(f"**Rationale**: {op.get('rationale', 'No description')}")
                            st.write(f"**Affected**: {op.get('affected_count', 0)} entities")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👁️ Preview", key=f"preview_{op_key}"):
                                    preview_result = preview_operation(op, doc, msp)
                                    if preview_result.get("success"):
                                        st.success(f"✅ Preview: {preview_result.get('message', 'Operation ready')}")
                                        st.write(f"Will affect: {preview_result.get('affected_count', 0)} entities")
                                    else:
                                        st.error(f"❌ Preview failed: {preview_result.get('error', 'Unknown error')}")
                            with col2:
                                if st.button("✅ Apply", key=f"apply_{op_key}"):
                                    apply_result = apply_operation(op, doc, msp)
                                    if apply_result.get("success"):
                                        st.success(f"✅ Applied: {apply_result.get('message', 'Operation completed')}")
                                        st.session_state["analysis_results"] = None
                                        safe_rerun()
                                    else:
                                        st.error(f"❌ Apply failed: {apply_result.get('error', 'Unknown error')}")
            else:
                st.info("✅ No recommendations - DXF file looks good!")
    else:
        st.info("👆 Click 'Run Analysis' to analyze the current DXF file")


def render_session_management():
    """Render session management UI in sidebar."""
    st.header("Session")
    
    session_path = st.session_state["state"]["session_path"]
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Session", key="save_session_btn"):
            try:
                save_session(session_path, st.session_state["state"])
                st.success("Session saved.")
            except Exception as e:
                st.error(f"Failed to save session: {e}")
    
    with col2:
        if st.button("📂 Load Session", key="load_session_btn"):
            try:
                loaded = load_session(session_path)
                st.session_state["state"] = loaded
                st.success("Session loaded.")
                safe_rerun()
            except Exception as e:
                st.error(f"Failed to load session: {e}")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main DXF Editor application."""
    st.set_page_config(page_title="DXF Editor", layout="wide", page_icon="📐")
    
    st.title("📐 DXF Editor")
    st.markdown("""
    Upload a DXF file to visualize, edit layers/groups, transform, draw, measure, validate, 
    and download the edited file.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # File upload
    uploaded = st.file_uploader("Upload DXF File", type=["dxf"], help="Select a DXF file to edit")
    
    if not uploaded:
        st.info("👆 Please upload a DXF file to get started")
        return
    
    # Load DXF document
    try:
        with st.spinner("Loading DXF file..."):
            doc = load_dxf(uploaded)
            msp = doc.modelspace()
    except Exception as e:
        st.error(f"Failed to load DXF file: {e}")
        if st.checkbox("Show detailed error", key="show_load_error"):
            import traceback
            st.code(traceback.format_exc())
        return
    
    # Get entities - support LINE, CIRCLE, LWPOLYLINE, ARC, POLYLINE, SPLINE
    entities = []
    for e in msp:
        if e.dxftype() in ["LINE", "CIRCLE", "LWPOLYLINE", "ARC", "POLYLINE", "SPLINE"]:
            entities.append(e)
    
    try:
        total_all = sum(1 for _ in msp)
    except Exception:
        total_all = len(entities)
    
    # Display entity count
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entities", total_all)
    with col2:
        st.metric("Supported Entities", len(entities))
    with col3:
        st.metric("Selected", len(st.session_state["state"]["selected"]))
    
    # Main layout: sidebar + main content
    # Always show sidebar with analysis (works on DXF file, not just entities)
    with st.sidebar:
        # Analysis section - always available
        render_analysis_and_recommendations(doc, msp, entities)
        st.divider()
        
        if len(entities) == 0:
            # No supported entities - show limited sidebar
            st.info("⚠️ No supported entities found. Analysis still available above.")
        else:
            # Has supported entities - show full editor tools
            render_layer_management(doc, get_layer_names(doc))
            st.divider()
            render_group_management(doc, entities)
            st.divider()
            render_selection_tools(doc, entities)
            st.divider()
            render_transform_tools(doc, msp, entities)
            st.divider()
            render_drawing_tools(doc, msp)
            st.divider()
            render_measurement_tools(entities)
            st.divider()
            render_validation_tools(entities)
            st.divider()
        
        # Session management - always available
        render_session_management()
    
    # Main content area
    if len(entities) == 0:
        # No supported entities - show fallback preview
        render_fallback_preview(doc, save_dxf)
    else:
        
        # Main preview area
        st.subheader("Preview")
        
        # Build warning markers
        warning_markers = build_warning_markers()
        
        # Render preview
        try:
            fig = plot_entities(
                entities,
                selected_handles=st.session_state["state"]["selected"],
                hidden_layers=st.session_state["state"]["hidden_layers"],
                color_by_layer=True,
                normalize_to_origin=True,
                warning_markers=warning_markers if warning_markers else None,
            )
            
            # Check if plot is empty
            if fig and len(fig.axes) > 0:
                ax = fig.axes[0]
                # Check if there are any lines/patches in the plot
                has_content = len(ax.lines) > 0 or len(ax.patches) > 0 or len(ax.collections) > 0
                if not has_content:
                    st.warning("⚠️ Entities found but preview is empty. This may indicate coordinate extraction issues. Trying fallback preview...")
                    # Try fallback preview
                    try:
                        render_fallback_preview(doc, save_dxf)
                    except Exception as fallback_error:
                        st.error(f"Fallback preview also failed: {fallback_error}")
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Failed to render preview: {e}")
            st.info("Attempting fallback polygonized preview...")
            try:
                render_fallback_preview(doc, save_dxf)
            except Exception as fallback_error:
                st.error(f"Fallback preview also failed: {fallback_error}")
            if st.checkbox("Show detailed error", key="show_preview_render_error"):
                import traceback
                st.code(traceback.format_exc())
    
    # Transform, Draw, Measure, Validate sections are in sidebar
    # Additional sections (Analysis, AI Analysis, etc.) would go here
    
    # Save/Export section
    st.divider()
    st.subheader("💾 Save / Export")
    
    default_filename = f"edited_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dxf"
    out_path = st.text_input("Save as", value=str(OUTPUT_DIR / default_filename), key="save_path_input")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save DXF", type="primary", key="save_dxf_btn"):
            try:
                save_dxf(doc, out_path)
                st.success(f"DXF saved to: {out_path}")
                
                # Download button
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "📥 Download Edited DXF",
                            data=f.read(),
                            file_name=os.path.basename(out_path),
                            mime="application/dxf",
                            key="download_dxf_btn"
                        )
            except Exception as e:
                st.error(f"Failed to save DXF: {e}")


if __name__ == "__main__":
    main()
