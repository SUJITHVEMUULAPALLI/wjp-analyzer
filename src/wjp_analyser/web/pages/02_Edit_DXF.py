# pages/02_Edit_DXF.py
import os
import sys
from pathlib import Path
import streamlit as st
from streamlit.components.v1 import html

# Ensure 'src' is on sys.path so modules are importable
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = str(_THIS_FILE.parents[3])  # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Import modules
from wjp_analyser.web.modules.dxf_utils import (
    SESSION_DXF_KEY, SESSION_PATH_KEY, SESSION_EDIT_LOG, SESSION_LAYER_VIS, SESSION_SELECTED, SESSION_HISTORY
)
from wjp_analyser.web.modules import dxf_editor_core as core
from wjp_analyser.web.modules.dxf_renderer import render_svg
from wjp_analyser.web.modules import dxf_export
from wjp_analyser.web.modules.dxf_viewport import add_zoom_pan_controls
from wjp_analyser.web.modules import dxf_reanalyze


def safe_rerun():
    """Safely trigger a Streamlit rerun, handling internal exceptions."""
    try:
        st.rerun()
    except Exception:
        # Streamlit's internal rerun exceptions should be re-raised
        # This prevents them from being displayed as user-facing errors
        import streamlit.runtime.scriptrunner.script_runner as script_runner
        if isinstance(script_runner.RerunException, type):
            raise
        # For other exceptions, just rerun normally
        st.rerun()

st.set_page_config(page_title="WJP – DXF Editor", layout="wide")

st.title("DXF Editor")

# ---- File Load Row ----
with st.container():
    col_a, col_b, col_c = st.columns([2,2,1])
    with col_a:
        st.caption("Source")
        st.write("Reads the DXF you analyzed earlier, or upload a new one.")
    with col_b:
        upload = st.file_uploader("Upload DXF (optional)", type=["dxf"], label_visibility="collapsed")
    with col_c:
        st.write("")  # spacing
        if st.button("Reload from Session", use_container_width=True):
            # Force re-load from Analyzer path
            st.session_state.pop(SESSION_DXF_KEY, None)
            st.session_state.pop(SESSION_PATH_KEY, None)

# Initialize & load
doc, working_path = core.load_from_analyzer_or_upload(st, upload)

if not doc:
    st.info("No DXF found in session and no file uploaded yet. From Analyzer, set `st.session_state['analyzed_dxf_path'] = <path>` or upload a DXF here.")
    st.stop()

# Sidebar: layers and actions
with st.sidebar:
    st.header("Layers")
    vis = st.session_state[SESSION_LAYER_VIS]
    if st.button("Show All"):
        for k in vis.keys():
            vis[k] = True
    if st.button("Hide All"):
        for k in vis.keys():
            vis[k] = False

    # Layer checklist
    for lname in sorted(vis.keys()):
        vis[lname] = st.checkbox(lname, value=vis[lname])

    st.divider()
    st.header("View Options")
    
    # Grid settings
    show_grid = st.checkbox("Show Grid", value=st.session_state.get("show_grid", False), key="show_grid")
    if show_grid:
        grid_size = st.number_input("Grid Size (mm)", value=10.0, min_value=1.0, max_value=100.0, step=1.0, key="grid_size")
    else:
        grid_size = 10.0
    
    # Zoom/Pan controls
    enable_zoom = st.checkbox("Enable Zoom/Pan", value=st.session_state.get("enable_zoom", True), key="enable_zoom")
    
    st.divider()
    st.header("Actions")
    
    # Undo/Redo buttons
    history = st.session_state.get(SESSION_HISTORY)
    if history:
        col_undo, col_redo = st.columns(2)
        with col_undo:
            undo_disabled = not history.can_undo()
            if st.button("↶ Undo", disabled=undo_disabled, use_container_width=True):
                if core.undo_last_action(st):
                    st.success("Undone")
                    safe_rerun()
        with col_redo:
            redo_disabled = not history.can_redo()
            if st.button("↷ Redo", disabled=redo_disabled, use_container_width=True):
                if core.redo_last_action(st):
                    st.success("Redone")
                    safe_rerun()
        
        # History info
        info = history.get_history_info()
        st.caption(f"Undo: {info['undo_count']} | Redo: {info['redo_count']}")
    
    st.divider()
    
    # Export options
    st.header("Export")
    export_format = st.radio("Format", ["DXF", "SVG", "JSON", "All"], horizontal=True, key="export_format")
    export_base = st.text_input("Export filename (no extension)", value=os.path.basename(working_path).replace(".dxf", "_export"))
    export_dir = st.text_input("Export directory", value=os.path.dirname(working_path))
    
    if st.button("📤 Export", use_container_width=True):
        try:
            base_path = os.path.join(export_dir, export_base)
            edit_log = st.session_state.get(SESSION_EDIT_LOG, [])
            
            if export_format == "DXF":
                out_path = f"{base_path}.dxf"
                result = dxf_export.export_dxf(doc, out_path, metadata={"export_date": str(os.path.getmtime(working_path))})
                st.success(f"Exported: {result}")
            elif export_format == "SVG":
                out_path = f"{base_path}.svg"
                result = dxf_export.export_svg(doc, out_path, layer_visibility=st.session_state[SESSION_LAYER_VIS],
                                               include_grid=show_grid, grid_size=grid_size)
                st.success(f"Exported: {result}")
            elif export_format == "JSON":
                out_path = f"{base_path}_metadata.json"
                result = dxf_export.export_json(doc, out_path, edit_log=edit_log)
                st.success(f"Exported: {result}")
            elif export_format == "All":
                results = dxf_export.export_all_formats(doc, base_path, 
                                                        layer_visibility=st.session_state[SESSION_LAYER_VIS],
                                                        edit_log=edit_log, include_grid=show_grid)
                st.success(f"Exported all formats:")
                for fmt, path in results.items():
                    st.write(f"  - {fmt.upper()}: {path}")
        except Exception as e:
            st.error(f"Export failed: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    st.divider()
    
    # Save
    save_name = st.text_input("Save as (filename.dxf)", value=os.path.basename(working_path).replace(".dxf", "_edited.dxf"))
    save_dir = st.text_input("Save directory", value=os.path.dirname(working_path))
    if st.button("💾 Save DXF"):
        out_path = os.path.join(save_dir, save_name)
        try:
            outp = core.save_as(st, out_path)
            st.success(f"Saved: {outp}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.divider()
    
    # Auto re-analyze warning
    edit_count = dxf_reanalyze.get_edit_count(st)
    if edit_count > 0:
        st.caption(f"Edit Count: {edit_count}")
        dxf_reanalyze.should_reanalyze(st, threshold=10, show_warning=True)
        if dxf_reanalyze.create_reanalyze_button(st, threshold=10):
            safe_rerun()
    
    st.divider()
    st.caption("Edit Log")
    if st.button("Clear Log"):
        st.session_state[SESSION_EDIT_LOG] = []
        if history:
            history.clear()
        dxf_reanalyze.reset_edit_count(st)
    st.json(st.session_state.get(SESSION_EDIT_LOG, []))

# Main layout: left table, right preview
left, right = st.columns([1.1, 1.9], gap="large")

with left:
    st.subheader("Objects")
    data = core.get_entity_table(st)
    if not data:
        st.warning("No entities in modelspace.")
    else:
        # Selection table
        selected_handles = st.session_state.get(SESSION_SELECTED) or set()

        # Compact listing grouped by layer
        layers = sorted(set(d["layer"] for d in data))
        for lname in layers:
            subset = [d for d in data if d["layer"] == lname]
            with st.expander(f"Layer: {lname} ({len(subset)})", expanded=False):
                for row in subset:
                    handle = row["handle"]
                    checked = handle in selected_handles
                    chk = st.checkbox(
                        f"{row['type']}  |  handle:{handle}  |  color:{row['color']}",
                        value=checked, key=f"sel_{handle}"
                    )
                    if chk:
                        selected_handles.add(handle)
                    else:
                        selected_handles.discard(handle)
        st.session_state[SESSION_SELECTED] = selected_handles

        # Buttons: delete selected, select none
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Delete Selected"):
                to_delete = list(st.session_state[SESSION_SELECTED])
                count = core.apply_delete(st, to_delete)
                if count:
                    st.success(f"Deleted {count} entities.")
                    st.session_state[SESSION_SELECTED] = set()
                    safe_rerun()
                else:
                    st.warning("Nothing deleted.")
        with c2:
            if st.button("Clear Selection"):
                st.session_state[SESSION_SELECTED] = set()
                st.info("Selection cleared.")
        
        # Transform tools
        if selected_handles:
            st.divider()
            st.subheader("Transform Selected")
            selected_list = list(selected_handles)
            
            # Move
            with st.expander("📍 Move", expanded=False):
                col_dx, col_dy = st.columns(2)
                with col_dx:
                    dx = st.number_input("ΔX (mm)", value=0.0, step=1.0, key="move_dx")
                with col_dy:
                    dy = st.number_input("ΔY (mm)", value=0.0, step=1.0, key="move_dy")
                if st.button("Apply Move", key="move_btn"):
                    count = core.apply_transform(st, selected_list, "move", dx=dx, dy=dy)
                    if count:
                        st.success(f"Moved {count} entities")
                        safe_rerun()
            
            # Rotate
            with st.expander("🔄 Rotate", expanded=False):
                angle = st.number_input("Angle (degrees)", value=0.0, step=15.0, key="rotate_angle")
                col_cx, col_cy = st.columns(2)
                with col_cx:
                    center_x = st.number_input("Center X", value=0.0, key="rotate_cx")
                with col_cy:
                    center_y = st.number_input("Center Y", value=0.0, key="rotate_cy")
                if st.button("Apply Rotate", key="rotate_btn"):
                    count = core.apply_transform(st, selected_list, "rotate", angle=angle, center=(center_x, center_y))
                    if count:
                        st.success(f"Rotated {count} entities")
                        safe_rerun()
            
            # Scale
            with st.expander("🔍 Scale", expanded=False):
                factor = st.number_input("Scale Factor", value=1.0, min_value=0.01, max_value=100.0, step=0.1, key="scale_factor")
                col_bx, col_by = st.columns(2)
                with col_bx:
                    base_x = st.number_input("Base Point X", value=0.0, key="scale_bx")
                with col_by:
                    base_y = st.number_input("Base Point Y", value=0.0, key="scale_by")
                if st.button("Apply Scale", key="scale_btn"):
                    count = core.apply_transform(st, selected_list, "scale", factor=factor, base_point=(base_x, base_y))
                    if count:
                        st.success(f"Scaled {count} entities")
                        safe_rerun()
            
            # Mirror
            with st.expander("🪞 Mirror", expanded=False):
                axis = st.radio("Axis", ["X", "Y"], horizontal=True, key="mirror_axis")
                if st.button("Apply Mirror", key="mirror_btn"):
                    count = core.apply_transform(st, selected_list, "mirror", axis=axis)
                    if count:
                        st.success(f"Mirrored {count} entities")
                        safe_rerun()

with right:
    st.subheader("Preview")
    try:
        # Get grid settings from session state
        show_grid = st.session_state.get("show_grid", False)
        grid_size = st.session_state.get("grid_size", 10.0)
        enable_zoom = st.session_state.get("enable_zoom", True)
        
        svg_text = render_svg(doc, layer_visibility=st.session_state[SESSION_LAYER_VIS],
                             include_grid=show_grid, grid_size=grid_size)
        
        # Add zoom/pan controls if enabled
        if enable_zoom:
            svg_with_controls = add_zoom_pan_controls(svg_text)
            html(svg_with_controls, height=600)
        else:
            # Render SVG in-page (native zoom via browser)
            html(
                f"""
                <div style="width:100%;height:75vh;border:1px solid #ddd;overflow:auto;background:#fff">
                    {svg_text}
                </div>
                """,
                height=600,
            )
    except Exception as e:
        st.error(f"Render error: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

# Footer details
st.caption(f"Working file: {working_path}")

