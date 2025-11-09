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
    SESSION_DXF_KEY, SESSION_PATH_KEY, SESSION_EDIT_LOG, SESSION_LAYER_VIS, SESSION_SELECTED
)
from wjp_analyser.web.modules import dxf_editor_core as core
from wjp_analyser.web.modules.dxf_renderer import render_svg


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
    st.header("Actions")
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
    st.caption("Edit Log")
    if st.button("Clear Log"):
        st.session_state[SESSION_EDIT_LOG] = []
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

with right:
    st.subheader("Preview")
    try:
        svg_text = render_svg(doc, layer_visibility=st.session_state[SESSION_LAYER_VIS])
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

