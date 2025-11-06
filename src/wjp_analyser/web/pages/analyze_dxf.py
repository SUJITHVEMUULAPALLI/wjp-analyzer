from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict

import streamlit as st

# Path shim
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from wjp_analyser.web._components import (
    ensure_workdir,
    run_analysis,
    plot_components,
    display_group_summary,
    display_metrics,
    display_quality,
    display_checklist,
    render_ai_status,
    display_ai_assist,
    run_variant_advisor,
)
# Import Phase 3 components
from wjp_analyser.web.components import (
    render_error,
    create_file_not_found_actions,
    create_dxf_error_actions,
    get_label,
    standardize_label,
)
# Use API client wrapper (falls back to services if API unavailable)
from wjp_analyser.web.api_client_wrapper import (
    analyze_dxf as api_analyze_dxf,
    estimate_cost,
    write_layered_dxf_from_report,
    export_components_csv,
    analyze_csv,
    summarize_for_quote,
)

# Import safely to avoid hard failure if heavy deps are missing
try:
    from wjp_analyser.analysis.dxf_analyzer import LAYER_NAMES  # type: ignore
except Exception:
    LAYER_NAMES = ["OUTER", "INNER", "COMPLEX", "HOLE", "DECOR"]

st.title("🔍 DXF Analyzer (KPIs · Quote · G‑code)")
st.caption("Object-level tools moved to DXF Editor. Use links below to switch.")

# AI Status
with st.sidebar.expander("AI Status", expanded=False):
    render_ai_status(compact=True)

# File Upload Section
st.markdown("## 📁 Upload DXF File")
tab_upload, tab_path = st.tabs(["Upload file", "Use local path"])

work = None
with tab_upload:
    uploaded = st.file_uploader("Upload DXF file", type=["dxf"], help="Select a DXF to analyze")
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        work = ensure_workdir(uploaded.name, file_bytes)

with tab_path:
    default_path = st.session_state.get("last_output_dxf", "") if hasattr(st, 'session_state') and st.session_state else ""
    local_path = st.text_input("DXF file path", value=str(default_path))
    if local_path and os.path.exists(local_path):
        try:
            b = open(local_path, "rb").read()
            work = ensure_workdir(os.path.basename(local_path), b)
            st.success(f"Loaded {local_path}")
        except FileNotFoundError as e:
            render_error(
                e,
                user_message=f"File not found: {local_path}",
                actions=create_file_not_found_actions(local_path),
                show_traceback=False,
            )
        except Exception as e:
            render_error(e, show_traceback=False)

if not work:
    st.info("Upload a DXF or provide a local path to begin.")
    st.stop()

# --- NEW WORKFLOW: Upload → Preview (0,0) → Target Size → Scale → Preview → Analyze → CSV → AI Analysis ---
from wjp_analyser.services.csv_analysis_service import analyze_csv
from wjp_analyser.services.editor_service import export_components_csv
from wjp_analyser.nesting.dxf_extractor import extract_polygons
import matplotlib.pyplot as plt
from datetime import datetime

def _preview_dxf_normalized(dxf_path: str, title: str = "DXF Preview"):
    """Preview DXF with X-Y axis starting from zero."""
    try:
        polys = extract_polygons(dxf_path)
        if not polys:
            st.warning("No polygons extracted for preview.")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Collect all points to find bounding box
        all_points = []
        for p in polys:
            pts = p.get("points") or []
            if len(pts) >= 3:
                all_points.extend(pts)
        
        if not all_points:
            st.warning("No valid points found in polygons.")
            return None
        
        # Calculate bounding box and normalize to (0,0)
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        
        # Normalize to origin (0,0)
        for p in polys:
            pts = p.get("points") or []
            if len(pts) >= 3:
                norm_pts = [(x - minx, y - miny) for (x, y) in pts]
                xs_norm = [x for x, y in norm_pts]
                ys_norm = [y for x, y in norm_pts]
                ax.fill(xs_norm, ys_norm, alpha=0.3, edgecolor='k', linewidth=0.5)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        
        # Show dimensions
        width = maxx - minx
        height = maxy - miny
        stats_text = f"Dimensions: {width:.2f} × {height:.2f} mm\nObjects: {len(polys)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        return fig
    except Exception as e:
        render_error(
            e,
            user_message="Failed to generate preview. The DXF file may be invalid or corrupted.",
            show_traceback=False,
        )
        return None

# Step 1: Initial Preview (normalized to 0,0)
st.markdown("## 📊 Step 1: Initial Preview (X-Y from Zero)")
initial_fig = _preview_dxf_normalized(str(work["dxf_path"]), "Initial DXF Preview (Normalized to Origin)")
if initial_fig:
    st.pyplot(initial_fig, clear_figure=True)
    plt.close(initial_fig)

# Step 2: Target Frame Size Input
st.markdown("## ⚙️ Step 2: Set Target Frame Size")
col1, col2 = st.columns(2)
with col1:
    target_width = st.number_input("Target Width (mm)", min_value=10.0, value=1000.0, step=50.0, key="target_w")
with col2:
    target_height = st.number_input("Target Height (mm)", min_value=10.0, value=1000.0, step=50.0, key="target_h")

normalize_origin = st.checkbox("Normalize origin to (0,0)", value=True, key="normalize_origin")

# Step 3: Scale DXF
if st.button("🔧 Scale DXF to Target Size", type="primary", key="scale_dxf"):
    with st.spinner("Scaling DXF to target size..."):
        try:
            # Use analysis service with proper normalization attributes
            args_overrides = {
                "normalize_mode": "fit",
                "target_frame_w_mm": float(target_width),
                "target_frame_h_mm": float(target_height),
                "frame_margin_mm": 0.0,
                "normalize_origin": bool(normalize_origin),
                "require_fit_within_frame": True
            }
            
            # Run analysis with normalization to get scaled DXF
            # Use os.path to avoid Path import issues
            base_dir = work["base_dir"]
            scaled_analysis_dir = os.path.join(base_dir, "scaled_analysis")
            scaled_report = api_analyze_dxf(
                str(work["dxf_path"]),
                out_dir=scaled_analysis_dir,
                args_overrides=args_overrides
            )
            
            # The analyzer doesn't write the layered DXF, we need to write it from components
            # Write the scaled DXF from the analysis report components
            scaled_dir = os.path.join(base_dir, "scaled")
            os.makedirs(scaled_dir, exist_ok=True)
            original_filename = os.path.basename(work['dxf_path'])
            scaled_dxf_path = os.path.join(scaled_dir, f"scaled_{original_filename}")
            
            # Write scaled DXF using service
            components = scaled_report.get("components", [])
            if components:
                try:
                    # Use layered DXF service instead of inline code
                    write_layered_dxf_from_report(
                        report=scaled_report,
                        output_path=scaled_dxf_path,
                        selected_only=False
                    )
                    st.session_state["_wjp_scaled_dxf_path"] = scaled_dxf_path
                    
                    # Get scale info from report
                    scale_info = scaled_report.get("scale", {}).get("normalize", {})
                    scale_factor = scale_info.get("factor", 1.0) if scale_info else 1.0
                    
                    st.success(f"DXF scaled to {target_width}×{target_height} mm (scale: {scale_factor:.4f}x). {len(components)} components saved!")
                except Exception as e:
                    render_error(
                        e,
                        user_message="Failed to write scaled DXF file. Check file permissions and disk space.",
                        actions=create_file_not_found_actions(str(scaled_dir)),
                        show_traceback=False,
                    )
            else:
                st.warning("Analysis completed but no components found. The DXF might be empty or could not be parsed.")
                # Show diagnostics
                diagnostics = scaled_report.get("diagnostics", {})
                if diagnostics:
                    st.json(diagnostics)
        except Exception as e:
            # Show more detailed error for debugging
            import traceback
            error_details = traceback.format_exc()
            
            # Check if it's a specific known error
            error_msg = str(e)
            if "sqlalchemy" in error_msg.lower() or "No module named" in error_msg:
                user_message = f"Failed to scale DXF due to missing dependency: {error_msg}. Please install: pip install sqlalchemy"
            elif "cannot read" in error_msg.lower() or "corrupted" in error_msg.lower():
                user_message = "Failed to scale DXF. The file may be corrupted or invalid."
            else:
                user_message = f"Failed to scale DXF: {error_msg}"
            
            render_error(
                e,
                user_message=user_message,
                actions=create_dxf_error_actions(str(work["dxf_path"])),
                show_traceback=True,  # Show traceback to help debug
            )

# Use scaled DXF if available, otherwise original
current_dxf_path = st.session_state.get("_wjp_scaled_dxf_path") or str(work["dxf_path"])

# Step 4: Preview Scaled DXF
if st.session_state.get("_wjp_scaled_dxf_path"):
    st.markdown("## 📊 Step 3: Scaled Preview (X-Y from Zero)")
    scaled_fig = _preview_dxf_normalized(current_dxf_path, "Scaled DXF Preview (Normalized to Origin)")
    if scaled_fig:
        st.pyplot(scaled_fig, clear_figure=True)
        plt.close(scaled_fig)

# Step 5: Analysis Settings
st.markdown("## ⚙️ Step 4: Analysis Settings")
material = st.text_input("Material", value="Steel", key="material_input")
thickness = st.number_input("Thickness (mm)", min_value=0.1, value=6.0, key="thickness_input")
kerf = st.number_input("Kerf (mm)", min_value=0.0, value=1.1, key="kerf_input")

# Step 6: Run Analysis
st.markdown("## 🔍 Step 5: Run Analysis")
if st.button("🚀 Analyze DXF", type="primary", key="run_analysis_workflow"):
    with st.spinner("Analyzing DXF file..."):
        try:
            report = api_analyze_dxf(current_dxf_path)
            st.session_state["_wjp_analysis_report_workflow"] = report
            
            # Export CSV
            csv_dir = os.path.join(work["base_dir"], "csv_exports")
            os.makedirs(csv_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
            csv_path = os.path.join(csv_dir, f"{timestamp}_export.csv")
            export_components_csv(report, csv_path)
            st.session_state["_wjp_analysis_csv_path"] = csv_path
            st.success("Analysis completed! CSV exported.")
        except Exception as e:
            render_error(
                e,
                user_message="Analysis failed. Please check your DXF file and try again.",
                actions=create_dxf_error_actions(current_dxf_path),
                show_traceback=False,
            )

# Step 7: Display Results and AI Analysis
if st.session_state.get("_wjp_analysis_report_workflow") and st.session_state.get("_wjp_analysis_csv_path"):
    report = st.session_state["_wjp_analysis_report_workflow"]
    csv_path = st.session_state["_wjp_analysis_csv_path"]
    
    # KPIs
    st.markdown("## 📊 Step 6: Analysis Results (KPIs)")
    kpi = summarize_for_quote(report)
    est = estimate_cost(current_dxf_path)
    eff_len_m = float(kpi.get("length_m", 0.0))
    eff_pierces = int(kpi.get("pierces", 0))
    mc = est.get("metrics", {}) if isinstance(est, dict) else {}
    if eff_len_m == 0.0 and mc:
        eff_len_m = float(mc.get("length_mm", 0.0)) / 1000.0
    if eff_pierces == 0 and mc:
        eff_pierces = int(mc.get("pierce_count", 0))
    
    cols = st.columns(3)
    cols[0].metric("Cutting length (m)", f"{eff_len_m:.2f}")
    cols[1].metric("Pierces", eff_pierces)
    cols[2].metric("Estimated cost (₹)", f"{est.get('total_cost', 0):,.0f}")
    
    # CSV Download
    st.markdown("## 📥 Step 7: Download CSV Export")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as fh:
            st.download_button(
                "Download Analysis CSV",
                data=fh.read(),
                file_name=os.path.basename(csv_path),
                mime="text/csv"
            )
    
    # AI Analysis and Recommendations
    st.markdown("## 🤖 Step 8: AI Analysis & Recommendations")
    if st.button("Analyze CSV with AI", type="primary", key="analyze_csv_ai"):
        with st.spinner("Analyzing CSV and generating recommendations..."):
            try:
                ai_analysis = analyze_csv(csv_path)
                
                if ai_analysis.get("success"):
                    stats = ai_analysis.get("statistics", {})
                    recs = ai_analysis.get("recommendations", {})
                    
                    # Display Statistics
                    st.markdown("### 📈 Detailed Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(get_label("total_objects", "Total Objects"), stats.get("total_objects", 0))
                        st.metric(get_label("selected_objects", "Selected"), stats.get("selected_count", 0))
                        st.metric(get_label("operable_objects", "Operable Objects"), stats.get("operable_objects", 0))
                    with col2:
                        st.metric("Total Area (mm²)", f"{stats.get('total_area_mm2', 0):,.2f}")
                        st.metric("Operable Area (mm²)", f"{stats.get('operable_area_mm2', 0):,.2f}")
                        st.metric("Total Perimeter (m)", f"{stats.get('total_perimeter_mm', 0) / 1000:.2f}")
                    with col3:
                        area_dist = stats.get("area_distribution", {})
                        st.metric("Zero Area Objects", area_dist.get("zero", 0))
                        st.metric("Tiny Objects (<1 mm²)", area_dist.get("tiny_lt1", 0))
                        st.metric("Large Objects (≥100 mm²)", area_dist.get("large_ge100", 0))
                    
                    # Layer Distribution
                    st.markdown("### 📁 Layer Distribution")
                    layer_dist = ai_analysis.get("layer_distribution", {})
                    if layer_dist:
                        layer_df_data = [{"Layer": k, "Count": v} for k, v in layer_dist.items()]
                        import pandas as pd
                        st.dataframe(pd.DataFrame(layer_df_data), use_container_width=True, hide_index=True)
                    
                    # Recommendations
                    st.markdown("### ⚠️ Warnings & Recommendations")
                    warnings = recs.get("warnings", [])
                    info = recs.get("info", [])
                    
                    for warning in warnings:
                        st.warning(f"**{warning.get('type', 'Warning')}**: {warning.get('message', '')}\n\n*Action*: {warning.get('action', '')}")
                    
                    for info_item in info:
                        st.info(f"**{info_item.get('type', 'Info')}**: {info_item.get('message', '')}\n\n*Action*: {info_item.get('action', '')}")
                    
                    # Viability Score
                    viability = recs.get("viability_score", "unknown")
                    viability_colors = {"good": "🟢", "fair": "🟡", "poor": "🔴"}
                    st.markdown(f"### Waterjet Viability: {viability_colors.get(viability, '⚪')} {viability.upper()}")
                    
                    # Summary
                    summary = ai_analysis.get("summary", {})
                    st.markdown("### 📋 Summary")
                    st.write(f"- **Cutting Length**: {summary.get('cutting_length_m', 0):.2f} m")
                    st.write(f"- **Estimated Pierces**: {summary.get('pierces_estimate', 0)}")
                    st.write(f"- **Net Operable Area**: {summary.get('net_operable_area_mm2', 0):,.2f} mm²")
                else:
                    st.error(f"AI Analysis failed: {ai_analysis.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"AI Analysis error: {e}")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Open in DXF Editor", url="http://127.0.0.1:8502", help="Open the app and choose 'DXF Editor' in the sidebar")
    with c2:
        st.link_button("Open G‑code Generator", url="http://127.0.0.1:8502", help="Open the app and choose 'G‑code' in the sidebar")

st.stop()

# Analysis Settings (Simplified)
st.markdown("## ⚙️ Analysis Settings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sheet Settings")
    sheet_w = st.number_input("Sheet width (mm)", min_value=100.0, value=1000.0, step=50.0, key="sheet_width")
    sheet_h = st.number_input("Sheet height (mm)", min_value=100.0, value=1000.0, step=50.0, key="sheet_height")
    frame_qty = st.number_input("Frames/Tiles quantity", min_value=1, value=1, step=1, key="frame_qty")

with col2:
    st.markdown("### Processing Options")
    scale_mode = st.selectbox(
        "Scale mode",
        options=["auto", "factor", "decade_fit"],
        index=2,
        key="scale_mode_select",
        help="Auto uses DXF $INSUNITS; Factor multiplies drawing units to mm; Decade-fit scales by 10x steps",
    )
    
    enable_fit = st.checkbox("Normalize to sheet size", value=True, help="Fit geometry into target frame size before nesting")
    
    enable_soften = st.checkbox("Soften lines", value=False, help="Apply smoothing/simplification to reduce jagged edges")
    
    enable_fillet = st.checkbox("Add fillets at sharp corners", value=False, help="Add rounded corners to reduce stress points")

# Scale options
scale_opts: Dict[str, object] | None = {"mode": scale_mode}
if scale_mode == "factor":
    scale_factor = st.number_input(
        "Units to mm factor", min_value=0.000001, value=1.0, step=0.1, help="e.g., 25.4 for inches to mm", key="scale_factor"
    )
    scale_opts["factor"] = scale_factor
elif scale_mode == "decade_fit":
    st.caption("Scale by powers of 10 until target frame is reached")
    direction = st.selectbox("Direction", options=["auto", "up", "down"], index=0, key="decade_direction_select")
    max_steps = st.number_input("Max steps", min_value=0, value=6, step=1, key="max_steps")
    allow_overshoot = st.checkbox("Allow overshoot", value=False, key="allow_overshoot")
    exact_fit = st.checkbox("Apply exact-fit after decade", value=False, key="exact_fit")
    scale_opts.update(
        {
            "decade_base": 10.0,
            "direction": direction,
            "max_steps": int(max_steps),
            "allow_overshoot": bool(allow_overshoot),
            "exact_fit": bool(exact_fit),
        }
    )

# Normalization options
normalize_opts: Dict[str, object] | None = None
if enable_fit:
    if st.checkbox("Override sheet size", value=False, key="override_sheet"):
        frame_w = st.number_input("Target width (mm)", min_value=10.0, value=float(sheet_w), step=10.0, key="frame_w")
        frame_h = st.number_input("Target height (mm)", min_value=10.0, value=float(sheet_h), step=10.0, key="frame_h")
    else:
        frame_w = float(sheet_w)
        frame_h = float(sheet_h)
    margin = st.number_input("Margin (mm)", min_value=0.0, value=0.0, step=1.0, key="margin")
    origin = st.checkbox("Translate to origin", value=True, key="origin")
    must_fit = st.checkbox("Must fit within frame", value=True, key="must_fit")

    import math

    def _to_tens(x: float) -> float:
        return float(int(round(x / 10.0)) * 10)

    fw_tens = _to_tens(frame_w)
    fh_tens = _to_tens(frame_h)
    if fw_tens != frame_w or fh_tens != frame_h:
        st.warning(f"Adjusted to nearest 10 mm: {fw_tens} x {fh_tens} mm")
    if scale_mode == "decade_fit" and must_fit:
        scale_opts["allow_overshoot"] = False
        scale_opts["exact_fit"] = False
    normalize_opts = {"mode": "fit", "frame_w": fw_tens, "frame_h": fh_tens, "margin": margin, "origin": origin, "must_fit": bool(must_fit)}

# Softening options
soften_opts: Dict[str, object] | None = None
if enable_soften:
    method = st.selectbox(
        "Method",
        options=[
            "simplify",            # Douglas-Peucker
            "simplify_topo",       # DP with topology preservation
            "rdp",                 # alias to DP
            "rdp_topo",            # alias to DP preserve topology
            "chaikin",             # corner cutting
            "visvalingam",         # VW: min-area removal
            "colinear",            # merge near-colinear vertices
            "decimate",            # keep every Nth vertex
            "resample",            # uniform step along path
            "measurement",         # measurement-guided (segment, deviation, radius, snap)
        ],
        index=0,
        key="soften_method_select",
    )

    if method in ("simplify", "simplify_topo", "rdp", "rdp_topo"):
        tol = st.number_input("Tolerance (mm)", min_value=0.001, value=0.2, step=0.05, key="tolerance")
        preserve = st.checkbox("Preserve topology (avoid self-crossing)", value=method.endswith("topo"), key="preserve_topo")
        soften_opts = {"method": method, "tolerance": float(tol), "preserve_topology": bool(preserve)}
    elif method == "chaikin":
        iters = st.number_input("Iterations", min_value=1, value=1, step=1, key="iterations")
        soften_opts = {"method": method, "iterations": int(iters)}
    elif method == "visvalingam":
        area_thr = st.number_input("Min triangle area (mm²)", min_value=0.0, value=0.5, step=0.1, key="vw_area")
        soften_opts = {"method": method, "vw_area_mm2": float(area_thr)}
    elif method == "colinear":
        ang = st.number_input("Colinear angle tolerance (deg)", min_value=0.1, value=2.0, step=0.1, key="colinear_ang")
        soften_opts = {"method": method, "colinear_angle_deg": float(ang)}
    elif method == "decimate":
        n = st.number_input("Keep every Nth vertex", min_value=2, value=2, step=1, key="decimate_n")
        soften_opts = {"method": method, "keep_every": int(n)}
    elif method == "resample":
        step = st.number_input("Resample step (mm)", min_value=0.001, value=1.0, step=0.1, key="resample_step")
        soften_opts = {"method": method, "step_mm": float(step)}
    elif method == "measurement":
        st.caption("Remove tiny segments, limit deviation, enforce min corner radius, and optional grid snapping.")
        min_seg = st.number_input("Min segment length (mm)", min_value=0.001, value=1.0, step=0.1, key="meas_min_seg")
        max_dev = st.number_input("Max deviation (mm)", min_value=0.001, value=0.2, step=0.05, key="meas_max_dev")
        min_rad = st.number_input("Min corner radius (mm)", min_value=0.0, value=1.0, step=0.1, key="meas_min_rad")
        snap = st.number_input("Snap grid (mm, 0=off)", min_value=0.0, value=0.0, step=0.5, key="meas_snap")
        soften_opts = {
            "method": method,
            "min_segment_mm": float(min_seg),
            "max_deviation_mm": float(max_dev),
            "min_corner_radius_mm": float(min_rad),
            "snap_grid_mm": float(snap),
        }

# Fillet options
fillet_opts: Dict[str, object] | None = None
if enable_fillet:
    fillet_radius = st.number_input("Fillet radius (mm)", min_value=0.01, value=1.0, step=0.1, key="fillet_radius")
    fillet_min_angle = st.number_input("Min angle to fillet (deg)", min_value=30.0, value=135.0, step=5.0, key="fillet_min_angle")
    fillet_opts = {"radius_mm": fillet_radius, "min_angle_deg": fillet_min_angle}

# Optional AI Assist before running analysis
st.markdown("## 🧠 AI Assist (Optional)")
try:
    # Display a minimal assist panel based on uploaded file (no raw geometry sent)
    dummy_report = st.session_state.get("_wjp_analysis_report_preview") or {"components": [], "layers": {}}
    display_ai_assist(dummy_report, [])
except Exception as _e:
    st.caption("AI Assist unavailable.")

# Run Analysis
st.markdown("## 🔍 Analysis Results")

if st.button("🚀 Run Analysis", type="primary", key="run_analysis"):
    with st.spinner("Analyzing DXF file..."):
        base_report = run_analysis(
            work,
            selected_groups=None,
            sheet_width=sheet_w,
            sheet_height=sheet_h,
            soften_opts=soften_opts,
            fillet_opts=fillet_opts,
            scale_opts=scale_opts,
            normalize_opts=normalize_opts,
            frame_quantity=int(frame_qty),
        )
        
        # Store report in session state
        if hasattr(st, 'session_state') and st.session_state:
            st.session_state["_wjp_analysis_report"] = base_report
        
        st.success("Analysis completed!")

# Display results if available
if hasattr(st, 'session_state') and st.session_state and "_wjp_analysis_report" in st.session_state:
    base_report = st.session_state["_wjp_analysis_report"]
    
    # Group selection
    group_names = list(base_report.get("groups", {}).keys())
    
    if group_names:
        st.markdown("### 📊 Select Groups to Include")
        c1, c2 = st.columns(2)
        if c1.button("Select All", key="select_all_groups"):
            if hasattr(st, 'session_state') and st.session_state:
                st.session_state["_wjp_groups_sel"] = group_names[:]
        if c2.button("Select None", key="select_none_groups"):
            if hasattr(st, 'session_state') and st.session_state:
                st.session_state["_wjp_groups_sel"] = []
        
        default_sel = st.session_state.get("_wjp_groups_sel", group_names) if hasattr(st, 'session_state') and st.session_state else group_names
        selected_groups: List[str] = st.multiselect(
            "Select groups to include",
            options=group_names,
            default=default_sel,
            key="groups_multiselect"
        )
    else:
        selected_groups = []
        st.write("No similarity groups detected; all objects will be used.")

    if group_names and not selected_groups:
        st.warning("No groups selected. Nesting/toolpath results will be empty.")

    # Layer reassignment
    group_layer_map: Dict[str, str] = st.session_state.get("_wjp_group_layer_map", {})
    if group_names:
        with st.expander("Layer Reassignment (by Group)", expanded=False):
            st.caption("Assign entire similarity groups to specific layers. Leave as 'Auto' to keep current classification.")
            cols = st.columns(2)
            updated_map: Dict[str, str] = {}
            options = ["Auto"] + LAYER_NAMES
            for idx, gname in enumerate(group_names):
                col = cols[idx % 2]
                current = group_layer_map.get(gname, "Auto")
                choice = col.selectbox(f"{gname}", options=options, index=options.index(current) if current in options else 0, key=f"layer_select_{idx}")
                if choice != "Auto":
                    updated_map[gname] = choice
            c_apply, c_reset = st.columns([1, 1])
            if c_apply.button("Apply Mapping", key="apply_mapping"):
                if hasattr(st, 'session_state') and st.session_state:
                    st.session_state["_wjp_group_layer_map"] = updated_map
                group_layer_map = updated_map
            if c_reset.button("Reset Mapping", key="reset_mapping"):
                if hasattr(st, 'session_state') and st.session_state:
                    st.session_state["_wjp_group_layer_map"] = {}
                group_layer_map = {}

    # Compute active report with mapping if present
    active_report = (
        base_report
        if (not group_names or set(selected_groups) == set(group_names)) and not group_layer_map
        else run_analysis(
            work,
            selected_groups=selected_groups,
            sheet_width=sheet_w,
            sheet_height=sheet_h,
            group_layer_map=group_layer_map if group_layer_map else None,
            soften_opts=soften_opts,
            fillet_opts=fillet_opts,
            scale_opts=scale_opts,
            normalize_opts=normalize_opts,
            frame_quantity=int(frame_qty),
        )
    )

    # DXF Preview
    st.markdown("## 📊 DXF Preview")
    
    # Preview options
    col1, col2 = st.columns(2)
    with col1:
        color_by = st.selectbox("Color by", ["Groups", "Layers"], index=0, key="preview_color_by_select")
    with col2:
        show_order = st.checkbox("Show order overlay (approx)", value=False, help="Nearest-neighbor centroid order", key="show_order")

    # Layer visibility controls for preview
    layers_present = list((active_report.get("layers") or {}).keys())
    if layers_present:
        selected_layers_for_preview = st.multiselect(
            "Show layers",
            options=layers_present,
            default=layers_present,
            key="layers_preview_multiselect",
            help="Affects on-screen preview only."
        )
    else:
        selected_layers_for_preview = []

    # Build filtered report copy for preview
    preview_report = dict(active_report)
    try:
        comps = active_report.get("components", [])
        if selected_layers_for_preview:
            comps = [c for c in comps if c.get("layer") in selected_layers_for_preview]
        preview_report["components"] = comps
    except Exception:
        pass

    # Display scale information
    _scale = active_report.get("scale", {}) or {}
    _smeta = _scale.get("normalize") or {}
    _dec = _smeta.get("decade") or {}
    _norm = _smeta.get("normalize") or {}
    _clamp = _smeta.get("clamp") or {}
    if _scale:
        _parts = []
        _parts.append(f"units={_scale.get('units_mode','auto')} (factor {_scale.get('units_factor')})")
        if _dec and _dec.get('mode') == 'decade_fit':
            try:
                _parts.append(f"10x steps: n={_dec.get('steps',0)}, factor={_dec.get('applied_factor',1.0):.4f}")
            except Exception:
                pass
        if _norm.get('mode') == 'fit':
            _bb = _norm.get('bbox_after') or {}
            try:
                _parts.append(f"fit: {_norm.get('factor',1.0):.3f}; final {_bb.get('width',0):.2f}x{_bb.get('height',0):.2f} mm")
            except Exception:
                pass
        if _clamp.get('applied'):
            _parts.append(f"clamped: factor {_clamp.get('factor',1.0):.4f}")
        st.caption(" | ".join(_parts))

    # Plot preview
    def _plot_preview(rep: dict, color_by: str = "Groups", show_order: bool = False):
        import matplotlib.pyplot as plt
        comps = rep.get("components", [])
        if not comps:
            st.info("No components to display.")
            return
        
        # Enhanced visualization with better colors and information
        if color_by.lower().startswith("layer"):
            names = list((rep.get("layers") or {}).keys()) or ["All"]
            key = "layer"
            title = "Layers"
        else:
            names = list(rep.get("groups", {}).keys()) or ["All"]
            key = "group"
            title = "Groups"
        
        # Use a better colormap with more distinct colors
        cmap = plt.get_cmap("Set3", max(1, len(names)))
        color_map = {n: cmap(i) for i, n in enumerate(names)}
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cents = []
        
        # Add object information to the plot
        for idx, c in enumerate(comps):
            pts = c.get("points", [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lab = c.get(key, "Unknown")
            col = color_map.get(lab, (0.3,0.3,0.3,1.0))
            a = 0.9 if c.get("selected", True) else 0.15
            
            # Enhanced visualization
            ax.fill(xs, ys, color=col, alpha=a, edgecolor='black', linewidth=0.5)
            ax.plot(xs, ys, color='black', linewidth=1.0, alpha=max(a,0.3))
            
            # Add object ID labels for small objects
            try:
                centroid_x = sum(xs)/len(xs)
                centroid_y = sum(ys)/len(ys)
                area = c.get("area", 0)
                if area < 500:  # Only label small objects to avoid clutter
                    ax.text(centroid_x, centroid_y, str(c.get("id", idx)), 
                           color='white', fontsize=6, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
                cents.append((idx, centroid_x, centroid_y))
            except Exception:
                cents.append((idx, 0.0, 0.0))
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"DXF Preview - {title}", fontsize=14, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        
        # Enhanced legend with object counts
        handles = []
        labels = []
        for name, col in color_map.items():
            count = sum(1 for c in comps if c.get(key) == name)
            handles.append(plt.Line2D([0],[0], color=col, marker='s', linestyle='', label=f"{name} ({count})"))
            labels.append(f"{name} ({count})")
        
        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize='small', 
                     title=f"{title} Legend", title_fontsize='small')
        
        # Order overlay with enhanced styling
        if show_order and len(cents) > 0:
            order = [i for i,_,_ in cents]
            if len(cents) > 1:
                import math
                rem = cents[:]
                start = min(range(len(rem)), key=lambda i: rem[i][1])
                seq = [rem.pop(start)]
                while rem:
                    last = seq[-1]
                    j = min(range(len(rem)), key=lambda k: math.hypot(rem[k][1]-last[1], rem[k][2]-last[2]))
                    seq.append(rem.pop(j))
                order = [i for (i,_,_) in seq]
            
            for n, comp_index in enumerate(order, start=1):
                try:
                    _, cx, cy = cents[comp_index]
                    ax.text(cx, cy, str(n), color='red', fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="circle", facecolor='yellow', alpha=0.8))
                except Exception:
                    pass
        
        # Add grid for better reference
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        total_objects = len(comps)
        total_groups = len(names)
        stats_text = f"Total Objects: {total_objects}\nTotal Groups: {total_groups}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        st.pyplot(fig, clear_figure=True, width="stretch")

    _plot_preview(preview_report, color_by=color_by, show_order=show_order)

    # Export filtered view to DXF
    with st.expander("Export (filtered preview)", expanded=False):
        def _write_dxf_from_components(components: List[dict], out_path: str) -> bool:
            """Write DXF from components using service."""
            try:
                from wjp_analyser.services.layered_dxf_service import write_layered_dxf_from_components
                write_layered_dxf_from_components(components, out_path)
                return True
            except Exception as exc:
                st.error(f"Failed to export DXF: {exc}")
                import traceback
                st.code(traceback.format_exc())
                return False

        if st.button("Export filtered preview to DXF", key="export_filtered"):
            filtered_path = os.path.join(str(work["base_dir"]), f"filtered_{os.path.basename(str(work['dxf_path']))}")
            ok = _write_dxf_from_components(preview_report.get("components", []), filtered_path)
            if ok and os.path.exists(filtered_path):
                with open(filtered_path, "rb") as fh:
                    st.download_button("Download filtered DXF", data=fh.read(), file_name=os.path.basename(filtered_path))

    # Results sections
    st.markdown("## 📈 Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Layer Summary")
        layer_counts = active_report.get("layers", {}) or {}
        if layer_counts:
            rows = [{"Layer": name, "Count": int(cnt)} for name, cnt in layer_counts.items()]
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.info("No layer data available")

    with col2:
        st.markdown("### Similarity Groups")
        display_group_summary(base_report, selected_groups or group_names)

    # Metrics and Quality
    st.markdown("### Selected Metrics")
    display_metrics(active_report)

    st.markdown("### Quality Checks")
    display_quality(active_report)

    st.markdown("### Checklist")
    display_checklist(active_report)

    # Variant Advisor
    st.markdown("## 🤖 Variant Advisor")
    with st.expander("Find best variant by workability score", expanded=False):
        use_fillet = st.checkbox("Include light fillet variants", value=True, key="advisor_use_fillet")
        max_variants_note = st.caption("Runs about 8–12 fast variants. Keep file sizes reasonable for responsiveness.")
        if st.button("Run Advisor", key="run_variant_advisor"):
            with st.spinner("Evaluating variants..."):
                results = run_variant_advisor(
                    work,
                    sheet_width=sheet_w,
                    sheet_height=sheet_h,
                    scale_opts=scale_opts,
                    normalize_opts=normalize_opts,
                    frame_quantity=int(frame_qty),
                    include_fillet=bool(use_fillet),
                )
                if results:
                    # Show top 5
                    topn = results[:5]
                    for i, r in enumerate(topn, start=1):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.subheader(f"#{i}  {r['id']}  —  Score {r['score']}")
                            st.write(f"Length: {r['metrics']['length_mm']:.1f} mm | Pierces: {r['metrics']['pierces']}")
                            if r.get("reasons"):
                                st.caption("; ".join(map(str, r["reasons"]))[:500])
                            st.code(str({k:v for k,v in (r.get('soften') or {}).items()}), language="json")
                            if r.get("fillet"):
                                st.code(str(r.get('fillet')), language="json")
                        with c2:
                            arts = r.get("artifacts") or {}
                            if arts.get("nc") and os.path.exists(arts["nc"]):
                                with open(arts["nc"], "rb") as fh:
                                    st.download_button(
                                        "Download NC",
                                        data=fh.read(),
                                        file_name=os.path.basename(arts["nc"]),
                                        key=f"dl_nc_{i}_{r['id']}",
                                    )
                            if arts.get("report") and os.path.exists(arts["report"]):
                                with open(arts["report"], "rb") as fh:
                                    st.download_button(
                                        "Report JSON",
                                        data=fh.read(),
                                        file_name=os.path.basename(arts["report"]),
                                        key=f"dl_report_{i}_{r['id']}",
                                    )
                            if arts.get("lengths_csv") and os.path.exists(arts["lengths_csv"]):
                                with open(arts["lengths_csv"], "rb") as fh:
                                    st.download_button(
                                        "Lengths CSV",
                                        data=fh.read(),
                                        file_name=os.path.basename(arts["lengths_csv"]),
                                        key=f"dl_lengths_{i}_{r['id']}",
                                    )
                            if arts.get("layered_dxf") and os.path.exists(arts["layered_dxf"]):
                                with open(arts["layered_dxf"], "rb") as fh:
                                    st.download_button(
                                        "Layered DXF",
                                        data=fh.read(),
                                        file_name=os.path.basename(arts["layered_dxf"]),
                                        key=f"dl_dxf_{i}_{r['id']}",
                                    )

    # Outputs
    st.markdown("## 📤 Download Results")
    artifacts = active_report.get("artifacts", {})
    cols = st.columns(4)
    artifact_items = [
        ("Layered DXF", artifacts.get("layered_dxf")),
        ("Report JSON", artifacts.get("report_json")),
        ("Lengths CSV", artifacts.get("lengths_csv")),
        ("Program NC", artifacts.get("gcode")),
    ]
    for col, (label, path) in zip(cols, artifact_items):
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                col.download_button(label=f"Download {label}", data=fh.read(), file_name=os.path.basename(path))
        else:
            col.write(f"{label}: not available")

else:
    st.info("Click 'Run Analysis' to process your DXF file.")