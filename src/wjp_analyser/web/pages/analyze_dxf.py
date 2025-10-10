from __future__ import annotations

import os
import sys
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
)

# Import safely to avoid hard failure if heavy deps are missing
try:
    from wjp_analyser.analysis.dxf_analyzer import LAYER_NAMES  # type: ignore
except Exception:
    LAYER_NAMES = ["OUTER", "INNER", "COMPLEX", "HOLE", "DECOR"]

st.title("🔍 DXF Analysis")
st.markdown("**Simple DXF analysis and processing**")

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
        except Exception as e:
            st.error(f"Could not read file: {e}")

if not work:
    st.info("Upload a DXF or provide a local path to begin.")
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
    method = st.selectbox("Method", options=["simplify", "chaikin"], index=0, key="soften_method_select")
    if method == "simplify":
        tol = st.number_input("Tolerance (mm)", min_value=0.01, value=0.2, step=0.05, key="tolerance")
        soften_opts = {"method": method, "tolerance": tol}
    else:
        iters = st.number_input("Iterations", min_value=1, value=1, step=1, key="iterations")
        soften_opts = {"method": method, "iterations": int(iters)}

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
        
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    _plot_preview(preview_report, color_by=color_by, show_order=show_order)

    # Export filtered view to DXF
    with st.expander("Export (filtered preview)", expanded=False):
        def _write_dxf_from_components(components: List[dict], out_path: str) -> bool:
            try:
                import ezdxf  # type: ignore
            except Exception:
                st.warning("ezdxf not available; install requirements to export DXF.")
                return False
            try:
                doc = ezdxf.new("R2010")
                msp = doc.modelspace()
                for c in components or []:
                    pts = c.get("points") or []
                    if len(pts) >= 2:
                        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], format="xy", close=True)
                doc.saveas(out_path)
                return True
            except Exception as exc:
                st.error(f"Failed to export DXF: {exc}")
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
            st.dataframe(rows, hide_index=True, use_container_width=True)
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