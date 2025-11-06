"""Streamlit-based DXF analyzer with size-grouping, must-fit scaling, and interactive selection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import hashlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import streamlit as st

# Ensure the project `src` directory is on sys.path when launched via streamlit
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # -> src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf, LAYER_NAMES
from wjp_analyser.web._components import render_ai_status


def _ensure_workdir(upload_name: str, file_bytes: bytes) -> dict:
    """Store uploaded DXF in a temporary working directory and return metadata."""
    work = st.session_state.get("_wjp_work")
    if work and work.get("name") == upload_name and work.get("bytes") == file_bytes:
        return work

    base_dir = Path(tempfile.mkdtemp(prefix="wjp_analyzer_"))
    dxf_path = base_dir / upload_name
    dxf_path.write_bytes(file_bytes)

    work = {
        "name": upload_name,
        "bytes": file_bytes,
        "base_dir": base_dir,
        "dxf_path": dxf_path,
        "reports": {},
    }
    st.session_state["_wjp_work"] = work
    return work


def _run_analysis(
    work: dict,
    selected_groups: List[str] | None = None,
    sheet_width: float | None = None,
    sheet_height: float | None = None,
    group_layer_map: Dict[str, str] | None = None,
    soften_opts: Dict[str, object] | None = None,
    fillet_opts: Dict[str, object] | None = None,
    scale_opts: Dict[str, object] | None = None,
    normalize_opts: Dict[str, object] | None = None,
    frame_quantity: int | None = None,
) -> dict:
    key = tuple(sorted(selected_groups)) if selected_groups else ("__all__",)
    map_sig = tuple(sorted((k, v) for k, v in (group_layer_map or {}).items())) if group_layer_map else ("__nomap__",)
    soft_sig = tuple(sorted((str(k), str(v)) for k, v in (soften_opts or {}).items())) if soften_opts else ("__nosoft__",)
    fillet_sig = tuple(sorted((str(k), str(v)) for k, v in (fillet_opts or {}).items())) if fillet_opts else ("__nofillet__",)
    scale_sig = tuple(sorted((str(k), str(v)) for k, v in (scale_opts or {}).items())) if scale_opts else ("__noscale__",)
    norm_sig = tuple(sorted((str(k), str(v)) for k, v in (normalize_opts or {}).items())) if normalize_opts else ("__nonorm__",)
    cache_key = (key, map_sig, soft_sig, fillet_sig, scale_sig, norm_sig, (frame_quantity or 1))
    if cache_key in work["reports"]:
        return work["reports"][cache_key]

    # Build a short, filesystem-friendly label to avoid Windows MAX_PATH issues
    if key == ("__all__",):
        klabel = "all"
    else:
        joined = ",".join(key)
        klabel = f"sel_{len(key)}_{hashlib.sha1(joined.encode('utf-8')).hexdigest()[:10]}"
    mlabel = "nomap" if not group_layer_map else hashlib.sha1(
        ";".join(f"{k}:{v}" for k, v in sorted(group_layer_map.items())).encode("utf-8")
    ).hexdigest()[:8]
    slabel = "nosoft" if not soften_opts else hashlib.sha1(
        ";".join(f"{k}:{v}" for k, v in sorted(soften_opts.items())).encode("utf-8")
    ).hexdigest()[:8]
    flabel = "nofillet" if not fillet_opts else hashlib.sha1(
        ";".join(f"{k}:{v}" for k, v in sorted(fillet_opts.items())).encode("utf-8")
    ).hexdigest()[:8]
    scalelabel = "noscale" if not scale_opts else hashlib.sha1(
        ";".join(f"{k}:{v}" for k, v in sorted(scale_opts.items())).encode("utf-8")
    ).hexdigest()[:8]
    normlabel = "nonorm" if not normalize_opts else hashlib.sha1(
        ";".join(f"{k}:{v}" for k, v in sorted(normalize_opts.items())).encode("utf-8")
    ).hexdigest()[:8]
    label = f"{klabel}_map_{mlabel}_soft_{slabel}_fillet_{flabel}_scale_{scalelabel}_norm_{normlabel}_qty_{frame_quantity or 1}"
    out_dir = work["base_dir"] / f"analysis_{label}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Fallback to a generic folder if the path is still problematic
        out_dir = work["base_dir"] / "analysis_sel"
        out_dir.mkdir(parents=True, exist_ok=True)

    args = AnalyzeArgs(out=str(out_dir))
    if sheet_width:
        args.sheet_width = float(sheet_width)
    if sheet_height:
        args.sheet_height = float(sheet_height)
    if isinstance(frame_quantity, int) and frame_quantity > 1:
        args.frame_quantity = int(frame_quantity)
    if soften_opts:
        method = str(soften_opts.get("method", "none"))
        args.soften_method = method
        if method == "simplify":
            args.soften_tolerance = float(soften_opts.get("tolerance", 0.2))
        elif method == "chaikin":
            args.soften_iterations = int(soften_opts.get("iterations", 1))
    if fillet_opts:
        args.fillet_radius_mm = float(fillet_opts.get("radius_mm", 0.0))
        args.fillet_min_angle_deg = float(fillet_opts.get("min_angle_deg", 135.0))
    if scale_opts:
        args.scale_mode = str(scale_opts.get("mode", "auto"))
        if args.scale_mode == "factor":
            args.scale_factor = float(scale_opts.get("factor", 1.0))
        if args.scale_mode == "decade_fit":
            # Optional decade-fit parameters
            if "decade_base" in scale_opts:
                args.scale_decade_base = float(scale_opts.get("decade_base", 10.0))
            if "direction" in scale_opts:
                args.scale_decade_direction = str(scale_opts.get("direction", "auto"))
            if "max_steps" in scale_opts:
                args.scale_decade_max_steps = int(scale_opts.get("max_steps", 6))
            if "allow_overshoot" in scale_opts:
                args.scale_decade_allow_overshoot = bool(scale_opts.get("allow_overshoot", True))
            if "exact_fit" in scale_opts:
                args.scale_decade_exact_fit = bool(scale_opts.get("exact_fit", False))
    if normalize_opts:
        args.normalize_mode = str(normalize_opts.get("mode", "none"))
        if args.normalize_mode == "fit":
            args.target_frame_w_mm = float(normalize_opts.get("frame_w", 1000.0))
            args.target_frame_h_mm = float(normalize_opts.get("frame_h", 1000.0))
            args.frame_margin_mm = float(normalize_opts.get("margin", 0.0))
            args.normalize_origin = bool(normalize_opts.get("origin", True))
        # Safety clamp toggle
        if isinstance(normalize_opts, dict) and "must_fit" in normalize_opts:
            args.require_fit_within_frame = bool(normalize_opts.get("must_fit"))

    report = analyze_dxf(
        str(work["dxf_path"]),
        args,
        selected_groups=selected_groups,
        group_layer_overrides=group_layer_map,
    )
    work["reports"][cache_key] = report
    return report


def _plot_components(report: dict, height: int = 600):
    components = report.get("components", [])
    if not components:
        st.info("No components to display.")
        return

    group_names = list(report.get("groups", {}).keys()) or ["All"]
    cmap = plt.get_cmap("tab20", max(1, len(group_names)))
    color_map: Dict[str, tuple] = {name: cmap(idx) for idx, name in enumerate(group_names)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for comp in components:
        pts = comp["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        group = comp.get("group", "Ungrouped")
        color = color_map.get(group, (0.3, 0.3, 0.3, 1.0))
        alpha = 0.9 if comp.get("selected", True) else 0.15
        ax.fill(xs, ys, color=color, alpha=alpha)
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=max(alpha, 0.3))

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("DXF Preview - Grouped Components")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    handles = [plt.Line2D([0], [0], color=color, marker="s", linestyle="", label=name) for name, color in color_map.items()]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize="small")

    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _display_group_summary(report: dict, selected_groups: List[str]):
    groups = report.get("groups", {})
    if not groups:
        st.write("No similarity groups detected.")
        return

    rows = []
    for name, meta in groups.items():
        rows.append(
            {
                "Group": name,
                "Count": meta.get("count", 0),
                "Vertices": meta.get("vcount"),
                "Avg Area (mm²)": round(meta.get("avg_area", 0.0), 2),
                "Avg Circularity": round(meta.get("avg_circ", 0.0), 3),
                "Complexity": meta.get("complexity", ""),
                "Selected": name in selected_groups,
            }
        )

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Group": st.column_config.TextColumn(width="small"),
            "Count": st.column_config.NumberColumn(format="%d", width="small"),
            "Vertices": st.column_config.NumberColumn(format="%d", width="small"),
            "Avg Area (mm²)": st.column_config.NumberColumn(format="%.2f"),
            "Avg Circularity": st.column_config.NumberColumn(format="%.3f"),
            "Complexity": st.column_config.TextColumn(),
            "Selected": st.column_config.CheckboxColumn(),
        },
    )


def _display_metrics(report: dict):
    metrics = report.get("metrics", {})
    cols = st.columns(3)
    cols[0].metric("Cutting length (mm)", metrics.get("length_internal_mm", 0))
    cols[1].metric("Pierce count", metrics.get("pierces", 0))
    cols[2].metric("Estimated cost", metrics.get("estimated_cutting_cost_inr", 0))


def _display_quality(report: dict):
    q = report.get("quality") or {}
    if not q:
        st.write("No quality data available.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Entities", q.get("Total Entities", 0))
    c2.metric("Polylines", q.get("Polylines", 0))
    c3.metric("Open Polylines", q.get("Open Polylines", 0))
    c4, c5, c6 = st.columns(3)
    c4.metric("Tiny Segments (< tol)", q.get("Tiny Segments", 0))
    c5.metric("Duplicate Candidates", q.get("Duplicate Candidates", 0))
    shaky = q.get("Shaky Polylines", [])
    c6.metric("Shaky Polylines", len(shaky))
    with st.expander("Details (by type)", expanded=False):
        st.write({k: v for k, v in q.items() if k in ("Lines", "Arcs", "Circles", "Polylines")})
    if shaky:
        with st.expander("Shaky polylines list", expanded=False):
            st.dataframe(shaky, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Waterjet DXF Analyzer", layout="wide")
    st.title("Waterjet DXF Analyzer")
    with st.sidebar.expander("AI Status", expanded=False):
        try:
            render_ai_status(compact=True)
        except Exception:
            st.caption("Status unavailable")
    st.caption("Group similar objects, preview, and select which to nest/toolpath")

    uploaded = st.file_uploader("Upload DXF file", type=["dxf"])
    if not uploaded:
        st.info("Upload a DXF file to begin.")
        return

    file_bytes = uploaded.getvalue()
    work = _ensure_workdir(uploaded.name, file_bytes)

    st.sidebar.header("Frame Settings")
    sheet_w = st.sidebar.number_input("Sheet width (mm)", min_value=100.0, value=1000.0, step=50.0)
    sheet_h = st.sidebar.number_input("Sheet height (mm)", min_value=100.0, value=1000.0, step=50.0)
    frame_qty = st.sidebar.number_input("Frames/Tiles quantity", min_value=1, value=1, step=1)

    # Line Softening
    st.sidebar.header("Line Softening")
    enable_soften = st.sidebar.checkbox("Soften lines", value=False, help="Apply smoothing/simplification to reduce jagged edges")
    soften_opts = None
    if enable_soften:
        method = st.sidebar.selectbox("Method", options=["simplify", "chaikin"], index=0)
        if method == "simplify":
            tol = st.sidebar.number_input("Tolerance (mm)", min_value=0.01, value=0.2, step=0.05)
            soften_opts = {"method": method, "tolerance": tol}
        else:
            iters = st.sidebar.number_input("Iterations", min_value=1, value=1, step=1)
            soften_opts = {"method": method, "iterations": int(iters)}

    # Corner fillets
    st.sidebar.header("Corner Fillet (arcs)")
    enable_fillet = st.sidebar.checkbox("Add fillets at sharp corners", value=False)
    fillet_opts = None
    if enable_fillet:
        fillet_radius = st.sidebar.number_input("Fillet radius (mm)", min_value=0.01, value=1.0, step=0.1)
        fillet_min_angle = st.sidebar.number_input("Min angle to fillet (deg)", min_value=30.0, value=135.0, step=5.0)
        fillet_opts = {"radius_mm": fillet_radius, "min_angle_deg": fillet_min_angle}

    # Units / Scale controls
    st.sidebar.header("Units / Scale")
    scale_mode = st.sidebar.selectbox(
        "Scale mode",
        options=["auto", "factor", "decade_fit"],
        index=2,
        help="Auto uses DXF $INSUNITS; Factor multiplies drawing units to mm; Decade-fit scales by 10× steps",
    )
    scale_opts: Dict[str, object] = {"mode": scale_mode}
    if scale_mode == "factor":
        scale_opts[" factor\] = st.sidebar.number_input(\Units to mm factor\, min_value=0.000001, value=1.0, step=0.1)
    elif scale_mode == "decade_fit":
        st.sidebar.caption("Scale by powers of 10 until target frame is reached")
        direction = st.sidebar.selectbox("Direction", options=["auto", "up", "down"], index=0)
        max_steps = st.sidebar.number_input("Max steps", min_value=0, value=6, step=1)
        allow_overshoot = st.sidebar.checkbox("Allow overshoot", value=False)
        exact_fit = st.sidebar.checkbox("Apply exact-fit after decade", value=False)
        scale_opts.update({
            "decade_base": 10.0,
            "direction": direction,
            "max_steps": int(max_steps),
            "allow_overshoot": bool(allow_overshoot),
            "exact_fit": bool(exact_fit),
        })

    # Frame Normalization controls
    st.sidebar.header("Frame Normalization")
    enable_fit = st.sidebar.checkbox("Normalize to frame (fit)", value=True)
    normalize_opts = None
    if enable_fit:
        frame_w = st.sidebar.number_input("Frame width (mm)", min_value=10.0, value=1000.0, step=10.0)
        frame_h = st.sidebar.number_input("Frame height (mm)", min_value=10.0, value=1000.0, step=10.0)
        margin = st.sidebar.number_input("Margin (mm)", min_value=0.0, value=0.0, step=1.0)
        origin = st.sidebar.checkbox("Translate to origin", value=True)
        must_fit = st.sidebar.checkbox("Must fit within frame", value=True)

        # Enforce tens-only values for frame size (with warning)
        import math

        def _to_tens(x: float) -> float:
            return float(int(round(x / 10.0)) * 10)

        fw_tens = _to_tens(frame_w)
        fh_tens = _to_tens(frame_h)
        if fw_tens != frame_w or fh_tens != frame_h:
            st.sidebar.warning(f"Frame adjusted to tens: {fw_tens} × {fh_tens} mm")

        # If must-fit is on, lock decade-fit to no overshoot and no exact-fit
        if scale_mode == "decade_fit" and must_fit:
            scale_opts["allow_overshoot"] = False
            scale_opts["exact_fit"] = False

        normalize_opts = {
            "mode": "fit",
            "frame_w": fw_tens,
            "frame_h": fh_tens,
            "margin": margin,
            "origin": origin,
            "must_fit": bool(must_fit),
        }

    base_report = _run_analysis(
        work,
        selected_groups=None,
        sheet_width=sheet_w,
        sheet_height=sheet_h,
        group_layer_map=None,
        soften_opts=soften_opts,
        fillet_opts=fillet_opts,
        scale_opts=scale_opts,
        normalize_opts=normalize_opts,
        frame_quantity=int(frame_qty),
    )

    group_names = list(base_report.get("groups", {}).keys())
    selection_placeholder = st.empty()
    if group_names:
        # quick toggles
        c1, c2 = st.columns(2)
        if c1.button("Select All"):
            st.session_state["_wjp_groups_sel"] = group_names[:]
        if c2.button("Select None"):
            st.session_state["_wjp_groups_sel"] = []

        default_sel = st.session_state.get("_wjp_groups_sel", group_names)
        selected_groups = selection_placeholder.multiselect(
            "Select groups to include",
            options=group_names,
            default=default_sel,
        )
    else:
        selected_groups = []
        selection_placeholder.write("No similarity groups detected; all objects will be used.")

    if group_names and not selected_groups:
        st.warning("No groups selected. Nesting/toolpath results will be empty.")

    # Layer reassignment UI
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
                choice = col.selectbox(f"{gname}", options=options, index=options.index(current) if current in options else 0)
                if choice != "Auto":
                    updated_map[gname] = choice
            c_apply, c_reset = st.columns([1, 1])
            if c_apply.button("Apply Mapping"):
                st.session_state["_wjp_group_layer_map"] = updated_map
                group_layer_map = updated_map
            if c_reset.button("Reset Mapping"):
                st.session_state["_wjp_group_layer_map"] = {}
                group_layer_map = {}

    active_report = (
        base_report
        if (not group_names or set(selected_groups) == set(group_names)) and not group_layer_map
        else _run_analysis(
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

    st.subheader("DXF Preview")
    # Layer visibility filter (preview-only)
    layers_present = list((active_report.get("layers") or {}).keys())
    with st.sidebar.expander("Layer Visibility", expanded=False):
        if layers_present:
            default_layers = st.session_state.get("_wjp_layers_sel_legacy", layers_present)
            selected_layers = st.multiselect(
                "Show layers",
                options=layers_present,
                default=default_layers,
                key="layers_preview_legacy",
            )
            st.session_state["_wjp_layers_sel_legacy"] = selected_layers or layers_present
        else:
            selected_layers = []
    # Scale/fit banner
    scale_info = active_report.get("scale", {}) or {}
    smeta = scale_info.get("normalize") or {}
    dec = smeta.get("decade") or {}
    normm = smeta.get("normalize") or {}
    clamp = smeta.get("clamp") or {}
    if scale_info:
        parts = []
        um = scale_info.get("units_mode", "auto")
        uf = scale_info.get("units_factor")
        parts.append(f"units={um} (factor {uf})")
        if dec and dec.get("mode") == "decade_fit":
            try:
                parts.append(f"10× steps: n={dec.get('steps', 0)}, factor={dec.get('applied_factor', 1.0):.4f}")
            except Exception:
                pass
        if normm.get("mode") == "fit":
            bb = normm.get("bbox_after") or {}
            try:
                parts.append(f"fit: {normm.get('factor', 1.0):.3f}; final {bb.get('width', 0):.2f}×{bb.get('height', 0):.2f} mm")
            except Exception:
                pass
        if clamp.get("applied"):
            parts.append(f"clamped: factor {clamp.get('factor', 1.0):.4f}")
        st.caption(" | ".join(parts))
    # Build preview copy
    preview_report = dict(active_report)
    try:
        comps = active_report.get("components", [])
        if selected_layers:
            comps = [c for c in comps if c.get("layer") in selected_layers]
        preview_report["components"] = comps
    except Exception:
        pass
    _plot_components(preview_report)

    st.subheader("Similarity Groups")
    _display_group_summary(base_report, selected_groups or group_names)

    st.subheader("Selected Metrics")
    _display_metrics(active_report)

    st.subheader("Quality Checks")
    _display_quality(active_report)

    # Checklist display
    st.subheader("Checklist")
    mc = active_report.get("mastery_checklist") or {}
    if mc:
        warns = mc.get("Warnings") or []
        cols = st.columns(3)
        cols[0].metric("Entities (types)", len(mc.get("Entities", {})))
        cols[1].metric("Total length (mm)", mc.get("TotalLength_mm", 0))
        cols[2].metric("Pierces", mc.get("Pierces", 0))
        if warns:
            st.warning("; ".join(map(str, warns)))
    else:
        st.caption("No checklist available for this analysis.")

    st.subheader("Outputs")
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
                col.download_button(label=f"Download {label}", data=fh.read(), file_name=Path(path).name)
        else:
            col.write(f"{label}: not available")

    st.subheader("Object Details")
    components = active_report.get("components", [])
    if components:
        table_rows = [
            {
                "ID": comp["id"],
                "Group": comp.get("group"),
                "Layer": comp.get("layer"),
                "Area (mm²)": comp.get("area"),
                "Perimeter (mm)": comp.get("perimeter"),
                "Vertices": comp.get("vertex_count"),
                "Size W (mm)": comp.get("size_w_mm", 0.0),
                "Size H (mm)": comp.get("size_h_mm", 0.0),
            }
            for comp in components
            if comp.get("selected", True)
        ]
        if table_rows:
            st.dataframe(
                table_rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "ID": st.column_config.NumberColumn(format="%d", width="small"),
                    "Group": st.column_config.TextColumn(width="small"),
                    "Layer": st.column_config.TextColumn(width="small"),
                    "Area (mm²)": st.column_config.NumberColumn(format="%.2f"),
                    "Perimeter (mm)": st.column_config.NumberColumn(format="%.2f"),
                    "Vertices": st.column_config.NumberColumn(format="%d", width="small"),
                    "Size W (mm)": st.column_config.NumberColumn(format="%.2f"),
                    "Size H (mm)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        else:
            st.write("No components selected.")
    else:
        st.write("No components available.")


if __name__ == "__main__":
    main()










