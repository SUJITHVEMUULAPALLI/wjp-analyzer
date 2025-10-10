"""Shared Streamlit components and helpers for the Waterjet DXF UI."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import streamlit as st
import os
from pathlib import Path
import json

# Path shim so imports work when launched via streamlit
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

"""
Note on imports:
- Heavy deps (ezdxf, shapely) are required by analyze_dxf, but may not be installed
  in minimal setups where only Image->DXF is used. To keep Streamlit pages loading,
  we import analyze_dxf lazily inside run_analysis and surface a friendly message
  instead of crashing at import time.
"""


def _load_env_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # optional fallback file for local dev
    try:
        p = Path("wjp.env.txt")
        if p.exists():
            for line in p.read_text(errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        return None
    return None


def get_ai_status(timeout: float = 2.0) -> dict:
    """Return a dict describing OpenAI and Ollama availability.

    This function is side-effect free; callers can decide how to render.
    """
    status: dict = {
        "openai_key_present": False,
        "openai_ok": None,
        "openai_error": None,
        "ollama_ok": None,
        "ollama_error": None,
    }
    key = _load_env_key()
    status["openai_key_present"] = bool(key)
    if key:
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=key)
            models = client.models.list()
            # if we can list models, mark OK
            _ = len(getattr(models, "data", []) or [])
            status["openai_ok"] = True
        except Exception as e:  # pragma: no cover
            status["openai_ok"] = False
            status["openai_error"] = str(e)[:200]
    try:
        import requests  # type: ignore

        r = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        status["ollama_ok"] = bool(r.status_code == 200)
        if r.status_code != 200:
            status["ollama_error"] = f"HTTP {r.status_code}"
    except Exception as e:  # pragma: no cover
        status["ollama_ok"] = False
        status["ollama_error"] = str(e)[:200]
    return status


def render_ai_status(compact: bool = True) -> None:
    """Render AI status in the current Streamlit container (sidebar or body)."""
    s = get_ai_status()
    oai = "OK" if s.get("openai_ok") else ("Not configured" if not s.get("openai_key_present") else "Error")
    oll = "OK" if s.get("ollama_ok") else "Unavailable"
    st.caption(f"OpenAI: {oai} | Ollama: {oll}")
    if not compact:
        with st.expander("Details", expanded=False):
            st.json({k: v for k, v in s.items() if v is not None})

def ensure_workdir(upload_name: str, file_bytes: bytes) -> dict:
    """Persist uploaded file in a temp working dir and return session metadata."""
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


def _selection_label(key: Tuple[str, ...]) -> str:
    if key == ("__all__",):
        return "all"
    joined = ",".join(key)
    sig = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"sel_{len(key)}_{sig}"


def run_analysis(
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
    # Lazy import to avoid hard failures if ezdxf/shapely are missing.
    try:
        from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf  # type: ignore
    except Exception as exc:
        st.error(
            "DXF analysis dependencies are missing. Please install requirements (ezdxf, shapely).\n"
            "Run: pip install -r requirements.txt\n\n"
            f"Details: {exc}"
        )
        # Return a minimal, well-formed report stub so pages can continue rendering.
        return {
            "file": work.get("name", "uploaded.dxf"),
            "metrics": {},
            "scale": {},
            "groups": {},
            "components": [],
            "layers": {},
            "toolpath": {},
            "nesting": {},
            "artifacts": {},
            "selection": {"groups": [], "component_ids": []},
            "quality": {},
        }
    key = tuple(sorted(selected_groups)) if selected_groups else ("__all__",)
    # Include mapping signature in cache key to avoid stale results
    map_sig: Tuple[Tuple[str, str], ...] | Tuple[str, ...]
    if group_layer_map:
        map_sig = tuple(sorted((k, v) for k, v in group_layer_map.items()))
    else:
        map_sig = ("__nomap__",)
    # softening signature
    if soften_opts:
        soft_sig = tuple(sorted((str(k), str(v)) for k, v in soften_opts.items()))
    else:
        soft_sig = ("__nosoft__",)
    if fillet_opts:
        fillet_sig = tuple(sorted((str(k), str(v)) for k, v in fillet_opts.items()))
    else:
        fillet_sig = ("__nofillet__",)
    if scale_opts:
        scale_sig = tuple(sorted((str(k), str(v)) for k, v in scale_opts.items()))
    else:
        scale_sig = ("__noscale__",)
    if normalize_opts:
        norm_sig = tuple(sorted((str(k), str(v)) for k, v in normalize_opts.items()))
    else:
        norm_sig = ("__nonorm__",)
    cache_key = (key, map_sig, soft_sig, fillet_sig, scale_sig, norm_sig, (frame_quantity or 1))
    if cache_key in work["reports"]:
        return work["reports"][cache_key]

    label = _selection_label(
        key
        + ("|map|",)
        + tuple([f"{k}:{v}" for k, v in (group_layer_map or {}).items()])
        + ("|soft|",)
        + tuple([f"{k}:{v}" for k, v in (soften_opts or {}).items()])
        + ("|fillet|",)
        + tuple([f"{k}:{v}" for k, v in (fillet_opts or {}).items()])
    )
    out_dir = work["base_dir"] / f"analysis_{label}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        out_dir = work["base_dir"] / "analysis_sel"
        out_dir.mkdir(parents=True, exist_ok=True)

    args = AnalyzeArgs(out=str(out_dir))
    if sheet_width:
        args.sheet_width = float(sheet_width)
    if sheet_height:
        args.sheet_height = float(sheet_height)
    if isinstance(frame_quantity, int) and frame_quantity > 1:
        args.frame_quantity = int(frame_quantity)
    # Soften options
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
        mode = str(scale_opts.get("mode", "auto"))
        args.scale_mode = mode
        if mode == "factor":
            args.scale_factor = float(scale_opts.get("factor", 1.0))
        if mode == "decade_fit":
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
        mode = str(normalize_opts.get("mode", "none"))
        args.normalize_mode = mode
        if mode == "fit":
            args.target_frame_w_mm = float(normalize_opts.get("frame_w", 1000.0))
            args.target_frame_h_mm = float(normalize_opts.get("frame_h", 1000.0))
            args.frame_margin_mm = float(normalize_opts.get("margin", 0.0))
            args.normalize_origin = bool(normalize_opts.get("origin", True))
        # must-fit enforcement
        mf = normalize_opts.get("must_fit") if isinstance(normalize_opts, dict) else None
        if mf is not None:
            args.require_fit_within_frame = bool(mf)
    report = analyze_dxf(
        str(work["dxf_path"]),
        args,
        selected_groups=selected_groups,
        group_layer_overrides=group_layer_map,
    )
    work["reports"][cache_key] = report
    return report


def plot_components(report: dict, height: int = 600) -> None:
    components = report.get("components", [])
    if not components:
        st.info("No components to display.")
        return

    group_names = list(report.get("groups", {}).keys()) or ["All"]

    # TODO (Codex): Rename all base entities currently labeled as "GroupX" to "ObjectX".
    # These represent individual DXF polylines before similarity grouping.
    # After similarity grouping (via group_similar_objects), use "GroupX" only for clustered similar shapes.
    #
    # Implementation details:
    # 1. When plotting or listing raw detected polylines (before grouping),
    #    label them as Object1, Object2, ... instead of Group1, Group2.
    # 2. After group_similar_objects() runs, its output keys remain as Group1, Group2, ...
    # 3. Update legends, tooltips, and any text in the UI/CSV to follow:
    #       - Pre-grouping stage â†’ "Object"
    #       - Post-grouping stage â†’ "Group"
    # 4. Ensure exported CSV reflects the same distinction.
    cmap = plt.get_cmap("tab20", max(1, len(group_names)))
    color_map: Dict[str, tuple] = {name: cmap(idx) for idx, name in enumerate(group_names)}

    # Ensure minimum figure size to avoid zero width error
    fig, ax = plt.subplots(figsize=(max(8, len(components) * 0.5), max(8, len(components) * 0.5)))
    
    # Check if we have valid components with points
    valid_components = [comp for comp in components if comp.get("points") and len(comp["points"]) >= 2]
    
    if not valid_components:
        st.warning("No valid components with sufficient points to display.")
        plt.close(fig)
        return
    
    for comp in valid_components:
        pts = comp["points"]
        if not pts or len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        group = comp.get("group", "Ungrouped")
        color = color_map.get(group, (0.3, 0.3, 0.3, 1.0))
        alpha = 0.9 if comp.get("selected", True) else 0.15
        ax.fill(xs, ys, color=color, alpha=alpha)
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=max(alpha, 0.3))

    ax.set_aspect("equal", adjustable="box")
    if report.get("groups"):
        ax.set_title("DXF Preview - Groups")
    else:
        ax.set_title("DXF Preview - Objects")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # Only show legend when similarity groups exist
    if report.get("groups"):
        handles = [
            plt.Line2D([0], [0], color=color, marker="s", linestyle="", label=name)
            for name, color in color_map.items()
        ]
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize="small")

    try:
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying plot: {e}")
        st.info("This may be due to empty or invalid component data.")
    finally:
        plt.close(fig)
    
    stage = "Groups" if report.get("groups") else "Objects"
    st.caption(f"Stage: {stage} • Components: {len(valid_components)}")


def run_objects_analysis(work: dict) -> dict:
    """Run analysis to get individual objects without grouping."""
    try:
        from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf_objects_only  # type: ignore
    except Exception as exc:
        st.error(
            "DXF analysis dependencies are missing. Please install requirements (ezdxf, shapely).\n"
            "Run: pip install -r requirements.txt\n\n"
            f"Details: {exc}"
        )
        return {
            "file": work.get("name", "uploaded.dxf"),
            "components": [],
            "metrics": {},
            "groups": {},
            "layers": {},
            "quality": {},
            "toolpath": {},
            "nesting": {},
            "artifacts": {},
            "selection": {"groups": [], "component_ids": []},
        }
    
    # Check cache first
    cache_key = ("__objects_only__",)
    if cache_key in work["reports"]:
        return work["reports"][cache_key]
    
    args = AnalyzeArgs(out=str(work["base_dir"] / "objects_analysis"))
    report = analyze_dxf_objects_only(str(work["dxf_path"]), args)
    
    work["reports"][cache_key] = report
    return report


def create_groups_by_layer(components: List[dict]) -> Dict[str, List[dict]]:
    """Group components by their DXF layer."""
    groups = {}
    for comp in components:
        layer = comp.get("layer", "Unknown")
        if layer not in groups:
            groups[layer] = []
        groups[layer].append(comp)
    return groups


def create_groups_by_size(components: List[dict]) -> Dict[str, List[dict]]:
    """Group components by size ranges."""
    groups = {
        "Small (<100mm²)": [],
        "Medium (100-1000mm²)": [],
        "Large (>1000mm²)": []
    }
    
    for comp in components:
        area = comp.get("area", 0)
        if area < 100:
            groups["Small (<100mm²)"].append(comp)
        elif area < 1000:
            groups["Medium (100-1000mm²)"].append(comp)
        else:
            groups["Large (>1000mm²)"].append(comp)
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def create_groups_by_shape(components: List[dict]) -> Dict[str, List[dict]]:
    """Group components by shape type."""
    groups = {
        "Circles": [],
        "Rectangles": [],
        "Polygons": [],
        "Complex": []
    }
    
    for comp in components:
        area = comp.get("area", 0)
        perimeter = comp.get("perimeter", 0)
        vcount = comp.get("vertex_count", 0)
        
        # Calculate circularity
        circularity = (4.0 * 3.14159 * area) / (perimeter * perimeter + 1e-9) if perimeter > 0 else 0
        
        if circularity > 0.8:
            groups["Circles"].append(comp)
        elif vcount <= 4 and circularity < 0.3:
            groups["Rectangles"].append(comp)
        elif vcount <= 6:
            groups["Polygons"].append(comp)
        else:
            groups["Complex"].append(comp)
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def display_drag_drop_grouping(components: List[dict], selected_objects: List[int]) -> None:
    """Display drag and drop interface for manual grouping."""
    st.markdown("### ✋ Manual Grouping - Drag & Drop Interface")
    st.info("🎯 Drag objects between groups to organize them manually")
    
    # Get selected components
    selected_components = [c for c in components if c["id"] in selected_objects]
    
    if not selected_components:
        st.warning("No objects selected for grouping.")
        return
    
    # Initialize groups in session state
    if "_wjp_manual_groups" not in st.session_state:
        st.session_state["_wjp_manual_groups"] = {
            "Group 1": [],
            "Group 2": [],
            "Group 3": [],
            "Ungrouped": selected_components.copy()
        }
    
    groups = st.session_state["_wjp_manual_groups"]
    
    # Group management controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➕ Add Group", key="add_group"):
            group_num = len([k for k in groups.keys() if k.startswith("Group")]) + 1
            groups[f"Group {group_num}"] = []
            st.session_state["_wjp_manual_groups"] = groups
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", key="clear_groups"):
            groups = {"Ungrouped": selected_components.copy()}
            st.session_state["_wjp_manual_groups"] = groups
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", key="reset_groups"):
            groups = {
                "Group 1": [],
                "Group 2": [],
                "Group 3": [],
                "Ungrouped": selected_components.copy()
            }
            st.session_state["_wjp_manual_groups"] = groups
            st.rerun()
    
    with col4:
        if st.button("📊 Preview", key="preview_manual_groups"):
            # Filter out empty groups
            non_empty_groups = {k: v for k, v in groups.items() if v}
            display_grouping_preview(non_empty_groups, "Manual Grouping")
    
    # Display groups in columns
    group_names = list(groups.keys())
    num_cols = min(4, len(group_names))
    cols = st.columns(num_cols)
    
    for i, (group_name, group_components) in enumerate(groups.items()):
        with cols[i % num_cols]:
            st.markdown(f"**📁 {group_name}** ({len(group_components)} objects)")
            
            # Group actions
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("✏️", key=f"rename_{group_name}", help="Rename group"):
                    new_name = st.text_input(f"New name for {group_name}:", key=f"rename_input_{group_name}")
                    if new_name and new_name != group_name:
                        groups[new_name] = groups.pop(group_name)
                        st.session_state["_wjp_manual_groups"] = groups
                        st.rerun()
            
            with action_col2:
                if group_name != "Ungrouped" and st.button("🗑️", key=f"delete_{group_name}", help="Delete group"):
                    # Move objects to Ungrouped
                    groups["Ungrouped"].extend(group_components)
                    del groups[group_name]
                    st.session_state["_wjp_manual_groups"] = groups
                    st.rerun()
            
            # Display objects in group
            if group_components:
                for comp in group_components:
                    obj_name = comp.get('name', f'Object {comp["id"]}')
                    
                    # Object container with drag/drop simulation
                    with st.container():
                        col_obj1, col_obj2, col_obj3 = st.columns([3, 1, 1])
                        
                        with col_obj1:
                            st.write(f"🔸 {obj_name}")
                            st.caption(f"{comp.get('area', 0):.1f} mm²")
                        
                        with col_obj2:
                            # Move to other groups
                            target_groups = [g for g in group_names if g != group_name]
                            if target_groups:
                                target_group = st.selectbox(
                                    "Move to:",
                                    options=target_groups,
                                    key=f"move_{comp['id']}_{group_name}",
                                    label_visibility="collapsed"
                                )
                                if st.button("→", key=f"move_btn_{comp['id']}_{group_name}", help="Move object"):
                                    # Move object to target group
                                    groups[group_name].remove(comp)
                                    groups[target_group].append(comp)
                                    st.session_state["_wjp_manual_groups"] = groups
                                    st.rerun()
                        
                        with col_obj3:
                            if st.button("❌", key=f"remove_{comp['id']}_{group_name}", help="Remove from group"):
                                groups[group_name].remove(comp)
                                st.session_state["_wjp_manual_groups"] = groups
                                st.rerun()
            else:
                st.info("Empty group")
    
    # Summary
    total_grouped = sum(len(group) for name, group in groups.items() if name != "Ungrouped")
    total_ungrouped = len(groups.get("Ungrouped", []))
    
    st.markdown("### 📊 Manual Grouping Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Groups", len([g for g in groups.values() if g]))
    
    with col2:
        st.metric("Grouped Objects", total_grouped)
    
    with col3:
        st.metric("Ungrouped Objects", total_ungrouped)
    
    with col4:
        completion = (total_grouped / len(selected_components) * 100) if selected_components else 0
        st.metric("Completion", f"{completion:.1f}%")


def display_drag_drop_layering(components: List[dict], selected_objects: List[int]) -> None:
    """Display drag and drop interface for layer assignment."""
    st.markdown("### 🎨 Layer Assignment - Drag & Drop Interface")
    st.info("🎯 Drag objects to different layers to organize them")
    
    # Get selected components
    selected_components = [c for c in components if c["id"] in selected_objects]
    
    if not selected_components:
        st.warning("No objects selected for layer assignment.")
        return
    
    # Define available layers
    available_layers = ["OUTER", "INNER", "COMPLEX", "HOLE", "DECOR", "CUT", "ENGRAVE", "OTHER"]
    
    # Initialize layer assignments in session state
    if "_wjp_layer_assignments" not in st.session_state:
        st.session_state["_wjp_layer_assignments"] = {}
        for comp in selected_components:
            st.session_state["_wjp_layer_assignments"][comp["id"]] = comp.get("layer", "OUTER")
    
    layer_assignments = st.session_state["_wjp_layer_assignments"]
    
    # Layer management controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Layers", key="reset_layers"):
            for comp in selected_components:
                layer_assignments[comp["id"]] = comp.get("layer", "OUTER")
            st.session_state["_wjp_layer_assignments"] = layer_assignments
            st.rerun()
    
    with col2:
        if st.button("📊 Preview", key="preview_layers"):
            # Group by assigned layers
            layer_groups = {}
            for comp in selected_components:
                layer = layer_assignments.get(comp["id"], comp.get("layer", "OUTER"))
                if layer not in layer_groups:
                    layer_groups[layer] = []
                layer_groups[layer].append(comp)
            display_grouping_preview(layer_groups, "Layer Assignment")
    
    with col3:
        if st.button("💾 Apply", key="apply_layers"):
            st.success("Layer assignments applied!")
            # TODO: Apply layer assignments to components
    
    # Display layers in columns
    num_cols = 4
    cols = st.columns(num_cols)
    
    for i, layer_name in enumerate(available_layers):
        with cols[i % num_cols]:
            # Get objects assigned to this layer
            layer_objects = [comp for comp in selected_components 
                           if layer_assignments.get(comp["id"], comp.get("layer", "OUTER")) == layer_name]
            
            st.markdown(f"**🎨 {layer_name}** ({len(layer_objects)} objects)")
            
            # Layer color indicator
            layer_colors = {
                "OUTER": "🔴", "INNER": "🔵", "COMPLEX": "🟡", 
                "HOLE": "⚫", "DECOR": "🟣", "CUT": "🟠", 
                "ENGRAVE": "🟢", "OTHER": "⚪"
            }
            st.write(f"{layer_colors.get(layer_name, '⚪')} {layer_name}")
            
            # Display objects in layer
            if layer_objects:
                for comp in layer_objects:
                    obj_name = comp.get('name', f'Object {comp["id"]}')
                    
                    # Object container
                    with st.container():
                        col_obj1, col_obj2 = st.columns([3, 1])
                        
                        with col_obj1:
                            st.write(f"🔸 {obj_name}")
                            st.caption(f"{comp.get('area', 0):.1f} mm²")
                        
                        with col_obj2:
                            # Move to other layers
                            other_layers = [l for l in available_layers if l != layer_name]
                            if other_layers:
                                new_layer = st.selectbox(
                                    "Move to:",
                                    options=other_layers,
                                    key=f"layer_{comp['id']}_{layer_name}",
                                    label_visibility="collapsed"
                                )
                                if st.button("→", key=f"layer_btn_{comp['id']}_{layer_name}", help="Move to layer"):
                                    # Update layer assignment
                                    layer_assignments[comp["id"]] = new_layer
                                    st.session_state["_wjp_layer_assignments"] = layer_assignments
                                    st.rerun()
            else:
                st.info("No objects")
    
    # Summary
    st.markdown("### 📊 Layer Assignment Summary")
    
    # Count objects per layer
    layer_counts = {}
    for comp in selected_components:
        layer = layer_assignments.get(comp["id"], comp.get("layer", "OUTER"))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    # Display layer statistics
    cols = st.columns(min(4, len(layer_counts)))
    for i, (layer, count) in enumerate(layer_counts.items()):
        with cols[i % 4]:
            color = layer_colors.get(layer, "⚪")
            st.metric(f"{color} {layer}", count)


def display_grouping_preview(groups: Dict[str, List[dict]], method: str) -> None:
    """Display preview of proposed groups."""
    st.markdown(f"### 👁️ {method} Preview")
    
    if not groups:
        st.warning("No groups created.")
        return
    
    # Group statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Groups", len(groups))
    
    with col2:
        total_objects = sum(len(group) for group in groups.values())
        st.metric("Total Objects", total_objects)
    
    with col3:
        total_area = sum(sum(c.get("area", 0) for c in group) for group in groups.values())
        st.metric("Total Area", f"{total_area:.1f} mm²")
    
    with col4:
        avg_group_size = total_objects / len(groups) if groups else 0
        st.metric("Avg Group Size", f"{avg_group_size:.1f}")
    
    # Group details
    st.markdown("**📋 Group Details:**")
    
    for group_name, group_components in groups.items():
        with st.expander(f"🔍 {group_name} ({len(group_components)} objects)", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown("**📊 Group Statistics:**")
                group_area = sum(c.get("area", 0) for c in group_components)
                group_perimeter = sum(c.get("perimeter", 0) for c in group_components)
                avg_area = group_area / len(group_components) if group_components else 0
                
                st.write(f"• **Objects:** {len(group_components)}")
                st.write(f"• **Total Area:** {group_area:.1f} mm²")
                st.write(f"• **Total Perimeter:** {group_perimeter:.1f} mm")
                st.write(f"• **Average Area:** {avg_area:.1f} mm²")
                
                # Layer information
                layers = set(c.get("layer", "Unknown") for c in group_components)
                st.write(f"• **Layers:** {', '.join(layers)}")
            
            with col2:
                st.markdown("**🎨 Visual Properties:**")
                
                # Size distribution
                small_count = sum(1 for c in group_components if c.get("area", 0) < 100)
                medium_count = sum(1 for c in group_components if 100 <= c.get("area", 0) < 1000)
                large_count = sum(1 for c in group_components if c.get("area", 0) >= 1000)
                
                if small_count > 0:
                    st.info(f"🔸 Small: {small_count}")
                if medium_count > 0:
                    st.info(f"🔹 Medium: {medium_count}")
                if large_count > 0:
                    st.info(f"🔶 Large: {large_count}")
            
            with col3:
                st.markdown("**⚙️ Actions:**")
                
                # Group actions
                if st.button(f"Rename Group", key=f"rename_{group_name}"):
                    st.info(f"Rename functionality for {group_name} will be implemented")
                
                if st.button(f"Split Group", key=f"split_{group_name}"):
                    st.info(f"Split functionality for {group_name} will be implemented")
                
                if st.button(f"Delete Group", key=f"delete_{group_name}"):
                    st.info(f"Delete functionality for {group_name} will be implemented")
            
            # Object list in group
            st.markdown("**📋 Objects in Group:**")
            for comp in group_components:
                obj_name = comp.get('name', f'Object {comp["id"]}')
                st.write(f"• {obj_name} - {comp.get('area', 0):.1f} mm²")


def _build_ai_features(report: dict, selected_ids: List[int]) -> dict:
    """Build a minimal, privacy-safe feature set for AI suggestions (no geometry)."""
    components = report.get("components", [])
    selected = [c for c in components if not selected_ids or c.get("id") in selected_ids]
    if not selected:
        selected = components

    areas = [float(c.get("area", 0.0)) for c in selected]
    perims = [float(c.get("perimeter", 0.0)) for c in selected]
    layers: Dict[str, int] = {}
    shapes: Dict[str, int] = {}
    for c in selected:
        layer = str(c.get("layer", "0"))
        shape = str(c.get("shape", "unknown"))
        layers[layer] = layers.get(layer, 0) + 1
        shapes[shape] = shapes.get(shape, 0) + 1

    total = len(selected)
    sorted_areas = sorted(areas)
    def _q(idx: float) -> float:
        return sorted_areas[int(min(max(idx * max(total - 1, 0), 0), max(total - 1, 0)))] if total else 0.0

    features = {
        "counts": {"total": total, "layers": layers, "shapes": shapes},
        "stats": {
            "area_sum": float(sum(areas)),
            "area_mean": float(sum(areas) / max(total, 1)),
            "perimeter_mean": float(sum(perims) / max(len(perims), 1)) if perims else 0.0,
        },
        "thresholds": {"small_area": float(_q(0.33)), "large_area": float(_q(0.66))},
    }
    return features


def _ai_group_layer_suggestions(features: dict) -> dict:
    """Attempt OpenAI suggestions; fallback to heuristics when unavailable."""
    suggestion = {
        "method": "size",
        "size_thresholds": features.get("thresholds", {}),
        "layer_policy": "keep_original",
        "notes": ["Heuristic: group by size terciles; keep original layers."],
    }

    # Try OpenAI if configured (safe, cost-aware)
    try:
        # Prefer secure config manager if available
        from wjp_analyser.config.secure_config import get_ai_config  # type: ignore
        ai_cfg = get_ai_config()
        api_key = getattr(ai_cfg, "openai_api_key", None)
        model = getattr(ai_cfg, "openai_model", "gpt-4o-mini")
        if not api_key:
            return suggestion

        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            prompt = (
                "You receive summarized DXF features (counts, layers, shapes, size thresholds).\n"
                "Recommend compact JSON with keys: method[size|layer|shape|auto], size_thresholds{small_area,large_area}, "
                "layer_policy[keep_original|by_shape|by_size], notes (short list).\n\n"
                f"Features: {features}"
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a waterjet DXF preparation assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            content = (resp.choices[0].message.content or "{}").strip()
            import re, json
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
            if m:
                data = json.loads(m.group(1))
            else:
                si, ei = content.find('{'), content.rfind('}')
                data = json.loads(content[si:ei+1]) if si != -1 and ei > si else {}
            if isinstance(data, dict):
                for k in ("method", "size_thresholds", "layer_policy", "notes"):
                    if k in data:
                        suggestion[k] = data[k]
        except Exception:
            return suggestion
    except Exception:
        return suggestion
    return suggestion


def display_ai_assist(report: dict, selected_ids: List[int]) -> None:
    """AI Assist panel suggesting grouping/layering. Works without API key."""
    with st.expander("🧠 AI Assist (optional)", expanded=False):
        # Availability
        try:
            render_ai_status(compact=True)
        except Exception:
            pass

        features = _build_ai_features(report, selected_ids)
        st.caption("The assistant uses only derived statistics, never raw geometry.")

        if st.button("Generate AI suggestions", key="ai_suggest_btn"):
            sugg = _ai_group_layer_suggestions(features)
            st.json(sugg)

            comps = report.get("components", [])
            sel = [c for c in comps if not selected_ids or c.get("id") in selected_ids] or comps
            method = str(sugg.get("method", "size"))
            if method == "layer":
                groups = create_groups_by_layer(sel)
                display_grouping_preview(groups, "AI: Group by Layer")
            elif method == "shape":
                groups = create_groups_by_shape(sel)
                display_grouping_preview(groups, "AI: Group by Shape")
            else:
                groups = create_groups_by_size(sel)
                display_grouping_preview(groups, "AI: Group by Size")

            # Apply suggestions to session state
            a1, a2 = st.columns(2)
            with a1:
                if st.button("Apply Grouping", key="ai_apply_grouping"):
                    try:
                        st.session_state["_wjp_manual_groups"] = {k: list(v) for k, v in groups.items()}
                        st.success("Applied suggested grouping to manual groups.")
                    except Exception as e:
                        st.error(f"Failed to apply grouping: {e}")
            with a2:
                if st.button("Apply Layer Policy", key="ai_apply_layers"):
                    try:
                        policy = str(sugg.get("layer_policy", "keep_original"))
                        layer_assignments = st.session_state.get("_wjp_layer_assignments", {})
                        if policy == "by_shape":
                            for c in sel:
                                shape = str(c.get("shape", "unknown")).lower()
                                if "circle" in shape:
                                    layer_assignments[c["id"]] = "INNER"
                                elif c.get("vertex_count", 0) <= 4:
                                    layer_assignments[c["id"]] = "CUT"
                                else:
                                    layer_assignments[c["id"]] = "DECOR"
                        elif policy == "by_size":
                            thr = sugg.get("size_thresholds", {})
                            s_thr = float(thr.get("small_area", 0.0) or 0.0)
                            l_thr = float(thr.get("large_area", 0.0) or 0.0)
                            for c in sel:
                                a = float(c.get("area", 0.0))
                                if a <= s_thr:
                                    layer_assignments[c["id"]] = "ENGRAVE"
                                elif a >= l_thr:
                                    layer_assignments[c["id"]] = "CUT"
                                else:
                                    layer_assignments[c["id"]] = "DECOR"
                        else:
                            for c in sel:
                                layer_assignments[c["id"]] = c.get("layer", "OUTER")
                        st.session_state["_wjp_layer_assignments"] = layer_assignments
                        st.success("Applied suggested layer policy.")
                    except Exception as e:
                        st.error(f"Failed to apply layers: {e}")

def display_object_selection(report: dict, selected_objects: List[int]) -> None:
    """Display object selection interface."""
    components = report.get("components", [])
    if not components:
        st.write("No objects found in DXF file.")
        return

    st.markdown("### 🔍 Object Discovery & Selection")
    st.write(f"**Found {len(components)} objects** - Select which ones to analyze")
    
    # Selection controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Select All"):
            if hasattr(st, 'session_state') and st.session_state:
                st.session_state["_wjp_objects_sel"] = [c["id"] for c in components]
    
    with col2:
        if st.button("Select None"):
            if hasattr(st, 'session_state') and st.session_state:
                st.session_state["_wjp_objects_sel"] = []
    
    with col3:
        if st.button("Select Small (<100mm²)"):
            if hasattr(st, 'session_state') and st.session_state:
                small_ids = [c["id"] for c in components if c.get("area", 0) < 100]
                st.session_state["_wjp_objects_sel"] = small_ids
    
    with col4:
        if st.button("Select Large (>1000mm²)"):
            if hasattr(st, 'session_state') and st.session_state:
                large_ids = [c["id"] for c in components if c.get("area", 0) > 1000]
                st.session_state["_wjp_objects_sel"] = large_ids
    
    # Object list with selection
    st.markdown("**📋 Objects List:**")
    
    # Filtering options
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        filter_size = st.selectbox("Filter by Size", ["All", "Small (<100mm²)", "Medium (100-1000mm²)", "Large (>1000mm²)"], index=0)
    
    with filter_col2:
        filter_layer = st.selectbox("Filter by Layer", ["All"] + list(set(c.get("layer", "Unknown") for c in components)), index=0)
    
    with filter_col3:
        filter_shape = st.selectbox("Filter by Shape", ["All", "Circle", "Rectangle", "Polygon", "Complex"], index=0)
    
    # Apply filters
    filtered_components = components
    if filter_size != "All":
        if filter_size == "Small (<100mm²)":
            filtered_components = [c for c in filtered_components if c.get("area", 0) < 100]
        elif filter_size == "Medium (100-1000mm²)":
            filtered_components = [c for c in filtered_components if 100 <= c.get("area", 0) < 1000]
        elif filter_size == "Large (>1000mm²)":
            filtered_components = [c for c in filtered_components if c.get("area", 0) >= 1000]
    
    if filter_layer != "All":
        filtered_components = [c for c in filtered_components if c.get("layer") == filter_layer]
    
    if filter_shape != "All":
        # Simple shape classification based on circularity
        if filter_shape == "Circle":
            filtered_components = [c for c in filtered_components if c.get("circularity", 0) > 0.8]
        elif filter_shape == "Rectangle":
            filtered_components = [c for c in filtered_components if c.get("vertex_count", 0) <= 4 and c.get("circularity", 0) < 0.3]
        elif filter_shape == "Polygon":
            filtered_components = [c for c in filtered_components if c.get("vertex_count", 0) <= 6]
        elif filter_shape == "Complex":
            filtered_components = [c for c in filtered_components if c.get("vertex_count", 0) > 6]
    
    st.write(f"**Showing {len(filtered_components)} objects** (filtered from {len(components)} total)")
    
    # Display filtered objects
    for comp in filtered_components:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            is_selected = comp["id"] in selected_objects
            if st.checkbox("Select", value=is_selected, key=f"select_obj_{comp['id']}", label_visibility="collapsed"):
                if comp["id"] not in selected_objects:
                    selected_objects.append(comp["id"])
            else:
                if comp["id"] in selected_objects:
                    selected_objects.remove(comp["id"])
        
        with col2:
            obj_name = comp.get('name', f'Object {comp["id"]}')
            st.write(f"**{obj_name}**")
            st.write(f"• Area: {comp.get('area', 0):.1f} mm²")
            st.write(f"• Perimeter: {comp.get('perimeter', 0):.1f} mm")
            st.write(f"• Vertices: {comp.get('vertex_count', 0)}")
            st.write(f"• Layer: {comp.get('layer', 'Unknown')}")
        
        with col3:
            # Visual indicators
            area = comp.get("area", 0)
            if area < 100:
                st.info("🔸 Small")
            elif area < 1000:
                st.info("🔹 Medium")
            else:
                st.info("🔶 Large")
            
            vcount = comp.get("vertex_count", 0)
            if vcount <= 4:
                st.success("✅ Simple")
            elif vcount <= 20:
                st.warning("⚠️ Moderate")
            else:
                st.error("❌ Complex")
    
    # Selection summary
    if selected_objects:
        selected_components = [c for c in components if c["id"] in selected_objects]
        total_area = sum(c.get("area", 0) for c in selected_components)
        avg_area = total_area / len(selected_components) if selected_components else 0
        
        st.markdown("### 📊 Selection Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Selected Objects", len(selected_objects))
        
        with col2:
            st.metric("Total Area", f"{total_area:.1f} mm²")
        
        with col3:
            st.metric("Average Area", f"{avg_area:.1f} mm²")
        
        with col4:
            layers_count = len(set(c.get("layer", "Unknown") for c in selected_components))
            st.metric("Unique Layers", layers_count)
    else:
        st.warning("No objects selected. Please select at least one object to proceed.")


def display_group_summary(report: dict, selected_groups: List[str]) -> None:
    groups = report.get("groups", {})
    if not groups:
        st.write("No similarity groups detected.")
        return

    # Enhanced group display with visual indicators
    st.markdown("### 📊 Object Groups Analysis")
    
    # Create columns for better layout
    cols = st.columns([2, 1, 1])
    
    with cols[0]:
        st.markdown("**Group Details**")
    
    with cols[1]:
        st.markdown("**Visual Preview**")
    
    with cols[2]:
        st.markdown("**Actions**")
    
    # Display each group with enhanced information
    for name, meta in groups.items():
        with st.expander(f"🔍 {name}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Group statistics
                st.markdown(f"**📈 Statistics:**")
                st.write(f"• **Count:** {meta.get('count', 0)} objects")
                st.write(f"• **Vertices:** {meta.get('vcount', 0)} per object")
                st.write(f"• **Area:** {meta.get('avg_area', 0.0):.2f} mm²")
                st.write(f"• **Perimeter:** {meta.get('avg_perimeter', 0.0):.2f} mm")
                st.write(f"• **Circularity:** {meta.get('avg_circ', 0.0):.3f}")
                
                # Complexity indicator
                complexity = meta.get("complexity", "simple")
                if complexity == "simple":
                    st.success("✅ Simple geometry - Easy to cut")
                elif complexity == "moderate":
                    st.warning("⚠️ Moderate complexity - Standard cutting")
                else:
                    st.error("❌ Complex geometry - Requires careful cutting")
            
            with col2:
                # Visual indicators
                st.markdown("**🎨 Visual Properties:**")
                
                # Size indicator
                area = meta.get("avg_area", 0)
                if area < 100:
                    st.info("🔸 Small object")
                elif area < 1000:
                    st.info("🔹 Medium object")
                else:
                    st.info("🔶 Large object")
                
                # Shape indicator
                circularity = meta.get("avg_circ", 0)
                if circularity > 0.8:
                    st.info("⭕ Circular shape")
                elif circularity < 0.3:
                    st.info("📐 Angular shape")
                else:
                    st.info("🔷 Mixed shape")
                
                # Layer information
                layers = meta.get("layers", {})
                if layers:
                    st.write("**📋 Layers:**")
                    for layer, count in layers.items():
                        st.write(f"• {layer}: {count} objects")
            
            with col3:
                # Actions and selection
                st.markdown("**⚙️ Actions:**")
                
                # Selection status
                is_selected = name in selected_groups
                if is_selected:
                    st.success("✅ Selected")
                else:
                    st.warning("⏸️ Not selected")
                
                # Quick actions
                if st.button(f"Toggle {name}", key=f"toggle_{name}"):
                    if hasattr(st, 'session_state') and st.session_state:
                        current_selection = st.session_state.get("_wjp_groups_sel", [])
                        if name in current_selection:
                            current_selection.remove(name)
                        else:
                            current_selection.append(name)
                        st.session_state["_wjp_groups_sel"] = current_selection
                        st.rerun()
                
                # Export individual group
                if st.button(f"Export {name}", key=f"export_{name}"):
                    st.info(f"Exporting group: {name}")
                    # TODO: Implement individual group export


def display_metrics(report: dict) -> None:
    metrics = report.get("metrics", {})
    cols = st.columns(3)
    cols[0].metric("Cutting length (mm)", metrics.get("length_internal_mm", 0))
    cols[1].metric("Pierce count", metrics.get("pierces", 0))
    cols[2].metric("Estimated cost", metrics.get("estimated_cutting_cost_inr", 0))


def display_quality(report: dict) -> None:
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


def display_checklist(report: dict) -> None:
    mc = report.get("mastery_checklist") or {}
    if not mc:
        st.write("No checklist available for this analysis.")
        return
    warnings = mc.get("Warnings") or []
    cols = st.columns(3)
    cols[0].metric("Entities (types)", len(mc.get("Entities", {})))
    cols[1].metric("Total length (mm)", mc.get("TotalLength_mm", 0))
    cols[2].metric("Pierces", mc.get("Pierces", 0))
    if warnings:
        st.warning("; ".join(map(str, warnings)))
    with st.expander("Checklist details", expanded=False):
        st.json({
            "Entities": mc.get("Entities"),
            "OpenPolylines": mc.get("OpenPolylines"),
            "ShakyPolylines": mc.get("ShakyPolylines"),
            "TinySegments": mc.get("TinySegments"),
            "Duplicates": mc.get("Duplicates"),
        })







