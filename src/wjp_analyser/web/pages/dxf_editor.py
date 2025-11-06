import os
from pathlib import Path
import sys

# Ensure 'src' is on sys.path so 'wjp_analyser' is importable when Streamlit runs this page directly
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = str(_THIS_FILE.parents[3])  # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st
import importlib
import time
from typing import List, Dict, Any
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

# Reload module to avoid stale imports during Streamlit hot-reload
import wjp_analyser.dxf_editor as dxf_mod
dxf_mod = importlib.reload(dxf_mod)


def _plot_entities_inline(entities):
    # Legacy plotter not used; preserved for reference
    return None


def main():
    st.title("DXF Editor")
    st.markdown("Upload a DXF, visualize, edit layers/groups, transform, draw, measure, validate, and download the edited file.")

    uploaded = st.file_uploader("Upload DXF File", type=["dxf"])

    if uploaded:
        doc = load_dxf(uploaded)
        msp = doc.modelspace()
        # Supported primitives used for direct editing tools
        entities = [e for e in msp.query("LINE CIRCLE LWPOLYLINE")] 
        # Also count all entities for visibility
        try:
            total_all = sum(1 for _ in msp)
        except Exception:
            total_all = len(entities)
        st.write(f"Total Entities (all): {total_all} · Supported for direct edit: {len(entities)}")
        if "state" not in st.session_state:
            st.session_state["state"] = {"selected": [], "hidden_layers": [], "session_path": "session.json"}

        if len(entities) == 0:
            # Display warning in red
            st.markdown("""
            <div style="background-color: #FFEBEE; border-left: 4px solid #F44336; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <p style="color: #C62828; font-weight: 500; margin: 0;">⚠️ <strong>Warning:</strong> No supported entities (LINE, CIRCLE, LWPOLYLINE) found. Showing polygonized preview for reference.</p>
            </div>
            """, unsafe_allow_html=True)
            # Enhanced fallback preview with normalization and layer coloring
            try:
                from wjp_analyser.nesting.dxf_extractor import extract_polygons
                from wjp_analyser.dxf_editor.preview_utils import (
                    normalize_to_origin,
                    classify_polygon_layers,
                    get_layer_color,
                    convert_hex_to_rgb
                )
                import matplotlib.pyplot as plt
                
                # Save to temp and extract
                temp_dir = Path("output") / "editor_preview"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / "preview_src.dxf"
                save_dxf(doc, str(temp_path))
                polys = extract_polygons(str(temp_path))
                
                if polys:
                    # Classify polygons into layers
                    classified = classify_polygon_layers(polys)
                    
                    # Normalize all polygons to origin (0,0)
                    # First, find the bounding box of all polygons
                    all_points = []
                    for p in polys:
                        pts = p.get("points", [])
                        if len(pts) >= 3:
                            all_points.extend(pts)
                    
                    if all_points:
                        min_x = min(p[0] for p in all_points)
                        min_y = min(p[1] for p in all_points)
                        offset = (min_x, min_y)
                        
                        # Create figure with normalized coordinates
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Render polygons with layer-based colors
                        # Render OUTER first (background), then others on top
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
                    else:
                        st.info("No valid polygons detected in this DXF.")
                else:
                    st.info("No closed polygons detected in this DXF.")
            except Exception as ex:
                import traceback
                st.error(f"Preview extraction unavailable: {ex}")
                st.code(traceback.format_exc())
        else:
            with st.sidebar:
                st.header("Layers")
                try:
                    layer_names = list(dxf_mod.get_layers(doc).keys())
                except Exception:
                    # Fallback: minimal safe enumeration
                    try:
                        layer_names = [getattr(e.dxf, "name", getattr(e, "name", "0")) for e in doc.layers]
                    except Exception:
                        layer_names = ["0"]
                new_layer = st.text_input("Create Layer", value="WJP_NEW")
                if st.button("Add Layer"):
                    ensure_layer(doc, new_layer, color=7)
                    st.success(f"Layer '{new_layer}' ensured.")
                layer_pick = st.selectbox("Pick Layer", options=layer_names if layer_names else ["0"])
                new_name = st.text_input("Rename Layer To", value=layer_pick)
                if st.button("Rename Layer"):
                    rename_layer(doc, layer_pick, new_name)
                    st.success(f"Renamed layer {layer_pick} → {new_name}")
                new_color = st.number_input("ACI Color (1-255)", value=7, min_value=1, max_value=255, step=1)
                if st.button("Recolor Layer"):
                    recolor_layer(doc, layer_pick, new_color)
                    st.success(f"Layer {layer_pick} recolored to {new_color}")

                st.subheader("Layer Visibility")
                vis_multi = st.multiselect("Hidden Layers", options=layer_names, default=st.session_state["state"]["hidden_layers"]) 
                st.session_state["state"]["hidden_layers"] = vis_multi

                st.header("Groups")
                st.write("Existing:", list_groups(doc))
                grp_name = st.text_input("New Group Name", value="group_1")
                if st.button("Group Selection"):
                    sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
                    create_group(doc, grp_name, [e.dxf.handle for e in sel])
                    st.success(f"Created group '{grp_name}' with {len(sel)} entities")

                st.header("Selection")
                st.write("Selected handles:", st.session_state["state"]["selected"])
                x = st.number_input("Pick X", value=0.0)
                y = st.number_input("Pick Y", value=0.0)
                tol = st.number_input("Pick Tolerance", value=2.0)
                if st.button("Pick @ (X,Y)"):
                    picked = pick_entity(entities, x, y, tol)
                    if picked and picked.dxf.handle not in st.session_state["state"]["selected"]:
                        st.session_state["state"]["selected"].append(picked.dxf.handle)
                        st.success(f"Selected {picked.dxftype()} on layer {picked.dxf.layer} handle {picked.dxf.handle}")
                    elif not picked:
                        st.warning("No entity near that point.")
                if st.button("Clear Selection"):
                    st.session_state["state"]["selected"] = []

                st.subheader("Move Selected to Layer")
                move_to = st.text_input("Target Layer", value="OUTER")
                if st.button("Move to Layer"):
                    ensure_layer(doc, move_to, color=7)
                    sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
                    move_entities_to_layer(sel, move_to)
                    st.success(f"Moved {len(sel)} entities to layer '{move_to}'")

                st.header("Session")
                if st.button("Save Session State"):
                    save_session(st.session_state["state"]["session_path"], st.session_state["state"])
                    st.success("Session saved.")
                if st.button("Load Session State"):
                    st.session_state["state"] = load_session(st.session_state["state"]["session_path"])
                    st.success("Session loaded.")

            # Enhanced preview with normalization and layer-based coloring
            # Note: Entity-based preview uses DXF layer names for coloring
            # (Polygonized preview uses geometric classification for more accurate OUTER/INNER detection)
            
            # Build warning markers from AI analysis if available
            warning_markers = {}
            if st.session_state.get("_editor_ai_analysis") and st.session_state.get("_editor_analysis_report"):
                ai_analysis = st.session_state.get("_editor_ai_analysis")
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
            
            fig = plot_entities(
                entities,
                selected_handles=st.session_state["state"]["selected"],
                hidden_layers=st.session_state["state"]["hidden_layers"],
                color_by_layer=True,
                normalize_to_origin=True,  # Normalize to origin (0,0)
                warning_markers=warning_markers if warning_markers else None,
            )
            st.pyplot(fig, use_container_width=True)

        # Analysis and Object Table (moved from Analyzer)
        with st.expander("Analysis and Object Table", expanded=False):
            try:
                from wjp_analyser.web.api_client_wrapper import analyze_dxf as api_analyze_dxf
                import pandas as pd
                # Persist current doc to a temp path
                tmp_dir = Path("output") / "editor_analysis"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / f"editor_tmp_{int(time.time())}.dxf"
                save_dxf(doc, str(tmp_path))
                report = api_analyze_dxf(str(tmp_path))
                comps: List[Dict[str, Any]] = report.get("components", []) or []
                rows = []
                for c in comps:
                    rows.append({
                        "ID": c.get("id"),
                        "Layer": c.get("layer", "0"),
                        "Group": c.get("group", "Ungrouped"),
                        "Area_mm2": float(c.get("area", 0.0)),
                        "Perimeter_mm": float(c.get("perimeter", 0.0)),
                        "Selected": True,
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    # Normalize dtypes and defaults to avoid ambiguous truth errors
                    try:
                        df["ID"] = df["ID"].astype(str)
                        df["Layer"] = df["Layer"].astype(str)
                        df["Group"] = df["Group"].fillna("Ungrouped").astype(str)
                        df["Area_mm2"] = pd.to_numeric(df["Area_mm2"], errors="coerce").fillna(0.0)
                        df["Perimeter_mm"] = pd.to_numeric(df["Perimeter_mm"], errors="coerce").fillna(0.0)
                        if "Selected" not in df.columns:
                            df["Selected"] = True
                        df["Selected"] = df["Selected"].fillna(False).astype(bool)
                    except Exception:
                        pass
                    st.markdown("**Objects (editable selection & layer)**")

                    # Filters: Layer and Group
                    try:
                        layer_options = sorted(df["Layer"].astype(str).unique().tolist())
                        group_options = sorted(df["Group"].astype(str).unique().tolist())
                    except Exception:
                        layer_options, group_options = [], []

                    fc1, fc2, fc3 = st.columns([2, 2, 2])
                    with fc1:
                        layer_filter = st.multiselect("Filter: Layer", options=layer_options, default=layer_options)
                    with fc2:
                        group_filter = st.multiselect("Filter: Group", options=group_options, default=group_options)
                    with fc3:
                        st.caption("Use filters to limit the table view. Actions apply to filtered rows.")

                    # Apply filters to table view
                    try:
                        vdf = df[(df["Layer"].isin(layer_filter)) & (df["Group"].isin(group_filter))].copy()
                    except Exception:
                        vdf = df.copy()

                    # Quick selection actions and soft refresh
                    ac1, ac2, ac3 = st.columns([2, 2, 2])
                    with ac1:
                        if st.button("Select Current Filter"):
                            try:
                                table = st.session_state.get("editor_objects_df", df).copy()
                                table["ID"] = table["ID"].astype(str)
                                sel_ids = set(vdf["ID"].astype(str).tolist())
                                table.loc[table["ID"].isin(sel_ids), "Selected"] = True
                                st.session_state["editor_objects_df"] = table
                                st.success(f"Selected {len(sel_ids)} rows in current filter.")
                            except Exception as ex:
                                st.error(f"Could not select rows: {ex}")
                    with ac2:
                        if st.button("Select None"):
                            try:
                                table = st.session_state.get("editor_objects_df", df).copy()
                                table["ID"] = table["ID"].astype(str)
                                sel_ids = set(vdf["ID"].astype(str).tolist())
                                table.loc[table["ID"].isin(sel_ids), "Selected"] = False
                                st.session_state["editor_objects_df"] = table
                                st.success("Deselected current filter.")
                            except Exception as ex:
                                st.error(f"Could not deselect rows: {ex}")
                    with ac3:
                        if st.button("Reanalyze (soft refresh)"):
                            try:
                                # Re-run analysis and rebuild table, preserving user edits by ID
                                new_report = api_analyze_dxf(str(tmp_path))
                                new_comps = new_report.get("components", []) or []
                                new_rows = []
                                for c in new_comps:
                                    new_rows.append({
                                        "ID": str(c.get("id")),
                                        "Layer": str(c.get("layer", "0")),
                                        "Group": str(c.get("group", "Ungrouped")),
                                        "Area_mm2": float(c.get("area", 0.0)),
                                        "Perimeter_mm": float(c.get("perimeter", 0.0)),
                                        "Selected": True,
                                    })
                                if new_rows:
                                    new_df = pd.DataFrame(new_rows)
                                    # Merge user edits
                                    old = st.session_state.get("editor_objects_df", df).copy()
                                    old["ID"] = old["ID"].astype(str)
                                    new_df["ID"] = new_df["ID"].astype(str)
                                    merged = new_df.set_index("ID").combine_first(
                                        old.set_index("ID")
                                    ).reset_index()
                                    # Normalize dtypes
                                    merged["Selected"] = merged["Selected"].fillna(False).astype(bool)
                                    merged["Layer"] = merged["Layer"].astype(str)
                                    merged["Group"] = merged["Group"].fillna("Ungrouped").astype(str)
                                    st.session_state["editor_objects_df"] = merged
                                    st.success("Reanalysis complete and table refreshed (preserved user edits).")
                                else:
                                    st.info("Reanalysis produced no components.")
                            except Exception as ex:
                                st.error(f"Soft refresh failed: {ex}")

                    # Show editable table for filtered view
                    edited = st.data_editor(
                        vdf,
                        use_container_width=True,
                        num_rows="dynamic",
                    )
                    # Persist normalized, edited DataFrame
                    try:
                        ed = edited.copy()  # this is filtered view
                        ed["ID"] = ed["ID"].astype(str)
                        ed["Layer"] = ed["Layer"].astype(str)
                        ed["Group"] = ed["Group"].fillna("Ungrouped").astype(str)
                        if "Selected" not in ed.columns:
                            ed["Selected"] = True
                        ed["Selected"] = ed["Selected"].fillna(False).astype(bool)
                        # Merge edited filtered rows back into full table
                        full = st.session_state.get("editor_objects_df", df).copy()
                        full["ID"] = full["ID"].astype(str)
                        full_indexed = full.set_index("ID")
                        ed_indexed = ed.set_index("ID")
                        full_indexed.update(ed_indexed)
                        st.session_state["editor_objects_df"] = full_indexed.reset_index()
                    except Exception:
                        st.session_state["editor_objects_df"] = edited
                    # Apply edited layers back to DXF by handle (ID)
                    c_apply1, c_apply2 = st.columns([1, 1])
                    with c_apply1:
                        if st.button("Apply Layer Changes", help="Writes layer values for selected rows back to DXF"):
                            try:
                                table = st.session_state.get("editor_objects_df", edited).copy()
                                if table.empty:
                                    st.warning("No rows available to apply.")
                                else:
                                    table["Selected"] = table["Selected"].fillna(False).astype(bool)
                                    table["ID"] = table["ID"].astype(str)
                                    table["Layer"] = table["Layer"].astype(str)
                                    # Build handle -> entity map
                                    handle_to_entity = {}
                                    for e in entities:
                                        try:
                                            h = getattr(e.dxf, "handle", None)
                                            if h:
                                                handle_to_entity[str(h)] = e
                                        except Exception:
                                            continue
                                    changed = 0
                                    for _, row in table.iterrows():
                                        try:
                                            if not bool(row.get("Selected", False)):
                                                continue
                                            handle = str(row.get("ID"))
                                            target_layer = str(row.get("Layer", "0"))
                                            ent = handle_to_entity.get(handle)
                                            if ent is not None:
                                                ensure_layer(doc, target_layer, color=7)
                                                try:
                                                    ent.dxf.layer = target_layer
                                                    changed += 1
                                                except Exception:
                                                    pass
                                        except Exception:
                                            continue
                                    st.success(f"Applied layer changes to {changed} entities. Use Save DXF below to persist.")
                            except Exception as ex:
                                st.error(f"Failed to apply changes: {ex}")
                    with c_apply2:
                        if st.button("Select Handles in View", help="Highlights selected rows in the current plot"):
                            try:
                                table = st.session_state.get("editor_objects_df", edited).copy()
                                if table.empty:
                                    st.warning("No rows available to select.")
                                else:
                                    table["Selected"] = table["Selected"].fillna(False).astype(bool)
                                    table["ID"] = table["ID"].astype(str)
                                    sel_handles = [str(r["ID"]) for _, r in table.iterrows() if bool(r.get("Selected", False))]
                                    st.session_state["state"]["selected"] = sel_handles
                                    st.success(f"Selected {len(sel_handles)} entities in view.")
                            except Exception as ex:
                                st.error(f"Could not update selection: {ex}")
                else:
                    st.info("No components detected by analyzer.")
            except Exception as e:
                st.warning(f"Analysis unavailable: {e}")

        # Transforms and Draw tools intentionally removed per updated workflow.

        # Helper functions for AI analysis (defined before use)
        def _calculate_readiness_score(ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate readiness score based on analysis."""
            stats = ai_analysis.get("statistics", {})
            recs = ai_analysis.get("recommendations", {})
            
            score = 100.0
            total_objects = stats.get("total_objects", 1)
            
            # Penalties
            zero_area = stats.get("area_distribution", {}).get("zero", 0)
            tiny_area = stats.get("area_distribution", {}).get("tiny_lt1", 0)
            tiny_perim = stats.get("perimeter_distribution", {}).get("tiny_lt5", 0)
            
            # Deduct points
            if total_objects > 0:
                zero_pct = (zero_area / total_objects) * 100
                tiny_pct = (tiny_area / total_objects) * 100
                tiny_perim_pct = (tiny_perim / total_objects) * 100
                
                score -= min(30, zero_pct * 0.3)  # Max 30 points for zero area
                score -= min(25, tiny_pct * 0.25)  # Max 25 points for tiny area
                score -= min(20, tiny_perim_pct * 0.2)  # Max 20 points for tiny perimeter
            
            score = max(0, min(100, score))
            
            # Determine level
            if score >= 80:
                level = "excellent"
            elif score >= 60:
                level = "good"
            elif score >= 40:
                level = "fair"
            else:
                level = "poor"
            
            return {"score": score, "level": level}
        
        def _apply_fix_zero_area(doc, report):
            """Remove zero-area objects from DXF."""
            components = report.get("components", [])
            zero_area_handles = []
            for comp in components:
                if comp.get("area", 0) == 0:
                    handle = comp.get("id")
                    if handle:
                        zero_area_handles.append(handle)
            
            msp = doc.modelspace()
            removed = 0
            for entity in list(msp):
                try:
                    if entity.dxf.handle in zero_area_handles:
                        msp.delete_entity(entity)
                        removed += 1
                except Exception:
                    pass
            return removed
        
        def _apply_fix_tiny_objects(doc, report):
            """Remove tiny objects (< 1 mm²) from DXF."""
            components = report.get("components", [])
            tiny_handles = []
            for comp in components:
                area = comp.get("area", 0)
                if 0 < area < 1.0:
                    handle = comp.get("id")
                    if handle:
                        tiny_handles.append(handle)
            
            msp = doc.modelspace()
            removed = 0
            for entity in list(msp):
                try:
                    if entity.dxf.handle in tiny_handles:
                        msp.delete_entity(entity)
                        removed += 1
                except Exception:
                    pass
            return removed
        
        def _create_enhanced_csv_with_recommendations(
            base_csv_path: str,
            ai_analysis: Dict[str, Any],
            readiness_score: Dict[str, Any],
            recommendations_state: Dict[str, Any]
        ) -> str:
            """Create enhanced CSV with component data + recommendation selections."""
            import csv
            
            # Read base CSV
            components_df = pd.read_csv(base_csv_path)
            
            # Create enhanced CSV path
            base_path = Path(base_csv_path)
            enhanced_path = base_path.parent / f"enhanced_{base_path.name}"
            
            # Write enhanced CSV with multiple sections
            with open(enhanced_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Section 1: Summary and Readiness Score
                writer.writerow(["=== DXF ANALYSIS REPORT ==="])
                writer.writerow([])
                writer.writerow(["READINESS SCORE"])
                writer.writerow(["Score", f"{readiness_score['score']:.0f}/100"])
                writer.writerow(["Level", readiness_score['level'].upper()])
                writer.writerow([])
                
                stats = ai_analysis.get("statistics", {})
                writer.writerow(["SUMMARY STATISTICS"])
                writer.writerow(["Total Objects", stats.get("total_objects", 0)])
                writer.writerow(["Operable Objects", stats.get("operable_objects", 0)])
                writer.writerow(["Total Area (mm²)", f"{stats.get('total_area_mm2', 0):,.2f}"])
                writer.writerow(["Total Perimeter (m)", f"{stats.get('total_perimeter_mm', 0) / 1000:.2f}"])
                writer.writerow(["Waterjet Viability", ai_analysis.get("recommendations", {}).get("viability_score", "unknown").upper()])
                writer.writerow([])
                
                # Section 2: Recommendation Selections
                writer.writerow(["=== RECOMMENDATION SELECTIONS ==="])
                writer.writerow([])
                writer.writerow(["Type", "Status", "Count", "Message", "Action"])
                
                warnings_state = recommendations_state.get("warnings", {})
                for w_type, w_data in warnings_state.items():
                    enabled = "ENABLED" if w_data.get("enabled", True) else "DISABLED"
                    warning = w_data.get("data", {})
                    writer.writerow([
                        warning.get("type", w_type),
                        enabled,
                        warning.get("count", 0),
                        warning.get("message", ""),
                        warning.get("action", "")
                    ])
                
                info_state = recommendations_state.get("info", {})
                for i_type, i_data in info_state.items():
                    enabled = "ENABLED" if i_data.get("enabled", True) else "DISABLED"
                    info_item = i_data.get("data", {})
                    writer.writerow([
                        info_item.get("type", i_type),
                        enabled,
                        "-",
                        info_item.get("message", ""),
                        info_item.get("action", "")
                    ])
                
                writer.writerow([])
                writer.writerow(["=== COMPONENT DATA ==="])
                writer.writerow([])
                
                # Section 3: Component Data
                # Write headers
                writer.writerow(list(components_df.columns))
                # Write component rows
                for _, row in components_df.iterrows():
                    writer.writerow(row.tolist())
            
            return str(enhanced_path)

        # AI Analysis and Recommendations with Preview and Readiness Score
        st.markdown("## 🤖 AI Analysis & Recommendations")
        with st.expander("Run AI Analysis and Get Recommendations", expanded=True):
            try:
                from wjp_analyser.web.api_client_wrapper import (
                    analyze_dxf as api_analyze_dxf,
                    export_components_csv,
                    analyze_csv,
                )
                from wjp_analyser.nesting.dxf_extractor import extract_polygons
                import pandas as pd
                import matplotlib.pyplot as plt
                from datetime import datetime
                
                # Persist current doc for analysis
                analysis_dir = Path("output") / "editor_ai_analysis"
                analysis_dir.mkdir(parents=True, exist_ok=True)
                analysis_dxf_path = analysis_dir / f"editor_analysis_{int(time.time())}.dxf"
                save_dxf(doc, str(analysis_dxf_path))
                
                if st.button("🔍 Run AI Analysis", type="primary", key="btn_run_editor_ai"):
                    with st.spinner("Running AI analysis..."):
                        # Run analysis via API wrapper
                        report = api_analyze_dxf(str(analysis_dxf_path))
                        
                        # Export CSV for AI analysis
                        csv_dir = analysis_dir / "csv_exports"
                        csv_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
                        csv_path = str(csv_dir / f"{timestamp}_editor_analysis.csv")
                        export_components_csv(report, csv_path)
                        
                        # Run AI analysis via API wrapper
                        ai_analysis = analyze_csv(csv_path, report=report)
                        
                        # Store in session state (using different keys to avoid widget conflicts)
                        st.session_state["_editor_ai_analysis"] = ai_analysis
                        st.session_state["_editor_analysis_report"] = report
                        st.session_state["_editor_analysis_csv"] = csv_path
                        st.success("AI Analysis completed!")
                
                # Display results if available
                if st.session_state.get("_editor_ai_analysis") and st.session_state.get("_editor_analysis_report"):
                    ai_analysis = st.session_state.get("_editor_ai_analysis")
                    report = st.session_state.get("_editor_analysis_report")
                    
                    if ai_analysis.get("success"):
                        stats = ai_analysis.get("statistics", {})
                        recs = ai_analysis.get("recommendations", {})
                        viability = recs.get("viability_score", "unknown")
                        
                        # Readiness Score Display
                        st.markdown("### 📊 Readiness Score")
                        readiness_score = _calculate_readiness_score(ai_analysis)
                        score_color = {
                            "excellent": "🟢",
                            "good": "🟡", 
                            "fair": "🟠",
                            "poor": "🔴"
                        }.get(readiness_score["level"], "⚪")
                        
                        col1, col2, col3 = st.columns([2, 2, 2])
                        with col1:
                            st.metric("Readiness Score", f"{readiness_score['score']:.0f}/100", f"{score_color} {readiness_score['level'].upper()}")
                        with col2:
                            st.metric("Operable Objects", stats.get("operable_objects", 0), f"of {stats.get('total_objects', 0)}")
                        with col3:
                            st.metric("Waterjet Viability", viability.upper(), f"{'✅ Ready' if viability == 'good' else '⚠️ Review' if viability == 'fair' else '❌ Issues'}")
                        
                        # Preview with warning markers
                        st.markdown("### 📊 Preview")
                        preview_tab1, preview_tab2 = st.tabs(["Current State", "Statistics"])
                        
                        with preview_tab1:
                            try:
                                polys = extract_polygons(str(analysis_dxf_path))
                                if polys:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    all_points = []
                                    
                                    # Get enabled warnings to show on preview
                                    enabled_warnings = {}
                                    if "editor_recommendations" in st.session_state:
                                        for w_type, w_data in st.session_state["editor_recommendations"].get("warnings", {}).items():
                                            if w_data.get("enabled", True):
                                                enabled_warnings[w_type] = w_data.get("data", {})
                                    
                                    # Map warnings to components by matching criteria
                                    components = report.get("components", [])
                                    warning_locations = {}  # component_id -> list of warning types
                                    
                                    for comp in components:
                                        comp_id = comp.get("id")
                                        comp_warnings = []
                                        
                                        # Check for zero area
                                        if comp.get("area", 0) == 0 and "zero_area" in enabled_warnings:
                                            comp_warnings.append("zero_area")
                                        
                                        # Check for tiny objects
                                        area = comp.get("area", 0)
                                        if 0 < area < 1.0 and "too_many_tiny" in enabled_warnings:
                                            comp_warnings.append("too_many_tiny")
                                        
                                        # Check for other warning conditions as needed
                                        if comp_warnings:
                                            warning_locations[comp_id] = comp_warnings
                                    
                                    for p in polys:
                                        pts = p.get("points") or []
                                        if len(pts) >= 3:
                                            all_points.extend(pts)
                                    
                                    if all_points:
                                        xs = [p[0] for p in all_points]
                                        ys = [p[1] for p in all_points]
                                        minx, maxx = min(xs), max(xs)
                                        miny, maxy = min(ys), max(ys)
                                        
                                        # Normalize to origin and render polygons
                                        for idx, p in enumerate(polys):
                                            pts = p.get("points") or []
                                            if len(pts) >= 3:
                                                norm_pts = [(x - minx, y - miny) for (x, y) in pts]
                                                xs_norm = [x for x, y in norm_pts]
                                                ys_norm = [y for x, y in norm_pts]
                                                
                                                # Check if this polygon has warnings
                                                has_warning = False
                                                for comp in components:
                                                    # Try to match polygon to component (simple matching by index or area)
                                                    if idx < len(components):
                                                        comp_id = components[idx].get("id")
                                                        if comp_id in warning_locations:
                                                            has_warning = True
                                                            break
                                                
                                                if has_warning:
                                                    # Highlight with red border
                                                    ax.fill(xs_norm, ys_norm, alpha=0.2, edgecolor='#F44336', 
                                                           linewidth=2.0, facecolor='#FFEBEE')
                                                else:
                                                    ax.fill(xs_norm, ys_norm, alpha=0.3, edgecolor='k', linewidth=0.5)
                                        
                                        # Add warning markers at problem locations
                                        if warning_locations:
                                            for comp in components:
                                                comp_id = comp.get("id")
                                                if comp_id in warning_locations:
                                                    # Get component center for marker placement
                                                    comp_points = comp.get("points", [])
                                                    if comp_points and len(comp_points) > 0:
                                                        # Calculate centroid
                                                        center_x = sum(p[0] for p in comp_points) / len(comp_points)
                                                        center_y = sum(p[1] for p in comp_points) / len(comp_points)
                                                        # Normalize
                                                        norm_x = center_x - minx
                                                        norm_y = center_y - miny
                                                        # Draw warning marker
                                                        ax.plot(norm_x, norm_y, 'ro', markersize=12, 
                                                               markeredgecolor='#C62828', markeredgewidth=2, 
                                                               markerfacecolor='#F44336', alpha=0.8, zorder=10)
                                                        # Overlay warning glyph
                                                        ax.text(norm_x, norm_y, '⚠', color='white', fontsize=10,
                                                                ha='center', va='center', zorder=11)
                                                        
                                                        # Add annotation
                                                        warning_types = warning_locations[comp_id]
                                                        warning_text = "\n".join([w.replace("_", " ").title() for w in warning_types])
                                                        ax.annotate(warning_text, (norm_x, norm_y), 
                                                                   xytext=(10, 10), textcoords='offset points',
                                                                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', 
                                                                            edgecolor='#F44336', linewidth=2),
                                                                   fontsize=8, color='#C62828', zorder=12)
                                        
                                        ax.set_aspect('equal')
                                        ax.grid(True, alpha=0.3)
                                        title = "DXF Preview (Normalized to Origin)"
                                        if warning_locations:
                                            title += f" - {len(warning_locations)} Warning(s) Shown"
                                        ax.set_title(title, fontsize=14, fontweight='bold')
                                        ax.set_xlabel('X (mm)', fontsize=12)
                                        ax.set_ylabel('Y (mm)', fontsize=12)
                                        width = maxx - minx
                                        height = maxy - miny
                                        stats_text = f"Dimensions: {width:.2f} × {height:.2f} mm\nObjects: {len(polys)}"
                                        if warning_locations:
                                            stats_text += f"\n⚠️ Warnings: {len(warning_locations)}"
                                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                                                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
                                        st.pyplot(fig, clear_figure=True)
                                        plt.close(fig)
                            except Exception as e:
                                st.warning(f"Preview unavailable: {e}")
                        
                        with preview_tab2:
                            st.markdown("**Key Statistics**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Total Objects:** {stats.get('total_objects', 0)}")
                                st.write(f"**Selected:** {stats.get('selected_count', 0)}")
                                st.write(f"**Operable:** {stats.get('operable_objects', 0)}")
                            with col2:
                                st.write(f"**Total Area:** {stats.get('total_area_mm2', 0):,.2f} mm²")
                                st.write(f"**Operable Area:** {stats.get('operable_area_mm2', 0):,.2f} mm²")
                                st.write(f"**Total Perimeter:** {stats.get('total_perimeter_mm', 0) / 1000:.2f} m")
                            with col3:
                                area_dist = stats.get("area_distribution", {})
                                st.write(f"**Zero Area:** {area_dist.get('zero', 0)}")
                                st.write(f"**Tiny (<1 mm²):** {area_dist.get('tiny_lt1', 0)}")
                                st.write(f"**Large (≥100 mm²):** {area_dist.get('large_ge100', 0)}")
                        
                        # Recommendations with Editable Actions
                        st.markdown("### ⚠️ Warnings & Recommendations (Editable)")
                        warnings = recs.get("warnings", [])
                        info_items = recs.get("info", [])
                        
                        # Create editable recommendations
                        if "editor_recommendations" not in st.session_state:
                            st.session_state["editor_recommendations"] = {
                                "warnings": {w.get("type"): {"enabled": True, "data": w} for w in warnings},
                                "info": {i.get("type"): {"enabled": True, "data": i} for i in info_items}
                            }
                        
                        # Display warnings with checkboxes to enable/disable fixes
                        # Style warnings in red
                        st.markdown("""
                        <style>
                        .warning-box {
                            background-color: #FFEBEE;
                            border-left: 4px solid #F44336;
                            padding: 10px;
                            margin: 10px 0;
                            border-radius: 4px;
                        }
                        .warning-text {
                            color: #C62828;
                            font-weight: 500;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        for warning in warnings:
                            w_type = warning.get("type", "unknown")
                            rec_state = st.session_state["editor_recommendations"]["warnings"].get(w_type, {"enabled": True, "data": warning})
                            warning_data = rec_state.get("data", warning)
                            
                            # Display warning in red-styled container
                            with st.container():
                                st.markdown(f'<div class="warning-box">', unsafe_allow_html=True)
                                
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    # Warning type and enable/disable
                                    warning_type_label = f"**{warning_data.get('type', 'Warning')}**"
                                    enabled = st.checkbox(
                                        warning_type_label,
                                        value=rec_state.get("enabled", True),
                                        key=f"rec_warn_enable_{w_type}"
                                    )
                                    
                                    # Editable message field
                                    current_message = warning_data.get("message", "")
                                    edited_message = st.text_input(
                                        "Warning Message:",
                                        value=current_message,
                                        key=f"rec_warn_msg_{w_type}"
                                    )
                                    
                                    # Editable action field
                                    current_action = warning_data.get("action", "")
                                    edited_action = st.text_input(
                                        "Action/Recommendation:",
                                        value=current_action,
                                        key=f"rec_warn_action_{w_type}"
                                    )
                                    
                                    # Update warning data if edited
                                    if edited_message != current_message or edited_action != current_action:
                                        warning_data = warning_data.copy()
                                        warning_data["message"] = edited_message
                                        warning_data["action"] = edited_action
                                    
                                    # Auto-fix buttons
                                    if enabled:
                                        fix_col1, fix_col2, fix_col3 = st.columns([1, 1, 2])
                                        with fix_col1:
                                            if w_type == "zero_area" and st.button("🔧 Remove Zero-Area", key=f"fix_zero_{w_type}"):
                                                removed = _apply_fix_zero_area(doc, report)
                                                st.success(f"Zero-area objects removed! ({removed} objects)")
                                                st.rerun()
                                        with fix_col2:
                                            if w_type == "too_many_tiny" and st.button("🔧 Filter Tiny", key=f"fix_tiny_{w_type}"):
                                                removed = _apply_fix_tiny_objects(doc, report)
                                                st.success(f"Tiny objects filtered! ({removed} objects)")
                                                st.rerun()
                                with col2:
                                    if enabled:
                                        st.markdown(f"⚠️ **{warning_data.get('count', 0)}**")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Store updated warning data
                                st.session_state["editor_recommendations"]["warnings"][w_type] = {
                                    "enabled": enabled,
                                    "data": warning_data
                                }
                                st.divider()

                        # Sync edited warnings back into AI analysis recommendations for downstream uses (preview/export)
                        try:
                            updated_list = []
                            warn_state = st.session_state.get("editor_recommendations", {}).get("warnings", {})
                            for w in warnings:
                                key = w.get("type", "unknown")
                                if key in warn_state and isinstance(warn_state[key], dict):
                                    updated_list.append(dict(warn_state[key].get("data", w)))
                                else:
                                    updated_list.append(w)
                            # Write back to local structures and session for persistence
                            recs["warnings"] = updated_list
                            ai_analysis["recommendations"]["warnings"] = updated_list
                            st.session_state["_editor_ai_analysis"] = ai_analysis
                        except Exception:
                            pass
                        
                        # Display info items
                        for info_item in info_items:
                            i_type = info_item.get("type", "unknown")
                            rec_state = st.session_state["editor_recommendations"]["info"].get(i_type, {"enabled": True, "data": info_item})
                            
                            enabled = st.checkbox(
                                f"**{info_item.get('type', 'Info')}**: {info_item.get('message', '')}",
                                value=rec_state["enabled"],
                                key=f"rec_info_{i_type}"
                            )
                            st.caption(f"*Action*: {info_item.get('action', '')}")
                            
                            st.session_state["editor_recommendations"]["info"][i_type] = {
                                "enabled": enabled,
                                "data": info_item
                            }
                            st.divider()
                        
                        # Apply All Enabled Fixes Button
                        enabled_count = sum(1 for w in st.session_state["editor_recommendations"]["warnings"].values() if w["enabled"])
                        if enabled_count > 0:
                            if st.button("🔧 Apply All Enabled Fixes", type="primary", key="apply_all_fixes"):
                                with st.spinner("Applying fixes..."):
                                    fixes_applied = []
                                    for w_type, w_data in st.session_state["editor_recommendations"]["warnings"].items():
                                        if w_data["enabled"]:
                                            if w_type == "zero_area":
                                                removed = _apply_fix_zero_area(doc, report)
                                                fixes_applied.append(f"Removed {removed} zero-area objects")
                                            elif w_type == "too_many_tiny":
                                                removed = _apply_fix_tiny_objects(doc, report)
                                                fixes_applied.append(f"Filtered {removed} tiny objects")
                                    if fixes_applied:
                                        st.success(f"Applied fixes: {', '.join(fixes_applied)}. Save DXF to persist changes.")
                                        st.rerun()
                        
                        # Download CSV (with recommendation selections)
                        base_csv_path = st.session_state.get("_editor_analysis_csv")
                        if base_csv_path and os.path.exists(base_csv_path):
                            # Create enhanced CSV with recommendation selections
                            enhanced_csv_path = _create_enhanced_csv_with_recommendations(
                                base_csv_path,
                                ai_analysis,
                                readiness_score,
                                st.session_state.get("editor_recommendations", {})
                            )
                            if enhanced_csv_path and os.path.exists(enhanced_csv_path):
                                with open(enhanced_csv_path, "rb") as fh:
                                    st.download_button(
                                        "📥 Download Analysis CSV (with Recommendations)",
                                        data=fh.read(),
                                        file_name=os.path.basename(enhanced_csv_path),
                                        mime="text/csv"
                                    )
                            else:
                                # Fallback to original CSV
                                with open(base_csv_path, "rb") as fh:
                                    st.download_button(
                                        "📥 Download Analysis CSV",
                                        data=fh.read(),
                                        file_name=os.path.basename(base_csv_path),
                                        mime="text/csv"
                                    )
                    else:
                        st.error(f"AI Analysis failed: {ai_analysis.get('error', 'Unknown error')}")
            except Exception as e:
                from wjp_analyser.web.components import render_error, create_dxf_error_actions
                render_error(
                    e,
                    user_message="AI Analysis unavailable. Please check your DXF file and try again.",
                    actions=create_dxf_error_actions(str(analysis_dxf_path)),
                    show_traceback=False,
                )

        # Measure / Validate
        m1, m2, m3 = st.columns(3)
        with m1:
            a = st.text_input("Point A (x,y)", value="0,0")
            b = st.text_input("Point B (x,y)", value="10,0")
            if st.button("Distance A→B"):
                try:
                    ax, ay = [float(v) for v in a.split(",")]
                    bx, by = [float(v) for v in b.split(",")]
                    st.info(f"Distance: {distance((ax,ay),(bx,by)):.3f} mm")
                except Exception:
                    st.error("Invalid input")
        with m2:
            kerf = st.number_input("Kerf preview (mm)", value=1.1)
            st.write(f"Kerf value: {kerf_preview_value(kerf)} mm")
        with m3:
            if st.button("Check Min Radius (selected)"):
                issues = []
                for e in entities:
                    if e.dxf.handle in st.session_state["state"]["selected"]:
                        ok, meta = check_min_radius(e, min_r=2.0)
                        if not ok:
                            issues.append((e.dxf.handle, meta))
                if issues:
                    st.warning(f"Violations: {issues}")
                else:
                    st.success("No radius violations (or not applicable).")

        # Save / Export
        st.subheader("Save / Export")
        output_dir = Path("output") / "dxf_editor"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = st.text_input("Save as", value=str(output_dir / "edited_output_v2.dxf"))
        if st.button("💾 Save DXF"):
            save_dxf(doc, out_path)
            with open(out_path, "rb") as f:
                st.download_button("Download Edited DXF", data=f, file_name=os.path.basename(out_path))

        # Waterjet viability pipeline
        with st.expander("Waterjet Clean-up (auto-scale, validate, export)", expanded=False):
            st.caption("Scale geometry to default frame, detect non-operable objects (tiny segments, tight corners, spacing), and export a clean DXF.")
            c1, c2, c3 = st.columns(3)
            with c1:
                frame_w = st.number_input("Target width (mm)", value=1000.0, min_value=10.0, step=10.0)
            with c2:
                frame_h = st.number_input("Target height (mm)", value=1000.0, min_value=10.0, step=10.0)
            with c3:
                autoscale = st.checkbox("Scale to target size", value=True)
            origin_zero = st.checkbox("Normalize origin to (0,0)", value=True)

            v1, v2, v3 = st.columns(3)
            with v1:
                min_segment = st.number_input("Min segment (mm)", value=1.0, min_value=0.01, step=0.1)
            with v2:
                min_corner_radius = st.number_input("Min corner radius (mm)", value=2.0, min_value=0.0, step=0.1)
            with v3:
                min_spacing = st.number_input("Min spacing (mm)", value=3.0, min_value=0.0, step=0.1)

            def _analyze_polygons(dxf_doc) -> tuple[list[dict], dict]:
                from wjp_analyser.nesting.dxf_extractor import extract_polygons
                temp_dir = Path("output") / "editor_clean"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / "clean_src.dxf"
                save_dxf(dxf_doc, str(temp_path))
                polys = extract_polygons(str(temp_path))
                return polys, {"path": str(temp_path)}

            def _scale_points(pts: list[tuple[float, float]], target_w: float, target_h: float, to_zero: bool = True) -> list[tuple[float, float]]:
                if not pts:
                    return pts
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                w = max(1e-9, maxx - minx); h = max(1e-9, maxy - miny)
                sx = target_w / w; sy = target_h / h; s = min(sx, sy)
                if to_zero:
                    # Shift so min corner is (0,0)
                    return [((x - minx) * s, (y - miny) * s) for (x, y) in pts]
                else:
                    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                    return [((x - cx) * s + target_w / 2.0, (y - cy) * s + target_h / 2.0) for (x, y) in pts]

            def _min_segment_len(pts: list[tuple[float, float]]) -> float:
                import math
                if len(pts) < 2:
                    return 0.0
                m = float("inf")
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    d = math.hypot(x2 - x1, y2 - y1)
                    if d < m:
                        m = d
                return m if m != float("inf") else 0.0

            def _min_corner_radius_approx(pts: list[tuple[float, float]]) -> float:
                # Approximate by circumradius of consecutive triplets
                import math
                if len(pts) < 3:
                    return 0.0
                def circumradius(a, b, c):
                    ax, ay = a; bx, by = b; cx, cy = c
                    ab = math.hypot(bx - ax, by - ay)
                    bc = math.hypot(cx - bx, cy - by)
                    ca = math.hypot(ax - cx, ay - cy)
                    s = (ab + bc + ca) / 2.0
                    area2 = max(1e-12, s * (s - ab) * (s - bc) * (s - ca))
                    area = math.sqrt(area2)
                    R = (ab * bc * ca) / (4.0 * area) if area > 0 else float("inf")
                    return R
                rmin = float("inf")
                for i in range(len(pts)):
                    a = pts[i - 1]
                    b = pts[i]
                    c = pts[(i + 1) % len(pts)]
                    r = circumradius(a, b, c)
                    if r < rmin:
                        rmin = r
                return 0.0 if rmin == float("inf") else rmin

            def _min_spacing_between(polys_pts: list[list[tuple[float, float]]]) -> float:
                try:
                    from shapely.geometry import Polygon
                    from shapely.ops import unary_union
                    if len(polys_pts) < 2:
                        return float("inf")
                    geoms = [Polygon(pts) for pts in polys_pts if len(pts) >= 3]
                    m = float("inf")
                    for i in range(len(geoms)):
                        for j in range(i + 1, len(geoms)):
                            try:
                                d = geoms[i].distance(geoms[j])
                                if d < m:
                                    m = d
                            except Exception:
                                continue
                    return m
                except Exception:
                    return float("inf")

            if st.button("Analyze for Waterjet", type="primary"):
                polys, meta = _analyze_polygons(doc)
                if not polys:
                    st.error("No polygons extracted.")
                else:
                    # Prepare points and optionally scale
                    pts_list = [p.get("points") or [] for p in polys]
                    if autoscale:
                        # Compute combined bbox and scale all together
                        # Flatten to get bbox
                        all_pts = [pt for pts in pts_list for pt in pts]
                        if all_pts:
                            # scale each set with shared factor
                            xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
                            minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
                            w = max(1e-9, maxx - minx); h = max(1e-9, maxy - miny)
                            s = min(frame_w / w, frame_h / h)
                            if origin_zero:
                                pts_list = [[((x - minx) * s, (y - miny) * s) for (x, y) in pts] for pts in pts_list]
                            else:
                                cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                                pts_list = [[((x - cx) * s + frame_w / 2.0, (y - cy) * s + frame_h / 2.0) for (x, y) in pts] for pts in pts_list]

                    # Validate each polygon
                    invalid_idx = []
                    for idx, pts in enumerate(pts_list):
                        if _min_segment_len(pts) < min_segment:
                            invalid_idx.append(idx); continue
                        if min_corner_radius > 0 and _min_corner_radius_approx(pts) < min_corner_radius:
                            invalid_idx.append(idx); continue
                    spacing = _min_spacing_between([p for i, p in enumerate(pts_list) if i not in invalid_idx])
                    if spacing < min_spacing:
                        # Mark smallest shapes until spacing improves (greedy)
                        sizes = [(i, len(pts_list[i])) for i in range(len(pts_list)) if i not in invalid_idx]
                        sizes.sort(key=lambda t: t[1])
                        for i, _n in sizes:
                            invalid_idx.append(i)
                            spacing = _min_spacing_between([p for j, p in enumerate(pts_list) if j not in invalid_idx])
                            if spacing >= min_spacing:
                                break

                    st.info(f"Detected {len(pts_list)} objects. Marked non-operable: {len(invalid_idx)}. Spacing after removal: {spacing if spacing != float('inf') else 0:.2f} mm")

                    # Store for export
                    st.session_state["wjp_clean_pts_list"] = pts_list
                    st.session_state["wjp_clean_invalid"] = set(invalid_idx)

            if st.session_state.get("wjp_clean_pts_list"):
                pts_list = st.session_state["wjp_clean_pts_list"]
                invalid_idx = st.session_state.get("wjp_clean_invalid", set())
                if st.button("Export Clean DXF", type="primary"):
                    try:
                        from wjp_analyser.web.api_client_wrapper import write_layered_dxf_from_report
                        from wjp_analyser.services.layered_dxf_service import write_layered_dxf_from_components
                        
                        # Convert pts_list to component format for service
                        components = []
                        for i, pts in enumerate(pts_list):
                            if i in invalid_idx:
                                continue
                            if len(pts) >= 2:
                                # Convert points to [x, y] format
                                points = [[float(p[0]), float(p[1])] for p in pts]
                                components.append({
                                    "points": points,
                                    "layer": "0",  # Default layer for clean export
                                    "id": f"clean_{i}"
                                })
                        
                        if components:
                            clean_path = output_dir / "clean_waterjet_ready.dxf"
                            write_layered_dxf_from_components(components, str(clean_path))
                            with open(clean_path, "rb") as f:
                                st.download_button("Download Clean DXF", data=f.read(), file_name=clean_path.name)
                        else:
                            st.warning("No valid components to export")
                    except Exception as ex:
                        st.error(f"Clean export failed: {ex}")
                        import traceback
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()


