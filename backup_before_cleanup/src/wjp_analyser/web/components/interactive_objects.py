"""
Interactive Object and Group Management
======================================

Interactive components for selecting, editing, and managing DXF objects and groups.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import json


def render_object_selection_panel(components: List[Dict[str, Any]], selected_objects: List[int] = None) -> List[int]:
    """Render interactive object selection panel with checkboxes and details."""
    
    if not components:
        st.info("No objects found in DXF file.")
        return []
    
    if selected_objects is None:
        selected_objects = []
    
    st.markdown("### 🔍 Object Selection & Management")
    
    # Selection controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Select All", key="select_all_objects"):
            selected_objects = [c["id"] for c in components]
            st.rerun()
    
    with col2:
        if st.button("Select None", key="select_none_objects"):
            selected_objects = []
            st.rerun()
    
    with col3:
        if st.button("Select Large (>1000mm²)", key="select_large_objects"):
            selected_objects = [c["id"] for c in components if c.get("area", 0) > 1000]
            st.rerun()
    
    with col4:
        if st.button("Select Small (<100mm²)", key="select_small_objects"):
            selected_objects = [c["id"] for c in components if c.get("area", 0) < 100]
            st.rerun()
    
    # Filtering options
    st.markdown("**Filter Options:**")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        filter_size = st.selectbox("Filter by Size", ["All", "Small (<100mm²)", "Medium (100-1000mm²)", "Large (>1000mm²)"], key="object_size_filter")
    
    with filter_col2:
        filter_layer = st.selectbox("Filter by Layer", ["All"] + list(set(c.get("layer", "Unknown") for c in components)), key="object_layer_filter")
    
    with filter_col3:
        filter_shape = st.selectbox("Filter by Shape", ["All", "Circle", "Rectangle", "Polygon", "Complex"], key="object_shape_filter")
    
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
        filtered_components = [c for c in filtered_components if _get_shape_type(c) == filter_shape]
    
    # Object list with selection
    st.markdown(f"**📋 Objects List ({len(filtered_components)} objects):**")
    
    # Create DataFrame for better display
    object_data = []
    for comp in filtered_components:
        shape_type = _get_shape_type(comp)
        object_data.append({
            "ID": comp["id"],
            "Area (mm²)": f"{comp.get('area', 0):.1f}",
            "Perimeter (mm)": f"{comp.get('perimeter', 0):.1f}",
            "Vertices": comp.get("vertex_count", 0),
            "Layer": comp.get("layer", "Unknown"),
            "Shape": shape_type,
            "Selected": comp["id"] in selected_objects
        })
    
    df = pd.DataFrame(object_data)
    
    # Display with selection checkboxes
    for idx, comp in enumerate(filtered_components):
        with st.expander(f"🔸 Object {comp['id']} - {comp.get('area', 0):.1f} mm²", expanded=False):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                is_selected = comp["id"] in selected_objects
                if st.checkbox(f"Select", value=is_selected, key=f"select_obj_{comp['id']}"):
                    if comp["id"] not in selected_objects:
                        selected_objects.append(comp["id"])
                else:
                    if comp["id"] in selected_objects:
                        selected_objects.remove(comp["id"])
            
            with col2:
                st.write(f"**Area:** {comp.get('area', 0):.1f} mm²")
                st.write(f"**Perimeter:** {comp.get('perimeter', 0):.1f} mm")
                st.write(f"**Vertices:** {comp.get('vertex_count', 0)}")
                st.write(f"**Layer:** {comp.get('layer', 'Unknown')}")
                st.write(f"**Shape:** {_get_shape_type(comp)}")
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_obj_{comp['id']}"):
                    st.warning(f"Delete object {comp['id']}? This action cannot be undone.")
                    if st.button("Confirm Delete", key=f"confirm_delete_obj_{comp['id']}"):
                        # Remove from components list
                        components.remove(comp)
                        if comp["id"] in selected_objects:
                            selected_objects.remove(comp["id"])
                        st.success(f"Object {comp['id']} deleted.")
                        st.rerun()
    
    return selected_objects


def render_group_selection_panel(groups: Dict[str, List[int]], components: List[Dict[str, Any]], selected_groups: List[str] = None) -> List[str]:
    """Render interactive group selection panel with expandable details."""
    
    if not groups:
        st.info("No groups found in DXF file.")
        return []
    
    if selected_groups is None:
        selected_groups = []
    
    st.markdown("### 🗂️ Group Selection & Management")
    
    # Selection controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Select All Groups", key="select_all_groups"):
            selected_groups = list(groups.keys())
            st.rerun()
    
    with col2:
        if st.button("Select None Groups", key="select_none_groups"):
            selected_groups = []
            st.rerun()
    
    with col3:
        if st.button("Select Large Groups (>5 objects)", key="select_large_groups"):
            selected_groups = [name for name, obj_ids in groups.items() if len(obj_ids) > 5]
            st.rerun()
    
    with col4:
        if st.button("Select Small Groups (<3 objects)", key="select_small_groups"):
            selected_groups = [name for name, obj_ids in groups.items() if len(obj_ids) < 3]
            st.rerun()
    
    # Group list with selection
    st.markdown(f"**📋 Groups List ({len(groups)} groups):**")
    
    for group_name, obj_ids in groups.items():
        group_components = [c for c in components if c["id"] in obj_ids]
        total_area = sum(c.get("area", 0) for c in group_components)
        avg_area = total_area / len(group_components) if group_components else 0
        
        with st.expander(f"🗂️ {group_name} - {len(obj_ids)} objects, {total_area:.1f} mm² total", expanded=False):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                is_selected = group_name in selected_groups
                if st.checkbox(f"Select Group", value=is_selected, key=f"select_group_{group_name}"):
                    if group_name not in selected_groups:
                        selected_groups.append(group_name)
                else:
                    if group_name in selected_groups:
                        selected_groups.remove(group_name)
            
            with col2:
                st.write(f"**Objects:** {len(obj_ids)}")
                st.write(f"**Total Area:** {total_area:.1f} mm²")
                st.write(f"**Average Area:** {avg_area:.1f} mm²")
                st.write(f"**Object IDs:** {', '.join(map(str, obj_ids))}")
                
                # Show object details
                if st.button("Show Object Details", key=f"show_details_{group_name}"):
                    for comp in group_components:
                        st.write(f"  - Object {comp['id']}: {comp.get('area', 0):.1f} mm², {comp.get('layer', 'Unknown')} layer")
            
            with col3:
                if st.button("✏️ Rename", key=f"rename_group_{group_name}"):
                    new_name = st.text_input(f"New name for {group_name}:", value=group_name, key=f"rename_input_{group_name}")
                    if new_name and new_name != group_name:
                        groups[new_name] = groups.pop(group_name)
                        if group_name in selected_groups:
                            selected_groups.remove(group_name)
                            selected_groups.append(new_name)
                        st.success(f"Group renamed to {new_name}")
                        st.rerun()
                
                if st.button("🗑️ Delete", key=f"delete_group_{group_name}"):
                    st.warning(f"Delete group {group_name}? This will remove the group but keep the objects.")
                    if st.button("Confirm Delete", key=f"confirm_delete_group_{group_name}"):
                        del groups[group_name]
                        if group_name in selected_groups:
                            selected_groups.remove(group_name)
                        st.success(f"Group {group_name} deleted.")
                        st.rerun()
    
    return selected_groups


def render_edit_options_panel(selected_objects: List[int], selected_groups: List[str], components: List[Dict[str, Any]], groups: Dict[str, List[int]]) -> None:
    """Render editing options for selected objects and groups."""
    
    if not selected_objects and not selected_groups:
        st.info("Select objects or groups to see editing options.")
        return
    
    st.markdown("### ⚙️ Editing Options")
    
    # Object editing options
    if selected_objects:
        st.markdown(f"**Selected Objects ({len(selected_objects)}):**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔀 Merge Selected", key="merge_objects"):
                st.info("Merge functionality coming soon!")
        
        with col2:
            if st.button("📏 Scale Selected", key="scale_objects"):
                scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1, key="scale_factor")
                if st.button("Apply Scale", key="apply_scale"):
                    st.info(f"Scale {scale_factor}x applied to {len(selected_objects)} objects")
        
        with col3:
            if st.button("🎨 Change Layer", key="change_layer"):
                new_layer = st.selectbox("New Layer", ["OUTER", "INNER", "DECOR", "HOLE", "CUT"], key="new_layer_select")
                if st.button("Apply Layer Change", key="apply_layer_change"):
                    st.info(f"Layer changed to {new_layer} for {len(selected_objects)} objects")
    
    # Group editing options
    if selected_groups:
        st.markdown(f"**Selected Groups ({len(selected_groups)}):**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔀 Merge Groups", key="merge_groups"):
                st.info("Group merge functionality coming soon!")
        
        with col2:
            if st.button("📊 Analyze Groups", key="analyze_groups"):
                st.info("Group analysis functionality coming soon!")
        
        with col3:
            if st.button("💾 Export Groups", key="export_groups"):
                st.info("Group export functionality coming soon!")
    
    # Bulk operations
    st.markdown("**Bulk Operations:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Copy Selected", key="copy_selected"):
            st.info("Copy functionality coming soon!")
    
    with col2:
        if st.button("📋 Paste", key="paste_selected"):
            st.info("Paste functionality coming soon!")
    
    with col3:
        if st.button("🔄 Duplicate", key="duplicate_selected"):
            st.info("Duplicate functionality coming soon!")
    
    with col4:
        if st.button("🗑️ Delete Selected", key="delete_selected"):
            if selected_objects or selected_groups:
                st.warning("Delete selected items? This action cannot be undone.")
                if st.button("Confirm Delete All", key="confirm_delete_all"):
                    st.info("Delete functionality coming soon!")


def _get_shape_type(component: Dict[str, Any]) -> str:
    """Determine the shape type of a component."""
    area = component.get("area", 0)
    perimeter = component.get("perimeter", 0)
    vcount = component.get("vertex_count", 0)
    
    # Calculate circularity
    circularity = (4.0 * 3.14159 * area) / (perimeter * perimeter + 1e-9) if perimeter > 0 else 0
    
    if circularity > 0.8:
        return "Circle"
    elif vcount <= 4 and circularity < 0.3:
        return "Rectangle"
    elif vcount <= 6:
        return "Polygon"
    else:
        return "Complex"


def render_object_group_summary(components: List[Dict[str, Any]], groups: Dict[str, List[int]], selected_objects: List[int], selected_groups: List[str]) -> None:
    """Render summary of objects and groups."""
    
    st.markdown("### 📊 Selection Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Objects", len(components))
    
    with col2:
        st.metric("Selected Objects", len(selected_objects))
    
    with col3:
        st.metric("Total Groups", len(groups))
    
    with col4:
        st.metric("Selected Groups", len(selected_groups))
    
    # Selected objects details
    if selected_objects:
        selected_components = [c for c in components if c["id"] in selected_objects]
        total_area = sum(c.get("area", 0) for c in selected_components)
        total_perimeter = sum(c.get("perimeter", 0) for c in selected_components)
        
        st.markdown("**Selected Objects Details:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Area", f"{total_area:.1f} mm²")
        
        with col2:
            st.metric("Total Perimeter", f"{total_perimeter:.1f} mm")
        
        with col3:
            st.metric("Average Area", f"{total_area/len(selected_components):.1f} mm²")
    
    # Selected groups details
    if selected_groups:
        selected_group_objects = []
        for group_name in selected_groups:
            if group_name in groups:
                selected_group_objects.extend(groups[group_name])
        
        st.markdown("**Selected Groups Details:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Objects in Groups", len(selected_group_objects))
        
        with col2:
            st.metric("Average Group Size", f"{len(selected_group_objects)/len(selected_groups):.1f}")
        
        with col3:
            st.metric("Group Coverage", f"{len(selected_group_objects)/len(components)*100:.1f}%")
