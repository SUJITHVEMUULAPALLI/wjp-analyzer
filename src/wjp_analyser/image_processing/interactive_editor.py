"""
Interactive Editing Interface for Image to DXF
==============================================

This module provides interactive editing capabilities for the Image to DXF pipeline,
including object selection, modification, and real-time preview generation.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .object_detector import ObjectDetector, ObjectProperties, DetectionParams


class InteractiveEditor:
    """Interactive editor for object manipulation and preview generation."""
    
    def __init__(self):
        self.detector: Optional[ObjectDetector] = None
        self.current_image: Optional[np.ndarray] = None
        self.binary_image: Optional[np.ndarray] = None
        self.preview_image: Optional[np.ndarray] = None
        self.edit_history: List[Dict[str, Any]] = []
        self.current_edit_index = -1
        
    def load_image(self, image_path: str) -> bool:
        """Load an image for editing."""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                return False
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            return True
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return False
    
    def set_binary_image(self, binary_image: np.ndarray):
        """Set the binary image for object detection."""
        self.binary_image = binary_image
        self.detect_objects()
    
    def detect_objects(self, params: Optional[DetectionParams] = None) -> List[ObjectProperties]:
        """Detect objects in the current binary image."""
        if self.binary_image is None:
            return []
            
        if self.detector is None:
            self.detector = ObjectDetector()
        self.detector.params = params or DetectionParams()
        objects = self.detector.detect_objects(self.binary_image, self.current_image)
        
        # Save state for undo/redo
        self._save_state("object_detection")
        
        return objects
    
    def _save_state(self, action: str):
        """Save current state for undo/redo functionality."""
        if self.detector is None:
            return
            
        state = {
            "action": action,
            "objects": [
                {
                    "id": obj.id,
                    "selected": obj.selected,
                    "visible": obj.visible,
                    "layer_type": obj.layer_type,
                    "color": obj.color
                }
                for obj in self.detector.objects
            ]
        }
        
        # Remove any states after current index
        self.edit_history = self.edit_history[:self.current_edit_index + 1]
        self.edit_history.append(state)
        self.current_edit_index = len(self.edit_history) - 1
        
        # Limit history size
        if len(self.edit_history) > 50:
            self.edit_history = self.edit_history[-50:]
            self.current_edit_index = len(self.edit_history) - 1
    
    def undo(self) -> bool:
        """Undo the last action."""
        if self.current_edit_index <= 0 or not self.detector:
            return False
            
        self.current_edit_index -= 1
        self._restore_state(self.edit_history[self.current_edit_index])
        return True
    
    def redo(self) -> bool:
        """Redo the next action."""
        if (self.current_edit_index >= len(self.edit_history) - 1 or 
            not self.detector):
            return False
            
        self.current_edit_index += 1
        self._restore_state(self.edit_history[self.current_edit_index])
        return True
    
    def _restore_state(self, state: Dict[str, Any]):
        """Restore a saved state."""
        if not self.detector:
            return
            
        state_objects = {obj["id"]: obj for obj in state["objects"]}
        
        for obj in self.detector.objects:
            if obj.id in state_objects:
                state_obj = state_objects[obj.id]
                obj.selected = state_obj["selected"]
                obj.visible = state_obj["visible"]
                obj.layer_type = state_obj["layer_type"]
                obj.color = state_obj["color"]
    
    def generate_preview(self, 
                        show_selection: bool = True,
                        show_bounding_boxes: bool = False,
                        overlay_on_original: bool = True,
                        show_layer_colors: bool = True) -> np.ndarray:
        """Generate a preview image with current object states."""
        if not self.detector:
            return np.ones((100, 100, 3), dtype=np.uint8) * 255
            
        return self.detector.generate_preview(
            show_selection=show_selection,
            show_bounding_boxes=show_bounding_boxes,
            overlay_on_original=overlay_on_original
        )
    
    def render_object_list(self) -> List[int]:
        """Render object list and return selected object IDs."""
        if not self.detector:
            st.info("No objects detected. Please run object detection first.")
            return []
        
        st.subheader("Detected Objects")
        
        # Object statistics
        stats = self.detector.get_statistics()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Objects", stats.get("total_objects", 0))
        col2.metric("Selected", stats.get("selected_objects", 0))
        col3.metric("Visible", stats.get("visible_objects", 0))
        col4.metric("Avg Area", f"{stats.get('avg_area', 0):.1f}")
        
        # Object type distribution
        type_counts = stats.get("type_counts", {})
        if type_counts:
            st.write("**Object Types:**")
            type_cols = st.columns(len(type_counts))
            for i, (obj_type, count) in enumerate(type_counts.items()):
                with type_cols[i]:
                    st.metric(obj_type.title(), count)
        
        # Object list with selection controls
        selected_ids = []
        
        with st.expander("Object List", expanded=True):
            # Batch selection controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Select All"):
                    for obj in self.detector.objects:
                        obj.selected = True
                    self._save_state("select_all")
                    
            with col2:
                if st.button("Deselect All"):
                    for obj in self.detector.objects:
                        obj.selected = False
                    self._save_state("deselect_all")
                    
            with col3:
                if st.button("Invert Selection"):
                    for obj in self.detector.objects:
                        obj.selected = not obj.selected
                    self._save_state("invert_selection")
            
            # Individual object controls
            for obj in self.detector.objects:
                col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])
                
                with col1:
                    obj.selected = st.checkbox(
                        "", 
                        value=obj.selected, 
                        key=f"select_{obj.id}"
                    )
                    if obj.selected:
                        selected_ids.append(obj.id)
                
                with col2:
                    st.write(f"**Object {obj.id}** ({obj.layer_type})")
                    st.caption(f"Area: {obj.area:.1f}, Circ: {obj.circularity:.2f}")
                
                with col3:
                    obj.visible = st.checkbox(
                        "Visible", 
                        value=obj.visible, 
                        key=f"visible_{obj.id}"
                    )
                
                with col4:
                    new_type = st.selectbox(
                        "Type",
                        ["edges", "stipple", "hatch", "contour"],
                        index=["edges", "stipple", "hatch", "contour"].index(obj.layer_type),
                        key=f"type_{obj.id}"
                    )
                    if new_type != obj.layer_type:
                        obj.layer_type = new_type
                        self._save_state(f"change_type_{obj.id}")
                
                with col5:
                    if st.button("Delete", key=f"delete_{obj.id}"):
                        self.detector.objects.remove(obj)
                        self._save_state(f"delete_{obj.id}")
                        st.rerun()
        
        return selected_ids
    
    def render_preview_controls(self):
        """Render preview control options."""
        st.subheader("Preview Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_selection = st.checkbox("Highlight Selected", value=True)
            show_bounding_boxes = st.checkbox("Show Bounding Boxes", value=False)
            overlay_on_original = st.checkbox("Overlay on Original", value=True)
        
        with col2:
            preview_size = st.slider("Preview Size", 200, 800, 400)
            line_width = st.slider("Line Width", 1, 5, 2)
        
        return {
            "show_selection": show_selection,
            "show_bounding_boxes": show_bounding_boxes,
            "overlay_on_original": overlay_on_original,
            "preview_size": preview_size,
            "line_width": line_width
        }
    
    def render_preview(self, controls: Dict[str, Any]):
        """Render the preview image."""
        if not self.detector:
            st.info("No objects to preview. Please run object detection first.")
            return
        
        # Generate preview
        preview = self.generate_preview(
            show_selection=controls["show_selection"],
            show_bounding_boxes=controls["show_bounding_boxes"],
            overlay_on_original=controls["overlay_on_original"]
        )
        
        # Convert to PIL Image for display
        preview_pil = Image.fromarray(preview)
        
        # Resize for display
        display_size = (controls["preview_size"], controls["preview_size"])
        preview_pil = preview_pil.resize(display_size, Image.Resampling.LANCZOS)
        
        st.image(preview_pil, caption="Object Preview", use_column_width=True)
        
        # Preview statistics
        stats = self.detector.get_statistics()
        st.caption(f"Showing {stats.get('visible_objects', 0)} visible objects, "
                  f"{stats.get('selected_objects', 0)} selected")
    
    def render_editing_tools(self):
        """Render editing tools and controls."""
        st.subheader("Editing Tools")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Undo", disabled=self.current_edit_index <= 0):
                self.undo()
                st.rerun()
                
        with col2:
            if st.button("Redo", disabled=self.current_edit_index >= len(self.edit_history) - 1):
                self.redo()
                st.rerun()
                
        with col3:
            if st.button("Reset All"):
                if self.detector:
                    for obj in self.detector.objects:
                        obj.selected = False
                        obj.visible = True
                    self._save_state("reset_all")
                    st.rerun()
        
        # Layer operations
        st.write("**Layer Operations:**")
        layer_cols = st.columns(4)
        
        with layer_cols[0]:
            if st.button("Select All Edges"):
                self.detector.select_objects_by_type("edges", True)
                self._save_state("select_edges")
                
        with layer_cols[1]:
            if st.button("Select All Stipple"):
                self.detector.select_objects_by_type("stipple", True)
                self._save_state("select_stipple")
                
        with layer_cols[2]:
            if st.button("Select All Hatch"):
                self.detector.select_objects_by_type("hatch", True)
                self._save_state("select_hatch")
                
        with layer_cols[3]:
            if st.button("Select All Contour"):
                self.detector.select_objects_by_type("contour", True)
                self._save_state("select_contour")
    
    def export_selected_objects(self, output_path: str) -> bool:
        """Export only selected objects to DXF."""
        if not self.detector:
            return False
        return self.detector.export_to_dxf(output_path, selected_only=True)
    
    def export_all_objects(self, output_path: str) -> bool:
        """Export all visible objects to DXF."""
        if not self.detector:
            return False
        return self.detector.export_to_dxf(output_path, selected_only=False)


def render_interactive_editor(editor: InteractiveEditor) -> Dict[str, Any]:
    """
    Render the complete interactive editing interface.
    
    Returns:
        Dictionary with editor state and controls
    """
    if not editor.detector:
        st.info("Please load an image and run object detection first.")
        return {}
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Preview controls
        preview_controls = editor.render_preview_controls()
        
        # Render preview
        editor.render_preview(preview_controls)
    
    with col2:
        # Object list and selection
        selected_ids = editor.render_object_list()
        
        # Editing tools
        editor.render_editing_tools()
        
        # Export options
        st.subheader("Export Options")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("Export Selected"):
                if selected_ids:
                    # Create temporary file
                    temp_path = Path("temp_selected.dxf")
                    if editor.export_selected_objects(str(temp_path)):
                        with open(temp_path, "rb") as f:
                            st.download_button(
                                "Download Selected Objects DXF",
                                data=f.read(),
                                file_name="selected_objects.dxf",
                                mime="application/dxf"
                            )
                        temp_path.unlink()
                else:
                    st.warning("No objects selected")
        
        with col_exp2:
            if st.button("Export All"):
                temp_path = Path("temp_all.dxf")
                if editor.export_all_objects(str(temp_path)):
                    with open(temp_path, "rb") as f:
                        st.download_button(
                            "Download All Objects DXF",
                            data=f.read(),
                            file_name="all_objects.dxf",
                            mime="application/dxf"
                        )
                    temp_path.unlink()
    
    return {
        "selected_ids": selected_ids,
        "preview_controls": preview_controls,
        "editor": editor
    }
