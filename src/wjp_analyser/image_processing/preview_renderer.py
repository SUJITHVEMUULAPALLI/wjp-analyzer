"""
Final Preview System for Image to DXF Pipeline
===============================================

This module provides comprehensive preview capabilities including:
- DXF vector overlay on original images
- Multi-layer preview with different visualization modes
- Interactive preview controls
- Export-ready preview generation
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import ezdxf

from .object_detector import ObjectDetector, ObjectProperties
from .interactive_editor import InteractiveEditor


class PreviewRenderer:
    """Renders comprehensive previews of DXF conversion results."""
    
    def __init__(self):
        self.original_image: Optional[np.ndarray] = None
        self.binary_image: Optional[np.ndarray] = None
        self.dxf_path: Optional[str] = None
        self.objects: List[ObjectProperties] = []
        
    def set_images(self, original: np.ndarray, binary: np.ndarray):
        """Set the original and binary images for preview."""
        self.original_image = original
        self.binary_image = binary
    
    def set_dxf_path(self, dxf_path: str):
        """Set the DXF file path for vector overlay."""
        self.dxf_path = dxf_path
    
    def set_objects(self, objects: List[ObjectProperties]):
        """Set detected objects for preview."""
        self.objects = objects
    
    def render_vector_overlay(self, 
                            alpha: float = 0.8,
                            line_width: float = 1.2,
                            flip_y: bool = False,
                            show_layer_colors: bool = True,
                            layer_opacity: Dict[str, float] = None) -> np.ndarray:
        """
        Render DXF vectors overlaid on the original image.
        
        Args:
            alpha: Overall opacity of vectors
            line_width: Width of vector lines
            flip_y: Whether to flip Y axis
            show_layer_colors: Use different colors for different layers
            layer_opacity: Custom opacity for each layer type
            
        Returns:
            Preview image with vector overlay
        """
        if self.original_image is None:
            return np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Start with original image
        preview = self.original_image.copy()
        h, w = preview.shape[:2]
        
        # Default layer colors and opacity
        layer_colors = {
            "edges": (255, 0, 0),      # Red
            "stipple": (0, 0, 255),     # Blue
            "hatch": (255, 165, 0),    # Orange
            "contour": (128, 0, 128),  # Purple
            "unknown": (64, 64, 64)    # Gray
        }
        
        if layer_opacity is None:
            layer_opacity = {layer: alpha for layer in layer_colors.keys()}
        
        # Render objects if available
        if self.objects:
            for obj in self.objects:
                if not obj.visible:
                    continue
                
                # Get color and opacity
                if show_layer_colors:
                    color = layer_colors.get(obj.layer_type, (64, 64, 64))
                else:
                    color = (255, 0, 0)  # Default red
                
                opacity = layer_opacity.get(obj.layer_type, alpha)
                
                # Draw contour
                contour_points = obj.contour.reshape(-1, 2)
                
                # Convert to image coordinates if needed
                if flip_y:
                    contour_points = contour_points.copy()
                    contour_points[:, 1] = h - contour_points[:, 1]
                
                # Draw contour lines
                for i in range(len(contour_points)):
                    start_point = tuple(map(int, contour_points[i]))
                    end_point = tuple(map(int, contour_points[(i + 1) % len(contour_points)]))
                    
                    # Create overlay for this line
                    overlay = preview.copy()
                    cv2.line(overlay, start_point, end_point, color, int(line_width))
                    
                    # Blend with original
                    preview = cv2.addWeighted(preview, 1 - opacity, overlay, opacity, 0)
        
        # Render DXF vectors if available
        elif self.dxf_path and Path(self.dxf_path).exists():
            preview = self._render_dxf_overlay(preview, alpha, line_width, flip_y)
        
        return preview
    
    def _render_dxf_overlay(self, 
                          base_image: np.ndarray,
                          alpha: float,
                          line_width: float,
                          flip_y: bool) -> np.ndarray:
        """Render DXF vectors over base image."""
        try:
            # Read DXF file
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()
            
            h, w = base_image.shape[:2]
            preview = base_image.copy()
            
            # Get DXF bounds
            all_points = []
            
            # Collect points from all entities
            for entity in msp:
                if entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points('xy'))
                    all_points.extend(points)
                elif entity.dxftype() == 'LINE':
                    all_points.extend([entity.dxf.start[:2], entity.dxf.end[:2]])
                elif entity.dxftype() == 'ARC':
                    # Sample arc points
                    center = entity.dxf.center[:2]
                    radius = entity.dxf.radius
                    start_angle = math.radians(entity.dxf.start_angle)
                    end_angle = math.radians(entity.dxf.end_angle)
                    
                    steps = 32
                    angles = np.linspace(start_angle, end_angle, steps)
                    arc_points = [(center[0] + radius * math.cos(a), 
                                 center[1] + radius * math.sin(a)) for a in angles]
                    all_points.extend(arc_points)
            
            if not all_points:
                return preview
            
            # Calculate scaling to fit image
            xs, ys = zip(*all_points)
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            
            spanx = max(1e-6, maxx - minx)
            spany = max(1e-6, maxy - miny)
            
            sx = (w - 1) / spanx
            sy = (h - 1) / spany
            scale = min(sx, sy)
            
            def map_point(x, y):
                xx = (x - minx) * scale
                yy = (y - miny) * scale
                if flip_y:
                    yy = h - 1 - yy
                return int(xx), int(yy)
            
            # Draw entities
            color = (255, 0, 0)  # Red for DXF vectors
            
            for entity in msp:
                if entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points('xy'))
                    mapped_points = [map_point(x, y) for x, y in points]
                    
                    for i in range(len(mapped_points)):
                        start = mapped_points[i]
                        end = mapped_points[(i + 1) % len(mapped_points)]
                        cv2.line(preview, start, end, color, int(line_width))
                
                elif entity.dxftype() == 'LINE':
                    start = map_point(entity.dxf.start.x, entity.dxf.start.y)
                    end = map_point(entity.dxf.end.x, entity.dxf.end.y)
                    cv2.line(preview, start, end, color, int(line_width))
                
                elif entity.dxftype() == 'ARC':
                    center = entity.dxf.center[:2]
                    radius = entity.dxf.radius
                    start_angle = math.radians(entity.dxf.start_angle)
                    end_angle = math.radians(entity.dxf.end_angle)
                    
                    steps = 32
                    angles = np.linspace(start_angle, end_angle, steps)
                    
                    for i in range(len(angles) - 1):
                        p1 = map_point(center[0] + radius * math.cos(angles[i]),
                                     center[1] + radius * math.sin(angles[i]))
                        p2 = map_point(center[0] + radius * math.cos(angles[i + 1]),
                                     center[1] + radius * math.sin(angles[i + 1]))
                        cv2.line(preview, p1, p2, color, int(line_width))
            
            return preview
            
        except Exception as e:
            st.error(f"Error rendering DXF overlay: {e}")
            return base_image
    
    def render_multi_layer_preview(self, 
                                 show_layers: List[str] = None,
                                 layer_opacity: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """
        Render separate previews for each layer type.
        
        Args:
            show_layers: List of layer types to show
            layer_opacity: Custom opacity for each layer
            
        Returns:
            Dictionary mapping layer names to preview images
        """
        if not self.objects:
            return {}
        
        if show_layers is None:
            show_layers = ["edges", "stipple", "hatch", "contour"]
        
        if layer_opacity is None:
            layer_opacity = {layer: 0.8 for layer in show_layers}
        
        previews = {}
        
        for layer_type in show_layers:
            # Filter objects by layer type
            layer_objects = [obj for obj in self.objects if obj.layer_type == layer_type]
            
            if not layer_objects:
                continue
            
            # Create preview for this layer
            if self.original_image is not None:
                preview = self.original_image.copy()
            else:
                preview = np.ones((400, 400, 3), dtype=np.uint8) * 255
            
            # Draw objects of this layer type
            layer_color = {
                "edges": (255, 0, 0),      # Red
                "stipple": (0, 0, 255),     # Blue
                "hatch": (255, 165, 0),    # Orange
                "contour": (128, 0, 128),  # Purple
            }.get(layer_type, (64, 64, 64))
            
            opacity = layer_opacity.get(layer_type, 0.8)
            
            for obj in layer_objects:
                if not obj.visible:
                    continue
                
                contour_points = obj.contour.reshape(-1, 2)
                
                # Draw contour
                for i in range(len(contour_points)):
                    start_point = tuple(map(int, contour_points[i]))
                    end_point = tuple(map(int, contour_points[(i + 1) % len(contour_points)]))
                    
                    overlay = preview.copy()
                    cv2.line(overlay, start_point, end_point, layer_color, 2)
                    preview = cv2.addWeighted(preview, 1 - opacity, overlay, opacity, 0)
            
            previews[layer_type] = preview
        
        return previews
    
    def render_comparison_view(self, 
                             show_original: bool = True,
                             show_binary: bool = True,
                             show_vectors: bool = True,
                             show_objects: bool = True) -> np.ndarray:
        """
        Render a comparison view showing multiple stages.
        
        Args:
            show_original: Show original image
            show_binary: Show binary processed image
            show_vectors: Show vector overlay
            show_objects: Show detected objects
            
        Returns:
            Combined comparison image
        """
        views = []
        
        if show_original and self.original_image is not None:
            views.append(("Original", self.original_image))
        
        if show_binary and self.binary_image is not None:
            # Convert binary to 3-channel
            binary_3ch = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2RGB)
            views.append(("Binary", binary_3ch))
        
        if show_vectors and self.dxf_path:
            vector_preview = self.render_vector_overlay()
            views.append(("Vectors", vector_preview))
        
        if show_objects and self.objects:
            object_preview = self._render_object_preview()
            views.append(("Objects", object_preview))
        
        if not views:
            return np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Combine views
        max_height = max(img.shape[0] for _, img in views)
        total_width = sum(img.shape[1] for _, img in views)
        
        combined = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
        
        x_offset = 0
        for title, img in views:
            h, w = img.shape[:2]
            # Resize to match max height
            if h != max_height:
                aspect_ratio = w / h
                new_w = int(max_height * aspect_ratio)
                img = cv2.resize(img, (new_w, max_height))
                w = new_w
            
            combined[0:h, x_offset:x_offset + w] = img
            x_offset += w
        
        return combined
    
    def _render_object_preview(self) -> np.ndarray:
        """Render detected objects preview."""
        if not self.objects or self.original_image is None:
            return np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        preview = self.original_image.copy()
        
        # Color map for object types
        color_map = {
            "edges": (255, 0, 0),      # Red
            "stipple": (0, 0, 255),     # Blue
            "hatch": (255, 165, 0),    # Orange
            "contour": (128, 0, 128),  # Purple
            "unknown": (64, 64, 64)    # Gray
        }
        
        for obj in self.objects:
            if not obj.visible:
                continue
            
            color = color_map.get(obj.layer_type, (64, 64, 64))
            
            # Draw contour
            cv2.drawContours(preview, [obj.contour], -1, color, 2)
            
            # Draw center point
            center = tuple(map(int, obj.center))
            cv2.circle(preview, center, 3, color, -1)
            
            # Draw bounding box
            x, y, w, h = obj.bounding_rect
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 1)
        
        return preview
    
    def generate_export_preview(self, 
                             size: Tuple[int, int] = (800, 800),
                             include_legend: bool = True,
                             include_stats: bool = True) -> np.ndarray:
        """
        Generate a high-quality preview for export.
        
        Args:
            size: Output image size
            include_legend: Include layer legend
            include_stats: Include statistics text
            
        Returns:
            High-quality preview image
        """
        # Start with vector overlay
        preview = self.render_vector_overlay(alpha=0.9, line_width=2.0)
        
        # Resize to target size
        preview = cv2.resize(preview, size)
        
        if include_legend or include_stats:
            # Convert to PIL for text rendering
            preview_pil = Image.fromarray(preview)
            draw = ImageDraw.Draw(preview_pil)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            y_offset = 10
            
            if include_legend:
                # Draw legend
                legend_items = [
                    ("Edges", (255, 0, 0)),
                    ("Stipple", (0, 0, 255)),
                    ("Hatch", (255, 165, 0)),
                    ("Contour", (128, 0, 128))
                ]
                
                for label, color in legend_items:
                    # Draw color box
                    draw.rectangle([10, y_offset, 30, y_offset + 15], fill=color)
                    # Draw label
                    draw.text((35, y_offset), label, fill=(0, 0, 0), font=font)
                    y_offset += 25
            
            if include_stats and self.objects:
                # Draw statistics
                stats = self._calculate_preview_stats()
                stats_text = [
                    f"Objects: {stats['total_objects']}",
                    f"Selected: {stats['selected_objects']}",
                    f"Total Area: {stats['total_area']:.1f}",
                    f"Avg Circularity: {stats['avg_circularity']:.2f}"
                ]
                
                for text in stats_text:
                    draw.text((10, y_offset), text, fill=(0, 0, 0), font=font)
                    y_offset += 20
            
            # Convert back to numpy
            preview = np.array(preview_pil)
        
        return preview
    
    def _calculate_preview_stats(self) -> Dict[str, Union[int, float]]:
        """Calculate statistics for preview display."""
        if not self.objects:
            return {}
        
        areas = [obj.area for obj in self.objects]
        circularities = [obj.circularity for obj in self.objects]
        
        return {
            "total_objects": len(self.objects),
            "selected_objects": len([obj for obj in self.objects if obj.selected]),
            "total_area": sum(areas),
            "avg_circularity": np.mean(circularities) if circularities else 0.0
        }


def render_final_preview_interface(renderer: PreviewRenderer) -> Dict[str, Any]:
    """
    Render the complete final preview interface.
    
    Returns:
        Dictionary with preview settings and generated images
    """
    st.subheader("Final Preview")
    
    # Preview controls
    col1, col2 = st.columns(2)
    
    with col1:
        preview_mode = st.selectbox(
            "Preview Mode",
            ["Vector Overlay", "Multi-Layer", "Comparison View", "Export Quality"],
            index=0
        )
        
        alpha = st.slider("Vector Opacity", 0.1, 1.0, 0.8)
        line_width = st.slider("Line Width", 0.5, 5.0, 1.2)
        flip_y = st.checkbox("Flip Y Axis", value=False, key="final_prev_flip_y")
    
    with col2:
        show_layer_colors = st.checkbox("Show Layer Colors", value=True, key="final_prev_show_layer_colors")
        show_bounding_boxes = st.checkbox("Show Bounding Boxes", value=False, key="final_prev_show_bboxes")
        show_center_points = st.checkbox("Show Center Points", value=False, key="final_prev_show_centers")
    
    # Layer-specific controls
    if preview_mode == "Multi-Layer":
        st.write("**Layer Selection:**")
        layer_cols = st.columns(4)
        
        layer_opacity = {}
        with layer_cols[0]:
            show_edges = st.checkbox("Edges", value=True, key="final_prev_show_edges")
            if show_edges:
                layer_opacity["edges"] = st.slider("Edges Opacity", 0.1, 1.0, 0.8, key="edges_opacity")
        
        with layer_cols[1]:
            show_stipple = st.checkbox("Stipple", value=True, key="final_prev_show_stipple")
            if show_stipple:
                layer_opacity["stipple"] = st.slider("Stipple Opacity", 0.1, 1.0, 0.8, key="stipple_opacity")
        
        with layer_cols[2]:
            show_hatch = st.checkbox("Hatch", value=True, key="final_prev_show_hatch")
            if show_hatch:
                layer_opacity["hatch"] = st.slider("Hatch Opacity", 0.1, 1.0, 0.8, key="hatch_opacity")
        
        with layer_cols[3]:
            show_contour = st.checkbox("Contour", value=True, key="final_prev_show_contour")
            if show_contour:
                layer_opacity["contour"] = st.slider("Contour Opacity", 0.1, 1.0, 0.8, key="contour_opacity")
    
    # Generate preview based on mode
    preview_image = None
    
    if preview_mode == "Vector Overlay":
        preview_image = renderer.render_vector_overlay(
            alpha=alpha,
            line_width=line_width,
            flip_y=flip_y,
            show_layer_colors=show_layer_colors
        )
        
    elif preview_mode == "Multi-Layer":
        show_layers = []
        if show_edges:
            show_layers.append("edges")
        if show_stipple:
            show_layers.append("stipple")
        if show_hatch:
            show_layers.append("hatch")
        if show_contour:
            show_layers.append("contour")
        
        layer_previews = renderer.render_multi_layer_preview(
            show_layers=show_layers,
            layer_opacity=layer_opacity
        )
        
        # Display layer previews
        if layer_previews:
            preview_cols = st.columns(len(layer_previews))
            for i, (layer_name, layer_img) in enumerate(layer_previews.items()):
                with preview_cols[i]:
                    st.image(layer_img, caption=f"{layer_name.title()} Layer", use_column_width=True)
        
        # Use first layer as main preview
        if layer_previews:
            preview_image = list(layer_previews.values())[0]
    
    elif preview_mode == "Comparison View":
        preview_image = renderer.render_comparison_view()
        
    elif preview_mode == "Export Quality":
        preview_size = st.slider("Export Size", 400, 1200, 800)
        include_legend = st.checkbox("Include Legend", value=True, key="final_prev_include_legend")
        include_stats = st.checkbox("Include Statistics", value=True, key="final_prev_include_stats")
        
        preview_image = renderer.generate_export_preview(
            size=(preview_size, preview_size),
            include_legend=include_legend,
            include_stats=include_stats
        )
    
    # Display main preview
    if preview_image is not None:
        st.image(preview_image, caption=f"{preview_mode} Preview", use_column_width=True)
        
        # Export options
        st.write("**Export Preview:**")
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("Download Preview Image"):
                # Convert to PIL and create download
                preview_pil = Image.fromarray(preview_image)
                img_buffer = io.BytesIO()
                preview_pil.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    "Download PNG",
                    data=img_buffer.getvalue(),
                    file_name="dxf_preview.png",
                    mime="image/png"
                )
        
        with col_exp2:
            if st.button("Download High-Res Preview"):
                # Generate high-resolution version
                high_res = renderer.generate_export_preview(
                    size=(1600, 1600),
                    include_legend=True,
                    include_stats=True
                )
                
                preview_pil = Image.fromarray(high_res)
                img_buffer = io.BytesIO()
                preview_pil.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    "Download High-Res PNG",
                    data=img_buffer.getvalue(),
                    file_name="dxf_preview_hires.png",
                    mime="image/png"
                )
    
    return {
        "preview_mode": preview_mode,
        "preview_image": preview_image,
        "settings": {
            "alpha": alpha,
            "line_width": line_width,
            "flip_y": flip_y,
            "show_layer_colors": show_layer_colors
        }
    }
