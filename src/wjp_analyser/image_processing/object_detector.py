"""
Object Detection and Analysis for Image to DXF Pipeline
======================================================

This module provides object identification, shape analysis, and interactive editing
capabilities for the Image to DXF conversion process.

Features:
- Contour detection and shape classification
- Object property analysis (area, perimeter, circularity, etc.)
- Interactive object selection and modification
- Real-time preview generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg


@dataclass
class ObjectProperties:
    """Properties of a detected object."""
    id: int
    contour: np.ndarray
    area: float
    perimeter: float
    circularity: float
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[float, float]
    aspect_ratio: float
    solidity: float
    convexity: float
    is_closed: bool
    layer_type: str = "unknown"  # edges, stipple, hatch, contour
    selected: bool = False
    visible: bool = True
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DetectionParams:
    """Parameters for object detection."""
    min_area: int = 100
    max_area: int = 1000000
    min_perimeter: int = 20
    min_circularity: float = 0.1
    max_circularity: float = 1.0
    min_solidity: float = 0.3
    min_convexity: float = 0.5
    merge_distance: float = 10.0
    simplify_tolerance: float = 2.0


class ObjectDetector:
    """Main class for object detection and analysis."""
    
    def __init__(self, params: Optional[DetectionParams] = None):
        self.params = params or DetectionParams()
        self.objects: List[ObjectProperties] = []
        self.image_shape: Tuple[int, int] = (0, 0)
        self.original_image: Optional[np.ndarray] = None
        
    def detect_objects(self, binary_image: np.ndarray, original_image: Optional[np.ndarray] = None) -> List[ObjectProperties]:
        """
        Detect objects in a binary image.
        
        Args:
            binary_image: Binary image (0/255 values)
            original_image: Original image for reference (optional)
            
        Returns:
            List of detected objects with properties
        """
        self.original_image = original_image
        self.image_shape = binary_image.shape[:2]
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        self.objects = []
        
        for i, contour in enumerate(contours):
            # Calculate basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter by area
            if area < self.params.min_area or area > self.params.max_area:
                continue
                
            # Filter by perimeter
            if perimeter < self.params.min_perimeter:
                continue
            
            # Calculate derived properties
            circularity = self._calculate_circularity(area, perimeter)
            if circularity < self.params.min_circularity or circularity > self.params.max_circularity:
                continue
                
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0.0
            
            # Center point
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx, cy = x + w/2, y + h/2
            
            # Solidity and convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            
            if solidity < self.params.min_solidity:
                continue
                
            convexity = perimeter / cv2.arcLength(hull, True) if cv2.arcLength(hull, True) > 0 else 0.0
            if convexity < self.params.min_convexity:
                continue
            
            # Check if contour is closed
            is_closed = cv2.arcLength(contour, True) == cv2.arcLength(contour, False)
            
            # Simplify contour if needed
            if self.params.simplify_tolerance > 0:
                epsilon = self.params.simplify_tolerance * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Create object properties
            obj = ObjectProperties(
                id=i,
                contour=contour,
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                bounding_rect=(x, y, w, h),
                center=(cx, cy),
                aspect_ratio=aspect_ratio,
                solidity=solidity,
                convexity=convexity,
                is_closed=is_closed,
                layer_type=self._classify_object_type(contour, area, circularity, solidity)
            )
            
            self.objects.append(obj)
        
        # Merge nearby objects if specified
        if self.params.merge_distance > 0:
            self._merge_nearby_objects()
            
        return self.objects
    
    def _calculate_circularity(self, area: float, perimeter: float) -> float:
        """Calculate circularity metric (4π*area/perimeter²)."""
        if perimeter == 0:
            return 0.0
        return 4 * math.pi * area / (perimeter * perimeter)
    
    def _classify_object_type(self, contour: np.ndarray, area: float, circularity: float, solidity: float) -> str:
        """Classify object type based on properties."""
        # Simple heuristic classification
        if circularity > 0.8:
            return "contour"  # Circular objects
        elif solidity > 0.9:
            return "edges"    # Solid, well-defined shapes
        elif area < 1000:
            return "stipple"  # Small objects
        else:
            return "hatch"    # Large, complex objects
    
    def _merge_nearby_objects(self):
        """Merge objects that are very close to each other."""
        if len(self.objects) < 2:
            return
            
        merged_indices = set()
        new_objects = []
        
        for i, obj1 in enumerate(self.objects):
            if i in merged_indices:
                continue
                
            merged_contour = obj1.contour.copy()
            merged_area = obj1.area
            merged_perimeter = obj1.perimeter
            
            # Find nearby objects to merge
            for j, obj2 in enumerate(self.objects[i+1:], i+1):
                if j in merged_indices:
                    continue
                    
                # Calculate distance between centers
                dx = obj1.center[0] - obj2.center[0]
                dy = obj1.center[1] - obj2.center[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < self.params.merge_distance:
                    # Merge contours
                    merged_contour = np.vstack([merged_contour, obj2.contour])
                    merged_area += obj2.area
                    merged_perimeter += obj2.perimeter
                    merged_indices.add(j)
            
            # Create merged object
            if len(merged_indices) > 0 or merged_contour.shape[0] != obj1.contour.shape[0]:
                # Recalculate properties for merged object
                area = cv2.contourArea(merged_contour)
                perimeter = cv2.arcLength(merged_contour, True)
                circularity = self._calculate_circularity(area, perimeter)
                
                x, y, w, h = cv2.boundingRect(merged_contour)
                moments = cv2.moments(merged_contour)
                if moments['m00'] != 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                else:
                    cx, cy = x + w/2, y + h/2
                
                hull = cv2.convexHull(merged_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0.0
                convexity = perimeter / cv2.arcLength(hull, True) if cv2.arcLength(hull, True) > 0 else 0.0
                
                merged_obj = ObjectProperties(
                    id=len(new_objects),
                    contour=merged_contour,
                    area=area,
                    perimeter=perimeter,
                    circularity=circularity,
                    bounding_rect=(x, y, w, h),
                    center=(cx, cy),
                    aspect_ratio=float(w) / h if h > 0 else 0.0,
                    solidity=solidity,
                    convexity=convexity,
                    is_closed=cv2.arcLength(merged_contour, True) == cv2.arcLength(merged_contour, False),
                    layer_type=obj1.layer_type
                )
                new_objects.append(merged_obj)
            else:
                new_objects.append(obj1)
        
        self.objects = new_objects
    
    def get_object_by_id(self, obj_id: int) -> Optional[ObjectProperties]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None
    
    def select_object(self, obj_id: int, selected: bool = True):
        """Select/deselect an object."""
        obj = self.get_object_by_id(obj_id)
        if obj:
            obj.selected = selected
    
    def select_objects_by_type(self, layer_type: str, selected: bool = True):
        """Select all objects of a specific type."""
        for obj in self.objects:
            if obj.layer_type == layer_type:
                obj.selected = selected
    
    def toggle_object_visibility(self, obj_id: int):
        """Toggle object visibility."""
        obj = self.get_object_by_id(obj_id)
        if obj:
            obj.visible = not obj.visible
    
    def get_selected_objects(self) -> List[ObjectProperties]:
        """Get all selected objects."""
        return [obj for obj in self.objects if obj.selected]
    
    def get_visible_objects(self) -> List[ObjectProperties]:
        """Get all visible objects."""
        return [obj for obj in self.objects if obj.visible]
    
    def generate_preview(self, 
                        show_selection: bool = True, 
                        show_bounding_boxes: bool = False,
                        overlay_on_original: bool = True) -> np.ndarray:
        """
        Generate a preview image showing detected objects.
        
        Args:
            show_selection: Highlight selected objects
            show_bounding_boxes: Show bounding rectangles
            overlay_on_original: Overlay on original image if available
            
        Returns:
            Preview image as numpy array
        """
        if overlay_on_original and self.original_image is not None:
            preview = self.original_image.copy()
            if len(preview.shape) == 3:
                preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        else:
            preview = np.ones((*self.image_shape, 3), dtype=np.uint8) * 255
        
        # Draw objects
        for obj in self.objects:
            if not obj.visible:
                continue
                
            # Choose color based on selection and type
            if show_selection and obj.selected:
                color = (0, 255, 0)  # Green for selected
            else:
                color_map = {
                    "edges": (255, 0, 0),      # Red
                    "stipple": (0, 0, 255),    # Blue
                    "hatch": (255, 165, 0),    # Orange
                    "contour": (128, 0, 128),  # Purple
                    "unknown": (64, 64, 64)    # Gray
                }
                color = color_map.get(obj.layer_type, (64, 64, 64))
            
            # Draw contour
            cv2.drawContours(preview, [obj.contour], -1, color, 2)
            
            # Draw bounding box if requested
            if show_bounding_boxes:
                x, y, w, h = obj.bounding_rect
                cv2.rectangle(preview, (x, y), (x + w, y + h), color, 1)
                
                # Draw center point
                cx, cy = int(obj.center[0]), int(obj.center[1])
                cv2.circle(preview, (cx, cy), 3, color, -1)
        
        return preview
    
    def export_to_dxf(self, output_path: str, selected_only: bool = False) -> bool:
        """
        Export detected objects to DXF format.
        
        Args:
            output_path: Path to save DXF file
            selected_only: Export only selected objects
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import ezdxf
            
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()
            
            # Create layers for different object types
            layer_colors = {
                "edges": 1,      # Red
                "stipple": 5,    # Blue
                "hatch": 30,     # Orange
                "contour": 6,    # Magenta
                "unknown": 7     # White
            }
            
            for layer_type, color in layer_colors.items():
                if layer_type not in doc.layers:
                    doc.layers.add(f"LAYER_{layer_type.upper()}", color=color)
            
            # Export objects
            objects_to_export = self.get_selected_objects() if selected_only else self.objects
            
            for obj in objects_to_export:
                if not obj.visible:
                    continue
                    
                layer_name = f"LAYER_{obj.layer_type.upper()}"
                
                # Convert contour to DXF polyline
                points = obj.contour.reshape(-1, 2)
                
                if len(points) > 1:
                    # Create polyline
                    polyline = msp.add_lwpolyline(points.tolist())
                    polyline.dxf.layer = layer_name
                    
                    # Close polyline if it's a closed contour
                    if obj.is_closed:
                        polyline.closed = True
            
            doc.saveas(output_path)
            return True
            
        except Exception as e:
            print(f"Error exporting to DXF: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get statistics about detected objects."""
        if not self.objects:
            return {}
            
        areas = [obj.area for obj in self.objects]
        perimeters = [obj.perimeter for obj in self.objects]
        circularities = [obj.circularity for obj in self.objects]
        
        type_counts = {}
        for obj in self.objects:
            type_counts[obj.layer_type] = type_counts.get(obj.layer_type, 0) + 1
        
        return {
            "total_objects": len(self.objects),
            "selected_objects": len(self.get_selected_objects()),
            "visible_objects": len(self.get_visible_objects()),
            "avg_area": np.mean(areas),
            "avg_perimeter": np.mean(perimeters),
            "avg_circularity": np.mean(circularities),
            "type_counts": type_counts,
            "total_area": sum(areas),
            "total_perimeter": sum(perimeters)
        }
