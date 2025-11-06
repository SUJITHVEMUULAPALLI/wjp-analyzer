#!/usr/bin/env python3
"""
Enhanced OpenCV Image-to-DXF Converter with Border Removal
Integrates ChatGPT-5 recommendations into the web interface.
"""

import cv2
import numpy as np
import ezdxf
from shapely.geometry import LineString
from typing import Dict, List, Optional, Tuple
import os

class EnhancedOpenCVImageToDXFConverter:
    """Enhanced OpenCV converter with border removal and multi-threshold processing."""
    
    def __init__(self, binary_threshold: int = 180, min_area: int = 1000, dxf_size: int = 1200):
        self.binary_threshold = binary_threshold
        self.min_area = min_area
        self.dxf_size = dxf_size
        self.border_margin = 0.05  # 5% margin for border detection
        
    def convert_image_to_dxf(self, input_image: str, output_dxf: str, preview_output: str) -> Dict:
        """Convert image to DXF with enhanced border removal."""
        try:
            # Load image
            image = cv2.imread(input_image)
            if image is None:
                return {"error": "Could not load image", "polygons": 0}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Multi-threshold approach for better contour detection
            contours = self._multi_threshold_contouring(blurred)
            
            # Filter contours by area and remove borders
            filtered_contours = self._filter_contours(contours, gray.shape)
            
            # Apply Douglas-Peucker simplification
            simplified_contours = self._simplify_contours(filtered_contours)
            
            # Convert to DXF
            polygon_count = self._create_dxf(simplified_contours, output_dxf)
            
            # Create preview
            self._create_preview(image, simplified_contours, preview_output)
            
            return {
                "polygons": polygon_count,
                "contours_found": len(contours),
                "contours_after_filtering": len(filtered_contours),
                "contours_after_simplification": len(simplified_contours)
            }
            
        except Exception as e:
            return {"error": str(e), "polygons": 0}
    
    def _multi_threshold_contouring(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply multi-threshold contouring for better detection."""
        all_contours = []
        
        # Use multiple thresholds for better contour detection
        thresholds = [self.binary_threshold - 20, self.binary_threshold, self.binary_threshold + 20]
        
        for threshold in thresholds:
            # Create binary image
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            
            # Remove image borders
            binary = self._remove_image_borders(binary)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out border contours
            filtered_contours = self._filter_border_contours(contours, image.shape)
            
            all_contours.extend(filtered_contours)
        
        # Remove duplicate contours
        return self._remove_duplicate_contours(all_contours)
    
    def _remove_image_borders(self, binary_image: np.ndarray) -> np.ndarray:
        """Remove image borders/outlines from binary image."""
        height, width = binary_image.shape
        
        # Define border margin (percentage of image dimensions)
        margin_h = int(height * 0.02)  # 2% margin
        margin_w = int(width * 0.02)   # 2% margin
        
        # Create a mask to remove borders
        mask = np.ones_like(binary_image, dtype=np.uint8) * 255
        
        # Set border areas to black (remove them)
        mask[:margin_h, :] = 0  # Top border
        mask[-margin_h:, :] = 0  # Bottom border
        mask[:, :margin_w] = 0  # Left border
        mask[:, -margin_w:] = 0  # Right border
        
        # Apply mask to remove borders
        result = cv2.bitwise_and(binary_image, mask)
        
        return result
    
    def _filter_border_contours(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Filter out contours that are likely image borders."""
        if not contours:
            return []
        
        height, width = image_shape
        filtered_contours = []
        
        # Define border detection parameters
        min_border_distance = min(height, width) * self.border_margin
        
        for contour in contours:
            # Calculate contour bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour is near image borders
            is_near_top = y < min_border_distance
            is_near_bottom = y + h > height - min_border_distance
            is_near_left = x < min_border_distance
            is_near_right = x + w > width - min_border_distance
            
            # Check if contour spans most of the image (likely a border)
            spans_width = w > width * 0.8  # Spans 80% of width
            spans_height = h > height * 0.8  # Spans 80% of height
            
            # Check if contour is very large (likely image border)
            contour_area = cv2.contourArea(contour)
            image_area = height * width
            is_large_contour = contour_area > image_area * 0.3  # More than 30% of image
            
            # Filter out border-like contours
            is_border_contour = (
                (is_near_top and is_near_bottom) or  # Spans top to bottom
                (is_near_left and is_near_right) or  # Spans left to right
                (spans_width and spans_height) or    # Spans most of image
                is_large_contour                     # Very large contour
            )
            
            if not is_border_contour:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def _remove_duplicate_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Remove duplicate or very similar contours."""
        if not contours:
            return []
        
        unique_contours = []
        similarity_threshold = 0.95  # 95% similarity threshold
        
        for contour in contours:
            is_duplicate = False
            
            for existing in unique_contours:
                # Calculate similarity using contour matching
                similarity = cv2.matchShapes(contour, existing, cv2.CONTOURS_MATCH_I1, 0)
                
                if similarity < (1 - similarity_threshold):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contours.append(contour)
        
        return unique_contours
    
    def _filter_contours(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Filter contours by area and other criteria."""
        filtered_contours = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Apply area filter
            if area >= self.min_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def _simplify_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Douglas-Peucker simplification to contours."""
        simplified_contours = []
        
        for contour in contours:
            # Apply Douglas-Peucker algorithm
            epsilon = 1.0  # Simplification tolerance
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Only keep contours with sufficient points
            if len(simplified) >= 3:
                simplified_contours.append(simplified)
        
        return simplified_contours
    
    def _create_dxf(self, contours: List[np.ndarray], output_path: str) -> int:
        """Create DXF file from contours."""
        doc = ezdxf.new()
        msp = doc.modelspace()
        
        polygon_count = 0
        
        for contour in contours:
            try:
                # Convert contour to LineString
                points = contour.reshape(-1, 2)
                if len(points) >= 2:
                    # Scale points to DXF size
                    scaled_points = self._scale_points(points)
                    
                    # Add as closed polyline
                    msp.add_lwpolyline(scaled_points, close=True)
                    polygon_count += 1
            except Exception as e:
                print(f"Error adding contour to DXF: {e}")
                continue
        
        doc.saveas(output_path)
        return polygon_count
    
    def _scale_points(self, points: np.ndarray) -> List[Tuple[float, float]]:
        """Scale points to DXF coordinate system."""
        if len(points) == 0:
            return []
        
        # Get bounding box
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        
        # Calculate scale factor
        width = max_x - min_x
        height = max_y - min_y
        max_dim = max(width, height)
        
        if max_dim == 0:
            return [(0, 0)]
        
        scale_factor = self.dxf_size / max_dim
        
        # Scale and center points
        scaled_points = []
        for point in points:
            x = (point[0] - min_x) * scale_factor
            y = (point[1] - min_y) * scale_factor
            scaled_points.append((x, y))
        
        return scaled_points
    
    def _create_preview(self, image: np.ndarray, contours: List[np.ndarray], preview_path: str):
        """Create preview image showing detected contours."""
        preview = image.copy()
        
        # Draw contours in different colors
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, contour in enumerate(contours):
            color = colors[i % len(colors)]
            cv2.drawContours(preview, [contour], -1, color, 2)
        
        cv2.imwrite(preview_path, preview)
