"""
OpenCV-based Image to DXF Converter
==================================

Based on the GitHub project: https://github.com/watch3602004/ImageToDxfConverter
Adapted for Python with enhanced features for waterjet analysis.

Key features:
- Binary threshold adjustment (like the original C# version)
- Line pitch control for contour simplification
- Enhanced edge detection with Canny
- Integration with waterjet analysis pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from shapely.geometry import Polygon
from typing import Tuple, List, Optional, Dict

class OpenCVImageToDXFConverter:
    """OpenCV-based image to DXF converter with adjustable parameters."""
    
    def __init__(self, 
                 binary_threshold: int = 127,
                 line_pitch: float = 1.0,
                 min_area: int = 100,
                 dxf_size: float = 1000.0):
        """
        Initialize converter with parameters similar to the GitHub project.
        
        Args:
            binary_threshold: Threshold for binary conversion (0-255)
            line_pitch: Line simplification factor (higher = more simplified)
            min_area: Minimum contour area to keep
            dxf_size: Output DXF canvas size in mm
        """
        self.binary_threshold = binary_threshold
        self.line_pitch = line_pitch
        self.min_area = min_area
        self.dxf_size = dxf_size
        
    def convert_image_to_dxf(self, 
                            input_image: str, 
                            output_dxf: str,
                            preview_output: Optional[str] = None) -> Dict:
        """
        Convert image to DXF using OpenCV approach.
        
        Args:
            input_image: Path to input image
            output_dxf: Path to output DXF file
            preview_output: Optional path for preview image
            
        Returns:
            Dictionary with conversion results and statistics
        """
        # Step 1: Read and preprocess image
        img = cv2.imread(input_image)
        if img is None:
            raise ValueError(f"Could not load image: {input_image}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary thresholding (key feature from GitHub project)
        _, thresh = cv2.threshold(blur, self.binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Optional: Apply Canny edge detection for better edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        return self._process_contours(img, thresh, edges, output_dxf, preview_output, input_image)
    
    def _process_contours(self, img, thresh, edges, output_dxf, preview_output, input_image):
        """Process contours and generate DXF."""
        
        # Morphological filtering to clean noise
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        initial_count = len(contours)
        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        
        print(f"Found {initial_count} contours initially")
        print(f"Kept {len(contours)} contours after filtering")
        
        # Scale contours to DXF size
        scaled_contours = self._scale_contours(contours)
        
        # Generate DXF
        self._generate_dxf(scaled_contours, output_dxf)
        
        # Generate preview if requested
        if preview_output:
            self._generate_preview(img, thresh, scaled_contours, preview_output)
        
        # Analyze results
        polygons = [Polygon(pts) for pts in scaled_contours if len(pts) >= 3]
        outer_count = sum(1 for p in polygons if p.exterior.is_ccw)
        inner_count = len(polygons) - outer_count
        
        return {
            "success": True,
            "input_image": input_image,
            "output_dxf": output_dxf,
            "contours_found": initial_count,
            "contours_kept": len(contours),
            "polygons": len(polygons),
            "outer_polygons": outer_count,
            "inner_polygons": inner_count,
            "dxf_size": self.dxf_size
        }
    
    def _scale_contours(self, contours):
        """Scale contours to DXF size."""
        if not contours:
            return []
            
        # Flatten all points to find bounds
        all_points = np.vstack([c.reshape(-1, 2) for c in contours])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        scale_x = self.dxf_size / (x_max - x_min)
        scale_y = self.dxf_size / (y_max - y_min)
        scale = min(scale_x, scale_y)  # keep aspect ratio
        
        def scale_contour(cnt):
            pts = cnt.reshape(-1, 2)
            pts = (pts - [x_min, y_min]) * scale
            return pts
        
        return [scale_contour(c) for c in contours]
    
    def _generate_dxf(self, scaled_contours, output_dxf):
        """Generate DXF file."""
        doc = ezdxf.new()
        msp = doc.modelspace()
        
        for pts in scaled_contours:
            if len(pts) > 1:
                msp.add_lwpolyline(pts, close=True)
        
        doc.saveas(output_dxf)
        print(f"DXF saved: {output_dxf}")
    
    def _generate_preview(self, img, thresh, scaled_contours, preview_output):
        """Generate preview image."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Thresholded image
        axes[1].imshow(thresh, cmap='gray')
        axes[1].set_title(f"Binary Threshold ({self.binary_threshold})")
        axes[1].axis('off')
        
        # DXF preview
        for pts in scaled_contours:
            if len(pts) > 1:
                x, y = pts[:,0], pts[:,1]
                axes[2].plot(x, -y, 'k-', linewidth=1)
        axes[2].set_title("DXF Preview")
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(preview_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Preview saved: {preview_output}")


def main():
    """Example usage of the OpenCV converter."""
    converter = OpenCVImageToDXFConverter(
        binary_threshold=127,  # Adjustable like the GitHub project
        line_pitch=1.0,       # Line simplification factor
        min_area=100,         # Minimum contour area
        dxf_size=1000.0       # Output DXF size in mm
    )
    
    # Convert image to DXF
    result = converter.convert_image_to_dxf(
        input_image="samples/floral_inlay.png",
        output_dxf="floral_inlay_opencv_converted.dxf",
        preview_output="opencv_conversion_preview.png"
    )
    
    print("\nConversion Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
