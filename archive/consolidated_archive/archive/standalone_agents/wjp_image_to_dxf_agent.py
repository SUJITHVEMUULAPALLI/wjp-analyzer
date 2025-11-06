#!/usr/bin/env python3
"""
WJP Image to DXF Agent
======================

This agent converts images to DXF files using metadata from the Designer Agent
and creates metadata for the next stage in the pipeline.
"""

import os
import sys
import json
import numpy as np
import cv2
import ezdxf
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from wjp_file_manager import WJPFileManager, JobMetadata, ProcessStage, MaterialCode

class ImageToDXFAgent:
    """Agent for converting images to DXF files."""
    
    def __init__(self):
        self.file_manager = WJPFileManager()
        self.output_dir = Path("output/image_to_dxf")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection parameters optimized for waterjet cutting
        self.detection_params = {
            "min_area": 25,
            "min_circularity": 0.03,
            "min_solidity": 0.05,
            "simplify_tolerance": 0.0,
            "merge_distance": 0.0
        }
    
    def run(self, metadata_path: str) -> Tuple[str, str]:
        """
        Convert image to DXF using metadata from Designer Agent.
        
        Args:
            metadata_path: Path to metadata JSON file from Designer Agent
            
        Returns:
            Tuple of (dxf_path, metadata_path)
        """
        print(f"🔄 **Image to DXF Agent - Processing**")
        
        # Load metadata from Designer Agent
        designer_metadata = self.file_manager.load_metadata(metadata_path)
        
        print(f"   Design Code: {designer_metadata.design_code}")
        print(f"   Material: {designer_metadata.material}")
        print(f"   Category: {designer_metadata.category}")
        
        # Find the corresponding image file
        image_path = self._find_image_file(designer_metadata)
        
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found for design {designer_metadata.design_code}")
        
        print(f"   Input Image: {os.path.basename(image_path)}")
        
        # Convert image to DXF
        dxf_path = self._convert_image_to_dxf(image_path, designer_metadata)
        
        # Create conversion metadata
        conversion_metadata = self._create_conversion_metadata(designer_metadata, image_path, dxf_path)
        
        # Save conversion metadata
        conversion_metadata_path = self.file_manager.save_metadata(
            designer_metadata.design_code, 
            conversion_metadata, 
            "converted_dxf"
        )
        
        print(f"✅ **Image to DXF Agent Complete**")
        print(f"   DXF: {os.path.basename(dxf_path)}")
        print(f"   Metadata: {os.path.basename(conversion_metadata_path)}")
        
        return dxf_path, conversion_metadata_path
    
    def _find_image_file(self, metadata: JobMetadata) -> Optional[str]:
        """Find the image file corresponding to the metadata."""
        design_code = metadata.design_code
        material_code = self._get_material_code(metadata.material)
        
        # Generate expected image filename
        image_filename = self.file_manager.generate_filename(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=25,  # Default for design stage
            process_stage=ProcessStage.DESIGN,
            version="V1",
            extension="png"
        )
        
        # Look in designer folder
        designer_folder = self.file_manager.get_stage_folder(design_code, "designer")
        image_path = designer_folder / image_filename
        
        return str(image_path) if image_path.exists() else None
    
    def _get_material_code(self, material: str) -> str:
        """Get material code from material name."""
        material_mapping = {
            "Tan Brown Granite": "TANB",
            "Marble": "MARB",
            "Stainless Steel": "STST",
            "Aluminum": "ALUM",
            "Brass": "BRAS",
            "Generic": "GENE"
        }
        return material_mapping.get(material, "GENE")
    
    def _convert_image_to_dxf(self, image_path: str, metadata: JobMetadata) -> str:
        """Convert image to DXF file."""
        
        # Load image
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply threshold - use INV because DesignerAgent creates white lines on dark background
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on detection parameters
        filtered_contours = self._filter_contours(contours)
        
        # Calculate scale factor
        scale_mm_per_px = self._calculate_scale_factor(metadata)
        
        # Create DXF file
        dxf_path = self._create_dxf_file(filtered_contours, scale_mm_per_px, metadata)
        
        return dxf_path
    
    def _filter_contours(self, contours: List) -> List:
        """Filter contours based on detection parameters."""
        filtered = []
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < self.detection_params["min_area"]:
                continue
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < self.detection_params["min_circularity"]:
                    continue
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.detection_params["min_solidity"]:
                    continue
            
            filtered.append(contour)
        
        return filtered
    
    def _calculate_scale_factor(self, metadata: JobMetadata) -> float:
        """Calculate scale factor from metadata dimensions."""
        # Convert inches to mm
        width_mm = metadata.dimensions_inch[0] * 25.4
        height_mm = metadata.dimensions_inch[1] * 25.4
        
        # Assume image is 100 DPI (100 pixels per inch)
        width_px = metadata.dimensions_inch[0] * 100
        height_px = metadata.dimensions_inch[1] * 100
        
        # Calculate scale factor
        scale_x = width_mm / width_px
        scale_y = height_mm / height_px
        
        # Use average scale factor
        scale_mm_per_px = (scale_x + scale_y) / 2
        
        return scale_mm_per_px
    
    def _create_dxf_file(self, contours: List, scale_mm_per_px: float, metadata: JobMetadata) -> str:
        """Create DXF file from contours."""
        
        # Generate DXF filename
        material_code = self._get_material_code(metadata.material)
        dxf_path = self.file_manager.get_file_path(
            design_code=metadata.design_code,
            material_code=material_code,
            thickness_mm=metadata.thickness_mm,
            process_stage=ProcessStage.RAW,
            stage_folder="converted_dxf",
            version="V1",
            extension="dxf"
        )
        
        # Create DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Add layers
        doc.layers.new('OUTER', dxfattribs={'color': 1})  # Red
        doc.layers.new('COMPLEX', dxfattribs={'color': 2})  # Yellow
        doc.layers.new('DECOR', dxfattribs={'color': 3})  # Green
        doc.layers.new('UNKNOWN', dxfattribs={'color': 7})  # White
        
        # Process contours
        total_contours = len(contours)
        open_contours_fixed = 0
        
        for i, contour in enumerate(contours):
            # Convert contour to points
            points = []
            for point in contour:
                x = point[0][0] * scale_mm_per_px
                y = point[0][1] * scale_mm_per_px
                points.append((x, y))
            
            # Determine layer based on contour properties
            layer = self._determine_layer(contour, scale_mm_per_px)
            
            # Create polyline
            polyline = msp.add_lwpolyline(points, dxfattribs={'layer': layer})
            
            # Check if polyline is closed
            if not polyline.closed and len(points) > 2:
                # Close the polyline
                points.append(points[0])
                polyline.set_points(points)
                polyline.closed = True
                open_contours_fixed += 1
        
        # Save DXF file
        doc.saveas(dxf_path)
        
        # Store conversion statistics
        self._conversion_stats = {
            "total_contours": total_contours,
            "open_contours_fixed": open_contours_fixed,
            "scale_mm_per_px": scale_mm_per_px
        }
        
        return dxf_path
    
    def _determine_layer(self, contour, scale_mm_per_px: float) -> str:
        """Determine appropriate layer for contour."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Convert to mm²
        area_mm2 = area * (scale_mm_per_px ** 2)
        perimeter_mm = perimeter * scale_mm_per_px
        
        # Layer classification logic
        if area_mm2 > 10000 and perimeter_mm > 500:
            return "OUTER"  # Large boundary objects
        elif area_mm2 < 1000 and perimeter_mm < 100:
            return "DECOR"  # Small decorative elements
        else:
            return "COMPLEX"  # Complex geometric objects
    
    def _create_conversion_metadata(self, designer_metadata: JobMetadata, 
                                   image_path: str, dxf_path: str) -> JobMetadata:
        """Create conversion metadata."""
        
        # Get conversion statistics
        stats = getattr(self, '_conversion_stats', {})
        
        # Create new metadata
        conversion_metadata = JobMetadata(
            design_code=designer_metadata.design_code,
            material=designer_metadata.material,
            thickness_mm=designer_metadata.thickness_mm,
            category=designer_metadata.category,
            dimensions_inch=designer_metadata.dimensions_inch,
            cut_spacing_mm=designer_metadata.cut_spacing_mm,
            min_radius_mm=designer_metadata.min_radius_mm,
            prompt_used=designer_metadata.prompt_used,
            next_stage="analyze_dxf",
            timestamp=datetime.now().isoformat()
        )
        
        # Add conversion-specific data
        conversion_data = {
            "input_image": os.path.basename(image_path),
            "scale_mm_per_px": stats.get("scale_mm_per_px", 0.0),
            "total_contours": stats.get("total_contours", 0),
            "open_contours_fixed": stats.get("open_contours_fixed", 0),
            "cleaning_status": "complete",
            "output_file": os.path.basename(dxf_path)
        }
        
        # Store additional data in metadata (extend the dataclass)
        conversion_metadata.__dict__.update(conversion_data)
        
        return conversion_metadata

def test_image_to_dxf_agent():
    """Test the Image to DXF Agent."""
    print("🔄 **Testing Image to DXF Agent**")
    print("=" * 50)
    
    # Create agent
    agent = ImageToDXFAgent()
    
    # First, create a test design using Designer Agent
    from wjp_designer_agent import DesignerAgent
    
    print("📋 **Creating test design first...**")
    designer = DesignerAgent()
    
    test_case = {
        "job_id": "SR06",
        "prompt": "Waterjet-safe Tan Brown granite tile with white marble inlay, 24x24 inch",
        "material": "Tan Brown Granite",
        "thickness_mm": 25,
        "category": "Inlay Tile",
        "dimensions_inch": [24, 24]
    }
    
    try:
        image_path, metadata_path = designer.run(**test_case)
        print(f"   ✅ Test design created: {os.path.basename(image_path)}")
        
        # Now test Image to DXF conversion
        print(f"\n🔄 **Testing Image to DXF conversion...**")
        
        dxf_path, conversion_metadata_path = agent.run(metadata_path)
        
        print(f"   ✅ DXF generated: {os.path.basename(dxf_path)}")
        print(f"   ✅ Conversion metadata: {os.path.basename(conversion_metadata_path)}")
        
        # Verify files exist
        if os.path.exists(dxf_path) and os.path.exists(conversion_metadata_path):
            print(f"   ✅ Files verified successfully")
            
            # Check DXF file
            try:
                doc = ezdxf.readfile(dxf_path)
                entities = list(doc.modelspace())
                print(f"   ✅ DXF contains {len(entities)} entities")
            except Exception as e:
                print(f"   ⚠️ DXF verification warning: {e}")
        else:
            print(f"   ❌ File verification failed")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 **Image to DXF Agent Test Completed!**")
    
    return agent

if __name__ == "__main__":
    test_image_to_dxf_agent()
