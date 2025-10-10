"""
Integrated Image-to-DXF Pipeline
===============================

Combines the OpenCV converter with the waterjet analysis pipeline.
This provides a complete workflow from image to waterjet-ready DXF.
"""

import os
import sys
from pathlib import Path

# Import from reorganized modules
from .converters.opencv_converter import OpenCVImageToDXFConverter
from ..analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf

class IntegratedImageToWaterjetPipeline:
    """Complete pipeline from image to waterjet analysis."""
    
    def __init__(self, 
                 binary_threshold: int = 127,
                 line_pitch: float = 1.0,
                 min_area: int = 100,
                 dxf_size: float = 1000.0,
                 material: str = "Tan Brown Granite",
                 thickness: float = 25.0,
                 kerf: float = 1.1,
                 rate_per_m: float = 825.0):
        """
        Initialize the integrated pipeline.
        
        Args:
            binary_threshold: Image processing threshold (0-255)
            line_pitch: Line simplification factor
            min_area: Minimum contour area
            dxf_size: DXF canvas size in mm
            material: Material name for analysis
            thickness: Material thickness in mm
            kerf: Kerf width in mm
            rate_per_m: Cutting rate per meter
        """
        self.converter = OpenCVImageToDXFConverter(
            binary_threshold=binary_threshold,
            line_pitch=line_pitch,
            min_area=min_area,
            dxf_size=dxf_size
        )
        
        self.analysis_args = AnalyzeArgs(
            material=material,
            thickness=thickness,
            kerf=kerf,
            rate_per_m=rate_per_m,
            out="output/analysis"
        )
    
    def process_image_to_waterjet(self, 
                                 input_image: str,
                                 output_dir: str = "output") -> dict:
        """
        Complete pipeline: Image -> DXF -> Waterjet Analysis.
        
        Args:
            input_image: Path to input image
            output_dir: Output directory for all files
            
        Returns:
            Dictionary with complete processing results
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        base_name = Path(input_image).stem
        dxf_path = os.path.join(output_dir, f"{base_name}_converted.dxf")
        preview_path = os.path.join(output_dir, f"{base_name}_preview.png")
        
        print(f"  Processing image: {input_image}")
        
        # Step 1: Convert image to DXF
        print(" Converting image to DXF...")
        conversion_result = self.converter.convert_image_to_dxf(
            input_image=input_image,
            output_dxf=dxf_path,
            preview_output=preview_path
        )
        
        # Step 2: Analyze DXF for waterjet cutting
        print(" Analyzing DXF for waterjet cutting...")
        analysis_result = analyze_dxf(dxf_path, self.analysis_args)
        
        # Combine results
        complete_result = {
            "input_image": input_image,
            "conversion": conversion_result,
            "analysis": analysis_result,
            "output_files": {
                "dxf": dxf_path,
                "preview": preview_path,
                "report": os.path.join(output_dir, "report.json"),
                "lengths": os.path.join(output_dir, "lengths.csv"),
                "gcode": os.path.join(output_dir, "program.nc")
            }
        }
        
        print(" Processing complete!")
        print(f" DXF file: {dxf_path}")
        print(f"  Preview: {preview_path}")
        print(f" Report: {complete_result['output_files']['report']}")
        
        return complete_result


def main():
    """Example usage of the integrated pipeline."""
    pipeline = IntegratedImageToWaterjetPipeline(
        binary_threshold=127,  # Adjustable threshold
        min_area=100,         # Filter small contours
        material="Tan Brown Granite",
        thickness=25.0,
        kerf=1.1,
        rate_per_m=825.0
    )
    
    # Process image
    result = pipeline.process_image_to_waterjet(
        input_image="Tile_1.png",
        output_dir="output/integrated"
    )
    
    print("\n Complete Pipeline Results:")
    print(f"  Contours found: {result['conversion']['contours_found']}")
    print(f"  Contours kept: {result['conversion']['contours_kept']}")
    print(f"  Polygons: {result['conversion']['polygons']}")
    print(f"  Outer polygons: {result['conversion']['outer_polygons']}")
    print(f"  Inner polygons: {result['conversion']['inner_polygons']}")


if __name__ == "__main__":
    main()
