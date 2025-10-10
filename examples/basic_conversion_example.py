#!/usr/bin/env python3
"""
Basic Image to DXF Conversion Example
=====================================

This example shows how to use the different image-to-DXF converters
available in the WJP ANALYSER project.
"""

import os
from pathlib import Path

def example_inkscape_conversion():
    """Example using Inkscape converter (recommended)."""
    print("🎨 Inkscape Conversion Example")
    print("-" * 40)
    
    try:
        from image_processing.converters.inkscape_converter import InkscapeImageToDXFConverter
        
        # Create converter with professional settings
        converter = InkscapeImageToDXFConverter(
            trace_method="autotrace",
            threshold=0.5,
            simplify=0.1,
            smooth_corners=True
        )
        
        # Convert image
        result = converter.convert_image_to_dxf(
            input_image="data/samples/floral_inlay.png",
            output_dxf="output/dxf/floral_inlay_inkscape.dxf",
            preview_output="output/reports/floral_inlay_preview.png"
        )
        
        print(f"✅ Conversion successful!")
        print(f"   DXF: {result['output_dxf']}")
        print(f"   Entities: {result['total_entities']}")
        
    except Exception as e:
        print(f"❌ Inkscape conversion failed: {e}")
        print("   Make sure Inkscape is installed and accessible")

def example_opencv_conversion():
    """Example using OpenCV converter."""
    print("\n🔧 OpenCV Conversion Example")
    print("-" * 40)
    
    try:
        from image_processing.converters.opencv_converter import OpenCVImageToDXFConverter
        
        # Create converter
        converter = OpenCVImageToDXFConverter(
            binary_threshold=127,
            min_area=100,
            dxf_size=1000.0
        )
        
        # Convert image
        result = converter.convert_image_to_dxf(
            input_image="data/samples/floral_inlay.png",
            output_dxf="output/dxf/floral_inlay_opencv.dxf",
            preview_output="output/reports/floral_inlay_opencv_preview.png"
        )
        
        print(f"✅ Conversion successful!")
        print(f"   DXF: {result['output_dxf']}")
        print(f"   Contours: {result['contours_kept']}")
        
    except Exception as e:
        print(f"❌ OpenCV conversion failed: {e}")

def example_complete_pipeline():
    """Example of complete image-to-waterjet pipeline."""
    print("\n🚀 Complete Pipeline Example")
    print("-" * 40)
    
    try:
        from image_processing.converters.inkscape_converter import InkscapeImageToDXFConverter
        from core.api import AnalyzeArgs, analyze_dxf
        
        # Step 1: Convert image to DXF
        print("Step 1: Converting image to DXF...")
        converter = InkscapeImageToDXFConverter()
        converter.convert_image_to_dxf(
            input_image="data/samples/floral_inlay.png",
            output_dxf="output/dxf/floral_inlay_pipeline.dxf"
        )
        
        # Step 2: Analyze DXF for waterjet cutting
        print("Step 2: Analyzing DXF for waterjet cutting...")
        args = AnalyzeArgs(
            material="Tan Brown Granite",
            thickness=25.0,
            kerf=1.1,
            rate_per_m=825.0,
            out="output/analysis"
        )
        analysis = analyze_dxf("output/dxf/floral_inlay_pipeline.dxf", args)
        
        print(f"✅ Complete pipeline successful!")
        print(f"   Analysis results saved to: output/analysis/")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

def main():
    """Run all examples."""
    print("🎯 WJP ANALYSER - Image to DXF Conversion Examples")
    print("=" * 60)
    
    # Ensure output directories exist
    os.makedirs("output/dxf", exist_ok=True)
    os.makedirs("output/reports", exist_ok=True)
    os.makedirs("output/analysis", exist_ok=True)
    
    # Run examples
    example_inkscape_conversion()
    example_opencv_conversion()
    example_complete_pipeline()
    
    print("\n📁 Check the 'output/' directory for generated files")
    print("🎉 Examples completed!")

if __name__ == "__main__":
    main()
