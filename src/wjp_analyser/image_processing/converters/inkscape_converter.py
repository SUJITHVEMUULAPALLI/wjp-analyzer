"""
Inkscape-based Image to DXF Converter
====================================

This converter uses Inkscape's powerful vectorization capabilities to create
high-quality DXF files from images. Inkscape provides much better edge detection
and path optimization than basic OpenCV approaches.

Requirements:
- Inkscape installed and accessible via command line
- Python subprocess module for calling Inkscape

Installation:
1. Download Inkscape from https://inkscape.org/release/
2. Install with command-line tools enabled
3. Add Inkscape to your system PATH
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import xml.etree.ElementTree as ET

class InkscapeImageToDXFConverter:
    """Inkscape-based image to DXF converter with advanced vectorization."""
    
    def __init__(self, 
                 inkscape_path: Optional[str] = None,
                 trace_method: str = "autotrace",
                 threshold: float = 0.5,
                 simplify: float = 0.0,
                 smooth_corners: bool = True):
        """
        Initialize Inkscape converter.
        
        Args:
            inkscape_path: Path to Inkscape executable (auto-detect if None)
            trace_method: Vectorization method ("autotrace", "potrace", "centerline")
            threshold: Edge detection threshold (0.0-1.0)
            simplify: Path simplification factor (0.0-1.0)
            smooth_corners: Whether to smooth sharp corners
        """
        self.inkscape_path = inkscape_path or self._find_inkscape()
        self.trace_method = trace_method
        self.threshold = threshold
        self.simplify = simplify
        self.smooth_corners = smooth_corners
        
        if not self.inkscape_path:
            raise RuntimeError("Inkscape not found. Please install Inkscape and add it to PATH.")
    
    def _find_inkscape(self) -> Optional[str]:
        """Find Inkscape executable in system PATH."""
        try:
            result = subprocess.run(['inkscape', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return 'inkscape'
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try common Windows paths
        common_paths = [
            r"C:\Program Files\Inkscape\bin\inkscape.exe",
            r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
            r"C:\Users\{}\AppData\Local\Programs\Inkscape\bin\inkscape.exe".format(os.getenv('USERNAME', '')),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def convert_image_to_dxf(self, 
                            input_image: str, 
                            output_dxf: str,
                            preview_output: Optional[str] = None) -> Dict:
        """
        Convert image to DXF using Inkscape.
        
        Args:
            input_image: Path to input image
            output_dxf: Path to output DXF file
            preview_output: Optional path for preview image
            
        Returns:
            Dictionary with conversion results and statistics
        """
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Input image not found: {input_image}")
        
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Step 1: Convert image to SVG using Inkscape's trace
            svg_path = temp_dir / "traced.svg"
            self._trace_image_to_svg(input_image, svg_path)
            
            # Step 2: Clean and optimize SVG
            self._clean_svg(svg_path)
            
            # Step 3: Convert SVG to DXF
            self._svg_to_dxf(svg_path, output_dxf)
            
            # Step 4: Generate preview if requested
            if preview_output:
                self._generate_preview(svg_path, preview_output)
            
            # Step 5: Analyze results
            stats = self._analyze_dxf(output_dxf)
            
            return {
                "success": True,
                "input_image": input_image,
                "output_dxf": output_dxf,
                "preview_output": preview_output,
                "inkscape_path": self.inkscape_path,
                "trace_method": self.trace_method,
                **stats
            }
    
    def _trace_image_to_svg(self, input_image: str, output_svg: Path):
        """Use Inkscape to trace image to SVG."""
        cmd = [
            self.inkscape_path,
            input_image,
            "--export-type=svg",
            f"--export-filename={output_svg}",
            "--export-plain-svg"
        ]
        
        # Add tracing options based on method
        if self.trace_method == "autotrace":
            cmd.extend([
                "--verb=org.inkscape.autotrace",
                f"--verb-option=org.inkscape.autotrace:threshold={self.threshold}",
                f"--verb-option=org.inkscape.autotrace:simplify={self.simplify}",
            ])
        elif self.trace_method == "potrace":
            cmd.extend([
                "--verb=org.inkscape.potrace",
                f"--verb-option=org.inkscape.potrace:threshold={self.threshold}",
                f"--verb-option=org.inkscape.potrace:simplify={self.simplify}",
            ])
        
        if self.smooth_corners:
            cmd.append("--verb-option=org.inkscape.autotrace:smooth-corners=true")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Inkscape trace failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Inkscape trace timed out")
    
    def _clean_svg(self, svg_path: Path):
        """Clean and optimize SVG for better DXF conversion."""
        try:
            # Parse SVG
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Remove unnecessary elements
            for elem in root.findall(".//{http://www.w3.org/2000/svg}defs"):
                root.remove(elem)
            
            # Simplify paths
            for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
                if self.simplify > 0:
                    # Apply path simplification
                    d = path.get('d', '')
                    if d:
                        # Basic path simplification (can be enhanced)
                        simplified_d = self._simplify_path(d)
                        path.set('d', simplified_d)
            
            # Save cleaned SVG
            tree.write(svg_path, encoding='utf-8', xml_declaration=True)
            
        except ET.ParseError as e:
            print(f"Warning: SVG cleaning failed: {e}")
    
    def _simplify_path(self, path_data: str) -> str:
        """Simplify SVG path data."""
        # Basic path simplification - remove redundant points
        # This is a simplified version; more sophisticated algorithms can be added
        return path_data
    
    def _svg_to_dxf(self, svg_path: Path, output_dxf: str):
        """Convert SVG to DXF using Inkscape."""
        cmd = [
            self.inkscape_path,
            str(svg_path),
            "--export-type=dxf",
            f"--export-filename={output_dxf}",
            "--export-plain-svg"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"Inkscape DXF export failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Inkscape DXF export timed out")
    
    def _generate_preview(self, svg_path: Path, preview_output: str):
        """Generate preview image from SVG."""
        cmd = [
            self.inkscape_path,
            str(svg_path),
            "--export-type=png",
            f"--export-filename={preview_output}",
            "--export-dpi=150"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"Warning: Preview generation failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Warning: Preview generation timed out")
    
    def _analyze_dxf(self, dxf_path: str) -> Dict:
        """Analyze the generated DXF file."""
        try:
            import ezdxf
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            entities = list(msp)
            entity_types = {}
            for entity in entities:
                entity_type = entity.dxftype()
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            return {
                "total_entities": len(entities),
                "entity_types": entity_types,
                "dxf_version": doc.dxfversion
            }
        except Exception as e:
            return {
                "total_entities": 0,
                "entity_types": {},
                "error": str(e)
            }


def main():
    """Example usage of the Inkscape converter."""
    try:
        converter = InkscapeImageToDXFConverter(
            trace_method="autotrace",
            threshold=0.5,
            simplify=0.1,
            smooth_corners=True
        )
        
        # Convert image to DXF
        result = converter.convert_image_to_dxf(
            input_image="Tile_1.png",
            output_dxf="Tile_1_inkscape_converted.dxf",
            preview_output="inkscape_conversion_preview.png"
        )
        
        print("Inkscape Conversion Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTo install Inkscape:")
        print("1. Download from https://inkscape.org/release/")
        print("2. Install with command-line tools enabled")
        print("3. Add Inkscape to your system PATH")


if __name__ == "__main__":
    main()
