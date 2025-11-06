#!/usr/bin/env python3
"""
WJP DXF Analyzer Agent with Cutting Report Module
================================================

This agent analyzes DXF files, validates geometry, calculates metrics,
and generates comprehensive cutting reports with CSV/JSON outputs.
"""

import os
import sys
import json
import csv
import numpy as np
import ezdxf
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from wjp_file_manager import WJPFileManager, JobMetadata, ProcessStage, MaterialCode

@dataclass
class CuttingMetrics:
    """Cutting metrics for analysis."""
    total_objects: int
    total_area_mm2: float
    total_cut_length_mtr: float
    cut_cost_inr: float
    machine_time_min: float
    violations: int
    complexity: str
    layer_breakdown: Dict[str, int]

@dataclass
class AnalysisResult:
    """Complete analysis result."""
    design_code: str
    material: str
    thickness_mm: int
    metrics: CuttingMetrics
    report_generated: bool
    output_image: str
    next_stage: str
    timestamp: str

class DXFAnalyzerAgent:
    """DXF Analyzer Agent with comprehensive cutting report capabilities."""
    
    def __init__(self):
        self.file_manager = WJPFileManager()
        self.output_dir = Path("output/dxf_analyzer")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Material cost rates (₹ per meter)
        self.material_rates = {
            "Tan Brown Granite": 850,
            "Marble": 750,
            "Stainless Steel": 1200,
            "Aluminum": 400,
            "Brass": 900,
            "Generic": 600
        }
        
        # Cutting speeds (mm/min)
        self.cutting_speeds = {
            "Tan Brown Granite": 800,
            "Marble": 1000,
            "Stainless Steel": 600,
            "Aluminum": 1200,
            "Brass": 700,
            "Generic": 1000
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_spacing_mm": 3.0,
            "min_radius_mm": 2.0,
            "max_complexity": 0.8
        }
    
    def run(self, conversion_metadata_path: str) -> Tuple[str, str, str, str]:
        """
        Analyze DXF file and generate cutting report.
        
        Args:
            conversion_metadata_path: Path to conversion metadata JSON file
            
        Returns:
            Tuple of (analysis_dxf_path, analysis_json_path, analysis_image_path, csv_path)
        """
        print(f"📊 **DXF Analyzer Agent - Processing**")
        
        # Load conversion metadata
        conversion_metadata = self.file_manager.load_metadata(conversion_metadata_path)
        
        print(f"   Design Code: {conversion_metadata.design_code}")
        print(f"   Material: {conversion_metadata.material}")
        print(f"   Input DXF: {getattr(conversion_metadata, 'output_file', 'Unknown')}")
        
        # Find the DXF file
        dxf_path = self._find_dxf_file(conversion_metadata)
        
        if not dxf_path or not os.path.exists(dxf_path):
            raise FileNotFoundError(f"DXF file not found for design {conversion_metadata.design_code}")
        
        print(f"   Analyzing DXF: {os.path.basename(dxf_path)}")
        
        # Analyze DXF file
        analysis_result = self._analyze_dxf_file(dxf_path, conversion_metadata)
        
        # Generate analysis DXF (cleaned version)
        analysis_dxf_path = self._create_analysis_dxf(dxf_path, analysis_result, conversion_metadata)
        
        # Generate analysis image (visualization)
        analysis_image_path = self._create_analysis_image(dxf_path, analysis_result, conversion_metadata)
        
        # Generate CSV report
        csv_path = self._create_csv_report(analysis_result, conversion_metadata)
        
        # Generate JSON report
        analysis_json_path = self._create_json_report(analysis_result, conversion_metadata)
        
        print(f"✅ **DXF Analyzer Agent Complete**")
        print(f"   Analysis DXF: {os.path.basename(analysis_dxf_path)}")
        print(f"   Analysis JSON: {os.path.basename(analysis_json_path)}")
        print(f"   Analysis Image: {os.path.basename(analysis_image_path)}")
        print(f"   CSV Report: {os.path.basename(csv_path)}")
        
        return analysis_dxf_path, analysis_json_path, analysis_image_path, csv_path
    
    def _find_dxf_file(self, metadata: JobMetadata) -> Optional[str]:
        """Find the DXF file corresponding to the metadata."""
        design_code = metadata.design_code
        material_code = self._get_material_code(metadata.material)
        
        # Generate expected DXF filename
        dxf_filename = self.file_manager.generate_filename(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=metadata.thickness_mm,
            process_stage=ProcessStage.RAW,
            version="V1",
            extension="dxf"
        )
        
        # Look in converted_dxf folder
        converted_folder = self.file_manager.get_stage_folder(design_code, "converted_dxf")
        dxf_path = converted_folder / dxf_filename
        
        return str(dxf_path) if dxf_path.exists() else None
    
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
    
    def _analyze_dxf_file(self, dxf_path: str, metadata: JobMetadata) -> AnalysisResult:
        """Analyze DXF file and calculate metrics."""
        
        # Load DXF file
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        # Initialize analysis variables
        total_objects = 0
        total_area_mm2 = 0.0
        total_cut_length_mtr = 0.0
        violations = 0
        layer_breakdown = {}
        
        # Analyze each entity
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE':
                total_objects += 1
                
                # Calculate area and perimeter
                area, perimeter = self._calculate_polyline_metrics(entity)
                total_area_mm2 += area
                total_cut_length_mtr += perimeter / 1000.0  # Convert mm to meters
                
                # Check for violations
                entity_violations = self._check_geometry_violations(entity, metadata)
                violations += entity_violations
                
                # Count by layer
                layer = entity.dxf.layer
                layer_breakdown[layer] = layer_breakdown.get(layer, 0) + 1
        
        # Calculate costs and time
        material_rate = self.material_rates.get(metadata.material, 600)
        cutting_speed = self.cutting_speeds.get(metadata.material, 1000)
        
        cut_cost_inr = total_cut_length_mtr * material_rate
        machine_time_min = total_cut_length_mtr * 1000 / cutting_speed  # Convert m to mm
        
        # Determine complexity
        complexity = self._determine_complexity(total_objects, total_area_mm2, violations)
        
        # Create metrics
        metrics = CuttingMetrics(
            total_objects=total_objects,
            total_area_mm2=total_area_mm2,
            total_cut_length_mtr=total_cut_length_mtr,
            cut_cost_inr=cut_cost_inr,
            machine_time_min=machine_time_min,
            violations=violations,
            complexity=complexity,
            layer_breakdown=layer_breakdown
        )
        
        # Create analysis result
        analysis_result = AnalysisResult(
            design_code=metadata.design_code,
            material=metadata.material,
            thickness_mm=metadata.thickness_mm,
            metrics=metrics,
            report_generated=True,
            output_image="",  # Will be set when image is created
            next_stage="report_generator",
            timestamp=datetime.now().isoformat()
        )
        
        return analysis_result
    
    def _calculate_polyline_metrics(self, polyline) -> Tuple[float, float]:
        """Calculate area and perimeter for a polyline."""
        try:
            # Get points
            points = list(polyline.get_points())
            
            if len(points) < 3:
                return 0.0, 0.0
            
            # Calculate perimeter
            perimeter = 0.0
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                perimeter += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Calculate area using shoelace formula
            area = 0.0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            area = abs(area) / 2.0
            
            return area, perimeter
            
        except Exception as e:
            print(f"Warning: Error calculating metrics for polyline: {e}")
            return 0.0, 0.0
    
    def _check_geometry_violations(self, polyline, metadata: JobMetadata) -> int:
        """Check for geometry violations."""
        violations = 0
        
        try:
            # Check minimum spacing
            min_spacing = metadata.cut_spacing_mm
            
            # Check minimum radius
            min_radius = metadata.min_radius_mm
            
            # Get points
            points = list(polyline.get_points())
            
            # Check for sharp corners (radius violations)
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                p3 = points[(i + 2) % len(points)]
                
                # Calculate angle at p2
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    
                    # If angle is too sharp, it violates radius requirement
                    if angle < np.pi / 6:  # 30 degrees
                        violations += 1
            
        except Exception as e:
            print(f"Warning: Error checking violations: {e}")
        
        return violations
    
    def _determine_complexity(self, total_objects: int, total_area_mm2: float, violations: int) -> str:
        """Determine complexity level."""
        # Simple complexity calculation
        complexity_score = (total_objects * 0.1) + (violations * 0.5) + (total_area_mm2 / 100000)
        
        if complexity_score < 0.3:
            return "Low"
        elif complexity_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _create_analysis_dxf(self, original_dxf_path: str, analysis_result: AnalysisResult, 
                           metadata: JobMetadata) -> str:
        """Create cleaned analysis DXF file."""
        
        # Generate analysis DXF filename
        material_code = self._get_material_code(metadata.material)
        analysis_dxf_path = self.file_manager.get_file_path(
            design_code=metadata.design_code,
            material_code=material_code,
            thickness_mm=metadata.thickness_mm,
            process_stage=ProcessStage.ANALYSIS,
            stage_folder="analyzed",
            version="V1",
            extension="dxf"
        )
        
        # Copy original DXF and add analysis information
        doc = ezdxf.readfile(original_dxf_path)
        
        # Add analysis metadata as text
        msp = doc.modelspace()
        
        # Add analysis summary text
        summary_text = f"""ANALYSIS SUMMARY
Design: {analysis_result.design_code}
Material: {analysis_result.material}
Objects: {analysis_result.metrics.total_objects}
Cut Length: {analysis_result.metrics.total_cut_length_mtr:.2f}m
Cost: ₹{analysis_result.metrics.cut_cost_inr:.0f}
Time: {analysis_result.metrics.machine_time_min:.1f}min
Violations: {analysis_result.metrics.violations}
Complexity: {analysis_result.metrics.complexity}"""
        
        msp.add_text(summary_text, dxfattribs={'height': 5, 'layer': 'ANALYSIS'})
        
        # Save analysis DXF
        doc.saveas(analysis_dxf_path)
        
        return analysis_dxf_path
    
    def _create_analysis_image(self, dxf_path: str, analysis_result: AnalysisResult, 
                              metadata: JobMetadata) -> str:
        """Create analysis visualization image."""
        
        # Generate analysis image filename
        material_code = self._get_material_code(metadata.material)
        analysis_image_path = self.file_manager.get_file_path(
            design_code=metadata.design_code,
            material_code=material_code,
            thickness_mm=metadata.thickness_mm,
            process_stage=ProcessStage.ANALYSIS,
            stage_folder="analyzed",
            version="V1",
            extension="png"
        )
        
        # Load DXF file
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color mapping for layers
        layer_colors = {
            'OUTER': 'red',
            'COMPLEX': 'yellow',
            'DECOR': 'green',
            'UNKNOWN': 'blue',
            '0': 'black'
        }
        
        # Plot entities
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE':
                layer = entity.dxf.layer
                color = layer_colors.get(layer, 'black')
                
                # Get points
                points = list(entity.get_points())
                if len(points) > 1:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    # Close the polygon if needed
                    if entity.closed and len(points) > 2:
                        x_coords.append(x_coords[0])
                        y_coords.append(y_coords[0])
                    
                    ax.plot(x_coords, y_coords, color=color, linewidth=1, alpha=0.7)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"DXF Analysis - {analysis_result.design_code}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        
        # Add legend
        legend_elements = []
        for layer, color in layer_colors.items():
            if layer in analysis_result.metrics.layer_breakdown:
                count = analysis_result.metrics.layer_breakdown[layer]
                legend_elements.append(plt.Line2D([0], [0], color=color, label=f"{layer} ({count})"))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        # Add analysis summary text
        summary_text = f"""Analysis Summary:
Objects: {analysis_result.metrics.total_objects}
Cut Length: {analysis_result.metrics.total_cut_length_mtr:.2f}m
Cost: ₹{analysis_result.metrics.cut_cost_inr:.0f}
Time: {analysis_result.metrics.machine_time_min:.1f}min
Violations: {analysis_result.metrics.violations}
Complexity: {analysis_result.metrics.complexity}"""
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save image
        plt.tight_layout()
        plt.savefig(analysis_image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Update analysis result with image path
        analysis_result.output_image = os.path.basename(analysis_image_path)
        
        return analysis_image_path
    
    def _create_csv_report(self, analysis_result: AnalysisResult, metadata: JobMetadata) -> str:
        """Create CSV report."""
        
        # Generate CSV filename
        material_code = self._get_material_code(metadata.material)
        csv_path = self.file_manager.get_file_path(
            design_code=metadata.design_code,
            material_code=material_code,
            thickness_mm=metadata.thickness_mm,
            process_stage=ProcessStage.ANALYSIS,
            stage_folder="analyzed",
            version="V1",
            extension="csv"
        )
        
        # Create CSV data
        csv_data = [
            ["Parameter", "Value"],
            ["Design Code", analysis_result.design_code],
            ["Material", analysis_result.material],
            ["Thickness (mm)", analysis_result.thickness_mm],
            ["Total Objects", analysis_result.metrics.total_objects],
            ["Total Area (mm²)", f"{analysis_result.metrics.total_area_mm2:.2f}"],
            ["Cut Length (mtr)", f"{analysis_result.metrics.total_cut_length_mtr:.2f}"],
            ["Cost (₹)", f"{analysis_result.metrics.cut_cost_inr:.2f}"],
            ["Machine Time (min)", f"{analysis_result.metrics.machine_time_min:.2f}"],
            ["Violations", analysis_result.metrics.violations],
            ["Complexity", analysis_result.metrics.complexity],
            ["", ""],  # Empty row
            ["Layer Breakdown", ""],
        ]
        
        # Add layer breakdown
        for layer, count in analysis_result.metrics.layer_breakdown.items():
            csv_data.append([f"  {layer}", count])
        
        # Write CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        
        return csv_path
    
    def _create_json_report(self, analysis_result: AnalysisResult, metadata: JobMetadata) -> str:
        """Create JSON report."""
        
        # Generate JSON filename
        material_code = self._get_material_code(metadata.material)
        json_path = self.file_manager.get_file_path(
            design_code=metadata.design_code,
            material_code=material_code,
            thickness_mm=metadata.thickness_mm,
            process_stage=ProcessStage.ANALYSIS,
            stage_folder="analyzed",
            version="V1",
            extension="json"
        )
        
        # Create JSON data
        json_data = {
            "design_code": analysis_result.design_code,
            "material": analysis_result.material,
            "thickness_mm": analysis_result.thickness_mm,
            "cut_length_mtr": analysis_result.metrics.total_cut_length_mtr,
            "cut_cost_inr": analysis_result.metrics.cut_cost_inr,
            "violations": analysis_result.metrics.violations,
            "complexity": analysis_result.metrics.complexity,
            "machine_time_min": analysis_result.metrics.machine_time_min,
            "total_objects": analysis_result.metrics.total_objects,
            "total_area_mm2": analysis_result.metrics.total_area_mm2,
            "layer_breakdown": analysis_result.metrics.layer_breakdown,
            "report_generated": analysis_result.report_generated,
            "output_image": analysis_result.output_image,
            "next_stage": analysis_result.next_stage,
            "timestamp": analysis_result.timestamp
        }
        
        # Write JSON file
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        return json_path

def test_dxf_analyzer_agent():
    """Test the DXF Analyzer Agent."""
    print("📊 **Testing DXF Analyzer Agent**")
    print("=" * 50)
    
    # Create agent
    agent = DXFAnalyzerAgent()
    
    # First, create test files using previous agents
    print("📋 **Creating test files using previous agents...**")
    
    from wjp_designer_agent import DesignerAgent
    from wjp_image_to_dxf_agent import ImageToDXFAgent
    
    # Create design
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
        print(f"   ✅ Test design created")
        
        # Convert to DXF
        image_to_dxf = ImageToDXFAgent()
        dxf_path, conversion_metadata_path = image_to_dxf.run(metadata_path)
        print(f"   ✅ DXF conversion completed")
        
        # Now test DXF analysis
        print(f"\n📊 **Testing DXF analysis...**")
        
        analysis_dxf_path, analysis_json_path, analysis_image_path, csv_path = agent.run(conversion_metadata_path)
        
        print(f"   ✅ Analysis DXF: {os.path.basename(analysis_dxf_path)}")
        print(f"   ✅ Analysis JSON: {os.path.basename(analysis_json_path)}")
        print(f"   ✅ Analysis Image: {os.path.basename(analysis_image_path)}")
        print(f"   ✅ CSV Report: {os.path.basename(csv_path)}")
        
        # Verify files exist
        files_to_check = [analysis_dxf_path, analysis_json_path, analysis_image_path, csv_path]
        all_exist = all(os.path.exists(f) for f in files_to_check)
        
        if all_exist:
            print(f"   ✅ All files verified successfully")
            
            # Check JSON content
            with open(analysis_json_path, 'r') as f:
                json_data = json.load(f)
                print(f"   ✅ JSON contains {len(json_data)} fields")
                print(f"   ✅ Cut Length: {json_data.get('cut_length_mtr', 0):.2f}m")
                print(f"   ✅ Cost: ₹{json_data.get('cut_cost_inr', 0):.0f}")
        else:
            print(f"   ❌ Some files missing")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 **DXF Analyzer Agent Test Completed!**")
    
    return agent

if __name__ == "__main__":
    test_dxf_analyzer_agent()
