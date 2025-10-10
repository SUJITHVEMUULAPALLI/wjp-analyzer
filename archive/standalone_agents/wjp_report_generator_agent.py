#!/usr/bin/env python3
"""
WJP Report Generator Agent - PDF Compilation
==========================================

This agent compiles all outputs into a professional PDF report.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from wjp_file_manager import WJPFileManager, JobMetadata, ProcessStage, MaterialCode

class ReportGeneratorAgent:
    """Report Generator Agent for creating professional PDF reports."""
    
    def __init__(self):
        self.file_manager = WJPFileManager()
        self.output_dir = Path("output/report_generator")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Header style
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        # Subheader style
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.darkgreen
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        ))
    
    def run(self, analysis_json_path: str) -> str:
        """
        Generate PDF report from analysis results.
        
        Args:
            analysis_json_path: Path to analysis JSON file
            
        Returns:
            Path to generated PDF report
        """
        print(f"📄 **Report Generator Agent - Processing**")
        
        # Load analysis data
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        design_code = analysis_data['design_code']
        print(f"   Design Code: {design_code}")
        
        # Find all related files
        files = self._find_related_files(design_code, analysis_data)
        
        print(f"   Found {len(files)} related files")
        
        # Generate PDF report
        pdf_path = self._create_pdf_report(analysis_data, files)
        
        print(f"✅ **Report Generator Agent Complete**")
        print(f"   PDF Report: {os.path.basename(pdf_path)}")
        
        return pdf_path
    
    def _find_related_files(self, design_code: str, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Find all related files for the design."""
        files = {}
        
        # Get material code
        material = analysis_data['material']
        material_code = self._get_material_code(material)
        thickness_mm = analysis_data['thickness_mm']
        
        # Look for files in different stages
        stages = [
            ("designer", "design", "png"),
            ("converted_dxf", "raw", "dxf"),
            ("analyzed", "analysis", "dxf"),
            ("analyzed", "analysis", "png"),
            ("analyzed", "analysis", "csv")
        ]
        
        for stage_folder, process_stage, extension in stages:
            try:
                file_path = self.file_manager.get_file_path(
                    design_code=design_code,
                    material_code=material_code,
                    thickness_mm=thickness_mm,
                    process_stage=ProcessStage(process_stage.upper()),
                    stage_folder=stage_folder,
                    version="V1",
                    extension=extension
                )
                
                if os.path.exists(file_path):
                    files[f"{stage_folder}_{extension}"] = file_path
                    
            except Exception as e:
                print(f"Warning: Could not find {stage_folder}/{extension} file: {e}")
        
        return files
    
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
    
    def _create_pdf_report(self, analysis_data: Dict[str, Any], files: Dict[str, str]) -> str:
        """Create PDF report."""
        
        # Generate PDF filename
        design_code = analysis_data['design_code']
        material_code = self._get_material_code(analysis_data['material'])
        thickness_mm = analysis_data['thickness_mm']
        
        pdf_path = self.file_manager.get_file_path(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=thickness_mm,
            process_stage=ProcessStage.REPORT,
            stage_folder="reports",
            version="V1",
            extension="pdf"
        )
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # Add title
        title = f"WJP Analysis Report - {design_code}"
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Add header information
        story.append(self._create_header_section(analysis_data))
        story.append(Spacer(1, 20))
        
        # Add images section
        if files:
            story.append(self._create_images_section(files))
            story.append(Spacer(1, 20))
        
        # Add analysis summary
        story.append(self._create_analysis_summary(analysis_data))
        story.append(Spacer(1, 20))
        
        # Add detailed metrics table
        story.append(self._create_metrics_table(analysis_data))
        story.append(Spacer(1, 20))
        
        # Add layer breakdown
        if 'layer_breakdown' in analysis_data:
            story.append(self._create_layer_breakdown(analysis_data['layer_breakdown']))
            story.append(Spacer(1, 20))
        
        # Add footer
        story.append(self._create_footer())
        
        # Build PDF
        doc.build(story)
        
        return pdf_path
    
    def _create_header_section(self, analysis_data: Dict[str, Any]) -> List:
        """Create header section with basic information."""
        elements = []
        
        # Header information
        header_data = [
            ["Design Code:", analysis_data['design_code']],
            ["Material:", analysis_data['material']],
            ["Thickness:", f"{analysis_data['thickness_mm']} mm"],
            ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Report Generated:", "WJP Analyzer System"]
        ]
        
        # Create table
        table = Table(header_data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(Paragraph("Project Information", self.styles['CustomHeading1']))
        elements.append(table)
        
        return elements
    
    def _create_images_section(self, files: Dict[str, str]) -> List:
        """Create images section."""
        elements = []
        
        elements.append(Paragraph("Design Visualizations", self.styles['CustomHeading1']))
        
        # Add design image
        if 'designer_png' in files:
            try:
                img_path = files['designer_png']
                img = RLImage(img_path, width=4*inch, height=4*inch)
                elements.append(Paragraph("Original Design", self.styles['CustomHeading2']))
                elements.append(img)
                elements.append(Spacer(1, 10))
            except Exception as e:
                elements.append(Paragraph(f"Design image not available: {e}", self.styles['CustomBody']))
        
        # Add analysis image
        if 'analyzed_png' in files:
            try:
                img_path = files['analyzed_png']
                img = RLImage(img_path, width=6*inch, height=4*inch)
                elements.append(Paragraph("Analysis Visualization", self.styles['CustomHeading2']))
                elements.append(img)
                elements.append(Spacer(1, 10))
            except Exception as e:
                elements.append(Paragraph(f"Analysis image not available: {e}", self.styles['CustomBody']))
        
        return elements
    
    def _create_analysis_summary(self, analysis_data: Dict[str, Any]) -> List:
        """Create analysis summary section."""
        elements = []
        
        elements.append(Paragraph("Analysis Summary", self.styles['CustomHeading1']))
        
        # Summary text
        summary_text = f"""
        This analysis report provides comprehensive metrics for the waterjet cutting project.
        The design contains {analysis_data.get('total_objects', 0)} objects with a total cut length of 
        {analysis_data.get('cut_length_mtr', 0):.2f} meters. The estimated cost is ₹{analysis_data.get('cut_cost_inr', 0):.0f} 
        with a machine time of {analysis_data.get('machine_time_min', 0):.1f} minutes.
        
        The design complexity is rated as {analysis_data.get('complexity', 'Unknown')} with 
        {analysis_data.get('violations', 0)} geometry violations detected.
        """
        
        elements.append(Paragraph(summary_text.strip(), self.styles['CustomBody']))
        
        return elements
    
    def _create_metrics_table(self, analysis_data: Dict[str, Any]) -> List:
        """Create detailed metrics table."""
        elements = []
        
        elements.append(Paragraph("Detailed Metrics", self.styles['CustomHeading1']))
        
        # Metrics data
        metrics_data = [
            ["Parameter", "Value", "Unit"],
            ["Total Objects", f"{analysis_data.get('total_objects', 0)}", "count"],
            ["Total Area", f"{analysis_data.get('total_area_mm2', 0):.2f}", "mm²"],
            ["Cut Length", f"{analysis_data.get('cut_length_mtr', 0):.2f}", "meters"],
            ["Cutting Cost", f"₹{analysis_data.get('cut_cost_inr', 0):.2f}", "INR"],
            ["Machine Time", f"{analysis_data.get('machine_time_min', 0):.2f}", "minutes"],
            ["Geometry Violations", f"{analysis_data.get('violations', 0)}", "count"],
            ["Complexity Rating", analysis_data.get('complexity', 'Unknown'), "level"]
        ]
        
        # Create table
        table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_layer_breakdown(self, layer_breakdown: Dict[str, int]) -> List:
        """Create layer breakdown section."""
        elements = []
        
        elements.append(Paragraph("Layer Breakdown", self.styles['CustomHeading1']))
        
        # Layer data
        layer_data = [["Layer", "Count"]]
        for layer, count in layer_breakdown.items():
            layer_data.append([layer, str(count)])
        
        # Create table
        table = Table(layer_data, colWidths=[2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_footer(self) -> List:
        """Create footer section."""
        elements = []
        
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("Generated by WJP Analyzer", self.styles['CustomBody']))
        elements.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                self.styles['CustomBody']))
        
        return elements

def test_report_generator_agent():
    """Test the Report Generator Agent."""
    print("📄 **Testing Report Generator Agent**")
    print("=" * 50)
    
    # Create agent
    agent = ReportGeneratorAgent()
    
    # First, create test files using previous agents
    print("📋 **Creating test files using previous agents...**")
    
    from wjp_designer_agent import DesignerAgent
    from wjp_image_to_dxf_agent import ImageToDXFAgent
    from wjp_dxf_analyzer_agent import DXFAnalyzerAgent
    
    try:
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
        
        image_path, metadata_path = designer.run(**test_case)
        print(f"   ✅ Test design created")
        
        # Convert to DXF
        image_to_dxf = ImageToDXFAgent()
        dxf_path, conversion_metadata_path = image_to_dxf.run(metadata_path)
        print(f"   ✅ DXF conversion completed")
        
        # Analyze DXF
        analyzer = DXFAnalyzerAgent()
        analysis_dxf_path, analysis_json_path, analysis_image_path, csv_path = analyzer.run(conversion_metadata_path)
        print(f"   ✅ DXF analysis completed")
        
        # Now test PDF report generation
        print(f"\n📄 **Testing PDF report generation...**")
        
        pdf_path = agent.run(analysis_json_path)
        
        print(f"   ✅ PDF Report: {os.path.basename(pdf_path)}")
        
        # Verify PDF exists
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"   ✅ PDF verified successfully ({file_size} bytes)")
        else:
            print(f"   ❌ PDF file not found")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 **Report Generator Agent Test Completed!**")
    
    return agent

if __name__ == "__main__":
    test_report_generator_agent()
