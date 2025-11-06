"""
Report Generator Service
=======================

Centralized service for generating reports in various formats.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from ..constants import ProcessStage
from .path_manager import PathManager

class ReportGenerator:
    """Handles generation of analysis and process reports."""
    
    def __init__(self, path_manager: PathManager):
        """Initialize the report generator.
        
        Args:
            path_manager (PathManager): Path management service
        """
        self.path_manager = path_manager
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def generate_analysis_report(
        self,
        design_code: str,
        analysis_result: Dict[str, Any],
        material_code: str,
        thickness_mm: float,
        output_format: str = "pdf"
    ) -> str:
        """Generate an analysis report in the specified format.
        
        Args:
            design_code (str): Design identifier
            analysis_result (Dict[str, Any]): Analysis results
            material_code (str): Material code
            thickness_mm (float): Material thickness
            output_format (str): Output format (pdf/json)
            
        Returns:
            str: Path to generated report
        """
        if output_format.lower() == "pdf":
            return self._generate_pdf_report(
                design_code, 
                analysis_result, 
                material_code, 
                thickness_mm
            )
        else:
            return self._generate_json_report(
                design_code, 
                analysis_result, 
                material_code, 
                thickness_mm
            )

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for PDF reports."""
        self.styles.add(
            ParagraphStyle(
                name='Title',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='Section',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='TableHeader',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.white,
                alignment=1
            )
        )

    def _generate_pdf_report(
        self,
        design_code: str,
        analysis_result: Dict[str, Any],
        material_code: str,
        thickness_mm: float
    ) -> str:
        """Generate a PDF format analysis report.
        
        Args:
            design_code (str): Design identifier
            analysis_result (Dict[str, Any]): Analysis results
            material_code (str): Material code
            thickness_mm (float): Material thickness
            
        Returns:
            str: Path to generated PDF
        """
        # Get output path
        output_path = self.path_manager.get_process_path(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=thickness_mm,
            process_stage=ProcessStage.ANALYSIS,
            extension="pdf"
        )
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        content = []
        
        # Add title
        content.append(Paragraph(f"Analysis Report: {design_code}", self.styles["Title"]))
        content.append(Spacer(1, 12))
        
        # Add summary section
        content.append(Paragraph("Summary", self.styles["Section"]))
        summary_data = [
            ["Material", f"{analysis_result['material']} ({thickness_mm}mm)"],
            ["Total Objects", str(analysis_result['metrics']['total_objects'])],
            ["Cut Length", f"{analysis_result['metrics']['total_cut_length_mtr']:.2f}m"],
            ["Cut Cost", f"₹{analysis_result['metrics']['cut_cost_inr']:.2f}"],
            ["Machine Time", f"{analysis_result['metrics']['machine_time_min']:.1f}min"]
        ]
        
        # Create and style the summary table
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (-1, -1), colors.beige),
            ('TEXTCOLOR', (1, 0), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(summary_table)
        
        # Build and save PDF
        doc.build(content)
        return output_path

    def _generate_json_report(
        self,
        design_code: str,
        analysis_result: Dict[str, Any],
        material_code: str,
        thickness_mm: float
    ) -> str:
        """Generate a JSON format analysis report.
        
        Args:
            design_code (str): Design identifier
            analysis_result (Dict[str, Any]): Analysis results
            material_code (str): Material code
            thickness_mm (float): Material thickness
            
        Returns:
            str: Path to generated JSON
        """
        # Get output path
        output_path = self.path_manager.get_process_path(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=thickness_mm,
            process_stage=ProcessStage.ANALYSIS,
            extension="json"
        )
        
        # Add metadata
        report_data = {
            "design_code": design_code,
            "material_code": material_code,
            "thickness_mm": thickness_mm,
            "generated_at": datetime.utcnow().isoformat(),
            "analysis": analysis_result
        }
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        return output_path