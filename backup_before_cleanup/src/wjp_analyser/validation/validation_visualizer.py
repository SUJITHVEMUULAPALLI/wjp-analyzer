"""
Enhanced DXF Validation Visualization Module

This module provides comprehensive visualization capabilities for DXF validation results,
including interactive reports, detailed analysis charts, and validation previews.
"""

from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import asdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from shapely.geometry import Polygon, Point
import ezdxf

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .enhanced_dxf_validator import ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class ValidationVisualizer:
    """Visualizer for DXF validation results."""
    
    def __init__(self, output_dir: str = "validation_reports"):
        """Initialize the visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        if HAS_SEABORN:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        else:
            plt.style.use('default')
    
    def create_comprehensive_report(self, 
                                  validation_result: ValidationResult,
                                  dxf_path: str,
                                  output_filename: Optional[str] = None) -> str:
        """
        Create a comprehensive validation report with visualizations.
        
        Args:
            validation_result: The validation result to visualize
            dxf_path: Path to the original DXF file
            output_filename: Optional custom output filename
            
        Returns:
            Path to the generated report file
        """
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(dxf_path))[0]
            output_filename = f"{base_name}_validation_report.pdf"
        
        report_path = os.path.join(self.output_dir, output_filename)
        
        with PdfPages(report_path) as pdf:
            # Title page
            self._create_title_page(pdf, validation_result, dxf_path)
            
            # Summary page
            self._create_summary_page(pdf, validation_result)
            
            # Issues analysis page
            self._create_issues_analysis_page(pdf, validation_result)
            
            # Geometry analysis page
            self._create_geometry_analysis_page(pdf, validation_result)
            
            # Manufacturing analysis page
            self._create_manufacturing_analysis_page(pdf, validation_result)
            
            # Performance analysis page
            self._create_performance_analysis_page(pdf, validation_result)
            
            # Detailed visualization page
            self._create_detailed_visualization_page(pdf, validation_result, dxf_path)
            
            # Recommendations page
            self._create_recommendations_page(pdf, validation_result)
        
        logger.info(f"Comprehensive validation report created: {report_path}")
        return report_path
    
    def _create_title_page(self, pdf: PdfPages, result: ValidationResult, dxf_path: str):
        """Create the title page of the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, 'DXF Validation Report', 
                fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # File information
        ax.text(0.5, 0.8, f'File: {os.path.basename(dxf_path)}', 
                fontsize=14, ha='center', transform=ax.transAxes)
        
        # Overall score
        score_color = self._get_score_color(result.overall_score)
        ax.text(0.5, 0.7, f'Overall Score: {result.overall_score:.1f}/100', 
                fontsize=18, fontweight='bold', ha='center', 
                color=score_color, transform=ax.transAxes)
        
        # Validation status
        status = "VALID" if result.is_valid else "INVALID"
        status_color = 'green' if result.is_valid else 'red'
        ax.text(0.5, 0.65, f'Status: {status}', 
                fontsize=16, fontweight='bold', ha='center', 
                color=status_color, transform=ax.transAxes)
        
        # Key statistics
        stats_text = f"""
        Total Entities: {result.total_entities}
        Polygons: {result.polygons}
        Open Contours: {result.open_contours}
        Issues Found: {len(result.issues)}
        Estimated Cutting Time: {result.estimated_cutting_time:.1f} min
        """
        
        ax.text(0.5, 0.5, stats_text, fontsize=12, ha='center', 
                va='top', transform=ax.transAxes)
        
        # Date
        from datetime import datetime
        ax.text(0.5, 0.1, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=10, ha='center', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_summary_page(self, pdf: PdfPages, result: ValidationResult):
        """Create the summary page with key metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Entity distribution pie chart
        entity_counts = [result.polygons, result.polylines, result.circles, 
                        result.arcs, result.lines]
        entity_labels = ['Polygons', 'Polylines', 'Circles', 'Arcs', 'Lines']
        
        # Filter out zero counts
        non_zero_counts = [(count, label) for count, label in zip(entity_counts, entity_labels) if count > 0]
        if non_zero_counts:
            counts, labels = zip(*non_zero_counts)
            ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Entity Distribution')
        else:
            ax1.text(0.5, 0.5, 'No entities found', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Entity Distribution')
        
        # Issue severity distribution
        severity_counts = {severity: 0 for severity in ValidationSeverity}
        for issue in result.issues:
            severity_counts[issue.severity] += 1
        
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = ['green', 'yellow', 'orange', 'red']
        
        bars = ax2.bar(severities, counts, color=colors[:len(severities)])
        ax2.set_title('Issue Severity Distribution')
        ax2.set_ylabel('Number of Issues')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # Quality scores radar chart
        categories = ['Geometry', 'Manufacturing', 'Performance', 'File Structure']
        scores = [
            self._calculate_geometry_score(result),
            result.cutting_feasibility_score * 100,
            result.efficiency_score * 100,
            self._calculate_file_structure_score(result)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        ax3.plot(angles, scores, 'o-', linewidth=2, color='blue')
        ax3.fill(angles, scores, alpha=0.25, color='blue')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 100)
        ax3.set_title('Quality Scores', pad=20)
        
        # Feature size distribution
        if result.min_feature_size > 0 and result.max_feature_size > result.min_feature_size:
            # Create a histogram of feature sizes (simplified)
            sizes = np.linspace(result.min_feature_size, result.max_feature_size, 10)
            ax4.hist(sizes, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_xlabel('Feature Size (mm)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Feature Size Distribution')
        else:
            ax4.text(0.5, 0.5, 'No feature size data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Size Distribution')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_issues_analysis_page(self, pdf: PdfPages, result: ValidationResult):
        """Create detailed issues analysis page."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Issues by rule type
        rule_counts = {}
        for issue in result.issues:
            rule_counts[issue.rule_name] = rule_counts.get(issue.rule_name, 0) + 1
        
        if rule_counts:
            rules = list(rule_counts.keys())
            counts = list(rule_counts.values())
            
            bars = ax1.barh(rules, counts, color='lightcoral')
            ax1.set_xlabel('Number of Issues')
            ax1.set_title('Issues by Rule Type')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), ha='left', va='center')
        else:
            ax1.text(0.5, 0.5, 'No issues found', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Issues by Rule Type')
        
        # Detailed issues list
        ax2.axis('off')
        ax2.set_title('Detailed Issues List', fontsize=14, fontweight='bold')
        
        if result.issues:
            y_pos = 0.95
            for i, issue in enumerate(result.issues[:20]):  # Limit to 20 issues
                color = self._get_severity_color(issue.severity)
                
                issue_text = f"{issue.severity.value.upper()}: {issue.message}"
                if issue.suggested_fix:
                    issue_text += f"\n    Fix: {issue.suggested_fix}"
                
                ax2.text(0.05, y_pos, issue_text, fontsize=10, color=color,
                        transform=ax2.transAxes, verticalalignment='top')
                y_pos -= 0.08
                
                if y_pos < 0.05:
                    ax2.text(0.05, y_pos, "... (more issues not shown)", 
                            fontsize=10, style='italic', transform=ax2.transAxes)
                    break
        else:
            ax2.text(0.5, 0.5, 'No issues found - DXF file is valid!', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, color='green', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_geometry_analysis_page(self, pdf: PdfPages, result: ValidationResult):
        """Create geometry analysis page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Contour analysis
        contour_data = [result.closed_contours, result.open_contours]
        contour_labels = ['Closed', 'Open']
        colors = ['lightgreen', 'lightcoral']
        
        bars = ax1.bar(contour_labels, contour_data, color=colors)
        ax1.set_title('Contour Analysis')
        ax1.set_ylabel('Number of Contours')
        
        for bar, count in zip(bars, contour_data):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Geometry issues
        geometry_issues = ['Self-intersections', 'Overlapping Features', 'Nested Contours']
        issue_counts = [result.self_intersections, result.overlapping_features, result.nested_contours]
        
        bars = ax2.bar(geometry_issues, issue_counts, color='orange')
        ax2.set_title('Geometry Issues')
        ax2.set_ylabel('Number of Issues')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, issue_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # Feature size analysis
        if result.min_feature_size > 0:
            ax3.scatter([result.min_feature_size], [result.max_feature_size], 
                       s=100, color='blue', alpha=0.7)
            ax3.set_xlabel('Minimum Feature Size (mm)')
            ax3.set_ylabel('Maximum Feature Size (mm)')
            ax3.set_title('Feature Size Range')
            
            # Add diagonal line for reference
            max_size = max(result.min_feature_size, result.max_feature_size)
            ax3.plot([0, max_size], [0, max_size], 'k--', alpha=0.5)
        else:
            ax3.text(0.5, 0.5, 'No feature size data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Size Range')
        
        # Bounding box visualization
        if result.bounding_box != (0, 0, 0, 0):
            min_x, min_y, max_x, max_y = result.bounding_box
            width = max_x - min_x
            height = max_y - min_y
            
            rect = patches.Rectangle((min_x, min_y), width, height, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax4.add_patch(rect)
            ax4.set_xlim(min_x - width*0.1, max_x + width*0.1)
            ax4.set_ylim(min_y - height*0.1, max_y + height*0.1)
            ax4.set_xlabel('X (mm)')
            ax4.set_ylabel('Y (mm)')
            ax4.set_title('Bounding Box')
            ax4.set_aspect('equal')
        else:
            ax4.text(0.5, 0.5, 'No bounding box data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Bounding Box')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_manufacturing_analysis_page(self, pdf: PdfPages, result: ValidationResult):
        """Create manufacturing analysis page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Manufacturing feasibility score
        feasibility_score = result.cutting_feasibility_score * 100
        ax1.pie([feasibility_score, 100 - feasibility_score], 
               labels=['Feasible', 'Issues'], 
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Manufacturing Feasibility')
        
        # Spacing violations
        spacing_data = [result.spacing_violations, result.total_entities - result.spacing_violations]
        spacing_labels = ['Violations', 'Compliant']
        colors = ['red', 'green']
        
        bars = ax2.bar(spacing_labels, spacing_data, color=colors)
        ax2.set_title('Spacing Compliance')
        ax2.set_ylabel('Number of Features')
        
        for bar, count in zip(bars, spacing_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Radius violations
        radius_data = [result.radius_violations, result.circles - result.radius_violations]
        radius_labels = ['Violations', 'Compliant']
        
        bars = ax3.bar(radius_labels, radius_data, color=colors)
        ax3.set_title('Radius Compliance')
        ax3.set_ylabel('Number of Circles')
        
        for bar, count in zip(bars, radius_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Kerf conflicts
        kerf_data = [result.kerf_conflicts, result.total_entities - result.kerf_conflicts]
        kerf_labels = ['Conflicts', 'Clear']
        
        bars = ax4.bar(kerf_labels, kerf_data, color=colors)
        ax4.set_title('Kerf Conflicts')
        ax4.set_ylabel('Number of Features')
        
        for bar, count in zip(bars, kerf_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_performance_analysis_page(self, pdf: PdfPages, result: ValidationResult):
        """Create performance analysis page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Cutting time estimation
        cutting_time = result.estimated_cutting_time
        ax1.bar(['Estimated'], [cutting_time], color='skyblue')
        ax1.set_title('Estimated Cutting Time')
        ax1.set_ylabel('Time (minutes)')
        
        # Add threshold line if applicable
        if hasattr(self, 'config') and hasattr(self.config, 'max_cutting_time_minutes'):
            ax1.axhline(y=self.config.max_cutting_time_minutes, color='red', 
                       linestyle='--', label='Max Threshold')
            ax1.legend()
        
        ax1.text(0, cutting_time + 1, f'{cutting_time:.1f} min', ha='center', va='bottom')
        
        # Cutting length
        cutting_length = result.estimated_cutting_length
        ax2.bar(['Estimated'], [cutting_length], color='lightgreen')
        ax2.set_title('Estimated Cutting Length')
        ax2.set_ylabel('Length (mm)')
        ax2.text(0, cutting_length + cutting_length*0.05, f'{cutting_length:.1f} mm', 
                ha='center', va='bottom')
        
        # Pierce count
        pierce_count = result.estimated_pierce_count
        ax3.bar(['Estimated'], [pierce_count], color='orange')
        ax3.set_title('Estimated Pierce Count')
        ax3.set_ylabel('Number of Pierces')
        ax3.text(0, pierce_count + pierce_count*0.05, str(pierce_count), 
                ha='center', va='bottom')
        
        # Efficiency score
        efficiency_score = result.efficiency_score * 100
        ax4.pie([efficiency_score, 100 - efficiency_score], 
               labels=['Efficient', 'Inefficient'], 
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Cutting Efficiency')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_detailed_visualization_page(self, pdf: PdfPages, result: ValidationResult, dxf_path: str):
        """Create detailed visualization page with DXF preview."""
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        
        try:
            # Load and visualize DXF entities
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # Set up the plot
            ax.set_title('DXF Preview with Validation Issues', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.grid(True, alpha=0.3)
            
            # Plot entities
            for entity in msp:
                if entity.dxftype() == 'LWPOLYLINE':
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) >= 2:
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        if entity.closed:
                            x_coords.append(x_coords[0])
                            y_coords.append(y_coords[0])
                            ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
                        else:
                            ax.plot(x_coords, y_coords, 'r--', linewidth=1, alpha=0.7)
                
                elif entity.dxftype() == 'CIRCLE':
                    center = (entity.dxf.center.x, entity.dxf.center.y)
                    radius = entity.dxf.radius
                    circle = plt.Circle(center, radius, fill=False, color='green', linewidth=2, alpha=0.7)
                    ax.add_patch(circle)
                
                elif entity.dxftype() == 'LINE':
                    start = (entity.dxf.start.x, entity.dxf.start.y)
                    end = (entity.dxf.end.x, entity.dxf.end.y)
                    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1, alpha=0.5)
            
            # Highlight issues if any
            for issue in result.issues:
                if issue.location:
                    ax.plot(issue.location[0], issue.location[1], 'ro', markersize=8, alpha=0.8)
                    ax.annotate(issue.rule_name, issue.location, xytext=(10, 10), 
                              textcoords='offset points', fontsize=8, 
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading DXF: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('DXF Preview (Error)')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_recommendations_page(self, pdf: PdfPages, result: ValidationResult):
        """Create recommendations page."""
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        ax.axis('off')
        
        ax.set_title('Recommendations', fontsize=16, fontweight='bold', pad=20)
        
        if result.recommendations:
            y_pos = 0.9
            for i, recommendation in enumerate(result.recommendations):
                ax.text(0.05, y_pos, f"{i+1}. {recommendation}", 
                       fontsize=12, transform=ax.transAxes, verticalalignment='top')
                y_pos -= 0.08
                
                if y_pos < 0.05:
                    ax.text(0.05, y_pos, "... (more recommendations available)", 
                           fontsize=10, style='italic', transform=ax.transAxes)
                    break
        else:
            ax.text(0.5, 0.5, 'No specific recommendations - DXF file is well-optimized!', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='green', fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score."""
        if score >= 90:
            return 'green'
        elif score >= 70:
            return 'orange'
        else:
            return 'red'
    
    def _get_severity_color(self, severity: ValidationSeverity) -> str:
        """Get color based on severity."""
        color_map = {
            ValidationSeverity.INFO: 'blue',
            ValidationSeverity.WARNING: 'orange',
            ValidationSeverity.ERROR: 'red',
            ValidationSeverity.CRITICAL: 'darkred'
        }
        return color_map.get(severity, 'black')
    
    def _calculate_geometry_score(self, result: ValidationResult) -> float:
        """Calculate geometry quality score."""
        score = 100.0
        
        # Deduct for geometry issues
        if result.self_intersections > 0:
            score -= min(30, result.self_intersections * 10)
        
        if result.open_contours > 0:
            score -= min(20, result.open_contours * 5)
        
        if result.overlapping_features > 0:
            score -= min(25, result.overlapping_features * 8)
        
        return max(0, score)
    
    def _calculate_file_structure_score(self, result: ValidationResult) -> float:
        """Calculate file structure quality score."""
        score = 100.0
        
        # Deduct for unused elements
        if result.unused_layers > 0:
            score -= min(20, result.unused_layers * 2)
        
        if result.unused_blocks > 0:
            score -= min(20, result.unused_blocks * 2)
        
        if result.text_entities > 0:
            score -= min(15, result.text_entities * 1)
        
        if result.dimension_entities > 0:
            score -= min(15, result.dimension_entities * 1)
        
        return max(0, score)
    
    def create_json_report(self, validation_result: ValidationResult, output_path: str) -> str:
        """Create a JSON report of the validation results."""
        # Convert result to dictionary
        result_dict = asdict(validation_result)
        
        # Convert enums to strings
        for issue in result_dict['issues']:
            issue['severity'] = issue['severity'].value
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"JSON validation report created: {output_path}")
        return output_path
    
    def create_summary_chart(self, validation_result: ValidationResult, output_path: str) -> str:
        """Create a summary chart image."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Overall score gauge
        score = validation_result.overall_score
        ax1.pie([score, 100 - score], labels=['Score', ''], 
               colors=['lightgreen' if score >= 70 else 'lightcoral', 'lightgray'],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Overall Score: {score:.1f}/100')
        
        # Issue severity distribution
        severity_counts = {severity: 0 for severity in ValidationSeverity}
        for issue in validation_result.issues:
            severity_counts[issue.severity] += 1
        
        severities = [s.value for s in severity_counts.keys()]
        counts = list(severity_counts.values())
        colors = ['green', 'yellow', 'orange', 'red']
        
        bars = ax2.bar(severities, counts, color=colors[:len(severities)])
        ax2.set_title('Issue Severity Distribution')
        ax2.set_ylabel('Number of Issues')
        
        for bar, count in zip(bars, counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Summary chart created: {output_path}")
        return output_path


def create_validation_report(validation_result: ValidationResult,
                           dxf_path: str,
                           output_dir: str = "validation_reports") -> Dict[str, str]:
    """
    Create comprehensive validation reports.
    
    Args:
        validation_result: The validation result to report on
        dxf_path: Path to the original DXF file
        output_dir: Directory to save reports
        
    Returns:
        Dictionary with paths to generated reports
    """
    visualizer = ValidationVisualizer(output_dir)
    
    # Create comprehensive PDF report
    pdf_report = visualizer.create_comprehensive_report(validation_result, dxf_path)
    
    # Create JSON report
    base_name = os.path.splitext(os.path.basename(dxf_path))[0]
    json_report = os.path.join(output_dir, f"{base_name}_validation.json")
    visualizer.create_json_report(validation_result, json_report)
    
    # Create summary chart
    chart_path = os.path.join(output_dir, f"{base_name}_summary_chart.png")
    visualizer.create_summary_chart(validation_result, chart_path)
    
    return {
        'pdf_report': pdf_report,
        'json_report': json_report,
        'summary_chart': chart_path
    }
