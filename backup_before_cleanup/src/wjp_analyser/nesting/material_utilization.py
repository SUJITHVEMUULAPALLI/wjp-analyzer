"""
Material Utilization Calculator and Reporting System

This module provides comprehensive material utilization analysis, cost calculation,
and reporting for nesting optimization results.
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .nesting_engine import NestingResult, PositionedObject
from ..object_management.layer_manager import CuttingLayer, MaterialSettings, CuttingSettings, CostAnalysis

logger = logging.getLogger(__name__)


@dataclass
class MaterialUtilizationReport:
    """Comprehensive material utilization report."""
    layer_id: str
    layer_name: str
    algorithm_used: str
    optimization_time: float
    
    # Material metrics
    material_width: float
    material_height: float
    material_area: float
    used_area: float
    waste_area: float
    utilization_percentage: float
    
    # Object metrics
    total_objects: int
    positioned_objects: int
    failed_objects: int
    positioning_success_rate: float
    
    # Cost analysis
    material_cost: float
    cutting_cost: float
    total_cost: float
    cost_per_object: float
    cost_per_area: float
    
    # Efficiency metrics
    cutting_length: float
    estimated_cutting_time: float
    pierce_count: int
    efficiency_score: float
    
    # Optimization metrics
    improvement_percentage: float
    iterations_completed: int
    convergence_rate: float
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MaterialUtilizationCalculator:
    """Calculator for material utilization metrics."""
    
    def __init__(self):
        """Initialize the calculator."""
        pass
    
    def calculate_utilization(self, result: NestingResult, 
                            material_settings: MaterialSettings) -> MaterialUtilizationReport:
        """Calculate comprehensive material utilization metrics."""
        
        # Basic material metrics
        material_area = material_settings.width * material_settings.height
        used_area = sum(obj.geometry.area for obj in result.positioned_objects)
        waste_area = material_area - used_area
        utilization_percentage = (used_area / material_area) * 100 if material_area > 0 else 0
        
        # Object metrics
        total_objects = len(result.positioned_objects) + len(result.failed_objects)
        positioned_objects = len(result.positioned_objects)
        failed_objects = len(result.failed_objects)
        positioning_success_rate = (positioned_objects / total_objects) * 100 if total_objects > 0 else 0
        
        # Cost analysis
        material_cost = self._calculate_material_cost(material_settings, material_area)
        cutting_cost = self._calculate_cutting_cost(result, material_settings)
        total_cost = material_cost + cutting_cost
        cost_per_object = total_cost / positioned_objects if positioned_objects > 0 else 0
        cost_per_area = total_cost / used_area if used_area > 0 else 0
        
        # Efficiency metrics
        cutting_length = self._calculate_cutting_length(result.positioned_objects)
        estimated_cutting_time = self._calculate_cutting_time(result, material_settings)
        pierce_count = len(result.positioned_objects)
        efficiency_score = self._calculate_efficiency_score(result, material_settings)
        
        # Optimization metrics
        improvement_percentage = result.improvement_percentage
        iterations_completed = result.iterations_completed
        convergence_rate = self._calculate_convergence_rate(result)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, material_settings, utilization_percentage)
        warnings = self._generate_warnings(result, material_settings)
        
        return MaterialUtilizationReport(
            layer_id="",  # Will be set by caller
            layer_name="",  # Will be set by caller
            algorithm_used=result.algorithm_used,
            optimization_time=result.optimization_time,
            material_width=material_settings.width,
            material_height=material_settings.height,
            material_area=material_area,
            used_area=used_area,
            waste_area=waste_area,
            utilization_percentage=utilization_percentage,
            total_objects=total_objects,
            positioned_objects=positioned_objects,
            failed_objects=failed_objects,
            positioning_success_rate=positioning_success_rate,
            material_cost=material_cost,
            cutting_cost=cutting_cost,
            total_cost=total_cost,
            cost_per_object=cost_per_object,
            cost_per_area=cost_per_area,
            cutting_length=cutting_length,
            estimated_cutting_time=estimated_cutting_time,
            pierce_count=pierce_count,
            efficiency_score=efficiency_score,
            improvement_percentage=improvement_percentage,
            iterations_completed=iterations_completed,
            convergence_rate=convergence_rate,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def _calculate_material_cost(self, material_settings: MaterialSettings, 
                               material_area: float) -> float:
        """Calculate material cost."""
        # Convert area to weight (assuming thickness in mm)
        volume_mm3 = material_area * material_settings.thickness
        volume_dm3 = volume_mm3 / (1000 * 1000 * 1000)  # Convert to dm³
        weight_kg = volume_dm3 * material_settings.density
        
        return weight_kg * material_settings.cost_per_kg
    
    def _calculate_cutting_cost(self, result: NestingResult, 
                              material_settings: MaterialSettings) -> float:
        """Calculate cutting cost."""
        # Estimate cutting time
        cutting_time = self._calculate_cutting_time(result, material_settings)
        
        # Assume machine cost per hour (this could be configurable)
        machine_cost_per_hour = 50.0  # USD/hour
        labor_cost_per_hour = 25.0  # USD/hour
        
        total_hourly_cost = machine_cost_per_hour + labor_cost_per_hour
        cutting_cost = (cutting_time / 60.0) * total_hourly_cost
        
        return cutting_cost
    
    def _calculate_cutting_length(self, positioned_objects: List[PositionedObject]) -> float:
        """Calculate total cutting length."""
        total_length = 0.0
        
        for obj in positioned_objects:
            # Calculate perimeter of positioned geometry
            total_length += obj.geometry.length
        
        return total_length
    
    def _calculate_cutting_time(self, result: NestingResult, 
                               material_settings: MaterialSettings) -> float:
        """Calculate estimated cutting time."""
        cutting_length = self._calculate_cutting_length(result.positioned_objects)
        pierce_count = len(result.positioned_objects)
        
        # Use cutting settings if available
        cutting_speed = 1200.0  # mm/min (default)
        pierce_time = 0.5  # seconds per pierce (default)
        
        # Calculate time
        cutting_time = cutting_length / cutting_speed  # minutes
        piercing_time = (pierce_count * pierce_time) / 60.0  # minutes
        
        return cutting_time + piercing_time
    
    def _calculate_efficiency_score(self, result: NestingResult, 
                                   material_settings: MaterialSettings) -> float:
        """Calculate efficiency score (0-1)."""
        factors = []
        
        # Material utilization factor
        material_area = material_settings.width * material_settings.height
        used_area = sum(obj.geometry.area for obj in result.positioned_objects)
        utilization_factor = used_area / material_area if material_area > 0 else 0
        factors.append(utilization_factor)
        
        # Positioning success factor
        total_objects = len(result.positioned_objects) + len(result.failed_objects)
        success_factor = len(result.positioned_objects) / total_objects if total_objects > 0 else 0
        factors.append(success_factor)
        
        # Optimization time factor (shorter is better)
        time_factor = max(0, 1.0 - (result.optimization_time / 300.0))  # 5 minutes max
        factors.append(time_factor)
        
        # Improvement factor
        improvement_factor = min(1.0, max(0, result.improvement_percentage / 50.0))  # 50% max improvement
        factors.append(improvement_factor)
        
        return np.mean(factors)
    
    def _calculate_convergence_rate(self, result: NestingResult) -> float:
        """Calculate convergence rate."""
        if result.iterations_completed == 0:
            return 0.0
        
        # Simple convergence rate based on iterations vs time
        expected_iterations = 1000  # Default max iterations
        convergence_rate = min(1.0, result.iterations_completed / expected_iterations)
        
        return convergence_rate
    
    def _generate_recommendations(self, result: NestingResult, 
                                material_settings: MaterialSettings,
                                utilization_percentage: float) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Material utilization recommendations
        if utilization_percentage < 70:
            recommendations.append("Consider using smaller material sheets or optimizing object sizes")
        elif utilization_percentage > 90:
            recommendations.append("Excellent material utilization achieved")
        
        # Failed objects recommendations
        if result.failed_objects:
            recommendations.append(f"Consider reducing object sizes or increasing material dimensions for {len(result.failed_objects)} failed objects")
        
        # Algorithm recommendations
        if result.algorithm_used == "bottom_left" and utilization_percentage < 80:
            recommendations.append("Try genetic algorithm or simulated annealing for better results")
        
        # Time recommendations
        if result.optimization_time > 300:
            recommendations.append("Consider reducing optimization time limit for faster processing")
        
        # Improvement recommendations
        if result.improvement_percentage < 10:
            recommendations.append("Current layout is already well-optimized")
        
        return recommendations
    
    def _generate_warnings(self, result: NestingResult, 
                          material_settings: MaterialSettings) -> List[str]:
        """Generate warnings based on results."""
        warnings = []
        
        # Failed objects warning
        if result.failed_objects:
            warnings.append(f"{len(result.failed_objects)} objects could not be positioned")
        
        # Low utilization warning
        utilization_percentage = (sum(obj.geometry.area for obj in result.positioned_objects) / 
                                 (material_settings.width * material_settings.height)) * 100
        if utilization_percentage < 50:
            warnings.append("Very low material utilization - consider redesigning layout")
        
        # Time limit warning
        if result.status.value == "time_limit":
            warnings.append("Optimization stopped due to time limit - results may not be optimal")
        
        # Iteration limit warning
        if result.status.value == "iteration_limit":
            warnings.append("Optimization stopped due to iteration limit - results may not be optimal")
        
        return warnings


class MaterialUtilizationReporter:
    """Reporter for material utilization analysis."""
    
    def __init__(self, output_dir: str = "utilization_reports"):
        """Initialize the reporter."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.calculator = MaterialUtilizationCalculator()
    
    def generate_comprehensive_report(self, layer: CuttingLayer, 
                                    result: NestingResult,
                                    output_filename: Optional[str] = None) -> str:
        """Generate comprehensive material utilization report."""
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{layer.name}_utilization_report_{timestamp}.pdf"
        
        report_path = os.path.join(self.output_dir, output_filename)
        
        # Calculate utilization metrics
        utilization_report = self.calculator.calculate_utilization(result, layer.material_settings)
        utilization_report.layer_id = layer.layer_id
        utilization_report.layer_name = layer.name
        
        # Generate PDF report
        with PdfPages(report_path) as pdf:
            self._create_title_page(pdf, utilization_report)
            self._create_summary_page(pdf, utilization_report)
            self._create_material_analysis_page(pdf, utilization_report)
            self._create_cost_analysis_page(pdf, utilization_report)
            self._create_efficiency_analysis_page(pdf, utilization_report)
            self._create_optimization_analysis_page(pdf, utilization_report)
            self._create_recommendations_page(pdf, utilization_report)
            self._create_visualization_page(pdf, result, layer.material_settings)
        
        logger.info(f"Material utilization report generated: {report_path}")
        return report_path
    
    def _create_title_page(self, pdf: PdfPages, report: MaterialUtilizationReport):
        """Create title page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, 'Material Utilization Report', 
                fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Layer information
        ax.text(0.5, 0.8, f'Layer: {report.layer_name}', 
                fontsize=16, ha='center', transform=ax.transAxes)
        
        # Algorithm
        ax.text(0.5, 0.75, f'Algorithm: {report.algorithm_used.title()}', 
                fontsize=14, ha='center', transform=ax.transAxes)
        
        # Key metrics
        utilization_color = 'green' if report.utilization_percentage >= 80 else 'orange' if report.utilization_percentage >= 60 else 'red'
        ax.text(0.5, 0.65, f'Material Utilization: {report.utilization_percentage:.1f}%', 
                fontsize=18, fontweight='bold', ha='center', 
                color=utilization_color, transform=ax.transAxes)
        
        # Summary statistics
        stats_text = f"""
        Objects Positioned: {report.positioned_objects}/{report.total_objects}
        Material Dimensions: {report.material_width:.1f} × {report.material_height:.1f} mm
        Used Area: {report.used_area:.1f} mm²
        Waste Area: {report.waste_area:.1f} mm²
        Total Cost: ${report.total_cost:.2f}
        Estimated Cutting Time: {report.estimated_cutting_time:.1f} min
        """
        
        ax.text(0.5, 0.5, stats_text, fontsize=12, ha='center', 
                va='top', transform=ax.transAxes)
        
        # Date
        ax.text(0.5, 0.1, f'Generated: {report.generated_at.strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=10, ha='center', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_summary_page(self, pdf: PdfPages, report: MaterialUtilizationReport):
        """Create summary page with key metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Material utilization pie chart
        used_area = report.used_area
        waste_area = report.waste_area
        
        ax1.pie([used_area, waste_area], 
               labels=['Used', 'Waste'], 
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Material Utilization')
        
        # Object positioning success
        positioned = report.positioned_objects
        failed = report.failed_objects
        
        ax2.pie([positioned, failed], 
               labels=['Positioned', 'Failed'], 
               colors=['lightblue', 'lightcoral'],
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Object Positioning Success')
        
        # Cost breakdown
        costs = [report.material_cost, report.cutting_cost]
        labels = ['Material', 'Cutting']
        colors = ['lightblue', 'lightgreen']
        
        bars = ax3.bar(labels, costs, color=colors)
        ax3.set_title('Cost Breakdown')
        ax3.set_ylabel('Cost ($)')
        
        for bar, cost in zip(bars, costs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'${cost:.2f}', ha='center', va='bottom')
        
        # Efficiency metrics
        metrics = ['Utilization', 'Success Rate', 'Efficiency Score']
        values = [report.utilization_percentage/100, 
                 report.positioning_success_rate/100, 
                 report.efficiency_score]
        
        bars = ax4.bar(metrics, values, color='skyblue')
        ax4.set_title('Efficiency Metrics')
        ax4.set_ylabel('Score (0-1)')
        ax4.set_ylim(0, 1)
        
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_material_analysis_page(self, pdf: PdfPages, report: MaterialUtilizationReport):
        """Create material analysis page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Material dimensions
        ax1.bar(['Width', 'Height'], [report.material_width, report.material_height], 
               color=['lightblue', 'lightgreen'])
        ax1.set_title('Material Dimensions')
        ax1.set_ylabel('Size (mm)')
        
        # Area breakdown
        areas = [report.used_area, report.waste_area]
        labels = ['Used Area', 'Waste Area']
        colors = ['lightgreen', 'lightcoral']
        
        ax2.pie(areas, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Area Breakdown')
        
        # Utilization over time (simulated)
        iterations = range(0, report.iterations_completed + 1, max(1, report.iterations_completed // 10))
        utilization_values = [report.utilization_percentage * (i / report.iterations_completed) 
                            for i in iterations]
        
        ax3.plot(iterations, utilization_values, 'b-', linewidth=2)
        ax3.set_title('Utilization Progress')
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Utilization (%)')
        ax3.grid(True, alpha=0.3)
        
        # Material cost analysis
        cost_breakdown = {
            'Material': report.material_cost,
            'Cutting': report.cutting_cost
        }
        
        ax4.pie(cost_breakdown.values(), labels=cost_breakdown.keys(), 
               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Cost Breakdown')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_cost_analysis_page(self, pdf: PdfPages, report: MaterialUtilizationReport):
        """Create cost analysis page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Total cost breakdown
        costs = [report.material_cost, report.cutting_cost]
        labels = ['Material Cost', 'Cutting Cost']
        colors = ['lightblue', 'lightgreen']
        
        bars = ax1.bar(labels, costs, color=colors)
        ax1.set_title('Total Cost Breakdown')
        ax1.set_ylabel('Cost ($)')
        
        for bar, cost in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'${cost:.2f}', ha='center', va='bottom')
        
        # Cost per object
        ax2.bar(['Cost per Object'], [report.cost_per_object], color='orange')
        ax2.set_title('Cost per Object')
        ax2.set_ylabel('Cost ($)')
        ax2.text(0, report.cost_per_object + 0.1, f'${report.cost_per_object:.2f}', 
                ha='center', va='bottom')
        
        # Cost per area
        ax3.bar(['Cost per Area'], [report.cost_per_area], color='purple')
        ax3.set_title('Cost per Area')
        ax3.set_ylabel('Cost ($/mm²)')
        ax3.text(0, report.cost_per_area + report.cost_per_area * 0.05, 
                f'${report.cost_per_area:.4f}', ha='center', va='bottom')
        
        # Cost efficiency
        efficiency_metrics = ['Total Cost', 'Cost per Object', 'Cost per Area']
        efficiency_values = [report.total_cost, report.cost_per_object, report.cost_per_area]
        
        ax4.bar(efficiency_metrics, efficiency_values, color='skyblue')
        ax4.set_title('Cost Efficiency Metrics')
        ax4.set_ylabel('Cost ($)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_efficiency_analysis_page(self, pdf: PdfPages, report: MaterialUtilizationReport):
        """Create efficiency analysis page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Efficiency score radar chart
        categories = ['Utilization', 'Success Rate', 'Time Efficiency', 'Improvement']
        scores = [report.utilization_percentage/100, 
                 report.positioning_success_rate/100,
                 max(0, 1.0 - report.optimization_time/300.0),
                 min(1.0, report.improvement_percentage/50.0)]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax1 = plt.subplot(2, 2, 1, projection='polar')
        ax1.plot(angles, scores, 'o-', linewidth=2, color='blue')
        ax1.fill(angles, scores, alpha=0.25, color='blue')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('Efficiency Score', pad=20)
        
        # Cutting metrics
        cutting_metrics = ['Cutting Length', 'Cutting Time', 'Pierce Count']
        cutting_values = [report.cutting_length, report.estimated_cutting_time, report.pierce_count]
        
        bars = ax2.bar(cutting_metrics, cutting_values, color='lightgreen')
        ax2.set_title('Cutting Metrics')
        ax2.set_ylabel('Value')
        ax2.tick_params(axis='x', rotation=45)
        
        # Optimization metrics
        opt_metrics = ['Iterations', 'Time (s)', 'Improvement (%)']
        opt_values = [report.iterations_completed, report.optimization_time, report.improvement_percentage]
        
        bars = ax3.bar(opt_metrics, opt_values, color='orange')
        ax3.set_title('Optimization Metrics')
        ax3.set_ylabel('Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # Performance comparison
        performance_metrics = ['Utilization', 'Success Rate', 'Efficiency']
        performance_values = [report.utilization_percentage, report.positioning_success_rate, report.efficiency_score * 100]
        
        bars = ax4.bar(performance_metrics, performance_values, color='purple')
        ax4.set_title('Performance Metrics')
        ax4.set_ylabel('Percentage')
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_optimization_analysis_page(self, pdf: PdfPages, report: MaterialUtilizationReport):
        """Create optimization analysis page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Algorithm performance
        ax1.bar(['Algorithm Used'], [1], color='lightblue')
        ax1.set_title(f'Algorithm: {report.algorithm_used.title()}')
        ax1.set_ylabel('Performance')
        ax1.text(0, 0.5, report.algorithm_used.title(), ha='center', va='center', fontsize=12)
        
        # Optimization time
        ax2.bar(['Optimization Time'], [report.optimization_time], color='orange')
        ax2.set_title('Optimization Time')
        ax2.set_ylabel('Time (seconds)')
        ax2.text(0, report.optimization_time + 1, f'{report.optimization_time:.1f}s', 
                ha='center', va='bottom')
        
        # Iterations completed
        ax3.bar(['Iterations'], [report.iterations_completed], color='green')
        ax3.set_title('Iterations Completed')
        ax3.set_ylabel('Count')
        ax3.text(0, report.iterations_completed + 10, str(report.iterations_completed), 
                ha='center', va='bottom')
        
        # Improvement percentage
        improvement_color = 'green' if report.improvement_percentage > 0 else 'red'
        ax4.bar(['Improvement'], [report.improvement_percentage], color=improvement_color)
        ax4.set_title('Improvement Percentage')
        ax4.set_ylabel('Percentage (%)')
        ax4.text(0, report.improvement_percentage + 1, f'{report.improvement_percentage:.1f}%', 
                ha='center', va='bottom')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_recommendations_page(self, pdf: PdfPages, report: MaterialUtilizationReport):
        """Create recommendations page."""
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        ax.axis('off')
        
        ax.set_title('Recommendations and Warnings', fontsize=16, fontweight='bold', pad=20)
        
        y_pos = 0.9
        
        # Recommendations
        if report.recommendations:
            ax.text(0.05, y_pos, 'Recommendations:', fontsize=14, fontweight='bold', 
                   transform=ax.transAxes, color='blue')
            y_pos -= 0.05
            
            for i, rec in enumerate(report.recommendations):
                ax.text(0.05, y_pos, f"{i+1}. {rec}", fontsize=12, 
                       transform=ax.transAxes, verticalalignment='top')
                y_pos -= 0.04
                
                if y_pos < 0.1:
                    break
        
        y_pos -= 0.05
        
        # Warnings
        if report.warnings:
            ax.text(0.05, y_pos, 'Warnings:', fontsize=14, fontweight='bold', 
                   transform=ax.transAxes, color='red')
            y_pos -= 0.05
            
            for i, warning in enumerate(report.warnings):
                ax.text(0.05, y_pos, f"{i+1}. {warning}", fontsize=12, 
                       transform=ax.transAxes, verticalalignment='top', color='red')
                y_pos -= 0.04
                
                if y_pos < 0.05:
                    break
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_visualization_page(self, pdf: PdfPages, result: NestingResult, 
                                 material_settings: MaterialSettings):
        """Create visualization page."""
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        
        # Draw material bounds
        material_rect = patches.Rectangle((0, 0), material_settings.width, material_settings.height,
                                        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(material_rect)
        
        # Draw positioned objects
        colors = plt.cm.Set3(np.linspace(0, 1, len(result.positioned_objects)))
        
        for i, pos_obj in enumerate(result.positioned_objects):
            # Get object bounds
            bounds = pos_obj.bounding_box
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            # Draw object
            obj_rect = patches.Rectangle((bounds[0], bounds[1]), width, height,
                                        linewidth=1, edgecolor='blue', facecolor=colors[i], alpha=0.7)
            ax.add_patch(obj_rect)
            
            # Add object ID
            ax.text(bounds[0] + width/2, bounds[1] + height/2, pos_obj.object.object_id,
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlim(-10, material_settings.width + 10)
        ax.set_ylim(-10, material_settings.height + 10)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Nesting Layout Visualization')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def export_json_report(self, report: MaterialUtilizationReport, output_path: str) -> str:
        """Export report as JSON."""
        report_dict = {
            'layer_id': report.layer_id,
            'layer_name': report.layer_name,
            'algorithm_used': report.algorithm_used,
            'optimization_time': report.optimization_time,
            'material_width': report.material_width,
            'material_height': report.material_height,
            'material_area': report.material_area,
            'used_area': report.used_area,
            'waste_area': report.waste_area,
            'utilization_percentage': report.utilization_percentage,
            'total_objects': report.total_objects,
            'positioned_objects': report.positioned_objects,
            'failed_objects': report.failed_objects,
            'positioning_success_rate': report.positioning_success_rate,
            'material_cost': report.material_cost,
            'cutting_cost': report.cutting_cost,
            'total_cost': report.total_cost,
            'cost_per_object': report.cost_per_object,
            'cost_per_area': report.cost_per_area,
            'cutting_length': report.cutting_length,
            'estimated_cutting_time': report.estimated_cutting_time,
            'pierce_count': report.pierce_count,
            'efficiency_score': report.efficiency_score,
            'improvement_percentage': report.improvement_percentage,
            'iterations_completed': report.iterations_completed,
            'convergence_rate': report.convergence_rate,
            'recommendations': report.recommendations,
            'warnings': report.warnings,
            'generated_at': report.generated_at.isoformat(),
            'metadata': report.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"JSON report exported: {output_path}")
        return output_path


# Convenience functions
def create_utilization_calculator() -> MaterialUtilizationCalculator:
    """Create a new utilization calculator."""
    return MaterialUtilizationCalculator()


def create_utilization_reporter(output_dir: str = "utilization_reports") -> MaterialUtilizationReporter:
    """Create a new utilization reporter."""
    return MaterialUtilizationReporter(output_dir)


def calculate_layer_utilization(layer: CuttingLayer, result: NestingResult) -> MaterialUtilizationReport:
    """Convenience function to calculate layer utilization."""
    calculator = MaterialUtilizationCalculator()
    report = calculator.calculate_utilization(result, layer.material_settings)
    report.layer_id = layer.layer_id
    report.layer_name = layer.name
    return report
