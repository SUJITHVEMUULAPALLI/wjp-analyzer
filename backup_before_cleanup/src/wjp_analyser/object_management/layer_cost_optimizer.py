"""
Layer-Based Cost Calculation and Optimization System

This module provides comprehensive cost analysis and optimization
for different layer types and cutting strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs in layer processing."""
    MATERIAL = "material"
    CUTTING = "cutting"
    SETUP = "setup"
    LABOR = "labor"
    WASTE = "waste"
    OVERHEAD = "overhead"


class OptimizationStrategy(Enum):
    """Optimization strategies for cost reduction."""
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_MATERIAL = "minimize_material"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCED = "balanced"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a layer."""
    material_cost: float = 0.0
    cutting_cost: float = 0.0
    setup_cost: float = 0.0
    labor_cost: float = 0.0
    waste_cost: float = 0.0
    overhead_cost: float = 0.0
    total_cost: float = 0.0
    
    def calculate_total(self):
        """Calculate total cost."""
        self.total_cost = (
            self.material_cost +
            self.cutting_cost +
            self.setup_cost +
            self.labor_cost +
            self.waste_cost +
            self.overhead_cost
        )


@dataclass
class CostOptimizationResult:
    """Result of cost optimization."""
    success: bool
    original_cost: float
    optimized_cost: float
    cost_savings: float
    savings_percentage: float
    optimization_strategy: OptimizationStrategy
    recommended_settings: Dict[str, Any]
    cost_breakdown: CostBreakdown
    efficiency_improvement: float
    processing_time_change: float
    material_usage_change: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayerCostCalculator:
    """Calculates costs for different layer types and configurations."""
    
    def __init__(self):
        """Initialize the cost calculator."""
        self.cost_models: Dict[str, Dict[str, float]] = {
            "steel": {
                "cost_per_kg": 2.5,
                "density": 7.85,  # kg/dm³
                "cost_per_sheet": 500.0,
                "waste_factor": 0.15
            },
            "aluminum": {
                "cost_per_kg": 3.2,
                "density": 2.7,
                "cost_per_sheet": 400.0,
                "waste_factor": 0.12
            },
            "stainless_steel": {
                "cost_per_kg": 8.5,
                "density": 7.9,
                "cost_per_sheet": 1200.0,
                "waste_factor": 0.18
            }
        }
        
        self.labor_rates = {
            "operator": 25.0,  # USD/hour
            "programmer": 45.0,
            "supervisor": 35.0
        }
        
        self.overhead_rates = {
            "equipment_depreciation": 0.15,
            "facility_cost": 0.10,
            "utilities": 0.05,
            "maintenance": 0.08
        }
    
    def calculate_layer_cost(self, layer, processing_result=None) -> CostBreakdown:
        """
        Calculate comprehensive cost breakdown for a layer.
        
        Args:
            layer: The cutting layer
            processing_result: Optional processing result for time calculations
            
        Returns:
            CostBreakdown with detailed cost analysis
        """
        breakdown = CostBreakdown()
        
        try:
            # Material cost calculation
            breakdown.material_cost = self._calculate_material_cost(layer)
            
            # Cutting cost calculation
            breakdown.cutting_cost = self._calculate_cutting_cost(layer, processing_result)
            
            # Setup cost calculation
            breakdown.setup_cost = self._calculate_setup_cost(layer)
            
            # Labor cost calculation
            breakdown.labor_cost = self._calculate_labor_cost(layer, processing_result)
            
            # Waste cost calculation
            breakdown.waste_cost = self._calculate_waste_cost(layer)
            
            # Overhead cost calculation
            breakdown.overhead_cost = self._calculate_overhead_cost(breakdown)
            
            # Calculate total
            breakdown.calculate_total()
            
            logger.info(f"Cost calculation completed for layer {layer.layer_id}: ${breakdown.total_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating layer cost: {e}")
            breakdown.total_cost = 0.0
        
        return breakdown
    
    def _calculate_material_cost(self, layer) -> float:
        """Calculate material cost for the layer."""
        material_settings = layer.material_settings
        material_name = material_settings.material_name.lower()
        
        if material_name not in self.cost_models:
            logger.warning(f"Unknown material: {material_name}, using steel rates")
            material_name = "steel"
        
        cost_model = self.cost_models[material_name]
        
        # Calculate material area used
        total_object_area = sum(obj.geometry.area for obj in layer.objects)
        
        # Add waste factor
        waste_factor = cost_model["waste_factor"]
        total_material_area = total_object_area * (1 + waste_factor)
        
        # Calculate cost based on sheet cost
        sheet_area = material_settings.width * material_settings.height
        sheets_needed = np.ceil(total_material_area / sheet_area)
        
        material_cost = sheets_needed * cost_model["cost_per_sheet"]
        
        return material_cost
    
    def _calculate_cutting_cost(self, layer, processing_result=None) -> float:
        """Calculate cutting operation cost."""
        cutting_settings = layer.cutting_settings
        
        # Base cutting cost per minute
        cutting_cost_per_minute = 15.0  # USD/minute
        
        # Calculate cutting time
        if processing_result and hasattr(processing_result, 'total_cutting_time'):
            cutting_time_minutes = processing_result.total_cutting_time / 60
        else:
            # Estimate cutting time based on object complexity
            total_perimeter = sum(obj.geometry.perimeter for obj in layer.objects)
            cutting_time_minutes = total_perimeter / cutting_settings.cutting_speed * 60
        
        cutting_cost = cutting_time_minutes * cutting_cost_per_minute
        
        return cutting_cost
    
    def _calculate_setup_cost(self, layer) -> float:
        """Calculate setup and preparation cost."""
        # Fixed setup cost per layer
        base_setup_cost = 50.0  # USD
        
        # Additional setup cost based on complexity
        complexity_factor = 0
        for obj in layer.objects:
            if obj.complexity.value == "complex":
                complexity_factor += 2
            elif obj.complexity.value == "moderate":
                complexity_factor += 1
            elif obj.complexity.value == "very_complex":
                complexity_factor += 3
        
        setup_cost = base_setup_cost + (complexity_factor * 10.0)
        
        return setup_cost
    
    def _calculate_labor_cost(self, layer, processing_result=None) -> float:
        """Calculate labor cost for the layer."""
        # Calculate processing time
        if processing_result and hasattr(processing_result, 'total_cutting_time'):
            processing_time_hours = processing_result.total_cutting_time / 3600
        else:
            # Estimate processing time
            total_area = sum(obj.geometry.area for obj in layer.objects)
            processing_time_hours = total_area / 10000  # Rough estimate
        
        # Add setup time
        setup_time_hours = 0.5  # 30 minutes setup
        
        total_time_hours = processing_time_hours + setup_time_hours
        
        # Calculate labor cost
        operator_cost = total_time_hours * self.labor_rates["operator"]
        programmer_cost = setup_time_hours * self.labor_rates["programmer"]
        
        labor_cost = operator_cost + programmer_cost
        
        return labor_cost
    
    def _calculate_waste_cost(self, layer) -> float:
        """Calculate waste material cost."""
        material_settings = layer.material_settings
        material_name = material_settings.material_name.lower()
        
        if material_name not in self.cost_models:
            material_name = "steel"
        
        cost_model = self.cost_models[material_name]
        
        # Calculate waste area
        total_object_area = sum(obj.geometry.area for obj in layer.objects)
        waste_factor = cost_model["waste_factor"]
        waste_area = total_object_area * waste_factor
        
        # Calculate waste cost
        sheet_area = material_settings.width * material_settings.height
        waste_cost_per_sheet = cost_model["cost_per_sheet"] * (waste_area / sheet_area)
        
        return waste_cost_per_sheet
    
    def _calculate_overhead_cost(self, breakdown: CostBreakdown) -> float:
        """Calculate overhead costs."""
        direct_costs = (
            breakdown.material_cost +
            breakdown.cutting_cost +
            breakdown.setup_cost +
            breakdown.labor_cost
        )
        
        # Apply overhead rates
        total_overhead_rate = sum(self.overhead_rates.values())
        overhead_cost = direct_costs * total_overhead_rate
        
        return overhead_cost


class LayerCostOptimizer:
    """Optimizes layer configurations for cost reduction."""
    
    def __init__(self):
        """Initialize the cost optimizer."""
        self.cost_calculator = LayerCostCalculator()
        self.optimization_history: List[CostOptimizationResult] = []
    
    def optimize_layer_cost(self, layer, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> CostOptimizationResult:
        """
        Optimize layer configuration for cost reduction.
        
        Args:
            layer: The cutting layer to optimize
            strategy: Optimization strategy to use
            
        Returns:
            CostOptimizationResult with optimization details
        """
        try:
            logger.info(f"Starting cost optimization for layer {layer.layer_id}")
            
            # Calculate original cost
            original_cost_breakdown = self.cost_calculator.calculate_layer_cost(layer)
            original_cost = original_cost_breakdown.total_cost
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(layer, strategy)
            
            # Apply optimizations
            optimized_layer = self._apply_optimizations(layer, recommendations)
            
            # Calculate optimized cost
            optimized_cost_breakdown = self.cost_calculator.calculate_layer_cost(optimized_layer)
            optimized_cost = optimized_cost_breakdown.total_cost
            
            # Calculate savings
            cost_savings = original_cost - optimized_cost
            savings_percentage = (cost_savings / original_cost * 100) if original_cost > 0 else 0
            
            # Calculate efficiency improvement
            efficiency_improvement = self._calculate_efficiency_improvement(
                original_cost_breakdown, optimized_cost_breakdown
            )
            
            result = CostOptimizationResult(
                success=True,
                original_cost=original_cost,
                optimized_cost=optimized_cost,
                cost_savings=cost_savings,
                savings_percentage=savings_percentage,
                optimization_strategy=strategy,
                recommended_settings=recommendations,
                cost_breakdown=optimized_cost_breakdown,
                efficiency_improvement=efficiency_improvement,
                processing_time_change=0.0,  # Would need processing results to calculate
                material_usage_change=0.0,  # Would need material analysis to calculate
                metadata={
                    "optimization_timestamp": datetime.now().isoformat(),
                    "layer_id": layer.layer_id,
                    "object_count": len(layer.objects)
                }
            )
            
            self.optimization_history.append(result)
            logger.info(f"Cost optimization completed: {savings_percentage:.1f}% savings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during cost optimization: {e}")
            return CostOptimizationResult(
                success=False,
                original_cost=0.0,
                optimized_cost=0.0,
                cost_savings=0.0,
                savings_percentage=0.0,
                optimization_strategy=strategy,
                recommended_settings={},
                cost_breakdown=CostBreakdown(),
                efficiency_improvement=0.0,
                processing_time_change=0.0,
                material_usage_change=0.0,
                errors=[f"Optimization error: {str(e)}"]
            )
    
    def _generate_optimization_recommendations(self, layer, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Generate optimization recommendations based on strategy."""
        recommendations = {}
        
        if strategy == OptimizationStrategy.MINIMIZE_MATERIAL:
            recommendations.update({
                "nesting_algorithm": "genetic",
                "rotation_enabled": True,
                "material_utilization_target": 0.95,
                "max_iterations": 2000,
                "sheet_optimization": True
            })
        
        elif strategy == OptimizationStrategy.MINIMIZE_TIME:
            recommendations.update({
                "nesting_algorithm": "bottom_left",
                "rotation_enabled": False,
                "material_utilization_target": 0.75,
                "max_iterations": 500,
                "cutting_speed_multiplier": 1.2
            })
        
        elif strategy == OptimizationStrategy.MINIMIZE_COST:
            recommendations.update({
                "nesting_algorithm": "genetic",
                "rotation_enabled": True,
                "material_utilization_target": 0.85,
                "max_iterations": 1000,
                "waste_reduction": True,
                "setup_optimization": True
            })
        
        elif strategy == OptimizationStrategy.MAXIMIZE_EFFICIENCY:
            recommendations.update({
                "nesting_algorithm": "simulated_annealing",
                "rotation_enabled": True,
                "material_utilization_target": 0.90,
                "max_iterations": 1500,
                "efficiency_focus": True
            })
        
        else:  # BALANCED
            recommendations.update({
                "nesting_algorithm": "genetic",
                "rotation_enabled": True,
                "material_utilization_target": 0.85,
                "max_iterations": 1000,
                "balanced_optimization": True
            })
        
        return recommendations
    
    def _apply_optimizations(self, layer, recommendations: Dict[str, Any]):
        """Apply optimization recommendations to layer settings."""
        # Create a copy of the layer for optimization
        optimized_layer = layer
        
        # Update nesting settings
        if "nesting_algorithm" in recommendations:
            optimized_layer.nesting_settings.algorithm = recommendations["nesting_algorithm"]
        
        if "rotation_enabled" in recommendations:
            optimized_layer.nesting_settings.rotation_enabled = recommendations["rotation_enabled"]
        
        if "material_utilization_target" in recommendations:
            optimized_layer.nesting_settings.material_utilization_target = recommendations["material_utilization_target"]
        
        if "max_iterations" in recommendations:
            optimized_layer.nesting_settings.max_iterations = recommendations["max_iterations"]
        
        # Update cutting settings
        if "cutting_speed_multiplier" in recommendations:
            optimized_layer.cutting_settings.cutting_speed *= recommendations["cutting_speed_multiplier"]
        
        return optimized_layer
    
    def _calculate_efficiency_improvement(self, original: CostBreakdown, optimized: CostBreakdown) -> float:
        """Calculate efficiency improvement percentage."""
        if original.total_cost == 0:
            return 0.0
        
        efficiency_improvement = ((original.total_cost - optimized.total_cost) / original.total_cost) * 100
        return max(0.0, efficiency_improvement)
    
    def compare_optimization_strategies(self, layer) -> Dict[str, CostOptimizationResult]:
        """Compare different optimization strategies for a layer."""
        strategies = [
            OptimizationStrategy.MINIMIZE_TIME,
            OptimizationStrategy.MINIMIZE_MATERIAL,
            OptimizationStrategy.MINIMIZE_COST,
            OptimizationStrategy.MAXIMIZE_EFFICIENCY,
            OptimizationStrategy.BALANCED
        ]
        
        results = {}
        for strategy in strategies:
            result = self.optimize_layer_cost(layer, strategy)
            results[strategy.value] = result
        
        return results
    
    def get_optimization_summary(self, layer_id: str) -> Dict[str, Any]:
        """Get optimization summary for a layer."""
        layer_optimizations = [opt for opt in self.optimization_history if opt.metadata.get("layer_id") == layer_id]
        
        if not layer_optimizations:
            return {"error": "No optimizations found for layer"}
        
        best_optimization = max(layer_optimizations, key=lambda x: x.savings_percentage)
        
        return {
            "total_optimizations": len(layer_optimizations),
            "best_strategy": best_optimization.optimization_strategy.value,
            "best_savings_percentage": best_optimization.savings_percentage,
            "best_cost_savings": best_optimization.cost_savings,
            "average_savings": sum(opt.savings_percentage for opt in layer_optimizations) / len(layer_optimizations),
            "optimization_history": [
                {
                    "strategy": opt.optimization_strategy.value,
                    "savings_percentage": opt.savings_percentage,
                    "cost_savings": opt.cost_savings,
                    "timestamp": opt.metadata.get("optimization_timestamp")
                }
                for opt in layer_optimizations
            ]
        }
