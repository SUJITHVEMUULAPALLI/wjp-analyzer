"""
Layer Management System

This module provides comprehensive layer management for organizing DXF objects
into different cutting layers with various optimization strategies.
"""

from __future__ import annotations

import os
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
from datetime import datetime
import json

from .dxf_object_manager import DXFObject, ObjectType, ObjectComplexity

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Types of cutting layers."""
    BASE = "base"              # Full DXF path cutting (original layout)
    NESTED = "nested"          # Optimized material utilization
    CUSTOM = "custom"          # User-defined groupings
    PRIORITY = "priority"      # High-priority objects first
    MATERIAL = "material"      # Grouped by material type
    COMPLEXITY = "complexity"  # Grouped by complexity level


class LayerStatus(Enum):
    """Status of layer processing."""
    CREATED = "created"
    CONFIGURED = "configured"
    OPTIMIZED = "optimized"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MaterialSettings:
    """Material settings for a layer."""
    material_name: str = "Steel"
    thickness: float = 25.0
    width: float = 2000.0
    height: float = 1000.0
    density: float = 7.85  # kg/dm³
    cost_per_kg: float = 2.5  # USD/kg
    cost_per_sheet: float = 500.0  # USD/sheet
    kerf_width: float = 1.1  # mm
    min_spacing: float = 3.0  # mm
    max_spacing: float = 10.0  # mm


@dataclass
class CuttingSettings:
    """Cutting settings for a layer."""
    cutting_speed: float = 1200.0  # mm/min
    piercing_speed: float = 200.0  # mm/min
    rapid_speed: float = 10000.0  # mm/min
    pierce_time: float = 0.5  # seconds
    lead_in_length: float = 5.0  # mm
    lead_out_length: float = 5.0  # mm
    lead_in_angle: float = 45.0  # degrees
    lead_out_angle: float = 45.0  # degrees
    overcut: float = 0.5  # mm
    quality_mode: str = "high"  # high, medium, fast


@dataclass
class NestingSettings:
    """Nesting optimization settings."""
    algorithm: str = "genetic"  # genetic, nfp, simulated_annealing, bottom_left
    rotation_enabled: bool = True
    allowed_rotations: List[float] = field(default_factory=lambda: [0, 90, 180, 270])
    min_rotation_step: float = 15.0  # degrees
    max_iterations: int = 1000
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    convergence_threshold: float = 0.001
    time_limit: float = 300.0  # seconds
    material_utilization_target: float = 0.85  # 85%


@dataclass
class CostAnalysis:
    """Cost analysis for a layer."""
    material_cost: float = 0.0
    cutting_time: float = 0.0  # minutes
    machine_cost: float = 0.0
    labor_cost: float = 0.0
    total_cost: float = 0.0
    material_utilization: float = 0.0
    efficiency_score: float = 0.0
    waste_area: float = 0.0
    waste_percentage: float = 0.0
    sheets_required: int = 0
    estimated_completion_time: float = 0.0  # minutes


@dataclass
class OptimizationResult:
    """Result of layer optimization."""
    success: bool = False
    algorithm_used: str = ""
    iterations_completed: int = 0
    optimization_time: float = 0.0  # seconds
    final_utilization: float = 0.0
    improvement_percentage: float = 0.0
    objects_positioned: int = 0
    objects_failed: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CuttingLayer:
    """Represents a cutting layer with objects and settings."""
    layer_id: str
    name: str
    layer_type: LayerType
    description: str = ""
    objects: List[DXFObject] = field(default_factory=list)
    material_settings: MaterialSettings = field(default_factory=MaterialSettings)
    cutting_settings: CuttingSettings = field(default_factory=CuttingSettings)
    nesting_settings: NestingSettings = field(default_factory=NestingSettings)
    cost_analysis: CostAnalysis = field(default_factory=CostAnalysis)
    optimization_result: Optional[OptimizationResult] = None
    status: LayerStatus = LayerStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayerManager:
    """Manages cutting layers and their configurations."""
    
    def __init__(self):
        """Initialize the layer manager."""
        self.layers: Dict[str, CuttingLayer] = {}
        self.layer_order: List[str] = []
        self._next_layer_id = 1
    
    def create_layer(self, 
                    name: str, 
                    layer_type: LayerType,
                    description: str = "",
                    material_settings: Optional[MaterialSettings] = None,
                    cutting_settings: Optional[CuttingSettings] = None,
                    nesting_settings: Optional[NestingSettings] = None) -> str:
        """
        Create a new cutting layer.
        
        Args:
            name: Name of the layer
            layer_type: Type of layer
            description: Description of the layer
            material_settings: Material configuration
            cutting_settings: Cutting configuration
            nesting_settings: Nesting configuration
            
        Returns:
            Layer ID of the created layer
        """
        layer_id = f"layer_{self._next_layer_id:04d}"
        self._next_layer_id += 1
        
        layer = CuttingLayer(
            layer_id=layer_id,
            name=name,
            layer_type=layer_type,
            description=description,
            material_settings=material_settings or MaterialSettings(),
            cutting_settings=cutting_settings or CuttingSettings(),
            nesting_settings=nesting_settings or NestingSettings()
        )
        
        self.layers[layer_id] = layer
        self.layer_order.append(layer_id)
        
        logger.info(f"Created layer '{name}' ({layer_type.value}) with ID: {layer_id}")
        return layer_id
    
    def delete_layer(self, layer_id: str) -> bool:
        """Delete a layer."""
        if layer_id in self.layers:
            layer_name = self.layers[layer_id].name
            del self.layers[layer_id]
            if layer_id in self.layer_order:
                self.layer_order.remove(layer_id)
            logger.info(f"Deleted layer '{layer_name}' ({layer_id})")
            return True
        return False
    
    def get_layer(self, layer_id: str) -> Optional[CuttingLayer]:
        """Get a layer by ID."""
        return self.layers.get(layer_id)
    
    def get_all_layers(self) -> List[CuttingLayer]:
        """Get all layers in order."""
        return [self.layers[layer_id] for layer_id in self.layer_order if layer_id in self.layers]
    
    def get_layers_by_type(self, layer_type: LayerType) -> List[CuttingLayer]:
        """Get all layers of a specific type."""
        return [layer for layer in self.layers.values() if layer.layer_type == layer_type]
    
    def add_object_to_layer(self, layer_id: str, object_id: str, object_manager) -> bool:
        """Add an object to a layer."""
        if layer_id not in self.layers:
            return False
        
        obj = object_manager.objects.get(object_id)
        if not obj:
            return False
        
        layer = self.layers[layer_id]
        
        # Check if object is already in another layer
        if obj.assigned_layer and obj.assigned_layer != layer_id:
            # Remove from previous layer
            self.remove_object_from_layer(obj.assigned_layer, object_id, object_manager)
        
        # Add to new layer
        layer.objects.append(obj)
        obj.assigned_layer = layer_id
        layer.updated_at = datetime.now()
        
        logger.info(f"Added object {object_id} to layer '{layer.name}' ({layer_id})")
        return True
    
    def remove_object_from_layer(self, layer_id: str, object_id: str, object_manager) -> bool:
        """Remove an object from a layer."""
        if layer_id not in self.layers:
            return False
        
        layer = self.layers[layer_id]
        
        # Find and remove object
        for i, obj in enumerate(layer.objects):
            if obj.object_id == object_id:
                del layer.objects[i]
                obj.assigned_layer = None
                layer.updated_at = datetime.now()
                logger.info(f"Removed object {object_id} from layer '{layer.name}' ({layer_id})")
                return True
        
        return False
    
    def move_object_between_layers(self, object_id: str, from_layer_id: str, to_layer_id: str, object_manager) -> bool:
        """Move an object from one layer to another."""
        if self.remove_object_from_layer(from_layer_id, object_id, object_manager):
            return self.add_object_to_layer(to_layer_id, object_id, object_manager)
        return False
    
    def duplicate_layer(self, layer_id: str, new_name: str) -> Optional[str]:
        """Duplicate a layer with a new name."""
        if layer_id not in self.layers:
            return None
        
        original_layer = self.layers[layer_id]
        
        # Create new layer with same settings
        new_layer_id = self.create_layer(
            name=new_name,
            layer_type=original_layer.layer_type,
            description=f"Copy of {original_layer.description}",
            material_settings=original_layer.material_settings,
            cutting_settings=original_layer.cutting_settings,
            nesting_settings=original_layer.nesting_settings
        )
        
        # Copy objects (without changing their assigned_layer)
        new_layer = self.layers[new_layer_id]
        for obj in original_layer.objects:
            new_layer.objects.append(obj)
            obj.assigned_layer = new_layer_id
        
        logger.info(f"Duplicated layer '{original_layer.name}' as '{new_name}' ({new_layer_id})")
        return new_layer_id
    
    def reorder_layers(self, new_order: List[str]) -> bool:
        """Reorder layers."""
        # Validate that all layer IDs exist
        if not all(layer_id in self.layers for layer_id in new_order):
            return False
        
        # Validate that all existing layers are included
        if set(new_order) != set(self.layer_order):
            return False
        
        self.layer_order = new_order
        logger.info("Reordered layers")
        return True
    
    def create_layer_from_selection(self, 
                                  selected_objects: List[DXFObject],
                                  layer_name: str,
                                  layer_type: LayerType = LayerType.CUSTOM) -> str:
        """Create a layer from selected objects."""
        layer_id = self.create_layer(
            name=layer_name,
            layer_type=layer_type,
            description=f"Layer created from {len(selected_objects)} selected objects"
        )
        
        layer = self.layers[layer_id]
        for obj in selected_objects:
            layer.objects.append(obj)
            obj.assigned_layer = layer_id
        
        logger.info(f"Created layer '{layer_name}' with {len(selected_objects)} objects")
        return layer_id
    
    def create_layers_by_type(self, objects: List[DXFObject]) -> Dict[str, str]:
        """Create layers grouped by object type."""
        type_layers = {}
        
        # Group objects by type
        objects_by_type = {}
        for obj in objects:
            obj_type = obj.object_type.value
            if obj_type not in objects_by_type:
                objects_by_type[obj_type] = []
            objects_by_type[obj_type].append(obj)
        
        # Create layers for each type
        for obj_type, type_objects in objects_by_type.items():
            layer_name = f"{obj_type.title()} Objects"
            layer_id = self.create_layer(
                name=layer_name,
                layer_type=LayerType.CUSTOM,
                description=f"Layer for {obj_type} objects"
            )
            
            layer = self.layers[layer_id]
            for obj in type_objects:
                layer.objects.append(obj)
                obj.assigned_layer = layer_id
            
            type_layers[obj_type] = layer_id
        
        logger.info(f"Created {len(type_layers)} layers grouped by object type")
        return type_layers
    
    def create_layers_by_complexity(self, objects: List[DXFObject]) -> Dict[str, str]:
        """Create layers grouped by complexity."""
        complexity_layers = {}
        
        # Group objects by complexity
        objects_by_complexity = {}
        for obj in objects:
            complexity = obj.complexity.value
            if complexity not in objects_by_complexity:
                objects_by_complexity[complexity] = []
            objects_by_complexity[complexity].append(obj)
        
        # Create layers for each complexity level
        for complexity, complexity_objects in objects_by_complexity.items():
            layer_name = f"{complexity.title()} Complexity"
            layer_id = self.create_layer(
                name=layer_name,
                layer_type=LayerType.COMPLEXITY,
                description=f"Layer for {complexity} complexity objects"
            )
            
            layer = self.layers[layer_id]
            for obj in complexity_objects:
                layer.objects.append(obj)
                obj.assigned_layer = layer_id
            
            complexity_layers[complexity] = layer_id
        
        logger.info(f"Created {len(complexity_layers)} layers grouped by complexity")
        return complexity_layers
    
    def create_base_layer(self, all_objects: List[DXFObject]) -> str:
        """Create a base layer with all objects in original positions."""
        layer_id = self.create_layer(
            name="Base Layer",
            layer_type=LayerType.BASE,
            description="Full DXF path cutting with original layout"
        )
        
        layer = self.layers[layer_id]
        for obj in all_objects:
            layer.objects.append(obj)
            obj.assigned_layer = layer_id
        
        logger.info(f"Created base layer with {len(all_objects)} objects")
        return layer_id
    
    def calculate_layer_statistics(self, layer_id: str) -> Dict[str, Any]:
        """Calculate statistics for a layer."""
        if layer_id not in self.layers:
            return {}
        
        layer = self.layers[layer_id]
        
        if not layer.objects:
            return {
                'object_count': 0,
                'total_area': 0.0,
                'total_perimeter': 0.0,
                'average_area': 0.0,
                'average_perimeter': 0.0,
                'type_distribution': {},
                'complexity_distribution': {}
            }
        
        # Basic statistics
        object_count = len(layer.objects)
        total_area = sum(obj.geometry.area for obj in layer.objects)
        total_perimeter = sum(obj.geometry.perimeter for obj in layer.objects)
        average_area = total_area / object_count
        average_perimeter = total_perimeter / object_count
        
        # Type distribution
        type_distribution = {}
        for obj in layer.objects:
            obj_type = obj.object_type.value
            type_distribution[obj_type] = type_distribution.get(obj_type, 0) + 1
        
        # Complexity distribution
        complexity_distribution = {}
        for obj in layer.objects:
            complexity = obj.complexity.value
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        return {
            'object_count': object_count,
            'total_area': total_area,
            'total_perimeter': total_perimeter,
            'average_area': average_area,
            'average_perimeter': average_perimeter,
            'type_distribution': type_distribution,
            'complexity_distribution': complexity_distribution,
            'layer_type': layer.layer_type.value,
            'status': layer.status.value,
            'created_at': layer.created_at.isoformat(),
            'updated_at': layer.updated_at.isoformat()
        }
    
    def export_layer_configuration(self, layer_id: str, file_path: str) -> bool:
        """Export layer configuration to file."""
        if layer_id not in self.layers:
            return False
        
        try:
            layer = self.layers[layer_id]
            
            export_data = {
                'layer_info': {
                    'layer_id': layer.layer_id,
                    'name': layer.name,
                    'layer_type': layer.layer_type.value,
                    'description': layer.description,
                    'status': layer.status.value,
                    'created_at': layer.created_at.isoformat(),
                    'updated_at': layer.updated_at.isoformat()
                },
                'material_settings': {
                    'material_name': layer.material_settings.material_name,
                    'thickness': layer.material_settings.thickness,
                    'width': layer.material_settings.width,
                    'height': layer.material_settings.height,
                    'density': layer.material_settings.density,
                    'cost_per_kg': layer.material_settings.cost_per_kg,
                    'cost_per_sheet': layer.material_settings.cost_per_sheet,
                    'kerf_width': layer.material_settings.kerf_width,
                    'min_spacing': layer.material_settings.min_spacing,
                    'max_spacing': layer.material_settings.max_spacing
                },
                'cutting_settings': {
                    'cutting_speed': layer.cutting_settings.cutting_speed,
                    'piercing_speed': layer.cutting_settings.piercing_speed,
                    'rapid_speed': layer.cutting_settings.rapid_speed,
                    'pierce_time': layer.cutting_settings.pierce_time,
                    'lead_in_length': layer.cutting_settings.lead_in_length,
                    'lead_out_length': layer.cutting_settings.lead_out_length,
                    'lead_in_angle': layer.cutting_settings.lead_in_angle,
                    'lead_out_angle': layer.cutting_settings.lead_out_angle,
                    'overcut': layer.cutting_settings.overcut,
                    'quality_mode': layer.cutting_settings.quality_mode
                },
                'nesting_settings': {
                    'algorithm': layer.nesting_settings.algorithm,
                    'rotation_enabled': layer.nesting_settings.rotation_enabled,
                    'allowed_rotations': layer.nesting_settings.allowed_rotations,
                    'min_rotation_step': layer.nesting_settings.min_rotation_step,
                    'max_iterations': layer.nesting_settings.max_iterations,
                    'population_size': layer.nesting_settings.population_size,
                    'mutation_rate': layer.nesting_settings.mutation_rate,
                    'crossover_rate': layer.nesting_settings.crossover_rate,
                    'elitism_rate': layer.nesting_settings.elitism_rate,
                    'convergence_threshold': layer.nesting_settings.convergence_threshold,
                    'time_limit': layer.nesting_settings.time_limit,
                    'material_utilization_target': layer.nesting_settings.material_utilization_target
                },
                'object_count': len(layer.objects),
                'object_ids': [obj.object_id for obj in layer.objects]
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported layer configuration for '{layer.name}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export layer configuration: {e}")
            return False
    
    def import_layer_configuration(self, file_path: str, object_manager) -> Optional[str]:
        """Import layer configuration from file."""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            layer_info = import_data['layer_info']
            material_settings = import_data['material_settings']
            cutting_settings = import_data['cutting_settings']
            nesting_settings = import_data['nesting_settings']
            
            # Create layer
            layer_id = self.create_layer(
                name=layer_info['name'],
                layer_type=LayerType(layer_info['layer_type']),
                description=layer_info['description']
            )
            
            layer = self.layers[layer_id]
            
            # Apply settings
            layer.material_settings = MaterialSettings(**material_settings)
            layer.cutting_settings = CuttingSettings(**cutting_settings)
            layer.nesting_settings = NestingSettings(**nesting_settings)
            
            # Add objects if they exist
            object_ids = import_data.get('object_ids', [])
            for obj_id in object_ids:
                if obj_id in object_manager.objects:
                    self.add_object_to_layer(layer_id, obj_id, object_manager)
            
            logger.info(f"Imported layer configuration for '{layer.name}' from {file_path}")
            return layer_id
            
        except Exception as e:
            logger.error(f"Failed to import layer configuration: {e}")
            return None
    
    def get_layer_summary(self) -> Dict[str, Any]:
        """Get summary of all layers."""
        summary = {
            'total_layers': len(self.layers),
            'layers_by_type': {},
            'total_objects': 0,
            'total_area': 0.0,
            'layers': []
        }
        
        # Count by type
        for layer_type in LayerType:
            type_layers = self.get_layers_by_type(layer_type)
            summary['layers_by_type'][layer_type.value] = len(type_layers)
        
        # Layer details
        for layer in self.get_all_layers():
            layer_stats = self.calculate_layer_statistics(layer.layer_id)
            summary['total_objects'] += layer_stats.get('object_count', 0)
            summary['total_area'] += layer_stats.get('total_area', 0.0)
            
            summary['layers'].append({
                'layer_id': layer.layer_id,
                'name': layer.name,
                'type': layer.layer_type.value,
                'status': layer.status.value,
                'object_count': layer_stats.get('object_count', 0),
                'total_area': layer_stats.get('total_area', 0.0),
                'created_at': layer.created_at.isoformat(),
                'updated_at': layer.updated_at.isoformat()
            })
        
        return summary


# Convenience functions
def create_layer_manager() -> LayerManager:
    """Create a new layer manager."""
    return LayerManager()


def create_default_layers(object_manager, all_objects: List[DXFObject]) -> Dict[str, str]:
    """Create default layer structure."""
    layer_manager = LayerManager()
    
    # Create base layer
    base_layer_id = layer_manager.create_base_layer(all_objects)
    
    # Create nested layer for optimization
    nested_layer_id = layer_manager.create_layer(
        name="Nested Layer",
        layer_type=LayerType.NESTED,
        description="Optimized material utilization"
    )
    
    # Create priority layer for important objects
    priority_layer_id = layer_manager.create_layer(
        name="Priority Layer",
        layer_type=LayerType.PRIORITY,
        description="High-priority objects"
    )
    
    return {
        'base': base_layer_id,
        'nested': nested_layer_id,
        'priority': priority_layer_id
    }
