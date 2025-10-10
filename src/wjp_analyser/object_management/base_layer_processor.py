"""
Base Layer Processing System

This module handles base layer processing for full DXF path cutting,
which maintains the original layout and cutting sequence.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import ezdxf
from shapely.geometry import Polygon, LineString, Point
import numpy as np

logger = logging.getLogger(__name__)


class BaseLayerStatus(Enum):
    """Status of base layer processing."""
    CREATED = "created"
    ANALYZED = "analyzed"
    OPTIMIZED = "optimized"
    READY_FOR_CUTTING = "ready_for_cutting"
    ERROR = "error"


@dataclass
class CuttingSequence:
    """Represents the cutting sequence for base layer."""
    sequence_id: str
    object_id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    cutting_path: List[Tuple[float, float]]
    cutting_time: float
    piercing_points: List[Tuple[float, float]]
    lead_in_length: float
    lead_out_length: float
    cutting_speed: float
    piercing_speed: float
    pierce_time: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseLayerResult:
    """Result of base layer processing."""
    success: bool
    total_cutting_time: float
    total_cutting_length: float
    piercing_count: int
    cutting_sequences: List[CuttingSequence]
    bounding_box: Tuple[float, float, float, float]
    material_usage: float
    efficiency_score: float
    status: BaseLayerStatus
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLayerProcessor:
    """Processes base layers for full DXF path cutting."""
    
    def __init__(self):
        """Initialize the base layer processor."""
        self.processed_layers: Dict[str, BaseLayerResult] = {}
        self._sequence_counter = 0
    
    def process_base_layer(self, layer, object_manager) -> BaseLayerResult:
        """
        Process a base layer for full DXF path cutting.
        
        Args:
            layer: The cutting layer to process
            object_manager: Object manager for accessing objects
            
        Returns:
            BaseLayerResult with processing results
        """
        try:
            logger.info(f"Processing base layer: {layer.layer_id}")
            
            # Initialize result
            result = BaseLayerResult(
                success=False,
                total_cutting_time=0.0,
                total_cutting_length=0.0,
                piercing_count=0,
                cutting_sequences=[],
                bounding_box=(0, 0, 0, 0),
                material_usage=0.0,
                efficiency_score=0.0,
                status=BaseLayerStatus.CREATED
            )
            
            if not layer.objects:
                result.errors.append("No objects in layer")
                result.status = BaseLayerStatus.ERROR
                return result
            
            # Analyze objects and create cutting sequences
            cutting_sequences = self._create_cutting_sequences(layer, object_manager)
            result.cutting_sequences = cutting_sequences
            
            # Calculate metrics
            result.total_cutting_time = sum(seq.cutting_time for seq in cutting_sequences)
            result.total_cutting_length = sum(
                self._calculate_path_length(seq.cutting_path) for seq in cutting_sequences
            )
            result.piercing_count = sum(len(seq.piercing_points) for seq in cutting_sequences)
            
            # Calculate bounding box
            result.bounding_box = self._calculate_bounding_box(layer.objects)
            
            # Calculate material usage
            result.material_usage = self._calculate_material_usage(layer.objects)
            
            # Calculate efficiency score
            result.efficiency_score = self._calculate_efficiency_score(result)
            
            result.success = True
            result.status = BaseLayerStatus.READY_FOR_CUTTING
            
            logger.info(f"Base layer processing completed: {len(cutting_sequences)} sequences")
            return result
            
        except Exception as e:
            logger.error(f"Error processing base layer: {e}")
            result.errors.append(f"Processing error: {str(e)}")
            result.status = BaseLayerStatus.ERROR
            return result
    
    def _create_cutting_sequences(self, layer, object_manager) -> List[CuttingSequence]:
        """Create cutting sequences for all objects in the layer."""
        sequences = []
        
        for obj in layer.objects:
            try:
                sequence = self._create_object_sequence(obj, layer.cutting_settings)
                sequences.append(sequence)
            except Exception as e:
                logger.warning(f"Failed to create sequence for object {obj.object_id}: {e}")
                continue
        
        # Sort sequences by priority and position
        sequences.sort(key=lambda s: (s.priority, s.start_point[0] + s.start_point[1]))
        
        return sequences
    
    def _create_object_sequence(self, obj, cutting_settings) -> CuttingSequence:
        """Create cutting sequence for a single object."""
        self._sequence_counter += 1
        
        # Extract cutting path from object geometry
        cutting_path = self._extract_cutting_path(obj)
        
        # Calculate start and end points
        start_point = cutting_path[0] if cutting_path else (0, 0)
        end_point = cutting_path[-1] if cutting_path else (0, 0)
        
        # Calculate cutting time
        cutting_time = self._calculate_cutting_time(cutting_path, cutting_settings)
        
        # Determine piercing points
        piercing_points = self._determine_piercing_points(obj, cutting_settings)
        
        sequence = CuttingSequence(
            sequence_id=f"seq_{self._sequence_counter:06d}",
            object_id=obj.object_id,
            start_point=start_point,
            end_point=end_point,
            cutting_path=cutting_path,
            cutting_time=cutting_time,
            piercing_points=piercing_points,
            lead_in_length=cutting_settings.lead_in_length,
            lead_out_length=cutting_settings.lead_out_length,
            cutting_speed=cutting_settings.cutting_speed,
            piercing_speed=cutting_settings.piercing_speed,
            pierce_time=cutting_settings.pierce_time,
            priority=self._calculate_priority(obj),
            metadata={
                "object_type": obj.object_type.value,
                "complexity": obj.complexity.value,
                "area": obj.geometry.area,
                "perimeter": obj.geometry.perimeter
            }
        )
        
        return sequence
    
    def _extract_cutting_path(self, obj) -> List[Tuple[float, float]]:
        """Extract cutting path from object geometry."""
        try:
            if obj.object_type.value == "circle":
                return self._extract_circle_path(obj)
            elif obj.object_type.value == "polyline":
                return self._extract_polyline_path(obj)
            elif obj.object_type.value == "polygon":
                return self._extract_polygon_path(obj)
            else:
                # Generic path extraction
                return self._extract_generic_path(obj)
        except Exception as e:
            logger.warning(f"Failed to extract cutting path for {obj.object_id}: {e}")
            return []
    
    def _extract_circle_path(self, obj) -> List[Tuple[float, float]]:
        """Extract cutting path for circle objects."""
        # Get circle parameters from entity
        entity = obj.entity
        center = (entity.dxf.center.x, entity.dxf.center.y)
        radius = entity.dxf.radius
        
        # Create circular path with appropriate resolution
        num_points = max(16, int(2 * np.pi * radius / 5))  # 5mm resolution
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        path = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            path.append((x, y))
        
        return path
    
    def _extract_polyline_path(self, obj) -> List[Tuple[float, float]]:
        """Extract cutting path for polyline objects."""
        try:
            entity = obj.entity
            path = []
            
            # Handle different polyline types
            if hasattr(entity, 'get_points'):
                points = entity.get_points()
                for point in points:
                    path.append((point[0], point[1]))
            elif hasattr(entity, 'points'):
                for point in entity.points:
                    path.append((point[0], point[1]))
            else:
                # Fallback to bounding box corners
                bbox = obj.geometry.bounding_box
                path = [
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[0], bbox[3]),
                    (bbox[0], bbox[1])  # Close the path
                ]
            
            return path
        except Exception as e:
            logger.warning(f"Failed to extract polyline path: {e}")
            return []
    
    def _extract_polygon_path(self, obj) -> List[Tuple[float, float]]:
        """Extract cutting path for polygon objects."""
        try:
            # Use Shapely geometry if available
            if hasattr(obj.geometry, 'exterior'):
                coords = list(obj.geometry.exterior.coords)
                return [(coord[0], coord[1]) for coord in coords]
            else:
                # Fallback to bounding box
                bbox = obj.geometry.bounding_box
                return [
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[0], bbox[3]),
                    (bbox[0], bbox[1])
                ]
        except Exception as e:
            logger.warning(f"Failed to extract polygon path: {e}")
            return []
    
    def _extract_generic_path(self, obj) -> List[Tuple[float, float]]:
        """Extract cutting path for generic objects."""
        # Use bounding box as fallback
        bbox = obj.geometry.bounding_box
        return [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
            (bbox[0], bbox[1])
        ]
    
    def _calculate_cutting_time(self, path: List[Tuple[float, float]], settings) -> float:
        """Calculate cutting time for a path."""
        if not path:
            return 0.0
        
        # Calculate path length
        total_length = self._calculate_path_length(path)
        
        # Calculate cutting time
        cutting_time = total_length / settings.cutting_speed * 60  # Convert to seconds
        
        return cutting_time
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total length of a cutting path."""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += distance
        
        return total_length
    
    def _determine_piercing_points(self, obj, settings) -> List[Tuple[float, float]]:
        """Determine piercing points for an object."""
        piercing_points = []
        
        # For closed objects, pierce at start point
        if obj.geometry.is_closed:
            bbox = obj.geometry.bounding_box
            # Pierce at bottom-left corner
            piercing_points.append((bbox[0], bbox[1]))
        
        return piercing_points
    
    def _calculate_priority(self, obj) -> int:
        """Calculate cutting priority for an object."""
        # Higher priority for smaller, simpler objects
        priority = 0
        
        # Size-based priority (smaller objects first)
        if obj.geometry.area < 100:
            priority += 3
        elif obj.geometry.area < 1000:
            priority += 2
        elif obj.geometry.area < 10000:
            priority += 1
        
        # Complexity-based priority (simpler objects first)
        if obj.complexity.value == "simple":
            priority += 2
        elif obj.complexity.value == "moderate":
            priority += 1
        
        return priority
    
    def _calculate_bounding_box(self, objects) -> Tuple[float, float, float, float]:
        """Calculate overall bounding box for all objects."""
        if not objects:
            return (0, 0, 0, 0)
        
        min_x = min(obj.geometry.bounding_box[0] for obj in objects)
        min_y = min(obj.geometry.bounding_box[1] for obj in objects)
        max_x = max(obj.geometry.bounding_box[2] for obj in objects)
        max_y = max(obj.geometry.bounding_box[3] for obj in objects)
        
        return (min_x, min_y, max_x, max_y)
    
    def _calculate_material_usage(self, objects) -> float:
        """Calculate total material usage."""
        return sum(obj.geometry.area for obj in objects)
    
    def _calculate_efficiency_score(self, result) -> float:
        """Calculate efficiency score for base layer processing."""
        if result.total_cutting_time == 0:
            return 0.0
        
        # Efficiency based on cutting time and material usage
        time_efficiency = 1.0 / (1.0 + result.total_cutting_time / 3600)  # Normalize to hours
        material_efficiency = min(1.0, result.material_usage / 1000000)  # Normalize to large area
        
        return (time_efficiency + material_efficiency) / 2.0
    
    def generate_gcode(self, result: BaseLayerResult, output_path: str) -> bool:
        """Generate G-code for base layer cutting."""
        try:
            with open(output_path, 'w') as f:
                f.write("; Base Layer G-code\n")
                f.write("; Generated by WJP Analyser\n")
                f.write(f"; Total cutting time: {result.total_cutting_time:.2f} seconds\n")
                f.write(f"; Total cutting length: {result.total_cutting_length:.2f} mm\n")
                f.write(f"; Piercing count: {result.piercing_count}\n\n")
                
                f.write("G21 ; Set units to millimeters\n")
                f.write("G90 ; Absolute positioning\n")
                f.write("G94 ; Feed rate per minute\n\n")
                
                for i, seq in enumerate(result.cutting_sequences):
                    f.write(f"; Sequence {i+1}: {seq.object_id}\n")
                    f.write(f"G0 X{seq.start_point[0]:.3f} Y{seq.start_point[1]:.3f}\n")
                    
                    # Piercing
                    for pierce_point in seq.piercing_points:
                        f.write(f"G0 X{pierce_point[0]:.3f} Y{pierce_point[1]:.3f}\n")
                        f.write(f"G1 X{pierce_point[0]:.3f} Y{pierce_point[1]:.3f} F{seq.piercing_speed:.1f}\n")
                        f.write(f"G4 P{seq.pierce_time:.3f}\n")
                    
                    # Cutting path
                    f.write(f"G1 F{seq.cutting_speed:.1f}\n")
                    for point in seq.cutting_path:
                        f.write(f"G1 X{point[0]:.3f} Y{point[1]:.3f}\n")
                    
                    f.write("\n")
                
                f.write("M30 ; End of program\n")
            
            logger.info(f"G-code generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate G-code: {e}")
            return False
    
    def get_processing_summary(self, layer_id: str) -> Dict[str, Any]:
        """Get processing summary for a layer."""
        if layer_id not in self.processed_layers:
            return {"error": "Layer not processed"}
        
        result = self.processed_layers[layer_id]
        return {
            "success": result.success,
            "status": result.status.value,
            "total_cutting_time": result.total_cutting_time,
            "total_cutting_length": result.total_cutting_length,
            "piercing_count": result.piercing_count,
            "sequence_count": len(result.cutting_sequences),
            "material_usage": result.material_usage,
            "efficiency_score": result.efficiency_score,
            "bounding_box": result.bounding_box,
            "warnings": result.warnings,
            "errors": result.errors
        }
