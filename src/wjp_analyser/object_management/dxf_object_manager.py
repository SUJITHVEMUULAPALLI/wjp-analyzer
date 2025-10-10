"""
DXF Object Management System

This module provides comprehensive object identification, classification, and management
for DXF files, enabling interactive object selection and layer-based processing.
"""

from __future__ import annotations

import os
import uuid
import math
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
from datetime import datetime
import ezdxf
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
import numpy as np

logger = logging.getLogger(__name__)


class ObjectType(Enum):
    """Types of DXF objects."""
    POLYGON = "polygon"
    CIRCLE = "circle"
    ARC = "arc"
    LINE = "line"
    POLYLINE = "polyline"
    COMPLEX = "complex"
    TEXT = "text"
    DIMENSION = "dimension"
    UNKNOWN = "unknown"


class ObjectComplexity(Enum):
    """Complexity levels for objects."""
    SIMPLE = "simple"      # Basic shapes (circles, rectangles)
    MODERATE = "moderate"  # Regular polygons, simple curves
    COMPLEX = "complex"    # Irregular shapes, multiple curves
    VERY_COMPLEX = "very_complex"  # Highly detailed, many vertices


@dataclass
class ObjectMetadata:
    """Metadata for DXF objects."""
    layer_name: str = "0"
    color: Optional[int] = None
    line_type: Optional[str] = None
    thickness: float = 0.0
    creation_time: Optional[str] = None
    modification_time: Optional[str] = None
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectGeometry:
    """Geometry information for objects."""
    bounding_box: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    width: float
    height: float
    aspect_ratio: float
    complexity_score: float
    vertex_count: int
    is_closed: bool
    is_convex: bool
    has_holes: bool
    hole_count: int


@dataclass
class DXFObject:
    """Represents a single object in DXF file."""
    object_id: str
    entity: Any
    object_type: ObjectType
    complexity: ObjectComplexity
    geometry: ObjectGeometry
    metadata: ObjectMetadata
    selected: bool = False
    assigned_layer: Optional[str] = None
    nesting_position: Optional[Tuple[float, float]] = None
    nesting_rotation: float = 0.0
    nesting_metadata: Dict[str, Any] = field(default_factory=dict)


class DXFObjectManager:
    """Manages DXF objects and their properties."""
    
    def __init__(self):
        """Initialize the object manager."""
        self.objects: Dict[str, DXFObject] = {}
        self.selected_objects: Set[str] = set()
        self.object_groups: Dict[str, List[str]] = {}
        self._next_object_id = 1
    
    def load_dxf_objects(self, dxf_path: str) -> List[DXFObject]:
        """
        Load and analyze all objects from a DXF file.
        
        Args:
            dxf_path: Path to the DXF file
            
        Returns:
            List of DXFObject instances
        """
        logger.info(f"Loading DXF objects from: {dxf_path}")
        
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            objects = []
            for entity in msp:
                try:
                    obj = self._create_object_from_entity(entity)
                    if obj:
                        objects.append(obj)
                        self.objects[obj.object_id] = obj
                except Exception as e:
                    logger.warning(f"Failed to process entity: {e}")
                    continue
            
            logger.info(f"Loaded {len(objects)} objects from DXF file")
            return objects
            
        except Exception as e:
            logger.error(f"Failed to load DXF file: {e}")
            return []
    
    def _create_object_from_entity(self, entity: Any) -> Optional[DXFObject]:
        """Create a DXFObject from an ezdxf entity."""
        object_id = f"obj_{self._next_object_id:06d}"
        self._next_object_id += 1
        
        # Extract geometry
        geometry = self._extract_geometry(entity)
        if not geometry:
            return None
        
        # Classify object type
        object_type = self._classify_object_type(entity)
        
        # Determine complexity
        complexity = self._determine_complexity(geometry, object_type)
        
        # Extract metadata
        metadata = self._extract_metadata(entity)
        
        return DXFObject(
            object_id=object_id,
            entity=entity,
            object_type=object_type,
            complexity=complexity,
            geometry=geometry,
            metadata=metadata
        )
    
    def _extract_geometry(self, entity: Any) -> Optional[ObjectGeometry]:
        """Extract geometry information from entity."""
        try:
            dxf_type = entity.dxftype()
            
            if dxf_type == 'LWPOLYLINE':
                return self._extract_polyline_geometry(entity)
            elif dxf_type == 'CIRCLE':
                return self._extract_circle_geometry(entity)
            elif dxf_type == 'ARC':
                return self._extract_arc_geometry(entity)
            elif dxf_type == 'LINE':
                return self._extract_line_geometry(entity)
            elif dxf_type == 'POLYLINE':
                return self._extract_polyline_geometry(entity)
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract geometry: {e}")
            return None
    
    def _extract_polyline_geometry(self, entity: Any) -> Optional[ObjectGeometry]:
        """Extract geometry from polyline entity."""
        try:
            points = [(p[0], p[1]) for p in entity.get_points()]
            if len(points) < 2:
                return None
            
            # Create Shapely geometry
            if entity.closed and len(points) >= 3:
                poly = Polygon(points)
                if not poly.is_valid:
                    return None
            else:
                line = LineString(points)
                poly = line.buffer(0.1)  # Convert to polygon for consistency
            
            # Calculate properties
            bounds = poly.bounds
            area = poly.area
            perimeter = poly.length
            centroid = poly.centroid.coords[0]
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_ratio = max(width, height) / max(min(width, height), 0.001)
            
            # Calculate complexity
            vertex_count = len(points)
            complexity_score = self._calculate_complexity_score(poly, vertex_count)
            
            # Check for holes
            has_holes = hasattr(poly, 'interiors') and len(poly.interiors) > 0
            hole_count = len(poly.interiors) if has_holes else 0
            
            return ObjectGeometry(
                bounding_box=bounds,
                area=area,
                perimeter=perimeter,
                centroid=centroid,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                complexity_score=complexity_score,
                vertex_count=vertex_count,
                is_closed=entity.closed if hasattr(entity, 'closed') else False,
                is_convex=poly.convex_hull.area == poly.area,
                has_holes=has_holes,
                hole_count=hole_count
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract polyline geometry: {e}")
            return None
    
    def _extract_circle_geometry(self, entity: Any) -> Optional[ObjectGeometry]:
        """Extract geometry from circle entity."""
        try:
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            
            # Create circle polygon
            circle = Point(center).buffer(radius, resolution=64)
            
            # Calculate properties
            bounds = circle.bounds
            area = circle.area
            perimeter = circle.length
            centroid = center
            width = height = radius * 2
            aspect_ratio = 1.0
            
            # Circle is always simple
            complexity_score = 1.0
            vertex_count = 64  # Resolution
            
            return ObjectGeometry(
                bounding_box=bounds,
                area=area,
                perimeter=perimeter,
                centroid=centroid,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                complexity_score=complexity_score,
                vertex_count=vertex_count,
                is_closed=True,
                is_convex=True,
                has_holes=False,
                hole_count=0
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract circle geometry: {e}")
            return None
    
    def _extract_arc_geometry(self, entity: Any) -> Optional[ObjectGeometry]:
        """Extract geometry from arc entity."""
        try:
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            start_angle = math.radians(entity.dxf.start_angle)
            end_angle = math.radians(entity.dxf.end_angle)
            
            # Create arc polygon (approximate)
            angles = np.linspace(start_angle, end_angle, 32)
            points = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]
            
            if len(points) >= 3:
                poly = Polygon(points)
            else:
                return None
            
            # Calculate properties
            bounds = poly.bounds
            area = poly.area
            perimeter = poly.length
            centroid = poly.centroid.coords[0]
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_ratio = max(width, height) / max(min(width, height), 0.001)
            
            complexity_score = 2.0  # Moderate complexity
            vertex_count = len(points)
            
            return ObjectGeometry(
                bounding_box=bounds,
                area=area,
                perimeter=perimeter,
                centroid=centroid,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                complexity_score=complexity_score,
                vertex_count=vertex_count,
                is_closed=False,
                is_convex=True,
                has_holes=False,
                hole_count=0
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract arc geometry: {e}")
            return None
    
    def _extract_line_geometry(self, entity: Any) -> Optional[ObjectGeometry]:
        """Extract geometry from line entity."""
        try:
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            
            # Create line polygon (buffer for area calculation)
            line = LineString([start, end])
            poly = line.buffer(0.1)  # Small buffer for area calculation
            
            # Calculate properties
            bounds = poly.bounds
            area = poly.area
            perimeter = poly.length
            centroid = line.centroid.coords[0]
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_ratio = max(width, height) / max(min(width, height), 0.001)
            
            complexity_score = 1.0  # Simple
            vertex_count = 2
            
            return ObjectGeometry(
                bounding_box=bounds,
                area=area,
                perimeter=perimeter,
                centroid=centroid,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                complexity_score=complexity_score,
                vertex_count=vertex_count,
                is_closed=False,
                is_convex=True,
                has_holes=False,
                hole_count=0
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract line geometry: {e}")
            return None
    
    def _classify_object_type(self, entity: Any) -> ObjectType:
        """Classify the type of object."""
        dxf_type = entity.dxftype()
        
        type_mapping = {
            'LWPOLYLINE': ObjectType.POLYLINE,
            'POLYLINE': ObjectType.POLYLINE,
            'CIRCLE': ObjectType.CIRCLE,
            'ARC': ObjectType.ARC,
            'LINE': ObjectType.LINE,
            'TEXT': ObjectType.TEXT,
            'MTEXT': ObjectType.TEXT,
            'DIMENSION': ObjectType.DIMENSION,
            'DIMENSION_ALIGNED': ObjectType.DIMENSION,
            'DIMENSION_ANGULAR': ObjectType.DIMENSION,
            'DIMENSION_DIAMETER': ObjectType.DIMENSION,
            'DIMENSION_RADIUS': ObjectType.DIMENSION
        }
        
        return type_mapping.get(dxf_type, ObjectType.UNKNOWN)
    
    def _determine_complexity(self, geometry: ObjectGeometry, object_type: ObjectType) -> ObjectComplexity:
        """Determine object complexity based on geometry."""
        if object_type in [ObjectType.CIRCLE, ObjectType.LINE]:
            return ObjectComplexity.SIMPLE
        
        if geometry.complexity_score < 2.0:
            return ObjectComplexity.SIMPLE
        elif geometry.complexity_score < 4.0:
            return ObjectComplexity.MODERATE
        elif geometry.complexity_score < 8.0:
            return ObjectComplexity.COMPLEX
        else:
            return ObjectComplexity.VERY_COMPLEX
    
    def _calculate_complexity_score(self, poly: Polygon, vertex_count: int) -> float:
        """Calculate complexity score for a polygon."""
        try:
            # Base score from vertex count
            vertex_score = min(vertex_count / 10.0, 5.0)
            
            # Convexity factor
            convexity_factor = 1.0 if poly.convex_hull.area == poly.area else 2.0
            
            # Aspect ratio factor
            bounds = poly.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_ratio = max(width, height) / max(min(width, height), 0.001)
            aspect_factor = min(aspect_ratio / 5.0, 2.0)
            
            # Hole factor
            hole_factor = 1.0 + len(poly.interiors) * 0.5
            
            return vertex_score * convexity_factor * aspect_factor * hole_factor
            
        except Exception:
            return 1.0
    
    def _extract_metadata(self, entity: Any) -> ObjectMetadata:
        """Extract metadata from entity."""
        try:
            return ObjectMetadata(
                layer_name=getattr(entity.dxf, 'layer', '0'),
                color=getattr(entity.dxf, 'color', None),
                line_type=getattr(entity.dxf, 'linetype', None),
                thickness=getattr(entity.dxf, 'thickness', 0.0)
            )
        except Exception:
            return ObjectMetadata()
    
    def select_object(self, object_id: str) -> bool:
        """Select an object."""
        if object_id in self.objects:
            self.objects[object_id].selected = True
            self.selected_objects.add(object_id)
            return True
        return False
    
    def deselect_object(self, object_id: str) -> bool:
        """Deselect an object."""
        if object_id in self.objects:
            self.objects[object_id].selected = False
            self.selected_objects.discard(object_id)
            return True
        return False
    
    def select_all_objects(self) -> int:
        """Select all objects."""
        count = 0
        for obj in self.objects.values():
            if not obj.selected:
                obj.selected = True
                self.selected_objects.add(obj.object_id)
                count += 1
        return count
    
    def deselect_all_objects(self) -> int:
        """Deselect all objects."""
        count = len(self.selected_objects)
        for obj_id in self.selected_objects:
            self.objects[obj_id].selected = False
        self.selected_objects.clear()
        return count
    
    def select_objects_by_type(self, object_type: ObjectType) -> int:
        """Select all objects of a specific type."""
        count = 0
        for obj in self.objects.values():
            if obj.object_type == object_type and not obj.selected:
                obj.selected = True
                self.selected_objects.add(obj.object_id)
                count += 1
        return count
    
    def select_objects_by_complexity(self, complexity: ObjectComplexity) -> int:
        """Select all objects of a specific complexity."""
        count = 0
        for obj in self.objects.values():
            if obj.complexity == complexity and not obj.selected:
                obj.selected = True
                self.selected_objects.add(obj.object_id)
                count += 1
        return count
    
    def filter_objects(self, 
                      min_area: Optional[float] = None,
                      max_area: Optional[float] = None,
                      min_perimeter: Optional[float] = None,
                      max_perimeter: Optional[float] = None,
                      object_types: Optional[List[ObjectType]] = None,
                      complexities: Optional[List[ObjectComplexity]] = None) -> List[DXFObject]:
        """Filter objects based on criteria."""
        filtered = []
        
        for obj in self.objects.values():
            # Area filter
            if min_area is not None and obj.geometry.area < min_area:
                continue
            if max_area is not None and obj.geometry.area > max_area:
                continue
            
            # Perimeter filter
            if min_perimeter is not None and obj.geometry.perimeter < min_perimeter:
                continue
            if max_perimeter is not None and obj.geometry.perimeter > max_perimeter:
                continue
            
            # Type filter
            if object_types is not None and obj.object_type not in object_types:
                continue
            
            # Complexity filter
            if complexities is not None and obj.complexity not in complexities:
                continue
            
            filtered.append(obj)
        
        return filtered
    
    def get_object_statistics(self) -> Dict[str, Any]:
        """Get statistics about all objects."""
        if not self.objects:
            return {}
        
        total_objects = len(self.objects)
        selected_objects = len(self.selected_objects)
        
        # Type distribution
        type_counts = {}
        for obj in self.objects.values():
            type_counts[obj.object_type.value] = type_counts.get(obj.object_type.value, 0) + 1
        
        # Complexity distribution
        complexity_counts = {}
        for obj in self.objects.values():
            complexity_counts[obj.complexity.value] = complexity_counts.get(obj.complexity.value, 0) + 1
        
        # Area statistics
        areas = [obj.geometry.area for obj in self.objects.values()]
        total_area = sum(areas)
        
        # Perimeter statistics
        perimeters = [obj.geometry.perimeter for obj in self.objects.values()]
        total_perimeter = sum(perimeters)
        
        return {
            'total_objects': total_objects,
            'selected_objects': selected_objects,
            'type_distribution': type_counts,
            'complexity_distribution': complexity_counts,
            'total_area': total_area,
            'average_area': total_area / total_objects if total_objects > 0 else 0,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'total_perimeter': total_perimeter,
            'average_perimeter': total_perimeter / total_objects if total_objects > 0 else 0,
            'min_perimeter': min(perimeters) if perimeters else 0,
            'max_perimeter': max(perimeters) if perimeters else 0
        }
    
    def export_selection(self, file_path: str) -> bool:
        """Export selected objects to file."""
        try:
            selected_objects = [self.objects[obj_id] for obj_id in self.selected_objects]
            
            export_data = {
                'selection_info': {
                    'total_selected': len(selected_objects),
                    'export_time': str(datetime.now()),
                    'file_path': file_path
                },
                'objects': []
            }
            
            for obj in selected_objects:
                obj_data = {
                    'object_id': obj.object_id,
                    'object_type': obj.object_type.value,
                    'complexity': obj.complexity.value,
                    'geometry': {
                        'area': obj.geometry.area,
                        'perimeter': obj.geometry.perimeter,
                        'width': obj.geometry.width,
                        'height': obj.geometry.height,
                        'aspect_ratio': obj.geometry.aspect_ratio,
                        'complexity_score': obj.geometry.complexity_score,
                        'vertex_count': obj.geometry.vertex_count,
                        'is_closed': obj.geometry.is_closed,
                        'is_convex': obj.geometry.is_convex,
                        'has_holes': obj.geometry.has_holes,
                        'hole_count': obj.geometry.hole_count
                    },
                    'metadata': {
                        'layer_name': obj.metadata.layer_name,
                        'color': obj.metadata.color,
                        'line_type': obj.metadata.line_type,
                        'thickness': obj.metadata.thickness
                    }
                }
                export_data['objects'].append(obj_data)
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(selected_objects)} objects to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export selection: {e}")
            return False
    
    def import_selection(self, file_path: str) -> bool:
        """Import object selection from file."""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            for obj_data in import_data.get('objects', []):
                object_id = obj_data.get('object_id')
                if object_id in self.objects:
                    self.select_object(object_id)
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} object selections from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import selection: {e}")
            return False


# Convenience functions
def load_dxf_objects(dxf_path: str) -> List[DXFObject]:
    """Convenience function to load DXF objects."""
    manager = DXFObjectManager()
    return manager.load_dxf_objects(dxf_path)


def create_object_manager() -> DXFObjectManager:
    """Create a new DXF object manager."""
    return DXFObjectManager()
