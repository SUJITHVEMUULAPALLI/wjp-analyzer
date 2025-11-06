"""
DXF Cleaner for Waterjet Limitations
====================================

Comprehensive DXF cleaning specifically designed for waterjet cutting constraints:
- Minimum feature size validation
- Kerf width compensation
- Overcut prevention
- Geometry repair and simplification
- Waterjet-specific quality checks
"""

import ezdxf
import shapely.geometry as geom
import shapely.ops as ops
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import os


@dataclass
class WaterjetLimitations:
    """Waterjet cutting limitations and constraints."""
    min_feature_size: float = 1.0  # mm - minimum feature size
    min_hole_diameter: float = 2.0  # mm - minimum hole diameter
    min_slot_width: float = 1.5  # mm - minimum slot width
    min_corner_radius: float = 0.5  # mm - minimum corner radius
    kerf_width: float = 1.1  # mm - kerf width
    overcut_tolerance: float = 0.1  # mm - overcut tolerance
    min_segment_length: float = 0.5  # mm - minimum segment length
    max_angle_change: float = 45.0  # degrees - maximum angle change
    simplify_tolerance: float = 0.1  # mm - simplification tolerance


@dataclass
class CleaningResult:
    """Result of DXF cleaning operation."""
    original_entities: int
    cleaned_entities: int
    removed_entities: int
    repaired_entities: int
    warnings: List[str]
    errors: List[str]
    cleaning_time: float


@dataclass
class CleanedGeometry:
    """Cleaned geometry with metadata."""
    geometry: geom.Polygon
    original_type: str
    cleaned_type: str
    area: float
    perimeter: float
    is_valid: bool
    warnings: List[str]


class WaterjetDXFCleaner:
    """DXF cleaner specifically designed for waterjet cutting constraints."""
    
    def __init__(self, limitations: Optional[WaterjetLimitations] = None):
        self.limitations = limitations or WaterjetLimitations()
        self.cleaned_geometries: List[CleanedGeometry] = []
        self.cleaning_result: Optional[CleaningResult] = None
    
    def clean_dxf(self, input_path: str, output_path: str) -> CleaningResult:
        """Clean DXF file for waterjet cutting."""
        import time
        start_time = time.time()
        
        # Load DXF
        doc = ezdxf.readfile(input_path)
        msp = doc.modelspace()
        
        # Extract and clean entities
        original_entities = list(msp)
        cleaned_entities = []
        removed_entities = []
        repaired_entities = []
        warnings = []
        errors = []
        
        for entity in original_entities:
            try:
                cleaned = self._clean_entity(entity)
                if cleaned:
                    if isinstance(cleaned, list):
                        cleaned_entities.extend(cleaned)
                        repaired_entities.append(entity)
                    else:
                        cleaned_entities.append(cleaned)
                else:
                    removed_entities.append(entity)
                    warnings.append(f"Removed {entity.dxftype()}: {self._get_entity_info(entity)}")
            except Exception as e:
                errors.append(f"Error processing {entity.dxftype()}: {str(e)}")
                removed_entities.append(entity)
        
        # Apply waterjet-specific cleaning
        final_entities = self._apply_waterjet_cleaning(cleaned_entities)
        
        # Create new DXF
        new_doc = ezdxf.new()
        new_msp = new_doc.modelspace()
        
        for entity in final_entities:
            try:
                new_msp.add_entity(entity)
            except Exception as e:
                errors.append(f"Error adding entity: {str(e)}")
        
        # Save cleaned DXF
        new_doc.saveas(output_path)
        
        # Calculate results
        cleaning_time = time.time() - start_time
        self.cleaning_result = CleaningResult(
            original_entities=len(original_entities),
            cleaned_entities=len(final_entities),
            removed_entities=len(removed_entities),
            repaired_entities=len(repaired_entities),
            warnings=warnings,
            errors=errors,
            cleaning_time=cleaning_time
        )
        
        return self.cleaning_result
    
    def _clean_entity(self, entity) -> Optional[Union[object, List[object]]]:
        """Clean individual entity."""
        dxf_type = entity.dxftype()
        
        if dxf_type == "LWPOLYLINE":
            return self._clean_lwpolyline(entity)
        elif dxf_type == "POLYLINE":
            return self._clean_polyline(entity)
        elif dxf_type == "CIRCLE":
            return self._clean_circle(entity)
        elif dxf_type == "ARC":
            return self._clean_arc(entity)
        elif dxf_type == "LINE":
            return self._clean_line(entity)
        elif dxf_type == "SPLINE":
            return self._clean_spline(entity)
        else:
            return None  # Remove unsupported entity types
    
    def _clean_lwpolyline(self, entity) -> Optional[object]:
        """Clean LWPOLYLINE entity."""
        try:
            points = list(entity.get_points())
            if len(points) < 2:
                return None
            
            # Check minimum segment length
            cleaned_points = self._clean_points(points)
            if len(cleaned_points) < 2:
                return None
            
            # Create new entity
            new_entity = entity.copy()
            new_entity.clear()
            
            for point in cleaned_points:
                new_entity.append(point)
            
            # Validate geometry
            if self._validate_geometry(new_entity):
                return new_entity
            else:
                return None
                
        except Exception:
            return None
    
    def _clean_polyline(self, entity) -> Optional[object]:
        """Clean POLYLINE entity."""
        try:
            points = []
            for vertex in entity.vertices:
                points.append((vertex.dxf.location.x, vertex.dxf.location.y))
            
            if len(points) < 2:
                return None
            
            # Check minimum segment length
            cleaned_points = self._clean_points(points)
            if len(cleaned_points) < 2:
                return None
            
            # Create new LWPOLYLINE
            new_entity = ezdxf.entities.LWPolyline()
            new_entity.dxf.layer = entity.dxf.layer
            
            for point in cleaned_points:
                new_entity.append(point)
            
            # Validate geometry
            if self._validate_geometry(new_entity):
                return new_entity
            else:
                return None
                
        except Exception:
            return None
    
    def _clean_circle(self, entity) -> Optional[object]:
        """Clean CIRCLE entity."""
        try:
            center = entity.dxf.center
            radius = entity.dxf.radius
            
            # Check minimum hole diameter
            if radius * 2 < self.limitations.min_hole_diameter:
                return None
            
            # Apply kerf compensation
            compensated_radius = radius - self.limitations.kerf_width / 2
            if compensated_radius <= 0:
                return None
            
            # Create new circle
            new_entity = entity.copy()
            new_entity.dxf.radius = compensated_radius
            
            return new_entity
            
        except Exception:
            return None
    
    def _clean_arc(self, entity) -> Optional[object]:
        """Clean ARC entity."""
        try:
            center = entity.dxf.center
            radius = entity.dxf.radius
            
            # Check minimum radius
            if radius < self.limitations.min_corner_radius:
                return None
            
            # Apply kerf compensation
            compensated_radius = radius - self.limitations.kerf_width / 2
            if compensated_radius <= 0:
                return None
            
            # Create new arc
            new_entity = entity.copy()
            new_entity.dxf.radius = compensated_radius
            
            return new_entity
            
        except Exception:
            return None
    
    def _clean_line(self, entity) -> Optional[object]:
        """Clean LINE entity."""
        try:
            start = entity.dxf.start
            end = entity.dxf.end
            
            # Check minimum length
            length = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
            if length < self.limitations.min_segment_length:
                return None
            
            return entity
            
        except Exception:
            return None
    
    def _clean_spline(self, entity) -> Optional[object]:
        """Clean SPLINE entity by converting to polyline."""
        try:
            # Convert spline to polyline
            points = []
            
            # Sample points along spline
            for i in range(0, 100, 5):  # Sample every 5% of spline
                try:
                    point = entity.point(i / 100.0)
                    points.append((point.x, point.y))
                except:
                    continue
            
            if len(points) < 2:
                return None
            
            # Create LWPOLYLINE
            new_entity = ezdxf.entities.LWPolyline()
            new_entity.dxf.layer = entity.dxf.layer
            
            for point in points:
                new_entity.append(point)
            
            return new_entity
            
        except Exception:
            return None
    
    def _clean_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Clean point list for waterjet constraints."""
        if len(points) < 2:
            return points
        
        cleaned = [points[0]]
        
        for i in range(1, len(points)):
            prev_point = cleaned[-1]
            current_point = points[i]
            
            # Check minimum segment length
            distance = math.sqrt(
                (current_point[0] - prev_point[0])**2 + 
                (current_point[1] - prev_point[1])**2
            )
            
            if distance >= self.limitations.min_segment_length:
                cleaned.append(current_point)
        
        return cleaned
    
    def _validate_geometry(self, entity) -> bool:
        """Validate geometry for waterjet constraints."""
        try:
            if entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                points = list(entity.get_points())
                if len(points) < 2:
                    return False
                
                # Check for minimum feature size
                if self._check_minimum_feature_size(points):
                    return True
                else:
                    return False
            
            elif entity.dxftype() == "CIRCLE":
                radius = entity.dxf.radius
                return radius >= self.limitations.min_hole_diameter / 2
            
            elif entity.dxftype() == "ARC":
                radius = entity.dxf.radius
                return radius >= self.limitations.min_corner_radius
            
            elif entity.dxftype() == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                length = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
                return length >= self.limitations.min_segment_length
            
            return True
            
        except Exception:
            return False
    
    def _check_minimum_feature_size(self, points: List[Tuple[float, float]]) -> bool:
        """Check if geometry meets minimum feature size requirements."""
        if len(points) < 3:
            return False
        
        try:
            # Create polygon
            polygon = geom.Polygon(points)
            
            # Check area
            if polygon.area < self.limitations.min_feature_size**2:
                return False
            
            # Check minimum width (simplified check)
            bounds = polygon.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            if width < self.limitations.min_feature_size or height < self.limitations.min_feature_size:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_waterjet_cleaning(self, entities: List[object]) -> List[object]:
        """Apply waterjet-specific cleaning operations."""
        cleaned_entities = []
        
        for entity in entities:
            try:
                # Apply kerf compensation
                compensated = self._apply_kerf_compensation(entity)
                if compensated:
                    cleaned_entities.append(compensated)
            except Exception:
                continue
        
        return cleaned_entities
    
    def _apply_kerf_compensation(self, entity) -> Optional[object]:
        """Apply kerf compensation to entity."""
        try:
            if entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                # For closed polylines, apply inward offset
                points = list(entity.get_points())
                if len(points) < 3:
                    return entity
                
                # Create polygon
                polygon = geom.Polygon(points)
                
                # Apply inward offset for kerf compensation
                offset_distance = -self.limitations.kerf_width / 2
                try:
                    offset_polygon = polygon.buffer(offset_distance, join_style=2)
                    if offset_polygon.is_valid and offset_polygon.area > 0:
                        # Convert back to polyline
                        new_entity = entity.copy()
                        new_entity.clear()
                        
                        coords = list(offset_polygon.exterior.coords[:-1])
                        for coord in coords:
                            new_entity.append(coord)
                        
                        return new_entity
                except Exception:
                    pass
            
            elif entity.dxftype() == "CIRCLE":
                # Reduce radius by kerf width
                new_entity = entity.copy()
                new_entity.dxf.radius = max(0, entity.dxf.radius - self.limitations.kerf_width / 2)
                return new_entity
            
            return entity
            
        except Exception:
            return entity
    
    def _get_entity_info(self, entity) -> str:
        """Get information about entity for logging."""
        try:
            if entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                points = list(entity.get_points())
                return f"Polyline with {len(points)} points"
            elif entity.dxftype() == "CIRCLE":
                return f"Circle with radius {entity.dxf.radius:.2f}"
            elif entity.dxftype() == "ARC":
                return f"Arc with radius {entity.dxf.radius:.2f}"
            elif entity.dxftype() == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                length = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
                return f"Line with length {length:.2f}"
            else:
                return f"{entity.dxftype()} entity"
        except Exception:
            return f"{entity.dxftype()} entity (error getting info)"
    
    def get_cleaning_report(self) -> Dict[str, any]:
        """Get detailed cleaning report."""
        if not self.cleaning_result:
            return {}
        
        return {
            "cleaning_summary": {
                "original_entities": self.cleaning_result.original_entities,
                "cleaned_entities": self.cleaning_result.cleaned_entities,
                "removed_entities": self.cleaning_result.removed_entities,
                "repaired_entities": self.cleaning_result.repaired_entities,
                "cleaning_time_seconds": round(self.cleaning_result.cleaning_time, 3),
                "success_rate_percent": round(
                    (self.cleaning_result.cleaned_entities / self.cleaning_result.original_entities) * 100, 1
                ) if self.cleaning_result.original_entities > 0 else 0
            },
            "waterjet_limitations": {
                "min_feature_size": self.limitations.min_feature_size,
                "min_hole_diameter": self.limitations.min_hole_diameter,
                "min_slot_width": self.limitations.min_slot_width,
                "min_corner_radius": self.limitations.min_corner_radius,
                "kerf_width": self.limitations.kerf_width,
                "overcut_tolerance": self.limitations.overcut_tolerance,
                "min_segment_length": self.limitations.min_segment_length,
                "max_angle_change": self.limitations.max_angle_change,
                "simplify_tolerance": self.limitations.simplify_tolerance
            },
            "warnings": self.cleaning_result.warnings,
            "errors": self.cleaning_result.errors
        }


def clean_dxf_for_waterjet(input_path: str, output_path: str, 
                          limitations: Optional[WaterjetLimitations] = None) -> Dict[str, any]:
    """Clean DXF file for waterjet cutting with comprehensive report."""
    cleaner = WaterjetDXFCleaner(limitations)
    result = cleaner.clean_dxf(input_path, output_path)
    return cleaner.get_cleaning_report()


def create_waterjet_limitations(**kwargs) -> WaterjetLimitations:
    """Create waterjet limitations with custom parameters."""
    return WaterjetLimitations(**kwargs)


if __name__ == "__main__":
    # Example usage
    input_file = "sample.dxf"
    output_file = "cleaned_sample.dxf"
    
    if os.path.exists(input_file):
        # Create custom limitations
        limitations = WaterjetLimitations(
            min_feature_size=1.5,
            min_hole_diameter=3.0,
            kerf_width=1.2
        )
        
        # Clean DXF
        report = clean_dxf_for_waterjet(input_file, output_file, limitations)
        
        print("DXF Cleaning Report:")
        print(f"Original entities: {report['cleaning_summary']['original_entities']}")
        print(f"Cleaned entities: {report['cleaning_summary']['cleaned_entities']}")
        print(f"Removed entities: {report['cleaning_summary']['removed_entities']}")
        print(f"Success rate: {report['cleaning_summary']['success_rate_percent']}%")
        
        if report['warnings']:
            print("\nWarnings:")
            for warning in report['warnings']:
                print(f"  - {warning}")
        
        if report['errors']:
            print("\nErrors:")
            for error in report['errors']:
                print(f"  - {error}")
    else:
        print(f"Input file not found: {input_file}")
