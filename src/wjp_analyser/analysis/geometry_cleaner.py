from __future__ import annotations

from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon, Point
from shapely.validation import make_valid
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import sys
import os

# Add config to path for material profiles
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
if config_path not in sys.path:
    sys.path.append(config_path)

try:
    from material_profiles import get_material_profile, get_material_parameters
except ImportError:
    # Fallback if material profiles not available
    def get_material_profile(material_name):
        return {"min_spacing": 2.0, "min_radius": 2.0, "kerf_width": 1.0, "cutting_speed": 1200.0, "pierce_time": 0.5}
    
    def get_material_parameters(material_name):
        return (2.0, 2.0, 1.0, 1200.0, 0.5)

logger = logging.getLogger(__name__)

def merge_and_polygonize(lines: list[LineString]) -> tuple[list[LineString], list[Polygon]]:
    """
    Returns merged lines and polygonized closed polygons.
    """
    if not lines:
        return [], []
    merged = linemerge(unary_union(lines))
    # merged can be LineString or MultiLineString
    if hasattr(merged, "geoms"):
        merged_lines = list(merged.geoms)
    else:
        merged_lines = [merged]
    polys = list(polygonize(merged_lines))
    return merged_lines, polys

def clean_geometry(entities: List[Any], tolerance: float = 0.1, 
                   min_radius: float = 2.0, min_spacing: float = 3.0) -> Dict[str, Any]:
    """
    Clean DXF entities for waterjet cutting:
    - Close open contours within tolerance
    - Remove duplicate lines
    - Simplify curves
    - Enforce waterjet spacing/radius rules
    
    Args:
        entities: List of DXF entities (lines, polylines, etc.)
        tolerance: Distance tolerance for closing gaps (mm)
        min_radius: Minimum radius for features (mm)
        min_spacing: Minimum spacing between features (mm)
    
    Returns:
        Dict containing cleaned entities and validation results
    """
    logger.info(f"Starting geometry cleaning with tolerance={tolerance}mm, min_radius={min_radius}mm, min_spacing={min_spacing}mm")
    
    # Convert DXF entities to Shapely geometries
    lines = []
    polygons = []
    violations = []
    warnings = []
    
    for entity in entities:
        try:
            geom = _dxf_to_shapely(entity)
            if geom:
                if isinstance(geom, LineString):
                    lines.append(geom)
                elif isinstance(geom, Polygon):
                    polygons.append(geom)
        except Exception as e:
            logger.warning(f"Failed to convert entity: {e}")
            warnings.append(f"Failed to convert entity: {str(e)}")
    
    # Step 1: Close open contours
    closed_lines, open_lines = _close_open_contours(lines, tolerance)
    logger.info(f"Closed {len(closed_lines)} contours, {len(open_lines)} remain open")
    
    # Step 2: Remove duplicates and overlapping lines
    unique_lines = _remove_duplicates(closed_lines + open_lines)
    logger.info(f"Removed duplicates, {len(unique_lines)} unique lines remaining")
    
    # Step 3: Simplify curves
    simplified_lines = _simplify_curves(unique_lines)
    logger.info(f"Simplified curves, {len(simplified_lines)} lines after simplification")
    
    # Step 4: Validate waterjet rules
    validation_results = _validate_waterjet_rules(simplified_lines, polygons, min_radius, min_spacing)
    violations.extend(validation_results['violations'])
    warnings.extend(validation_results['warnings'])
    
    # Step 5: Merge and polygonize
    merged_lines, new_polygons = merge_and_polygonize(simplified_lines)
    all_polygons = polygons + new_polygons
    
    # Step 6: Classify contours (inner vs outer)
    classified_contours = _classify_contours(merged_lines, all_polygons)

    # If classification is empty or missing polygon-only contours, classify by polygon containment
    if not classified_contours and all_polygons:
        for i, poly_i in enumerate(all_polygons):
            is_inner = any(
                (j != i) and all_polygons[j].contains(poly_i)
                for j in range(len(all_polygons))
            )
            classified_contours.append({
                'geometry': poly_i,
                'is_inner': bool(is_inner),
                'length': poly_i.exterior.length,
                'bounds': poly_i.bounds
            })
    
    logger.info(f"Geometry cleaning complete: {len(classified_contours)} contours, {len(violations)} violations, {len(warnings)} warnings")
    
    # Ensure at least one contour for simple closed polygons (outer if none classified)
    if not classified_contours and all_polygons:
        for poly in all_polygons:
            classified_contours.append({
                'geometry': poly,
                'is_inner': False,
                'length': poly.exterior.length,
                'bounds': poly.bounds
            })

    return {
        'contours': classified_contours,
        'polygons': all_polygons,
        'violations': violations,
        'warnings': warnings,
        'stats': {
            'original_lines': len(lines),
            'closed_contours': len(closed_lines),
            'open_contours': len(open_lines),
            'final_contours': len(classified_contours),
            'violations_count': len(violations),
            'warnings_count': len(warnings)
        }
    }

def _dxf_to_shapely(entity) -> Any:
    """Convert DXF entity to Shapely geometry"""
    try:
        if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
            # Line entity
            start = entity.dxf.start
            end = entity.dxf.end
            return LineString([(float(start.x), float(start.y)), (float(end.x), float(end.y))])
        elif hasattr(entity, 'get_points'):
            # Polyline entity
            points = entity.get_points()
            if len(points) >= 2:
                coords = [(float(p[0]), float(p[1])) for p in points]
                if len(coords) >= 3 and coords[0] == coords[-1]:
                    return Polygon(coords)
                else:
                    return LineString(coords)
        elif hasattr(entity, 'control_points'):
            # SPLINE entity
            control_points = entity.control_points
            if len(control_points) >= 2:
                coords = [(float(p[0]), float(p[1])) for p in control_points]
                return LineString(coords)
        elif hasattr(entity, 'fit_points'):
            # SPLINE entity with fit points
            fit_points = entity.fit_points
            if len(fit_points) >= 2:
                coords = [(float(p[0]), float(p[1])) for p in fit_points]
                return LineString(coords)
    except Exception as e:
        logger.debug(f"Error converting entity: {e}")
    return None

def _close_open_contours(lines: List[LineString], tolerance: float) -> Tuple[List[LineString], List[LineString]]:
    """Close open contours within tolerance"""
    closed_lines = []
    open_lines = []
    
    for line in lines:
        if line.is_closed:
            closed_lines.append(line)
        else:
            # Try to close the line
            coords = list(line.coords)
            start = Point(coords[0])
            end = Point(coords[-1])
            
            if start.distance(end) <= tolerance:
                # Close the contour
                closed_coords = coords + [coords[0]]
                closed_lines.append(LineString(closed_coords))
            else:
                open_lines.append(line)
    
    return closed_lines, open_lines

def _remove_duplicates(lines: List[LineString]) -> List[LineString]:
    """Remove duplicate and overlapping lines"""
    unique_lines = []
    
    for line in lines:
        is_duplicate = False
        for existing in unique_lines:
            if line.equals(existing) or line.overlaps(existing):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_lines.append(line)
    
    return unique_lines

def _simplify_curves(lines: List[LineString], tolerance: float = 0.2) -> List[LineString]:
    """Simplify curves to reduce node count while preserving geometry"""
    simplified = []
    
    for line in lines:
        try:
            simplified_line = line.simplify(tolerance, preserve_topology=True)
            simplified.append(simplified_line)
        except Exception as e:
            logger.warning(f"Failed to simplify line: {e}")
            simplified.append(line)
    
    return simplified

def _validate_waterjet_rules(lines: List[LineString], polygons: List[Polygon], 
                           min_radius: float, min_spacing: float) -> Dict[str, List[str]]:
    """Validate waterjet manufacturing rules"""
    violations = []
    warnings = []
    
    # Check minimum radius for corners
    for i, line in enumerate(lines):
        coords = list(line.coords)
        for j in range(len(coords) - 2):
            p1, p2, p3 = coords[j], coords[j+1], coords[j+2]
            # Calculate angle and radius
            angle = _calculate_angle(p1, p2, p3)
            if abs(angle) < np.pi / 4:  # Sharp corner
                violations.append(f"Line {i}: Sharp corner at ({p2[0]:.1f}, {p2[1]:.1f}) - radius < {min_radius}mm")
    
    # Check minimum spacing between features
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:], i+1):
            distance = line1.distance(line2)
            if distance < min_spacing:
                violations.append(f"Lines {i} and {j}: Spacing {distance:.1f}mm < {min_spacing}mm")
    
    # Check polygon spacing
    for i, poly1 in enumerate(polygons):
        for j, poly2 in enumerate(polygons[i+1:], i+1):
            distance = poly1.distance(poly2)
            if distance < min_spacing:
                violations.append(f"Polygons {i} and {j}: Spacing {distance:.1f}mm < {min_spacing}mm")
    
    return {'violations': violations, 'warnings': warnings}

def _calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Calculate angle between three points"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def _classify_contours(lines: List[LineString], polygons: List[Polygon]) -> List[Dict[str, Any]]:
    """Classify contours as inner or outer"""
    classified = []
    
    for line in lines:
        # Create a temporary polygon to check containment
        try:
            temp_poly = Polygon(line.coords)
            is_inner = False
            
            # Check if this contour is inside any other polygon
            for poly in polygons:
                if temp_poly.within(poly) and not temp_poly.equals(poly):
                    is_inner = True
                    break
            
            classified.append({
                'geometry': line,
                'is_inner': is_inner,
                'is_closed': line.is_closed,
                'length': line.length,
                'bounds': line.bounds
            })
        except Exception as e:
            logger.warning(f"Failed to classify contour: {e}")
            classified.append({
                'geometry': line,
                'is_inner': False,
                'is_closed': line.is_closed,
                'length': line.length,
                'bounds': line.bounds
            })
    
    return classified


def clean_geometry_with_material(entities: List[Any], material_name: str = "steel", 
                                tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Clean DXF entities using material-specific parameters.
    
    Args:
        entities: List of DXF entities (lines, polylines, etc.)
        material_name: Name of the material (e.g., "steel", "granite", "aluminum")
        tolerance: Distance tolerance for closing gaps (mm)
    
    Returns:
        Dict containing cleaned entities and validation results
    """
    # Get material-specific parameters
    min_spacing, min_radius, kerf_width, cutting_speed, pierce_time = get_material_parameters(material_name)
    
    logger.info(f"Using material profile: {material_name}")
    logger.info(f"Parameters: min_spacing={min_spacing}mm, min_radius={min_radius}mm, kerf_width={kerf_width}mm")
    
    # Use the main cleaning function with material parameters
    result = clean_geometry(entities, tolerance, min_radius, min_spacing)
    
    # Add material information to the result
    result['material_profile'] = {
        'name': material_name,
        'min_spacing': min_spacing,
        'min_radius': min_radius,
        'kerf_width': kerf_width,
        'cutting_speed': cutting_speed,
        'pierce_time': pierce_time
    }
    
    return result
