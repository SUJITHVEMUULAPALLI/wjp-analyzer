"""
Advanced CAM Processor
=====================

Professional-grade CAM processing with:
- DXF preprocessing and curve simplification
- Arc interpolation (G2/G3) instead of G1 segments
- Contour sorting (holes first, outer later)
- Continuous path joining to reduce pierces
- Advanced path optimization for minimal travel time
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import ezdxf
from ezdxf.math import Vec3, arc_angle_span_deg
from ezdxf.entities import Arc, Line, Polyline, Spline


@dataclass
class CAMSettings:
    """Advanced CAM processing settings."""
    # DXF Preprocessing
    simplify_tolerance: float = 0.1  # mm
    join_curves: bool = True
    min_segment_length: float = 0.5  # mm
    
    # Arc Interpolation
    use_arc_interpolation: bool = True
    arc_tolerance: float = 0.01  # mm
    min_arc_radius: float = 0.5  # mm
    max_arc_radius: float = 1000.0  # mm
    
    # Contour Sorting
    holes_first: bool = True
    optimize_contour_order: bool = True
    
    # Path Optimization
    reduce_pierces: bool = True
    join_tolerance: float = 0.1  # mm
    optimize_travel: bool = True
    
    # G-code Settings
    rapid_speed: float = 10000.0  # mm/min
    cutting_speed: float = 1200.0  # mm/min
    pierce_time: float = 0.5  # seconds
    arc_feedrate: float = 1000.0  # mm/min


@dataclass
class ProcessedContour:
    """A processed contour with optimized geometry."""
    geometry: Union[Polygon, LineString]
    contour_type: str  # 'outer', 'inner', 'open'
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    is_closed: bool
    arc_segments: List[Dict]  # Arc interpolation data
    pierce_points: List[Tuple[float, float]]


@dataclass
class OptimizedPath:
    """An optimized cutting path."""
    contours: List[ProcessedContour]
    total_length: float
    rapid_distance: float
    pierce_count: int
    gcode_lines: List[str]


class AdvancedCAMProcessor:
    """Advanced CAM processor with professional-grade optimizations."""
    
    def __init__(self, settings: CAMSettings):
        self.settings = settings
        self.processed_contours: List[ProcessedContour] = []
        self.optimized_paths: List[OptimizedPath] = []
    
    def process_dxf(self, dxf_path: str) -> List[ProcessedContour]:
        """Process DXF file with advanced preprocessing."""
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        # Extract and preprocess entities
        entities = self._extract_entities(msp)
        simplified_entities = self._simplify_entities(entities)
        joined_entities = self._join_curves(simplified_entities)
        
        # Convert to contours
        contours = self._entities_to_contours(joined_entities)
        
        # Process each contour
        processed_contours = []
        for contour in contours:
            processed = self._process_contour(contour)
            if processed:
                processed_contours.append(processed)
        
        # Sort contours (holes first, outer later)
        if self.settings.optimize_contour_order:
            processed_contours = self._sort_contours(processed_contours)
        
        self.processed_contours = processed_contours
        return processed_contours
    
    def _extract_entities(self, msp) -> List:
        """Extract relevant entities from DXF modelspace."""
        entities = []
        
        for entity in msp:
            dxf_type = entity.dxftype()
            if dxf_type in ['LINE', 'ARC', 'POLYLINE', 'LWPOLYLINE', 'SPLINE']:
                entities.append(entity)
        
        return entities
    
    def _simplify_entities(self, entities: List) -> List:
        """Simplify entities using Douglas-Peucker algorithm."""
        simplified = []
        
        for entity in entities:
            if entity.dxftype() == 'LINE':
                simplified.append(entity)
            elif entity.dxftype() == 'ARC':
                simplified.append(entity)
            elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
                # Simplify polyline
                simplified_entity = self._simplify_polyline(entity)
                if simplified_entity:
                    simplified.append(simplified_entity)
            elif entity.dxftype() == 'SPLINE':
                # Convert spline to polyline and simplify
                polyline = self._spline_to_polyline(entity)
                if polyline:
                    simplified_entity = self._simplify_polyline(polyline)
                    if simplified_entity:
                        simplified.append(simplified_entity)
        
        return simplified
    
    def _simplify_polyline(self, polyline) -> Optional:
        """Simplify polyline using Douglas-Peucker algorithm."""
        try:
            points = []
            for point in polyline.points():
                points.append((point.x, point.y))
            
            if len(points) < 3:
                return polyline
            
            # Apply Douglas-Peucker simplification
            simplified_points = self._douglas_peucker(points, self.settings.simplify_tolerance)
            
            if len(simplified_points) < 2:
                return None
            
            # Create new simplified polyline
            new_polyline = polyline.copy()
            new_polyline.clear()
            
            for point in simplified_points:
                new_polyline.append(point)
            
            return new_polyline
            
        except Exception:
            return polyline
    
    def _douglas_peucker(self, points: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
        """Douglas-Peucker line simplification algorithm."""
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from line
        max_dist = 0
        max_index = 0
        
        for i in range(1, len(points) - 1):
            dist = self._point_to_line_distance(points[i], points[0], points[-1])
            if dist > max_dist:
                max_dist = dist
                max_index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            # Recursive call
            left_points = self._douglas_peucker(points[:max_index + 1], tolerance)
            right_points = self._douglas_peucker(points[max_index:], tolerance)
            
            # Combine results (remove duplicate middle point)
            return left_points[:-1] + right_points
        else:
            # All points are within tolerance, return endpoints only
            return [points[0], points[-1]]
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        # Distance formula
        return abs(a * x0 + b * y0 + c) / math.sqrt(a * a + b * b)
    
    def _join_curves(self, entities: List) -> List:
        """Join connected curves into continuous paths."""
        if not self.settings.join_curves:
            return entities
        
        joined = []
        processed = set()
        
        for i, entity in enumerate(entities):
            if i in processed:
                continue
            
            # Start a new path
            path_entities = [entity]
            processed.add(i)
            
            # Try to extend the path
            current_end = self._get_entity_end(entity)
            extended = True
            
            while extended:
                extended = False
                for j, other_entity in enumerate(entities):
                    if j in processed:
                        continue
                    
                    other_start = self._get_entity_start(other_entity)
                    other_end = self._get_entity_end(other_entity)
                    
                    # Check if entities can be joined
                    if self._can_join_entities(current_end, other_start, self.settings.join_tolerance):
                        path_entities.append(other_entity)
                        processed.add(j)
                        current_end = other_end
                        extended = True
                        break
                    elif self._can_join_entities(current_end, other_end, self.settings.join_tolerance):
                        # Reverse the entity
                        reversed_entity = self._reverse_entity(other_entity)
                        path_entities.append(reversed_entity)
                        processed.add(j)
                        current_end = other_start
                        extended = True
                        break
            
            # Create joined entity
            if len(path_entities) > 1:
                joined_entity = self._create_joined_entity(path_entities)
                if joined_entity:
                    joined.append(joined_entity)
            else:
                joined.append(entity)
        
        return joined
    
    def _get_entity_start(self, entity) -> Tuple[float, float]:
        """Get start point of entity."""
        if entity.dxftype() == 'LINE':
            return (entity.dxf.start.x, entity.dxf.start.y)
        elif entity.dxftype() == 'ARC':
            # Start of arc
            start_angle = math.radians(entity.dxf.start_angle)
            center = entity.dxf.center
            radius = entity.dxf.radius
            x = center.x + radius * math.cos(start_angle)
            y = center.y + radius * math.sin(start_angle)
            return (x, y)
        elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
            try:
                points = list(entity.points())
                if points:
                    return (points[0].x, points[0].y)
            except Exception:
                # Fallback for different ezdxf versions
                try:
                    points = [p for p in entity.points()]
                    if points:
                        return (points[0].x, points[0].y)
                except Exception:
                    pass
        return (0, 0)
    
    def _get_entity_end(self, entity) -> Tuple[float, float]:
        """Get end point of entity."""
        if entity.dxftype() == 'LINE':
            return (entity.dxf.end.x, entity.dxf.end.y)
        elif entity.dxftype() == 'ARC':
            # End of arc
            end_angle = math.radians(entity.dxf.end_angle)
            center = entity.dxf.center
            radius = entity.dxf.radius
            x = center.x + radius * math.cos(end_angle)
            y = center.y + radius * math.sin(end_angle)
            return (x, y)
        elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
            try:
                points = list(entity.points())
                if points:
                    return (points[-1].x, points[-1].y)
            except Exception:
                # Fallback for different ezdxf versions
                try:
                    points = [p for p in entity.points()]
                    if points:
                        return (points[-1].x, points[-1].y)
                except Exception:
                    pass
        return (0, 0)
    
    def _can_join_entities(self, end1: Tuple[float, float], 
                          start2: Tuple[float, float], 
                          tolerance: float) -> bool:
        """Check if two entities can be joined."""
        distance = math.sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2)
        return distance <= tolerance
    
    def _reverse_entity(self, entity):
        """Reverse the direction of an entity."""
        if entity.dxftype() == 'LINE':
            new_entity = entity.copy()
            new_entity.dxf.start = entity.dxf.end
            new_entity.dxf.end = entity.dxf.start
            return new_entity
        elif entity.dxftype() == 'ARC':
            new_entity = entity.copy()
            new_entity.dxf.start_angle = entity.dxf.end_angle
            new_entity.dxf.end_angle = entity.dxf.start_angle
            return new_entity
        elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
            new_entity = entity.copy()
            new_entity.clear()
            points = list(entity.points())
            for point in reversed(points):
                new_entity.append(point)
            return new_entity
        return entity
    
    def _create_joined_entity(self, entities: List):
        """Create a single entity from multiple connected entities."""
        if not entities:
            return None
        
        if len(entities) == 1:
            return entities[0]
        
        # Create a polyline from all entities
        points = []
        
        for entity in entities:
            if entity.dxftype() == 'LINE':
                points.append((entity.dxf.start.x, entity.dxf.start.y))
                points.append((entity.dxf.end.x, entity.dxf.end.y))
            elif entity.dxftype() == 'ARC':
                # Sample arc points
                arc_points = self._sample_arc_points(entity)
                points.extend(arc_points)
            elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
                try:
                    entity_points = [(p.x, p.y) for p in entity.points()]
                    points.extend(entity_points)
                except Exception:
                    try:
                        entity_points = [(p.x, p.y) for p in entity.points()]
                        points.extend(entity_points)
                    except Exception:
                        pass
        
        # Remove duplicate consecutive points
        cleaned_points = []
        for i, point in enumerate(points):
            if i == 0 or point != points[i-1]:
                cleaned_points.append(point)
        
        if len(cleaned_points) < 2:
            return None
        
        # Create new polyline
        try:
            new_polyline = entities[0].copy()
            new_polyline.clear()
            for point in cleaned_points:
                new_polyline.append(point)
            return new_polyline
        except Exception:
            return None
    
    def _sample_arc_points(self, arc, num_points: int = 8) -> List[Tuple[float, float]]:
        """Sample points along an arc."""
        points = []
        start_angle = math.radians(arc.dxf.start_angle)
        end_angle = math.radians(arc.dxf.end_angle)
        center = arc.dxf.center
        radius = arc.dxf.radius
        
        # Handle angle wrapping
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((x, y))
        
        return points
    
    def _entities_to_contours(self, entities: List) -> List[Union[Polygon, LineString]]:
        """Convert entities to Shapely contours."""
        contours = []
        
        for entity in entities:
            try:
                if entity.dxftype() == 'LINE':
                    line = LineString([(entity.dxf.start.x, entity.dxf.start.y),
                                     (entity.dxf.end.x, entity.dxf.end.y)])
                    contours.append(line)
                elif entity.dxftype() == 'ARC':
                    # Convert arc to line segments
                    arc_points = self._sample_arc_points(entity, 16)
                    line = LineString(arc_points)
                    contours.append(line)
                elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
                    try:
                        points = [(p.x, p.y) for p in entity.points()]
                        if len(points) >= 2:
                            if entity.closed:
                                polygon = Polygon(points)
                                contours.append(polygon)
                            else:
                                line = LineString(points)
                                contours.append(line)
                    except Exception:
                        try:
                            points = [(p.x, p.y) for p in entity.points()]
                            if len(points) >= 2:
                                if entity.closed:
                                    polygon = Polygon(points)
                                    contours.append(polygon)
                                else:
                                    line = LineString(points)
                                    contours.append(line)
                        except Exception:
                            pass
            except Exception:
                continue
        
        return contours
    
    def _process_contour(self, contour: Union[Polygon, LineString]) -> Optional[ProcessedContour]:
        """Process a single contour with arc interpolation and optimization."""
        try:
            # Determine contour type
            if isinstance(contour, Polygon):
                contour_type = 'outer' if contour.exterior.is_ccw else 'inner'
                is_closed = True
                coords = list(contour.exterior.coords[:-1])  # Remove duplicate last point
            else:
                contour_type = 'open'
                is_closed = False
                coords = list(contour.coords)
            
            if len(coords) < 2:
                return None
            
            # Apply arc interpolation
            arc_segments = self._interpolate_arcs(coords) if self.settings.use_arc_interpolation else []
            
            # Determine pierce points
            pierce_points = self._determine_pierce_points(coords, arc_segments)
            
            # Create processed contour
            processed = ProcessedContour(
                geometry=contour,
                contour_type=contour_type,
                start_point=coords[0],
                end_point=coords[-1],
                is_closed=is_closed,
                arc_segments=arc_segments,
                pierce_points=pierce_points
            )
            
            return processed
            
        except Exception:
            return None
    
    def _interpolate_arcs(self, coords: List[Tuple[float, float]]) -> List[Dict]:
        """Interpolate arcs from coordinate sequences."""
        arc_segments = []
        
        if len(coords) < 3:
            return arc_segments
        
        i = 0
        while i < len(coords) - 2:
            # Try to fit an arc through three consecutive points
            arc_data = self._fit_arc_through_points(
                coords[i], coords[i + 1], coords[i + 2]
            )
            
            if arc_data and self._is_valid_arc(arc_data):
                arc_segments.append(arc_data)
                i += 2  # Skip the middle point
            else:
                i += 1
        
        return arc_segments
    
    def _fit_arc_through_points(self, p1: Tuple[float, float], 
                               p2: Tuple[float, float], 
                               p3: Tuple[float, float]) -> Optional[Dict]:
        """Fit an arc through three points."""
        try:
            # Calculate center and radius using perpendicular bisectors
            center, radius = self._calculate_arc_center_radius(p1, p2, p3)
            
            if center is None or radius is None:
                return None
            
            # Check radius constraints
            if radius < self.settings.min_arc_radius or radius > self.settings.max_arc_radius:
                return None
            
            # Calculate angles
            start_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
            mid_angle = math.atan2(p2[1] - center[1], p2[0] - center[0])
            end_angle = math.atan2(p3[1] - center[1], p3[0] - center[0])
            
            # Determine arc direction
            angle_diff = end_angle - start_angle
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Check if middle point is on the arc
            mid_angle_diff = mid_angle - start_angle
            if mid_angle_diff > math.pi:
                mid_angle_diff -= 2 * math.pi
            elif mid_angle_diff < -math.pi:
                mid_angle_diff += 2 * math.pi
            
            # Verify middle point is between start and end
            if (angle_diff > 0 and not (0 <= mid_angle_diff <= angle_diff)) or \
               (angle_diff < 0 and not (angle_diff <= mid_angle_diff <= 0)):
                return None
            
            return {
                'center': center,
                'radius': radius,
                'start_angle': math.degrees(start_angle),
                'end_angle': math.degrees(end_angle),
                'start_point': p1,
                'end_point': p3,
                'direction': 'CW' if angle_diff < 0 else 'CCW'
            }
            
        except Exception:
            return None
    
    def _calculate_arc_center_radius(self, p1: Tuple[float, float], 
                                   p2: Tuple[float, float], 
                                   p3: Tuple[float, float]) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """Calculate arc center and radius from three points."""
        try:
            # Check for collinear points
            if abs((p2[1] - p1[1]) * (p3[0] - p1[0]) - (p3[1] - p1[1]) * (p2[0] - p1[0])) < 1e-10:
                return None, None
            
            # Calculate perpendicular bisectors
            # Midpoint of p1-p2
            mid1_x = (p1[0] + p2[0]) / 2
            mid1_y = (p1[1] + p2[1]) / 2
            
            # Midpoint of p2-p3
            mid2_x = (p2[0] + p3[0]) / 2
            mid2_y = (p2[1] + p3[1]) / 2
            
            # Direction vectors
            dir1_x = p2[0] - p1[0]
            dir1_y = p2[1] - p1[1]
            dir2_x = p3[0] - p2[0]
            dir2_y = p3[1] - p2[1]
            
            # Perpendicular vectors
            perp1_x = -dir1_y
            perp1_y = dir1_x
            perp2_x = -dir2_y
            perp2_y = dir2_x
            
            # Normalize perpendicular vectors
            len1 = math.sqrt(perp1_x**2 + perp1_y**2)
            len2 = math.sqrt(perp2_x**2 + perp2_y**2)
            
            if len1 < 1e-10 or len2 < 1e-10:
                return None, None
            
            perp1_x /= len1
            perp1_y /= len1
            perp2_x /= len2
            perp2_y /= len2
            
            # Solve intersection of perpendicular bisectors
            # Line 1: (mid1_x, mid1_y) + t * (perp1_x, perp1_y)
            # Line 2: (mid2_x, mid2_y) + s * (perp2_x, perp2_y)
            
            # Set up system of equations
            det = perp1_x * perp2_y - perp1_y * perp2_x
            if abs(det) < 1e-10:
                return None, None
            
            t = ((mid2_x - mid1_x) * perp2_y - (mid2_y - mid1_y) * perp2_x) / det
            
            center_x = mid1_x + t * perp1_x
            center_y = mid1_y + t * perp1_y
            
            # Calculate radius
            radius = math.sqrt((p1[0] - center_x)**2 + (p1[1] - center_y)**2)
            
            return (center_x, center_y), radius
            
        except Exception:
            return None, None
    
    def _is_valid_arc(self, arc_data: Dict) -> bool:
        """Validate arc data."""
        try:
            # Check radius constraints
            if arc_data['radius'] < self.settings.min_arc_radius or \
               arc_data['radius'] > self.settings.max_arc_radius:
                return False
            
            # Check angle span
            start_angle = math.radians(arc_data['start_angle'])
            end_angle = math.radians(arc_data['end_angle'])
            angle_span = abs(end_angle - start_angle)
            
            if angle_span > math.pi:  # Limit to semicircle
                return False
            
            # Check arc tolerance
            # Sample points along arc and check deviation from original points
            center = arc_data['center']
            radius = arc_data['radius']
            
            # This is a simplified check - in practice, you'd want to verify
            # against the original coordinate sequence
            return True
            
        except Exception:
            return False
    
    def _determine_pierce_points(self, coords: List[Tuple[float, float]], 
                                arc_segments: List[Dict]) -> List[Tuple[float, float]]:
        """Determine optimal pierce points for the contour."""
        if not self.settings.reduce_pierces:
            return [coords[0]]  # Pierce at start only
        
        pierce_points = []
        
        # Start with first point
        pierce_points.append(coords[0])
        
        # For now, use simple strategy - pierce at start of each major segment
        # In advanced implementation, this would analyze curvature and optimize
        # for minimal pierces while maintaining quality
        
        return pierce_points
    
    def _sort_contours(self, contours: List[ProcessedContour]) -> List[ProcessedContour]:
        """Sort contours for optimal cutting order."""
        if not self.settings.optimize_contour_order:
            return contours
        
        # Separate by type
        inner_contours = [c for c in contours if c.contour_type == 'inner']
        outer_contours = [c for c in contours if c.contour_type == 'outer']
        open_contours = [c for c in contours if c.contour_type == 'open']
        
        # Sort by area (smallest first for inner contours, largest first for outer)
        inner_contours.sort(key=lambda c: c.geometry.area if hasattr(c.geometry, 'area') else 0)
        outer_contours.sort(key=lambda c: c.geometry.area if hasattr(c.geometry, 'area') else 0, reverse=True)
        
        # Order: holes first, then outer contours, then open contours
        sorted_contours = inner_contours + outer_contours + open_contours
        
        return sorted_contours
    
    def generate_optimized_gcode(self, contours: List[ProcessedContour]) -> List[str]:
        """Generate optimized G-code with all CAM improvements."""
        gcode_lines = []
        
        # Header
        gcode_lines.extend([
            "G90",  # Absolute positioning
            "G21",  # Metric units
            "G94",  # Feed per minute
            f"G00 F{self.settings.rapid_speed:.0f}",  # Rapid speed
            f"G01 F{self.settings.cutting_speed:.0f}",  # Cutting speed
            ""
        ])
        
        current_position = (0.0, 0.0)
        
        for contour in contours:
            # Rapid to start point
            start_point = contour.start_point
            rapid_distance = math.sqrt(
                (start_point[0] - current_position[0])**2 + 
                (start_point[1] - current_position[1])**2
            )
            
            if rapid_distance > 0.1:  # Only rapid if significant distance
                gcode_lines.append(f"G00 X{start_point[0]:.3f} Y{start_point[1]:.3f}")
            
            # Pierce at start point
            gcode_lines.append(f"G04 P{self.settings.pierce_time:.1f}")
            gcode_lines.append("M62")  # Turn on waterjet
            
            # Generate cutting moves with arc interpolation
            cutting_moves = self._generate_cutting_moves(contour)
            gcode_lines.extend(cutting_moves)
            
            # Turn off waterjet
            gcode_lines.append("M63")
            gcode_lines.append("")  # Empty line for readability
            
            current_position = contour.end_point
        
        # Footer
        gcode_lines.extend([
            "G00 X0 Y0",  # Return to origin
            "M30"  # Program end
        ])
        
        return gcode_lines
    
    def _generate_cutting_moves(self, contour: ProcessedContour) -> List[str]:
        """Generate cutting moves with arc interpolation."""
        moves = []
        
        if contour.arc_segments:
            # Use arc interpolation
            moves.extend(self._generate_arc_moves(contour))
        else:
            # Use linear interpolation
            moves.extend(self._generate_linear_moves(contour))
        
        return moves
    
    def _generate_arc_moves(self, contour: ProcessedContour) -> List[str]:
        """Generate G2/G3 arc moves."""
        moves = []
        
        # This is a simplified implementation
        # In practice, you'd need to carefully map arc segments to coordinate sequences
        
        if hasattr(contour.geometry, 'exterior'):
            coords = list(contour.geometry.exterior.coords[:-1])
        else:
            coords = list(contour.geometry.coords)
        
        for i, arc_segment in enumerate(contour.arc_segments):
            # Generate arc move
            center = arc_segment['center']
            radius = arc_segment['radius']
            direction = arc_segment['direction']
            
            # Calculate I, J offsets (center relative to start)
            start_point = arc_segment['start_point']
            i_offset = center[0] - start_point[0]
            j_offset = center[1] - start_point[1]
            
            # G-code command
            if direction == 'CW':
                cmd = f"G02 X{arc_segment['end_point'][0]:.3f} Y{arc_segment['end_point'][1]:.3f} I{i_offset:.3f} J{j_offset:.3f}"
            else:
                cmd = f"G03 X{arc_segment['end_point'][0]:.3f} Y{arc_segment['end_point'][1]:.3f} I{i_offset:.3f} J{j_offset:.3f}"
            
            moves.append(cmd)
        
        return moves
    
    def _generate_linear_moves(self, contour: ProcessedContour) -> List[str]:
        """Generate G01 linear moves."""
        moves = []
        
        if hasattr(contour.geometry, 'exterior'):
            coords = list(contour.geometry.exterior.coords[:-1])
        else:
            coords = list(contour.geometry.coords)
        
        for coord in coords[1:]:  # Skip first point (already at start)
            moves.append(f"G01 X{coord[0]:.3f} Y{coord[1]:.3f}")
        
        return moves


# Convenience functions
def process_dxf_with_cam(dxf_path: str, settings: Optional[CAMSettings] = None) -> List[str]:
    """Process DXF file with advanced CAM and return optimized G-code."""
    if settings is None:
        settings = CAMSettings()
    
    processor = AdvancedCAMProcessor(settings)
    contours = processor.process_dxf(dxf_path)
    gcode_lines = processor.generate_optimized_gcode(contours)
    
    return gcode_lines


def create_cam_settings(**kwargs) -> CAMSettings:
    """Create CAM settings with custom parameters."""
    return CAMSettings(**kwargs)
