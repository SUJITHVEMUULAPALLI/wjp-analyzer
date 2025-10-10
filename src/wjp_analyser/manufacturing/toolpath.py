from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from typing import List, Dict, Tuple, Optional
import math
import numpy as np
from dataclasses import dataclass
from .cam_processor import AdvancedCAMProcessor, CAMSettings, process_dxf_with_cam


@dataclass
class CuttingPath:
    """Represents a single cutting path with optimization data."""
    polygon: Polygon
    polygon_index: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    cutting_direction: str  # 'clockwise' or 'counterclockwise'
    entry_angle: float
    exit_angle: float
    rapid_distance: float  # Distance to next path
    cutting_length: float


@dataclass
class ToolpathOptimization:
    """Toolpath optimization parameters."""
    kerf_compensation: float = 1.1
    rapid_speed: float = 10000.0  # mm/min
    cutting_speed: float = 1200.0  # mm/min
    pierce_time: float = 0.5  # seconds
    min_rapid_distance: float = 5.0  # mm
    optimize_rapids: bool = True
    optimize_direction: bool = True
    entry_strategy: str = "tangent"  # 'tangent', 'perpendicular', 'angle'
    
    # Advanced CAM settings
    use_advanced_cam: bool = True
    simplify_tolerance: float = 0.1  # mm
    join_curves: bool = True
    use_arc_interpolation: bool = True
    arc_tolerance: float = 0.01  # mm
    min_arc_radius: float = 0.5  # mm
    max_arc_radius: float = 1000.0  # mm
    holes_first: bool = True
    reduce_pierces: bool = True
    join_tolerance: float = 0.1  # mm


class AdvancedToolpathPlanner:
    """Advanced toolpath planning with cutting path optimization."""
    
    def __init__(self, optimization: ToolpathOptimization):
        self.optimization = optimization
        self.cutting_paths: List[CuttingPath] = []
        self.rapid_moves: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    
    def plan_cutting_paths(self, polys: List[Polygon], classes: Dict[int, str]) -> List[CuttingPath]:
        """Plan optimized cutting paths for all polygons."""
        self.cutting_paths = []
        
        # Group polygons by type for optimal cutting order
        inner_polys = [(i, p) for i, p in enumerate(polys) if classes.get(i) == "inner"]
        outer_polys = [(i, p) for i, p in enumerate(polys) if classes.get(i) == "outer"]
        inlay_polys = [(i, p) for i, p in enumerate(polys) if classes.get(i) == "inlay"]
        
        # Process in optimal order: inners first, then outers, then inlays
        all_polys = inner_polys + outer_polys + inlay_polys
        
        current_position = (0.0, 0.0)  # Start at origin
        
        for poly_idx, polygon in all_polys:
            path = self._create_cutting_path(polygon, poly_idx, current_position)
            self.cutting_paths.append(path)
            current_position = path.end_point
        
        return self.cutting_paths
    
    def _create_cutting_path(self, polygon: Polygon, poly_idx: int, 
                           current_position: Tuple[float, float]) -> CuttingPath:
        """Create an optimized cutting path for a single polygon."""
        
        # Get polygon boundary coordinates
        if hasattr(polygon, 'exterior'):
            coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point
        else:
            coords = []
        
        if not coords:
            # Fallback for empty polygon
            return CuttingPath(
                polygon=polygon,
                polygon_index=poly_idx,
                start_point=current_position,
                end_point=current_position,
                cutting_direction='clockwise',
                entry_angle=0.0,
                exit_angle=0.0,
                rapid_distance=0.0,
                cutting_length=0.0
            )
        
        # Find optimal entry point (closest to current position)
        entry_point = self._find_optimal_entry_point(coords, current_position)
        entry_idx = coords.index(entry_point)
        
        # Determine cutting direction for optimal finish
        cutting_direction = self._determine_cutting_direction(polygon, entry_point)
        
        # Calculate entry and exit angles
        entry_angle = self._calculate_entry_angle(coords, entry_idx, cutting_direction)
        exit_angle = self._calculate_exit_angle(coords, entry_idx, cutting_direction)
        
        # Calculate cutting length
        cutting_length = polygon.exterior.length if hasattr(polygon, 'exterior') else 0.0
        
        # Calculate rapid distance to next path
        rapid_distance = self._calculate_rapid_distance(current_position, entry_point)
        
        return CuttingPath(
            polygon=polygon,
            polygon_index=poly_idx,
            start_point=entry_point,
            end_point=entry_point,  # Closed path ends where it starts
            cutting_direction=cutting_direction,
            entry_angle=entry_angle,
            exit_angle=exit_angle,
            rapid_distance=rapid_distance,
            cutting_length=cutting_length
        )
    
    def _find_optimal_entry_point(self, coords: List[Tuple[float, float]], 
                                current_position: Tuple[float, float]) -> Tuple[float, float]:
        """Find the optimal entry point on the polygon boundary."""
        min_distance = float('inf')
        optimal_point = coords[0]
        
        for point in coords:
            distance = math.sqrt((point[0] - current_position[0])**2 + 
                               (point[1] - current_position[1])**2)
            if distance < min_distance:
                min_distance = distance
                optimal_point = point
        
        return optimal_point
    
    def _determine_cutting_direction(self, polygon: Polygon, 
                                   entry_point: Tuple[float, float]) -> str:
        """Determine optimal cutting direction (clockwise/counterclockwise)."""
        if not hasattr(polygon, 'exterior'):
            return 'clockwise'
        
        # Check if polygon is oriented counter-clockwise (standard for outer boundaries)
        coords = list(polygon.exterior.coords[:-1])
        
        # Calculate signed area to determine orientation
        area = 0.0
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            area += (coords[j][0] - coords[i][0]) * (coords[j][1] + coords[i][1])
        
        # Positive area = counter-clockwise, negative = clockwise
        return 'counterclockwise' if area > 0 else 'clockwise'
    
    def _calculate_entry_angle(self, coords: List[Tuple[float, float]], 
                            entry_idx: int, direction: str) -> float:
        """Calculate optimal entry angle for smooth cutting start."""
        if len(coords) < 2:
            return 0.0
        
        # Get the tangent at entry point
        prev_idx = (entry_idx - 1) % len(coords)
        next_idx = (entry_idx + 1) % len(coords)
        
        prev_point = coords[prev_idx]
        next_point = coords[next_idx]
        
        # Calculate angle of tangent
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        
        angle = math.atan2(dy, dx)
        
        # Adjust for cutting direction
        if direction == 'clockwise':
            angle += math.pi / 2  # Perpendicular entry
        else:
            angle -= math.pi / 2
        
        return math.degrees(angle)
    
    def _calculate_exit_angle(self, coords: List[Tuple[float, float]], 
                            entry_idx: int, direction: str) -> float:
        """Calculate optimal exit angle for smooth cutting finish."""
        # For closed paths, exit angle is same as entry angle
        return self._calculate_entry_angle(coords, entry_idx, direction)
    
    def _calculate_rapid_distance(self, start: Tuple[float, float], 
                                end: Tuple[float, float]) -> float:
        """Calculate rapid move distance between two points."""
        return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    def optimize_path_order(self) -> List[int]:
        """Optimize the order of cutting paths to minimize rapid moves."""
        if not self.cutting_paths:
            return []
        
        # Use nearest neighbor algorithm for path optimization
        remaining_paths = self.cutting_paths.copy()
        optimized_order = []
        current_position = (0.0, 0.0)
        
        while remaining_paths:
            # Find closest path to current position
            min_distance = float('inf')
            closest_path = None
            closest_idx = -1
            
            for i, path in enumerate(remaining_paths):
                distance = self._calculate_rapid_distance(current_position, path.start_point)
                if distance < min_distance:
                    min_distance = distance
                    closest_path = path
                    closest_idx = i
            
            if closest_path:
                optimized_order.append(closest_path.polygon_index)
                current_position = closest_path.end_point
                remaining_paths.pop(closest_idx)
        
        return optimized_order
    
    def generate_optimized_gcode(self, paths: List[CuttingPath]) -> List[str]:
        """Generate optimized G-code for cutting paths."""
        gcode_lines = []
        
        # Header
        gcode_lines.extend([
            "G90",  # Absolute positioning
            "G21",  # Metric units
            "G94",  # Feed per minute
            f"G00 F{self.optimization.rapid_speed:.0f}",  # Rapid speed
            f"G01 F{self.optimization.cutting_speed:.0f}",  # Cutting speed
            ""
        ])
        
        current_position = (0.0, 0.0)
        
        for path in paths:
            # Rapid to entry point
            rapid_distance = self._calculate_rapid_distance(current_position, path.start_point)
            if rapid_distance > self.optimization.min_rapid_distance:
                gcode_lines.append(f"G00 X{path.start_point[0]:.3f} Y{path.start_point[1]:.3f}")
            
            # Start cutting: jet ON then pierce dwell
            gcode_lines.append("M62")  # Turn on waterjet
            gcode_lines.append(f"G04 P{self.optimization.pierce_time:.1f}")
            
            # Cut the polygon boundary
            if hasattr(path.polygon, 'exterior'):
                coords = list(path.polygon.exterior.coords[:-1])
                
                # Find entry point in coordinates
                entry_idx = 0
                min_dist = float('inf')
                for i, coord in enumerate(coords):
                    dist = math.sqrt((coord[0] - path.start_point[0])**2 + 
                                   (coord[1] - path.start_point[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        entry_idx = i
                
                # Generate cutting moves
                if path.cutting_direction == 'clockwise':
                    # Cut in reverse order
                    cutting_coords = coords[entry_idx:] + coords[:entry_idx]
                    cutting_coords.reverse()
                else:
                    # Cut in forward order
                    cutting_coords = coords[entry_idx:] + coords[:entry_idx]
                
                for coord in cutting_coords:
                    gcode_lines.append(f"G01 X{coord[0]:.3f} Y{coord[1]:.3f}")
            
            # End cutting
            gcode_lines.append("M63")  # Turn off waterjet
            gcode_lines.append("")  # Empty line for readability
            
            current_position = path.end_point
        
        # Footer
        gcode_lines.extend([
            "G00 X0 Y0",  # Return to origin
            "M30"  # Program end
        ])
        
        return gcode_lines
    
    def generate_cam_optimized_gcode(self, dxf_path: str) -> List[str]:
        """Generate G-code using advanced CAM processor."""
        if not self.optimization.use_advanced_cam:
            return self.generate_optimized_gcode(self.cutting_paths)
        
        # Create CAM settings from toolpath optimization
        cam_settings = CAMSettings(
            simplify_tolerance=self.optimization.simplify_tolerance,
            join_curves=self.optimization.join_curves,
            use_arc_interpolation=self.optimization.use_arc_interpolation,
            arc_tolerance=self.optimization.arc_tolerance,
            min_arc_radius=self.optimization.min_arc_radius,
            max_arc_radius=self.optimization.max_arc_radius,
            holes_first=self.optimization.holes_first,
            reduce_pierces=self.optimization.reduce_pierces,
            join_tolerance=self.optimization.join_tolerance,
            rapid_speed=self.optimization.rapid_speed,
            cutting_speed=self.optimization.cutting_speed,
            pierce_time=self.optimization.pierce_time
        )
        
        # Process DXF with advanced CAM
        return process_dxf_with_cam(dxf_path, cam_settings)


def _closest_dist_to_point(poly: Polygon, x: float, y: float) -> Tuple[float, Tuple[float, float]]:
    """Return the minimal Euclidean distance from (x,y) to any exterior vertex and that vertex."""
    ext = list(poly.exterior.coords) if hasattr(poly, "exterior") else []
    best_d = float("inf")
    best_xy = (x, y)
    for px, py in ext:
        d = (px - x) * (px - x) + (py - y) * (py - y)
        if d < best_d:
            best_d = d
            best_xy = (px, py)
    return math.sqrt(best_d), best_xy


def plan_order(polys: list[Polygon], classes: dict[int, str]) -> list[int]:
    """
    Plan cut order prioritizing:
    1) Internal features before outers (to avoid part movement)
    2) Within each set, nearest-neighbor traversal to minimize rapids
    """
    internals = [(i, p) for i, p in enumerate(polys) if classes.get(i) != "outer"]
    outers = [(i, p) for i, p in enumerate(polys) if classes.get(i) == "outer"]

    def nn_order(items: list[tuple[int, Polygon]], start_xy: Tuple[float, float]) -> tuple[list[int], Tuple[float, float]]:
        remaining = items[:]
        order: list[int] = []
        curx, cury = start_xy
        while remaining:
            # pick nearest polygon to current head
            best_k = 0
            best_d = float("inf")
            for k, (_, poly) in enumerate(remaining):
                d, _ = _closest_dist_to_point(poly, curx, cury)
                if d < best_d:
                    best_d = d
                    best_k = k
            idx, chosen = remaining.pop(best_k)
            order.append(idx)
            # advance current to closest entry point of chosen
            _, (curx, cury) = _closest_dist_to_point(chosen, curx, cury)
        return order, (curx, cury)

    # Start from origin (0,0) which is typical for sheet setups
    order_i, head = nn_order(internals, (0.0, 0.0))
    order_o, _ = nn_order(outers, head)
    return order_i + order_o


def plan_advanced_toolpath(polys: list[Polygon], classes: dict[int, str], 
                          optimization: Optional[ToolpathOptimization] = None) -> List[CuttingPath]:
    """
    Plan advanced cutting paths with optimization.
    
    Args:
        polys: List of polygons to cut
        classes: Classification of each polygon (inner/outer/inlay)
        optimization: Optional optimization parameters
        
    Returns:
        List of optimized cutting paths
    """
    if optimization is None:
        optimization = ToolpathOptimization()
    
    planner = AdvancedToolpathPlanner(optimization)
    return planner.plan_cutting_paths(polys, classes)


def generate_optimized_gcode(polys: list[Polygon], classes: dict[int, str],
                            optimization: Optional[ToolpathOptimization] = None) -> List[str]:
    """
    Generate optimized G-code for cutting paths.
    
    Args:
        polys: List of polygons to cut
        classes: Classification of each polygon
        optimization: Optional optimization parameters
        
    Returns:
        List of G-code lines
    """
    if optimization is None:
        optimization = ToolpathOptimization()
    
    planner = AdvancedToolpathPlanner(optimization)
    paths = planner.plan_cutting_paths(polys, classes)
    return planner.generate_optimized_gcode(paths)

def kerf_preview(polys: list[Polygon], kerf_mm: float=1.1) -> list[Polygon]:
    # half-kerf offset for preview (not true tool compensation)
    half = kerf_mm/2.0
    out: list[Polygon] = []
    for p in polys:
        g = p.buffer(-half)
        if g.is_empty:
            continue
        # Flatten MultiPolygons and filter to Polygons only
        if hasattr(g, "geoms"):
            for sub in g.geoms:
                if hasattr(sub, "exterior"):
                    out.append(sub)  # type: ignore[arg-type]
        else:
            if hasattr(g, "exterior"):
                out.append(g)  # type: ignore[arg-type]
    return out
