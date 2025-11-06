"""
Memory Optimization Utilities
==============================

Optimizations for large polygon sets and geometry operations.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings


def optimize_coordinates(
    points: List[Tuple[float, float]],
    precision: int = 3,
    use_float32: bool = False,
) -> List[Tuple[float, float]]:
    """
    Optimize coordinate precision to reduce memory usage.
    
    Args:
        points: List of (x, y) tuples
        precision: Decimal precision (number of decimal places)
        use_float32: Use float32 instead of float64
        
    Returns:
        Optimized points
    """
    if not points:
        return points
    
    # Round to specified precision
    multiplier = 10 ** precision
    
    optimized = []
    for x, y in points:
        if use_float32:
            x = np.float32(round(x * multiplier) / multiplier)
            y = np.float32(round(y * multiplier) / multiplier)
        else:
            x = round(x * multiplier) / multiplier
            y = round(y * multiplier) / multiplier
        optimized.append((x, y))
    
    return optimized


def filter_tiny_segments(
    points: List[Tuple[float, float]],
    epsilon: float = 0.01,
) -> List[Tuple[float, float]]:
    """
    Filter out edges smaller than epsilon to reduce point count.
    
    Args:
        points: List of (x, y) tuples
        epsilon: Minimum segment length (mm)
        
    Returns:
        Filtered points
    """
    if len(points) < 2:
        return points
    
    filtered = [points[0]]
    
    for i in range(1, len(points)):
        prev = filtered[-1]
        curr = points[i]
        
        # Calculate distance
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dist = (dx * dx + dy * dy) ** 0.5
        
        # Keep if distance >= epsilon
        if dist >= epsilon:
            filtered.append(curr)
    
    # Ensure closed polygons remain closed
    if len(filtered) > 2 and points[0] == points[-1]:
        if filtered[0] != filtered[-1]:
            filtered.append(filtered[0])
    
    return filtered


def paginate_geometry(
    polygons: List,
    page_size: int = 100,
    page: int = 0,
) -> Tuple[List, int]:
    """
    Paginate geometry for large datasets.
    
    Args:
        polygons: List of polygons
        page_size: Number of items per page
        page: Page number (0-indexed)
        
    Returns:
        (paginated_polygons, total_pages)
    """
    total = len(polygons)
    total_pages = (total + page_size - 1) // page_size
    
    start = page * page_size
    end = min(start + page_size, total)
    
    return polygons[start:end], total_pages


def use_strtree_for_queries() -> bool:
    """
    Check if STRtree is available and recommended.
    
    Returns:
        True if STRtree should be used
    """
    try:
        from shapely.strtree import STRtree
        from shapely import __version__
        
        # Shapely 2.0+ has improved STRtree
        major_version = int(__version__.split('.')[0])
        return major_version >= 2
    except (ImportError, ValueError, AttributeError):
        return False


def create_spatial_index(polygons: List) -> Optional[Any]:
    """
    Create STRtree spatial index for efficient spatial queries.
    
    Args:
        polygons: List of Shapely polygons
        
    Returns:
        STRtree index or None
    """
    try:
        from shapely.strtree import STRtree
        
        if use_strtree_for_queries():
            return STRtree(polygons)
    except ImportError:
        pass
    
    return None


def optimize_polygon_set(
    polygons: List,
    coordinate_precision: int = 3,
    min_segment_length: float = 0.01,
    use_float32: bool = False,
) -> List:
    """
    Optimize a set of polygons for memory usage.
    
    Args:
        polygons: List of Shapely polygons or point lists
        coordinate_precision: Decimal precision for coordinates
        min_segment_length: Minimum segment length (mm)
        use_float32: Use float32 coordinates
        
    Returns:
        Optimized polygons
    """
    optimized = []
    
    for poly in polygons:
        try:
            # Extract points if polygon object
            if hasattr(poly, 'exterior'):
                points = list(poly.exterior.coords[:-1])  # Exclude duplicate closing point
            elif isinstance(poly, (list, tuple)):
                points = list(poly)
            else:
                optimized.append(poly)
                continue
            
            # Optimize coordinates
            points = optimize_coordinates(points, coordinate_precision, use_float32)
            
            # Filter tiny segments
            points = filter_tiny_segments(points, min_segment_length)
            
            # Reconstruct polygon
            if hasattr(poly, 'exterior'):
                from shapely.geometry import Polygon
                if len(points) >= 3:
                    opt_poly = Polygon(points)
                    if opt_poly.is_valid:
                        optimized.append(opt_poly)
            else:
                optimized.append(points)
                
        except Exception:
            # Keep original if optimization fails
            optimized.append(poly)
    
    return optimized


def estimate_memory_usage(polygons: List) -> Dict[str, float]:
    """
    Estimate memory usage of polygon set.
    
    Args:
        polygons: List of polygons
        
    Returns:
        Dictionary with memory estimates (MB)
    """
    import sys
    
    total_points = 0
    total_size = 0
    
    for poly in polygons:
        try:
            if hasattr(poly, 'exterior'):
                points = list(poly.exterior.coords)
            elif isinstance(poly, (list, tuple)):
                points = list(poly)
            else:
                continue
            
            total_points += len(points)
            # Estimate: 8 bytes per float, 2 floats per point
            total_size += len(points) * 2 * 8
        except Exception:
            pass
    
    return {
        'total_points': total_points,
        'estimated_size_bytes': total_size,
        'estimated_size_mb': total_size / (1024 * 1024),
    }





