"""
Geometry Hygiene for Production-Grade Nesting
==============================================

Robust polygonization, hole handling, winding rules, and tolerance unification
for reliable nesting operations.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid
import numpy as np


class GeometryHygiene:
    """Ensures geometry is clean and ready for nesting."""
    
    def __init__(
        self,
        tolerance_microns: float = 1.0,  # µm-scale tolerance
        min_area_mm2: float = 0.01,
        fix_winding: bool = True,
        handle_holes: bool = True,
    ):
        """
        Initialize geometry hygiene processor.
        
        Args:
            tolerance_microns: Tolerance in micrometers (default: 1.0 µm)
            min_area_mm2: Minimum area to keep (mm²)
            fix_winding: Fix winding order (CCW for exterior, CW for holes)
            handle_holes: Process and preserve holes
        """
        self.tolerance_mm = tolerance_microns / 1000.0  # Convert to mm
        self.min_area_mm2 = min_area_mm2
        self.fix_winding = fix_winding
        self.handle_holes = handle_holes
    
    def clean_polygon(self, polygon: Polygon) -> Optional[Polygon]:
        """
        Clean a single polygon: fix validity, winding, holes.
        
        Args:
            polygon: Input polygon (may be invalid)
            
        Returns:
            Cleaned polygon or None if too small
        """
        if polygon is None or polygon.is_empty:
            return None
        
        # Make valid (fix self-intersections, etc.)
        try:
            fixed = make_valid(polygon)
        except Exception:
            return None
        
        # Handle MultiPolygon (split if needed)
        if isinstance(fixed, MultiPolygon):
            # Take largest polygon
            polygons = list(fixed.geoms)
            if not polygons:
                return None
            fixed = max(polygons, key=lambda p: p.area)
        
        if not isinstance(fixed, Polygon):
            return None
        
        # Check minimum area
        if abs(fixed.area) < self.min_area_mm2:
            return None
        
        # Fix winding order
        if self.fix_winding:
            fixed = self._fix_winding(fixed)
        
        # Simplify with tolerance
        try:
            fixed = fixed.simplify(self.tolerance_mm, preserve_topology=True)
        except Exception:
            pass
        
        # Validate again after simplification
        if not fixed.is_valid:
            try:
                fixed = make_valid(fixed)
                if isinstance(fixed, MultiPolygon):
                    fixed = max(fixed.geoms, key=lambda p: p.area)
            except Exception:
                return None
        
        return fixed if isinstance(fixed, Polygon) else None
    
    def _fix_winding(self, polygon: Polygon) -> Polygon:
        """
        Fix winding order: CCW for exterior, CW for holes.
        
        Args:
            polygon: Input polygon
            
        Returns:
            Polygon with correct winding
        """
        # Fix exterior (should be CCW)
        exterior_coords = list(polygon.exterior.coords[:-1])
        if not self._is_ccw(exterior_coords):
            exterior_coords.reverse()
        
        # Fix holes (should be CW)
        holes = []
        for hole in polygon.interiors:
            hole_coords = list(hole.coords[:-1])
            if self._is_ccw(hole_coords):
                hole_coords.reverse()
            holes.append(hole_coords)
        
        # Reconstruct polygon
        try:
            return Polygon(exterior_coords, holes)
        except Exception:
            # Fallback: try with original
            return polygon
    
    def _is_ccw(self, coords: List[Tuple[float, float]]) -> bool:
        """
        Check if coordinates are counter-clockwise.
        
        Uses shoelace formula.
        """
        if len(coords) < 3:
            return True
        
        area = 0.0
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            area += (coords[j][0] - coords[i][0]) * (coords[j][1] + coords[i][1])
        
        return area < 0  # CCW if area is negative
    
    def process_holes(
        self,
        polygon: Polygon,
        min_hole_area_mm2: float = 1.0,
        merge_nearby_holes: bool = True,
        hole_merge_distance_mm: float = 0.1,
    ) -> Polygon:
        """
        Process and clean holes in a polygon.
        
        Args:
            polygon: Input polygon with holes
            min_hole_area_mm2: Minimum hole area to keep
            merge_nearby_holes: Merge holes that are very close
            hole_merge_distance_mm: Distance threshold for merging
            
        Returns:
            Polygon with processed holes
        """
        if not self.handle_holes or polygon.interiors is None or len(polygon.interiors) == 0:
            return polygon
        
        # Filter holes by minimum area
        valid_holes = []
        for hole in polygon.interiors:
            hole_poly = Polygon(hole.coords)
            if abs(hole_poly.area) >= min_hole_area_mm2:
                valid_holes.append(list(hole.coords[:-1]))
        
        # Merge nearby holes if enabled
        if merge_nearby_holes and len(valid_holes) > 1:
            valid_holes = self._merge_nearby_holes(valid_holes, hole_merge_distance_mm)
        
        # Reconstruct polygon with processed holes
        try:
            exterior_coords = list(polygon.exterior.coords[:-1])
            return Polygon(exterior_coords, valid_holes)
        except Exception:
            return polygon
    
    def _merge_nearby_holes(
        self,
        holes: List[List[Tuple[float, float]]],
        distance_threshold: float,
    ) -> List[List[Tuple[float, float]]]:
        """Merge holes that are within distance threshold."""
        if len(holes) < 2:
            return holes
        
        merged = []
        used = set()
        
        for i, hole1 in enumerate(holes):
            if i in used:
                continue
            
            poly1 = Polygon(hole1)
            merged_hole = hole1
            
            # Find nearby holes to merge
            for j, hole2 in enumerate(holes[i+1:], start=i+1):
                if j in used:
                    continue
                
                poly2 = Polygon(hole2)
                
                # Check distance
                if poly1.distance(poly2) < distance_threshold:
                    # Merge holes
                    try:
                        merged_poly = unary_union([poly1, poly2])
                        if isinstance(merged_poly, Polygon):
                            merged_hole = list(merged_poly.exterior.coords[:-1])
                            poly1 = merged_poly
                            used.add(j)
                    except Exception:
                        pass
            
            merged.append(merged_hole)
        
        return merged
    
    def unify_tolerance(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Unify tolerance across multiple polygons.
        
        Simplifies all polygons with the same tolerance to ensure
        consistent geometry quality.
        
        Args:
            polygons: List of polygons
            
        Returns:
            List of simplified polygons
        """
        unified = []
        for poly in polygons:
            if poly is None or poly.is_empty:
                continue
            
            try:
                # Simplify with unified tolerance
                simplified = poly.simplify(
                    self.tolerance_mm,
                    preserve_topology=True,
                )
                
                # Ensure valid
                if not simplified.is_valid:
                    simplified = make_valid(simplified)
                    if isinstance(simplified, MultiPolygon):
                        simplified = max(simplified.geoms, key=lambda p: p.area)
                
                if isinstance(simplified, Polygon) and not simplified.is_empty:
                    unified.append(simplified)
            except Exception:
                # Keep original if simplification fails
                if poly.is_valid:
                    unified.append(poly)
        
        return unified
    
    def clean_polygon_list(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Clean a list of polygons.
        
        Args:
            polygons: List of input polygons
            
        Returns:
            List of cleaned polygons
        """
        cleaned = []
        
        for poly in polygons:
            # Clean polygon
            clean_poly = self.clean_polygon(poly)
            if clean_poly is None:
                continue
            
            # Process holes if enabled
            if self.handle_holes:
                clean_poly = self.process_holes(clean_poly)
            
            if clean_poly is not None and not clean_poly.is_empty:
                cleaned.append(clean_poly)
        
        # Unify tolerance
        cleaned = self.unify_tolerance(cleaned)
        
        return cleaned
    
    def validate_for_nesting(self, polygon: Polygon) -> Tuple[bool, List[str]]:
        """
        Validate polygon is ready for nesting.
        
        Args:
            polygon: Polygon to validate
            
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        if polygon is None or polygon.is_empty:
            return False, ["Polygon is empty"]
        
        if not polygon.is_valid:
            warnings.append("Polygon is invalid")
            return False, warnings
        
        if abs(polygon.area) < self.min_area_mm2:
            warnings.append(f"Polygon area ({abs(polygon.area):.3f} mm²) below minimum")
            return False, warnings
        
        # Check for very thin features
        if polygon.length > 0:
            area_to_perimeter_ratio = abs(polygon.area) / polygon.length
            if area_to_perimeter_ratio < 0.01:  # Very thin
                warnings.append("Polygon has very thin features")
        
        # Check hole sizes
        for i, hole in enumerate(polygon.interiors):
            hole_poly = Polygon(hole.coords)
            if abs(hole_poly.area) < 0.1:
                warnings.append(f"Hole {i} is very small ({abs(hole_poly.area):.3f} mm²)")
        
        return True, warnings








