#!/usr/bin/env python3
"""
Advanced DXF Cleaner with Multi-Level Options
Implements ChatGPT-5 recommendations for waterjet optimization.
"""

import ezdxf
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
import argparse
import sys
import os
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum

class CleaningLevel(Enum):
    STRICT = "strict"
    BALANCED = "balanced" 
    RELAXED = "relaxed"

class AdvancedDXFCleaner:
    """Advanced DXF cleaner with multiple cleaning levels and comprehensive reporting."""
    
    def __init__(self, cleaning_level: CleaningLevel = CleaningLevel.BALANCED):
        self.cleaning_level = cleaning_level
        self.report = {
            "cleaning_level": cleaning_level.value,
            "input_file": "",
            "output_file": "",
            "contours_processed": 0,
            "open_contours_fixed": 0,
            "spacing_violations_resolved": 0,
            "features_removed": 0,
            "features_retained": 0,
            "processing_time": 0,
            "violations_before": [],
            "violations_after": []
        }
        
        # Set parameters based on cleaning level
        self._set_parameters()
    
    def _set_parameters(self):
        """Set cleaning parameters based on selected level."""
        if self.cleaning_level == CleaningLevel.STRICT:
            self.min_length = 2.0
            self.min_area = 5.0
            self.endpoint_tolerance = 0.1
            self.min_spacing = 3.5
            self.simplify_tolerance = 0.05
        elif self.cleaning_level == CleaningLevel.BALANCED:
            self.min_length = 1.0
            self.min_area = 2.0
            self.endpoint_tolerance = 0.2
            self.min_spacing = 3.0
            self.simplify_tolerance = 0.1
        else:  # RELAXED
            self.min_length = 0.5
            self.min_area = 1.0
            self.endpoint_tolerance = 0.3
            self.min_spacing = 2.5
            self.simplify_tolerance = 0.2
    
    def load_dxf(self, filepath: str):
        """Load DXF file and extract geometries."""
        try:
            doc = ezdxf.readfile(filepath)
            msp = doc.modelspace()
            self.report["input_file"] = filepath
            return doc, msp
        except Exception as e:
            print(f"Error loading DXF: {e}")
            return None, None
    
    def extract_geometries(self, msp):
        """Extract and filter geometries from DXF."""
        geometries = []
        
        for entity in msp.query("LWPOLYLINE LINE POLYLINE CIRCLE ARC"):
            try:
                geom = self._process_entity(entity)
                if geom and self._meets_minimum_criteria(geom):
                    geometries.append(geom)
            except Exception as e:
                print(f"Skipped entity: {e}")
                continue
        
        self.report["contours_processed"] = len(geometries)
        return geometries
    
    def _process_entity(self, entity) -> Optional[LineString]:
        """Process individual DXF entity."""
        if entity.dxftype() == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            return LineString([(start[0], start[1]), (end[0], end[1])])
            
        elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            points = [(p[0], p[1]) for p in entity.get_points()]
            if len(points) >= 2:
                return LineString(points)
                
        elif entity.dxftype() == "CIRCLE":
            center = entity.dxf.center
            radius = entity.dxf.radius
            if radius >= self.min_length / (2 * np.pi):
                return self._circle_to_linestring(center, radius)
                
        elif entity.dxftype() == "ARC":
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            return self._arc_to_linestring(center, radius, start_angle, end_angle)
        
        return None
    
    def _circle_to_linestring(self, center: Tuple[float, float], radius: float) -> LineString:
        """Convert circle to LineString approximation."""
        num_points = max(8, int(radius * 4))
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append((x, y))
        return LineString(points)
    
    def _arc_to_linestring(self, center: Tuple[float, float], radius: float, 
                          start_angle: float, end_angle: float) -> LineString:
        """Convert arc to LineString approximation."""
        num_points = max(8, int(radius * 2))
        points = []
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append((x, y))
        return LineString(points)
    
    def _meets_minimum_criteria(self, geom: LineString) -> bool:
        """Check if geometry meets minimum criteria."""
        if geom.length < self.min_length:
            return False
        
        # Check area for closed geometries
        if geom.is_closed:
            try:
                poly = Polygon(geom)
                if poly.area < self.min_area:
                    return False
            except:
                pass
        
        return True
    
    def apply_douglas_peucker_simplification(self, geometries: List[LineString]) -> List[LineString]:
        """Apply Douglas-Peucker simplification to reduce node count."""
        simplified = []
        for geom in geometries:
            try:
                # Convert to numpy array for processing
                coords = np.array(geom.coords)
                if len(coords) > 2:
                    # Apply Douglas-Peucker algorithm
                    simplified_coords = self._douglas_peucker(coords, self.simplify_tolerance)
                    if len(simplified_coords) >= 2:
                        simplified.append(LineString(simplified_coords))
                    else:
                        simplified.append(geom)
                else:
                    simplified.append(geom)
            except Exception as e:
                print(f"Simplification failed: {e}")
                simplified.append(geom)
        
        return simplified
    
    def _douglas_peucker(self, points: np.ndarray, tolerance: float) -> np.ndarray:
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
            # Recursive call on both segments
            left_points = self._douglas_peucker(points[:max_index + 1], tolerance)
            right_points = self._douglas_peucker(points[max_index:], tolerance)
            
            # Combine results (remove duplicate point)
            return np.vstack([left_points[:-1], right_points])
        else:
            # Return endpoints only
            return np.array([points[0], points[-1]])
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line."""
        A = point - line_start
        B = line_end - line_start
        B_norm = np.linalg.norm(B)
        
        if B_norm == 0:
            return np.linalg.norm(A)
        
        return np.linalg.norm(A - np.dot(A, B) / B_norm * B)
    
    def apply_endpoint_snapping(self, geometries: List[LineString]) -> List[LineString]:
        """Apply endpoint snapping to close open contours."""
        snapped = []
        fixed_count = 0
        
        for geom in geometries:
            if geom.is_closed:
                snapped.append(geom)
                continue
            
            coords = list(geom.coords)
            start = Point(coords[0])
            end = Point(coords[-1])
            
            # If endpoints are close, snap them together
            if start.distance(end) <= self.endpoint_tolerance:
                coords.append(coords[0])  # Close the loop
                snapped.append(LineString(coords))
                fixed_count += 1
            else:
                snapped.append(geom)
        
        self.report["open_contours_fixed"] = fixed_count
        return snapped
    
    def apply_graph_based_stitching(self, geometries: List[LineString]) -> List[LineString]:
        """Use graph-based approach to stitch remaining open contours."""
        stitched = []
        open_geometries = [g for g in geometries if not g.is_closed]
        closed_geometries = [g for g in geometries if g.is_closed]
        
        if not open_geometries:
            return geometries
        
        # Create graph of open geometries
        G = nx.Graph()
        for i, geom in enumerate(open_geometries):
            G.add_node(i, geometry=geom)
        
        # Connect geometries with close endpoints
        for i in range(len(open_geometries)):
            for j in range(i + 1, len(open_geometries)):
                geom1 = open_geometries[i]
                geom2 = open_geometries[j]
                
                # Check all endpoint combinations
                endpoints1 = [Point(geom1.coords[0]), Point(geom1.coords[-1])]
                endpoints2 = [Point(geom2.coords[0]), Point(geom2.coords[-1])]
                
                min_dist = float('inf')
                for ep1 in endpoints1:
                    for ep2 in endpoints2:
                        dist = ep1.distance(ep2)
                        if dist < min_dist:
                            min_dist = dist
                
                if min_dist <= self.endpoint_tolerance:
                    G.add_edge(i, j, weight=min_dist)
        
        # Find connected components and stitch them
        components = list(nx.connected_components(G))
        for component in components:
            if len(component) == 1:
                # Single geometry, keep as is
                idx = list(component)[0]
                stitched.append(open_geometries[idx])
            else:
                # Multiple geometries, stitch them together
                stitched_geom = self._stitch_geometries([open_geometries[i] for i in component])
                if stitched_geom:
                    stitched.append(stitched_geom)
        
        # Add closed geometries
        stitched.extend(closed_geometries)
        
        return stitched
    
    def _stitch_geometries(self, geometries: List[LineString]) -> Optional[LineString]:
        """Stitch multiple geometries into one continuous line."""
        if not geometries:
            return None
        
        if len(geometries) == 1:
            return geometries[0]
        
        # Start with first geometry
        result_coords = list(geometries[0].coords)
        remaining = geometries[1:]
        
        while remaining:
            last_point = Point(result_coords[-1])
            best_match = None
            best_distance = float('inf')
            best_reverse = False
            
            for i, geom in enumerate(remaining):
                start_point = Point(geom.coords[0])
                end_point = Point(geom.coords[-1])
                
                dist_start = last_point.distance(start_point)
                dist_end = last_point.distance(end_point)
                
                if dist_start < best_distance:
                    best_distance = dist_start
                    best_match = i
                    best_reverse = False
                
                if dist_end < best_distance:
                    best_distance = dist_end
                    best_match = i
                    best_reverse = True
            
            if best_match is not None and best_distance <= self.endpoint_tolerance:
                geom = remaining[best_match]
                if best_reverse:
                    result_coords.extend(reversed(geom.coords[1:]))
                else:
                    result_coords.extend(geom.coords[1:])
                remaining.pop(best_match)
            else:
                # Can't stitch further, return what we have
                break
        
        return LineString(result_coords) if len(result_coords) >= 2 else None
    
    def apply_spacing_validation(self, geometries: List[LineString]) -> List[LineString]:
        """Apply spacing validation and auto-fix using Shapely buffers."""
        if len(geometries) <= 1:
            return geometries
        
        # Create spatial index for efficient nearest neighbor search
        from shapely.strtree import STRtree
        
        # Build spatial index
        spatial_index = STRtree(geometries)
        
        fixed_geometries = []
        violations_resolved = 0
        
        for i, geom in enumerate(geometries):
            try:
                # Create buffer around geometry
                buffered = geom.buffer(self.min_spacing / 2)
                
                # Find potential conflicts - fix the query result handling
                potential_conflicts = spatial_index.query(buffered)
                # Convert to list and filter out the geometry itself
                conflicts = []
                for g in potential_conflicts:
                    if g != geom and hasattr(g, 'buffer'):
                        if g.buffer(self.min_spacing / 2).intersects(buffered):
                            conflicts.append(g)
                
                if not conflicts:
                    fixed_geometries.append(geom)
                else:
                    # Try to resolve conflicts by offsetting
                    resolved_geom = self._resolve_spacing_conflicts(geom, conflicts)
                    if resolved_geom:
                        fixed_geometries.append(resolved_geom)
                        violations_resolved += len(conflicts)
                    else:
                        # If resolution fails, keep original
                        fixed_geometries.append(geom)
                        
            except Exception as e:
                print(f"Spacing validation failed for geometry {i}: {e}")
                fixed_geometries.append(geom)
        
        self.report["spacing_violations_resolved"] = violations_resolved
        return fixed_geometries
    
    def _resolve_spacing_conflicts(self, geom: LineString, conflicts: List[LineString]) -> Optional[LineString]:
        """Resolve spacing conflicts by offsetting geometry."""
        try:
            # Try offsetting in different directions
            offset_distance = self.min_spacing / 4
            
            for direction in ['left', 'right']:
                try:
                    offset_geom = geom.parallel_offset(offset_distance, direction)
                    if not offset_geom.is_empty:
                        # Check if offset resolves conflicts
                        buffered = offset_geom.buffer(self.min_spacing / 2)
                        still_conflicts = any(g.buffer(self.min_spacing / 2).intersects(buffered) for g in conflicts)
                        if not still_conflicts:
                            return offset_geom
                except:
                    continue
            
            return None
        except:
            return None
    
    def create_clean_dxf(self, geometries: List[LineString], output_path: str):
        """Create a new DXF file with cleaned geometries."""
        doc = ezdxf.new()
        msp = doc.modelspace()
        
        for geom in geometries:
            try:
                coords = list(geom.coords)
                if len(coords) >= 2:
                    msp.add_lwpolyline(coords, close=geom.is_closed)
            except Exception as e:
                print(f"Error adding geometry: {e}")
        
        doc.saveas(output_path)
        self.report["output_file"] = output_path
        self.report["features_retained"] = len(geometries)
    
    def generate_report(self, output_path: str):
        """Generate comprehensive JSON report."""
        report_path = output_path.replace('.dxf', '_cleaning_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"Cleaning report saved: {report_path}")
    
    def clean_dxf(self, input_path: str, output_path: str) -> bool:
        """Main cleaning pipeline."""
        import time
        start_time = time.time()
        
        # Load DXF
        doc, msp = self.load_dxf(input_path)
        if not doc:
            return False
        
        # Extract geometries
        geometries = self.extract_geometries(msp)
        print(f"Extracted {len(geometries)} geometries")
        
        # Apply cleaning pipeline
        geometries = self.apply_douglas_peucker_simplification(geometries)
        print(f"After simplification: {len(geometries)} geometries")
        
        geometries = self.apply_endpoint_snapping(geometries)
        print(f"After endpoint snapping: {len(geometries)} geometries")
        
        geometries = self.apply_graph_based_stitching(geometries)
        print(f"After graph stitching: {len(geometries)} geometries")
        
        geometries = self.apply_spacing_validation(geometries)
        print(f"After spacing validation: {len(geometries)} geometries")
        
        # Create output
        self.create_clean_dxf(geometries, output_path)
        
        # Generate report
        self.report["processing_time"] = time.time() - start_time
        self.generate_report(output_path)
        
        print(f"Cleaning complete! Output: {output_path}")
        print(f"Level: {self.cleaning_level.value}")
        print(f"Features retained: {self.report['features_retained']}")
        print(f"Open contours fixed: {self.report['open_contours_fixed']}")
        print(f"Spacing violations resolved: {self.report['spacing_violations_resolved']}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Advanced DXF cleaner with multiple cleaning levels")
    parser.add_argument("input", help="Input DXF file path")
    parser.add_argument("-o", "--output", help="Output DXF file path")
    parser.add_argument("-l", "--level", choices=["strict", "balanced", "relaxed"], 
                       default="balanced", help="Cleaning level")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_cleaned_{args.level}.dxf"
    
    # Create cleaner
    cleaning_level = CleaningLevel(args.level)
    cleaner = AdvancedDXFCleaner(cleaning_level)
    
    # Run cleaning
    success = cleaner.clean_dxf(args.input, output_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
