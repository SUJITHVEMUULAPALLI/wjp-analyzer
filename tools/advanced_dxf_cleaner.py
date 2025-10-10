#!/usr/bin/env python3
"""
Advanced DXF Cleaner for Waterjet Cutting
Fixes open contours and spacing violations for waterjet manufacturing.
"""

import ezdxf
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
import argparse
import sys
import os

def load_dxf(filepath):
    """Load DXF file and extract geometries."""
    try:
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        return doc, msp
    except Exception as e:
        print(f"Error loading DXF: {e}")
        return None, None

def extract_geometries(msp, min_length=1.0):
    """Extract and filter geometries from DXF."""
    geometries = []
    
    for entity in msp.query("LWPOLYLINE LINE POLYLINE CIRCLE ARC"):
        try:
            if entity.dxftype() == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                line = LineString([(start[0], start[1]), (end[0], end[1])])
                if line.length >= min_length:
                    geometries.append(line)
                    
            elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                points = [(p[0], p[1]) for p in entity.get_points()]
                if len(points) >= 2:
                    poly = LineString(points)
                    if poly.length >= min_length:
                        geometries.append(poly)
                        
            elif entity.dxftype() == "CIRCLE":
                center = entity.dxf.center
                radius = entity.dxf.radius
                if radius >= min_length / (2 * np.pi):
                    # Convert circle to polygon approximation
                    num_points = max(8, int(radius * 4))
                    points = []
                    for i in range(num_points):
                        angle = 2 * np.pi * i / num_points
                        x = center[0] + radius * np.cos(angle)
                        y = center[1] + radius * np.sin(angle)
                        points.append((x, y))
                    circle_poly = LineString(points)
                    geometries.append(circle_poly)
                    
            elif entity.dxftype() == "ARC":
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                
                # Convert arc to line segments
                num_points = max(8, int(radius * 2))
                points = []
                for i in range(num_points + 1):
                    angle = start_angle + (end_angle - start_angle) * i / num_points
                    x = center[0] + radius * np.cos(angle)
                    y = center[1] + radius * np.sin(angle)
                    points.append((x, y))
                arc_line = LineString(points)
                if arc_line.length >= min_length:
                    geometries.append(arc_line)
                    
        except Exception as e:
            print(f"Skipped entity: {e}")
            continue
    
    return geometries

def fix_open_contours(geometries, tolerance=0.1):
    """Attempt to close open contours by connecting nearby endpoints."""
    closed_geometries = []
    open_geometries = []
    
    for geom in geometries:
        if geom.is_closed:
            closed_geometries.append(geom)
        else:
            open_geometries.append(geom)
    
    # Try to close open contours
    for geom in open_geometries:
        coords = list(geom.coords)
        start = Point(coords[0])
        end = Point(coords[-1])
        
        # If start and end are close, close the contour
        if start.distance(end) <= tolerance:
            coords.append(coords[0])  # Close the loop
            closed_geom = LineString(coords)
            closed_geometries.append(closed_geom)
        else:
            # Keep as open contour
            closed_geometries.append(geom)
    
    return closed_geometries

def fix_spacing_violations(geometries, min_spacing=3.0):
    """Fix spacing violations by offsetting geometries."""
    if len(geometries) <= 1:
        return geometries
    
    fixed_geometries = []
    
    for i, geom in enumerate(geometries):
        try:
            # Create buffer around geometry
            buffered = geom.buffer(min_spacing / 2)
            
            # Check for overlaps with other geometries
            overlaps = False
            for j, other_geom in enumerate(geometries):
                if i != j:
                    other_buffered = other_geom.buffer(min_spacing / 2)
                    if buffered.intersects(other_buffered):
                        overlaps = True
                        break
            
            if not overlaps:
                fixed_geometries.append(geom)
            else:
                # Try to offset the geometry
                try:
                    # Simple offset approach - move geometry slightly
                    offset_distance = min_spacing / 4
                    offset_geom = geom.parallel_offset(offset_distance, 'left')
                    if offset_geom.is_empty:
                        offset_geom = geom.parallel_offset(offset_distance, 'right')
                    
                    if not offset_geom.is_empty:
                        fixed_geometries.append(offset_geom)
                    else:
                        # If offset fails, keep original but mark as problematic
                        fixed_geometries.append(geom)
                except:
                    # If all else fails, keep original
                    fixed_geometries.append(geom)
                    
        except Exception as e:
            print(f"Error processing geometry {i}: {e}")
            fixed_geometries.append(geom)
    
    return fixed_geometries

def create_clean_dxf(geometries, output_path):
    """Create a new DXF file with cleaned geometries."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    for geom in geometries:
        try:
            coords = list(geom.coords)
            if len(coords) >= 2:
                # Add as closed polyline
                msp.add_lwpolyline(coords, close=True)
        except Exception as e:
            print(f"Error adding geometry: {e}")
    
    doc.saveas(output_path)
    print(f"Saved cleaned DXF: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced DXF cleaner for waterjet cutting")
    parser.add_argument("input", help="Input DXF file path")
    parser.add_argument("-o", "--output", help="Output DXF file path")
    parser.add_argument("--min-length", type=float, default=1.0, help="Minimum geometry length")
    parser.add_argument("--min-spacing", type=float, default=3.0, help="Minimum spacing between features")
    parser.add_argument("--close-tolerance", type=float, default=0.1, help="Tolerance for closing contours")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_waterjet_ready.dxf"
    
    print(f"Loading DXF: {args.input}")
    doc, msp = load_dxf(args.input)
    if not doc:
        return 1
    
    print("Extracting geometries...")
    geometries = extract_geometries(msp, args.min_length)
    print(f"Found {len(geometries)} geometries")
    
    print("Fixing open contours...")
    geometries = fix_open_contours(geometries, args.close_tolerance)
    
    print("Fixing spacing violations...")
    geometries = fix_spacing_violations(geometries, args.min_spacing)
    
    print(f"Creating cleaned DXF with {len(geometries)} geometries...")
    create_clean_dxf(geometries, output_path)
    
    print("Cleaning complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
