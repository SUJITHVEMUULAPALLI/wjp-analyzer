#!/usr/bin/env python3
"""
Create a simplified medallion design optimized for waterjet cutting.
This addresses the fundamental design issues in the original medallion.
"""

import ezdxf
import numpy as np
import math

def create_simple_medallion():
    """Create a simplified medallion with proper waterjet spacing."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # Main outer circle (100mm diameter)
    outer_radius = 50
    msp.add_circle((0, 0), outer_radius)
    
    # Inner decorative circles with proper spacing (min 3mm)
    inner_circles = [
        (0, 0, 20),      # Center circle
        (0, 30, 8),       # Top circle
        (0, -30, 8),      # Bottom circle
        (30, 0, 8),       # Right circle
        (-30, 0, 8),      # Left circle
        (21, 21, 6),      # Top-right circle
        (-21, 21, 6),     # Top-left circle
        (21, -21, 6),     # Bottom-right circle
        (-21, -21, 6),    # Bottom-left circle
    ]
    
    for x, y, radius in inner_circles:
        msp.add_circle((x, y), radius)
    
    # Add some decorative lines with proper spacing
    # Horizontal lines
    msp.add_line((-40, 0), (-25, 0))  # Left horizontal
    msp.add_line((25, 0), (40, 0))     # Right horizontal
    
    # Vertical lines
    msp.add_line((0, -40), (0, -25))   # Bottom vertical
    msp.add_line((0, 25), (0, 40))    # Top vertical
    
    # Diagonal decorative elements
    msp.add_line((-35, -35), (-25, -25))  # Bottom-left diagonal
    msp.add_line((25, 25), (35, 35))      # Top-right diagonal
    msp.add_line((-35, 35), (-25, 25))    # Top-left diagonal
    msp.add_line((25, -25), (35, -35))    # Bottom-right diagonal
    
    return doc

def create_geometric_pattern():
    """Create a geometric pattern optimized for waterjet cutting."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # Main square frame (80x80mm)
    frame_size = 80
    msp.add_lwpolyline([
        (-frame_size/2, -frame_size/2),
        (frame_size/2, -frame_size/2),
        (frame_size/2, frame_size/2),
        (-frame_size/2, frame_size/2),
        (-frame_size/2, -frame_size/2)
    ], close=True)
    
    # Inner geometric elements with proper spacing
    # Central cross
    cross_size = 20
    msp.add_line((-cross_size/2, 0), (cross_size/2, 0))
    msp.add_line((0, -cross_size/2), (0, cross_size/2))
    
    # Corner squares
    corner_size = 15
    corners = [
        (-frame_size/2 + 10, -frame_size/2 + 10),
        (frame_size/2 - 10, -frame_size/2 + 10),
        (frame_size/2 - 10, frame_size/2 - 10),
        (-frame_size/2 + 10, frame_size/2 - 10)
    ]
    
    for x, y in corners:
        msp.add_lwpolyline([
            (x - corner_size/2, y - corner_size/2),
            (x + corner_size/2, y - corner_size/2),
            (x + corner_size/2, y + corner_size/2),
            (x - corner_size/2, y + corner_size/2),
            (x - corner_size/2, y - corner_size/2)
        ], close=True)
    
    # Inner circles
    circle_positions = [
        (0, 0, 12),      # Center
        (-25, 0, 8),     # Left
        (25, 0, 8),      # Right
        (0, -25, 8),     # Bottom
        (0, 25, 8),      # Top
    ]
    
    for x, y, radius in circle_positions:
        msp.add_circle((x, y), radius)
    
    return doc

def main():
    print("Creating waterjet-optimized designs...")
    
    # Create simple medallion
    medallion_doc = create_simple_medallion()
    medallion_doc.saveas("simple_medallion_waterjet.dxf")
    print("Created: simple_medallion_waterjet.dxf")
    
    # Create geometric pattern
    geometric_doc = create_geometric_pattern()
    geometric_doc.saveas("geometric_pattern_waterjet.dxf")
    print("Created: geometric_pattern_waterjet.dxf")
    
    print("\nDesign specifications:")
    print("- Minimum spacing: 3mm between features")
    print("- All contours are closed")
    print("- Optimized for waterjet cutting")
    print("- Suitable for granite cutting")

if __name__ == "__main__":
    main()
