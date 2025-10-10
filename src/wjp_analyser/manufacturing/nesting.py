"""
Nesting functionality for arranging multiple DXF files on a sheet.
"""
import os
import json
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Polygon, box
from shapely.affinity import translate as shp_translate
from shapely.ops import unary_union
import ezdxf
from ezdxf import recover


class NestingEngine:
    """Engine for nesting multiple DXF files on a sheet."""
    
    def __init__(self, sheet_width: float = 3000.0, sheet_height: float = 1500.0):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.sheet_bounds = box(0, 0, sheet_width, sheet_height)
        
    def load_dxf_polygons(self, dxf_path: str) -> List[Polygon]:
        """Load polygons from a DXF file."""
        try:
            doc, auditor = recover.readfile(dxf_path)
            msp = doc.modelspace()
            
            polygons = []
            for entity in msp:
                if entity.dxftype() == 'LWPOLYLINE':
                    # Convert LWPOLYLINE to polygon
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) >= 3:
                        try:
                            poly = Polygon(points)
                            if poly.is_valid:
                                polygons.append(poly)
                        except:
                            continue
                elif entity.dxftype() == 'POLYLINE':
                    # Convert POLYLINE to polygon
                    points = [(p[0], p[1]) for p in entity.points()]
                    if len(points) >= 3:
                        try:
                            poly = Polygon(points)
                            if poly.is_valid:
                                polygons.append(poly)
                        except:
                            continue
                            
            return polygons
        except Exception as e:
            print(f"Error loading DXF {dxf_path}: {e}")
            return []
    
    def calculate_bounds(self, polygons: List[Polygon]) -> Tuple[float, float]:
        """Calculate bounding box dimensions for a set of polygons."""
        if not polygons:
            return 0, 0
            
        # Get union of all polygons
        union_poly = unary_union(polygons)
        bounds = union_poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        return width, height
    
    def simple_nesting(self, dxf_files: List[str], spacing: float = 10.0) -> Dict:
        """
        Simple nesting algorithm - arrange files in rows.
        Returns nesting information including positions and sheet utilization.
        """
        nested_items = []
        current_x = spacing
        current_y = spacing
        max_height_in_row = 0
        row_count = 0
        
        total_area = 0
        used_area = 0
        
        for dxf_path in dxf_files:
            polygons = self.load_dxf_polygons(dxf_path)
            if not polygons:
                continue
                
            width, height = self.calculate_bounds(polygons)
            total_area += width * height
            
            # Check if item fits in current row
            if current_x + width + spacing > self.sheet_width:
                # Move to next row
                current_x = spacing
                current_y += max_height_in_row + spacing
                max_height_in_row = 0
                row_count += 1
            
            # Check if item fits on sheet
            if current_y + height + spacing > self.sheet_height:
                print(f"Warning: {os.path.basename(dxf_path)} doesn't fit on sheet")
                continue
            
            # Place item
            nested_items.append({
                'dxf_path': dxf_path,
                'x': current_x,
                'y': current_y,
                'width': width,
                'height': height,
                'polygons': polygons,
                'row': row_count
            })
            
            used_area += width * height
            current_x += width + spacing
            max_height_in_row = max(max_height_in_row, height)
        
        utilization = (used_area / (self.sheet_width * self.sheet_height)) * 100
        
        return {
            'items': nested_items,
            'sheet_width': self.sheet_width,
            'sheet_height': self.sheet_height,
            'utilization_percent': round(utilization, 2),
            'total_items': len(nested_items),
            'spacing': spacing
        }
    
    def generate_nested_dxf(self, nesting_result: Dict, output_path: str) -> bool:
        """Generate a DXF file with all nested items."""
        try:
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            for item in nesting_result['items']:
                # Add each polygon from the item
                for poly in item['polygons']:
                    # Translate polygon to nesting position
                    translated_poly = shp_translate(poly, xoff=item['x'], yoff=item['y'])
                    
                    # Convert to LWPOLYLINE
                    points = list(translated_poly.exterior.coords)
                    msp.add_lwpolyline(points, format="xy", close=True)
            
            doc.saveas(output_path)
            return True
        except Exception as e:
            print(f"Error generating nested DXF: {e}")
            return False
    
    def generate_nesting_report(self, nesting_result: Dict, output_path: str) -> bool:
        """Generate a JSON report of the nesting."""
        try:
            report = {
                'sheet_dimensions': {
                    'width': nesting_result['sheet_width'],
                    'height': nesting_result['sheet_height']
                },
                'utilization_percent': nesting_result['utilization_percent'],
                'total_items': nesting_result['total_items'],
                'spacing': nesting_result['spacing'],
                'items': []
            }
            
            for item in nesting_result['items']:
                report['items'].append({
                    'file': os.path.basename(item['dxf_path']),
                    'position': {'x': item['x'], 'y': item['y']},
                    'dimensions': {'width': item['width'], 'height': item['height']},
                    'row': item['row']
                })
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            return True
        except Exception as e:
            print(f"Error generating nesting report: {e}")
            return False
