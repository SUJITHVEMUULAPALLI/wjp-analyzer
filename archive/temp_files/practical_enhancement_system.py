#!/usr/bin/env python3
"""
Practical Agent Enhancement System
=================================

This module implements practical enhancements that integrate with our
existing working agent system, providing professional-grade features
without overwhelming complexity.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

class LayerType(Enum):
    """Layer classification types."""
    OUTER = "OUTER"
    COMPLEX = "COMPLEX"
    DECOR = "DECOR"
    UNKNOWN = "UNKNOWN"

class MaterialType(Enum):
    """Material types for cost calculation."""
    GRANITE = "GRANITE"
    MARBLE = "MARBLE"
    STAINLESS_STEEL = "STAINLESS_STEEL"
    ALUMINUM = "ALUMINUM"
    BRASS = "BRASS"
    GENERIC = "GENERIC"

@dataclass
class MaterialProfile:
    """Material cost and cutting parameters."""
    name: str
    cost_per_sq_mm: float
    cutting_speed_mm_min: float
    kerf_width_mm: float
    pierce_time_sec: float
    setup_cost: float

@dataclass
class EnhancedObjectInfo:
    """Enhanced object information for layer classification."""
    id: int
    area: float
    perimeter: float
    circularity: float
    solidity: float
    bounding_rect: Tuple[int, int, int, int]
    center: Tuple[float, float]
    aspect_ratio: float
    layer_type: LayerType
    complexity_score: float
    quality_score: float

class PracticalEnhancementSystem:
    """Practical enhancement system that works with existing agents."""
    
    def __init__(self):
        self.material_db = self._create_material_database()
    
    def _create_material_database(self):
        """Create material database."""
        return {
            MaterialType.GRANITE: MaterialProfile(
                name="Granite",
                cost_per_sq_mm=0.0012,  # ₹1.2 per sq mm
                cutting_speed_mm_min=800,
                kerf_width_mm=1.0,
                pierce_time_sec=3.0,
                setup_cost=500.0
            ),
            MaterialType.MARBLE: MaterialProfile(
                name="Marble",
                cost_per_sq_mm=0.0008,  # ₹0.8 per sq mm
                cutting_speed_mm_min=1000,
                kerf_width_mm=0.8,
                pierce_time_sec=2.5,
                setup_cost=400.0
            ),
            MaterialType.STAINLESS_STEEL: MaterialProfile(
                name="Stainless Steel",
                cost_per_sq_mm=0.002,  # ₹2.0 per sq mm
                cutting_speed_mm_min=600,
                kerf_width_mm=1.2,
                pierce_time_sec=4.0,
                setup_cost=800.0
            ),
            MaterialType.ALUMINUM: MaterialProfile(
                name="Aluminum",
                cost_per_sq_mm=0.0005,  # ₹0.5 per sq mm
                cutting_speed_mm_min=1200,
                kerf_width_mm=0.6,
                pierce_time_sec=1.5,
                setup_cost=300.0
            ),
            MaterialType.BRASS: MaterialProfile(
                name="Brass",
                cost_per_sq_mm=0.0015,  # ₹1.5 per sq mm
                cutting_speed_mm_min=700,
                kerf_width_mm=0.9,
                pierce_time_sec=3.5,
                setup_cost=600.0
            ),
            MaterialType.GENERIC: MaterialProfile(
                name="Generic Material",
                cost_per_sq_mm=0.0008,  # ₹0.8 per sq mm
                cutting_speed_mm_min=1000,
                kerf_width_mm=1.0,
                pierce_time_sec=3.0,
                setup_cost=400.0
            )
        }
    
    def enhance_existing_analysis(self, dxf_path: str, material_type: MaterialType = MaterialType.GENERIC) -> Dict[str, Any]:
        """Enhance existing DXF analysis with professional features."""
        try:
            # Load DXF and extract objects
            import ezdxf
            doc = ezdxf.readfile(dxf_path)
            
            # Extract objects from DXF
            objects = self._extract_objects_from_dxf(doc)
            
            # Classify objects by layer
            classified_objects = self._classify_objects_by_layer(objects)
            
            # Calculate enhanced costs
            cost_data = self._calculate_enhanced_costs(classified_objects, material_type)
            
            # Generate professional reports
            reports = self._generate_professional_reports(classified_objects, cost_data, dxf_path)
            
            return {
                "success": True,
                "objects": classified_objects,
                "cost_data": cost_data,
                "reports": reports,
                "enhancement_features": {
                    "layer_classification": True,
                    "material_integration": True,
                    "professional_reporting": True,
                    "csv_output": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "enhancement_features": {}
            }
    
    def _extract_objects_from_dxf(self, doc) -> List[Dict[str, Any]]:
        """Extract objects from DXF file."""
        objects = []
        
        for entity in doc.modelspace():
            if entity.dxftype() == 'LWPOLYLINE':
                # Get entity properties
                points = list(entity.get_points())
                
                if len(points) >= 3:
                    # Calculate basic properties
                    area = self._calculate_polyline_area(points)
                    perimeter = self._calculate_polyline_perimeter(points)
                    
                    # Bounding rectangle
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    
                    # Center
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    
                    # Aspect ratio
                    width = max_x - min_x
                    height = max_y - min_y
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Geometric properties
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    solidity = 1.0  # Simplified for polylines
                    
                    obj = {
                        "id": len(objects),
                        "area": area,
                        "perimeter": perimeter,
                        "circularity": circularity,
                        "solidity": solidity,
                        "bounding_rect": (int(min_x), int(min_y), int(width), int(height)),
                        "center": (center_x, center_y),
                        "aspect_ratio": aspect_ratio,
                        "points": points,
                        "layer": entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
                    }
                    
                    objects.append(obj)
        
        return objects
    
    def _calculate_polyline_area(self, points: List[Tuple[float, float]]) -> float:
        """Calculate area of polyline using shoelace formula."""
        if len(points) < 3:
            return 0.0
        
        # Close the polygon
        if points[0] != points[-1]:
            points = points + [points[0]]
        
        area = 0.0
        n = len(points)
        for i in range(n - 1):
            area += points[i][0] * points[i + 1][1]
            area -= points[i + 1][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def _calculate_polyline_perimeter(self, points: List[Tuple[float, float]]) -> float:
        """Calculate perimeter of polyline."""
        if len(points) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            perimeter += np.sqrt(dx * dx + dy * dy)
        
        return perimeter
    
    def _classify_objects_by_layer(self, objects: List[Dict[str, Any]]) -> List[EnhancedObjectInfo]:
        """Classify objects by layer type."""
        enhanced_objects = []
        
        for obj in objects:
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(obj)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(obj)
            
            # Classify layer type
            layer_type = self._classify_layer_type(obj, complexity_score)
            
            enhanced_obj = EnhancedObjectInfo(
                id=obj["id"],
                area=obj["area"],
                perimeter=obj["perimeter"],
                circularity=obj["circularity"],
                solidity=obj["solidity"],
                bounding_rect=obj["bounding_rect"],
                center=obj["center"],
                aspect_ratio=obj["aspect_ratio"],
                layer_type=layer_type,
                complexity_score=complexity_score,
                quality_score=quality_score
            )
            
            enhanced_objects.append(enhanced_obj)
        
        return enhanced_objects
    
    def _calculate_complexity_score(self, obj: Dict[str, Any]) -> float:
        """Calculate complexity score for object."""
        # Based on number of points and shape irregularity
        point_count = len(obj["points"])
        perimeter = obj["perimeter"]
        
        # Complexity based on point density
        complexity = point_count / perimeter if perimeter > 0 else 0
        
        # Add circularity factor
        circularity_factor = 1 - obj["circularity"]
        
        # Combined complexity
        total_complexity = complexity * 0.7 + circularity_factor * 0.3
        
        return min(total_complexity, 1.0)
    
    def _calculate_quality_score(self, obj: Dict[str, Any]) -> float:
        """Calculate quality score for object."""
        # Based on circularity and aspect ratio
        circularity = obj["circularity"]
        aspect_ratio = obj["aspect_ratio"]
        
        # Ideal aspect ratio is close to 1
        aspect_score = 1 - abs(1 - aspect_ratio) if aspect_ratio > 0 else 0
        
        # Quality is combination of circularity and aspect ratio
        quality = circularity * 0.6 + aspect_score * 0.4
        
        return min(quality, 1.0)
    
    def _classify_layer_type(self, obj: Dict[str, Any], complexity_score: float) -> LayerType:
        """Classify object by layer type."""
        area = obj["area"]
        perimeter = obj["perimeter"]
        
        # OUTER: Large boundary objects
        if area > 10000 and perimeter > 500:
            return LayerType.OUTER
        
        # COMPLEX: Complex geometric objects
        elif complexity_score > 0.05 or (area > 1000 and perimeter > 200):
            return LayerType.COMPLEX
        
        # DECOR: Small decorative elements
        elif area < 1000 and perimeter < 100:
            return LayerType.DECOR
        
        # UNKNOWN: Default category
        else:
            return LayerType.UNKNOWN
    
    def _calculate_enhanced_costs(self, objects: List[EnhancedObjectInfo], 
                                material_type: MaterialType) -> Dict[str, Any]:
        """Calculate enhanced costs with material integration."""
        material = self.material_db[material_type]
        
        # Calculate costs by layer
        layer_costs = {}
        total_cutting_length = 0
        total_area = 0
        total_pierces = 0
        
        for layer_type in LayerType:
            layer_objects = [obj for obj in objects if obj.layer_type == layer_type]
            
            if layer_objects:
                layer_length = sum(obj.perimeter for obj in layer_objects)
                layer_area = sum(obj.area for obj in layer_objects)
                layer_pierces = len(layer_objects)
                
                # Material cost
                material_cost = layer_area * material.cost_per_sq_mm
                
                # Cutting cost
                cutting_time = layer_length / material.cutting_speed_mm_min
                cutting_cost = cutting_time * 100  # ₹100 per minute
                
                # Pierce cost
                pierce_cost = layer_pierces * material.pierce_time_sec * 2  # ₹2 per second
                
                # Total layer cost
                total_layer_cost = material_cost + cutting_cost + pierce_cost
                
                layer_costs[layer_type.value] = {
                    "count": len(layer_objects),
                    "length_mm": layer_length,
                    "area_mm2": layer_area,
                    "pierces": layer_pierces,
                    "material_cost": material_cost,
                    "cutting_cost": cutting_cost,
                    "pierce_cost": pierce_cost,
                    "total_cost": total_layer_cost,
                    "cutting_time_min": cutting_time
                }
                
                total_cutting_length += layer_length
                total_area += layer_area
                total_pierces += layer_pierces
        
        # Overall costs
        setup_cost = material.setup_cost
        total_material_cost = total_area * material.cost_per_sq_mm
        total_cutting_cost = (total_cutting_length / material.cutting_speed_mm_min) * 100
        total_pierce_cost = total_pierces * material.pierce_time_sec * 2
        
        grand_total = setup_cost + total_material_cost + total_cutting_cost + total_pierce_cost
        
        return {
            "material_type": material_type.value,
            "material_profile": asdict(material),
            "layer_breakdown": layer_costs,
            "totals": {
                "total_length_mm": total_cutting_length,
                "total_area_mm2": total_area,
                "total_pierces": total_pierces,
                "setup_cost": setup_cost,
                "material_cost": total_material_cost,
                "cutting_cost": total_cutting_cost,
                "pierce_cost": total_pierce_cost,
                "grand_total": grand_total,
                "cutting_time_min": total_cutting_length / material.cutting_speed_mm_min
            }
        }
    
    def _generate_professional_reports(self, objects: List[EnhancedObjectInfo], 
                                     cost_data: Dict[str, Any], dxf_path: str) -> Dict[str, str]:
        """Generate professional reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output/reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate CSV report
        csv_path = self._generate_csv_report(objects, cost_data, timestamp, output_dir)
        
        # Generate JSON report
        json_path = self._generate_json_report(objects, cost_data, timestamp, output_dir)
        
        # Generate summary report
        summary_path = self._generate_summary_report(cost_data, timestamp, output_dir)
        
        return {
            "csv_report": csv_path,
            "json_report": json_path,
            "summary_report": summary_path
        }
    
    def _generate_csv_report(self, objects: List[EnhancedObjectInfo], 
                           cost_data: Dict[str, Any], timestamp: str, output_dir: str) -> str:
        """Generate professional CSV report."""
        csv_data = []
        
        for obj in objects:
            csv_data.append({
                "object_id": obj.id,
                "layer_type": obj.layer_type.value,
                "area_mm2": round(obj.area, 2),
                "perimeter_mm": round(obj.perimeter, 2),
                "circularity": round(obj.circularity, 3),
                "solidity": round(obj.solidity, 3),
                "aspect_ratio": round(obj.aspect_ratio, 3),
                "complexity_score": round(obj.complexity_score, 3),
                "quality_score": round(obj.quality_score, 3),
                "center_x": round(obj.center[0], 2),
                "center_y": round(obj.center[1], 2),
                "bounding_width": obj.bounding_rect[2],
                "bounding_height": obj.bounding_rect[3]
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"enhanced_analysis_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def _generate_json_report(self, objects: List[EnhancedObjectInfo], 
                            cost_data: Dict[str, Any], timestamp: str, output_dir: str) -> str:
        """Generate professional JSON report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "enhanced_practical",
            "total_objects": len(objects),
            "layer_breakdown": {
                layer.value: len([obj for obj in objects if obj.layer_type == layer])
                for layer in LayerType
            },
            "cost_analysis": cost_data,
            "object_details": [
                {
                    "id": obj.id,
                    "layer_type": obj.layer_type.value,
                    "area": obj.area,
                    "perimeter": obj.perimeter,
                    "complexity_score": obj.complexity_score,
                    "quality_score": obj.quality_score
                }
                for obj in objects
            ]
        }
        
        json_path = os.path.join(output_dir, f"enhanced_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return json_path
    
    def _generate_summary_report(self, cost_data: Dict[str, Any], timestamp: str, output_dir: str) -> str:
        """Generate executive summary report."""
        summary = f"""# Enhanced Analysis Summary - {timestamp}

## Cost Analysis
- **Material Type**: {cost_data['material_type']}
- **Total Length**: {cost_data['totals']['total_length_mm']:.2f} mm
- **Total Area**: {cost_data['totals']['total_area_mm2']:.2f} mm²
- **Total Pierces**: {cost_data['totals']['total_pierces']}
- **Grand Total Cost**: ₹{cost_data['totals']['grand_total']:.2f}
- **Cutting Time**: {cost_data['totals']['cutting_time_min']:.2f} minutes

## Layer Breakdown
"""
        
        for layer, data in cost_data['layer_breakdown'].items():
            summary += f"- **{layer}**: {data['count']} objects, ₹{data['total_cost']:.2f}\n"
        
        summary_path = os.path.join(output_dir, f"enhanced_summary_{timestamp}.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return summary_path

def test_practical_enhancements():
    """Test practical enhancements with existing DXF files."""
    print("🚀 **Testing Practical Agent Enhancements**")
    print("=" * 60)
    
    # Test with existing enhanced DXF file
    test_dxf = "C:\\WJP ANALYSER\\output\\dxf\\Tile 21_converted_enhanced_fixed.dxf"
    
    if not os.path.exists(test_dxf):
        print(f"❌ Test DXF not found: {test_dxf}")
        return
    
    try:
        # Initialize enhancement system
        enhancer = PracticalEnhancementSystem()
        
        # Enhance existing analysis
        print("1. 🔍 Enhancing existing DXF analysis...")
        result = enhancer.enhance_existing_analysis(test_dxf, MaterialType.GRANITE)
        
        if result["success"]:
            objects = result["objects"]
            cost_data = result["cost_data"]
            reports = result["reports"]
            
            print(f"   ✅ Enhanced {len(objects)} objects")
            
            # Show layer classification
            layer_counts = {}
            for layer in LayerType:
                count = len([obj for obj in objects if obj.layer_type == layer])
                layer_counts[layer.value] = count
                if count > 0:
                    print(f"   📋 {layer.value}: {count} objects")
            
            # Show cost analysis
            print(f"\n2. 💰 Enhanced cost analysis...")
            print(f"   💵 Total Cost: ₹{cost_data['totals']['grand_total']:.2f}")
            print(f"   ⏱️ Cutting Time: {cost_data['totals']['cutting_time_min']:.2f} minutes")
            print(f"   📏 Total Length: {cost_data['totals']['total_length_mm']:.2f} mm")
            
            # Show reports
            print(f"\n3. 📄 Professional reports generated...")
            print(f"   📊 CSV Report: {os.path.basename(reports['csv_report'])}")
            print(f"   📋 JSON Report: {os.path.basename(reports['json_report'])}")
            print(f"   📝 Summary Report: {os.path.basename(reports['summary_report'])}")
            
            # Show enhancement features
            print(f"\n4. 🎯 Enhancement features applied...")
            features = result["enhancement_features"]
            for feature, enabled in features.items():
                status = "✅" if enabled else "❌"
                print(f"   {status} {feature.replace('_', ' ').title()}")
            
            print("\n🎉 **Practical Enhancement Test Completed Successfully!**")
            
            return result
        else:
            print(f"❌ Enhancement failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_practical_enhancements()
