#!/usr/bin/env python3
"""
Material-Based Kerf Table for Waterjet Cutting
Implements ChatGPT-5 recommendations for accurate kerf compensation.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

class MaterialType(Enum):
    GRANITE = "granite"
    MARBLE = "marble"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    STAINLESS_STEEL = "stainless_steel"
    BRASS = "brass"
    COPPER = "copper"
    CARBON_STEEL = "carbon_steel"
    TITANIUM = "titanium"
    CERAMIC = "ceramic"
    GLASS = "glass"
    PLASTIC = "plastic"
    RUBBER = "rubber"
    WOOD = "wood"
    COMPOSITE = "composite"

@dataclass
class KerfData:
    """Kerf data for a specific material and thickness."""
    material: MaterialType
    thickness_mm: float
    kerf_mm: float
    cutting_speed_mm_min: float
    pierce_time_sec: float
    quality_factor: float  # 0.0-1.0, affects surface finish
    notes: str = ""

class KerfTable:
    """Material-based kerf table for waterjet cutting."""
    
    def __init__(self):
        self.kerf_data = self._initialize_kerf_table()
    
    def _initialize_kerf_table(self) -> Dict[MaterialType, Dict[float, KerfData]]:
        """Initialize comprehensive kerf table."""
        kerf_table = {}
        
        # Granite
        kerf_table[MaterialType.GRANITE] = {
            6.0: KerfData(MaterialType.GRANITE, 6.0, 1.0, 800, 0.8, 0.9, "Standard granite"),
            12.0: KerfData(MaterialType.GRANITE, 12.0, 1.1, 600, 1.0, 0.9, "Medium granite"),
            20.0: KerfData(MaterialType.GRANITE, 20.0, 1.2, 400, 1.2, 0.8, "Thick granite"),
            25.0: KerfData(MaterialType.GRANITE, 25.0, 1.3, 350, 1.5, 0.8, "Heavy granite"),
            40.0: KerfData(MaterialType.GRANITE, 40.0, 1.4, 250, 2.0, 0.7, "Very thick granite"),
        }
        
        # Marble
        kerf_table[MaterialType.MARBLE] = {
            6.0: KerfData(MaterialType.MARBLE, 6.0, 0.8, 1000, 0.6, 0.95, "Standard marble"),
            12.0: KerfData(MaterialType.MARBLE, 12.0, 0.9, 800, 0.8, 0.95, "Medium marble"),
            20.0: KerfData(MaterialType.MARBLE, 20.0, 1.0, 600, 1.0, 0.9, "Thick marble"),
            25.0: KerfData(MaterialType.MARBLE, 25.0, 1.1, 500, 1.2, 0.9, "Heavy marble"),
            40.0: KerfData(MaterialType.MARBLE, 40.0, 1.2, 350, 1.8, 0.8, "Very thick marble"),
        }
        
        # Steel
        kerf_table[MaterialType.STEEL] = {
            3.0: KerfData(MaterialType.STEEL, 3.0, 0.6, 1200, 0.5, 0.9, "Thin steel"),
            6.0: KerfData(MaterialType.STEEL, 6.0, 0.7, 1000, 0.7, 0.9, "Standard steel"),
            12.0: KerfData(MaterialType.STEEL, 12.0, 0.8, 800, 1.0, 0.85, "Medium steel"),
            20.0: KerfData(MaterialType.STEEL, 20.0, 0.9, 600, 1.5, 0.8, "Thick steel"),
            25.0: KerfData(MaterialType.STEEL, 25.0, 1.0, 500, 2.0, 0.8, "Heavy steel"),
            40.0: KerfData(MaterialType.STEEL, 40.0, 1.1, 350, 3.0, 0.75, "Very thick steel"),
        }
        
        # Stainless Steel
        kerf_table[MaterialType.STAINLESS_STEEL] = {
            3.0: KerfData(MaterialType.STAINLESS_STEEL, 3.0, 0.7, 1000, 0.6, 0.9, "Thin stainless"),
            6.0: KerfData(MaterialType.STAINLESS_STEEL, 6.0, 0.8, 800, 0.8, 0.9, "Standard stainless"),
            12.0: KerfData(MaterialType.STAINLESS_STEEL, 12.0, 0.9, 600, 1.2, 0.85, "Medium stainless"),
            20.0: KerfData(MaterialType.STAINLESS_STEEL, 20.0, 1.0, 400, 2.0, 0.8, "Thick stainless"),
            25.0: KerfData(MaterialType.STAINLESS_STEEL, 25.0, 1.1, 350, 2.5, 0.8, "Heavy stainless"),
        }
        
        # Aluminum
        kerf_table[MaterialType.ALUMINUM] = {
            3.0: KerfData(MaterialType.ALUMINUM, 3.0, 0.5, 1500, 0.3, 0.95, "Thin aluminum"),
            6.0: KerfData(MaterialType.ALUMINUM, 6.0, 0.6, 1200, 0.5, 0.95, "Standard aluminum"),
            12.0: KerfData(MaterialType.ALUMINUM, 12.0, 0.7, 1000, 0.8, 0.9, "Medium aluminum"),
            20.0: KerfData(MaterialType.ALUMINUM, 20.0, 0.8, 800, 1.2, 0.9, "Thick aluminum"),
            25.0: KerfData(MaterialType.ALUMINUM, 25.0, 0.9, 600, 1.5, 0.85, "Heavy aluminum"),
        }
        
        # Brass
        kerf_table[MaterialType.BRASS] = {
            3.0: KerfData(MaterialType.BRASS, 3.0, 0.6, 1200, 0.4, 0.9, "Thin brass"),
            6.0: KerfData(MaterialType.BRASS, 6.0, 0.7, 1000, 0.6, 0.9, "Standard brass"),
            12.0: KerfData(MaterialType.BRASS, 12.0, 0.8, 800, 1.0, 0.85, "Medium brass"),
            20.0: KerfData(MaterialType.BRASS, 20.0, 0.9, 600, 1.5, 0.8, "Thick brass"),
        }
        
        # Copper
        kerf_table[MaterialType.COPPER] = {
            3.0: KerfData(MaterialType.COPPER, 3.0, 0.6, 1200, 0.4, 0.9, "Thin copper"),
            6.0: KerfData(MaterialType.COPPER, 6.0, 0.7, 1000, 0.6, 0.9, "Standard copper"),
            12.0: KerfData(MaterialType.COPPER, 12.0, 0.8, 800, 1.0, 0.85, "Medium copper"),
            20.0: KerfData(MaterialType.COPPER, 20.0, 0.9, 600, 1.5, 0.8, "Thick copper"),
        }
        
        # Titanium
        kerf_table[MaterialType.TITANIUM] = {
            3.0: KerfData(MaterialType.TITANIUM, 3.0, 0.8, 800, 0.8, 0.85, "Thin titanium"),
            6.0: KerfData(MaterialType.TITANIUM, 6.0, 0.9, 600, 1.2, 0.85, "Standard titanium"),
            12.0: KerfData(MaterialType.TITANIUM, 12.0, 1.0, 400, 2.0, 0.8, "Medium titanium"),
            20.0: KerfData(MaterialType.TITANIUM, 20.0, 1.1, 300, 3.0, 0.75, "Thick titanium"),
        }
        
        # Glass
        kerf_table[MaterialType.GLASS] = {
            3.0: KerfData(MaterialType.GLASS, 3.0, 0.4, 2000, 0.2, 0.95, "Thin glass"),
            6.0: KerfData(MaterialType.GLASS, 6.0, 0.5, 1500, 0.4, 0.95, "Standard glass"),
            12.0: KerfData(MaterialType.GLASS, 12.0, 0.6, 1000, 0.8, 0.9, "Medium glass"),
            20.0: KerfData(MaterialType.GLASS, 20.0, 0.7, 600, 1.5, 0.85, "Thick glass"),
        }
        
        # Ceramic
        kerf_table[MaterialType.CERAMIC] = {
            3.0: KerfData(MaterialType.CERAMIC, 3.0, 0.5, 1000, 0.5, 0.9, "Thin ceramic"),
            6.0: KerfData(MaterialType.CERAMIC, 6.0, 0.6, 800, 0.8, 0.9, "Standard ceramic"),
            12.0: KerfData(MaterialType.CERAMIC, 12.0, 0.7, 600, 1.2, 0.85, "Medium ceramic"),
            20.0: KerfData(MaterialType.CERAMIC, 20.0, 0.8, 400, 2.0, 0.8, "Thick ceramic"),
        }
        
        return kerf_table
    
    def get_kerf_data(self, material: MaterialType, thickness_mm: float) -> Optional[KerfData]:
        """Get kerf data for specific material and thickness."""
        if material not in self.kerf_data:
            return None
        
        thickness_data = self.kerf_data[material]
        
        # Find exact match
        if thickness_mm in thickness_data:
            return thickness_data[thickness_mm]
        
        # Find closest thickness
        available_thicknesses = sorted(thickness_data.keys())
        
        # If thickness is smaller than minimum, use minimum
        if thickness_mm < available_thicknesses[0]:
            return thickness_data[available_thicknesses[0]]
        
        # If thickness is larger than maximum, use maximum
        if thickness_mm > available_thicknesses[-1]:
            return thickness_data[available_thicknesses[-1]]
        
        # Find closest thickness
        closest_thickness = min(available_thicknesses, key=lambda x: abs(x - thickness_mm))
        return thickness_data[closest_thickness]
    
    def interpolate_kerf_data(self, material: MaterialType, thickness_mm: float) -> Optional[KerfData]:
        """Interpolate kerf data for thickness between available values."""
        if material not in self.kerf_data:
            return None
        
        thickness_data = self.kerf_data[material]
        available_thicknesses = sorted(thickness_data.keys())
        
        if thickness_mm <= available_thicknesses[0]:
            return thickness_data[available_thicknesses[0]]
        
        if thickness_mm >= available_thicknesses[-1]:
            return thickness_data[available_thicknesses[-1]]
        
        # Find surrounding thicknesses
        lower_thickness = None
        upper_thickness = None
        
        for i, t in enumerate(available_thicknesses):
            if t <= thickness_mm:
                lower_thickness = t
            if t >= thickness_mm and upper_thickness is None:
                upper_thickness = t
                break
        
        if lower_thickness is None or upper_thickness is None:
            return None
        
        # Interpolate
        lower_data = thickness_data[lower_thickness]
        upper_data = thickness_data[upper_thickness]
        
        if lower_thickness == upper_thickness:
            return lower_data
        
        # Linear interpolation
        ratio = (thickness_mm - lower_thickness) / (upper_thickness - lower_thickness)
        
        interpolated_kerf = lower_data.kerf_mm + ratio * (upper_data.kerf_mm - lower_data.kerf_mm)
        interpolated_speed = lower_data.cutting_speed_mm_min + ratio * (upper_data.cutting_speed_mm_min - lower_data.cutting_speed_mm_min)
        interpolated_pierce = lower_data.pierce_time_sec + ratio * (upper_data.pierce_time_sec - lower_data.pierce_time_sec)
        interpolated_quality = lower_data.quality_factor + ratio * (upper_data.quality_factor - lower_data.quality_factor)
        
        return KerfData(
            material=material,
            thickness_mm=thickness_mm,
            kerf_mm=interpolated_kerf,
            cutting_speed_mm_min=interpolated_speed,
            pierce_time_sec=interpolated_pierce,
            quality_factor=interpolated_quality,
            notes=f"Interpolated between {lower_thickness}mm and {upper_thickness}mm"
        )
    
    def get_available_materials(self) -> list:
        """Get list of available materials."""
        return list(self.kerf_data.keys())
    
    def get_available_thicknesses(self, material: MaterialType) -> list:
        """Get available thicknesses for a material."""
        if material not in self.kerf_data:
            return []
        return sorted(self.kerf_data[material].keys())
    
    def print_material_info(self, material: MaterialType):
        """Print detailed information about a material."""
        if material not in self.kerf_data:
            print(f"Material {material.value} not found in kerf table")
            return
        
        print(f"\n=== {material.value.upper()} KERF DATA ===")
        thickness_data = self.kerf_data[material]
        
        for thickness, data in sorted(thickness_data.items()):
            print(f"Thickness: {thickness}mm")
            print(f"  Kerf: {data.kerf_mm}mm")
            print(f"  Cutting Speed: {data.cutting_speed_mm_min} mm/min")
            print(f"  Pierce Time: {data.pierce_time_sec}s")
            print(f"  Quality Factor: {data.quality_factor}")
            print(f"  Notes: {data.notes}")
            print()

def main():
    """Test the kerf table."""
    kerf_table = KerfTable()
    
    # Test granite at 25mm thickness
    granite_data = kerf_table.get_kerf_data(MaterialType.GRANITE, 25.0)
    if granite_data:
        print(f"Granite 25mm: Kerf={granite_data.kerf_mm}mm, Speed={granite_data.cutting_speed_mm_min}mm/min")
    
    # Test interpolation
    granite_interp = kerf_table.interpolate_kerf_data(MaterialType.GRANITE, 15.0)
    if granite_interp:
        print(f"Granite 15mm (interpolated): Kerf={granite_interp.kerf_mm}mm, Speed={granite_interp.cutting_speed_mm_min}mm/min")
    
    # Print all materials
    print("\nAvailable materials:")
    for material in kerf_table.get_available_materials():
        print(f"  - {material.value}")
    
    # Print granite details
    kerf_table.print_material_info(MaterialType.GRANITE)

if __name__ == "__main__":
    main()
