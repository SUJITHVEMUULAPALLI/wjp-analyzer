"""
Preset Configuration Loader
===========================

Utility for loading and applying toolpath optimization presets.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..manufacturing.toolpath import ToolpathOptimization

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class PresetConfig:
    """Configuration for loading presets."""
    presets_dir: str = "config/presets"
    default_preset: str = "standard"


class PresetLoader:
    """Loads and applies toolpath optimization presets."""
    
    def __init__(self, config: Optional[PresetConfig] = None):
        self.config = config or PresetConfig()
        self.presets_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a preset configuration."""
        if preset_name in self.presets_cache:
            return self.presets_cache[preset_name]
        
        if not YAML_AVAILABLE:
            # Return default preset if yaml is not available
            return self._get_default_preset()
        
        preset_file = os.path.join(self.config.presets_dir, f"{preset_name}.yaml")
        if not os.path.exists(preset_file):
            # Try loading from advanced_toolpath.yaml
            advanced_file = os.path.join(self.config.presets_dir, "advanced_toolpath.yaml")
            if os.path.exists(advanced_file):
                with open(advanced_file, 'r') as f:
                    all_presets = yaml.safe_load(f)
                    if preset_name in all_presets:
                        preset_data = all_presets[preset_name]
                        self.presets_cache[preset_name] = preset_data
                        return preset_data
        
        if os.path.exists(preset_file):
            with open(preset_file, 'r') as f:
                preset_data = yaml.safe_load(f)
                self.presets_cache[preset_name] = preset_data
                return preset_data
        
        # Return default preset if not found
        return self._get_default_preset()
    
    def _get_default_preset(self) -> Dict[str, Any]:
        """Get default preset configuration."""
        return {
            "kerf_compensation": 1.1,
            "rapid_speed": 10000.0,
            "cutting_speed": 1200.0,
            "pierce_time": 0.5,
            "min_rapid_distance": 5.0,
            "optimize_rapids": True,
            "optimize_direction": True,
            "entry_strategy": "tangent"
        }
    
    def create_toolpath_optimization(self, preset_name: str, **overrides) -> ToolpathOptimization:
        """Create a ToolpathOptimization from a preset with optional overrides."""
        preset_data = self.load_preset(preset_name)
        
        # Apply overrides
        config_data = {**preset_data, **overrides}
        
        return ToolpathOptimization(
            kerf_compensation=config_data.get("kerf_compensation", 1.1),
            rapid_speed=config_data.get("rapid_speed", 10000.0),
            cutting_speed=config_data.get("cutting_speed", 1200.0),
            pierce_time=config_data.get("pierce_time", 0.5),
            min_rapid_distance=config_data.get("min_rapid_distance", 5.0),
            optimize_rapids=config_data.get("optimize_rapids", True),
            optimize_direction=config_data.get("optimize_direction", True),
            entry_strategy=config_data.get("entry_strategy", "tangent")
        )
    
    def list_available_presets(self) -> list[str]:
        """List all available presets."""
        presets = []
        
        if not YAML_AVAILABLE:
            return ["standard"]  # Return default preset if yaml not available
        
        # Check individual preset files
        if os.path.exists(self.config.presets_dir):
            for file in os.listdir(self.config.presets_dir):
                if file.endswith('.yaml'):
                    preset_name = file[:-5]  # Remove .yaml extension
                    presets.append(preset_name)
        
        # Check advanced_toolpath.yaml for nested presets
        advanced_file = os.path.join(self.config.presets_dir, "advanced_toolpath.yaml")
        if os.path.exists(advanced_file):
            with open(advanced_file, 'r') as f:
                all_presets = yaml.safe_load(f)
                if isinstance(all_presets, dict):
                    presets.extend(all_presets.keys())
        
        return sorted(list(set(presets)))
    
    def get_material_preset(self, material: str) -> Optional[str]:
        """Get preset name for a specific material."""
        if not YAML_AVAILABLE:
            return None
            
        advanced_file = os.path.join(self.config.presets_dir, "advanced_toolpath.yaml")
        if os.path.exists(advanced_file):
            with open(advanced_file, 'r') as f:
                all_presets = yaml.safe_load(f)
                materials = all_presets.get("materials", {})
                return materials.get(material.lower())
        return None
    
    def get_thickness_preset(self, thickness: float) -> Optional[str]:
        """Get preset name for a specific thickness."""
        if not YAML_AVAILABLE:
            return None
            
        advanced_file = os.path.join(self.config.presets_dir, "advanced_toolpath.yaml")
        if os.path.exists(advanced_file):
            with open(advanced_file, 'r') as f:
                all_presets = yaml.safe_load(f)
                thickness_presets = all_presets.get("thickness", {})
                
                # Find closest thickness preset
                closest_preset = None
                min_diff = float('inf')
                
                for preset_name, preset_data in thickness_presets.items():
                    # Extract thickness from preset name (e.g., "thin_10mm" -> 10)
                    if "_" in preset_name:
                        try:
                            preset_thickness = float(preset_name.split("_")[1].replace("mm", ""))
                            diff = abs(thickness - preset_thickness)
                            if diff < min_diff:
                                min_diff = diff
                                closest_preset = preset_name
                        except (ValueError, IndexError):
                            continue
                
                return closest_preset
        return None


# Convenience functions
def load_preset(preset_name: str) -> Dict[str, Any]:
    """Load a preset configuration."""
    loader = PresetLoader()
    return loader.load_preset(preset_name)


def create_optimization_from_preset(preset_name: str, **overrides) -> ToolpathOptimization:
    """Create a ToolpathOptimization from a preset."""
    loader = PresetLoader()
    return loader.create_toolpath_optimization(preset_name, **overrides)


def list_presets() -> list[str]:
    """List all available presets."""
    loader = PresetLoader()
    return loader.list_available_presets()
