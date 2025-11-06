# Material Tolerance Profiles for Waterjet Cutting
# ===============================================

# Cutting profiles for different materials with optimized parameters
MATERIAL_PROFILES = {
    "granite": {
        "name": "Granite",
        "min_spacing": 3.0,      # mm - minimum spacing between features
        "min_radius": 2.0,       # mm - minimum corner radius
        "kerf_width": 1.2,       # mm - typical kerf width
        "cutting_speed": 800.0,  # mm/min - optimal cutting speed
        "pierce_time": 0.8,      # seconds - pierce delay
        "description": "Hard stone material requiring conservative spacing"
    },
    
    "steel": {
        "name": "Steel",
        "min_spacing": 1.0,      # mm - minimum spacing between features
        "min_radius": 1.5,       # mm - minimum corner radius
        "kerf_width": 0.8,       # mm - typical kerf width
        "cutting_speed": 1200.0, # mm/min - optimal cutting speed
        "pierce_time": 0.5,      # seconds - pierce delay
        "description": "Metal material allowing tighter tolerances"
    },
    
    "aluminum": {
        "name": "Aluminum",
        "min_spacing": 1.5,      # mm - minimum spacing between features
        "min_radius": 1.5,       # mm - minimum corner radius
        "kerf_width": 0.6,       # mm - typical kerf width
        "cutting_speed": 1500.0, # mm/min - optimal cutting speed
        "pierce_time": 0.3,      # seconds - pierce delay
        "description": "Soft metal with good cutting characteristics"
    },
    
    "stainless_steel": {
        "name": "Stainless Steel",
        "min_spacing": 1.2,      # mm - minimum spacing between features
        "min_radius": 1.8,       # mm - minimum corner radius
        "kerf_width": 0.9,       # mm - typical kerf width
        "cutting_speed": 1000.0, # mm/min - optimal cutting speed
        "pierce_time": 0.6,      # seconds - pierce delay
        "description": "Hard metal requiring careful handling"
    },
    
    "titanium": {
        "name": "Titanium",
        "min_spacing": 2.0,      # mm - minimum spacing between features
        "min_radius": 2.5,       # mm - minimum corner radius
        "kerf_width": 1.0,       # mm - typical kerf width
        "cutting_speed": 900.0,  # mm/min - optimal cutting speed
        "pierce_time": 0.7,      # seconds - pierce delay
        "description": "Exotic metal requiring conservative approach"
    },
    
    "glass": {
        "name": "Glass",
        "min_spacing": 4.0,      # mm - minimum spacing between features
        "min_radius": 3.0,       # mm - minimum corner radius
        "kerf_width": 1.5,       # mm - typical kerf width
        "cutting_speed": 600.0,  # mm/min - optimal cutting speed
        "pierce_time": 1.0,      # seconds - pierce delay
        "description": "Brittle material requiring maximum spacing"
    },
    
    "ceramic": {
        "name": "Ceramic",
        "min_spacing": 3.5,      # mm - minimum spacing between features
        "min_radius": 2.5,       # mm - minimum corner radius
        "kerf_width": 1.3,       # mm - typical kerf width
        "cutting_speed": 700.0,  # mm/min - optimal cutting speed
        "pierce_time": 0.9,      # seconds - pierce delay
        "description": "Hard ceramic material with brittle properties"
    },
    
    "composite": {
        "name": "Composite",
        "min_spacing": 2.5,      # mm - minimum spacing between features
        "min_radius": 2.0,       # mm - minimum corner radius
        "kerf_width": 1.1,       # mm - typical kerf width
        "cutting_speed": 1100.0, # mm/min - optimal cutting speed
        "pierce_time": 0.4,      # seconds - pierce delay
        "description": "Composite material with variable properties"
    }
}

# Default profile for unknown materials
DEFAULT_PROFILE = {
    "name": "Default",
    "min_spacing": 2.0,      # mm - conservative default
    "min_radius": 2.0,       # mm - conservative default
    "kerf_width": 1.0,        # mm - typical kerf width
    "cutting_speed": 1200.0,  # mm/min - standard speed
    "pierce_time": 0.5,      # seconds - standard pierce delay
    "description": "Conservative default profile for unknown materials"
}

def get_material_profile(material_name: str) -> dict:
    """
    Get material profile by name.
    
    Args:
        material_name: Name of the material (case-insensitive)
        
    Returns:
        Dictionary with material parameters
    """
    material_key = material_name.lower().replace(" ", "_").replace("-", "_")
    
    if material_key in MATERIAL_PROFILES:
        return MATERIAL_PROFILES[material_key]
    else:
        print(f"Warning: Unknown material '{material_name}', using default profile")
        return DEFAULT_PROFILE.copy()

def list_available_materials() -> list:
    """
    Get list of available material names.
    
    Returns:
        List of material names
    """
    return [profile["name"] for profile in MATERIAL_PROFILES.values()]

def get_material_parameters(material_name: str) -> tuple:
    """
    Get key parameters for geometry cleaning.
    
    Args:
        material_name: Name of the material
        
    Returns:
        Tuple of (min_spacing, min_radius, kerf_width, cutting_speed, pierce_time)
    """
    profile = get_material_profile(material_name)
    return (
        profile["min_spacing"],
        profile["min_radius"],
        profile["kerf_width"],
        profile["cutting_speed"],
        profile["pierce_time"]
    )
