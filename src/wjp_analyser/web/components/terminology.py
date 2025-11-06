"""
Terminology Standardization
============================

Standardized terminology mappings to ensure consistent language across all UI pages.
"""

from __future__ import annotations

from typing import Dict

# Standard terminology definitions
TERMINOLOGY = {
    # Core entities
    "object": "Object",  # Individual DXF entity before grouping
    "group": "Group",    # Similar objects clustered together
    "component": "Component",  # Processed/analyzed entity
    "entity": "Entity",  # Generic DXF entity
    
    # Actions
    "analyze": "Analyze",
    "process": "Process",
    "convert": "Convert",
    "export": "Export",
    "download": "Download",
    "upload": "Upload",
    
    # Status
    "operable": "Operable",  # Can be cut
    "non-operable": "Non-operable",  # Cannot be cut
    "selected": "Selected",
    "deselected": "Deselected",
    
    # Metrics
    "area": "Area",
    "perimeter": "Perimeter",
    "length": "Length",
    "pierces": "Pierces",
    "cutting_length": "Cutting Length",
    
    # File types
    "dxf": "DXF",
    "gcode": "G-Code",
    "csv": "CSV",
    "image": "Image",
    
    # Processes
    "analysis": "Analysis",
    "nesting": "Nesting",
    "costing": "Costing",
    "conversion": "Conversion",
}

# Abbreviations
ABBREVIATIONS = {
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "mm²": "mm²",
    "cm²": "cm²",
    "m²": "m²",
}


def standardize_term(term: str) -> str:
    """
    Standardize a terminology term.
    
    Args:
        term: Term to standardize
        
    Returns:
        Standardized term
    """
    term_lower = term.lower().strip()
    
    # Check exact match
    if term_lower in TERMINOLOGY:
        return TERMINOLOGY[term_lower]
    
    # Check partial matches (e.g., "objects" -> "object")
    for key, value in TERMINOLOGY.items():
        if term_lower.startswith(key) or key in term_lower:
            return value
    
    # Return title case if not found
    return term.strip().title()


def standardize_label(label: str) -> str:
    """
    Standardize a UI label.
    
    Args:
        label: Label to standardize
        
    Returns:
        Standardized label
    """
    # Handle common patterns
    replacements = {
        "objects": "Objects",
        "groups": "Groups",
        "components": "Components",
        "entities": "Entities",
        "analyze": "Analyze",
        "process": "Process",
        "export": "Export",
        "download": "Download",
    }
    
    result = label
    for old, new in replacements.items():
        # Case-insensitive replacement
        import re
        result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
    
    return result


def get_label(key: str, default: str = None) -> str:
    """
    Get standardized label for a key.
    
    Args:
        key: Label key
        default: Default value if key not found
        
    Returns:
        Standardized label
    """
    return TERMINOLOGY.get(key.lower(), default or standardize_term(key))


# Common label mappings
LABELS = {
    "total_objects": "Total Objects",
    "total_groups": "Total Groups",
    "total_components": "Total Components",
    "selected_objects": "Selected Objects",
    "operable_objects": "Operable Objects",
    "non_operable_objects": "Non-operable Objects",
    "total_area": "Total Area",
    "cutting_length": "Cutting Length",
    "pierce_count": "Pierce Count",
    "analysis": "Analysis",
    "nesting": "Nesting",
    "costing": "Costing",
    "conversion": "Conversion",
}





