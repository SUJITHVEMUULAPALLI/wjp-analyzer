"""
Constants module for WJP ANALYSER
================================

Centralized location for all constant values used throughout the application.
"""

from enum import Enum

# Material code mappings
MATERIAL_CODES = {
    "Tan Brown Granite": "TANB",
    "Marble": "MARB",
    "Stainless Steel": "STST",
    "Aluminum": "ALUM",
    "Brass": "BRAS",
    "Generic": "GENE"
}

# Process stages
class ProcessStage(Enum):
    DESIGN = "design"
    ANALYSIS = "analysis"
    CONVERSION = "conversion"
    NESTING = "nesting"
    MACHINING = "machining"

# File stage folders
STAGE_FOLDERS = {
    ProcessStage.DESIGN: "designer",
    ProcessStage.ANALYSIS: "analyzed",
    ProcessStage.CONVERSION: "converted_dxf",
    ProcessStage.NESTING: "nested",
    ProcessStage.MACHINING: "machining"
}

# Default file extensions
FILE_EXTENSIONS = {
    "design": "png",
    "analysis": "dxf",
    "report": "pdf",
    "data": "json",
    "raw": "dxf"
}

# Default version string
DEFAULT_VERSION = "V1"

# Analysis metrics thresholds
ANALYSIS_THRESHOLDS = {
    "min_cut_length": 1.0,  # mm
    "max_cut_length": 10000.0,  # mm
    "min_object_size": 1.0,  # mm²
    "max_complexity": 100,
    "max_violations": 10
}