"""
Type Hints Utilities
====================

Common type definitions and utilities for the WJP ANALYSER system.
"""

from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import os

# Common type aliases
FilePath = str
UserId = str
ProjectId = str
AnalysisId = str
ConversionId = str
NestingId = str
TaskId = str

# Configuration types
@dataclass
class AnalysisParams:
    material: str = "steel"
    thickness: float = 6.0
    kerf: float = 1.1
    cutting_speed: float = 1200.0
    cost_per_meter: float = 50.0
    sheet_width: float = 3000.0
    sheet_height: float = 1500.0
    spacing: float = 10.0

@dataclass
class NestingParams:
    sheet_width: float = 3000.0
    sheet_height: float = 1500.0
    spacing: float = 10.0
    algorithm: str = "rectangular"

# Result types
@dataclass
class AnalysisResult:
    success: bool
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class NestingResult:
    success: bool
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

# File validation types
def validate_file_path(file_path: FilePath, must_exist: bool = True) -> bool:
    """Validate file path."""
    if not file_path or not isinstance(file_path, str):
        return False
    
    if ".." in file_path or file_path.startswith("/"):
        return False
    
    if must_exist and not os.path.exists(file_path):
        return False
    
    return True

def validate_user_id(user_id: UserId) -> bool:
    """Validate user ID format."""
    return isinstance(user_id, str) and len(user_id) > 0

def validate_project_id(project_id: ProjectId) -> bool:
    """Validate project ID format."""
    return isinstance(project_id, str) and len(project_id) > 0
