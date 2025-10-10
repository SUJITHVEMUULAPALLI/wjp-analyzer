"""
Input Validation System for WJP Analyser
========================================

This module provides comprehensive input validation for file uploads,
parameters, and user inputs with security-focused validation.
"""

import os
import re
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
try:
    import magic  # python-magic for file type detection
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None
from PIL import Image
import logging

from .error_handler import ValidationError, raise_validation_error

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Optional[Any] = None
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class FileValidator:
    """File validation with security focus."""
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        'dxf': ['application/dxf', 'image/vnd.dxf'],
        'jpg': ['image/jpeg'],
        'jpeg': ['image/jpeg'],
        'png': ['image/png'],
        'bmp': ['image/bmp'],
        'tiff': ['image/tiff'],
        'tif': ['image/tiff']
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'dxf': 50 * 1024 * 1024,  # 50MB
        'jpg': 10 * 1024 * 1024,  # 10MB
        'jpeg': 10 * 1024 * 1024,  # 10MB
        'png': 10 * 1024 * 1024,  # 10MB
        'bmp': 10 * 1024 * 1024,  # 10MB
        'tiff': 20 * 1024 * 1024,  # 20MB
        'tif': 20 * 1024 * 1024,  # 20MB
    }
    
    # Dangerous file patterns
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'<script',  # Script injection
        r'javascript:',  # JavaScript injection
        r'data:',  # Data URI
        r'vbscript:',  # VBScript
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_filename(self, filename: str) -> ValidationResult:
        """Validate filename for security issues."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not filename:
            result.add_error("Filename cannot be empty")
            return result
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                result.add_error(f"Filename contains dangerous pattern: {pattern}")
        
        # Check filename length
        if len(filename) > 255:
            result.add_error("Filename too long (maximum 255 characters)")
        
        # Check for valid characters (alphanumeric, dots, dashes, underscores)
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            result.add_error("Filename contains invalid characters")
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        name_without_ext = Path(filename).stem.upper()
        if name_without_ext in reserved_names:
            result.add_error(f"Filename uses reserved name: {name_without_ext}")
        
        return result
    
    def validate_file_extension(self, filename: str) -> ValidationResult:
        """Validate file extension."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not filename:
            result.add_error("Filename cannot be empty")
            return result
        
        # Get file extension
        ext = Path(filename).suffix.lower().lstrip('.')
        
        if not ext:
            result.add_error("File must have an extension")
            return result
        
        if ext not in self.ALLOWED_EXTENSIONS:
            allowed = ', '.join(sorted(self.ALLOWED_EXTENSIONS.keys()))
            result.add_error(f"File extension '{ext}' not allowed. Allowed: {allowed}")
        
        return result
    
    def validate_file_size(self, file_path: str, max_size: Optional[int] = None) -> ValidationResult:
        """Validate file size."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            file_size = os.path.getsize(file_path)
            
            # Determine max size based on extension
            if max_size is None:
                ext = Path(file_path).suffix.lower().lstrip('.')
                max_size = self.MAX_FILE_SIZES.get(ext, 10 * 1024 * 1024)  # Default 10MB
            
            if file_size > max_size:
                result.add_error(f"File too large: {file_size} bytes (max: {max_size} bytes)")
            
            if file_size == 0:
                result.add_error("File is empty")
            
        except OSError as e:
            result.add_error(f"Cannot determine file size: {e}")
        
        return result
    
    def validate_file_type(self, file_path: str) -> ValidationResult:
        """Validate file type using magic bytes."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not MAGIC_AVAILABLE:
            result.add_warning("File type detection not available (python-magic not installed)")
            return result
        
        try:
            # Use python-magic to detect file type
            file_type = magic.from_file(file_path, mime=True)
            
            # Get expected extension
            ext = Path(file_path).suffix.lower().lstrip('.')
            
            if ext not in self.ALLOWED_EXTENSIONS:
                result.add_error(f"File extension '{ext}' not allowed")
                return result
            
            # Check if detected MIME type matches expected
            expected_types = self.ALLOWED_EXTENSIONS.get(ext, [])
            
            if expected_types and file_type not in expected_types:
                # For DXF files, be more lenient as MIME detection can be inconsistent
                if ext == 'dxf' and 'application' in file_type:
                    result.add_warning(f"File type detection uncertain for DXF file: {file_type}")
                else:
                    result.add_error(f"File type mismatch. Expected: {expected_types}, Got: {file_type}")
            
        except Exception as e:
            result.add_warning(f"Could not detect file type: {e}")
        
        return result
    
    def validate_image_file(self, file_path: str) -> ValidationResult:
        """Validate image file specifically."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            with Image.open(file_path) as img:
                # Check image dimensions
                width, height = img.size
                
                if width > 8192 or height > 8192:
                    result.add_warning(f"Image very large: {width}x{height} pixels")
                
                if width < 10 or height < 10:
                    result.add_error("Image too small (minimum 10x10 pixels)")
                
                # Check image format
                if img.format not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
                    result.add_error(f"Unsupported image format: {img.format}")
                
                # Check for corruption
                img.verify()
                
        except Exception as e:
            result.add_error(f"Invalid image file: {e}")
        
        return result
    
    def validate_dxf_file(self, file_path: str) -> ValidationResult:
        """Validate DXF file specifically."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            # Basic DXF file validation
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1000 characters
                
                if not content.startswith('0'):
                    result.add_warning("DXF file may not start with proper header")
                
                # Check for basic DXF structure
                if 'SECTION' not in content:
                    result.add_warning("DXF file may be missing sections")
                
        except Exception as e:
            result.add_error(f"Cannot read DXF file: {e}")
        
        return result
    
    def validate_file(self, file_path: str, filename: Optional[str] = None) -> ValidationResult:
        """Comprehensive file validation."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Use provided filename or extract from path
        if filename is None:
            filename = Path(file_path).name
        
        # Validate filename
        filename_result = self.validate_filename(filename)
        result.errors.extend(filename_result.errors)
        result.warnings.extend(filename_result.warnings)
        if not filename_result.is_valid:
            result.is_valid = False
        
        # Validate file extension
        ext_result = self.validate_file_extension(filename)
        result.errors.extend(ext_result.errors)
        result.warnings.extend(ext_result.warnings)
        if not ext_result.is_valid:
            result.is_valid = False
        
        # Check if file exists
        if not os.path.exists(file_path):
            result.add_error("File does not exist")
            return result
        
        # Validate file size
        size_result = self.validate_file_size(file_path)
        result.errors.extend(size_result.errors)
        result.warnings.extend(size_result.warnings)
        if not size_result.is_valid:
            result.is_valid = False
        
        # Validate file type
        type_result = self.validate_file_type(file_path)
        result.errors.extend(type_result.errors)
        result.warnings.extend(type_result.warnings)
        if not type_result.is_valid:
            result.is_valid = False
        
        # File-specific validation
        ext = Path(filename).suffix.lower().lstrip('.')
        
        if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:
            img_result = self.validate_image_file(file_path)
            result.errors.extend(img_result.errors)
            result.warnings.extend(img_result.warnings)
            if not img_result.is_valid:
                result.is_valid = False
        
        elif ext == 'dxf':
            dxf_result = self.validate_dxf_file(file_path)
            result.errors.extend(dxf_result.errors)
            result.warnings.extend(dxf_result.warnings)
            if not dxf_result.is_valid:
                result.is_valid = False
        
        return result


class ParameterValidator:
    """Parameter validation for application inputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_numeric_parameter(self, value: Any, param_name: str, 
                                 min_val: Optional[float] = None, 
                                 max_val: Optional[float] = None,
                                 required: bool = True) -> ValidationResult:
        """Validate numeric parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if value is None:
            if required:
                result.add_error(f"Parameter '{param_name}' is required")
            return result
        
        try:
            num_value = float(value)
            
            if min_val is not None and num_value < min_val:
                result.add_error(f"Parameter '{param_name}' must be >= {min_val}")
            
            if max_val is not None and num_value > max_val:
                result.add_error(f"Parameter '{param_name}' must be <= {max_val}")
            
            result.sanitized_value = num_value
            
        except (ValueError, TypeError):
            result.add_error(f"Parameter '{param_name}' must be a number")
        
        return result
    
    def validate_string_parameter(self, value: Any, param_name: str,
                                 max_length: Optional[int] = None,
                                 allowed_chars: Optional[str] = None,
                                 required: bool = True) -> ValidationResult:
        """Validate string parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if value is None:
            if required:
                result.add_error(f"Parameter '{param_name}' is required")
            return result
        
        str_value = str(value).strip()
        
        if not str_value and required:
            result.add_error(f"Parameter '{param_name}' cannot be empty")
            return result
        
        if max_length and len(str_value) > max_length:
            result.add_error(f"Parameter '{param_name}' too long (max {max_length} characters)")
        
        if allowed_chars and not re.match(f"^[{re.escape(allowed_chars)}]+$", str_value):
            result.add_error(f"Parameter '{param_name}' contains invalid characters")
        
        result.sanitized_value = str_value
        return result
    
    def validate_file_path(self, path: str, param_name: str = "file_path",
                          must_exist: bool = True) -> ValidationResult:
        """Validate file paths."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not path:
            result.add_error(f"Parameter '{param_name}' cannot be empty")
            return result
        
        # Check for dangerous patterns
        dangerous_patterns = [r'\.\./', r'\.\.\\', r'[<>:"|?*]']
        for pattern in dangerous_patterns:
            if re.search(pattern, path):
                result.add_error(f"Parameter '{param_name}' contains dangerous characters")
        
        if must_exist and not os.path.exists(path):
            result.add_error(f"File path '{param_name}' does not exist")
        
        result.sanitized_value = os.path.normpath(path)
        return result
    
    def validate_material_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate material-related parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Validate material type
        material = params.get('material', '')
        if not material:
            result.add_error("Material type is required")
        elif material not in ['steel', 'aluminum', 'stainless_steel', 'titanium', 'brass', 'copper']:
            result.add_warning(f"Unknown material type: {material}")
        
        # Validate thickness
        thickness_result = self.validate_numeric_parameter(
            params.get('thickness'), 'thickness', min_val=0.1, max_val=100.0
        )
        result.errors.extend(thickness_result.errors)
        result.warnings.extend(thickness_result.warnings)
        if not thickness_result.is_valid:
            result.is_valid = False
        
        # Validate kerf
        kerf_result = self.validate_numeric_parameter(
            params.get('kerf'), 'kerf', min_val=0.05, max_val=2.0
        )
        result.errors.extend(kerf_result.errors)
        result.warnings.extend(kerf_result.warnings)
        if not kerf_result.is_valid:
            result.is_valid = False
        
        # Validate cutting speed
        speed_result = self.validate_numeric_parameter(
            params.get('cutting_speed'), 'cutting_speed', min_val=100, max_val=3000
        )
        result.errors.extend(speed_result.errors)
        result.warnings.extend(speed_result.warnings)
        if not speed_result.is_valid:
            result.is_valid = False
        
        return result
    
    def validate_image_processing_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate image processing parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Validate edge threshold
        threshold_result = self.validate_numeric_parameter(
            params.get('edge_threshold'), 'edge_threshold', min_val=0.0, max_val=1.0
        )
        result.errors.extend(threshold_result.errors)
        result.warnings.extend(threshold_result.warnings)
        if not threshold_result.is_valid:
            result.is_valid = False
        
        # Validate contour area
        area_result = self.validate_numeric_parameter(
            params.get('min_contour_area'), 'min_contour_area', min_val=1, max_val=10000
        )
        result.errors.extend(area_result.errors)
        result.warnings.extend(area_result.warnings)
        if not area_result.is_valid:
            result.is_valid = False
        
        # Validate simplification tolerance
        tolerance_result = self.validate_numeric_parameter(
            params.get('simplify_tolerance'), 'simplify_tolerance', min_val=0.0, max_val=1.0
        )
        result.errors.extend(tolerance_result.errors)
        result.warnings.extend(tolerance_result.warnings)
        if not tolerance_result.is_valid:
            result.is_valid = False
        
        return result


# Global validator instances
file_validator = FileValidator()
parameter_validator = ParameterValidator()


# Convenience functions
def validate_uploaded_file(file_path: str, filename: str) -> ValidationResult:
    """Validate uploaded file."""
    return file_validator.validate_file(file_path, filename)


def validate_material_params(params: Dict[str, Any]) -> ValidationResult:
    """Validate material parameters."""
    return parameter_validator.validate_material_parameters(params)


def validate_image_params(params: Dict[str, Any]) -> ValidationResult:
    """Validate image processing parameters."""
    return parameter_validator.validate_image_processing_parameters(params)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)
    
    # Remove path traversal patterns
    sanitized = re.sub(r'\.\./', '', sanitized)
    sanitized = re.sub(r'\.\.\\', '', sanitized)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized or "unnamed_file"
