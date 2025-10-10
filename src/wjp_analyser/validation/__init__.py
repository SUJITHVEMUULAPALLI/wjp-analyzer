"""
Enhanced DXF Validation Module

This module provides comprehensive DXF validation capabilities specifically designed
for waterjet cutting operations. It includes:

1. Enhanced geometry validation with self-intersection detection
2. Manufacturing-specific validation (kerf compensation, cutting feasibility)
3. File structure validation (layers, blocks, text entities)
4. Performance validation (cutting time estimation, efficiency metrics)
5. Configurable validation rules and thresholds
6. Comprehensive visualization and reporting
7. Easy integration with existing workflow

Main Components:
- EnhancedDXFValidator: Core validation engine
- ValidationVisualizer: Report generation and visualization
- ValidationIntegration: Workflow integration and batch processing
- ValidationConfig: Configuration management
- ValidationResult: Comprehensive result data structure

Usage Examples:

Basic validation:
```python
from wjp_analyser.validation import validate_dxf_with_enhanced_features

result, reports = validate_dxf_with_enhanced_features("file.dxf")
print(f"Validation score: {result.overall_score}")
```

Batch validation:
```python
from wjp_analyser.validation import batch_validate_directory_with_enhanced_features

results = batch_validate_directory_with_enhanced_features("dxf_files/")
for file_path, (result, reports) in results.items():
    print(f"{file_path}: {result.overall_score:.1f}")
```

Custom configuration:
```python
from wjp_analyser.validation import EnhancedValidationManager, ValidationIntegrationConfig, ValidationLevel

config = ValidationIntegrationConfig(
    validation_level=ValidationLevel.COMPREHENSIVE,
    create_reports=True,
    output_dir="my_reports"
)
manager = EnhancedValidationManager(config)
result, reports = manager.validate_dxf_file("file.dxf")
```
"""

from .enhanced_dxf_validator import (
    EnhancedDXFValidator,
    ValidationConfig,
    ValidationLevel,
    ValidationSeverity,
    ValidationRule,
    ValidationResult,
    ValidationIssue,
    create_validation_config,
    validate_dxf_file
)

from .validation_visualizer import (
    ValidationVisualizer,
    create_validation_report
)

from .validation_integration import (
    ValidationIntegrationConfig,
    EnhancedValidationManager,
    validate_dxf_with_enhanced_features,
    batch_validate_directory_with_enhanced_features,
    create_validation_config_from_preset
)

__all__ = [
    # Core validation classes
    'EnhancedDXFValidator',
    'ValidationConfig',
    'ValidationLevel',
    'ValidationSeverity',
    'ValidationRule',
    'ValidationResult',
    'ValidationIssue',
    
    # Visualization
    'ValidationVisualizer',
    'create_validation_report',
    
    # Integration
    'ValidationIntegrationConfig',
    'EnhancedValidationManager',
    
    # Convenience functions
    'validate_dxf_file',
    'validate_dxf_with_enhanced_features',
    'batch_validate_directory_with_enhanced_features',
    'create_validation_config',
    'create_validation_config_from_preset'
]

# Version information
__version__ = "1.0.0"
__author__ = "WJP Analyser Team"
__description__ = "Enhanced DXF validation for waterjet cutting operations"
