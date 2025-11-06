# Enhanced DXF Validation System

## Overview

The Enhanced DXF Validation System is a comprehensive validation framework specifically designed for waterjet cutting operations. It provides advanced geometry analysis, manufacturing-specific validation, performance optimization, and detailed reporting capabilities.

## Features

### 🔍 **Comprehensive Validation Levels**

- **Basic**: Essential checks only (entity counting, basic geometry)
- **Standard**: Common manufacturing checks (spacing, radii, cutting feasibility)
- **Comprehensive**: All validation checks including file structure analysis
- **Custom**: User-defined validation rules and thresholds

### 🏗️ **Advanced Geometry Validation**

- **Self-intersection Detection**: Identifies overlapping or intersecting geometry
- **Overlapping Features**: Detects features that are too close together
- **Nested Contours**: Analyzes contour relationships and nesting
- **Open/Closed Contour Analysis**: Validates contour completeness
- **Feature Size Analysis**: Checks minimum and maximum feature dimensions
- **Corner Radius Validation**: Ensures adequate corner radii for cutting

### ⚙️ **Manufacturing-Specific Validation**

- **Kerf Compensation Analysis**: Validates spacing considering kerf width
- **Cutting Feasibility Scoring**: Rates how well-suited the geometry is for cutting
- **Material Constraints**: Validates against material-specific limitations
- **Minimum Hole Diameter**: Ensures holes are large enough for cutting
- **Spacing Violations**: Checks minimum spacing between features
- **Radius Violations**: Validates corner and hole radii

### 📊 **Performance Analysis**

- **Cutting Time Estimation**: Calculates estimated cutting time
- **Cutting Length Calculation**: Measures total cutting path length
- **Pierce Count Analysis**: Counts required pierce operations
- **Efficiency Scoring**: Rates cutting efficiency
- **Performance Thresholds**: Validates against time and complexity limits

### 📁 **File Structure Validation**

- **Layer Analysis**: Checks layer usage and identifies unused layers
- **Block Analysis**: Analyzes block definitions and usage
- **Text Entity Detection**: Identifies non-cutting text elements
- **Dimension Entity Detection**: Finds dimensioning elements
- **File Cleanup Recommendations**: Suggests structural improvements

### 📈 **Advanced Visualization & Reporting**

- **Comprehensive PDF Reports**: Multi-page detailed analysis reports
- **Interactive Charts**: Visual representation of validation results
- **DXF Preview**: Visual preview with issue highlighting
- **JSON Export**: Machine-readable validation results
- **Summary Charts**: Quick overview visualizations
- **Batch Processing Reports**: Multi-file analysis summaries

## Installation & Setup

### Prerequisites

```bash
pip install ezdxf shapely matplotlib seaborn numpy
```

### Basic Usage

```python
from wjp_analyser.validation import validate_dxf_with_enhanced_features

# Validate a single DXF file
result, reports = validate_dxf_with_enhanced_features("file.dxf")
print(f"Validation score: {result.overall_score:.1f}")
```

### Batch Processing

```python
from wjp_analyser.validation import batch_validate_directory_with_enhanced_features

# Validate all DXF files in a directory
results = batch_validate_directory_with_enhanced_features("dxf_files/")
for file_path, (result, reports) in results.items():
    print(f"{file_path}: {result.overall_score:.1f}")
```

## Configuration

### Validation Levels

```python
from wjp_analyser.validation import ValidationLevel, ValidationIntegrationConfig

# Configure validation level
config = ValidationIntegrationConfig(
    validation_level=ValidationLevel.COMPREHENSIVE,
    create_reports=True,
    output_dir="validation_reports"
)
```

### Custom Rules

```python
from wjp_analyser.validation import EnhancedValidationManager, ValidationIntegrationConfig

config = ValidationIntegrationConfig(
    custom_rules=[
        {
            'name': 'custom_min_size',
            'description': 'Custom minimum feature size check',
            'severity': 'warning',
            'enabled': True,
            'threshold': 5.0
        }
    ]
)

manager = EnhancedValidationManager(config)
result, reports = manager.validate_dxf_file("file.dxf")
```

### Preset Configurations

```python
from wjp_analyser.validation import create_validation_config_from_preset

# Use predefined presets
config = create_validation_config_from_preset('waterjet_optimized')
```

Available presets:
- `basic`: Essential checks only
- `standard`: Common manufacturing checks
- `comprehensive`: All validation checks
- `waterjet_optimized`: Optimized for waterjet cutting

## API Reference

### Core Classes

#### `EnhancedDXFValidator`

Main validation engine class.

```python
from wjp_analyser.validation import EnhancedDXFValidator, ValidationConfig

validator = EnhancedDXFValidator(ValidationConfig())
result = validator.validate_dxf("file.dxf")
```

#### `ValidationResult`

Comprehensive validation result data structure.

```python
# Key properties
result.is_valid          # Boolean validity status
result.overall_score     # Overall score (0-100)
result.total_entities    # Total number of entities
result.issues           # List of ValidationIssue objects
result.recommendations  # List of improvement suggestions
```

#### `ValidationIssue`

Individual validation issue.

```python
# Key properties
issue.rule_name        # Name of the validation rule
issue.severity         # Severity level (INFO, WARNING, ERROR, CRITICAL)
issue.message          # Human-readable issue description
issue.suggested_fix    # Suggested solution
issue.location         # Optional location coordinates
```

### Validation Levels

#### `ValidationLevel.BASIC`
- Entity counting
- Basic geometry analysis
- File structure validation

#### `ValidationLevel.STANDARD`
- All basic checks
- Manufacturing-specific validation
- Performance analysis
- Spacing and radius checks

#### `ValidationLevel.COMPREHENSIVE`
- All standard checks
- File structure analysis
- Layer and block analysis
- Text and dimension detection

### Severity Levels

#### `ValidationSeverity.INFO`
Informational messages that don't affect processing.

#### `ValidationSeverity.WARNING`
Issues that may cause problems but don't prevent processing.

#### `ValidationSeverity.ERROR`
Issues that will cause problems and should be addressed.

#### `ValidationSeverity.CRITICAL`
Critical issues that prevent processing.

## Validation Rules

### Geometry Rules

- **min_feature_size**: Minimum feature dimension
- **max_feature_size**: Maximum feature dimension
- **min_corner_radius**: Minimum corner radius
- **self_intersections**: Self-intersecting geometry detection
- **open_contours**: Open contour detection

### Manufacturing Rules

- **min_spacing**: Minimum spacing between features
- **min_hole_diameter**: Minimum hole diameter
- **kerf_width**: Kerf compensation validation
- **cutting_feasibility**: Overall cutting feasibility

### Performance Rules

- **cutting_time**: Maximum cutting time threshold
- **pierce_count**: Maximum pierce count threshold
- **efficiency**: Minimum efficiency threshold

## Report Formats

### PDF Reports

Comprehensive multi-page reports including:
- Title page with summary
- Entity distribution charts
- Issue analysis
- Geometry analysis
- Manufacturing analysis
- Performance analysis
- DXF preview with issue highlighting
- Recommendations

### JSON Reports

Machine-readable validation results for integration with other systems.

### Summary Charts

Quick visual overviews of validation results.

## Integration

### Workflow Integration

The enhanced validation system integrates seamlessly with the existing workflow manager:

```python
# Automatic integration in workflow
from wjp_analyser.workflow import WorkflowManager

workflow = WorkflowManager()
result = workflow.process_file("file.dxf")  # Uses enhanced validation automatically
```

### Custom Integration

```python
from wjp_analyser.validation import EnhancedValidationManager, ValidationIntegrationConfig

config = ValidationIntegrationConfig(
    validation_level=ValidationLevel.STANDARD,
    integrate_with_workflow=True,
    stop_on_critical_errors=True
)

manager = EnhancedValidationManager(config)
result, reports = manager.validate_dxf_file("file.dxf")
```

## Examples

### Basic Validation

```python
from wjp_analyser.validation import validate_dxf_with_enhanced_features

result, reports = validate_dxf_with_enhanced_features("sample.dxf")

print(f"Score: {result.overall_score:.1f}")
print(f"Valid: {result.is_valid}")
print(f"Issues: {len(result.issues)}")

for issue in result.issues:
    print(f"- {issue.severity.value}: {issue.message}")
```

### Advanced Configuration

```python
from wjp_analyser.validation import (
    EnhancedValidationManager, 
    ValidationIntegrationConfig, 
    ValidationLevel
)

config = ValidationIntegrationConfig(
    validation_level=ValidationLevel.COMPREHENSIVE,
    create_reports=True,
    output_dir="custom_reports",
    custom_rules=[
        {
            'name': 'custom_spacing',
            'description': 'Custom spacing check',
            'severity': 'error',
            'threshold': 5.0
        }
    ]
)

manager = EnhancedValidationManager(config)
result, reports = manager.validate_dxf_file("complex_file.dxf")
```

### Batch Processing

```python
from wjp_analyser.validation import batch_validate_directory_with_enhanced_features

results = batch_validate_directory_with_enhanced_features("dxf_files/")

# Get summary
from wjp_analyser.validation import EnhancedValidationManager
manager = EnhancedValidationManager()
summary = manager.get_validation_summary(results)

print(f"Total files: {summary['total_files']}")
print(f"Valid files: {summary['valid_files']}")
print(f"Average score: {summary['average_score']:.1f}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Check file paths and permissions
3. **Memory Issues**: For large files, use BASIC validation level
4. **Report Generation**: Ensure output directory is writable

### Performance Tips

1. Use BASIC validation level for quick checks
2. Use STANDARD level for most applications
3. Use COMPREHENSIVE level only when detailed analysis is needed
4. Disable report generation for batch processing if not needed

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

The enhanced validation system is designed to be extensible. To add new validation rules:

1. Create a new validation rule in `ValidationRule`
2. Implement the validation logic in `EnhancedDXFValidator`
3. Add appropriate tests
4. Update documentation

## License

This enhanced validation system is part of the WJP Analyser project and follows the same licensing terms.
