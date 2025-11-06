"""
Enhanced DXF Validation Integration Module

This module integrates the enhanced DXF validation functionality with the existing
workflow manager and provides easy-to-use interfaces for validation.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

from .enhanced_dxf_validator import (
    EnhancedDXFValidator, ValidationConfig, ValidationLevel, 
    ValidationResult, ValidationSeverity, create_validation_config
)
from .validation_visualizer import ValidationVisualizer, create_validation_report

logger = logging.getLogger(__name__)


@dataclass
class ValidationIntegrationConfig:
    """Configuration for validation integration."""
    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    auto_fix_enabled: bool = False
    create_reports: bool = True
    report_format: str = "pdf"  # pdf, json, both
    
    # Integration settings
    integrate_with_workflow: bool = True
    stop_on_critical_errors: bool = True
    warn_on_warnings: bool = True
    
    # Custom validation rules
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Output settings
    output_dir: str = "validation_reports"
    include_visualizations: bool = True


class EnhancedValidationManager:
    """Manager for enhanced DXF validation with workflow integration."""
    
    def __init__(self, config: Optional[ValidationIntegrationConfig] = None):
        """Initialize the validation manager."""
        self.config = config or ValidationIntegrationConfig()
        self.validator = None
        self.visualizer = None
        self._setup_validator()
    
    def _setup_validator(self):
        """Setup the validator with configuration."""
        # Create validation config
        validation_config = ValidationConfig(
            validation_level=self.config.validation_level
        )
        
        # Add custom rules if provided
        if self.config.custom_rules:
            from .enhanced_dxf_validator import ValidationRule
            custom_validation_rules = []
            for rule_dict in self.config.custom_rules:
                rule = ValidationRule(
                    name=rule_dict.get('name', 'custom_rule'),
                    description=rule_dict.get('description', 'Custom validation rule'),
                    severity=ValidationSeverity(rule_dict.get('severity', 'warning')),
                    enabled=rule_dict.get('enabled', True),
                    threshold=rule_dict.get('threshold', 0.0),
                    custom_params=rule_dict.get('custom_params', {})
                )
                custom_validation_rules.append(rule)
            validation_config.custom_rules = custom_validation_rules
        
        self.validator = EnhancedDXFValidator(validation_config)
        
        if self.config.create_reports:
            self.visualizer = ValidationVisualizer(self.config.output_dir)
    
    def validate_dxf_file(self, dxf_path: str) -> Tuple[ValidationResult, Dict[str, str]]:
        """
        Validate a DXF file and create reports.
        
        Args:
            dxf_path: Path to the DXF file to validate
            
        Returns:
            Tuple of (validation_result, report_paths)
        """
        logger.info(f"Starting enhanced validation for: {dxf_path}")
        
        # Perform validation
        validation_result = self.validator.validate_dxf(dxf_path)
        
        # Create reports if enabled
        report_paths = {}
        if self.config.create_reports and self.visualizer:
            try:
                report_paths = create_validation_report(
                    validation_result, 
                    dxf_path, 
                    self.config.output_dir
                )
                logger.info(f"Validation reports created: {list(report_paths.keys())}")
            except Exception as e:
                logger.error(f"Failed to create validation reports: {e}")
                report_paths = {}
        
        # Log validation summary
        self._log_validation_summary(validation_result)
        
        return validation_result, report_paths
    
    def _log_validation_summary(self, result: ValidationResult):
        """Log a summary of validation results."""
        logger.info(f"Validation Summary:")
        logger.info(f"  Overall Score: {result.overall_score:.1f}/100")
        logger.info(f"  Status: {'VALID' if result.is_valid else 'INVALID'}")
        logger.info(f"  Total Entities: {result.total_entities}")
        logger.info(f"  Issues Found: {len(result.issues)}")
        
        if result.issues:
            severity_counts = {severity: 0 for severity in ValidationSeverity}
            for issue in result.issues:
                severity_counts[issue.severity] += 1
            
            logger.info(f"  Issue Breakdown:")
            for severity, count in severity_counts.items():
                if count > 0:
                    logger.info(f"    {severity.value.upper()}: {count}")
        
        logger.info(f"  Estimated Cutting Time: {result.estimated_cutting_time:.1f} min")
        logger.info(f"  Cutting Feasibility: {result.cutting_feasibility_score:.2f}")
        logger.info(f"  Efficiency Score: {result.efficiency_score:.2f}")
    
    def validate_multiple_files(self, dxf_paths: List[str]) -> Dict[str, Tuple[ValidationResult, Dict[str, str]]]:
        """
        Validate multiple DXF files.
        
        Args:
            dxf_paths: List of DXF file paths to validate
            
        Returns:
            Dictionary mapping file paths to (validation_result, report_paths)
        """
        results = {}
        
        for dxf_path in dxf_paths:
            try:
                result, reports = self.validate_dxf_file(dxf_path)
                results[dxf_path] = (result, reports)
            except Exception as e:
                logger.error(f"Failed to validate {dxf_path}: {e}")
                # Create error result
                error_result = ValidationResult(
                    is_valid=False,
                    overall_score=0.0,
                    issues=[ValidationIssue(
                        rule_name="validation_error",
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validation failed: {str(e)}"
                    )]
                )
                results[dxf_path] = (error_result, {})
        
        return results
    
    def batch_validate_directory(self, directory_path: str, 
                                file_pattern: str = "*.dxf") -> Dict[str, Tuple[ValidationResult, Dict[str, str]]]:
        """
        Validate all DXF files in a directory.
        
        Args:
            directory_path: Directory containing DXF files
            file_pattern: File pattern to match (default: *.dxf)
            
        Returns:
            Dictionary mapping file paths to (validation_result, report_paths)
        """
        import glob
        
        dxf_files = glob.glob(os.path.join(directory_path, file_pattern))
        logger.info(f"Found {len(dxf_files)} DXF files in {directory_path}")
        
        return self.validate_multiple_files(dxf_files)
    
    def get_validation_summary(self, results: Dict[str, Tuple[ValidationResult, Dict[str, str]]]) -> Dict[str, Any]:
        """
        Generate a summary of validation results for multiple files.
        
        Args:
            results: Results from validate_multiple_files or batch_validate_directory
            
        Returns:
            Summary dictionary with statistics
        """
        total_files = len(results)
        valid_files = sum(1 for result, _ in results.values() if result.is_valid)
        invalid_files = total_files - valid_files
        
        # Calculate average scores
        scores = [result.overall_score for result, _ in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Count issues by severity
        total_issues = 0
        severity_counts = {severity: 0 for severity in ValidationSeverity}
        
        for result, _ in results.values():
            total_issues += len(result.issues)
            for issue in result.issues:
                severity_counts[issue.severity] += 1
        
        # Calculate total cutting time
        total_cutting_time = sum(result.estimated_cutting_time for result, _ in results.values())
        
        summary = {
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'validation_rate': valid_files / total_files if total_files > 0 else 0,
            'average_score': avg_score,
            'total_issues': total_issues,
            'severity_breakdown': {severity.value: count for severity, count in severity_counts.items()},
            'total_cutting_time': total_cutting_time,
            'files': {}
        }
        
        # Add individual file summaries
        for file_path, (result, reports) in results.items():
            summary['files'][file_path] = {
                'is_valid': result.is_valid,
                'score': result.overall_score,
                'issues_count': len(result.issues),
                'cutting_time': result.estimated_cutting_time,
                'reports': reports
            }
        
        return summary
    
    def export_validation_summary(self, summary: Dict[str, Any], output_path: str) -> str:
        """
        Export validation summary to JSON file.
        
        Args:
            summary: Summary from get_validation_summary
            output_path: Path to save the summary
            
        Returns:
            Path to the saved summary file
        """
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Validation summary exported to: {output_path}")
        return output_path
    
    def create_batch_report(self, results: Dict[str, Tuple[ValidationResult, Dict[str, str]]], 
                           output_path: str) -> str:
        """
        Create a comprehensive batch validation report.
        
        Args:
            results: Results from validate_multiple_files or batch_validate_directory
            output_path: Path to save the batch report
            
        Returns:
            Path to the created report
        """
        summary = self.get_validation_summary(results)
        
        # Create detailed batch report
        report_content = {
            'batch_validation_summary': summary,
            'detailed_results': {}
        }
        
        for file_path, (result, reports) in results.items():
            # Convert result to dictionary for JSON serialization
            result_dict = {
                'is_valid': result.is_valid,
                'overall_score': result.overall_score,
                'total_entities': result.total_entities,
                'polygons': result.polygons,
                'polylines': result.polylines,
                'circles': result.circles,
                'arcs': result.arcs,
                'lines': result.lines,
                'open_contours': result.open_contours,
                'closed_contours': result.closed_contours,
                'self_intersections': result.self_intersections,
                'overlapping_features': result.overlapping_features,
                'nested_contours': result.nested_contours,
                'spacing_violations': result.spacing_violations,
                'radius_violations': result.radius_violations,
                'kerf_conflicts': result.kerf_conflicts,
                'cutting_feasibility_score': result.cutting_feasibility_score,
                'estimated_cutting_time': result.estimated_cutting_time,
                'estimated_cutting_length': result.estimated_cutting_length,
                'estimated_pierce_count': result.estimated_pierce_count,
                'efficiency_score': result.efficiency_score,
                'layer_count': result.layer_count,
                'block_count': result.block_count,
                'unused_layers': result.unused_layers,
                'unused_blocks': result.unused_blocks,
                'bounding_box': result.bounding_box,
                'min_feature_size': result.min_feature_size,
                'max_feature_size': result.max_feature_size,
                'issues': [
                    {
                        'rule_name': issue.rule_name,
                        'severity': issue.severity.value,
                        'message': issue.message,
                        'location': issue.location,
                        'entity_type': issue.entity_type,
                        'entity_id': issue.entity_id,
                        'suggested_fix': issue.suggested_fix,
                        'metadata': issue.metadata
                    }
                    for issue in result.issues
                ],
                'recommendations': result.recommendations,
                'reports': reports
            }
            
            report_content['detailed_results'][file_path] = result_dict
        
        with open(output_path, 'w') as f:
            json.dump(report_content, f, indent=2)
        
        logger.info(f"Batch validation report created: {output_path}")
        return output_path


# Convenience functions for easy integration
def validate_dxf_with_enhanced_features(dxf_path: str, 
                                      validation_level: ValidationLevel = ValidationLevel.STANDARD,
                                      create_reports: bool = True,
                                      output_dir: str = "validation_reports") -> Tuple[ValidationResult, Dict[str, str]]:
    """
    Convenience function to validate a DXF file with enhanced features.
    
    Args:
        dxf_path: Path to the DXF file
        validation_level: Level of validation to perform
        create_reports: Whether to create validation reports
        output_dir: Directory to save reports
        
    Returns:
        Tuple of (validation_result, report_paths)
    """
    config = ValidationIntegrationConfig(
        validation_level=validation_level,
        create_reports=create_reports,
        output_dir=output_dir
    )
    
    manager = EnhancedValidationManager(config)
    return manager.validate_dxf_file(dxf_path)


def batch_validate_directory_with_enhanced_features(directory_path: str,
                                                   validation_level: ValidationLevel = ValidationLevel.STANDARD,
                                                   create_reports: bool = True,
                                                   output_dir: str = "validation_reports") -> Dict[str, Tuple[ValidationResult, Dict[str, str]]]:
    """
    Convenience function to batch validate all DXF files in a directory.
    
    Args:
        directory_path: Directory containing DXF files
        validation_level: Level of validation to perform
        create_reports: Whether to create validation reports
        output_dir: Directory to save reports
        
    Returns:
        Dictionary mapping file paths to (validation_result, report_paths)
    """
    config = ValidationIntegrationConfig(
        validation_level=validation_level,
        create_reports=create_reports,
        output_dir=output_dir
    )
    
    manager = EnhancedValidationManager(config)
    return manager.batch_validate_directory(directory_path)


def create_validation_config_from_preset(preset_name: str) -> ValidationConfig:
    """
    Create a validation configuration from a preset.
    
    Args:
        preset_name: Name of the preset (basic, standard, comprehensive, waterjet_optimized)
        
    Returns:
        ValidationConfig with preset settings
    """
    presets = {
        'basic': {
            'validation_level': ValidationLevel.BASIC,
            'min_feature_size': 0.5,
            'min_spacing': 1.0,
            'kerf_width': 1.1
        },
        'standard': {
            'validation_level': ValidationLevel.STANDARD,
            'min_feature_size': 1.0,
            'min_spacing': 2.0,
            'kerf_width': 1.1,
            'min_corner_radius': 0.5,
            'min_hole_diameter': 2.0
        },
        'comprehensive': {
            'validation_level': ValidationLevel.COMPREHENSIVE,
            'min_feature_size': 0.5,
            'min_spacing': 1.5,
            'kerf_width': 1.1,
            'min_corner_radius': 0.5,
            'min_hole_diameter': 2.0,
            'max_cutting_time_minutes': 120.0,
            'max_pierce_count': 1000,
            'check_layers': True,
            'check_blocks': True,
            'check_text_entities': True,
            'check_dimensions': True
        },
        'waterjet_optimized': {
            'validation_level': ValidationLevel.STANDARD,
            'min_feature_size': 2.0,
            'min_spacing': 3.0,
            'kerf_width': 1.1,
            'min_corner_radius': 1.0,
            'min_hole_diameter': 3.0,
            'max_cutting_time_minutes': 60.0,
            'max_pierce_count': 500
        }
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")
    
    return create_validation_config(**presets[preset_name])
