"""
WJP Image Analyzer Integration Hook

This module provides integration hooks for the existing image-to-DXF pipeline.
It acts as a pre-processing gate that analyzes images before conversion.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict

from .core import analyze_image_for_wjp, AnalyzerConfig


logger = logging.getLogger(__name__)


class ImageAnalyzerGate:
    """
    Integration gate for image analysis before DXF conversion.
    
    This class provides methods to:
    1. Analyze images for DXF conversion suitability
    2. Generate reports and save them
    3. Determine if an image should proceed to conversion
    4. Provide actionable feedback
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None, 
                 min_score_threshold: float = 75.0,
                 auto_fix_enabled: bool = False):
        """
        Initialize the analyzer gate.
        
        Args:
            config: Analyzer configuration (uses default if None)
            min_score_threshold: Minimum score required to proceed (0-100)
            auto_fix_enabled: Whether to attempt automatic fixes
        """
        self.config = config or AnalyzerConfig()
        self.min_score_threshold = min_score_threshold
        self.auto_fix_enabled = auto_fix_enabled
        
    def analyze_and_decide(self, image_path: str, 
                          output_dir: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze an image and decide whether it should proceed to DXF conversion.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save analysis report (optional)
            
        Returns:
            Tuple of (should_proceed, analysis_report)
        """
        try:
            # Perform analysis
            report = analyze_image_for_wjp(image_path, self.config)
            
            # Save report if output directory specified
            if output_dir:
                self._save_report(report, output_dir)
            
            # Determine if should proceed
            should_proceed = self._should_proceed(report)
            
            # Add decision metadata
            report['gate_decision'] = {
                'should_proceed': should_proceed,
                'min_threshold': self.min_score_threshold,
                'reason': self._get_decision_reason(report)
            }
            
            logger.info(f"Image analysis complete: {image_path} - Score: {report['score']}, Proceed: {should_proceed}")
            
            return should_proceed, report
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            error_report = {
                'file': image_path,
                'error': str(e),
                'gate_decision': {
                    'should_proceed': False,
                    'reason': f"Analysis failed: {e}"
                }
            }
            return False, error_report
    
    def _should_proceed(self, report: Dict[str, Any]) -> bool:
        """Determine if image should proceed based on analysis."""
        score = report.get('score', 0)
        
        # Check score threshold
        if score < self.min_score_threshold:
            return False
            
        # Check for critical failures
        flags = report.get('flags', {})
        critical_failures = [
            'tight_spacing',  # Could cause cutting issues
            'closed_ratio_low'  # Open contours problematic
        ]
        
        for flag in critical_failures:
            if flags.get(flag, False):
                return False
                
        return True
    
    def _get_decision_reason(self, report: Dict[str, Any]) -> str:
        """Get human-readable reason for the decision."""
        score = report.get('score', 0)
        
        if score >= self.min_score_threshold:
            return f"Score {score} meets threshold {self.min_score_threshold}"
        else:
            return f"Score {score} below threshold {self.min_score_threshold}"
    
    def _save_report(self, report: Dict[str, Any], output_dir: str) -> str:
        """Save analysis report to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on input image
        input_path = Path(report['file'])
        report_filename = f"{input_path.stem}_analysis_{hash(input_path.name) % 10000:04d}.json"
        report_path = output_path / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Analysis report saved to: {report_path}")
        return str(report_path)
    
    def get_suggestions_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Get a structured summary of suggestions."""
        suggestions = report.get('suggestions', [])
        flags = report.get('flags', {})
        
        # Categorize suggestions
        categories = {
            'contrast': [],
            'geometry': [],
            'orientation': [],
            'manufacturing': [],
            'other': []
        }
        
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if any(word in suggestion_lower for word in ['contrast', 'sharpen', 'flat', 'busy']):
                categories['contrast'].append(suggestion)
            elif any(word in suggestion_lower for word in ['contour', 'shape', 'watertight']):
                categories['geometry'].append(suggestion)
            elif any(word in suggestion_lower for word in ['rotate', 'deskew']):
                categories['orientation'].append(suggestion)
            elif any(word in suggestion_lower for word in ['gap', 'spacing', 'radius', 'fillet']):
                categories['manufacturing'].append(suggestion)
            else:
                categories['other'].append(suggestion)
        
        return {
            'total_suggestions': len(suggestions),
            'categories': categories,
            'critical_issues': [flag for flag, value in flags.items() if value and flag in ['tight_spacing', 'closed_ratio_low']]
        }


def create_analyzer_gate(config_dict: Optional[Dict[str, Any]] = None,
                        min_score: float = 75.0) -> ImageAnalyzerGate:
    """
    Factory function to create an analyzer gate with custom configuration.
    
    Args:
        config_dict: Dictionary with configuration overrides
        min_score: Minimum score threshold for proceeding
        
    Returns:
        Configured ImageAnalyzerGate instance
    """
    config = AnalyzerConfig()
    
    if config_dict:
        # Override config values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return ImageAnalyzerGate(config=config, min_score_threshold=min_score)


# Convenience function for direct integration
def quick_analyze(image_path: str, min_score: float = 75.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Quick analysis function for simple integration.
    
    Args:
        image_path: Path to image file
        min_score: Minimum score threshold
        
    Returns:
        Tuple of (should_proceed, analysis_report)
    """
    gate = create_analyzer_gate(min_score=min_score)
    return gate.analyze_and_decide(image_path)

