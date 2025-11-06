"""
WJP Image Analyzer Module

This module provides intelligent pre-processing analysis for images
before DXF conversion, ensuring waterjet cutting suitability.

Phase 1: Diagnostic Core - analyzes images and provides suitability scores
Phase 2: Visualization Layer - Streamlit UI for interactive analysis
Phase 3: Integration Hook - connects to image-to-DXF pipeline
"""

from .core import analyze_image_for_wjp, AnalyzerConfig
from .integration import ImageAnalyzerGate, create_analyzer_gate, quick_analyze
from .cli import main as cli_main

__all__ = [
    'analyze_image_for_wjp', 
    'AnalyzerConfig',
    'ImageAnalyzerGate',
    'create_analyzer_gate', 
    'quick_analyze',
    'cli_main'
]
