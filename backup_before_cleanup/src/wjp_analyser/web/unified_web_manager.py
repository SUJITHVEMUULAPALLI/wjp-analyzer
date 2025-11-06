#!/usr/bin/env python3
"""
Unified Web Interface Manager
============================

This module provides a unified approach to managing all web interfaces in the WJP ANALYSER system.
It consolidates the different web interface approaches (Flask, Streamlit) and provides consistent
import handling, error management, and configuration.

Key Features:
- Unified import management with proper fallbacks
- Consistent error handling across all interfaces
- Standardized configuration access
- Common utility functions
- Proper dependency management
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WebInterfaceConfig:
    """Configuration for web interfaces."""
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 5000
    upload_folder: str = "output/temp"
    output_folder: str = "output"
    log_level: str = "INFO"
    max_upload_size: int = 32 * 1024 * 1024  # 32MB
    allowed_extensions: set = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {"dxf", "jpg", "jpeg", "png", "bmp", "tiff"}


class UnifiedWebManager:
    """
    Unified manager for all web interfaces in the WJP ANALYSER system.
    
    This class provides consistent access to:
    - Configuration management
    - Import handling with fallbacks
    - Error management
    - Common utilities
    """
    
    def __init__(self, config: Optional[WebInterfaceConfig] = None):
        """Initialize the unified web manager."""
        self.config = config or WebInterfaceConfig()
        self._import_cache: Dict[str, Any] = {}
        self._setup_paths()
        
    def _setup_paths(self):
        """Setup import paths for the web interfaces."""
        # Add src directory to path
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            
        # Add project root to path
        project_root = src_dir.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    
    def safe_import(self, module_name: str, fallback_value: Any = None) -> Any:
        """
        Safely import a module with fallback handling.
        
        Args:
            module_name: Name of the module to import
            fallback_value: Value to return if import fails
            
        Returns:
            Imported module or fallback value
        """
        if module_name in self._import_cache:
            return self._import_cache[module_name]
            
        try:
            module = __import__(module_name, fromlist=[''])
            self._import_cache[module_name] = module
            logger.info(f"Successfully imported {module_name}")
            return module
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")
            self._import_cache[module_name] = fallback_value
            return fallback_value
    
    
    def get_dxf_analyzer(self):
        """Get the DXF analyzer."""
        dxf_analyzer_module = self.safe_import("wjp_analyser.analysis.dxf_analyzer")
        if dxf_analyzer_module:
            return getattr(dxf_analyzer_module, 'analyze_dxf', None)
        return None
    
    def get_analyze_args(self):
        """Get the AnalyzeArgs class."""
        dxf_analyzer_module = self.safe_import("wjp_analyser.analysis.dxf_analyzer")
        if dxf_analyzer_module:
            return getattr(dxf_analyzer_module, 'AnalyzeArgs', None)
        return None
    
    def get_interactive_components(self) -> Dict[str, Any]:
        """Get interactive editing components."""
        components = {}
        
        
        return components
    
    def get_ai_components(self) -> Dict[str, Any]:
        """Get AI-related components."""
        components = {}
        
        # OpenAI client
        openai_module = self.safe_import("wjp_analyser.ai.openai_client")
        if openai_module:
            components['OpenAIAnalyzer'] = getattr(openai_module, 'OpenAIAnalyzer', None)
            components['OpenAIConfig'] = getattr(openai_module, 'OpenAIConfig', None)
        
        # Ollama client
        ollama_module = self.safe_import("wjp_analyser.ai.ollama_client")
        if ollama_module:
            components['OllamaAnalyzer'] = getattr(ollama_module, 'OllamaAnalyzer', None)
            components['OllamaConfig'] = getattr(ollama_module, 'OllamaConfig', None)
        
        return components
    
    def get_agents_interface(self):
        """Get OpenAI agents interface."""
        try:
            from .pages.openai_agents import render_agents_interface
            return render_agents_interface
        except ImportError as e:
            logger.warning(f"Failed to import agents interface: {e}")
            return None
    
    def get_agents_manager(self):
        """Get OpenAI agents manager."""
        try:
            from ..ai.openai_agents_manager import get_agents_manager
            return get_agents_manager()
        except ImportError as e:
            logger.warning(f"Failed to import agents manager: {e}")
            return None
    
    def get_config_components(self) -> Dict[str, Any]:
        """Get configuration components."""
        components = {}
        
        # Secure config
        secure_config_module = self.safe_import("wjp_analyser.config.secure_config")
        if secure_config_module:
            components['get_security_config'] = getattr(secure_config_module, 'get_security_config', None)
            components['get_ai_config'] = getattr(secure_config_module, 'get_ai_config', None)
            components['get_app_config'] = getattr(secure_config_module, 'get_app_config', None)
            components['validate_config'] = getattr(secure_config_module, 'validate_config', None)
        
        return components
    
    def get_utility_components(self) -> Dict[str, Any]:
        """Get utility components."""
        components = {}
        
        # Error handler
        error_handler_module = self.safe_import("wjp_analyser.utils.error_handler")
        if error_handler_module:
            components['handle_errors'] = getattr(error_handler_module, 'handle_errors', None)
            components['error_handler'] = getattr(error_handler_module, 'error_handler', None)
        
        # Input validator
        input_validator_module = self.safe_import("wjp_analyser.utils.input_validator")
        if input_validator_module:
            components['validate_uploaded_file'] = getattr(input_validator_module, 'validate_uploaded_file', None)
            components['validate_material_params'] = getattr(input_validator_module, 'validate_material_params', None)
            components['validate_image_params'] = getattr(input_validator_module, 'validate_image_params', None)
        
        # Logging config
        logging_config_module = self.safe_import("wjp_analyser.utils.logging_config")
        if logging_config_module:
            components['initialize_logging'] = getattr(logging_config_module, 'initialize_logging', None)
            components['log_startup'] = getattr(logging_config_module, 'log_startup', None)
            components['log_shutdown'] = getattr(logging_config_module, 'log_shutdown', None)
        
        # Cache manager
        cache_manager_module = self.safe_import("wjp_analyser.utils.cache_manager")
        if cache_manager_module:
            components['initialize_cache'] = getattr(cache_manager_module, 'initialize_cache', None)
        
        return components
    
    
    def create_analyze_args(self, **kwargs) -> Any:
        """Create analyze arguments with defaults."""
        AnalyzeArgs = self.get_analyze_args()
        if AnalyzeArgs is None:
            # Fallback to basic dict
            defaults = {
                "material": "Generic Material",
                "thickness": 10.0,
                "kerf": 1.0,
                "rate_per_m": 800.0,
                "pierce_cost": 5.0,
                "out": "out"
            }
            defaults.update(kwargs)
            return defaults
        
        return AnalyzeArgs(**kwargs)
    
    def validate_file_upload(self, filename: str) -> bool:
        """Validate uploaded file."""
        if not filename:
            return False
            
        # Check extension
        ext = Path(filename).suffix.lower().lstrip('.')
        return ext in self.config.allowed_extensions
    
    def get_upload_path(self, filename: str) -> Path:
        """Get upload path for a file."""
        upload_dir = Path(self.config.upload_folder)
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """Get output path for a file."""
        output_dir = Path(self.config.output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    
    def log_web_event(self, event_type: str, details: Dict[str, Any]):
        """Log web interface events."""
        logger.info(f"Web Event - {event_type}: {details}")
    
    def handle_web_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle web interface errors consistently."""
        error_details = {
            "error": str(error),
            "context": context,
            "type": type(error).__name__
        }
        
        logger.error(f"Web Error - {context}: {error}")
        return error_details


# Global web manager instance
web_manager = UnifiedWebManager()


# Convenience functions for backward compatibility
def get_unified_converter():
    """Get the unified image to DXF converter."""
    return web_manager.get_unified_converter()


def get_conversion_params():
    """Get the conversion parameters class."""
    return web_manager.get_conversion_params()


def get_interactive_components():
    """Get interactive editing components."""
    return web_manager.get_interactive_components()


def create_conversion_params(**kwargs):
    """Create conversion parameters with defaults."""
    return web_manager.create_conversion_params(**kwargs)


def create_analyze_args(**kwargs):
    """Create analyze arguments with defaults."""
    return web_manager.create_analyze_args(**kwargs)


def validate_file_upload(filename: str) -> bool:
    """Validate uploaded file."""
    return web_manager.validate_file_upload(filename)


def get_upload_path(filename: str) -> Path:
    """Get upload path for a file."""
    return web_manager.get_upload_path(filename)


def get_output_path(filename: str) -> Path:
    """Get output path for a file."""
    return web_manager.get_output_path(filename)


def get_agents_interface():
    """Get OpenAI agents interface."""
    return web_manager.get_agents_interface()


def get_agents_manager():
    """Get OpenAI agents manager."""
    return web_manager.get_agents_manager()


if __name__ == "__main__":
    # Test the unified web manager
    print("Testing Unified Web Manager...")
    
    # Test imports
    converter = get_unified_converter()
    print(f"Unified Converter: {converter is not None}")
    
    params = get_conversion_params()
    print(f"Conversion Params: {params is not None}")
    
    # Test component loading
    interactive = get_interactive_components()
    print(f"Interactive Components: {len(interactive)} loaded")
    
    # Test parameter creation
    conv_params = create_conversion_params(binary_threshold=150)
    print(f"Conversion Params Created: {conv_params is not None}")
    
    print("Unified Web Manager test completed!")
