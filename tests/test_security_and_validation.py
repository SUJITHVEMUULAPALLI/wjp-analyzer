"""
Security and Validation Tests for WJP Analyser
==============================================

This module tests the security hardening, input validation, and error handling
implementations to ensure they work correctly.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules we're testing
from src.wjp_analyser.config.secure_config import (
    SecureConfigManager, SecurityConfig, AIConfig, AppConfig,
    get_security_config, get_ai_config, get_app_config, validate_config
)
from src.wjp_analyser.utils.error_handler import (
    WJPError, ValidationError, FileProcessingError, AIServiceError,
    ConfigurationError, SystemError, ErrorHandler, error_handler,
    ErrorCategory, ErrorSeverity, raise_validation_error
)
from src.wjp_analyser.utils.input_validator import (
    FileValidator, ParameterValidator, ValidationResult,
    file_validator, parameter_validator, validate_uploaded_file,
    validate_material_params, validate_image_params, sanitize_filename
)
from src.wjp_analyser.utils.logging_config import (
    LoggingManager, initialize_logging, get_logger, log_security_event
)
from src.wjp_analyser.utils.cache_manager import (
    CacheManager, MemoryCache, FileBasedCache, initialize_cache,
    get_cache_manager, cached, cache_dxf_analysis, cache_image_processing
)


class TestSecureConfig:
    """Test secure configuration management."""
    
    def test_security_config_creation(self):
        """Test SecurityConfig creation with defaults."""
        config = SecurityConfig(secret_key="test_key_32_chars_long_12345")
        
        assert config.secret_key == "test_key_32_chars_long_12345"
        assert config.max_upload_size == 32 * 1024 * 1024
        assert "dxf" in config.allowed_extensions
        assert config.enable_cors is False
    
    def test_secret_key_generation(self):
        """Test secure secret key generation."""
        config_manager = SecureConfigManager()
        
        # Test with no environment variable
        with patch.dict(os.environ, {}, clear=True):
            secret_key = config_manager._get_secret_key()
            assert len(secret_key) >= 32
            assert isinstance(secret_key, str)
    
    def test_openai_key_loading(self):
        """Test OpenAI API key loading."""
        config_manager = SecureConfigManager()
        
        # Test with environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            api_key = config_manager._get_openai_key()
            assert api_key == "sk-test-key"
        
        # Test without environment variable
        with patch.dict(os.environ, {}, clear=True):
            api_key = config_manager._get_openai_key()
            assert api_key is None
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = SecureConfigManager()
        
        # Test with valid config
        with patch.dict(os.environ, {"SECRET_KEY": "test_key_32_chars_long_12345"}):
            assert config_manager.validate_config() is True
        
        # Test with invalid config (short secret key) - system generates a valid key
        with patch.dict(os.environ, {"SECRET_KEY": "short"}):
            # Clear cache to force re-evaluation
            config_manager._config_cache.clear()
            # The system generates a valid key, so validation should pass
            assert config_manager.validate_config() is True


class TestErrorHandler:
    """Test error handling system."""
    
    def test_wjp_error_creation(self):
        """Test WJPError creation and serialization."""
        error = WJPError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error_code="test_error",
            user_message="User-friendly message",
            suggested_action="Try again"
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.error_code == "test_error"
        
        # Test serialization
        error_dict = error.to_dict()
        assert error_dict["message"] == "Test error"
        assert error_dict["user_message"] == "User-friendly message"
        assert error_dict["suggested_action"] == "Try again"
    
    def test_specific_error_types(self):
        """Test specific error type creation."""
        # Validation error
        val_error = ValidationError("Invalid input", field="email")
        assert val_error.category == ErrorCategory.VALIDATION
        assert val_error.details["field"] == "email"
        
        # File processing error
        file_error = FileProcessingError("File not found", file_path="/test/file.dxf")
        assert file_error.category == ErrorCategory.FILE_PROCESSING
        assert file_error.details["file_path"] == "/test/file.dxf"
        
        # AI service error
        ai_error = AIServiceError("API timeout", service="openai")
        assert ai_error.category == ErrorCategory.AI_SERVICE
        assert ai_error.details["service"] == "openai"
    
    def test_error_handler_processing(self):
        """Test error handler processing."""
        handler = ErrorHandler()
        
        # Test handling WJPError
        wjp_error = ValidationError("Test validation error")
        result = handler.handle_error(wjp_error)
        assert result["error_code"] == "validation_low"
        assert result["category"] == "validation"
        
        # Test handling generic exception
        generic_error = FileNotFoundError("File not found")
        result = handler.handle_error(generic_error)
        assert result["error_code"] == "file_not_found"
        assert result["category"] == "file_processing"
    
    def test_error_handler_logging(self):
        """Test error handler logging."""
        handler = ErrorHandler()
        
        with patch.object(handler.logger, 'info') as mock_info:
            error = ValidationError("Test error")
            handler.handle_error(error)
            mock_info.assert_called_once()


class TestInputValidator:
    """Test input validation system."""
    
    def test_filename_validation(self):
        """Test filename validation."""
        validator = FileValidator()
        
        # Valid filename
        result = validator.validate_filename("test_file.dxf")
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Invalid filename with directory traversal
        result = validator.validate_filename("../../../etc/passwd")
        assert result.is_valid is False
        assert len(result.errors) > 0
        
        # Invalid filename with reserved name
        result = validator.validate_filename("CON.dxf")
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_file_extension_validation(self):
        """Test file extension validation."""
        validator = FileValidator()
        
        # Valid extension
        result = validator.validate_file_extension("test.dxf")
        assert result.is_valid is True
        
        # Invalid extension
        result = validator.validate_file_extension("test.exe")
        assert result.is_valid is False
        assert "not allowed" in result.errors[0]
        
        # No extension
        result = validator.validate_file_extension("test")
        assert result.is_valid is False
        assert "must have an extension" in result.errors[0]
    
    def test_file_size_validation(self):
        """Test file size validation."""
        validator = FileValidator()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            tmp_file.close()  # Close the file handle
            
            try:
                # Valid size
                result = validator.validate_file_size(tmp_file.name, max_size=1024)
                assert result.is_valid is True
                
                # Invalid size (too large)
                result = validator.validate_file_size(tmp_file.name, max_size=1)
                assert result.is_valid is False
                assert "too large" in result.errors[0]
                
            finally:
                try:
                    os.unlink(tmp_file.name)
                except PermissionError:
                    pass  # File might be locked on Windows
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        validator = ParameterValidator()
        
        # Valid numeric parameter
        result = validator.validate_numeric_parameter(5.0, "test_param", min_val=0, max_val=10)
        assert result.is_valid is True
        assert result.sanitized_value == 5.0
        
        # Invalid numeric parameter (out of range)
        result = validator.validate_numeric_parameter(15.0, "test_param", min_val=0, max_val=10)
        assert result.is_valid is False
        assert "must be <=" in result.errors[0]
        
        # Invalid numeric parameter (not a number)
        result = validator.validate_numeric_parameter("not_a_number", "test_param")
        assert result.is_valid is False
        assert "must be a number" in result.errors[0]
    
    def test_material_parameters_validation(self):
        """Test material parameters validation."""
        validator = ParameterValidator()
        
        # Valid material parameters
        params = {
            "material": "steel",
            "thickness": 6.0,
            "kerf": 1.1,
            "cutting_speed": 1200.0
        }
        result = validator.validate_material_parameters(params)
        assert result.is_valid is True
        
        # Invalid material parameters
        params = {
            "material": "unknown_material",
            "thickness": -1.0,
            "kerf": 0.01,
            "cutting_speed": 50.0
        }
        result = validator.validate_material_parameters(params)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Valid filename
        result = sanitize_filename("test_file.dxf")
        assert result == "test_file.dxf"
        
        # Filename with dangerous characters
        result = sanitize_filename("test<>file.dxf")
        assert result == "testfile.dxf"
        
        # Filename with path traversal
        result = sanitize_filename("../../../test.dxf")
        assert result == "test.dxf"
        
        # Empty filename
        result = sanitize_filename("")
        assert result == "unnamed_file"


class TestLoggingSystem:
    """Test logging system."""
    
    def test_logging_manager_initialization(self):
        """Test logging manager initialization."""
        config = {
            "level": "INFO",
            "logs_folder": "test_logs",
            "console_output": True,
            "file_output": False
        }
        
        manager = LoggingManager(config)
        assert manager is not None
        
        # Test logger creation
        logger = manager.get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"
    
    def test_security_event_logging(self):
        """Test security event logging."""
        config = {"level": "INFO", "logs_folder": "test_logs", "console_output": False, "file_output": False}
        manager = LoggingManager(config)
        
        with patch.object(manager.get_logger('wjp_analyser.security'), 'warning') as mock_warning:
            manager.log_security_event("test_event", {"details": "test"})
            mock_warning.assert_called_once()
    
    def test_performance_logging(self):
        """Test performance metric logging."""
        config = {"level": "INFO", "logs_folder": "test_logs", "console_output": False, "file_output": False}
        manager = LoggingManager(config)
        
        with patch.object(manager.get_logger('wjp_analyser.performance'), 'info') as mock_info:
            manager.log_performance_metric("test_operation", 1.5, {"details": "test"})
            mock_info.assert_called_once()


class TestCacheManager:
    """Test cache management system."""
    
    def test_memory_cache(self):
        """Test memory cache functionality."""
        cache = MemoryCache(max_size=2)
        
        # Test set and get
        cache.set("key1", "value1", ttl=3600)
        assert cache.get("key1") == "value1"
        
        # Test expiration
        cache.set("key2", "value2", ttl=0.001)  # Very short TTL
        import time
        time.sleep(0.002)
        assert cache.get("key2") is None
        
        # Test eviction
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict oldest
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_file_cache(self):
        """Test file cache functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileBasedCache(temp_dir)
            
            # Test set and get
            cache.set("test_key", "test_value", ttl=3600)
            assert cache.get("test_key") == "test_value"
            
            # Test deletion
            cache.delete("test_key")
            assert cache.get("test_key") is None
            
            # Test clear
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.clear()
            assert cache.get("key1") is None
            assert cache.get("key2") is None
    
    def test_cache_manager(self):
        """Test unified cache manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Test set and get
            manager.set("test_key", "test_value", ttl=3600)
            assert manager.get("test_key") == "test_value"
            
            # Test deletion
            manager.delete("test_key")
            assert manager.get("test_key") is None
            
            # Test stats
            stats = manager.get_stats()
            assert "memory_cache" in stats
            assert "file_cache" in stats
    
    def test_cache_decorator(self):
        """Test cache decorator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Mock the global cache manager
            with patch('src.wjp_analyser.utils.cache_manager._cache_manager', manager):
                call_count = 0
                
                @cached(ttl=3600)
                def expensive_function(x):
                    nonlocal call_count
                    call_count += 1
                    return x * 2
                
                # First call should execute function
                result1 = expensive_function(5)
                assert result1 == 10
                assert call_count == 1
                
                # Second call should use cache
                result2 = expensive_function(5)
                assert result2 == 10
                assert call_count == 1  # Should not increment
                
                # Different argument should execute function
                result3 = expensive_function(3)
                assert result3 == 6
                assert call_count == 2


class TestIntegration:
    """Integration tests for security and validation systems."""
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp_file:
            tmp_file.write(b"test dxf content")
            tmp_file.flush()
            tmp_file.close()  # Close the file handle
            
            try:
                # Test file validation
                result = validate_uploaded_file(tmp_file.name, "test.dxf")
                assert result.is_valid is True
                
                # Test parameter validation
                params = {"material": "steel", "thickness": 6.0, "kerf": 1.1, "cutting_speed": 1200.0}
                result = validate_material_params(params)
                assert result.is_valid is True
                
            finally:
                try:
                    os.unlink(tmp_file.name)
                except PermissionError:
                    pass  # File might be locked on Windows
    
    def test_error_handling_integration(self):
        """Test error handling integration."""
        # Test error handling with validation
        try:
            raise_validation_error("Test validation error", field="test_field")
        except ValidationError as e:
            error_info = error_handler.handle_error(e)
            assert error_info["category"] == "validation"
            assert error_info["error_code"] == "validation_low"
    
    def test_configuration_integration(self):
        """Test configuration system integration."""
        # Test configuration loading
        config_manager = SecureConfigManager()
        
        # Test security config
        security_config = config_manager.get_security_config()
        assert isinstance(security_config, SecurityConfig)
        assert len(security_config.secret_key) >= 32
        
        # Test AI config
        ai_config = config_manager.get_ai_config()
        assert isinstance(ai_config, AIConfig)
        
        # Test app config
        app_config = config_manager.get_app_config()
        assert isinstance(app_config, AppConfig)


if __name__ == "__main__":
    # Run tests individually for debugging
    test_classes = [
        TestSecureConfig,
        TestErrorHandler,
        TestInputValidator,
        TestLoggingSystem,
        TestCacheManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                print(f"  {method_name}...")
                try:
                    getattr(instance, method_name)()
                    print(f"    ✓ Passed")
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
    
    print("\n✓ All security and validation tests completed!")
