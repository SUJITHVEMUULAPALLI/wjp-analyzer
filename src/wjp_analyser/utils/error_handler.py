"""
Centralized Error Handling System for WJP Analyser
=================================================

This module provides standardized error handling, logging, and user-friendly
error messages across the application.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import sys

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization."""
    VALIDATION = "validation"
    FILE_PROCESSING = "file_processing"
    AI_SERVICE = "ai_service"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    USER_INPUT = "user_input"


@dataclass
class WJPError(Exception):
    """Base exception class for WJP Analyser with enhanced error information."""
    
    message: str
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = None
    suggested_action: Optional[str] = None
    
    def __post_init__(self):
        if self.user_message is None:
            self.user_message = self.message
        if self.error_code is None:
            self.error_code = f"{self.category.value}_{self.severity.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details or {},
            "suggested_action": self.suggested_action
        }


class ValidationError(WJPError):
    """Validation-related errors."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        if field:
            self.details = self.details or {}
            self.details["field"] = field


class FileProcessingError(WJPError):
    """File processing related errors."""
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if file_path:
            self.details = self.details or {}
            self.details["file_path"] = file_path


class AIServiceError(WJPError):
    """AI service related errors."""
    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AI_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if service:
            self.details = self.details or {}
            self.details["service"] = service


class ConfigurationError(WJPError):
    """Configuration related errors."""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if config_key:
            self.details = self.details or {}
            self.details["config_key"] = config_key


class SystemError(WJPError):
    """System-level errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handler with logging and user-friendly messages."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: Optional[str] = None) -> Dict[str, Any]:
        """Handle any exception and return user-friendly error information."""
        
        # Log the error
        self._log_error(error, context)
        
        # Convert to WJPError if needed
        if isinstance(error, WJPError):
            return error.to_dict()
        
        # Handle common exception types
        if isinstance(error, FileNotFoundError):
            return self._handle_file_not_found(error)
        elif isinstance(error, PermissionError):
            return self._handle_permission_error(error)
        elif isinstance(error, ValueError):
            return self._handle_value_error(error)
        elif isinstance(error, ConnectionError):
            return self._handle_connection_error(error)
        elif isinstance(error, TimeoutError):
            return self._handle_timeout_error(error)
        else:
            return self._handle_generic_error(error)
    
    def _log_error(self, error: Exception, context: Optional[str] = None):
        """Log error with appropriate level."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        if isinstance(error, WJPError):
            if error.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"Critical error: {error_info}")
            elif error.severity == ErrorSeverity.HIGH:
                self.logger.error(f"High severity error: {error_info}")
            elif error.severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"Medium severity error: {error_info}")
            else:
                self.logger.info(f"Low severity error: {error_info}")
        else:
            self.logger.error(f"Unhandled error: {error_info}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def _handle_file_not_found(self, error: FileNotFoundError) -> Dict[str, Any]:
        """Handle file not found errors."""
        return {
            "error_code": "file_not_found",
            "message": str(error),
            "user_message": "The requested file could not be found.",
            "category": ErrorCategory.FILE_PROCESSING.value,
            "severity": ErrorSeverity.MEDIUM.value,
            "suggested_action": "Please check the file path and ensure the file exists."
        }
    
    def _handle_permission_error(self, error: PermissionError) -> Dict[str, Any]:
        """Handle permission errors."""
        return {
            "error_code": "permission_denied",
            "message": str(error),
            "user_message": "You don't have permission to access this resource.",
            "category": ErrorCategory.SYSTEM.value,
            "severity": ErrorSeverity.HIGH.value,
            "suggested_action": "Please check file permissions or contact your administrator."
        }
    
    def _handle_value_error(self, error: ValueError) -> Dict[str, Any]:
        """Handle value errors."""
        return {
            "error_code": "invalid_value",
            "message": str(error),
            "user_message": "Invalid input value provided.",
            "category": ErrorCategory.VALIDATION.value,
            "severity": ErrorSeverity.LOW.value,
            "suggested_action": "Please check your input and try again."
        }
    
    def _handle_connection_error(self, error: ConnectionError) -> Dict[str, Any]:
        """Handle connection errors."""
        return {
            "error_code": "connection_failed",
            "message": str(error),
            "user_message": "Unable to connect to the required service.",
            "category": ErrorCategory.NETWORK.value,
            "severity": ErrorSeverity.MEDIUM.value,
            "suggested_action": "Please check your internet connection and try again."
        }
    
    def _handle_timeout_error(self, error: TimeoutError) -> Dict[str, Any]:
        """Handle timeout errors."""
        return {
            "error_code": "timeout",
            "message": str(error),
            "user_message": "The operation timed out.",
            "category": ErrorCategory.NETWORK.value,
            "severity": ErrorSeverity.MEDIUM.value,
            "suggested_action": "Please try again or contact support if the problem persists."
        }
    
    def _handle_generic_error(self, error: Exception) -> Dict[str, Any]:
        """Handle generic errors."""
        return {
            "error_code": "unknown_error",
            "message": str(error),
            "user_message": "An unexpected error occurred.",
            "category": ErrorCategory.SYSTEM.value,
            "severity": ErrorSeverity.HIGH.value,
            "suggested_action": "Please try again or contact support if the problem persists."
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(context: Optional[str] = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = error_handler.handle_error(e, context or func.__name__)
                
                # For web applications, you might want to raise a specific exception
                # For CLI applications, you might want to print and exit
                if 'web' in sys.modules:
                    # Running in web context
                    from flask import abort, jsonify
                    abort(500, error_info)
                else:
                    # Running in CLI context
                    print(f"Error: {error_info['user_message']}")
                    if error_info['suggested_action']:
                        print(f"Suggestion: {error_info['suggested_action']}")
                    sys.exit(1)
        
        return wrapper
    return decorator


def safe_execute(func, *args, context: Optional[str] = None, **kwargs) -> tuple[Any, Optional[Dict[str, Any]]]:
    """Safely execute a function and return result with error information."""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_info = error_handler.handle_error(e, context)
        return None, error_info


# Convenience functions for common error types
def raise_validation_error(message: str, field: Optional[str] = None, **kwargs):
    """Raise a validation error."""
    raise ValidationError(message, field, **kwargs)


def raise_file_processing_error(message: str, file_path: Optional[str] = None, **kwargs):
    """Raise a file processing error."""
    raise FileProcessingError(message, file_path, **kwargs)


def raise_ai_service_error(message: str, service: Optional[str] = None, **kwargs):
    """Raise an AI service error."""
    raise AIServiceError(message, service, **kwargs)


def raise_configuration_error(message: str, config_key: Optional[str] = None, **kwargs):
    """Raise a configuration error."""
    raise ConfigurationError(message, config_key, **kwargs)


def raise_system_error(message: str, **kwargs):
    """Raise a system error."""
    raise SystemError(message, **kwargs)
