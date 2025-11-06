"""
Comprehensive Error Handling System
==================================

Standardized error handling with custom exceptions, logging, and recovery mechanisms.
"""

import logging
import traceback
import functools
from typing import Dict, Any, Optional, Callable, Type, Union, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    FILE_IO = "file_io"
    NETWORK = "network"
    DATABASE = "database"
    AI_SERVICE = "ai_service"
    DXF_ANALYSIS = "dxf_analysis"
    NESTING = "nesting"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Error context information."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    file_path: Optional[str] = None
    operation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class WJPAnalyserError(Exception):
    """Base exception for WJP ANALYSER."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "file_path": self.context.file_path,
                "operation": self.context.operation,
                "parameters": self.context.parameters
            },
            "original_error": str(self.original_error) if self.original_error else None
        }


class AuthenticationError(WJPAnalyserError):
    """Authentication related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH,
                        context, original_error)


class AuthorizationError(WJPAnalyserError):
    """Authorization related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.AUTHORIZATION, ErrorSeverity.HIGH,
                        context, original_error)


class ValidationError(WJPAnalyserError):
    """Input validation errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM,
                        context, original_error)


class FileIOError(WJPAnalyserError):
    """File I/O related errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        if context is None:
            context = ErrorContext(file_path=file_path)
        elif file_path:
            context.file_path = file_path
        super().__init__(message, ErrorCategory.FILE_IO, ErrorSeverity.MEDIUM,
                        context, original_error)


class NetworkError(WJPAnalyserError):
    """Network related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM,
                        context, original_error)


class DatabaseError(WJPAnalyserError):
    """Database related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.DATABASE, ErrorSeverity.HIGH,
                        context, original_error)


class AIServiceError(WJPAnalyserError):
    """AI service related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.AI_SERVICE, ErrorSeverity.MEDIUM,
                        context, original_error)


class ImageProcessingError(WJPAnalyserError):
    """Image processing related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.IMAGE_PROCESSING, ErrorSeverity.MEDIUM,
                        context, original_error)


class DXFAnalysisError(WJPAnalyserError):
    """DXF analysis related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.DXF_ANALYSIS, ErrorSeverity.MEDIUM,
                        context, original_error)


class NestingError(WJPAnalyserError):
    """Nesting optimization related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.NESTING, ErrorSeverity.MEDIUM,
                        context, original_error)


class ConfigurationError(WJPAnalyserError):
    """Configuration related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH,
                        context, original_error)


class SystemError(WJPAnalyserError):
    """System related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL,
                        context, original_error)


class ErrorHandler:
    """Centralized error handler."""
    
    def __init__(self):
        self.error_log_file = "logs/errors.json"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup error logging."""
        # Resolve a safe log directory
        log_path = self.error_log_file
        log_dir = os.path.dirname(log_path)
        try:
            os.makedirs(log_dir, exist_ok=True)
            safe_file = log_path
        except Exception:
            # Fall back to LOCALAPPDATA or TEMP
            base = os.getenv("LOCALAPPDATA") or os.getenv("TMP") or os.getenv("TEMP") or os.getcwd()
            fallback_dir = os.path.join(base, "WJP_ANALYSER", "logs")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                safe_file = os.path.join(fallback_dir, "errors.json")
            except Exception:
                safe_file = None
        
        # Create error logger
        self.error_logger = logging.getLogger("wjp_analyser.errors")
        self.error_logger.setLevel(logging.ERROR)
        
        # File handler for structured error logging if available
        if safe_file:
            try:
                file_handler = logging.FileHandler(safe_file)
                file_handler.setLevel(logging.ERROR)
                formatter = logging.Formatter('%(message)s')
                file_handler.setFormatter(formatter)
                self.error_logger.addHandler(file_handler)
                self.error_log_file = safe_file
            except Exception:
                self.error_log_file = None
        else:
            self.error_log_file = None
        
        self.error_logger.propagate = False
    
    def handle_error(self, error: WJPAnalyserError, log_to_file: bool = True,
                    log_to_console: bool = True) -> Dict[str, Any]:
        """Handle and log error."""
        error_data = error.to_dict()
        
        # Log to file
        if log_to_file:
            self.error_logger.error(json.dumps(error_data))
        
        # Log to console
        if log_to_console:
            logger.error(f"Error occurred: {error.message}")
            logger.error(f"Category: {error.category.value}, Severity: {error.severity.value}")
            if error.original_error:
                logger.error(f"Original error: {error.original_error}")
        
        # Return error response
        return {
            "success": False,
            "error": {
                "type": error.__class__.__name__,
                "message": error.message,
                "category": error.category.value,
                "severity": error.severity.value,
                "timestamp": error.timestamp.isoformat()
            }
        }
    
    def handle_exception(self, exception: Exception, context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Handle generic exception."""
        # Convert to WJP ANALYSER error
        if isinstance(exception, WJPAnalyserError):
            return self.handle_error(exception)
        
        # Create appropriate error type based on exception
        if isinstance(exception, FileNotFoundError):
            error = FileIOError(f"File not found: {exception}", context=context, original_error=exception)
        elif isinstance(exception, PermissionError):
            error = FileIOError(f"Permission denied: {exception}", context=context, original_error=exception)
        elif isinstance(exception, ConnectionError):
            error = NetworkError(f"Connection error: {exception}", context=context, original_error=exception)
        elif isinstance(exception, ValueError):
            error = ValidationError(f"Invalid value: {exception}", context=context, original_error=exception)
        elif isinstance(exception, KeyError):
            error = ValidationError(f"Missing key: {exception}", context=context, original_error=exception)
        else:
            error = SystemError(f"Unexpected error: {exception}", context=context, original_error=exception)
        
        return self.handle_error(error)


# Global error handler instance
error_handler = ErrorHandler()


def error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN,
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           reraise: bool = False):
    """Decorator for error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except WJPAnalyserError as e:
                # Already a WJP ANALYSER error, just handle it
                result = error_handler.handle_error(e)
                if reraise:
                    raise
                return result
            except Exception as e:
                # Convert to WJP ANALYSER error
                context = ErrorContext(operation=func.__name__)
                error = WJPAnalyserError(str(e), category, severity, context, e)
                result = error_handler.handle_error(error)
                if reraise:
                    raise error
                return result
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Safely execute a function with error handling."""
    try:
        result = func(*args, **kwargs)
        return {"success": True, "result": result}
    except WJPAnalyserError as e:
        return error_handler.handle_error(e)
    except Exception as e:
        context = ErrorContext(operation=func.__name__)
        return error_handler.handle_exception(e, context)


def validate_file_path(file_path: str, must_exist: bool = True) -> None:
    """Validate file path."""
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    if not isinstance(file_path, str):
        raise ValidationError("File path must be a string")
    
    # Check for path traversal
    if ".." in file_path or file_path.startswith("/"):
        raise ValidationError("Invalid file path: path traversal not allowed")
    
    if must_exist and not os.path.exists(file_path):
        raise FileIOError(f"File does not exist: {file_path}", file_path)


def validate_user_input(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate user input data."""
    if not isinstance(data, dict):
        raise ValidationError("Input data must be a dictionary")
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Check for empty values
    empty_fields = [field for field in required_fields if not data.get(field)]
    if empty_fields:
        raise ValidationError(f"Empty values not allowed for fields: {', '.join(empty_fields)}")


def create_error_response(error: WJPAnalyserError, include_details: bool = False) -> Dict[str, Any]:
    """Create standardized error response."""
    response = {
        "success": False,
        "error": {
            "type": error.__class__.__name__,
            "message": error.message,
            "category": error.category.value,
            "severity": error.severity.value,
            "timestamp": error.timestamp.isoformat()
        }
    }
    
    if include_details:
        response["error"]["context"] = {
            "user_id": error.context.user_id,
            "session_id": error.context.session_id,
            "request_id": error.context.request_id,
            "file_path": error.context.file_path,
            "operation": error.context.operation
        }
        
        if error.original_error:
            response["error"]["original_error"] = str(error.original_error)
    
    return response


def log_error_with_context(error: WJPAnalyserError, additional_context: Optional[Dict[str, Any]] = None):
    """Log error with additional context."""
    error_data = error.to_dict()
    
    if additional_context:
        error_data["additional_context"] = additional_context
    
    error_handler.error_logger.error(json.dumps(error_data))


# Error recovery mechanisms
class ErrorRecovery:
    """Error recovery mechanisms."""
    
    @staticmethod
    def retry_with_backoff(func: Callable, max_retries: int = 3, 
                          backoff_factor: float = 2.0, *args, **kwargs):
        """Retry function with exponential backoff."""
        import time
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    @staticmethod
    def fallback_value(func: Callable, fallback_value: Any, *args, **kwargs):
        """Execute function with fallback value on error."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Function failed, using fallback value: {e}")
            return fallback_value
    
    @staticmethod
    def graceful_degradation(func: Callable, degraded_func: Callable, *args, **kwargs):
        """Execute function with graceful degradation on error."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed, using degraded version: {e}")
            return degraded_func(*args, **kwargs)