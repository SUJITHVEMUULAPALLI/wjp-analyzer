"""
Logging Configuration for WJP Analyser
======================================

This module provides centralized logging configuration with proper
log rotation, formatting, and security considerations.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        r'api[_-]?key["\s]*[:=]["\s]*([^"\s\n]+)',
        r'secret[_-]?key["\s]*[:=]["\s]*([^"\s\n]+)',
        r'password["\s]*[:=]["\s]*([^"\s\n]+)',
        r'token["\s]*[:=]["\s]*([^"\s\n]+)',
        r'sk-[a-zA-Z0-9]{20,}',
        r'pk_[a-zA-Z0-9]{20,}',
    ]
    
    def filter(self, record):
        """Filter sensitive information from log records."""
        if hasattr(record, 'msg'):
            import re
            msg = str(record.msg)
            for pattern in self.SENSITIVE_PATTERNS:
                msg = re.sub(pattern, '***REDACTED***', msg, flags=re.IGNORECASE)
            record.msg = msg
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        return super().format(record)


class LoggingManager:
    """Centralized logging manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup root logger with basic configuration."""
        # Get configuration
        log_level = self.config.get('level', 'INFO')
        log_dir = self.config.get('logs_folder', 'logs')
        max_bytes = self.config.get('file_rotation', {}).get('max_bytes', 10 * 1024 * 1024)
        backup_count = self.config.get('file_rotation', {}).get('backup_count', 5)
        console_output = self.config.get('console_output', True)
        file_output = self.config.get('file_output', True)
        
        # Create logs directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            # Use colored formatter for console
            console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            console_formatter = ColoredFormatter(console_format)
            console_handler.setFormatter(console_formatter)
            
            # Add security filter
            console_handler.addFilter(SecurityFilter())
            
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if file_output:
            log_file = Path(log_dir) / 'wjp_analyser.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            
            # Use JSON formatter for file
            file_formatter = JSONFormatter()
            file_handler.setFormatter(file_formatter)
            
            # Add security filter
            file_handler.addFilter(SecurityFilter())
            
            root_logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = Path(log_dir) / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # Use JSON formatter for error file
        error_formatter = JSONFormatter()
        error_handler.setFormatter(error_formatter)
        
        # Add security filter
        error_handler.addFilter(SecurityFilter())
        
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger with given name."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def configure_module_logger(self, module_name: str, level: Optional[str] = None):
        """Configure logger for specific module."""
        logger = self.get_logger(module_name)
        
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        
        return logger
    
    def log_application_start(self, version: str, config_info: Dict[str, Any]):
        """Log application startup information."""
        logger = self.get_logger('wjp_analyser.startup')
        
        startup_info = {
            'event': 'application_start',
            'version': version,
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'config': config_info
        }
        
        logger.info("WJP Analyser started", extra=startup_info)
    
    def log_application_stop(self):
        """Log application shutdown."""
        logger = self.get_logger('wjp_analyser.shutdown')
        logger.info("WJP Analyser shutting down")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events."""
        logger = self.get_logger('wjp_analyser.security')
        
        security_info = {
            'event': 'security_event',
            'event_type': event_type,
            'details': details,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        logger.warning(f"Security event: {event_type}", extra=security_info)
    
    def log_performance_metric(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Log performance metrics."""
        logger = self.get_logger('wjp_analyser.performance')
        
        perf_info = {
            'event': 'performance_metric',
            'operation': operation,
            'duration_seconds': duration,
            'details': details or {}
        }
        
        if duration > 10.0:  # Log slow operations as warnings
            logger.warning(f"Slow operation: {operation} took {duration:.2f}s", extra=perf_info)
        else:
            logger.info(f"Operation: {operation} took {duration:.2f}s", extra=perf_info)
    
    def log_user_action(self, action: str, user_id: Optional[str] = None, details: Dict[str, Any] = None):
        """Log user actions."""
        logger = self.get_logger('wjp_analyser.user_actions')
        
        action_info = {
            'event': 'user_action',
            'action': action,
            'user_id': user_id,
            'details': details or {}
        }
        
        logger.info(f"User action: {action}", extra=action_info)
    
    def log_file_operation(self, operation: str, file_path: str, success: bool, details: Dict[str, Any] = None):
        """Log file operations."""
        logger = self.get_logger('wjp_analyser.file_operations')
        
        file_info = {
            'event': 'file_operation',
            'operation': operation,
            'file_path': file_path,
            'success': success,
            'details': details or {}
        }
        
        if success:
            logger.info(f"File operation: {operation} on {file_path}", extra=file_info)
        else:
            logger.error(f"File operation failed: {operation} on {file_path}", extra=file_info)
    
    def log_ai_request(self, service: str, model: str, success: bool, duration: float, details: Dict[str, Any] = None):
        """Log AI service requests."""
        logger = self.get_logger('wjp_analyser.ai_requests')
        
        ai_info = {
            'event': 'ai_request',
            'service': service,
            'model': model,
            'success': success,
            'duration_seconds': duration,
            'details': details or {}
        }
        
        if success:
            logger.info(f"AI request: {service}/{model} completed in {duration:.2f}s", extra=ai_info)
        else:
            logger.error(f"AI request failed: {service}/{model} after {duration:.2f}s", extra=ai_info)


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def initialize_logging(config: Optional[Dict[str, Any]] = None) -> LoggingManager:
    """Initialize logging system."""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    return _logging_manager


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    if _logging_manager is None:
        # Fallback to basic logging if not initialized
        return logging.getLogger(name)
    
    return _logging_manager.get_logger(name)


def get_logging_manager() -> Optional[LoggingManager]:
    """Get logging manager instance."""
    return _logging_manager


# Convenience functions for common logging operations
def log_startup(version: str, config_info: Dict[str, Any]):
    """Log application startup."""
    if _logging_manager:
        _logging_manager.log_application_start(version, config_info)


def log_shutdown():
    """Log application shutdown."""
    if _logging_manager:
        _logging_manager.log_application_stop()


def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security event."""
    if _logging_manager:
        _logging_manager.log_security_event(event_type, details)


def log_performance(operation: str, duration: float, details: Dict[str, Any] = None):
    """Log performance metric."""
    if _logging_manager:
        _logging_manager.log_performance_metric(operation, duration, details)


def log_user_action(action: str, user_id: Optional[str] = None, details: Dict[str, Any] = None):
    """Log user action."""
    if _logging_manager:
        _logging_manager.log_user_action(action, user_id, details)


def log_file_operation(operation: str, file_path: str, success: bool, details: Dict[str, Any] = None):
    """Log file operation."""
    if _logging_manager:
        _logging_manager.log_file_operation(operation, file_path, success, details)


def log_ai_request(service: str, model: str, success: bool, duration: float, details: Dict[str, Any] = None):
    """Log AI request."""
    if _logging_manager:
        _logging_manager.log_ai_request(service, model, success, duration, details)
