"""
Logging Service for WJP Analyser
================================

Centralized logging service that handles all logging operations across the application.
Supports multiple logging handlers and formats.
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class LoggingService:
    """Manages centralized logging for the application."""

    def __init__(self, log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 enable_file_logging: bool = True,
                 enable_json_logging: bool = True):
        """
        Initialize logging service.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (default: INFO)
            enable_file_logging: Enable logging to file
            enable_json_logging: Enable JSON format logging
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up main logger
        self.logger = logging.getLogger("wjp_analyser")
        self.logger.setLevel(log_level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if enable_file_logging:
            self._setup_file_handler()

        # JSON handler
        if enable_json_logging:
            self._setup_json_handler()

    def _setup_file_handler(self):
        """Set up logging to regular text file."""
        log_file = self.log_dir / "wjp_analyser.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _setup_json_handler(self):
        """Set up logging to JSON file."""
        json_log_file = self.log_dir / "wjp_analyser.json"
        json_handler = logging.FileHandler(json_log_file)
        json_handler.setLevel(self.log_level)

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'funcName': record.funcName,
                    'lineNo': record.lineno
                }
                if hasattr(record, 'extra_data'):
                    log_data.update(record.extra_data)
                return json.dumps(log_data)

        json_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(json_handler)

    def log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log a message with optional extra data.

        Args:
            level: Logging level (e.g., logging.INFO)
            message: Log message
            extra: Optional dictionary of extra data to include
        """
        if extra:
            # Create a new logger adapter to inject extra data
            extra_data = {'extra_data': extra}
            adapter = logging.LoggerAdapter(self.logger, extra_data)
            adapter.log(level, message)
        else:
            self.logger.log(level, message)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log at INFO level."""
        self.log(logging.INFO, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log at ERROR level."""
        self.log(logging.ERROR, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log at WARNING level."""
        self.log(logging.WARNING, message, extra)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log at DEBUG level."""
        self.log(logging.DEBUG, message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log at CRITICAL level."""
        self.log(logging.CRITICAL, message, extra)

    def get_logs(self, n: int = 100) -> list[dict]:
        """
        Retrieve the last n log entries from the JSON log file.
        
        Args:
            n: Number of log entries to retrieve
            
        Returns:
            List of log entries as dictionaries
        """
        json_log_file = self.log_dir / "wjp_analyser.json"
        if not json_log_file.exists():
            return []
            
        logs = []
        try:
            with open(json_log_file) as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                        
            return logs[-n:]
        except Exception:
            return []