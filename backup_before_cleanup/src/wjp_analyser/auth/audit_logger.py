"""
Security Audit Logger
====================

Logs security-related events for audit and monitoring purposes.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os


class SecurityEventType(Enum):
    """Types of security events."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PASSWORD_CHANGE = "password_change"
    ROLE_CHANGE = "role_change"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    CONFIGURATION_CHANGE = "configuration_change"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: SecurityEventType
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: Optional[str]
    action: Optional[str]
    success: bool
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class SecurityAuditLogger:
    """Logs security events for audit purposes."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize security audit logger.
        
        Args:
            log_file: Path to audit log file. If None, uses default location.
        """
        self.log_file = log_file or os.path.join("logs", "security_audit.log")
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for audit log
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def log_event(self, event: SecurityEvent) -> None:
        """
        Log a security event.
        
        Args:
            event: Security event to log
        """
        try:
            # Log as JSON for easy parsing
            log_data = event.to_dict()
            self.logger.info(json.dumps(log_data))
            
            # Also log to console for immediate visibility of critical events
            if event.event_type in [
                SecurityEventType.LOGIN_FAILURE,
                SecurityEventType.PERMISSION_DENIED,
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecurityEventType.SUSPICIOUS_ACTIVITY
            ]:
                print(f"SECURITY ALERT: {event.event_type.value} - {event.details}")
                
        except Exception as e:
            # Fallback logging if JSON serialization fails
            self.logger.error(f"Failed to log security event: {e}")
            self.logger.info(f"Security Event: {event.event_type.value} - {event.details}")
    
    def log_login_success(self, user_id: str, ip_address: str, user_agent: str) -> None:
        """Log successful login."""
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="authentication",
            action="login",
            success=True,
            details={"message": "User logged in successfully"},
            timestamp=datetime.utcnow()
        )
        self.log_event(event)
    
    def log_login_failure(self, email: str, ip_address: str, user_agent: str, reason: str) -> None:
        """Log failed login attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            user_id=None,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="authentication",
            action="login",
            success=False,
            details={
                "email": email,
                "reason": reason,
                "message": "Login attempt failed"
            },
            timestamp=datetime.utcnow()
        )
        self.log_event(event)
    
    def log_permission_denied(self, user_id: str, ip_address: str, user_agent: str, 
                            resource: str, action: str, required_permission: str) -> None:
        """Log permission denied event."""
        event = SecurityEvent(
            event_type=SecurityEventType.PERMISSION_DENIED,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            success=False,
            details={
                "required_permission": required_permission,
                "message": "Access denied due to insufficient permissions"
            },
            timestamp=datetime.utcnow()
        )
        self.log_event(event)
    
    def log_rate_limit_exceeded(self, ip_address: str, user_agent: str, endpoint: str) -> None:
        """Log rate limit exceeded event."""
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            user_id=None,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=endpoint,
            action="rate_limit",
            success=False,
            details={
                "message": "Rate limit exceeded",
                "endpoint": endpoint
            },
            timestamp=datetime.utcnow()
        )
        self.log_event(event)
    
    def log_file_upload(self, user_id: str, ip_address: str, user_agent: str, 
                       filename: str, file_size: int, file_type: str) -> None:
        """Log file upload event."""
        event = SecurityEvent(
            event_type=SecurityEventType.FILE_UPLOAD,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="file_upload",
            action="upload",
            success=True,
            details={
                "filename": filename,
                "file_size": file_size,
                "file_type": file_type,
                "message": "File uploaded successfully"
            },
            timestamp=datetime.utcnow()
        )
        self.log_event(event)
    
    def log_suspicious_activity(self, user_id: Optional[str], ip_address: str, 
                              user_agent: str, activity: str, details: Dict[str, Any]) -> None:
        """Log suspicious activity."""
        event = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="system",
            action="suspicious_activity",
            success=False,
            details={
                "activity": activity,
                "message": "Suspicious activity detected",
                **details
            },
            timestamp=datetime.utcnow()
        )
        self.log_event(event)
    
    def get_recent_events(self, hours: int = 24) -> list[SecurityEvent]:
        """
        Get recent security events from the log file.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent security events
        """
        events = []
        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        event_time = datetime.fromisoformat(data['timestamp']).timestamp()
                        
                        if event_time >= cutoff_time:
                            # Convert back to SecurityEvent object
                            event = SecurityEvent(
                                event_type=SecurityEventType(data['event_type']),
                                user_id=data['user_id'],
                                ip_address=data['ip_address'],
                                user_agent=data['user_agent'],
                                resource=data['resource'],
                                action=data['action'],
                                success=data['success'],
                                details=data['details'],
                                timestamp=datetime.fromisoformat(data['timestamp'])
                            )
                            events.append(event)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # Skip malformed log entries
                        continue
                        
        except FileNotFoundError:
            # Log file doesn't exist yet
            pass
        
        return events
