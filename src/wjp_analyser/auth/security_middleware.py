"""
Rate Limiting and CSRF Protection
=================================

Advanced security middleware for API protection.
"""

import time
import hashlib
import secrets
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # seconds


@dataclass
class RateLimitRule:
    """Rate limiting rule for specific endpoints."""
    endpoint: str
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    window_size: int = 60


class RateLimiter:
    """Advanced rate limiter with sliding window."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.burst_tokens: Dict[str, int] = defaultdict(lambda: self.config.burst_limit)
        self.last_burst_refill: Dict[str, float] = defaultdict(time.time)
        
        # Endpoint-specific rules
        self.endpoint_rules: Dict[str, RateLimitRule] = {
            '/api/v1/auth/login': RateLimitRule('/api/v1/auth/login', 5, 20, 3),
            '/api/v1/auth/register': RateLimitRule('/api/v1/auth/register', 3, 10, 2),
            '/api/v1/projects': RateLimitRule('/api/v1/projects', 30, 200, 5),
            '/api/v1/analyze': RateLimitRule('/api/v1/analyze', 10, 50, 3),
            '/api/v1/convert': RateLimitRule('/api/v1/convert', 10, 50, 3),
            '/api/v1/nesting': RateLimitRule('/api/v1/nesting', 5, 20, 2),
        }
    
    def _get_client_key(self, client_ip: str, user_id: Optional[str] = None) -> str:
        """Generate unique key for client."""
        if user_id:
            return f"user:{user_id}"
        return f"ip:{client_ip}"
    
    def _cleanup_old_requests(self, client_key: str, window_size: int):
        """Remove old requests outside the window."""
        current_time = time.time()
        requests_queue = self.requests[client_key]
        
        # Remove requests older than window_size
        while requests_queue and requests_queue[0] < current_time - window_size:
            requests_queue.popleft()
    
    def _refill_burst_tokens(self, client_key: str, burst_limit: int):
        """Refill burst tokens based on time elapsed."""
        current_time = time.time()
        last_refill = self.last_burst_refill[client_key]
        time_elapsed = current_time - last_refill
        
        # Refill tokens based on time elapsed (1 token per second)
        tokens_to_add = int(time_elapsed)
        if tokens_to_add > 0:
            self.burst_tokens[client_key] = min(
                burst_limit,
                self.burst_tokens[client_key] + tokens_to_add
            )
            self.last_burst_refill[client_key] = current_time
    
    def is_allowed(self, client_ip: str, endpoint: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if request is allowed."""
        client_key = self._get_client_key(client_ip, user_id)
        current_time = time.time()
        
        # Get endpoint-specific rule or use default
        rule = self.endpoint_rules.get(endpoint)
        if rule:
            requests_per_minute = rule.requests_per_minute
            requests_per_hour = rule.requests_per_hour
            burst_limit = rule.burst_limit
            window_size = rule.window_size
        else:
            requests_per_minute = self.config.requests_per_minute
            requests_per_hour = self.config.requests_per_hour
            burst_limit = self.config.burst_limit
            window_size = self.config.window_size
        
        # Clean up old requests
        self._cleanup_old_requests(client_key, window_size)
        
        # Check burst limit first
        self._refill_burst_tokens(client_key, burst_limit)
        
        if self.burst_tokens[client_key] <= 0:
            return {
                "allowed": False,
                "reason": "burst_limit_exceeded",
                "retry_after": 1,
                "limit_type": "burst"
            }
        
        # Check minute limit
        minute_requests = len([req for req in self.requests[client_key] if req > current_time - 60])
        if minute_requests >= requests_per_minute:
            return {
                "allowed": False,
                "reason": "minute_limit_exceeded",
                "retry_after": 60,
                "limit_type": "minute"
            }
        
        # Check hour limit
        hour_requests = len([req for req in self.requests[client_key] if req > current_time - 3600])
        if hour_requests >= requests_per_hour:
            return {
                "allowed": False,
                "reason": "hour_limit_exceeded",
                "retry_after": 3600,
                "limit_type": "hour"
            }
        
        # Request is allowed
        self.requests[client_key].append(current_time)
        self.burst_tokens[client_key] -= 1
        
        return {
            "allowed": True,
            "remaining_minute": requests_per_minute - minute_requests - 1,
            "remaining_hour": requests_per_hour - hour_requests - 1,
            "remaining_burst": self.burst_tokens[client_key]
        }
    
    def get_client_stats(self, client_ip: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get client rate limiting statistics."""
        client_key = self._get_client_key(client_ip, user_id)
        current_time = time.time()
        
        # Clean up old requests
        self._cleanup_old_requests(client_key, 3600)  # 1 hour window
        
        minute_requests = len([req for req in self.requests[client_key] if req > current_time - 60])
        hour_requests = len([req for req in self.requests[client_key] if req > current_time - 3600])
        
        return {
            "client_key": client_key,
            "requests_last_minute": minute_requests,
            "requests_last_hour": hour_requests,
            "burst_tokens": self.burst_tokens[client_key],
            "last_request": self.requests[client_key][-1] if self.requests[client_key] else None
        }


class CSRFProtection:
    """CSRF protection middleware."""
    
    def __init__(self):
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.token_expiry = 3600  # 1 hour
        self.header_name = 'X-CSRF-Token'
        self.cookie_name = 'csrf_token'
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session."""
        token = secrets.token_urlsafe(32)
        current_time = time.time()
        
        self.tokens[session_id] = {
            'token': token,
            'created_at': current_time,
            'expires_at': current_time + self.token_expiry
        }
        
        logger.info(f"Generated CSRF token for session: {session_id}")
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token."""
        if session_id not in self.tokens:
            logger.warning(f"CSRF token not found for session: {session_id}")
            return False
        
        token_data = self.tokens[session_id]
        current_time = time.time()
        
        # Check if token is expired
        if current_time > token_data['expires_at']:
            logger.warning(f"CSRF token expired for session: {session_id}")
            del self.tokens[session_id]
            return False
        
        # Check if token matches
        if token_data['token'] != token:
            logger.warning(f"CSRF token mismatch for session: {session_id}")
            return False
        
        logger.info(f"CSRF token validated for session: {session_id}")
        return True
    
    def cleanup_expired_tokens(self):
        """Clean up expired CSRF tokens."""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, token_data in self.tokens.items()
            if current_time > token_data['expires_at']
        ]
        
        for session_id in expired_sessions:
            del self.tokens[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired CSRF tokens")


class SecurityMiddleware:
    """Comprehensive security middleware."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.csrf_protection = CSRFProtection()
        self.blocked_ips: Dict[str, float] = {}
        self.block_duration = 3600  # 1 hour
        self.suspicious_patterns = [
            'admin', 'root', 'test', 'api', 'login', 'password',
            'select', 'insert', 'update', 'delete', 'drop', 'union'
        ]
    
    def check_request(self, client_ip: str, endpoint: str, method: str, 
                     headers: Dict[str, str], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive request security check."""
        result = {
            "allowed": True,
            "reason": None,
            "security_checks": {}
        }
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            if time.time() < self.blocked_ips[client_ip]:
                result["allowed"] = False
                result["reason"] = "ip_blocked"
                result["retry_after"] = self.blocked_ips[client_ip] - time.time()
                return result
            else:
                # Remove expired block
                del self.blocked_ips[client_ip]
        
        # Check for suspicious patterns in headers
        suspicious_score = self._check_suspicious_patterns(headers)
        if suspicious_score > 5:
            logger.warning(f"Suspicious request from {client_ip}: score {suspicious_score}")
            result["security_checks"]["suspicious_patterns"] = suspicious_score
        
        # Rate limiting check
        rate_limit_result = self.rate_limiter.is_allowed(client_ip, endpoint, user_id)
        if not rate_limit_result["allowed"]:
            result["allowed"] = False
            result["reason"] = rate_limit_result["reason"]
            result["retry_after"] = rate_limit_result["retry_after"]
            result["limit_type"] = rate_limit_result["limit_type"]
            
            # Block IP if too many violations
            if rate_limit_result["limit_type"] == "burst":
                self.blocked_ips[client_ip] = time.time() + self.block_duration
                logger.warning(f"Blocked IP {client_ip} due to burst limit violations")
            
            return result
        
        result["security_checks"]["rate_limit"] = rate_limit_result
        
        # CSRF protection for state-changing methods
        if method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            csrf_token = headers.get('X-CSRF-Token')
            session_id = headers.get('X-Session-ID')
            
            if not csrf_token or not session_id:
                result["allowed"] = False
                result["reason"] = "csrf_token_missing"
                return result
            
            if not self.csrf_protection.validate_token(session_id, csrf_token):
                result["allowed"] = False
                result["reason"] = "csrf_token_invalid"
                return result
        
        return result
    
    def _check_suspicious_patterns(self, headers: Dict[str, str]) -> int:
        """Check for suspicious patterns in request headers."""
        score = 0
        header_text = ' '.join(headers.values()).lower()
        
        for pattern in self.suspicious_patterns:
            if pattern in header_text:
                score += 1
        
        return score
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "blocked_ips": len(self.blocked_ips),
            "active_csrf_tokens": len(self.csrf_protection.tokens),
            "rate_limiter_stats": {
                "active_clients": len(self.rate_limiter.requests),
                "burst_tokens": len(self.rate_limiter.burst_tokens)
            }
        }


# Global security middleware instance
security_middleware = SecurityMiddleware()


def rate_limit(endpoint: str, requests_per_minute: int = 60):
    """Decorator for rate limiting specific endpoints."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be implemented based on the web framework
            # For now, we'll just return the function
            return func(*args, **kwargs)
        return wrapper
    return decorator


def csrf_protect(func):
    """Decorator for CSRF protection."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # This would be implemented based on the web framework
        # For now, we'll just return the function
        return func(*args, **kwargs)
    return wrapper


def security_check(client_ip: str, endpoint: str, method: str, 
                  headers: Dict[str, str], user_id: Optional[str] = None) -> Dict[str, Any]:
    """Perform comprehensive security check."""
    return security_middleware.check_request(client_ip, endpoint, method, headers, user_id)
