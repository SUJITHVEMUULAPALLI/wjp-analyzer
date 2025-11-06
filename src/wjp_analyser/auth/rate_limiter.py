"""
Rate Limiting System
====================

Implements rate limiting for API endpoints and user actions.
"""

import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limits."""
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    BURST = "burst"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    burst_window: int = 60  # seconds


@dataclass
class RateLimitInfo:
    """Rate limit information for a request."""
    allowed: bool
    remaining: int
    reset_time: int
    limit: int
    retry_after: Optional[int] = None


class RateLimiter:
    """In-memory rate limiter."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        
        # In-memory storage for rate limit data
        # In production, this should be Redis or similar
        self.minute_requests: Dict[str, list] = {}
        self.hour_requests: Dict[str, list] = {}
        self.day_requests: Dict[str, list] = {}
        self.burst_requests: Dict[str, list] = {}
    
    def _get_client_key(self, ip_address: str, user_id: Optional[str] = None) -> str:
        """
        Get unique key for rate limiting.
        
        Args:
            ip_address: Client IP address
            user_id: User ID if authenticated
            
        Returns:
            Unique key for rate limiting
        """
        if user_id:
            return f"user:{user_id}"
        return f"ip:{ip_address}"
    
    def _cleanup_old_requests(self, requests: list, window_seconds: int) -> list:
        """
        Remove old requests outside the time window.
        
        Args:
            requests: List of request timestamps
            window_seconds: Time window in seconds
            
        Returns:
            Cleaned list of requests
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        return [req_time for req_time in requests if req_time > cutoff_time]
    
    def check_rate_limit(self, ip_address: str, user_id: Optional[str] = None,
                        endpoint: str = "default") -> RateLimitInfo:
        """
        Check if request is within rate limits.
        
        Args:
            ip_address: Client IP address
            user_id: User ID if authenticated
            endpoint: API endpoint being accessed
            
        Returns:
            Rate limit information
        """
        client_key = self._get_client_key(ip_address, user_id)
        current_time = time.time()
        
        # Check minute limit
        if client_key not in self.minute_requests:
            self.minute_requests[client_key] = []
        
        self.minute_requests[client_key] = self._cleanup_old_requests(
            self.minute_requests[client_key], 60
        )
        
        if len(self.minute_requests[client_key]) >= self.config.requests_per_minute:
            reset_time = int(current_time + 60)
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                limit=self.config.requests_per_minute,
                retry_after=60
            )
        
        # Check hour limit
        if client_key not in self.hour_requests:
            self.hour_requests[client_key] = []
        
        self.hour_requests[client_key] = self._cleanup_old_requests(
            self.hour_requests[client_key], 3600
        )
        
        if len(self.hour_requests[client_key]) >= self.config.requests_per_hour:
            reset_time = int(current_time + 3600)
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                limit=self.config.requests_per_hour,
                retry_after=3600
            )
        
        # Check day limit
        if client_key not in self.day_requests:
            self.day_requests[client_key] = []
        
        self.day_requests[client_key] = self._cleanup_old_requests(
            self.day_requests[client_key], 86400
        )
        
        if len(self.day_requests[client_key]) >= self.config.requests_per_day:
            reset_time = int(current_time + 86400)
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                limit=self.config.requests_per_day,
                retry_after=86400
            )
        
        # Check burst limit
        if client_key not in self.burst_requests:
            self.burst_requests[client_key] = []
        
        self.burst_requests[client_key] = self._cleanup_old_requests(
            self.burst_requests[client_key], self.config.burst_window
        )
        
        if len(self.burst_requests[client_key]) >= self.config.burst_limit:
            reset_time = int(current_time + self.config.burst_window)
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                limit=self.config.burst_limit,
                retry_after=self.config.burst_window
            )
        
        # Request is allowed
        self.minute_requests[client_key].append(current_time)
        self.hour_requests[client_key].append(current_time)
        self.day_requests[client_key].append(current_time)
        self.burst_requests[client_key].append(current_time)
        
        remaining_minute = self.config.requests_per_minute - len(self.minute_requests[client_key])
        remaining_hour = self.config.requests_per_hour - len(self.hour_requests[client_key])
        remaining_day = self.config.requests_per_day - len(self.day_requests[client_key])
        
        # Return the most restrictive remaining count
        remaining = min(remaining_minute, remaining_hour, remaining_day)
        
        return RateLimitInfo(
            allowed=True,
            remaining=remaining,
            reset_time=int(current_time + 60),  # Next minute reset
            limit=self.config.requests_per_minute
        )
    
    def get_rate_limit_status(self, ip_address: str, user_id: Optional[str] = None) -> Dict[str, int]:
        """
        Get current rate limit status for a client.
        
        Args:
            ip_address: Client IP address
            user_id: User ID if authenticated
            
        Returns:
            Dictionary with current request counts
        """
        client_key = self._get_client_key(ip_address, user_id)
        current_time = time.time()
        
        minute_count = len(self._cleanup_old_requests(
            self.minute_requests.get(client_key, []), 60
        ))
        hour_count = len(self._cleanup_old_requests(
            self.hour_requests.get(client_key, []), 3600
        ))
        day_count = len(self._cleanup_old_requests(
            self.day_requests.get(client_key, []), 86400
        ))
        burst_count = len(self._cleanup_old_requests(
            self.burst_requests.get(client_key, []), self.config.burst_window
        ))
        
        return {
            "minute_requests": minute_count,
            "hour_requests": hour_count,
            "day_requests": day_count,
            "burst_requests": burst_count,
            "minute_limit": self.config.requests_per_minute,
            "hour_limit": self.config.requests_per_hour,
            "day_limit": self.config.requests_per_day,
            "burst_limit": self.config.burst_limit
        }
    
    def reset_rate_limit(self, ip_address: str, user_id: Optional[str] = None) -> None:
        """
        Reset rate limit for a client.
        
        Args:
            ip_address: Client IP address
            user_id: User ID if authenticated
        """
        client_key = self._get_client_key(ip_address, user_id)
        
        self.minute_requests.pop(client_key, None)
        self.hour_requests.pop(client_key, None)
        self.day_requests.pop(client_key, None)
        self.burst_requests.pop(client_key, None)
        
        logger.info(f"Rate limit reset for client: {client_key}")
    
    def cleanup_expired_entries(self) -> int:
        """
        Clean up expired rate limit entries.
        
        Returns:
            Number of entries cleaned up
        """
        current_time = time.time()
        cleaned_count = 0
        
        # Clean up minute requests
        for client_key in list(self.minute_requests.keys()):
            old_count = len(self.minute_requests[client_key])
            self.minute_requests[client_key] = self._cleanup_old_requests(
                self.minute_requests[client_key], 60
            )
            if len(self.minute_requests[client_key]) == 0:
                del self.minute_requests[client_key]
            cleaned_count += old_count - len(self.minute_requests.get(client_key, []))
        
        # Clean up hour requests
        for client_key in list(self.hour_requests.keys()):
            old_count = len(self.hour_requests[client_key])
            self.hour_requests[client_key] = self._cleanup_old_requests(
                self.hour_requests[client_key], 3600
            )
            if len(self.hour_requests[client_key]) == 0:
                del self.hour_requests[client_key]
            cleaned_count += old_count - len(self.hour_requests.get(client_key, []))
        
        # Clean up day requests
        for client_key in list(self.day_requests.keys()):
            old_count = len(self.day_requests[client_key])
            self.day_requests[client_key] = self._cleanup_old_requests(
                self.day_requests[client_key], 86400
            )
            if len(self.day_requests[client_key]) == 0:
                del self.day_requests[client_key]
            cleaned_count += old_count - len(self.day_requests.get(client_key, []))
        
        # Clean up burst requests
        for client_key in list(self.burst_requests.keys()):
            old_count = len(self.burst_requests[client_key])
            self.burst_requests[client_key] = self._cleanup_old_requests(
                self.burst_requests[client_key], self.config.burst_window
            )
            if len(self.burst_requests[client_key]) == 0:
                del self.burst_requests[client_key]
            cleaned_count += old_count - len(self.burst_requests.get(client_key, []))
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired rate limit entries")
        
        return cleaned_count
