"""
JWT Token Handler
=================

Handles JWT token creation, validation, and refresh for authentication.
"""

import jwt
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT token payload structure."""
    user_id: str
    email: str
    role: str
    permissions: list
    iat: datetime
    exp: datetime
    jti: str  # JWT ID for token revocation


class JWTHandler:
    """Handles JWT token operations."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize JWT handler.
        
        Args:
            secret_key: JWT secret key. If None, uses environment variable.
        """
        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY')
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY environment variable is required")
        
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(self, user_id: str, email: str, role: str, permissions: list) -> str:
        """
        Create a new access token.
        
        Args:
            user_id: User identifier
            email: User email
            role: User role
            permissions: List of user permissions
            
        Returns:
            JWT access token
        """
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "email": email,
            "role": role,
            "permissions": permissions,
            "iat": now,
            "exp": now + timedelta(minutes=self.access_token_expire_minutes),
            "jti": f"{user_id}_{now.timestamp()}"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created access token for user {user_id}")
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create a new refresh token.
        
        Args:
            user_id: User identifier
            
        Returns:
            JWT refresh token
        """
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(days=self.refresh_token_expire_days),
            "jti": f"refresh_{user_id}_{now.timestamp()}"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created refresh token for user {user_id}")
        return token
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            TokenPayload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return TokenPayload(
                user_id=payload["user_id"],
                email=payload["email"],
                role=payload["role"],
                permissions=payload["permissions"],
                iat=datetime.fromtimestamp(payload["iat"]),
                exp=datetime.fromtimestamp(payload["exp"]),
                jti=payload["jti"]
            )
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Create a new access token from a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token if refresh token is valid, None otherwise
        """
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                logger.warning("Token is not a refresh token")
                return None
            
            user_id = payload["user_id"]
            
            # In a real implementation, you would fetch user data from database
            # For now, we'll return a placeholder
            return self.create_access_token(
                user_id=user_id,
                email="user@example.com",  # Fetch from database
                role="user",  # Fetch from database
                permissions=[]  # Fetch from database
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
