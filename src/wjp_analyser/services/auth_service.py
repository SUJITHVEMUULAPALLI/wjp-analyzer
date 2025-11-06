"""
Authentication Service
=====================

Centralized service for handling authentication and authorization.
"""

from typing import Optional, List
from datetime import datetime, timedelta
import jwt
from ..database.models import User
from ..config.config_manager import ConfigManager

class AuthService:
    """Handles authentication and authorization."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the auth service.
        
        Args:
            config_manager (ConfigManager): Application configuration manager
        """
        self.config = config_manager
        self.secret_key = self.config.get_security_config()["secret_key"]
        self.token_expiry = timedelta(hours=24)

    def create_access_token(self, user_id: str, permissions: List[str]) -> str:
        """Create a new JWT access token.
        
        Args:
            user_id (str): User identifier
            permissions (List[str]): List of user permissions
            
        Returns:
            str: JWT token
        """
        expiration = datetime.utcnow() + self.token_expiry
        
        token_data = {
            "sub": user_id,
            "permissions": permissions,
            "exp": expiration
        }
        
        return jwt.encode(token_data, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token.
        
        Args:
            token (str): JWT token to verify
            
        Returns:
            Optional[dict]: Decoded token data if valid, None otherwise
        """
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return None

    def check_permission(self, required_permission: str, user: User) -> bool:
        """Check if a user has the required permission.
        
        Args:
            required_permission (str): Required permission name
            user (User): User to check
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        if not user:
            return False
            
        # Admin users have all permissions
        if getattr(user, "roles", None) and "admin" in user.roles:
            return True
            
        user_permissions = set(getattr(user, "permissions", []) or [])
        return required_permission in user_permissions

    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify an API key and return associated user ID.
        
        Args:
            api_key (str): API key to verify
            
        Returns:
            Optional[str]: User ID if valid, None otherwise
        """
        # Get API key configuration
        api_keys = self.config.get_security_config().get("api_keys", {})
        
        # Check if API key exists and is valid
        if api_key in api_keys:
            key_data = api_keys[api_key]
            expiry = datetime.fromisoformat(key_data["expiry"])
            
            if expiry > datetime.utcnow():
                return key_data["user_id"]
        
        return None

    def get_user_permissions(self, user: User) -> List[str]:
        """Get list of permissions for a user.
        
        Args:
            user (User): User object
            
        Returns:
            List[str]: List of permission strings
        """
        permissions = set(user.permissions or [])
        
        # Add role-based permissions
        role_permissions = self.config.get_security_config().get("role_permissions", {})
        for role in (user.roles or []):
            if role in role_permissions:
                permissions.update(role_permissions[role])
        
        return list(permissions)