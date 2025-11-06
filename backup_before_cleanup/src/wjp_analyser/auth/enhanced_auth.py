"""
Enhanced Authentication System with JWT and RBAC
================================================

Comprehensive authentication system with role-based access control.
"""

import jwt
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles with hierarchical permissions."""
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(Enum):
    """System permissions."""
    # Authentication
    LOGIN = "auth:login"
    REGISTER = "auth:register"
    LOGOUT = "auth:logout"
    
    # User Management
    VIEW_PROFILE = "user:view_profile"
    EDIT_PROFILE = "user:edit_profile"
    CHANGE_PASSWORD = "user:change_password"
    
    # Project Management
    CREATE_PROJECT = "project:create"
    VIEW_PROJECT = "project:view"
    EDIT_PROJECT = "project:edit"
    DELETE_PROJECT = "project:delete"
    SHARE_PROJECT = "project:share"
    
    # Analysis Operations
    ANALYZE_DXF = "analysis:analyze_dxf"
    VIEW_ANALYSIS = "analysis:view"
    EXPORT_ANALYSIS = "analysis:export"
    
    # Image Processing
    CONVERT_IMAGE = "conversion:convert_image"
    VIEW_CONVERSION = "conversion:view"
    EXPORT_CONVERSION = "conversion:export"
    
    # Nesting Operations
    CREATE_NESTING = "nesting:create"
    VIEW_NESTING = "nesting:view"
    OPTIMIZE_NESTING = "nesting:optimize"
    
    # System Administration
    VIEW_USERS = "admin:view_users"
    EDIT_USERS = "admin:edit_users"
    DELETE_USERS = "admin:delete_users"
    VIEW_SYSTEM_LOGS = "admin:view_logs"
    MANAGE_SYSTEM = "admin:manage_system"
    
    # API Access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"


@dataclass
class User:
    """User data structure."""
    id: str
    email: str
    password_hash: str
    role: UserRole
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class RBACManager:
    """Role-Based Access Control Manager."""
    
    def __init__(self):
        self.role_permissions = {
            UserRole.GUEST: [
                Permission.LOGIN,
                Permission.VIEW_PROFILE,
                Permission.VIEW_PROJECT,
                Permission.VIEW_ANALYSIS,
                Permission.VIEW_CONVERSION,
                Permission.VIEW_NESTING,
                Permission.API_READ
            ],
            UserRole.USER: [
                Permission.LOGIN,
                Permission.REGISTER,
                Permission.LOGOUT,
                Permission.VIEW_PROFILE,
                Permission.EDIT_PROFILE,
                Permission.CHANGE_PASSWORD,
                Permission.CREATE_PROJECT,
                Permission.VIEW_PROJECT,
                Permission.EDIT_PROJECT,
                Permission.SHARE_PROJECT,
                Permission.ANALYZE_DXF,
                Permission.VIEW_ANALYSIS,
                Permission.EXPORT_ANALYSIS,
                Permission.CONVERT_IMAGE,
                Permission.VIEW_CONVERSION,
                Permission.EXPORT_CONVERSION,
                Permission.CREATE_NESTING,
                Permission.VIEW_NESTING,
                Permission.OPTIMIZE_NESTING,
                Permission.API_READ,
                Permission.API_WRITE
            ],
            UserRole.POWER_USER: [
                # All USER permissions plus:
                Permission.DELETE_PROJECT,
            ],
            UserRole.ADMIN: [
                # All POWER_USER permissions plus:
                Permission.VIEW_USERS,
                Permission.EDIT_USERS,
                Permission.VIEW_SYSTEM_LOGS,
                Permission.API_ADMIN
            ],
            UserRole.SUPER_ADMIN: [
                # All permissions
                *[p for p in Permission]
            ]
        }
    
    def get_role_permissions(self, role: UserRole) -> List[Permission]:
        """Get permissions for a role."""
        return self.role_permissions.get(role, [])
    
    def has_permission(self, role: UserRole, permission: Permission) -> bool:
        """Check if role has specific permission."""
        return permission in self.get_role_permissions(role)
    
    def can_access_resource(self, role: UserRole, resource: str, action: str) -> bool:
        """Check if role can perform action on resource."""
        permission_name = f"{resource}:{action}"
        try:
            permission = Permission(permission_name)
            return self.has_permission(role, permission)
        except ValueError:
            # Unknown permission, deny by default
            return False


class PasswordManager:
    """Secure password management."""
    
    def __init__(self):
        self.min_length = 8
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        
        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "strength": self._calculate_strength(password)
        }
    
    def _calculate_strength(self, password: str) -> str:
        """Calculate password strength."""
        score = 0
        
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if any(c.isupper() for c in password):
            score += 1
        if any(c.islower() for c in password):
            score += 1
        if any(c.isdigit() for c in password):
            score += 1
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        
        if score <= 2:
            return "weak"
        elif score <= 4:
            return "medium"
        else:
            return "strong"
    
    def hash_password(self, password: str) -> str:
        """Hash password using PBKDF2."""
        salt = secrets.token_hex(32)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return f"{salt}:{pwd_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hash_hex = password_hash.split(':')
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return pwd_hash.hex() == hash_hex
        except (ValueError, TypeError):
            return False


class JWTHandler:
    """JWT token management."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY', 'change-this-in-production')
        self.algorithm = 'HS256'
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(self, user_id: str, email: str, role: UserRole, permissions: List[Permission]) -> str:
        """Create access token."""
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'email': email,
            'role': role.value,
            'permissions': [p.value for p in permissions],
            'iat': now,
            'exp': now + timedelta(minutes=self.access_token_expire_minutes),
            'type': 'access'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created access token for user {user_id}")
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token."""
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'iat': now,
            'exp': now + timedelta(days=self.refresh_token_expire_days),
            'type': 'refresh'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created refresh token for user {user_id}")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token."""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get('type') != 'refresh':
            return None
        
        user_id = payload.get('sub')
        if not user_id:
            return None
        
        # Get user data (this would typically come from database)
        # For now, we'll create a basic token
        return self.create_access_token(
            user_id=user_id,
            email=payload.get('email', ''),
            role=UserRole.USER,  # Default role
            permissions=[]
        )


class AuthManager:
    """Main authentication manager."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.password_manager = PasswordManager()
        self.jwt_handler = JWTHandler()
        self.rbac_manager = RBACManager()
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_email = "admin@wjp-analyser.com"
        admin_password = "Admin123!@#"
        
        if admin_email not in self.users:
            self.register_user(
                email=admin_email,
                password=admin_password,
                role=UserRole.SUPER_ADMIN,
                first_name="System",
                last_name="Administrator"
            )
            logger.info("Created default admin user")
    
    def register_user(self, email: str, password: str, role: UserRole = UserRole.USER,
                     first_name: Optional[str] = None, last_name: Optional[str] = None) -> Dict[str, Any]:
        """Register new user."""
        # Validate email
        if not self._is_valid_email(email):
            return {"success": False, "error": "Invalid email format"}
        
        # Check if user exists
        if email in self.users:
            return {"success": False, "error": "User already exists"}
        
        # Validate password
        password_validation = self.password_manager.validate_password(password)
        if not password_validation["valid"]:
            return {"success": False, "error": "Password validation failed", "details": password_validation["errors"]}
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        password_hash = self.password_manager.hash_password(password)
        
        user = User(
            id=user_id,
            email=email,
            password_hash=password_hash,
            role=role,
            first_name=first_name,
            last_name=last_name
        )
        
        self.users[email] = user
        
        logger.info(f"User registered: {email}")
        return {"success": True, "user_id": user_id}
    
    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user."""
        # Check if user exists
        if email not in self.users:
            logger.warning(f"Login attempt for non-existent user: {email}")
            return {"success": False, "error": "Invalid credentials"}
        
        user = self.users[email]
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            logger.warning(f"Login attempt for locked account: {email}")
            return {"success": False, "error": "Account is locked due to too many failed attempts"}
        
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failures
            if user.failed_login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
                logger.warning(f"Account locked due to failed attempts: {email}")
            
            logger.warning(f"Failed login attempt for user: {email}")
            return {"success": False, "error": "Invalid credentials"}
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Generate tokens
        permissions = self.rbac_manager.get_role_permissions(user.role)
        access_token = self.jwt_handler.create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            permissions=permissions
        )
        refresh_token = self.jwt_handler.create_refresh_token(user.id)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        logger.info(f"User authenticated successfully: {email}")
        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session_id,
            "user": {
                "id": user.id,
                "email": user.email,
                "role": user.role.value,
                "first_name": user.first_name,
                "last_name": user.last_name
            }
        }
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        payload = self.jwt_handler.verify_token(token)
        if not payload:
            return None
        
        # Check if user still exists and is active
        email = payload.get('email')
        if email not in self.users:
            return None
        
        user = self.users[email]
        if not user.is_active:
            return None
        
        return payload
    
    def has_permission(self, token: str, permission: Permission) -> bool:
        """Check if token has specific permission."""
        payload = self.verify_token(token)
        if not payload:
            return False
        
        role = UserRole(payload.get('role', UserRole.GUEST.value))
        return self.rbac_manager.has_permission(role, permission)
    
    def logout_user(self, session_id: str) -> bool:
        """Logout user by session ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"User logged out: {session_id}")
            return True
        return False
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


# Global auth manager instance
auth_manager = AuthManager()


def require_auth(permission: Optional[Permission] = None):
    """Decorator to require authentication and optionally specific permission."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be implemented based on the web framework
            # For now, we'll just return the function
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """Get current user from token."""
    payload = auth_manager.verify_token(token)
    if not payload:
        return None
    
    email = payload.get('email')
    if email in auth_manager.users:
        user = auth_manager.users[email]
        return {
            "id": user.id,
            "email": user.email,
            "role": user.role.value,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "permissions": [p.value for p in auth_manager.rbac_manager.get_role_permissions(user.role)]
        }
    
    return None
