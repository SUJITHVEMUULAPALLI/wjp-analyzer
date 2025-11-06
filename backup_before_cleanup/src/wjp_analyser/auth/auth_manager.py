"""
Authentication Manager
=====================

Main authentication manager that coordinates all authentication components.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .jwt_handler import JWTHandler, TokenPayload
from .password_manager import PasswordManager
from .rbac import RBACManager, Role, Permission
from .audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)


class AuthManager:
    """Main authentication manager."""
    
    def __init__(self, jwt_secret_key: Optional[str] = None):
        """
        Initialize authentication manager.
        
        Args:
            jwt_secret_key: JWT secret key for token signing
        """
        self.jwt_handler = JWTHandler(jwt_secret_key)
        self.password_manager = PasswordManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = SecurityAuditLogger()
        
        # In-memory user store (replace with database in production)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default admin user
        self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """Create default admin user for initial setup."""
        admin_email = "admin@wjp-analyser.com"
        admin_password = "Admin123!@#"
        
        if admin_email not in self.users:
            self.register_user(
                email=admin_email,
                password=admin_password,
                role=Role.SUPER_ADMIN,
                first_name="System",
                last_name="Administrator"
            )
            logger.info("Created default admin user")
    
    def register_user(self, email: str, password: str, role: Role = Role.USER,
                     first_name: str = "", last_name: str = "") -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            email: User email address
            password: Plain text password
            role: User role
            first_name: User's first name
            last_name: User's last name
            
        Returns:
            User data dictionary
            
        Raises:
            ValueError: If email already exists or password is weak
        """
        # Validate email uniqueness
        if email in self.users:
            raise ValueError("Email already registered")
        
        # Validate password strength
        is_valid, errors = self.password_manager.validate_password_strength(password)
        if not is_valid:
            raise ValueError(f"Password validation failed: {', '.join(errors)}")
        
        # Hash password
        password_hash = self.password_manager.hash_password(password)
        
        # Create user
        user_id = f"user_{len(self.users) + 1}"
        user_data = {
            "user_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "role": role,
            "first_name": first_name,
            "last_name": last_name,
            "created_at": datetime.utcnow(),
            "last_login": None,
            "is_active": True
        }
        
        self.users[email] = user_data
        
        # Log registration
        from .audit_logger import SecurityEvent, SecurityEventType
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            user_id=user_id,
            ip_address="127.0.0.1",  # Default for registration
            user_agent="system",
            resource="user_registration",
            action="register",
            success=True,
            details={"message": "User registered successfully"},
            timestamp=datetime.utcnow()
        )
        self.audit_logger.log_event(event)
        
        logger.info(f"User registered: {email}")
        return user_data
    
    def authenticate_user(self, email: str, password: str, ip_address: str = "127.0.0.1",
                        user_agent: str = "unknown") -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with email and password.
        
        Args:
            email: User email
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Authentication result with tokens if successful, None otherwise
        """
        # Check if user exists
        if email not in self.users:
            self.audit_logger.log_login_failure(email, ip_address, user_agent, "User not found")
            return None
        
        user_data = self.users[email]
        
        # Check if user is active
        if not user_data.get("is_active", True):
            self.audit_logger.log_login_failure(email, ip_address, user_agent, "Account deactivated")
            return None
        
        # Verify password
        if not self.password_manager.verify_password(password, user_data["password_hash"]):
            self.audit_logger.log_login_failure(email, ip_address, user_agent, "Invalid password")
            return None
        
        # Generate tokens
        role = user_data["role"]
        permissions = list(self.rbac_manager.get_role_permissions(role))
        
        access_token = self.jwt_handler.create_access_token(
            user_id=user_data["user_id"],
            email=email,
            role=role.value,
            permissions=[p.value for p in permissions]
        )
        
        refresh_token = self.jwt_handler.create_refresh_token(user_data["user_id"])
        
        # Update last login
        user_data["last_login"] = datetime.utcnow()
        
        # Create session
        session_id = f"session_{user_data['user_id']}_{datetime.utcnow().timestamp()}"
        self.sessions[session_id] = {
            "user_id": user_data["user_id"],
            "email": email,
            "role": role,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        
        # Log successful login
        self.audit_logger.log_login_success(user_data["user_id"], ip_address, user_agent)
        
        logger.info(f"User authenticated: {email}")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session_id,
            "user": {
                "user_id": user_data["user_id"],
                "email": email,
                "role": role.value,
                "permissions": [p.value for p in permissions],
                "first_name": user_data["first_name"],
                "last_name": user_data["last_name"]
            }
        }
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """
        Verify a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None otherwise
        """
        return self.jwt_handler.verify_token(token)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh an access token using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token if refresh token is valid, None otherwise
        """
        return self.jwt_handler.refresh_access_token(refresh_token)
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User identifier
            permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        # Find user by ID
        user_data = None
        for email, data in self.users.items():
            if data["user_id"] == user_id:
                user_data = data
                break
        
        if not user_data:
            return False
        
        role = user_data["role"]
        return self.rbac_manager.has_permission(role, permission)
    
    def logout_user(self, session_id: str, ip_address: str = "127.0.0.1",
                   user_agent: str = "unknown") -> bool:
        """
        Logout a user by invalidating their session.
        
        Args:
            session_id: Session identifier
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if logout successful, False otherwise
        """
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
            
            # Log logout
            self.audit_logger.log_event(
                self.audit_logger.SecurityEvent(
                    event_type=self.audit_logger.SecurityEventType.LOGOUT,
                    user_id=session_data["user_id"],
                    ip_address=ip_address,
                    user_agent=user_agent,
                    resource="authentication",
                    action="logout",
                    success=True,
                    details={"message": "User logged out successfully"},
                    timestamp=datetime.utcnow()
                )
            )
            
            # Remove session
            del self.sessions[session_id]
            logger.info(f"User logged out: {session_data['email']}")
            return True
        
        return False
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user data by user ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User data if found, None otherwise
        """
        for email, data in self.users.items():
            if data["user_id"] == user_id:
                return data
        return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user data by email.
        
        Args:
            email: User email
            
        Returns:
            User data if found, None otherwise
        """
        return self.users.get(email)
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User identifier
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully, False otherwise
        """
        user_data = self.get_user_by_id(user_id)
        if not user_data:
            return False
        
        # Verify old password
        if not self.password_manager.verify_password(old_password, user_data["password_hash"]):
            return False
        
        # Validate new password
        is_valid, errors = self.password_manager.validate_password_strength(new_password)
        if not is_valid:
            raise ValueError(f"New password validation failed: {', '.join(errors)}")
        
        # Hash new password
        new_password_hash = self.password_manager.hash_password(new_password)
        
        # Update password
        user_data["password_hash"] = new_password_hash
        
        # Log password change
        self.audit_logger.log_event(
            self.audit_logger.SecurityEvent(
                event_type=self.audit_logger.SecurityEventType.PASSWORD_CHANGE,
                user_id=user_id,
                ip_address="127.0.0.1",  # Default for password change
                user_agent="system",
                resource="password",
                action="change",
                success=True,
                details={"message": "Password changed successfully"},
                timestamp=datetime.utcnow()
            )
        )
        
        logger.info(f"Password changed for user: {user_data['email']}")
        return True
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active sessions.
        
        Returns:
            Dictionary of active sessions
        """
        return self.sessions.copy()
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        expired_sessions = []
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # 24 hour session timeout
        
        for session_id, session_data in self.sessions.items():
            if session_data["last_activity"] < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
