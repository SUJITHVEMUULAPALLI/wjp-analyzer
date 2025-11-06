"""
Authentication and Authorization Module
======================================

This module provides JWT-based authentication and role-based access control
for the WJP ANALYSER system.

Key Features:
- JWT token management
- Role-based access control (RBAC)
- Password hashing with bcrypt
- Session management
- Security audit logging
"""

from .auth_manager import AuthManager
from .rbac import Role, Permission, RBACManager
from .jwt_handler import JWTHandler
from .password_manager import PasswordManager
from .audit_logger import SecurityAuditLogger

__all__ = [
    'AuthManager',
    'Role', 
    'Permission',
    'RBACManager',
    'JWTHandler',
    'PasswordManager',
    'SecurityAuditLogger'
]
