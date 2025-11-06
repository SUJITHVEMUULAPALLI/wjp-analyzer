"""
Role-Based Access Control (RBAC)
================================

Implements role-based access control system for WJP ANALYSER.
"""

from enum import Enum
from typing import List, Set, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    
    # Project management
    CREATE_PROJECT = "create_project"
    READ_PROJECT = "read_project"
    UPDATE_PROJECT = "update_project"
    DELETE_PROJECT = "delete_project"
    
    # DXF analysis
    ANALYZE_DXF = "analyze_dxf"
    READ_ANALYSIS = "read_analysis"
    EXPORT_ANALYSIS = "export_analysis"
    
    # Image conversion
    CONVERT_IMAGE = "convert_image"
    READ_CONVERSION = "read_conversion"
    EXPORT_CONVERSION = "export_conversion"
    
    # Nesting
    CREATE_NESTING = "create_nesting"
    READ_NESTING = "read_nesting"
    OPTIMIZE_NESTING = "optimize_nesting"
    
    # System administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_CONFIG = "manage_config"
    
    # AI features
    USE_AI_ANALYSIS = "use_ai_analysis"
    MANAGE_AI_MODELS = "manage_ai_models"


class Role(Enum):
    """System roles."""
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class RolePermission:
    """Role permission mapping."""
    role: Role
    permissions: Set[Permission]


class RBACManager:
    """Manages role-based access control."""
    
    def __init__(self):
        """Initialize RBAC manager with default role permissions."""
        self.role_permissions = self._initialize_role_permissions()
    
    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role permissions."""
        return {
            Role.GUEST: {
                Permission.READ_PROJECT,
                Permission.READ_ANALYSIS,
                Permission.READ_CONVERSION,
                Permission.READ_NESTING,
            },
            
            Role.USER: {
                # All guest permissions
                Permission.READ_PROJECT,
                Permission.READ_ANALYSIS,
                Permission.READ_CONVERSION,
                Permission.READ_NESTING,
                
                # User-specific permissions
                Permission.CREATE_PROJECT,
                Permission.UPDATE_PROJECT,
                Permission.DELETE_PROJECT,
                Permission.ANALYZE_DXF,
                Permission.CONVERT_IMAGE,
                Permission.CREATE_NESTING,
                Permission.USE_AI_ANALYSIS,
                Permission.EXPORT_ANALYSIS,
                Permission.EXPORT_CONVERSION,
            },
            
            Role.POWER_USER: {
                # All user permissions
                Permission.READ_PROJECT,
                Permission.READ_ANALYSIS,
                Permission.READ_CONVERSION,
                Permission.READ_NESTING,
                Permission.CREATE_PROJECT,
                Permission.UPDATE_PROJECT,
                Permission.DELETE_PROJECT,
                Permission.ANALYZE_DXF,
                Permission.CONVERT_IMAGE,
                Permission.CREATE_NESTING,
                Permission.USE_AI_ANALYSIS,
                Permission.EXPORT_ANALYSIS,
                Permission.EXPORT_CONVERSION,
                
                # Power user specific
                Permission.OPTIMIZE_NESTING,
                Permission.READ_USER,
            },
            
            Role.ADMIN: {
                # All power user permissions
                Permission.READ_PROJECT,
                Permission.READ_ANALYSIS,
                Permission.READ_CONVERSION,
                Permission.READ_NESTING,
                Permission.CREATE_PROJECT,
                Permission.UPDATE_PROJECT,
                Permission.DELETE_PROJECT,
                Permission.ANALYZE_DXF,
                Permission.CONVERT_IMAGE,
                Permission.CREATE_NESTING,
                Permission.USE_AI_ANALYSIS,
                Permission.EXPORT_ANALYSIS,
                Permission.EXPORT_CONVERSION,
                Permission.OPTIMIZE_NESTING,
                Permission.READ_USER,
                
                # Admin specific
                Permission.CREATE_USER,
                Permission.UPDATE_USER,
                Permission.DELETE_USER,
                Permission.MANAGE_SYSTEM,
                Permission.VIEW_AUDIT_LOGS,
                Permission.MANAGE_CONFIG,
                Permission.MANAGE_AI_MODELS,
            },
            
            Role.SUPER_ADMIN: {
                # All permissions
                *[p for p in Permission]
            }
        }
    
    def has_permission(self, role: Role, permission: Permission) -> bool:
        """
        Check if a role has a specific permission.
        
        Args:
            role: User role
            permission: Required permission
            
        Returns:
            True if role has permission, False otherwise
        """
        role_perms = self.role_permissions.get(role, set())
        has_perm = permission in role_perms
        logger.debug(f"Permission check: {role.value} -> {permission.value}: {has_perm}")
        return has_perm
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """
        Get all permissions for a role.
        
        Args:
            role: User role
            
        Returns:
            Set of permissions for the role
        """
        return self.role_permissions.get(role, set())
    
    def can_access_resource(self, role: Role, resource: str, action: str) -> bool:
        """
        Check if a role can access a resource with a specific action.
        
        Args:
            role: User role
            resource: Resource name (e.g., 'project', 'user')
            action: Action name (e.g., 'create', 'read', 'update', 'delete')
            
        Returns:
            True if access is allowed, False otherwise
        """
        permission_name = f"{action}_{resource}"
        
        try:
            permission = Permission(permission_name)
            return self.has_permission(role, permission)
        except ValueError:
            logger.warning(f"Unknown permission: {permission_name}")
            return False
    
    def add_role_permission(self, role: Role, permission: Permission) -> None:
        """
        Add a permission to a role.
        
        Args:
            role: Role to modify
            permission: Permission to add
        """
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        
        self.role_permissions[role].add(permission)
        logger.info(f"Added permission {permission.value} to role {role.value}")
    
    def remove_role_permission(self, role: Role, permission: Permission) -> None:
        """
        Remove a permission from a role.
        
        Args:
            role: Role to modify
            permission: Permission to remove
        """
        if role in self.role_permissions:
            self.role_permissions[role].discard(permission)
            logger.info(f"Removed permission {permission.value} from role {role.value}")
    
    def create_custom_role(self, role_name: str, permissions: Set[Permission]) -> Role:
        """
        Create a custom role with specific permissions.
        
        Args:
            role_name: Name of the custom role
            permissions: Set of permissions for the role
            
        Returns:
            New custom role
        """
        # In a real implementation, you would store custom roles in a database
        # For now, we'll create a new enum value dynamically
        logger.info(f"Created custom role '{role_name}' with {len(permissions)} permissions")
        
        # This is a simplified implementation
        # In production, you'd want to use a more robust approach
        custom_role = Role(role_name.lower())
        self.role_permissions[custom_role] = permissions
        return custom_role
