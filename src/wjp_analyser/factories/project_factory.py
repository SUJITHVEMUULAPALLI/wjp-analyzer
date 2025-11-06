"""
Project Factory
==============

Factory for creating and managing projects.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from ..database.models import Project, User
from ..services.path_manager import PathManager

class ProjectFactory:
    """Factory for creating and managing projects."""
    
    def __init__(self, path_manager: PathManager):
        """Initialize the project factory.
        
        Args:
            path_manager (PathManager): Path management service
        """
        self.path_manager = path_manager

    def create_project(
        self,
        user: User,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Project:
        """Create a new project.
        
        Args:
            user (User): Project owner
            name (str): Project name
            description (Optional[str]): Project description
            settings (Optional[Dict[str, Any]]): Project settings
            
        Returns:
            Project: Created project instance
        """
        # Create project instance
        project = Project(
            user_id=user.id,
            name=name,
            description=description,
            settings=settings or {},
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Create project directory structure
        project_path = self.path_manager.get_process_path(
            design_code=project.id,
            material_code="NONE",
            thickness_mm=0,
            process_stage="project",
            extension=None,
            create_dirs=True
        )
        
        # Create necessary subdirectories
        subdirs = ["uploads", "analysis", "conversion", "reports"]
        for subdir in subdirs:
            self.path_manager.ensure_directory_exists(
                os.path.join(project_path, subdir)
            )
        
        return project

    def archive_project(self, project: Project) -> None:
        """Archive a project.
        
        Args:
            project (Project): Project to archive
        """
        project.status = "archived"
        project.updated_at = datetime.utcnow()

    def delete_project(self, project: Project) -> None:
        """Delete a project.
        
        Args:
            project (Project): Project to delete
        """
        project.status = "deleted"
        project.updated_at = datetime.utcnow()

    def update_project(
        self,
        project: Project,
        name: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update project details.
        
        Args:
            project (Project): Project to update
            name (Optional[str]): New project name
            description (Optional[str]): New project description
            settings (Optional[Dict[str, Any]]): New project settings
        """
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if settings is not None:
            project.settings.update(settings)
            
        project.updated_at = datetime.utcnow()