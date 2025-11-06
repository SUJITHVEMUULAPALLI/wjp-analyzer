"""
Path Manager Service
===================

Centralized service for managing file paths and directory structures.
"""

import os
from typing import Optional
from ..constants import ProcessStage, STAGE_FOLDERS, FILE_EXTENSIONS, DEFAULT_VERSION

class PathManager:
    """Manages file paths and directory structures for the application."""
    
    def __init__(self, base_path: str):
        """Initialize the path manager.
        
        Args:
            base_path (str): Base directory for all file operations
        """
        self.base_path = base_path

    def get_process_path(
        self,
        design_code: str,
        material_code: str,
        thickness_mm: float,
        process_stage: ProcessStage,
        extension: Optional[str] = None,
        version: str = DEFAULT_VERSION,
        create_dirs: bool = True
    ) -> str:
        """Generate a standardized file path for process outputs.
        
        Args:
            design_code (str): Unique design identifier
            material_code (str): Material code (e.g., STST, ALUM)
            thickness_mm (float): Material thickness in millimeters
            process_stage (ProcessStage): Current process stage
            extension (str, optional): File extension. If None, uses default for stage
            version (str, optional): Version string. Defaults to V1
            create_dirs (bool, optional): Create directories if they don't exist
            
        Returns:
            str: Complete file path
        """
        # Get stage folder
        stage_folder = STAGE_FOLDERS[process_stage]
        
        # Get file extension (use default if not specified)
        if extension is None:
            extension = FILE_EXTENSIONS.get(process_stage.value, "txt")
        
        # Build the directory structure
        dir_path = os.path.join(
            self.base_path,
            stage_folder,
            f"{material_code}_{thickness_mm}mm",
            design_code
        )
        
        # Create directories if needed
        if create_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # Build the complete file path
        filename = f"{design_code}_{process_stage.value}_{version}.{extension}"
        return os.path.join(dir_path, filename)

    def get_temp_path(self, prefix: str, extension: str) -> str:
        """Generate a temporary file path.
        
        Args:
            prefix (str): Prefix for the temporary file
            extension (str): File extension
            
        Returns:
            str: Path to temporary file
        """
        import tempfile
        temp_dir = os.path.join(self.base_path, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a unique temporary filename
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f"{prefix}_",
            suffix=f".{extension}",
            dir=temp_dir,
            delete=False
        )
        return temp_file.name

    def ensure_directory_exists(self, path: str) -> None:
        """Ensure a directory exists, creating it if necessary.
        
        Args:
            path (str): Directory path to check/create
        """
        os.makedirs(path, exist_ok=True)

    def clean_temp_files(self, max_age_hours: int = 24) -> None:
        """Clean up temporary files older than specified age.
        
        Args:
            max_age_hours (int): Maximum age of temp files in hours
        """
        import time
        temp_dir = os.path.join(self.base_path, "temp")
        if not os.path.exists(temp_dir):
            return
            
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass  # Ignore errors during cleanup