#!/usr/bin/env python3
"""
WJP Automation Pipeline - File Naming and Folder Structure
==========================================================

This module implements the standardized file naming and folder structure
for the WJP automation pipeline.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ProcessStage(Enum):
    """Process stages in the WJP pipeline."""
    DESIGN = "DESIGN"
    RAW = "RAW"
    CONVERT = "CONVERT"
    ANALYSIS = "ANALYSIS"
    REPORT = "REPORT"
    NC = "NC"
    LAY = "LAY"
    NEST = "NEST"

class MaterialCode(Enum):
    """Material codes for file naming."""
    TANB = "TANB"  # Tan Brown Granite
    MARB = "MARB"  # Marble
    STST = "STST"  # Stainless Steel
    ALUM = "ALUM"  # Aluminum
    BRAS = "BRAS"  # Brass
    GENERIC = "GENE"  # Generic

@dataclass
class JobMetadata:
    """Job metadata structure for JSON files."""
    design_code: str
    material: str
    thickness_mm: int
    category: str
    dimensions_inch: list
    cut_spacing_mm: float
    min_radius_mm: float
    prompt_used: str
    next_stage: str
    timestamp: str
    version: str = "V1"

class WJPFileManager:
    """Manages file naming and folder structure for WJP automation."""
    
    def __init__(self, base_dir: str = "WJP_PROJECTS"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create main project folders
        self._create_folder_structure()
    
    def _create_folder_structure(self):
        """Create the standard WJP folder structure."""
        folders = [
            "01_DESIGNER",
            "02_CONVERTED_DXF", 
            "03_ANALYZED",
            "04_REPORTS",
            "05_ARCHIVE"
        ]
        
        for folder in folders:
            (self.base_dir / folder).mkdir(exist_ok=True)
    
    def generate_filename(self, 
                         design_code: str,
                         material_code: str,
                         thickness_mm: int,
                         process_stage: ProcessStage,
                         version: str = "V1",
                         extension: str = "dxf") -> str:
        """Generate standardized filename."""
        date_str = datetime.now().strftime("%Y%m%d")
        
        filename = f"WJP_{design_code}_{material_code}_{thickness_mm}_{process_stage.value}_{version}_{date_str}.{extension}"
        
        return filename
    
    def get_project_folder(self, design_code: str) -> Path:
        """Get project folder for a design code."""
        project_folder = self.base_dir / design_code
        project_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        subfolders = [
            "01_DESIGNER",
            "02_CONVERTED_DXF",
            "03_ANALYZED", 
            "04_REPORTS",
            "05_ARCHIVE"
        ]
        
        for subfolder in subfolders:
            (project_folder / subfolder).mkdir(exist_ok=True)
        
        return project_folder
    
    def get_stage_folder(self, design_code: str, stage: str) -> Path:
        """Get folder for specific processing stage."""
        project_folder = self.get_project_folder(design_code)
        
        stage_mapping = {
            "designer": "01_DESIGNER",
            "converted_dxf": "02_CONVERTED_DXF",
            "analyzed": "03_ANALYZED",
            "reports": "04_REPORTS",
            "archive": "05_ARCHIVE"
        }
        
        stage_folder = stage_mapping.get(stage, "05_ARCHIVE")
        return project_folder / stage_folder
    
    def save_metadata(self, 
                     design_code: str,
                     metadata: JobMetadata,
                     stage: str) -> str:
        """Save job metadata to JSON file."""
        stage_folder = self.get_stage_folder(design_code, stage)
        
        # Generate metadata filename
        filename = self.generate_filename(
            design_code=design_code,
            material_code=metadata.material.replace(" ", "").upper()[:4],
            thickness_mm=metadata.thickness_mm,
            process_stage=ProcessStage.DESIGN,  # Metadata files use DESIGN stage
            version=metadata.version,
            extension="json"
        )
        
        # Replace DESIGN with META in filename
        filename = filename.replace("_DESIGN_", "_META_")
        
        filepath = stage_folder / filename
        
        # Save metadata
        with open(filepath, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        return str(filepath)
    
    def load_metadata(self, filepath: str) -> JobMetadata:
        """Load job metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return JobMetadata(**data)
    
    def get_file_path(self, 
                     design_code: str,
                     material_code: str,
                     thickness_mm: int,
                     process_stage: ProcessStage,
                     stage_folder: str,
                     version: str = "V1",
                     extension: str = "dxf") -> str:
        """Get full file path for a specific file."""
        stage_path = self.get_stage_folder(design_code, stage_folder)
        filename = self.generate_filename(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=thickness_mm,
            process_stage=process_stage,
            version=version,
            extension=extension
        )
        
        return str(stage_path / filename)
    
    def archive_files(self, design_code: str, files_to_archive: list):
        """Archive files to the archive folder."""
        archive_folder = self.get_stage_folder(design_code, "archive")
        
        for file_path in files_to_archive:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                archive_path = archive_folder / filename
                
                # Move file to archive
                os.rename(file_path, archive_path)
                print(f"📦 Archived: {filename}")

def test_file_manager():
    """Test the WJP file manager."""
    print("🗂️ **Testing WJP File Manager**")
    print("=" * 50)
    
    # Create file manager
    fm = WJPFileManager()
    
    # Test filename generation
    print("📝 **Testing Filename Generation:**")
    
    test_cases = [
        ("SR06", "TANB", 25, ProcessStage.DESIGN, "png"),
        ("SR06", "TANB", 25, ProcessStage.RAW, "dxf"),
        ("SR06", "TANB", 25, ProcessStage.ANALYSIS, "json"),
        ("SR06", "TANB", 25, ProcessStage.REPORT, "pdf"),
        ("SR06", "TANB", 25, ProcessStage.NC, "nc")
    ]
    
    for design_code, material_code, thickness, stage, ext in test_cases:
        filename = fm.generate_filename(design_code, material_code, thickness, stage, "V1", ext)
        print(f"   {stage.value:<10}: {filename}")
    
    # Test metadata creation
    print("\n📋 **Testing Metadata Creation:**")
    
    metadata = JobMetadata(
        design_code="SR06",
        material="Tan Brown Granite",
        thickness_mm=25,
        category="Inlay Tile",
        dimensions_inch=[24, 24],
        cut_spacing_mm=3.0,
        min_radius_mm=2.0,
        prompt_used="Waterjet-safe Tan Brown granite tile with white marble inlay, 24x24 inch",
        next_stage="image_to_dxf",
        timestamp=datetime.now().isoformat()
    )
    
    # Save metadata
    metadata_path = fm.save_metadata("SR06", metadata, "designer")
    print(f"   ✅ Metadata saved: {os.path.basename(metadata_path)}")
    
    # Load metadata
    loaded_metadata = fm.load_metadata(metadata_path)
    print(f"   ✅ Metadata loaded: {loaded_metadata.design_code}")
    
    # Test file paths
    print("\n📁 **Testing File Paths:**")
    
    file_paths = [
        ("SR06", "TANB", 25, ProcessStage.DESIGN, "designer", "png"),
        ("SR06", "TANB", 25, ProcessStage.RAW, "converted_dxf", "dxf"),
        ("SR06", "TANB", 25, ProcessStage.ANALYSIS, "analyzed", "json"),
        ("SR06", "TANB", 25, ProcessStage.REPORT, "reports", "pdf")
    ]
    
    for design_code, material_code, thickness, stage, folder, ext in file_paths:
        path = fm.get_file_path(design_code, material_code, thickness, stage, folder, "V1", ext)
        print(f"   {stage.value:<10}: {os.path.basename(path)}")
    
    print("\n🎉 **WJP File Manager Test Completed Successfully!**")
    
    return fm

if __name__ == "__main__":
    test_file_manager()
