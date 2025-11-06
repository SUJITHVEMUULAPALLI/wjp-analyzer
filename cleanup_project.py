#!/usr/bin/env python3
"""
WJP ANALYSER - Project Cleanup Script
====================================

This script cleans up the WJP ANALYSER project by:
1. Removing duplicate and obsolete files
2. Consolidating configurations
3. Archiving old modules
4. Updating documentation

Usage:
    python cleanup_project.py [options]

Options:
    --dry-run     Show what would be deleted without actually deleting
    --backup      Create backup before cleanup
    --force       Force cleanup without confirmation
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

class WJPProjectCleanup:
    """WJP ANALYSER Project Cleanup Manager."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.setup_logging()
        
        # Files and directories to remove/archive
        self.files_to_remove = []
        self.dirs_to_remove = []
        self.files_to_archive = []
        
    def setup_logging(self):
        """Setup logging for cleanup operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('WJPCleanup')
        
    def identify_duplicate_files(self):
        """Identify duplicate files across the project."""
        self.logger.info("🔍 Identifying duplicate files...")
        
        # List of files that are duplicates or obsolete
        duplicate_files = [
            # Old launchers (replaced by unified launcher)
            "run_web_ui.py",
            "run_one_click.py", 
            "launcher.py",
            
            # Old main entry points (replaced by unified)
            "main.py",
            "app.py",
            
            # Old config files (consolidated into unified config)
            "config/app_config.yaml",
            "config/ai_config.yaml",
            "config/unified_config.yaml",
            
            # Old web interfaces (consolidated into unified)
            "src/wjp_analyser/web/app.py",
            "src/wjp_analyser/web/enhanced_app.py",
            "src/wjp_analyser/web/supervisor_dashboard.py",
            
            # Old batch files (replaced by unified launcher)
            "run_evaluation.bat",
            "run_one_click.bat", 
            "run_web_ui.bat",
            "start_wjp_analyser.bat",
            "start_wjp_analyser.ps1",
            
            # Old documentation files (consolidated)
            "AI_PROJECT_DOCUMENTATION.md",
            "AI_TRAINING_DATA.md",
            "API_DOCUMENTATION.md",
            "CONSOLIDATION_SUMMARY.md",
            "FINAL_STATUS_REPORT.md",
            "IMPLEMENTATION_SUMMARY.md",
            "INSTALLATION_GUIDE.md",
            "QUICK_START_GUIDE.md",
            "TECHNICAL_SPECIFICATIONS.md",
            "TRANSFORMATION_COMPLETE.md",
            "UPDATES_SUMMARY.md",
            "USER_MANUAL.md",
            "WJP_ANALYSER_EVALUATION_REPORT.md",
            "WJP_ANALYSER_RE_EVALUATION_REPORT.md",
            "TEST_RESULTS_SUMMARY.md",
            
            # Old test files (consolidated)
            "test_performance.py",
            "scaffold_builder.py",
            
            # Old Docker files (consolidated)
            "docker-compose.monitoring.yml",
            "docker-compose.prod.yml",
            
            # Old migration files
            "scripts/migrate_config.py",
        ]
        
        for file_path in duplicate_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.files_to_remove.append(full_path)
                self.logger.info(f"📄 Marked for removal: {file_path}")
                
    def identify_obsolete_directories(self):
        """Identify obsolete directories."""
        self.logger.info("🔍 Identifying obsolete directories...")
        
        # Directories that are obsolete or can be cleaned up
        obsolete_dirs = [
            # Old archive directories (already archived)
            "archive/old_files",
            "archive/temp_files",
            "archive/test_results",
            
            # Old output directories (can be cleaned)
            "output/temp",
            "output/analysis",
            "output/designer", 
            "output/dxf",
            "output/image_analyzer",
            "output/image_to_dxf",
            "output/learning",
            "output/reports",
            
            # Old test directories
            "test_logs",
            
            # Old cache directories
            "__pycache__",
            "src/__pycache__",
            "src/wjp_analyser/__pycache__",
            "src/wjp_analyser/web/__pycache__",
            "src/wjp_analyser/web/components/__pycache__",
            "src/wjp_analyser/web/pages/__pycache__",
        ]
        
        for dir_path in obsolete_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                if full_path.is_dir():
                    self.dirs_to_remove.append(full_path)
                    self.logger.info(f"📁 Marked for removal: {dir_path}")
                    
    def identify_files_to_archive(self):
        """Identify files that should be archived instead of deleted."""
        self.logger.info("🔍 Identifying files to archive...")
        
        # Files that should be archived for reference
        archive_files = [
            # Old standalone interfaces
            "archive/standalone_interfaces/wjp_guided_interface.py",
            "archive/standalone_interfaces/wjp_streamlit_interface.py",
            "archive/standalone_interfaces/advanced_batch_interface.py",
            "archive/standalone_interfaces/intelligent_supervisor_agent.py",
            
            # Old standalone agents
            "archive/standalone_agents/wjp_designer_agent.py",
            "archive/standalone_agents/wjp_dxf_analyzer_agent.py",
            "archive/standalone_agents/wjp_file_manager.py",
            "archive/standalone_agents/wjp_image_to_dxf_agent.py",
            "archive/standalone_agents/wjp_report_generator_agent.py",
            "archive/standalone_agents/wjp_supervisor_agent.py",
            
            # Old converters
            "archive/old_converters",
            
            # Old documentation
            "archive/docs",
            "archive/documentation",
        ]
        
        for file_path in archive_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.files_to_archive.append(full_path)
                self.logger.info(f"📦 Marked for archiving: {file_path}")
                
    def create_backup(self):
        """Create backup of the project before cleanup."""
        self.logger.info("💾 Creating backup...")
        
        backup_dir = self.project_root / "backup_before_cleanup"
        backup_dir.mkdir(exist_ok=True)
        
        # Copy important files to backup
        important_files = [
            "config",
            "src",
            "requirements.txt",
            "pyproject.toml",
            "README.md"
        ]
        
        for item in important_files:
            src = self.project_root / item
            dst = backup_dir / item
            
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                    
        self.logger.info(f"✅ Backup created at: {backup_dir}")
        
    def remove_files(self, dry_run: bool = False):
        """Remove identified files."""
        self.logger.info(f"{'🔍 DRY RUN: ' if dry_run else '🗑️ '}Removing files...")
        
        for file_path in self.files_to_remove:
            if dry_run:
                self.logger.info(f"Would remove: {file_path}")
            else:
                try:
                    file_path.unlink()
                    self.logger.info(f"✅ Removed: {file_path}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to remove {file_path}: {e}")
                    
    def remove_directories(self, dry_run: bool = False):
        """Remove identified directories."""
        self.logger.info(f"{'🔍 DRY RUN: ' if dry_run else '🗑️ '}Removing directories...")
        
        for dir_path in self.dirs_to_remove:
            if dry_run:
                self.logger.info(f"Would remove directory: {dir_path}")
            else:
                try:
                    shutil.rmtree(dir_path)
                    self.logger.info(f"✅ Removed directory: {dir_path}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to remove directory {dir_path}: {e}")
                    
    def archive_files(self, dry_run: bool = False):
        """Archive identified files."""
        self.logger.info(f"{'🔍 DRY RUN: ' if dry_run else '📦 '}Archiving files...")
        
        archive_dir = self.project_root / "archive" / "consolidated_archive"
        
        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            
        for file_path in self.files_to_archive:
            if dry_run:
                self.logger.info(f"Would archive: {file_path}")
            else:
                try:
                    # Create archive subdirectory structure
                    rel_path = file_path.relative_to(self.project_root)
                    archive_path = archive_dir / rel_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if file_path.is_dir():
                        shutil.copytree(file_path, archive_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(file_path, archive_path)
                        
                    self.logger.info(f"✅ Archived: {file_path}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to archive {file_path}: {e}")
                    
    def update_documentation(self):
        """Update documentation to reflect consolidated structure."""
        self.logger.info("📝 Updating documentation...")
        
        # Create new consolidated README
        readme_content = """# WJP ANALYSER - Unified System

## Overview
WJP ANALYSER is a comprehensive waterjet DXF analysis and processing system that has been consolidated into a single, unified application.

## Quick Start

### Launch the Unified Interface
```bash
python wjp_analyser_unified.py
```

### Available Commands
- `web-ui`: Launch unified web interface (default)
- `cli`: Launch command-line interface  
- `api`: Launch API server
- `demo`: Run demo pipeline
- `test`: Run tests
- `status`: Show system status
- `all-interfaces`: Launch all interfaces simultaneously

### Examples
```bash
# Launch web UI
python wjp_analyser_unified.py web-ui

# Launch with specific interface
python wjp_analyser_unified.py web-ui --interface streamlit

# Launch API server
python wjp_analyser_unified.py api --host 0.0.0.0 --port 5000

# Run demo
python wjp_analyser_unified.py demo

# Show system status
python wjp_analyser_unified.py status
```

## Features
- **Unified Interface**: Single entry point for all features
- **AI Analysis**: Intelligent DXF analysis and design generation
- **Image Conversion**: Convert images to DXF format
- **Nesting**: Optimize part placement for material efficiency
- **Cost Estimation**: Calculate cutting costs and time
- **Guided Mode**: Step-by-step guidance for beginners
- **Batch Processing**: Process multiple files efficiently
- **Real-time Monitoring**: Track system performance

## Configuration
All configuration is consolidated in `config/wjp_unified_config.yaml`.

## Architecture
The system has been consolidated from multiple separate applications into:
- **Unified Entry Point**: `wjp_analyser_unified.py`
- **Unified Web Interface**: `src/wjp_analyser/web/unified_web_app.py`
- **Unified Configuration**: `config/wjp_unified_config.yaml`
- **Consolidated Features**: All features integrated into single system

## Migration Notes
- Old entry points (`main.py`, `app.py`, `launcher.py`) have been consolidated
- Old web interfaces have been merged into unified interface
- Old configuration files have been consolidated
- All features are now accessible through single unified system

## Support
For issues or questions, please refer to the system status:
```bash
python wjp_analyser_unified.py status
```
"""
        
        readme_path = self.project_root / "README.md"
        readme_path.write_text(readme_content)
        self.logger.info("✅ Updated README.md")
        
    def generate_cleanup_report(self):
        """Generate a report of cleanup operations."""
        self.logger.info("📊 Generating cleanup report...")
        
        report_content = f"""# WJP ANALYSER - Project Cleanup Report

## Summary
- Files marked for removal: {len(self.files_to_remove)}
- Directories marked for removal: {len(self.dirs_to_remove)}
- Files marked for archiving: {len(self.files_to_archive)}

## Files Removed
"""
        
        for file_path in self.files_to_remove:
            report_content += f"- {file_path.relative_to(self.project_root)}\n"
            
        report_content += "\n## Directories Removed\n"
        for dir_path in self.dirs_to_remove:
            report_content += f"- {dir_path.relative_to(self.project_root)}\n"
            
        report_content += "\n## Files Archived\n"
        for file_path in self.files_to_archive:
            report_content += f"- {file_path.relative_to(self.project_root)}\n"
            
        report_content += """
## New Unified Structure
- **Entry Point**: `wjp_analyser_unified.py`
- **Web Interface**: `src/wjp_analyser/web/unified_web_app.py`
- **Configuration**: `config/wjp_unified_config.yaml`
- **Documentation**: `README.md` (consolidated)

## Benefits
- Single entry point for all functionality
- Consolidated configuration management
- Reduced complexity and maintenance overhead
- Improved user experience with unified interface
- Better organization and structure
"""
        
        report_path = self.project_root / "CLEANUP_REPORT.md"
        report_path.write_text(report_content)
        self.logger.info("✅ Generated cleanup report")
        
    def run_cleanup(self, dry_run: bool = False, backup: bool = False, force: bool = False):
        """Run the complete cleanup process."""
        self.logger.info("🚀 Starting WJP ANALYSER project cleanup...")
        
        if not force and not dry_run:
            response = input("Are you sure you want to proceed with cleanup? (y/N): ")
            if response.lower() != 'y':
                self.logger.info("❌ Cleanup cancelled by user")
                return
                
        # Create backup if requested
        if backup and not dry_run:
            self.create_backup()
            
        # Identify files and directories
        self.identify_duplicate_files()
        self.identify_obsolete_directories()
        self.identify_files_to_archive()
        
        # Perform cleanup operations
        self.remove_files(dry_run)
        self.remove_directories(dry_run)
        self.archive_files(dry_run)
        
        # Update documentation
        if not dry_run:
            self.update_documentation()
            self.generate_cleanup_report()
            
        self.logger.info("✅ Cleanup completed successfully!")
        
        if dry_run:
            self.logger.info("🔍 This was a dry run. No files were actually modified.")
        else:
            self.logger.info("📊 Check CLEANUP_REPORT.md for detailed information.")


def main():
    """Main entry point for cleanup script."""
    parser = argparse.ArgumentParser(
        description="WJP ANALYSER Project Cleanup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true", 
        help="Create backup before cleanup"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cleanup without confirmation"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Create cleanup manager
    cleanup = WJPProjectCleanup(project_root)
    
    # Run cleanup
    cleanup.run_cleanup(
        dry_run=args.dry_run,
        backup=args.backup,
        force=args.force
    )


if __name__ == "__main__":
    main()
