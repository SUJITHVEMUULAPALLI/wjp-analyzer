#!/usr/bin/env python3
"""
WJP Analyzer - Clean Organization Script
========================================

This script organizes all the scattered files into a clean, unified structure.
"""

import os
import shutil
from pathlib import Path

def organize_files():
    """Organize all scattered files into clean structure."""
    
    print("🧹 **WJP ANALYZER - CLEAN ORGANIZATION**")
    print("=" * 50)
    
    # Define organization structure
    organization_plan = {
        "📁 Core System": {
            "files": [
                "app.py",
                "main.py", 
                "run_web_ui.py",
                "run_one_click.py",
                "run_one_click.bat",
                "run_web_ui.bat",
                "run_evaluation.bat"
            ],
            "description": "Main application files"
        },
        
        "📁 Documentation": {
            "files": [
                "README.md",
                "QUICK_START_GUIDE.md",
                "USER_MANUAL.md",
                "TECHNICAL_SPECIFICATIONS.md",
                "API_DOCUMENTATION.md",
                "AI_PROJECT_DOCUMENTATION.md",
                "AI_TRAINING_DATA.md"
            ],
            "description": "All documentation files"
        },
        
        "📁 Configuration": {
            "files": [
                "config/",
                "pyproject.toml",
                "requirements.txt",
                "pytest.ini"
            ],
            "description": "Configuration and setup files"
        },
        
        "📁 Source Code": {
            "files": [
                "src/",
                "wjp_agents/",
                "tools/",
                "tests/"
            ],
            "description": "Core source code and agents"
        },
        
        "📁 Data & Output": {
            "files": [
                "data/",
                "output/",
                "uploads/",
                "logs/"
            ],
            "description": "Data, output, and log files"
        },
        
        "📁 Templates & UI": {
            "files": [
                "templates/",
                "examples/"
            ],
            "description": "UI templates and examples"
        },
        
        "📁 Cleanup Candidates": {
            "files": [
                # Standalone agent files (should be integrated)
                "wjp_designer_agent.py",
                "wjp_image_to_dxf_agent.py", 
                "wjp_dxf_analyzer_agent.py",
                "wjp_report_generator_agent.py",
                "wjp_supervisor_agent.py",
                "wjp_file_manager.py",
                
                # Standalone interfaces (should be integrated)
                "wjp_guided_interface.py",
                "wjp_guided_batch_interface.py",
                "wjp_streamlit_interface.py",
                "advanced_batch_interface.py",
                "intelligent_supervisor_agent.py",
                
                # Launchers (should be consolidated)
                "launch_guided_interfaces.py",
                "launch_advanced_batch.py",
                "launch_wjp_automation.py",
                "wjp_interface_launcher.py",
                
                # Test and temporary files
                "test_wjp_pipeline.py",
                "system_summary.py",
                "practical_enhancement_system.py",
                
                # Documentation duplicates
                "WJP_AUTOMATION_PIPELINE_COMPLETE.md",
                "WJP_GUIDED_INTERFACES_DOCUMENTATION.md",
                "WJP_GUIDED_INTERFACES_INTEGRATION_COMPLETE.md",
                "WJP_GUIDED_INTERFACES_PROPER_INTEGRATION_COMPLETE.md",
                "ADVANCED_BATCH_PROCESSING_DOCUMENTATION.md",
                "AGENT_LEARNING_SUMMARY.md",
                "AGENT_TOOLING_DOCUMENTATION.md",
                "ENHANCED_AGENTS_SUMMARY.md",
                "CLEANUP_SUMMARY.md",
                
                # Test results and logs
                "*.json",
                "*.md",
                "agent_requirements.txt",
                "agent_tooling_report.md",
                "comparison_summary_*.md",
                "designer_test_summary_*.md",
                "enhanced_test_summary_*.md",
                "real_world_test_summary_*.md",
                "streamlit_cli_comparison_*.json",
                "test_results_*.json",
                "learning_demonstration_*.json",
                "enhanced_agent_test_*.json",
                "comprehensive_designer_test_*.json",
                "real_world_image_test_*.json",
                "tooling_summary_*.json",
                
                # Batch files (should be consolidated)
                "launch_*.bat",
                
                # Temporary files
                "debug_export*.dxf",
                "test_export*.dxf",
                "wjp.env.txt"
            ],
            "description": "Files to clean up or consolidate"
        }
    }
    
    # Show current structure
    print("\n📊 **CURRENT FILE STRUCTURE:**")
    for category, info in organization_plan.items():
        print(f"\n{category}")
        print(f"  {info['description']}")
        if isinstance(info['files'], list):
            for file in info['files']:
                if os.path.exists(file):
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (not found)")
    
    # Show cleanup recommendations
    print("\n🧹 **CLEANUP RECOMMENDATIONS:**")
    print("\n1. **Consolidate Agent Files**")
    print("   - Move standalone agent files into wjp_agents/")
    print("   - Integrate guided interfaces into src/wjp_analyser/web/pages/")
    print("   - Remove duplicate functionality")
    
    print("\n2. **Consolidate Documentation**")
    print("   - Merge similar documentation files")
    print("   - Keep only essential documentation")
    print("   - Remove test result files")
    
    print("\n3. **Consolidate Launchers**")
    print("   - Merge all launcher scripts into run_web_ui.py")
    print("   - Remove standalone launcher files")
    print("   - Keep only essential batch files")
    
    print("\n4. **Clean Test Files**")
    print("   - Remove temporary test files")
    print("   - Keep only essential test results")
    print("   - Clean up debug files")
    
    return organization_plan

def create_clean_structure():
    """Create a clean, organized file structure."""
    
    print("\n🏗️ **CREATING CLEAN STRUCTURE:**")
    
    # Create organized directories
    clean_dirs = [
        "docs/guides/",
        "docs/api/", 
        "docs/examples/",
        "scripts/launchers/",
        "scripts/tools/",
        "archive/old_files/",
        "archive/test_results/",
        "archive/documentation/"
    ]
    
    for dir_path in clean_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    print("\n📋 **ORGANIZATION COMPLETE!**")
    print("\nNext steps:")
    print("1. Review the cleanup recommendations")
    print("2. Move files to appropriate locations")
    print("3. Remove duplicate and temporary files")
    print("4. Update documentation to reflect new structure")

if __name__ == "__main__":
    organization_plan = organize_files()
    create_clean_structure()
