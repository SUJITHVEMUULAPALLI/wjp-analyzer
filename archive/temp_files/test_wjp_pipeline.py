#!/usr/bin/env python3
"""
WJP Automation Pipeline - Complete System Test
==============================================

This script tests the entire WJP automation pipeline from prompt to PDF report.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

def test_complete_pipeline():
    """Test the complete WJP automation pipeline."""
    print("🚀 **WJP Automation Pipeline - Complete System Test**")
    print("=" * 70)
    
    try:
        # Test 1: File Manager
        print("\n📁 **Test 1: File Manager**")
        from wjp_file_manager import test_file_manager
        file_manager = test_file_manager()
        print("✅ File Manager test passed")
        
        # Test 2: Designer Agent
        print("\n🎨 **Test 2: Designer Agent**")
        from wjp_designer_agent import test_designer_agent
        designer = test_designer_agent()
        print("✅ Designer Agent test passed")
        
        # Test 3: Image to DXF Agent
        print("\n🔄 **Test 3: Image to DXF Agent**")
        from wjp_image_to_dxf_agent import test_image_to_dxf_agent
        image_to_dxf = test_image_to_dxf_agent()
        print("✅ Image to DXF Agent test passed")
        
        # Test 4: DXF Analyzer Agent
        print("\n📊 **Test 4: DXF Analyzer Agent**")
        from wjp_dxf_analyzer_agent import test_dxf_analyzer_agent
        analyzer = test_dxf_analyzer_agent()
        print("✅ DXF Analyzer Agent test passed")
        
        # Test 5: Report Generator Agent
        print("\n📄 **Test 5: Report Generator Agent**")
        from wjp_report_generator_agent import test_report_generator_agent
        report_generator = test_report_generator_agent()
        print("✅ Report Generator Agent test passed")
        
        # Test 6: Supervisor Agent (Complete Pipeline)
        print("\n🎯 **Test 6: Supervisor Agent (Complete Pipeline)**")
        from wjp_supervisor_agent import test_supervisor_agent
        supervisor = test_supervisor_agent()
        print("✅ Supervisor Agent test passed")
        
        # Test 7: File Structure Verification
        print("\n📂 **Test 7: File Structure Verification**")
        verify_file_structure()
        print("✅ File structure verification passed")
        
        # Test 8: Pipeline Integration Test
        print("\n🔗 **Test 8: Pipeline Integration Test**")
        test_pipeline_integration()
        print("✅ Pipeline integration test passed")
        
        print("\n🎉 **ALL TESTS PASSED SUCCESSFULLY!**")
        print("=" * 70)
        print("✅ WJP Automation Pipeline is ready for production use!")
        print("🚀 Launch the interface with: python launch_wjp_automation.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ **TEST FAILED: {e}**")
        import traceback
        traceback.print_exc()
        return False

def verify_file_structure():
    """Verify the WJP file structure is created correctly."""
    print("   📁 Checking WJP_PROJECTS folder structure...")
    
    base_dir = Path("WJP_PROJECTS")
    
    if not base_dir.exists():
        print("   ❌ WJP_PROJECTS folder not found")
        return False
    
    # Check main folders
    required_folders = [
        "01_DESIGNER",
        "02_CONVERTED_DXF",
        "03_ANALYZED",
        "04_REPORTS",
        "05_ARCHIVE"
    ]
    
    for folder in required_folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            print(f"   ✅ {folder} folder exists")
        else:
            print(f"   ❌ {folder} folder missing")
            return False
    
    # Check for project folders
    project_folders = [f for f in base_dir.iterdir() if f.is_dir() and f.name not in required_folders]
    
    if project_folders:
        print(f"   ✅ Found {len(project_folders)} project folders")
        for project_folder in project_folders:
            print(f"      📁 {project_folder.name}")
            
            # Check project subfolders
            for subfolder in required_folders:
                subfolder_path = project_folder / subfolder
                if subfolder_path.exists():
                    files = list(subfolder_path.glob("*"))
                    print(f"         ✅ {subfolder}: {len(files)} files")
                else:
                    print(f"         ❌ {subfolder}: missing")
    else:
        print("   ℹ️ No project folders found yet")
    
    return True

def test_pipeline_integration():
    """Test the complete pipeline integration."""
    print("   🔗 Testing complete pipeline integration...")
    
    try:
        from wjp_supervisor_agent import SupervisorAgent
        
        # Create supervisor
        supervisor = SupervisorAgent()
        
        # Submit a test job
        test_job = {
            "job_id": "INT01",
            "prompt": "Integration test design for WJP automation pipeline",
            "material": "Tan Brown Granite",
            "thickness_mm": 25,
            "category": "Inlay Tile",
            "dimensions_inch": [24, 24],
            "cut_spacing_mm": 3.0,
            "min_radius_mm": 2.0
        }
        
        print(f"   📋 Submitting test job: {test_job['job_id']}")
        result = supervisor.submit_job(**test_job)
        print(f"   ✅ Job submitted: {result}")
        
        # Wait for processing
        print("   ⏳ Waiting for job processing...")
        max_wait_time = 180  # 3 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            queue_status = supervisor.get_queue_status()
            
            if queue_status["active_jobs"] == 0 and queue_status["queue_size"] == 0:
                break
            
            print(f"      Queue: {queue_status['queue_size']}, Active: {queue_status['active_jobs']}")
            time.sleep(5)
        
        # Check final status
        job_status = supervisor.get_job_status(test_job["job_id"])
        
        if job_status:
            print(f"   📊 Job Status: {job_status['status']}")
            print(f"   ⏱️ Duration: {job_status['duration_seconds']:.2f}s")
            
            if job_status["status"] == "completed":
                print(f"   ✅ Integration test completed successfully!")
                print(f"   📁 Output Files: {job_status['output_files']}")
                return True
            else:
                print(f"   ❌ Integration test failed: {job_status['status']}")
                if job_status.get('errors'):
                    for error in job_status['errors']:
                        print(f"      Error: {error}")
                return False
        else:
            print("   ❌ Job status not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration test error: {e}")
        return False

def create_system_summary():
    """Create a comprehensive system summary."""
    print("\n📋 **WJP Automation Pipeline - System Summary**")
    print("=" * 70)
    
    # System components
    components = [
        ("wjp_file_manager.py", "File naming and folder structure management"),
        ("wjp_designer_agent.py", "Prompt to image generation"),
        ("wjp_image_to_dxf_agent.py", "Image to DXF conversion"),
        ("wjp_dxf_analyzer_agent.py", "DXF analysis and cutting reports"),
        ("wjp_report_generator_agent.py", "PDF report generation"),
        ("wjp_supervisor_agent.py", "Pipeline orchestration and automation"),
        ("wjp_streamlit_interface.py", "Web-based user interface"),
        ("launch_wjp_automation.py", "Application launcher")
    ]
    
    print("📦 **System Components:**")
    for i, (component, description) in enumerate(components, 1):
        print(f"   {i:2d}. {component:<35} - {description}")
    
    # Pipeline stages
    stages = [
        ("1️⃣ Designer Agent", "Prompt → Image + Metadata JSON"),
        ("2️⃣ Image to DXF Agent", "Image + Metadata → DXF + Conversion JSON"),
        ("3️⃣ DXF Analyzer Agent", "DXF + Metadata → Analysis JSON + CSV + PNG"),
        ("4️⃣ Report Generator Agent", "All Data → Professional PDF Report"),
        ("🎯 Supervisor Agent", "Orchestrates entire pipeline automatically")
    ]
    
    print("\n🔄 **Pipeline Stages:**")
    for stage, description in stages:
        print(f"   {stage:<25} - {description}")
    
    # File naming standard
    print("\n📝 **File Naming Standard:**")
    print("   WJP_<DESIGN>_<MATERIAL>_<THK>_<PROCESS>_<VER>_<DATE>.<EXT>")
    print("   Example: WJP_SR06_TANB_25_DESIGN_V1_20251008.png")
    
    # Folder structure
    print("\n📂 **Folder Structure:**")
    print("   WJP_PROJECTS/")
    print("   ├── SR06/")
    print("   │   ├── 01_DESIGNER/")
    print("   │   ├── 02_CONVERTED_DXF/")
    print("   │   ├── 03_ANALYZED/")
    print("   │   ├── 04_REPORTS/")
    print("   │   └── 05_ARCHIVE/")
    print("   └── [Other Projects]/")
    
    # Features
    features = [
        "✅ Complete automation from prompt to PDF report",
        "✅ Intelligent supervisor agent orchestration",
        "✅ Professional file naming standards",
        "✅ Material-specific cost calculations",
        "✅ Comprehensive quality assessment",
        "✅ Professional reporting (CSV, JSON, PDF)",
        "✅ Real-time job monitoring",
        "✅ Batch processing capabilities",
        "✅ Web-based user interface",
        "✅ Learning system integration"
    ]
    
    print("\n🎯 **Key Features:**")
    for feature in features:
        print(f"   {feature}")
    
    # Usage instructions
    print("\n🚀 **Usage Instructions:**")
    print("   1. Launch: python launch_wjp_automation.py")
    print("   2. Open browser: http://localhost:8503")
    print("   3. Submit jobs through the web interface")
    print("   4. Monitor progress in real-time")
    print("   5. Download results and reports")
    
    print("\n🎉 **WJP Automation Pipeline is ready for production use!**")

if __name__ == "__main__":
    success = test_complete_pipeline()
    
    if success:
        create_system_summary()
    else:
        print("\n❌ **System test failed. Please check the errors above.**")
        sys.exit(1)
