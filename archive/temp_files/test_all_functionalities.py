#!/usr/bin/env python3
"""
WJP Analyzer - Comprehensive Functionality Test
==============================================

This script tests all functionalities after the cleanup to ensure everything works.
"""

import os
import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("🔍 **TESTING IMPORTS**")
    print("=" * 40)
    
    tests = [
        ("wjp_agents.designer_agent", "DesignerAgent"),
        ("wjp_agents.image_to_dxf_agent", "ImageToDXFAgent"),
        ("wjp_agents.analyze_dxf_agent", "AnalyzeDXFAgent"),
        ("wjp_agents.learning_agent", "LearningAgent"),
        ("wjp_agents.report_agent", "ReportAgent"),
        ("wjp_agents.supervisor_agent", "SupervisorAgent"),
        ("src.wjp_analyser.web.streamlit_app", "streamlit_app"),
        ("src.wjp_analyser.image_processing.interactive_editor", "InteractiveEditor"),
        ("src.wjp_analyser.image_processing.object_detector", "ObjectDetector"),
    ]
    
    results = {}
    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            results[module_name] = "✅ SUCCESS"
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            results[module_name] = f"❌ FAILED: {e}"
            print(f"❌ {module_name}.{class_name}: {e}")
    
    return results

def test_agent_initialization():
    """Test agent initialization."""
    print("\n🤖 **TESTING AGENT INITIALIZATION**")
    print("=" * 40)
    
    # Add project root to path
    current_dir = Path(__file__).parent
    project_root = current_dir
    sys.path.insert(0, str(project_root))
    
    # Add wjp_agents to path
    wjp_agents_path = project_root / "wjp_agents"
    sys.path.insert(0, str(wjp_agents_path))
    
    agents_to_test = [
        ("DesignerAgent", "wjp_agents.designer_agent"),
        ("ImageToDXFAgent", "wjp_agents.image_to_dxf_agent"),
        ("AnalyzeDXFAgent", "wjp_agents.analyze_dxf_agent"),
        ("LearningAgent", "wjp_agents.learning_agent"),
        ("ReportAgent", "wjp_agents.report_agent"),
        ("SupervisorAgent", "wjp_agents.supervisor_agent"),
    ]
    
    results = {}
    for agent_name, module_name in agents_to_test:
        try:
            module = __import__(module_name, fromlist=[agent_name])
            agent_class = getattr(module, agent_name)
            agent = agent_class()
            results[agent_name] = "✅ SUCCESS"
            print(f"✅ {agent_name} initialized successfully")
        except Exception as e:
            results[agent_name] = f"❌ FAILED: {e}"
            print(f"❌ {agent_name} failed: {e}")
            traceback.print_exc()
    
    return results

def test_designer_agent_functionality():
    """Test DesignerAgent functionality."""
    print("\n🎨 **TESTING DESIGNER AGENT**")
    print("=" * 40)
    
    try:
        from wjp_agents.designer_agent import DesignerAgent
        
        agent = DesignerAgent()
        print(f"✅ DesignerAgent initialized")
        print(f"📁 Output directory: {agent.output_dir}")
        print(f"🔑 API key loaded: {'Yes' if agent.api_key else 'No'}")
        
        # Test with a simple prompt
        test_prompt = "Simple geometric medallion design"
        print(f"🔄 Testing with prompt: {test_prompt}")
        
        result = agent.run(test_prompt)
        print(f"✅ Generation result: {result}")
        
        # Check if image was created
        image_path = result.get("image_path")
        if image_path and os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            print(f"✅ Image created: {image_path} ({file_size} bytes)")
            return True
        else:
            print(f"❌ Image not found: {image_path}")
            return False
            
    except Exception as e:
        print(f"❌ DesignerAgent test failed: {e}")
        traceback.print_exc()
        return False

def test_image_to_dxf_functionality():
    """Test Image to DXF functionality."""
    print("\n🖼️ **TESTING IMAGE TO DXF**")
    print("=" * 40)
    
    try:
        from wjp_agents.image_to_dxf_agent import ImageToDXFAgent
        
        agent = ImageToDXFAgent()
        print(f"✅ ImageToDXFAgent initialized")
        
        # Check if we have a test image
        test_image_path = None
        designer_output = "output/designer"
        if os.path.exists(designer_output):
            for file in os.listdir(designer_output):
                if file.endswith('.png'):
                    test_image_path = os.path.join(designer_output, file)
                    break
        
        if test_image_path and os.path.exists(test_image_path):
            print(f"📁 Using test image: {test_image_path}")
            
            # Test conversion
            result = agent.run(test_image_path)
            print(f"✅ Conversion result: {result}")
            
            # Check if DXF was created
            dxf_path = result.get("dxf_path")
            if dxf_path and os.path.exists(dxf_path):
                file_size = os.path.getsize(dxf_path)
                print(f"✅ DXF created: {dxf_path} ({file_size} bytes)")
                return True
            else:
                print(f"❌ DXF not found: {dxf_path}")
                return False
        else:
            print("⚠️ No test image found, skipping conversion test")
            return True
            
    except Exception as e:
        print(f"❌ ImageToDXF test failed: {e}")
        traceback.print_exc()
        return False

def test_dxf_analyzer_functionality():
    """Test DXF Analyzer functionality."""
    print("\n📊 **TESTING DXF ANALYZER**")
    print("=" * 40)
    
    try:
        from wjp_agents.analyze_dxf_agent import AnalyzeDXFAgent
        
        agent = AnalyzeDXFAgent()
        print(f"✅ AnalyzeDXFAgent initialized")
        
        # Check if we have a test DXF
        test_dxf_path = None
        dxf_output = "output/dxf"
        if os.path.exists(dxf_output):
            for file in os.listdir(dxf_output):
                if file.endswith('.dxf'):
                    test_dxf_path = os.path.join(dxf_output, file)
                    break
        
        if test_dxf_path and os.path.exists(test_dxf_path):
            print(f"📁 Using test DXF: {test_dxf_path}")
            
            # Test analysis
            result = agent.run(test_dxf_path)
            print(f"✅ Analysis result: {result}")
            
            # Check if report was created
            report_path = result.get("report_path")
            if report_path and os.path.exists(report_path):
                file_size = os.path.getsize(report_path)
                print(f"✅ Report created: {report_path} ({file_size} bytes)")
                return True
            else:
                print(f"❌ Report not found: {report_path}")
                return False
        else:
            print("⚠️ No test DXF found, skipping analysis test")
            return True
            
    except Exception as e:
        print(f"❌ DXF Analyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_pages():
    """Test Streamlit pages functionality."""
    print("\n🌐 **TESTING STREAMLIT PAGES**")
    print("=" * 40)
    
    pages_to_test = [
        "src/wjp_analyser/web/pages/designer.py",
        "src/wjp_analyser/web/pages/image_to_dxf.py",
        "src/wjp_analyser/web/pages/analyze_dxf.py",
        "src/wjp_analyser/web/pages/nesting.py",
    ]
    
    results = {}
    for page_path in pages_to_test:
        try:
            if os.path.exists(page_path):
                # Try to import the page
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_page", page_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                results[page_path] = "✅ SUCCESS"
                print(f"✅ {page_path}")
            else:
                results[page_path] = "❌ FILE NOT FOUND"
                print(f"❌ {page_path} - File not found")
        except Exception as e:
            results[page_path] = f"❌ FAILED: {e}"
            print(f"❌ {page_path}: {e}")
    
    return results

def test_configuration():
    """Test configuration files."""
    print("\n⚙️ **TESTING CONFIGURATION**")
    print("=" * 40)
    
    config_files = [
        "config/api_keys.yaml",
        "config/ai_config.yaml",
        "config/material_profiles.py",
        "requirements.txt",
        "pyproject.toml",
    ]
    
    results = {}
    for config_file in config_files:
        try:
            if os.path.exists(config_file):
                file_size = os.path.getsize(config_file)
                results[config_file] = f"✅ EXISTS ({file_size} bytes)"
                print(f"✅ {config_file} ({file_size} bytes)")
            else:
                results[config_file] = "❌ NOT FOUND"
                print(f"❌ {config_file} - Not found")
        except Exception as e:
            results[config_file] = f"❌ ERROR: {e}"
            print(f"❌ {config_file}: {e}")
    
    return results

def test_file_structure():
    """Test file structure integrity."""
    print("\n📁 **TESTING FILE STRUCTURE**")
    print("=" * 40)
    
    essential_dirs = [
        "src/wjp_analyser/",
        "wjp_agents/",
        "config/",
        "data/",
        "output/",
        "templates/",
        "tools/",
        "tests/",
    ]
    
    results = {}
    for dir_path in essential_dirs:
        try:
            if os.path.exists(dir_path):
                file_count = len(os.listdir(dir_path))
                results[dir_path] = f"✅ EXISTS ({file_count} files)"
                print(f"✅ {dir_path} ({file_count} files)")
            else:
                results[dir_path] = "❌ NOT FOUND"
                print(f"❌ {dir_path} - Not found")
        except Exception as e:
            results[dir_path] = f"❌ ERROR: {e}"
            print(f"❌ {dir_path}: {e}")
    
    return results

def run_comprehensive_test():
    """Run comprehensive functionality test."""
    print("🧪 **WJP ANALYZER - COMPREHENSIVE FUNCTIONALITY TEST**")
    print("=" * 60)
    print(f"🕐 Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_results = {}
    
    test_results["imports"] = test_imports()
    test_results["agent_init"] = test_agent_initialization()
    test_results["designer"] = test_designer_agent_functionality()
    test_results["image_to_dxf"] = test_image_to_dxf_functionality()
    test_results["dxf_analyzer"] = test_dxf_analyzer_functionality()
    test_results["streamlit_pages"] = test_streamlit_pages()
    test_results["configuration"] = test_configuration()
    test_results["file_structure"] = test_file_structure()
    
    # Summary
    print("\n📊 **TEST SUMMARY**")
    print("=" * 40)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, results in test_results.items():
        print(f"\n🔍 {test_name.upper()}:")
        if isinstance(results, dict):
            for item, result in results.items():
                total_tests += 1
                if "✅" in result:
                    passed_tests += 1
                print(f"  {result}")
        else:
            total_tests += 1
            if results:
                passed_tests += 1
            print(f"  {'✅ PASSED' if results else '❌ FAILED'}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 **OVERALL RESULTS:**")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"📊 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 **SYSTEM STATUS: EXCELLENT**")
    elif success_rate >= 60:
        print("⚠️ **SYSTEM STATUS: GOOD**")
    else:
        print("❌ **SYSTEM STATUS: NEEDS ATTENTION**")
    
    print(f"\n🕐 Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_test()
