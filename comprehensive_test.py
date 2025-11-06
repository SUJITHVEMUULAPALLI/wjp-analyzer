#!/usr/bin/env python3
"""Comprehensive test script to check all dependencies."""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_import(module_name, import_statement):
    """Test importing a module."""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: Imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False

print("Testing all dependencies...")
print("=" * 50)

# Test basic dependencies
tests = [
    ("NumPy", "import numpy as np"),
    ("Matplotlib", "import matplotlib.pyplot as plt"),
    ("PIL/Pillow", "from PIL import Image"),
    ("Streamlit", "import streamlit as st"),
    ("OpenCV", "import cv2"),
    ("Shapely", "from shapely.geometry import Polygon"),
    ("EzDXF", "import ezdxf"),
    ("Pydantic", "import pydantic"),
]

all_passed = True
for name, import_stmt in tests:
    if not test_import(name, import_stmt):
        all_passed = False

print("\n" + "=" * 50)
print("Testing WJP components...")

# Test WJP components
wjp_tests = [
    ("WJP Package", "import wjp_analyser"),
    ("Web Components", "from wjp_analyser.web._components import ensure_workdir"),
    ("Streamlit App", "from wjp_analyser.web.streamlit_app import main"),
    ("Unified Web App", "from wjp_analyser.web.unified_web_app import WJPUnifiedWebApp"),
]

for name, import_stmt in wjp_tests:
    if not test_import(name, import_stmt):
        all_passed = False

print("\n" + "=" * 50)
if all_passed:
    print("🎉 ALL TESTS PASSED! System is ready.")
else:
    print("❌ Some tests failed. Check the errors above.")
