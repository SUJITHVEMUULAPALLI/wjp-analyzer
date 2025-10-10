#!/usr/bin/env python3
"""
WJP ANALYSER - Main Entry Point

Waterjet DXF Analysis Tool with AI Integration
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cli.main import main

if __name__ == "__main__":
    main()