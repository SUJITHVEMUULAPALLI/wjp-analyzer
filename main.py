#!/usr/bin/env python3
"""
WJP ANALYSER - Main Entry Point

Waterjet DXF Analysis Tool with AI Integration

DEPRECATED: Use 'wjp' command instead.
This entry point is kept for backward compatibility.
"""

import sys
import os
import warnings

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Show deprecation warning
warnings.warn(
    "Using 'main.py' is deprecated. Use 'wjp' command instead:\n"
    "  python -m wjp_analyser.cli.wjp_cli web\n"
    "  or: python -m src.wjp_analyser.cli.wjp_cli web",
    DeprecationWarning,
    stacklevel=2
)

# Try new CLI first, fallback to old CLI
try:
    from wjp_analyser.cli.wjp_cli import wjp
    if __name__ == "__main__":
        wjp()
except ImportError:
    # Fallback to old CLI
    from cli.main import main
    if __name__ == "__main__":
        main()