#!/usr/bin/env python3
"""
WJP ANALYSER - Minimal Startup Script
=====================================

Quick start for the WJP Analyser web interface.

Usage:
    python run.py              # Start web UI (default: http://127.0.0.1:8501)
    python run.py --port 8080  # Start on custom port
    python run.py --no-browser # Don't open browser automatically
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Launch WJP Analyser web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="WJP Analyser - Minimal Startup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to bind to (default: 8501)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    # Get paths
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    app_path = src_dir / "wjp_analyser" / "web" / "unified_web_app.py"
    
    # Check if app exists
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}", file=sys.stderr)
        sys.exit(1)
    
    # Build streamlit command
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    
    # Add headless flag if no-browser is requested
    if args.no_browser:
        streamlit_cmd.extend(["--server.headless", "true"])
    
    # Quick dependency check
    try:
        import streamlit
        import sqlalchemy
        print("✓ Core dependencies found")
    except ImportError as e:
        print(f"\n⚠️  Warning: Missing dependency: {e}", file=sys.stderr)
        print("   Install with: pip install -r requirements.txt", file=sys.stderr)
        print("   Continuing anyway...\n", file=sys.stderr)
    
    # Set environment variables
    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")
    
    # Display startup message
    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("WJP ANALYSER - Starting Web Interface")
    print("=" * 60)
    print(f"\n🌐 Opening browser at: {url}")
    if args.no_browser:
        print("   (Browser auto-open disabled)")
    print(f"\n💡 If browser doesn't open, manually navigate to: {url}")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Launch streamlit
        subprocess.run(streamlit_cmd, env=env, cwd=project_root)
    except KeyboardInterrupt:
        print("\n\n⏹️  Shutting down WJP Analyser...")
        sys.exit(0)
    except FileNotFoundError:
        print(f"\n❌ Error: Python executable not found: {sys.executable}", file=sys.stderr)
        print("   Make sure Python is installed and in your PATH", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting WJP Analyser: {e}", file=sys.stderr)
        print("\n💡 Troubleshooting:", file=sys.stderr)
        print("   1. Make sure dependencies are installed:", file=sys.stderr)
        print("      pip install -r requirements.txt", file=sys.stderr)
        print("   2. Check if Streamlit is installed:", file=sys.stderr)
        print("      pip install streamlit", file=sys.stderr)
        print(f"   3. Try accessing manually: {url}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

