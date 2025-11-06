"""
WJP ANALYSER - Unified CLI Entry Point
=======================================

This is the canonical CLI entry point for WJP ANALYSER.
All commands should go through this interface.

Usage:
    wjp web          # Launch Streamlit UI
    wjp api          # Launch FastAPI server
    wjp worker       # Start queue workers
    wjp demo         # Run demo pipeline
    wjp test         # Run tests
    wjp status       # Show system status
"""

from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Error: click is required. Install with: pip install click")
    sys.exit(1)


# Add src to Python path
_THIS_DIR = Path(__file__).parent
_SRC_DIR = _THIS_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@click.group()
@click.version_option(version="2.0.0", prog_name="WJP ANALYSER")
def wjp():
    """WJP ANALYSER - Waterjet DXF Analysis Tool"""
    pass


@wjp.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", type=int, default=8501, help="Port to bind to")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def web(host: str, port: int, no_browser: bool):
    """Launch Streamlit web UI"""
    import streamlit.web.cli as stcli
    
    # Build streamlit command
    streamlit_args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_SRC_DIR / "wjp_analyser" / "web" / "unified_web_app.py"),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    
    if no_browser:
        streamlit_args.append("--server.headless")
        streamlit_args.append("true")
    
    click.echo(f"Starting Streamlit UI at http://{host}:{port}")
    os.execv(sys.executable, streamlit_args)


@wjp.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", type=int, default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def api(host: str, port: int, reload: bool):
    """Launch FastAPI API server"""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is required. Install with: pip install uvicorn", err=True)
        sys.exit(1)
    
    click.echo(f"Starting FastAPI server at http://{host}:{port}")
    click.echo(f"API docs available at http://{host}:{port}/docs")
    
    # Import FastAPI app
    from wjp_analyser.api.fastapi_app import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


@wjp.command()
@click.option("--queues", default="default", help="Comma-separated queue names (default: default)")
@click.option("--burst", is_flag=True, help="Run in burst mode (exit when no jobs)")
@click.option("--name", default=None, help="Worker name")
def worker(queues: str, burst: bool, name: Optional[str]):
    """Start RQ workers for background jobs"""
    try:
        from wjp_analyser.api.worker import main as worker_main
        import sys
        
        # Build arguments for worker
        worker_args = ["--queues", queues]
        if burst:
            worker_args.append("--burst")
        if name:
            worker_args.extend(["--name", name])
        
        # Set sys.argv for worker main()
        original_argv = sys.argv
        sys.argv = ["worker"] + worker_args
        try:
            worker_main()
        finally:
            sys.argv = original_argv
    except ImportError:
        click.echo("Error: RQ and redis are required. Install with: pip install rq redis", err=True)
        click.echo("  Install Redis: https://redis.io/docs/getting-started/installation/")
        click.echo("  Or use Docker: docker run -d -p 6379:6379 redis")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting worker: {e}", err=True)
        sys.exit(1)


@wjp.command()
def demo():
    """Run demo pipeline"""
    click.echo("Running demo pipeline...")
    # TODO: Implement demo
    click.echo("Demo pipeline not yet implemented")


@wjp.command()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test(verbose: bool):
    """Run tests"""
    import pytest
    
    test_dir = _SRC_DIR.parent / "tests"
    if not test_dir.exists():
        click.echo(f"Test directory not found: {test_dir}", err=True)
        sys.exit(1)
    
    args = [str(test_dir)]
    if verbose:
        args.append("-v")
    
    sys.exit(pytest.main(args))


@wjp.command()
def status():
    """Show system status"""
    click.echo("WJP ANALYSER Status")
    click.echo("=" * 50)
    
    # Check dependencies
    click.echo("\nDependencies:")
    deps = {
        "ezdxf": "DXF file handling",
        "opencv-python": "Image processing",
        "streamlit": "Web UI",
        "shapely": "Geometric operations",
        "numpy": "Numerical operations",
    }
    
    for dep, desc in deps.items():
        try:
            __import__(dep.replace("-", "_"))
            click.echo(f"  ✓ {dep:20s} - {desc}")
        except ImportError:
            click.echo(f"  ✗ {dep:20s} - {desc} (MISSING)")
    
    # Check configuration
    click.echo("\nConfiguration:")
    config_file = _SRC_DIR.parent / "config" / "wjp_unified_config.yaml"
    if config_file.exists():
        click.echo(f"  ✓ Config file found: {config_file}")
    else:
        click.echo(f"  ✗ Config file missing: {config_file}")
    
    # Check data directories
    click.echo("\nData Directories:")
    data_dirs = [
        _SRC_DIR.parent / "data",
        _SRC_DIR.parent / "output",
    ]
    for data_dir in data_dirs:
        if data_dir.exists():
            click.echo(f"  ✓ {data_dir}")
        else:
            click.echo(f"  ✗ {data_dir} (will be created on first use)")


def main():
    """Main entry point"""
    wjp()


if __name__ == "__main__":
    main()

