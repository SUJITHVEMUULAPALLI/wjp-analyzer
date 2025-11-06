#!/usr/bin/env python3
"""
WJP ANALYSER - Unified Application Entry Point
==============================================

DEPRECATED: This entry point is deprecated. Use the new unified CLI instead:
  python -m wjp_analyser.cli.wjp_cli web
  python -m wjp_analyser.cli.wjp_cli api
  python -m wjp_analyser.cli.wjp_cli worker

This entry point is kept for backward compatibility.

Features:
- DXF Analysis and Processing
- Image to DXF Conversion
- AI-Powered Design Generation
- Cost Estimation and Nesting
- Multiple Interface Options (Web UI, CLI, API)
- Guided Mode for Beginners
- Batch Processing
- Real-time Monitoring

Usage:
    python wjp_analyser_unified.py [command] [options]
    
    DEPRECATED: Use 'wjp' command instead.

Commands:
    web-ui         Launch unified web interface (default)
    cli            Launch command-line interface
    api            Launch API server
    demo           Run demo pipeline
    test           Run tests
    status         Show system status
    help           Show this help message

Examples:
    python wjp_analyser_unified.py                    # Launch web UI
    python wjp_analyser_unified.py web-ui --port 8501 # Launch on specific port
    python wjp_analyser_unified.py cli analyze sample.dxf
    python wjp_analyser_unified.py api --host 0.0.0.0 --port 5000
    python wjp_analyser_unified.py demo
    python wjp_analyser_unified.py status
"""

import argparse
import os
import sys
import subprocess
import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class WJPUnifiedApp:
    """Unified WJP ANALYSER Application Manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.config_dir = self.project_root / "config"
        self.output_dir = self.project_root / "output"
        self.logs_dir = self.project_root / "logs"
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
        # Determine Python executable
        self.python_cmd = self._get_python_executable()
        
    def setup_logging(self):
        """Setup unified logging system."""
        log_file = self.logs_dir / "wjp_unified.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('WJPUnified')
        
    def load_config(self) -> Dict[str, Any]:
        """Load unified configuration."""
        config_files = [
            self.config_dir / "unified_config.yaml",
            self.config_dir / "app_config.yaml",
            self.config_dir / "ai_config.yaml"
        ]
        
        config = {}
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        file_config = yaml.safe_load(f)
                        config.update(file_config)
                        self.logger.info(f"Loaded config from {config_file}")
                except Exception as e:
                    self.logger.warning(f"Could not load config from {config_file}: {e}")
        
        # Set defaults
        config.setdefault('server', {})
        config['server'].setdefault('host', '127.0.0.1')
        config['server'].setdefault('port', 8501)
        config['server'].setdefault('debug', False)
        
        config.setdefault('features', {})
        config['features'].setdefault('ai_analysis', True)
        config['features'].setdefault('image_conversion', True)
        config['features'].setdefault('nesting', True)
        config['features'].setdefault('cost_estimation', True)
        config['features'].setdefault('guided_mode', True)
        config['features'].setdefault('batch_processing', True)
        
        return config
        
    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable."""
        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)
        return sys.executable
        
    def launch_web_ui(self, args: argparse.Namespace) -> int:
        """Launch the unified web interface."""
        self.logger.info("Launching WJP ANALYSER Unified Web Interface...")
        
        # Determine which interface to use
        interface_mode = getattr(args, 'interface', 'streamlit')
        
        if interface_mode == 'streamlit':
            return self._launch_streamlit_ui(args)
        elif interface_mode == 'flask':
            return self._launch_flask_ui(args)
        elif interface_mode == 'enhanced':
            return self._launch_enhanced_ui(args)
        elif interface_mode == 'supervisor':
            return self._launch_supervisor_ui(args)
        else:
            self.logger.error(f"Unknown interface mode: {interface_mode}")
            return 1
            
    def _launch_streamlit_ui(self, args: argparse.Namespace) -> int:
        """Launch Streamlit interface."""
        script_path = self.src_dir / "wjp_analyser" / "web" / "streamlit_app.py"
        
        if not script_path.exists():
            self.logger.error(f"Streamlit app not found at {script_path}")
            return 1
            
        cmd = [
            self.python_cmd,
            "-m", "streamlit", "run",
            str(script_path),
            "--server.address", args.host,
            "--server.port", str(args.port)
        ]
        
        if args.headless:
            cmd.extend(["--server.headless", "true"])
            
        if args.guided:
            cmd.extend(["--", "--guided"])
            
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"URL: http://{args.host}:{args.port}")
        
        try:
            env = os.environ.copy()
            env.update({
                "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "0",
                "USE_STUB_IMAGES": "1",
                "PYTHONUTF8": "1"
            })
            
            if args.guided:
                env["WJP_GUIDED_MODE"] = "true"
                
            return subprocess.run(cmd, env=env, cwd=self.project_root).returncode
        except KeyboardInterrupt:
            self.logger.info("Streamlit stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error launching Streamlit: {e}")
            return 1
            
    def _launch_flask_ui(self, args: argparse.Namespace) -> int:
        """Launch Flask interface."""
        script_path = self.project_root / "app.py"
        
        if not script_path.exists():
            self.logger.error(f"Flask app not found at {script_path}")
            return 1
            
        cmd = [self.python_cmd, str(script_path)]
        
        # Set environment variables for Flask
        env = os.environ.copy()
        env.update({
            "FLASK_HOST": args.host,
            "FLASK_PORT": str(args.port),
            "FLASK_DEBUG": str(args.debug).lower()
        })
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"URL: http://{args.host}:{args.port}")
        
        try:
            return subprocess.run(cmd, env=env, cwd=self.project_root).returncode
        except KeyboardInterrupt:
            self.logger.info("Flask stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error launching Flask: {e}")
            return 1
            
    def _launch_enhanced_ui(self, args: argparse.Namespace) -> int:
        """Launch Enhanced interface."""
        script_path = self.src_dir / "wjp_analyser" / "web" / "enhanced_app.py"
        
        if not script_path.exists():
            self.logger.error(f"Enhanced app not found at {script_path}")
            return 1
            
        cmd = [self.python_cmd, str(script_path)]
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"URL: http://{args.host}:{args.port}")
        
        try:
            return subprocess.run(cmd, cwd=self.project_root).returncode
        except KeyboardInterrupt:
            self.logger.info("Enhanced UI stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error launching Enhanced UI: {e}")
            return 1
            
    def _launch_supervisor_ui(self, args: argparse.Namespace) -> int:
        """Launch Supervisor Dashboard."""
        script_path = self.src_dir / "wjp_analyser" / "web" / "supervisor_dashboard.py"
        
        if not script_path.exists():
            self.logger.error(f"Supervisor dashboard not found at {script_path}")
            return 1
            
        cmd = [
            self.python_cmd,
            "-m", "streamlit", "run",
            str(script_path),
            "--server.address", args.host,
            "--server.port", str(args.port)
        ]
        
        if args.headless:
            cmd.extend(["--server.headless", "true"])
            
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"URL: http://{args.host}:{args.port}")
        
        try:
            env = os.environ.copy()
            env.update({
                "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "0",
                "USE_STUB_IMAGES": "1",
                "PYTHONUTF8": "1"
            })
            
            return subprocess.run(cmd, env=env, cwd=self.project_root).returncode
        except KeyboardInterrupt:
            self.logger.info("Supervisor Dashboard stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error launching Supervisor Dashboard: {e}")
            return 1
            
    def launch_cli(self, args: argparse.Namespace) -> int:
        """Launch command-line interface."""
        self.logger.info("Launching WJP ANALYSER CLI...")
        
        try:
            from cli.main import main as cli_main
            return cli_main()
        except ImportError as e:
            self.logger.error(f"CLI not available: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"Error launching CLI: {e}")
            return 1
            
    def launch_api(self, args: argparse.Namespace) -> int:
        """Launch API server."""
        self.logger.info("Launching WJP ANALYSER API Server...")
        
        script_path = self.project_root / "app.py"
        
        if not script_path.exists():
            self.logger.error(f"API server not found at {script_path}")
            return 1
            
        cmd = [self.python_cmd, str(script_path)]
        
        # Set environment variables for API
        env = os.environ.copy()
        env.update({
            "FLASK_HOST": args.host,
            "FLASK_PORT": str(args.port),
            "FLASK_DEBUG": str(args.debug).lower(),
            "WJP_API_MODE": "true"
        })
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"API URL: http://{args.host}:{args.port}")
        
        try:
            return subprocess.run(cmd, env=env, cwd=self.project_root).returncode
        except KeyboardInterrupt:
            self.logger.info("API Server stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error launching API Server: {e}")
            return 1
            
    def run_demo(self, args: argparse.Namespace) -> int:
        """Run demo pipeline."""
        self.logger.info("Running WJP ANALYSER Demo Pipeline...")
        
        try:
            # Create demo output directory
            demo_output = self.project_root / "output" / "demo"
            demo_output.mkdir(parents=True, exist_ok=True)
            
            # Create a simple demo DXF
            demo_dxf = demo_output / "demo_circle.dxf"
            
            if not demo_dxf.exists():
                try:
                    import ezdxf
                    doc = ezdxf.new('R2010')
                    msp = doc.modelspace()
                    msp.add_circle((0, 0), 50)
                    doc.saveas(demo_dxf)
                    self.logger.info(f"Created demo DXF: {demo_dxf}")
                except ImportError:
                    self.logger.warning("ezdxf not available, creating simple DXF")
                    # Create a simple DXF without ezdxf
                    simple_dxf = """0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
TABLES
0
ENDSEC
0
SECTION
2
BLOCKS
0
ENDSEC
0
SECTION
2
ENTITIES
0
CIRCLE
5
1
8
0
10
0.0
20
0.0
30
0.0
40
100.0
0
ENDSEC
0
EOF
"""
                    demo_dxf.write_text(simple_dxf)
            
            # Analyze the demo DXF
            try:
                from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf
                
                analyze_args = AnalyzeArgs(
                    material="Steel",
                    thickness=6.0,
                    kerf=1.1,
                    out=str(demo_output)
                )
                
                report = analyze_dxf(str(demo_dxf), analyze_args)
                
                self.logger.info("Demo completed successfully!")
                
                # Show results
                if report:
                    metrics = report.get('metrics', {})
                    if metrics:
                        self.logger.info(f"Demo Results:")
                        self.logger.info(f"  Outer Length: {metrics.get('length_outer_mm', 0):.1f} mm")
                        self.logger.info(f"  Pierces: {metrics.get('pierce_count', 0)}")
                        self.logger.info(f"  Estimated Cost: INR {metrics.get('estimated_cutting_cost_inr', 0):.0f}")
                
                return 0
                
            except ImportError as e:
                self.logger.warning(f"DXF analyzer not available: {e}")
                self.logger.info("Demo DXF created successfully!")
                return 0
                
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            return 1
            
    def run_tests(self, args: argparse.Namespace) -> int:
        """Run tests."""
        self.logger.info("Running WJP ANALYSER tests...")
        
        cmd = [self.python_cmd, "-m", "pytest"]
        
        if args.verbose:
            cmd.append("-v")
            
        if args.coverage:
            cmd.extend(["--cov=src", "--cov-report=html"])
            
        if args.file:
            cmd.append(args.file)
            
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            return subprocess.run(cmd, cwd=self.project_root).returncode
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return 1
            


    def show_status(self) -> int:
        """Show system status."""
        print("WJP ANALYSER Unified System Status")
        print("=" * 50)

        print(f"Python executable: {self.python_cmd}")
        is_virtual_env = ".venv" in self.python_cmd
        virtual_status = "Active" if is_virtual_env else "Not found"
        print(f"Virtual environment: {virtual_status}")

        print("\nKey directories:")
        print(f"  Project Root: {self.project_root}")
        print(f"  Source: {self.src_dir}")
        print(f"  Config: {self.config_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Logs: {self.logs_dir}")

        print("\nKey files:")
        key_files = [
            ("Main Entry", "wjp_analyser_unified.py"),
            ("Streamlit App", "src/wjp_analyser/web/streamlit_app.py"),
            ("Flask App", "app.py"),
            ("Enhanced App", "src/wjp_analyser/web/enhanced_app.py"),
            ("Supervisor Dashboard", "src/wjp_analyser/web/supervisor_dashboard.py"),
            ("Unified Config", "config/unified_config.yaml"),
            ("Requirements", "requirements.txt"),
        ]

        for name, path_value in key_files:
            full_path = self.project_root / path_value
            status = "OK" if full_path.exists() else "Missing"
            print(f"  {status:>7} - {name}: {path_value}")

        print("\nDependencies:")
        try:
            import streamlit  # type: ignore
            print(f"  Streamlit: v{streamlit.__version__}")
        except ImportError:
            print("  Streamlit: Not available")

        try:
            import flask  # type: ignore
            print(f"  Flask: v{flask.__version__}")
        except ImportError:
            print("  Flask: Not available")

        try:
            import openai  # type: ignore
            print(f"  OpenAI: v{openai.__version__}")
        except ImportError:
            print("  OpenAI: Not available")

        print("\nConfigured features:")
        features = self.config.get("features", {})
        for feature, enabled in features.items():
            status = "Enabled" if enabled else "Disabled"
            label = feature.replace("_", " ").title()
            print(f"  {label}: {status}")

        return 0

    def launch_all_interfaces(self, args: argparse.Namespace) -> int:
        """Launch all interfaces simultaneously."""
        self.logger.info("Launching all WJP ANALYSER interfaces...")
        
        interfaces = [
            {
                "name": "Streamlit UI",
                "method": self._launch_streamlit_ui,
                "port": args.port,
                "description": "Main unified interface"
            },
            {
                "name": "Flask API",
                "method": self._launch_flask_ui,
                "port": args.port + 1,
                "description": "REST API server"
            },
            {
                "name": "Supervisor Dashboard",
                "method": self._launch_supervisor_ui,
                "port": args.port + 2,
                "description": "Agent supervisor dashboard"
            }
        ]
        
        threads = []
        
        for interface in interfaces:
            interface_args = argparse.Namespace(
                host=args.host,
                port=interface["port"],
                headless=True,
                debug=False
            )
            
            def launch_interface(cmd_method, cmd_args, name, port):
                try:
                    self.logger.info(f"Starting {name} on port {port}...")
                    cmd_method(cmd_args)
                except Exception as e:
                    self.logger.error(f"{name} failed: {e}")
            
            thread = threading.Thread(
                target=launch_interface,
                args=(interface["method"], interface_args, interface["name"], interface["port"]),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            
            time.sleep(2)  # Small delay between launches
            
        print("\nAll interfaces launched successfully.")
        print("=" * 50)
        for interface in interfaces:
            print(f"{interface['name']}: http://{args.host}:{interface['port']}/")
            print(f"   Description: {interface['description']}")
        print("\nPress Ctrl+C to stop all interfaces")
        
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.logger.info("Stopping all interfaces...")
            return 0
            
        return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WJP ANALYSER - Unified Application Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Web UI command
    webui_parser = subparsers.add_parser("web-ui", help="Launch unified web interface")
    webui_parser.add_argument("--interface", choices=["streamlit", "flask", "enhanced", "supervisor"], 
                             default="streamlit", help="Interface to use")
    webui_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    webui_parser.add_argument("--port", type=int, default=8501, help="Port number")
    webui_parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    webui_parser.add_argument("--guided", action="store_true", help="Enable guided mode")
    webui_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Launch command-line interface")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Launch API server")
    api_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    api_parser.add_argument("--port", type=int, default=5000, help="Port number")
    api_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo pipeline")
    demo_parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    demo_parser.add_argument("--upgrade", action="store_true", help="Upgrade dependencies")
    demo_parser.add_argument("--open-preview", action="store_true", help="Open preview after demo")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    test_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    test_parser.add_argument("--file", help="Specific test file to run")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    # All interfaces command
    all_parser = subparsers.add_parser("all-interfaces", help="Launch all interfaces")
    all_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    all_parser.add_argument("--port", type=int, default=8501, help="Base port number")
    
    # Help command
    subparsers.add_parser("help", help="Show help message")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if not args.command or args.command == "help":
        print(__doc__)
        return 0
    
    app = WJPUnifiedApp()
    
    if args.command == "web-ui":
        return app.launch_web_ui(args)
    elif args.command == "cli":
        return app.launch_cli(args)
    elif args.command == "api":
        return app.launch_api(args)
    elif args.command == "demo":
        return app.run_demo(args)
    elif args.command == "test":
        return app.run_tests(args)
    elif args.command == "status":
        return app.show_status()
    elif args.command == "all-interfaces":
        return app.launch_all_interfaces(args)
    else:
        print(f"Unknown command: {args.command}")
        print("Use 'python wjp_analyser_unified.py help' for available commands")
        return 1


if __name__ == "__main__":
    sys.exit(main())
