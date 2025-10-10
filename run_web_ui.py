#!/usr/bin/env python3
"""Launch the Streamlit-based Waterjet DXF Analyzer UI."""

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Streamlit Waterjet DXF Analyzer UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--dashboard", action="store_true", help="Launch Supervisor Dashboard instead of legacy app")
    parser.add_argument("--guided", action="store_true", help="Launch Guided Individual Interface")
    parser.add_argument("--batch-guided", action="store_true", help="Launch Guided Batch Interface")
    parser.add_argument("--all-interfaces", action="store_true", help="Launch all interfaces (main, guided, batch)")
    return parser.parse_args()


def _streamlit_command() -> list[str]:
    base_dir = os.path.dirname(sys.executable)
    if os.name == "nt":
        candidate = os.path.join(base_dir, "streamlit.exe")
    else:
        candidate = os.path.join(base_dir, "streamlit")
    if os.path.exists(candidate):
        return [candidate]
    return [sys.executable, "-m", "streamlit"]


def launch_guided_interface(host: str, port: int, no_browser: bool) -> int:
    """Launch the guided individual interface using the integrated approach."""
    repo_root = os.path.dirname(__file__)
    script_path = os.path.join(repo_root, "src", "wjp_analyser", "web", "streamlit_app.py")
    
    if not os.path.exists(script_path):
        print(f"Streamlit app not found at {script_path}.")
        return 1
    
    command = _streamlit_command() + [
        "run",
        script_path,
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    
    # Add UTF-8 mode for Python to prevent encoding issues
    if command[0].endswith("python.exe") or command[0].endswith("python"):
        command = [command[0], "-X", "utf8"] + command[1:]
    
    if no_browser:
        command.extend(["--server.headless", "true"])

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")
    # Set guided mode environment variable
    env["WJP_GUIDED_MODE"] = "true"

    print("Starting Guided Interface (Integrated)...")
    print(f"Listening on http://{host}:{port}/")
    print("🎯 Guided Mode: Enabled - Use the guided pages in the sidebar")

    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"Guided interface exited with status {exc.returncode}")
        return exc.returncode
    except KeyboardInterrupt:
        print("\nStopping Guided Interface...")
        return 0
    return 0


def launch_batch_guided_interface(host: str, port: int, no_browser: bool) -> int:
    """Launch the guided batch interface."""
    repo_root = os.path.dirname(__file__)
    script_path = os.path.join(repo_root, "wjp_guided_batch_interface.py")
    
    if not os.path.exists(script_path):
        print(f"Guided batch interface not found at {script_path}.")
        return 1
    
    command = _streamlit_command() + [
        "run",
        script_path,
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    
    if no_browser:
        command.extend(["--server.headless", "true"])

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")

    print("Starting Guided Batch Interface...")
    print(f"Listening on http://{host}:{port}/")

    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"Guided batch interface exited with status {exc.returncode}")
        return exc.returncode
    except KeyboardInterrupt:
        print("\nStopping Guided Batch Interface...")
        return 0
    return 0


def launch_all_interfaces(host: str, base_port: int, no_browser: bool) -> int:
    """Launch all interfaces on different ports."""
    import threading
    import time
    
    print("🚀 **Launching All WJP Interfaces**")
    print("=" * 50)
    
    # Define interfaces
    interfaces = [
        {
            "name": "Main Interface",
            "script": "src/wjp_analyser/web/streamlit_app.py",
            "port": base_port,
            "description": "Complete WJP analysis tools"
        },
        {
            "name": "Guided Individual Interface", 
            "script": "wjp_guided_interface.py",
            "port": base_port + 1,
            "description": "Step-by-step individual project guidance"
        },
        {
            "name": "Guided Batch Interface",
            "script": "wjp_guided_batch_interface.py", 
            "port": base_port + 2,
            "description": "Intelligent batch processing guidance"
        }
    ]
    
    # Check all scripts exist
    repo_root = os.path.dirname(__file__)
    for interface in interfaces:
        script_path = os.path.join(repo_root, interface["script"])
        if not os.path.exists(script_path):
            print(f"❌ {interface['name']} script not found: {script_path}")
            return 1
    
    print("✅ All interface scripts found!")
    print("")
    
    # Launch each interface
    threads = []
    for interface in interfaces:
        script_path = os.path.join(repo_root, interface["script"])
        
        command = _streamlit_command() + [
            "run",
            script_path,
            "--server.address",
            host,
            "--server.port",
            str(interface["port"]),
        ]
        
        if no_browser:
            command.extend(["--server.headless", "true"])

        env = os.environ.copy()
        env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")
        
        def launch_interface(cmd, env_vars, name, port):
            try:
                print(f"🌐 Starting {name} on port {port}...")
                subprocess.run(cmd, check=True, env=env_vars)
            except subprocess.CalledProcessError as exc:
                print(f"❌ {name} exited with status {exc.returncode}")
            except KeyboardInterrupt:
                print(f"\n🛑 {name} stopped by user")
        
        thread = threading.Thread(
            target=launch_interface,
            args=(command, env, interface["name"], interface["port"]),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        
        # Small delay between launches
        time.sleep(2)
    
    print("")
    print("🎯 **All Interfaces Launched Successfully!**")
    print("=" * 50)
    for interface in interfaces:
        print(f"🌐 {interface['name']}: http://{host}:{interface['port']}/")
        print(f"   📝 {interface['description']}")
    print("")
    print("🔄 Press Ctrl+C to stop all interfaces")
    
    try:
        # Wait for all threads
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all interfaces...")
        return 0
    
    return 0


def main() -> int:
    args = parse_args()

    # Handle guided interfaces
    if args.guided:
        return launch_guided_interface(args.host, args.port, args.no_browser)
    
    if args.batch_guided:
        return launch_batch_guided_interface(args.host, args.port, args.no_browser)
    
    if args.all_interfaces:
        return launch_all_interfaces(args.host, args.port, args.no_browser)

    # Original logic for main interface
    repo_root = os.path.dirname(__file__)
    script_path = os.path.join(repo_root, "src", "wjp_analyser", "web", "supervisor_dashboard.py") if args.dashboard else os.path.join(repo_root, "src", "wjp_analyser", "web", "streamlit_app.py")
    if not os.path.exists(script_path):
        print(f"Streamlit UI entry not found at {script_path}.")
        return 1
    print(f"Using {'Supervisor Dashboard' if args.dashboard else 'legacy UI'} -> {script_path}")

    command = _streamlit_command() + [
        "run",
        script_path,
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    
    # Add UTF-8 mode for Python to prevent encoding issues
    if command[0].endswith("python.exe") or command[0].endswith("python"):
        command = [command[0], "-X", "utf8"] + command[1:]

    if args.no_browser:
        command.extend(["--server.headless", "true"])

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")

    print("Starting Streamlit Waterjet DXF Analyzer UI...")
    print(f"Listening on http://{args.host}:{args.port}/")

    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"Streamlit exited with status {exc.returncode}")
        return exc.returncode
    except KeyboardInterrupt:
        print("\nStopping Streamlit UI...")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
