#!/usr/bin/env python3
"""One-click launcher for the Waterjet DXF Analyzer project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs as Args, analyze_dxf as analyze
from cli.main import command_gcode
from tools.make_sample_dxf import make_sample

REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
DEFAULT_OUT = REPO_ROOT / "oneclick_out"


def install_dependencies(skip: bool, upgrade: bool) -> None:
    """Install project requirements unless explicitly skipped."""
    if skip:
        print("[skip] Skipping dependency installation as requested.")
        return
    if not REQUIREMENTS.exists():
        print(f"[warn] requirements.txt not found at {REQUIREMENTS}, skipping install.")
        return

    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(["-r", str(REQUIREMENTS)])

    print("[setup] Installing dependencies...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external command
        raise SystemExit(f"Dependency installation failed (exit code {exc.returncode}).") from exc
    print("[setup] Dependencies ready.")


def ensure_sample_dxf() -> Path:
    """Ensure a sample DXF is available for the demo pipeline."""
    sample_dir = REPO_ROOT / "data" / "samples" / "dxf"
    sample_path = sample_dir / "medallion_sample.dxf"
    if not sample_path.exists():
        sample_dir.mkdir(parents=True, exist_ok=True)
        make_sample(str(sample_path))
        print(f"[demo] Created sample DXF at {sample_path}")
    return sample_path


def run_sample_analysis(dxf_path: Path, output_dir: Path) -> Dict[str, float]:
    """Run the analysis pipeline (demo mode)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    args = Args(
        material="Tan Brown Granite",
        thickness=25.0,
        kerf=1.1,
        rate_per_m=825.0,
        out=str(output_dir),
        use_advanced_toolpath=True,
        rapid_speed=10000.0,
        cutting_speed=1200.0,
        pierce_time=0.5,
        optimize_rapids=True,
        optimize_direction=True,
        entry_strategy="tangent",
        # Pierce-reduction preset
        soften_method="simplify",
        soften_tolerance=0.10,
        fillet_radius_mm=0.50,
        fillet_min_angle_deg=120.0,
        # Normalize to a standard frame for predictable sizing
        normalize_mode="fit",
        target_frame_w_mm=1000.0,
        target_frame_h_mm=1000.0,
        frame_margin_mm=0.0,
        normalize_origin=True,
        require_fit_within_frame=True,
    )
    analyze(str(dxf_path), args)

    report_path = output_dir / "report.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            return data.get("metrics", {})
        except json.JSONDecodeError:
            print("[demo] Warning: could not parse report.json")
    return {}


def run_sample_gcode(dxf_path: Path, output_dir: Path) -> None:
    """Emit toy G-code for the demo pipeline."""
    params = SimpleNamespace(
        dxf=str(dxf_path),
        feed=1200.0,
        m_on="M62",
        m_off="M63",
        pierce_ms=500,
        out=str(output_dir),
    )
    command_gcode(params)


def maybe_open_preview(output_dir: Path) -> None:
    preview = output_dir / "preview.png"
    if preview.exists():
        try:
            webbrowser.open(preview.resolve().as_uri())
            print(f"[demo] Opened preview image -> {preview}")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[demo] Could not open preview automatically: {exc}")


def run_demo(skip_install: bool, upgrade: bool, open_preview: bool) -> None:
    install_dependencies(skip_install, upgrade)
    dxf_path = ensure_sample_dxf()
    output_dir = DEFAULT_OUT

    print("[demo] Running DXF analysis...")
    metrics = run_sample_analysis(dxf_path, output_dir)

    print("[demo] Generating toy G-code...")
    run_sample_gcode(dxf_path, output_dir)

    print("\n[demo] Results ready in 'oneclick_out':")
    print(f"  report.json -> {output_dir / 'report.json'}")
    print(f"  preview.png -> {output_dir / 'preview.png'}")
    print(f"  program.nc -> {output_dir / 'program.nc'}")

    if metrics:
        outer_mm = metrics.get("length_outer_mm", 0.0)
        inner_mm = metrics.get("length_internal_mm", 0.0)
        pierces = metrics.get("pierces", 0)
        cost = metrics.get("cost_inr", 0.0)
        print("\n[demo] Key metrics:")
        print(f"  Outer length: {outer_mm:.1f} mm")
        print(f"  Internal length: {inner_mm:.1f} mm")
        print(f"  Pierces: {pierces}")
        print(f"  Estimated cost: INR {cost:.0f}")

    if open_preview:
        maybe_open_preview(output_dir)


def launch_web_ui(host: str, port: int, open_browser: bool) -> None:
    """Launch the Flask web UI using the existing runner helpers."""
    from run_web_ui import flask_app, wait_and_open

    url = f"http://{host}:{port}/"
    print("[ui] Starting Waterjet DXF Analyzer web UI...")
    print(f"[ui] Listening on {url}")

    if open_browser:
        opener = threading.Thread(target=wait_and_open, args=(url,), daemon=True)
        opener.start()

    try:
        flask_app.run(host=host, port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[ui] Stopping web server...")


def run_ui(skip_install: bool, upgrade: bool, host: str, port: int, no_browser: bool) -> None:
    install_dependencies(skip_install, upgrade)
    launch_web_ui(host=host, port=port, open_browser=not no_browser)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap, install requirements, and either run the UI or the demo pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "demo", "guided", "batch-guided", "all-interfaces"],
        default="ui",
        help="Which workflow to run (default: ui).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to pip while installing requirements.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/interface for the web UI (ui mode).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web UI (ui mode).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser when starting the UI.",
    )
    parser.add_argument(
        "--open-preview",
        action="store_true",
        help="Open the generated preview image after the demo pipeline completes.",
    )
    return parser.parse_args()


def launch_guided_interface(host: str, port: int, no_browser: bool) -> None:
    """Launch the guided individual interface using run_web_ui."""
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "run_web_ui.py",
        "--host", host,
        "--port", str(port),
        "--guided"
    ]
    
    if no_browser:
        cmd.append("--no-browser")
    
    print(f"[guided] Starting Guided Individual Interface on {host}:{port}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[guided] Failed to launch guided interface: {exc}")
    except KeyboardInterrupt:
        print("\n[guided] Stopping guided interface...")


def launch_batch_guided_interface(host: str, port: int, no_browser: bool) -> None:
    """Launch the guided batch interface using run_web_ui."""
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "run_web_ui.py",
        "--host", host,
        "--port", str(port),
        "--batch-guided"
    ]
    
    if no_browser:
        cmd.append("--no-browser")
    
    print(f"[batch-guided] Starting Guided Batch Interface on {host}:{port}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[batch-guided] Failed to launch batch guided interface: {exc}")
    except KeyboardInterrupt:
        print("\n[batch-guided] Stopping batch guided interface...")


def launch_all_interfaces(host: str, port: int, no_browser: bool) -> None:
    """Launch all interfaces using run_web_ui."""
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "run_web_ui.py",
        "--host", host,
        "--port", str(port),
        "--all-interfaces"
    ]
    
    if no_browser:
        cmd.append("--no-browser")
    
    print(f"[all-interfaces] Starting All Interfaces from port {port}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[all-interfaces] Failed to launch all interfaces: {exc}")
    except KeyboardInterrupt:
        print("\n[all-interfaces] Stopping all interfaces...")


def main() -> int:
    args = parse_args()

    if args.mode == "demo":
        run_demo(args.skip_install, args.upgrade, args.open_preview)
    elif args.mode == "guided":
        install_dependencies(args.skip_install, args.upgrade)
        launch_guided_interface(args.host, args.port, args.no_browser)
    elif args.mode == "batch-guided":
        install_dependencies(args.skip_install, args.upgrade)
        launch_batch_guided_interface(args.host, args.port, args.no_browser)
    elif args.mode == "all-interfaces":
        install_dependencies(args.skip_install, args.upgrade)
        launch_all_interfaces(args.host, args.port, args.no_browser)
    else:
        run_ui(args.skip_install, args.upgrade, args.host, args.port, args.no_browser)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
