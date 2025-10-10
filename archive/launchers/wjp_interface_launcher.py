#!/usr/bin/env python3
"""
WJP Interface Launcher - Help and Quick Start
=============================================

This script provides help and quick access to all WJP interfaces.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def show_help():
    """Display comprehensive help for all WJP interfaces."""
    print("🚀 **WJP ANALYZER - INTERFACE LAUNCHER**")
    print("=" * 60)
    print("")
    print("📋 **AVAILABLE INTERFACES:**")
    print("")
    print("1️⃣ **Main Interface** (Port 8501)")
    print("   🎯 Complete WJP analysis tools")
    print("   📁 Individual pages for each function")
    print("   🔧 Advanced features and customization")
    print("   🌐 URL: http://localhost:8501")
    print("")
    print("2️⃣ **Guided Individual Interface** (Port 8504)")
    print("   🎯 Step-by-step guidance for individual projects")
    print("   📝 From prompt to PDF report")
    print("   💡 Intelligent tips and warnings")
    print("   🌐 URL: http://localhost:8504")
    print("")
    print("3️⃣ **Guided Batch Interface** (Port 8505)")
    print("   📦 Intelligent batch processing")
    print("   🧠 Smart optimization suggestions")
    print("   📊 Comprehensive analysis and reporting")
    print("   🌐 URL: http://localhost:8505")
    print("")
    print("4️⃣ **All Interfaces** (Ports 8501, 8504, 8505)")
    print("   🚀 Launch all interfaces simultaneously")
    print("   🔄 Access all features from different ports")
    print("   📊 Complete WJP ecosystem")
    print("")
    print("=" * 60)
    print("")
    print("🛠️ **LAUNCH COMMANDS:**")
    print("")
    print("**Using run_web_ui.py:**")
    print("  python run_web_ui.py                    # Main interface")
    print("  python run_web_ui.py --guided            # Guided individual")
    print("  python run_web_ui.py --batch-guided      # Guided batch")
    print("  python run_web_ui.py --all-interfaces    # All interfaces")
    print("")
    print("**Using run_one_click.py:**")
    print("  python run_one_click.py --mode ui         # Main interface")
    print("  python run_one_click.py --mode guided     # Guided individual")
    print("  python run_one_click.py --mode batch-guided # Guided batch")
    print("  python run_one_click.py --mode all-interfaces # All interfaces")
    print("")
    print("**Using dedicated launchers:**")
    print("  python launch_guided_interfaces.py       # Guided interfaces only")
    print("  python launch_wjp_automation.py          # WJP automation pipeline")
    print("")
    print("=" * 60)
    print("")
    print("🎯 **QUICK START RECOMMENDATIONS:**")
    print("")
    print("**For Beginners:**")
    print("  🎯 Start with: python run_one_click.py --mode guided")
    print("  📚 Get step-by-step guidance for your first project")
    print("  💡 Learn the workflow with intelligent tips")
    print("")
    print("**For Regular Users:**")
    print("  🚀 Start with: python run_one_click.py --mode all-interfaces")
    print("  🔄 Access all interfaces from different ports")
    print("  📊 Use the most appropriate interface for each task")
    print("")
    print("**For Batch Processing:**")
    print("  📦 Start with: python run_one_click.py --mode batch-guided")
    print("  🧠 Get intelligent batch processing guidance")
    print("  📈 Optimize your workflow with suggestions")
    print("")
    print("**For Advanced Users:**")
    print("  🔧 Start with: python run_one_click.py --mode ui")
    print("  ⚙️ Access all advanced features directly")
    print("  🎛️ Full control over all parameters")
    print("")
    print("=" * 60)
    print("")
    print("📚 **FEATURE COMPARISON:**")
    print("")
    print("| Feature | Main UI | Guided Individual | Guided Batch |")
    print("|---------|---------|-------------------|--------------|")
    print("| Step-by-step guidance | ❌ | ✅ | ✅ |")
    print("| Individual projects | ✅ | ✅ | ❌ |")
    print("| Batch processing | ✅ | ❌ | ✅ |")
    print("| Advanced features | ✅ | ❌ | ❌ |")
    print("| Intelligent tips | ❌ | ✅ | ✅ |")
    print("| Progress tracking | ❌ | ✅ | ✅ |")
    print("| Optimization suggestions | ❌ | ❌ | ✅ |")
    print("| Learning system | ✅ | ✅ | ✅ |")
    print("")
    print("=" * 60)
    print("")
    print("🔧 **ADVANCED OPTIONS:**")
    print("")
    print("**Custom Ports:**")
    print("  python run_web_ui.py --port 9000 --guided")
    print("")
    print("**No Browser Auto-Open:**")
    print("  python run_web_ui.py --no-browser --guided")
    print("")
    print("**Custom Host:**")
    print("  python run_web_ui.py --host 0.0.0.0 --guided")
    print("")
    print("**Skip Dependency Installation:**")
    print("  python run_one_click.py --skip-install --mode guided")
    print("")
    print("=" * 60)
    print("")
    print("❓ **NEED HELP?**")
    print("")
    print("📖 Documentation:")
    print("  - WJP_GUIDED_INTERFACES_DOCUMENTATION.md")
    print("  - WJP_AUTOMATION_PIPELINE_COMPLETE.md")
    print("  - README.md")
    print("")
    print("🎥 Tutorials:")
    print("  - Watch guided interface tutorials")
    print("  - Follow step-by-step examples")
    print("  - Learn best practices")
    print("")
    print("🆘 Support:")
    print("  - Check system requirements")
    print("  - Verify file permissions")
    print("  - Review error messages")
    print("")
    print("=" * 60)

def launch_interface(interface_type: str, port: int = None):
    """Launch a specific interface."""
    if interface_type == "main":
        cmd = [sys.executable, "run_web_ui.py"]
        if port:
            cmd.extend(["--port", str(port)])
        print("🚀 Launching Main Interface...")
        
    elif interface_type == "guided":
        cmd = [sys.executable, "run_web_ui.py", "--guided"]
        if port:
            cmd.extend(["--port", str(port)])
        print("🎯 Launching Guided Individual Interface...")
        
    elif interface_type == "batch":
        cmd = [sys.executable, "run_web_ui.py", "--batch-guided"]
        if port:
            cmd.extend(["--port", str(port)])
        print("📦 Launching Guided Batch Interface...")
        
    elif interface_type == "all":
        cmd = [sys.executable, "run_web_ui.py", "--all-interfaces"]
        if port:
            cmd.extend(["--port", str(port)])
        print("🚀 Launching All Interfaces...")
        
    else:
        print(f"❌ Unknown interface type: {interface_type}")
        return False
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to launch interface: {exc}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Interface stopped by user")
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="WJP Interface Launcher - Help and Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--help-full",
        action="store_true",
        help="Show comprehensive help and interface information"
    )
    
    parser.add_argument(
        "--launch",
        choices=["main", "guided", "batch", "all"],
        help="Launch a specific interface"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Custom port for the interface"
    )
    
    args = parser.parse_args()
    
    if args.help_full:
        show_help()
        return 0
    
    if args.launch:
        success = launch_interface(args.launch, args.port)
        return 0 if success else 1
    
    # Default: show basic help
    print("🚀 **WJP ANALYZER - INTERFACE LAUNCHER**")
    print("=" * 50)
    print("")
    print("🎯 **QUICK LAUNCH OPTIONS:**")
    print("")
    print("1. Main Interface:")
    print("   python run_web_ui.py")
    print("")
    print("2. Guided Individual Interface:")
    print("   python run_web_ui.py --guided")
    print("")
    print("3. Guided Batch Interface:")
    print("   python run_web_ui.py --batch-guided")
    print("")
    print("4. All Interfaces:")
    print("   python run_web_ui.py --all-interfaces")
    print("")
    print("5. One-Click Launcher:")
    print("   python run_one_click.py --mode guided")
    print("")
    print("=" * 50)
    print("")
    print("📚 **For detailed help and feature comparison:**")
    print("   python wjp_interface_launcher.py --help-full")
    print("")
    print("🚀 **For quick launch:**")
    print("   python wjp_interface_launcher.py --launch guided")
    print("")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
