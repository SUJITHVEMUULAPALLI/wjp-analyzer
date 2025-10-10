#!/usr/bin/env python3
"""
WJP Automation Pipeline - Streamlit App Launcher
================================================

This script launches the WJP automation Streamlit interface.
"""

import subprocess
import sys
import os
from pathlib import Path

def launch_wjp_automation_interface():
    """Launch the WJP automation Streamlit interface."""
    print("🚀 **Launching WJP Automation Pipeline Interface**")
    print("=" * 60)
    
    # Get the script directory
    script_dir = Path(__file__).parent
    interface_script = script_dir / "wjp_streamlit_interface.py"
    
    if not interface_script.exists():
        print(f"❌ Interface script not found: {interface_script}")
        return False
    
    try:
        # Launch Streamlit app
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(interface_script),
            "--server.port", "8503",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🌐 Starting Streamlit server on port 8503...")
        print(f"📱 Open your browser and go to: http://localhost:8503")
        print(f"🔄 Press Ctrl+C to stop the server")
        print(f"")
        print(f"🎯 **WJP Automation Pipeline Features:**")
        print(f"   ✅ Complete automation from Prompt → Image → DXF → Analysis → PDF")
        print(f"   ✅ Intelligent supervisor agent orchestration")
        print(f"   ✅ Professional file naming standards")
        print(f"   ✅ Comprehensive reporting and visualization")
        print(f"   ✅ Batch processing capabilities")
        print(f"   ✅ Real-time job monitoring")
        print(f"   ✅ Material-specific cost calculations")
        print(f"")
        
        # Run the command
        subprocess.run(cmd, check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = launch_wjp_automation_interface()
    if success:
        print("✅ WJP Automation Pipeline Interface launched successfully!")
    else:
        print("❌ Failed to launch WJP Automation Pipeline Interface")
        sys.exit(1)
