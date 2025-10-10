#!/usr/bin/env python3
"""
Advanced Batch Processing Streamlit App Launcher
===============================================

This script launches the advanced batch processing Streamlit interface.
"""

import subprocess
import sys
import os
from pathlib import Path

def launch_advanced_batch_interface():
    """Launch the advanced batch processing Streamlit interface."""
    print("🚀 **Launching Advanced Batch Processing Interface**")
    print("=" * 60)
    
    # Get the script directory
    script_dir = Path(__file__).parent
    interface_script = script_dir / "advanced_batch_interface.py"
    
    if not interface_script.exists():
        print(f"❌ Interface script not found: {interface_script}")
        return False
    
    try:
        # Launch Streamlit app
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(interface_script),
            "--server.port", "8502",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🌐 Starting Streamlit server on port 8502...")
        print(f"📱 Open your browser and go to: http://localhost:8502")
        print(f"🔄 Press Ctrl+C to stop the server")
        
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
    success = launch_advanced_batch_interface()
    if success:
        print("✅ Advanced Batch Processing Interface launched successfully!")
    else:
        print("❌ Failed to launch Advanced Batch Processing Interface")
        sys.exit(1)
