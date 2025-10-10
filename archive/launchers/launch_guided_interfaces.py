#!/usr/bin/env python3
"""
WJP Guided Interfaces - Launcher
================================

This script launches the guided interfaces for both individual and batch processing.
"""

import subprocess
import sys
import os
from pathlib import Path

def launch_guided_interface():
    """Launch the guided individual processing interface."""
    print("🎯 **Launching WJP Guided Interface**")
    print("=" * 50)
    
    script_dir = Path(__file__).parent
    interface_script = script_dir / "wjp_guided_interface.py"
    
    if not interface_script.exists():
        print(f"❌ Interface script not found: {interface_script}")
        return False
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(interface_script),
            "--server.port", "8504",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🌐 Starting Guided Interface on port 8504...")
        print(f"📱 Open your browser and go to: http://localhost:8504")
        print(f"🔄 Press Ctrl+C to stop the server")
        print(f"")
        print(f"🎯 **Guided Interface Features:**")
        print(f"   ✅ Step-by-step guidance for individual projects")
        print(f"   ✅ Intelligent tips and warnings")
        print(f"   ✅ Progress tracking and validation")
        print(f"   ✅ Contextual help and recommendations")
        print(f"   ✅ Complete workflow from prompt to PDF")
        print(f"")
        
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Guided Interface: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def launch_guided_batch_interface():
    """Launch the guided batch processing interface."""
    print("📦 **Launching WJP Guided Batch Interface**")
    print("=" * 50)
    
    script_dir = Path(__file__).parent
    interface_script = script_dir / "wjp_guided_batch_interface.py"
    
    if not interface_script.exists():
        print(f"❌ Interface script not found: {interface_script}")
        return False
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(interface_script),
            "--server.port", "8505",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🌐 Starting Guided Batch Interface on port 8505...")
        print(f"📱 Open your browser and go to: http://localhost:8505")
        print(f"🔄 Press Ctrl+C to stop the server")
        print(f"")
        print(f"📦 **Guided Batch Interface Features:**")
        print(f"   ✅ Step-by-step guidance for batch processing")
        print(f"   ✅ Intelligent batch planning and optimization")
        print(f"   ✅ Real-time progress monitoring")
        print(f"   ✅ Comprehensive results analysis")
        print(f"   ✅ Optimization suggestions and learning")
        print(f"")
        
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Guided Batch Interface: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main launcher function."""
    print("🚀 **WJP Guided Interfaces Launcher**")
    print("=" * 60)
    
    print("Choose which guided interface to launch:")
    print("1. 🎯 Individual Project Guidance (Port 8504)")
    print("2. 📦 Batch Processing Guidance (Port 8505)")
    print("3. 🚀 Launch Both Interfaces")
    print("4. ❌ Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        success = launch_guided_interface()
        if success:
            print("✅ Guided Interface launched successfully!")
        else:
            print("❌ Failed to launch Guided Interface")
            sys.exit(1)
    
    elif choice == "2":
        success = launch_guided_batch_interface()
        if success:
            print("✅ Guided Batch Interface launched successfully!")
        else:
            print("❌ Failed to launch Guided Batch Interface")
            sys.exit(1)
    
    elif choice == "3":
        print("🚀 Launching both interfaces...")
        print("Note: You'll need to run this script twice or use separate terminals")
        print("First, launch individual guidance:")
        success1 = launch_guided_interface()
        if success1:
            print("✅ Individual interface launched!")
        print("\nThen launch batch guidance:")
        success2 = launch_guided_batch_interface()
        if success2:
            print("✅ Batch interface launched!")
    
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
