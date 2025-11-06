# WJP Analyser - Complete Installation Guide

## 🚀 Quick Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Optional Tools

#### Potrace (Required for Image-to-DXF Conversion)
**Windows:**
```bash
# Option 1: WSL (recommended - automatically detected)
wsl bash -c "sudo apt update && sudo apt install -y potrace"

# Option 2: Chocolatey (if available)
choco install potrace

# Option 3: Manual download
# Download from: http://potrace.sourceforge.net/
# Add to PATH: C:\Program Files\potrace\
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install potrace
```

**Linux (RHEL/CentOS/Fedora):**
```bash
sudo yum install potrace
# or
sudo dnf install potrace
```

**macOS:**
```bash
brew install potrace
```

#### Ollama (Optional - for AI Analysis)
```bash
# Download from: https://ollama.ai/
# Follow platform-specific installation instructions
```

#### Inkscape (Optional - for advanced vectorization)
**Windows:**
```bash
# Download from: https://inkscape.org/release/
# Install with command-line tools enabled
# Add to PATH: C:\Program Files\Inkscape\bin\
```

**Linux:**
```bash
sudo apt-get install inkscape
```

**macOS:**
```bash
brew install inkscape
```

## 🔧 Verification

### Check Python Dependencies
```bash
python -c "import streamlit, opencv-python, ezdxf, shapely; print('✅ All Python dependencies installed')"
```

### Check Potrace Installation
```bash
# Windows with WSL
wsl potrace --version
# Should output: potrace 1.16

# Linux/macOS
potrace --version
# Should output: potrace 1.16
```

### Check Ollama Installation (if installed)
```bash
ollama --version
```

### Check Inkscape Installation (if installed)
```bash
inkscape --version
```

## 🚨 Troubleshooting

### Common Issues

**"Potrace is not installed or not found in PATH"**
- Install Potrace using the instructions above
- **Windows**: Install via WSL: `wsl bash -c "sudo apt update && sudo apt install -y potrace"`
- **Linux/macOS**: Ensure Potrace is in your system PATH
- Restart your terminal/command prompt after installation

**"No module named 'streamlit'"**
- Run: `pip install streamlit>=1.34.0`
- Or: `pip install -r requirements.txt`

**"No module named 'ezdxf'"**
- Run: `pip install ezdxf>=1.3.0`
- Or: `pip install -r requirements.txt`

**Web interface won't start**
- Check if port 8501 (Streamlit) is available: `netstat -an | findstr :8501`
- Start Streamlit on a different port: `python run_one_click.py --mode ui --ui-backend streamlit --port 8502`
- Alternatively, try Flask backend on port 5000: `python run_one_click.py --mode ui --ui-backend flask --port 5000`

**Image-to-DXF conversion fails**
- Ensure Potrace is installed and accessible
- Check image format (PNG, JPG supported)
- Try with a simpler image first

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.10+ (3.11+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+

### Recommended Requirements
- **Python**: 3.11+
- **RAM**: 16GB+
- **Storage**: 10GB+ free space
- **GPU**: Optional, for faster image processing

## 🎯 Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt as Administrator for installations
- Chocolatey package manager recommended for easy tool installation
- Add installation directories to PATH environment variable

### Linux
- Use package manager (apt, yum, dnf) for system tools
- Virtual environment recommended for Python dependencies
- May need to install additional system libraries

### macOS
- Homebrew package manager recommended
- Xcode command line tools may be required
- Python installed via Homebrew works best

## 🔄 Updates

### Update Python Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Update System Tools
**Windows (Chocolatey):**
```bash
choco upgrade potrace inkscape
```

**Linux:**
```bash
sudo apt-get update && sudo apt-get upgrade potrace inkscape
```

**macOS:**
```bash
brew upgrade potrace inkscape
```

## ✅ Installation Complete

Once all dependencies are installed, you can:

1. **Start the web interface (Streamlit)**: `python run_one_click.py --mode ui --ui-backend streamlit --host 127.0.0.1 --port 8501`
2. **Start the web interface (Flask)**: `python run_one_click.py --mode ui --ui-backend flask --host 127.0.0.1 --port 5000`
3. **Run command-line tools**: `python main.py --help`
4. **Test image-to-DXF conversion**: Upload an image in the web interface

For more information, see the [User Manual](USER_MANUAL.md) and [Quick Start Guide](QUICK_START_GUIDE.md).

---

**Need help?** Check the troubleshooting section above or refer to the project documentation.
