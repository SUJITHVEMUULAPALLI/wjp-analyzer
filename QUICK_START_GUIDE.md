# Waterjet Analyser - Quick Start Guide

## 🚀 Getting Started (Tomorrow)

### 1. Start the System
```bash
# Navigate to project directory
cd "C:\WJP ANALYSER"

# Start web interface (Streamlit)
C:\Python313\python.exe run_one_click.py --mode ui --ui-backend streamlit --host 127.0.0.1 --port 8501

# Open browser to: http://localhost:8501
```

### 2. Basic DXF Analysis
```bash
# Analyze a DXF file
C:\Python313\python.exe main.py analyze data/samples/dxf/medallion_sample.dxf --out analysis_output

# View results in: analysis_output/report.json
```

### 3. AI-Powered Analysis
```bash
# Ollama analysis (recommended)
C:\Python313\python.exe main.py ollama-analyze data/samples/dxf/medallion_sample.dxf --out ai_output --model waterjet:latest

# Design suggestions
C:\Python313\python.exe main.py ollama-design "geometric pattern for waterjet cutting" --out design_output
```

### 4. Clean DXF Files
```bash
# Basic cleaning
C:\Python313\python.exe tools/clean_dxf.py

# Advanced cleaning with spacing fixes
C:\Python313\python.exe tools/advanced_dxf_cleaner.py data/samples/dxf/medallion_sample.dxf -o cleaned.dxf --min-spacing 3.0

# Create waterjet-ready designs
C:\Python313\python.exe tools/create_simple_medallion.py
```

## 📋 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `analyze` | Basic DXF analysis | `main.py analyze file.dxf --out output` |
| `ollama-analyze` | AI manufacturing analysis | `main.py ollama-analyze file.dxf --out ai_output` |
| `ollama-design` | AI design suggestions | `main.py ollama-design "pattern description" --out design` |
| `opencv` | Image to DXF conversion | `main.py opencv image.png --out dxf_output` |
| `advanced-toolpath` | Optimized toolpath generation | `main.py advanced-toolpath file.dxf --out toolpath` |
| `cam-process` | Professional CAM processing | `main.py cam-process file.dxf --out cam_output` |

## 🤖 AI Models Available

- **`waterjet:latest`** - Custom waterjet model (20.9B parameters)
- **`llama3.2-vision:latest`** - Vision model (10.7B parameters)
- **`gpt-oss:20b`** - Open source GPT (20.9B parameters)

## 📁 Key Directories

- **`data/samples/`** - Sample DXF files and images
- **`tools/`** - DXF cleaning and design tools
- **`src/wjp_analyser/`** - Core analysis modules
- **`config/`** - Configuration files
- **`output/`** - Analysis results

## 🔧 Troubleshooting

### Potrace Not Found
```bash
# Check if Potrace is installed
potrace --version

# Windows: Install via Chocolatey
choco install potrace

# Linux: Install via package manager
sudo apt-get install potrace  # Ubuntu/Debian
sudo yum install potrace      # RHEL/CentOS

# macOS: Install via Homebrew
brew install potrace
```

### Ollama Not Responding
```bash
# Check if Ollama is running
netstat -an | findstr :11434

# Test simple request
C:\Python313\python.exe -c "import requests; print(requests.get('http://localhost:11434/api/tags').json())"
```

### Web Interface Issues
```bash
# Check if port 8501 (Streamlit) is in use
netstat -an | findstr :8501

# Start Streamlit on a different port
C:\Python313\python.exe run_one_click.py --mode ui --ui-backend streamlit --port 8502

# Alternatively, use Flask backend on port 5000
C:\Python313\python.exe run_one_click.py --mode ui --ui-backend flask --port 5000
```

### DXF Analysis Errors
```bash
# Check file exists
dir data\samples\dxf\

# Test with simple file
C:\Python313\python.exe main.py analyze data/samples/dxf/medallion_sample.dxf --out test
```

## 📊 Understanding Analysis Results

### Report Structure
- **`metrics`** - Cutting length, time, cost estimates
- **`violations`** - Open contours, spacing issues
- **`entities`** - Geometry counts and types
- **`toolpath`** - Optimized cutting order

### Key Metrics
- **Cutting Length**: Total distance to cut
- **Estimated Time**: Manufacturing time in minutes
- **Cost**: Estimated cost in INR
- **Pierces**: Number of pierce points needed

## 🎯 Best Practices

1. **Always check violations** before manufacturing
2. **Use waterjet-ready designs** for best results
3. **Run AI analysis** for manufacturing insights
4. **Clean DXF files** before processing
5. **Test with web interface** for user-friendly workflow

## 📞 Quick Help

- **Web Interface**: http://localhost:8501 (Streamlit default)
- **Ollama Status**: http://localhost:11434/api/tags
- **Project Root**: C:\WJP ANALYSER
- **Python Path**: C:\Python313\python.exe

**Ready to analyze and optimize your waterjet cutting projects! 🚀**
