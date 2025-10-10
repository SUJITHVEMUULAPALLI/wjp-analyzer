# 🚀 WJP ANALYSER - Complete Usage Guide

## Quick Start Commands

### 1. Image-to-DXF Conversion

#### OpenCV Converter (Recommended)
```bash
# Basic usage
python image_processing/converters/opencv_converter.py

# Custom parameters (modify the script)
# - binary_threshold: 0-255 (lower = more sensitive)
# - min_area: minimum contour area to keep
# - dxf_size: output DXF canvas size in mm
```

#### Multi-Shade Converter
```bash
# For complex images with multiple materials
python image_processing/converters/multishade.py
# Creates BACKGROUND, GOLD, WHITE layers automatically
```

#### Basic Converter
```bash
# Simple threshold-based conversion
python image_processing/converters/basic.py
```

### 2. DXF Analysis & G-code Generation

```bash
# Analyze DXF for waterjet cutting
python -m cli.wjdx analyze your_file.dxf --out output --material "Tan Brown Granite" --thickness 25 --kerf 1.1 --rate-per-m 825

# Generate G-code
python -m cli.wjdx gcode your_file.dxf --out output --post generic

# Convert image to DXF (built-in CLI)
python -m cli.wjdx image your_image.png --out output --scale 1.0 --edge-threshold 0.33 --min-area 100
```

### 3. One-Click Launchers

#### Web Interface
```bash
python run_web_ui.py
# Opens browser at http://localhost:5000
# Upload images/DXF files, adjust parameters, generate reports
```

#### Demo Mode
```bash
python run_one_click.py --mode demo
# Runs complete sample pipeline with medallion_sample.dxf
# Generates analysis reports and opens preview
```

#### Full Demo with Web UI
```bash
python run_one_click.py
# Installs dependencies, starts Flask UI, opens browser
```

### 4. Utility Tools

```bash
# Clean existing DXF files
python tools/clean_dxf.py

# Generate sample DXF
python scripts/make_sample_dxf.py samples/medallion_sample.dxf
```

## 📁 File Structure

```
WJP_ANALYSER/
├── 📁 core/                    # Core analysis engine
│   ├── api.py                  # Main API functions
│   ├── io_dxf.py              # DXF I/O operations
│   ├── geom_clean.py          # Geometry cleaning
│   ├── topology.py            # Topology analysis
│   ├── classify.py             # Classification logic
│   ├── checks.py              # Validation checks
│   ├── metrics.py              # Metrics calculation
│   ├── toolpath.py             # Toolpath planning
│   ├── gcode.py               # G-code generation
│   ├── report.py              # Report generation
│   ├── viz.py                 # Visualization
│   └── nesting.py             # Nesting algorithms
│
├── 📁 image_processing/        # Image-to-DXF conversion
│   ├── converters/
│   │   ├── opencv_converter.py # OpenCV-based (recommended)
│   │   ├── multishade.py      # Multi-shade K-means
│   │   └── basic.py           # Basic threshold
│   └── integrated_pipeline.py  # Complete pipeline
│
├── 📁 cli/                    # Command-line interface
│   └── wjdx.py               # CLI commands
│
├── 📁 web/                    # Web interface
│   ├── app.py                # Flask application
│   ├── templates/             # HTML templates
│   └── static/               # CSS/JS assets
│
├── 📁 tools/                  # Utility tools
│   └── clean_dxf.py          # DXF cleaning utility
│
├── 📁 data/samples/           # Sample files
├── 📁 output/                # Output directory
└── 📁 config/                # Configuration files
```

## 🔧 Advanced Usage

### Custom Converter Parameters

```python
from image_processing.converters.opencv_converter import OpenCVImageToDXFConverter

# Create converter with custom settings
converter = OpenCVImageToDXFConverter(
    binary_threshold=100,  # 0-255, lower = more sensitive
    line_pitch=1.0,        # Line simplification factor
    min_area=200,          # Minimum contour area (pixels)
    dxf_size=1000.0        # Output DXF size (mm)
)

# Convert image
result = converter.convert_image_to_dxf(
    input_image="your_image.png",
    output_dxf="output.dxf",
    preview_output="preview.png"
)

print(f"Found {result['contours_found']} contours")
print(f"Kept {result['contours_kept']} after filtering")
print(f"Generated {result['polygons']} polygons")
```

### Complete Pipeline Integration

```python
from image_processing.converters.opencv_converter import OpenCVImageToDXFConverter
from core.api import AnalyzeArgs, analyze_dxf

# Step 1: Convert image to DXF
converter = OpenCVImageToDXFConverter()
converter.convert_image_to_dxf("image.png", "output.dxf")

# Step 2: Analyze for waterjet cutting
args = AnalyzeArgs(
    material="Tan Brown Granite",
    thickness=25.0,
    kerf=1.1,
    rate_per_m=825.0,
    out="output"
)
analysis = analyze_dxf("output.dxf", args)

# Step 3: Generate G-code
from cli.wjdx import command_gcode
command_gcode(SimpleNamespace(dxf="output.dxf", out="output", post="generic"))
```

## 📊 Output Files

### Image Conversion Outputs
- `*.dxf` - Generated DXF file
- `*_preview.png` - 3-panel preview (original → thresholded → DXF)
- `segmentation_preview.png` - K-means clustering visualization (multishade only)

### Analysis Outputs
- `report.json` - Detailed analysis report
- `lengths.csv` - Cutting lengths and metrics
- `preview.png` - Analysis visualization
- `program.nc` - Generated G-code

## 🎯 Parameter Guide

### OpenCV Converter Parameters
- **binary_threshold** (0-255): Lower values = more sensitive to edges
- **min_area** (pixels): Filter out small contours/noise
- **dxf_size** (mm): Output DXF canvas size
- **line_pitch** (factor): Line simplification (higher = more simplified)

### Analysis Parameters
- **material**: Material name (affects cutting parameters)
- **thickness** (mm): Material thickness
- **kerf** (mm): Kerf width for compensation
- **rate_per_m** (units): Cutting rate per meter

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Try demo**: `python run_one_click.py --mode demo`
3. **Convert your image**: `python image_processing/converters/opencv_converter.py`
4. **Analyze DXF**: `python -m cli.wjdx analyze your_file.dxf --out output`
5. **Generate G-code**: `python -m cli.wjdx gcode your_file.dxf --out output`

## 📞 Support

- **Documentation**: See README.md for detailed information
- **Examples**: Check `examples/` directory for usage patterns
- **Samples**: Use `data/samples/` for testing
- **Issues**: Report bugs and feature requests via GitHub
