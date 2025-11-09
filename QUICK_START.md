# WJP ANALYSER - Quick Start Guide

## 🚀 Getting Started

### Recommended: New Unified CLI

```bash
# Launch Streamlit web UI (recommended)
python -m wjp_analyser.cli.wjp_cli web

# The app will open at http://localhost:8501
```

### Alternative Methods (Deprecated)

```bash
# DEPRECATED: Legacy launcher
python run_one_click.py --mode ui --ui-backend streamlit

# DEPRECATED: Old unified launcher
python wjp_analyser_unified.py web-ui
```

## 📋 Basic Workflows

### 1. Analyze a DXF File

**Via Web UI:**
1. Launch: `python -m wjp_analyser.cli.wjp_cli web`
2. Navigate to "DXF Analyzer" page
3. Upload your DXF file
4. Review analysis results and AI recommendations
5. Download optimized DXF or CSV report

**Via Python API:**
```python
from wjp_analyser.services.analysis_service import run_analysis

report = run_analysis("your_file.dxf", out_dir="output")
print(f"Objects: {report['metrics']['object_count']}")
print(f"Cutting length: {report['metrics']['length_internal_mm']} mm")
```

**Via FastAPI (if server running):**
```bash
curl -X POST http://localhost:8000/analyze-dxf \
  -H "Content-Type: application/json" \
  -d '{"dxf_path": "your_file.dxf"}'
```

### 2. Convert Image to DXF

**Via Web UI:**
1. Navigate to "Image → DXF" page
2. Upload image (PNG, JPG, etc.)
3. Adjust preprocessing settings
4. Preview and edit detected objects
5. Export to DXF

**Via Python:**
```python
from wjp_analyser.image_processing.converters.enhanced_opencv_converter import EnhancedOpenCVImageToDXFConverter

converter = EnhancedOpenCVImageToDXFConverter()
dxf_path = converter.convert("image.png", output_path="output.dxf")
```

### 3. Run Advanced Nesting

**Via Web UI:**
1. Navigate to "Nesting" page
2. Upload multiple DXF files
3. Configure constraints (kerf margin, min web, etc.)
4. Select algorithm (BLF, NFP, Genetic)
5. Run optimization
6. Review utilization and download nested layout

**Via Python:**
```python
from wjp_analyser.nesting import GeometryHygiene, BottomLeftFillEngine, ConstraintSet
from wjp_analyser.nesting.constraints import HardConstraints, SoftConstraints

# Clean geometry
hygiene = GeometryHygiene(tolerance_microns=1.0)
cleaned_polygons = hygiene.clean_polygon_list(polygons)

# Setup constraints
hard = HardConstraints(kerf_margin=1.0, min_web=2.0)
soft = SoftConstraints(compactness_weight=0.5)
constraints = ConstraintSet(hard, soft)

# Run nesting
blf = BottomLeftFillEngine(sheet_width=1000, sheet_height=1000)
placements = blf.place_objects(cleaned_polygons)
```

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# Enable API mode (auto-detected by default)
export WJP_USE_API=true

# Redis for async jobs
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Performance tuning
export WJP_STREAMING_THRESHOLD=10485760  # 10MB
```

### Material Settings

Edit `config/material_profiles.py` to customize material parameters.

## 📊 Performance Tips

### For Large Files (>10MB)
- Streaming parser automatically enabled
- Results are cached (subsequent analyses instant)
- Use memory optimization: `coordinate_precision=3, use_float32=True`

### For Fast Iteration
- Enable caching (automatic)
- Use API with async mode for long operations
- Start background worker: `python -m wjp_analyser.cli.wjp_cli worker`

## 🐛 Troubleshooting

### App Won't Start
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt
```

### Large Files Cause Errors
- Automatic streaming parser handles this
- Check available memory
- Reduce `coordinate_precision` or enable `use_float32`

### Slow Performance
- Enable caching (automatic)
- Use async mode for long operations
- Start Redis worker for background processing

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [WJP_ANALYSER_IMPROVEMENT_ROADMAP.md](WJP_ANALYSER_IMPROVEMENT_ROADMAP.md) for feature details
- Explore API documentation at `http://localhost:8000/docs` (when API running)

---

**Need help?** Check the main README or open an issue on GitHub.








