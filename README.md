# WJP ANALYSER - Waterjet Cutting Analysis System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive, production-ready waterjet cutting analysis system with DXF processing, AI-powered analysis, image-to-DXF conversion, advanced nesting optimization, and performance enhancements for enterprise use.

## 🚀 Quick Start

### Minimal Startup (Simplest)

```bash
# Just run this to start the web UI
python run.py
```

That's it! The web interface will open at `http://127.0.0.1:8501`

### Recommended: New Unified CLI

```bash
# Launch Streamlit web UI (recommended)
python -m wjp_analyser.cli.wjp_cli web

# Launch FastAPI server
python -m wjp_analyser.cli.wjp_cli api

# Start background worker for async jobs
python -m wjp_analyser.cli.wjp_cli worker

# View all commands
python -m wjp_analyser.cli.wjp_cli --help
```

### Alternative: Legacy Launchers (Deprecated)

```bash
# DEPRECATED: Use 'wjp' command instead
python run_one_click.py --mode ui --ui-backend streamlit

# DEPRECATED: Use 'wjp api' instead
python wjp_analyser_unified.py api
```

## ✨ Key Features

### 📊 Advanced DXF Analysis Engine
- **Performance Optimized**: Streaming parser for large files (>10MB), memory optimization, intelligent caching
- **Geometric Analysis**: Comprehensive analysis of DXF entities with quality checks
- **AI-Powered Recommendations**: Executable operation suggestions with auto-apply options
- **Shape Grouping**: Intelligent grouping of similar shapes for efficient processing
- **Cost Estimation**: Material-specific cost calculation with caching
- **Quality Assessment**: Detection of issues (tiny segments, open contours, min radius/spacing violations)

### 🏭 Production-Grade Nesting
- **Geometry Hygiene**: Robust polygonization, hole handling, winding rules, tolerance unification
- **Advanced Placement**: Bottom-Left Fill, NFP refinement, genetic algorithms, simulated annealing
- **Constraint System**: Hard constraints (kerf margin, min web, pierce zones) and soft constraints (priorities, grain direction)
- **Determinism Mode**: Reproducible nesting results with seed control
- **Spatial Indexing**: STRtree-based collision detection for fast performance

### 🖼️ Image to DXF Conversion
- **Multiple Algorithms**: Support for Potrace, OpenCV-enhanced, and texture-aware conversion
- **Object Detection**: Advanced contour detection with shape classification
- **Interactive Editing**: Live editing interface with object selection and modification
- **Batch Processing**: Support for processing multiple images

### 🤖 AI-Powered Features
- **Executable Recommendations**: Rule+AI hybrid system with auto-apply operations
- **Design Analysis**: AI-driven analysis with actionable fixes
- **Readiness Scoring**: Automatic scoring of DXF files for waterjet readiness
- **Material Recommendations**: AI-powered material selection

### 🌐 Modern Architecture
- **API-First Design**: FastAPI backend with automatic fallback to direct services
- **Async Job Processing**: Redis Queue (RQ) for background tasks
- **Service Layer**: Centralized business logic (analysis, costing, editing, caching)
- **Performance Optimizations**: Streaming parsers, memory optimization, intelligent caching
- **Unified CLI**: Single entry point (`wjp` command) for all operations

### 🎨 Web Interface
- **Multi-Page Design**: Separate pages for different workflows
- **Real-Time Preview**: Live preview with jobs drawer for async operations
- **Interactive Editing**: Point-and-click editing with error handling
- **Wizard Flows**: Guided workflows for complex operations
- **Professional UX**: Actionable error messages, terminology standardization

## 📁 Project Structure

```
WJP ANALYSER/
├── src/
│   └── wjp_analyser/
│       ├── analysis/              # DXF analysis engine
│       │   └── dxf_analyzer.py   # Core analyzer with performance optimizations
│       ├── services/              # Service layer (NEW)
│       │   ├── analysis_service.py
│       │   ├── costing_service.py
│       │   ├── editor_service.py
│       │   ├── csv_analysis_service.py
│       │   └── layered_dxf_service.py
│       ├── performance/           # Performance optimizations (NEW)
│       │   ├── streaming_parser.py
│       │   ├── cache_manager.py
│       │   └── memory_optimizer.py
│       ├── nesting/               # Production-grade nesting (NEW)
│       │   ├── geometry_hygiene.py
│       │   ├── placement_engine.py
│       │   └── constraints.py
│       ├── api/                   # FastAPI backend (NEW)
│       │   ├── fastapi_app.py
│       │   ├── queue_manager.py
│       │   └── worker.py
│       ├── web/                   # Web interface
│       │   ├── unified_web_app.py
│       │   ├── api_client.py
│       │   ├── api_client_wrapper.py
│       │   ├── components/        # Reusable UI components (NEW)
│       │   │   ├── wizard.py
│       │   │   ├── jobs_drawer.py
│       │   │   ├── error_handler.py
│       │   │   └── terminology.py
│       │   └── pages/
│       ├── ai/                    # AI integration
│       │   └── recommendation_engine.py  # Executable operations (NEW)
│       ├── cli/                   # Command-line tools
│       │   └── wjp_cli.py        # Unified CLI (NEW)
│       └── image_processing/      # Image-to-DXF conversion
├── config/                        # Configuration files
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- **Python**: 3.10 or higher
- **Redis**: For async job processing (optional, see below)
- **Potrace**: Optional, for advanced vectorization

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Key dependencies include:
# - Core: numpy, opencv-python, pillow, matplotlib, pandas
# - DXF Processing: ezdxf, shapely
# - Web: streamlit, fastapi, uvicorn
# - Performance: rq, redis (for job queue)
# - AI: openai, requests
```

### Optional: Redis for Async Jobs

For background job processing:

```bash
# Windows (Docker)
docker run -d -p 6379:6379 redis

# Linux
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis
```

### Optional: Potrace (Advanced Vectorization)

```bash
# Windows (Chocolatey)
choco install potrace

# Linux (Ubuntu/Debian)
sudo apt-get install potrace

# macOS
brew install potrace
```

## 🚀 Usage

### Web Interface

```bash
# Minimal startup (simplest)
python run.py

# With custom port
python run.py --port 8080

# With custom host and port
python run.py --host 0.0.0.0 --port 8501

# Or use the full CLI
python -m wjp_analyser.cli.wjp_cli web
```

### API Server

```bash
# Launch FastAPI server
python -m wjp_analyser.cli.wjp_cli api

# With auto-reload (development)
python -m wjp_analyser.cli.wjp_cli api --reload

# Custom host/port
python -m wjp_analyser.cli.wjp_cli api --host 0.0.0.0 --port 8000
```

### Background Worker (for async jobs)

```bash
# Start worker (requires Redis)
python -m wjp_analyser.cli.wjp_cli worker

# Specific queues
python -m wjp_analyser.cli.wjp_cli worker --queues analysis,conversion

# Burst mode (exit when no jobs)
python -m wjp_analyser.cli.wjp_cli worker --burst
```

### Command Line Analysis

```bash
# Using Python API directly
python -c "from wjp_analyser.services.analysis_service import run_analysis; print(run_analysis('file.dxf'))"
```

## 📖 Workflows

### 1. DXF Analysis Workflow

1. **Upload DXF** through web interface or API
2. **Automatic Analysis** with performance optimizations:
   - Large files use streaming parser
   - Results cached for repeated analyses
   - Memory optimized for large polygon sets
3. **AI Recommendations** with executable operations:
   - Auto-fix critical issues (zero-area, open contours)
   - Suggestions for optimization (simplification, grouping)
   - Readiness score calculation
4. **Review & Apply** fixes interactively
5. **Export** optimized DXF, CSV reports, G-code

### 2. Image to DXF Workflow

1. Upload image file (PNG, JPG, etc.)
2. Configure preprocessing (threshold, blur, invert)
3. Detect objects with advanced contour detection
4. Edit objects interactively
5. Preview with vector overlay
6. Export to DXF format

### 3. Advanced Nesting Workflow

1. Load multiple DXF files
2. Configure constraints:
   - Hard: kerf margin, min web, pierce zones
   - Soft: priorities, grain direction
3. Run optimization (BLF, NFP, genetic algorithms)
4. Review utilization and placement
5. Export nested layout

## ⚙️ Configuration

### Environment Variables

```bash
# Redis connection (for job queue)
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_URL=redis://localhost:6379/0

# API configuration
export WJP_USE_API=true  # Use FastAPI (default: auto-detect)
export WJP_API_URL=http://localhost:8000

# Performance tuning
export WJP_CACHE_DIR=.cache
export WJP_STREAMING_THRESHOLD=10485760  # 10MB in bytes
```

### Material Profiles (`config/material_profiles.py`)

```python
MATERIAL_PROFILES = {
    "steel": {
        "thickness": 6.0,
        "cutting_speed": 100.0,
        "pierce_time": 2.0,
        "cost_per_mm": 0.05,
        "kerf_width": 1.1
    },
    # ... more materials
}
```

## 🔧 API Usage

### FastAPI Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Analyze DXF (synchronous)
curl -X POST http://localhost:8000/analyze-dxf \
  -H "Content-Type: application/json" \
  -d '{"dxf_path": "file.dxf", "material": "steel"}'

# Analyze DXF (asynchronous)
curl -X POST "http://localhost:8000/analyze-dxf?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"dxf_path": "file.dxf"}'

# Check job status
curl http://localhost:8000/jobs/{job_id}

# API documentation
open http://localhost:8000/docs
```

### Python API Client

```python
from wjp_analyser.web.api_client_wrapper import analyze_dxf

# Automatically uses API if available, falls back to direct service
report = analyze_dxf("file.dxf", {"material": "steel", "thickness": 6.0})
print(f"Cutting length: {report['metrics']['length_internal_mm']} mm")
```

### Direct Service Usage

```python
from wjp_analyser.services.analysis_service import run_analysis
from wjp_analyser.services.costing_service import estimate_cost

# Analysis
report = run_analysis("file.dxf", out_dir="output")

# Costing (with caching)
cost = estimate_cost("file.dxf", {"rate_per_m": 825.0})
print(f"Total cost: ${cost['total_cost']:.2f}")
```

## 📊 Performance Features

### Large File Handling
- **Streaming Parser**: Handles files >10MB without OOM
- **Early Simplification**: Douglas-Peucker reduction during parsing
- **Entity Normalization**: Automatic SPLINE/ELLIPSE conversion

### Caching
- **Function-Level Memoization**: Automatic caching of expensive operations
- **Artifact Caching**: DXF, G-code, CSV paths cached with job hash
- **Cost Caching**: Repeated cost calculations return instantly

### Memory Optimization
- **Coordinate Precision**: Configurable decimal precision (default: 3)
- **float32 Support**: 50% memory reduction for large polygon sets
- **Segment Filtering**: Automatic removal of tiny segments (< epsilon)

### Async Processing
- **Background Jobs**: Long-running tasks processed asynchronously
- **Job Queue**: Redis Queue (RQ) with multiple queues per workload type
- **Idempotency**: Duplicate jobs return existing results

## 📋 Supported Formats

### Input
- **DXF**: R12-R2018 (AutoCAD DXF files)
- **Images**: PNG, JPG, JPEG, BMP, TIFF

### Output
- **DXF**: Optimized, layered DXF files
- **CSV**: Component analysis exports
- **JSON**: Analysis reports
- **NC/G-code**: Toolpath files (via separate workflow)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_dxf_analyzer.py

# With coverage
python -m pytest --cov=src/wjp_analyser tests/
```

## 🚀 Performance Benchmarks

### DXF Analysis
- **Small files (<1MB)**: <1 second (with cache: instant)
- **Medium files (1-10MB)**: 5-30 seconds (with streaming: <10 seconds)
- **Large files (>10MB)**: Streaming parser prevents OOM, 30-120 seconds
- **Memory usage**: 50-70% reduction with optimizations

### Nesting
- **Simple nesting (10 parts)**: <1 second
- **Complex nesting (100+ parts)**: 10-60 seconds
- **With metaheuristics**: 30-300 seconds (configurable)

## 🔄 Migration from Legacy Launchers

### Old Way (Deprecated)
```bash
python run_one_click.py --mode ui
python main.py analyze file.dxf
python wjp_analyser_unified.py web-ui
```

### New Way (Recommended)
```bash
# Web UI
python -m wjp_analyser.cli.wjp_cli web

# API
python -m wjp_analyser.cli.wjp_cli api

# Worker
python -m wjp_analyser.cli.wjp_cli worker

# Direct Python API (no CLI needed)
from wjp_analyser.services import run_analysis
```

## 📝 Recent Updates (Phases 1-5)

### Phase 1-2: Service Layer & API
- ✅ Centralized service layer (analysis, costing, editing)
- ✅ FastAPI backend with async job processing
- ✅ API client with automatic fallback
- ✅ Unified CLI (`wjp` command)

### Phase 3: UX Improvements
- ✅ Wizard components for guided workflows
- ✅ Jobs drawer for real-time status
- ✅ Actionable error handling
- ✅ Terminology standardization

### Phase 4: Performance Optimization
- ✅ Streaming DXF parser for large files
- ✅ Enhanced caching with job hashing
- ✅ Memory optimization (float32, segment filtering)
- ✅ Job idempotency

### Phase 5: Advanced Nesting
- ✅ Geometry hygiene (polygonization, holes, winding)
- ✅ Advanced placement engines (BLF, NFP, metaheuristics)
- ✅ Constraint system (hard/soft constraints)
- ✅ Determinism mode

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for image processing capabilities
- Streamlit team for the web framework
- FastAPI team for the modern API framework
- The waterjet cutting industry for inspiration and requirements

## 📞 Support

- **Documentation**: Check the `/docs` directory and phase progress files
- **Issues**: Report bugs and request features on GitHub Issues
- **API Docs**: Available at `http://localhost:8000/docs` when API server is running

---

**Made with ❤️ for the waterjet cutting industry**
