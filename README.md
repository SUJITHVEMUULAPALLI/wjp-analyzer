# WJP ANALYSER - Waterjet Cutting Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive waterjet cutting analysis system that provides DXF file processing, AI-powered analysis, image-to-DXF conversion, and interactive editing capabilities.

## 🚀 Features

### 📊 DXF Analysis Engine
- **Geometric Analysis**: Comprehensive analysis of DXF entities (polylines, arcs, circles, lines)
- **Shape Grouping**: Intelligent grouping of similar shapes for efficient processing
- **Cutting Optimization**: Calculation of optimal cutting paths and pierce points
- **Cost Estimation**: Material-specific cost calculation based on cutting length and complexity
- **Quality Assessment**: Detection of potential issues like tiny segments, shaky polylines, and duplicates

### 🖼️ Image to DXF Conversion
- **Multiple Algorithms**: Support for Potrace, OpenCV, and custom texture-aware conversion
- **Object Detection**: Advanced contour detection with shape classification
- **Interactive Editing**: Live editing interface with object selection and modification
- **Preview System**: Comprehensive preview with vector overlay and multi-layer visualization
- **Batch Processing**: Support for processing multiple images

### 🤖 AI-Powered Features
- **Intelligent Analysis**: AI-driven analysis of DXF files for optimization suggestions
- **Design Generation**: Automated generation of designs based on requirements
- **Material Recommendations**: AI-powered material selection based on design characteristics
- **Quality Prediction**: Prediction of cutting quality and potential issues

### 🌐 Web Interface
- **Multi-Page Design**: Separate pages for different workflows
- **Real-Time Preview**: Live preview of analysis results and conversions
- **Interactive Editing**: Point-and-click editing of detected objects
- **File Management**: Upload, download, and management of DXF files
- **Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
WJP ANALYSER/
├── src/
│   ├── wjp_analyser/
│   │   ├── analysis/           # DXF analysis engine
│   │   │   ├── dxf_analyzer.py
│   │   │   ├── grouping.py
│   │   │   ├── cost_estimator.py
│   │   │   └── quality_checker.py
│   │   ├── image_processing/   # Image-to-DXF conversion
│   │   │   ├── object_detector.py
│   │   │   ├── interactive_editor.py
│   │   │   ├── preview_renderer.py
│   │   │   ├── potrace_pipeline.py
│   │   │   └── texture_pipeline.py
│   │   ├── web/               # Web interface
│   │   │   ├── app.py
│   │   │   ├── _components.py
│   │   │   └── pages/
│   │   ├── ai/                # AI integration
│   │   │   ├── openai_client.py
│   │   │   └── ollama_client.py
│   │   ├── cli/               # Command-line tools
│   │   │   ├── analyze_cli.py
│   │   │   └── batch_analyze.py
│   │   └── utils/             # Utility functions
│   └── scripts/               # Standalone scripts
├── config/                    # Configuration files
│   ├── ai_config.yaml
│   ├── material_profiles.py
│   └── api_keys.yaml
├── data/                      # Sample data and templates
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── tools/                     # Utility tools
└── templates/                 # Web templates
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Dependencies
- **Core**: numpy, opencv-python, pillow, matplotlib
- **DXF Processing**: ezdxf, shapely
- **Web Interface**: streamlit
- **AI Integration**: openai, requests
- **Image Processing**: scikit-image

### Optional Dependencies
- **Potrace**: For advanced vectorization (install separately)
  - **Windows**: Download from http://potrace.sourceforge.net/ or install via Chocolatey: `choco install potrace`
  - **Linux**: `sudo apt-get install potrace` (Ubuntu/Debian) or `sudo yum install potrace` (RHEL/CentOS)
  - **macOS**: `brew install potrace`
- **Ollama**: For local AI models (install separately)

## 🚀 Quick Start

### Web Interface
```bash
streamlit run src/wjp_analyser/web/app.py
```

### Command Line Analysis
```bash
python -m src.wjp_analyser.cli.analyze_cli --input sample.dxf --output results/
```

### Batch Processing
```bash
python -m src.wjp_analyser.cli.batch_analyze --input-dir dxf_files/ --output-dir results/
```

## 📖 Usage

### 1. DXF Analysis Workflow
1. Upload DXF file through web interface
2. System automatically analyzes geometry and groups similar shapes
3. Review analysis results including cutting length, pierce points, and cost
4. Apply optimizations (softening, filleting, scaling)
5. Generate optimized DXF and toolpath files
6. Download results

### 2. Image to DXF Workflow
1. Upload image file (PNG, JPG, etc.)
2. Crop and adjust image boundaries
3. Configure preprocessing parameters (threshold, blur, etc.)
4. Detect objects using advanced contour detection
5. Edit objects interactively (select, modify, organize)
6. Preview final DXF with vector overlay
7. Export to DXF format

### 3. AI Design Generation Workflow
1. Describe design requirements in natural language
2. AI generates design suggestions
3. Review and refine generated designs
4. Convert to DXF format
5. Analyze and optimize for cutting

## ⚙️ Configuration

### AI Configuration (`config/ai_config.yaml`)
```yaml
openai:
  api_key: "your-openai-api-key"
  model: "gpt-4"
  max_tokens: 4000
  temperature: 0.7

ollama:
  base_url: "http://localhost:11434"
  model: "llama2"
  timeout: 30
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
    "aluminum": {
        "thickness": 3.0,
        "cutting_speed": 150.0,
        "pierce_time": 1.5,
        "cost_per_mm": 0.03,
        "kerf_width": 0.8
    }
}
```

## 🔧 API Usage

### Basic DXF Analysis
```python
from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf

args = AnalyzeArgs(out="output/")
args.sheet_width = 1000.0
args.sheet_height = 1000.0

report = analyze_dxf("input.dxf", args)
print(f"Cutting length: {report['metrics']['length_internal_mm']} mm")
```

### Image to DXF Conversion
```python
from wjp_analyser.image_processing.object_detector import ObjectDetector, DetectionParams
from wjp_analyser.image_processing.interactive_editor import InteractiveEditor

editor = InteractiveEditor()
editor.load_image("input.png")

params = DetectionParams(min_area=100)
objects = editor.detect_objects(params)

editor.export_all_objects("output.dxf")
```

### AI Analysis
```python
from wjp_analyser.ai.openai_client import OpenAIClient

client = OpenAIClient(api_key="your-key")
analysis = client.analyze_dxf("input.dxf", analysis_report)
print(analysis['recommendations'])
```

## 📊 Supported Formats

### Input Formats
- **DXF**: R12-R2018 (AutoCAD DXF files)
- **Images**: PNG, JPG, JPEG, BMP, TIFF
- **Text**: Natural language descriptions for AI generation

### Output Formats
- **DXF**: Optimized DXF files
- **SVG**: Vector graphics
- **NC**: G-code toolpaths
- **JSON**: Analysis reports
- **CSV**: Data exports

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/
```

### Run Specific Test
```bash
python -m pytest tests/test_grouping.py
```

### Test Coverage
```bash
python -m pytest --cov=src/wjp_analyser tests/
```

## 🚀 Performance

### Benchmarks
- **DXF Analysis**: Handles files up to 10MB with complex geometries
- **Image Processing**: Supports images up to 4K resolution
- **Real-time Preview**: Updates in <100ms for typical files
- **Batch Processing**: Can process multiple files simultaneously

### Optimization Tips
- Use caching for repeated analysis
- Process files in batches for CLI tools
- Enable parallel processing for multiple files
- Use appropriate image resolution for conversion

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
- OpenAI for AI integration capabilities
- The waterjet cutting industry for inspiration and requirements

## 📞 Support

- **Documentation**: Check the `/docs` directory for detailed documentation
- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Professional Support**: Available for enterprise users

## 🔮 Roadmap

### Upcoming Features
- **3D Analysis**: Support for 3D DXF files
- **Advanced AI**: Integration with specialized CAD AI models
- **Cloud Processing**: Cloud-based processing for large files
- **Mobile App**: Native mobile application
- **REST API**: RESTful API for external integrations

### Performance Improvements
- **GPU Acceleration**: CUDA support for image processing
- **Distributed Processing**: Multi-node processing for large batches
- **Advanced Caching**: Intelligent caching strategies
- **Algorithm Optimization**: Better performance algorithms

---

**Made with ❤️ for the waterjet cutting industry**