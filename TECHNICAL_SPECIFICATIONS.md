# Waterjet DXF Analyzer - Technical Specifications

## 🏗️ System Architecture

### Core Components
```
Waterjet DXF Analyzer
├── Web Interface (Flask)
├── Analysis Engine
├── AI Integration Layer
├── Manufacturing Module
├── Image Processing Pipeline
└── Export System
```

### Module Dependencies
```
src/wjp_analyser/
├── analysis/
│   ├── dxf_analyzer.py      # Main DXF analysis engine
│   ├── geometry_cleaner.py  # Geometric operations
│   ├── topology.py          # Containment analysis
│   ├── classification.py    # Part classification
│   └── quality_checks.py   # Validation rules
├── ai/
│   ├── ollama_client.py     # Local AI integration
│   └── openai_client.py     # Cloud AI integration
├── manufacturing/
│   ├── nesting.py           # Part nesting algorithms
│   ├── toolpath.py          # Cutting path optimization
│   ├── gcode_generator.py   # G-code generation
│   ├── cost_calculator.py   # Pricing calculations
│   └── path_optimizer.py    # Advanced path optimization
├── image_processing/
│   └── pipeline.py          # Image-to-DXF conversion
├── io/
│   ├── dxf_io.py           # DXF file operations
│   ├── visualization.py    # Preview generation
│   └── quote_export.py     # PDF/Excel export
└── web/
    ├── app.py              # Flask application
    └── templates/          # HTML templates
```

## 🔧 Technical Requirements

### System Requirements
- **Python**: 3.13+
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.15+, Linux Ubuntu 20.04+

### Python Dependencies
```python
# Core DXF Processing
ezdxf>=1.3.0              # DXF file operations
shapely>=2.0.2            # Geometric operations
numpy>=1.26.4             # Numerical computing

# Web Framework
flask>=3.0.3              # Web application
werkzeug>=3.0.0           # WSGI utilities

# Image Processing
opencv-python>=4.10.0     # Computer vision
pillow>=10.4.0            # Image manipulation
scikit-image>=0.24.0      # Advanced image processing

# AI Integration
openai>=1.0.0             # OpenAI API client
pyyaml>=6.0               # Configuration files
requests>=2.25.0          # HTTP requests

# CLI and Utilities
rich>=13.7.1              # Terminal formatting
pydantic>=2.7.0           # Data validation
click>=8.1.0              # Command line interface

# Development
pytest>=6.0.0             # Testing framework
pytest-cov>=3.0.0         # Coverage reporting
```

## 📊 Data Models

### ManufacturingAnalysis (AI Output)
```python
class ManufacturingAnalysis(BaseModel):
    feasibility_score: float           # 0-100 manufacturing score
    complexity_level: str              # Simple/Moderate/Complex
    estimated_time: str                # Cutting time estimate
    material_recommendations: List[str] # Suggested materials
    toolpath_suggestions: List[str]    # Cutting strategies
    potential_issues: List[str]        # Manufacturing concerns
    optimization_tips: List[str]       # Efficiency improvements
    cost_considerations: List[str]     # Budget insights
    model_used: Optional[str] = None   # AI model identifier
```

### DXF Analysis Results
```python
class AnalysisReport:
    metrics: {
        length_outer_mm: float        # External cutting length
        length_internal_mm: float      # Internal cutting length
        pierces: int                   # Number of pierce points
        cost_inr: float               # Cost in Indian Rupees
        est_time_min: float           # Estimated time in minutes
    }
    entities: {
        polygons: int                 # Number of polygons
        lines: int                    # Number of lines
        outer: int                    # External contours
        inner: int                    # Internal contours
    }
    violations: List[Dict]            # Manufacturing violations
    warnings: List[Dict]              # Manufacturing warnings
```

### Nesting Results
```python
class NestingReport:
    sheet_dimensions: {
        width: float                  # Sheet width in mm
        height: float                 # Sheet height in mm
    }
    utilization_percent: float        # Material utilization %
    total_items: int                  # Number of nested parts
    spacing: float                    # Part spacing in mm
    items: List[{
        file: str                     # DXF filename
        position: {x: float, y: float} # Part position
        dimensions: {width: float, height: float} # Part size
        row: int                      # Nesting row number
    }]
```

## 🔄 Processing Workflows

### 1. DXF Analysis Pipeline
```
DXF File Input
    ↓
Load DXF Lines & Layers
    ↓
Merge & Polygonize Geometry
    ↓
Containment Depth Analysis
    ↓
Part Classification
    ↓
Quality Validation
    ↓
Cost Calculation
    ↓
Generate Report & Visualizations
```

### 2. AI Analysis Pipeline
```
DXF Analysis Data
    ↓
AI Model Selection (Ollama/OpenAI)
    ↓
Manufacturing Prompt Generation
    ↓
AI Processing
    ↓
Response Parsing & Validation
    ↓
ManufacturingAnalysis Object
    ↓
Results Display
```

### 3. Nesting Pipeline
```
Multiple DXF Files
    ↓
Part Dimension Analysis
    ↓
Sheet Parameter Configuration
    ↓
Nesting Algorithm (Simple Row-based)
    ↓
Layout Optimization
    ↓
Nested DXF Generation
    ↓
Utilization Report
```

### 4. Image Conversion Pipeline
```
Image Input
    ↓
Preprocessing (Blur, Edge Detection)
    ↓
Contour Extraction
    ↓
Geometric Simplification
    ↓
DXF Generation
    ↓
Quality Validation
    ↓
Output DXF File
```

## 🎛️ Configuration Parameters

### DXF Analysis Parameters
```yaml
material: "steel"                    # Material type
thickness_mm: 25.0                   # Material thickness
kerf_mm: 0.2                        # Kerf width
rate_per_m: 50.0                     # Cost per meter
cutting_speed: 1200.0               # Cutting speed mm/min
rapid_speed: 6000.0                 # Rapid speed mm/min
pierce_time: 0.5                    # Pierce time seconds
```

### AI Configuration
```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "waterjet:latest"
  timeout: 120

openai:
  api_key: "your-api-key-here"
  model: "gpt-4"
  max_tokens: 2000
  temperature: 0.7
```

### Nesting Parameters
```yaml
sheet_width: 3000.0                 # Sheet width mm
sheet_height: 1500.0                # Sheet height mm
spacing: 10.0                       # Part spacing mm
```

### Image Processing Parameters
```yaml
edge_threshold: 100                 # Edge detection threshold
canny_low: 50                       # Canny low threshold
canny_high: 150                     # Canny high threshold
min_contour_area: 100              # Minimum contour area
simplify_tolerance: 0.5             # Simplification tolerance
blur_kernel_size: 5                 # Blur kernel size
```

## 🔒 Security & Validation

### Input Validation
- **File Type Checking**: DXF, PNG, JPG, JPEG validation
- **File Size Limits**: Maximum 50MB per file
- **Path Sanitization**: Prevent directory traversal attacks
- **Parameter Validation**: Pydantic model validation

### Error Handling
- **Graceful Degradation**: Fallback mechanisms for AI failures
- **User-Friendly Messages**: Clear error descriptions
- **Logging**: Comprehensive error logging
- **Recovery**: Automatic retry mechanisms

### Data Protection
- **Temporary Files**: Automatic cleanup
- **API Key Security**: Environment variable storage
- **Session Management**: Secure session handling
- **File Access**: Restricted file system access

## 📈 Performance Metrics

### Processing Times
- **DXF Analysis**: 1-5 seconds per file
- **AI Analysis**: 2-10 seconds (depending on model)
- **Nesting**: 5-15 seconds for multiple parts
- **Image Conversion**: 3-8 seconds per image

### Resource Usage
- **Memory**: 100-500MB typical usage
- **CPU**: Single-threaded processing
- **Storage**: Temporary files cleaned automatically
- **Network**: Minimal (AI API calls only)

### Scalability
- **Concurrent Users**: 10-50 users (Flask development server)
- **File Processing**: Sequential processing
- **AI Requests**: Rate-limited by API providers
- **Storage**: Local file system only

## 🧪 Testing Strategy

### Unit Tests
- **Module Testing**: Individual component testing
- **Data Validation**: Pydantic model testing
- **Error Handling**: Exception scenario testing
- **Edge Cases**: Boundary condition testing

### Integration Tests
- **Workflow Testing**: End-to-end process testing
- **API Integration**: AI service testing
- **File Processing**: DXF/image processing testing
- **Web Interface**: UI functionality testing

### Performance Tests
- **Load Testing**: Multiple concurrent users
- **Memory Testing**: Large file processing
- **Timeout Testing**: AI service reliability
- **Resource Testing**: System resource usage

## 🔧 Deployment Options

### Development Environment
```bash
# Local development setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python run_web_ui.py
```

### Production Environment
```bash
# Production deployment
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.wjp_analyser.web.app:app
```

### Docker Deployment
```dockerfile
FROM python:3.13-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.wjp_analyser.web.app:app"]
```

## 📋 Maintenance & Updates

### Regular Maintenance
- **Dependency Updates**: Monthly security updates
- **AI Model Updates**: Quarterly model refreshes
- **Performance Monitoring**: Weekly performance checks
- **Error Log Review**: Daily error log analysis

### Version Control
- **Git Repository**: Full version control
- **Release Tags**: Semantic versioning
- **Change Log**: Detailed change documentation
- **Rollback Plan**: Quick rollback procedures

### Backup Strategy
- **Configuration Backup**: Daily config file backup
- **Code Backup**: Git repository backup
- **Data Backup**: User file backup (if implemented)
- **System Backup**: Full system backup monthly

---

**Document Version**: 1.0.0  
**Last Updated**: September 28, 2025  
**Maintained By**: Development Team
