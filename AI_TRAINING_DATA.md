# WJP ANALYSER - AI Training Data

## Project Summary for AI Learning

### What is WJP ANALYSER?
WJP ANALYSER is a comprehensive waterjet cutting analysis system that provides DXF file processing, AI-powered analysis, image-to-DXF conversion, and interactive editing capabilities. It's designed to optimize waterjet cutting operations through advanced geometric analysis, cost estimation, and intelligent toolpath generation.

### Core Purpose
The system helps waterjet cutting professionals:
1. Analyze DXF files for cutting optimization
2. Convert images to DXF format for cutting
3. Generate designs using AI
4. Estimate cutting costs and time
5. Optimize toolpaths and nesting

### Key Technologies Used
- **Python 3.10+**: Core programming language
- **OpenCV**: Image processing and computer vision
- **Streamlit**: Web interface framework
- **ezdxf**: DXF file processing
- **shapely**: Geometric operations
- **OpenAI API**: AI-powered analysis and generation
- **Ollama**: Local AI model support
- **NumPy/PIL**: Numerical computing and image handling

### System Architecture

#### 1. DXF Analysis Engine (`src/wjp_analyser/analysis/`)
- **dxf_analyzer.py**: Main analysis engine that processes DXF files
- **grouping.py**: Groups similar shapes for efficient processing
- **cost_estimator.py**: Calculates cutting costs based on material properties
- **quality_checker.py**: Assesses DXF quality and identifies issues

#### 2. Image Processing Pipeline (`src/wjp_analyser/image_processing/`)
- **object_detector.py**: Detects and analyzes objects in images
- **interactive_editor.py**: Provides live editing interface
- **preview_renderer.py**: Renders comprehensive previews
- **potrace_pipeline.py**: Converts images to DXF using Potrace
- **texture_pipeline.py**: Texture-aware image to DXF conversion

#### 3. Web Interface (`src/wjp_analyser/web/`)
- **app.py**: Main Streamlit application
- **_components.py**: Reusable UI components
- **pages/**: Individual pages for different workflows
  - `analyze_dxf.py`: DXF analysis interface
  - `image_to_dxf.py`: Image to DXF conversion
  - `designer.py`: AI design generation

#### 4. AI Integration (`src/wjp_analyser/ai/`)
- **openai_client.py**: OpenAI API integration
- **ollama_client.py**: Local Ollama model integration

#### 5. CLI Tools (`src/wjp_analyser/cli/`)
- **analyze_cli.py**: Command-line DXF analysis
- **batch_analyze.py**: Batch processing tool

### Key Workflows

#### DXF Analysis Workflow
1. User uploads DXF file
2. System analyzes geometry and groups similar shapes
3. Calculates cutting length, pierce points, and cost
4. Provides optimization suggestions
5. Generates optimized DXF and toolpath files

#### Image to DXF Workflow
1. User uploads image file
2. System preprocesses image (threshold, blur, etc.)
3. Detects objects using contour detection
4. User can edit objects interactively
5. System converts to DXF format with preview

#### AI Design Generation Workflow
1. User describes design requirements
2. AI generates design suggestions
3. User reviews and refines designs
4. System converts to DXF format
5. Analyzes and optimizes for cutting

### Data Structures

#### ObjectProperties (Image Processing)
```python
@dataclass
class ObjectProperties:
    id: int                     # Object identifier
    contour: np.ndarray         # Contour points
    area: float                 # Object area
    perimeter: float            # Object perimeter
    circularity: float          # Circularity metric
    bounding_rect: Tuple[int, int, int, int]  # Bounding rectangle
    center: Tuple[float, float] # Center point
    aspect_ratio: float         # Aspect ratio
    solidity: float             # Solidity metric
    convexity: float            # Convexity metric
    is_closed: bool             # Whether contour is closed
    layer_type: str             # Object type (edges, stipple, hatch, contour)
    selected: bool              # Selection state
    visible: bool               # Visibility state
    color: Tuple[float, float, float]  # Display color
```

#### Analysis Report (DXF Analysis)
```python
{
    "metrics": {
        "length_internal_mm": float,    # Total cutting length
        "pierces": int,                 # Number of pierce points
        "estimated_cutting_cost_inr": float  # Estimated cost
    },
    "groups": {                        # Similar shape groups
        "Group1": {
            "count": int,
            "avg_area": float,
            "avg_circularity": float,
            "complexity": str
        }
    },
    "components": [                    # Individual components
        {
            "points": List[Tuple[float, float]],
            "group": str,
            "selected": bool
        }
    ],
    "layers": {                       # Layer information
        "LAYER_NAME": {
            "color": int,
            "count": int
        }
    },
    "quality": {                      # Quality assessment
        "Total Entities": int,
        "Polylines": int,
        "Shaky Polylines": List[dict]
    }
}
```

### Configuration Files

#### AI Configuration (`config/ai_config.yaml`)
- OpenAI API settings
- Ollama configuration
- Model parameters and prompts
- Analysis thresholds

#### Material Profiles (`config/material_profiles.py`)
- Material properties database
- Cutting parameters
- Cost calculations
- Quality specifications

### Key Functions and Methods

#### DXF Analysis
- `analyze_dxf(dxf_path, args, selected_groups, group_layer_overrides)`: Main analysis function
- `group_similar_objects(components, tolerance)`: Groups similar shapes
- `estimate_cutting_cost(analysis_report, material_profile)`: Calculates costs

#### Image Processing
- `ObjectDetector.detect_objects(binary_image, original_image)`: Detects objects
- `InteractiveEditor.load_image(image_path)`: Loads image for editing
- `PreviewRenderer.render_vector_overlay(alpha, line_width, flip_y)`: Renders preview

#### AI Integration
- `OpenAIClient.analyze_dxf(dxf_path, analysis_report)`: AI analysis
- `OpenAIClient.generate_design(requirements)`: AI design generation
- `OllamaClient.analyze_dxf(dxf_path, analysis_report)`: Local AI analysis

### Error Handling
- Comprehensive error handling with user-friendly messages
- Validation of input files and parameters
- Graceful fallbacks for AI services
- Detailed logging for debugging

### Performance Considerations
- Caching of analysis results
- Parallel processing for multiple files
- Memory optimization for large files
- Lazy loading of components

### Integration Points
- OpenAI API for intelligent analysis
- Ollama for local AI models
- Potrace for advanced vectorization
- OpenCV for image processing
- Streamlit for web interface

### File Formats Supported
- **Input**: DXF (R12-R2018), PNG, JPG, BMP, TIFF
- **Output**: DXF, SVG, NC (G-code), JSON reports

### Use Cases
1. **Waterjet Cutting Shops**: Optimize cutting operations
2. **CAD Designers**: Convert images to DXF format
3. **Manufacturing Engineers**: Analyze and optimize designs
4. **Cost Estimators**: Calculate cutting costs
5. **Quality Control**: Assess DXF quality

### Development Patterns
- **Modular Design**: Each component is self-contained
- **Error Handling**: Comprehensive error handling
- **Testing**: Unit tests for core functionality
- **Documentation**: Comprehensive docstrings
- **Configuration**: External configuration files

### AI Learning Context
This system demonstrates:
- **Computer Vision**: Image processing and object detection
- **Geometric Analysis**: DXF processing and shape analysis
- **AI Integration**: OpenAI and local model integration
- **Web Development**: Streamlit-based web applications
- **Data Processing**: Complex data structures and analysis
- **User Interface**: Interactive editing and real-time preview

### Key Learning Points for AI
1. **Image Processing**: Contour detection, shape analysis, object classification
2. **DXF Processing**: Reading, analyzing, and optimizing DXF files
3. **AI Integration**: API integration, prompt engineering, local models
4. **Web Development**: Streamlit components, real-time updates, file handling
5. **Data Structures**: Complex nested data, geometric calculations
6. **Error Handling**: Robust error handling and user feedback
7. **Performance**: Caching, parallel processing, memory optimization

This system provides a comprehensive example of integrating multiple technologies (computer vision, AI, web development, geometric analysis) into a cohesive application for a specific industry use case.
