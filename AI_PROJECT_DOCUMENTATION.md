# WJP ANALYSER - Waterjet Cutting Analysis System

## Project Overview

The WJP ANALYSER is a comprehensive waterjet cutting analysis system that provides DXF file processing, AI-powered analysis, image-to-DXF conversion, and interactive editing capabilities. The system is designed to optimize waterjet cutting operations through advanced geometric analysis, cost estimation, and intelligent toolpath generation.

## System Architecture

### Core Components

1. **DXF Analysis Engine** (`src/wjp_analyser/analysis/`)
   - Geometric analysis of DXF files
   - Shape classification and grouping
   - Cutting length and pierce point calculation
   - Cost estimation based on material properties
   - Quality assessment and validation

2. **Image Processing Pipeline** (`src/wjp_analyser/image_processing/`)
   - Image-to-DXF conversion using multiple algorithms
   - Object detection and shape analysis
   - Interactive editing interface
   - Preview rendering system
   - Texture-aware vectorization

3. **Web Interface** (`src/wjp_analyser/web/`)
   - Streamlit-based web application
   - Multi-page interface for different workflows
   - Real-time preview and editing
   - File upload and download capabilities

4. **AI Integration** (`src/wjp_analyser/ai/`)
   - OpenAI API integration for intelligent analysis
   - Ollama local model support
   - Automated design generation
   - Intelligent material recommendations

5. **CLI Tools** (`src/wjp_analyser/cli/`)
   - Command-line interface for batch processing
   - Automated analysis workflows
   - Integration with external systems

## Key Features

### DXF Analysis
- **Geometric Analysis**: Comprehensive analysis of DXF entities including polylines, arcs, circles, and lines
- **Shape Grouping**: Intelligent grouping of similar shapes for efficient processing
- **Cutting Optimization**: Calculation of optimal cutting paths and pierce points
- **Cost Estimation**: Material-specific cost calculation based on cutting length and complexity
- **Quality Assessment**: Detection of potential issues like tiny segments, shaky polylines, and duplicates

### Image to DXF Conversion
- **Multiple Algorithms**: Support for Potrace, OpenCV, and custom texture-aware conversion
- **Object Detection**: Advanced contour detection with shape classification
- **Interactive Editing**: Live editing interface with object selection and modification
- **Preview System**: Comprehensive preview with vector overlay and multi-layer visualization
- **Batch Processing**: Support for processing multiple images

### AI-Powered Features
- **Intelligent Analysis**: AI-driven analysis of DXF files for optimization suggestions
- **Design Generation**: Automated generation of designs based on requirements
- **Material Recommendations**: AI-powered material selection based on design characteristics
- **Quality Prediction**: Prediction of cutting quality and potential issues

### Web Interface
- **Multi-Page Design**: Separate pages for different workflows
- **Real-Time Preview**: Live preview of analysis results and conversions
- **Interactive Editing**: Point-and-click editing of detected objects
- **File Management**: Upload, download, and management of DXF files
- **Responsive Design**: Works on desktop and mobile devices

## File Structure

```
WJP ANALYSER/
├── src/
│   ├── wjp_analyser/
│   │   ├── analysis/           # DXF analysis engine
│   │   ├── image_processing/   # Image-to-DXF conversion
│   │   ├── web/               # Web interface
│   │   ├── ai/                # AI integration
│   │   ├── cli/               # Command-line tools
│   │   └── utils/             # Utility functions
│   └── scripts/               # Standalone scripts
├── config/                    # Configuration files
├── data/                      # Sample data and templates
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── tools/                     # Utility tools
└── templates/                 # Web templates
```

## Configuration

### AI Configuration (`config/ai_config.yaml`)
- OpenAI API settings
- Ollama configuration
- Model parameters and prompts
- Analysis thresholds

### Material Profiles (`config/material_profiles.py`)
- Material properties database
- Cutting parameters
- Cost calculations
- Quality specifications

### API Keys (`config/api_keys.yaml`)
- OpenAI API key configuration
- External service credentials
- Security settings

## Usage Workflows

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

## Technical Specifications

### Dependencies
- **Core**: Python 3.10+, NumPy, OpenCV, PIL
- **DXF Processing**: ezdxf, shapely
- **Web Interface**: Streamlit, matplotlib
- **AI Integration**: openai, requests
- **Image Processing**: scikit-image, potrace

### Performance
- **DXF Analysis**: Handles files up to 10MB with complex geometries
- **Image Processing**: Supports images up to 4K resolution
- **Real-time Preview**: Updates in <100ms for typical files
- **Batch Processing**: Can process multiple files simultaneously

### Supported Formats
- **Input**: DXF (R12-R2018), PNG, JPG, BMP, TIFF
- **Output**: DXF, SVG, NC (G-code), JSON reports

## API Integration

### OpenAI Integration
- **Models**: GPT-4, GPT-3.5-turbo
- **Use Cases**: Design analysis, optimization suggestions, natural language processing
- **Rate Limiting**: Configurable rate limits and retry logic
- **Error Handling**: Robust error handling with fallback options

### Ollama Integration
- **Local Models**: Support for local LLM models
- **Privacy**: Complete data privacy for sensitive designs
- **Performance**: Optimized for local processing
- **Models**: Llama, CodeLlama, Mistral

## Development Guidelines

### Code Organization
- **Modular Design**: Each component is self-contained with clear interfaces
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Testing**: Unit tests for core functionality
- **Documentation**: Comprehensive docstrings and comments

### Performance Optimization
- **Caching**: Intelligent caching of analysis results
- **Parallel Processing**: Multi-threaded processing where appropriate
- **Memory Management**: Efficient memory usage for large files
- **Lazy Loading**: Components loaded only when needed

### Security
- **API Key Management**: Secure storage and handling of API keys
- **File Validation**: Strict validation of uploaded files
- **Error Sanitization**: Sanitized error messages to prevent information leakage
- **Access Control**: Configurable access controls for different features

## Deployment

### Local Development
```bash
pip install -r requirements.txt
streamlit run src/wjp_analyser/web/app.py
```

### Production Deployment
- Docker containerization support
- Environment variable configuration
- Logging and monitoring
- Health checks and metrics

## Future Enhancements

### Planned Features
- **3D Analysis**: Support for 3D DXF files
- **Advanced AI**: Integration with specialized CAD AI models
- **Cloud Processing**: Cloud-based processing for large files
- **Mobile App**: Native mobile application
- **API**: RESTful API for external integrations

### Performance Improvements
- **GPU Acceleration**: CUDA support for image processing
- **Distributed Processing**: Multi-node processing for large batches
- **Caching**: Advanced caching strategies
- **Optimization**: Algorithm optimizations for better performance

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **API Errors**: Check API key configuration and network connectivity
3. **Memory Issues**: Reduce file size or enable memory optimization
4. **Performance**: Check system resources and optimize settings

### Support
- **Documentation**: Comprehensive documentation in `/docs`
- **Examples**: Sample files and usage examples
- **Community**: GitHub issues and discussions
- **Professional Support**: Available for enterprise users

## License and Credits

This project is developed for waterjet cutting optimization and analysis. It integrates multiple open-source libraries and tools to provide a comprehensive solution for the waterjet cutting industry.

---

*This documentation is designed to help AI systems understand the project structure, capabilities, and usage patterns for effective assistance and development.*
