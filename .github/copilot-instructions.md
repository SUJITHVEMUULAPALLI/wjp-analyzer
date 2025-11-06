# WJP ANALYSER AI Agent Instructions

## Project Overview
WJP ANALYSER is a waterjet cutting analysis system that processes DXF files, performs AI-powered analysis, converts images to DXF, and provides interactive editing capabilities. The system is built in Python with a Streamlit web interface.

## Key Architectural Components

### 1. Core Analysis Engine (`src/wjp_analyser/analysis/`)
- `dxf_analyzer.py`: Entry point for DXF analysis
- `geometry_cleaner.py`: Handles geometric operations
- `topology.py`: Analyzes shape containment
- Uses Shapely for geometry and ezdxf for DXF operations

### 2. AI Integration (`src/wjp_analyser/ai/`)
- Supports both local (Ollama) and cloud (OpenAI) models
- Configuration in `config/ai_config.yaml`
- AI responses must follow `ManufacturingAnalysis` model structure

### 3. Image Processing (`src/wjp_analyser/image_processing/`)
- Uses OpenCV for contour detection and processing
- Configuration parameters in `image_processing/pipeline.py`
- Supports multiple conversion algorithms (Potrace, OpenCV)

### 4. Web Interface (`src/wjp_analyser/web/`)
- Built with Streamlit (legacy Flask/FastAPI removed)
- Main entry: `python wjp_analyser_unified.py web-ui`
- Page components in `web/components/`

## Development Workflows

### Running the Application
```bash
# Development mode
python wjp_analyser_unified.py web-ui --interface streamlit --host 127.0.0.1 --port 8501

# CLI Analysis
python -m src.wjp_analyser.cli.analyze_cli --input sample.dxf --output results/
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# With coverage
python -m pytest --cov=src/wjp_analyser tests/
```

## Project-Specific Conventions

### Code Organization
1. Core logic in `src/wjp_analyser/`
2. CLI tools in `src/wjp_analyser/cli/`
3. Web components in `src/wjp_analyser/web/`
4. Configuration in `config/`

### Error Handling
- Use custom exceptions from `src/wjp_analyser/utils/exceptions.py`
- Always validate DXF files before processing
- Log errors using the configured logger

### Configuration Management
- AI settings in `config/ai_config.yaml`
- Material profiles in `config/material_profiles.py`
- Use environment variables for sensitive data

## Integration Points

### AI Integration
```python
from wjp_analyser.ai.openai_client import OpenAIClient

client = OpenAIClient()
analysis = client.analyze_dxf("input.dxf", analysis_report)
```

### Image Processing
```python
from wjp_analyser.image_processing.object_detector import ObjectDetector

detector = ObjectDetector()
objects = detector.detect_objects(image_path, params)
```

### DXF Analysis
```python
from wjp_analyser.analysis.dxf_analyzer import analyze_dxf

report = analyze_dxf("input.dxf", AnalyzeArgs(out="output/"))
```

## Cross-Component Communication

### Data Flow
1. Web UI → Analysis Engine
2. Analysis Engine → AI Integration
3. AI Integration → Report Generation
4. Report Generation → Web UI

### Event System
- Use the event system in `src/wjp_analyser/utils/events.py`
- Subscribe to events using `@event.subscribe('event_name')`
- Emit events using `event.emit('event_name', data)`

## Performance Considerations
- Cache analysis results for repeated operations
- Use batch processing for multiple files
- Clean up temporary files after processing
- Monitor memory usage with large DXF files

## Resource Paths
- Input files: `data/uploads/`
- Output files: `data/output/`
- Logs: `logs/`
- Templates: `templates/`

## Common Pitfalls
- Always close DXF files after reading
- Handle large files in chunks
- Validate AI responses before processing
- Check image dimensions before conversion

## Example Implementations
See `examples/basic_conversion_example.py` for common usage patterns.