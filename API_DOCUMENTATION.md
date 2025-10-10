# WJP ANALYSER API Documentation

## Overview

This document provides comprehensive API documentation for the WJP ANALYSER system, designed for AI systems to understand and interact with the codebase effectively.

## Core Modules and Classes

### 1. DXF Analysis Engine

#### `src/wjp_analyser/analysis/dxf_analyzer.py`

**Main Class**: `AnalyzeArgs`
```python
class AnalyzeArgs:
    out: str                    # Output directory
    sheet_width: float          # Sheet width in mm
    sheet_height: float         # Sheet height in mm
    frame_quantity: int         # Number of frames
    soften_method: str          # Softening method (none, simplify, chaikin)
    soften_tolerance: float     # Softening tolerance
    soften_iterations: int     # Softening iterations
    fillet_radius_mm: float    # Fillet radius
    fillet_min_angle_deg: float # Minimum angle for filleting
    scale_mode: str            # Scaling mode (auto, factor, decade_fit)
    scale_factor: float        # Scale factor
    normalize_mode: str        # Normalization mode
    target_frame_w_mm: float   # Target frame width
    target_frame_h_mm: float  # Target frame height
    frame_margin_mm: float     # Frame margin
    normalize_origin: bool     # Normalize to origin
    require_fit_within_frame: bool # Require fit within frame
```

**Main Function**: `analyze_dxf(dxf_path: str, args: AnalyzeArgs, selected_groups: List[str] = None, group_layer_overrides: Dict[str, str] = None) -> dict`

**Returns**: Analysis report dictionary with:
- `metrics`: Cutting length, pierce count, estimated cost
- `groups`: Similar shape groups with metadata
- `components`: Individual geometric components
- `layers`: Layer information
- `toolpath`: Toolpath data
- `nesting`: Nesting optimization results
- `quality`: Quality assessment results

#### `src/wjp_analyser/analysis/grouping.py`

**Main Function**: `group_similar_objects(components: List[dict], tolerance: float = 0.1) -> Dict[str, dict]`

**Parameters**:
- `components`: List of geometric components
- `tolerance`: Similarity tolerance for grouping

**Returns**: Dictionary of grouped objects with metadata

#### `src/wjp_analyser/analysis/cost_estimator.py`

**Main Function**: `estimate_cutting_cost(analysis_report: dict, material_profile: dict) -> dict`

**Parameters**:
- `analysis_report`: DXF analysis results
- `material_profile`: Material properties and costs

**Returns**: Cost breakdown including material, cutting, and total costs

### 2. Image Processing Pipeline

#### `src/wjp_analyser/image_processing/object_detector.py`

**Main Class**: `ObjectDetector`
```python
class ObjectDetector:
    def __init__(self, params: DetectionParams = None)
    def detect_objects(self, binary_image: np.ndarray, original_image: np.ndarray = None) -> List[ObjectProperties]
    def get_object_by_id(self, obj_id: int) -> Optional[ObjectProperties]
    def select_object(self, obj_id: int, selected: bool = True)
    def select_objects_by_type(self, layer_type: str, selected: bool = True)
    def generate_preview(self, show_selection: bool = True, show_bounding_boxes: bool = False, overlay_on_original: bool = True) -> np.ndarray
    def export_to_dxf(self, output_path: str, selected_only: bool = False) -> bool
    def get_statistics(self) -> Dict[str, Union[int, float]]
```

**Data Class**: `ObjectProperties`
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

**Data Class**: `DetectionParams`
```python
@dataclass
class DetectionParams:
    min_area: int = 100         # Minimum object area
    max_area: int = 1000000    # Maximum object area
    min_perimeter: int = 20     # Minimum perimeter
    min_circularity: float = 0.1 # Minimum circularity
    max_circularity: float = 1.0 # Maximum circularity
    min_solidity: float = 0.3   # Minimum solidity
    min_convexity: float = 0.5  # Minimum convexity
    merge_distance: float = 10.0 # Distance for merging objects
    simplify_tolerance: float = 2.0 # Contour simplification tolerance
```

#### `src/wjp_analyser/image_processing/interactive_editor.py`

**Main Class**: `InteractiveEditor`
```python
class InteractiveEditor:
    def __init__(self)
    def load_image(self, image_path: str) -> bool
    def set_binary_image(self, binary_image: np.ndarray)
    def detect_objects(self, params: DetectionParams = None) -> List[ObjectProperties]
    def generate_preview(self, show_selection: bool = True, show_bounding_boxes: bool = False, overlay_on_original: bool = True) -> np.ndarray
    def undo(self) -> bool
    def redo(self) -> bool
    def export_selected_objects(self, output_path: str) -> bool
    def export_all_objects(self, output_path: str) -> bool
```

**Main Function**: `render_interactive_editor(editor: InteractiveEditor) -> Dict[str, Any]`

#### `src/wjp_analyser/image_processing/preview_renderer.py`

**Main Class**: `PreviewRenderer`
```python
class PreviewRenderer:
    def __init__(self)
    def set_images(self, original: np.ndarray, binary: np.ndarray)
    def set_dxf_path(self, dxf_path: str)
    def set_objects(self, objects: List[ObjectProperties])
    def render_vector_overlay(self, alpha: float = 0.8, line_width: float = 1.2, flip_y: bool = False, show_layer_colors: bool = True, layer_opacity: Dict[str, float] = None) -> np.ndarray
    def render_multi_layer_preview(self, show_layers: List[str] = None, layer_opacity: Dict[str, float] = None) -> Dict[str, np.ndarray]
    def render_comparison_view(self, show_original: bool = True, show_binary: bool = True, show_vectors: bool = True, show_objects: bool = True) -> np.ndarray
    def generate_export_preview(self, size: Tuple[int, int] = (800, 800), include_legend: bool = True, include_stats: bool = True) -> np.ndarray
```

**Main Function**: `render_final_preview_interface(renderer: PreviewRenderer) -> Dict[str, Any]`

#### `src/wjp_analyser/image_processing/potrace_pipeline.py`

**Main Function**: `preprocess_and_vectorize(image_path: str, out_dir: Path, target_size_mm: float = 1000.0, threshold_type: str = "global", threshold_value: int = 180, adaptive_block_size: int = 21, adaptive_C: int = 5, gaussian_blur_ksize: int = 5, use_canny: bool = False, morph_op: str = "open", morph_ksize: int = 3, morph_iters: int = 1, output_route: str = "potrace_dxf", simplify_tolerance: float = 0.0, invert: bool = False, potrace_turdsize: int = 2, potrace_alphamax: float = 1.0, potrace_opttolerance: float = 0.2) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]`

**Returns**: Tuple of (dxf_path, preview_path, svg_path)

#### `src/wjp_analyser/image_processing/texture_pipeline.py`

**Data Classes**:
```python
@dataclass
class PreprocessParams:
    working_px: int = 1000      # Working pixel size
    adaptive_block: int = 35    # Adaptive threshold block size
    adaptive_C: int = 2         # Adaptive threshold C value
    morph_kernel: int = 3       # Morphology kernel size
    morph_iters: int = 1        # Morphology iterations

@dataclass
class TextureClassifyParams:
    tile: int = 32              # Tile size for classification
    clusters: int = 4           # Number of clusters

@dataclass
class TextureVectorizeParams:
    mode: str = "auto"          # Vectorization mode
    dxf_size_mm: float = 1000.0 # DXF canvas size
    dot_spacing_mm: float = 1.5 # Dot spacing for stipple
    dot_radius_mm: float = 0.5  # Dot radius for stipple
    hatch_spacing_mm: float = 2.0 # Hatch spacing
    hatch_angle_deg: float = 45.0 # Hatch angle
    cross_hatch: bool = False   # Cross hatch
    contour_bands: int = 6      # Contour bands
    min_feature_size_mm: float = 1.0 # Minimum feature size
    simplify_tol_mm: float = 0.2 # Simplify tolerance
    kerf_mm: float = 1.1       # Kerf width
    kerf_offset_mm: float = 0.0 # Kerf offset
    kerf_inout: bool = False    # Kerf inside/outside
    preserve_arcs: bool = True  # Preserve arcs
    min_feature_area_mm2: float = 1.0 # Minimum feature area
    merge_angle_deg: float = 3.0 # Merge angle
```

**Main Function**: `generate_texture_dxf(image_path: str, out_dir: str | Path, preprocess_params: Optional[PreprocessParams] = None, classify_params: Optional[TextureClassifyParams] = None, vec_params: Optional[TextureVectorizeParams] = None) -> Tuple[Path, Path]`

### 3. Web Interface

#### `src/wjp_analyser/web/app.py`

**Main Application**: Streamlit web application with multiple pages

**Pages**:
- `analyze_dxf.py`: DXF analysis interface
- `image_to_dxf.py`: Image to DXF conversion interface
- `designer.py`: AI design generation interface
- `supervisor_dashboard.py`: Supervisor dashboard

#### `src/wjp_analyser/web/_components.py`

**Main Functions**:
```python
def get_ai_status(timeout: float = 2.0) -> dict
def render_ai_status(compact: bool = True) -> None
def ensure_workdir(upload_name: str, file_bytes: bytes) -> dict
def run_analysis(work: dict, selected_groups: List[str] = None, sheet_width: float = None, sheet_height: float = None, group_layer_map: Dict[str, str] = None, soften_opts: Dict[str, object] = None, fillet_opts: Dict[str, object] = None, scale_opts: Dict[str, object] = None, normalize_opts: Dict[str, object] = None, frame_quantity: int = None) -> dict
def plot_components(report: dict, height: int = 600) -> None
def display_group_summary(report: dict, selected_groups: List[str]) -> None
def display_metrics(report: dict) -> None
def display_quality(report: dict) -> None
def display_checklist(report: dict) -> None
```

### 4. AI Integration

#### `src/wjp_analyser/ai/openai_client.py`

**Main Class**: `OpenAIClient`
```python
class OpenAIClient:
    def __init__(self, api_key: str = None)
    def analyze_dxf(self, dxf_path: str, analysis_report: dict) -> dict
    def generate_design(self, requirements: str) -> dict
    def optimize_cutting_path(self, analysis_report: dict) -> dict
    def suggest_material(self, design_characteristics: dict) -> dict
```

#### `src/wjp_analyser/ai/ollama_client.py`

**Main Class**: `OllamaClient`
```python
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434")
    def analyze_dxf(self, dxf_path: str, analysis_report: dict) -> dict
    def generate_design(self, requirements: str) -> dict
    def optimize_cutting_path(self, analysis_report: dict) -> dict
```

### 5. CLI Tools

#### `src/wjp_analyser/cli/analyze_cli.py`

**Main Function**: `main()` - Command-line DXF analysis tool

**Arguments**:
- `--input`: Input DXF file path
- `--output`: Output directory
- `--sheet-width`: Sheet width in mm
- `--sheet-height`: Sheet height in mm
- `--soften-method`: Softening method
- `--fillet-radius`: Fillet radius in mm
- `--scale-mode`: Scaling mode

#### `src/wjp_analyser/cli/batch_analyze.py`

**Main Function**: `main()` - Batch analysis tool for multiple files

## Configuration Files

### `config/ai_config.yaml`
```yaml
openai:
  api_key: "your-api-key"
  model: "gpt-4"
  max_tokens: 4000
  temperature: 0.7

ollama:
  base_url: "http://localhost:11434"
  model: "llama2"
  timeout: 30

analysis:
  default_tolerance: 0.1
  max_file_size: 10485760  # 10MB
  cache_enabled: true
```

### `config/material_profiles.py`
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

## Error Handling

### Common Exceptions
- `DXFReadError`: DXF file reading errors
- `AnalysisError`: Analysis processing errors
- `ImageProcessingError`: Image processing errors
- `AIError`: AI service errors
- `ValidationError`: Input validation errors

### Error Response Format
```python
{
    "error": "Error type",
    "message": "Human-readable error message",
    "details": "Technical details",
    "suggestion": "Suggested resolution"
}
```

## Performance Considerations

### Memory Usage
- DXF files: ~1MB per 1000 entities
- Images: ~4MB per 1MP image
- Analysis cache: ~100KB per file

### Processing Time
- DXF analysis: ~1-5 seconds per file
- Image processing: ~2-10 seconds per image
- AI analysis: ~5-30 seconds depending on complexity

### Optimization Tips
- Use caching for repeated analysis
- Process files in batches for CLI tools
- Enable parallel processing for multiple files
- Use appropriate image resolution for conversion

## Integration Examples

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

---

*This API documentation provides comprehensive information for AI systems to understand and interact with the WJP ANALYSER codebase effectively.*
