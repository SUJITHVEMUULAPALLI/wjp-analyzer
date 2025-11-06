# WJP Image Analyzer

The WJP Image Analyzer is an intelligent pre-processing module that evaluates images for DXF conversion suitability before waterjet cutting. It acts as a quality gate in your image-to-DXF pipeline, providing detailed diagnostic reports and actionable suggestions.

## 🎯 Overview

The analyzer evaluates images across multiple dimensions:

- **Basic Info**: Dimensions, aspect ratio, grayscale statistics
- **Contrast & Clarity**: Edge density, edge contrast ratio
- **Texture & Noise**: Entropy, FFT high-frequency energy analysis
- **Orientation**: Skew angle detection and correction suggestions
- **Topology**: Contour analysis, closed/open path detection
- **Manufacturability**: Minimum spacing, curve radius analysis
- **Suitability Score**: Composite score (0-100) with pass/warn/fail thresholds

## 🚀 Quick Start

### Basic Usage

```python
from src.wjp_analyser.image_analyzer import analyze_image_for_wjp, AnalyzerConfig

# Analyze an image
report = analyze_image_for_wjp('path/to/image.png')
print(f"Suitability Score: {report['score']}/100")
print(f"Suggestions: {report['suggestions']}")
```

### Integration Gate

```python
from src.wjp_analyser.image_analyzer import quick_analyze

# Quick analysis with decision
should_proceed, report = quick_analyze('path/to/image.png', min_score=75.0)

if should_proceed:
    print("✅ Image ready for DXF conversion")
    # Proceed with your DXF conversion pipeline
else:
    print("❌ Image needs preprocessing")
    print("Suggestions:", report['suggestions'])
```

### CLI Tool

```bash
# Basic analysis
python -m src.wjp_analyser.image_analyzer.cli test_image.png

# Verbose output with summary
python -m src.wjp_analyser.image_analyzer.cli test_image.png --verbose

# Save report to file
python -m src.wjp_analyser.image_analyzer.cli test_image.png --output report.json
```

## 📊 Analysis Report Structure

```json
{
  "file": "image.png",
  "score": 83.0,
  "basic_stats": {
    "mean_gray": 128.5,
    "std_gray": 45.2,
    "aspect_ratio": 1.33
  },
  "orientation": {
    "skew_angle_deg": 2.8
  },
  "texture_metrics": {
    "edge_density": 0.11,
    "edge_contrast_ratio": 15.3,
    "entropy": 5.2,
    "fft_highfreq_energy": 0.08
  },
  "topology_preview": {
    "total_contours": 152,
    "closed_contours": 118,
    "closed_ratio": 0.776,
    "small_features_count": 5
  },
  "manufacturability": {
    "min_spacing_unit": 2.2,
    "min_radius_unit": 1.8
  },
  "flags": {
    "low_edge_density": false,
    "high_texture_noise": false,
    "tight_spacing": true,
    "skew_detected_deg": 2.8
  },
  "suggestions": [
    "Increase minimum gap to ≥ 3.00 units.",
    "Rotate by 2.8° to deskew before conversion."
  ]
}
```

## ⚙️ Configuration

### AnalyzerConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size_px` | 2000 | Resize longest edge if larger (for performance) |
| `px_to_unit` | 1.0 | Pixel-to-unit conversion (set mm/px if known) |
| `min_spacing_unit` | 3.0 | Minimum safe spacing for waterjet cutting |
| `min_radius_unit` | 1.5 | Minimum safe radius for curves |
| `good_edge_density` | (0.05, 0.25) | Acceptable edge density range |
| `max_texture_fft_energy` | 0.15 | Maximum texture/noise threshold |
| `min_closed_ratio_good` | 0.80 | Minimum closed contour ratio for "good" |

### Custom Configuration Example

```python
from src.wjp_analyser.image_analyzer import AnalyzerConfig, ImageAnalyzerGate

# Create custom config
config = AnalyzerConfig(
    px_to_unit=0.1,  # 0.1 mm per pixel
    min_spacing_unit=2.0,  # 2mm minimum spacing
    min_radius_unit=1.0,   # 1mm minimum radius
    max_texture_fft_energy=0.25,  # More lenient texture detection
    invert=True  # For white shapes on black background
)

# Use with analyzer gate
gate = ImageAnalyzerGate(config=config, min_score_threshold=60.0)
should_proceed, report = gate.analyze_and_decide('image.png')
```

## 🔧 Integration Examples

### 1. Pre-Conversion Gate

```python
from src.wjp_analyser.image_analyzer import create_analyzer_gate

def process_image_to_dxf(image_path):
    # Analyze before conversion
    gate = create_analyzer_gate(min_score=75.0)
    should_proceed, report = gate.analyze_and_decide(image_path)
    
    if not should_proceed:
        print(f"Image not suitable: {report['gate_decision']['reason']}")
        print("Suggestions:", report['suggestions'])
        return None
    
    # Proceed with DXF conversion
    print(f"✅ Proceeding with conversion (score: {report['score']})")
    # ... your DXF conversion code here ...
```

### 2. Batch Processing

```python
def process_image_batch(image_paths, min_score=75.0):
    gate = create_analyzer_gate(min_score=min_score)
    
    results = {'ready': [], 'needs_work': [], 'failed': []}
    
    for image_path in image_paths:
        should_proceed, report = gate.analyze_and_decide(image_path)
        
        if should_proceed:
            results['ready'].append((image_path, report['score']))
        elif report['score'] >= 50:
            results['needs_work'].append((image_path, report['score']))
        else:
            results['failed'].append((image_path, report['score']))
    
    return results
```

### 3. Web API Integration

```python
from flask import Flask, request, jsonify
from src.wjp_analyser.image_analyzer import quick_analyze

app = Flask(__name__)

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    min_score = request.form.get('min_score', 75.0, type=float)
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    try:
        should_proceed, report = quick_analyze(temp_path, min_score)
        return jsonify({
            'should_proceed': should_proceed,
            'score': report['score'],
            'suggestions': report['suggestions'],
            'report': report
        })
    finally:
        # Clean up temp file
        os.remove(temp_path)
```

## 📈 Scoring System

The analyzer uses a weighted scoring system (0-100):

- **Edge Density**: -15 points if too low/high
- **Texture Noise**: -20 points if excessive
- **Entropy**: -10 points if too flat/busy
- **Closed Contours**: -10 to -25 points based on ratio
- **Spacing**: -20 points if too tight
- **Radius**: -10 points if too small

### Score Interpretation

- **75-100**: ✅ **Excellent** - Ready for DXF conversion
- **50-74**: ⚠️ **Moderate** - Review suggestions, may need preprocessing
- **0-49**: ❌ **Poor** - Significant issues, requires substantial work

## 🎨 Visual Analysis (Phase 2 - Coming Soon)

Future versions will include:

- Streamlit web interface
- Visual overlay showing problem areas
- Interactive parameter adjustment
- Real-time analysis preview

## 🔍 Troubleshooting

### Common Issues

1. **"Image is too flat"** - Increase contrast or use histogram equalization
2. **"High texture noise"** - Simplify background or use edge detection preprocessing
3. **"Tight spacing"** - Increase gaps between features or use larger scale
4. **"Open contours"** - Ensure shapes are closed or use contour closing algorithms

### Performance Tips

- Use `max_size_px` to limit analysis resolution for large images
- Set `px_to_unit` correctly for accurate manufacturability checks
- Adjust thresholds based on your specific waterjet requirements

## 📚 Dependencies

- `opencv-python` - Image processing and computer vision
- `numpy` - Numerical computations
- `pillow` - Optional image format support

## 🚀 Next Steps

1. **Phase 2**: Streamlit visualization interface
2. **Phase 3**: Automatic preprocessing suggestions
3. **Phase 4**: Machine learning-based optimization
4. **Phase 5**: Real-time analysis integration

---

For more examples and advanced usage, see `examples/image_analyzer_integration.py`.
