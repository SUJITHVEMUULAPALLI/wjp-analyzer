# 🎨 Inkscape Integration Guide

## Why Inkscape?

Inkscape provides **superior vectorization** compared to basic OpenCV approaches:

- **Better edge detection** - Advanced algorithms for clean vectorization
- **Path optimization** - Automatic simplification and smoothing
- **Multiple trace methods** - Autotrace, Potrace, Centerline
- **Professional quality** - Used by graphic designers worldwide
- **DXF export** - Native support for CAD formats

## 📥 Installation

### Windows
1. **Download Inkscape**: https://inkscape.org/release/
2. **Install with command-line tools** (important!)
3. **Add to PATH**: Add `C:\Program Files\Inkscape\bin\` to your system PATH
4. **Verify installation**:
   ```bash
   inkscape --version
   ```

### Alternative: Portable Version
1. Download portable Inkscape
2. Extract to a folder (e.g., `C:\Inkscape\`)
3. Use full path in converter: `inkscape_path="C:\Inkscape\bin\inkscape.exe"`

## 🚀 Usage

### Basic Usage
```python
from image_processing.converters.inkscape_converter import InkscapeImageToDXFConverter

converter = InkscapeImageToDXFConverter()
result = converter.convert_image_to_dxf(
    input_image="your_image.png",
    output_dxf="output.dxf",
    preview_output="preview.png"
)
```

### Advanced Configuration
```python
converter = InkscapeImageToDXFConverter(
    trace_method="autotrace",    # or "potrace", "centerline"
    threshold=0.5,               # Edge detection sensitivity (0.0-1.0)
    simplify=0.1,               # Path simplification (0.0-1.0)
    smooth_corners=True          # Smooth sharp corners
)
```

### CLI Integration
```bash
python -m cli.wjdx inkscape your_image.png --out output --threshold 0.5 --simplify 0.1
```

## 🔧 Parameters Explained

- **trace_method**: 
  - `"autotrace"` - Best for photos and complex images
  - `"potrace"` - Good for simple graphics and logos
  - `"centerline"` - Best for line drawings

- **threshold**: 
  - `0.0` - Very sensitive (more details)
  - `1.0` - Less sensitive (fewer details)
  - `0.5` - Balanced (recommended)

- **simplify**: 
  - `0.0` - No simplification (more points)
  - `1.0` - Maximum simplification (fewer points)
  - `0.1` - Light simplification (recommended)

## 🎯 Advantages Over OpenCV

| Feature | Inkscape | OpenCV |
|---------|----------|--------|
| Edge Detection | ✅ Advanced algorithms | ❌ Basic thresholding |
| Path Optimization | ✅ Automatic | ❌ Manual |
| Noise Reduction | ✅ Built-in | ❌ Requires filters |
| Path Smoothing | ✅ Professional | ❌ Basic |
| DXF Quality | ✅ CAD-ready | ❌ Needs cleanup |
| User Control | ✅ Many options | ❌ Limited |

## 🛠️ Troubleshooting

### "Inkscape not found" Error
```bash
# Check if Inkscape is in PATH
inkscape --version

# If not found, use full path
converter = InkscapeImageToDXFConverter(
    inkscape_path="C:\\Program Files\\Inkscape\\bin\\inkscape.exe"
)
```

### Conversion Fails
1. **Check image format** - PNG, JPG, BMP work best
2. **Reduce image size** - Large images may timeout
3. **Adjust parameters** - Try different threshold values
4. **Check file permissions** - Ensure write access to output directory

### Poor Quality Results
1. **Increase threshold** - For cleaner edges
2. **Enable smoothing** - For smoother curves
3. **Try different trace method** - Autotrace vs Potrace
4. **Preprocess image** - Clean up image before conversion

## 📊 Expected Results

With Inkscape, you should get:
- **Cleaner vector paths** - No jagged edges
- **Better DXF files** - Ready for CAD software
- **Fewer artifacts** - Less noise and unwanted details
- **Professional quality** - Suitable for manufacturing

## 🔄 Integration with Existing Pipeline

The Inkscape converter integrates seamlessly:
1. **Web Interface** - Available as conversion option
2. **CLI Commands** - New `inkscape` command
3. **Batch Processing** - Process multiple images
4. **Quality Control** - Better results than OpenCV

Try Inkscape for your `Tile_1.png` - you should see much better DXF conversion results!
