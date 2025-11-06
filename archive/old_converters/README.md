# Old Image to DXF Converters

This directory contains the old, conflicting image to DXF converters that have been replaced by the unified converter.

## Files Archived

- `basic.py` - Basic image to DXF converter with hardcoded parameters
- `opencv_converter.py` - OpenCV-based converter with adjustable parameters
- `enhanced_opencv_converter.py` - Enhanced converter with border removal

## Why These Were Replaced

These converters had several issues:

1. **Inconsistent Parameters**: Different threshold values, min_area settings, and DXF sizes
2. **Conflicting Logic**: Some used `THRESH_BINARY_INV`, others used `THRESH_BINARY`
3. **Integration Problems**: Different web interfaces used different converters
4. **Missing Dependencies**: References to non-existent modules
5. **Inconsistent Output**: Different coordinate systems and metadata handling

## Replacement

All functionality has been consolidated into:
- `src/wjp_analyser/image_processing/converters/unified_converter.py`

The unified converter provides:
- Standardized parameters with sensible defaults
- Consistent threshold logic (`THRESH_BINARY`)
- Proper error handling and logging
- Consistent output format and metadata
- Backward compatibility through convenience functions

## Migration Guide

If you need to use the old converters, you can:

1. Copy them back to the converters directory
2. Update imports in your code
3. Be aware of the parameter inconsistencies

However, it's recommended to use the unified converter instead.
