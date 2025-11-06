# DXF Editor Enhancement Plan
## Visual Preview Improvements & Layer Coloring System

**Date:** Current  
**Status:** Planning Phase

---

## 🎯 Objectives

1. **Normalize preview to origin (0,0)** - All DXF previews start from origin for consistency
2. **Layer-based coloring** - Outer layers (blue), inner objects (single color)
3. **Warning visualization** - Red color for warnings/issues
4. **Customization system** - Framework for advanced user preferences

---

## 📋 Implementation Plan

### Phase 1: Preview Normalization to Origin

#### 1.1 Initial Preview Normalization
**File:** `src/wjp_analyser/web/pages/dxf_editor.py`  
**Location:** Lines 70-95 (fallback preview) and line 164-170 (main preview)

**Changes:**
- Extract bounding box from all entities/polygons
- Calculate offset to move bottom-left corner to (0,0)
- Apply transformation to preview coordinates
- Display normalized preview with origin at (0,0)

**Implementation Steps:**
```python
def _normalize_to_origin(points_list):
    """Normalize all points so minimum x,y is at (0,0)"""
    all_points = []
    for pts in points_list:
        all_points.extend(pts)
    
    if not all_points:
        return points_list, (0, 0)
    
    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    
    normalized = []
    for pts in points_list:
        normalized.append([(x - min_x, y - min_y) for x, y in pts])
    
    return normalized, (min_x, min_y)
```

**Benefits:**
- Consistent preview positioning
- Easier comparison between files
- Standard waterjet workflow (origin-based)

---

### Phase 2: Layer Classification & Coloring System

#### 2.1 Layer Classification
**File:** `src/wjp_analyser/analysis/dxf_analyzer.py`  
**Reuse:** `_classify_polylines()` function (already exists)

**Layer Types:**
- **OUTER** - Largest boundary polygon (Blue: `#0066CC` or `#2196F3`)
- **INNER** - Objects inside outer boundary (Single color: `#666666` or `#9E9E9E`)
- **HOLE** - Holes/cutouts (Inner color variant)
- **COMPLEX** - High vertex count (May use distinct color)
- **DECOR** - Decorative elements (Inner color)

#### 2.2 Coloring Logic

**Standard Color Scheme:**
```python
LAYER_COLORS = {
    "OUTER": "#0066CC",      # Blue - primary boundaries
    "INNER": "#666666",      # Gray - interior objects  
    "HOLE": "#666666",       # Gray - same as inner
    "COMPLEX": "#666666",    # Gray - same as inner
    "DECOR": "#666666",      # Gray - same as inner
}
```

**Visual Rules:**
1. **Outer layers** = Blue (`#0066CC`) - Always distinct
2. **All inner objects** = Single gray color (`#666666`)
3. **Selected entities** = Red (existing behavior)
4. **Warnings** = Red background/border

---

### Phase 3: Enhanced Preview Rendering

#### 3.1 Update Preview Functions

**File 1:** `src/wjp_analyser/web/pages/dxf_editor.py`
- Lines 70-95: Fallback preview (polygonized)
- Line 164-170: Main entity preview

**File 2:** `src/wjp_analyser/dxf_editor/visualize.py`
- `plot_entities()` function enhancement

**Changes:**

1. **Fallback Preview (Polygonized):**
   - Apply normalization
   - Classify polygons using existing logic
   - Color by layer type
   - Blue for OUTER, gray for others

2. **Main Entity Preview:**
   - Apply normalization
   - Use layer classification for supported entities
   - Fallback to entity-based classification for unsupported types

#### 3.2 New Preview Function

```python
def _render_normalized_preview(
    entities_or_polygons,
    classify_layers=True,
    normalize=True,
    show_warnings=True
):
    """
    Render preview with:
    - Normalized to origin (0,0)
    - Layer-based coloring (OUTER=blue, INNER=gray)
    - Warning highlighting (red)
    """
    # 1. Normalize to origin
    # 2. Classify layers (OUTER vs INNER)
    # 3. Apply color scheme
    # 4. Highlight warnings
    pass
```

---

### Phase 4: Warning Visualization System

#### 4.1 Warning Detection
**Location:** Already exists in analysis warnings

**Warning Types:**
- `single_outer` - Only 1 object on OUTER layer
- `all_selected` - All objects selected
- `min_radius_violations` - Minimum radius issues
- `open_contours` - Unclosed shapes
- `tiny_segments` - Very small segments

#### 4.2 Warning Display
**Color Scheme:**
- **Warning background:** Light red (`#FFEBEE` or `#FCE4EC`)
- **Warning border:** Red (`#F44336` or `#E91E63`)
- **Warning text:** Dark red (`#C62828`)
- **Warning icon:** Red

**Implementation:**
```python
def _highlight_warnings(ax, warnings_list, bbox):
    """Overlay warning indicators on preview"""
    for warning in warnings_list:
        # Draw red border/background around affected area
        # Add warning annotation
        pass
```

---

### Phase 5: Customization System (Future Enhancement)

#### 5.1 Color Customization
**UI Components:**
- Color picker for OUTER layer
- Color picker for INNER layer
- Preset color schemes (Default, High Contrast, Colorblind-friendly)
- Save preferences to session state

#### 5.2 Preview Options
**Settings Panel:**
- Toggle normalization on/off
- Choose classification method (by layer vs by geometry)
- Adjust preview resolution/quality
- Show/hide grid, axes, dimensions

#### 5.3 Advanced Features (Future)
- Custom layer color mapping
- Transparency controls per layer type
- Export preview settings
- Keyboard shortcuts for common actions

---

## 🔧 Technical Implementation Details

### Files to Modify

1. **`src/wjp_analyser/web/pages/dxf_editor.py`**
   - Add normalization function
   - Update preview rendering (lines 70-95, 164-170)
   - Integrate layer classification
   - Add warning display

2. **`src/wjp_analyser/dxf_editor/visualize.py`**
   - Enhance `plot_entities()` with layer colors
   - Add normalization support
   - Add warning highlighting

3. **`src/wjp_analyser/analysis/dxf_analyzer.py`**
   - Reuse `_classify_polylines()` function
   - Export classification function for editor use

4. **New File: `src/wjp_analyser/dxf_editor/preview_utils.py`**
   - Normalization functions
   - Color scheme definitions
   - Warning visualization helpers

---

## 🎨 Color Scheme Specifications

### Default Color Palette

```python
COLOR_SCHEME = {
    # Layer Colors
    "OUTER": {
        "fill": "#0066CC",      # Blue
        "edge": "#004499",      # Darker blue
        "alpha": 0.7
    },
    "INNER": {
        "fill": "#666666",      # Gray
        "edge": "#444444",      # Darker gray
        "alpha": 0.6
    },
    "HOLE": {
        "fill": "#666666",      # Same as inner
        "edge": "#444444",
        "alpha": 0.6
    },
    # UI Colors
    "WARNING": {
        "background": "#FFEBEE",  # Light red
        "border": "#F44336",      # Red
        "text": "#C62828"         # Dark red
    },
    "SELECTED": {
        "fill": "#FF0000",        # Red
        "edge": "#CC0000",
        "alpha": 0.8
    }
}
```

---

## 📊 Preview Flow Diagram

```
DXF Upload
    ↓
Extract Entities/Polygons
    ↓
[Phase 1] Normalize to Origin (0,0)
    ↓
[Phase 2] Classify Layers (OUTER vs INNER)
    ↓
[Phase 4] Detect Warnings
    ↓
[Phase 3] Apply Color Scheme
    ├─ OUTER → Blue
    ├─ INNER → Gray
    └─ Warnings → Red highlights
    ↓
Render Preview
```

---

## ✅ Implementation Checklist

### Phase 1: Normalization
- [ ] Create `_normalize_to_origin()` function
- [ ] Apply to fallback preview (polygonized)
- [ ] Apply to main entity preview
- [ ] Test with various DXF files
- [ ] Verify origin alignment

### Phase 2: Layer Classification
- [ ] Integrate `_classify_polylines()` into editor
- [ ] Create layer color mapping
- [ ] Update preview rendering to use layer colors
- [ ] Test classification accuracy

### Phase 3: Preview Enhancement
- [ ] Update `plot_entities()` with layer colors
- [ ] Enhance polygonized preview coloring
- [ ] Ensure normalization works with both previews
- [ ] Add legend showing layer types

### Phase 4: Warning System
- [ ] Create warning detection function
- [ ] Implement warning overlay rendering
- [ ] Style warnings with red color scheme
- [ ] Add warning tooltips/annotations
- [ ] Test warning display

### Phase 5: Customization (Future)
- [ ] Design color customization UI
- [ ] Create preset color schemes
- [ ] Implement preference storage
- [ ] Add preview options panel

---

## 🧪 Testing Requirements

1. **Normalization Testing:**
   - Files with entities at various positions
   - Files already at origin
   - Very large coordinates
   - Negative coordinates

2. **Layer Classification Testing:**
   - Files with clear OUTER boundaries
   - Nested structures (OUTER → INNER → HOLE)
   - Complex geometry
   - Multiple disconnected outer boundaries

3. **Color Rendering Testing:**
   - Verify blue for OUTER layers
   - Verify gray for INNER objects
   - Check warning highlighting
   - Test with various layer combinations

4. **Performance Testing:**
   - Large files (1000+ entities)
   - Complex polygons
   - Real-time preview updates

---

## 📝 Code Structure

### New Functions to Create

```python
# In preview_utils.py
def normalize_to_origin(points_list) -> tuple
def classify_layers_for_preview(polygons) -> dict
def apply_layer_colors(fig, ax, classified_data, color_scheme) -> None
def highlight_warnings(ax, warnings, bbox) -> None

# In dxf_editor.py
def _render_enhanced_preview(doc, entities, warnings=None) -> matplotlib.figure
```

---

## 🚀 Rollout Plan

1. **Week 1:** Phase 1 & 2 (Normalization + Basic Coloring)
2. **Week 2:** Phase 3 (Enhanced Preview)
3. **Week 3:** Phase 4 (Warning System)
4. **Week 4:** Testing & Refinement
5. **Future:** Phase 5 (Customization)

---

## 📚 References

- Existing layer classification: `_classify_polylines()` in `dxf_analyzer.py`
- Preview rendering: `plot_entities()` in `visualize.py`
- Polygon extraction: `extract_polygons()` in `dxf_extractor.py`
- Normalization example: `_preview_dxf_normalized()` in `analyze_dxf.py`

---

## 🎯 Success Criteria

✅ All previews normalize to origin (0,0)  
✅ Outer layers displayed in blue  
✅ Inner objects displayed in single gray color  
✅ Warnings displayed in red  
✅ Customization system planned and documented  
✅ No performance degradation  
✅ Works with both entity-based and polygonized previews  

---

**Next Steps:** Begin Phase 1 implementation


