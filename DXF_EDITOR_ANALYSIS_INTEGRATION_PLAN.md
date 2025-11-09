# DXF Editor - Analysis Recommendations Integration Plan

## Overview
Integrate DXF analysis and recommendation engine into the DXF Editor to allow users to:
1. Run analysis on the current DXF file
2. View recommendations and issues
3. Apply fixes automatically or manually
4. Preview changes before applying
5. Export cleaned/fixed DXF files

---

## Architecture

### Components

1. **Analysis Service Integration**
   - Use existing `analyze_dxf()` from `dxf_analyzer.py`
   - Use `RecommendationEngine` from `ai/recommendation_engine.py`
   - Generate executable operations

2. **Operation Executors**
   - Create operation executors that apply fixes to DXF entities
   - Map `OperationType` enum to actual DXF modification functions
   - Support preview mode (dry-run) and apply mode

3. **UI Components**
   - Analysis panel in sidebar or main area
   - Recommendations list with severity indicators
   - Apply buttons (individual or batch)
   - Preview before/after comparison
   - Progress indicators

---

## Implementation Plan

### Phase 1: Analysis Integration

#### 1.1 Add Analysis Section to Editor
**Location**: `src/wjp_analyser/web/pages/dxf_editor.py`

**Features**:
- "Run Analysis" button
- Display analysis results summary
- Show readiness score
- List of issues found

**UI Layout**:
```
┌─────────────────────────────┐
│ 🔍 Analysis & Recommendations│
├─────────────────────────────┤
│ [Run Analysis]              │
│                             │
│ Readiness: 🟢 85% (Good)    │
│                             │
│ Issues Found:               │
│ • 3 Critical                │
│ • 5 Errors                  │
│ • 2 Warnings                │
└─────────────────────────────┘
```

#### 1.2 Run Analysis Function
```python
def run_dxf_analysis(doc, temp_path: str) -> Dict[str, Any]:
    """Run DXF analysis on current document."""
    # Save doc to temp file
    doc.saveas(temp_path)
    
    # Run analysis
    from wjp_analyser.analysis.dxf_analyzer import analyze_dxf
    report = analyze_dxf(temp_path)
    
    # Get recommendations
    from wjp_analyser.ai.recommendation_engine import analyze_and_recommend
    recommendations = analyze_and_recommend(report)
    
    return {
        "report": report,
        "recommendations": recommendations,
        "operations": recommendations["operations"]
    }
```

---

### Phase 2: Recommendations Display

#### 2.1 Recommendations Panel
**Location**: New section in sidebar or expandable panel in main area

**Features**:
- Grouped by severity (Critical, Error, Warning, Info)
- Show operation type, rationale, affected count
- Auto-apply indicator
- Individual apply buttons
- Batch apply for auto-apply operations

**UI Layout**:
```
┌─────────────────────────────────────┐
│ 📋 Recommendations                  │
├─────────────────────────────────────┤
│ 🔴 Critical (3)                     │
│ ┌─────────────────────────────────┐ │
│ │ Remove Zero-Area Objects        │ │
│ │ 3 objects affected              │ │
│ │ [Auto-Apply] [Preview] [Apply]  │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ⚠️ Errors (5)                       │
│ ┌─────────────────────────────────┐ │
│ │ Fix Minimum Radius Violations   │ │
│ │ 5 corners need fixing           │ │
│ │ [Preview] [Apply]               │ │
│ └─────────────────────────────────┘ │
│                                     │
│ [Apply All Auto-Fixes]              │
└─────────────────────────────────────┘
```

#### 2.2 Recommendation Card Component
```python
def render_recommendation_card(operation: Operation, index: int):
    """Render a single recommendation card."""
    severity_colors = {
        "critical": "🔴",
        "error": "⚠️",
        "warning": "🟡",
        "info": "ℹ️"
    }
    
    with st.container():
        st.markdown(f"**{severity_colors.get(operation.severity, '')} {operation.operation.value}**")
        st.caption(operation.rationale)
        st.write(f"Affected: {operation.affected_count} entities")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Preview", key=f"preview_{index}"):
                preview_operation(operation)
        with col2:
            if st.button("Apply", key=f"apply_{index}"):
                apply_operation(operation)
        with col3:
            if operation.auto_apply:
                st.badge("Auto-Apply", type="success")
```

---

### Phase 3: Operation Executors

#### 3.1 Create Operation Executor Module
**Location**: `src/wjp_analyser/dxf_editor/operation_executor.py`

**Purpose**: Map operation types to actual DXF modification functions

**Structure**:
```python
class OperationExecutor:
    """Execute operations on DXF documents."""
    
    def __init__(self, doc, msp):
        self.doc = doc
        self.msp = msp
        self.entities = list(msp)
    
    def execute(self, operation: Operation, preview: bool = False) -> Dict[str, Any]:
        """Execute an operation on the DXF."""
        executor_map = {
            OperationType.REMOVE_ZERO_AREA: self._remove_zero_area,
            OperationType.CLOSE_OPEN_CONTOUR: self._close_open_contours,
            OperationType.FILLET_MIN_RADIUS: self._fillet_min_radius,
            OperationType.FILTER_TINY: self._filter_tiny,
            OperationType.SIMPLIFY_EPS: self._simplify_eps,
            OperationType.FIX_MIN_SPACING: self._fix_min_spacing,
            OperationType.REMOVE_DUPLICATE: self._remove_duplicates,
            OperationType.ASSIGN_LAYER: self._assign_layer,
        }
        
        executor = executor_map.get(operation.operation)
        if executor:
            return executor(operation.parameters, preview)
        else:
            return {"success": False, "error": f"Unknown operation: {operation.operation}"}
    
    def _remove_zero_area(self, params: Dict, preview: bool) -> Dict:
        """Remove zero-area objects."""
        # Implementation
        pass
    
    def _close_open_contours(self, params: Dict, preview: bool) -> Dict:
        """Close open contours."""
        # Use geometry_cleaner.close_small_gaps
        pass
    
    # ... other executors
```

#### 3.2 Integration with Existing Cleaners
- Use `geometry_cleaner.py` for geometry operations
- Use `dxf_cleaner.py` for entity-level cleaning
- Use `repair.py` for gap closing and duplicate removal

---

### Phase 4: Preview & Apply Workflow

#### 4.1 Preview Mode
**Features**:
- Show what will change before applying
- Highlight affected entities
- Show before/after metrics
- Side-by-side comparison option

**Implementation**:
```python
def preview_operation(operation: Operation, doc, msp):
    """Preview operation without applying."""
    executor = OperationExecutor(doc, msp)
    result = executor.execute(operation, preview=True)
    
    # Show preview
    st.subheader("Preview Changes")
    st.write(f"**Operation**: {operation.operation.value}")
    st.write(f"**Rationale**: {operation.rationale}")
    
    # Show affected entities
    if result.get("affected_entities"):
        st.write(f"**Entities to modify**: {len(result['affected_entities'])}")
        # Highlight in preview
    
    # Show metrics
    if result.get("metrics"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before", result["metrics"]["before"])
        with col2:
            st.metric("After", result["metrics"]["after"])
    
    # Apply button
    if st.button("Apply This Fix", type="primary"):
        apply_operation(operation)
```

#### 4.2 Apply Mode
**Features**:
- Apply operation to DXF document
- Update entity list
- Refresh preview
- Show success/error messages
- Undo capability (optional)

**Implementation**:
```python
def apply_operation(operation: Operation, doc, msp):
    """Apply operation to DXF."""
    executor = OperationExecutor(doc, msp)
    result = executor.execute(operation, preview=False)
    
    if result.get("success"):
        st.success(f"✅ Applied: {operation.operation.value}")
        st.write(f"Modified {result.get('affected_count', 0)} entities")
        st.rerun()  # Refresh editor
    else:
        st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
```

---

### Phase 5: Batch Operations

#### 5.1 Apply All Auto-Fixes
**Features**:
- Button to apply all auto-apply operations
- Progress indicator
- Summary of changes
- Option to undo all

**Implementation**:
```python
def apply_all_auto_fixes(operations: List[Operation], doc, msp):
    """Apply all auto-apply operations."""
    auto_ops = [op for op in operations if op.auto_apply]
    
    if not auto_ops:
        st.info("No auto-apply operations available")
        return
    
    progress_bar = st.progress(0)
    results = []
    
    for i, op in enumerate(auto_ops):
        progress_bar.progress((i + 1) / len(auto_ops))
        executor = OperationExecutor(doc, msp)
        result = executor.execute(op, preview=False)
        results.append(result)
    
    # Show summary
    success_count = sum(1 for r in results if r.get("success"))
    st.success(f"✅ Applied {success_count}/{len(auto_ops)} auto-fixes")
    st.rerun()
```

---

## File Structure

```
src/wjp_analyser/dxf_editor/
├── __init__.py
├── io_utils.py
├── visualize.py
├── transform_utils.py
├── selection.py
├── layers.py
├── groups.py
├── draw.py
├── measure.py
├── validate.py
├── repair.py
├── session.py
├── operation_executor.py  # NEW - Execute operations
└── analysis_integration.py  # NEW - Analysis UI components
```

---

## UI Integration Points

### Sidebar Section
Add new section after "Validation Tools":
```python
st.divider()
render_analysis_and_recommendations(doc, msp, entities)
```

### Main Area Tab (Optional)
Add tabs to main area:
- "Edit" (current functionality)
- "Analysis" (analysis results and recommendations)
- "Preview" (before/after comparison)

---

## Operation Executor Implementation Details

### 1. REMOVE_ZERO_AREA
```python
def _remove_zero_area(self, params: Dict, preview: bool) -> Dict:
    """Remove entities with zero area."""
    removed = []
    for entity in self.entities:
        if self._has_zero_area(entity):
            if not preview:
                self.msp.delete_entity(entity)
            removed.append(entity)
    return {
        "success": True,
        "affected_count": len(removed),
        "removed_entities": [e.dxf.handle for e in removed]
    }
```

### 2. CLOSE_OPEN_CONTOUR
```python
def _close_open_contours(self, params: Dict, preview: bool) -> Dict:
    """Close open contours."""
    tolerance = params.get("tolerance_mm", 0.1)
    from wjp_analyser.analysis.geometry_cleaner import clean_geometry
    
    result = clean_geometry(self.entities, tolerance=tolerance)
    # Apply changes to DXF
    return result
```

### 3. FILLET_MIN_RADIUS
```python
def _fillet_min_radius(self, params: Dict, preview: bool) -> Dict:
    """Apply fillets to sharp corners."""
    min_radius = params.get("min_radius_mm", 2.0)
    # Use geometry operations to add fillets
    # This is complex - may need Shapely operations
    pass
```

### 4. SIMPLIFY_EPS
```python
def _simplify_eps(self, params: Dict, preview: bool) -> Dict:
    """Simplify geometry with Douglas-Peucker."""
    tolerance = params.get("tolerance_mm", 0.05)
    from shapely.geometry import LineString
    from shapely.ops import simplify
    
    simplified = []
    for entity in self.entities:
        # Convert to LineString, simplify, convert back
        pass
    return {"success": True, "simplified_count": len(simplified)}
```

---

## Testing Plan

### Unit Tests
- Test each operation executor
- Test preview mode
- Test error handling

### Integration Tests
- Test full analysis → recommendations → apply workflow
- Test batch operations
- Test undo functionality

### User Testing
- Test with various DXF files
- Test with different issue types
- Test UI responsiveness

---

## Future Enhancements

1. **Undo/Redo System**
   - Track operations history
   - Allow undo/redo

2. **Custom Rules**
   - User-defined validation rules
   - Custom operation parameters

3. **AI Explanations**
   - LLM-powered explanations for recommendations
   - Context-aware suggestions

4. **Comparison View**
   - Side-by-side before/after
   - Metrics comparison
   - Visual diff

5. **Export Options**
   - Export cleaned DXF
   - Export report with applied fixes
   - Export operation log

---

## Implementation Priority

### High Priority (Phase 1-2)
1. ✅ Analysis integration
2. ✅ Recommendations display
3. ✅ Basic operation executors (remove zero area, close contours)

### Medium Priority (Phase 3-4)
4. ⚠️ Preview mode
5. ⚠️ Apply operations
6. ⚠️ Batch auto-fix

### Low Priority (Phase 5+)
7. 📝 Advanced operations (fillet, simplify)
8. 📝 Undo/redo
9. 📝 Comparison view

---

## Success Criteria

✅ Users can run analysis on DXF files in editor
✅ Recommendations are clearly displayed with severity
✅ Critical issues can be auto-fixed with one click
✅ Users can preview changes before applying
✅ All operations update the DXF document correctly
✅ Editor preview refreshes after applying fixes
✅ Users can export cleaned DXF files

---

## Notes

- Reuse existing analysis and cleaning functions where possible
- Keep operation executors stateless where possible
- Use session state to track applied operations for undo
- Consider performance for large DXF files
- Provide clear feedback for all operations


