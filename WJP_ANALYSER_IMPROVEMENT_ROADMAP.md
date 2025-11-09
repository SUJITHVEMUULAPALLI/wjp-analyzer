# WJP ANALYSER - Improvement Roadmap
## Based on Comprehensive Architecture Review

**Review Date**: 2025-01-01  
**Status**: Implementation Plan

---

## Executive Summary

This roadmap addresses architectural consolidation, feature enhancements, performance optimization, UX improvements, and integration capabilities. Implementation is phased over 8-10 working days with clear priorities.

---

## Phase 1: Infrastructure & Operations (Days 1-2)

### 1.1 Single CLI Entry Point

**Current State**: Multiple entry points (`run_one_click.py`, `wjp_analyser_unified.py`, direct Streamlit execution)

**Target**: One canonical CLI `wjp` (click-based)

**Implementation**:
```python
# New file: cli/wjp_cli.py
import click

@click.group()
def wjp():
    """WJP ANALYSER - Unified CLI"""
    pass

@wjp.command()
def web():
    """Launch Streamlit UI"""
    # Starts Streamlit as UI shell only

@wjp.command()
def api():
    """Launch FastAPI server"""
    # Starts FastAPI API only

@wjp.command()
def worker():
    """Start queue workers"""
    # Starts RQ/Celery workers

@wjp.command()
def demo():
    """Run demo pipeline"""
    pass

@wjp.command()
def test():
    """Run tests"""
    pass

@wjp.command()
def status():
    """Show system status"""
    pass
```

**Actions**:
- [ ] Create `cli/wjp_cli.py` with click-based CLI
- [ ] Update `main.py` to use new CLI
- [ ] Add deprecation warnings to old launchers
- [ ] Update documentation to use `wjp` CLI

---

### 1.2 FastAPI Core API

**Current State**: Logic mixed in Streamlit pages, Flask legacy API

**Target**: FastAPI as core, Streamlit as thin client

**Implementation**:
```python
# New file: api/fastapi_app.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="WJP ANALYSER API")

# Endpoints
@app.post("/analyze-dxf")
async def analyze_dxf(dxf_file: UploadFile):
    """DXF analysis endpoint"""
    pass

@app.post("/convert-image")
async def convert_image(image_file: UploadFile):
    """Image to DXF conversion"""
    pass

@app.post("/nest")
async def nest_objects(job_id: str):
    """Nesting optimization"""
    pass

@app.post("/gcode")
async def generate_gcode(job_id: str):
    """G-code generation"""
    pass

@app.get("/cost")
async def calculate_cost(dxf_path: str, material: str, thickness: float):
    """Cost calculation"""
    pass

@app.post("/csv/ai-analysis")
async def analyze_csv(csv_file: UploadFile):
    """AI CSV analysis"""
    pass

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Job status and results"""
    pass
```

**Actions**:
- [ ] Create `api/fastapi_app.py` with all endpoints
- [ ] Move all business logic from Streamlit pages to API
- [ ] Add authentication middleware
- [ ] Add CORS middleware for Streamlit client
- [ ] Implement job queue integration

---

### 1.3 Job Queue System

**Current State**: Synchronous processing, blocks UI

**Target**: Async job queue with status tracking

**Implementation**:
- Use **RQ (Redis Queue)** for simplicity, or Celery for advanced features
- Redis for queue backend
- Job status: Queued → Running → Done/Failed
- WebSocket or polling for status updates

**Actions**:
- [ ] Install and configure Redis
- [ ] Install RQ or Celery
- [ ] Create job models
- [ ] Implement job workers
- [ ] Add job status endpoints

---

### 1.4 Storage Upgrade

**Current State**: SQLite + local file system

**Target**: 
- Postgres for multi-user (keep SQLite for dev)
- S3-compatible storage (MinIO local) for artifacts

**Actions**:
- [ ] Add Postgres support (optional, keep SQLite for single-user)
- [ ] Set up MinIO or S3-compatible storage
- [ ] Create bucket structure: `uploads/`, `artifacts/`, `reports/`
- [ ] Migrate file storage to object store
- [ ] DB stores pointers only

---

## Phase 2: Service Layer & API Migration (Days 3-4)

### 2.1 Complete Service Layer Adoption

**Current State**: Partial adoption, some logic still in UI

**Target**: All business logic in services, UI only calls API

**New Services Needed**:
- `nesting_service.py` - Nesting operations
- `gcode_service.py` - G-code generation
- `image_conversion_service.py` - Image to DXF conversion
- `layered_dxf_service.py` - ⭐ CRITICAL: Layered DXF writing

**Actions**:
- [ ] Create `layered_dxf_service.py` - First-class layered DXF writing
- [ ] Remove all duplicate costing logic, use `costing_service` only
- [ ] Create remaining service modules
- [ ] Update Streamlit pages to call API only (no direct service calls)
- [ ] Remove business logic from UI pages

---

### 2.2 API-First Refactoring

**Pattern**:
```
Streamlit Page → FastAPI Endpoint → Service → Engine → Response
```

**Actions**:
- [ ] Refactor all Streamlit pages to use API client
- [ ] Move service calls from pages to API endpoints
- [ ] Implement API client library for Streamlit
- [ ] Add proper error handling and user feedback

---

## Phase 3: UX & UI Improvements (Days 5-6)

### 3.1 Wizard Mode

**Implementation**:
- Left-to-right wizard flow per workflow
- Progress indicators
- Step validation
- Back/Next navigation

**Actions**:
- [ ] Create wizard components
- [ ] Implement step-by-step flows
- [ ] Add progress tracking
- [ ] Add validation between steps

---

### 3.2 Jobs Drawer

**Implementation**:
- Sidebar drawer showing all jobs
- Real-time status updates
- Click to view results
- Download artifacts

**Actions**:
- [ ] Create jobs drawer component
- [ ] Implement WebSocket or polling for status
- [ ] Add job result viewing
- [ ] Add artifact download UI

---

### 3.3 Error Model Improvements

**Current State**: Stack traces shown to users

**Target**: Actionable error messages with fix buttons

**Actions**:
- [ ] Create error model with fix actions
- [ ] Replace stack traces with user-friendly messages
- [ ] Add "auto-fix" buttons for known issues
- [ ] Implement error recovery flows

---

### 3.4 Consistent Terminology

**Current State**: Mixed use of "Group" and "Object"

**Target**: Consistent terminology across all pages
- **Object** = Individual DXF entity before grouping
- **Group** = Similar objects clustered together

**Actions**:
- [ ] Update all UI labels to use consistent terms
- [ ] Update CSV exports
- [ ] Update documentation
- [ ] Update variable names where visible to users

---

### 3.5 Keyboard Operations

**Implementation**:
- Keyboard shortcuts for selection
- Layer assignment shortcuts
- Bulk operations via keyboard

**Actions**:
- [ ] Add keyboard event handlers in Editor
- [ ] Implement selection shortcuts
- [ ] Add layer assignment shortcuts
- [ ] Document keyboard operations

---

## Phase 4: Performance Optimization (Days 7-8)

### 4.1 Large DXF Handling

**Implementation**:
- Streaming parse (ezdxf)
- Tile/chunk entities by layer/space
- Early simplification (Douglas-Peucker)
- Entity normalization (explode SPLINE/ELLIPSE)
- Cache tessellations per doc hash

**Actions**:
- [ ] Implement streaming DXF parser
- [ ] Add tiling/chunking logic
- [ ] Add early simplification pass
- [ ] Implement entity normalization
- [ ] Add tessellation caching

---

### 4.2 Caching Strategy

**Implementation**:
- Function-level memoization (job hash → artifacts)
- Artifact cache in object store
- Warm caches on upload
- Lazy heavy metrics

**Actions**:
- [ ] Implement job hash system
- [ ] Add artifact caching in object store
- [ ] Add warm cache on upload
- [ ] Implement lazy metric calculation

---

### 4.3 Memory Optimization

**Implementation**:
- Use Shapely 2 + pygeos backend
- STRtree everywhere for spatial queries
- float32 coordinates where tolerable
- Segment filters (drop edges below ε)
- Paginate editor table
- Never hold full geometry in Streamlit session state

**Actions**:
- [ ] Upgrade to Shapely 2
- [ ] Replace all spatial queries with STRtree
- [ ] Add coordinate precision optimization
- [ ] Implement segment filtering
- [ ] Add pagination to editor table
- [ ] Remove full geometry from session state

---

### 4.4 Distributed Batch Processing

**Implementation**:
- RQ/Celery with work queues per workload
- Idempotent jobs (content hash + parameters)
- Return existing results if present

**Actions**:
- [ ] Set up work queues per workload type
- [ ] Implement job idempotency
- [ ] Add result caching for repeated jobs

---

## Phase 5: Advanced Features (Days 9-10)

### 5.1 Production-Grade Nesting

**Implementation Steps**:
1. **Geometry Hygiene**:
   - Robust polygonization
   - Hole handling
   - Winding rules
   - Tolerance unification (µm-scale)

2. **Placement Engine**:
   - BLF (fast first pass)
   - NFP refinement for top-K pieces
   - Metaheuristic (GA/SA) on order & rotations
   - Utilize STRtree for collision checks
   - Batch-intersect using envelopes first

3. **Constraints**:
   - Hard: kerf margin, min web, pierce-avoid zones
   - Soft: part priority, grain direction, re-use offcuts

4. **Determinism**: Switch for reproducible runs

**Actions**:
- [ ] Implement geometry hygiene checks
- [ ] Enhance placement engine pipeline
- [ ] Add constraint system
- [ ] Add determinism mode
- [ ] Performance testing with real-world files

---

### 5.2 Actionable AI Recommendations

**Current State**: Advisory recommendations

**Target**: Rule+AI hybrid with executable operations

**Implementation**:
```python
# New: ai/recommendation_engine.py
from pydantic import BaseModel
from enum import Enum

class OperationType(str, Enum):
    REMOVE_ZERO_AREA = "remove_zero_area"
    SIMPLIFY_EPS = "simplify_eps"
    FILLET_MIN_RADIUS = "fillet_min_radius"
    FILTER_TINY = "filter_tiny"
    # ... more

class Recommendation(BaseModel):
    operation: OperationType
    parameters: dict
    rationale: str
    estimated_impact: dict  # length/time/cost deltas
    auto_apply: bool  # Rules engine decides
```

**Actions**:
- [ ] Create rules engine for must-fix issues
- [ ] Implement operation system
- [ ] Enhance LLM to explain fixes and suggest strategies
- [ ] Output executable operations with rationale
- [ ] Integrate with Editor auto-fix buttons

---

### 5.3 Layered DXF Service (CRITICAL FIX)

**Current Issue**: Analyzer doesn't write layered DXF by default (workaround in UI)

**Solution**: First-class service

**Implementation**:
```python
# New: services/layered_dxf_service.py
def write_layered_dxf(
    components: List[Dict],
    layers: Dict[str, List[Dict]],
    output_path: str,
    scaled: bool = False,
    normalized: bool = True
) -> str:
    """Write layered DXF from analysis components."""
    import ezdxf
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    
    # Create layers
    for layer_name in layers.keys():
        if layer_name not in doc.layers:
            doc.layers.new(name=layer_name)
    
    # Write components as polylines
    for comp in components:
        pts = comp.get("points", [])
        if len(pts) >= 2:
            layer = comp.get("layer", "0")
            polyline_points = [(float(p[0]), float(p[1])) for p in pts]
            try:
                msp.add_lwpolyline(polyline_points, dxfattribs={"layer": layer}, close=True)
            except Exception:
                msp.add_lwpolyline(polyline_points, dxfattribs={"layer": layer})
    
    doc.saveas(output_path)
    return output_path
```

**Actions**:
- [ ] Create `layered_dxf_service.py`
- [ ] Move layered DXF writing from UI to service
- [ ] Update analyzer to optionally write layered DXF
- [ ] Update API endpoint to use service
- [ ] Remove UI workarounds

---

## High-Impact Quick Fixes (Can Do Immediately)

### Fix 1: Layered DXF Service ⭐ CRITICAL
**Impact**: Unblocks downstream workflows
**Effort**: 2-3 hours
**File**: `src/wjp_analyser/services/layered_dxf_service.py` (new)

### Fix 2: Remove Duplicate Costing Logic
**Impact**: Consistency, easier maintenance
**Effort**: 1-2 hours
**Files**: All pages calling costing directly → use `costing_service` only

### Fix 3: Single CLI Entry Point
**Impact**: Clarity for users
**Effort**: 3-4 hours
**File**: `cli/wjp_cli.py` (new), update documentation

### Fix 4: AI+Rules Actions
**Impact**: More actionable recommendations
**Effort**: 4-6 hours
**Files**: Enhance `csv_analysis_service.py`, create `ai/recommendation_engine.py`

---

## Implementation Priority

### Critical (Do First)
1. ✅ **Layered DXF Service** - Blocks downstream features
2. ✅ **Remove Duplicate Costing** - Technical debt
3. ✅ **Single CLI Entry Point** - User confusion

### High Priority (Phase 1-2)
4. FastAPI core API
5. Job queue system
6. Complete service layer adoption

### Medium Priority (Phase 3-4)
7. UX improvements (wizard, jobs drawer, errors)
8. Performance optimizations
9. Consistent terminology

### Lower Priority (Phase 5)
10. Production-grade nesting
11. Advanced AI recommendations
12. Real-time collaboration
13. GPU acceleration

---

## Technical Decisions

### Queue System: RQ vs Celery
**Recommendation**: **RQ** for simplicity
- Easier setup and debugging
- Sufficient for current needs
- Can migrate to Celery later if needed

### Storage: SQLite vs Postgres
**Recommendation**: **SQLite for now, Postgres optional**
- Current single-user focus
- Add Postgres when multi-user needed
- Migration path should be easy

### API Framework: FastAPI
**Recommendation**: **FastAPI**
- Modern async support
- Automatic OpenAPI docs
- Better type hints
- Easy WebSocket support

---

## Success Metrics

### Phase 1 Completion
- [ ] Single `wjp` CLI works for all operations
- [ ] FastAPI API responds to all endpoint calls
- [ ] Job queue processes background tasks
- [ ] Artifacts stored in object store

### Phase 2 Completion
- [ ] All business logic in services
- [ ] Streamlit pages only call API
- [ ] No duplicate costing logic
- [ ] Layered DXF service works end-to-end

### Phase 3 Completion
- [ ] Wizard flows work for all workflows
- [ ] Jobs drawer shows real-time status
- [ ] Error messages are actionable
- [ ] Consistent terminology throughout

### Phase 4 Completion
- [ ] Handles 10MB+ DXFs without issues
- [ ] Caching reduces redundant computation
- [ ] Memory usage optimized for large files
- [ ] Batch processing works via queue

### Phase 5 Completion
- [ ] Nesting produces production-quality results
- [ ] AI recommendations include executable operations
- [ ] Editor auto-fix buttons work seamlessly

---

## Migration Strategy

### Backward Compatibility
- Keep old entry points with deprecation warnings
- Maintain old API endpoints during transition
- Gradual migration of pages to new API

### Testing Strategy
- Unit tests for all new services
- Integration tests for API endpoints
- E2E tests for critical workflows
- Performance tests for large files

### Rollout Plan
1. Internal testing (Phase 1-2)
2. Beta testing with select users (Phase 3)
3. Full rollout (Phase 4-5)

---

## Notes

- All citations reference the comprehensive report
- Implementation should be incremental
- Keep existing functionality working during migration
- Document all API changes
- Provide migration guide for users

---

**Next Steps**: Begin with high-impact quick fixes, then proceed through phases systematically.








