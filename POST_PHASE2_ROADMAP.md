# WJP Analyzer – Post-Phase-2 Roadmap

**Current Status**: Phase-2 Complete ✅ (95% coverage, 44 tests passing)

---

## 1. Optional: 100% Coverage Push (Polish Stage)

**Goal**: Achieve 100% test coverage for DXF Editor modules

**Missing Coverage**: 7 defensive lines

| Module | Lines | Issue | Fix Strategy |
|--------|-------|-------|--------------|
| `dxf_editor_core.py` | 50–52 | Work dict path check | Mock `_wjp_work` dict in `st.session_state` |
| `dxf_utils.py` | 55–57 | Exception handler in `entity_summary` | Inject malformed DXF entity to trigger exception |
| `dxf_utils.py` | 71–72 | Exception handler in `delete_entities_by_handle` | Force deletion failure via fake entity |

**Effort**: Low (1-2 hours)  
**Impact**: Completeness, 100% coverage badge  
**Priority**: Optional (95% is already excellent)

**Implementation**:
- Add test fixtures for malformed entities
- Mock session state with `_wjp_work` dict
- Create test cases that trigger exception handlers

---

## 2. Phase-3: Streamlit UI Integration Tests

**Goal**: Confirm Analyzer ↔ Editor pages cooperate correctly in a live session

### Test Scenarios

1. **Session State Handoff**
   - Load Analyzer page → set `st.session_state['analyzed_dxf_path']`
   - Navigate to Editor → confirm SVG render loads
   - Verify document is accessible

2. **Edit Workflow**
   - Perform delete → save → reload
   - Verify edit-log updates
   - Confirm entity count changes persist

3. **UI Controls**
   - Validate sidebar layer toggles
   - Test "Show All / Hide All" controls
   - Verify layer visibility affects SVG rendering

4. **Error Handling**
   - Test with invalid DXF files
   - Test with missing session state
   - Verify error messages are user-friendly

### Tools Required

```bash
pip install streamlit-testing pytest-playwright
```

### Test Structure

```
tests/integration/
├── __init__.py
├── conftest.py          # Streamlit test fixtures
├── test_analyzer_editor_workflow.py
└── test_ui_controls.py
```

### Implementation Approach

- Use headless browser snapshots to assert UI behavior
- Mock Streamlit session state transitions
- Test page navigation and state persistence
- Validate UI component interactions

**Effort**: Medium (4-6 hours)  
**Impact**: High (validates end-to-end workflows)  
**Priority**: High (critical for user experience)

---

## 3. Phase-4: Performance & Stress Tests

**Goal**: Validate performance with large DXF files

### Test Scenarios

1. **Large File Handling**
   - Create synthetic DXFs with 10k+ entities
   - Measure load times
   - Measure render times
   - Measure save times

2. **Memory Usage**
   - Monitor memory consumption
   - Test with files >10MB
   - Verify streaming parser works correctly

3. **Concurrent Operations**
   - Test multiple simultaneous loads
   - Test concurrent edits
   - Verify thread safety

### Test Structure

```
tests/perf/
├── __init__.py
├── conftest.py          # Performance test fixtures
├── test_large_file_handling.py
├── test_memory_usage.py
└── test_concurrent_operations.py
```

### Benchmark Reports

Store results in `/reports/perf_summary.md`:
- Load time vs file size
- Render time vs entity count
- Memory usage trends
- Performance regressions

**Effort**: Medium (3-4 hours)  
**Impact**: Medium (ensures scalability)  
**Priority**: Medium (important for production)

---

## 4. Phase-5: Continuous Integration

**Goal**: Automate testing and enforce quality gates

### GitHub Actions Workflow

**File**: `.github/workflows/tests.yml`

```yaml
name: WJP Analyzer Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      
      - name: Run tests with coverage
        run: |
          pytest tests/ \
            --cov=wjp_analyser \
            --cov-report=xml \
            --cov-report=term \
            --cov-fail-under=90
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
```

### Quality Gates

- **Coverage Threshold**: 90% minimum
- **Test Failure**: Blocks merge
- **Coverage Drop**: Blocks merge
- **Linting**: Optional (can add flake8/black)

### Additional Workflows

1. **Linting** (optional)
   ```yaml
   - name: Run linters
     run: |
       pip install flake8 black mypy
       flake8 src/
       black --check src/
       mypy src/
   ```

2. **Security Scanning** (optional)
   ```yaml
   - name: Run security scan
     run: |
       pip install bandit
       bandit -r src/
   ```

**Effort**: Low (1-2 hours)  
**Impact**: High (automated quality assurance)  
**Priority**: High (essential for team development)

---

## 5. Phase-6: Functional Extension

**Goal**: Add advanced editing features and test them

### New Features

1. **Transformation Tools**
   - Move entities
   - Rotate entities
   - Scale entities
   - Mirror entities

2. **Undo/Redo Stack**
   - Track edit history
   - Implement undo/redo operations
   - Persist history in session state

3. **Re-Analyze Integration**
   - Send modified DXF back to Analyzer
   - Update analysis results
   - Refresh recommendations

### Test Requirements

For each new feature:
- Unit tests for transformation logic
- Integration tests for UI interactions
- Performance tests for large selections
- Edge case tests (invalid inputs, boundary conditions)

### Implementation Order

1. **Move Tool** (simplest)
   - Implement move operation
   - Add UI controls
   - Write tests

2. **Rotate Tool**
   - Implement rotation logic
   - Add UI controls
   - Write tests

3. **Scale Tool**
   - Implement scaling logic
   - Add UI controls
   - Write tests

4. **Mirror Tool**
   - Implement mirroring logic
   - Add UI controls
   - Write tests

5. **Undo/Redo Stack**
   - Design history structure
   - Implement stack operations
   - Add UI controls
   - Write tests

6. **Re-Analyze Integration**
   - Implement re-analysis trigger
   - Update session state
   - Refresh UI
   - Write tests

**Effort**: High (20-30 hours)  
**Impact**: High (significantly improves editor functionality)  
**Priority**: Medium (enhancement, not critical)

---

## Implementation Priority

### Immediate (This Week)
1. ✅ **Phase-2 Complete** - 95% coverage achieved
2. **Optional**: 100% coverage push (if desired)
3. **Phase-5**: Set up CI/CD (high value, low effort)

### Short Term (Next 2 Weeks)
4. **Phase-3**: Streamlit UI Integration Tests
5. **Phase-4**: Performance & Stress Tests

### Medium Term (Next Month)
6. **Phase-6**: Functional Extensions (start with Move tool)

---

## Success Metrics

### Phase-3 (Integration Tests)
- ✅ All Analyzer ↔ Editor workflows tested
- ✅ UI controls validated
- ✅ Error handling verified

### Phase-4 (Performance)
- ✅ Large files (>10MB) handled correctly
- ✅ Performance benchmarks documented
- ✅ No memory leaks detected

### Phase-5 (CI/CD)
- ✅ Automated tests on every PR
- ✅ Coverage threshold enforced
- ✅ Test results visible in PRs

### Phase-6 (Extensions)
- ✅ All transformation tools implemented
- ✅ Undo/redo working
- ✅ Re-analyze integration complete

---

## Notes

- **100% Coverage**: Optional but nice to have. 95% is already excellent.
- **CI/CD**: Should be prioritized early (high value, low effort)
- **Integration Tests**: Critical for validating user workflows
- **Performance Tests**: Important for production readiness
- **Extensions**: Can be done incrementally

---

**Last Updated**: 2025-01-XX  
**Current Coverage**: 95%  
**Current Tests**: 44 passing

