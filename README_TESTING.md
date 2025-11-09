# WJP Analyzer - Testing Guide

## Current Status

✅ **Phase-1 Complete**: 27 tests, 55% coverage  
✅ **Phase-2 Complete**: 44 tests, 95% coverage

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/ --cov=wjp_analyser --cov-report=term-missing
```

### Specific Test Suites
```bash
# Unit tests only
pytest tests/unit/ -v

# DXF Editor module tests
pytest tests/unit/web/ -v

# Integration tests (when available)
pytest tests/integration/ -v

# Performance tests (when available)
pytest tests/perf/ -v
```

### Coverage Reports
```bash
# Terminal report
pytest tests/ --cov=wjp_analyser --cov-report=term

# HTML report
pytest tests/ --cov=wjp_analyser --cov-report=html
open htmlcov/index.html
```

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── web/                # DXF Editor module tests
│   │   ├── test_dxf_utils.py
│   │   ├── test_dxf_renderer.py
│   │   └── test_dxf_editor_core.py
│   ├── analysis/           # Analysis module tests
│   ├── services/           # Service layer tests
│   └── ...
├── integration/            # Integration tests (Phase-3)
│   ├── test_analyzer_editor_workflow.py
│   └── test_ui_controls.py
└── perf/                   # Performance tests (Phase-4)
    ├── test_large_file_handling.py
    └── test_memory_usage.py
```

## Coverage Goals

- **Current**: 95% (DXF Editor modules)
- **Target**: 90% (overall project)
- **Stretch**: 100% (DXF Editor modules)

## CI/CD

Tests run automatically on:
- Every push to `main` or `develop`
- Every pull request
- Coverage threshold: 90% minimum

See `.github/workflows/tests.yml` for details.

## Writing New Tests

### Unit Test Template
```python
import pytest
from wjp_analyser.module import function

def test_function_success():
    """Test successful execution."""
    result = function(input)
    assert result == expected

def test_function_error():
    """Test error handling."""
    with pytest.raises(ExpectedError):
        function(invalid_input)
```

### Integration Test Template
```python
import pytest
from wjp_analyser.web.modules import dxf_editor_core as core

def test_workflow(mock_streamlit_session, sample_dxf):
    """Test complete workflow."""
    # Setup
    mock_streamlit_session["analyzed_dxf_path"] = sample_dxf
    
    # Execute
    doc, path = core.load_from_analyzer_or_upload(mock_streamlit_session, None)
    
    # Verify
    assert doc is not None
    assert path is not None
```

## Test Fixtures

### Available Fixtures
- `sample_dxf`: Small test DXF file (from `tests/conftest.py`)
- `temp_dir`: Temporary directory for test outputs
- `large_dxf_10k_entities`: Large DXF for performance testing
- `mock_streamlit_session`: Mock Streamlit session state

## Best Practices

1. **Test Naming**: Use descriptive names (`test_function_scenario`)
2. **Arrange-Act-Assert**: Structure tests clearly
3. **Isolation**: Each test should be independent
4. **Fixtures**: Use fixtures for common setup
5. **Coverage**: Aim for 100% coverage of new code
6. **Documentation**: Add docstrings to test functions

## Troubleshooting

### Streamlit Warnings
If you see `missing ScriptRunContext` warnings, these are expected when running tests outside Streamlit. They can be ignored.

### Import Errors
Make sure `src/` is in your Python path. The `conftest.py` files handle this automatically.

### Coverage Issues
- Use `--cov-report=term-missing` to see which lines are missing
- Focus on testing public APIs first
- Exception handlers are lower priority

