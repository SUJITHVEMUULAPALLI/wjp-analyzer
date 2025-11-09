# WJP Analyzer

A comprehensive DXF analysis and editing tool for waterjet cutting operations.

## Features

- 🔍 **DXF Analysis**: Analyze DXF files for waterjet cutting
- ✏️ **DXF Editor**: Visualize, edit, and modify DXF files
- 🤖 **AI-Powered Recommendations**: Get intelligent suggestions for DXF optimization
- 📊 **Performance Metrics**: Track cutting length, pierces, and costs
- 🎯 **Nesting**: Optimize material usage with nesting algorithms
- 📐 **G-code Generation**: Generate G-code for waterjet machines

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer.git
cd wjp-analyzer

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Streamlit web interface
python run.py

# Or use the one-click launcher
python run_one_click.py
```

The application will open in your browser at `http://127.0.0.1:8501`

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=wjp_analyser --cov-report=term-missing
```

### Run Specific Test Suites

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/perf/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Test Coverage

- **Total Tests**: 66+ passing
- **Unit Tests**: 51
- **Integration Tests**: 8
- **Performance Tests**: 10+
- **Coverage**: 100% for DXF Editor modules, >90% overall

## CI/CD

The project uses GitHub Actions for automated testing:

- ✅ Runs on every push/PR
- ✅ Tests on Python 3.10, 3.11, 3.12
- ✅ Enforces 90% coverage threshold
- ✅ Uploads coverage to Codecov

**Status**: [![Tests](https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer/workflows/WJP%20Analyzer%20Tests/badge.svg)](https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer/actions)

## Documentation

- **Testing Guide**: [README_TESTING.md](README_TESTING.md)
- **CI/CD Monitoring**: [docs/MONITOR_CI_CD.md](docs/MONITOR_CI_CD.md)
- **Codecov Setup**: [docs/CODECOV_SETUP.md](docs/CODECOV_SETUP.md)
- **Performance Tests**: [reports/perf_summary.md](reports/perf_summary.md)
- **Roadmap**: [POST_PHASE2_ROADMAP.md](POST_PHASE2_ROADMAP.md)

## Project Structure

```
wjp-analyzer/
├── src/wjp_analyser/      # Main source code
│   ├── web/               # Streamlit web interface
│   ├── dxf_editor/        # DXF editing utilities
│   ├── analysis/          # DXF analysis engine
│   └── nesting/           # Nesting algorithms
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── perf/              # Performance tests
├── .github/workflows/     # CI/CD configuration
└── docs/                  # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass and coverage ≥90%
6. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Links

- **Repository**: https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer
- **Issues**: https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer/issues
- **Actions**: https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer/actions
