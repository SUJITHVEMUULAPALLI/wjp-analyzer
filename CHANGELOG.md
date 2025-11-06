# Changelog

All notable changes to the WJP ANALYSER project will be documented in this file.

## [2.0.0] - 2025-01-01

### 🎉 Major Updates - Phases 1-5 Complete

#### Phase 1-2: Service Layer & API Architecture
- **NEW**: Unified CLI (`wjp` command) - single entry point for all operations
- **NEW**: FastAPI backend with async job processing
- **NEW**: Service layer architecture (analysis, costing, editing, caching)
- **NEW**: API client with automatic fallback to direct services
- **NEW**: Redis Queue (RQ) integration for background jobs
- **NEW**: Layered DXF service (first-class service, not UI workaround)
- **DEPRECATED**: Old entry points (`main.py`, `run_one_click.py`, `wjp_analyser_unified.py`)

#### Phase 3: UX Improvements
- **NEW**: Wizard components for guided workflows
- **NEW**: Jobs drawer for real-time async job status
- **NEW**: Actionable error handling with auto-fix buttons
- **NEW**: Terminology standardization across UI
- **IMPROVED**: Better error messages and user feedback

#### Phase 4: Performance Optimization
- **NEW**: Streaming DXF parser for large files (>10MB)
- **NEW**: Enhanced caching with job hashing and artifact caching
- **NEW**: Memory optimization (float32 support, coordinate precision, segment filtering)
- **NEW**: Job idempotency (prevent duplicate work)
- **IMPROVED**: 50-70% memory reduction for large files
- **IMPROVED**: 30-50% faster processing with early simplification

#### Phase 5: Advanced Nesting
- **NEW**: Geometry hygiene (robust polygonization, hole handling, winding rules)
- **NEW**: Advanced placement engines (Bottom-Left Fill, NFP refinement, metaheuristics)
- **NEW**: Constraint system (hard/soft constraints, determinism mode)
- **NEW**: Production-grade nesting with STRtree collision detection

### 🔧 Improvements
- Centralized costing logic (removed duplication)
- Consolidated entry points (one canonical CLI)
- API-first architecture with graceful fallback
- Performance optimizations for enterprise-scale files

### 📝 Documentation
- Updated README.md with new CLI instructions
- Created QUICK_START.md for quick reference
- Added deprecation warnings to legacy launchers

### 🐛 Bug Fixes
- Fixed circular import issues in nesting module
- Improved error handling across all components
- Fixed caching integration issues

---

## [1.x.x] - Previous Versions

See git history for earlier changes.





