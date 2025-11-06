from __future__ import annotations

from typing import Any, Dict
from pathlib import Path


def run_analysis(dxf_path: str, out_dir: str | None = None, args_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Thin wrapper around core analyze_dxf to provide a stable DTO for UI layers.
    
    Now includes performance optimizations:
    - Caching for repeated analyses
    - Streaming parser for large files
    - Memory optimization for large polygon sets
    """
    try:
        from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf
    except ImportError as e:
        # If dxf_analyzer can't be imported (e.g., due to SQLAlchemy dependency), provide helpful error
        error_msg = str(e)
        if "sqlalchemy" in error_msg.lower() or "No module named 'sqlalchemy'" in error_msg:
            raise ImportError(
                f"DXF analyzer requires SQLAlchemy: {error_msg}. "
                "Install it with: pip install sqlalchemy"
            ) from e
        raise
    
    try:
        from wjp_analyser.performance import get_cache_manager, compute_file_hash
        from wjp_analyser.analysis.cache_utils import build_cache_key
    except ImportError:
        # Fallback if performance module not available
        get_cache_manager = None
        compute_file_hash = None
    
    # Setup cache (if available)
    cache = None
    cache_key = None
    if get_cache_manager:
        try:
            cache_dir = str(Path(out_dir or "out") / ".cache")
            cache = get_cache_manager(cache_dir)
            
            # Generate cache key from file hash and parameters
            if compute_file_hash:
                file_hash = compute_file_hash(dxf_path)
                overrides_dict = args_overrides or {}
                cache_key = build_cache_key(file_hash, overrides_dict)
                
                # Check cache first
                cached_result = cache.get(cache_key)
                if cached_result:
                    return cached_result
        except Exception:
            # Cache unavailable, continue without it
            cache = None
    
    # Prepare arguments
    args = AnalyzeArgs(out=out_dir or "out")
    if args_overrides:
        for k, v in (args_overrides or {}).items():
            setattr(args, k, v)
    
    # Check file size for streaming parser
    file_size_mb = Path(dxf_path).stat().st_size / (1024 * 1024)
    use_streaming = file_size_mb > 10  # Use streaming for files > 10MB
    
    if use_streaming:
        # For large files, we'll enable streaming in the analyzer
        args.streaming_mode = True
    
    # Run analysis
    report = analyze_dxf(dxf_path, args)
    
    # Cache result (if cache available)
    if cache and compute_file_hash and cache_key:
        try:
            cache.set(cache_key, report)
        except Exception:
            # Cache write failed, continue anyway
            pass
    
    return report


def summarize_for_quote(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    total_length_mm = float(metrics.get("length_internal_mm", 0.0)) + float(metrics.get("length_outer_mm", 0.0))
    length_m = total_length_mm / 1000.0
    pierces = int(metrics.get("pierce_count", metrics.get("pierces", 0)))
    return {
        "length_m": length_m,
        "pierces": pierces,
    }
