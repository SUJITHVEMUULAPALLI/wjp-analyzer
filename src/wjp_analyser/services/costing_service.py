from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

# Import performance optimizations
try:
    from ..performance import get_cache_manager, memoize
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False


def estimate_cost(dxf_path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Estimate cost from DXF file path.
    
    This is the primary costing function that works with DXF files.
    It analyzes the DXF and calculates costs based on cutting length,
    pierces, material, and machine time.
    
    Args:
        dxf_path: Path to DXF file
        overrides: Optional dict to override default parameters:
            - rate_per_m: Rate per meter (default: 825.0)
            - machine_rate_per_min: Machine rate per minute (default: 20.0)
            - pierce_cost: Cost per pierce (default: 1.5)
            - setup_cost: Setup cost (default: 15.0)
            - pierce_ms: Pierce time in milliseconds (default: 500)
            - feed: Feed rate mm/min (default: 1200.0)
            
    Returns:
        Dict with cost breakdown and metrics
    """
    # Check cache if performance module available (Phase 4)
    if PERFORMANCE_AVAILABLE:
        cache = get_cache_manager(".cache")
        cache_key = f"cost_{dxf_path}_{hash(str(sorted((overrides or {}).items())))}"
        cached = cache.get(cache_key)
        if cached:
            return cached
    
    from wjp_analyser.web.api_utils import calculate_costs_from_dxf

    # Extract overrides for calculate_costs_from_dxf
    params = {}
    if overrides:
        param_mapping = {
            "rate_per_m": "rate_per_m",
            "machine_rate_per_min": "machine_rate_per_min",
            "pierce_cost": "pierce_cost",
            "setup_cost": "setup_cost",
            "pierce_ms": "pierce_ms",
            "feed": "feed",
            "rate_per_meter": "rate_per_m",  # Alias
        }
        for key, api_key in param_mapping.items():
            if key in overrides:
                params[api_key] = overrides[key]
    
    base = calculate_costs_from_dxf(dxf_path, **params) if params else calculate_costs_from_dxf(dxf_path)
    
    # Apply any remaining overrides that aren't API params
    if overrides:
        for key, value in overrides.items():
            if key not in params and key not in base:
                base[key] = value
    
    # Cache result if performance module available (Phase 4)
    if PERFORMANCE_AVAILABLE:
        cache = get_cache_manager(".cache")
        cache_key = f"cost_{dxf_path}_{hash(str(sorted((overrides or {}).items())))}"
        cache.set(cache_key, base)
    
    return base


def estimate_cost_from_toolpath(
    toolpath: Iterable[List[Tuple[float, float]]],
    rate_per_m: float = 825.0,
    pierce_cost: float = 1.5,
    setup_cost: float = 0.0,
) -> Dict[str, Any]:
    """
    Estimate cost from already-computed toolpath.
    
    This function works with toolpath polylines that have already been
    generated from analysis. Use this when you have a toolpath but not
    the original DXF file.
    
    Args:
        toolpath: Iterable of polylines (each polyline is list of (x, y) tuples)
        rate_per_m: Rate per meter (default: 825.0)
        pierce_cost: Cost per pierce (default: 1.5)
        setup_cost: Setup cost (default: 0.0)
        
    Returns:
        Dict with cost breakdown:
            - cutting_length_mm: Total cutting length in mm
            - cutting_length_m: Total cutting length in meters
            - pierce_count: Number of pierces
            - cutting_cost: Cost based on cutting length
            - pierce_cost_total: Total pierce cost
            - setup_cost: Setup cost
            - total_cost: Total cost
    """
    import math
    
    total_length = 0.0
    pierce_count = 0
    
    for poly in toolpath:
        pierce_count += 1
        length = 0.0
        for idx in range(len(poly) - 1):
            x1, y1 = poly[idx]
            x2, y2 = poly[idx + 1]
            length += math.dist((x1, y1), (x2, y2))
        total_length += length
    
    cutting_length_m = total_length / 1000.0
    cutting_cost = cutting_length_m * rate_per_m
    pierce_cost_total = pierce_count * pierce_cost
    total_cost = cutting_cost + pierce_cost_total + setup_cost
    
    return {
        "cutting_length_mm": round(total_length, 2),
        "cutting_length_m": round(cutting_length_m, 2),
        "pierce_count": pierce_count,
        "cutting_cost": round(cutting_cost, 2),
        "pierce_cost": round(pierce_cost_total, 2),
        "setup_cost": round(setup_cost, 2),
        "total_cost": round(total_cost, 2),
    }


# Legacy alias for backward compatibility
def calculate_cost(toolpath: Iterable[List[Tuple[float, float]]], rate_per_mtr: float, pierce_cost: float) -> dict:
    """
    Legacy function for backward compatibility.
    
    Use estimate_cost_from_toolpath() instead.
    """
    result = estimate_cost_from_toolpath(toolpath, rate_per_m=rate_per_mtr, pierce_cost=pierce_cost)
    # Return in old format for compatibility
    return {
        "cutting_length_mm": result["cutting_length_mm"],
        "pierce_count": result["pierce_count"],
        "cutting_cost": result["cutting_cost"],
        "total_cost": result["total_cost"],
    }

