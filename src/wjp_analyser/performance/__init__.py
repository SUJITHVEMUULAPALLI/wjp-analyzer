"""
Performance Optimization Module
================================

Utilities for optimizing performance of large DXF files and geometry operations.
"""

from .streaming_parser import (
    StreamingDXFParser,
    parse_with_early_simplification,
    normalize_entities,
    compute_file_hash,
)

from .cache_manager import (
    CacheManager,
    memoize,
    get_cache_manager,
    clear_cache,
)

from .memory_optimizer import (
    optimize_coordinates,
    filter_tiny_segments,
    paginate_geometry,
    use_strtree_for_queries,
    create_spatial_index,
    optimize_polygon_set,
    estimate_memory_usage,
)

__all__ = [
    # Streaming parser
    "StreamingDXFParser",
    "parse_with_early_simplification",
    "normalize_entities",
    "compute_file_hash",
    # Cache manager
    "CacheManager",
    "memoize",
    "get_cache_manager",
    "clear_cache",
    # Memory optimizer
    "optimize_coordinates",
    "filter_tiny_segments",
    "paginate_geometry",
    "use_strtree_for_queries",
    "create_spatial_index",
    "optimize_polygon_set",
    "estimate_memory_usage",
]





