"""
Memory usage tests for DXF file handling.

Tests memory consumption with large files and verifies no memory leaks.
"""
from __future__ import annotations

import sys
import pytest
import tracemalloc

from wjp_analyser.web.modules import dxf_utils as du
from wjp_analyser.web.modules import dxf_renderer as dr


def test_memory_usage_load_large_file(large_dxf_10k_entities):
    """Test memory usage when loading large DXF files."""
    tracemalloc.start()
    
    # Get baseline memory
    snapshot1 = tracemalloc.take_snapshot()
    
    # Load document
    doc = du.load_document(large_dxf_10k_entities)
    
    # Get memory after load
    snapshot2 = tracemalloc.take_snapshot()
    
    # Calculate memory increase
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats)
    memory_mb = total_memory / (1024 * 1024)
    
    tracemalloc.stop()
    
    assert doc is not None
    
    # Memory assertion: should use reasonable memory (< 100 MB for 10k entities)
    assert memory_mb < 100.0, f"Memory usage {memory_mb:.2f}MB exceeds 100MB threshold"
    
    # Log memory metric
    print(f"\n[MEMORY] Load 10k entities: {memory_mb:.2f} MB")


def test_memory_usage_render_large_file(large_dxf_10k_entities):
    """Test memory usage when rendering large DXF files to SVG."""
    doc = du.load_document(large_dxf_10k_entities)
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Render SVG
    svg_text = dr.render_svg(doc)
    
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats)
    memory_mb = total_memory / (1024 * 1024)
    
    tracemalloc.stop()
    
    assert svg_text is not None
    
    # Memory assertion: rendering should use reasonable memory (< 50 MB)
    assert memory_mb < 50.0, f"Render memory usage {memory_mb:.2f}MB exceeds 50MB threshold"
    
    # Log memory metric
    print(f"\n[MEMORY] Render 10k entities: {memory_mb:.2f} MB")


def test_memory_usage_entity_summary(large_dxf_10k_entities):
    """Test memory usage when generating entity summary."""
    doc = du.load_document(large_dxf_10k_entities)
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Generate summary
    summary = du.entity_summary(doc)
    
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats)
    memory_mb = total_memory / (1024 * 1024)
    
    tracemalloc.stop()
    
    assert len(summary) >= 10000
    
    # Memory assertion: summary should use reasonable memory (< 10 MB)
    assert memory_mb < 10.0, f"Summary memory usage {memory_mb:.2f}MB exceeds 10MB threshold"
    
    # Log memory metric
    print(f"\n[MEMORY] Entity summary 10k entities: {memory_mb:.2f} MB")


def test_no_memory_leak_multiple_operations(large_dxf_10k_entities):
    """Test that multiple operations don't cause memory leaks."""
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Perform multiple operations
    for i in range(5):
        doc = du.load_document(large_dxf_10k_entities)
        summary = du.entity_summary(doc)
        svg_text = dr.render_svg(doc)
        del doc, summary, svg_text  # Explicit cleanup
    
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats)
    memory_mb = total_memory / (1024 * 1024)
    
    tracemalloc.stop()
    
    # Memory assertion: should not leak significantly (< 200 MB for 5 iterations)
    assert memory_mb < 200.0, f"Memory leak detected: {memory_mb:.2f}MB after 5 iterations"
    
    # Log memory metric
    print(f"\n[MEMORY] 5 iterations (load+summary+render): {memory_mb:.2f} MB")


@pytest.mark.slow
def test_very_large_file_memory(very_large_dxf_50k_entities):
    """Test memory usage with very large files (50k entities)."""
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Load very large file
    doc = du.load_document(very_large_dxf_50k_entities)
    
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats)
    memory_mb = total_memory / (1024 * 1024)
    
    tracemalloc.stop()
    
    assert doc is not None
    
    # Memory assertion: should use reasonable memory (< 500 MB for 50k entities)
    assert memory_mb < 500.0, f"Memory usage {memory_mb:.2f}MB exceeds 500MB threshold"
    
    # Log memory metric
    print(f"\n[MEMORY] Load 50k entities: {memory_mb:.2f} MB")

