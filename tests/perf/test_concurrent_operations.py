"""
Concurrent operations tests for DXF file handling.

Tests thread safety and concurrent access patterns.
"""
from __future__ import annotations

import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from wjp_analyser.web.modules import dxf_utils as du
from wjp_analyser.web.modules import dxf_renderer as dr


def test_concurrent_loads(large_dxf_10k_entities):
    """Test concurrent loading of the same DXF file."""
    def load_dxf():
        doc = du.load_document(large_dxf_10k_entities)
        entity_count = len(list(doc.modelspace()))
        return entity_count
    
    # Run 5 concurrent loads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(load_dxf) for _ in range(5)]
        results = [f.result() for f in as_completed(futures)]
    
    # All should succeed and return same count
    assert all(count >= 10000 for count in results)
    assert len(set(results)) == 1  # All should be identical
    
    print(f"\n[CONCURRENT] 5 concurrent loads: all succeeded ({results[0]} entities each)")


def test_concurrent_renders(large_dxf_10k_entities):
    """Test concurrent rendering of the same DXF file."""
    doc = du.load_document(large_dxf_10k_entities)
    
    def render_dxf():
        svg_text = dr.render_svg(doc)
        return len(svg_text)
    
    # Run 3 concurrent renders
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(render_dxf) for _ in range(3)]
        results = [f.result() for f in as_completed(futures)]
    
    # All should succeed
    assert all(size > 0 for size in results)
    
    print(f"\n[CONCURRENT] 3 concurrent renders: all succeeded ({len(results)} results)")


def test_concurrent_summaries(large_dxf_10k_entities):
    """Test concurrent entity summary generation."""
    doc = du.load_document(large_dxf_10k_entities)
    
    def generate_summary():
        summary = du.entity_summary(doc)
        return len(summary)
    
    # Run 5 concurrent summaries
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_summary) for _ in range(5)]
        results = [f.result() for f in as_completed(futures)]
    
    # All should succeed and return same count
    assert all(count >= 10000 for count in results)
    assert len(set(results)) == 1  # All should be identical
    
    print(f"\n[CONCURRENT] 5 concurrent summaries: all succeeded ({results[0]} entities each)")


def test_concurrent_saves(large_dxf_10k_entities, tmp_path):
    """Test concurrent saving of DXF files."""
    doc = du.load_document(large_dxf_10k_entities)
    
    def save_dxf(index):
        output_path = tmp_path / f"concurrent_save_{index}.dxf"
        saved_path = du.save_document(doc, str(output_path))
        return Path(saved_path).exists()
    
    # Run 3 concurrent saves
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(save_dxf, i) for i in range(3)]
        results = [f.result() for f in as_completed(futures)]
    
    # All should succeed
    assert all(results)
    
    # Verify all files exist
    for i in range(3):
        assert (tmp_path / f"concurrent_save_{i}.dxf").exists()
    
    print(f"\n[CONCURRENT] 3 concurrent saves: all succeeded")


@pytest.mark.slow
def test_mixed_concurrent_operations(large_dxf_10k_entities, tmp_path):
    """Test mixed concurrent operations (load, render, save, summary)."""
    def load_operation():
        doc = du.load_document(large_dxf_10k_entities)
        return len(list(doc.modelspace()))
    
    def render_operation():
        doc = du.load_document(large_dxf_10k_entities)
        svg_text = dr.render_svg(doc)
        return len(svg_text)
    
    def summary_operation():
        doc = du.load_document(large_dxf_10k_entities)
        summary = du.entity_summary(doc)
        return len(summary)
    
    def save_operation(index):
        doc = du.load_document(large_dxf_10k_entities)
        output_path = tmp_path / f"mixed_concurrent_{index}.dxf"
        du.save_document(doc, str(output_path))
        return Path(output_path).exists()
    
    # Run mixed concurrent operations
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        futures.append(executor.submit(load_operation))
        futures.append(executor.submit(render_operation))
        futures.append(executor.submit(summary_operation))
        futures.extend([executor.submit(save_operation, i) for i in range(3)])
        
        results = [f.result() for f in as_completed(futures)]
    
    # All should succeed
    assert all(results)
    
    print(f"\n[CONCURRENT] Mixed operations (8 concurrent): all succeeded")

