"""
Performance tests for large DXF file handling.

Tests load, render, and save performance with large files (10k+ entities).
"""
from __future__ import annotations

import time
import pytest
from pathlib import Path

from wjp_analyser.web.modules import dxf_utils as du
from wjp_analyser.web.modules import dxf_renderer as dr


def test_load_large_dxf_performance(large_dxf_10k_entities):
    """Test loading performance for large DXF files (10k entities)."""
    start_time = time.time()
    doc = du.load_document(large_dxf_10k_entities)
    load_time = time.time() - start_time
    
    assert doc is not None
    entity_count = len(list(doc.modelspace()))
    assert entity_count >= 10000
    
    # Performance assertion: should load in reasonable time (< 5 seconds)
    assert load_time < 5.0, f"Load time {load_time:.2f}s exceeds 5s threshold"
    
    # Log performance metric
    print(f"\n[PERF] Load 10k entities: {load_time:.3f}s ({entity_count} entities)")


def test_render_large_dxf_performance(large_dxf_10k_entities):
    """Test SVG rendering performance for large DXF files."""
    doc = du.load_document(large_dxf_10k_entities)
    
    start_time = time.time()
    svg_text = dr.render_svg(doc)
    render_time = time.time() - start_time
    
    assert svg_text is not None
    assert len(svg_text) > 0
    assert "<svg" in svg_text
    
    # Performance assertion: should render in reasonable time (< 10 seconds)
    assert render_time < 10.0, f"Render time {render_time:.2f}s exceeds 10s threshold"
    
    # Log performance metric
    print(f"\n[PERF] Render 10k entities to SVG: {render_time:.3f}s ({len(svg_text)} bytes)")


def test_save_large_dxf_performance(large_dxf_10k_entities, tmp_path):
    """Test saving performance for large DXF files."""
    doc = du.load_document(large_dxf_10k_entities)
    output_path = tmp_path / "large_saved.dxf"
    
    start_time = time.time()
    saved_path = du.save_document(doc, str(output_path))
    save_time = time.time() - start_time
    
    assert Path(saved_path).exists()
    
    # Performance assertion: should save in reasonable time (< 3 seconds)
    assert save_time < 3.0, f"Save time {save_time:.2f}s exceeds 3s threshold"
    
    # Log performance metric
    file_size = Path(saved_path).stat().st_size
    print(f"\n[PERF] Save 10k entities: {save_time:.3f}s ({file_size} bytes)")


def test_entity_summary_large_dxf_performance(large_dxf_10k_entities):
    """Test entity summary generation performance for large files."""
    doc = du.load_document(large_dxf_10k_entities)
    
    start_time = time.time()
    summary = du.entity_summary(doc)
    summary_time = time.time() - start_time
    
    assert len(summary) >= 10000
    
    # Performance assertion: should generate summary in reasonable time (< 2 seconds)
    assert summary_time < 2.0, f"Summary time {summary_time:.2f}s exceeds 2s threshold"
    
    # Log performance metric
    print(f"\n[PERF] Entity summary 10k entities: {summary_time:.3f}s ({len(summary)} entries)")


def test_very_large_dxf_handling(very_large_dxf_50k_entities):
    """Test handling of very large DXF files (50k entities)."""
    start_time = time.time()
    doc = du.load_document(very_large_dxf_50k_entities)
    load_time = time.time() - start_time
    
    assert doc is not None
    entity_count = len(list(doc.modelspace()))
    assert entity_count >= 50000
    
    # Performance assertion: should load in reasonable time (< 15 seconds)
    assert load_time < 15.0, f"Load time {load_time:.2f}s exceeds 15s threshold"
    
    # Log performance metric
    print(f"\n[PERF] Load 50k entities: {load_time:.3f}s ({entity_count} entities)")


def test_render_very_large_dxf_performance(very_large_dxf_50k_entities):
    """Test SVG rendering performance for very large DXF files."""
    doc = du.load_document(very_large_dxf_50k_entities)
    
    start_time = time.time()
    svg_text = dr.render_svg(doc)
    render_time = time.time() - start_time
    
    assert svg_text is not None
    assert len(svg_text) > 0
    
    # Performance assertion: should render in reasonable time (< 30 seconds)
    assert render_time < 30.0, f"Render time {render_time:.2f}s exceeds 30s threshold"
    
    # Log performance metric
    print(f"\n[PERF] Render 50k entities to SVG: {render_time:.3f}s ({len(svg_text)} bytes)")


def test_delete_performance_large_selection(large_dxf_10k_entities):
    """Test deletion performance with large entity selections."""
    doc = du.load_document(large_dxf_10k_entities)
    
    # Get handles for first 1000 entities
    summary = du.entity_summary(doc)
    handles_to_delete = [e["handle"] for e in summary[:1000]]
    
    start_time = time.time()
    deleted_count = du.delete_entities_by_handle(doc, handles_to_delete)
    delete_time = time.time() - start_time
    
    assert deleted_count == 1000
    
    # Performance assertion: should delete in reasonable time (< 1 second)
    assert delete_time < 1.0, f"Delete time {delete_time:.2f}s exceeds 1s threshold"
    
    # Log performance metric
    print(f"\n[PERF] Delete 1000 entities: {delete_time:.3f}s")


@pytest.mark.slow
def test_end_to_end_large_file_workflow(large_dxf_10k_entities, tmp_path):
    """End-to-end performance test: load → modify → save → reload."""
    # 1. Load
    load_start = time.time()
    doc = du.load_document(large_dxf_10k_entities)
    load_time = time.time() - load_start
    
    # 2. Generate summary
    summary_start = time.time()
    summary = du.entity_summary(doc)
    summary_time = time.time() - summary_start
    
    # 3. Delete some entities
    delete_start = time.time()
    handles_to_delete = [e["handle"] for e in summary[:100]]
    du.delete_entities_by_handle(doc, handles_to_delete)
    delete_time = time.time() - delete_start
    
    # 4. Save
    save_start = time.time()
    output_path = tmp_path / "workflow_test.dxf"
    du.save_document(doc, str(output_path))
    save_time = time.time() - save_start
    
    # 5. Reload
    reload_start = time.time()
    reloaded_doc = du.load_document(str(output_path))
    reload_time = time.time() - reload_start
    
    # Verify
    assert len(list(reloaded_doc.modelspace())) == len(list(doc.modelspace()))
    
    # Performance assertions
    total_time = load_time + summary_time + delete_time + save_time + reload_time
    assert total_time < 10.0, f"Total workflow time {total_time:.2f}s exceeds 10s threshold"
    
    # Log performance metrics
    print(f"\n[PERF] End-to-end workflow:")
    print(f"  Load: {load_time:.3f}s")
    print(f"  Summary: {summary_time:.3f}s")
    print(f"  Delete: {delete_time:.3f}s")
    print(f"  Save: {save_time:.3f}s")
    print(f"  Reload: {reload_time:.3f}s")
    print(f"  Total: {total_time:.3f}s")

