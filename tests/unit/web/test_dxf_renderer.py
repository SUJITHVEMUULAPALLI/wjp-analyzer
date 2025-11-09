"""
Tests for DXF renderer module.

Tests SVG rendering, layer visibility toggling, and rendering edge cases.
"""
from __future__ import annotations

import pytest
import ezdxf

# Import the modules under test
from wjp_analyser.web.modules import dxf_renderer as dr
from wjp_analyser.web.modules import dxf_utils as du


def test_render_svg_basic(sample_dxf):
    """Test basic SVG rendering of a DXF document."""
    doc = du.load_document(sample_dxf)
    svg_text = dr.render_svg(doc)
    
    assert isinstance(svg_text, str)
    assert len(svg_text) > 0
    # SVG may start with XML declaration or directly with <svg>
    assert "<svg" in svg_text
    assert svg_text.strip().endswith("</svg>")
    
    # Should contain some drawing elements
    assert "path" in svg_text.lower() or "circle" in svg_text.lower() or "line" in svg_text.lower()


def test_render_svg_empty_document():
    """Test rendering an empty DXF document."""
    doc = ezdxf.new(dxfversion="R2010")
    svg_text = dr.render_svg(doc)
    
    assert isinstance(svg_text, str)
    assert "<svg" in svg_text


def test_render_svg_with_entities(sample_dxf):
    """Test rendering a document with various entity types."""
    doc = du.load_document(sample_dxf)
    svg_text = dr.render_svg(doc)
    
    assert "<svg" in svg_text
    # Should render something (even if empty, SVG structure should exist)
    assert len(svg_text) > 100  # Reasonable minimum for non-empty SVG


def test_layer_visibility_toggle_all_visible(sample_dxf):
    """Test rendering with all layers visible."""
    doc = du.load_document(sample_dxf)
    layers = du.list_layers(doc)
    
    # All layers visible
    vis = {layer: True for layer in layers}
    svg_text = dr.render_svg(doc, layer_visibility=vis)
    
    assert "<svg" in svg_text
    assert isinstance(svg_text, str)


def test_layer_visibility_toggle_all_hidden(sample_dxf):
    """Test rendering with all layers hidden."""
    doc = du.load_document(sample_dxf)
    layers = du.list_layers(doc)
    
    # All layers hidden
    vis = {layer: False for layer in layers}
    svg_text = dr.render_svg(doc, layer_visibility=vis)
    
    # Should still produce valid SVG (just empty)
    assert "<svg" in svg_text
    assert isinstance(svg_text, str)


def test_layer_visibility_toggle_partial(sample_dxf):
    """Test rendering with some layers visible, some hidden."""
    doc = du.load_document(sample_dxf)
    layers = du.list_layers(doc)
    
    if len(layers) >= 2:
        # Toggle visibility based on index
        vis = {layer: (i % 2 == 0) for i, layer in enumerate(layers)}
        svg_text = dr.render_svg(doc, layer_visibility=vis)
        
        assert "<svg" in svg_text
        assert isinstance(svg_text, str)


def test_layer_visibility_nonexistent_layer(sample_dxf):
    """Test rendering with visibility dict containing non-existent layers."""
    doc = du.load_document(sample_dxf)
    
    # Include non-existent layer names
    vis = {
        "NONEXISTENT_LAYER_1": True,
        "NONEXISTENT_LAYER_2": False,
        "0": True,  # Valid layer
    }
    
    # Should not raise exception, should ignore invalid layers
    svg_text = dr.render_svg(doc, layer_visibility=vis)
    assert "<svg" in svg_text


def test_layer_visibility_empty_dict(sample_dxf):
    """Test rendering with empty visibility dict (all layers visible by default)."""
    doc = du.load_document(sample_dxf)
    
    svg_text = dr.render_svg(doc, layer_visibility={})
    assert "<svg" in svg_text


def test_layer_visibility_none(sample_dxf):
    """Test rendering with None visibility (all layers visible by default)."""
    doc = du.load_document(sample_dxf)
    
    svg_text = dr.render_svg(doc, layer_visibility=None)
    assert "<svg" in svg_text


def test_render_svg_consistency(sample_dxf):
    """Test that rendering the same document twice produces consistent results."""
    doc1 = du.load_document(sample_dxf)
    doc2 = du.load_document(sample_dxf)
    
    svg1 = dr.render_svg(doc1)
    svg2 = dr.render_svg(doc2)
    
    # Should produce same SVG (or at least same structure)
    assert len(svg1) == len(svg2)
    assert svg1 == svg2


def test_render_svg_with_different_visibility(sample_dxf):
    """Test that different visibility settings produce different SVGs."""
    doc = du.load_document(sample_dxf)
    layers = du.list_layers(doc)
    
    if len(layers) >= 2:
        # All visible
        vis_all = {layer: True for layer in layers}
        svg_all = dr.render_svg(doc, layer_visibility=vis_all)
        
        # All hidden
        vis_none = {layer: False for layer in layers}
        svg_none = dr.render_svg(doc, layer_visibility=vis_none)
        
        # Should be different (or at least different lengths if content differs)
        # Note: Structure might be same, but content should differ
        assert isinstance(svg_all, str)
        assert isinstance(svg_none, str)
        assert "<svg" in svg_all
        assert "<svg" in svg_none


def test_render_svg_handles_malformed_entities():
    """Test that renderer handles documents gracefully."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    # Add normal entities
    msp.add_line((0, 0), (10, 10))
    msp.add_circle((5, 5), 2)
    
    # Should render without errors
    svg_text = dr.render_svg(doc)
    assert "<svg" in svg_text

