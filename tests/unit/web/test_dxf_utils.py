"""
Tests for DXF Editor utilities (dxf_utils module).

Tests file handling, layer management, entity operations, and document I/O.
"""
from __future__ import annotations

import os
import pytest
import ezdxf
from pathlib import Path

# Import the modules under test
from wjp_analyser.web.modules import dxf_utils as du


def test_load_document_success(sample_dxf):
    """Test loading a valid DXF file."""
    doc = du.load_document(sample_dxf)
    assert doc is not None
    # ezdxf.readfile returns a Document object
    assert hasattr(doc, 'modelspace')
    assert hasattr(doc, 'layers')


def test_load_document_not_found():
    """Test loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        du.load_document("nonexistent_file.dxf")


def test_list_layers(sample_dxf):
    """Test listing layers from a DXF document."""
    doc = du.load_document(sample_dxf)
    layers = du.list_layers(doc)
    
    assert isinstance(layers, list)
    assert "0" in layers  # Default layer should always exist
    assert "LAYER1" in layers
    assert "LAYER2" in layers


def test_entity_summary_counts(sample_dxf):
    """Test entity summary returns correct entity types and counts."""
    doc = du.load_document(sample_dxf)
    summary = du.entity_summary(doc)
    
    assert isinstance(summary, list)
    assert len(summary) >= 5  # At least: line, circle, text, line2, circle2
    
    types = [d["type"] for d in summary]
    assert "LINE" in types
    assert "CIRCLE" in types
    assert "TEXT" in types


def test_entity_summary_structure(sample_dxf):
    """Test entity summary has correct structure."""
    doc = du.load_document(sample_dxf)
    summary = du.entity_summary(doc)
    
    assert len(summary) > 0
    for entity in summary:
        assert "handle" in entity
        assert "type" in entity
        assert "layer" in entity
        assert "color" in entity
        assert isinstance(entity["handle"], str)
        assert isinstance(entity["type"], str)
        assert isinstance(entity["layer"], str)


def test_entity_summary_layer_assignment(sample_dxf):
    """Test entity summary correctly captures layer assignments."""
    doc = du.load_document(sample_dxf)
    summary = du.entity_summary(doc)
    
    layers_found = set(e["layer"] for e in summary)
    assert "0" in layers_found
    assert "LAYER1" in layers_found
    assert "LAYER2" in layers_found


def test_delete_entities_by_handle(sample_dxf):
    """Test deleting entities by handle."""
    doc = du.load_document(sample_dxf)
    msp = doc.modelspace()
    
    # Get initial count
    initial_count = len(list(msp))
    assert initial_count >= 5
    
    # Get handles of first entity
    entities = list(msp)
    handles_to_delete = [entities[0].dxf.handle]
    
    # Delete one entity
    deleted_count = du.delete_entities_by_handle(doc, handles_to_delete)
    assert deleted_count == 1
    
    # Verify count decreased
    remaining_count = len(list(msp))
    assert remaining_count == initial_count - 1


def test_delete_entities_multiple(sample_dxf):
    """Test deleting multiple entities by handle."""
    doc = du.load_document(sample_dxf)
    msp = doc.modelspace()
    
    initial_count = len(list(msp))
    entities = list(msp)
    
    # Delete first two entities
    handles_to_delete = [e.dxf.handle for e in entities[:2]]
    deleted_count = du.delete_entities_by_handle(doc, handles_to_delete)
    
    assert deleted_count == 2
    assert len(list(msp)) == initial_count - 2


def test_delete_entities_nonexistent_handle(sample_dxf):
    """Test deleting with non-existent handles returns 0."""
    doc = du.load_document(sample_dxf)
    
    deleted_count = du.delete_entities_by_handle(doc, ["INVALID_HANDLE"])
    assert deleted_count == 0


def test_delete_entities_empty_list(sample_dxf):
    """Test deleting with empty handle list returns 0."""
    doc = du.load_document(sample_dxf)
    
    deleted_count = du.delete_entities_by_handle(doc, [])
    assert deleted_count == 0


def test_save_document(sample_dxf, temp_dir):
    """Test saving a DXF document."""
    doc = du.load_document(sample_dxf)
    out_path = os.path.join(temp_dir, "out.dxf")
    
    saved_path = du.save_document(doc, out_path)
    
    assert os.path.exists(saved_path)
    assert saved_path == out_path
    
    # Verify we can load the saved file
    loaded_doc = du.load_document(saved_path)
    assert loaded_doc is not None
    assert len(list(loaded_doc.modelspace())) == len(list(doc.modelspace()))


def test_save_document_creates_directory(temp_dir):
    """Test saving creates parent directories if they don't exist."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 10))
    
    nested_path = os.path.join(temp_dir, "nested", "dir", "out.dxf")
    saved_path = du.save_document(doc, nested_path)
    
    assert os.path.exists(saved_path)
    assert os.path.exists(os.path.dirname(saved_path))


def test_temp_copy(sample_dxf):
    """Test creating a temporary copy of a DXF file."""
    copy_path = du.temp_copy(sample_dxf)
    
    assert os.path.exists(copy_path)
    assert copy_path != sample_dxf
    
    # Verify copy is readable
    doc = du.load_document(copy_path)
    assert doc is not None
    
    # Clean up
    if os.path.exists(copy_path):
        os.remove(copy_path)


def test_temp_copy_preserves_content(sample_dxf):
    """Test temporary copy preserves DXF content."""
    original_doc = du.load_document(sample_dxf)
    original_count = len(list(original_doc.modelspace()))
    
    copy_path = du.temp_copy(sample_dxf)
    try:
        copy_doc = du.load_document(copy_path)
        copy_count = len(list(copy_doc.modelspace()))
        
        assert copy_count == original_count
    finally:
        if os.path.exists(copy_path):
            os.remove(copy_path)


def test_session_keys_are_strings():
    """Test that session key constants are strings."""
    assert isinstance(du.SESSION_DXF_KEY, str)
    assert isinstance(du.SESSION_PATH_KEY, str)
    assert isinstance(du.SESSION_EDIT_LOG, str)
    assert isinstance(du.SESSION_LAYER_VIS, str)
    assert isinstance(du.SESSION_SELECTED, str)

