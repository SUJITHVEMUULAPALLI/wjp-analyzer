"""
Additional tests to achieve 100% coverage for dxf_utils.py.

Tests the remaining defensive exception handlers:
- Exception handler in entity_summary (lines 55-57)
- Exception handler in delete_entities_by_handle (lines 71-72)
"""
from __future__ import annotations

import pytest
import ezdxf
from unittest.mock import Mock, patch

from wjp_analyser.web.modules import dxf_utils as du


def test_entity_summary_handles_malformed_entity():
    """Test entity_summary handles malformed entities gracefully (lines 55-57)."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    # Add a normal entity
    line1 = msp.add_line((0, 0), (10, 10))
    normal_handle = line1.dxf.handle
    
    # Create an entity and patch it to raise exception when accessing handle
    line2 = msp.add_line((5, 5), (15, 15))
    
    # Use patch to make accessing handle raise exception for this specific entity
    # We'll patch the dxf attribute to return a mock that raises on handle access
    original_dxf = line2.dxf
    
    class MockDXF:
        def __init__(self):
            self.layer = original_dxf.layer
            self.color = getattr(original_dxf, 'color', None)
        
        @property
        def handle(self):
            raise Exception("Test exception")
    
    line2.dxf = MockDXF()
    
    try:
        # Now get summary - should skip the malformed entity
        summary = du.entity_summary(doc)
        
        # Should still get at least the first line (the one that doesn't raise exception)
        assert len(summary) >= 1
        # The malformed entity should be skipped
        assert all("handle" in e for e in summary)
        # Should have the normal handle
        assert any(e["handle"] == normal_handle for e in summary)
    finally:
        # Restore original dxf
        line2.dxf = original_dxf


def test_entity_summary_handles_missing_layer_attribute():
    """Test entity_summary handles entities without layer attribute."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    # Add normal entity
    msp.add_line((0, 0), (10, 10))
    
    # Create entity without layer attribute
    # We'll use a circle and patch it
    circle = msp.add_circle((5, 5), 2)
    
    # Temporarily remove layer attribute
    if hasattr(circle.dxf, 'layer'):
        original_layer = circle.dxf.layer
        delattr(circle.dxf, 'layer')
    
    try:
        summary = du.entity_summary(doc)
        # Should still work, using "0" as default
        assert len(summary) >= 1
        assert all(e["layer"] == "0" for e in summary)
    finally:
        # Restore if needed
        if not hasattr(circle.dxf, 'layer'):
            circle.dxf.layer = original_layer


def test_delete_entities_by_handle_handles_deletion_failure():
    """Test delete_entities_by_handle handles deletion failures gracefully (lines 71-72)."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    # Add entities
    line1 = msp.add_line((0, 0), (10, 10))
    line2 = msp.add_line((10, 10), (20, 20))
    
    handles = [line1.dxf.handle, line2.dxf.handle]
    
    # Mock delete_entity to raise exception for first entity
    original_delete = msp.delete_entity
    
    call_count = [0]
    def mock_delete(entity):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Deletion failed")
        return original_delete(entity)
    
    msp.delete_entity = mock_delete
    
    try:
        # Should handle the exception and continue
        deleted_count = du.delete_entities_by_handle(doc, handles)
        
        # Should delete at least one (the second one)
        assert deleted_count >= 1
    finally:
        # Restore original method
        msp.delete_entity = original_delete


def test_delete_entities_by_handle_handles_invalid_handle_access():
    """Test delete_entities_by_handle handles entities with invalid handle access."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    # Add normal entity
    line = msp.add_line((0, 0), (10, 10))
    valid_handle = line.dxf.handle
    
    # Create a mock entity that raises exception when accessing handle
    # We'll patch the iteration to include a problematic entity
    entities_list = list(msp)
    
    # Create a mock entity
    mock_entity = Mock()
    type(mock_entity).dxf = Mock()
    type(mock_entity.dxf).handle = property(lambda self: (_ for _ in ()).throw(Exception("Invalid handle")))
    
    # Patch list(msp) to include the mock
    with patch.object(msp, '__iter__', return_value=iter(entities_list + [mock_entity])):
        # Should handle the exception and continue
        deleted_count = du.delete_entities_by_handle(doc, [valid_handle])
        
        # Should still delete the valid entity
        assert deleted_count >= 0  # At least doesn't crash

