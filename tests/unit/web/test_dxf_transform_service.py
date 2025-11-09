"""
Unit tests for DXF transformation service.
"""
from __future__ import annotations

import pytest
import ezdxf
import math

from wjp_analyser.web.modules import dxf_transform_service as trans


def test_move_entity_basic():
    """Test basic entity movement."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    trans.move_entity(line, 5, 5)
    assert line.dxf.start == (5, 5)
    assert line.dxf.end == (6, 5)


def test_move_entity_negative():
    """Test moving entity with negative values."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((10, 10), (20, 20))
    trans.move_entity(line, -5, -3)
    assert line.dxf.start == (5, 7)
    assert line.dxf.end == (15, 17)


def test_rotate_entity_basic():
    """Test basic entity rotation."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    circle = msp.add_circle((0, 0), 5)
    trans.rotate_entity(circle, 90)
    # Center should remain the same
    assert abs(circle.dxf.center[0] - 0) < 0.001
    assert abs(circle.dxf.center[1] - 0) < 0.001


def test_rotate_entity_around_center():
    """Test rotation around a specific center point."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    trans.rotate_entity(line, 90, center=(0, 0))
    # After 90° rotation, start should be at (0, 0) and end at (0, 1)
    assert abs(line.dxf.start[0] - 0) < 0.001
    assert abs(line.dxf.start[1] - 0) < 0.001


def test_scale_entity_basic():
    """Test basic entity scaling."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    circle = msp.add_circle((0, 0), 2)
    trans.scale_entity(circle, 2)
    assert abs(circle.dxf.radius - 4) < 0.001


def test_scale_entity_from_base_point():
    """Test scaling from a specific base point."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    circle = msp.add_circle((5, 5), 2)
    trans.scale_entity(circle, 2, base_point=(5, 5))
    # Center should remain at (5, 5), radius should double
    assert abs(circle.dxf.center[0] - 5) < 0.001
    assert abs(circle.dxf.center[1] - 5) < 0.001
    assert abs(circle.dxf.radius - 4) < 0.001


def test_mirror_entity_x_axis():
    """Test mirroring across X axis."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((1, 1), (2, 2))
    original_start = line.dxf.start
    trans.mirror_entity(line, axis='X')
    # After mirroring across X axis, y coordinates should be negated
    assert abs(line.dxf.start[0] - original_start[0]) < 0.001
    assert abs(line.dxf.start[1] + original_start[1]) < 0.001


def test_mirror_entity_y_axis():
    """Test mirroring across Y axis."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((1, 1), (2, 2))
    original_start = line.dxf.start
    trans.mirror_entity(line, axis='Y')
    # After mirroring across Y axis, x coordinates should be negated
    assert abs(line.dxf.start[0] + original_start[0]) < 0.001
    assert abs(line.dxf.start[1] - original_start[1]) < 0.001


def test_move_entity_error_handling():
    """Test error handling for invalid move operations."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    
    # Create a mock entity that will fail
    class BadEntity:
        def translate(self, dx, dy, dz):
            raise Exception("Test error")
        
        def dxftype(self):
            return "UNKNOWN"
    
    bad_entity = BadEntity()
    with pytest.raises(RuntimeError, match="Move failed|Unsupported entity type"):
        trans.move_entity(bad_entity, 1, 1)


def test_transform_entities_multiple():
    """Test transforming multiple entities."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    line1 = msp.add_line((0, 0), (1, 0))
    line2 = msp.add_line((2, 0), (3, 0))
    circle = msp.add_circle((5, 5), 2)
    
    handles = [line1.dxf.handle, line2.dxf.handle, circle.dxf.handle]
    count = trans.transform_entities(doc, handles, 'move', dx=10, dy=10)
    
    assert count == 3
    assert line1.dxf.start == (10, 10)
    assert line2.dxf.start == (12, 10)
    assert abs(circle.dxf.center[0] - 15) < 0.001
    assert abs(circle.dxf.center[1] - 15) < 0.001


def test_transform_entities_invalid_handle():
    """Test transforming with invalid handles."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    
    handles = [line.dxf.handle, "INVALID_HANDLE"]
    count = trans.transform_entities(doc, handles, 'move', dx=5, dy=5)
    
    # Should transform valid handle, skip invalid one
    assert count == 1
    assert line.dxf.start == (5, 5)


def test_transform_entities_rotate():
    """Test rotating multiple entities."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    circle1 = msp.add_circle((0, 0), 2)
    circle2 = msp.add_circle((5, 5), 3)
    
    handles = [circle1.dxf.handle, circle2.dxf.handle]
    count = trans.transform_entities(
        doc, handles, 'rotate',
        angle=90, center=(0, 0)
    )
    
    assert count == 2
    # Centers should be rotated 90 degrees around (0, 0)
    assert abs(circle1.dxf.center[0] - 0) < 0.001
    assert abs(circle1.dxf.center[1] - 0) < 0.001


def test_transform_entities_scale():
    """Test scaling multiple entities."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    circle1 = msp.add_circle((0, 0), 2)
    circle2 = msp.add_circle((5, 5), 3)
    
    handles = [circle1.dxf.handle, circle2.dxf.handle]
    count = trans.transform_entities(
        doc, handles, 'scale',
        factor=2.0, base_point=(0, 0)
    )
    
    assert count == 2
    assert abs(circle1.dxf.radius - 4) < 0.001
    assert abs(circle2.dxf.radius - 6) < 0.001


def test_transform_entities_mirror():
    """Test mirroring multiple entities."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    line1 = msp.add_line((1, 1), (2, 2))
    line2 = msp.add_line((3, 3), (4, 4))
    
    handles = [line1.dxf.handle, line2.dxf.handle]
    count = trans.transform_entities(doc, handles, 'mirror', axis='X')
    
    assert count == 2


def test_transform_entities_invalid_operation():
    """Test transforming with invalid operation."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    
    handles = [line.dxf.handle]
    count = trans.transform_entities(doc, handles, 'invalid_op', dx=1)
    
    # Should return 0 for invalid operation
    assert count == 0

