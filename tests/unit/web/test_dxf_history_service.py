"""
Unit tests for DXF history service (undo/redo).
"""
from __future__ import annotations

import pytest
import ezdxf

from wjp_analyser.web.modules import dxf_history_service as hist


def test_history_manager_initialization():
    """Test HistoryManager initialization."""
    h = hist.HistoryManager()
    assert len(h.undo_stack) == 0
    assert len(h.redo_stack) == 0
    assert not h.can_undo()
    assert not h.can_redo()


def test_history_record():
    """Test recording an action."""
    h = hist.HistoryManager()
    h.record('move', 'handle123', {'dx': 10, 'dy': 5})
    
    assert len(h.undo_stack) == 1
    assert len(h.redo_stack) == 0
    assert h.can_undo()
    assert not h.can_redo()
    
    action, handle, params = h.undo_stack[0]
    assert action == 'move'
    assert handle == 'handle123'
    assert params == {'dx': 10, 'dy': 5}


def test_history_record_clears_redo():
    """Test that recording clears redo stack."""
    h = hist.HistoryManager()
    h.record('move', 'handle1', {'dx': 1, 'dy': 1})
    h.undo(ezdxf.new())  # This moves to redo stack
    assert h.can_redo()
    
    h.record('move', 'handle2', {'dx': 2, 'dy': 2})
    assert not h.can_redo()  # Redo should be cleared


def test_history_undo_move():
    """Test undoing a move operation."""
    from wjp_analyser.web.modules import dxf_transform_service as trans
    
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    original_start = line.dxf.start
    
    h = hist.HistoryManager()
    h.record('move', line.dxf.handle, {'dx': 10, 'dy': 10})
    
    # Move the entity
    trans.move_entity(line, 10, 10)
    assert line.dxf.start != original_start
    
    # Undo
    h.undo(doc)
    assert abs(line.dxf.start[0] - original_start[0]) < 0.001
    assert abs(line.dxf.start[1] - original_start[1]) < 0.001
    assert h.can_redo()


def test_history_redo_move():
    """Test redoing a move operation."""
    from wjp_analyser.web.modules import dxf_transform_service as trans
    
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    original_start = line.dxf.start
    
    h = hist.HistoryManager()
    h.record('move', line.dxf.handle, {'dx': 10, 'dy': 10})
    
    # Move and undo
    trans.move_entity(line, 10, 10)
    h.undo(doc)
    assert line.dxf.start == original_start
    
    # Redo
    h.redo(doc)
    assert abs(line.dxf.start[0] - (original_start[0] + 10)) < 0.001
    assert abs(line.dxf.start[1] - (original_start[1] + 10)) < 0.001


def test_history_undo_rotate():
    """Test undoing a rotate operation."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((1, 0), (2, 0))
    original_start = line.dxf.start
    
    h = hist.HistoryManager()
    h.record('rotate', line.dxf.handle, {'angle': 90, 'center': (0, 0)})
    
    # Rotate manually (since entities don't all have rotate method)
    import math
    from wjp_analyser.web.modules import dxf_transform_service as trans
    trans.rotate_entity(line, 90, (0, 0))
    
    # Undo
    h.undo(doc)
    assert abs(line.dxf.start[0] - original_start[0]) < 0.001
    assert abs(line.dxf.start[1] - original_start[1]) < 0.001


def test_history_undo_scale():
    """Test undoing a scale operation."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    circle = msp.add_circle((0, 0), 2)
    original_radius = circle.dxf.radius
    
    h = hist.HistoryManager()
    h.record('scale', circle.dxf.handle, {'factor': 2.0, 'base_point': (0, 0)})
    
    # Scale
    from wjp_analyser.web.modules import dxf_transform_service as trans
    trans.scale_entity(circle, 2.0, (0, 0))
    assert abs(circle.dxf.radius - 4) < 0.001
    
    # Undo
    h.undo(doc)
    assert abs(circle.dxf.radius - original_radius) < 0.001


def test_history_undo_without_actions():
    """Test undo when no actions are available."""
    doc = ezdxf.new(dxfversion="R2010")
    h = hist.HistoryManager()
    assert not h.undo(doc)
    assert not h.can_undo()


def test_history_redo_without_actions():
    """Test redo when no actions are available."""
    doc = ezdxf.new(dxfversion="R2010")
    h = hist.HistoryManager()
    assert not h.redo(doc)
    assert not h.can_redo()


def test_history_clear():
    """Test clearing history."""
    h = hist.HistoryManager()
    h.record('move', 'handle1', {'dx': 1, 'dy': 1})
    h.record('move', 'handle2', {'dx': 2, 'dy': 2})
    
    h.clear()
    assert len(h.undo_stack) == 0
    assert len(h.redo_stack) == 0
    assert not h.can_undo()
    assert not h.can_redo()


def test_history_max_size():
    """Test that history is limited to max_history."""
    h = hist.HistoryManager(max_history=3)
    
    for i in range(5):
        h.record('move', f'handle{i}', {'dx': i, 'dy': i})
    
    # Should only keep last 3
    assert len(h.undo_stack) == 3
    # Last action should be handle4
    assert h.undo_stack[-1][1] == 'handle4'


def test_history_record_batch():
    """Test recording batch actions."""
    h = hist.HistoryManager()
    handles = ['handle1', 'handle2', 'handle3']
    h.record_batch('move', handles, {'dx': 10, 'dy': 10})
    
    assert len(h.undo_stack) == 3
    assert all(action == 'move' for action, _, _ in h.undo_stack)


def test_history_get_info():
    """Test getting history information."""
    h = hist.HistoryManager()
    h.record('move', 'handle1', {'dx': 1, 'dy': 1})
    h.record('move', 'handle2', {'dx': 2, 'dy': 2})
    
    info = h.get_history_info()
    assert info['undo_count'] == 2
    assert info['redo_count'] == 0
    assert info['can_undo'] is True
    assert info['can_redo'] is False


def test_history_undo_redo_cycle():
    """Test complete undo/redo cycle."""
    from wjp_analyser.web.modules import dxf_transform_service as trans
    
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    line = msp.add_line((0, 0), (1, 0))
    original_start = line.dxf.start
    
    h = hist.HistoryManager()
    h.record('move', line.dxf.handle, {'dx': 10, 'dy': 10})
    
    # Move
    trans.move_entity(line, 10, 10)
    moved_start = line.dxf.start
    
    # Undo
    h.undo(doc)
    assert line.dxf.start == original_start
    
    # Redo
    h.redo(doc)
    assert abs(line.dxf.start[0] - moved_start[0]) < 0.001
    assert abs(line.dxf.start[1] - moved_start[1]) < 0.001
    
    # Undo again
    h.undo(doc)
    assert line.dxf.start == original_start

