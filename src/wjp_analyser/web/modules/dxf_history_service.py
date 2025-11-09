"""
DXF History Service

Manages undo/redo operations for DXF edits.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional
from ezdxf.math import Vec3


class HistoryManager:
    """Manages undo/redo operations for DXF edits."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize history manager.
        
        Args:
            max_history: Maximum number of undo operations to store
        """
        self.undo_stack: List[Tuple[str, str, Dict]] = []
        self.redo_stack: List[Tuple[str, str, Dict]] = []
        self.max_history = max_history
    
    def record(self, action: str, entity_handle: str, params: Dict) -> None:
        """
        Record an action for undo/redo.
        
        Args:
            action: Action type ('move', 'rotate', 'scale', 'delete', etc.)
            entity_handle: Handle of the affected entity
            params: Action parameters (dx, dy, angle, factor, etc.)
        """
        self.undo_stack.append((action, entity_handle, params.copy()))
        self.redo_stack.clear()
        
        # Limit history size
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
    
    def record_batch(self, action: str, handles: List[str], params: Dict) -> None:
        """
        Record a batch action affecting multiple entities.
        
        Args:
            action: Action type
            handles: List of entity handles
            params: Action parameters
        """
        for handle in handles:
            self.record(action, handle, params)
    
    def undo(self, doc) -> bool:
        """
        Undo the last action.
        
        Args:
            doc: DXF document
        
        Returns:
            True if undo was successful, False if no actions to undo
        """
        if not self.undo_stack:
            return False
        
        action, handle, params = self.undo_stack.pop()
        self._reverse_action(doc, action, handle, params)
        self.redo_stack.append((action, handle, params))
        return True
    
    def redo(self, doc) -> bool:
        """
        Redo the last undone action.
        
        Args:
            doc: DXF document
        
        Returns:
            True if redo was successful, False if no actions to redo
        """
        if not self.redo_stack:
            return False
        
        action, handle, params = self.redo_stack.pop()
        self._apply_action(doc, action, handle, params)
        self.undo_stack.append((action, handle, params))
        return True
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.redo_stack) > 0
    
    def clear(self) -> None:
        """Clear all history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
    
    def get_history_info(self) -> Dict:
        """Get information about current history state."""
        return {
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
            "can_undo": self.can_undo(),
            "can_redo": self.can_redo(),
        }
    
    def _apply_action(self, doc, action: str, handle: str, params: Dict):
        """Apply an action to an entity."""
        from . import dxf_transform_service as trans
        
        entity = doc.entitydb.get(handle)
        if not entity:
            return
        
        if action == 'move':
            trans.move_entity(entity, params.get('dx', 0), params.get('dy', 0))
        elif action == 'rotate':
            trans.rotate_entity(
                entity,
                params.get('angle', 0),
                params.get('center', (0, 0))
            )
        elif action == 'scale':
            trans.scale_entity(
                entity,
                params.get('factor', 1.0),
                params.get('base_point', (0, 0))
            )
        elif action == 'mirror':
            trans.mirror_entity(entity, params.get('axis', 'X'))
    
    def _reverse_action(self, doc, action: str, handle: str, params: Dict):
        """Reverse an action on an entity."""
        from . import dxf_transform_service as trans
        
        entity = doc.entitydb.get(handle)
        if not entity:
            return
        
        if action == 'move':
            trans.move_entity(entity, -params.get('dx', 0), -params.get('dy', 0))
        elif action == 'rotate':
            trans.rotate_entity(
                entity,
                -params.get('angle', 0),
                params.get('center', (0, 0))
            )
        elif action == 'scale':
            inv_factor = 1 / params.get('factor', 1.0) if params.get('factor', 1.0) != 0 else 1.0
            trans.scale_entity(
                entity,
                inv_factor,
                params.get('base_point', (0, 0))
            )
        elif action == 'mirror':
            # Mirror is its own inverse
            trans.mirror_entity(entity, params.get('axis', 'X'))
        elif action == 'delete':
            # For delete, we would need to restore the entity
            # This is more complex and would require storing entity data
            pass

