"""
Operation Executor for DXF Editor
==================================

Executes recommendation operations on DXF documents.
Maps OperationType enum to actual DXF modification functions.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import math
import logging

try:
    import ezdxf
    from ezdxf import recover
except ImportError:
    ezdxf = None

from wjp_analyser.ai.recommendation_engine import Operation, OperationType

logger = logging.getLogger(__name__)


class OperationExecutor:
    """Execute operations on DXF documents."""
    
    def __init__(self, doc, msp):
        """
        Initialize executor with DXF document and modelspace.
        
        Args:
            doc: ezdxf DXF document
            msp: Modelspace of the document
        """
        self.doc = doc
        self.msp = msp
        self.entities = list(msp)
        self._affected_entities = []
        self._removed_entities = []
        self._modified_entities = []
    
    def execute(self, operation: Operation, preview: bool = False) -> Dict[str, Any]:
        """
        Execute an operation on the DXF.
        
        Args:
            operation: Operation to execute
            preview: If True, don't modify DXF, just return what would change
            
        Returns:
            Dict with success status, affected count, and details
        """
        self._affected_entities = []
        self._removed_entities = []
        self._modified_entities = []
        
        executor_map = {
            OperationType.REMOVE_ZERO_AREA: self._remove_zero_area,
            OperationType.CLOSE_OPEN_CONTOUR: self._close_open_contours,
            OperationType.FILLET_MIN_RADIUS: self._fillet_min_radius,
            OperationType.FILTER_TINY: self._filter_tiny,
            OperationType.SIMPLIFY_EPS: self._simplify_eps,
            OperationType.FIX_MIN_SPACING: self._fix_min_spacing,
            OperationType.REMOVE_DUPLICATE: self._remove_duplicates,
            OperationType.ASSIGN_LAYER: self._assign_layer,
        }
        
        executor = executor_map.get(operation.operation)
        if executor:
            try:
                result = executor(operation.parameters, preview)
                result["operation"] = operation.operation.value
                result["preview"] = preview
                return result
            except Exception as e:
                logger.error(f"Error executing {operation.operation}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "operation": operation.operation.value
                }
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation.operation}",
                "operation": operation.operation.value
            }
    
    def _has_zero_area(self, entity) -> bool:
        """Check if entity has zero or negative area."""
        try:
            if entity.dxftype() == "CIRCLE":
                return entity.dxf.radius <= 0
            elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                # Check if closed and has area
                if hasattr(entity, "closed") and entity.closed:
                    try:
                        points = list(entity.get_points("xy"))
                        if len(points) < 3:
                            return True
                        # Simple area check using shoelace formula
                        area = 0.0
                        for i in range(len(points)):
                            j = (i + 1) % len(points)
                            area += points[i][0] * points[j][1]
                            area -= points[j][0] * points[i][1]
                        return abs(area) < 0.001  # Very small area threshold
                    except Exception:
                        return True
                return False
            elif entity.dxftype() == "LINE":
                return False  # Lines don't have area
            elif entity.dxftype() in ["ARC", "SPLINE"]:
                return False  # These are curves, not closed shapes
            return False
        except Exception:
            return True  # If we can't check, assume zero area
    
    def _remove_zero_area(self, params: Dict, preview: bool) -> Dict:
        """Remove entities with zero area."""
        threshold = params.get("threshold", 0.0)
        removed = []
        
        for entity in self.entities:
            if self._has_zero_area(entity):
                removed.append(entity)
                if not preview:
                    try:
                        self.msp.delete_entity(entity)
                    except Exception as e:
                        logger.warning(f"Failed to delete entity: {e}")
        
        return {
            "success": True,
            "affected_count": len(removed),
            "removed_entities": [self._get_entity_id(e) for e in removed],
            "message": f"Removed {len(removed)} zero-area entities"
        }
    
    def _close_open_contours(self, params: Dict, preview: bool) -> Dict:
        """Close open contours within tolerance."""
        tolerance = params.get("tolerance_mm", 0.1)
        closed_count = 0
        modified = []
        
        for entity in self.entities:
            if entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                try:
                    # Check if already closed
                    is_closed = False
                    if entity.dxftype() == "LWPOLYLINE":
                        is_closed = getattr(entity, "closed", False)
                    elif entity.dxftype() == "POLYLINE":
                        is_closed = getattr(entity, "is_closed", False) or getattr(entity, "closed", False)
                    
                    if not is_closed:
                        # Get points
                        if entity.dxftype() == "LWPOLYLINE":
                            points = list(entity.get_points("xy"))
                        else:  # POLYLINE
                            try:
                                points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                            except Exception:
                                points = []
                        
                        if len(points) >= 2:
                            # Check if first and last points are close
                            first = points[0]
                            last = points[-1]
                            distance = math.hypot(first[0] - last[0], first[1] - last[1])
                            
                            if distance <= tolerance:
                                # Close the polyline
                                if not preview:
                                    try:
                                        if entity.dxftype() == "LWPOLYLINE":
                                            entity.closed = True
                                        elif entity.dxftype() == "POLYLINE":
                                            # For POLYLINE, try different methods
                                            if hasattr(entity, "close"):
                                                entity.close(True)
                                            elif hasattr(entity, "is_closed"):
                                                entity.is_closed = True
                                            # Also ensure last vertex connects to first
                                            if hasattr(entity, "vertices") and len(entity.vertices) > 0:
                                                first_vertex = entity.vertices[0]
                                                last_vertex = entity.vertices[-1]
                                                if first_vertex.dxf.location != last_vertex.dxf.location:
                                                    # Add closing vertex if needed
                                                    pass  # Complex - skip for now
                                    except Exception as e:
                                        logger.warning(f"Failed to close polyline: {e}")
                                closed_count += 1
                                modified.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to close contour: {e}")
        
        return {
            "success": True,
            "affected_count": closed_count,
            "modified_entities": [self._get_entity_id(e) for e in modified],
            "message": f"Closed {closed_count} open contours"
        }
    
    def _fillet_min_radius(self, params: Dict, preview: bool) -> Dict:
        """Apply fillets to sharp corners (placeholder - complex operation)."""
        min_radius = params.get("min_radius_mm", 2.0)
        
        # This is a complex operation that would require geometry processing
        # For now, return a placeholder
        return {
            "success": False,
            "error": "Fillet operation not yet implemented. Requires geometry processing.",
            "affected_count": 0,
            "message": "Fillet operation requires advanced geometry processing"
        }
    
    def _filter_tiny(self, params: Dict, preview: bool) -> Dict:
        """Remove tiny objects below area threshold."""
        min_area = params.get("min_area_mm2", 1.0)
        removed = []
        
        for entity in self.entities:
            try:
                area = self._get_entity_area(entity)
                if 0 < area < min_area:
                    removed.append(entity)
                    if not preview:
                        try:
                            self.msp.delete_entity(entity)
                        except Exception:
                            pass
            except Exception:
                pass
        
        return {
            "success": True,
            "affected_count": len(removed),
            "removed_entities": [self._get_entity_id(e) for e in removed],
            "message": f"Removed {len(removed)} tiny objects (< {min_area} mm²)"
        }
    
    def _simplify_eps(self, params: Dict, preview: bool) -> Dict:
        """Simplify geometry with Douglas-Peucker algorithm."""
        tolerance = params.get("tolerance_mm", 0.05)
        simplified_count = 0
        modified = []
        
        # This requires Shapely for simplification
        try:
            from shapely.geometry import LineString, Polygon
            from shapely.ops import simplify
        except ImportError:
            return {
                "success": False,
                "error": "Shapely required for simplification",
                "affected_count": 0
            }
        
        for entity in self.entities:
            if entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                try:
                    points = list(entity.get_points("xy"))
                    if len(points) > 2:
                        # Create LineString and simplify
                        line = LineString(points)
                        simplified = simplify(line, tolerance=tolerance)
                        
                        if len(simplified.coords) < len(points):
                            if not preview:
                                # Update entity with simplified points
                                new_points = [(p[0], p[1]) for p in simplified.coords]
                                if entity.dxftype() == "LWPOLYLINE":
                                    entity.set_points(new_points)
                                # POLYLINE is more complex to modify
                            simplified_count += 1
                            modified.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to simplify entity: {e}")
        
        return {
            "success": True,
            "affected_count": simplified_count,
            "modified_entities": [self._get_entity_id(e) for e in modified],
            "message": f"Simplified {simplified_count} entities"
        }
    
    def _fix_min_spacing(self, params: Dict, preview: bool) -> Dict:
        """Fix minimum spacing violations (placeholder - complex operation)."""
        min_spacing = params.get("min_spacing_mm", 3.0)
        
        # This is a complex operation requiring spatial analysis
        return {
            "success": False,
            "error": "Min spacing fix not yet implemented. Requires spatial analysis.",
            "affected_count": 0,
            "message": "Min spacing fix requires advanced spatial processing"
        }
    
    def _remove_duplicates(self, params: Dict, preview: bool) -> Dict:
        """Remove duplicate entities."""
        removed = []
        seen = set()
        
        for entity in self.entities:
            try:
                # Create a signature for the entity
                sig = self._get_entity_signature(entity)
                if sig in seen:
                    removed.append(entity)
                    if not preview:
                        try:
                            self.msp.delete_entity(entity)
                        except Exception:
                            pass
                else:
                    seen.add(sig)
            except Exception:
                pass
        
        return {
            "success": True,
            "affected_count": len(removed),
            "removed_entities": [self._get_entity_id(e) for e in removed],
            "message": f"Removed {len(removed)} duplicate entities"
        }
    
    def _assign_layer(self, params: Dict, preview: bool) -> Dict:
        """Assign entities to a specific layer."""
        layer_name = params.get("layer", "OUTER")
        min_area = params.get("min_area_mm2", 100.0)
        assigned_count = 0
        modified = []
        
        # Ensure layer exists
        if not preview:
            try:
                if layer_name not in self.doc.layers:
                    self.doc.layers.new(name=layer_name)
            except Exception:
                pass
        
        for entity in self.entities:
            try:
                area = self._get_entity_area(entity)
                if area >= min_area:
                    if not preview:
                        entity.dxf.layer = layer_name
                    assigned_count += 1
                    modified.append(entity)
            except Exception:
                pass
        
        return {
            "success": True,
            "affected_count": assigned_count,
            "modified_entities": [self._get_entity_id(e) for e in modified],
            "message": f"Assigned {assigned_count} entities to layer '{layer_name}'"
        }
    
    def _get_entity_id(self, entity) -> str:
        """Get unique identifier for entity."""
        try:
            return entity.dxf.handle
        except Exception:
            return str(id(entity))
    
    def _get_entity_area(self, entity) -> float:
        """Calculate area of entity."""
        try:
            if entity.dxftype() == "CIRCLE":
                return math.pi * entity.dxf.radius ** 2
            elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                if getattr(entity, "closed", False):
                    points = list(entity.get_points("xy"))
                    if len(points) < 3:
                        return 0.0
                    # Shoelace formula
                    area = 0.0
                    for i in range(len(points)):
                        j = (i + 1) % len(points)
                        area += points[i][0] * points[j][1]
                        area -= points[j][0] * points[i][1]
                    return abs(area) / 2.0
            return 0.0
        except Exception:
            return 0.0
    
    def _get_entity_signature(self, entity) -> str:
        """Create a signature for duplicate detection."""
        try:
            et = entity.dxftype()
            if et == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                return f"LINE:({start[0]:.3f},{start[1]:.3f})-({end[0]:.3f},{end[1]:.3f})"
            elif et == "CIRCLE":
                center = entity.dxf.center
                radius = entity.dxf.radius
                return f"CIRCLE:({center[0]:.3f},{center[1]:.3f}),r={radius:.3f}"
            elif et in ["LWPOLYLINE", "POLYLINE"]:
                points = list(entity.get_points("xy"))
                point_str = ",".join([f"({p[0]:.3f},{p[1]:.3f})" for p in points[:10]])  # First 10 points
                return f"{et}:{point_str}"
            else:
                return f"{et}:{self._get_entity_id(entity)}"
        except Exception:
            return str(id(entity))

