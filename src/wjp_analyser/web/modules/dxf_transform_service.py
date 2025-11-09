"""
DXF Transformation Service

Provides functions for transforming DXF entities: move, rotate, scale, mirror.
"""
from __future__ import annotations

import math
from typing import Tuple
from ezdxf.math import Matrix44


def _rot_point(point: Tuple[float, float], angle_rad: float, center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
    """Rotate a point around a center by angle (radians)."""
    x, y = point[0] - center[0], point[1] - center[1]
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return (
        x * cos_a - y * sin_a + center[0],
        x * sin_a + y * cos_a + center[1]
    )


def move_entity(entity, dx: float, dy: float) -> None:
    """Translate an entity by dx, dy."""
    try:
        entity.translate(dx, dy, 0)  # dz=0 for 2D translation
    except Exception:
        # Fallback for entities that don't support translate
        dxftype = entity.dxftype()
        if dxftype == "LINE":
            entity.dxf.start = (entity.dxf.start[0] + dx, entity.dxf.start[1] + dy)
            entity.dxf.end = (entity.dxf.end[0] + dx, entity.dxf.end[1] + dy)
        elif dxftype == "CIRCLE":
            entity.dxf.center = (entity.dxf.center[0] + dx, entity.dxf.center[1] + dy)
        elif dxftype == "ARC":
            entity.dxf.center = (entity.dxf.center[0] + dx, entity.dxf.center[1] + dy)
        elif dxftype == "LWPOLYLINE":
            new_pts = []
            for p in entity.get_points():
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = p[0] + dx, p[1] + dy
                    if len(p) == 2:
                        new_pts.append((x, y))
                    else:
                        new_pts.append((x, y, *p[2:]))
            entity.set_points(new_pts)
        else:
            raise RuntimeError(f"Unsupported entity type for move: {dxftype}")


def rotate_entity(entity, angle_deg: float, center: Tuple[float, float] = (0, 0)) -> None:
    """Rotate an entity around a center point by angle (degrees)."""
    try:
        angle_rad = math.radians(angle_deg)
        dxftype = entity.dxftype()
        
        if dxftype == "LINE":
            entity.dxf.start = _rot_point(entity.dxf.start, angle_rad, center)
            entity.dxf.end = _rot_point(entity.dxf.end, angle_rad, center)
        elif dxftype == "CIRCLE":
            entity.dxf.center = _rot_point(entity.dxf.center, angle_rad, center)
        elif dxftype == "ARC":
            entity.dxf.center = _rot_point(entity.dxf.center, angle_rad, center)
            entity.dxf.start_angle += angle_deg
            entity.dxf.end_angle += angle_deg
        elif dxftype == "LWPOLYLINE":
            new_pts = []
            for p in entity.get_points():
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = _rot_point((p[0], p[1]), angle_rad, center)
                    if len(p) == 2:
                        new_pts.append((x, y))
                    else:
                        new_pts.append((x, y, *p[2:]))
            entity.set_points(new_pts)
        elif dxftype == "POLYLINE":
            for vertex in entity.vertices:
                loc = vertex.dxf.location
                vertex.dxf.location = _rot_point((loc.x, loc.y), angle_rad, center)
        elif dxftype == "SPLINE":
            try:
                for cp in entity.control_points:
                    x, y = _rot_point((cp[0], cp[1]), angle_rad, center)
                    cp[0] = x
                    cp[1] = y
            except Exception:
                try:
                    for fp in entity.fit_points:
                        x, y = _rot_point((fp[0], fp[1]), angle_rad, center)
                        fp[0] = x
                        fp[1] = y
                except Exception:
                    raise RuntimeError(f"Could not rotate SPLINE entity")
        else:
            raise RuntimeError(f"Unsupported entity type for rotate: {dxftype}")
    except Exception as e:
        raise RuntimeError(f"Rotate failed: {e}")


def scale_entity(entity, factor: float, base_point: Tuple[float, float] = (0, 0)) -> None:
    """Scale an entity from a base point."""
    try:
        dxftype = entity.dxftype()
        
        if dxftype == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            entity.dxf.start = (
                base_point[0] + (start[0] - base_point[0]) * factor,
                base_point[1] + (start[1] - base_point[1]) * factor
            )
            entity.dxf.end = (
                base_point[0] + (end[0] - base_point[0]) * factor,
                base_point[1] + (end[1] - base_point[1]) * factor
            )
        elif dxftype == "CIRCLE":
            center = entity.dxf.center
            entity.dxf.center = (
                base_point[0] + (center[0] - base_point[0]) * factor,
                base_point[1] + (center[1] - base_point[1]) * factor
            )
            entity.dxf.radius *= factor
        elif dxftype == "ARC":
            center = entity.dxf.center
            entity.dxf.center = (
                base_point[0] + (center[0] - base_point[0]) * factor,
                base_point[1] + (center[1] - base_point[1]) * factor
            )
            entity.dxf.radius *= factor
        elif dxftype == "LWPOLYLINE":
            new_pts = []
            for p in entity.get_points():
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x = base_point[0] + (p[0] - base_point[0]) * factor
                    y = base_point[1] + (p[1] - base_point[1]) * factor
                    if len(p) == 2:
                        new_pts.append((x, y))
                    else:
                        new_pts.append((x, y, *p[2:]))
            entity.set_points(new_pts)
        else:
            # Try built-in scale method as fallback
            entity.scale(factor, base_point)
    except Exception as e:
        raise RuntimeError(f"Scale failed: {e}")


def mirror_entity(entity, axis: str = 'X') -> None:
    """Mirror an entity across the X or Y axis."""
    try:
        if axis.upper() == 'Y':
            m = Matrix44.scale(-1, 1, 1)
        else:
            m = Matrix44.scale(1, -1, 1)
        entity.transform(m)
    except Exception:
        # Fallback: manual mirroring
        dxftype = entity.dxftype()
        if dxftype == "LINE":
            if axis.upper() == 'Y':
                entity.dxf.start = (-entity.dxf.start[0], entity.dxf.start[1])
                entity.dxf.end = (-entity.dxf.end[0], entity.dxf.end[1])
            else:
                entity.dxf.start = (entity.dxf.start[0], -entity.dxf.start[1])
                entity.dxf.end = (entity.dxf.end[0], -entity.dxf.end[1])
        elif dxftype == "CIRCLE":
            if axis.upper() == 'Y':
                entity.dxf.center = (-entity.dxf.center[0], entity.dxf.center[1])
            else:
                entity.dxf.center = (entity.dxf.center[0], -entity.dxf.center[1])
        else:
            raise RuntimeError(f"Unsupported entity type for mirror: {dxftype}")
    except Exception as e:
        raise RuntimeError(f"Mirror failed: {e}")


def transform_entities(doc, handles: list[str], operation: str, **params) -> int:
    """
    Transform multiple entities by their handles.
    
    Args:
        doc: DXF document
        handles: List of entity handles to transform
        operation: 'move', 'rotate', 'scale', or 'mirror'
        **params: Operation-specific parameters
    
    Returns:
        Number of entities successfully transformed
    """
    msp = doc.modelspace()
    count = 0
    
    for handle in handles:
        try:
            entity = doc.entitydb.get(handle)
            if not entity or entity not in msp:
                continue
            
            if operation == 'move':
                move_entity(entity, params.get('dx', 0), params.get('dy', 0))
            elif operation == 'rotate':
                rotate_entity(
                    entity,
                    params.get('angle', 0),
                    params.get('center', (0, 0))
                )
            elif operation == 'scale':
                scale_entity(
                    entity,
                    params.get('factor', 1.0),
                    params.get('base_point', (0, 0))
                )
            elif operation == 'mirror':
                mirror_entity(entity, params.get('axis', 'X'))
            else:
                continue
            
            count += 1
        except Exception:
            # Skip entities that can't be transformed
            continue
    
    return count

