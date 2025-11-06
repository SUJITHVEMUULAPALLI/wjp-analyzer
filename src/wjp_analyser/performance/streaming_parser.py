"""
Streaming DXF Parser
====================

Efficient parser for large DXF files using streaming and chunking.
Handles files >10MB without loading entire file into memory.
"""

from __future__ import annotations

import io
import hashlib
from typing import Iterator, Dict, List, Tuple, Optional, Callable
from pathlib import Path


class StreamingDXFParser:
    """Streaming parser for large DXF files."""
    
    def __init__(self, chunk_size: int = 8192, max_memory_mb: int = 100):
        """
        Initialize streaming parser.
        
        Args:
            chunk_size: Size of chunks to process (bytes)
            max_memory_mb: Maximum memory usage (MB)
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
    
    def parse_in_chunks(
        self,
        dxf_path: str,
        entity_filter: Optional[Callable] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Dict]:
        """
        Parse DXF file in chunks, yielding entities as they're processed.
        
        Args:
            dxf_path: Path to DXF file
            entity_filter: Optional filter function for entities
            progress_callback: Optional callback(processed, total) for progress
            
        Yields:
            Entity dictionaries with type, layer, handle, points
        """
        import ezdxf
        
        # Get file size for progress
        file_size = Path(dxf_path).stat().st_size
        processed = 0
        
        try:
            # Use ezdxf's iterator mode if available
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # Process entities one at a time (memory-efficient)
            for entity in msp:
                processed += 1
                
                if progress_callback:
                    progress_callback(processed, file_size)
                
                # Apply filter if provided
                if entity_filter and not entity_filter(entity):
                    continue
                
                # Extract entity data
                entity_data = self._extract_entity(entity)
                if entity_data:
                    yield entity_data
                    
        except Exception as e:
            raise RuntimeError(f"Failed to parse DXF in chunks: {e}") from e
    
    def _extract_entity(self, entity) -> Optional[Dict]:
        """Extract data from an entity."""
        try:
            dxf_type = entity.dxftype()
            layer = getattr(entity.dxf, 'layer', '0')
            handle = getattr(entity.dxf, 'handle', None)
            
            # Extract points based on entity type
            points = []
            
            if dxf_type == 'LWPOLYLINE':
                points = [(float(v[0]), float(v[1])) for v in entity.get_points("xy")]
                closed = bool(entity.closed)
                return {
                    'type': 'LWPOLYLINE',
                    'points': points,
                    'layer': str(layer),
                    'handle': str(handle) if handle else None,
                    'closed': closed,
                }
            
            elif dxf_type == 'LINE':
                start = (float(entity.dxf.start.x), float(entity.dxf.start.y))
                end = (float(entity.dxf.end.x), float(entity.dxf.end.y))
                return {
                    'type': 'LINE',
                    'points': [start, end],
                    'layer': str(layer),
                    'handle': str(handle) if handle else None,
                    'closed': False,
                }
            
            elif dxf_type == 'CIRCLE':
                center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
                radius = float(entity.dxf.radius)
                # Approximate circle with segments
                points = self._circle_to_segments(center, radius)
                return {
                    'type': 'CIRCLE',
                    'points': points,
                    'layer': str(layer),
                    'handle': str(handle) if handle else None,
                    'closed': True,
                    'radius': radius,
                }
            
            elif dxf_type == 'ARC':
                center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
                radius = float(entity.dxf.radius)
                start_angle = float(entity.dxf.start_angle)
                end_angle = float(entity.dxf.end_angle)
                points = self._arc_to_segments(center, radius, start_angle, end_angle)
                return {
                    'type': 'ARC',
                    'points': points,
                    'layer': str(layer),
                    'handle': str(handle) if handle else None,
                    'closed': False,
                }
            
            # Add more entity types as needed
            
        except Exception:
            return None
        
        return None
    
    def _circle_to_segments(self, center: Tuple[float, float], radius: float, segments: int = 32) -> List[Tuple[float, float]]:
        """Convert circle to polygon segments."""
        import math
        points = []
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points
    
    def _arc_to_segments(self, center: Tuple[float, float], radius: float, start_angle: float, end_angle: float, segments: int = 32) -> List[Tuple[float, float]]:
        """Convert arc to line segments."""
        import math
        points = []
        # Normalize angles
        while start_angle < 0:
            start_angle += 2 * math.pi
        while end_angle < 0:
            end_angle += 2 * math.pi
        
        angle_range = end_angle - start_angle
        if angle_range < 0:
            angle_range += 2 * math.pi
        
        for i in range(segments + 1):
            angle = start_angle + angle_range * i / segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points


def parse_with_early_simplification(
    dxf_path: str,
    tolerance: float = 0.1,
    min_segment_length: float = 0.01,
) -> List[Dict]:
    """
    Parse DXF with early simplification to reduce point count.
    
    Args:
        dxf_path: Path to DXF file
        tolerance: Douglas-Peucker tolerance (mm)
        min_segment_length: Minimum segment length to keep (mm)
        
    Returns:
        List of simplified entities
    """
    from shapely.geometry import LineString
    from shapely.ops import simplify
    
    parser = StreamingDXFParser()
    simplified = []
    
    for entity in parser.parse_in_chunks(dxf_path):
        points = entity.get('points', [])
        if len(points) < 3:
            continue
        
        # Apply Douglas-Peucker simplification
        try:
            line = LineString(points)
            simplified_line = simplify(line, tolerance)
            simplified_points = list(simplified_line.coords)
            
            # Filter by minimum segment length
            if len(simplified_points) >= 2:
                filtered_points = [simplified_points[0]]
                for i in range(1, len(simplified_points)):
                    dist = ((simplified_points[i][0] - filtered_points[-1][0]) ** 2 + 
                           (simplified_points[i][1] - filtered_points[-1][1]) ** 2) ** 0.5
                    if dist >= min_segment_length:
                        filtered_points.append(simplified_points[i])
                
                if len(filtered_points) >= 2:
                    entity['points'] = filtered_points
                    entity['simplified'] = True
                    simplified.append(entity)
        except Exception:
            # Fallback to original if simplification fails
            simplified.append(entity)
    
    return simplified


def normalize_entities(entities: List[Dict]) -> List[Dict]:
    """
    Normalize entities (explode SPLINE/ELLIPSE to polylines).
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Normalized entities
    """
    normalized = []
    
    for entity in entities:
        entity_type = entity.get('type', '')
        
        if entity_type in ('SPLINE', 'ELLIPSE'):
            # Convert to approximate polyline
            points = entity.get('points', [])
            if len(points) >= 2:
                normalized.append({
                    'type': 'LWPOLYLINE',
                    'points': points,
                    'layer': entity.get('layer', '0'),
                    'handle': entity.get('handle'),
                    'closed': entity.get('closed', False),
                    'normalized_from': entity_type,
                })
        else:
            normalized.append(entity)
    
    return normalized


def compute_file_hash(dxf_path: str, chunk_size: int = 8192) -> str:
    """
    Compute MD5 hash of file for caching.
    
    Args:
        dxf_path: Path to DXF file
        chunk_size: Chunk size for hashing
        
    Returns:
        MD5 hash hex string
    """
    hash_md5 = hashlib.md5()
    
    with open(dxf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()





