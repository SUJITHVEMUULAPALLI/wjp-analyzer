"""
Performance Optimized DXF Analysis
==================================

High-performance DXF analysis with caching, parallel processing, and optimized algorithms.
"""

import os
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import ezdxf
from ezdxf.math import Vec3
import threading
from dataclasses import dataclass

from ..monitoring.metrics import track_dxf_analysis, metrics_collector
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Global cache for DXF parsing results
_dxf_cache = {}
_cache_lock = threading.Lock()


@dataclass
class PerformanceConfig:
    """Performance configuration for DXF analysis."""
    max_workers: int = min(mp.cpu_count(), 8)
    cache_size: int = 1000
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    simplify_tolerance: float = 0.1
    max_polygon_vertices: int = 1000


class OptimizedDXFAnalyzer:
    """High-performance DXF analyzer with caching and parallel processing."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.cache_manager = CacheManager()
        self._setup_performance_monitoring()
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        self.start_time = 0
        self.stage_times = {}
    
    def _start_timer(self, stage: str = 'total'):
        """Start timing a stage."""
        self.stage_times[stage] = time.time()
        if stage == 'total':
            self.start_time = time.time()
    
    def _end_timer(self, stage: str) -> float:
        """End timing a stage and return duration."""
        if stage in self.stage_times:
            duration = time.time() - self.stage_times[stage]
            logger.debug(f"{stage} took {duration:.3f}s")
            return duration
        return 0.0
    
    @lru_cache(maxsize=1000)
    def _parse_dxf_cached(self, file_path: str, file_hash: str) -> Dict[str, Any]:
        """Cached DXF parsing with file hash validation."""
        try:
            doc = ezdxf.readfile(file_path)
            msp = doc.modelspace()
            
            # Extract entities efficiently
            entities = {
                'lines': [],
                'arcs': [],
                'circles': [],
                'polylines': [],
                'lwpolylines': [],
                'splines': [],
                'blocks': []
            }
            
            for entity in msp:
                entity_type = entity.dxftype()
                if entity_type in entities:
                    entities[entity_type].append(entity)
            
            return {
                'entities': entities,
                'bounds': msp.extents(),
                'layer_count': len(doc.layers),
                'entity_count': len(msp)
            }
            
        except Exception as e:
            logger.error(f"Failed to parse DXF file {file_path}: {e}")
            raise
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for caching."""
        try:
            stat = os.stat(file_path)
            return f"{stat.st_mtime}_{stat.st_size}"
        except OSError:
            return str(time.time())
    
    def _extract_polygons_parallel(self, entities: Dict[str, Any]) -> List[Polygon]:
        """Extract polygons using parallel processing."""
        polygons = []
        
        if not self.config.enable_parallel_processing:
            return self._extract_polygons_sequential(entities)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            # Process different entity types in parallel
            for entity_type, entity_list in entities.items():
                if entity_list:
                    future = executor.submit(
                        self._process_entities_batch, 
                        entity_type, 
                        entity_list
                    )
                    futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    batch_polygons = future.result(timeout=30)
                    polygons.extend(batch_polygons)
                except Exception as e:
                    logger.error(f"Error processing entity batch: {e}")
        
        return polygons
    
    def _extract_polygons_sequential(self, entities: Dict[str, Any]) -> List[Polygon]:
        """Extract polygons sequentially (fallback)."""
        polygons = []
        
        for entity_type, entity_list in entities.items():
            batch_polygons = self._process_entities_batch(entity_type, entity_list)
            polygons.extend(batch_polygons)
        
        return polygons
    
    def _process_entities_batch(self, entity_type: str, entities: List) -> List[Polygon]:
        """Process a batch of entities of the same type."""
        polygons = []
        
        try:
            if entity_type == 'lwpolylines':
                polygons.extend(self._process_lwpolylines(entities))
            elif entity_type == 'polylines':
                polygons.extend(self._process_polylines(entities))
            elif entity_type == 'circles':
                polygons.extend(self._process_circles(entities))
            elif entity_type == 'arcs':
                polygons.extend(self._process_arcs(entities))
            elif entity_type == 'lines':
                polygons.extend(self._process_lines(entities))
                
        except Exception as e:
            logger.error(f"Error processing {entity_type}: {e}")
        
        return polygons
    
    def _process_lwpolylines(self, entities: List) -> List[Polygon]:
        """Process LWPOLYLINE entities efficiently."""
        polygons = []
        
        for entity in entities:
            try:
                points = list(entity.get_points())
                if len(points) >= 3:
                    # Convert to numpy for faster processing
                    coords = np.array([(p[0], p[1]) for p in points])
                    
                    # Simplify if too many vertices
                    if len(coords) > self.config.max_polygon_vertices:
                        coords = self._simplify_coordinates(coords)
                    
                    if len(coords) >= 3:
                        polygon = Polygon(coords)
                        if polygon.is_valid:
                            polygons.append(polygon)
                            
            except Exception as e:
                logger.debug(f"Error processing LWPOLYLINE: {e}")
                continue
        
        return polygons
    
    def _process_polylines(self, entities: List) -> List[Polygon]:
        """Process POLYLINE entities efficiently."""
        polygons = []
        
        for entity in entities:
            try:
                points = []
                for vertex in entity.vertices:
                    points.append((vertex.dxf.location.x, vertex.dxf.location.y))
                
                if len(points) >= 3:
                    coords = np.array(points)
                    
                    # Simplify if too many vertices
                    if len(coords) > self.config.max_polygon_vertices:
                        coords = self._simplify_coordinates(coords)
                    
                    if len(coords) >= 3:
                        polygon = Polygon(coords)
                        if polygon.is_valid:
                            polygons.append(polygon)
                            
            except Exception as e:
                logger.debug(f"Error processing POLYLINE: {e}")
                continue
        
        return polygons
    
    def _process_circles(self, entities: List) -> List[Polygon]:
        """Process CIRCLE entities efficiently."""
        polygons = []
        
        for entity in entities:
            try:
                center = entity.dxf.center
                radius = entity.dxf.radius
                
                # Create circle as polygon with reasonable number of segments
                segments = max(16, min(64, int(radius * 4)))
                angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
                
                coords = []
                for angle in angles:
                    x = center.x + radius * np.cos(angle)
                    y = center.y + radius * np.sin(angle)
                    coords.append((x, y))
                
                polygon = Polygon(coords)
                if polygon.is_valid:
                    polygons.append(polygon)
                    
            except Exception as e:
                logger.debug(f"Error processing CIRCLE: {e}")
                continue
        
        return polygons
    
    def _process_arcs(self, entities: List) -> List[Polygon]:
        """Process ARC entities efficiently."""
        polygons = []
        
        for entity in entities:
            try:
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = np.radians(entity.dxf.start_angle)
                end_angle = np.radians(entity.dxf.end_angle)
                
                # Create arc as polygon
                segments = max(8, min(32, int(radius * 2)))
                angles = np.linspace(start_angle, end_angle, segments)
                
                coords = []
                for angle in angles:
                    x = center.x + radius * np.cos(angle)
                    y = center.y + radius * np.sin(angle)
                    coords.append((x, y))
                
                # Close the arc
                coords.append((center.x, center.y))
                
                polygon = Polygon(coords)
                if polygon.is_valid:
                    polygons.append(polygon)
                    
            except Exception as e:
                logger.debug(f"Error processing ARC: {e}")
                continue
        
        return polygons
    
    def _process_lines(self, entities: List) -> List[Polygon]:
        """Process LINE entities efficiently."""
        polygons = []
        
        # Group lines by layer for potential polygon reconstruction
        lines_by_layer = {}
        for entity in entities:
            layer = entity.dxf.layer
            if layer not in lines_by_layer:
                lines_by_layer[layer] = []
            lines_by_layer[layer].append(entity)
        
        # Try to reconstruct polygons from lines
        for layer, lines in lines_by_layer.items():
            try:
                # Simple line-to-polygon conversion (can be enhanced)
                if len(lines) >= 3:
                    points = []
                    for line in lines:
                        points.append((line.dxf.start.x, line.dxf.start.y))
                        points.append((line.dxf.end.x, line.dxf.end.y))
                    
                    # Remove duplicates and create polygon
                    unique_points = list(set(points))
                    if len(unique_points) >= 3:
                        coords = np.array(unique_points)
                        polygon = Polygon(coords)
                        if polygon.is_valid:
                            polygons.append(polygon)
                            
            except Exception as e:
                logger.debug(f"Error processing lines for layer {layer}: {e}")
                continue
        
        return polygons
    
    def _simplify_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Simplify coordinates using Douglas-Peucker algorithm."""
        try:
            from scipy.spatial.distance import pdist, squareform
            
            # Simple simplification - keep every nth point
            if len(coords) > self.config.max_polygon_vertices:
                step = len(coords) // self.config.max_polygon_vertices
                return coords[::max(1, step)]
            
            return coords
            
        except ImportError:
            # Fallback to simple decimation
            if len(coords) > self.config.max_polygon_vertices:
                step = len(coords) // self.config.max_polygon_vertices
                return coords[::max(1, step)]
            
            return coords
    
    def _calculate_metrics_parallel(self, polygons: List[Polygon]) -> Dict[str, Any]:
        """Calculate metrics using parallel processing."""
        if not polygons:
            return self._get_empty_metrics()
        
        if not self.config.enable_parallel_processing:
            return self._calculate_metrics_sequential(polygons)
        
        # Split polygons into batches for parallel processing
        batch_size = max(1, len(polygons) // self.config.max_workers)
        batches = [polygons[i:i + batch_size] for i in range(0, len(polygons), batch_size)]
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self._calculate_batch_metrics, batch) for batch in batches]
            
            # Collect results
            batch_results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error calculating batch metrics: {e}")
                    batch_results.append(self._get_empty_metrics())
        
        # Combine batch results
        return self._combine_batch_metrics(batch_results)
    
    def _calculate_metrics_sequential(self, polygons: List[Polygon]) -> Dict[str, Any]:
        """Calculate metrics sequentially (fallback)."""
        if not polygons:
            return self._get_empty_metrics()
        
        return self._calculate_batch_metrics(polygons)
    
    def _calculate_batch_metrics(self, polygons: List[Polygon]) -> Dict[str, Any]:
        """Calculate metrics for a batch of polygons."""
        if not polygons:
            return self._get_empty_metrics()
        
        try:
            # Calculate basic metrics
            total_area = sum(polygon.area for polygon in polygons if polygon.is_valid)
            total_perimeter = sum(polygon.length for polygon in polygons if polygon.is_valid)
            
            # Calculate bounding box
            all_bounds = [polygon.bounds for polygon in polygons if polygon.is_valid]
            if all_bounds:
                minx = min(bounds[0] for bounds in all_bounds)
                miny = min(bounds[1] for bounds in all_bounds)
                maxx = max(bounds[2] for bounds in all_bounds)
                maxy = max(bounds[3] for bounds in all_bounds)
                bounding_box = (minx, miny, maxx, maxy)
            else:
                bounding_box = (0, 0, 0, 0)
            
            # Calculate complexity metrics
            vertex_counts = [len(polygon.exterior.coords) for polygon in polygons if polygon.is_valid]
            avg_vertices = np.mean(vertex_counts) if vertex_counts else 0
            
            # Calculate pierce points (simplified)
            pierce_points = len(polygons)  # Simplified calculation
            
            return {
                'polygons_count': len(polygons),
                'valid_polygons': len([p for p in polygons if p.is_valid]),
                'total_area': total_area,
                'total_perimeter': total_perimeter,
                'bounding_box': bounding_box,
                'avg_vertices_per_polygon': avg_vertices,
                'pierce_points': pierce_points,
                'complexity_score': len(polygons) * avg_vertices
            }
            
        except Exception as e:
            logger.error(f"Error calculating batch metrics: {e}")
            return self._get_empty_metrics()
    
    def _combine_batch_metrics(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine metrics from multiple batches."""
        if not batch_results:
            return self._get_empty_metrics()
        
        combined = {
            'polygons_count': sum(result.get('polygons_count', 0) for result in batch_results),
            'valid_polygons': sum(result.get('valid_polygons', 0) for result in batch_results),
            'total_area': sum(result.get('total_area', 0) for result in batch_results),
            'total_perimeter': sum(result.get('total_perimeter', 0) for result in batch_results),
            'pierce_points': sum(result.get('pierce_points', 0) for result in batch_results),
            'complexity_score': sum(result.get('complexity_score', 0) for result in batch_results)
        }
        
        # Calculate averages
        valid_results = [r for r in batch_results if r.get('polygons_count', 0) > 0]
        if valid_results:
            combined['avg_vertices_per_polygon'] = np.mean([
                r.get('avg_vertices_per_polygon', 0) for r in valid_results
            ])
        else:
            combined['avg_vertices_per_polygon'] = 0
        
        # Calculate overall bounding box
        all_bounds = [r.get('bounding_box', (0, 0, 0, 0)) for r in batch_results if r.get('bounding_box')]
        if all_bounds:
            minx = min(bounds[0] for bounds in all_bounds)
            miny = min(bounds[1] for bounds in all_bounds)
            maxx = max(bounds[2] for bounds in all_bounds)
            maxy = max(bounds[3] for bounds in all_bounds)
            combined['bounding_box'] = (minx, miny, maxx, maxy)
        else:
            combined['bounding_box'] = (0, 0, 0, 0)
        
        return combined
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Get empty metrics structure."""
        return {
            'polygons_count': 0,
            'valid_polygons': 0,
            'total_area': 0,
            'total_perimeter': 0,
            'bounding_box': (0, 0, 0, 0),
            'avg_vertices_per_polygon': 0,
            'pierce_points': 0,
            'complexity_score': 0
        }
    
    @track_dxf_analysis
    def analyze_dxf_optimized(self, file_path: str, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimized DXF analysis with performance monitoring.
        
        Args:
            file_path: Path to DXF file
            analysis_params: Analysis parameters
            
        Returns:
            Analysis results with performance metrics
        """
        self._start_timer('total')
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = f"dxf_analysis:{file_path}:{hash(str(analysis_params))}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    logger.debug(f"Using cached result for {file_path}")
                    return cached_result
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_hash = self._get_file_hash(file_path)
            
            # Parse DXF file
            self._start_timer('parsing')
            parsed_data = self._parse_dxf_cached(file_path, file_hash)
            parse_time = self._end_timer('parsing')
            
            # Extract polygons
            self._start_timer('polygon_extraction')
            polygons = self._extract_polygons_parallel(parsed_data['entities'])
            extraction_time = self._end_timer('polygon_extraction')
            
            # Calculate metrics
            self._start_timer('metrics_calculation')
            metrics = self._calculate_metrics_parallel(polygons)
            metrics_time = self._end_timer('metrics_calculation')
            
            # Calculate cost estimation
            self._start_timer('cost_estimation')
            cost_estimation = self._estimate_cutting_cost(metrics, analysis_params)
            cost_time = self._end_timer('cost_estimation')
            
            # Generate quality assessment
            self._start_timer('quality_assessment')
            quality_assessment = self._assess_cutting_quality(metrics, analysis_params)
            quality_time = self._end_timer('quality_assessment')
            
            total_time = self._end_timer('total')
            
            # Compile results
            result = {
                'file_info': {
                    'path': file_path,
                    'size_bytes': file_size,
                    'hash': file_hash
                },
                'parsing_info': {
                    'entity_count': parsed_data['entity_count'],
                    'layer_count': parsed_data['layer_count'],
                    'bounds': parsed_data['bounds']
                },
                'metrics': metrics,
                'cost_estimation': cost_estimation,
                'quality_assessment': quality_assessment,
                'performance': {
                    'total_time': total_time,
                    'parse_time': parse_time,
                    'extraction_time': extraction_time,
                    'metrics_time': metrics_time,
                    'cost_time': cost_time,
                    'quality_time': quality_time,
                    'polygons_per_second': metrics['polygons_count'] / max(total_time, 0.001)
                }
            }
            
            # Cache result
            if self.config.enable_caching:
                self.cache_manager.set(cache_key, result, ttl=3600)  # 1 hour TTL
            
            # Record metrics
            metrics_collector.record_dxf_analysis(
                status='success',
                duration=total_time,
                file_size=file_size,
                polygons_count=metrics['polygons_count']
            )
            
            logger.info(f"DXF analysis completed in {total_time:.3f}s for {file_path}")
            return result
            
        except Exception as e:
            total_time = self._end_timer('total')
            
            # Record error metrics
            metrics_collector.record_dxf_analysis(
                status='failure',
                duration=total_time,
                file_size=0,
                polygons_count=0
            )
            
            logger.error(f"DXF analysis failed for {file_path}: {e}")
            raise
    
    def _estimate_cutting_cost(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cutting cost based on metrics and parameters."""
        try:
            # Cost factors
            material_cost_per_mm2 = params.get('material_cost_per_mm2', 0.001)
            cutting_cost_per_mm = params.get('cutting_cost_per_mm', 0.01)
            pierce_cost = params.get('pierce_cost', 0.1)
            
            # Calculate costs
            material_cost = metrics['total_area'] * material_cost_per_mm2
            cutting_cost = metrics['total_perimeter'] * cutting_cost_per_mm
            pierce_cost_total = metrics['pierce_points'] * pierce_cost
            
            total_cost = material_cost + cutting_cost + pierce_cost_total
            
            return {
                'material_cost': material_cost,
                'cutting_cost': cutting_cost,
                'pierce_cost': pierce_cost_total,
                'total_cost': total_cost,
                'cost_per_polygon': total_cost / max(metrics['polygons_count'], 1),
                'cost_per_mm2': total_cost / max(metrics['total_area'], 1)
            }
            
        except Exception as e:
            logger.error(f"Error estimating cutting cost: {e}")
            return {'total_cost': 0, 'error': str(e)}
    
    def _assess_cutting_quality(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cutting quality based on metrics and parameters."""
        try:
            # Quality factors
            complexity_score = metrics.get('complexity_score', 0)
            avg_vertices = metrics.get('avg_vertices_per_polygon', 0)
            polygons_count = metrics.get('polygons_count', 0)
            
            # Calculate quality scores
            complexity_quality = max(0, 100 - (complexity_score / 1000))
            detail_quality = max(0, 100 - (avg_vertices / 10))
            efficiency_quality = min(100, polygons_count * 2)
            
            overall_quality = (complexity_quality + detail_quality + efficiency_quality) / 3
            
            return {
                'overall_quality': overall_quality,
                'complexity_quality': complexity_quality,
                'detail_quality': detail_quality,
                'efficiency_quality': efficiency_quality,
                'recommendations': self._generate_quality_recommendations(metrics, params)
            }
            
        except Exception as e:
            logger.error(f"Error assessing cutting quality: {e}")
            return {'overall_quality': 0, 'error': str(e)}
    
    def _generate_quality_recommendations(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if metrics['avg_vertices_per_polygon'] > 50:
            recommendations.append("Consider simplifying complex polygons for better cutting quality")
        
        if metrics['polygons_count'] > 1000:
            recommendations.append("Large number of polygons may require optimized cutting strategy")
        
        if metrics['complexity_score'] > 10000:
            recommendations.append("High complexity design - consider breaking into smaller sections")
        
        if not recommendations:
            recommendations.append("Design appears suitable for waterjet cutting")
        
        return recommendations


# Global optimized analyzer instance
optimized_analyzer = OptimizedDXFAnalyzer()


def analyze_dxf_fast(file_path: str, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast DXF analysis function with performance optimization.
    
    Args:
        file_path: Path to DXF file
        analysis_params: Analysis parameters
        
    Returns:
        Analysis results
    """
    return optimized_analyzer.analyze_dxf_optimized(file_path, analysis_params)
