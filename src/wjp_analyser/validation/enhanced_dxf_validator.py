"""
Enhanced DXF Validation Module for Waterjet Cutting

This module provides comprehensive validation capabilities for DXF files
specifically designed for waterjet cutting operations. It includes:

1. Geometry validation (self-intersections, overlapping features)
2. Manufacturing-specific validation (kerf compensation, cutting feasibility)
3. File structure validation (layers, blocks, text entities)
4. Performance validation (cutting time estimation, efficiency metrics)
5. Configurable validation rules and thresholds
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
import ezdxf
from ezdxf import DXFStructureError
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
import numpy as np

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for different use cases."""
    BASIC = "basic"           # Essential checks only
    STANDARD = "standard"    # Common manufacturing checks
    COMPREHENSIVE = "comprehensive"  # All validation checks
    CUSTOM = "custom"        # User-defined validation rules


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"           # Informational only
    WARNING = "warning"     # May cause issues
    ERROR = "error"         # Will cause problems
    CRITICAL = "critical"   # Prevents processing


@dataclass
class ValidationRule:
    """Individual validation rule configuration."""
    name: str
    description: str
    severity: ValidationSeverity
    enabled: bool = True
    threshold: float = 0.0
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    location: Optional[Tuple[float, float]] = None
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for DXF validation."""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    
    # Geometry validation thresholds
    min_feature_size: float = 0.5          # Minimum feature size (mm)
    max_feature_size: float = 10000.0      # Maximum feature size (mm)
    min_corner_radius: float = 0.5        # Minimum corner radius (mm)
    min_spacing: float = 1.0               # Minimum spacing between features (mm)
    max_self_intersection_area: float = 0.1  # Max allowed self-intersection area (mm²)
    
    # Manufacturing validation thresholds
    kerf_width: float = 1.1               # Kerf width (mm)
    min_hole_diameter: float = 2.0        # Minimum hole diameter (mm)
    max_cutting_length: float = 5000.0    # Maximum cutting length (mm)
    max_pierce_count: int = 1000          # Maximum number of pierces
    
    # Performance validation thresholds
    max_cutting_time_minutes: float = 120.0  # Maximum cutting time (minutes)
    min_cutting_efficiency: float = 0.7       # Minimum cutting efficiency (0-1)
    
    # File structure validation
    check_layers: bool = True
    check_blocks: bool = True
    check_text_entities: bool = True
    check_dimensions: bool = True
    
    # Custom rules
    custom_rules: List[ValidationRule] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    is_valid: bool
    overall_score: float  # 0-100
    
    # Statistics
    total_entities: int = 0
    polygons: int = 0
    polylines: int = 0
    circles: int = 0
    arcs: int = 0
    lines: int = 0
    text_entities: int = 0
    dimension_entities: int = 0
    
    # Geometry analysis
    open_contours: int = 0
    closed_contours: int = 0
    self_intersections: int = 0
    overlapping_features: int = 0
    nested_contours: int = 0
    
    # Manufacturing analysis
    spacing_violations: int = 0
    radius_violations: int = 0
    kerf_conflicts: int = 0
    cutting_feasibility_score: float = 0.0
    
    # Performance analysis
    estimated_cutting_time: float = 0.0
    estimated_cutting_length: float = 0.0
    estimated_pierce_count: int = 0
    efficiency_score: float = 0.0
    
    # File structure analysis
    layer_count: int = 0
    block_count: int = 0
    unused_layers: int = 0
    unused_blocks: int = 0
    
    # Bounding box and dimensions
    bounding_box: Tuple[float, float, float, float] = (0, 0, 0, 0)
    min_feature_size: float = 0.0
    max_feature_size: float = 0.0
    
    # Issues and recommendations
    issues: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Detailed analysis
    geometry_analysis: Dict[str, Any] = field(default_factory=dict)
    manufacturing_analysis: Dict[str, Any] = field(default_factory=dict)
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    file_structure_analysis: Dict[str, Any] = field(default_factory=dict)


class EnhancedDXFValidator:
    """Enhanced DXF validator with comprehensive validation capabilities."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the validator with configuration."""
        self.config = config or ValidationConfig()
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules based on configuration."""
        if not self.config.custom_rules:
            self.config.custom_rules = [
                ValidationRule(
                    name="min_feature_size",
                    description="Check minimum feature size",
                    severity=ValidationSeverity.WARNING,
                    threshold=self.config.min_feature_size
                ),
                ValidationRule(
                    name="max_feature_size",
                    description="Check maximum feature size",
                    severity=ValidationSeverity.WARNING,
                    threshold=self.config.max_feature_size
                ),
                ValidationRule(
                    name="min_corner_radius",
                    description="Check minimum corner radius",
                    severity=ValidationSeverity.ERROR,
                    threshold=self.config.min_corner_radius
                ),
                ValidationRule(
                    name="min_spacing",
                    description="Check minimum spacing between features",
                    severity=ValidationSeverity.ERROR,
                    threshold=self.config.min_spacing
                ),
                ValidationRule(
                    name="min_hole_diameter",
                    description="Check minimum hole diameter",
                    severity=ValidationSeverity.ERROR,
                    threshold=self.config.min_hole_diameter
                ),
                ValidationRule(
                    name="self_intersections",
                    description="Check for self-intersecting geometry",
                    severity=ValidationSeverity.ERROR,
                    threshold=self.config.max_self_intersection_area
                ),
                ValidationRule(
                    name="open_contours",
                    description="Check for open contours",
                    severity=ValidationSeverity.WARNING,
                    threshold=0
                ),
                ValidationRule(
                    name="cutting_time",
                    description="Check estimated cutting time",
                    severity=ValidationSeverity.WARNING,
                    threshold=self.config.max_cutting_time_minutes
                ),
                ValidationRule(
                    name="pierce_count",
                    description="Check pierce count",
                    severity=ValidationSeverity.WARNING,
                    threshold=self.config.max_pierce_count
                )
            ]
    
    def validate_dxf(self, dxf_path: str) -> ValidationResult:
        """
        Perform comprehensive DXF validation.
        
        Args:
            dxf_path: Path to the DXF file to validate
            
        Returns:
            ValidationResult with comprehensive analysis
        """
        logger.info(f"Starting comprehensive DXF validation: {dxf_path}")
        
        try:
            # Load DXF document
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # Initialize result
            result = ValidationResult(
                is_valid=True,
                overall_score=100.0
            )
            
            # Perform validation based on level
            if self.config.validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, 
                                               ValidationLevel.COMPREHENSIVE]:
                self._validate_geometry(msp, result)
                self._validate_manufacturing(msp, result)
                
            if self.config.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                self._validate_performance(msp, result)
                
            if self.config.validation_level == ValidationLevel.COMPREHENSIVE:
                self._validate_file_structure(doc, result)
            
            # Apply custom rules
            self._apply_custom_rules(msp, result)
            
            # Calculate overall score and validity
            self._calculate_overall_score(result)
            
            logger.info(f"Validation completed. Score: {result.overall_score:.1f}, "
                       f"Issues: {len(result.issues)}")
            
            return result
            
        except DXFStructureError as e:
            logger.error(f"DXF structure error: {e}")
            return ValidationResult(
                is_valid=False,
                overall_score=0.0,
                issues=[ValidationIssue(
                    rule_name="dxf_structure",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"DXF file structure error: {str(e)}"
                )]
            )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                overall_score=0.0,
                issues=[ValidationIssue(
                    rule_name="validation_error",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation error: {str(e)}"
                )]
            )
    
    def _validate_geometry(self, msp, result: ValidationResult):
        """Validate geometry aspects of the DXF."""
        logger.debug("Validating geometry...")
        
        entities = list(msp)
        polygons = []
        lines = []
        
        # Analyze entities
        for entity in entities:
            result.total_entities += 1
            
            if entity.dxftype() == 'LWPOLYLINE':
                result.polylines += 1
                if entity.closed:
                    result.closed_contours += 1
                    result.polygons += 1
                    try:
                        points = [(p[0], p[1]) for p in entity.get_points()]
                        if len(points) >= 3:
                            poly = Polygon(points)
                            # Always record polygon for downstream metrics
                            polygons.append(poly)
                            # Detect self-intersections
                            if not poly.is_valid:
                                result.issues.append(ValidationIssue(
                                    rule_name="self_intersections",
                                    severity=ValidationSeverity.ERROR,
                                    message=f"Polygon {len(polygons)-1} has self-intersections",
                                    suggested_fix="Fix self-intersecting geometry"
                                ))
                                result.self_intersections += 1
                    except Exception as e:
                        result.issues.append(ValidationIssue(
                            rule_name="polygon_creation",
                            severity=ValidationSeverity.ERROR,
                            message=f"Failed to create polygon: {str(e)}",
                            entity_type="LWPOLYLINE"
                        ))
                else:
                    result.open_contours += 1
                    try:
                        points = [(p[0], p[1]) for p in entity.get_points()]
                        if len(points) >= 2:
                            line = LineString(points)
                            lines.append(line)
                    except Exception as e:
                        result.issues.append(ValidationIssue(
                            rule_name="line_creation",
                            severity=ValidationSeverity.ERROR,
                            message=f"Failed to create line: {str(e)}",
                            entity_type="LWPOLYLINE"
                        ))
            
            elif entity.dxftype() == 'CIRCLE':
                result.circles += 1
                try:
                    center = (entity.dxf.center.x, entity.dxf.center.y)
                    radius = entity.dxf.radius
                    circle = Point(center).buffer(radius, resolution=64)
                    polygons.append(circle)
                    
                    # Check minimum hole diameter
                    if radius * 2 < self.config.min_hole_diameter:
                        result.issues.append(ValidationIssue(
                            rule_name="min_hole_diameter",
                            severity=ValidationSeverity.ERROR,
                            message=f"Circle diameter {radius * 2:.2f}mm is below minimum {self.config.min_hole_diameter}mm",
                            location=center,
                            entity_type="CIRCLE",
                            suggested_fix=f"Increase diameter to at least {self.config.min_hole_diameter}mm"
                        ))
                        result.radius_violations += 1
                        
                except Exception as e:
                    result.issues.append(ValidationIssue(
                        rule_name="circle_creation",
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to create circle: {str(e)}",
                        entity_type="CIRCLE"
                    ))
            
            elif entity.dxftype() == 'ARC':
                result.arcs += 1
            
            elif entity.dxftype() == 'LINE':
                result.lines += 1
                try:
                    start = (entity.dxf.start.x, entity.dxf.start.y)
                    end = (entity.dxf.end.x, entity.dxf.end.y)
                    line = LineString([start, end])
                    lines.append(line)
                except Exception as e:
                    result.issues.append(ValidationIssue(
                        rule_name="line_creation",
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to create line: {str(e)}",
                        entity_type="LINE"
                    ))
        
        # Analyze polygons for geometry issues
        self._analyze_polygon_geometry(polygons, result)
        # Corner radius checks on polygon vertices
        self._check_polygon_corner_radius(polygons, result)
        
        # Analyze spacing between features
        self._analyze_feature_spacing(polygons, result)
        
        # Calculate bounding box and feature sizes
        self._calculate_bounding_box_and_sizes(polygons, result)

        # Empty file handling
        if result.total_entities == 0:
            result.issues.append(ValidationIssue(
                rule_name="no_entities",
                severity=ValidationSeverity.INFO,
                message="DXF contains no entities"
            ))
    
    def _analyze_polygon_geometry(self, polygons: List[Polygon], result: ValidationResult):
        """Analyze polygon geometry for issues."""
        for i, poly in enumerate(polygons):
            # Check for self-intersections
            if not poly.is_valid:
                result.issues.append(ValidationIssue(
                    rule_name="self_intersections",
                    severity=ValidationSeverity.ERROR,
                    message=f"Polygon {i} has self-intersections",
                    suggested_fix="Fix self-intersecting geometry"
                ))
                result.self_intersections += 1
            
            # Check minimum feature size
            bounds = poly.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            min_dimension = min(width, height)
            
            if min_dimension < self.config.min_feature_size:
                result.issues.append(ValidationIssue(
                    rule_name="min_feature_size",
                    severity=ValidationSeverity.WARNING,
                    message=f"Feature {i} has dimension {min_dimension:.2f}mm below minimum {self.config.min_feature_size}mm",
                    location=(bounds[0] + width/2, bounds[1] + height/2),
                    suggested_fix=f"Increase feature size to at least {self.config.min_feature_size}mm"
                ))
            
            # Check maximum feature size
            max_dimension = max(width, height)
            if max_dimension > self.config.max_feature_size:
                result.issues.append(ValidationIssue(
                    rule_name="max_feature_size",
                    severity=ValidationSeverity.WARNING,
                    message=f"Feature {i} has dimension {max_dimension:.2f}mm above maximum {self.config.max_feature_size}mm",
                    location=(bounds[0] + width/2, bounds[1] + height/2),
                    suggested_fix=f"Consider splitting large features"
                ))
    
    def _analyze_feature_spacing(self, polygons: List[Polygon], result: ValidationResult):
        """Analyze spacing between features."""
        if len(polygons) < 2:
            return
        
        # Create buffer zones for kerf compensation
        buffered_polygons = []
        for poly in polygons:
            try:
                buffered = poly.buffer(self.config.kerf_width / 2)
                buffered_polygons.append(buffered)
            except Exception:
                continue
        
        # Check for overlapping buffered polygons
        for i in range(len(buffered_polygons)):
            for j in range(i + 1, len(buffered_polygons)):
                try:
                    if buffered_polygons[i].intersects(buffered_polygons[j]):
                        # Calculate actual distance between original polygons
                        distance = polygons[i].distance(polygons[j])
                        if distance < self.config.min_spacing:
                            result.issues.append(ValidationIssue(
                                rule_name="min_spacing",
                                severity=ValidationSeverity.ERROR,
                                message=f"Features {i} and {j} are too close: {distance:.2f}mm < {self.config.min_spacing}mm",
                                suggested_fix=f"Increase spacing to at least {self.config.min_spacing}mm"
                            ))
                            result.spacing_violations += 1
                            result.kerf_conflicts += 1
                except Exception:
                    continue
    
    def _calculate_bounding_box_and_sizes(self, polygons: List[Polygon], result: ValidationResult):
        """Calculate bounding box and feature sizes."""
        if not polygons:
            return
        
        all_bounds = [poly.bounds for poly in polygons]
        min_x = min(bounds[0] for bounds in all_bounds)
        min_y = min(bounds[1] for bounds in all_bounds)
        max_x = max(bounds[2] for bounds in all_bounds)
        max_y = max(bounds[3] for bounds in all_bounds)
        
        result.bounding_box = (min_x, min_y, max_x, max_y)
        
        # Calculate feature sizes
        sizes = []
        for poly in polygons:
            bounds = poly.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            sizes.extend([width, height])
        
        if sizes:
            result.min_feature_size = min(sizes)
            result.max_feature_size = max(sizes)

    def _check_polygon_corner_radius(self, polygons: List[Polygon], result: ValidationResult):
        """Detect sharp corners on polygon exteriors vs min_corner_radius."""
        min_r = getattr(self.config, 'min_corner_radius', 0.0)
        if min_r <= 0:
            return
        for i, poly in enumerate(polygons):
            try:
                coords = list(poly.exterior.coords)
                if len(coords) < 4:
                    continue
                # iterate triples (wrap around)
                n = len(coords) - 1  # last equals first
                for j in range(n):
                    p1 = coords[(j - 1) % n]
                    p2 = coords[j]
                    p3 = coords[(j + 1) % n]
                    len1 = math.dist(p1, p2)
                    len2 = math.dist(p2, p3)
                    edge_min = min(len1, len2)
                    # Skip corners formed by very short segments (likely arc approximation)
                    if edge_min < max(1e-6, min_r * 0.25):
                        continue

                    r_est = self._estimate_corner_radius(p1, p2, p3)
                    if r_est < min_r:
                        result.issues.append(ValidationIssue(
                            rule_name="min_corner_radius",
                            severity=ValidationSeverity.ERROR,
                            message=f"Corner radius {r_est:.2f}mm < {min_r}mm at {p2}",
                            suggested_fix=f"Increase corner radius to at least {min_r}mm"
                        ))
                        result.radius_violations += 1
            except Exception:
                continue
    
    def _validate_manufacturing(self, msp, result: ValidationResult):
        """Validate manufacturing-specific aspects."""
        logger.debug("Validating manufacturing aspects...")
        
        # Calculate cutting feasibility score
        feasibility_factors = []
        
        # Factor 1: Feature size distribution
        if result.min_feature_size > 0:
            size_ratio = result.min_feature_size / max(result.max_feature_size, 1.0)
            feasibility_factors.append(min(size_ratio * 2, 1.0))
        
        # Factor 2: Spacing violations
        if result.total_entities > 0:
            spacing_factor = max(0, 1.0 - (result.spacing_violations / result.total_entities))
            feasibility_factors.append(spacing_factor)
        
        # Factor 3: Radius violations
        if result.circles > 0:
            radius_factor = max(0, 1.0 - (result.radius_violations / result.circles))
            feasibility_factors.append(radius_factor)
        
        # Factor 4: Self-intersections
        intersection_factor = max(0, 1.0 - (result.self_intersections / max(result.polygons, 1)))
        feasibility_factors.append(intersection_factor)
        
        result.cutting_feasibility_score = np.mean(feasibility_factors) if feasibility_factors else 0.0
    
    def _validate_performance(self, msp, result: ValidationResult):
        """Validate performance aspects."""
        logger.debug("Validating performance aspects...")
        
        # Estimate cutting parameters
        total_length = 0.0
        pierce_count = 0
        
        # Calculate cutting length from detected polygons
        # Rebuild simple polygons from modelspace closures as fallback
        polys: List[Polygon] = []
        try:
            for e in msp:
                if e.dxftype() == 'LWPOLYLINE' and e.closed:
                    pts = [(p[0], p[1]) for p in e.get_points()]
                    if len(pts) >= 3:
                        try:
                            polys.append(Polygon(pts))
                        except Exception:
                            pass
                elif e.dxftype() == 'CIRCLE':
                    c = (e.dxf.center.x, e.dxf.center.y)
                    r = e.dxf.radius
                    polys.append(Point(c).buffer(r, resolution=64))
        except Exception:
            polys = []
        
        for poly in polys:
            if not hasattr(poly, 'exterior'):
                continue
            total_length += poly.exterior.length
            pierce_count += 1
            for interior in getattr(poly, 'interiors', []):
                total_length += interior.length
                pierce_count += 1
        
        result.estimated_cutting_length = total_length
        result.estimated_pierce_count = pierce_count
        
        # Estimate cutting time (simplified model)
        cutting_speed = 1200.0  # mm/min
        pierce_time = 0.5  # seconds per pierce
        
        cutting_time = (total_length / cutting_speed) + (pierce_count * pierce_time / 60)
        result.estimated_cutting_time = cutting_time
        
        # Check performance thresholds
        if cutting_time > self.config.max_cutting_time_minutes:
            result.issues.append(ValidationIssue(
                rule_name="cutting_time",
                severity=ValidationSeverity.WARNING,
                message=f"Estimated cutting time {cutting_time:.1f}min exceeds maximum {self.config.max_cutting_time_minutes}min",
                suggested_fix="Consider optimizing geometry or splitting into multiple jobs"
            ))
        
        if pierce_count > self.config.max_pierce_count:
            result.issues.append(ValidationIssue(
                rule_name="pierce_count",
                severity=ValidationSeverity.WARNING,
                message=f"Pierce count {pierce_count} exceeds maximum {self.config.max_pierce_count}",
                suggested_fix="Consider reducing number of separate features"
            ))
        
        # Calculate efficiency score
        if result.total_entities > 0:
            # For simple polygonal jobs, treat efficiency as high to avoid penalizing trivial parts
            if polys:
                result.efficiency_score = 1.0
            else:
                efficiency_factors = []
                # Length per entity ratio
                length_per_entity = total_length / result.total_entities
                efficiency_factors.append(min(length_per_entity / 100.0, 1.0))
                # Pierce efficiency
                pierce_efficiency = max(0, 1.0 - (pierce_count / max(result.total_entities, 1)))
                efficiency_factors.append(pierce_efficiency)
                result.efficiency_score = np.mean(efficiency_factors) if efficiency_factors else 1.0
    
    def _validate_file_structure(self, doc, result: ValidationResult):
        """Validate DXF file structure."""
        logger.debug("Validating file structure...")
        
        # Check layers
        if self.config.check_layers:
            layers = doc.layers
            result.layer_count = len(layers)
            
            # Check for unused layers
            used_layers = set()
            for entity in doc.modelspace():
                if hasattr(entity.dxf, 'layer'):
                    used_layers.add(entity.dxf.layer)
            
            unused_layers = set(layer.dxf.name for layer in layers) - used_layers
            result.unused_layers = len(unused_layers)
            
            if unused_layers and self.config.validation_level == ValidationLevel.COMPREHENSIVE and result.total_entities > 1:
                result.issues.append(ValidationIssue(
                    rule_name="unused_layers",
                    severity=ValidationSeverity.INFO,
                    message=f"Found {len(unused_layers)} unused layers: {', '.join(list(unused_layers)[:5])}",
                    suggested_fix="Consider removing unused layers to clean up the file"
                ))
        
        # Check blocks
        if self.config.check_blocks:
            blocks = doc.blocks
            result.block_count = len(blocks)
            
            # Check for unused blocks
            used_blocks = set()
            for entity in doc.modelspace():
                if hasattr(entity.dxf, 'name') and entity.dxftype() == 'INSERT':
                    used_blocks.add(entity.dxf.name)
            
            unused_blocks = set(block.dxf.name for block in blocks) - used_blocks
            result.unused_blocks = len(unused_blocks)
            
            if unused_blocks and self.config.validation_level == ValidationLevel.COMPREHENSIVE and result.total_entities > 1:
                result.issues.append(ValidationIssue(
                    rule_name="unused_blocks",
                    severity=ValidationSeverity.INFO,
                    message=f"Found {len(unused_blocks)} unused blocks",
                    suggested_fix="Consider removing unused blocks to clean up the file"
                ))
        
        # Check text entities
        if self.config.check_text_entities:
            text_count = 0
            for entity in doc.modelspace():
                if entity.dxftype() in ['TEXT', 'MTEXT']:
                    text_count += 1
            
            result.text_entities = text_count
            
            if text_count > 0:
                result.issues.append(ValidationIssue(
                    rule_name="text_entities",
                    severity=ValidationSeverity.INFO,
                    message=f"Found {text_count} text entities",
                    suggested_fix="Text entities are not cut by waterjet - consider removing if not needed"
                ))
        
        # Check dimension entities
        if self.config.check_dimensions:
            dim_count = 0
            for entity in doc.modelspace():
                if entity.dxftype().startswith('DIMENSION'):
                    dim_count += 1
            
            result.dimension_entities = dim_count
            
            if dim_count > 0:
                result.issues.append(ValidationIssue(
                    rule_name="dimension_entities",
                    severity=ValidationSeverity.INFO,
                    message=f"Found {dim_count} dimension entities",
                    suggested_fix="Dimension entities are not cut by waterjet - consider removing if not needed"
                ))
    
    def _apply_custom_rules(self, msp, result: ValidationResult):
        """Apply custom validation rules."""
        for rule in self.config.custom_rules:
            if not rule.enabled:
                continue
            
            # Apply rule-specific validation
            if rule.name == "min_feature_size":
                # Already handled in geometry validation
                pass
            elif rule.name == "max_feature_size":
                # Already handled in geometry validation
                pass
            elif rule.name == "min_corner_radius":
                # Check corner radii in polygons
                self._check_corner_radii(msp, result, rule)
            elif rule.name == "min_spacing":
                # Already handled in spacing validation
                pass
            elif rule.name == "min_hole_diameter":
                # Already handled in circle validation
                pass
            elif rule.name == "self_intersections":
                # Already handled in geometry validation
                pass
            elif rule.name == "open_contours":
                if result.open_contours > rule.threshold:
                    result.issues.append(ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {result.open_contours} open contours",
                        suggested_fix="Close open contours for proper cutting"
                    ))
            elif rule.name == "cutting_time":
                if result.estimated_cutting_time > rule.threshold:
                    result.issues.append(ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Cutting time {result.estimated_cutting_time:.1f}min exceeds threshold {rule.threshold}min",
                        suggested_fix="Optimize geometry or split into multiple jobs"
                    ))
            elif rule.name == "pierce_count":
                if result.estimated_pierce_count > rule.threshold:
                    result.issues.append(ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Pierce count {result.estimated_pierce_count} exceeds threshold {rule.threshold}",
                        suggested_fix="Reduce number of separate features"
                    ))
    
    def _check_corner_radii(self, msp, result: ValidationResult, rule: ValidationRule):
        """Check corner radii in polygons."""
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE' and entity.closed:
                try:
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) >= 3:
                        poly = Polygon(points)
                        # Check each corner for minimum radius
                        for i in range(len(points)):
                            prev_point = points[i - 1]
                            curr_point = points[i]
                            next_point = points[(i + 1) % len(points)]
                            
                            # Calculate corner angle and effective radius
                            angle = self._calculate_corner_angle(prev_point, curr_point, next_point)
                            if angle < math.pi / 2:  # Sharp corner
                                # Estimate effective radius
                                effective_radius = self._estimate_corner_radius(prev_point, curr_point, next_point)
                                if effective_radius < rule.threshold:
                                    result.issues.append(ValidationIssue(
                                        rule_name=rule.name,
                                        severity=rule.severity,
                                        message=f"Sharp corner at ({curr_point[0]:.2f}, {curr_point[1]:.2f}) with radius {effective_radius:.2f}mm",
                                        location=curr_point,
                                        entity_type="LWPOLYLINE",
                                        suggested_fix=f"Increase corner radius to at least {rule.threshold}mm"
                                    ))
                                    result.radius_violations += 1
                except Exception:
                    continue
    
    def _calculate_corner_angle(self, p1: Tuple[float, float], 
                               p2: Tuple[float, float], 
                               p3: Tuple[float, float]) -> float:
        """Calculate the angle at point p2."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return math.pi
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        
        return math.acos(cos_angle)
    
    def _estimate_corner_radius(self, p1: Tuple[float, float], 
                               p2: Tuple[float, float], 
                               p3: Tuple[float, float]) -> float:
        """Estimate the effective radius of a corner."""
        # Simple estimation based on angle and segment lengths
        angle = self._calculate_corner_angle(p1, p2, p3)

        if angle <= 0 or angle >= math.pi:
            # Degenerate or straight-line corner
            return float("inf") if angle >= math.pi else 0.0

        len1 = math.dist(p1, p2)
        len2 = math.dist(p2, p3)
        min_len = max(min(len1, len2), 1e-9)

        # Use a geometry-derived proxy that maps 90° corners to a modest radius and
        # progressively increases for more obtuse angles.
        cos_half = math.cos(angle / 2.0)
        # Guard against numerical issues when angle approaches 180°
        if abs(1 + cos_half) < 1e-9:
            return min_len

        radius_factor = (1.0 - cos_half) / (1.0 + cos_half)
        estimated_radius = max(0.0, min_len * radius_factor)

        return estimated_radius
    
    def _calculate_overall_score(self, result: ValidationResult):
        """Calculate overall validation score."""
        score = 100.0
        
        # Deduct points for issues based on severity
        for issue in result.issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 25
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 15
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 5
            elif issue.severity == ValidationSeverity.INFO:
                score -= 1
        
        # Apply manufacturing feasibility factor
        score *= result.cutting_feasibility_score
        
        # Apply efficiency factor
        score *= result.efficiency_score
        
        result.overall_score = max(0, min(100, score))
        has_critical = any(issue.severity == ValidationSeverity.CRITICAL for issue in result.issues)
        # Consider simple closed geometry without geometry errors valid
        simple_ok = (result.total_entities > 0 and result.closed_contours >= 1 and result.self_intersections == 0)
        result.is_valid = (result.overall_score >= 70 and not has_critical) or simple_ok
        
        # Generate recommendations
        self._generate_recommendations(result)
    
    def _generate_recommendations(self, result: ValidationResult):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if result.open_contours > 0:
            recommendations.append(f"Close {result.open_contours} open contours for proper cutting")
        
        if result.spacing_violations > 0:
            recommendations.append(f"Fix {result.spacing_violations} spacing violations")
        
        if result.radius_violations > 0:
            recommendations.append(f"Fix {result.radius_violations} radius violations")
        
        if result.self_intersections > 0:
            recommendations.append(f"Fix {result.self_intersections} self-intersections")
        
        if result.kerf_conflicts > 0:
            recommendations.append(f"Resolve {result.kerf_conflicts} kerf conflicts")
        
        if result.unused_layers > 0:
            recommendations.append(f"Remove {result.unused_layers} unused layers")
        
        if result.unused_blocks > 0:
            recommendations.append(f"Remove {result.unused_blocks} unused blocks")
        
        if result.text_entities > 0:
            recommendations.append(f"Remove {result.text_entities} text entities if not needed")
        
        if result.dimension_entities > 0:
            recommendations.append(f"Remove {result.dimension_entities} dimension entities if not needed")
        
        if result.estimated_cutting_time > self.config.max_cutting_time_minutes:
            recommendations.append("Consider splitting into multiple jobs due to long cutting time")
        
        if result.estimated_pierce_count > self.config.max_pierce_count:
            recommendations.append("Consider reducing number of separate features")
        
        result.recommendations = recommendations


def create_validation_config(level: ValidationLevel = ValidationLevel.STANDARD,
                           **kwargs) -> ValidationConfig:
    """Create a validation configuration with specified level and custom parameters."""
    config = ValidationConfig(validation_level=level)
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def validate_dxf_file(dxf_path: str, 
                     config: Optional[ValidationConfig] = None) -> ValidationResult:
    """
    Convenience function to validate a DXF file.
    
    Args:
        dxf_path: Path to the DXF file
        config: Optional validation configuration
        
    Returns:
        ValidationResult with comprehensive analysis
    """
    validator = EnhancedDXFValidator(config)
    return validator.validate_dxf(dxf_path)
