"""
Constraint System for Production-Grade Nesting
==============================================

Hard and soft constraints for realistic nesting scenarios.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, rotate


class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Preferable but not required


@dataclass
class Constraint:
    """Represents a nesting constraint."""
    constraint_type: ConstraintType
    name: str
    description: str
    weight: float = 1.0  # For soft constraints
    satisfied: bool = True


@dataclass
class HardConstraints:
    """Hard constraints (must be satisfied)."""
    kerf_margin: float = 1.0  # Minimum spacing between parts (mm)
    min_web: float = 2.0  # Minimum web width (mm)
    pierce_avoid_zones: List[Polygon] = field(default_factory=list)  # Zones to avoid for pierces
    sheet_bounds: Tuple[float, float] = (1000.0, 1000.0)  # (width, height) in mm
    min_edge_distance: float = 5.0  # Minimum distance from sheet edges (mm)
    
    def check_kerf_margin(
        self,
        polygon: Polygon,
        existing_polygons: List[Polygon],
    ) -> Tuple[bool, float]:
        """
        Check if polygon satisfies kerf margin constraint.
        
        Returns:
            (is_satisfied, minimum_distance)
        """
        if not existing_polygons:
            return True, float('inf')
        
        min_distance = float('inf')
        for existing in existing_polygons:
            distance = polygon.distance(existing)
            min_distance = min(min_distance, distance)
            if distance < self.kerf_margin:
                return False, min_distance
        
        return min_distance >= self.kerf_margin, min_distance
    
    def check_bounds(self, polygon: Polygon) -> Tuple[bool, List[str]]:
        """
        Check if polygon is within sheet bounds.
        
        Returns:
            (is_within_bounds, list_of_violations)
        """
        violations = []
        bounds = polygon.bounds
        
        # Check left edge
        if bounds[0] < self.min_edge_distance:
            violations.append(f"Too close to left edge ({bounds[0]:.2f} < {self.min_edge_distance})")
        
        # Check bottom edge
        if bounds[1] < self.min_edge_distance:
            violations.append(f"Too close to bottom edge ({bounds[1]:.2f} < {self.min_edge_distance})")
        
        # Check right edge
        if bounds[2] > self.sheet_bounds[0] - self.min_edge_distance:
            violations.append(f"Too close to right edge ({bounds[2]:.2f} > {self.sheet_bounds[0] - self.min_edge_distance})")
        
        # Check top edge
        if bounds[3] > self.sheet_bounds[1] - self.min_edge_distance:
            violations.append(f"Too close to top edge ({bounds[3]:.2f} > {self.sheet_bounds[1] - self.min_edge_distance})")
        
        return len(violations) == 0, violations
    
    def check_pierce_zones(self, polygon: Polygon) -> Tuple[bool, int]:
        """
        Check if polygon avoids pierce zones.
        
        Returns:
            (is_satisfied, number_of_violations)
        """
        if not self.pierce_avoid_zones:
            return True, 0
        
        violations = 0
        for zone in self.pierce_avoid_zones:
            if polygon.intersects(zone):
                violations += 1
        
        return violations == 0, violations
    
    def check_min_web(
        self,
        polygon: Polygon,
        existing_polygons: List[Polygon],
    ) -> Tuple[bool, float]:
        """
        Check if minimum web width is maintained.
        
        Returns:
            (is_satisfied, minimum_web_width)
        """
        if not existing_polygons:
            return True, float('inf')
        
        min_web = float('inf')
        for existing in existing_polygons:
            # Calculate web width (minimum gap between polygons)
            distance = polygon.distance(existing)
            
            # If polygons are close, check actual gap geometry
            if distance < self.min_web:
                # Create buffer zone and check intersection
                buffer_poly = polygon.buffer(distance / 2)
                buffer_existing = existing.buffer(distance / 2)
                
                if buffer_poly.intersects(buffer_existing):
                    gap = distance
                    min_web = min(min_web, gap)
                    if gap < self.min_web:
                        return False, min_web
            else:
                min_web = min(min_web, distance)
        
        return min_web >= self.min_web, min_web
    
    def validate_placement(
        self,
        polygon: Polygon,
        existing_polygons: List[Polygon],
    ) -> Tuple[bool, List[Constraint]]:
        """
        Validate placement against all hard constraints.
        
        Returns:
            (is_valid, list_of_violated_constraints)
        """
        violated = []
        
        # Check bounds
        in_bounds, bound_violations = self.check_bounds(polygon)
        if not in_bounds:
            violated.append(Constraint(
                constraint_type=ConstraintType.HARD,
                name="bounds",
                description="; ".join(bound_violations),
                satisfied=False,
            ))
        
        # Check kerf margin
        kerf_ok, min_dist = self.check_kerf_margin(polygon, existing_polygons)
        if not kerf_ok:
            violated.append(Constraint(
                constraint_type=ConstraintType.HARD,
                name="kerf_margin",
                description=f"Minimum distance ({min_dist:.3f} mm) < kerf margin ({self.kerf_margin} mm)",
                satisfied=False,
            ))
        
        # Check minimum web
        web_ok, min_web = self.check_min_web(polygon, existing_polygons)
        if not web_ok:
            violated.append(Constraint(
                constraint_type=ConstraintType.HARD,
                name="min_web",
                description=f"Web width ({min_web:.3f} mm) < minimum ({self.min_web} mm)",
                satisfied=False,
            ))
        
        # Check pierce zones
        pierce_ok, violations = self.check_pierce_zones(polygon)
        if not pierce_ok:
            violated.append(Constraint(
                constraint_type=ConstraintType.HARD,
                name="pierce_zones",
                description=f"Intersects {violations} pierce avoidance zone(s)",
                satisfied=False,
            ))
        
        return len(violated) == 0, violated


@dataclass
class SoftConstraints:
    """Soft constraints (preferable but not required)."""
    part_priorities: Dict[int, float] = field(default_factory=dict)  # Priority weights by part index
    grain_direction: Optional[float] = None  # Preferred grain direction (degrees)
    reuse_offcuts: bool = True  # Prefer to reuse offcut areas
    compactness_weight: float = 0.5  # Weight for compact placement
    edge_alignment_weight: float = 0.3  # Weight for edge alignment
    
    def calculate_priority_score(self, part_index: int) -> float:
        """Calculate priority score for a part."""
        return self.part_priorities.get(part_index, 1.0)
    
    def calculate_grain_score(
        self,
        polygon: Polygon,
        rotation: float,
    ) -> float:
        """
        Calculate score based on grain direction alignment.
        
        Returns:
            Score (0-1, higher is better)
        """
        if self.grain_direction is None:
            return 0.5  # Neutral
        
        # Check if rotation aligns with grain direction
        rotation_diff = abs((rotation % 180) - (self.grain_direction % 180))
        if rotation_diff > 90:
            rotation_diff = 180 - rotation_diff
        
        # Score: 1.0 if perfectly aligned, 0.0 if perpendicular
        score = 1.0 - (rotation_diff / 90.0)
        return max(0.0, min(1.0, score))
    
    def calculate_compactness_score(
        self,
        polygon: Polygon,
        placed_polygons: List[Polygon],
        sheet_bounds: Tuple[float, float],
    ) -> float:
        """
        Calculate compactness score (how well parts are grouped together).
        
        Returns:
            Score (0-1, higher is better)
        """
        if not placed_polygons:
            return 0.5  # Neutral
        
        # Calculate centroid
        centroid = polygon.centroid
        
        # Find nearest neighbor distance
        min_dist = float('inf')
        for placed in placed_polygons:
            dist = centroid.distance(placed.centroid)
            min_dist = min(min_dist, dist)
        
        # Normalize by sheet diagonal
        sheet_diagonal = (sheet_bounds[0]**2 + sheet_bounds[1]**2)**0.5
        normalized_dist = min_dist / sheet_diagonal
        
        # Score: closer neighbors = higher score
        score = 1.0 - min(1.0, normalized_dist * 2)
        return max(0.0, min(1.0, score))
    
    def calculate_edge_alignment_score(
        self,
        polygon: Polygon,
        sheet_bounds: Tuple[float, float],
    ) -> float:
        """
        Calculate edge alignment score.
        
        Returns:
            Score (0-1, higher is better)
        """
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Check alignment with edges
        alignments = []
        
        # Left edge alignment
        if bounds[0] < 1.0:  # Very close to left edge
            alignments.append(1.0)
        else:
            alignments.append(0.0)
        
        # Bottom edge alignment
        if bounds[1] < 1.0:  # Very close to bottom edge
            alignments.append(1.0)
        else:
            alignments.append(0.0)
        
        # Right edge alignment
        if bounds[2] > sheet_bounds[0] - 1.0:
            alignments.append(1.0)
        else:
            alignments.append(0.0)
        
        # Top edge alignment
        if bounds[3] > sheet_bounds[1] - 1.0:
            alignments.append(1.0)
        else:
            alignments.append(0.0)
        
        return sum(alignments) / len(alignments) if alignments else 0.0
    
    def calculate_total_soft_score(
        self,
        polygon: Polygon,
        part_index: int,
        rotation: float,
        placed_polygons: List[Polygon],
        sheet_bounds: Tuple[float, float],
    ) -> float:
        """
        Calculate total soft constraint score.
        
        Returns:
            Combined score (0-1, higher is better)
        """
        # Priority score
        priority_score = self.calculate_priority_score(part_index)
        
        # Grain direction score
        grain_score = self.calculate_grain_score(polygon, rotation)
        
        # Compactness score
        compact_score = self.calculate_compactness_score(
            polygon,
            placed_polygons,
            sheet_bounds,
        )
        
        # Edge alignment score
        edge_score = self.calculate_edge_alignment_score(polygon, sheet_bounds)
        
        # Weighted combination
        total_score = (
            priority_score * 0.3 +
            grain_score * 0.2 +
            compact_score * self.compactness_weight +
            edge_score * self.edge_alignment_weight
        )
        
        return max(0.0, min(1.0, total_score))


@dataclass
class ConstraintSet:
    """Complete constraint set for nesting."""
    hard: HardConstraints
    soft: SoftConstraints
    deterministic: bool = False  # Enable determinism mode
    random_seed: Optional[int] = None  # Seed for deterministic runs
    
    def __post_init__(self):
        """Initialize random seed if deterministic."""
        if self.deterministic and self.random_seed is not None:
            import random
            random.seed(self.random_seed)
            import numpy as np
            np.random.seed(self.random_seed)





