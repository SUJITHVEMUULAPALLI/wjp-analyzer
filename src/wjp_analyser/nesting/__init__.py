"""
Nesting Module
==============

Production-grade nesting optimization for waterjet cutting.
"""

from .nesting_engine import (
    NestingEngine,
    NestingAlgorithm,
    NestingResult,
    PositionedObject,
    OptimizationStatus,
    NoFitPolygonAlgorithm,
    GeneticAlgorithm,
)

from .geometry_hygiene import (
    GeometryHygiene,
)

from .placement_engine import (
    BottomLeftFillEngine,
    NFPRefinementEngine,
    MetaheuristicOptimizer,
    PlacementCandidate,
)

from .constraints import (
    Constraint,
    ConstraintType,
    HardConstraints,
    SoftConstraints,
    ConstraintSet,
)

__all__ = [
    # Main engine
    "NestingEngine",
    "NestingAlgorithm",
    "NestingResult",
    "PositionedObject",
    "OptimizationStatus",
    "NoFitPolygonAlgorithm",
    "GeneticAlgorithm",
    # Geometry hygiene
    "GeometryHygiene",
    # Placement engines
    "BottomLeftFillEngine",
    "NFPRefinementEngine",
    "MetaheuristicOptimizer",
    "PlacementCandidate",
    # Constraints
    "Constraint",
    "ConstraintType",
    "HardConstraints",
    "SoftConstraints",
    "ConstraintSet",
]
