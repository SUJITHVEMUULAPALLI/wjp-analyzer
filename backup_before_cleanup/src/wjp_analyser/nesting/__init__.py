"""
Nesting Optimization Module

This module provides comprehensive nesting optimization algorithms for maximizing
material utilization in waterjet cutting operations.

Main Components:
- NestingEngine: Core optimization engine with multiple algorithms
- MaterialUtilizationCalculator: Calculate utilization metrics and costs
- MaterialUtilizationReporter: Generate comprehensive reports
- Multiple optimization algorithms (NFP, Genetic, Simulated Annealing, Bottom-Left)

Usage Examples:

Basic nesting optimization:
```python
from wjp_analyser.nesting import NestingEngine, optimize_layer_nesting

engine = NestingEngine()
result = engine.optimize_nesting(layer)
```

Material utilization analysis:
```python
from wjp_analyser.nesting import MaterialUtilizationCalculator, calculate_layer_utilization

report = calculate_layer_utilization(layer, result)
print(f"Utilization: {report.utilization_percentage:.1f}%")
```

Comprehensive reporting:
```python
from wjp_analyser.nesting import MaterialUtilizationReporter

reporter = MaterialUtilizationReporter()
report_path = reporter.generate_comprehensive_report(layer, result)
```

Algorithm comparison:
```python
from wjp_analyser.nesting import compare_nesting_algorithms

results = compare_nesting_algorithms(layer)
best_algorithm = max(results.keys(), key=lambda k: results[k].final_utilization)
```
"""

from .nesting_engine import (
    NestingEngine,
    NestingAlgorithm,
    OptimizationStatus,
    PositionedObject,
    NestingResult,
    NoFitPolygonAlgorithm,
    GeneticAlgorithm,
    SimulatedAnnealingAlgorithm,
    BottomLeftFillAlgorithm,
    create_nesting_engine,
    optimize_layer_nesting,
    compare_nesting_algorithms
)

from .material_utilization import (
    MaterialUtilizationCalculator,
    MaterialUtilizationReporter,
    MaterialUtilizationReport,
    create_utilization_calculator,
    create_utilization_reporter,
    calculate_layer_utilization
)

__all__ = [
    # Core nesting engine
    'NestingEngine',
    'NestingAlgorithm',
    'OptimizationStatus',
    'PositionedObject',
    'NestingResult',
    
    # Algorithms
    'NoFitPolygonAlgorithm',
    'GeneticAlgorithm',
    'SimulatedAnnealingAlgorithm',
    'BottomLeftFillAlgorithm',
    
    # Material utilization
    'MaterialUtilizationCalculator',
    'MaterialUtilizationReporter',
    'MaterialUtilizationReport',
    
    # Convenience functions
    'create_nesting_engine',
    'optimize_layer_nesting',
    'compare_nesting_algorithms',
    'create_utilization_calculator',
    'create_utilization_reporter',
    'calculate_layer_utilization'
]

# Version information
__version__ = "1.0.0"
__author__ = "WJP Analyser Team"
__description__ = "Nesting optimization and material utilization for waterjet cutting"
