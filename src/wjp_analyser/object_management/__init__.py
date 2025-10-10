"""
Object Management Module

This module provides comprehensive object identification, classification, and management
for DXF files, enabling interactive object selection and layer-based processing.

Main Components:
- DXFObjectManager: Object identification and management
- LayerManager: Layer creation and organization
- Object selection and filtering capabilities
- Layer-based processing workflows

Usage Examples:

Basic object management:
```python
from wjp_analyser.object_management import DXFObjectManager

manager = DXFObjectManager()
objects = manager.load_dxf_objects("file.dxf")
manager.select_object("obj_000001")
```

Layer management:
```python
from wjp_analyser.object_management import LayerManager, LayerType

layer_manager = LayerManager()
layer_id = layer_manager.create_layer("My Layer", LayerType.NESTED)
layer_manager.add_object_to_layer(layer_id, "obj_000001", object_manager)
```

Interactive workflow:
```python
from wjp_analyser.object_management import create_interactive_workflow

workflow = create_interactive_workflow("file.dxf")
workflow.present_object_selection()
workflow.create_layers_from_selection()
workflow.optimize_layers()
```
"""

from .dxf_object_manager import (
    DXFObjectManager,
    DXFObject,
    ObjectType,
    ObjectComplexity,
    ObjectMetadata,
    ObjectGeometry,
    load_dxf_objects,
    create_object_manager
)

from .layer_manager import (
    LayerManager,
    CuttingLayer,
    LayerType,
    LayerStatus,
    MaterialSettings,
    CuttingSettings,
    NestingSettings,
    CostAnalysis,
    OptimizationResult,
    create_layer_manager,
    create_default_layers
)

from .interactive_interface import create_app
from .polygon_layer_classifier import (
    classify_polylines as classify_polylines_simple,
    create_layers as export_layered_dxf,
)

from .base_layer_processor import (
    BaseLayerProcessor,
    BaseLayerStatus,
    CuttingSequence,
    BaseLayerResult
)

from .layer_cost_optimizer import (
    LayerCostCalculator,
    LayerCostOptimizer,
    CostBreakdown,
    CostOptimizationResult,
    CostCategory,
    OptimizationStrategy
)

__all__ = [
    # Object management
    'DXFObjectManager',
    'DXFObject',
    'ObjectType',
    'ObjectComplexity',
    'ObjectMetadata',
    'ObjectGeometry',
    'load_dxf_objects',
    'create_object_manager',
    
    # Layer management
    'LayerManager',
    'CuttingLayer',
    'LayerType',
    'LayerStatus',
    'MaterialSettings',
    'CuttingSettings',
    'NestingSettings',
    'CostAnalysis',
    'OptimizationResult',
    'create_layer_manager',
    'create_default_layers',
    
    # Interactive interface
    'create_app',
    
    # Base layer processing
    'BaseLayerProcessor',
    'BaseLayerStatus',
    'CuttingSequence',
    'BaseLayerResult',
    
    # Cost optimization
    'LayerCostCalculator',
    'LayerCostOptimizer',
    'CostBreakdown',
    'CostOptimizationResult',
    'CostCategory',
    'OptimizationStrategy'
]

# Utilities
__all__ += [
    'classify_polylines_simple',
    'export_layered_dxf',
]

# Version information
__version__ = "1.0.0"
__author__ = "WJP Analyser Team"
__description__ = "Object management and layer-based processing for DXF files"
