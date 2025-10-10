#!/usr/bin/env python3
"""
Comprehensive Waterjet Analyser Workflow Manager
Handles both Image Upload and DXF Upload workflows with complete analysis pipeline.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import ezdxf
from shapely.geometry import LineString, Polygon
import numpy as np

# Import layer management components
from ..object_management import (
    DXFObjectManager, LayerManager, LayerType, 
    BaseLayerProcessor, LayerCostOptimizer, OptimizationStrategy
)

class WorkflowType(Enum):
    IMAGE_UPLOAD = "image_upload"
    DXF_UPLOAD = "dxf_upload"

class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

@dataclass
class WorkflowConfig:
    """Configuration for workflow processing."""
    workflow_type: WorkflowType
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_toolpath_analysis: bool = True
    enable_cost_estimation: bool = True
    enable_nesting: bool = True
    enable_layer_management: bool = True

@dataclass
class DXFValidationResult:
    """Results from DXF validation."""
    is_valid: bool
    total_entities: int
    polygons: int
    polylines: int
    circles: int
    arcs: int
    lines: int
    open_contours: int
    closed_contours: int
    spacing_violations: int
    min_feature_size: float
    max_feature_size: float
    bounding_box: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    issues: List[str]
    recommendations: List[str]
    quality_score: float  # 0-100

@dataclass
class ToolpathParameters:
    """Toolpath analysis parameters."""
    cutting_speed_mm_min: float
    pierce_time_s: float
    kerf_width_mm: float
    lead_in_length_mm: float
    lead_out_length_mm: float
    total_cutting_length_mm: float
    estimated_cutting_time_min: float
    material_consumption_mm2: float
    quality_level: str  # "rough", "standard", "precision"

@dataclass
class CostEstimate:
    """Cost estimation results."""
    material_cost: float
    cutting_time_cost: float
    setup_cost: float
    total_cost: float
    cost_per_unit: float
    currency: str = "USD"

@dataclass
class LayerInfo:
    """Information about a DXF layer."""
    name: str
    color: int
    line_type: str
    entity_count: int
    entities: List[Any]
    bounding_box: Tuple[float, float, float, float]

@dataclass
class NestingResult:
    """Nesting optimization results."""
    total_parts: int
    nested_parts: int
    material_utilization: float  # percentage
    cutting_paths: List[List[Tuple[float, float]]]
    nesting_efficiency: float  # percentage
    recommended_sheet_size: Tuple[float, float]  # (width, height)

class WorkflowManager:
    """Manages the complete Waterjet Analyser workflow."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.workflow_id = f"workflow_{int(time.time())}"
        
        # Initialize layer management components
        self.object_manager = DXFObjectManager()
        self.layer_manager = LayerManager()
        self.base_layer_processor = BaseLayerProcessor()
        self.cost_optimizer = LayerCostOptimizer()
        
        self.results = {
            "workflow_id": self.workflow_id,
            "workflow_type": config.workflow_type.value,
            "start_time": time.time(),
            "steps_completed": [],
            "current_step": None,
            "validation_result": None,
            "toolpath_parameters": None,
            "cost_estimate": None,
            "layer_info": [],
            "nesting_result": None,
            "object_analysis": None,
            "layer_processing": None,
            "cost_optimization": None,
            "errors": [],
            "warnings": []
        }
    
    def execute_image_upload_workflow(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Execute complete image upload workflow."""
        try:
            self.results["current_step"] = "image_conversion"
            
            # Step 1: Convert image to DXF
            dxf_path = self._convert_image_to_dxf(image_path, output_dir)
            if not dxf_path:
                raise Exception("Image to DXF conversion failed")
            
            self.results["steps_completed"].append("image_conversion")
            self.results["dxf_path"] = dxf_path
            
            # Step 2: Validate DXF
            self.results["current_step"] = "dxf_validation"
            validation_result = self._validate_dxf(dxf_path)
            self.results["validation_result"] = asdict(validation_result)
            self.results["steps_completed"].append("dxf_validation")
            
            # Step 3: Generate validation report
            self.results["current_step"] = "validation_report"
            validation_report = self._generate_validation_report(validation_result)
            self.results["validation_report"] = validation_report
            self.results["steps_completed"].append("validation_report")
            
            # Step 4: Toolpath analysis
            if self.config.enable_toolpath_analysis:
                self.results["current_step"] = "toolpath_analysis"
                toolpath_params = self._analyze_toolpath(dxf_path, validation_result)
                self.results["toolpath_parameters"] = asdict(toolpath_params)
                self.results["steps_completed"].append("toolpath_analysis")
            
            # Step 5: Cost estimation
            if self.config.enable_cost_estimation:
                self.results["current_step"] = "cost_estimation"
                cost_estimate = self._estimate_costs(toolpath_params, validation_result)
                self.results["cost_estimate"] = asdict(cost_estimate)
                self.results["steps_completed"].append("cost_estimation")
            
            self.results["current_step"] = "completed"
            self.results["end_time"] = time.time()
            self.results["total_time"] = self.results["end_time"] - self.results["start_time"]
            
            return self.results
            
        except Exception as e:
            self.results["errors"].append(str(e))
            self.results["current_step"] = "failed"
            return self.results
    
    def execute_dxf_upload_workflow(self, dxf_path: str, user_inputs: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Execute complete DXF upload workflow."""
        try:
            self.results["current_step"] = "dxf_resize"
            self.results["user_inputs"] = user_inputs
            
            # Step 1: Resize DXF based on user inputs
            resized_dxf_path = self._resize_dxf(dxf_path, user_inputs, output_dir)
            self.results["resized_dxf_path"] = resized_dxf_path
            self.results["steps_completed"].append("dxf_resize")
            
            # Step 2: Layer management
            if self.config.enable_layer_management:
                self.results["current_step"] = "layer_management"
                layer_info = self._analyze_layers(resized_dxf_path)
                self.results["layer_info"] = [asdict(layer) for layer in layer_info]
                self.results["steps_completed"].append("layer_management")
            
            # Step 3: Nesting optimization
            if self.config.enable_nesting:
                self.results["current_step"] = "nesting"
                nesting_result = self._optimize_nesting(resized_dxf_path, layer_info, user_inputs)
                self.results["nesting_result"] = asdict(nesting_result)
                self.results["steps_completed"].append("nesting")
            
            # Step 4: Toolpath analysis
            if self.config.enable_toolpath_analysis:
                self.results["current_step"] = "toolpath_analysis"
                toolpath_params = self._analyze_toolpath(resized_dxf_path, None, user_inputs)
                self.results["toolpath_parameters"] = asdict(toolpath_params)
                self.results["steps_completed"].append("toolpath_analysis")
            
            # Step 5: Cost estimation
            if self.config.enable_cost_estimation:
                self.results["current_step"] = "cost_estimation"
                cost_estimate = self._estimate_costs(toolpath_params, None, user_inputs)
                self.results["cost_estimate"] = asdict(cost_estimate)
                self.results["steps_completed"].append("cost_estimation")
            
            self.results["current_step"] = "completed"
            self.results["end_time"] = time.time()
            self.results["total_time"] = self.results["end_time"] - self.results["start_time"]
            
            return self.results
            
        except Exception as e:
            self.results["errors"].append(str(e))
            self.results["current_step"] = "failed"
            return self.results
    
    def _convert_image_to_dxf(self, image_path: str, output_dir: str) -> Optional[str]:
        """Convert image to DXF using enhanced converter."""
        try:
            from tools.enhanced_image_to_dxf import EnhancedImageToDXF, ContourQuality
            
            converter = EnhancedImageToDXF(quality=ContourQuality.HIGH)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            dxf_name = f"{base_name}_converted.dxf"
            
            # The converter returns boolean, but we need the actual file path
            success = converter.convert_image_to_dxf(image_path, dxf_name)
            if success:
                # The file should be created in the current directory
                dxf_path = os.path.join(os.getcwd(), dxf_name)
                if os.path.exists(dxf_path):
                    # Move to output directory
                    output_path = os.path.join(output_dir, dxf_name)
                    os.makedirs(output_dir, exist_ok=True)
                    import shutil
                    shutil.move(dxf_path, output_path)
                    return output_path
                else:
                    # Try to find the file
                    for root, dirs, files in os.walk('.'):
                        if dxf_name in files:
                            found_path = os.path.join(root, dxf_name)
                            output_path = os.path.join(output_dir, dxf_name)
                            os.makedirs(output_dir, exist_ok=True)
                            import shutil
                            shutil.move(found_path, output_path)
                            return output_path
            return None
            
        except Exception as e:
            self.results["errors"].append(f"Image conversion error: {str(e)}")
            return None
    
    def _validate_dxf(self, dxf_path: str) -> DXFValidationResult:
        """Comprehensive DXF validation using enhanced validation system."""
        try:
            # Use enhanced validation if available
            try:
                from ..validation import validate_dxf_with_enhanced_features, ValidationLevel
                
                # Perform enhanced validation
                enhanced_result, reports = validate_dxf_with_enhanced_features(
                    dxf_path, 
                    validation_level=ValidationLevel.STANDARD,
                    create_reports=False  # Don't create reports in workflow
                )
                
                # Convert enhanced result to legacy format
                return self._convert_enhanced_to_legacy_result(enhanced_result)
                
            except ImportError:
                # Fallback to original validation if enhanced validation not available
                logger.warning("Enhanced validation not available, using legacy validation")
                return self._legacy_validate_dxf(dxf_path)
                
        except Exception as e:
            logger.error(f"DXF validation error: {e}")
            return DXFValidationResult(
                is_valid=False,
                total_entities=0,
                polygons=0,
                polylines=0,
                circles=0,
                arcs=0,
                lines=0,
                open_contours=0,
                closed_contours=0,
                spacing_violations=0,
                min_feature_size=0,
                max_feature_size=0,
                bounding_box=(0, 0, 0, 0),
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Check DXF file format and content"],
                quality_score=0
            )
    
    def _convert_enhanced_to_legacy_result(self, enhanced_result) -> DXFValidationResult:
        """Convert enhanced validation result to legacy format."""
        # Convert issues to simple strings
        issues = []
        for issue in enhanced_result.issues:
            issues.append(f"{issue.severity.value.upper()}: {issue.message}")
            if issue.suggested_fix:
                issues.append(f"  Suggested fix: {issue.suggested_fix}")
        
        return DXFValidationResult(
            is_valid=enhanced_result.is_valid,
            total_entities=enhanced_result.total_entities,
            polygons=enhanced_result.polygons,
            polylines=enhanced_result.polylines,
            circles=enhanced_result.circles,
            arcs=enhanced_result.arcs,
            lines=enhanced_result.lines,
            open_contours=enhanced_result.open_contours,
            closed_contours=enhanced_result.closed_contours,
            spacing_violations=enhanced_result.spacing_violations,
            min_feature_size=enhanced_result.min_feature_size,
            max_feature_size=enhanced_result.max_feature_size,
            bounding_box=enhanced_result.bounding_box,
            issues=issues,
            recommendations=enhanced_result.recommendations,
            quality_score=enhanced_result.overall_score
        )
    
    def _legacy_validate_dxf(self, dxf_path: str) -> DXFValidationResult:
        """Legacy DXF validation (original implementation)."""
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # Initialize counters
            total_entities = 0
            polygons = 0
            polylines = 0
            circles = 0
            arcs = 0
            lines = 0
            open_contours = 0
            closed_contours = 0
            spacing_violations = 0
            min_feature_size = float('inf')
            max_feature_size = 0
            bounding_box = [float('inf'), float('inf'), float('-inf'), float('-inf')]
            issues = []
            recommendations = []
            
            # Analyze entities
            for entity in msp:
                total_entities += 1
                
                if entity.dxftype() == 'LWPOLYLINE':
                    polylines += 1
                    if entity.closed:
                        closed_contours += 1
                        polygons += 1
                    else:
                        open_contours += 1
                    
                    # Calculate feature size
                    points = list(entity.get_points())
                    if len(points) >= 2:
                        # Calculate bounding box
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        
                        feature_size = max(max_x - min_x, max_y - min_y)
                        min_feature_size = min(min_feature_size, feature_size)
                        max_feature_size = max(max_feature_size, feature_size)
                        
                        # Update global bounding box
                        bounding_box[0] = min(bounding_box[0], min_x)
                        bounding_box[1] = min(bounding_box[1], min_y)
                        bounding_box[2] = max(bounding_box[2], max_x)
                        bounding_box[3] = max(bounding_box[3], max_y)
                
                elif entity.dxftype() == 'CIRCLE':
                    circles += 1
                    radius = entity.dxf.radius
                    min_feature_size = min(min_feature_size, radius * 2)
                    max_feature_size = max(max_feature_size, radius * 2)
                
                elif entity.dxftype() == 'ARC':
                    arcs += 1
                
                elif entity.dxftype() == 'LINE':
                    lines += 1
            
            # Check for issues
            if open_contours > 0:
                issues.append(f"Found {open_contours} open contours")
                recommendations.append("Consider closing open contours for better cutting")
            
            if min_feature_size < 1.0:  # Less than 1mm
                issues.append("Very small features detected")
                recommendations.append("Consider removing or enlarging features smaller than 1mm")
            
            if total_entities == 0:
                issues.append("No entities found in DXF")
                recommendations.append("Check DXF file content")
            
            # Calculate quality score
            quality_score = 100
            if open_contours > 0:
                quality_score -= min(30, open_contours * 5)
            if min_feature_size < 1.0:
                quality_score -= 20
            if total_entities == 0:
                quality_score = 0
            
            return DXFValidationResult(
                is_valid=total_entities > 0 and open_contours == 0,
                total_entities=total_entities,
                polygons=polygons,
                polylines=polylines,
                circles=circles,
                arcs=arcs,
                lines=lines,
                open_contours=open_contours,
                closed_contours=closed_contours,
                spacing_violations=spacing_violations,
                min_feature_size=min_feature_size if min_feature_size != float('inf') else 0,
                max_feature_size=max_feature_size,
                bounding_box=tuple(bounding_box),
                issues=issues,
                recommendations=recommendations,
                quality_score=max(0, quality_score)
            )
            
        except Exception as e:
            self.results["errors"].append(f"DXF validation error: {str(e)}")
            return DXFValidationResult(
                is_valid=False,
                total_entities=0,
                polygons=0,
                polylines=0,
                circles=0,
                arcs=0,
                lines=0,
                open_contours=0,
                closed_contours=0,
                spacing_violations=0,
                min_feature_size=0,
                max_feature_size=0,
                bounding_box=(0, 0, 0, 0),
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Check DXF file format and content"],
                quality_score=0
            )
    
    def _generate_validation_report(self, validation_result: DXFValidationResult) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            "summary": {
                "is_valid": validation_result.is_valid,
                "quality_score": validation_result.quality_score,
                "total_entities": validation_result.total_entities,
                "bounding_box": validation_result.bounding_box
            },
            "entity_breakdown": {
                "polygons": validation_result.polygons,
                "polylines": validation_result.polylines,
                "circles": validation_result.circles,
                "arcs": validation_result.arcs,
                "lines": validation_result.lines
            },
            "quality_metrics": {
                "open_contours": validation_result.open_contours,
                "closed_contours": validation_result.closed_contours,
                "min_feature_size": validation_result.min_feature_size,
                "max_feature_size": validation_result.max_feature_size,
                "spacing_violations": validation_result.spacing_violations
            },
            "issues": validation_result.issues,
            "recommendations": validation_result.recommendations
        }
    
    def _analyze_toolpath(self, dxf_path: str, validation_result: Optional[DXFValidationResult] = None, user_inputs: Optional[Dict[str, Any]] = None) -> ToolpathParameters:
        """Analyze toolpath and calculate cutting parameters."""
        try:
            # Load material parameters
            material = user_inputs.get("material", "steel") if user_inputs else "steel"
            thickness = user_inputs.get("thickness", 10.0) if user_inputs else 10.0
            
            # Get kerf data
            from src.wjp_analyser.manufacturing.kerf_table import KerfTable, MaterialType
            
            kerf_table = KerfTable()
            material_type = MaterialType.STEEL  # Default
            
            # Map material names to enum
            material_mapping = {
                "steel": MaterialType.STEEL,
                "aluminum": MaterialType.ALUMINUM,
                "granite": MaterialType.GRANITE,
                "marble": MaterialType.MARBLE,
                "brass": MaterialType.BRASS,
                "copper": MaterialType.COPPER
            }
            
            if material.lower() in material_mapping:
                material_type = material_mapping[material.lower()]
            
            kerf_data = kerf_table.get_kerf_data(material_type, thickness)
            
            if not kerf_data:
                # Default values if no kerf data available
                cutting_speed = 1000.0
                pierce_time = 2.0
                kerf_width = 0.8
                quality_factor = 1.0
            else:
                # Access KerfData object attributes
                cutting_speed = kerf_data.cutting_speed_mm_min
                pierce_time = kerf_data.pierce_time_s
                kerf_width = kerf_data.kerf_mm
                quality_factor = kerf_data.quality_factor
            
            # Calculate total cutting length
            total_cutting_length = 0
            if validation_result:
                # Estimate based on bounding box and entity count
                bbox = validation_result.bounding_box
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                perimeter = 2 * (width + height)
                total_cutting_length = perimeter * validation_result.total_entities / 10  # Rough estimate
            
            # Calculate cutting time
            estimated_cutting_time = total_cutting_length / cutting_speed if cutting_speed > 0 else 0
            
            # Calculate material consumption
            if validation_result:
                bbox = validation_result.bounding_box
                material_consumption = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * thickness
            else:
                material_consumption = 0
            
            # Determine quality level
            quality_level = "standard"
            if quality_factor > 1.2:
                quality_level = "precision"
            elif quality_factor < 0.8:
                quality_level = "rough"
            
            return ToolpathParameters(
                cutting_speed_mm_min=cutting_speed,
                pierce_time_s=pierce_time,
                kerf_width_mm=kerf_width,
                lead_in_length_mm=2.0,
                lead_out_length_mm=2.0,
                total_cutting_length_mm=total_cutting_length,
                estimated_cutting_time_min=estimated_cutting_time,
                material_consumption_mm2=material_consumption,
                quality_level=quality_level
            )
            
        except Exception as e:
            self.results["errors"].append(f"Toolpath analysis error: {str(e)}")
            # Return default parameters
            return ToolpathParameters(
                cutting_speed_mm_min=1000.0,
                pierce_time_s=2.0,
                kerf_width_mm=0.8,
                lead_in_length_mm=2.0,
                lead_out_length_mm=2.0,
                total_cutting_length_mm=0,
                estimated_cutting_time_min=0,
                material_consumption_mm2=0,
                quality_level="standard"
            )
    
    def _estimate_costs(self, toolpath_params: ToolpathParameters, validation_result: Optional[DXFValidationResult] = None, user_inputs: Optional[Dict[str, Any]] = None) -> CostEstimate:
        """Estimate costs based on toolpath parameters."""
        try:
            # Material cost calculation
            material = user_inputs.get("material", "steel") if user_inputs else "steel"
            thickness = user_inputs.get("thickness", 10.0) if user_inputs else 10.0
            
            # Material cost per mm³ (rough estimates)
            material_costs = {
                "steel": 0.001,  # $0.001 per mm³
                "aluminum": 0.002,
                "granite": 0.005,
                "marble": 0.008,
                "brass": 0.003,
                "copper": 0.004
            }
            
            material_cost_per_mm3 = material_costs.get(material.lower(), 0.001)
            material_cost = toolpath_params.material_consumption_mm2 * thickness * material_cost_per_mm3
            
            # Cutting time cost (assuming $50/hour)
            hourly_rate = 50.0
            cutting_time_cost = (toolpath_params.estimated_cutting_time_min / 60) * hourly_rate
            
            # Setup cost (fixed cost per job)
            setup_cost = 25.0
            
            # Total cost
            total_cost = material_cost + cutting_time_cost + setup_cost
            
            # Cost per unit (assuming single part)
            cost_per_unit = total_cost
            
            return CostEstimate(
                material_cost=material_cost,
                cutting_time_cost=cutting_time_cost,
                setup_cost=setup_cost,
                total_cost=total_cost,
                cost_per_unit=cost_per_unit,
                currency="USD"
            )
            
        except Exception as e:
            self.results["errors"].append(f"Cost estimation error: {str(e)}")
            return CostEstimate(
                material_cost=0,
                cutting_time_cost=0,
                setup_cost=0,
                total_cost=0,
                cost_per_unit=0,
                currency="USD"
            )
    
    def _resize_dxf(self, dxf_path: str, user_inputs: Dict[str, Any], output_dir: str) -> str:
        """Resize DXF based on user inputs without disturbing contents."""
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # Get target dimensions from user inputs
            target_width = user_inputs.get("width", 100.0)
            target_height = user_inputs.get("height", 100.0)
            
            # Calculate current bounding box
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for entity in msp:
                if hasattr(entity, 'get_points'):
                    points = list(entity.get_points())
                    for point in points:
                        min_x = min(min_x, point[0])
                        min_y = min(min_y, point[1])
                        max_x = max(max_x, point[0])
                        max_y = max(max_y, point[1])
            
            current_width = max_x - min_x
            current_height = max_y - min_y
            
            # Calculate scale factors
            scale_x = target_width / current_width if current_width > 0 else 1.0
            scale_y = target_height / current_height if current_height > 0 else 1.0
            
            # Use uniform scaling to maintain aspect ratio
            scale = min(scale_x, scale_y)
            
            # Apply scaling
            for entity in msp:
                if hasattr(entity, 'scale'):
                    entity.scale(scale)
                elif hasattr(entity, 'get_points'):
                    # Manual scaling for entities without scale method
                    points = list(entity.get_points())
                    scaled_points = [(p[0] * scale, p[1] * scale) for p in points]
                    # Update entity points (implementation depends on entity type)
            
            # Save resized DXF
            base_name = os.path.splitext(os.path.basename(dxf_path))[0]
            resized_path = os.path.join(output_dir, f"{base_name}_resized.dxf")
            doc.saveas(resized_path)
            
            return resized_path
            
        except Exception as e:
            self.results["errors"].append(f"DXF resize error: {str(e)}")
            return dxf_path  # Return original if resize fails
    
    def _analyze_layers(self, dxf_path: str) -> List[LayerInfo]:
        """Analyze DXF layers and group entities."""
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            layers = {}
            
            for entity in msp:
                layer_name = entity.dxf.layer
                
                if layer_name not in layers:
                    layers[layer_name] = {
                        'entities': [],
                        'min_x': float('inf'),
                        'min_y': float('inf'),
                        'max_x': float('-inf'),
                        'max_y': float('-inf')
                    }
                
                layers[layer_name]['entities'].append(entity)
                
                # Update bounding box
                if hasattr(entity, 'get_points'):
                    points = list(entity.get_points())
                    for point in points:
                        layers[layer_name]['min_x'] = min(layers[layer_name]['min_x'], point[0])
                        layers[layer_name]['min_y'] = min(layers[layer_name]['min_y'], point[1])
                        layers[layer_name]['max_x'] = max(layers[layer_name]['max_x'], point[0])
                        layers[layer_name]['max_y'] = max(layers[layer_name]['max_y'], point[1])
            
            # Convert to LayerInfo objects
            layer_info = []
            for layer_name, layer_data in layers.items():
                layer_info.append(LayerInfo(
                    name=layer_name,
                    color=0,  # Default color
                    line_type="CONTINUOUS",
                    entity_count=len(layer_data['entities']),
                    entities=layer_data['entities'],
                    bounding_box=(
                        layer_data['min_x'] if layer_data['min_x'] != float('inf') else 0,
                        layer_data['min_y'] if layer_data['min_y'] != float('inf') else 0,
                        layer_data['max_x'] if layer_data['max_x'] != float('-inf') else 0,
                        layer_data['max_y'] if layer_data['max_y'] != float('-inf') else 0
                    )
                ))
            
            return layer_info
            
        except Exception as e:
            self.results["errors"].append(f"Layer analysis error: {str(e)}")
            return []
    
    def _optimize_nesting(self, dxf_path: str, layer_info: List[LayerInfo], user_inputs: Dict[str, Any]) -> NestingResult:
        """Optimize nesting based on layers and user inputs."""
        try:
            # Get sheet size from user inputs
            sheet_width = user_inputs.get("sheet_width", 1000.0)
            sheet_height = user_inputs.get("sheet_height", 1000.0)
            
            total_parts = sum(layer.entity_count for layer in layer_info)
            nested_parts = 0
            cutting_paths = []
            
            # Simple nesting algorithm (can be enhanced)
            current_x = 0
            current_y = 0
            max_height_in_row = 0
            
            for layer in layer_info:
                for entity in layer.entities:
                    if hasattr(entity, 'get_points'):
                        points = list(entity.get_points())
                        if len(points) >= 2:
                            # Calculate entity dimensions
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            entity_width = max(x_coords) - min(x_coords)
                            entity_height = max(y_coords) - min(y_coords)
                            
                            # Check if entity fits in current position
                            if current_x + entity_width <= sheet_width and current_y + entity_height <= sheet_height:
                                # Add to cutting paths
                                cutting_paths.append(points)
                                nested_parts += 1
                                
                                # Update position
                                current_x += entity_width + 5  # 5mm spacing
                                max_height_in_row = max(max_height_in_row, entity_height)
                            else:
                                # Move to next row
                                current_x = 0
                                current_y += max_height_in_row + 5
                                max_height_in_row = 0
                                
                                # Check if still fits
                                if current_y + entity_height <= sheet_height:
                                    cutting_paths.append(points)
                                    nested_parts += 1
                                    current_x = entity_width + 5
                                    max_height_in_row = entity_height
            
            # Calculate utilization
            total_area = sheet_width * sheet_height
            used_area = sum(len(path) * 10 for path in cutting_paths)  # Rough estimate
            material_utilization = (used_area / total_area) * 100 if total_area > 0 else 0
            
            # Calculate nesting efficiency
            nesting_efficiency = (nested_parts / total_parts) * 100 if total_parts > 0 else 0
            
            return NestingResult(
                total_parts=total_parts,
                nested_parts=nested_parts,
                material_utilization=material_utilization,
                cutting_paths=cutting_paths,
                nesting_efficiency=nesting_efficiency,
                recommended_sheet_size=(sheet_width, sheet_height)
            )
            
        except Exception as e:
            self.results["errors"].append(f"Nesting optimization error: {str(e)}")
            return NestingResult(
                total_parts=0,
                nested_parts=0,
                material_utilization=0,
                cutting_paths=[],
                nesting_efficiency=0,
                recommended_sheet_size=(0, 0)
            )
    
    def analyze_objects_for_layers(self, dxf_path: str) -> Dict[str, Any]:
        """Analyze DXF objects for layer management."""
        try:
            self.results["current_step"] = "object_analysis"
            
            # Load and analyze objects
            objects = self.object_manager.load_dxf_objects(dxf_path)
            
            # Analyze object characteristics
            object_analysis = {
                "total_objects": len(objects),
                "object_types": {},
                "complexity_distribution": {},
                "size_distribution": {},
                "area_statistics": {
                    "total_area": sum(obj.geometry.area for obj in objects),
                    "average_area": 0,
                    "min_area": 0,
                    "max_area": 0
                }
            }
            
            if objects:
                # Object type distribution
                for obj in objects:
                    obj_type = obj.object_type.value
                    object_analysis["object_types"][obj_type] = object_analysis["object_types"].get(obj_type, 0) + 1
                
                # Complexity distribution
                for obj in objects:
                    complexity = obj.complexity.value
                    object_analysis["complexity_distribution"][complexity] = object_analysis["complexity_distribution"].get(complexity, 0) + 1
                
                # Size distribution
                areas = [obj.geometry.area for obj in objects]
                object_analysis["area_statistics"]["average_area"] = sum(areas) / len(areas)
                object_analysis["area_statistics"]["min_area"] = min(areas)
                object_analysis["area_statistics"]["max_area"] = max(areas)
                
                # Size categories
                for obj in objects:
                    area = obj.geometry.area
                    if area < 100:
                        size_cat = "small"
                    elif area < 1000:
                        size_cat = "medium"
                    elif area < 10000:
                        size_cat = "large"
                    else:
                        size_cat = "very_large"
                    
                    object_analysis["size_distribution"][size_cat] = object_analysis["size_distribution"].get(size_cat, 0) + 1
            
            self.results["object_analysis"] = object_analysis
            self.results["steps_completed"].append("object_analysis")
            
            return object_analysis
            
        except Exception as e:
            error_msg = f"Object analysis error: {str(e)}"
            self.results["errors"].append(error_msg)
            return {"error": error_msg}
    
    def create_default_layers(self, dxf_path: str) -> Dict[str, Any]:
        """Create default layers for DXF processing."""
        try:
            self.results["current_step"] = "layer_creation"
            
            # Load objects
            objects = self.object_manager.load_dxf_objects(dxf_path)
            
            # Create base layer (full DXF path)
            base_layer_id = self.layer_manager.create_layer(
                name="Base Layer",
                layer_type=LayerType.BASE,
                description="Full DXF path cutting"
            )
            
            # Create nested layer for optimization
            nested_layer_id = self.layer_manager.create_layer(
                name="Nested Layer",
                layer_type=LayerType.NESTED,
                description="Optimized material utilization"
            )
            
            # Assign objects to layers
            for obj in objects:
                # Assign to base layer
                self.layer_manager.add_object_to_layer(base_layer_id, obj.object_id, self.object_manager)
                
                # Assign to nested layer for optimization
                self.layer_manager.add_object_to_layer(nested_layer_id, obj.object_id, self.object_manager)
            
            layer_info = {
                "base_layer_id": base_layer_id,
                "nested_layer_id": nested_layer_id,
                "total_objects": len(objects),
                "layers_created": 2
            }
            
            self.results["layer_processing"] = layer_info
            self.results["steps_completed"].append("layer_creation")
            
            return layer_info
            
        except Exception as e:
            error_msg = f"Layer creation error: {str(e)}"
            self.results["errors"].append(error_msg)
            return {"error": error_msg}
    
    def process_base_layer(self, layer_id: str) -> Dict[str, Any]:
        """Process base layer for full DXF path cutting."""
        try:
            self.results["current_step"] = "base_layer_processing"
            
            layer = self.layer_manager.get_layer(layer_id)
            if not layer:
                raise ValueError(f"Layer {layer_id} not found")
            
            # Process base layer
            result = self.base_layer_processor.process_base_layer(layer, self.object_manager)
            
            # Store result
            self.base_layer_processor.processed_layers[layer_id] = result
            
            processing_info = {
                "layer_id": layer_id,
                "success": result.success,
                "total_cutting_time": result.total_cutting_time,
                "total_cutting_length": result.total_cutting_length,
                "piercing_count": result.piercing_count,
                "sequence_count": len(result.cutting_sequences),
                "material_usage": result.material_usage,
                "efficiency_score": result.efficiency_score,
                "status": result.status.value
            }
            
            self.results["layer_processing"] = processing_info
            self.results["steps_completed"].append("base_layer_processing")
            
            return processing_info
            
        except Exception as e:
            error_msg = f"Base layer processing error: {str(e)}"
            self.results["errors"].append(error_msg)
            return {"error": error_msg}
    
    def optimize_layer_costs(self, layer_id: str, strategy: str = "balanced") -> Dict[str, Any]:
        """Optimize layer costs using different strategies."""
        try:
            self.results["current_step"] = "cost_optimization"
            
            layer = self.layer_manager.get_layer(layer_id)
            if not layer:
                raise ValueError(f"Layer {layer_id} not found")
            
            # Convert strategy string to enum
            strategy_enum = OptimizationStrategy.BALANCED
            if strategy == "minimize_time":
                strategy_enum = OptimizationStrategy.MINIMIZE_TIME
            elif strategy == "minimize_material":
                strategy_enum = OptimizationStrategy.MINIMIZE_MATERIAL
            elif strategy == "minimize_cost":
                strategy_enum = OptimizationStrategy.MINIMIZE_COST
            elif strategy == "maximize_efficiency":
                strategy_enum = OptimizationStrategy.MAXIMIZE_EFFICIENCY
            
            # Run cost optimization
            result = self.cost_optimizer.optimize_layer_cost(layer, strategy_enum)
            
            optimization_info = {
                "layer_id": layer_id,
                "strategy": strategy,
                "success": result.success,
                "original_cost": result.original_cost,
                "optimized_cost": result.optimized_cost,
                "cost_savings": result.cost_savings,
                "savings_percentage": result.savings_percentage,
                "efficiency_improvement": result.efficiency_improvement,
                "recommended_settings": result.recommended_settings
            }
            
            self.results["cost_optimization"] = optimization_info
            self.results["steps_completed"].append("cost_optimization")
            
            return optimization_info
            
        except Exception as e:
            error_msg = f"Cost optimization error: {str(e)}"
            self.results["errors"].append(error_msg)
            return {"error": error_msg}
    
    def generate_layer_reports(self, output_dir: str) -> Dict[str, Any]:
        """Generate comprehensive reports for all layers."""
        try:
            self.results["current_step"] = "report_generation"
            
            reports = {}
            
            # Generate base layer reports
            for layer_id, result in self.base_layer_processor.processed_layers.items():
                layer = self.layer_manager.get_layer(layer_id)
                if layer:
                    # Generate G-code
                    gcode_path = os.path.join(output_dir, f"{layer.name}_base_layer.nc")
                    success = self.base_layer_processor.generate_gcode(result, gcode_path)
                    
                    reports[layer_id] = {
                        "layer_name": layer.name,
                        "gcode_generated": success,
                        "gcode_path": gcode_path if success else None,
                        "processing_summary": self.base_layer_processor.get_processing_summary(layer_id)
                    }
            
            # Generate cost optimization reports
            for layer_id in self.layer_manager.get_all_layers():
                layer = self.layer_manager.get_layer(layer_id)
                if layer:
                    optimization_summary = self.cost_optimizer.get_optimization_summary(layer_id)
                    if layer_id not in reports:
                        reports[layer_id] = {}
                    reports[layer_id]["cost_optimization"] = optimization_summary
            
            self.results["layer_processing"] = reports
            self.results["steps_completed"].append("report_generation")
            
            return reports
            
        except Exception as e:
            error_msg = f"Report generation error: {str(e)}"
            self.results["errors"].append(error_msg)
            return {"error": error_msg}
    
    def save_results(self, output_path: str):
        """Save workflow results to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            self.results["errors"].append(f"Save results error: {str(e)}")

def main():
    """Example usage of the workflow manager."""
    # Example 1: Image Upload Workflow
    config = WorkflowConfig(WorkflowType.IMAGE_UPLOAD)
    manager = WorkflowManager(config)
    
    # Execute workflow
    results = manager.execute_image_upload_workflow(
        "data/samples/images/jali_panel.png",
        "output/workflow_test"
    )
    
    # Save results
    manager.save_results("output/workflow_test/image_workflow_results.json")
    
    print("Image Upload Workflow Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
