"""
Interactive Object Selection and Layer Management Web Interface

This module provides a web-based interface for interactive object selection
and layer management, enabling users to organize DXF objects into cutting layers.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, request, jsonify, send_file
import uuid

from .dxf_object_manager import DXFObjectManager, ObjectType, ObjectComplexity
from .layer_manager import LayerManager, LayerType, MaterialSettings, CuttingSettings, NestingSettings
from ..nesting import NestingEngine, MaterialUtilizationReporter
from ..web.api_utils import (
    DXFProcessingError,
    calculate_costs_from_dxf,
    generate_gcode_from_dxf,
)

logger = logging.getLogger(__name__)

from flask import Blueprint

# Create blueprint for layer management
layers_bp = Blueprint('layers', __name__, 
                     template_folder='../../templates',
                     static_folder='../../static')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')
UPLOAD_ROOT = os.path.join(PROJECT_ROOT, 'uploads')

def create_app():
    """Create and configure the Flask blueprint."""
    return layers_bp

# Global managers (in production, use proper session management)
object_manager = None
layer_manager = None
current_dxf_path = None


@layers_bp.route('/')
def index():
    """Main interface page."""
    return render_template('object_selection.html')


@layers_bp.route('/api/upload-dxf', methods=['POST'])
def upload_dxf():
    """Upload and analyze DXF file."""
    global object_manager, layer_manager, current_dxf_path
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.dxf'):
            return jsonify({'success': False, 'error': 'Please upload a DXF file'})
        
        # Save uploaded file
        os.makedirs(UPLOAD_ROOT, exist_ok=True)
        file_path = os.path.join(UPLOAD_ROOT, f"{uuid.uuid4().hex}.dxf")
        file.save(file_path)
        
        # Initialize managers
        global object_manager, layer_manager, current_dxf_path
        object_manager = DXFObjectManager()
        layer_manager = LayerManager()
        current_dxf_path = file_path
        
        # Load objects
        objects = object_manager.load_dxf_objects(file_path)
        
        # Prepare response with simplified object data for frontend
        response = {
            'success': True,
            'file_path': file_path,
            'objects': []
        }
        
        # Add object data (simplified for frontend compatibility)
        for obj in objects:
            response['objects'].append({
                'object_id': obj.object_id,
                'object_type': obj.object_type.value,
                'complexity': obj.complexity.value,
                'name': getattr(obj, 'name', 'Unnamed'),
                'geometry': {
                    'area': obj.geometry.area,
                    'perimeter': obj.geometry.perimeter,
                    'width': obj.geometry.width,
                    'height': obj.geometry.height
                }
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error uploading DXF: {e}")
        return jsonify({'success': False, 'error': str(e)})


@layers_bp.route('/api/objects', methods=['GET'])
def get_objects():
    """Get all objects."""
    if not object_manager:
        return jsonify({'error': 'No DXF file loaded'}), 400
    
    objects = list(object_manager.objects.values())
    response = []
    
    for obj in objects:
        response.append({
            'object_id': obj.object_id,
            'object_type': obj.object_type.value,
            'complexity': obj.complexity.value,
            'area': obj.geometry.area,
            'perimeter': obj.geometry.perimeter,
            'width': obj.geometry.width,
            'height': obj.geometry.height,
            'aspect_ratio': obj.geometry.aspect_ratio,
            'vertex_count': obj.geometry.vertex_count,
            'is_closed': obj.geometry.is_closed,
            'is_convex': obj.geometry.is_convex,
            'has_holes': obj.geometry.has_holes,
            'hole_count': obj.geometry.hole_count,
            'selected': obj.selected,
            'assigned_layer': obj.assigned_layer,
            'metadata': {
                'layer_name': obj.metadata.layer_name,
                'color': obj.metadata.color,
                'line_type': obj.metadata.line_type,
                'thickness': obj.metadata.thickness
            }
        })
    
    return jsonify(response)


@layers_bp.route('/api/objects/<object_id>/select', methods=['POST'])
def select_object(object_id):
    """Select or deselect an object."""
    if not object_manager:
        return jsonify({'error': 'No DXF file loaded'}), 400
    
    action = request.json.get('action', 'toggle')
    
    if action == 'select':
        success = object_manager.select_object(object_id)
    elif action == 'deselect':
        success = object_manager.deselect_object(object_id)
    else:  # toggle
        obj = object_manager.objects.get(object_id)
        if obj:
            success = object_manager.deselect_object(object_id) if obj.selected else object_manager.select_object(object_id)
        else:
            success = False
    
    if success:
        obj = object_manager.objects[object_id]
        return jsonify({
            'success': True,
            'object_id': object_id,
            'selected': obj.selected
        })
    else:
        return jsonify({'error': 'Object not found'}), 404


@layers_bp.route('/api/objects/select-all', methods=['POST'])
def select_all_objects():
    """Select all objects."""
    if not object_manager:
        return jsonify({'error': 'No DXF file loaded'}), 400
    
    count = object_manager.select_all_objects()
    return jsonify({
        'success': True,
        'selected_count': count,
        'total_count': len(object_manager.objects)
    })


@layers_bp.route('/api/objects/deselect-all', methods=['POST'])
def deselect_all_objects():
    """Deselect all objects."""
    if not object_manager:
        return jsonify({'error': 'No DXF file loaded'}), 400
    
    count = object_manager.deselect_all_objects()
    return jsonify({
        'success': True,
        'deselected_count': count
    })


@layers_bp.route('/api/objects/filter', methods=['POST'])
def filter_objects():
    """Filter objects based on criteria."""
    if not object_manager:
        return jsonify({'error': 'No DXF file loaded'}), 400
    
    filters = request.json
    filtered_objects = object_manager.filter_objects(
        min_area=filters.get('min_area'),
        max_area=filters.get('max_area'),
        min_perimeter=filters.get('min_perimeter'),
        max_perimeter=filters.get('max_perimeter'),
        object_types=[ObjectType(t) for t in filters.get('object_types', [])],
        complexities=[ObjectComplexity(c) for c in filters.get('complexities', [])]
    )
    
    response = []
    for obj in filtered_objects:
        response.append({
            'object_id': obj.object_id,
            'object_type': obj.object_type.value,
            'complexity': obj.complexity.value,
            'area': obj.geometry.area,
            'perimeter': obj.geometry.perimeter,
            'selected': obj.selected,
            'assigned_layer': obj.assigned_layer
        })
    
    return jsonify(response)


@layers_bp.route('/api/layers', methods=['GET'])
def get_layers():
    """Get all layers."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    layers = layer_manager.get_all_layers()
    response = []
    
    for layer in layers:
        stats = layer_manager.calculate_layer_statistics(layer.layer_id)
        response.append({
            'layer_id': layer.layer_id,
            'name': layer.name,
            'layer_type': layer.layer_type.value,
            'description': layer.description,
            'status': layer.status.value,
            'object_count': stats.get('object_count', 0),
            'total_area': stats.get('total_area', 0.0),
            'created_at': layer.created_at.isoformat(),
            'updated_at': layer.updated_at.isoformat()
        })
    
    return jsonify(response)


@layers_bp.route('/api/layers', methods=['POST'])
def create_layer():
    """Create a new layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    data = request.json
    layer_id = layer_manager.create_layer(
        name=data['name'],
        layer_type=LayerType(data['layer_type']),
        description=data.get('description', '')
    )
    
    return jsonify({
        'success': True,
        'layer_id': layer_id
    })


@layers_bp.route('/api/layers/<layer_id>', methods=['DELETE'])
def delete_layer(layer_id):
    """Delete a layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    success = layer_manager.delete_layer(layer_id)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Layer not found'}), 404


@layers_bp.route('/api/layers/<layer_id>/objects', methods=['POST'])
def assign_objects_to_layer(layer_id):
    """Assign objects to a layer."""
    if not layer_manager or not object_manager:
        return jsonify({'error': 'Managers not available'}), 400
    
    data = request.json
    object_ids = data.get('object_ids', [])
    
    success_count = 0
    for obj_id in object_ids:
        if layer_manager.add_object_to_layer(layer_id, obj_id, object_manager):
            success_count += 1
    
    return jsonify({
        'success': True,
        'assigned_count': success_count,
        'total_requested': len(object_ids)
    })


@layers_bp.route('/api/layers/<layer_id>/objects/<object_id>', methods=['DELETE'])
def remove_object_from_layer(layer_id, object_id):
    """Remove an object from a layer."""
    if not layer_manager or not object_manager:
        return jsonify({'error': 'Managers not available'}), 400
    
    success = layer_manager.remove_object_from_layer(layer_id, object_id, object_manager)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Object or layer not found'}), 404


@layers_bp.route('/api/layers/<layer_id>/statistics', methods=['GET'])
def get_layer_statistics(layer_id):
    """Get statistics for a layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    stats = layer_manager.calculate_layer_statistics(layer_id)
    return jsonify(stats)


@layers_bp.route('/api/layers/<layer_id>/duplicate', methods=['POST'])
def duplicate_layer(layer_id):
    """Duplicate a layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    data = request.json
    new_name = data.get('name', f'Copy of Layer {layer_id}')
    
    new_layer_id = layer_manager.duplicate_layer(layer_id, new_name)
    if new_layer_id:
        return jsonify({
            'success': True,
            'new_layer_id': new_layer_id
        })
    else:
        return jsonify({'error': 'Layer not found'}), 404


@layers_bp.route('/api/layers/reorder', methods=['POST'])
def reorder_layers():
    """Reorder layers."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    data = request.json
    new_order = data.get('layer_order', [])
    
    success = layer_manager.reorder_layers(new_order)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Invalid layer order'}), 400


@layers_bp.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall statistics."""
    if not object_manager:
        return jsonify({'error': 'No DXF file loaded'}), 400
    
    object_stats = object_manager.get_object_statistics()
    layer_summary = layer_manager.get_layer_summary() if layer_manager else {}
    
    return jsonify({
        'object_statistics': object_stats,
        'layer_summary': layer_summary
    })


@layers_bp.route('/api/export/selection', methods=['POST'])
def export_selection():
    """Export selected objects."""
    if not object_manager:
        return jsonify({'error': 'No DXF file loaded'}), 400
    
    export_dir = 'exports'
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, f"selection_{uuid.uuid4().hex}.json")
    
    success = object_manager.export_selection(file_path)
    if success:
        return jsonify({
            'success': True,
            'file_path': file_path
        })
    else:
        return jsonify({'error': 'Export failed'}), 500


@layers_bp.route('/api/export/layer/<layer_id>', methods=['POST'])
def export_layer(layer_id):
    """Export layer configuration."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    export_dir = 'exports'
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, f"layer_{layer_id}_{uuid.uuid4().hex}.json")
    
    success = layer_manager.export_layer_configuration(layer_id, file_path)
    if success:
        return jsonify({
            'success': True,
            'file_path': file_path
        })
    else:
        return jsonify({'error': 'Export failed'}), 500


@layers_bp.route('/api/layers/<layer_id>/optimize', methods=['POST'])
def optimize_layer_nesting(layer_id):
    """Optimize nesting for a layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    layer = layer_manager.get_layer(layer_id)
    if not layer:
        return jsonify({'error': 'Layer not found'}), 404
    
    if not layer.objects:
        return jsonify({'error': 'No objects in layer to optimize'}), 400
    
    try:
        # Initialize nesting engine
        nesting_engine = NestingEngine()
        
        # Run optimization
        result = nesting_engine.optimize_nesting(layer)
        
        # Prepare response
        response = {
            'success': result.success,
            'algorithm_used': result.algorithm_used,
            'optimization_time': result.optimization_time,
            'iterations_completed': result.iterations_completed,
            'final_utilization': result.final_utilization,
            'improvement_percentage': result.improvement_percentage,
            'positioned_objects': len(result.positioned_objects),
            'failed_objects': len(result.failed_objects),
            'waste_area': result.waste_area,
            'waste_percentage': result.waste_percentage,
            'status': result.status.value,
            'warnings': result.warnings,
            'errors': result.errors
        }
        
        # Add positioned object details
        positioned_details = []
        for pos_obj in result.positioned_objects:
            positioned_details.append({
                'object_id': pos_obj.object.object_id,
                'x': pos_obj.x,
                'y': pos_obj.y,
                'rotation': pos_obj.rotation,
                'area': pos_obj.geometry.area,
                'positioning_time': pos_obj.positioning_time
            })
        response['positioned_details'] = positioned_details
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Nesting optimization failed: {e}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500


@layers_bp.route('/api/layers/<layer_id>/compare-algorithms', methods=['POST'])
def compare_nesting_algorithms(layer_id):
    """Compare different nesting algorithms for a layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    layer = layer_manager.get_layer(layer_id)
    if not layer:
        return jsonify({'error': 'Layer not found'}), 404
    
    if not layer.objects:
        return jsonify({'error': 'No objects in layer to optimize'}), 400
    
    try:
        # Initialize nesting engine
        nesting_engine = NestingEngine()
        
        # Compare algorithms
        results = nesting_engine.compare_algorithms(layer)
        
        # Prepare response
        comparison = {}
        for algorithm_name, result in results.items():
            comparison[algorithm_name] = {
                'success': result.success,
                'optimization_time': result.optimization_time,
                'iterations_completed': result.iterations_completed,
                'final_utilization': result.final_utilization,
                'improvement_percentage': result.improvement_percentage,
                'positioned_objects': len(result.positioned_objects),
                'failed_objects': len(result.failed_objects),
                'waste_area': result.waste_area,
                'waste_percentage': result.waste_percentage,
                'status': result.status.value,
                'warnings': result.warnings,
                'errors': result.errors
            }
        
        # Find best algorithm
        best_algorithm = max(comparison.keys(), 
                           key=lambda k: comparison[k]['final_utilization'])
        
        return jsonify({
            'success': True,
            'comparison': comparison,
            'best_algorithm': best_algorithm,
            'best_utilization': comparison[best_algorithm]['final_utilization']
        })
        
    except Exception as e:
        logger.error(f"Algorithm comparison failed: {e}")
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500


@layers_bp.route('/api/layers/<layer_id>/utilization-report', methods=['POST'])
def generate_utilization_report(layer_id):
    """Generate material utilization report for a layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    layer = layer_manager.get_layer(layer_id)
    if not layer:
        return jsonify({'error': 'Layer not found'}), 404
    
    if not layer.optimization_result:
        return jsonify({'error': 'Layer has not been optimized yet'}), 400
    
    try:
        # Initialize reporter
        reporter = MaterialUtilizationReporter()
        
        # Generate report
        report_path = reporter.generate_comprehensive_report(layer, layer.optimization_result)
        
        return jsonify({
            'success': True,
            'report_path': report_path,
            'message': 'Utilization report generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500


@layers_bp.route('/api/layers/<layer_id>/utilization-analysis', methods=['GET'])
def get_utilization_analysis(layer_id):
    """Get utilization analysis for a layer."""
    if not layer_manager:
        return jsonify({'error': 'No layer manager available'}), 400
    
    layer = layer_manager.get_layer(layer_id)
    if not layer:
        return jsonify({'error': 'Layer not found'}), 404
    
    if not layer.optimization_result:
        return jsonify({'error': 'Layer has not been optimized yet'}), 400
    
    try:
        from ..nesting import calculate_layer_utilization
        
        # Calculate utilization metrics
        utilization_report = calculate_layer_utilization(layer, layer.optimization_result)
        
        # Prepare response
        response = {
            'layer_id': utilization_report.layer_id,
            'layer_name': utilization_report.layer_name,
            'algorithm_used': utilization_report.algorithm_used,
            'optimization_time': utilization_report.optimization_time,
            'material_width': utilization_report.material_width,
            'material_height': utilization_report.material_height,
            'material_area': utilization_report.material_area,
            'used_area': utilization_report.used_area,
            'waste_area': utilization_report.waste_area,
            'utilization_percentage': utilization_report.utilization_percentage,
            'total_objects': utilization_report.total_objects,
            'positioned_objects': utilization_report.positioned_objects,
            'failed_objects': utilization_report.failed_objects,
            'positioning_success_rate': utilization_report.positioning_success_rate,
            'material_cost': utilization_report.material_cost,
            'cutting_cost': utilization_report.cutting_cost,
            'total_cost': utilization_report.total_cost,
            'cost_per_object': utilization_report.cost_per_object,
            'cost_per_area': utilization_report.cost_per_area,
            'cutting_length': utilization_report.cutting_length,
            'estimated_cutting_time': utilization_report.estimated_cutting_time,
            'pierce_count': utilization_report.pierce_count,
            'efficiency_score': utilization_report.efficiency_score,
            'improvement_percentage': utilization_report.improvement_percentage,
            'iterations_completed': utilization_report.iterations_completed,
            'convergence_rate': utilization_report.convergence_rate,
            'recommendations': utilization_report.recommendations,
            'warnings': utilization_report.warnings
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Utilization analysis failed: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


# ----------------------- Additional API endpoints used by enhanced DXF page -----------------------

@layers_bp.route('/api/generate-gcode', methods=['POST'])
def api_generate_gcode():
    """Generate G-code for the currently loaded DXF."""
    try:
        if not current_dxf_path or not os.path.exists(current_dxf_path):
            return jsonify({'success': False, 'error': 'No DXF file loaded'}), 400

        payload = request.get_json(silent=True) or {}
        feed = float(payload.get('feed', 1200.0))
        m_on = payload.get('m_on', 'M62')
        m_off = payload.get('m_off', 'M63')
        pierce_ms = int(payload.get('pierce_ms', 500))

        result = generate_gcode_from_dxf(
            current_dxf_path,
            OUTPUT_ROOT,
            feed=feed,
            m_on=m_on,
            m_off=m_off,
            pierce_ms=pierce_ms,
        )

        return jsonify({
            'success': True,
            'gcode': result['gcode_preview'],
            'line_count': result['line_count'],
            'estimated_time': f"{result['estimated_time_minutes']:.1f} min",
            'gcode_id': result['gcode_id'],
            'gcode_path': result['gcode_path'],
            'metrics': result['metrics'],
        })
    except (DXFProcessingError, ValueError, FileNotFoundError) as exc:
        logger.error(f"G-code generation failed: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        logger.error(f"G-code generation failed: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@layers_bp.route('/api/calculate-costs', methods=['POST'])
def api_calculate_costs():
    """Calculate manufacturing costs for the currently loaded DXF."""
    try:
        if not current_dxf_path or not os.path.exists(current_dxf_path):
            return jsonify({'success': False, 'error': 'No DXF file loaded'}), 400

        payload = request.get_json(silent=True) or {}
        from wjp_analyser.services.costing_service import estimate_cost
        costs = estimate_cost(current_dxf_path)

        return jsonify({'success': True, 'costs': costs})
    except (DXFProcessingError, ValueError, FileNotFoundError) as exc:
        logger.error(f"Cost calculation failed: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        logger.error(f"Cost calculation failed: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


def create_default_layers(object_manager, all_objects):
    """Create default layer structure."""
    # Create base layer
    base_layer_id = layer_manager.create_base_layer(all_objects)
    
    # Create nested layer for optimization
    nested_layer_id = layer_manager.create_layer(
        name="Nested Layer",
        layer_type=LayerType.NESTED,
        description="Optimized material utilization"
    )
    
    # Create priority layer for important objects
    priority_layer_id = layer_manager.create_layer(
        name="Priority Layer",
        layer_type=LayerType.PRIORITY,
        description="High-priority objects"
    )
    
    return {
        'base': base_layer_id,
        'nested': nested_layer_id,
        'priority': priority_layer_id
    }


def run_interactive_interface(host='localhost', port=5000, debug=True):
    """Run the interactive interface."""
    logger.info(f"Starting interactive interface on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_interactive_interface()
