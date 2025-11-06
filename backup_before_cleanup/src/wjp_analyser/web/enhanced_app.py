#!/usr/bin/env python3
"""
Enhanced Web Interface for Comprehensive Waterjet Analyser Workflow
Integrates both Image Upload and DXF Upload workflows with real-time processing.
"""

import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from typing import Dict, Any, Optional
import time

# Import workflow manager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.wjp_analyser.workflow.workflow_manager import (
        WorkflowManager, WorkflowConfig, WorkflowType, ValidationLevel
    )
except ImportError:
    # Fallback import for when running from different directory
    from workflow.workflow_manager import (
        WorkflowManager, WorkflowConfig, WorkflowType, ValidationLevel
    )

app = Flask(__name__)
app.secret_key = 'waterjet_analyser_secret_key'

# Configuration
UPLOAD_FOLDER = 'output/temp'
OUTPUT_FOLDER = 'output'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
ALLOWED_DXF_EXTENSIONS = {'dxf'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route("/")
def index():
    """Display the main workflow selection page."""
    return render_template("workflow_index.html")

@app.route("/image-workflow")
def image_workflow():
    """Display the image upload workflow page."""
    return render_template("image_workflow.html")

@app.route("/dxf-workflow")
def dxf_workflow():
    """Display the DXF upload workflow page."""
    return render_template("dxf_workflow.html")

@app.route("/process-image", methods=["POST"])
def process_image():
    """Process image upload workflow."""
    try:
        # Get uploaded file
        uploaded_file = request.files.get("file")
        if not uploaded_file or uploaded_file.filename == "":
            flash("Please choose an image file to upload.")
            return redirect(url_for("image_workflow"))
        
        if not allowed_file(uploaded_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash("Unsupported file type. Please upload JPG, JPEG, PNG, BMP, or TIFF.")
            return redirect(url_for("image_workflow"))
        
        # Save uploaded file
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)
        
        # Create workflow session
        workflow_id = f"img_{uuid.uuid4().hex[:8]}"
        session['workflow_id'] = workflow_id
        session['workflow_type'] = 'image'
        session['input_file'] = file_path
        
        # Create output directory
        output_dir = os.path.join(OUTPUT_FOLDER, workflow_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure workflow
        config = WorkflowConfig(
            workflow_type=WorkflowType.IMAGE_UPLOAD,
            validation_level=ValidationLevel.STANDARD,
            enable_toolpath_analysis=True,
            enable_cost_estimation=True
        )
        
        # Execute workflow
        manager = WorkflowManager(config)
        results = manager.execute_image_upload_workflow(file_path, output_dir)
        
        # Save results
        results_path = os.path.join(output_dir, "workflow_results.json")
        manager.save_results(results_path)
        
        # Store results in session
        session['workflow_results'] = results
        
        # Redirect to results page
        return redirect(url_for("workflow_results", workflow_id=workflow_id))
        
    except Exception as e:
        flash(f"Error processing image: {str(e)}")
        return redirect(url_for("image_workflow"))

@app.route("/process-dxf", methods=["POST"])
def process_dxf():
    """Process DXF upload workflow."""
    try:
        # Get uploaded file
        uploaded_file = request.files.get("file")
        if not uploaded_file or uploaded_file.filename == "":
            flash("Please choose a DXF file to upload.")
            return redirect(url_for("dxf_workflow"))
        
        if not allowed_file(uploaded_file.filename, ALLOWED_DXF_EXTENSIONS):
            flash("Unsupported file type. Please upload a DXF file.")
            return redirect(url_for("dxf_workflow"))
        
        # Save uploaded file
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)
        
        # Get user inputs
        user_inputs = {
            "width": float(request.form.get("width", 100.0)),
            "height": float(request.form.get("height", 100.0)),
            "material": request.form.get("material", "steel"),
            "thickness": float(request.form.get("thickness", 10.0)),
            "sheet_width": float(request.form.get("sheet_width", 1000.0)),
            "sheet_height": float(request.form.get("sheet_height", 1000.0)),
            "quality_level": request.form.get("quality_level", "standard")
        }
        
        # Create workflow session
        workflow_id = f"dxf_{uuid.uuid4().hex[:8]}"
        session['workflow_id'] = workflow_id
        session['workflow_type'] = 'dxf'
        session['input_file'] = file_path
        session['user_inputs'] = user_inputs
        
        # Create output directory
        output_dir = os.path.join(OUTPUT_FOLDER, workflow_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure workflow
        config = WorkflowConfig(
            workflow_type=WorkflowType.DXF_UPLOAD,
            validation_level=ValidationLevel.STANDARD,
            enable_toolpath_analysis=True,
            enable_cost_estimation=True,
            enable_nesting=True,
            enable_layer_management=True
        )
        
        # Execute workflow
        manager = WorkflowManager(config)
        results = manager.execute_dxf_upload_workflow(file_path, user_inputs, output_dir)
        
        # Save results
        results_path = os.path.join(output_dir, "workflow_results.json")
        manager.save_results(results_path)
        
        # Store results in session
        session['workflow_results'] = results
        
        # Redirect to results page
        return redirect(url_for("workflow_results", workflow_id=workflow_id))
        
    except Exception as e:
        flash(f"Error processing DXF: {str(e)}")
        return redirect(url_for("dxf_workflow"))

@app.route("/workflow-results/<workflow_id>")
def workflow_results(workflow_id: str):
    """Display workflow results."""
    try:
        # Get results from session or load from file
        results = session.get('workflow_results')
        if not results:
            results_path = os.path.join(OUTPUT_FOLDER, workflow_id, "workflow_results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
            else:
                flash("Workflow results not found.")
                return redirect(url_for("index"))
        
        # Determine workflow type
        workflow_type = results.get('workflow_type', 'unknown')
        
        # Render appropriate results template
        if workflow_type == 'image_upload':
            return render_template("image_workflow_results.html", 
                                results=results, 
                                workflow_id=workflow_id)
        elif workflow_type == 'dxf_upload':
            return render_template("dxf_workflow_results.html", 
                                results=results, 
                                workflow_id=workflow_id)
        else:
            return render_template("workflow_results.html", 
                                results=results, 
                                workflow_id=workflow_id)
        
    except Exception as e:
        flash(f"Error displaying results: {str(e)}")
        return redirect(url_for("index"))

@app.route("/api/workflow-status/<workflow_id>")
def workflow_status(workflow_id: str):
    """API endpoint for real-time workflow status."""
    try:
        results_path = os.path.join(OUTPUT_FOLDER, workflow_id, "workflow_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            return jsonify({
                "status": "completed",
                "current_step": results.get("current_step", "unknown"),
                "steps_completed": results.get("steps_completed", []),
                "errors": results.get("errors", []),
                "warnings": results.get("warnings", [])
            })
        else:
            return jsonify({
                "status": "not_found",
                "current_step": "unknown",
                "steps_completed": [],
                "errors": ["Workflow not found"],
                "warnings": []
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "current_step": "error",
            "steps_completed": [],
            "errors": [str(e)],
            "warnings": []
        })

@app.route("/api/layer-management/<workflow_id>")
def layer_management(workflow_id: str):
    """API endpoint for real-time layer management."""
    try:
        results_path = os.path.join(OUTPUT_FOLDER, workflow_id, "workflow_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            layer_info = results.get("layer_info", [])
            return jsonify({
                "layers": layer_info,
                "total_layers": len(layer_info),
                "status": "success"
            })
        else:
            return jsonify({
                "layers": [],
                "total_layers": 0,
                "status": "not_found"
            })
    except Exception as e:
        return jsonify({
            "layers": [],
            "total_layers": 0,
            "status": "error",
            "error": str(e)
        })

@app.route("/download-results/<workflow_id>")
def download_results(workflow_id: str):
    """Download workflow results as JSON."""
    try:
        results_path = os.path.join(OUTPUT_FOLDER, workflow_id, "workflow_results.json")
        if os.path.exists(results_path):
            return jsonify({"download_url": f"/api/download-file/{workflow_id}/workflow_results.json"})
        else:
            return jsonify({"error": "Results not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download-file/<workflow_id>/<filename>")
def download_file(workflow_id: str, filename: str):
    """Download specific file from workflow output."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, workflow_id, filename)
        if os.path.exists(file_path):
            from flask import send_file
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/preview-dxf/<workflow_id>")
def preview_dxf(workflow_id: str):
    """Generate DXF preview."""
    try:
        results_path = os.path.join(OUTPUT_FOLDER, workflow_id, "workflow_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Get DXF path
            dxf_path = results.get("dxf_path") or results.get("resized_dxf_path")
            if dxf_path and os.path.exists(dxf_path):
                # Generate preview (implement preview generation)
                preview_path = dxf_path.replace(".dxf", "_preview.png")
                # TODO: Implement DXF preview generation
                return jsonify({"preview_url": preview_path})
            else:
                return jsonify({"error": "DXF file not found"}), 404
        else:
            return jsonify({"error": "Workflow results not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
