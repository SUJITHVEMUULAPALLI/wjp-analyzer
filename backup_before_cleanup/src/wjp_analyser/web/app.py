#!/usr/bin/env python3
"""
Waterjet DXF Analyzer - Web Interface
=====================================

Flask application for uploading DXF files or raster images, converting images
into DXF via contour extraction, and displaying analysis results.
"""

import json
import os
import sys
import uuid
from typing import Dict, Tuple

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
    abort,
)
from werkzeug.utils import secure_filename
import requests

# Add src directory to path for image processing imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import AI image interface
from .ai_image_interface import ai_image_bp

# Import layer selection interface
from ..object_management.interactive_interface import create_app as create_layer_app

# Import image analyzer
from ..image_analyzer import quick_analyze, ImageAnalyzerGate, AnalyzerConfig

# Import unified web manager
from .unified_web_manager import web_manager, get_unified_converter, get_conversion_params, create_conversion_params, create_analyze_args
# Optional quote export (PDF/XLSX); provide safe fallbacks when unavailable
try:  # Attempt to import if present in project
    from ..io.quote_export import make_pdf, make_xlsx  # type: ignore
except Exception:  # Fallback stubs to avoid runtime NameError
    def make_pdf(*args, **kwargs):  # pragma: no cover - feature not yet implemented
        raise NotImplementedError("Quote export is not configured in this build.")

    def make_xlsx(*args, **kwargs):  # pragma: no cover - feature not yet implemented
        raise NotImplementedError("Quote export is not configured in this build.")

from importlib.util import find_spec
from importlib import resources as importlib_resources
import logging

# Import analysis components through unified manager
analyze_dxf = web_manager.get_dxf_analyzer()
AnalyzeArgs = web_manager.get_analyze_args()

# Import other components with fallbacks
from .interactive_api import api_bp as interactive_api_bp

# AI components through unified manager
ai_components = web_manager.get_ai_components()
OllamaAnalyzer = ai_components.get('OllamaAnalyzer')
OllamaConfig = ai_components.get('OllamaConfig')
OpenAIAnalyzer = ai_components.get('OpenAIAnalyzer')
OpenAIConfig = ai_components.get('OpenAIConfig')
import time
import traceback

# Use local paths for templates and static files
template_folder = os.path.join(os.path.dirname(__file__), "templates")
static_folder = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

# Register AI image interface blueprint
try:
    app.register_blueprint(ai_image_bp)
except Exception as e:
    print(f"Warning: Could not register AI image blueprint: {e}")

# Authentication routes
@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    """User login endpoint."""
    if request.method == 'GET':
        return render_template('auth/login.html')
    
    # Security check
    client_ip = request.remote_addr
    security_result = security_check(client_ip, '/auth/login', 'POST', dict(request.headers))
    
    if not security_result['allowed']:
        flash(f"Security check failed: {security_result['reason']}", 'error')
        return render_template('auth/login.html')
    
    email = request.form.get('email')
    password = request.form.get('password')
    
    if not email or not password:
        flash('Email and password are required', 'error')
        return render_template('auth/login.html')
    
    # Authenticate user
    auth_result = auth_manager.authenticate_user(email, password)
    
    if auth_result['success']:
        # Set session data
        session['user_id'] = auth_result['user']['id']
        session['email'] = auth_result['user']['email']
        session['role'] = auth_result['user']['role']
        session['access_token'] = auth_result['access_token']
        session['csrf_token'] = security_middleware.csrf_protection.generate_token(auth_result['session_id'])
        
        flash('Login successful!', 'success')
        return redirect(url_for('index'))
    else:
        flash(f"Login failed: {auth_result['error']}", 'error')
        return render_template('auth/login.html')

@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    """User registration endpoint."""
    if request.method == 'GET':
        return render_template('auth/register.html')
    
    # Security check
    client_ip = request.remote_addr
    security_result = security_check(client_ip, '/auth/register', 'POST', dict(request.headers))
    
    if not security_result['allowed']:
        flash(f"Security check failed: {security_result['reason']}", 'error')
        return render_template('auth/register.html')
    
    email = request.form.get('email')
    password = request.form.get('password')
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    
    if not email or not password:
        flash('Email and password are required', 'error')
        return render_template('auth/register.html')
    
    # Register user
    reg_result = auth_manager.register_user(email, password, first_name=first_name, last_name=last_name)
    
    if reg_result['success']:
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    else:
        flash(f"Registration failed: {reg_result['error']}", 'error')
        return render_template('auth/register.html')

@app.route('/auth/logout')
def logout():
    """User logout endpoint."""
    session_id = session.get('session_id')
    if session_id:
        auth_manager.logout_user(session_id)
    
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))

@app.route('/auth/profile')
def profile():
    """User profile page."""
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    # Get user data
    user_data = auth_manager.users.get(session.get('email'))
    if not user_data:
        flash('User not found', 'error')
        return redirect(url_for('login'))
    
    return render_template('auth/profile.html', user=user_data)

# Import configuration and utilities through unified manager
config_components = web_manager.get_config_components()
utility_components = web_manager.get_utility_components()

get_security_config = config_components.get('get_security_config')
get_ai_config = config_components.get('get_ai_config')
get_app_config = config_components.get('get_app_config')
validate_config = config_components.get('validate_config')

handle_errors = utility_components.get('handle_errors')
error_handler = utility_components.get('error_handler')
validate_uploaded_file = utility_components.get('validate_uploaded_file')
validate_material_params = utility_components.get('validate_material_params')
validate_image_params = utility_components.get('validate_image_params')
initialize_logging = utility_components.get('initialize_logging')
log_startup = utility_components.get('log_startup')
log_shutdown = utility_components.get('log_shutdown')
initialize_cache = utility_components.get('initialize_cache')

# Initialize secure configuration with fallbacks
security_config = get_security_config() if get_security_config else web_manager.config
ai_config = get_ai_config() if get_ai_config else web_manager.config
app_config = get_app_config() if get_app_config else web_manager.config

# Register layer selection interface blueprint
try:
    layer_app = create_layer_app()
    if layer_app is not None:
        app.register_blueprint(layer_app, url_prefix='/layers')
except Exception as e:
    print(f"Warning: Could not register layer app: {e}")

try:
    app.register_blueprint(interactive_api_bp)
except Exception as e:
    print(f"Warning: Could not register interactive API: {e}")
# Apply security configuration
if security_config is not None:
    app.config["MAX_CONTENT_LENGTH"] = security_config.max_upload_size
else:
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB default

# Initialize logging system
def _resolve_default_logs_dir() -> str:
    base = os.getenv("LOCALAPPDATA") or os.getenv("TMP") or os.getenv("TEMP") or os.getcwd()
    return os.path.join(base, "WJP_ANALYSER", "logs")

logging_config = {
    "level": app_config.log_level if app_config is not None else "INFO",
    "logs_folder": _resolve_default_logs_dir(),
    "console_output": True,
    "file_output": True
}
if initialize_logging is not None:
    initialize_logging(logging_config)

# Initialize cache system
def _resolve_default_cache_dir() -> str:
    base = os.getenv("LOCALAPPDATA") or os.getenv("TMP") or os.getenv("TEMP") or os.getcwd()
    return os.path.join(base, "WJP_ANALYSER", "cache")

if initialize_cache is not None:
    initialize_cache(_resolve_default_cache_dir(), memory_size=1000, file_size_mb=100)

# Validate configuration
if validate_config is not None and not validate_config():
    print("WARNING: Configuration validation failed. Check logs for details.")

# Apply security configuration with enhanced authentication
from ..auth.enhanced_auth import auth_manager, require_auth, Permission
from ..auth.api_key_manager import api_key_manager, migrate_existing_keys
from ..auth.security_middleware import security_middleware, security_check

# Migrate existing API keys to encrypted storage
migrate_existing_keys()

# Set secure secret key
if security_config is not None and hasattr(security_config, 'secret_key'):
    app.secret_key = security_config.secret_key
else:
    # Generate secure random key
    import secrets
    app.secret_key = secrets.token_urlsafe(32)
    logger.warning("Using generated secret key. Set WJP_SECRET_KEY environment variable for production.")

# Log application startup
if log_startup is not None:
    log_startup("0.1.0", {
        "security_config": {
            "max_upload_size": security_config.max_upload_size if security_config is not None else 32 * 1024 * 1024,
            "allowed_extensions": list(security_config.allowed_extensions) if security_config is not None else ["dxf", "jpg", "jpeg", "png", "bmp", "tiff"]
        },
        "ai_config": {
            "openai_available": ai_config.openai_api_key is not None if ai_config is not None else False,
            "ollama_url": ai_config.ollama_base_url if ai_config is not None else "http://localhost:11434"
        }
    })

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, app_config.upload_folder if app_config is not None else "output/temp")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, app_config.output_folder if app_config is not None else "output")
ALLOWED_EXTENSIONS = security_config.allowed_extensions if security_config is not None else {"dxf", "jpg", "jpeg", "png", "bmp", "tiff"}

DEFAULT_FORM = {
    "kerf": 1.1,
}


DEFAULT_NESTING_PARAMS = {
    "sheet_width": 3000.0,
    "sheet_height": 1500.0,
    "spacing": 10.0,
}

DEFAULT_GCODE_PARAMS = {
    "feed": 1200.0,
    "m_on": "M62",
    "m_off": "M63",
    "pierce_ms": 500,
}

ANALYSIS_FILES = {
    "report": "report.json",
    "lengths": "lengths.csv",
    "preview": "preview.png",
    "preview_full": "preview_full.png",
    "gcode": "program.nc",
    "image_metadata": "image_metadata.json",
    "image_gray": "image_gray.png",
    "image_threshold": "image_threshold.png",
    "image_edges": "image_edges.png",
}

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------- Path safety helpers ---------
import re

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_id(identifier: str) -> bool:
    return bool(SAFE_ID_RE.match(identifier or ""))


def _safe_join(base: str, *paths: str) -> str:
    base_real = os.path.realpath(base)
    target = os.path.realpath(os.path.join(base, *paths))
    if not target.startswith(base_real + os.sep) and target != base_real:
        raise ValueError("Unsafe path traversal detected")
    return target


def allowed_file(filename: str) -> bool:
    """Check if the incoming filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_int(raw: str, default: int) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default




def render_index(form_defaults=None):
    """Render the upload form with provided defaults."""
    form_values = {**DEFAULT_FORM, **(form_defaults or {})}
    return render_template(
        "index.html",
        form_defaults=form_values,
        allowed_extensions=sorted(ALLOWED_EXTENSIONS),
    )


@app.route("/")
def index():
    """Display the workflow steps."""
    return render_template("index.html")


@app.route("/files/<path:filename>")
def serve_uploaded_file(filename: str):
    """Serve files from the upload/temp directory for viewer consumption."""
    try:
        safe_path = _safe_join(UPLOAD_FOLDER, filename)
    except ValueError:
        abort(400)
    if not os.path.exists(safe_path):
        abort(404)
    # Best-effort content type; most viewers detect by extension
    return send_file(safe_path, as_attachment=False)


@app.route("/image-analysis")
def image_analysis():
    """Display image analysis upload page."""
    return render_template("image_analysis.html")


@app.route("/image-analysis/<analysis_id>")
def image_analysis_results(analysis_id):
    """Display image analysis results."""
    try:
        # Load analysis report
        output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
        report_path = os.path.join(output_dir, "image_analysis_report.json")
        
        if not os.path.exists(report_path):
            flash("Analysis report not found.")
            return redirect(url_for("index"))
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return render_template("image_analysis_results.html", 
                             analysis_id=analysis_id,
                             report=report)
        
    except Exception as e:
        flash(f"Error loading analysis results: {e}")
        return redirect(url_for("index"))


@app.route("/image-to-dxf")
def image_to_dxf():
    abort(404)


@app.route("/handle-relayer")
def handle_relayer():
    """Simple UI to move entities by handles to a target layer."""
    return render_template(
        "handle_relayer.html",
        upload_folder=UPLOAD_FOLDER,
        default_sheet_w=DEFAULT_NESTING_PARAMS["sheet_width"],
        default_sheet_h=DEFAULT_NESTING_PARAMS["sheet_height"],
    )


@app.route("/viewer")
def simple_dxf_viewer():
    """Render a DXF file to an image and display it.

    Usage: /viewer?file=<filename.dxf>
    The file must be under UPLOAD_FOLDER. The rendered PNG is cached until the DXF changes.
    """
    fname = request.args.get("file", "").strip()
    if not fname:
        abort(400)
    try:
        safe_dxf = _safe_join(UPLOAD_FOLDER, fname)
    except ValueError:
        abort(400)
    if not os.path.exists(safe_dxf):
        abort(404)

    png_name = os.path.splitext(fname)[0] + ".png"
    safe_png = os.path.join(UPLOAD_FOLDER, png_name)

    try:
        dxf_mtime = os.path.getmtime(safe_dxf)
        png_mtime = os.path.getmtime(safe_png) if os.path.exists(safe_png) else 0
        needs_render = (png_mtime < dxf_mtime)
    except Exception:
        needs_render = True

    if needs_render:
        try:
            # Headless rendering
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import ezdxf
            from ezdxf.addons.drawing import RenderContext
            from ezdxf.addons.drawing import matplotlib as ezdxf_mpl

            doc = ezdxf.readfile(safe_dxf)
            msp = doc.modelspace()

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_aspect('equal')
            ax.axis('off')

            ctx = RenderContext(doc)
            out = ezdxf_mpl.MatplotlibBackend(ax)
            ezdxf_mpl.draw_layout(msp, ctx, out, finalize=True)

            os.makedirs(os.path.dirname(safe_png), exist_ok=True)
            fig.savefig(safe_png, dpi=150, transparent=True)
            plt.close(fig)
        except Exception:
            err = traceback.format_exc()
            return render_template("viewer_error.html", error=err, filename=fname), 500

    file_url = url_for('serve_uploaded_file', filename=fname)
    png_url = url_for('serve_uploaded_file', filename=png_name)
    return render_template("simple_viewer.html", filename=fname, file_url=file_url, png_url=png_url)


@app.route("/dxf-analysis")
def dxf_analysis():
    """Display the enhanced DXF analysis form with layer management and nesting."""
    form_values = {**DEFAULT_FORM}
    return render_template(
        "dxf_analysis.html",
        form_defaults=form_values,
    )


@app.route("/nesting")
def nesting():
    """Redirect to the advanced layer selection interface."""
    return redirect(url_for('layers.index'))

# Enhanced DXF Analysis API endpoints
@app.route("/api/upload-dxf", methods=["POST"])
def api_upload_dxf():
    """API endpoint for uploading and analyzing DXF files."""
    try:
        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"success": False, "error": "No file uploaded"})

        if not allowed_file(uploaded_file.filename):
            return jsonify({"success": False, "error": "Invalid file type"})

        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)

        # Analyze DXF objects
        from ..object_management import DXFObjectManager
        object_manager = DXFObjectManager()
        objects = object_manager.analyze_dxf_objects(file_path)
        
        return jsonify({
            "success": True,
            "objects": [obj.to_dict() for obj in objects],
            "file_path": file_path
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/layers", methods=["POST"])
def api_create_layer():
    """API endpoint for creating layers."""
    try:
        data = request.get_json()
        layer_name = data.get("name")
        layer_type = data.get("layer_type", "custom")
        description = data.get("description", "")
        
        if not layer_name:
            return jsonify({"success": False, "error": "Layer name is required"})
        
        # Create layer using LayerManager
        from ..object_management import LayerManager, LayerType
        layer_manager = LayerManager()
        
        # Convert string to enum
        layer_type_enum = LayerType.CUSTOM
        if layer_type == "base":
            layer_type_enum = LayerType.BASE
        elif layer_type == "nested":
            layer_type_enum = LayerType.NESTED
        
        layer_id = layer_manager.create_layer(
            name=layer_name,
            layer_type=layer_type_enum,
            description=description
        )
        
        layer = layer_manager.get_layer(layer_id)
        
        return jsonify({
            "success": True,
            "layer": {
                "layer_id": layer_id,
                "name": layer.name,
                "layer_type": layer.layer_type.value,
                "object_count": len(layer.objects),
                "description": layer.description
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/layers/<layer_id>/optimize", methods=["POST"])
def api_optimize_layer(layer_id):
    """API endpoint for optimizing layer nesting."""
    try:
        from ..object_management import LayerManager
        from ..nesting import NestingEngine
        
        layer_manager = LayerManager()
        layer = layer_manager.get_layer(layer_id)
        
        if not layer:
            return jsonify({"success": False, "error": "Layer not found"})
        
        # Run nesting optimization
        nesting_engine = NestingEngine()
        result = nesting_engine.optimize_nesting(layer)
        
        return jsonify({
            "success": True,
            "result": {
                "utilization": result.final_utilization,
                "sheets_required": result.sheets_required,
                "total_cost": result.total_cost,
                "cutting_time": result.cutting_time
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/generate-gcode", methods=["POST"])
def api_generate_gcode():
    """API endpoint for generating G-code."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        
        if not layer_id:
            return jsonify({"success": False, "error": "Layer ID is required"})
        
        # Generate G-code (placeholder implementation)
        gcode_lines = [
            "G21 ; Set units to millimeters",
            "G90 ; Absolute positioning",
            "G0 X0 Y0 ; Move to origin",
            "M3 S1000 ; Start spindle",
            "G1 X10 Y10 F1000 ; Cut line",
            "M5 ; Stop spindle",
            "G0 X0 Y0 ; Return to origin",
            "M30 ; End program"
        ]
        
        return jsonify({
            "success": True,
            "gcode": gcode_lines,
            "line_count": len(gcode_lines),
            "estimated_time": "12.5 min"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/calculate-costs", methods=["POST"])
def api_calculate_costs():
    """API endpoint for calculating costs."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        
        if not layer_id:
            return jsonify({"success": False, "error": "Layer ID is required"})
        
        # Calculate costs (placeholder implementation)
        costs = {
            "total_cost": 245.50,
            "material_cost": 180.25,
            "cutting_cost": 45.75,
            "setup_cost": 19.50,
            "waste_cost": 12.00
        }
        
        return jsonify({
            "success": True,
            "costs": costs
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/gcode-generation")
def gcode_generation():
    """Display the G-code generation form."""
    gcode_values = {**DEFAULT_GCODE_PARAMS}
    return render_template(
        "gcode_generation.html",
        gcode_defaults=gcode_values,
    )


@app.route("/ai-status")
def ai_status():
    """Check AI model status and connectivity."""
    try:
        from ..ai.ollama_client import OllamaAnalyzer, OllamaConfig
        
        # Test Ollama connection
        ollama_config = OllamaConfig()
        ollama_analyzer = OllamaAnalyzer(ollama_config)
        
        ollama_connected = ollama_analyzer._check_ollama_connection()
        available_models = ollama_analyzer._get_available_models()
        
        # Test OpenAI connection
        openai_available = False
        try:
            from ..ai.openai_client import OpenAIAnalyzer, OpenAIConfig
            openai_config = OpenAIConfig()
            openai_analyzer = OpenAIAnalyzer(openai_config)
            openai_available = openai_analyzer.client is not None
        except Exception:
            openai_available = False
        
        status = {
            "ollama_connected": ollama_connected,
            "ollama_models": available_models,
            "openai_available": openai_available,
            "timestamp": str(uuid.uuid4().hex[:8])
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ai-analysis")
def ai_analysis():
    """Display the AI analysis form."""
    return render_template("ai_analysis.html")


@app.route("/run-ai-analysis", methods=["POST"])
def run_ai_analysis():
    """Run AI analysis on DXF file."""
    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please choose a DXF file to analyze.")
        return redirect(url_for("ai_analysis"))

    if not allowed_file(uploaded_file.filename):
        flash("Please upload a DXF file.")
        return redirect(url_for("ai_analysis"))

    # Get user selections
    selected_model = request.form.get("ai_model", "auto")
    timeout_seconds = int(request.form.get("timeout", "300"))

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)

    base_name, file_ext = os.path.splitext(filename)
    file_ext = file_ext.lower().lstrip(".")
    
    if file_ext != "dxf":
        flash("Please upload a valid DXF file.")
        return redirect(url_for("ai_analysis"))

    ai_analysis_id = f"{base_name}_ai_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(OUTPUT_FOLDER, ai_analysis_id)
    os.makedirs(output_dir, exist_ok=True)

    # First run basic DXF analysis to get data for AI
    if Args is not None:
        analysis_args = Args(
            material="DXF Validation",
            thickness=25.0,
            kerf=1.1,
            rate_per_m=825.0,
            out=output_dir,
        )
    else:
        flash("DXF analyzer not available. Please check installation.")
        return render_index(form_values, {})

    try:
        if analyze is not None:
            analyze(file_path, analysis_args)
        else:
            flash("DXF analyzer not available. Please check installation.")
            return render_index(form_values, {})
        
        # Load analysis data for AI
        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
            
            # Run AI analysis based on user selection
            ai_analysis = None
            
            if selected_model == "auto":
                # Auto mode: try models in order of preference (skip OpenAI if no key)
                models_to_try = [
                    ("waterjet:latest", "Ollama (waterjet:latest)"),
                    ("gpt-oss:20b", "Ollama (gpt-oss:20b)"),
                    ("llama3.2-vision:latest", "Ollama (llama3.2-vision)"),
                ]
                
                # Only add OpenAI if API key is available
                try:
                    from ..ai.openai_client import OpenAIAnalyzer, OpenAIConfig
                    openai_config = OpenAIConfig()
                    openai_analyzer = OpenAIAnalyzer(openai_config)
                    if openai_analyzer.client is not None:
                        models_to_try.append(("openai", "OpenAI (gpt-4)"))
                except Exception:
                    pass  # Skip OpenAI if not available
                
                for model_name, display_name in models_to_try:
                    try:
                        if model_name == "openai":
                            openai_config = OpenAIConfig()
                            openai_analyzer = OpenAIAnalyzer(openai_config)
                            ai_analysis = openai_analyzer.analyze_dxf_manufacturing(file_path, analysis_data)
                        else:
                            ollama_config = OllamaConfig(model=model_name, timeout=timeout_seconds)
                            ollama_analyzer = OllamaAnalyzer(ollama_config)
                            ai_analysis = ollama_analyzer.analyze_dxf_manufacturing(file_path, analysis_data)
                        
                        if ai_analysis:
                            # Add model_used attribute to the analysis
                            ai_analysis_dict = ai_analysis.dict()
                            ai_analysis_dict['model_used'] = display_name
                            ai_analysis = ManufacturingAnalysis(**ai_analysis_dict)
                            break
                            
                    except Exception as e:
                        print(f"Model {model_name} failed: {e}")
                        continue
                        
            else:
                # Specific model selected
                try:
                    if selected_model == "openai":
                        # Check if OpenAI is available
                        try:
                            from ..ai.openai_client import OpenAIAnalyzer, OpenAIConfig
                            openai_config = OpenAIConfig()
                            openai_analyzer = OpenAIAnalyzer(openai_config)
                            if openai_analyzer.client is None:
                                flash("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable or update config/api_keys.yaml")
                                return redirect(url_for("ai_analysis"))
                            
                            ai_analysis = openai_analyzer.analyze_dxf_manufacturing(file_path, analysis_data)
                            if ai_analysis:
                                # Update model_used attribute
                                ai_analysis.model_used = "OpenAI (gpt-4)"
                        except Exception as e:
                            flash(f"OpenAI not available: {str(e)}. Please configure your API key.")
                            return redirect(url_for("ai_analysis"))
                    else:
                        # Use Ollama model
                        ollama_config = OllamaConfig(model=selected_model, timeout=timeout_seconds)
                        ollama_analyzer = OllamaAnalyzer(ollama_config)
                        
                        # Check if model is available
                        available_models = ollama_analyzer._get_available_models()
                        if selected_model not in available_models:
                            flash(f"Model '{selected_model}' not available. Available models: {', '.join(available_models)}")
                            return redirect(url_for("ai_analysis"))
                        
                        ai_analysis = ollama_analyzer.analyze_dxf_manufacturing(file_path, analysis_data)
                        if ai_analysis:
                            # Update model_used attribute
                            ai_analysis.model_used = f"Ollama ({selected_model})"
                            
                except Exception as e:
                    print(f"Selected model {selected_model} failed: {e}")
                    flash(f"Selected model '{selected_model}' failed: {str(e)}")
                    return redirect(url_for("ai_analysis"))
            
            if ai_analysis:
                # Save AI analysis
                ai_report_path = os.path.join(output_dir, "ai_analysis.json")
                with open(ai_report_path, "w", encoding="utf-8") as f:
                    json.dump(ai_analysis.dict(), f, indent=2)
                
                flash(f"AI analysis completed successfully using {ai_analysis.model_used}!")
                return redirect(url_for("ai_results", analysis_id=ai_analysis_id))
            else:
                flash("AI analysis failed. No AI model was able to process the request. Please check your AI configuration and try a different model.")
                return redirect(url_for("ai_analysis"))
                
        else:
            flash("Basic analysis failed. Cannot run AI analysis.")
            return redirect(url_for("ai_analysis"))
            
    except Exception as exc:
        flash(f"Error during analysis: {exc}")
        return redirect(url_for("ai_analysis"))


@app.route("/ai-results/<analysis_id>")
def ai_results(analysis_id: str):
    """Display AI analysis results."""
    if not _validate_id(analysis_id):
        flash("AI analysis results not found.")
        return redirect(url_for("ai_analysis"))
    
    output_dir = _safe_join(OUTPUT_FOLDER, analysis_id)
    if not os.path.isdir(output_dir):
        flash("AI analysis results not found.")
        return redirect(url_for("ai_analysis"))

    # Load basic analysis report
    report_path = os.path.join(output_dir, "report.json")
    if not os.path.exists(report_path):
        flash("Analysis report not found.")
        return redirect(url_for("ai_analysis"))

    with open(report_path, "r", encoding="utf-8") as f:
        basic_report = json.load(f)

    # Load AI analysis
    ai_report_path = os.path.join(output_dir, "ai_analysis.json")
    ai_analysis = None
    if os.path.exists(ai_report_path):
        with open(ai_report_path, "r", encoding="utf-8") as f:
            ai_data = json.load(f)
            ai_analysis = ManufacturingAnalysis(**ai_data)

    return render_template(
        "ai_results.html",
        analysis_id=analysis_id,
        basic_report=basic_report,
        ai_analysis=ai_analysis,
    )


@app.route("/analyze-dxf", methods=["POST"])
def analyze_dxf():
    """Handle DXF analysis."""
    form_values = {
        "kerf": parse_float(request.form.get("kerf"), DEFAULT_FORM["kerf"]),
    }

    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please choose a DXF file to upload.")
        return redirect(url_for("dxf_analysis"))

    if not allowed_file(uploaded_file.filename):
        flash("Please upload a DXF file.")
        return redirect(url_for("dxf_analysis"))

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)

    base_name, file_ext = os.path.splitext(filename)
    file_ext = file_ext.lower().lstrip(".")
    
    if file_ext != "dxf":
        flash("Please upload a valid DXF file.")
        return redirect(url_for("dxf_analysis"))

    analysis_id = f"{base_name}_analysis_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
    os.makedirs(output_dir, exist_ok=True)

    if Args is not None:
        analysis_args = Args(
            material="DXF Validation",  # Default for validation
            thickness=25.0,  # Default thickness
            kerf=form_values["kerf"],
            rate_per_m=825.0,  # Default rate
            out=output_dir,
        )
    else:
        flash("DXF analyzer not available. Please check installation.")
        return redirect(url_for("dxf_analysis"))

    try:
        if analyze is not None:
            analyze(file_path, analysis_args)
            flash("DXF analysis completed successfully!")
            return redirect(url_for("results", analysis_id=analysis_id))
        else:
            flash("DXF analyzer not available. Please check installation.")
            return redirect(url_for("dxf_analysis"))
    except Exception as exc:
        flash(f"Error during analysis: {exc}")
        return redirect(url_for("dxf_analysis"))


@app.route("/generate-gcode", methods=["POST"])
def generate_gcode():
    """Handle G-code generation."""
    gcode_values = {
        "feed": parse_float(request.form.get("feed"), DEFAULT_GCODE_PARAMS["feed"]),
        "m_on": request.form.get("m_on") or DEFAULT_GCODE_PARAMS["m_on"],
        "m_off": request.form.get("m_off") or DEFAULT_GCODE_PARAMS["m_off"],
        "pierce_ms": parse_int(request.form.get("pierce_ms"), DEFAULT_GCODE_PARAMS["pierce_ms"]),
    }

    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please choose a DXF file to upload.")
        return redirect(url_for("gcode_generation"))

    if not allowed_file(uploaded_file.filename):
        flash("Please upload a DXF file.")
        return redirect(url_for("gcode_generation"))

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)

    base_name, file_ext = os.path.splitext(filename)
    file_ext = file_ext.lower().lstrip(".")
    
    if file_ext != "dxf":
        flash("Please upload a valid DXF file.")
        return redirect(url_for("gcode_generation"))

    gcode_id = f"gcode_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(OUTPUT_FOLDER, gcode_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from ..io.dxf_io import load_dxf_lines_with_layers
        from ..analysis.geometry_cleaner import merge_and_polygonize
        from ..analysis.topology import containment_depth
        from ..analysis.classification import classify_by_depth_and_layers
        from ..manufacturing.toolpath import plan_order
        from ..manufacturing.gcode_generator import write_gcode
        
        lines, line_layers = load_dxf_lines_with_layers(file_path)
        _, polys = merge_and_polygonize(lines)
        depths = containment_depth(polys)
        classes = classify_by_depth_and_layers(polys, depths, lines, line_layers)
        order = plan_order(polys, classes)
        
        # Generate G-code
        gcode_path = os.path.join(output_dir, "program.nc")
        write_gcode(gcode_path, polys, order, 
                   feed=gcode_values["feed"], 
                   m_on=gcode_values["m_on"], 
                   m_off=gcode_values["m_off"], 
                   pierce_ms=gcode_values["pierce_ms"])
        
        # Save generation report
        report = {
            "source_dxf": file_path,
            "output_gcode": gcode_path,
            "parameters": gcode_values,
            "polygon_count": len(polys),
            "gcode_id": gcode_id,
        }
        
        report_path = os.path.join(output_dir, "gcode_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        flash(f"G-code generated successfully! {len(polys)} polygons processed.")
        return redirect(url_for("gcode_results", gcode_id=gcode_id))
        
    except Exception as exc:
        flash(f"Error during G-code generation: {exc}")
        return redirect(url_for("gcode_generation"))


@app.route("/create-nest", methods=["POST"])
def create_nest():
    """Handle nesting creation."""
    nesting_values = {
        "sheet_width": parse_float(request.form.get("sheet_width"), DEFAULT_NESTING_PARAMS["sheet_width"]),
        "sheet_height": parse_float(request.form.get("sheet_height"), DEFAULT_NESTING_PARAMS["sheet_height"]),
        "spacing": parse_float(request.form.get("spacing"), DEFAULT_NESTING_PARAMS["spacing"]),
    }

    uploaded_files = request.files.getlist("files")
    if not uploaded_files or all(f.filename == "" for f in uploaded_files):
        flash("Please select DXF files to nest.")
        return redirect(url_for("nesting"))

    # Filter valid DXF files
    dxf_files = []
    for file in uploaded_files:
        if file.filename and file.filename.lower().endswith('.dxf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            dxf_files.append(file_path)

    if not dxf_files:
        flash("No valid DXF files found.")
        return redirect(url_for("nesting"))

    nesting_id = f"nest_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(OUTPUT_FOLDER, nesting_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from ..manufacturing.nesting import NestingEngine
        
        engine = NestingEngine(
            sheet_width=nesting_values["sheet_width"],
            sheet_height=nesting_values["sheet_height"]
        )
        # Inspect part dimensions first
        parts = []
        for path in dxf_files:
            polys = engine.load_dxf_polygons(path)
            w, h = engine.calculate_bounds(polys)
            parts.append({
                "file": os.path.basename(path),
                "path": path,
                "width": w,
                "height": h,
            })

        # Stash inspection metadata
        inspect_meta = {
            "sheet": nesting_values,
            "parts": parts,
        }
        with open(os.path.join(output_dir, "inspection.json"), "w", encoding="utf-8") as f:
            json.dump(inspect_meta, f, indent=2)

        return render_template(
            "nesting.html",
            nesting_defaults=DEFAULT_NESTING_PARAMS,
            inspection=inspect_meta,
            nesting_id=nesting_id,
        )
        
    except Exception as exc:
        flash(f"Error during nesting: {exc}")
        return redirect(url_for("nesting"))


@app.route("/confirm-nest", methods=["POST"])
def confirm_nest():
    nesting_id = request.form.get("nesting_id")
    if not nesting_id:
        flash("Missing nesting session.")
        return redirect(url_for("nesting"))
    if not _validate_id(nesting_id):
        flash("Invalid nesting id.")
        return redirect(url_for("nesting"))
    output_dir = _safe_join(OUTPUT_FOLDER, nesting_id)
    inspect_path = os.path.join(output_dir, "inspection.json")
    if not os.path.exists(inspect_path):
        flash("Nesting inspection not found.")
        return redirect(url_for("nesting"))
    try:
        with open(inspect_path, "r", encoding="utf-8") as f:
            inspect_meta = json.load(f)

        # Expand parts based on quantities
        expanded_paths = []
        for idx, part in enumerate(inspect_meta.get("parts", [])):
            qty = parse_int(request.form.get(f"qty_{idx}"), 1)
            expanded_paths.extend([part["path"]] * max(1, qty))

        from ..manufacturing.nesting import NestingEngine
        engine = NestingEngine(
            sheet_width=inspect_meta["sheet"]["sheet_width"],
            sheet_height=inspect_meta["sheet"]["sheet_height"],
        )
        nesting_result = engine.simple_nesting(expanded_paths, inspect_meta["sheet"]["spacing"])

        # Generate outputs
        nested_dxf_path = os.path.join(output_dir, "nested_layout.dxf")
        engine.generate_nested_dxf(nesting_result, nested_dxf_path)
        report_path = os.path.join(output_dir, "nesting_report.json")
        engine.generate_nesting_report(nesting_result, report_path)

        return redirect(url_for("nesting_results", nesting_id=nesting_id))
    except Exception as exc:
        flash(f"Failed to create nest: {exc}")
        return redirect(url_for("nesting"))


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Analyze an uploaded image for DXF conversion suitability."""
    try:
        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"success": False, "error": "No file uploaded"})

        # Check if it's an image file
        filename = uploaded_file.filename.lower()
        if not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
            return jsonify({"success": False, "error": "Please upload an image file (JPG, PNG, BMP, TIFF)"})

        # Save uploaded file temporarily
        filename = secure_filename(uploaded_file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
        uploaded_file.save(temp_path)

        try:
            # Get analysis parameters from form
            min_score = float(request.form.get("min_score", 75.0))
            px_to_unit = float(request.form.get("px_to_unit", 1.0))
            
            # Create custom config if parameters provided
            config = AnalyzerConfig(px_to_unit=px_to_unit) if px_to_unit != 1.0 else None
            
            # Analyze the image
            should_proceed, report = quick_analyze(temp_path, min_score=min_score)
            
            # Save report to output directory
            analysis_id = f"img_analysis_{uuid.uuid4().hex[:8]}"
            output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
            os.makedirs(output_dir, exist_ok=True)
            
            report_path = os.path.join(output_dir, "image_analysis_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Prepare response
            response = {
                "success": True,
                "analysis_id": analysis_id,
                "should_proceed": should_proceed,
                "score": report["score"],
                "suggestions": report["suggestions"],
                "flags": report["flags"],
                "basic_stats": report["basic_stats"],
                "orientation": report["orientation"],
                "texture_metrics": report["texture_metrics"],
                "topology_preview": report["topology_preview"],
                "manufacturability": report["manufacturability"],
                "gate_decision": report.get("gate_decision", {}),
                "report_path": report_path
            }
            
            return jsonify(response)
            
        except Exception as analysis_error:
            return jsonify({
                "success": False, 
                "error": f"Analysis failed: {str(analysis_error)}",
                "details": str(analysis_error)
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": f"Upload failed: {str(e)}",
            "details": str(e)
        })


@app.route("/convert-image", methods=["POST"])
def convert_image():
    abort(404)


def _mcp_call(tool: str, args: dict, url: str = "http://127.0.0.1:8000/mcp") -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data.get("result", {})


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file uploads and run analysis/image conversion."""
    # Log user action
    if log_user_action is not None:
        log_user_action("file_upload", details={"method": "POST"})
    
    # Validate material parameters
    material_params = {
        "material": request.form.get("material", "steel"),
        "thickness": parse_float(request.form.get("thickness"), DEFAULT_FORM.get("thickness", 6.0)),
        "kerf": parse_float(request.form.get("kerf"), DEFAULT_FORM.get("kerf", 1.1)),
        "rate_per_m": parse_float(request.form.get("rate_per_m"), DEFAULT_FORM.get("rate_per_m", 50.0)),
    }
    
    validation_result = validate_material_params(material_params) if validate_material_params is not None else None
    if validation_result is not None and not validation_result.is_valid:
        if log_security_event is not None:
            log_security_event("invalid_parameters", {
                "errors": validation_result.errors,
                "params": material_params
            })
        flash(f"Invalid parameters: {', '.join(validation_result.errors)}")
        return render_index(material_params, {})
    
    form_values = material_params

    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please choose a DXF file to upload.")
        return render_index(form_values)

    # Validate uploaded file
    filename = uploaded_file.filename
    validation_result = validate_uploaded_file(uploaded_file.filename, filename) if validate_uploaded_file is not None else None
    
    if validation_result is not None and not validation_result.is_valid:
        if log_security_event is not None:
            log_security_event("invalid_file_upload", {
                "filename": filename,
                "errors": validation_result.errors
            })
        flash(f"Invalid file: {', '.join(validation_result.errors)}")
        return render_index(form_values)
    
    if validation_result is not None and validation_result.warnings:
        for warning in validation_result.warnings:
            flash(f"Warning: {warning}", "warning")

    # Sanitize filename and save file
    filename = secure_filename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)
    
    # Log successful file upload
    if log_user_action is not None:
        log_user_action("file_upload_success", details={
            "filename": filename,
        "file_size": os.path.getsize(file_path),
        "file_type": filename.split('.')[-1].lower()
    })

    base_name, file_ext = os.path.splitext(filename)
    file_ext = file_ext.lower().lstrip(".")

    analysis_id = f"{base_name}_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
    os.makedirs(output_dir, exist_ok=True)

    dxf_path = file_path
    if file_ext in {"jpg", "jpeg", "png", "bmp", "tiff"}:
        # Analyze image for DXF conversion suitability
        try:
            min_score = 75.0  # Default threshold
            should_proceed, report = quick_analyze(file_path, min_score=min_score)
            
            # Save analysis report
            report_path = os.path.join(output_dir, "image_analysis_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            if should_proceed:
                flash(f"✅ Image analysis complete! Score: {report['score']}/100 - Ready for DXF conversion.", "success")
            else:
                flash(f"⚠️ Image analysis complete! Score: {report['score']}/100 - Needs preprocessing. Check suggestions.", "warning")
            
            # Redirect to image analysis results
            return redirect(url_for("image_analysis_results", analysis_id=analysis_id))
            
        except Exception as e:
            flash(f"Error analyzing image: {e}", "error")
            return redirect(url_for("index"))

    if Args is not None:
        analysis_args = Args(
            material="DXF Validation",  # Default for validation
            thickness=25.0,  # Default thickness
            kerf=form_values["kerf"],
            rate_per_m=825.0,  # Default rate
            out=output_dir,
        )
    else:
        flash("DXF analyzer not available. Please check installation.")
        return render_index(form_values, image_values)

    try:
        analyze(dxf_path, analysis_args)
    except Exception as exc:  # pragma: no cover - surface error to UI
        flash(f"Error during analysis: {exc}")
        return render_index(form_values, image_values)

    return redirect(url_for("results", analysis_id=analysis_id))


def _resolve_analysis_file(analysis_id: str, file_type: str) -> Tuple[str, str]:
    if file_type not in ANALYSIS_FILES:
        raise KeyError("Invalid file type")
    if not _validate_id(analysis_id):
        raise KeyError("Invalid analysis id")
    output_dir = _safe_join(OUTPUT_FOLDER, analysis_id)
    file_path = os.path.join(output_dir, ANALYSIS_FILES[file_type])
    return ANALYSIS_FILES[file_type], file_path


@app.route("/results/<analysis_id>")
def results(analysis_id: str):
    """Render the analysis results page."""
    if not _validate_id(analysis_id):
        flash("Analysis results not found. Try running the analysis again.")
        return redirect(url_for("index"))
    output_dir = _safe_join(OUTPUT_FOLDER, analysis_id)
    if not os.path.isdir(output_dir):
        flash("Analysis results not found. Try running the analysis again.")
        return redirect(url_for("index"))

    try:
        _, report_path = _resolve_analysis_file(analysis_id, "report")
    except KeyError:
        flash("Analysis report not found.")
        return redirect(url_for("index"))

    if not os.path.exists(report_path):
        flash("Analysis report not found.")
        return redirect(url_for("index"))

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    lengths = []
    try:
        _, lengths_path = _resolve_analysis_file(analysis_id, "lengths")
        if os.path.exists(lengths_path):
            with open(lengths_path, "r", encoding="utf-8") as handle:
                rows = handle.readlines()
            for row in rows[1:]:  # Skip header
                parts = row.strip().split(",")
                if len(parts) >= 4:
                    lengths.append(
                        {
                            "id": parts[0],
                            "class": parts[1],
                            "perimeter": f"{float(parts[2]):.1f}",
                            "area": f"{float(parts[3]):.1f}",
                        }
                    )
    except Exception:
        lengths = []

    available_assets = {}
    for key in ANALYSIS_FILES:
        _, path = _resolve_analysis_file(analysis_id, key)
        available_assets[key] = os.path.exists(path)

    image_metadata = None
    if available_assets.get("image_metadata"):
        _, meta_path = _resolve_analysis_file(analysis_id, "image_metadata")
        with open(meta_path, "r", encoding="utf-8") as handle:
            image_metadata = json.load(handle)

    download_order = [
        ("report", "Report (JSON)"),
        ("lengths", "Lengths (CSV)"),
        ("preview", "Preview (PNG)"),
        ("gcode", "G-code (NC)"),
        ("image_metadata", "Image Metadata (JSON)"),
        ("image_gray", "Grayscale Input (PNG)"),
        ("image_threshold", "Threshold Mask (PNG)"),
        ("image_edges", "Edge Map (PNG)"),
    ]

    download_links = [
        {
            "label": label,
            "href": url_for("download_file", analysis_id=analysis_id, file_type=key),
        }
        for key, label in download_order
        if available_assets.get(key)
    ]

    debug_images = [
        {
            "label": label,
            "url": url_for("analysis_asset", analysis_id=analysis_id, file_type=key),
        }
        for key, label in [
            ("preview", "Toolpath Preview"),
            ("preview_full", "Preview (Detailed)"),
            ("image_gray", "Grayscale"),
            ("image_threshold", "Threshold"),
            ("image_edges", "Edges"),
        ]
        if available_assets.get(key)
    ]

    return render_template(
        "results.html",
        analysis_id=analysis_id,
        report=report,
        lengths=lengths,
        download_links=download_links,
        debug_images=debug_images,
        available_assets=available_assets,
    )


@app.route("/export-quote/<analysis_id>/pdf")
def export_quote_pdf(analysis_id: str):
    output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
    report_path = os.path.join(output_dir, "report.json")
    csv_path = os.path.join(output_dir, "lengths.csv")
    if not os.path.exists(report_path):
        flash("Analysis report not found.")
        return redirect(url_for("results", analysis_id=analysis_id))
    out_path = os.path.join(output_dir, "quotation.pdf")
    try:
        make_pdf(report_path, csv_path if os.path.exists(csv_path) else None, out_path)
        return send_file(out_path, as_attachment=True, download_name="quotation.pdf")
    except Exception as exc:
        flash(f"Failed to export PDF: {exc}")
        return redirect(url_for("results", analysis_id=analysis_id))


@app.route("/export-quote/<analysis_id>/xlsx")
def export_quote_xlsx(analysis_id: str):
    output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
    report_path = os.path.join(output_dir, "report.json")
    csv_path = os.path.join(output_dir, "lengths.csv")
    if not os.path.exists(report_path):
        flash("Analysis report not found.")
        return redirect(url_for("results", analysis_id=analysis_id))
    out_path = os.path.join(output_dir, "quotation.xlsx")
    try:
        make_xlsx(report_path, csv_path if os.path.exists(csv_path) else None, out_path)
        return send_file(out_path, as_attachment=True, download_name="quotation.xlsx")
    except Exception as exc:
        flash(f"Failed to export XLSX: {exc}")
        return redirect(url_for("results", analysis_id=analysis_id))


@app.route("/route-converted-dxf", methods=["POST"])
def route_converted_dxf():
    """Route the converted DXF to other flows (analysis, nesting, gcode)."""
    analysis_id = request.form.get("analysis_id")
    target = (request.form.get("route_target") or "").strip().lower()
    if not analysis_id:
        flash("Missing analysis context.")
        return redirect(url_for("index"))

    # Locate converted DXF from image metadata
    try:
        _, meta_path = _resolve_analysis_file(analysis_id, "image_metadata")
    except KeyError:
        flash("No converted DXF associated with this analysis.")
        return redirect(url_for("results", analysis_id=analysis_id))

    if not os.path.exists(meta_path):
        flash("Converted DXF metadata not found.")
        return redirect(url_for("results", analysis_id=analysis_id))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    dxf_path = meta.get("output_dxf")
    if not dxf_path or not os.path.exists(dxf_path):
        flash("Converted DXF file is missing.")
        return redirect(url_for("results", analysis_id=analysis_id))

    # Redirect to the selected target with DXF preselected if supported
    if target == "analysis":
        return redirect(url_for("dxf_analysis") + f"?dxf_path={dxf_path}")
    if target == "nesting":
        return redirect(url_for("nesting") + f"?dxf_path={dxf_path}")
    if target == "gcode":
        return redirect(url_for("gcode_generation") + f"?dxf_path={dxf_path}")

    flash("Unknown target selected.")
    return redirect(url_for("results", analysis_id=analysis_id))


@app.route("/gcode-results/<gcode_id>")
def gcode_results(gcode_id: str):
    """Display G-code generation results."""
    if not _validate_id(gcode_id):
        flash("G-code results not found.")
        return redirect(url_for("gcode_generation"))
    output_dir = _safe_join(OUTPUT_FOLDER, gcode_id)
    if not os.path.isdir(output_dir):
        flash("G-code results not found.")
        return redirect(url_for("gcode_generation"))

    report_path = os.path.join(output_dir, "gcode_report.json")
    gcode_path = os.path.join(output_dir, "program.nc")
    
    if not os.path.exists(report_path):
        flash("G-code report not found.")
        return redirect(url_for("gcode_generation"))

    with open(report_path, "r", encoding="utf-8") as f:
        gcode_results = json.load(f)

    return render_template(
        "gcode_generation.html",
        gcode_defaults=DEFAULT_GCODE_PARAMS,
        gcode_results=gcode_results,
        gcode_id=gcode_id,
    )


# ------------------ Flooring Visualizer ------------------

def _ft_to_mm(ft: float) -> float:
    return float(ft) * 304.8


@app.route("/flooring", methods=["GET", "POST"])
def flooring():
    if request.method == "GET":
        return render_template("flooring.html", result=None)

    # POST: compute
    try:
        w_ft = parse_float(request.form.get("room_w_ft"), 10.0)
        l_ft = parse_float(request.form.get("room_l_ft"), 12.0)
        tile_ft = parse_int(request.form.get("tile_size_ft"), 2)
        border_in = parse_float(request.form.get("border_in"), 0.0)
        rate_per_m = parse_float(request.form.get("rate_per_m"), DEFAULT_FORM["rate_per_m"])
        material = (request.form.get("material") or DEFAULT_FORM["material"]).strip()
        thickness = parse_float(request.form.get("thickness"), DEFAULT_FORM["thickness"])

        W = _ft_to_mm(w_ft)
        L = _ft_to_mm(l_ft)
        tile_size_mm = _ft_to_mm(tile_ft)

        cols = int(W // tile_size_mm)
        rows = int(L // tile_size_mm)
        total_tiles = cols * rows

        waste_w = W - cols * tile_size_mm
        waste_l = L - rows * tile_size_mm

        border_len_m = 0.0
        if border_in > 0.0:
            border_len_m = 2.0 * (W + L) / 1000.0

        est_border_cost = border_len_m * rate_per_m

        result = {
            "room": {"W_mm": round(W,1), "L_mm": round(L,1)},
            "tile": {"size_mm": round(tile_size_mm,1)},
            "grid": {"cols": cols, "rows": rows, "total_tiles": total_tiles},
            "waste": {"width_mm": round(waste_w,1), "length_mm": round(waste_l,1)},
            "border": {"length_m": round(border_len_m,2), "cost_inr": round(est_border_cost,0)},
            "pricing": {"rate_per_m": rate_per_m},
            "material": {"name": material, "thickness_mm": thickness},
        }
        return render_template("flooring.html", result=result)
    except Exception as exc:
        flash(f"Flooring calculation failed: {exc}")
        return render_template("flooring.html", result=None)


@app.route("/nesting-results/<nesting_id>")
def nesting_results(nesting_id: str):
    """Display nesting results."""
    if not _validate_id(nesting_id):
        flash("Nesting results not found.")
        return redirect(url_for("nesting"))
    output_dir = _safe_join(OUTPUT_FOLDER, nesting_id)
    if not os.path.isdir(output_dir):
        flash("Nesting results not found.")
        return redirect(url_for("nesting"))

    report_path = os.path.join(output_dir, "nesting_report.json")
    nested_dxf_path = os.path.join(output_dir, "nested_layout.dxf")
    
    if not os.path.exists(report_path):
        flash("Nesting report not found.")
        return redirect(url_for("nesting"))

    with open(report_path, "r", encoding="utf-8") as f:
        nesting_results = json.load(f)

    return render_template(
        "nesting.html",
        nesting_defaults=DEFAULT_NESTING_PARAMS,
        nesting_results=nesting_results,
        nesting_id=nesting_id,
        nested_dxf_path=nested_dxf_path,
    )


@app.route("/conversion-results/<conversion_id>")
def conversion_results(conversion_id: str):
    abort(404)


@app.route("/mcp-reprocess-conversion/<conversion_id>", methods=["POST"])
def mcp_reprocess_conversion(conversion_id: str):
    """Image conversion is no longer available."""
    flash("Image conversion feature has been removed.", "error")
    return redirect(url_for("main.index"))


@app.route("/download-conversion-dxf/<conversion_id>")
def download_converted_dxf(conversion_id: str):
    """Download the converted DXF from image conversion."""
    if not _validate_id(conversion_id):
        flash("Invalid conversion id.")
        return redirect(url_for("image_to_dxf"))
    output_dir = _safe_join(OUTPUT_FOLDER, conversion_id)
    metadata_path = os.path.join(output_dir, "image_metadata.json")
    
    if not os.path.exists(metadata_path):
        flash("Conversion metadata not found.")
        return redirect(url_for("image_to_dxf"))
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    dxf_path = metadata.get("output_dxf")
    if not dxf_path or not os.path.exists(dxf_path):
        flash("Converted DXF file is missing.")
        return redirect(url_for("image_to_dxf"))
        
    name = os.path.basename(dxf_path)
    return send_file(dxf_path, as_attachment=True, download_name=name)


@app.route("/download-analysis-dxf/<analysis_id>")
def download_converted_dxf_analysis(analysis_id: str):
    """Download the converted DXF produced from an image upload."""
    if not _validate_id(analysis_id):
        flash("Invalid analysis id.")
        return redirect(url_for("index"))
    try:
        _, meta_path = _resolve_analysis_file(analysis_id, "image_metadata")
    except KeyError:
        flash("No converted DXF found for this analysis.")
        return redirect(url_for("results", analysis_id=analysis_id))
    if not os.path.exists(meta_path):
        flash("No converted DXF available.")
        return redirect(url_for("results", analysis_id=analysis_id))
    with open(meta_path, "r", encoding="utf-8") as h:
        meta = json.load(h)
    dxf_path = meta.get("output_dxf")
    if not dxf_path or not os.path.exists(dxf_path):
        flash("Converted DXF file is missing.")
        return redirect(url_for("results", analysis_id=analysis_id))
    name = os.path.basename(dxf_path)
    return send_file(dxf_path, as_attachment=True, download_name=name)


@app.route("/reprocess/<analysis_id>", methods=["POST"])
def reprocess(analysis_id: str):
    """Image conversion is no longer available."""
    flash("Image conversion feature has been removed.", "error")
    return redirect(url_for("main.index"))






    # Re-run conversion using the unified OpenCV-based converter
    try:
        from wjp_analyser.image_processing.converters.unified_converter import (
            UnifiedImageToDXFConverter,
            ConversionParams,
        )

        binary_threshold = 180
        min_area = int(image_values["min_contour_area"]) if image_values else 1000
        dxf_size = 1200

        params = ConversionParams(
            binary_threshold=binary_threshold,
            min_area=min_area,
            dxf_size=dxf_size,
        )
        converter = UnifiedImageToDXFConverter(params)
        preview_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(dxf_path))[0]}_opencv_preview.png",
        )
        result = converter.convert_image_to_dxf(
            input_image=source_image,
            output_dxf=dxf_path,
            preview_output=preview_path,
        )
        if result.get("polygons", 0) <= 0:
            flash("No contours detected with the chosen parameters.")
            return redirect(url_for("results", analysis_id=analysis_id))
    except Exception as exc:
        flash(f"Image processing failed: {exc}")
        return redirect(url_for("results", analysis_id=analysis_id))

    # Re-run analysis
    # Use prior material/kerf from report.json if available, else defaults
    report_path = os.path.join(output_dir, "report.json")
    material = DEFAULT_FORM["material"]
    thickness = DEFAULT_FORM["thickness"]
    kerf = DEFAULT_FORM["kerf"]
    rate = DEFAULT_FORM["rate_per_m"]
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as h:
                rep = json.load(h)
            material = rep.get("material", {}).get("name", material)
            thickness = rep.get("material", {}).get("thickness_mm", thickness)
            kerf = rep.get("kerf_mm", kerf)
        except Exception:
            pass
    if Args is not None:
        analysis_args = Args(
            material=material,
            thickness=thickness,
            kerf=kerf,
            rate_per_m=rate,
            out=output_dir,
        )
        try:
            if analyze is not None:
                analyze(dxf_path, analysis_args)
            else:
                flash("DXF analyzer not available. Please check installation.")
        except Exception as exc:
            flash(f"Analysis failed after reprocess: {exc}")
    else:
        flash("DXF analyzer not available. Please check installation.")
    
    return redirect(url_for("results", analysis_id=analysis_id))


@app.route("/download-gcode/<gcode_id>")
def download_gcode(gcode_id: str):
    """Download the generated G-code file."""
    if not _validate_id(gcode_id):
        flash("Invalid G-code id.")
        return redirect(url_for("gcode_generation"))
    output_dir = _safe_join(OUTPUT_FOLDER, gcode_id)
    gcode_path = os.path.join(output_dir, "program.nc")
    
    if not os.path.exists(gcode_path):
        flash("G-code file not found.")
        return redirect(url_for("gcode_generation"))
        
    return send_file(gcode_path, as_attachment=True, download_name="program.nc")


@app.route("/download-gcode-report/<gcode_id>")
def download_gcode_report(gcode_id: str):
    """Download the G-code generation report."""
    if not _validate_id(gcode_id):
        flash("Invalid G-code id.")
        return redirect(url_for("gcode_generation"))
    output_dir = _safe_join(OUTPUT_FOLDER, gcode_id)
    report_path = os.path.join(output_dir, "gcode_report.json")
    
    if not os.path.exists(report_path):
        flash("G-code report not found.")
        return redirect(url_for("gcode_generation"))
        
    return send_file(report_path, as_attachment=True, download_name="gcode_report.json")


@app.route("/download-nested-dxf/<nesting_id>")
def download_nested_dxf(nesting_id: str):
    """Download the nested DXF file."""
    if not _validate_id(nesting_id):
        flash("Invalid nesting id.")
        return redirect(url_for("nesting"))
    output_dir = _safe_join(OUTPUT_FOLDER, nesting_id)
    nested_dxf_path = os.path.join(output_dir, "nested_layout.dxf")
    
    if not os.path.exists(nested_dxf_path):
        flash("Nested DXF file not found.")
        return redirect(url_for("nesting"))
        
    return send_file(nested_dxf_path, as_attachment=True, download_name="nested_layout.dxf")


@app.route("/download-nesting-report/<nesting_id>")
def download_nesting_report(nesting_id: str):
    """Download the nesting report."""
    if not _validate_id(nesting_id):
        flash("Invalid nesting id.")
        return redirect(url_for("nesting"))
    output_dir = _safe_join(OUTPUT_FOLDER, nesting_id)
    report_path = os.path.join(output_dir, "nesting_report.json")
    
    if not os.path.exists(report_path):
        flash("Nesting report not found.")
        return redirect(url_for("nesting"))
        
    return send_file(report_path, as_attachment=True, download_name="nesting_report.json")


@app.route("/download-conversion/<conversion_id>/<file_type>")
def download_conversion_file(conversion_id: str, file_type: str):
    """Download conversion files as attachments."""
    if not _validate_id(conversion_id):
        flash("Invalid conversion id.")
        return redirect(url_for("image_to_dxf"))
    output_dir = _safe_join(OUTPUT_FOLDER, conversion_id)
    file_path = os.path.join(output_dir, f"{file_type}.png")
    
    if not os.path.exists(file_path):
        flash("Requested file is not available.")
        return redirect(url_for("image_to_dxf"))
        
    return send_file(file_path, as_attachment=True, download_name=f"{file_type}.png")


@app.route("/download/<analysis_id>/<file_type>")
def download_file(analysis_id: str, file_type: str):
    """Download analysis artefacts as attachments."""
    try:
        name, file_path = _resolve_analysis_file(analysis_id, file_type)
    except KeyError:
        flash("Invalid file requested.")
        return redirect(url_for("index"))

    if not os.path.exists(file_path):
        flash("Requested file is not available.")
        return redirect(url_for("index"))

    return send_file(file_path, as_attachment=True, download_name=name)


@app.route("/conversion-assets/<conversion_id>/<file_type>")
def conversion_asset(conversion_id: str, file_type: str):
    """Serve conversion debug images inline for previews."""
    if not _validate_id(conversion_id):
        flash("Invalid conversion id.")
        return redirect(url_for("image_to_dxf"))
    output_dir = _safe_join(OUTPUT_FOLDER, conversion_id)
    file_path = os.path.join(output_dir, f"{file_type}.png")
    
    if not os.path.exists(file_path):
        flash("Asset not available.")
        return redirect(url_for("image_to_dxf"))
        
    return send_file(file_path, as_attachment=False)


@app.route("/assets/<analysis_id>/<file_type>")
def analysis_asset(analysis_id: str, file_type: str):
    """Serve analysis artefacts inline for previews."""
    try:
        _, file_path = _resolve_analysis_file(analysis_id, file_type)
    except KeyError:
        flash("Invalid asset requested.")
        return redirect(url_for("index"))

    if not os.path.exists(file_path):
        flash("Asset not available.")
        return redirect(url_for("index"))

    return send_file(file_path, as_attachment=False)


@app.route("/sample")
def generate_sample():
    """Generate and serve a sample DXF file for testing."""
    try:
        from scripts.make_sample_dxf import make_sample

        sample_path = os.path.join(UPLOAD_FOLDER, "sample_medallion.dxf")
        make_sample(sample_path)
        return send_file(sample_path, as_attachment=True, download_name="sample_medallion.dxf")
    except Exception as exc:  # pragma: no cover
        flash(f"Error generating sample: {exc}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    print("Starting Waterjet DXF Analyzer Web Interface...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
