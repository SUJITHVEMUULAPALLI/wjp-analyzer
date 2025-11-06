"""
Web Application Routes
=====================

Flask routes for the WJP ANALYSER web interface.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session, send_file, abort, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from typing import Dict, Any, Optional

from ..auth.enhanced_auth import auth_manager, Permission, require_auth
from ..auth.security_middleware import security_check
from ..database.models import get_db_session
from ..database.models import Project, Analysis, Conversion, Nesting
from ..utils.error_handler import WJPAnalyserError, ErrorContext, safe_execute

logger = logging.getLogger(__name__)

# Create main blueprint
main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Main page."""
    user_id = session.get('user_id')
    if not user_id:
        return render_template('index.html')
    
    # Get user projects
    db_session = get_db_session()
    try:
        projects = db_session.query(Project).filter(
            Project.user_id == user_id,
            Project.status == 'active'
        ).order_by(Project.created_at.desc()).limit(10).all()
        
        return render_template('dashboard.html', projects=projects)
    finally:
        db_session.close()


@main_bp.route('/projects')
@require_auth(Permission.VIEW_PROJECT)
def projects():
    """Projects page."""
    user_id = session.get('user_id')
    try:
        db_session = get_db_session()
    except Exception as e:
        logger.warning(f"Database unavailable for /projects: {e}")
        flash('Database unavailable; showing empty project list', 'error')
        return render_template('projects.html', projects=[])
    
    try:
        projects = db_session.query(Project).filter(
            Project.user_id == user_id,
            Project.status == 'active'
        ).order_by(Project.created_at.desc()).all()
        
        return render_template('projects.html', projects=projects)
    finally:
        db_session.close()


@main_bp.route('/projects/new', methods=['GET', 'POST'])
@require_auth(Permission.CREATE_PROJECT)
def new_project():
    """Create new project."""
    if request.method == 'GET':
        return render_template('new_project.html')
    
    # Security check
    client_ip = request.remote_addr
    security_result = security_check(client_ip, '/projects/new', 'POST', dict(request.headers))
    
    if not security_result['allowed']:
        flash(f"Security check failed: {security_result['reason']}", 'error')
        return render_template('new_project.html')
    
    name = request.form.get('name')
    description = request.form.get('description')
    
    if not name:
        flash('Project name is required', 'error')
        return render_template('new_project.html')
    
    user_id = session.get('user_id')
    db_session = get_db_session()
    
    try:
        from ..database import get_database_manager
        db_manager = get_database_manager()
        
        project = db_manager.create_project(
            session=db_session,
            user_id=user_id,
            name=name,
            description=description
        )
        
        flash('Project created successfully!', 'success')
        return redirect(url_for('main.project_detail', project_id=project.id))
        
    except Exception as e:
        flash(f"Failed to create project: {e}", 'error')
        return render_template('new_project.html')
    finally:
        db_session.close()


@main_bp.route('/projects/<project_id>')
@require_auth(Permission.VIEW_PROJECT)
def project_detail(project_id: str):
    """Project detail page."""
    user_id = session.get('user_id')
    db_session = get_db_session()
    
    try:
        project = db_session.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user_id
        ).first()
        
        if not project:
            abort(404)
        
        # Get project analyses, conversions, and nestings
        analyses = db_session.query(Analysis).filter(
            Analysis.project_id == project_id
        ).order_by(Analysis.created_at.desc()).all()
        
        conversions = db_session.query(Conversion).filter(
            Conversion.project_id == project_id
        ).order_by(Conversion.created_at.desc()).all()
        
        nestings = db_session.query(Nesting).filter(
            Nesting.project_id == project_id
        ).order_by(Nesting.created_at.desc()).all()
        
        return render_template('project_detail.html', 
                             project=project, 
                             analyses=analyses,
                             conversions=conversions,
                             nestings=nestings)
    finally:
        db_session.close()


@main_bp.route('/upload', methods=['GET', 'POST'])
@require_auth(Permission.CREATE_PROJECT)
def upload():
    """File upload page."""
    if request.method == 'GET':
        return render_template('upload.html')
    
    # Security check
    client_ip = request.remote_addr
    security_result = security_check(client_ip, '/upload', 'POST', dict(request.headers))
    
    if not security_result['allowed']:
        flash(f"Security check failed: {security_result['reason']}", 'error')
        return render_template('upload.html')
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return render_template('upload.html')
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return render_template('upload.html')
    
    if file:
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{file_id}{file_extension}"
        
        # Save file
        upload_folder = "output/temp"
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, new_filename)
        file.save(file_path)
        
        flash(f'File uploaded successfully: {filename}', 'success')
        return redirect(url_for('main.process_file', file_id=file_id))
    
    return render_template('upload.html')


@main_bp.route('/process/<file_id>')
@require_auth(Permission.ANALYZE_DXF)
def process_file(file_id: str):
    """Process uploaded file."""
    # Find file
    upload_folder = "output/temp"
    file_path = None
    
    for filename in os.listdir(upload_folder):
        if filename.startswith(file_id):
            file_path = os.path.join(upload_folder, filename)
            break
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found', 'error')
        return redirect(url_for('main.upload'))
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.dxf':
        return redirect(url_for('main.analyze_dxf', file_id=file_id))
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return redirect(url_for('main.convert_image', file_id=file_id))
    else:
        flash('Unsupported file type', 'error')
        return redirect(url_for('main.upload'))


@main_bp.route('/analyze/<file_id>', methods=['GET', 'POST'])
@require_auth(Permission.ANALYZE_DXF)
def analyze_dxf(file_id: str):
    """Analyze DXF file."""
    # Find file
    upload_folder = "output/temp"
    file_path = None
    
    for filename in os.listdir(upload_folder):
        if filename.startswith(file_id):
            file_path = os.path.join(upload_folder, filename)
            break
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found', 'error')
        return redirect(url_for('main.upload'))
    
    if request.method == 'GET':
        return render_template('analyze_dxf.html', file_id=file_id, file_path=file_path)
    
    # Process analysis parameters
    analysis_params = {
        'material': request.form.get('material', 'steel'),
        'thickness': float(request.form.get('thickness', 6.0)),
        'kerf': float(request.form.get('kerf', 1.1)),
        'cutting_speed': float(request.form.get('cutting_speed', 1200.0)),
        'cost_per_meter': float(request.form.get('cost_per_meter', 50.0)),
        'sheet_width': float(request.form.get('sheet_width', 3000.0)),
        'sheet_height': float(request.form.get('sheet_height', 1500.0)),
        'spacing': float(request.form.get('spacing', 10.0))
    }
    
    # Create analysis record
    user_id = session.get('user_id')
    db_session = get_db_session()
    
    try:
        from ..database import get_database_manager
        db_manager = get_database_manager()
        
        # Create project if needed
        project_name = f"Analysis {file_id[:8]}"
        project = db_manager.create_project(
            session=db_session,
            user_id=user_id,
            name=project_name,
            description=f"DXF analysis for {os.path.basename(file_path)}"
        )
        
        # Create analysis record
        analysis = db_manager.create_analysis(
            session=db_session,
            project_id=str(project.id),
            name=f"Analysis {file_id[:8]}",
            dxf_file_path=file_path,
            analysis_type='geometric',
            parameters=analysis_params
        )
        
        # Start background task
        from ..tasks import analyze_dxf_task
        task = analyze_dxf_task.delay(
            str(analysis.id),
            file_path,
            user_id,
            analysis_params
        )
        
        flash('Analysis started! You can check the progress in your projects.', 'success')
        return redirect(url_for('main.project_detail', project_id=project.id))
        
    except Exception as e:
        flash(f"Failed to start analysis: {e}", 'error')
        return render_template('analyze_dxf.html', file_id=file_id, file_path=file_path)
    finally:
        db_session.close()


@main_bp.route('/convert/<file_id>', methods=['GET', 'POST'])
@require_auth(Permission.CONVERT_IMAGE)
def convert_image(file_id: str):
    """Convert image to DXF."""
    # Find file
    upload_folder = "output/temp"
    file_path = None
    
    for filename in os.listdir(upload_folder):
        if filename.startswith(file_id):
            file_path = os.path.join(upload_folder, filename)
            break
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found', 'error')
        return redirect(url_for('main.upload'))
    
    if request.method == 'GET':
        return render_template('convert_image.html', file_id=file_id, file_path=file_path)
    
    # Process conversion parameters
    conversion_params = {
        'edge_threshold': float(request.form.get('edge_threshold', 0.33)),
        'min_contour_area': int(request.form.get('min_contour_area', 100)),
        'simplify_tolerance': float(request.form.get('simplify_tolerance', 0.02)),
        'blur_kernel_size': int(request.form.get('blur_kernel_size', 5)),
        'canny_low': int(request.form.get('canny_low', 50)),
        'canny_high': int(request.form.get('canny_high', 150))
    }
    
    # Create conversion record
    user_id = session.get('user_id')
    db_session = get_db_session()
    
    try:
        from ..database import get_database_manager
        db_manager = get_database_manager()
        
        # Create project if needed
        project_name = f"Conversion {file_id[:8]}"
        project = db_manager.create_project(
            session=db_session,
            user_id=user_id,
            name=project_name,
            description=f"Image conversion for {os.path.basename(file_path)}"
        )
        
        # Create conversion record
        conversion = db_manager.create_conversion(
            session=db_session,
            project_id=str(project.id),
            name=f"Conversion {file_id[:8]}",
            image_file_path=file_path,
            conversion_algorithm='unified',
            parameters=conversion_params
        )
        
        # Image conversion is no longer available
        return jsonify({'error': 'Image conversion feature has been removed'}), 404
        
        flash('Conversion started! You can check the progress in your projects.', 'success')
        return redirect(url_for('main.project_detail', project_id=project.id))
        
    except Exception as e:
        flash(f"Failed to start conversion: {e}", 'error')
        return render_template('convert_image.html', file_id=file_id, file_path=file_path)
    finally:
        db_session.close()


@main_bp.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'healthy', 'timestamp': '2024-01-01T00:00:00Z'}
