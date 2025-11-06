#!/usr/bin/env python3
"""
Waterjet DXF Analyzer - Web Interface
=====================================

Refactored Flask application with modular structure.
"""

import os
import sys
import logging
from flask import Flask

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import modules
from .routes import main_bp
from .auth_routes import auth_bp
from .ai_image_interface import ai_image_bp
from ..config.unified_config_manager import get_config_manager
from ..database import init_database_from_config
from ..auth.enhanced_auth import auth_manager
from ..auth.api_key_manager import migrate_existing_keys
from ..auth.security_middleware import security_middleware

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), "templates"),
                static_folder=os.path.join(os.path.dirname(__file__), "static"))
    
    # Initialize configuration
    config_manager = get_config_manager()
    config = config_manager.get_config()
    
    # Set secret key
    if config.security.secret_key:
        app.secret_key = config.security.secret_key
    else:
        import secrets
        app.secret_key = secrets.token_urlsafe(32)
        logger.warning("Using generated secret key. Set WJP_SECRET_KEY environment variable for production.")
    
    # Initialize database
    try:
        init_database_from_config()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Migrate existing API keys
    migrate_existing_keys()
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(ai_image_bp, url_prefix='/ai')
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Setup security middleware
    setup_security_middleware(app)
    
    logger.info("Flask application created successfully")
    return app


def setup_error_handlers(app: Flask):
    """Setup error handlers."""
    from ..utils.error_handler import WJPAnalyserError, create_error_response
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(WJPAnalyserError)
    def handle_wjp_error(error):
        response = create_error_response(error)
        return response, 400


def setup_security_middleware(app: Flask):
    """Setup security middleware."""
    from flask import request, g
    
    @app.before_request
    def security_check():
        """Perform security check on each request."""
        client_ip = request.remote_addr
        endpoint = request.endpoint
        method = request.method
        headers = dict(request.headers)
        
        # Skip security check for static files and health checks
        if endpoint in ['static', 'health']:
            return
        
        security_result = security_middleware.check_request(
            client_ip, endpoint, method, headers
        )
        
        if not security_result['allowed']:
            from flask import abort
            abort(403, description=f"Security check failed: {security_result['reason']}")


# Create application instance
app = create_app()


if __name__ == '__main__':
    config = get_config_manager().get_config()
    app.run(
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug,
        threaded=config.server.threaded
    )
