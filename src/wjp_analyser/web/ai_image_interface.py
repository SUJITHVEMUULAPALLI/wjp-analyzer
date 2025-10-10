#!/usr/bin/env python3
"""
AI-Enhanced Image-to-DXF Web Interface
=====================================

Live interactive interface for AI-powered image conversion with Ollama and OpenAI integration.
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import os
import json
import time
from typing import Dict, Any, Optional
import threading
import queue

from ..image_processing.converters.ai_enhanced_converter import AIEnhancedImageConverter
from ..config.preset_loader import PresetLoader

# Create blueprint
ai_image_bp = Blueprint('ai_image', __name__, url_prefix='/ai-image')

# Global converter instance
converter = None
conversion_queue = queue.Queue()
conversion_results = {}

def init_converter():
    """Initialize the AI converter with configuration."""
    global converter
    
    # Load configuration
    preset_loader = PresetLoader()
    config = preset_loader.load_preset("standard")
    
    # Get API keys from environment or config
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        # Try to load from config file
        try:
            with open('config/api_keys.yaml', 'r') as f:
                import yaml
                api_config = yaml.safe_load(f)
                openai_key = api_config.get('openai', {}).get('api_key')
        except:
            pass
    
    converter = AIEnhancedImageConverter(
        ollama_url="http://localhost:11434",
        openai_api_key=openai_key
    )

@ai_image_bp.route('/')
def index():
    """Main AI image interface page."""
    return render_template('ai_image_interface.html')

@ai_image_bp.route('/generate', methods=['POST'])
def generate_image():
    """Generate image from text prompt using OpenAI."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        style = data.get('style', 'waterjet')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        if not converter:
            init_converter()
        
        # Generate image
        image_path = converter.generate_image_from_prompt(prompt, style)
        
        if image_path:
            return jsonify({
                'success': True,
                'image_path': image_path,
                'message': 'Image generated successfully'
            })
        else:
            return jsonify({'error': 'Failed to generate image'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_image_bp.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze image using Ollama AI."""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        analysis_type = data.get('analysis_type', 'waterjet')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Invalid image path'}), 400
        
        if not converter:
            init_converter()
        
        # Perform AI analysis
        analysis = converter.analyze_with_ollama(image_path, analysis_type)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_image_bp.route('/convert', methods=['POST'])
def convert_image():
    """Convert image to DXF with AI guidance."""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        output_name = data.get('output_name', 'converted_design')
        user_preferences = data.get('preferences', {})
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Invalid image path'}), 400
        
        if not converter:
            init_converter()
        
        # Generate output paths
        timestamp = int(time.time())
        output_dxf = os.path.join('output', 'dxf', f'{output_name}_{timestamp}.dxf')
        os.makedirs(os.path.dirname(output_dxf), exist_ok=True)
        
        # Start conversion in background
        conversion_id = f"conv_{timestamp}"
        conversion_queue.put({
            'id': conversion_id,
            'image_path': image_path,
            'output_dxf': output_dxf,
            'preferences': user_preferences
        })
        
        return jsonify({
            'success': True,
            'conversion_id': conversion_id,
            'message': 'Conversion started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_image_bp.route('/status/<conversion_id>')
def get_conversion_status(conversion_id):
    """Get conversion status and results."""
    if conversion_id in conversion_results:
        result = conversion_results[conversion_id]
        return jsonify({
            'success': True,
            'status': 'complete',
            'result': result
        })
    else:
        return jsonify({
            'success': True,
            'status': 'processing',
            'message': 'Conversion in progress...'
        })

@ai_image_bp.route('/live-convert', methods=['POST'])
def live_convert():
    """Perform live interactive conversion."""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        output_name = data.get('output_name', 'live_converted')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Invalid image path'}), 400
        
        if not converter:
            init_converter()
        
        # Generate output paths
        timestamp = int(time.time())
        output_dxf = os.path.join('output', 'dxf', f'{output_name}_{timestamp}.dxf')
        os.makedirs(os.path.dirname(output_dxf), exist_ok=True)
        
        # Progress tracking
        progress_updates = []
        
        def progress_callback(message, percentage):
            progress_updates.append({
                'message': message,
                'percentage': percentage,
                'timestamp': time.time()
            })
        
        # Perform live conversion
        result = converter.live_interactive_conversion(
            image_path, output_dxf, progress_callback
        )
        
        # Add progress updates to result
        result['progress_updates'] = progress_updates
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_image_bp.route('/preview/<filename>')
def get_preview(filename):
    """Serve preview images."""
    try:
        preview_path = os.path.join('output', 'temp', filename)
        if os.path.exists(preview_path):
            from flask import send_file
            return send_file(preview_path)
        else:
            return jsonify({'error': 'Preview not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def background_conversion_worker():
    """Background worker for processing conversion queue."""
    global converter, conversion_results
    
    if not converter:
        init_converter()
    
    while True:
        try:
            # Get next conversion task
            task = conversion_queue.get(timeout=1)
            
            conversion_id = task['id']
            image_path = task['image_path']
            output_dxf = task['output_dxf']
            preferences = task['preferences']
            
            # Perform conversion
            result = converter.convert_with_ai_guidance(
                image_path, output_dxf, None, preferences
            )
            
            # Store result
            conversion_results[conversion_id] = result
            
            # Mark task as done
            conversion_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Background conversion error: {e}")
            if 'conversion_id' in locals():
                conversion_results[conversion_id] = {'error': str(e)}

# Start background worker
conversion_thread = threading.Thread(target=background_conversion_worker, daemon=True)
conversion_thread.start()
