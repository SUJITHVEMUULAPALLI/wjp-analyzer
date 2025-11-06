"""
DXF to SVG Converter with Interactive Object IDs
===============================================

Converts DXF files to interactive SVG with clickable object IDs for bidirectional selection.
"""

import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import json

try:
    import ezdxf
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.svg import SVGBackend
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

try:
    import svgwrite
    SVGWRITE_AVAILABLE = True
except ImportError:
    SVGWRITE_AVAILABLE = False


def convert_dxf_to_interactive_svg(dxf_path: str, components: List[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Convert DXF file to interactive SVG with object IDs.
    
    Args:
        dxf_path: Path to DXF file
        components: List of component data from analysis (optional)
    
    Returns:
        Tuple of (svg_code, object_mapping)
    """
    
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf is required for DXF to SVG conversion. Install with: pip install ezdxf")
    
    try:
        # Read DXF file
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        # Create SVG backend
        backend = SVGBackend()
        frontend = Frontend(RenderContext(doc), backend)
        
        # Draw all entities
        frontend.draw_layout(msp)
        
        # Get base SVG - fix the get_string() call
        try:
            # Try different method signatures for different ezdxf versions
            if hasattr(backend, 'get_string'):
                # Check if get_string needs arguments
                import inspect
                sig = inspect.signature(backend.get_string)
                if len(sig.parameters) > 0:
                    # Try with msp argument
                    svg_code = backend.get_string(msp)
                else:
                    # Try without arguments
                    svg_code = backend.get_string()
            else:
                # Fallback if get_string doesn't exist
                svg_code = create_fallback_svg(components)
        except Exception as e:
            # Create a simple SVG as fallback
            print(f"ezdxf SVG creation failed: {e}")
            svg_code = create_fallback_svg(components)
        
        # Create object mapping from components if provided
        object_mapping = {}
        if components:
            for i, comp in enumerate(components):
                obj_id = f"OBJ-{comp.get('id', i+1):03d}"
                object_mapping[obj_id] = {
                    'id': comp.get('id', i+1),
                    'area': comp.get('area', 0),
                    'perimeter': comp.get('perimeter', 0),
                    'layer': comp.get('layer', 'Unknown'),
                    'vertex_count': comp.get('vertex_count', 0),
                    'points': comp.get('points', [])
                }
        
        # Enhance SVG with interactive IDs
        enhanced_svg = enhance_svg_with_object_ids(svg_code, object_mapping)
        
        return enhanced_svg, object_mapping
        
    except Exception as e:
        raise Exception(f"Failed to convert DXF to SVG: {str(e)}")


def enhance_svg_with_object_ids(svg_code: str, object_mapping: Dict[str, Any]) -> str:
    """
    Enhance SVG code with interactive object IDs and styling.
    
    Args:
        svg_code: Base SVG code from ezdxf
        object_mapping: Mapping of object IDs to component data
    
    Returns:
        Enhanced SVG code with interactive elements
    """
    
    # Add interactive styling and JavaScript
    interactive_style = """
    <style>
        .dxf-object {
            cursor: pointer;
            transition: stroke-width 0.2s ease;
        }
        .dxf-object:hover {
            stroke-width: 3px !important;
            stroke: #3B82F6 !important;
        }
        .dxf-object.selected {
            stroke-width: 4px !important;
            stroke: #EF4444 !important;
            fill: rgba(239, 68, 68, 0.1) !important;
        }
        .dxf-object.highlighted {
            stroke-width: 3px !important;
            stroke: #F59E0B !important;
            fill: rgba(245, 158, 11, 0.1) !important;
        }
    </style>
    """
    
    # Add JavaScript for click handling
    interactive_script = """
    <script>
        let selectedObjectId = null;
        
        function handleObjectClick(event) {
            const target = event.target;
            const objectId = target.getAttribute('data-object-id');
            
            if (objectId) {
                // Remove previous selection
                if (selectedObjectId) {
                    const prevElement = document.querySelector(`[data-object-id="${selectedObjectId}"]`);
                    if (prevElement) {
                        prevElement.classList.remove('selected');
                    }
                }
                
                // Add selection to clicked object
                target.classList.add('selected');
                selectedObjectId = objectId;
                
                // Send selection to Streamlit
                if (window.parent && window.parent.postMessage) {
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: objectId
                    }, '*');
                }
                
                // Show tooltip
                showTooltip(event, objectId);
            }
        }
        
        function highlightObject(objectId) {
            // Remove previous highlights
            document.querySelectorAll('.dxf-object.highlighted').forEach(el => {
                el.classList.remove('highlighted');
            });
            
            // Add highlight to specified object
            if (objectId) {
                const element = document.querySelector(`[data-object-id="${objectId}"]`);
                if (element) {
                    element.classList.add('highlighted');
                }
            }
        }
        
        function showTooltip(event, objectId) {
            // Remove existing tooltip
            const existingTooltip = document.getElementById('dxf-tooltip');
            if (existingTooltip) {
                existingTooltip.remove();
            }
            
            // Create tooltip
            const tooltip = document.createElement('div');
            tooltip.id = 'dxf-tooltip';
            tooltip.style.cssText = `
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                max-width: 200px;
            `;
            tooltip.innerHTML = `Object: ${objectId}`;
            
            document.body.appendChild(tooltip);
            
            // Position tooltip
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY - 30) + 'px';
            
            // Remove tooltip after 3 seconds
            setTimeout(() => {
                if (tooltip.parentNode) {
                    tooltip.remove();
                }
            }, 3000);
        }
        
        // Add click listeners to all objects
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.dxf-object').forEach(element => {
                element.addEventListener('click', handleObjectClick);
            });
        });
        
        // Listen for highlight messages from Streamlit
        window.addEventListener('message', function(event) {
            if (event.data && event.data.type === 'highlight-object') {
                highlightObject(event.data.objectId);
            }
        });
    </script>
    """
    
    # Modify SVG to add interactive attributes
    enhanced_svg = svg_code
    
    # Add interactive class and data attributes to paths
    import re
    
    # Pattern to match SVG paths
    path_pattern = r'<path([^>]*?)>'
    
    def add_interactive_attributes(match):
        path_attrs = match.group(1)
        
        # Add interactive class
        if 'class=' not in path_attrs:
            path_attrs += ' class="dxf-object"'
        else:
            path_attrs = re.sub(r'class="([^"]*)"', r'class="\1 dxf-object"', path_attrs)
        
        # Add data-object-id (use a simple counter for now)
        if 'data-object-id=' not in path_attrs:
            # Generate a simple ID based on path index
            path_attrs += ' data-object-id="OBJ-001"'
        
        return f'<path{path_attrs}>'
    
    # Apply interactive attributes to paths
    enhanced_svg = re.sub(path_pattern, add_interactive_attributes, enhanced_svg)
    
    # Insert styles and scripts into SVG
    if '<defs>' in enhanced_svg:
        enhanced_svg = enhanced_svg.replace('<defs>', f'<defs>{interactive_style}')
    else:
        enhanced_svg = enhanced_svg.replace('<svg', f'<svg><defs>{interactive_style}</defs>')
    
    # Add script before closing SVG
    enhanced_svg = enhanced_svg.replace('</svg>', f'{interactive_script}</svg>')
    
    return enhanced_svg


def create_simple_interactive_svg(components: List[Dict[str, Any]], width: int = 800, height: int = 600) -> str:
    """
    Create a simple interactive SVG from component data.
    
    Args:
        components: List of component data
        width: SVG width
        height: SVG height
    
    Returns:
        Interactive SVG code
    """
    
    if not SVGWRITE_AVAILABLE:
        # Fallback to basic SVG
        return create_basic_svg(components, width, height)
    
    try:
        # Create SVG drawing
        dwg = svgwrite.Drawing(size=(width, height))
        
        # Add styles
        dwg.defs.add(dwg.style("""
            .dxf-object {
                cursor: pointer;
                transition: stroke-width 0.2s ease;
            }
            .dxf-object:hover {
                stroke-width: 3px !important;
                stroke: #3B82F6 !important;
            }
            .dxf-object.selected {
                stroke-width: 4px !important;
                stroke: #EF4444 !important;
                fill: rgba(239, 68, 68, 0.1) !important;
            }
        """))
        
        # Add components as SVG elements
        for i, comp in enumerate(components):
            obj_id = f"OBJ-{comp.get('id', i+1):03d}"
            points = comp.get('points', [])
            
            if len(points) >= 2:
                # Create path from points
                path_data = f"M {points[0][0]},{points[0][1]}"
                for point in points[1:]:
                    path_data += f" L {point[0]},{point[1]}"
                path_data += " Z"  # Close path
                
                # Add path to SVG
                dwg.add(dwg.path(
                    d=path_data,
                    class_="dxf-object",
                    data_object_id=obj_id,
                    stroke="#1F2937",
                    stroke_width=1,
                    fill="none"
                ))
        
        # Add JavaScript for interactivity
        script = """
        <script>
            function handleClick(event) {
                const objectId = event.target.getAttribute('data-object-id');
                if (objectId && window.parent) {
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: objectId
                    }, '*');
                }
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                document.querySelectorAll('.dxf-object').forEach(el => {
                    el.addEventListener('click', handleClick);
                });
            });
        </script>
        """
        
        svg_code = dwg.tostring()
        svg_code = svg_code.replace('</svg>', f'{script}</svg>')
        
        return svg_code
        
    except Exception as e:
        # Fallback to basic SVG
        return create_basic_svg(components, width, height)


def create_basic_svg(components: List[Dict[str, Any]], width: int = 800, height: int = 600) -> str:
    """
    Create a basic interactive SVG as fallback.
    
    Args:
        components: List of component data
        width: SVG width
        height: SVG height
    
    Returns:
        Basic interactive SVG code
    """
    
    svg_elements = []
    
    for i, comp in enumerate(components):
        obj_id = f"OBJ-{comp.get('id', i+1):03d}"
        points = comp.get('points', [])
        
        if len(points) >= 2:
            # Create path from points
            path_data = f"M {points[0][0]},{points[0][1]}"
            for point in points[1:]:
                path_data += f" L {point[0]},{point[1]}"
            path_data += " Z"  # Close path
            
            svg_elements.append(f'''
                <path d="{path_data}" 
                      class="dxf-object" 
                      data-object-id="{obj_id}"
                      stroke="#1F2937" 
                      stroke-width="1" 
                      fill="none"
                      onclick="handleClick(event, '{obj_id}')" />
            ''')
    
    svg_code = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .dxf-object {{
                cursor: pointer;
                transition: stroke-width 0.2s ease;
            }}
            .dxf-object:hover {{
                stroke-width: 3px !important;
                stroke: #3B82F6 !important;
            }}
            .dxf-object.selected {{
                stroke-width: 4px !important;
                stroke: #EF4444 !important;
                fill: rgba(239, 68, 68, 0.1) !important;
            }}
        </style>
        {''.join(svg_elements)}
        <script>
            function handleClick(event, objectId) {{
                // Remove previous selection
                document.querySelectorAll('.dxf-object.selected').forEach(el => {{
                    el.classList.remove('selected');
                }});
                
                // Add selection to clicked object
                event.target.classList.add('selected');
                
                // Send to Streamlit
                if (window.parent && window.parent.postMessage) {{
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: objectId
                    }}, '*');
                }}
            }}
        </script>
    </svg>
    '''
    
    return svg_code


def create_fallback_svg(components: List[Dict[str, Any]], width: int = 800, height: int = 600) -> str:
    """
    Create an enhanced interactive SVG from component data.
    
    Args:
        components: List of component data
        width: SVG width
        height: SVG height
    
    Returns:
        Enhanced interactive SVG code
    """
    
    svg_elements = []
    
    # Calculate bounding box for proper scaling
    all_points = []
    for comp in components:
        points = comp.get('points', [])
        all_points.extend(points)
    
    if all_points:
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        # Calculate scale and offset for proper display
        scale_x = (width - 100) / (max_x - min_x) if max_x > min_x else 1
        scale_y = (height - 100) / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y, 1)  # Don't scale up
        
        offset_x = (width - (max_x - min_x) * scale) / 2 - min_x * scale
        offset_y = (height - (max_y - min_y) * scale) / 2 - min_y * scale
    else:
        scale = 1
        offset_x = 0
        offset_y = 0
    
    # Create SVG elements for each component
    for i, comp in enumerate(components):
        obj_id = f"OBJ-{comp.get('id', i+1):03d}"
        points = comp.get('points', [])
        layer = comp.get('layer', 'Unknown')
        
        # Define colors based on layer
        layer_colors = {
            'OUTER': '#3B82F6',
            'INNER': '#10B981', 
            'DECOR': '#EF4444',
            'HOLE': '#F59E0B',
            'CUT': '#8B5CF6',
            'ENGRAVE': '#6B7280'
        }
        stroke_color = layer_colors.get(layer, '#1F2937')
        
        if len(points) >= 2:
            # Transform points with scale and offset
            transformed_points = []
            for point in points:
                x = point[0] * scale + offset_x
                y = point[1] * scale + offset_y
                transformed_points.append((x, y))
            
            # Create path from transformed points
            path_data = f"M {transformed_points[0][0]},{transformed_points[0][1]}"
            for point in transformed_points[1:]:
                path_data += f" L {point[0]},{point[1]}"
            path_data += " Z"  # Close path
            
            svg_elements.append(f'''
                <path d="{path_data}" 
                      class="dxf-object" 
                      data-object-id="{obj_id}"
                      data-layer="{layer}"
                      stroke="{stroke_color}" 
                      stroke-width="2" 
                      fill="none"
                      onclick="handleClick(event, '{obj_id}')" />
            ''')
    
    # Create enhanced SVG with better styling and functionality
    svg_code = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
        <defs>
            <style>
                .dxf-object {{
                    cursor: pointer;
                    transition: all 0.2s ease;
                }}
                .dxf-object:hover {{
                    stroke-width: 4px !important;
                    filter: drop-shadow(0 0 4px currentColor);
                }}
                .dxf-object.selected {{
                    stroke-width: 5px !important;
                    stroke: #EF4444 !important;
                    fill: rgba(239, 68, 68, 0.15) !important;
                    filter: drop-shadow(0 0 8px #EF4444);
                }}
                .dxf-object.highlighted {{
                    stroke-width: 4px !important;
                    stroke: #F59E0B !important;
                    fill: rgba(245, 158, 11, 0.1) !important;
                    filter: drop-shadow(0 0 6px #F59E0B);
                }}
                .grid {{
                    stroke: #E5E7EB;
                    stroke-width: 0.5;
                    opacity: 0.3;
                }}
            </style>
        </defs>
        
        <!-- Grid background -->
        <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#E5E7EB" stroke-width="0.5" opacity="0.3"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" class="grid"/>
        
        <!-- DXF Objects -->
        {''.join(svg_elements)}
        
        <script>
            let selectedObjectId = null;
            
            function handleClick(event, objectId) {{
                // Remove previous selection
                if (selectedObjectId) {{
                    const prevElement = document.querySelector(`[data-object-id="${{selectedObjectId}}"]`);
                    if (prevElement) {{
                        prevElement.classList.remove('selected');
                    }}
                }}
                
                // Add selection to clicked object
                event.target.classList.add('selected');
                selectedObjectId = objectId;
                
                // Send to Streamlit
                if (window.parent && window.parent.postMessage) {{
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: objectId
                    }}, '*');
                }}
                
                // Show selection info
                showSelectionInfo(objectId);
            }}
            
            function showSelectionInfo(objectId) {{
                const element = document.querySelector(`[data-object-id="${{objectId}}"]`);
                if (element) {{
                    const layer = element.getAttribute('data-layer') || 'Unknown';
                    const info = `Selected: ${{objectId}} (Layer: ${{layer}})`;
                    
                    // Create or update info display
                    let infoDiv = document.getElementById('selection-info');
                    if (!infoDiv) {{
                        infoDiv = document.createElement('div');
                        infoDiv.id = 'selection-info';
                        infoDiv.style.cssText = `
                            position: absolute;
                            bottom: 10px;
                            left: 10px;
                            background: rgba(0, 0, 0, 0.8);
                            color: white;
                            padding: 8px 12px;
                            border-radius: 4px;
                            font-size: 12px;
                            z-index: 1000;
                        `;
                        document.body.appendChild(infoDiv);
                    }}
                    infoDiv.textContent = info;
                }}
            }}
            
            function highlightObject(objectId) {{
                // Remove previous highlights
                document.querySelectorAll('.dxf-object.highlighted').forEach(el => {{
                    el.classList.remove('highlighted');
                }});
                
                // Add highlight to specified object
                if (objectId) {{
                    const element = document.querySelector(`[data-object-id="${{objectId}}"]`);
                    if (element) {{
                        element.classList.add('highlighted');
                    }}
                }}
            }}
            
            // Listen for messages from Streamlit
            window.addEventListener('message', function(event) {{
                if (event.data && event.data.type === 'highlight-object') {{
                    highlightObject(event.data.objectId);
                }}
            }});
        </script>
    </svg>
    '''
    
    return svg_code


def get_object_info(object_id: str, object_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about a specific object.
    
    Args:
        object_id: Object ID (e.g., "OBJ-001")
        object_mapping: Mapping of object IDs to component data
    
    Returns:
        Object information dictionary
    """
    
    return object_mapping.get(object_id, {
        'id': object_id,
        'area': 0,
        'perimeter': 0,
        'layer': 'Unknown',
        'vertex_count': 0,
        'points': []
    })
