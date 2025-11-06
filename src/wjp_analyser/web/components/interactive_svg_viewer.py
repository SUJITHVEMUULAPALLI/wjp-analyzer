"""
Interactive SVG Viewer Component
===============================

Streamlit component for displaying interactive SVG with click detection and bidirectional selection.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Any, Optional, Tuple
import json
import tempfile
import os


def interactive_svg_viewer(
    svg_code: str,
    object_mapping: Dict[str, Any] = None,
    selected_object_id: str = None,
    width: int = 800,
    height: int = 600
) -> Optional[str]:
    """
    Display interactive SVG with click detection.
    
    Args:
        svg_code: SVG code to display
        object_mapping: Mapping of object IDs to component data
        selected_object_id: Currently selected object ID
        width: Viewer width
        height: Viewer height
        key: Streamlit component key
    
    Returns:
        Clicked object ID or None
    """
    
    # Create HTML wrapper with interactive features
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 10px;
                font-family: Arial, sans-serif;
                background: #f8fafc;
            }}
            .svg-container {{
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                background: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .svg-header {{
                background: #1f2937;
                color: white;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: 600;
            }}
            .svg-content {{
                position: relative;
            }}
            .tooltip {{
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                max-width: 200px;
                display: none;
            }}
            .controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255, 255, 255, 0.9);
                padding: 8px;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .control-btn {{
                background: #3b82f6;
                color: white;
                border: none;
                padding: 4px 8px;
                margin: 2px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
            }}
            .control-btn:hover {{
                background: #2563eb;
            }}
        </style>
    </head>
    <body>
        <div class="svg-container">
            <div class="svg-header">
                🎯 Interactive DXF Preview - Click objects to select them
            </div>
            <div class="svg-content">
                <div class="controls">
                    <button class="control-btn" onclick="resetView()">Reset View</button>
                    <button class="control-btn" onclick="fitToScreen()">Fit Screen</button>
                    <button class="control-btn" onclick="toggleGrid()">Grid</button>
                </div>
                <div id="svg-wrapper">
                    {svg_code}
                </div>
                <div id="tooltip" class="tooltip"></div>
            </div>
        </div>
        
        <script>
            let selectedObjectId = null;
            let showGrid = false;
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {{
                setupInteractivity();
                if ('{selected_object_id}') {{
                    highlightObject('{selected_object_id}');
                }}
            }});
            
            function setupInteractivity() {{
                // Add click listeners to all objects
                document.querySelectorAll('.dxf-object, [data-object-id]').forEach(element => {{
                    element.addEventListener('click', handleObjectClick);
                    element.addEventListener('mouseenter', handleMouseEnter);
                    element.addEventListener('mouseleave', handleMouseLeave);
                }});
            }}
            
            function handleObjectClick(event) {{
                const target = event.target;
                const objectId = target.getAttribute('data-object-id') || target.getAttribute('id');
                
                if (objectId) {{
                    // Remove previous selection
                    if (selectedObjectId) {{
                        const prevElement = document.querySelector(`[data-object-id="${{selectedObjectId}}"]`);
                        if (prevElement) {{
                            prevElement.classList.remove('selected');
                        }}
                    }}
                    
                    // Add selection to clicked object
                    target.classList.add('selected');
                    selectedObjectId = objectId;
                    
                    // Send selection to Streamlit
                    if (window.parent && window.parent.postMessage) {{
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: objectId
                        }}, '*');
                    }}
                    
                    // Show selection info
                    showSelectionInfo(objectId);
                }}
            }}
            
            function handleMouseEnter(event) {{
                const objectId = event.target.getAttribute('data-object-id') || event.target.getAttribute('id');
                if (objectId) {{
                    showTooltip(event, objectId);
                }}
            }}
            
            function handleMouseLeave(event) {{
                hideTooltip();
            }}
            
            function showTooltip(event, objectId) {{
                const tooltip = document.getElementById('tooltip');
                const objectInfo = getObjectInfo(objectId);
                
                tooltip.innerHTML = `
                    <strong>Object: ${{objectId}}</strong><br>
                    Layer: ${{objectInfo.layer}}<br>
                    Area: ${{objectInfo.area.toFixed(1)}} mm²<br>
                    Vertices: ${{objectInfo.vertex_count}}
                `;
                
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX - event.target.closest('.svg-content').offsetLeft + 10) + 'px';
                tooltip.style.top = (event.clientY - event.target.closest('.svg-content').offsetTop - 60) + 'px';
            }}
            
            function hideTooltip() {{
                const tooltip = document.getElementById('tooltip');
                tooltip.style.display = 'none';
            }}
            
            function showSelectionInfo(objectId) {{
                const objectInfo = getObjectInfo(objectId);
                const info = `
                    <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 4px; padding: 10px; margin: 10px 0;">
                        <strong>Selected: ${{objectId}}</strong><br>
                        Layer: ${{objectInfo.layer}}<br>
                        Area: ${{objectInfo.area.toFixed(1)}} mm²<br>
                        Perimeter: ${{objectInfo.perimeter.toFixed(1)}} mm<br>
                        Vertices: ${{objectInfo.vertex_count}}
                    </div>
                `;
                
                // Create or update info display
                let infoDiv = document.getElementById('selection-info');
                if (!infoDiv) {{
                    infoDiv = document.createElement('div');
                    infoDiv.id = 'selection-info';
                    infoDiv.style.cssText = 'position: absolute; bottom: 10px; left: 10px; right: 10px; z-index: 100;';
                    document.querySelector('.svg-content').appendChild(infoDiv);
                }}
                infoDiv.innerHTML = info;
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
            
            function getObjectInfo(objectId) {{
                // Mock object info - in real implementation, this would come from the mapping
                const objectMapping = {json.dumps(object_mapping or {})};
                return objectMapping[objectId] || {{
                    id: objectId,
                    area: 0,
                    perimeter: 0,
                    layer: 'Unknown',
                    vertex_count: 0
                }};
            }}
            
            function resetView() {{
                const svg = document.querySelector('svg');
                if (svg) {{
                    svg.style.transform = 'scale(1) translate(0, 0)';
                }}
            }}
            
            function fitToScreen() {{
                const svg = document.querySelector('svg');
                const container = document.querySelector('.svg-content');
                if (svg && container) {{
                    const svgRect = svg.getBoundingClientRect();
                    const containerRect = container.getBoundingClientRect();
                    const scale = Math.min(containerRect.width / svgRect.width, containerRect.height / svgRect.height) * 0.9;
                    svg.style.transform = `scale(${{scale}})`;
                }}
            }}
            
            function toggleGrid() {{
                showGrid = !showGrid;
                const svg = document.querySelector('svg');
                if (svg) {{
                    if (showGrid) {{
                        // Add grid pattern
                        const defs = svg.querySelector('defs') || svg.insertBefore(document.createElementNS('http://www.w3.org/2000/svg', 'defs'), svg.firstChild);
                        const pattern = document.createElementNS('http://www.w3.org/2000/svg', 'pattern');
                        pattern.setAttribute('id', 'grid');
                        pattern.setAttribute('width', '20');
                        pattern.setAttribute('height', '20');
                        pattern.setAttribute('patternUnits', 'userSpaceOnUse');
                        
                        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        path.setAttribute('d', 'M 20 0 L 0 0 0 20');
                        path.setAttribute('fill', 'none');
                        path.setAttribute('stroke', '#e5e7eb');
                        path.setAttribute('stroke-width', '1');
                        
                        pattern.appendChild(path);
                        defs.appendChild(pattern);
                        
                        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                        rect.setAttribute('width', '100%');
                        rect.setAttribute('height', '100%');
                        rect.setAttribute('fill', 'url(#grid)');
                        svg.insertBefore(rect, svg.firstChild);
                    }} else {{
                        // Remove grid
                        const gridRect = svg.querySelector('rect[fill="url(#grid)"]');
                        if (gridRect) {{
                            gridRect.remove();
                        }}
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
    </body>
    </html>
    """
    
    # Display the interactive SVG
    clicked_object = components.html(
        html_code,
        height=height + 100  # Add extra height for header and controls
    )
    
    return clicked_object


def create_object_selection_panel(
    components: List[Dict[str, Any]],
    selected_object_id: str = None,
    object_mapping: Dict[str, Any] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Create an object selection panel with bidirectional linking.
    
    Args:
        components: List of component data
        selected_object_id: Currently selected object ID
        object_mapping: Mapping of object IDs to component data
    
    Returns:
        Tuple of (selected_object_id, updated_object_mapping)
    """
    
    st.markdown("### 🎯 Object Selection")
    
    # Create object list
    object_list = []
    for i, comp in enumerate(components):
        obj_id = f"OBJ-{comp.get('id', i+1):03d}"
        object_list.append({
            'id': obj_id,
            'component_id': comp.get('id', i+1),
            'area': comp.get('area', 0),
            'perimeter': comp.get('perimeter', 0),
            'layer': comp.get('layer', 'Unknown'),
            'vertex_count': comp.get('vertex_count', 0)
        })
    
    # Update object mapping
    if object_mapping is None:
        object_mapping = {}
    
    for obj in object_list:
        object_mapping[obj['id']] = obj
    
    # Object selection dropdown
    object_options = [obj['id'] for obj in object_list]
    current_selection = selected_object_id if selected_object_id in object_options else object_options[0] if object_options else None
    
    selected_object = st.selectbox(
        "Select Object:",
        object_options,
        index=object_options.index(current_selection) if current_selection in object_options else 0,
        key="object_selection"
    )
    
    # Display selected object info
    if selected_object and selected_object in object_mapping:
        obj_info = object_mapping[selected_object]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Area", f"{obj_info['area']:.1f} mm²")
        
        with col2:
            st.metric("Perimeter", f"{obj_info['perimeter']:.1f} mm")
        
        with col3:
            st.metric("Vertices", obj_info['vertex_count'])
        
        st.info(f"**Layer:** {obj_info['layer']}")
    
    return selected_object, object_mapping


def create_layer_filter_panel(layers: Dict[str, int]) -> List[str]:
    """
    Create a layer filter panel.
    
    Args:
        layers: Dictionary of layer names and object counts
    
    Returns:
        List of selected layer names
    """
    
    st.markdown("### 🎨 Layer Filter")
    
    if not layers:
        st.info("No layers found.")
        return []
    
    # Layer checkboxes
    selected_layers = []
    for layer_name, count in layers.items():
        if st.checkbox(f"{layer_name} ({count} objects)", value=True, key=f"layer_{layer_name}"):
            selected_layers.append(layer_name)
    
    return selected_layers


def create_interactive_preview_controls() -> Dict[str, Any]:
    """
    Create interactive preview controls.
    
    Returns:
        Dictionary of control settings
    """
    
    st.markdown("### ⚙️ Preview Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_grid = st.checkbox("Show Grid", value=False)
        show_labels = st.checkbox("Show Object Labels", value=False)
        auto_fit = st.checkbox("Auto Fit to Screen", value=True)
    
    with col2:
        zoom_level = st.slider("Zoom Level", 0.1, 3.0, 1.0, 0.1)
        highlight_color = st.color_picker("Highlight Color", "#EF4444")
        stroke_width = st.slider("Stroke Width", 1, 5, 2)
    
    return {
        'show_grid': show_grid,
        'show_labels': show_labels,
        'auto_fit': auto_fit,
        'zoom_level': zoom_level,
        'highlight_color': highlight_color,
        'stroke_width': stroke_width
    }
