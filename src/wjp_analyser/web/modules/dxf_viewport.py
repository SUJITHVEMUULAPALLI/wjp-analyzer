"""
DXF Viewport Utilities

Provides grid overlay, zoom/pan controls, and enhanced SVG rendering.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict
import re


def add_grid_overlay(svg_text: str, grid_size: float = 10.0, grid_color: str = "#e0e0e0", 
                     show_axes: bool = True, axes_color: str = "#888888") -> str:
    """
    Add a grid overlay to SVG content.
    
    Args:
        svg_text: Original SVG content
        grid_size: Grid spacing in DXF units
        grid_color: Color for grid lines
        show_axes: Whether to show X/Y axes
        axes_color: Color for axes lines
    
    Returns:
        SVG text with grid overlay
    """
    # Extract viewBox from SVG
    viewbox_match = re.search(r'viewBox="([^"]+)"', svg_text)
    if not viewbox_match:
        # Try to extract width/height
        width_match = re.search(r'width="([^"]+)"', svg_text)
        height_match = re.search(r'height="([^"]+)"', svg_text)
        if width_match and height_match:
            try:
                width = float(re.sub(r'[^\d.]', '', width_match.group(1)))
                height = float(re.sub(r'[^\d.]', '', height_match.group(1)))
                viewbox = f"0 0 {width} {height}"
            except:
                viewbox = "0 0 1000 1000"  # Default
        else:
            viewbox = "0 0 1000 1000"  # Default
    else:
        viewbox = viewbox_match.group(1)
    
    # Parse viewBox
    try:
        coords = [float(x) for x in viewbox.split()]
        min_x, min_y, width, height = coords
        max_x = min_x + width
        max_y = min_y + height
    except:
        min_x, min_y, max_x, max_y = -500, -500, 500, 500
    
    # Generate grid lines
    grid_lines = []
    
    # Vertical lines
    x = min_x
    while x <= max_x:
        grid_lines.append(
            f'<line x1="{x}" y1="{min_y}" x2="{x}" y2="{max_y}" '
            f'stroke="{grid_color}" stroke-width="0.5" opacity="0.5"/>'
        )
        x += grid_size
    
    # Horizontal lines
    y = min_y
    while y <= max_y:
        grid_lines.append(
            f'<line x1="{min_x}" y1="{y}" x2="{max_x}" y2="{y}" '
            f'stroke="{grid_color}" stroke-width="0.5" opacity="0.5"/>'
        )
        y += grid_size
    
    # Add axes if requested
    axes_lines = []
    if show_axes:
        # X axis
        axes_lines.append(
            f'<line x1="{min_x}" y1="0" x2="{max_x}" y2="0" '
            f'stroke="{axes_color}" stroke-width="1" opacity="0.8"/>'
        )
        # Y axis
        axes_lines.append(
            f'<line x1="0" y1="{min_y}" x2="0" y2="{max_y}" '
            f'stroke="{axes_color}" stroke-width="1" opacity="0.8"/>'
        )
    
    # Insert grid before existing content
    grid_svg = '\n'.join(grid_lines + axes_lines)
    
    # Find the opening <svg> tag and insert grid after it
    svg_pattern = r'(<svg[^>]*>)'
    if re.search(svg_pattern, svg_text):
        svg_text = re.sub(
            svg_pattern,
            r'\1\n<g id="grid-overlay">' + grid_svg + '\n</g>',
            svg_text,
            count=1
        )
    else:
        # Fallback: prepend to content
        svg_text = f'<g id="grid-overlay">{grid_svg}</g>\n' + svg_text
    
    return svg_text


def add_zoom_pan_controls(svg_text: str, initial_zoom: float = 1.0, 
                          min_zoom: float = 0.1, max_zoom: float = 10.0) -> str:
    """
    Add zoom and pan controls to SVG using JavaScript.
    
    Args:
        svg_text: Original SVG content
        initial_zoom: Initial zoom level
        min_zoom: Minimum zoom level
        max_zoom: Maximum zoom level
    
    Returns:
        SVG text with zoom/pan controls
    """
    # Wrap SVG in a container with controls
    controls_html = f"""
    <div id="svg-container" style="position: relative; width: 100%; height: 100%;">
        <div style="position: absolute; top: 10px; right: 10px; z-index: 1000; background: white; padding: 5px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            <button onclick="zoomIn()" style="margin: 2px; padding: 5px 10px;">+</button>
            <button onclick="zoomOut()" style="margin: 2px; padding: 5px 10px;">-</button>
            <button onclick="resetView()" style="margin: 2px; padding: 5px 10px;">Reset</button>
        </div>
        <div id="svg-wrapper" style="transform-origin: 0 0; transform: scale({initial_zoom});">
            {svg_text}
        </div>
    </div>
    <script>
        let currentZoom = {initial_zoom};
        const minZoom = {min_zoom};
        const maxZoom = {max_zoom};
        let panStart = {{x: 0, y: 0}};
        let isPanning = false;
        
        function zoomIn() {{
            currentZoom = Math.min(currentZoom * 1.2, maxZoom);
            updateZoom();
        }}
        
        function zoomOut() {{
            currentZoom = Math.max(currentZoom / 1.2, minZoom);
            updateZoom();
        }}
        
        function resetView() {{
            currentZoom = {initial_zoom};
            const wrapper = document.getElementById('svg-wrapper');
            wrapper.style.transform = `scale(${{currentZoom}})`;
            wrapper.style.transformOrigin = '0 0';
        }}
        
        function updateZoom() {{
            const wrapper = document.getElementById('svg-wrapper');
            wrapper.style.transform = `scale(${{currentZoom}})`;
        }}
        
        // Mouse wheel zoom
        const container = document.getElementById('svg-container');
        if (container) {{
            container.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                currentZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom * delta));
                updateZoom();
            }});
            
            // Pan with mouse drag
            container.addEventListener('mousedown', (e) => {{
                if (e.button === 0) {{
                    isPanning = true;
                    panStart = {{x: e.clientX, y: e.clientY}};
                }}
            }});
            
            container.addEventListener('mousemove', (e) => {{
                if (isPanning) {{
                    const wrapper = document.getElementById('svg-wrapper');
                    const dx = e.clientX - panStart.x;
                    const dy = e.clientY - panStart.y;
                    const currentTransform = wrapper.style.transform || '';
                    // Simple pan implementation
                    wrapper.style.transform = currentTransform + ` translate(${{dx}}px, ${{dy}}px)`;
                    panStart = {{x: e.clientX, y: e.clientY}};
                }}
            }});
            
            container.addEventListener('mouseup', () => {{
                isPanning = false;
            }});
        }}
    </script>
    """
    
    return controls_html

