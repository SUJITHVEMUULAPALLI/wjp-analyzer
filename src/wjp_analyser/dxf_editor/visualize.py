from typing import Iterable, Optional, Dict, Any, List
import math

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def plot_entities(
    entities: Iterable, 
    selected_handles: Optional[Iterable] = None, 
    hidden_layers: Optional[Iterable] = None, 
    color_by_layer: bool = False,
    normalize_to_origin: bool = True,
    layer_classification: Optional[Dict[str, Any]] = None,
    warning_markers: Optional[Dict[str, List[str]]] = None
):
    """
    Plot entities with optional normalization and layer-based coloring.
    
    Args:
        entities: Iterable of DXF entities
        selected_handles: Set of selected entity handles
        hidden_layers: Set of hidden layer names
        color_by_layer: Use layer-based colors (OUTER=blue, INNER=gray)
        normalize_to_origin: Normalize coordinates to (0,0)
        layer_classification: Optional dict mapping entity handles to layer types
        warning_markers: Optional dict mapping entity handles to list of warning types
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    
    selected = set(selected_handles or [])
    hidden = set(hidden_layers or [])
    warnings = warning_markers or {}
    
    # Collect all points for normalization
    all_coords = []
    entity_data = []
    
    for e in entities:
        coords = []
        handle = None
        try:
            handle = e.dxf.handle
        except Exception:
            pass
        
        if e.dxftype() == "LINE":
            sx, sy = e.dxf.start[0], e.dxf.start[1]
            ex, ey = e.dxf.end[0], e.dxf.end[1]
            coords = [(sx, sy), (ex, ey)]
        elif e.dxftype() == "CIRCLE":
            c = e.dxf.center
            cx, cy = (c[0], c[1]) if isinstance(c, (list, tuple)) else (float(c.x), float(c.y))
            r = e.dxf.radius
            # Approximate circle with points for normalization
            for angle in range(0, 360, 15):
                rad = math.radians(angle)
                coords.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))
        elif e.dxftype() == "LWPOLYLINE":
            raw_pts = list(e.get_points())
            coords = [(p[0], p[1]) for p in raw_pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        
        if coords:
            all_coords.extend(coords)
            entity_data.append((e, coords, handle))
    
    # Calculate normalization offset
    min_x, min_y = (0.0, 0.0)
    if normalize_to_origin and all_coords:
        min_x = min(p[0] for p in all_coords)
        min_y = min(p[1] for p in all_coords)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Import color utilities if layer-based coloring is enabled
    if color_by_layer:
        try:
            from wjp_analyser.dxf_editor.preview_utils import get_layer_color, convert_hex_to_rgb
        except ImportError:
            color_by_layer = False
    
    # Render entities
    warning_centers = []  # Store warning locations for marker rendering
    for e, coords, handle in entity_data:
        layer_name = getattr(e.dxf, "layer", "0")
        is_selected = handle in selected if handle else False
        is_hidden = layer_name in hidden
        has_warning = handle in warnings if handle else False
        
        alpha = 0.2 if is_hidden else (0.8 if is_selected else 1.0)
        
        # Determine color
        if is_selected:
            color = "#FF0000"  # Red for selected
        elif has_warning:
            color = "#F44336"  # Red for warnings
        elif color_by_layer and layer_classification:
            # Use layer classification if available
            entity_layer = layer_classification.get(handle, "INNER")
            color_info = get_layer_color(entity_layer, is_selected=False)
            color = color_info["fill"]
        elif color_by_layer:
            # Fallback: assume OUTER if layer name suggests it, otherwise INNER
            if "OUTER" in layer_name.upper() or layer_name in ["0", "OUTLINE"]:
                color_info = get_layer_color("OUTER", is_selected=False)
                color = color_info["fill"]
            else:
                color_info = get_layer_color("INNER", is_selected=False)
                color = color_info["fill"]
        else:
            color = "gray" if color_by_layer else "black"
        
        # Normalize coordinates
        if normalize_to_origin:
            coords = [(x - min_x, y - min_y) for x, y in coords]
        
        # Render entity with warning highlighting
        line_width = 2.0 if has_warning else 1.2
        if e.dxftype() == "LINE":
            ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]], 
                   color=color, linewidth=line_width, alpha=alpha)
            if has_warning and handle:
                # Store midpoint for warning marker
                mid_x = (coords[0][0] + coords[1][0]) / 2
                mid_y = (coords[0][1] + coords[1][1]) / 2
                warning_centers.append((mid_x, mid_y, warnings[handle]))
        elif e.dxftype() == "CIRCLE":
            # Recalculate center from normalized coords
            if coords:
                center_x = sum(p[0] for p in coords) / len(coords)
                center_y = sum(p[1] for p in coords) / len(coords)
                # Estimate radius (average distance from center)
                if len(coords) > 1:
                    radius = sum(math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) 
                               for p in coords[:4]) / min(4, len(coords))
                    circ = plt.Circle((center_x, center_y), radius, color=color, 
                                    fill=False, linewidth=line_width, alpha=alpha)
                    ax.add_patch(circ)
                    if has_warning and handle:
                        warning_centers.append((center_x, center_y, warnings[handle]))
        elif e.dxftype() == "LWPOLYLINE":
            if coords:
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                # Check if closed
                if len(coords) > 2 and coords[0] == coords[-1]:
                    if has_warning:
                        # Highlight with warning color
                        ax.fill(xs, ys, color="#FFEBEE", alpha=alpha*0.2, edgecolor=color, linewidth=line_width)
                    else:
                        ax.fill(xs, ys, color=color, alpha=alpha*0.3, edgecolor=color, linewidth=line_width)
                else:
                    ax.plot(xs, ys, color=color, linewidth=line_width, alpha=alpha)
                
                if has_warning and handle:
                    # Store centroid for warning marker
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)
                    warning_centers.append((center_x, center_y, warnings[handle]))
    
    # Add warning markers
    for center_x, center_y, warning_types in warning_centers:
        # Draw warning marker
        ax.plot(center_x, center_y, 'ro', markersize=10, 
               markeredgecolor='#C62828', markeredgewidth=2, 
               markerfacecolor='#F44336', alpha=0.9, zorder=10)
        
        # Add annotation with warning text
        warning_text = "\n".join([w.replace("_", " ").title() for w in warning_types[:3]])  # Limit to 3 warnings
        if len(warning_types) > 3:
            warning_text += f"\n+{len(warning_types) - 3} more"
        ax.annotate(warning_text, (center_x, center_y), 
                   xytext=(12, 12), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', 
                            edgecolor='#F44336', linewidth=2),
                   fontsize=7, color='#C62828', zorder=12)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Build title with warning count
    title = "DXF Preview (Normalized to Origin)" if normalize_to_origin else "DXF Preview"
    if warning_centers:
        title += f" - {len(warning_centers)} Warning(s)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    
    return fig


