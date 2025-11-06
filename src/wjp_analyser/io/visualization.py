import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web server
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Dict, Tuple, Any
import os

def _color_for(cls: str) -> str:
    if cls == "outer":
        return "#1f77b4"  # blue
    if cls == "inner":
        return "#ff7f0e"  # orange
    return "#2ca02c"      # green fallback

def preview_png(
    path: str,
    polys: list[Polygon],
    classes: dict[int, str],
    order: list[int],
    kerf_preview: list[Polygon] | None = None,
    annotate_labels: bool = False,
    max_labels: int = 60,
) -> None:
    """Render a clean DXF preview.

    - Shows only DXF objects and contours
    - Outer contours in solid lines
    - Inner contours in dashed lines
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    n = len(polys)

    # Determine which indices to label
    label_every = 1
    if annotate_labels and n > max_labels:
        label_every = max(1, n // max_labels)

    for idx, p in enumerate(polys):
        x, y = p.exterior.xy
        cls = classes.get(idx, "")
        ax.plot(
            x,
            y,
            color=_color_for(cls),
            linewidth=1.0 if cls != "outer" else 1.5,
            linestyle="-" if cls == "outer" else "--",
            alpha=0.9 if cls == "outer" else 0.7,
        )
        # sparse labels for order
        if annotate_labels and (idx % label_every == 0):
            if p.exterior.coords:
                x0, y0 = p.exterior.coords[0]
                try:
                    k = order.index(idx) + 1
                    ax.text(
                        x0,
                        y0,
                        str(k),
                        fontsize=7,
                        color="#111",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, linewidth=0.0),
                    )
                except ValueError:
                    pass

    # kerf overlay
    if kerf_preview:
        for kp in kerf_preview:
            if not kp or kp.is_empty:
                continue
            pieces = []
            if hasattr(kp, "geoms"):
                pieces = [g for g in kp.geoms if hasattr(g, "exterior")]
            else:
                pieces = [kp]
            for pg in pieces:
                xk, yk = pg.exterior.xy
                ax.plot(xk, yk, color="#444", linewidth=0.6, alpha=0.6)

    # Set aspect ratio to equal for proper scaling
    ax.set_aspect("equal", adjustable="box")
    
    # Remove axes and frame for clean preview
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Ensure white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Save with minimal padding and high resolution
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    plt.close(fig)


def simulate_toolpath(sequence: List[Dict[str, Any]], save_path: str = None, 
                     show_order: bool = True, show_pierce_points: bool = True) -> str:
    """
    Simulate toolpath with cutting and rapid moves visualization.
    
    Args:
        sequence: List of optimized contours with sequence numbers and pierce points
        save_path: Path to save the simulation image
        show_order: Whether to show cutting order numbers
        show_pierce_points: Whether to highlight pierce points
    
    Returns:
        Path to saved simulation image
    """
    if not sequence:
        raise ValueError("No contours in sequence")
    
    # Create figure with subplots for before/after comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original order (before optimization)
    _plot_toolpath_sequence(ax1, sequence, original_order=True, 
                           show_order=show_order, show_pierce_points=show_pierce_points)
    ax1.set_title("Before Optimization", fontsize=14, fontweight='bold')
    
    # Plot optimized order (after optimization)
    _plot_toolpath_sequence(ax2, sequence, original_order=False,
                           show_order=show_order, show_pierce_points=show_pierce_points)
    ax2.set_title("After Optimization", fontsize=14, fontweight='bold')
    
    # Add overall title
    fig.suptitle("Toolpath Simulation", fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Cutting Moves'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Rapid Moves'),
        plt.Line2D([0], [0], marker='o', color='green', lw=0, markersize=8, label='Pierce Points')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Toolpath simulation saved to: {save_path}")
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return ""


def _plot_toolpath_sequence(ax, sequence: List[Dict[str, Any]], original_order: bool = False,
                           show_order: bool = True, show_pierce_points: bool = True):
    """Plot a single toolpath sequence."""
    
    # Sort contours for display
    if original_order:
        # Original order (by geometry bounds)
        display_sequence = sorted(sequence, key=lambda c: c['geometry'].bounds[0])
    else:
        # Optimized order (by sequence number)
        display_sequence = sorted(sequence, key=lambda c: c.get('sequence_number', 0))
    
    # Plot contours
    colors = plt.cm.Set3(np.linspace(0, 1, len(display_sequence)))
    
    for i, contour in enumerate(display_sequence):
        geometry = contour['geometry']
        color = colors[i]
        
        # Plot contour outline
        if hasattr(geometry, 'exterior'):
            x, y = geometry.exterior.xy
            ax.plot(x, y, color=color, linewidth=2, alpha=0.8)
            ax.fill(x, y, color=color, alpha=0.2)
        else:
            x, y = geometry.xy
            ax.plot(x, y, color=color, linewidth=2, alpha=0.8)
        
        # Add contour number
        centroid = geometry.centroid
        ax.text(centroid.x, centroid.y, str(i + 1), 
               ha='center', va='center', fontweight='bold', fontsize=10,
               bbox=dict(boxstyle="circle", facecolor='white', alpha=0.8))
    
    # Plot toolpath moves
    if len(display_sequence) > 1:
        _plot_toolpath_moves(ax, display_sequence, show_pierce_points)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')


def _plot_toolpath_moves(ax, sequence: List[Dict[str, Any]], show_pierce_points: bool = True):
    """Plot cutting and rapid moves between contours."""
    
    current_pos = Point(0, 0)  # Start at origin
    
    for i, contour in enumerate(sequence):
        geometry = contour['geometry']
        
        # Get pierce point
        if 'pierce_point' in contour:
            pierce_x, pierce_y = contour['pierce_point']
        else:
            pierce_x, pierce_y = geometry.centroid.x, geometry.centroid.y
        
        pierce_point = Point(pierce_x, pierce_y)
        
        # Rapid move to pierce point (dashed red line)
        ax.plot([current_pos.x, pierce_point.x], [current_pos.y, pierce_point.y], 
               color='red', linewidth=1, linestyle='--', alpha=0.7)
        
        # Pierce point marker
        if show_pierce_points:
            ax.plot(pierce_point.x, pierce_point.y, 'go', markersize=8, alpha=0.8)
            
            # Pierce point number
            ax.text(pierce_point.x, pierce_point.y + 5, f"P{i+1}", 
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
        # Cutting move around contour (solid blue line)
        if hasattr(geometry, 'exterior'):
            x, y = geometry.exterior.xy
            ax.plot(x, y, color='blue', linewidth=2, alpha=0.8)
        else:
            x, y = geometry.xy
            ax.plot(x, y, color='blue', linewidth=2, alpha=0.8)
        
        # Update current position to end of this contour
        current_pos = pierce_point


def create_toolpath_comparison(original_sequence: List[Dict[str, Any]], 
                              optimized_sequence: List[Dict[str, Any]], 
                              save_path: str = None) -> str:
    """
    Create side-by-side comparison of original vs optimized toolpath.
    
    Args:
        original_sequence: Original contour sequence
        optimized_sequence: Optimized contour sequence
        save_path: Path to save comparison image
    
    Returns:
        Path to saved comparison image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original sequence
    _plot_toolpath_sequence(ax1, original_sequence, show_order=True, show_pierce_points=True)
    ax1.set_title("Original Toolpath", fontsize=14, fontweight='bold')
    
    # Plot optimized sequence
    _plot_toolpath_sequence(ax2, optimized_sequence, show_order=True, show_pierce_points=True)
    ax2.set_title("Optimized Toolpath", fontsize=14, fontweight='bold')
    
    # Add overall title
    fig.suptitle("Toolpath Optimization Comparison", fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Cutting Moves'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Rapid Moves'),
        plt.Line2D([0], [0], marker='o', color='green', lw=0, markersize=8, label='Pierce Points')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Toolpath comparison saved to: {save_path}")
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return ""
