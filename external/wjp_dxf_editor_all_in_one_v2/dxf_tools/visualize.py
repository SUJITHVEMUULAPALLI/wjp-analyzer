import matplotlib.pyplot as plt

def _draw_line(ax, e, color, alpha=1.0, lw=1.0):
    ax.plot([e.dxf.start[0], e.dxf.end[0]], [e.dxf.start[1], e.dxf.end[1]],
            color=color, linewidth=lw, alpha=alpha)

def _draw_circle(ax, e, color, alpha=1.0, lw=1.0):
    circ = plt.Circle(e.dxf.center, e.dxf.radius, fill=False, color=color, alpha=alpha, linewidth=lw)
    ax.add_patch(circ)

def _draw_lwpoly(ax, e, color, alpha=1.0, lw=1.0):
    pts = list(e.get_points())
    if len(pts) >= 2:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha)

def plot_entities(entities, selected_handles=None, hidden_layers=None, color_by_layer=False):
    selected_handles = set(selected_handles or [])
    hidden_layers = set(hidden_layers or [])
    fig, ax = plt.subplots()
    for e in entities:
        layer = getattr(e.dxf, "layer", "0")
        if layer in hidden_layers:
            alpha = 0.2
        else:
            alpha = 1.0
        color = "black"
        if e.dxf.handle in selected_handles:
            color = "red"
            alpha = 1.0
        # Optional: simple color mapping by layer index (fallback black)
        if color_by_layer and color != "red":
            # ezdxf color is ACI index (1..255); matplotlib can accept a gray fallback
            color = "gray"
        if e.dxftype() == "LINE":
            _draw_line(ax, e, color, alpha)
        elif e.dxftype() == "CIRCLE":
            _draw_circle(ax, e, color, alpha)
        elif e.dxftype() == "LWPOLYLINE":
            _draw_lwpoly(ax, e, color, alpha)
    ax.axis("equal")
    ax.grid(True)
    return fig
