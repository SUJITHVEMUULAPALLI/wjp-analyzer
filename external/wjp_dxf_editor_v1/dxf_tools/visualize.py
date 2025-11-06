import matplotlib.pyplot as plt

def plot_entities(entities):
    fig, ax = plt.subplots()
    for e in entities:
        if e.dxftype() == "LINE":
            ax.plot([e.dxf.start[0], e.dxf.end[0]],
                    [e.dxf.start[1], e.dxf.end[1]], color="black", linewidth=0.8)
        elif e.dxftype() == "CIRCLE":
            circ = plt.Circle(e.dxf.center, e.dxf.radius, color="gray", fill=False)
            ax.add_patch(circ)
        elif e.dxftype() == "LWPOLYLINE":
            pts = list(e.get_points())
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="blue", linewidth=0.8)
    ax.axis("equal")
    ax.grid(True)
    return fig
