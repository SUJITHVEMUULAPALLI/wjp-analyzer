import matplotlib.pyplot as plt
from shapely.geometry import box
from descartes import PolygonPatch

def render(result, frame_w, frame_h):
    fig, ax = plt.subplots(figsize=(6,6))
    frame_poly = box(0, 0, frame_w, frame_h)
    ax.add_patch(PolygonPatch(frame_poly, fill=False, edgecolor='black', linewidth=1))
    for p in result['placed']:
        ax.plot(p['x'], p['y'], 'ro', markersize=3)
    ax.set_xlim(0, frame_w)
    ax.set_ylim(0, frame_h)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    img_path = '/mnt/data/nesting_preview.png'
    fig.savefig(img_path)
    plt.close(fig)
    return img_path
