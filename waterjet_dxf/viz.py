import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web server
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import List, Dict

def preview_png(path: str, polys: list[Polygon], classes: dict[int,str], order: list[int], kerf_preview: list[Polygon] | None = None):
    plt.figure()
    for i,p in enumerate(polys):
        x,y = p.exterior.xy
        style = {"linewidth":1.5}
        if classes.get(i)=="outer":
            style["linestyle"] = "-"
        else:
            style["linestyle"] = "--"
        plt.plot(x,y, **style)
        # order index label
        if p.exterior.coords:
            x0,y0 = p.exterior.coords[0]
            try:
                k = order.index(i)+1
                plt.text(x0, y0, str(k), fontsize=8)
            except ValueError:
                pass
    if kerf_preview:
        for kp in kerf_preview:
            if kp and not kp.is_empty:
                xk, yk = kp.exterior.xy
                plt.plot(xk, yk, linewidth=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
