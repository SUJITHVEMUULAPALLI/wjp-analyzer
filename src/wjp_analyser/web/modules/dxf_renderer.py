# modules/dxf_renderer.py
from __future__ import annotations

import io
from typing import Iterable, Optional, Set, Dict
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.addons.drawing.layout import Page


def render_svg(doc, layer_visibility: Optional[Dict[str, bool]] = None) -> str:
    """
    Render modelspace to SVG. Layers set False in `layer_visibility` are hidden.
    """
    ctx = RenderContext(doc)
    if layer_visibility:
        for layer_name, is_on in layer_visibility.items():
            try:
                ctx.layers.set_visibility(layer_name, bool(is_on))
            except Exception:
                # Ignore missing layers safely
                pass
    backend = SVGBackend()
    frontend = Frontend(ctx, backend)
    msp = doc.modelspace()
    frontend.draw_layout(msp)
    
    # Create a Page object from the layout for get_string()
    try:
        page = Page.from_dxf_layout(msp)
        svg_text = backend.get_string(page)
    except Exception:
        # Fallback: try with msp directly (older ezdxf versions)
        try:
            svg_text = backend.get_string(msp)
        except Exception:
            # Ultimate fallback: create minimal SVG
            svg_text = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    
    return svg_text



