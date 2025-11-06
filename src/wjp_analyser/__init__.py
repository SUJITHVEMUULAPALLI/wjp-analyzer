"""
WJP ANALYSER - Waterjet DXF Analysis Tool

A comprehensive tool for analyzing DXF files for waterjet cutting,
with AI-powered manufacturing insights and optimization recommendations.
"""

__version__ = "2.0.0"
__author__ = "WJP ANALYSER Team"
__description__ = "Waterjet DXF Analysis Tool with AI Integration"

# Compatibility shims (loaded early)
try:
    import ezdxf
    try:
        from ezdxf.entities import Text as _EzdxfText
    except Exception:
        _EzdxfText = None
    # Provide Text.set_pos((x,y)) for tests that expect older API
    if _EzdxfText is not None and not hasattr(_EzdxfText, "set_pos"):
        def _set_pos(self, location, align=None):  # type: ignore
            try:
                # Newer ezdxf provides set_placement
                set_placement = getattr(self, "set_placement", None)
                if callable(set_placement):
                    set_placement(location, align)
                else:
                    # Fallback: set insertion point directly
                    self.dxf.insert = (float(location[0]), float(location[1]), 0.0)
            except Exception:
                # Best-effort fallback
                try:
                    self.dxf.insert = (float(location[0]), float(location[1]), 0.0)
                except Exception:
                    pass
            return self
        setattr(_EzdxfText, "set_pos", _set_pos)
except Exception:
    pass

# Lightweight package init: avoid importing heavy subpackages at import time to
# prevent circular imports and speed up tools like Streamlit. Import modules
# directly where needed, e.g. `from wjp_analyser.analysis import dxf_analyzer`.

__all__ = [
    "__version__",
    "__author__",
    "__description__",
]
