"""Shim for sample DXF generator used by tests and web UI.

Delegates to the implementation in tools/make_sample_dxf.py to avoid code
duplication while providing a stable import path `scripts.make_sample_dxf`.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_TOOLS = os.path.join(_ROOT, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from make_sample_dxf import make_sample  # type: ignore

__all__ = ["make_sample"]

