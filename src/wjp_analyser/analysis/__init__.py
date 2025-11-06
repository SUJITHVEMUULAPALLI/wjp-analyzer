"""
Analysis modules for DXF file processing and geometry analysis.

Exports are resolved lazily to avoid importing the heaviest modules unless needed.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

_MODULE_PATHS = {
    "dxf_analyzer": ".dxf_analyzer",
    "geometry_cleaner": ".geometry_cleaner",
    "topology": ".topology",
    "classification": ".classification",
    "quality_checks": ".quality_checks",
}

_MODULE_CACHE: Dict[str, Any] = {}


def _load_module(module_key: str):
    module = _MODULE_CACHE.get(module_key)
    if module is None:
        module = importlib.import_module(_MODULE_PATHS[module_key], __name__)
        _MODULE_CACHE[module_key] = module
    return module


def __getattr__(name: str) -> Any:
    for key in _MODULE_PATHS:
        module = _load_module(key)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    attributes = set(globals().keys())
    for key in _MODULE_PATHS:
        module = _load_module(key)
        attributes.update(dir(module))
    return sorted(attributes)


__all__ = list(_MODULE_PATHS.keys())
