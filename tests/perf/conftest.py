"""Performance test fixtures."""
from __future__ import annotations

import pytest
import ezdxf
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def large_dxf_10k_entities(tmp_path_factory):
    """Create a DXF file with 10,000+ entities for performance testing."""
    path = tmp_path_factory.mktemp("perf_data") / "large_10k.dxf"
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    # Add 10,000 entities (mix of lines, circles, polylines)
    for i in range(10000):
        x = (i % 100) * 10.0
        y = (i // 100) * 10.0
        
        if i % 3 == 0:
            msp.add_line((x, y), (x + 5, y + 5))
        elif i % 3 == 1:
            msp.add_circle((x, y), 2.0)
        else:
            points = [(x, y), (x + 3, y), (x + 3, y + 3), (x, y + 3)]
            msp.add_lwpolyline(points, close=True)
    
    doc.saveas(path)
    return str(path)


@pytest.fixture(scope="session")
def very_large_dxf_50k_entities(tmp_path_factory):
    """Create a DXF file with 50,000+ entities for stress testing."""
    path = tmp_path_factory.mktemp("perf_data") / "very_large_50k.dxf"
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    # Add 50,000 entities
    for i in range(50000):
        x = (i % 500) * 10.0
        y = (i // 500) * 10.0
        
        if i % 4 == 0:
            msp.add_line((x, y), (x + 5, y + 5))
        elif i % 4 == 1:
            msp.add_circle((x, y), 2.0)
        elif i % 4 == 2:
            points = [(x, y), (x + 3, y), (x + 3, y + 3), (x, y + 3)]
            msp.add_lwpolyline(points, close=True)
        else:
            msp.add_text(f"TEXT{i}").set_pos((x, y))
    
    doc.saveas(path)
    return str(path)

