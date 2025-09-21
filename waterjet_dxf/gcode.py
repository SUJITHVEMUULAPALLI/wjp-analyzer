from shapely.geometry import Polygon
from typing import List, Dict, TextIO

HEADER = """(Program: waterjet)
G90
G21
"""

FOOTER = """M30
"""

def _emit_path(f: TextIO, poly: Polygon, feed: float, m_on: str, m_off: str, pierce_ms: int):
    ext = list(poly.exterior.coords)
    if not ext:
        return
    x0,y0 = ext[0]
    f.write(f"G0 X{round(x0,3)} Y{round(y0,3)}\n")
    f.write(f"{m_on}\n")
    f.write(f"G4 P{pierce_ms}\n")
    for (x,y) in ext[1:]:
        f.write(f"G1 X{round(x,3)} Y{round(y,3)} F{feed}\n")
    f.write(f"{m_off}\n")

def write_gcode(path: str, polys: list[Polygon], order: list[int], feed: float=1200.0, m_on="M62", m_off="M63", pierce_ms: int=500):
    with open(path, "w") as f:
        f.write(HEADER)
        for idx in order:
            _emit_path(f, polys[idx], feed, m_on, m_off, pierce_ms)
        f.write(FOOTER)
