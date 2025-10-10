import math
from pathlib import Path
from typing import Tuple

import ezdxf


def estimate_cost(dxf_path: str | Path, rate: float) -> Tuple[float, float]:
    """Estimate cutting cost based on DXF geometry.

    Parameters
    ----------
    dxf_path : str or Path
        Path to the DXF file.
    rate : float
        Cost rate in ₹ per centimeter of cut.

    Returns
    -------
    tuple
        ``(length_cm, cost)`` where ``length_cm`` is the total cut length in
        centimeters and ``cost`` is the estimated price in ₹. Returns
        ``(0.0, 0.0)`` if the file cannot be read or parsed.
    """
    path = Path(dxf_path)
    try:
        doc = ezdxf.readfile(path)
    except (OSError, IOError):
        # File not found or inaccessible
        return 0.0, 0.0
    except ezdxf.DXFError:
        # Parsing error
        return 0.0, 0.0

    msp = doc.modelspace()
    total_length = 0.0

    for entity in msp.query("LINE ARC SPLINE"):
        if entity.dxftype() == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            total_length += start.distance(end)
        elif entity.dxftype() == "ARC":
            angle_rad = math.radians((entity.dxf.end_angle - entity.dxf.start_angle) % 360)
            total_length += angle_rad * entity.dxf.radius
        elif entity.dxftype() == "SPLINE":
            try:
                spline = entity.construction_tool()
                points = spline.approximate(100)
            except ezdxf.DXFError:
                continue
            total_length += sum(p1.distance(p2) for p1, p2 in zip(points, points[1:]))

    length_cm = total_length / 10.0
    cost = length_cm * rate
    return length_cm, cost
