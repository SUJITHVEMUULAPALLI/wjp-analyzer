# Placeholders for Trim/Extend/Mirror/Offset/Join/Simplify
# Implement robust versions later using Shapely as needed.
import math

def mirror_entity(entity, axis="X"):
    if axis.upper() not in ("X","Y"):
        return
    def m(x,y):
        return (x, -y) if axis.upper()=="X" else (-x, y)
    if entity.dxftype() == "LINE":
        entity.dxf.start = m(*entity.dxf.start)
        entity.dxf.end = m(*entity.dxf.end)
    elif entity.dxftype() == "CIRCLE":
        entity.dxf.center = m(*entity.dxf.center)
    elif entity.dxftype() == "LWPOLYLINE":
        pts = [m(p[0], p[1]) for p in entity.get_points()]
        entity.set_points(pts)

def join_open_contours(entity_list, tol=0.5):
    # TODO: real path join; placeholder returns input
    return entity_list

def offset_preview_notch_width(value_mm=1.1):
    return float(value_mm)
