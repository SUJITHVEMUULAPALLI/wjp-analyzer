import math

def translate(entity, dx, dy):
    if entity.dxftype() == "LINE":
        entity.dxf.start = (entity.dxf.start[0] + dx, entity.dxf.start[1] + dy)
        entity.dxf.end = (entity.dxf.end[0] + dx, entity.dxf.end[1] + dy)
    elif entity.dxftype() == "CIRCLE":
        entity.dxf.center = (entity.dxf.center[0] + dx, entity.dxf.center[1] + dy)
    elif entity.dxftype() == "LWPOLYLINE":
        pts = [(p[0] + dx, p[1] + dy) for p in entity.get_points()]
        entity.set_points(pts)

def scale(entity, factor):
    if entity.dxftype() == "LINE":
        entity.dxf.start = (entity.dxf.start[0] * factor, entity.dxf.start[1] * factor)
        entity.dxf.end = (entity.dxf.end[0] * factor, entity.dxf.end[1] * factor)
    elif entity.dxftype() == "CIRCLE":
        entity.dxf.center = (entity.dxf.center[0] * factor, entity.dxf.center[1] * factor)
        entity.dxf.radius *= factor
    elif entity.dxftype() == "LWPOLYLINE":
        pts = [(p[0] * factor, p[1] * factor) for p in entity.get_points()]
        entity.set_points(pts)

def rotate(entity, angle_deg):
    angle = math.radians(angle_deg)
    def rot(x, y):
        return (x*math.cos(angle) - y*math.sin(angle), x*math.sin(angle) + y*math.cos(angle))
    if entity.dxftype() == "LINE":
        entity.dxf.start = rot(*entity.dxf.start)
        entity.dxf.end = rot(*entity.dxf.end)
    elif entity.dxftype() == "CIRCLE":
        entity.dxf.center = rot(*entity.dxf.center)
    elif entity.dxftype() == "LWPOLYLINE":
        pts = [rot(p[0], p[1]) for p in entity.get_points()]
        entity.set_points(pts)
