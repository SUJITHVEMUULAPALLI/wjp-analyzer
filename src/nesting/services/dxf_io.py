import ezdxf
from shapely.geometry import Polygon

def read_objects(dxf_path, selected_handles):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    objs = []
    for e in msp:
        if e.dxf.handle in selected_handles and e.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
            pts = [tuple(p[:2]) for p in e.get_points()]
            poly = Polygon(pts)
            objs.append({'handle': e.dxf.handle, 'layer': e.dxf.layer, 'polygon': poly, 'bbox': poly.bounds, 'area': poly.area})
    return objs

def write_nested(base_dxf, placements, out_path):
    doc = ezdxf.readfile(base_dxf)
    msp = doc.modelspace()
    for p in placements:
        blk = msp.add_circle(center=(p['x'], p['y']), radius=2.0)
        blk.dxf.layer = 'NESTED'
    doc.saveas(out_path)
