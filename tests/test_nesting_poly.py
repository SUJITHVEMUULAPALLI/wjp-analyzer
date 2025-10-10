from shapely.geometry import Polygon
from wjpanalyser.app.services.nesting.pack_poly import pack_polygons


def test_simple_poly_pack_no_overlap():
    l_shape = Polygon([(0,0),(60,0),(60,20),(20,20),(20,60),(0,60)])
    square = Polygon([(0,0),(30,0),(30,30),(0,30)])
    parts = [
        (l_shape.area, "L", l_shape),
        (square.area, "S", square),
    ]
    placed = pack_polygons(parts, 200, 200, gap=3.0)
    poses = [pose for pid, ang, pose in placed if pose is not None]
    assert len(poses) >= 2
    inter = poses[0].intersection(poses[1])
    assert inter.is_empty or inter.area == 0

