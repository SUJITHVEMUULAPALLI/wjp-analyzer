from shapely.geometry import Polygon
from wjpanalyser.app.services.geometry import offset_polygon


def test_buffer_in_out_area_change():
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    grown = offset_polygon(square, 1.0)
    shrunk = offset_polygon(square, -1.0)
    assert grown.area > square.area
    assert shrunk.area < square.area

