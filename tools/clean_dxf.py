import ezdxf
from shapely.geometry import LineString

# === Input and Output files ===
input_file = "data/samples/dxf/medallion_sample.dxf"   # using existing file with content
output_file = "medallion_sample_cleaned.dxf"

# === Parameters ===
MIN_LENGTH = 5.0  # threshold: remove anything shorter than this (drawing units)

print(f"Opening DXF: {input_file}")
doc = ezdxf.readfile(input_file)
msp = doc.modelspace()

# Collect geometries
geoms = []
for e in msp.query("LWPOLYLINE LINE POLYLINE CIRCLE"):
    try:
        if e.dxftype() == "LINE":
            line = LineString([(e.dxf.start[0], e.dxf.start[1]),
                               (e.dxf.end[0], e.dxf.end[1])])
            if line.length >= MIN_LENGTH:
                geoms.append(line)
        elif e.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            points = [(p[0], p[1]) for p in e.get_points()]
            poly = LineString(points)
            if poly.length >= MIN_LENGTH:
                geoms.append(poly)
        elif e.dxftype() == "CIRCLE":
            # Convert circle to polygon approximation
            center = e.dxf.center
            radius = e.dxf.radius
            if radius * 2 * 3.14159 >= MIN_LENGTH:  # circumference check
                # Create a polygon approximation of the circle
                import math
                num_points = max(8, int(radius * 4))  # More points for larger circles
                points = []
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append((x, y))
                circle_poly = LineString(points)
                geoms.append(circle_poly)
    except Exception as ex:
        print("Skipped entity:", ex)

print(f"Kept {len(geoms)} geometries after filtering")

# === Create new DXF ===
new_doc = ezdxf.new()
new_msp = new_doc.modelspace()

for g in geoms:
    coords = list(g.coords)
    new_msp.add_lwpolyline(coords, close=True)

new_doc.saveas(output_file)
print(f"Saved cleaned DXF as: {output_file}")
