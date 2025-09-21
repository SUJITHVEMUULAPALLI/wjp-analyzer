import ezdxf, sys, math

def make_sample(path: str):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # Outer square 100x100
    msp.add_lwpolyline([(0,0),(100,0),(100,100),(0,100),(0,0)], format="xy")
    # Inner circle (cutout)
    msp.add_circle(center=(50,50), radius=20)
    # Inlay-like ring: two concentric circles (groove visualization)
    msp.add_circle(center=(50,50), radius=35)
    msp.add_circle(center=(50,50), radius=32)
    # Four small holes
    for dx,dy in [(25,25),(75,25),(75,75),(25,75)]:
        msp.add_circle(center=(dx,dy), radius=3)
    doc.saveas(path)
    print("Wrote", path)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv)>1 else "samples/medallion_sample.dxf"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    make_sample(out)
