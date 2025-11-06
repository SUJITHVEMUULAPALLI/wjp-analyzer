import cv2
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from shapely.geometry import Polygon

def basic_image_to_dxf_converter():
    """Basic image to DXF converter function."""
    # === Input / Output ===
    input_image = "Tile_1.png"          # replace with your uploaded image
    output_dxf = "Tile_1_converted.dxf"
    DXF_SIZE = 1000.0                   # default canvas size in mm

    # --- Step 1: Read & Preprocess Image ---
    img = cv2.imread(input_image)
    if img is None:
        print(f"Warning: Could not load image {input_image}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

    # Morphological filtering to clean noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Preview original vs grayscale-thresholded
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Original")
    plt.subplot(1,2,2), plt.imshow(thresh, cmap="gray"), plt.title("Preprocessed + Filtered")
    plt.savefig("preprocessing_preview.png", dpi=150, bbox_inches='tight')
    plt.close()

    # --- Step 2: Find Contours with filtering ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only contours above a certain area threshold
    AREA_THRESHOLD = 500
    initial_count = len(contours)
    contours = [c for c in contours if cv2.contourArea(c) > AREA_THRESHOLD]

    print(f"Found {initial_count} contours initially")
    print(f"Kept {len(contours)} contours after filtering")

    # --- Step 3: Normalize to 1000mm x 1000mm ---
    # Flatten all points to find bounds
    all_points = np.vstack([c.reshape(-1, 2) for c in contours])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    scale_x = DXF_SIZE / (x_max - x_min)
    scale_y = DXF_SIZE / (y_max - y_min)
    scale = min(scale_x, scale_y)  # keep aspect ratio

    def scale_contour(cnt):
        pts = cnt.reshape(-1, 2)
        pts = (pts - [x_min, y_min]) * scale
        return pts

    scaled_contours = [scale_contour(c) for c in contours]

    # --- Step 4: Save to DXF ---
    doc = ezdxf.new()
    msp = doc.modelspace()
    for pts in scaled_contours:
        msp.add_lwpolyline(pts, close=True)

    doc.saveas(output_dxf)
    print(f"DXF saved: {output_dxf}")

    # --- Step 5: DXF Analysis ---
    polygons = [Polygon(pts) for pts in scaled_contours if len(pts) >= 3]
    outer_count = sum(1 for p in polygons if p.exterior.is_ccw)  # counter-clockwise
    inner_count = len(polygons) - outer_count
    print(f"Polys: {len(polygons)} | Outer: {outer_count} | Inner: {inner_count}")

    # --- Step 6: DXF Preview ---
    plt.figure(figsize=(6,6))
    for pts in scaled_contours:
        if len(pts) > 1:
            x, y = pts[:,0], pts[:,1]
            plt.plot(x, -y, 'k')  # invert Y for DXF-style view
    plt.title("DXF Preview (cutting paths)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("dxf_preview.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPreviews saved:")
    print(f"- preprocessing_preview.png")
    print(f"- dxf_preview.png")

# Only run if this file is executed directly
if __name__ == "__main__":
    basic_image_to_dxf_converter()