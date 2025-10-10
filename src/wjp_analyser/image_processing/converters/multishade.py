import cv2
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from shapely.geometry import Polygon

# === Input / Output ===
input_image = "Tile_1.png"          # replace with your uploaded tile image
output_dxf = "Tile_1_multishade.dxf"
DXF_SIZE = 1000.0                   # default canvas size in mm

# --- Step 1: Read Image ---
img = cv2.imread(input_image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 2: K-means Clustering into 3 shades ---
Z = gray.reshape((-1,1))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3  # background, gold, white
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

labels_2d = labels.reshape(gray.shape)
segmented = centers[labels.flatten()].reshape(gray.shape).astype(np.uint8)

# Sort cluster centers (darkest -> brightest) and assign names
sorted_idx = np.argsort(centers.flatten())
layer_names = ["BACKGROUND", "GOLD", "WHITE"]

# --- Step 3: Generate Masks for each cluster ---
masks = []
for rank, idx in enumerate(sorted_idx):
    mask = np.uint8(labels_2d == idx) * 255
    masks.append((layer_names[rank], mask))

# Show segmentation preview
plt.figure(figsize=(12,4))
for i, (name, mask) in enumerate(masks):
    plt.subplot(1,K,i+1)
    plt.imshow(mask, cmap="gray")
    plt.title(name)
plt.savefig("segmentation_preview.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Step 4: Extract contours per mask ---
contour_data = []
for layer_name, mask in masks:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter small areas
    AREA_THRESHOLD = 500
    contours = [c for c in contours if cv2.contourArea(c) > AREA_THRESHOLD]
    contour_data.append((layer_name, contours))

print("Contour extraction complete.")

# --- Step 5: Normalize all contours to 1000mm x 1000mm ---
all_points = np.vstack([c.reshape(-1,2) for _, cs in contour_data for c in cs])
x_min, y_min = np.min(all_points, axis=0)
x_max, y_max = np.max(all_points, axis=0)

scale_x = DXF_SIZE / (x_max - x_min)
scale_y = DXF_SIZE / (y_max - y_min)
scale = min(scale_x, scale_y)

def scale_contour(cnt):
    pts = cnt.reshape(-1, 2)
    pts = (pts - [x_min, y_min]) * scale
    return pts

scaled_data = []
for layer_name, contours in contour_data:
    scaled_contours = [scale_contour(c) for c in contours]
    scaled_data.append((layer_name, scaled_contours))

# --- Step 6: Save DXF with named layers ---
doc = ezdxf.new()
msp = doc.modelspace()

for layer_name, contours in scaled_data:
    if layer_name not in doc.layers:
        doc.layers.add(name=layer_name)
    for pts in contours:
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer_name})

doc.saveas(output_dxf)
print(f"DXF saved: {output_dxf}")

# --- Step 7: Analyze & Preview ---
total_polys = 0
for layer_name, contours in scaled_data:
    total_polys += len(contours)
    print(f"{layer_name}: {len(contours)} contours")

print(f"Total polygons: {total_polys}")

# DXF-style preview
plt.figure(figsize=(6,6))
colors = {"BACKGROUND":"k","GOLD":"orange","WHITE":"gray"}
for layer_name, contours in scaled_data:
    for pts in contours:
        if len(pts) > 1:
            x, y = pts[:,0], pts[:,1]
            plt.plot(x, -y, color=colors[layer_name], label=layer_name)
plt.title("DXF Preview (multi-shade layers)")
plt.gca().set_aspect("equal", adjustable="box")
plt.savefig("multishade_dxf_preview.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPreviews saved:")
print(f"- segmentation_preview.png")
print(f"- multishade_dxf_preview.png")
