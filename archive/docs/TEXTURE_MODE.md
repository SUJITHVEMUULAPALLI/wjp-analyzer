Image-to-DXF Texture Mode
=========================

This module provides texture-aware conversion of images (photo/texture scan/material sample) into layered DXF suitable for waterjet workflows.

Where to find it (UI)
- Launch the Streamlit UI (see `run_web_ui.py`) and open the page: Image -> DXF.
- Crop and confirm the preview.
- In the sidebar, open "Texture Mode" and configure parameters.
- Click "Run Texture Vectorizer" to generate a layered DXF and a preview.

Layers
- `LAYER_EDGES`: Outlines from high-edge-density regions
- `LAYER_STIPPLE`: Blue-noise stipple dots (CIRCLE entities)
- `LAYER_HATCH`: Directional hatch stripes (polylines)
- `LAYER_CONTOUR_*`: Smooth-tone contour bands across multiple threshold levels

Parameters
- DXF canvas (mm): target size for the 1000x1000 working canvas scaling
- Tile size (px): size of tiles for texture classification (32 is a good start)
- Stipple: dot spacing and radius (mm)
- Hatch: spacing (mm), angle (deg), optional cross-hatching
- Contour bands: number of evenly spaced intensity levels (2–12)
- Min feature size (mm): removes tiny closed shapes and short open paths; drops dots below diameter
- Simplify tolerance (mm): applies Douglas–Peucker simplification to polylines
- Kerf compensation: choose outward/inward offset by ±kerf/2; set kerf width (mm)

Notes
- Potrace is used for vectorization when installed; the pipeline falls back to OpenCV contours when unavailable.
- Generated DXF is scaled to real-world millimeters and organized into layers for downstream processing.

- Preserve ARC entities: keep native arcs intact during cleanup
- Advanced per-layer cleanup (overrides 0 = use global):
  - Edges/Hatch/Contour simplify tolerance (mm)
  - Edges/Hatch/Contour min area (mm^2)

