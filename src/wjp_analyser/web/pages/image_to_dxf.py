from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_WS_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

try:
    from wjp_analyser.image_processing.converters.opencv_converter import (
        OpenCVImageToDXFConverter,
    )
except Exception:
    OpenCVImageToDXFConverter = None

try:
    from wjp_analyser.image_processing.texture_pipeline import (
        PreprocessParams as TexPreParams,
        TextureClassifyParams as TexClassParams,
        TextureVectorizeParams as TexVecParams,
        generate_texture_dxf,
    )
except Exception:
    TexPreParams = None  # type: ignore
    TexClassParams = None  # type: ignore
    TexVecParams = None  # type: ignore
    generate_texture_dxf = None  # type: ignore

# Import new interactive editing components
try:
    from wjp_analyser.image_processing.object_detector import ObjectDetector, DetectionParams
    from wjp_analyser.image_processing.interactive_editor import InteractiveEditor, render_interactive_editor
    from wjp_analyser.image_processing.preview_renderer import PreviewRenderer, render_final_preview_interface
except Exception as e:
    st.error(f"Failed to import interactive editing components: {e}")
    ObjectDetector = None
    DetectionParams = None
    InteractiveEditor = None
    render_interactive_editor = None
    PreviewRenderer = None
    render_final_preview_interface = None


st.title("Image -> DXF")

from wjp_analyser.web._components import render_ai_status
with st.sidebar.expander("AI Status", expanded=False):
    try:
        render_ai_status(compact=True)
    except Exception:
        st.caption("Status unavailable")

# Reset gating if a new image is chosen
def _reset_crop_state():
    for k in [
        "_img2dxf_confirmed",
        "_img2dxf_preview_path",
        "_img2dxf_cropped_path",
        "_img2dxf_crop_left",
        "_img2dxf_crop_right",
        "_img2dxf_crop_top",
        "_img2dxf_crop_bottom",
    ]:
        st.session_state.pop(k, None)


tab_upload, tab_path = st.tabs(["Upload", "Use local path"]) 

image_path: str | None = None
image_path: str | None = None
# Preload image from Designer page if available
if not image_path:
    try:
        if st.session_state.get("designer_image_path"):
            image_path = st.session_state.get("designer_image_path")
            st.info(f"Loaded image from Designer: {image_path}")
    except Exception:
        pass
with tab_upload:
    img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "tiff"], on_change=_reset_crop_state)
    if img is not None:
        # Use absolute path for output directory
        project_root = Path(__file__).parent.parent.parent.parent
        base_dir = project_root / "output" / "image_to_dxf"
        base_dir.mkdir(parents=True, exist_ok=True)
        st.session_state["_wjp_img_base"] = str(base_dir)
        image_path = str(base_dir / Path(img.name).name)
        Path(image_path).write_bytes(img.getvalue())
        st.success(f"Saved to {image_path}")

with tab_path:
    p = st.text_input("Image file path", value=image_path or "")
    if p:
        if p != st.session_state.get("_img2dxf_image_path"):
            _reset_crop_state()
        st.session_state["_img2dxf_image_path"] = p
        image_path = p

# Step 1: show original preview and allow boundary adjustment (crop)
if image_path:
    st.subheader("Original Preview")
    try:
        st.image(image_path, use_column_width=True)
    except Exception:
        st.warning("Could not render image preview; proceeding anyway.")

    # Transform & Orientation (Rotate / Angle-of-view tilt)
    with st.expander("Transform & Orientation", expanded=False):
        st.caption("Apply orientation fixes before cropping and conversion. Saves a new image copy when applied.")

        import cv2  # local import to avoid global dependency at module load

        def _load_bgr(path: str):
            return cv2.imread(path, cv2.IMREAD_COLOR)

        def _rotate_image(bgr, angle_deg: float):
            h, w = bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        def _tilt_image(bgr, tilt_x: float, tilt_y: float):
            h, w = bgr.shape[:2]
            dx = int(w * tilt_x * 0.25)
            dy = int(h * tilt_y * 0.25)
            src = np.float32([[0,0],[w,0],[w,h],[0,h]])
            dst = np.float32([[0+dx,0+dy],[w-dx,0+dy],[w-dx,h-dy],[0+dx,h-dy]])
            P = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(bgr, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        c1, c2, c3 = st.columns(3)
        with c1:
            angle = st.slider("Rotation (°)", -180, 180, 0, 1)
        with c2:
            tilt_x = st.slider("Tilt X", -0.5, 0.5, 0.0, 0.01)
        with c3:
            tilt_y = st.slider("Tilt Y", -0.5, 0.5, 0.0, 0.01)

        col_t1, col_t2 = st.columns(2)
        preview_transform = col_t1.button("Preview Transform")
        apply_transform = col_t2.button("Save as New Working Image")

        if preview_transform or apply_transform:
            try:
                bgr = _load_bgr(image_path)
                if bgr is None:
                    st.error("Failed to load image for transform")
                else:
                    out = _rotate_image(bgr, angle) if angle else bgr
                    if abs(tilt_x) > 1e-6 or abs(tilt_y) > 1e-6:
                        out = _tilt_image(out, float(tilt_x), float(tilt_y))
                    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption="Transformed Preview", use_column_width=True)

                    if apply_transform:
                        project_root = Path(__file__).parent.parent.parent.parent
                        base_dir = project_root / "output" / "image_to_dxf"
                        base_dir.mkdir(parents=True, exist_ok=True)
                        suffix = f"rot{int(angle)}_tx{int(tilt_x*100)}_ty{int(tilt_y*100)}"
                        save_path = base_dir / f"{Path(image_path).stem}_{suffix}.png"
                        cv2.imwrite(str(save_path), out)
                        # Set as new working image and reset downstream state
                        st.session_state["_img2dxf_image_path"] = str(save_path)
                        image_path = str(save_path)
                        _reset_crop_state()
                        st.success(f"Saved and set as working image: {save_path}")
            except Exception as e:
                st.error(f"Transform failed: {e}")

    with st.sidebar.expander("Adjust Boundaries (Crop)", expanded=True):
        left = st.slider("Left trim (%)", 0, 50, int(st.session_state.get("_img2dxf_crop_left", 0)))
        right = st.slider("Right trim (%)", 0, 50, int(st.session_state.get("_img2dxf_crop_right", 0)))
        top = st.slider("Top trim (%)", 0, 50, int(st.session_state.get("_img2dxf_crop_top", 0)))
        bottom = st.slider("Bottom trim (%)", 0, 50, int(st.session_state.get("_img2dxf_crop_bottom", 0)))

        colp1, colp2 = st.columns(2)
        if colp1.button("Preview Crop"):
            try:
                im = Image.open(image_path)
                w, h = im.size
                lpx = int(w * (left / 100.0))
                rpx = int(w - w * (right / 100.0))
                tpx = int(h * (top / 100.0))
                bpx = int(h - h * (bottom / 100.0))
                # clamp
                lpx = max(0, min(lpx, w - 1))
                rpx = max(lpx + 1, min(rpx, w))
                tpx = max(0, min(tpx, h - 1))
                bpx = max(tpx + 1, min(bpx, h))
                crop = im.crop((lpx, tpx, rpx, bpx))
                # Use absolute path for output directory
                project_root = Path(__file__).parent.parent.parent.parent
                base_dir = project_root / "output" / "image_to_dxf"
                base_dir.mkdir(parents=True, exist_ok=True)
                preview_path = base_dir / f"{Path(image_path).stem}_crop_preview.png"
                crop.save(preview_path)
                st.session_state["_img2dxf_preview_path"] = str(preview_path)
                st.session_state["_img2dxf_crop_left"] = left
                st.session_state["_img2dxf_crop_right"] = right
                st.session_state["_img2dxf_crop_top"] = top
                st.session_state["_img2dxf_crop_bottom"] = bottom
                st.success("Cropped preview generated below.")
            except Exception as e:
                st.error(f"Cropping failed: {e}")

        if colp2.button("Confirm Cropped Preview", disabled=not bool(st.session_state.get("_img2dxf_preview_path"))):
            st.session_state["_img2dxf_confirmed"] = True
            st.session_state["_img2dxf_cropped_path"] = st.session_state.get("_img2dxf_preview_path")
            st.success("Cropped preview confirmed. You can proceed to conversion.")

    # Show cropped preview if available
    if st.session_state.get("_img2dxf_preview_path"):
        st.subheader("Cropped Preview")
        st.image(st.session_state.get("_img2dxf_preview_path"), use_column_width=True)

# Gate the rest of the pipeline on confirmation
confirmed = bool(st.session_state.get("_img2dxf_confirmed")) and bool(st.session_state.get("_img2dxf_cropped_path"))
if not confirmed:
    st.info("Adjust boundaries and confirm the cropped preview to continue.")

if confirmed:
    # Initialize interactive editor if components are available
    if InteractiveEditor is not None:
        if "_interactive_editor" not in st.session_state:
            st.session_state["_interactive_editor"] = InteractiveEditor()
        
        editor = st.session_state["_interactive_editor"]
        
        # Load image into editor
        input_for_pipeline = st.session_state.get("_img2dxf_cropped_path") or image_path
        if input_for_pipeline and Path(input_for_pipeline).exists():
            editor.load_image(input_for_pipeline)
    
    with st.sidebar.expander("Preprocessing", expanded=True):
        threshold_type = st.selectbox("Threshold type", ["global", "adaptive"], index=0)
        binary_threshold = st.slider("Global threshold", min_value=0, max_value=255, value=180)
        adaptive_block = st.slider("Adaptive block size", min_value=3, max_value=101, value=21, step=2)
        adaptive_C = st.slider("Adaptive C", min_value=-20, max_value=20, value=5)
        gaussian_blur = st.number_input("Gaussian blur ksize (0=off)", min_value=0, value=5, step=1)
        use_canny = st.checkbox("Enhance edges (Canny)", value=False)
        morph_op = st.selectbox("Morphology", ["open", "close", "none"], index=0)
        morph_ksize = st.number_input("Morph kernel size", min_value=1, value=3, step=1)
        morph_iters = st.number_input("Morph iterations", min_value=1, value=1, step=1)
        invert = st.checkbox("Invert colors for Potrace (black foreground)", value=False, help="Potrace traces black shapes by default. Enable if your design is white on black.")
        st.markdown("---")
        skeletonize_on = st.checkbox("🦴 Line Thinning (Skeletonize)", value=False, help="Reduce thick strokes to centerlines before tracing")
        simplify_before = st.checkbox("🧹 Simplify Before Convert", value=True, help="Clean stray specks and enforce binary edges")
        skeleton_thresh = st.slider("Skeletonize threshold", min_value=100, max_value=240, value=200, step=5)

    with st.sidebar.expander("Vectorization", expanded=True):
        route = st.selectbox("Route", ["Potrace DXF (arcs)", "SVG then DXF"], index=0)
        dxf_size = st.number_input("Target size (mm)", min_value=100.0, value=1000.0, step=50.0)
        simplify_tol = st.number_input("Simplify tolerance (SVG route)", min_value=0.0, value=0.0, step=0.1)
        post_tol = st.number_input("⚙️ Tolerance (mm) — Post DXF simplify", min_value=0.0, value=0.5, step=0.1)
        potrace_path = st.text_input("Potrace executable (optional)", value=st.session_state.get("_potrace_exe", ""), help="Leave blank to use PATH/WSL. Provide full path like C:\\Tools\\potrace\\potrace.exe")
        colpa, colpb, colpc = st.columns(3)
        potrace_turd = colpa.number_input("Speckle cleanup (turdsize)", min_value=0, value=2, step=1, help="Remove tiny speckles before tracing")
        potrace_alpha = colpb.number_input("Corner smoothness (alphamax)", min_value=0.0, value=1.0, step=0.1, help="Lower = sharper corners, higher = smoother")
        potrace_opt = colpc.number_input("Curve opt tol", min_value=0.0, value=0.2, step=0.05, help="Arc/Bezier optimization tolerance")

    if st.button("Convert", type="primary", disabled=not confirmed):
        # Use absolute path for output directory
        project_root = Path(__file__).parent.parent.parent.parent
        base_dir = project_root / "output" / "image_to_dxf"
        base_dir.mkdir(parents=True, exist_ok=True)
        input_for_pipeline = st.session_state.get("_img2dxf_cropped_path") or image_path
        try:
            # Optional skeletonization pre-pass
            if skeletonize_on and input_for_pipeline:
                try:
                    from wjp_analyser.skeletonize_preprocessor import centerline_preprocess_for_vectorization
                    input_for_pipeline = centerline_preprocess_for_vectorization(
                        input_for_pipeline,
                        out_dir=str(base_dir),
                        invert=not invert,  # if we invert for potrace later, keep foreground handling consistent
                        thresh=int(skeleton_thresh),
                        open_kernel=(3, 3),
                    )
                    st.info("Applied skeletonization (centerline thinning)")
                except Exception as e:
                    st.warning(f"Skeletonization skipped: {e}")

            # Prefer Potrace pipeline if available
            from wjp_analyser.image_processing.potrace_pipeline import (
                preprocess_and_vectorize,
                simplify_dxf_inplace,
                compute_dxf_complexity,
            )
            route_key = "potrace_dxf" if route.startswith("Potrace") else "svg_then_dxf"
            dxf_path, preview_path, svg_path = preprocess_and_vectorize(
                str(input_for_pipeline),
                out_dir=base_dir,
                target_size_mm=float(dxf_size),
                threshold_type=str(threshold_type),
                threshold_value=int(binary_threshold),
                adaptive_block_size=int(adaptive_block),
                adaptive_C=int(adaptive_C),
                gaussian_blur_ksize=int(gaussian_blur),
                use_canny=bool(use_canny),
                morph_op=str(morph_op),
                morph_ksize=int(morph_ksize),
                morph_iters=int(morph_iters),
                output_route=route_key,
                simplify_tolerance=float(simplify_tol),
                invert=bool(invert),
                potrace_turdsize=int(potrace_turd),
                potrace_alphamax=float(potrace_alpha),
                potrace_opttolerance=float(potrace_opt),
                potrace_exe=str(potrace_path).strip() or None,
            )
            if dxf_path is None:
                raise RuntimeError("Potrace pipeline did not produce a DXF.")
            st.success("Conversion complete (Potrace)")
            st.image(str(preview_path), caption="Preprocessed binary", use_column_width=True)
            # Post-DXF simplification
            if post_tol > 0:
                try:
                    ok = simplify_dxf_inplace(Path(dxf_path), float(post_tol))
                    if ok:
                        st.info(f"Applied post-DXF simplification (tol={post_tol} mm)")
                except Exception as e:
                    st.warning(f"Post simplification failed: {e}")

            # Complexity report
            try:
                n_entities, n_nodes, total_len = compute_dxf_complexity(Path(dxf_path))
                with st.expander("📊 Complexity Report", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entities", n_entities)
                    c2.metric("Nodes", n_nodes)
                    c3.metric("Total Length (mm)", f"{total_len:.1f}")
            except Exception as e:
                st.caption(f"Complexity report unavailable: {e}")

        except Exception as e:
            # Fallback: OpenCV converter
            if OpenCVImageToDXFConverter is None:
                st.error(f"Conversion failed and OpenCV fallback unavailable: {e}")
            else:
                st.info(f"Potrace pipeline unavailable or failed ({e}). Falling back to OpenCV converter.")
                preview_path = base_dir / f"{Path(input_for_pipeline).stem}_preview.png"
                dxf_path = base_dir / f"{Path(input_for_pipeline).stem}_converted.dxf"
                try:
                    converter = OpenCVImageToDXFConverter(
                        binary_threshold=int(binary_threshold),
                        min_area=500,
                        dxf_size=float(dxf_size),
                    )
                    converter.convert_image_to_dxf(
                        input_image=str(input_for_pipeline),
                        output_dxf=str(dxf_path),
                        preview_output=str(preview_path),
                    )
                    st.success("Conversion complete (OpenCV)")
                    st.image(str(preview_path), caption="Preview", use_column_width=True)
                    col1, col2, col3 = st.columns(3)
                    with open(dxf_path, "rb") as fh:
                        col1.download_button("Download DXF", data=fh.read(), file_name=Path(dxf_path).name)
                    with open(preview_path, "rb") as fh:
                        col2.download_button("Download Preview", data=fh.read(), file_name=Path(preview_path).name)
                    if col3.button("Open in Analyzer"):
                        st.session_state["last_output_dxf"] = str(dxf_path)
                        st.success("DXF set. Open Analyze DXF page.")
                except Exception as e2:
                    st.error(f"Conversion failed: {e2}")

    # ---------------------------------------
    # Auto Pre-Processor (one-click pipeline)
    # ---------------------------------------
    st.markdown("---")
    st.subheader("Auto Pre-Processor (Pipeline)")
    colA, colB, colC, colD = st.columns(4)
    auto_run_potrace = colA.checkbox("Run Potrace", value=True)
    auto_export_dxf = colB.checkbox("Export DXF", value=True)
    auto_simplify_mm = colC.number_input("Post simplify (mm)", min_value=0.0, value=0.5, step=0.1)
    auto_px_per_mm = colD.number_input("Pixels per mm", min_value=0.1, value=10.0, step=0.1)
    if st.button("Run Auto Pre-Processor", disabled=not confirmed):
        try:
            from wjp_analyser.auto_preprocess import auto_process
            project_root = Path(__file__).parent.parent.parent.parent
            base_dir = project_root / "output" / "image_to_dxf"
            base_dir.mkdir(parents=True, exist_ok=True)
            input_for_pipeline = st.session_state.get("_img2dxf_cropped_path") or image_path
            res = auto_process(
                input_path=str(input_for_pipeline),
                outdir=str(base_dir),
                scale_px_per_mm=float(auto_px_per_mm),
                run_potrace_flag=bool(auto_run_potrace),
                export_dxf_flag=bool(auto_export_dxf),
                dxf_simplify_mm=float(auto_simplify_mm) if auto_simplify_mm > 0 else None,
            )
            st.success(f"Auto processing done. Mode: {res.mode}")
            if res.cleaned_png and Path(res.cleaned_png).exists():
                st.image(res.cleaned_png, caption="Auto cleaned PNG", use_column_width=True)
            if res.dxf_simplified_path and Path(res.dxf_simplified_path).exists():
                with open(res.dxf_simplified_path, "rb") as fh:
                    st.download_button("Download Simplified DXF", data=fh.read(), file_name=Path(res.dxf_simplified_path).name)
            elif res.dxf_path and Path(res.dxf_path).exists():
                with open(res.dxf_path, "rb") as fh:
                    st.download_button("Download DXF", data=fh.read(), file_name=Path(res.dxf_path).name)
            st.json(res.logs or {})
        except Exception as e:
            st.error(f"Auto pre-processor failed: {e}")
        except Exception as e:
            # Fallback: OpenCV converter
            if OpenCVImageToDXFConverter is None:
                st.error(f"Conversion failed and OpenCV fallback unavailable: {e}")
            else:
                st.info(f"Potrace pipeline unavailable or failed ({e}). Falling back to OpenCV converter.")
                preview_path = base_dir / f"{Path(input_for_pipeline).stem}_preview.png"
                dxf_path = base_dir / f"{Path(input_for_pipeline).stem}_converted.dxf"
                try:
                    converter = OpenCVImageToDXFConverter(
                        binary_threshold=int(binary_threshold),
                        min_area=500,
                        dxf_size=float(dxf_size),
                    )
                    converter.convert_image_to_dxf(
                        input_image=str(input_for_pipeline),
                        output_dxf=str(dxf_path),
                        preview_output=str(preview_path),
                    )
                    st.success("Conversion complete (OpenCV)")
                    st.image(str(preview_path), caption="Preview", use_column_width=True)
                    col1, col2, col3 = st.columns(3)
                    with open(dxf_path, "rb") as fh:
                        col1.download_button("Download DXF", data=fh.read(), file_name=Path(dxf_path).name)
                    with open(preview_path, "rb") as fh:
                        col2.download_button("Download Preview", data=fh.read(), file_name=Path(preview_path).name)
                    if col3.button("Open in Analyzer"):
                        st.session_state["last_output_dxf"] = str(dxf_path)
                        st.success("DXF set. Open Analyze DXF page.")
                except Exception as e2:
                    st.error(f"Conversion failed: {e2}")

    # ------------------------------
    # Interactive Object Editing
    # ------------------------------
    if InteractiveEditor is not None and editor:
        st.markdown("---")
        st.subheader("Interactive Object Editing")
        
        # Object Detection Controls
        with st.expander("Object Detection Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                min_area = st.number_input("Min Area", min_value=10, value=100, step=10)
                max_area = st.number_input("Max Area", min_value=1000, value=1000000, step=1000)
                min_perimeter = st.number_input("Min Perimeter", min_value=5, value=20, step=5)
            
            with col2:
                min_circularity = st.slider("Min Circularity", 0.0, 1.0, 0.1, 0.01)
                min_solidity = st.slider("Min Solidity", 0.0, 1.0, 0.3, 0.01)
                merge_distance = st.number_input("Merge Distance", min_value=0.0, value=10.0, step=1.0)
        
        # Detect Objects Button
        if st.button("Detect Objects", type="primary"):
            # Generate binary image for object detection
            try:
                import cv2
                from PIL import Image
                
                # Load and preprocess image
                img = Image.open(input_for_pipeline)
                img_array = np.array(img)
                
                # Convert to grayscale
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                # Apply threshold
                _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
                
                # Apply preprocessing
                if gaussian_blur > 0:
                    binary = cv2.GaussianBlur(binary, (gaussian_blur, gaussian_blur), 0)
                
                if use_canny:
                    binary = cv2.Canny(binary, 50, 150)
                
                # Set binary image in editor
                editor.set_binary_image(binary)
                
                # Create detection parameters
                params = DetectionParams(
                    min_area=min_area,
                    max_area=max_area,
                    min_perimeter=min_perimeter,
                    min_circularity=min_circularity,
                    min_solidity=min_solidity,
                    merge_distance=merge_distance
                )
                
                # Detect objects
                objects = editor.detect_objects(params)
                
                st.success(f"Detected {len(objects)} objects")
                
            except Exception as e:
                st.error(f"Object detection failed: {e}")
        
        # Render Interactive Editor
        if editor.detector and editor.detector.objects:
            render_interactive_editor(editor)
        
        # Final Preview System
        if PreviewRenderer is not None and editor.detector:
            st.markdown("---")
            st.subheader("Final Preview")
            
            # Initialize preview renderer
            if "_preview_renderer" not in st.session_state:
                st.session_state["_preview_renderer"] = PreviewRenderer()
            
            preview_renderer = st.session_state["_preview_renderer"]
            
            # Set data for preview renderer
            if editor.current_image is not None:
                preview_renderer.set_images(editor.current_image, editor.binary_image)
            if editor.detector:
                preview_renderer.set_objects(editor.detector.objects)
            
            # Render final preview interface
            render_final_preview_interface(preview_renderer)

# ------------------------------
# Texture Mode (Auto/Per-type)
# ------------------------------
if confirmed and generate_texture_dxf is not None:
    st.subheader("Texture Mode (Auto Mix / Per-Type)")
    with st.sidebar.expander("Texture Mode", expanded=False):
        mode = st.selectbox(
            "Texture Mode",
            ["Auto Mix", "Edges", "Stipple", "Hatch", "Contour"],
            index=0,
            help="Auto Mix classifies tiles into edges/stipple/hatch/contour and vectorizes accordingly",
        )
        dxf_size = st.number_input("DXF canvas (mm)", min_value=100.0, max_value=5000.0, value=1000.0, step=50.0)
        tile = st.slider("Tile size (px)", min_value=16, max_value=128, value=32, step=8)
        # stipple
        dot_spacing = st.number_input("Dot spacing (mm)", min_value=0.2, max_value=10.0, value=1.5, step=0.1)
        dot_radius = st.number_input("Dot radius (mm)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        # hatch
        hatch_spacing = st.number_input("Hatch spacing (mm)", min_value=0.2, max_value=10.0, value=2.0, step=0.1)
        hatch_angle = st.slider("Hatch angle (deg)", min_value=0, max_value=179, value=45)
        cross_hatch = st.checkbox("Cross hatch (add 90 deg)", value=False)
        # contour
        contour_bands = st.slider("Contour bands", min_value=2, max_value=12, value=6)
        # cleanup + kerf
        # cleanup + kerf
        min_feature = st.number_input("Min feature size (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                      help="Remove closed shapes smaller than this; remove open paths shorter than this; drop dots below diameter.")
        simplify_tol = st.number_input("Simplify tolerance (mm)", min_value=0.0, max_value=5.0, value=0.2, step=0.05,
                                       help="Douglas-Peucker tolerance; 0 disables.")
        kerf_width = st.number_input("Kerf width (mm)", min_value=0.0, max_value=5.0, value=1.1, step=0.1)
        kerf_mode = st.selectbox("Kerf compensation", ["None", "Outward (+kerf/2)", "Inward (-kerf/2)", "Inside/Outside (+/- kerf/2)"], index=0, help="Inside/Outside applies +/- kerf/2 based on polygon containment (outer vs inner).")
        # area + colinear merge
        min_area = st.number_input("Min feature area (mm^2)", min_value=0.1, max_value=200.0, value=1.0, step=0.1)
        merge_angle = st.slider("Merge near-colinear angle (deg)", min_value=0, max_value=10, value=3,
                                help="Removes vertices when direction change is below threshold.")
        preserve_arcs = st.checkbox("Preserve ARC entities", value=True)

    with st.sidebar.expander("Advanced: Per-layer cleanup", expanded=False):
        st.caption("Overrides (0 = use global)")
        colA, colB = st.columns(2)
        with colA:
            simplify_edges = st.number_input("Edges simplify (mm)", min_value=0.0, max_value=5.0, value=0.0, step=0.05)
            simplify_hatch = st.number_input("Hatch simplify (mm)", min_value=0.0, max_value=5.0, value=0.0, step=0.05)
            simplify_contour = st.number_input("Contour simplify (mm)", min_value=0.0, max_value=5.0, value=0.0, step=0.05)
        with colB:
            min_area_edges = st.number_input("Edges min area (mm^2)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            min_area_hatch = st.number_input("Hatch min area (mm^2)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            min_area_contour = st.number_input("Contour min area (mm^2)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        # NOTE: advanced per-layer overrides only; global cleanup/kerf controls are above.

    if st.button("Run Texture Vectorizer"):
        try:
            base_dir = Path(st.session_state.get("_wjp_img_base", Path.cwd() / "output" / "image_to_dxf"))
            base_dir.mkdir(parents=True, exist_ok=True)
            input_for_pipeline = st.session_state.get("_img2dxf_cropped_path") or image_path
            if not input_for_pipeline:
                raise RuntimeError("No input image selected.")

            pre = TexPreParams(working_px=1000)
            cls = TexClassParams(tile=int(tile), clusters=4)
            m = mode.lower().split()[0]
            if m == "auto":
                m = "auto"
            vec = TexVecParams(
                mode=m if m in ("edges", "stipple", "hatch", "contour") else "auto",
                dxf_size_mm=float(dxf_size),
                dot_spacing_mm=float(dot_spacing),
                dot_radius_mm=float(dot_radius),
                hatch_spacing_mm=float(hatch_spacing),
                hatch_angle_deg=float(hatch_angle),
                cross_hatch=bool(cross_hatch),
                contour_bands=int(contour_bands),
                min_feature_size_mm=float(min_feature),
                simplify_tol_mm=float(simplify_tol),
                kerf_mm=float(kerf_width),
                kerf_offset_mm=(0.5*float(kerf_width) if kerf_mode.startswith("Out") else (-0.5*float(kerf_width) if kerf_mode.startswith("Inw") else 0.0)),
                kerf_inout=bool(kerf_mode.startswith("Inside/Outside")),
                preserve_arcs=bool(preserve_arcs),
                simplify_tol_edges_mm=(float(simplify_edges) if simplify_edges>0 else None),
                simplify_tol_hatch_mm=(float(simplify_hatch) if simplify_hatch>0 else None),
                simplify_tol_contour_mm=(float(simplify_contour) if simplify_contour>0 else None),
                min_area_edges_mm2=(float(min_area_edges) if min_area_edges>0 else None),
                min_area_hatch_mm2=(float(min_area_hatch) if min_area_hatch>0 else None),
                min_area_contour_mm2=(float(min_area_contour) if min_area_contour>0 else None),
                min_feature_area_mm2=float(min_area),
                merge_angle_deg=float(merge_angle),
            )

            dxf_path, preview_path = generate_texture_dxf(
                image_path=str(input_for_pipeline),
                out_dir=str(base_dir),
                preprocess_params=pre,
                classify_params=cls,
                vec_params=vec,
            )
            st.success("Texture vectorization complete")
            st.image(str(preview_path), caption="Texture Preview", use_column_width=True)
            col1, col2, col3 = st.columns(3)
            with open(dxf_path, "rb") as fh:
                col1.download_button("Download DXF", data=fh.read(), file_name=Path(dxf_path).name)
            with open(preview_path, "rb") as fh:
                col2.download_button("Download Preview", data=fh.read(), file_name=Path(preview_path).name)
            if col3.button("Open in Analyzer"):
                st.session_state["last_output_dxf"] = str(dxf_path)
                st.success("DXF set. Open Analyze DXF page.")
        except Exception as e:
            st.error(f"Texture vectorization failed: {e}")


def _read_dxf_polylines(dxf_path: str):
    import ezdxf
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    polys = []
    # LWPOLYLINE
    for e in msp.query("LWPOLYLINE"):
        try:
            pts = [(v[0], v[1]) for v in e.get_points("xy")]
            polys.append(pts)
        except Exception:
            continue
    # LINE
    for e in msp.query("LINE"):
        try:
            pts = [(float(e.dxf.start.x), float(e.dxf.start.y)), (float(e.dxf.end.x), float(e.dxf.end.y))]
            polys.append(pts)
        except Exception:
            continue
    # ARC (sampled)
    for e in msp.query("ARC"):
        try:
            cx, cy = float(e.dxf.center.x), float(e.dxf.center.y)
            r = float(e.dxf.radius)
            a1 = math.radians(float(e.dxf.start_angle))
            a2 = math.radians(float(e.dxf.end_angle))
            # ensure correct direction and sampling
            steps = 64
            if a2 < a1:
                a2 += 2 * math.pi
            ts = np.linspace(a1, a2, steps)
            pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in ts]
            polys.append(pts)
        except Exception:
            continue
    return polys


def _render_overlay(preview_png: str | Path, dxf_path: str | Path, alpha: float = 0.8, lw: float = 1.2, flip_y: bool = False):
    # Load image
    im = Image.open(preview_png).convert("L")
    W, H = im.size
    im_arr = np.array(im)

    # Read vectors
    polylines = _read_dxf_polylines(str(dxf_path))
    if not polylines:
        raise RuntimeError("No polylines found in DXF for overlay.")

    # Compute DXF bounds
    all_x = []
    all_y = []
    for pts in polylines:
        if not pts:
            continue
        xs, ys = zip(*pts)
        all_x.extend(xs)
        all_y.extend(ys)
    minx, maxx = min(all_x), max(all_x)
    miny, maxy = min(all_y), max(all_y)
    spanx = max(1e-6, maxx - minx)
    spany = max(1e-6, maxy - miny)
    sx = (W - 1) / spanx
    sy = (H - 1) / spany
    s = min(sx, sy)

    def map_pt(x, y):
        xx = (x - minx) * s
        yy = (y - miny) * s
        if flip_y:
            yy = H - 1 - yy
        return xx, yy

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im_arr, cmap="gray", origin="lower" if not flip_y else "upper")
    for pts in polylines:
        mapped = [map_pt(x, y) for (x, y) in pts]
        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        ax.plot(xs, ys, color=(1, 0, 0, alpha), linewidth=lw)
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Overlay: DXF vectors over binary")
    fig.tight_layout()
    return fig









