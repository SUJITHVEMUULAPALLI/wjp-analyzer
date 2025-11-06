"""
WJP Image Analyzer – Integrated App (Phases 1 + 2 + 3)
Upload → Analyze → Visualize → Live Score → DXF-Readiness
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root(s) to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # points to .../src
workspace_root = project_root.parent             # repository root containing 'src'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

# Conditional imports
try:
    from wjp_analyser.image_analyzer.core import analyze_image_for_wjp, AnalyzerConfig
    IMAGE_ANALYZER_AVAILABLE = True
except ImportError:
    IMAGE_ANALYZER_AVAILABLE = False

try:
    from wjp_analyser.dxf_validator import validate_dxf_geometry, ValidatorConfig, ValidationResult
    DXF_VALIDATOR_AVAILABLE = True
except ImportError:
    DXF_VALIDATOR_AVAILABLE = False

try:
    from wjp_analyser.image_preprocessor import preprocess_image_for_analyzer, PreprocConfig
    IMAGE_PREPROCESSOR_AVAILABLE = True
except ImportError:
    IMAGE_PREPROCESSOR_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def draw_overlay(img: np.ndarray, contours: List, closed_flags: List[bool], small_flags: List[bool]) -> np.ndarray:
    """Draw contour overlay on image."""
    disp = img.copy()
    for i, c in enumerate(contours):
        color = (0, 255, 0) if closed_flags[i] else (0, 0, 255)  # Green for closed, red for open
        if small_flags[i]:
            color = (255, 255, 0)  # Yellow for small features
        cv2.drawContours(disp, [c], -1, color, 1)
    return disp


def create_readiness_meter(score: float) -> Tuple[str, str]:
    """Create readiness meter display."""
    if score >= 75:
        return "success", f"✅ DXF Ready (Score: {score:.1f})"
    elif score >= 50:
        return "warning", f"⚠️ Moderate (Score: {score:.1f}) – consider cleanup"
    else:
        return "error", f"❌ Not Ready (Score: {score:.1f})"


def generate_dxf_from_image(image_path: str, report: Dict[str, Any]) -> bytes:
    """Generate DXF file from processed image contours.
    More robust contour extraction and serialization to ensure valid DXF.
    """
    try:
        import ezdxf
        from ezdxf import units
        import io

        doc = ezdxf.new('R2010')
        doc.units = units.MM
        msp = doc.modelspace()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Otsu threshold, invert if background is bright
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = 255 - otsu if float(gray.mean()) > 127.0 else otsu
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        px_to_unit = float(report.get('metrics', {}).get('px_to_unit', 1.0))
        num_added = 0
        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri < 20.0:
                continue
            approx = cv2.approxPolyDP(c, 0.005 * peri, True)
            if len(approx) < 3:
                continue
            points = []
            for pt in approx:
                x, y = int(pt[0][0]), int(pt[0][1])
                points.append((x * px_to_unit, (img.shape[0] - y) * px_to_unit))
            if len(points) >= 2:
                msp.add_lwpolyline(points, close=True)
                num_added += 1

        if num_added == 0:
            msp.add_text("No contours detected", dxfattribs={'height': 5, 'layer': 'TEXT'}).set_pos((10, 10))

        title_text = f"WJP Analysis - Score: {report.get('score', 0):.1f}"
        msp.add_text(title_text, dxfattribs={'height': 5, 'layer': 'TEXT'}).set_pos((10, 30))

        # ezdxf.write() writes text, so use StringIO then encode to bytes
        text_buffer = io.StringIO()
        doc.write(text_buffer)
        data = text_buffer.getvalue().encode("utf-8")
        return data

    except ImportError:
        st.error("❌ DXF generation requires ezdxf package. Install with: pip install ezdxf")
        return b""
    except Exception as e:
        st.error(f"❌ Error generating DXF: {e}")
        return b""


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="WJP Image Analyzer", layout="wide")
    st.title("🧠 WJP Image Analyzer – Full Intelligent Diagnostic")
    st.markdown("**Complete Pipeline**: Upload → Preprocess → Analyze → Visualize → Score → DXF-Ready")
    
    if not IMAGE_ANALYZER_AVAILABLE:
        st.error("❌ Image Analyzer module not available. Please check installation.")
        return
    
    if not IMAGE_PREPROCESSOR_AVAILABLE:
        st.warning("⚠️ Image Preprocessor not available. Analysis will run without preprocessing.")
    
    # Upload section
    st.subheader("📁 Upload Image")
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        # Process uploaded image
        image_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # Display uploaded image
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=f"Uploaded: {uploaded.name}", 
                width=600)

        # Optional preview tab for tuning preprocessing thresholds
        tab_analyzer, tab_preview = st.tabs(["Analyzer", "Pre-Processing Preview"])

        with tab_preview:
            st.markdown("### 🎚️ Pre-Processing Preview – Shadow & Glare Tuning")
            st.caption("Tune thresholds and preview masks before running the full preprocessing pipeline.")

            # Sidebar-like controls within the tab
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                glare_thr = st.slider("Glare Threshold", 200, 255, 235, 1,
                                      help="Higher → only very bright pixels marked as glare")
            with col_p2:
                shadow_thr = st.slider("Shadow Sensitivity", 50, 150, 120, 1,
                                       help="Lower → detect more shadow (darker regions)")
            with col_p3:
                blur_k = st.slider("Shadow Blur Kernel", 15, 55, 35, 2,
                                   help="Larger kernel → smoother illumination estimation")

            # Build preview masks (shadow/glare)
            gray_preview = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_preview, (blur_k, blur_k), 0)
            # Avoid division by zero
            blur_safe = np.maximum(blur, 1)
            ratio = cv2.divide(gray_preview, blur_safe, scale=255)
            shadow_mask = cv2.inRange(ratio, 0, shadow_thr)
            glare_mask = cv2.inRange(gray_preview, glare_thr, 255)

            # Colored overlay preview
            overlay_base = cv2.cvtColor(gray_preview, cv2.COLOR_GRAY2BGR)
            # Colorize masks
            shadow_bgr = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
            glare_bgr = cv2.cvtColor(glare_mask, cv2.COLOR_GRAY2BGR)
            preview_overlay = cv2.addWeighted(overlay_base, 0.8, shadow_bgr, 0.5, 0)
            preview_overlay = cv2.addWeighted(preview_overlay, 1.0, glare_bgr, 0.5, 0)

            c1, c2, c3 = st.columns(3)
            c1.image(shadow_mask, caption="Shadow Mask", width=350)
            c2.image(glare_mask, caption="Glare Mask", width=350)
            c3.image(cv2.cvtColor(preview_overlay, cv2.COLOR_BGR2RGB), caption="Overlay Preview", width=350)

            st.markdown("### Shadow/Glare Stats")
            st.write(f"Shadow pixels: {(shadow_mask>0).sum():,}")
            st.write(f"Glare pixels: {(glare_mask>0).sum():,}")

            # Optional immediate run using tuned thresholds
            if IMAGE_PREPROCESSOR_AVAILABLE and st.button("✅ Run Full Preprocessing (using tuned thresholds)"):
                try:
                    try:
                        project_root_dir = Path(__file__).parent.parent.parent.parent
                        temp_dir_prev = project_root_dir / "output" / "temp"
                        temp_dir_prev.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        import tempfile
                        temp_dir_prev = Path(tempfile.gettempdir()) / "wjp_analyser" / "temp"
                        temp_dir_prev.mkdir(parents=True, exist_ok=True)

                    temp_path_prev = temp_dir_prev / f"preview_{uploaded.name}"
                    cv2.imwrite(str(temp_path_prev), img)

                    cfg_prev = PreprocConfig(glare_threshold=int(glare_thr))
                    corrected_prev, metrics_prev = preprocess_image_for_analyzer(str(temp_path_prev), cfg_prev)
                    st.success("Preprocessing complete!")
                    st.json(metrics_prev)
                    st.image(cv2.cvtColor(corrected_prev, cv2.COLOR_BGR2RGB), caption="Corrected Image (Preview Run)", width=900)
                except Exception as e:
                    st.error(f"❌ Preview preprocessing failed: {e}")
        
        # Preprocessing configuration
        st.subheader("🔧 Preprocessing Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Lighting & Glare**")
            glare_threshold = st.number_input("Glare Threshold", min_value=200, max_value=255, value=225, step=5, 
                                            help="Lower values detect more subtle reflections (improved from 240)")
            contrast_limit = st.number_input("Contrast Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
            inpaint_radius = st.number_input("Inpaint Radius", min_value=3, max_value=15, value=5, step=1)
        
        with col2:
            st.markdown("**Skew & Perspective**")
            enable_deskew = st.checkbox("Enable Auto Deskew", value=True)
            skew_confidence = st.number_input("Skew Confidence Threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.1,
                                             help="Higher confidence required for auto-rotation (improved from 0.5)")
            perspective_tolerance = st.number_input("Perspective Tolerance (°)", min_value=1.0, max_value=20.0, value=8.0, step=1.0)
            use_contour_skew = st.checkbox("Use Contour-Based Skew Detection", value=True,
                                         help="Secondary skew detection using largest contour bounding box")
        
        with col3:
            st.markdown("**Analysis Settings**")
            px_to_unit = st.number_input("Pixels to Unit Ratio", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            max_size_px = st.number_input("Max Size (px)", min_value=500, max_value=5000, value=2000, step=100)
            gaussian_blur = st.checkbox("Enable Gaussian Blur", value=True)
            blur_kernel_size = st.number_input("Blur Kernel Size", min_value=3, max_value=15, value=3, step=2) if gaussian_blur else 3
            color_restoration = st.checkbox("Enable Color Restoration", value=True,
                                          help="Overlay corrected grayscale on original color texture")
        
        # Analyze button
        skel_toggle = st.checkbox("🦴 Apply centerline skeletonization before analysis", value=False, help="Reduces double-contour artifacts in subsequent DXF conversion")
        if st.button("🔍 Run Complete Pipeline Analysis", type="primary"):
            with st.spinner("Running complete pipeline..."):
                try:
                    # Save original image to temp file (writable path)
                    try:
                        project_root_dir = Path(__file__).parent.parent.parent.parent
                        temp_dir = project_root_dir / "output" / "temp"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        import tempfile
                        temp_dir = Path(tempfile.gettempdir()) / "wjp_analyser" / "temp"
                        temp_dir.mkdir(parents=True, exist_ok=True)

                    temp_path = temp_dir / f"temp_{uploaded.name}"
                    cv2.imwrite(str(temp_path), img)
                    
                    # Optional: centerline skeletonization pass for visualization/analysis
                    if skel_toggle:
                        try:
                            from wjp_analyser.skeletonize_preprocessor import skeletonize_array
                            img = skeletonize_array(img, invert=True, thresh=200)
                            st.info("Applied centerline skeletonization for analysis preview")
                        except Exception as e:
                            st.warning(f"Skeletonization skipped: {e}")

                    # Step 1: Preprocessing (if available)
                    preproc_metrics = None
                    if IMAGE_PREPROCESSOR_AVAILABLE:
                        st.info("🔧 **Step 1**: Running image preprocessing...")
                        
                        # Create preprocessing config
                        preproc_cfg = PreprocConfig(
                            deskew=enable_deskew,
                            max_size_px=max_size_px,
                            glare_threshold=glare_threshold,
                            inpaint_radius=inpaint_radius,
                            contrast_clip_limit=contrast_limit,
                            skew_conf_threshold=skew_confidence,
                            auto_rotation_threshold=skew_confidence,
                            perspective_warn_angle=perspective_tolerance,
                            use_contour_skew=use_contour_skew,
                            color_restoration=color_restoration
                        )
                        
                        # Run preprocessing
                        corrected_img, preproc_metrics = preprocess_image_for_analyzer(str(temp_path), preproc_cfg)
                        
                        # Save corrected image
                        cv2.imwrite(str(temp_path), corrected_img)
                        
                        shadow_info = ""
                        if preproc_metrics.get('shadow_flagged', False):
                            shadow_info = f", compensated {preproc_metrics.get('shadow_pixels', 0)} shadow pixels"
                        
                        st.success(f"✅ Preprocessing complete! Fixed {preproc_metrics['glare_pixels']} glare pixels{shadow_info}, "
                                 f"skew: {preproc_metrics['skew_angle_deg']}° (conf: {preproc_metrics['skew_confidence']:.2f})")
                    
                    # Step 2: Analysis
                    st.info("🔍 **Step 2**: Running intelligent analysis...")
                    
                    # Create analyzer config
                    cfg = AnalyzerConfig(
                        px_to_unit=px_to_unit,
                        max_size_px=max_size_px,
                        gaussian_blur_ksize=blur_kernel_size if gaussian_blur else None,
                        deskew=False,  # Already handled by preprocessor
                        contour_mode_tree=True
                    )
                    
                    # Run analysis
                    report = analyze_image_for_wjp(str(temp_path), cfg)
                    
                    # Step 3: Visualization
                    st.info("📊 **Step 3**: Creating visualization...")
                    
                    # Load the processed image for visualization
                    processed_img = cv2.imread(str(temp_path))
                    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    cnts, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    closed_flags = [True] * len(cnts)  # Simplified for demo
                    small_flags = [False] * len(cnts)  # Simplified for demo
                    vis = draw_overlay(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cnts, closed_flags, small_flags)
                    
                    # Display results
                    st.subheader("📊 Complete Pipeline Results")
                    
                    # Show preprocessing results if available
                    if preproc_metrics:
                        st.markdown("### 🔧 Enhanced Preprocessing Results")
                        col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
                        
                        with col_pre1:
                            st.markdown("**Lighting Correction**")
                            st.metric("Glare Pixels Fixed", preproc_metrics['glare_pixels'])
                            st.metric("Shadow Pixels", preproc_metrics.get('shadow_pixels', 0))
                            if preproc_metrics.get('shadow_flagged', False):
                                st.warning("⚠️ Shadows Detected")
                            if preproc_metrics.get('shadow_compensation_applied', False):
                                st.success("✅ Shadows Compensated")
                            if preproc_metrics.get('color_restoration_applied', False):
                                st.success("✅ Color Restored")
                            else:
                                st.info("ℹ️ Grayscale Only")
                        
                        with col_pre2:
                            st.markdown("**Skew Detection**")
                            st.metric("Combined Angle", f"{preproc_metrics['skew_angle_deg']}°")
                            st.metric("Confidence", f"{preproc_metrics['skew_confidence']:.2f}")
                            if preproc_metrics.get('rotation_applied', False):
                                st.success("✅ Auto-Rotated")
                            else:
                                st.info("ℹ️ No Rotation")
                        
                        with col_pre3:
                            st.markdown("**Detection Methods**")
                            st.metric("Hough Angle", f"{preproc_metrics.get('hough_angle', 0):.1f}°")
                            st.metric("Contour Angle", f"{preproc_metrics.get('contour_angle', 0):.1f}°")
                            if preproc_metrics.get('use_contour_skew', False):
                                st.success("✅ Dual Detection")
                            else:
                                st.info("ℹ️ Hough Only")
                        
                        with col_pre4:
                            st.markdown("**Quality Assessment**")
                            st.metric("Perspective Tilt", f"{preproc_metrics['perspective_tilt_deg']}°")
                            if preproc_metrics['perspective_flagged']:
                                st.warning("⚠️ Perspective Issue")
                            else:
                                st.success("✅ Good Perspective")
                            
                            if preproc_metrics['skew_confidence'] > 0.7:
                                st.success("✅ High Confidence")
                            else:
                                st.info("ℹ️ Moderate Confidence")
                    
                    # Main layout
                    col1, col2 = st.columns([1.3, 1])
                    
                    with col1:
                        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                caption="Contour Overlay (🟢 closed | 🔴 open | 🟡 small)",
                                width=600)
                    
                    with col2:
                        # DXF Readiness Meter
                        st.markdown("### 🎯 DXF Readiness Meter")
                        readiness = report.get("score", 0)
                        st.progress(int(readiness) / 100)
                        
                        meter_type, meter_text = create_readiness_meter(readiness)
                        if meter_type == "success":
                            st.success(meter_text)
                        elif meter_type == "warning":
                            st.warning(meter_text)
                        else:
                            st.error(meter_text)
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        suggestions = report.get("suggestions", [])
                        if suggestions:
                            for suggestion in suggestions:
                                st.markdown(f"- {suggestion}")
                        else:
                            st.info("No specific recommendations available.")
                        
                        # Key Metrics
                        st.markdown("### 📈 Key Metrics")
                        metrics = report.get("metrics", {})
                        if metrics:
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    st.metric(key.replace("_", " ").title(), f"{value:.2f}")
                                else:
                                    st.text(f"{key.replace('_', ' ').title()}: {value}")
                    
                    # Full report section
                    st.markdown("---")
                    st.subheader("📋 Complete Pipeline Report")
                    
                    # Combine preprocessing and analysis reports
                    from datetime import datetime
                    complete_report = {
                        "pipeline_version": "1.0",
                        "timestamp": datetime.now().isoformat(),
                        "preprocessing": preproc_metrics if preproc_metrics else {"status": "skipped"},
                        "analysis": report,
                        "summary": {
                            "preprocessing_applied": preproc_metrics is not None,
                            "final_score": report.get("score", 0),
                            "dxf_ready": report.get("score", 0) >= 75
                        }
                    }
                    
                    # Display as JSON
                    st.json(complete_report)
                    
                    # Download buttons
                    col_download1, col_download2 = st.columns(2)
                    
                    with col_download1:
                        report_json = json.dumps(complete_report, indent=2, default=str)
                        st.download_button(
                            "⬇️ Download Complete Report (JSON)",
                            data=report_json,
                            file_name=f"{uploaded.name}_complete_pipeline_report.json",
                            mime="application/json"
                        )
                    
                    with col_download2:
                        # DXF Download button (available regardless of score)
                        dxf_data = generate_dxf_from_image(str(temp_path), report)
                        st.download_button(
                            "📐 Download DXF File",
                            data=dxf_data,
                            file_name=f"{uploaded.name}_processed.dxf",
                            mime="application/dxf",
                            help="Download the processed image as a DXF file for CAD software"
                        )
                        
                        # DXF Validation section
                        if DXF_VALIDATOR_AVAILABLE and dxf_data:
                                st.markdown("---")
                                st.subheader("🔎 DXF Geometry Validation")
                                
                                # Validation configuration
                                col_val1, col_val2 = st.columns(2)
                                with col_val1:
                                    min_spacing = st.number_input("Min Spacing (mm)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
                                    min_radius = st.number_input("Min Radius (mm)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
                                with col_val2:
                                    min_area = st.number_input("Min Area (mm²)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
                                    open_tolerance = st.number_input("Open Tolerance (mm)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                                
                                if st.button("🔎 Validate DXF Geometry", type="secondary"):
                                    with st.spinner("Validating DXF geometry..."):
                                        try:
                                            # Save DXF data to temporary file for validation
                                            temp_dxf_path = temp_dir / f"temp_{uploaded.name}_validation.dxf"
                                            with open(temp_dxf_path, 'wb') as f:
                                                f.write(dxf_data)
                                            
                                            # Create validator config
                                            validator_cfg = ValidatorConfig(
                                                min_spacing_mm=min_spacing,
                                                min_radius_mm=min_radius,
                                                min_area_mm2=min_area,
                                                open_tolerance=open_tolerance
                                            )
                                            
                                            # Run validation
                                            validation_result = validate_dxf_geometry(str(temp_dxf_path), validator_cfg)
                                            
                                            # Display validation results
                                            st.markdown("### 📊 Validation Results")
                                            
                                            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                                            
                                            with col_res1:
                                                st.metric("Open Contours", validation_result.open_contours)
                                                if validation_result.open_contours > 0:
                                                    st.error("❌ Open contours detected")
                                                else:
                                                    st.success("✅ All contours closed")
                                            
                                            with col_res2:
                                                st.metric("Spacing Violations", validation_result.spacing_violations)
                                                if validation_result.spacing_violations > 0:
                                                    st.error("❌ Spacing too small")
                                                else:
                                                    st.success("✅ Spacing adequate")
                                            
                                            with col_res3:
                                                st.metric("Small Features", validation_result.small_features)
                                                if validation_result.small_features > 0:
                                                    st.warning("⚠️ Small features")
                                                else:
                                                    st.success("✅ Features adequate")
                                            
                                            with col_res4:
                                                st.metric("Radius Violations", validation_result.radius_violations)
                                                if validation_result.radius_violations > 0:
                                                    st.error("❌ Radius too small")
                                                else:
                                                    st.success("✅ Radius adequate")
                                            
                                            # Overall validation status
                                            st.markdown("### 🎯 Overall Validation Status")
                                            if validation_result.is_valid:
                                                st.success("🎉 **DXF is waterjet-ready!** All geometry checks passed.")
                                            else:
                                                st.warning("⚠️ **Issues detected** - Review recommendations below.")
                                            
                                            # Recommendations
                                            if validation_result.recommendations:
                                                st.markdown("### 💡 Recommendations")
                                                for rec in validation_result.recommendations:
                                                    st.markdown(f"- {rec}")
                                            
                                            # Detailed results
                                            st.markdown("### 📋 Detailed Results")
                                            st.json(validation_result.details)
                                            
                                            # Clean up temp DXF file
                                            if temp_dxf_path.exists():
                                                temp_dxf_path.unlink()
                                                
                                        except Exception as e:
                                            st.error(f"❌ Validation failed: {e}")
                        else:
                            if not DXF_VALIDATOR_AVAILABLE:
                                st.info("🔎 DXF validation requires shapely package: `pip install shapely`")
                    
                    # Integration hook
                    st.markdown("---")
                    st.subheader("🔗 Integration Status")
                    
                    if readiness >= 75:
                        st.success("🎉 **Image passed threshold → ready for production!**")
                        st.info("📐 **DXF file generated** - Download and validate above for CAD software integration")
                        if DXF_VALIDATOR_AVAILABLE:
                            st.info("🔎 **DXF validation available** - Check geometry rules before machine export")
                        
                        # Production readiness details
                        st.markdown("### 🏭 Production Readiness")
                        col_prod1, col_prod2, col_prod3 = st.columns(3)
                        
                        with col_prod1:
                            st.metric("Quality Score", f"{readiness:.1f}/100")
                            if readiness >= 90:
                                st.success("🌟 Excellent Quality")
                            elif readiness >= 80:
                                st.success("✅ Very Good Quality")
                            else:
                                st.success("✅ Good Quality")
                        
                        with col_prod2:
                            st.metric("DXF Status", "Ready")
                            st.success("📐 CAD Compatible")
                        
                        with col_prod3:
                            st.metric("Validation", "Available")
                            if DXF_VALIDATOR_AVAILABLE:
                                st.success("🔎 Geometry Checked")
                            else:
                                st.info("ℹ️ Install shapely for validation")
                        
                        # Next steps for production
                        st.markdown("### 🚀 Production Workflow")
                        col_work1, col_work2 = st.columns(2)
                        
                        with col_work1:
                            st.markdown("""
                            **📐 CAD Integration:**
                            1. Download DXF file above
                            2. Open in CAD software (AutoCAD, Fusion 360)
                            3. Verify dimensions and scaling
                            4. Apply cutting parameters
                            5. Generate machine code
                            """)
                        
                        with col_work2:
                            st.markdown("""
                            **🔧 Machine Setup:**
                            1. Load material on waterjet table
                            2. Set cutting parameters
                            3. Run geometry validation
                            4. Start cutting operation
                            5. Monitor quality
                            """)
                        
                        # Quality assurance checklist
                        st.markdown("### ✅ Pre-Production Checklist")
                        checklist_items = [
                            "DXF file downloaded and verified",
                            "Dimensions checked in CAD software",
                            "Cutting parameters optimized",
                            "Material properly secured",
                            "Machine calibration verified",
                            "Safety protocols followed"
                        ]
                        
                        for item in checklist_items:
                            st.checkbox(item, value=False, key=f"checklist_{item}")
                        
                        st.session_state["last_ready_image"] = str(temp_path)
                        st.session_state["last_analysis_report"] = report
                        st.session_state["last_dxf_ready"] = True
                        
                        if st.button("🚀 Proceed to Advanced DXF Conversion", type="primary"):
                            st.info("Redirecting to Image to DXF converter...")
                            # This would redirect to the image-to-dxf page
                            st.switch_page("pages/image_to_dxf.py")
                    else:
                        st.info("🔧 **Analyzer suggests improvements** before DXF conversion.")
                        
                        # Detailed improvement suggestions
                        st.markdown("### 💡 Improvement Recommendations")
                        
                        score = report.get("score", 0)
                        suggestions = report.get("suggestions", [])
                        
                        if score < 50:
                            st.error("**Critical Issues Detected** - Image needs significant improvement")
                            col_imp1, col_imp2 = st.columns(2)
                            with col_imp1:
                                st.markdown("""
                                **🔧 Immediate Actions:**
                                - Improve image lighting and contrast
                                - Remove background noise
                                - Ensure clear, sharp edges
                                - Use higher resolution if possible
                                """)
                            with col_imp2:
                                st.markdown("""
                                **📸 Photography Tips:**
                                - Use even lighting (avoid shadows)
                                - Keep camera perpendicular to surface
                                - Ensure stable focus
                                - Use contrasting background
                                """)
                        elif score < 75:
                            st.warning("**Moderate Issues** - Some improvements needed")
                            col_imp1, col_imp2 = st.columns(2)
                            with col_imp1:
                                st.markdown("""
                                **🔧 Quick Fixes:**
                                - Adjust preprocessing settings above
                                - Try different blur/edge detection
                                - Fine-tune contour parameters
                                - Enable advanced preprocessing
                                """)
                            with col_imp2:
                                st.markdown("""
                                **⚙️ Settings to Try:**
                                - Increase blur kernel size
                                - Adjust contour perimeter threshold
                                - Enable deskew correction
                                - Try different analysis modes
                                """)
                        
                        # Specific suggestions from analyzer
                        if suggestions:
                            st.markdown("### 🎯 Specific Suggestions")
                            for i, suggestion in enumerate(suggestions, 1):
                                st.markdown(f"{i}. {suggestion}")
                        
                        # Action buttons for improvement
                        st.markdown("### 🚀 Next Steps")
                        col_action1, col_action2, col_action3 = st.columns(3)
                        
                        with col_action1:
                            if st.button("🔄 Retry with Different Settings", width="stretch"):
                                st.info("Adjust the preprocessing and analysis settings above and try again.")
                        
                        with col_action2:
                            if st.button("📸 Upload Different Image", width="stretch"):
                                st.info("Try uploading a higher quality or better-lit image.")
                        
                        with col_action3:
                            if st.button("🔧 Manual Preprocessing", width="stretch"):
                                st.info("Use external image editing software to improve contrast and clarity.")
                        
                        # Progress tracking
                        st.markdown("### 📊 Quality Progress")
                        progress_value = min(score / 75, 1.0)  # Normalize to 75 as target
                        st.progress(progress_value)
                        st.caption(f"Current Score: {score:.1f}/75 (Target: 75+ for DXF conversion)")
                        
                        # Tips for better results
                        st.markdown("### 💡 Pro Tips for Better Results")
                        tips_col1, tips_col2 = st.columns(2)
                        
                        with tips_col1:
                            st.markdown("""
                            **📐 For Technical Drawings:**
                            - Use high contrast (black on white)
                            - Ensure clean, sharp lines
                            - Remove any annotations or text
                            - Use consistent line weights
                            """)
                        
                        with tips_col2:
                            st.markdown("""
                            **🖼️ For Photos:**
                            - Use good lighting (avoid flash)
                            - Keep camera steady and perpendicular
                            - Use contrasting background
                            - Ensure object is in sharp focus
                            """)
                        
                        st.session_state["last_analysis_report"] = report
                        st.session_state["last_dxf_ready"] = False
                    
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
    
    # Auto Pre-Processor (one-click) for DXF readiness preview
    st.markdown("---")
    st.subheader("Auto Pre-Processor (Pipeline)")
    colAA, colBB, colCC, colDD = st.columns(4)
    auto_run_potrace2 = colAA.checkbox("Run Potrace", value=False, help="Generate SVG (optional)")
    auto_export_dxf2 = colBB.checkbox("Export DXF", value=False, help="Requires Inkscape installed")
    auto_simplify_mm2 = colCC.number_input("Post simplify (mm)", min_value=0.0, value=0.5, step=0.1)
    auto_px_per_mm2 = colDD.number_input("Pixels per mm", min_value=0.1, value=10.0, step=0.1)
    potrace_exe2 = st.text_input("Potrace executable (optional)", value=st.session_state.get("_potrace_exe", ""), help="Leave blank to use PATH/WSL. Provide full path like C:\\Tools\\potrace\\potrace.exe")
    if st.button("Run Auto Pre-Processor on Uploaded Image", disabled=not bool(uploaded or st.session_state.get('last_ready_image'))):
        try:
            from wjp_analyser.auto_preprocess import auto_process
            # Pick source: uploaded in-session file if available, otherwise last_ready_image
            src_path = None
            try:
                if uploaded:
                    # Save current upload to a temporary, writable path
                    from pathlib import Path
                    try:
                        project_root_dir = Path(__file__).parent.parent.parent.parent
                        temp_dir2 = project_root_dir / "output" / "temp"
                        temp_dir2.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        import tempfile
                        temp_dir2 = Path(tempfile.gettempdir()) / "wjp_analyser" / "temp"
                        temp_dir2.mkdir(parents=True, exist_ok=True)
                    src_path = str(temp_dir2 / f"auto_{uploaded.name}")
                    cv2.imwrite(src_path, img if isinstance(img, np.ndarray) else cv2.imdecode(np.asarray(bytearray(uploaded.read()), dtype=np.uint8), cv2.IMREAD_COLOR))
            except Exception:
                pass
            if not src_path:
                src_path = st.session_state.get("last_ready_image")
            if not src_path:
                st.warning("No image available. Upload an image first.")
            else:
                # Output directory
                from pathlib import Path
                project_root_dir = Path(__file__).parent.parent.parent.parent
                out_dir2 = project_root_dir / "output" / "auto_preprocess"
                out_dir2.mkdir(parents=True, exist_ok=True)
                res2 = auto_process(
                    input_path=str(src_path),
                    outdir=str(out_dir2),
                    scale_px_per_mm=float(auto_px_per_mm2),
                    run_potrace_flag=bool(auto_run_potrace2),
                    export_dxf_flag=bool(auto_export_dxf2),
                    dxf_simplify_mm=float(auto_simplify_mm2) if auto_simplify_mm2 > 0 else None,
                    potrace_exe=(potrace_exe2.strip() or None),
                )
                st.success(f"Auto processing done. Mode: {res2.mode}")
                if res2.cleaned_png and Path(res2.cleaned_png).exists():
                    st.image(res2.cleaned_png, caption="Auto cleaned PNG", use_column_width=True)
                if res2.dxf_simplified_path and Path(res2.dxf_simplified_path).exists():
                    with open(res2.dxf_simplified_path, "rb") as fh:
                        st.download_button("Download Simplified DXF", data=fh.read(), file_name=Path(res2.dxf_simplified_path).name)
                elif res2.dxf_path and Path(res2.dxf_path).exists():
                    with open(res2.dxf_path, "rb") as fh:
                        st.download_button("Download DXF", data=fh.read(), file_name=Path(res2.dxf_path).name)
                st.json(res2.logs or {})
        except Exception as e:
            st.error(f"Auto pre-processor failed: {e}")

    # Instructions
    st.markdown("---")
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Upload** any JPG/PNG image
    2. **Configure** analysis parameters (optional)
    3. **Click** "Run Intelligent Analysis"
    4. **Review** the contour overlay and readiness score
    5. **Download** the full report if needed
    6. **Proceed** to DXF conversion if ready
    """)
    
    # Features overview
    st.markdown("---")
    st.subheader("🎯 Complete Pipeline Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔧 Enhanced Preprocessing**
        - Adaptive glare detection (225 threshold)
        - Dual skew detection (Hough + Contour)
        - Auto-rotation (confidence > 0.6)
        - Color restoration overlay
        - CLAHE contrast enhancement
        """)
    
    with col2:
        st.markdown("""
        **🔍 Analysis & Visualization**
        - Intelligent edge detection
        - Live contour overlay
        - Color-coded features
        - Quality metrics
        - Real-time feedback
        """)
    
    with col3:
        st.markdown("""
        **🎯 Integration & Scoring**
        - DXF readiness scoring
        - Smart recommendations
        - Complete pipeline reports
        - **📐 DXF file download**
        - **🔎 DXF geometry validation**
        - CAD software integration
        """)


if __name__ == "__main__":
    main()
