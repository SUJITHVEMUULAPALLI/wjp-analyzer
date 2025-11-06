"""Image Analyzer Page for WJP ANALYSER Streamlit Interface."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st

# Conditional imports
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    NUMPY_AVAILABLE = False
    plt = None
    np = None

# Path shim
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_WS_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

try:
    from wjp_analyser.image_analyzer import analyze_image_for_wjp, AnalyzerConfig, quick_analyze
    from wjp_analyser.image_analyzer.integration import ImageAnalyzerGate
except ImportError as e:
    st.error(f"Image analyzer not available: {e}")
    st.stop()

def main():
    st.set_page_config(page_title="Image Analyzer", layout="wide")
    st.title("🖼️ Image Analyzer")
    st.markdown("Analyze images for waterjet cutting suitability before DXF conversion")
    
    # Check if guided mode is enabled
    guided_mode = st.session_state.get('guided_mode', False)
    if guided_mode:
        st.info("🎯 **Guided Mode**: Follow the steps below for guided image analysis.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Analysis Configuration")
    
    # Basic settings
    st.sidebar.subheader("Basic Settings")
    px_to_unit = st.sidebar.number_input(
        "Pixels to Unit Ratio (px/mm)",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1,
        help="Conversion ratio from pixels to millimeters"
    )
    
    max_size_px = st.sidebar.number_input(
        "Max Analysis Size (pixels)",
        min_value=100,
        max_value=4000,
        value=1024,
        step=100,
        help="Maximum image size for analysis (larger images are resized)"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        gaussian_blur = st.checkbox("Enable Gaussian Blur", value=False)
        blur_kernel_size = st.number_input(
            "Blur Kernel Size",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
            disabled=not gaussian_blur
        )
        
        deskew_enabled = st.checkbox("Enable Deskew Detection", value=True)
        hough_min_line_len = st.number_input(
            "Hough Min Line Length",
            min_value=10,
            max_value=200,
            value=50,
            disabled=not deskew_enabled
        )
        
        contour_mode = st.selectbox(
            "Contour Detection Mode",
            ["Tree", "External", "List"],
            index=0
        )
        
        min_contour_area = st.number_input(
            "Min Contour Area (pixels)",
            min_value=1,
            max_value=1000,
            value=10
        )
    
    # File upload
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image to analyze for waterjet cutting suitability"
    )
    
    if uploaded_file:
        # Save uploaded file to a writable location
        try:
            project_root_dir = Path(__file__).parent.parent.parent.parent
            upload_dir = project_root_dir / "output" / "temp"
            upload_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            import tempfile
            upload_dir = Path(tempfile.gettempdir()) / "wjp_analyser" / "temp"
            upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        st.subheader("📷 Uploaded Image")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Image info
            st.markdown("**Image Information:**")
            st.text(f"Name: {uploaded_file.name}")
            st.text(f"Size: {uploaded_file.size:,} bytes")
            st.text(f"Type: {uploaded_file.type}")
        
        # Analysis configuration
        st.subheader("🔍 Analysis Configuration")
        
        # Create analyzer config
        config = AnalyzerConfig(
            px_to_unit=px_to_unit,
            max_size_px=max_size_px,
            gaussian_blur_ksize=blur_kernel_size if gaussian_blur else None,
            deskew=deskew_enabled,
            hough_min_line_len=hough_min_line_len,
            contour_mode_tree=(contour_mode == "Tree"),
            min_contour_perimeter_px=min_contour_area
        )
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            min_score_threshold = st.number_input(
                "Minimum Score Threshold",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=5.0,
                help="Minimum suitability score to pass analysis"
            )
        
        with col2:
            auto_fix_enabled = st.checkbox(
                "Enable Auto-Fix Suggestions",
                value=False,
                help="Generate automatic fix suggestions"
            )
        
        # Run analysis button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Run analysis
                    report = analyze_image_for_wjp(str(file_path), config)
                    
                    # Display results
                    display_analysis_results(report, min_score_threshold, auto_fix_enabled, uploaded_file.name)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.exception(e)
    
    else:
        # Show example or instructions
        st.info("👆 Upload an image to begin analysis")
        
        # Show analysis features
        st.subheader("🎯 Analysis Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Basic Analysis:**
            - Image dimensions and aspect ratio
            - Grayscale statistics
            - Color distribution analysis
            
            **🔍 Edge Detection:**
            - Edge density calculation
            - Edge contrast ratio
            - Contour detection and analysis
            """)
        
        with col2:
            st.markdown("""
            **🎨 Texture Analysis:**
            - Entropy calculation
            - FFT high-frequency energy
            - Noise level assessment
            
            **⚙️ Manufacturability:**
            - Minimum spacing analysis
            - Curve radius detection
            - Suitability scoring
            """)


def display_analysis_results(report: Dict[str, Any], min_threshold: float, auto_fix: bool, uploaded_filename: str):
    """Display comprehensive analysis results."""
    
    st.subheader("📊 Analysis Results")
    
    # Overall score
    score = report.get('score', 0)
    score_color = "green" if score >= min_threshold else "red"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Suitability Score",
            f"{score:.1f}/100",
            delta=f"{'✅ Pass' if score >= min_threshold else '❌ Fail'}"
        )
    
    with col2:
        st.metric(
            "Image Dimensions",
            f"{report.get('width', 0)}×{report.get('height', 0)}",
            delta=f"AR: {report.get('aspect_ratio', 0):.2f}"
        )
    
    with col3:
        st.metric(
            "Contours Found",
            report.get('total_contours', 0),
            delta=f"Closed: {report.get('closed_contours', 0)}"
        )
    
    # Detailed metrics
    st.subheader("📈 Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Edge Analysis:**")
        st.text(f"Edge Density: {report.get('edge_density', 0):.3f}")
        st.text(f"Edge Contrast: {report.get('edge_contrast_ratio', 0):.3f}")
        
        st.markdown("**🎨 Texture Analysis:**")
        st.text(f"Entropy: {report.get('entropy', 0):.3f}")
        st.text(f"FFT High Freq: {report.get('fft_highfreq_energy', 0):.3f}")
    
    with col2:
        st.markdown("**📐 Geometry:**")
        st.text(f"Skew Angle: {report.get('skew_angle_deg', 0):.1f}°")
        st.text(f"Min Radius: {report.get('min_radius_unit', 0):.2f} mm")
        st.text(f"Small Features: {report.get('small_features_count', 0)}")
        
        st.markdown("**⚙️ Quality:**")
        st.text(f"Contrast Score: {report.get('contrast_score', 0):.1f}")
        st.text(f"Clarity Score: {report.get('clarity_score', 0):.1f}")
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    # Create visualizations if data is available
    if 'edge_density' in report and 'entropy' in report:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Score breakdown
        categories = ['Contrast', 'Clarity', 'Geometry', 'Manufacturability']
        scores = [
            report.get('contrast_score', 0),
            report.get('clarity_score', 0),
            report.get('geometry_score', 0),
            report.get('manufacturability_score', 0)
        ]
        
        ax1.bar(categories, scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        ax1.set_title('Score Breakdown')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 100)
        
        # Overall score gauge
        ax2.pie([score, 100-score], labels=['Score', 'Remaining'], 
                colors=['#4ecdc4', '#e0e0e0'], startangle=90)
        ax2.set_title(f'Overall Score: {score:.1f}/100')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Suggestions and recommendations
    st.subheader("💡 Recommendations")
    
    suggestions = report.get('suggestions', [])
    warnings = report.get('warnings', [])
    
    if suggestions:
        st.markdown("**✅ Suggestions:**")
        for suggestion in suggestions:
            st.text(f"• {suggestion}")
    
    if warnings:
        st.markdown("**⚠️ Warnings:**")
        for warning in warnings:
            st.text(f"• {warning}")
    
    # Auto-fix suggestions
    if auto_fix and score < min_threshold:
        st.subheader("🔧 Auto-Fix Suggestions")
        
        if report.get('skew_angle_deg', 0) > 1.0:
            st.info("🔄 **Deskew**: Consider rotating the image to correct skew angle")
        
        if report.get('edge_density', 0) < 0.1:
            st.info("📈 **Enhance Contrast**: Increase image contrast for better edge detection")
        
        if report.get('small_features_count', 0) > 10:
            st.info("🔍 **Simplify**: Consider removing small features that may be too fine for cutting")
        
        if report.get('min_radius_unit', 0) < 1.0:
            st.info("📐 **Increase Radius**: Some curves may be too tight for waterjet cutting")
    
    # Export options
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Report"):
            # Save report to JSON
            try:
                project_root_dir = Path(__file__).parent.parent.parent.parent
                output_dir = project_root_dir / "output" / "image_analyzer"
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                import tempfile
                output_dir = Path(tempfile.gettempdir()) / "wjp_analyser" / "image_analyzer"
                output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / f"analysis_{uploaded_filename.split('.')[0]}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            
            st.success(f"Report saved to: {report_file}")
    
    with col2:
        if st.button("📊 Download Report"):
            # Create downloadable JSON
            json_str = json.dumps(report, indent=2)
            st.download_button(
                "📥 Download JSON Report",
                json_str,
                file_name=f"image_analysis_{uploaded_filename.split('.')[0]}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
