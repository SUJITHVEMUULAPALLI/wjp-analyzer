"""Multipage Streamlit UI for Waterjet DXF Analyzer."""

from __future__ import annotations

import os
import sys
import streamlit as st

# Path shim so `wjp_analyser` is importable when run via `streamlit run`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def main():
    st.set_page_config(page_title="Waterjet DXF Analyzer", layout="wide")
    st.title("Waterjet DXF Analyzer")
    
    # Check if guided mode is enabled via environment variable or checkbox
    guided_mode_env = os.environ.get("WJP_GUIDED_MODE", "false").lower() == "true"
    guided_mode_checkbox = st.sidebar.checkbox("🎯 Enable Guided Mode", value=guided_mode_env, help="Enable step-by-step guidance for all features")
    guided_mode = guided_mode_env or guided_mode_checkbox
    
    if guided_mode:
        st.info("🎯 **Guided Mode Enabled** - You'll receive step-by-step guidance through all features. Use the guided pages in the sidebar for the best experience.")
        st.markdown("""
        ### 🎯 Guided Pages Available:
        - **Guided Designer**: Step-by-step design creation with intelligent tips
        - **Guided Image to DXF**: Guided image conversion with parameter optimization
        - **Analyze DXF**: Advanced analysis with guided workflow
        - **Nesting**: Intelligent nesting with guided optimization
        """)
        
        # Show guided mode features
        st.markdown("""
        ### 🎯 Guided Mode Features:
        - **Step-by-step guidance** through each process
        - **Intelligent tips and warnings** based on your experience level
        - **Progress tracking** with visual indicators
        - **Contextual help** at every step
        - **Quality validation** and recommendations
        """)
    else:
        st.info("Use the pages in the left sidebar to navigate. Enable Guided Mode for step-by-step assistance.")
        st.write("Use the pages in the left sidebar to navigate:")
        st.markdown("""
        - Analyze DXF: Upload a DXF, group parts, select groups, nest and export artifacts. Advanced Toolpath is integrated here.
        - Image to DXF: Convert images to DXF via OpenCV and feed into analysis.
        - Designer: Build waterjet-aware prompts and generate a stub image.
        - Nesting: Place parts onto a sheet (rect/irregular heuristic) and export nested DXF.
        """)
    
    # Store guided mode in session state
    st.session_state.guided_mode = guided_mode


if __name__ == "__main__":
    main()


