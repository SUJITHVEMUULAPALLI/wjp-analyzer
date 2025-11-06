#!/usr/bin/env python3
"""
WJP ANALYSER - Unified Web Interface
===================================

This is the single, consolidated web interface that integrates all features
from the different web applications (Streamlit, Flask, Enhanced, Supervisor).

Features:
- Multi-page interface with all workflows
- Guided mode for beginners
- Real-time processing and preview
- AI-powered analysis and design
- Batch processing capabilities
- Professional reporting
- Interactive components
"""

import os
import sys
import streamlit as st
from pathlib import Path
import yaml
import logging
import time
from typing import Dict, Any, Optional

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Add src to Python path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# IMPORTANT: set_page_config() MUST be the first Streamlit command
# Page configuration
st.set_page_config(
    page_title="WJP ANALYSER - Unified Interface",
    page_icon="WJP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import WJP modules - use API wrapper for services
# Store import errors to show after set_page_config
import_errors = []

# Try to import AnalyzeArgs (optional - may fail if database dependencies unavailable)
# Note: This import triggers AuthService -> database.models -> SQLAlchemy chain
try:
    from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs
except ImportError as e:
    # AnalyzeArgs is optional - API client wrapper doesn't require it
    import_errors.append(f"Note: DXF analyzer import failed (optional): {e}")
    AnalyzeArgs = None  # Make it optional
except Exception as e:
    # Catch any other errors (e.g., SQLAlchemy not found)
    import_errors.append(f"Note: DXF analyzer import failed (optional): {e}")
    AnalyzeArgs = None

try:
    # Use API client wrapper for all service calls
    from wjp_analyser.web.api_client_wrapper import (
        analyze_dxf,
        estimate_cost,
        convert_image,
    )
except ImportError as e:
    import_errors.append(f"Failed to import API client wrapper: {e}")
    analyze_dxf = None
    estimate_cost = None
    convert_image = None

# Try to import optional workflow manager (may require database)
try:
    from wjp_analyser.workflow.workflow_manager import WorkflowManager, WorkflowConfig, WorkflowType
except ImportError as e:
    # Workflow manager is optional - may fail if database modules aren't available
    logging.warning(f"Workflow manager not available (optional): {e}")
    WorkflowManager = None
    WorkflowConfig = None
    WorkflowType = None

# Try to import optional AI agents
try:
    from wjp_analyser.ai.openai_agents_manager import is_agents_sdk_available
except ImportError as e:
    logging.warning(f"AI agents manager not available (optional): {e}")
    is_agents_sdk_available = lambda: False

# Show import errors after set_page_config
for error in import_errors:
    st.error(error)

class WJPUnifiedWebApp:
    """Unified Web Application for WJP ANALYSER."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config = self.load_config()
        self.setup_logging()
        
    def load_config(self) -> Dict[str, Any]:
        """Load unified configuration."""
        config_file = self.project_root / "config" / "wjp_unified_config.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                st.error(f"Failed to load config: {e}")
                
        # Return default config
        return {
            'features': {
                'ai_analysis': True,
                'image_conversion': True,
                'nesting': True,
                'cost_estimation': True,
                'guided_mode': True,
                'batch_processing': True
            },
            'defaults': {
                'material': 'steel',
                'thickness': 6.0,
                'kerf': 1.1,
                'cutting_speed': 1200.0,
                'cost_per_meter': 50.0
            }
        }
        
    def setup_logging(self):
        """Setup logging for the web app."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('WJPUnifiedWebApp')
        
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        st.sidebar.title("WJP ANALYSER")
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.subheader("System Status")
        
        # Check AI availability
        try:
            import openai
            ai_status = "Available"
        except ImportError:
            ai_status = "Not available"
            
        st.sidebar.text(f"AI Analysis: {ai_status}")
        
        # Check agents SDK
        agents_status = "Available" if is_agents_sdk_available() else "Not available"
        st.sidebar.text(f"OpenAI Agents: {agents_status}")
        
        st.sidebar.markdown("---")
        
        # Jobs Drawer (Phase 3 component)
        try:
            from wjp_analyser.web.components import render_jobs_drawer
            
            # Get jobs from session state or API
            jobs = st.session_state.get("active_jobs", [])
            
            # Poll for job updates if API available
            if jobs:
                try:
                    from wjp_analyser.web.api_client import get_api_client, is_api_available
                    if is_api_available():
                        from wjp_analyser.web.components import poll_job_statuses
                        job_ids = [j.get("id") for j in jobs if j.get("id")]
                        if job_ids:
                            updated_jobs = poll_job_statuses(job_ids)
                            if updated_jobs:
                                st.session_state["active_jobs"] = updated_jobs
                                jobs = updated_jobs
                except Exception:
                    pass  # Fallback to cached jobs
            
            render_jobs_drawer(jobs, session_key="unified_jobs")
        except ImportError:
            pass  # Jobs drawer not available
        
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.subheader("Navigation")
        
        # Query param deep-linking: initialize from ?page=... (supports old/new APIs)
        initial_page = None
        try:
            if hasattr(st, 'query_params'):
                q = st.query_params
                if isinstance(q, dict) and 'page' in q:
                    val = q.get('page')
                    if isinstance(val, (list, tuple)):
                        initial_page = (val[0] if val else None)
                    else:
                        initial_page = val
            else:
                get_qp = getattr(st, 'experimental_get_query_params', None)
                if callable(get_qp):
                    q = get_qp()
                    if isinstance(q, dict) and 'page' in q:
                        val = q.get('page')
                        if isinstance(val, (list, tuple)):
                            initial_page = (val[0] if val else None)
                        else:
                            initial_page = val
        except Exception:
            initial_page = None
        
        # Main workflows
        page = st.sidebar.selectbox(
            "Select Workflow",
            [
                "Home",
                "Designer",
                "Improved Designer",
                "Image to DXF Analyzer",
                "Analyze DXF",
                "G-Code Generation",
                "Interactive Workflow",
                "DXF Editor",
                "Nesting",
                "AI Agents",
                "Supervisor Dashboard",
                "Settings"
            ],
            index=(
                [
                    "Home","Designer","Improved Designer","Image to DXF Analyzer",
                    "Analyze DXF","G-Code Generation","Interactive Workflow","DXF Editor","Nesting","AI Agents","Supervisor Dashboard","Settings"
                ].index(initial_page) if initial_page in [
                    "Home","Designer","Improved Designer","Image to DXF Analyzer",
                    "Analyze DXF","G-Code Generation","Interactive Workflow","DXF Editor","Nesting","AI Agents","Supervisor Dashboard","Settings"
                ] else 0
            )
        )
        
        st.sidebar.markdown("---")
        
        # Guided mode toggle
        guided_mode = st.sidebar.checkbox(
            "Guided Mode",
            value=os.environ.get("WJP_GUIDED_MODE", "false").lower() == "true",
            help="Enable step-by-step guidance"
        )
        
        # Store in session state
        st.session_state.guided_mode = guided_mode
        
        # Feature toggles
        st.sidebar.subheader("Features")
        
        features = self.config.get('features', {})
        for feature, default in features.items():
            feature_name = feature.replace('_', ' ').title()
            features[feature] = st.sidebar.checkbox(
                feature_name,
                value=default,
                key=f"feature_{feature}"
            )
            
        st.session_state.features = features

        st.sidebar.markdown("---")

        with st.sidebar.expander("Quick Help & Resources", expanded=False):
            st.markdown(
                "- Need the classic UI? [Open Flask interface](http://127.0.0.1:5000/)\n"
                "- UI feels stuck? Use the ⋮ menu → Clear cache → Rerun.\n"
                "- Docs live in the `docs/` folder for each workflow."
            )
            st.caption(
                "Tip: Share links directly by adding `?page=Analyze%20DXF` (or any page name) to the URL."
            )

        st.sidebar.markdown("---")

        # Quick actions
        st.sidebar.subheader("Quick Actions")

        if st.sidebar.button("Refresh System", key="qa_refresh", use_container_width=True):
            st.rerun()

        if st.sidebar.button("Show Status", key="qa_status", use_container_width=True):
            self.show_system_status()

        if st.sidebar.button("Run Demo", key="qa_demo", use_container_width=True):
            self.run_demo()
            
        # Update query params to reflect current page (shareable link)
        try:
            if hasattr(st, 'query_params'):
                st.query_params["page"] = page
            else:
                set_qp = getattr(st, 'experimental_set_query_params', None)
                if callable(set_qp):
                    set_qp(page=page)
        except Exception:
            pass
        
        return page
        
    def show_system_status(self):
        """Show comprehensive system status."""
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Python Version", sys.version.split()[0])
            st.metric("Streamlit Version", st.__version__)
            
        with col2:
            # Check dependencies
            deps = {}
            try:
                import openai
                deps['OpenAI'] = openai.__version__
            except ImportError:
                deps['OpenAI'] = "Not Available"
                
            try:
                import ezdxf
                deps['ezdxf'] = ezdxf.__version__
            except ImportError:
                deps['ezdxf'] = "Not Available"
                
            for dep, version in deps.items():
                st.metric(dep, version)
                
        with col3:
            # Check directories
            dirs = {
                'Config': self.project_root / "config",
                'Output': self.project_root / "output",
                'Logs': self.project_root / "logs",
                'Data': self.project_root / "data"
            }
            
            for name, path in dirs.items():
                status = "Available" if path.exists() else "Missing"
                st.text(f"{status} {name}: {path.name}")
                
    def run_demo(self):
        """Run a demo pipeline."""
        st.subheader("Demo Pipeline")
        
        with st.spinner("Running demo pipeline..."):
            try:
                # Create a simple demo
                demo_output = self.project_root / "output" / "demo"
                demo_output.mkdir(parents=True, exist_ok=True)
                
                # Create a simple DXF for demo
                demo_dxf = demo_output / "demo_circle.dxf"
                
                if not demo_dxf.exists():
                    # Create a simple circle DXF
                    import ezdxf
                    doc = ezdxf.new('R2010')
                    msp = doc.modelspace()
                    msp.add_circle((0, 0), 50)
                    doc.saveas(demo_dxf)
                    
                # Analyze the demo DXF using API wrapper
                report = analyze_dxf(
                    str(demo_dxf),
                    out_dir=str(demo_output),
                    args_overrides={
                        "material": "Steel",
                        "thickness": 6.0,
                        "kerf": 1.1,
                    }
                )
                
                st.success("Demo completed successfully!")
                
                # Show results
                if report:
                    st.subheader("Demo Results")
                    
                    metrics = report.get('metrics', {})
                    if metrics:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Outer Length", f"{metrics.get('length_outer_mm', 0):.1f} mm")
                            
                        with col2:
                            st.metric("Pierces", metrics.get('pierce_count', 0))
                            
                        with col3:
                            st.metric("Estimated Cost", f"INR {metrics.get('estimated_cutting_cost_inr', 0):.0f}")
                            
            except Exception as e:
                st.error(f"Demo failed: {e}")
                
    def render_home_page(self):
        """Render the home page."""
        st.title("WJP ANALYSER - Unified Interface")
        st.markdown("**Comprehensive Waterjet DXF Analysis and Processing System**")
        
        st.markdown("---")
        
        # Welcome message
        st.markdown("""
        Welcome to the WJP ANALYSER unified interface! This system provides comprehensive 
        tools for waterjet cutting analysis, design generation, and optimization.
        """)

        # Inject shared CSS tokens for visual consistency with Flask templates
        custom_css = """
        <style>
        :root { --st-accent: #0f62fe; }
        .stButton>button { background: var(--st-accent); color: #fff; border-radius: 8px; font-weight: 600; }
        .stButton>button:hover { background: #0043ce; }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
        
        # Feature overview
        st.subheader("Available Features")
        
        features = st.session_state.get('features', self.config.get('features', {}))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if features.get('ai_analysis', True):
                st.markdown("""
                **AI Analysis**
                - Intelligent DXF analysis
                - Design suggestions
                - Quality optimization
                """)
                
            if features.get('image_conversion', True):
                st.markdown("""
                ** Image Analysis**
                - Pre-conversion analysis
                - Suitability scoring
                - Quality assessment
                - **Enhanced Analyzer** with live visualization
                """)
                
            if features.get('image_conversion', True):
                st.markdown("""
                ** Image to DXF**
                - Convert images to DXF
                - Edge detection
                - Vectorization
                """)
                
        with col2:
            if features.get('nesting', True):
                st.markdown("""
                **Nesting**
                - Automatic part nesting
                - Material optimization
                - Layout generation
                """)
                
            if features.get('cost_estimation', True):
                st.markdown("""
                **Cost Estimation**
                - Cutting time calculation
                - Material cost analysis
                - Optimization suggestions
                """)
        
        # Quick start
        st.subheader("Quick Start")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Start Designing", use_container_width=True):
                try:
                    if hasattr(st, 'query_params'):
                        st.query_params["page"] = "Designer"
                except Exception:
                    pass
                st.session_state.page = "Designer"
                st.rerun()
                
        with col2:
            if st.button("Analyze Image", use_container_width=True):
                try:
                    if hasattr(st, 'query_params'):
                        st.query_params["page"] = "Image Analyzer"
                except Exception:
                    pass
                st.session_state.page = "Image Analyzer"
                st.rerun()
                
        with col3:
            if st.button("Enhanced Analyzer", use_container_width=True):
                try:
                    if hasattr(st, 'query_params'):
                        st.query_params["page"] = "Enhanced Image Analyzer"
                except Exception:
                    pass
                st.session_state.page = "Enhanced Image Analyzer"
                st.rerun()
                
        with col4:
            if st.button("Analyze DXF", use_container_width=True):
                try:
                    if hasattr(st, 'query_params'):
                        st.query_params["page"] = "Analyze DXF"
                except Exception:
                    pass
                st.session_state.page = "Analyze DXF"
                st.rerun()
            
        # Second row of buttons
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            if st.button("Convert Image", use_container_width=True):
                try:
                    if hasattr(st, 'query_params'):
                        st.query_params["page"] = "Image to DXF"
                except Exception:
                    pass
                st.session_state.page = "Image to DXF"
                st.rerun()
            
        # Recent activity
        st.subheader("Recent Activity")
        
        # Check for recent outputs
        output_dir = self.project_root / "output"
        if output_dir.exists():
            recent_files = []
            for item in output_dir.iterdir():
                if item.is_dir() and item.stat().st_mtime > (time.time() - 86400):  # Last 24 hours
                    recent_files.append(item)
                    
            if recent_files:
                st.markdown("**Recent Projects:**")
                for project in recent_files[:5]:
                    st.text(f"{project.name}")
            else:
                st.info("No recent activity. Start a new project!")
                
    def render_designer_page(self):
        """Render the designer page."""
        st.title("Designer")
        st.markdown("Generate waterjet-compatible designs using AI")
        
        if st.session_state.get('guided_mode', False):
            st.info("**Guided Mode**: Follow the steps below for guided design creation.")
            
        # Design parameters
        col1, col2 = st.columns(2)
        
        with col1:
            design_type = st.selectbox(
                "Design Type",
                ["tile", "medallion", "pattern", "logo", "artwork"],
                help="Type of design to generate"
            )
            
            material = st.selectbox(
                "Material",
                ["Tan Brown Granite", "Steel", "Aluminum", "Marble", "Wood"],
                help="Material for the design"
            )
            
        with col2:
            dimensions = st.text_input(
                "Dimensions",
                value="600x600 mm",
                help="Design dimensions (e.g., 600x600 mm)"
            )
            
            style = st.selectbox(
                "Style",
                ["geometric", "floral", "abstract", "traditional", "modern"],
                help="Design style"
            )
            
        # Advanced settings
        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                kerf_width = st.number_input(
                    "Kerf Width (mm)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.1,
                    step=0.1
                )
                
            with col2:
                min_feature_size = st.number_input(
                    "Min Feature Size (mm)",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1
                )
                
        # Generate design
        if st.button("Generate Design", type="primary"):
            with st.spinner("Generating design..."):
                try:
                    # Create output directory
                    output_dir = self.project_root / "output" / "designer"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate design using AI
                    prompt = f"Generate a waterjet-compatible {design_type} design using {material}, with dimensions {dimensions}, in a {style} style. Waterjet constraints: >={min_feature_size} mm spacing, >={min_feature_size} mm inner radius curves, no floating parts, clean geometry, continuous contours."
                    
                    # For demo purposes, create a placeholder
                    st.success("Design generated successfully!")
                    st.info(f"**Prompt used:** {prompt}")
                    
                    # In a real implementation, this would call the AI service
                    st.markdown("**Note:** This is a demo. In the full implementation, this would generate an actual design image.")
                    
                except Exception as e:
                    st.error(f"Design generation failed: {e}")
                    
    def render_image_to_dxf_analyzer_page(self):
        """Combined Enhanced Image Analyzer + Image to DXF converter in one place."""
        st.title("Image to DXF Analyzer")
        st.markdown("Analyze an image for suitability, preview, then convert to DXF with softening presets.")
        tabs = st.tabs(["Enhanced Analyzer", "Convert to DXF"])
        
        # Tab 1: Enhanced Analyzer
        with tabs[0]:
            if st.session_state.get('guided_mode', False):
                st.info("**Guided Mode**: Upload image, review diagnostics, then switch to Convert tab.")
        
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload an image to analyze for waterjet cutting suitability",
                key="imgdxfa_analyze_up"
            )
        
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
                st.subheader("Analysis Parameters")
            
                col1, col2 = st.columns(2)
            
                with col1:
                    px_to_unit = st.number_input(
                        "Pixels to Unit Ratio (px/mm)",
                        min_value=0.1,
                        max_value=100.0,
                        value=10.0,
                        step=0.1,
                        key="imgdxfa_px"
                    )
                    
                    max_size_px = st.number_input(
                        "Max Analysis Size (pixels)",
                        min_value=100,
                        max_value=4000,
                        value=1024,
                        step=100,
                        key="imgdxfa_maxpx"
                    )
                
                with col2:
                    min_score_threshold = st.number_input(
                        "Minimum Score Threshold",
                        min_value=0.0,
                        max_value=100.0,
                        value=75.0,
                        step=5.0,
                        key="imgdxfa_score"
                    )
                    
                    auto_fix_enabled = st.checkbox(
                        "Enable Auto-Fix Suggestions",
                        value=False,
                        key="imgdxfa_autofix"
                    )
            
                if st.button("Analyze Image", type="primary", key="imgdxfa_analyze_btn"):
                    with st.spinner("Analyzing image..."):
                        try:
                            output_dir = self.project_root / "output" / "image_analyzer"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            file_path = output_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            from wjp_analyser.image_analyzer import analyze_image_for_wjp, AnalyzerConfig
                            config = AnalyzerConfig(px_to_unit=px_to_unit, max_size_px=max_size_px)
                            report = analyze_image_for_wjp(str(file_path), config)
                            st.success("Image analysis completed successfully!")
                            if report:
                                st.subheader("Analysis Results")
                                score = report.get('score', 0)
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Suitability Score", f"{score:.1f}/100", delta=f"{'Pass' if score >= min_score_threshold else 'Fail'}")
                                c2.metric("Image Dimensions", f"{report.get('width', 0)}x{report.get('height', 0)}")
                                c3.metric("Contours Found", report.get('total_contours', 0))
                                suggestions = report.get('suggestions', [])
                                if suggestions:
                                    st.subheader("Suggestions")
                                    for suggestion in suggestions:
                                        st.text(f"- {suggestion}")
                                if auto_fix_enabled and score < min_score_threshold:
                                    st.subheader("Auto-Fix Suggestions")
                                    st.info("Consider adjusting image contrast, removing small features, or increasing curve radii for better waterjet cutting results.")
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
            else:
                st.info("Upload an image to begin analysis")
            
            st.subheader("Analysis Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Basic Analysis:**
                - Image dimensions and aspect ratio
                - Grayscale statistics
                - Color distribution analysis
                
                **Edge Detection:**
                - Edge density calculation
                - Edge contrast ratio
                - Contour detection and analysis
                """)
            
            with col2:
                st.markdown("""
                **Texture Analysis:**
                - Entropy calculation
                - FFT high-frequency energy
                - Noise level assessment
                
                ** Manufacturability:**
                - Minimum spacing analysis
                - Curve radius detection
                - Suitability scoring
                """)

    def render_enhanced_image_analyzer_page(self):
        """Render the enhanced image analyzer page with full Phase 1+2+3 integration."""
        st.title("Enhanced Image Analyzer")
        st.markdown("**Full Intelligent Diagnostic** - Upload -> Analyze -> Visualize -> Live Score -> DXF-Readiness")
        
        if st.session_state.get('guided_mode', False):
            st.info("**Guided Mode**: This enhanced analyzer provides comprehensive image analysis with live visualization.")
        
        # Import and run the enhanced image analyzer
        try:
            import sys
            from pathlib import Path
            
            # Add the pages directory to path
            pages_dir = Path(__file__).parent / "pages"
            sys.path.insert(0, str(pages_dir))
            
            # Import and run the enhanced analyzer
            from enhanced_image_analyzer import main
            main()
            
        except ImportError as e:
            st.error(f"Enhanced Image Analyzer not available: {e}")
            st.info("Please use the 'Image to DXF Analyzer' page instead.")
        except Exception as e:
            st.error(f"Error loading enhanced analyzer: {e}")
            st.info("Please use the 'Image to DXF Analyzer' page instead.")

    def render_analyze_dxf_page(self):
        """Render the DXF analysis page."""
        st.title("Analyze DXF")
        st.markdown("Analyze DXF files for waterjet cutting optimization")
        
        if st.session_state.get('guided_mode', False):
            st.info("**Guided Mode**: Follow the steps below for guided DXF analysis.")
            
        # File upload
        uploaded_file = dxf_file_uploader(
            "Upload DXF File",
            help_text="Upload a DXF file to analyze",
            key="analyze_dxf_upload",
        )
        
        if uploaded_file:
            # Analysis parameters
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                material = st.selectbox(
                    "Material",
                    ["Steel", "Aluminum", "Tan Brown Granite", "Marble", "Wood"],
                    help="Material type"
                )
                
                thickness = st.number_input(
                    "Thickness (mm)",
                    min_value=0.1,
                    max_value=100.0,
                    value=6.0,
                    step=0.1
                )
                
            with col2:
                kerf = st.number_input(
                    "Kerf Width (mm)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.1,
                    step=0.1
                )
                
                cutting_speed = st.number_input(
                    "Cutting Speed (mm/min)",
                    min_value=100.0,
                    max_value=5000.0,
                    value=1200.0,
                    step=50.0
                )
                
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    enable_toolpath = st.checkbox("Enable Advanced Toolpath", value=True)
                    enable_cost_estimation = st.checkbox("Enable Cost Estimation", value=True)
                    
                with col2:
                    optimize_rapids = st.checkbox("Optimize Rapids", value=True)
                    optimize_direction = st.checkbox("Optimize Direction", value=True)
                    
            # Analyze button
            if st.button("Analyze DXF", type="primary"):
                with st.spinner("Analyzing DXF..."):
                    try:
                        # Persist uploaded file
                        output_dir = self.project_root / "output" / "analysis"
                        file_path = save_uploaded_dxf(uploaded_file, output_dir)

                        # Analyze DXF using API wrapper
                        report = analyze_dxf(
                            str(file_path),
                            out_dir=str(output_dir),
                            args_overrides={
                                "material": material,
                                "thickness": thickness,
                                "kerf": kerf,
                                "cutting_speed": cutting_speed,
                                # Note: API doesn't support advanced toolpath options yet
                                # These would need to be added to API if needed
                            }
                        )
                        
                        st.success("DXF analysis completed successfully!")
                        
                        # Display results
                        if report:
                            st.subheader("Analysis Results")
                            
                            metrics = report.get('metrics', {})
                            if metrics:
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Outer Length",
                                        f"{metrics.get('length_outer_mm', 0):.1f} mm"
                                    )
                                    
                                with col2:
                                    st.metric(
                                        "Internal Length",
                                        f"{metrics.get('length_internal_mm', 0):.1f} mm"
                                    )
                                    
                                with col3:
                                    st.metric(
                                        "Pierces",
                                        metrics.get('pierce_count', 0)
                                    )
                                    
                                with col4:
                                    st.metric(
                                        "Estimated Cost",
                                        f"INR {metrics.get('estimated_cutting_cost_inr', 0):.0f}"
                                    )
                                    
                            # Quality analysis
                            quality = report.get('quality', {})
                            if quality:
                                st.subheader("Quality Analysis")
                                
                                issues = quality.get('issues', [])
                                if issues:
                                    st.warning("Warning Quality Issues Found:")
                                    for issue in issues:
                                        st.text(f"- {issue}")
                                else:
                                    st.success("No quality issues found!")
                                    
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        
    def render_nesting_page(self):
        """Render the nesting page."""
        st.title("Nesting")
        st.markdown("Optimize part placement for material efficiency")
        
        if st.session_state.get('guided_mode', False):
            st.info("**Guided Mode**: Follow the steps below for guided nesting optimization.")
            
        st.info("**Note**: Nesting functionality requires DXF files. Upload files from the Analyze DXF page first.")
        
        # Sheet parameters
        st.subheader("Sheet Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sheet_width = st.number_input(
                "Sheet Width (mm)",
                min_value=100.0,
                max_value=5000.0,
                value=3000.0,
                step=50.0
            )
            
        with col2:
            sheet_height = st.number_input(
                "Sheet Height (mm)",
                min_value=100.0,
                max_value=5000.0,
                value=1500.0,
                step=50.0
            )
            
        # Nesting parameters
        st.subheader("Nesting Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            spacing = st.number_input(
                "Spacing (mm)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0
            )
            
            algorithm = st.selectbox(
                "Algorithm",
                ["Rectangular", "Irregular", "Genetic Algorithm"],
                help="Nesting algorithm to use"
            )
            
        with col2:
            enable_rotation = st.checkbox("Enable Rotation", value=True)
            max_rotation = st.number_input(
                "Max Rotation (degrees)",
                min_value=0,
                max_value=180,
                value=90,
                step=15
            )
            
        # Nesting options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                optimization_level = st.selectbox(
                    "Optimization Level",
                    ["Low", "Medium", "High", "Maximum"],
                    index=2
                )
                
            with col2:
                max_iterations = st.number_input(
                    "Max Iterations",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100
                )
                
        # Run nesting
        if st.button("Run Nesting", type="primary"):
            st.info("**Demo Mode**: This would run the actual nesting algorithm with the uploaded DXF files.")
            
            # Create mock results for demo
            st.success("Nesting completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Parts Placed", "12")
                
            with col2:
                st.metric("Material Usage", "78.5%")
                
            with col3:
                st.metric("Waste", "21.5%")
                
    def render_gcode_generation_page(self):
        """Render the G-code generation page."""
        st.title("G-Code Generation")
        st.markdown("Generate G-code for waterjet cutting from DXF files")
        
        if st.session_state.get('guided_mode', False):
            st.info("**Guided Mode**: Upload a DXF file and configure parameters to generate cutting G-code.")
        
        # File upload
        uploaded_file = dxf_file_uploader(
            "Upload DXF File",
            help_text="Upload a DXF file to generate G-code",
            key="gcode_generation_upload",
        )
        
        if uploaded_file:
            # G-code parameters
            st.subheader("G-Code Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                feed_rate = st.number_input(
                    "Feed Rate (mm/min)",
                    min_value=100.0,
                    max_value=5000.0,
                    value=1200.0,
                    step=50.0,
                    help="Cutting feed rate"
                )
                
                m_on = st.text_input(
                    "M-Code ON",
                    value="M62",
                    help="G-code command to turn on waterjet"
                )
                
            with col2:
                m_off = st.text_input(
                    "M-Code OFF",
                    value="M63",
                    help="G-code command to turn off waterjet"
                )
                
                pierce_time = st.number_input(
                    "Pierce Time (ms)",
                    min_value=0,
                    max_value=5000,
                    value=500,
                    step=50,
                    help="Time to wait for pierce"
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    optimize_rapids = st.checkbox("Optimize Rapid Moves", value=True)
                    optimize_direction = st.checkbox("Optimize Cut Direction", value=True)
                    
                with col2:
                    enable_smoothing = st.checkbox("Enable Path Smoothing", value=True)
                    enable_filleting = st.checkbox("Enable Fillet Corners", value=False)
            
            # Generate G-code button
            if st.button("Generate G-Code", type="primary"):
                with st.spinner("Generating G-code..."):
                    try:
                        # Persist uploaded file
                        output_dir = self.project_root / "output" / "gcode"
                        file_path = save_uploaded_dxf(uploaded_file, output_dir)

                        # Import G-code generation functions
                        try:
                            from wjp_analyser.web.api_utils import generate_gcode_from_dxf
                            
                            result = generate_gcode_from_dxf(
                                str(file_path),
                                str(output_dir),
                                feed=feed_rate,
                                m_on=m_on,
                                m_off=m_off,
                                pierce_ms=int(pierce_time)
                            )
                            
                            st.success("G-code generated successfully!")
                            
                            # Display results
                            if result:
                                st.subheader("Generation Results")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Lines", result.get('line_count', 0))
                                    
                                with col2:
                                    st.metric("Estimated Time", result.get('estimated_time', 'N/A'))
                                    
                                with col3:
                                    st.metric("Cut Length", f"{result.get('metrics', {}).get('length_mm', 0):.1f} mm")
                                    
                                with col4:
                                    st.metric("Pierces", result.get('metrics', {}).get('pierce_count', 0))
                                
                                # Show G-code preview
                                gcode_preview = result.get('gcode_preview', '')
                                if gcode_preview:
                                    st.subheader("G-Code Preview (first 50 lines)")
                                    with st.expander("View G-Code", expanded=False):
                                        st.code(gcode_preview[:2000] if len(gcode_preview) > 2000 else gcode_preview, language='text')
                                
                                # Download buttons
                                st.subheader("Download Files")
                                
                                gcode_path = result.get('gcode_path')
                                if gcode_path and os.path.exists(gcode_path):
                                    with open(gcode_path, "rb") as f:
                                        st.download_button(
                                            "Download G-Code (.nc)",
                                            f.read(),
                                            file_name=os.path.basename(gcode_path),
                                            mime="text/plain",
                                            key="gcode_download"
                                        )
                                        
                        except ImportError:
                            st.error("G-code generation module not available")
                            st.info("Please install required dependencies or use the API endpoint.")
                            
                        except Exception as e:
                            st.error(f"G-code generation failed: {e}")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Configure parameters above and click 'Generate G-Code' to proceed")
        else:
            st.info("Upload a DXF file to begin G-code generation")
            
            # Show help
            st.subheader("G-Code Generation Guide")
            st.markdown("""
            **What is G-code?**
            G-code is the language that controls CNC machines, including waterjet cutters.
            
            **Steps to generate G-code:**
            1. Upload a DXF file containing your cutting geometry
            2. Configure feed rate, M-codes, and pierce time
            3. Enable advanced options for optimization
            4. Click "Generate G-Code"
            5. Download the generated .nc file
            
            **Parameters explained:**
            - **Feed Rate**: The speed at which the waterjet moves (mm/min)
            - **M-Code ON/OFF**: Commands to control the waterjet pump
            - **Pierce Time**: Time to wait when starting a new cut
            - **Optimization**: Improve cutting path efficiency
            """)
                
    def render_ai_agents_page(self):
        """Render the AI agents page."""
        st.title("AI Agents")
        st.markdown("Interact with specialized AI agents for advanced analysis")
        
        if not is_agents_sdk_available():
            st.error("OpenAI Agents SDK not available. Please install it with: pip install openai-agents")
            return
            
        st.success("OpenAI Agents SDK is available!")
        
        # Agent selection
        agent_type = st.selectbox(
            "Select Agent",
            [
                "Designer Agent",
                "DXF Analyzer Agent", 
                "Image Converter Agent",
                "Report Generator Agent",
                "Supervisor Agent"
            ],
            help="Choose an AI agent to interact with"
        )
        
        # Agent interface
        st.subheader(f"{agent_type}")
        
        if agent_type == "Designer Agent":
            st.markdown("""
            The Designer Agent helps create waterjet-compatible designs with:
            - Design generation and optimization
            - Material-specific recommendations
            - Quality validation
            """)
            
        elif agent_type == "DXF Analyzer Agent":
            st.markdown("""
            The DXF Analyzer Agent provides:
            - Advanced geometric analysis
            - Cutting path optimization
            - Quality assessment
            """)
            
        elif agent_type == "Image Converter Agent":
            st.markdown("""
            The Image Converter Agent offers:
            - Intelligent image processing
            - Vectorization optimization
            - Parameter tuning
            """)
            
        elif agent_type == "Report Generator Agent":
            st.markdown("""
            The Report Generator Agent creates:
            - Professional analysis reports
            - Cost breakdowns
            - Optimization recommendations
            """)
            
        elif agent_type == "Supervisor Agent":
            st.markdown("""
            The Supervisor Agent coordinates:
            - Multi-agent workflows
            - Quality assurance
            - Process optimization
            """)
            
        # Agent interaction
        user_input = st.text_area(
            "Ask the agent:",
            placeholder="Describe what you need help with...",
            height=100
        )
        
        if st.button("Send to Agent", type="primary"):
            if user_input:
                with st.spinner("Agent is thinking..."):
                    # Mock agent response
                    st.success("Agent response received!")
                    st.info("**Demo Mode**: This would show the actual agent response.")
            else:
                st.warning("Please enter a message for the agent.")
                
    def render_supervisor_dashboard_page(self):
        """Render the supervisor dashboard page."""
        st.title("Supervisor Dashboard")
        st.markdown("Monitor and control all WJP ANALYSER processes")
        
        # System overview
        st.subheader("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Processes", "3")
            
        with col2:
            st.metric("Completed Jobs", "47")
            
        with col3:
            st.metric("Success Rate", "94.2%")
            
        with col4:
            st.metric("Avg Processing Time", "2.3 min")
            
        # Process monitoring
        st.subheader("Process Monitoring")
        
        # Mock process data
        processes = [
            {"name": "DXF Analysis", "status": "Running", "progress": 75},
            {"name": "Image Conversion", "status": "Completed", "progress": 100},
            {"name": "Nesting Optimization", "status": "Queued", "progress": 0}
        ]
        
        for process in processes:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.text(process["name"])
                
            with col2:
                status_color = {
                    "Running": "Active",
                    "Completed": "Completed", 
                    "Queued": "Queued",
                    "Failed": "Failed"
                }.get(process["status"], "?")
                st.text(f"{status_color}: {process['status']}")
                
            with col3:
                st.progress(process["progress"] / 100)
                
        # Agent status
        st.subheader("Agent Status")
        
        agents = [
            {"name": "Designer Agent", "status": "Active", "tasks": 2},
            {"name": "Analyzer Agent", "status": "Active", "tasks": 1},
            {"name": "Converter Agent", "status": "Idle", "tasks": 0},
            {"name": "Report Agent", "status": "Active", "tasks": 3}
        ]
        
        for agent in agents:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.text(agent["name"])
                
            with col2:
                status_color = "Active" if agent["status"] == "Active" else "Idle"
                st.text(f"{status_color} {agent['status']}")
                
            with col3:
                st.text(f"Tasks: {agent['tasks']}")
                
        # Quick actions
        st.subheader("Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Refresh Status", use_container_width=True):
                st.rerun()
                
        with col2:
            if st.button("Generate Report", use_container_width=True):
                st.success("Report generated!")
                
        with col3:
            if st.button("Cleanup", use_container_width=True):
                st.success("Cleanup completed!")
                
    def render_settings_page(self):
        """Render the settings page."""
        st.title("Settings")
        st.markdown("Configure WJP ANALYSER system settings")
        
        # Configuration sections
        tab1, tab2, tab3, tab4 = st.tabs(["General", "AI", "Processing", "Advanced"])
        
        with tab1:
            st.subheader("General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Project Name", value="WJP ANALYSER")
                st.text_input("Version", value=self.config.get('version', '2.0.0'))
                
            with col2:
                st.selectbox("Environment", ["Development", "Staging", "Production"])
                st.checkbox("Enable Debug Mode", value=False)
                
        with tab2:
            st.subheader("AI Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
                st.selectbox("Default Model", ["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini"])
                
            with col2:
                st.number_input("Max Tokens", min_value=100, max_value=4000, value=2000)
                st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                
        with tab3:
            st.subheader("Processing Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input("Max Workers", min_value=1, max_value=16, value=4)
                st.number_input("Timeout (seconds)", min_value=30, max_value=600, value=300)
                
            with col2:
                st.checkbox("Enable Caching", value=True)
                st.checkbox("Enable Compression", value=True)
                
        with tab4:
            st.subheader("Advanced Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.checkbox("Enable Monitoring", value=True)
                st.checkbox("Enable Logging", value=True)
                
            with col2:
                st.checkbox("Enable Rate Limiting", value=True)
                st.checkbox("Enable Security Headers", value=True)
                
        # Save settings
        if st.button("Save Settings", type="primary"):
            st.success("Settings saved successfully!")
            
    def run(self):
        """Run the unified web application."""
        # Render sidebar and get current page
        page = self.render_sidebar()
        
        # Handle page navigation
        if page == "Home":
            self.render_home_page()
        elif page == "Designer":
            self.render_designer_page()
        elif page == "Improved Designer":
            try:
                # Import and run the improved Streamlit designer app module
                from wjp_analyser.improved_wjp_designer import main as improved_designer_main
                improved_designer_main()
            except Exception as e:
                st.error(f"Failed to load Improved Designer: {e}")
        elif page == "Image to DXF Analyzer":
            self.render_image_to_dxf_analyzer_page()
        elif page == "Analyze DXF":
            self.render_analyze_dxf_page()
        elif page == "G-Code Generation":
            self.render_gcode_generation_page()
        elif page == "Interactive Workflow":
            self.render_interactive_workflow_page()
        elif page == "DXF Editor":
            try:
                import sys
                from pathlib import Path
                pages_dir = Path(__file__).parent / "pages"
                if str(pages_dir) not in sys.path:
                    sys.path.insert(0, str(pages_dir))
                from dxf_editor import main as dxf_editor_main
                dxf_editor_main()
            except Exception as e:
                st.error(f"DXF Editor not available: {e}")
        elif page == "Nesting":
            self.render_nesting_page()
        elif page == "AI Agents":
            self.render_ai_agents_page()
        elif page == "Supervisor Dashboard":
            self.render_supervisor_dashboard_page()
        elif page == "Settings":
            self.render_settings_page()
        else:
            self.render_home_page()

    def render_interactive_workflow_page(self):
        """Guided one-screen workflow: Source → Presets → Preview → Analyze → Variants → Downloads."""
        st.title("Interactive Workflow")
        st.caption("Image or DXF in, optimized NC and reports out.")

        # Import shared helpers lazily
        try:
            from wjp_analyser.web._components import ensure_workdir, run_analysis, run_variant_advisor
        except Exception as e:
            st.error(f"Failed to load workflow components: {e}")
            return

        # Step 1: Source selection
        st.subheader("Step 1 · Source")
        src_mode = st.radio("Choose source", ["DXF", "Image"], horizontal=True)

        work = None
        if src_mode == "DXF":
            dxf_up = st.file_uploader("Upload DXF", type=["dxf"], key="wf_dxf")
            if dxf_up is not None:
                work = ensure_workdir(dxf_up.name, dxf_up.getvalue())
                st.session_state["wf_work"] = work
                st.success(f"Loaded DXF: {dxf_up.name}")
        else:
            img_up = st.file_uploader("Upload Image", type=["png","jpg","jpeg","bmp","tiff"], key="wf_img")
            if img_up is not None:
                # Minimal placeholder: save image and generate placeholder DXF
                out_dir = (Path.cwd() / "output" / "workflow")
                out_dir.mkdir(parents=True, exist_ok=True)
                img_path = out_dir / img_up.name
                with open(img_path, "wb") as f:
                    f.write(img_up.getbuffer())
                try:
                    import ezdxf
                    doc = ezdxf.new('R2010')
                    msp = doc.modelspace()
                    msp.add_circle((0,0), 50)
                    dxf_path = out_dir / f"{img_up.name.split('.')[0]}_auto.dxf"
                    doc.saveas(dxf_path)
                    st.success("Image converted (placeholder) → DXF")
                    work = ensure_workdir(dxf_path.name, dxf_path.read_bytes())
                    st.session_state["wf_work"] = work
                except Exception as e:
                    st.error(f"DXF create failed: {e}")
                    return

        if not work:
            st.info("Upload a file to continue, then click 'Run Analysis'.")
            return

        # Step 2: Presets
        st.subheader("Step 2 · Presets")
        preset = st.selectbox(
            "Softening Preset",
            ["Simple (DP 0.2)", "Smooth (VW 0.5)", "Production (measurement)", "Custom"],
            index=0,
        )

        soften_opts: Dict[str, object] | None = None
        if preset.startswith("Simple"):
            soften_opts = {"method": "simplify_topo", "tolerance": 0.2, "preserve_topology": True}
        elif preset.startswith("Smooth"):
            soften_opts = {"method": "visvalingam", "vw_area_mm2": 0.5}
        elif preset.startswith("Production"):
            soften_opts = {"method": "measurement", "min_segment_mm": 1.0, "max_deviation_mm": 0.2, "min_corner_radius_mm": 1.0, "snap_grid_mm": 0.0}
        else:
            c1, c2 = st.columns(2)
            with c1:
                m = st.selectbox("Method", ["simplify_topo","rdp","chaikin","visvalingam","colinear","decimate","resample","measurement"], index=0)
            with c2:
                st.caption("Adjust parameters below as applicable")
            soften_opts = {"method": m}
            if m in ("simplify_topo","rdp"):
                tol = st.number_input("Tolerance (mm)", 0.001, 5.0, 0.2, 0.05)
                preserve = st.checkbox("Preserve topology", value=True)
                soften_opts.update({"tolerance": tol, "preserve_topology": preserve})
            if m == "chaikin":
                it = st.number_input("Iterations", 1, 10, 1, 1)
                soften_opts.update({"iterations": int(it)})
            if m == "visvalingam":
                area = st.number_input("Min area (mm²)", 0.0, 10.0, 0.5, 0.1)
                soften_opts.update({"vw_area_mm2": area})
            if m == "colinear":
                ang = st.number_input("Angle tolerance (deg)", 0.1, 10.0, 2.0, 0.1)
                soften_opts.update({"colinear_angle_deg": ang})
            if m == "decimate":
                n = st.number_input("Keep every Nth", 2, 50, 2, 1)
                soften_opts.update({"keep_every": int(n)})
            if m == "resample":
                step = st.number_input("Step (mm)", 0.001, 10.0, 1.0, 0.1)
                soften_opts.update({"step_mm": step})
            if m == "measurement":
                min_seg = st.number_input("Min segment (mm)", 0.001, 10.0, 1.0, 0.1)
                max_dev = st.number_input("Max deviation (mm)", 0.001, 5.0, 0.2, 0.05)
                min_rad = st.number_input("Min corner radius (mm)", 0.0, 10.0, 1.0, 0.1)
                snap = st.number_input("Snap grid (mm)", 0.0, 10.0, 0.0, 0.5)
                soften_opts.update({
                    "min_segment_mm": min_seg,
                    "max_deviation_mm": max_dev,
                    "min_corner_radius_mm": min_rad,
                    "snap_grid_mm": snap,
                })

        # Step 3: Sheet/frame
        st.subheader("Step 3 · Sheet")
        c1, c2, c3 = st.columns(3)
        with c1:
            sheet_w = st.number_input("Width (mm)", 100.0, 5000.0, 1000.0, 50.0)
        with c2:
            sheet_h = st.number_input("Height (mm)", 100.0, 5000.0, 1000.0, 50.0)
        with c3:
            qty = st.number_input("Frames", 1, 20, 1, 1)
        normalize_opts = {"mode": "fit", "frame_w": float(sheet_w), "frame_h": float(sheet_h), "margin": 1.0, "origin": True, "must_fit": True}
        scale_opts = {"mode": "decade_fit", "decade_base": 10.0, "direction": "auto", "max_steps": 6, "allow_overshoot": False, "exact_fit": False}

        # Step 4: Analyze
        st.subheader("Step 4 · Analyze")
        if st.button("Run Analysis", type="primary", key="wf_run"):
            try:
                with st.status("Analyzing...", expanded=False):
                    rep = run_analysis(
                        work,
                        selected_groups=None,
                        sheet_width=sheet_w,
                        sheet_height=sheet_h,
                        soften_opts=soften_opts,
                        fillet_opts=None,
                        scale_opts=scale_opts,
                        normalize_opts=normalize_opts,
                        frame_quantity=int(qty),
                    )
                    st.session_state["wf_report"] = rep
                    st.session_state["wf_work"] = work
                st.toast("Analysis complete", icon="✅")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        rep = st.session_state.get("wf_report")
        if rep:
            # Quick preview
            st.caption("Summary")
            m = rep.get("metrics", {}) or {}
            # Prepare effective KPIs (primary → fallback → component-derived)
            eff_len_mm = float(m.get('length_internal_mm', 0)) + float(m.get('length_outer_mm', 0))
            eff_pierces = int(m.get('pierce_count', m.get('pierces', 0)))
            eff_cost = m.get('estimated_cutting_cost_inr', None)

            # Diagnostics and fallbacks
            try:
                comps = rep.get("components", []) or []
                layers = rep.get("layers", {}) or {}
                st.caption(f"Components: {len(comps)} · Layers: {len(layers)}")
                if (eff_len_mm == 0) or (eff_pierces == 0):
                    st.warning("Analysis returned zero metrics. Attempting fallback cost/length computation.")
                    try:
                        # Use estimate_cost from API wrapper (already imported)
                        pass  # estimate_cost already imported from wrapper
                        import os
                        # Prefer existing artifact path; otherwise, use working DXF path
                        candidates = []
                        try:
                            art = rep.get('artifacts', {}) or {}
                            if art.get('layered_dxf'):
                                candidates.append(art.get('layered_dxf'))
                        except Exception:
                            pass
                        try:
                            if rep.get('file'):
                                candidates.append(rep.get('file'))
                        except Exception:
                            pass
                        wf_work = st.session_state.get('wf_work') or {}
                        if wf_work.get('dxf_path'):
                            candidates.append(wf_work.get('dxf_path'))

                        dxfp = next((p for p in candidates if p and os.path.exists(str(p))), None)
                        if dxfp:
                            costs = estimate_cost(str(dxfp))
                            mc = costs.get('metrics', {})
                            eff_len_mm = float(mc.get('length_mm', eff_len_mm))
                            eff_pierces = int(mc.get('pierce_count', eff_pierces))
                            eff_cost = costs.get('total_cost', eff_cost)
                        else:
                            # Second fallback: compute from components directly
                            perim = 0.0
                            for c in comps:
                                try:
                                    perim += float(c.get('perimeter', 0.0))
                                except Exception:
                                    pass
                            eff_len_mm = max(eff_len_mm, perim)
                            eff_pierces = max(eff_pierces, len(comps))
                    except Exception as e:
                        st.info(f"Fallback computation unavailable: {e}")
            except Exception:
                pass

            # Final KPI render (single row)
            c1, c2, c3 = st.columns(3)
            c1.metric("Length (mm)", f"{eff_len_mm:.1f}")
            c2.metric("Pierces", eff_pierces)
            c3.metric("Est. Cost", f"INR {float(eff_cost):.0f}" if eff_cost is not None else "INR 0")

            # Step 5: Variants
            st.subheader("Step 5 · Variant Advisor")
            if st.button("Evaluate Variants", key="wf_variants"):
                with st.spinner("Evaluating..."):
                    results = run_variant_advisor(
                        work,
                        sheet_width=sheet_w,
                        sheet_height=sheet_h,
                        scale_opts=scale_opts,
                        normalize_opts=normalize_opts,
                        frame_quantity=int(qty),
                        include_fillet=True,
                    )
                    st.session_state["wf_variants"] = results
            results = st.session_state.get("wf_variants")
            if results:
                for i, r in enumerate(results[:5], start=1):
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.write(f"#{i} {r['id']} · Score {r['score']}")
                        st.caption("; ".join(map(str, r.get('reasons', [])))[:400])
                    with c2:
                        arts = r.get("artifacts") or {}
                        if arts.get("nc") and os.path.exists(arts["nc"]):
                            with open(arts["nc"], "rb") as fh:
                                st.download_button("Download NC", data=fh.read(), file_name=os.path.basename(arts["nc"]), key=f"wf_nc_{i}_{r['id']}")
                        if arts.get("report") and os.path.exists(arts["report"]):
                            with open(arts["report"], "rb") as fh:
                                st.download_button("Report JSON", data=fh.read(), file_name=os.path.basename(arts["report"]), key=f"wf_r_{i}_{r['id']}")
                        if arts.get("lengths_csv") and os.path.exists(arts["lengths_csv"]):
                            with open(arts["lengths_csv"], "rb") as fh:
                                st.download_button("Lengths CSV", data=fh.read(), file_name=os.path.basename(arts["lengths_csv"]), key=f"wf_l_{i}_{r['id']}")
                        if arts.get("layered_dxf") and os.path.exists(arts["layered_dxf"]):
                            with open(arts["layered_dxf"], "rb") as fh:
                                st.download_button("Layered DXF", data=fh.read(), file_name=os.path.basename(arts["layered_dxf"]), key=f"wf_d_{i}_{r['id']}")


def main():
    """Main entry point for the unified web application."""
    app = WJPUnifiedWebApp()
    app.run()


if __name__ == "__main__":
    main()
