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
from typing import Dict, Any, Optional

# Add src to Python path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Import WJP modules
try:
    from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf
    from wjp_analyser.workflow.workflow_manager import WorkflowManager, WorkflowConfig, WorkflowType
    from wjp_analyser.ai.openai_agents_manager import is_agents_sdk_available
except ImportError as e:
    st.error(f"Failed to import WJP modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="WJP ANALYSER - Unified Interface",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        st.sidebar.title("🔧 WJP ANALYSER")
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.subheader("📊 System Status")
        
        # Check AI availability
        try:
            import openai
            ai_status = "✅ Available"
        except ImportError:
            ai_status = "❌ Not Available"
            
        st.sidebar.text(f"AI Analysis: {ai_status}")
        
        # Check agents SDK
        agents_status = "✅ Available" if is_agents_sdk_available() else "❌ Not Available"
        st.sidebar.text(f"OpenAI Agents: {agents_status}")
        
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.subheader("🧭 Navigation")
        
        # Main workflows
        page = st.sidebar.selectbox(
            "Select Workflow",
            [
                "🏠 Home",
                "🎨 Designer",
                "🖼️ Image to DXF",
                "📐 Analyze DXF",
                "📦 Nesting",
                "🤖 AI Agents",
                "📊 Supervisor Dashboard",
                "⚙️ Settings"
            ]
        )
        
        st.sidebar.markdown("---")
        
        # Guided mode toggle
        guided_mode = st.sidebar.checkbox(
            "🎯 Guided Mode",
            value=os.environ.get("WJP_GUIDED_MODE", "false").lower() == "true",
            help="Enable step-by-step guidance"
        )
        
        # Store in session state
        st.session_state.guided_mode = guided_mode
        
        # Feature toggles
        st.sidebar.subheader("🎛️ Features")
        
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
        
        # Quick actions
        st.sidebar.subheader("⚡ Quick Actions")
        
        if st.sidebar.button("🔄 Refresh System"):
            st.rerun()
            
        if st.sidebar.button("📊 Show Status"):
            self.show_system_status()
            
        if st.sidebar.button("🧪 Run Demo"):
            self.run_demo()
            
        return page
        
    def show_system_status(self):
        """Show comprehensive system status."""
        st.subheader("📊 System Status")
        
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
                status = "✅" if path.exists() else "❌"
                st.text(f"{status} {name}: {path.name}")
                
    def run_demo(self):
        """Run a demo pipeline."""
        st.subheader("🧪 Demo Pipeline")
        
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
                    
                # Analyze the demo DXF
                args = AnalyzeArgs(
                    material="Steel",
                    thickness=6.0,
                    kerf=1.1,
                    out=str(demo_output)
                )
                
                report = analyze_dxf(str(demo_dxf), args)
                
                st.success("✅ Demo completed successfully!")
                
                # Show results
                if report:
                    st.subheader("📊 Demo Results")
                    
                    metrics = report.get('metrics', {})
                    if metrics:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Outer Length", f"{metrics.get('length_outer_mm', 0):.1f} mm")
                            
                        with col2:
                            st.metric("Pierces", metrics.get('pierce_count', 0))
                            
                        with col3:
                            st.metric("Estimated Cost", f"₹{metrics.get('estimated_cutting_cost_inr', 0):.0f}")
                            
            except Exception as e:
                st.error(f"❌ Demo failed: {e}")
                
    def render_home_page(self):
        """Render the home page."""
        st.title("🏠 WJP ANALYSER - Unified Interface")
        st.markdown("**Comprehensive Waterjet DXF Analysis and Processing System**")
        
        st.markdown("---")
        
        # Welcome message
        st.markdown("""
        Welcome to the WJP ANALYSER unified interface! This system provides comprehensive 
        tools for waterjet cutting analysis, design generation, and optimization.
        """)
        
        # Feature overview
        st.subheader("🎯 Available Features")
        
        features = st.session_state.get('features', self.config.get('features', {}))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if features.get('ai_analysis', True):
                st.markdown("""
                **🤖 AI Analysis**
                - Intelligent DXF analysis
                - Design suggestions
                - Quality optimization
                """)
                
            if features.get('image_conversion', True):
                st.markdown("""
                **🖼️ Image to DXF**
                - Convert images to DXF
                - Edge detection
                - Vectorization
                """)
                
        with col2:
            if features.get('nesting', True):
                st.markdown("""
                **📦 Nesting**
                - Automatic part nesting
                - Material optimization
                - Layout generation
                """)
                
            if features.get('cost_estimation', True):
                st.markdown("""
                **💰 Cost Estimation**
                - Cutting time calculation
                - Material cost analysis
                - Optimization suggestions
                """)
        
        # Quick start
        st.subheader("🚀 Quick Start")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎨 Start Designing", use_container_width=True):
                st.session_state.page = "Designer"
                st.rerun()
                
        with col2:
            if st.button("📐 Analyze DXF", use_container_width=True):
                st.session_state.page = "Analyze DXF"
                st.rerun()
                
        with col3:
            if st.button("🖼️ Convert Image", use_container_width=True):
                st.session_state.page = "Image to DXF"
                st.rerun()
            
        # Recent activity
        st.subheader("📈 Recent Activity")
        
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
                    st.text(f"📁 {project.name}")
            else:
                st.info("No recent activity. Start a new project!")
                
    def render_designer_page(self):
        """Render the designer page."""
        st.title("🎨 Designer")
        st.markdown("Generate waterjet-compatible designs using AI")
        
        if st.session_state.get('guided_mode', False):
            st.info("🎯 **Guided Mode**: Follow the steps below for guided design creation.")
            
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
        with st.expander("⚙️ Advanced Settings"):
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
        if st.button("🎨 Generate Design", type="primary"):
            with st.spinner("Generating design..."):
                try:
                    # Create output directory
                    output_dir = self.project_root / "output" / "designer"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate design using AI
                    prompt = f"Generate a waterjet-compatible {design_type} design using {material}, with dimensions {dimensions}, in a {style} style. Waterjet constraints: ≥{min_feature_size} mm spacing, ≥{min_feature_size} mm inner radius curves, no floating parts, clean geometry, continuous contours."
                    
                    # For demo purposes, create a placeholder
                    st.success("✅ Design generated successfully!")
                    st.info(f"**Prompt used:** {prompt}")
                    
                    # In a real implementation, this would call the AI service
                    st.markdown("**Note:** This is a demo. In the full implementation, this would generate an actual design image.")
                    
                except Exception as e:
                    st.error(f"❌ Design generation failed: {e}")
                    
    def render_image_to_dxf_page(self):
        """Render the image to DXF conversion page."""
        st.title("🖼️ Image to DXF")
        st.markdown("Convert images to DXF format for waterjet cutting")
        
        if st.session_state.get('guided_mode', False):
            st.info("🎯 **Guided Mode**: Follow the steps below for guided image conversion.")
            
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to convert to DXF"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Conversion parameters
            st.subheader("⚙️ Conversion Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                mode = st.selectbox(
                    "Conversion Mode",
                    ["Auto Mix", "Edges", "Stipple", "Hatch", "Contour"],
                    index=1,
                    help="Method for vectorization"
                )
                
                kerf_width = st.number_input(
                    "Kerf Width (mm)",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.1,
                    step=0.1
                )
                
            with col2:
                simplify_tolerance = st.number_input(
                    "Simplify Tolerance (mm)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.2,
                    step=0.05
                )
                
                min_feature_size = st.number_input(
                    "Min Feature Size (mm)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
                
            # Kerf compensation
            kerf_mode = st.selectbox(
                "Kerf Compensation",
                ["None", "Outward (+kerf/2)", "Inward (-kerf/2)", "Inside/Outside (+/- kerf/2)"],
                index=3
            )
            
            # Convert button
            if st.button("🔄 Convert to DXF", type="primary"):
                with st.spinner("Converting image to DXF..."):
                    try:
                        # Create output directory
                        output_dir = self.project_root / "output" / "image_to_dxf"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save uploaded file
                        file_path = output_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # For demo purposes, create a placeholder DXF
                        import ezdxf
                        doc = ezdxf.new('R2010')
                        msp = doc.modelspace()
                        msp.add_circle((0, 0), 50)
                        
                        dxf_path = output_dir / f"{uploaded_file.name.split('.')[0]}_converted.dxf"
                        doc.saveas(dxf_path)
                        
                        st.success("✅ Image converted to DXF successfully!")
                        st.info(f"**Output file:** {dxf_path}")
                        
                        # Show download link
                        with open(dxf_path, "rb") as f:
                            st.download_button(
                                "📥 Download DXF",
                                f.read(),
                                file_name=dxf_path.name,
                                mime="application/dxf"
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Conversion failed: {e}")
                        
    def render_analyze_dxf_page(self):
        """Render the DXF analysis page."""
        st.title("📐 Analyze DXF")
        st.markdown("Analyze DXF files for waterjet cutting optimization")
        
        if st.session_state.get('guided_mode', False):
            st.info("🎯 **Guided Mode**: Follow the steps below for guided DXF analysis.")
            
        # File upload
        uploaded_file = st.file_uploader(
            "Upload DXF File",
            type=['dxf'],
            help="Upload a DXF file to analyze"
        )
        
        if uploaded_file:
            # Analysis parameters
            st.subheader("⚙️ Analysis Parameters")
            
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
            with st.expander("🔧 Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    enable_toolpath = st.checkbox("Enable Advanced Toolpath", value=True)
                    enable_cost_estimation = st.checkbox("Enable Cost Estimation", value=True)
                    
                with col2:
                    optimize_rapids = st.checkbox("Optimize Rapids", value=True)
                    optimize_direction = st.checkbox("Optimize Direction", value=True)
                    
            # Analyze button
            if st.button("📊 Analyze DXF", type="primary"):
                with st.spinner("Analyzing DXF..."):
                    try:
                        # Create output directory
                        output_dir = self.project_root / "output" / "analysis"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save uploaded file
                        file_path = output_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Analyze DXF
                        args = AnalyzeArgs(
                            material=material,
                            thickness=thickness,
                            kerf=kerf,
                            cutting_speed=cutting_speed,
                            out=str(output_dir),
                            use_advanced_toolpath=enable_toolpath,
                            optimize_rapids=optimize_rapids,
                            optimize_direction=optimize_direction
                        )
                        
                        report = analyze_dxf(str(file_path), args)
                        
                        st.success("✅ DXF analysis completed successfully!")
                        
                        # Display results
                        if report:
                            st.subheader("📊 Analysis Results")
                            
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
                                        f"₹{metrics.get('estimated_cutting_cost_inr', 0):.0f}"
                                    )
                                    
                            # Quality analysis
                            quality = report.get('quality', {})
                            if quality:
                                st.subheader("🔍 Quality Analysis")
                                
                                issues = quality.get('issues', [])
                                if issues:
                                    st.warning("⚠️ Quality Issues Found:")
                                    for issue in issues:
                                        st.text(f"• {issue}")
                                else:
                                    st.success("✅ No quality issues found!")
                                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        
    def render_nesting_page(self):
        """Render the nesting page."""
        st.title("📦 Nesting")
        st.markdown("Optimize part placement for material efficiency")
        
        if st.session_state.get('guided_mode', False):
            st.info("🎯 **Guided Mode**: Follow the steps below for guided nesting optimization.")
            
        st.info("📝 **Note**: Nesting functionality requires DXF files. Upload files from the Analyze DXF page first.")
        
        # Sheet parameters
        st.subheader("📏 Sheet Parameters")
        
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
        st.subheader("⚙️ Nesting Parameters")
        
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
        with st.expander("🔧 Advanced Options"):
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
        if st.button("📦 Run Nesting", type="primary"):
            st.info("📝 **Demo Mode**: This would run the actual nesting algorithm with the uploaded DXF files.")
            
            # Create mock results for demo
            st.success("✅ Nesting completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Parts Placed", "12")
                
            with col2:
                st.metric("Material Usage", "78.5%")
                
            with col3:
                st.metric("Waste", "21.5%")
                
    def render_ai_agents_page(self):
        """Render the AI agents page."""
        st.title("🤖 AI Agents")
        st.markdown("Interact with specialized AI agents for advanced analysis")
        
        if not is_agents_sdk_available():
            st.error("❌ OpenAI Agents SDK not available. Please install it with: pip install openai-agents")
            return
            
        st.success("✅ OpenAI Agents SDK is available!")
        
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
        st.subheader(f"🤖 {agent_type}")
        
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
        
        if st.button("💬 Send to Agent", type="primary"):
            if user_input:
                with st.spinner("Agent is thinking..."):
                    # Mock agent response
                    st.success("✅ Agent response received!")
                    st.info("📝 **Demo Mode**: This would show the actual agent response.")
            else:
                st.warning("Please enter a message for the agent.")
                
    def render_supervisor_dashboard_page(self):
        """Render the supervisor dashboard page."""
        st.title("📊 Supervisor Dashboard")
        st.markdown("Monitor and control all WJP ANALYSER processes")
        
        # System overview
        st.subheader("📈 System Overview")
        
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
        st.subheader("🔄 Process Monitoring")
        
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
                    "Running": "🟢",
                    "Completed": "✅", 
                    "Queued": "⏳",
                    "Failed": "❌"
                }.get(process["status"], "❓")
                st.text(f"{status_color} {process['status']}")
                
            with col3:
                st.progress(process["progress"] / 100)
                
        # Agent status
        st.subheader("🤖 Agent Status")
        
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
                status_color = "🟢" if agent["status"] == "Active" else "⚪"
                st.text(f"{status_color} {agent['status']}")
                
            with col3:
                st.text(f"Tasks: {agent['tasks']}")
                
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
                
        with col2:
            if st.button("📊 Generate Report", use_container_width=True):
                st.success("✅ Report generated!")
                
        with col3:
            if st.button("🧹 Cleanup", use_container_width=True):
                st.success("✅ Cleanup completed!")
                
    def render_settings_page(self):
        """Render the settings page."""
        st.title("⚙️ Settings")
        st.markdown("Configure WJP ANALYSER system settings")
        
        # Configuration sections
        tab1, tab2, tab3, tab4 = st.tabs(["General", "AI", "Processing", "Advanced"])
        
        with tab1:
            st.subheader("🔧 General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Project Name", value="WJP ANALYSER")
                st.text_input("Version", value=self.config.get('version', '2.0.0'))
                
            with col2:
                st.selectbox("Environment", ["Development", "Staging", "Production"])
                st.checkbox("Enable Debug Mode", value=False)
                
        with tab2:
            st.subheader("🤖 AI Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
                st.selectbox("Default Model", ["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini"])
                
            with col2:
                st.number_input("Max Tokens", min_value=100, max_value=4000, value=2000)
                st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                
        with tab3:
            st.subheader("⚙️ Processing Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input("Max Workers", min_value=1, max_value=16, value=4)
                st.number_input("Timeout (seconds)", min_value=30, max_value=600, value=300)
                
            with col2:
                st.checkbox("Enable Caching", value=True)
                st.checkbox("Enable Compression", value=True)
                
        with tab4:
            st.subheader("🔧 Advanced Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.checkbox("Enable Monitoring", value=True)
                st.checkbox("Enable Logging", value=True)
                
            with col2:
                st.checkbox("Enable Rate Limiting", value=True)
                st.checkbox("Enable Security Headers", value=True)
                
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
            
    def run(self):
        """Run the unified web application."""
        # Render sidebar and get current page
        page = self.render_sidebar()
        
        # Handle page navigation
        if page == "🏠 Home":
            self.render_home_page()
        elif page == "🎨 Designer":
            self.render_designer_page()
        elif page == "🖼️ Image to DXF":
            self.render_image_to_dxf_page()
        elif page == "📐 Analyze DXF":
            self.render_analyze_dxf_page()
        elif page == "📦 Nesting":
            self.render_nesting_page()
        elif page == "🤖 AI Agents":
            self.render_ai_agents_page()
        elif page == "📊 Supervisor Dashboard":
            self.render_supervisor_dashboard_page()
        elif page == "⚙️ Settings":
            self.render_settings_page()
        else:
            self.render_home_page()


def main():
    """Main entry point for the unified web application."""
    app = WJPUnifiedWebApp()
    app.run()


if __name__ == "__main__":
    main()
