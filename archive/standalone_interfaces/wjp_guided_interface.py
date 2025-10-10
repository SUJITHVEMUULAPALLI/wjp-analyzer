#!/usr/bin/env python3
"""
WJP Guided Interface - Step-by-Step User Guidance
=================================================

This module creates intelligent step-by-step guidance for users in both
the individual and batch processing interfaces.
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

class GuidanceStep(Enum):
    """Guidance steps for user workflow."""
    WELCOME = "welcome"
    MATERIAL_SELECTION = "material_selection"
    DESIGN_CONFIGURATION = "design_configuration"
    PROMPT_CREATION = "prompt_creation"
    JOB_SUBMISSION = "job_submission"
    PROCESSING_MONITORING = "processing_monitoring"
    RESULTS_REVIEW = "results_review"
    REPORT_DOWNLOAD = "report_download"
    COMPLETION = "completion"

class GuidanceLevel(Enum):
    """Guidance levels for different user types."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class GuidanceMessage:
    """Guidance message structure."""
    step: GuidanceStep
    title: str
    message: str
    action_required: str
    tips: List[str]
    warnings: List[str]
    next_step: Optional[GuidanceStep]
    is_complete: bool = False

class WJPGuidanceSystem:
    """Intelligent guidance system for WJP interfaces."""
    
    def __init__(self):
        self.current_step = GuidanceStep.WELCOME
        self.guidance_level = GuidanceLevel.BEGINNER
        self.completed_steps = set()
        self.user_preferences = {}
        
        # Initialize guidance messages
        self.guidance_messages = self._initialize_guidance_messages()
    
    def _initialize_guidance_messages(self) -> Dict[GuidanceStep, GuidanceMessage]:
        """Initialize all guidance messages."""
        return {
            GuidanceStep.WELCOME: GuidanceMessage(
                step=GuidanceStep.WELCOME,
                title="🎯 Welcome to WJP Automation Pipeline",
                message="Welcome! I'll guide you through creating professional waterjet projects from prompt to PDF report. Let's start by understanding your needs.",
                action_required="Choose your experience level and project type",
                tips=[
                    "This system automates the entire workflow from design prompt to professional report",
                    "You can process individual projects or batch multiple projects",
                    "All files are automatically organized with professional naming standards"
                ],
                warnings=[],
                next_step=GuidanceStep.MATERIAL_SELECTION
            ),
            
            GuidanceStep.MATERIAL_SELECTION: GuidanceMessage(
                step=GuidanceStep.MATERIAL_SELECTION,
                title="🏗️ Material Selection",
                message="Choose the material for your waterjet project. This affects cost calculations and cutting parameters.",
                action_required="Select material type and thickness",
                tips=[
                    "Granite: Premium quality, ₹850/meter cutting cost",
                    "Marble: High quality, ₹750/meter cutting cost", 
                    "Stainless Steel: Industrial grade, ₹1200/meter cutting cost",
                    "Aluminum: Cost-effective, ₹400/meter cutting cost",
                    "Thickness affects cutting time and material cost"
                ],
                warnings=[
                    "Thicker materials require longer cutting times",
                    "Some materials may have minimum thickness requirements"
                ],
                next_step=GuidanceStep.DESIGN_CONFIGURATION
            ),
            
            GuidanceStep.DESIGN_CONFIGURATION: GuidanceMessage(
                step=GuidanceStep.DESIGN_CONFIGURATION,
                title="📐 Design Configuration",
                message="Configure your design dimensions and technical parameters for optimal waterjet cutting.",
                action_required="Set dimensions and cutting parameters",
                tips=[
                    "Standard tile sizes: 24×24 inch, 36×36 inch",
                    "Minimum cut spacing: 3mm recommended",
                    "Minimum radius: 2mm for clean cuts",
                    "Larger designs may require longer processing times"
                ],
                warnings=[
                    "Very small features (<2mm) may be difficult to cut",
                    "Complex designs increase processing time and cost"
                ],
                next_step=GuidanceStep.PROMPT_CREATION
            ),
            
            GuidanceStep.PROMPT_CREATION: GuidanceMessage(
                step=GuidanceStep.PROMPT_CREATION,
                title="💭 Design Prompt Creation",
                message="Create a detailed prompt describing your design. Be specific about patterns, styles, and requirements.",
                action_required="Write a clear, detailed design prompt",
                tips=[
                    "Include material type and dimensions in your prompt",
                    "Specify design style: modern, traditional, geometric, floral",
                    "Mention any special requirements: symmetry, borders, inlays",
                    "Example: 'Waterjet-safe Tan Brown granite tile with white marble inlay, 24×24 inch, geometric pattern'"
                ],
                warnings=[
                    "Vague prompts may result in unexpected designs",
                    "Very complex prompts may take longer to process"
                ],
                next_step=GuidanceStep.JOB_SUBMISSION
            ),
            
            GuidanceStep.JOB_SUBMISSION: GuidanceMessage(
                step=GuidanceStep.JOB_SUBMISSION,
                title="🚀 Job Submission",
                message="Review your configuration and submit the job for processing. The system will automatically handle the entire pipeline.",
                action_required="Review and submit your job",
                tips=[
                    "Double-check all parameters before submission",
                    "Job ID will be automatically generated",
                    "Processing typically takes 1-3 minutes",
                    "You can monitor progress in real-time"
                ],
                warnings=[
                    "Once submitted, parameters cannot be changed",
                    "Large or complex designs may take longer to process"
                ],
                next_step=GuidanceStep.PROCESSING_MONITORING
            ),
            
            GuidanceStep.PROCESSING_MONITORING: GuidanceMessage(
                step=GuidanceStep.PROCESSING_MONITORING,
                title="📊 Processing Monitoring",
                message="Monitor your job progress through the automated pipeline. The system processes: Prompt → Image → DXF → Analysis → Report.",
                action_required="Monitor job progress and wait for completion",
                tips=[
                    "Stage 1: Designer Agent creates your design image",
                    "Stage 2: Image to DXF Agent converts to cutting file",
                    "Stage 3: DXF Analyzer calculates costs and validates geometry",
                    "Stage 4: Report Generator creates professional PDF report"
                ],
                warnings=[
                    "Don't close the browser during processing",
                    "Network issues may affect real-time updates"
                ],
                next_step=GuidanceStep.RESULTS_REVIEW
            ),
            
            GuidanceStep.RESULTS_REVIEW: GuidanceMessage(
                step=GuidanceStep.RESULTS_REVIEW,
                title="📈 Results Review",
                message="Review your completed project results. Check the analysis, costs, and quality metrics.",
                action_required="Review analysis results and quality metrics",
                tips=[
                    "Check cutting cost and time estimates",
                    "Review geometry violations and complexity rating",
                    "Examine layer breakdown (OUTER, COMPLEX, DECOR)",
                    "Quality score indicates design manufacturability"
                ],
                warnings=[
                    "High violation counts may indicate design issues",
                    "Very high complexity may increase cutting costs"
                ],
                next_step=GuidanceStep.REPORT_DOWNLOAD
            ),
            
            GuidanceStep.REPORT_DOWNLOAD: GuidanceMessage(
                step=GuidanceStep.REPORT_DOWNLOAD,
                title="📥 Report Download",
                message="Download your professional reports and files. Multiple formats are available for different uses.",
                action_required="Download reports and files",
                tips=[
                    "PDF Report: Professional presentation format",
                    "CSV Report: Data analysis and integration",
                    "JSON Report: System integration and automation",
                    "DXF File: Direct use in cutting machines",
                    "PNG Images: Visual reference and documentation"
                ],
                warnings=[
                    "Save files to a secure location",
                    "PDF reports contain all project information"
                ],
                next_step=GuidanceStep.COMPLETION
            ),
            
            GuidanceStep.COMPLETION: GuidanceMessage(
                step=GuidanceStep.COMPLETION,
                title="🎉 Project Complete",
                message="Congratulations! Your waterjet project has been successfully processed. You can now submit additional projects or start batch processing.",
                action_required="Choose next action",
                tips=[
                    "Submit another individual project",
                    "Start batch processing for multiple projects",
                    "Review system statistics and performance",
                    "Archive completed projects for future reference"
                ],
                warnings=[],
                next_step=None
            )
        }
    
    def get_current_guidance(self) -> GuidanceMessage:
        """Get current guidance message."""
        return self.guidance_messages[self.current_step]
    
    def advance_step(self, step: GuidanceStep):
        """Advance to next step."""
        if step in self.guidance_messages:
            self.current_step = step
            self.completed_steps.add(step)
    
    def set_guidance_level(self, level: GuidanceLevel):
        """Set guidance level for user."""
        self.guidance_level = level
    
    def get_step_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        total_steps = len(GuidanceStep)
        completed_count = len(self.completed_steps)
        current_index = list(GuidanceStep).index(self.current_step)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_count,
            "current_step_index": current_index,
            "progress_percentage": (completed_count / total_steps) * 100,
            "current_step": self.current_step.value,
            "is_complete": self.current_step == GuidanceStep.COMPLETION
        }
    
    def get_contextual_tips(self, context: str) -> List[str]:
        """Get contextual tips based on current context."""
        contextual_tips = {
            "material_selection": [
                "Consider your budget when selecting materials",
                "Aluminum offers the best cost-to-quality ratio",
                "Granite provides premium appearance and durability"
            ],
            "design_configuration": [
                "Start with standard dimensions for your first project",
                "Use conservative spacing values for reliable results",
                "Consider your cutting machine's capabilities"
            ],
            "prompt_creation": [
                "Be specific about design elements you want",
                "Include color and material preferences",
                "Mention any functional requirements"
            ],
            "processing_monitoring": [
                "Processing time varies with design complexity",
                "You can submit multiple jobs while one is processing",
                "Check the system status page for overall performance"
            ]
        }
        
        return contextual_tips.get(context, [])
    
    def get_warnings_for_context(self, context: str) -> List[str]:
        """Get warnings for specific context."""
        contextual_warnings = {
            "material_selection": [
                "Some materials may not be suitable for very thin designs",
                "Check material availability before finalizing"
            ],
            "design_configuration": [
                "Very small features may be difficult to cut accurately",
                "Complex geometries increase processing time"
            ],
            "prompt_creation": [
                "Avoid overly complex prompts for your first projects",
                "Test with simple designs before attempting complex ones"
            ],
            "processing_monitoring": [
                "Don't interrupt processing once started",
                "Monitor system resources if processing multiple jobs"
            ]
        }
        
        return contextual_warnings.get(context, [])

def create_guided_interface():
    """Create the guided interface for WJP automation."""
    st.set_page_config(
        page_title="WJP Guided Interface",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 WJP Automation Pipeline - Guided Interface")
    st.markdown("**Step-by-step guidance for professional waterjet project creation**")
    
    # Initialize guidance system
    if "guidance_system" not in st.session_state:
        st.session_state.guidance_system = WJPGuidanceSystem()
    
    guidance = st.session_state.guidance_system
    
    # Sidebar for guidance
    with st.sidebar:
        st.header("🎯 Guidance")
        
        # Progress bar
        progress = guidance.get_step_progress()
        st.progress(progress["progress_percentage"] / 100)
        st.caption(f"Step {progress['current_step_index'] + 1} of {progress['total_steps']}")
        
        # Current step
        current_guidance = guidance.get_current_guidance()
        st.markdown(f"### {current_guidance.title}")
        st.write(current_guidance.message)
        
        # Action required
        st.markdown("#### 📋 Action Required:")
        st.write(current_guidance.action_required)
        
        # Tips
        if current_guidance.tips:
            st.markdown("#### 💡 Tips:")
            for tip in current_guidance.tips:
                st.write(f"• {tip}")
        
        # Warnings
        if current_guidance.warnings:
            st.markdown("#### ⚠️ Warnings:")
            for warning in current_guidance.warnings:
                st.write(f"• {warning}")
        
        # Navigation
        st.markdown("#### 🧭 Navigation:")
        
        # Step selection
        step_options = [step.value for step in GuidanceStep]
        current_step_index = step_options.index(guidance.current_step.value)
        
        selected_step = st.selectbox(
            "Go to step:",
            step_options,
            index=current_step_index
        )
        
        if selected_step != guidance.current_step.value:
            guidance.advance_step(GuidanceStep(selected_step))
            st.rerun()
        
        # Guidance level
        st.markdown("#### 🎓 Guidance Level:")
        level = st.selectbox(
            "Experience Level:",
            ["beginner", "intermediate", "advanced"],
            index=0
        )
        guidance.set_guidance_level(GuidanceLevel(level))
    
    # Main content based on current step
    if guidance.current_step == GuidanceStep.WELCOME:
        create_welcome_step(guidance)
    elif guidance.current_step == GuidanceStep.MATERIAL_SELECTION:
        create_material_selection_step(guidance)
    elif guidance.current_step == GuidanceStep.DESIGN_CONFIGURATION:
        create_design_configuration_step(guidance)
    elif guidance.current_step == GuidanceStep.PROMPT_CREATION:
        create_prompt_creation_step(guidance)
    elif guidance.current_step == GuidanceStep.JOB_SUBMISSION:
        create_job_submission_step(guidance)
    elif guidance.current_step == GuidanceStep.PROCESSING_MONITORING:
        create_processing_monitoring_step(guidance)
    elif guidance.current_step == GuidanceStep.RESULTS_REVIEW:
        create_results_review_step(guidance)
    elif guidance.current_step == GuidanceStep.REPORT_DOWNLOAD:
        create_report_download_step(guidance)
    elif guidance.current_step == GuidanceStep.COMPLETION:
        create_completion_step(guidance)

def create_welcome_step(guidance: WJPGuidanceSystem):
    """Create welcome step interface."""
    st.header("🎯 Welcome to WJP Automation Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 What is WJP Automation Pipeline?
        
        The WJP Automation Pipeline is a complete system that automates your waterjet project workflow from initial design prompt to professional PDF report. Our intelligent agents guide you through every step.
        
        #### 🔄 Complete Workflow:
        1. **Design Creation** - Generate design images from your prompts
        2. **DXF Conversion** - Convert images to cutting-ready DXF files
        3. **Analysis & Validation** - Calculate costs, validate geometry, assess quality
        4. **Professional Reporting** - Generate comprehensive PDF reports
        
        #### 🎯 Key Benefits:
        - **Complete Automation** - No manual intervention required
        - **Professional Standards** - Industry-standard file naming and organization
        - **Material Integration** - Accurate cost calculations for 6 material types
        - **Quality Assessment** - Comprehensive geometry validation and analysis
        - **Real-time Monitoring** - Track progress through every stage
        """)
        
        # Project type selection
        st.markdown("### 📋 Choose Your Project Type:")
        
        project_type = st.radio(
            "What would you like to create?",
            ["Individual Project", "Batch Processing", "Explore System Features"],
            help="Select the type of project you want to work on"
        )
        
        if project_type == "Individual Project":
            st.success("✅ Perfect! We'll guide you through creating a single waterjet project.")
        elif project_type == "Batch Processing":
            st.info("ℹ️ Great choice! We'll help you process multiple projects efficiently.")
        else:
            st.info("ℹ️ Let's explore the system capabilities and features.")
    
    with col2:
        st.markdown("### 🎓 Your Experience Level")
        
        experience_level = st.selectbox(
            "How familiar are you with waterjet cutting?",
            ["Beginner", "Intermediate", "Advanced"],
            help="This helps us provide appropriate guidance"
        )
        
        st.markdown("### 📊 System Status")
        
        # Mock system status
        st.metric("Active Jobs", "0")
        st.metric("Completed Today", "12")
        st.metric("Success Rate", "94.2%")
        
        st.markdown("### 🚀 Quick Start")
        
        if st.button("🎯 Start Guided Process", type="primary"):
            guidance.advance_step(GuidanceStep.MATERIAL_SELECTION)
            st.rerun()
        
        if st.button("📚 View Documentation"):
            st.info("Documentation will open in a new tab")
        
        if st.button("🎥 Watch Tutorial"):
            st.info("Tutorial video will be displayed")

def create_material_selection_step(guidance: WJPGuidanceSystem):
    """Create material selection step interface."""
    st.header("🏗️ Material Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Choose Your Material")
        
        # Material selection with detailed information
        materials = {
            "Tan Brown Granite": {
                "cost_per_meter": 850,
                "cutting_speed": 800,
                "description": "Premium quality granite with natural brown tones",
                "best_for": "High-end architectural projects, medallions",
                "thickness_range": "10-50mm"
            },
            "Marble": {
                "cost_per_meter": 750,
                "cutting_speed": 1000,
                "description": "Elegant marble with smooth finish",
                "best_for": "Decorative elements, nameplates",
                "thickness_range": "10-30mm"
            },
            "Stainless Steel": {
                "cost_per_meter": 1200,
                "cutting_speed": 600,
                "description": "Industrial-grade stainless steel",
                "best_for": "Functional components, drainage covers",
                "thickness_range": "1-10mm"
            },
            "Aluminum": {
                "cost_per_meter": 400,
                "cutting_speed": 1200,
                "description": "Lightweight and cost-effective",
                "best_for": "Prototypes, cost-sensitive projects",
                "thickness_range": "1-20mm"
            },
            "Brass": {
                "cost_per_meter": 900,
                "cutting_speed": 700,
                "description": "Decorative brass with golden finish",
                "best_for": "Decorative panels, signage",
                "thickness_range": "2-15mm"
            },
            "Generic": {
                "cost_per_meter": 600,
                "cutting_speed": 1000,
                "description": "Standard material for general use",
                "best_for": "Testing, general projects",
                "thickness_range": "5-25mm"
            }
        }
        
        selected_material = st.selectbox(
            "Select Material:",
            list(materials.keys()),
            help="Choose the material for your project"
        )
        
        # Display material details
        material_info = materials[selected_material]
        
        st.markdown(f"#### 📋 {selected_material} Details:")
        st.write(f"**Description:** {material_info['description']}")
        st.write(f"**Best For:** {material_info['best_for']}")
        st.write(f"**Thickness Range:** {material_info['thickness_range']}")
        
        # Thickness selection
        st.markdown("### 📏 Thickness Selection")
        
        thickness = st.slider(
            "Material Thickness (mm):",
            min_value=1,
            max_value=50,
            value=25,
            help="Select the thickness of your material"
        )
        
        # Cost estimation
        st.markdown("### 💰 Cost Estimation")
        
        estimated_length = st.number_input(
            "Estimated Cut Length (meters):",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Estimate the total cutting length for cost calculation"
        )
        
        material_cost = estimated_length * material_info["cost_per_meter"]
        cutting_time = (estimated_length * 1000) / material_info["cutting_speed"]
        
        col_cost, col_time = st.columns(2)
        
        with col_cost:
            st.metric("Estimated Cost", f"₹{material_cost:,.0f}")
        
        with col_time:
            st.metric("Cutting Time", f"{cutting_time:.1f} min")
    
    with col2:
        st.markdown("### 💡 Material Selection Tips")
        
        tips = guidance.get_contextual_tips("material_selection")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Important Considerations")
        
        warnings = guidance.get_warnings_for_context("material_selection")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 🎯 Recommendation")
        
        if selected_material == "Aluminum":
            st.success("✅ Great choice for cost-effective projects!")
        elif selected_material == "Tan Brown Granite":
            st.success("✅ Premium choice for high-quality projects!")
        elif selected_material == "Stainless Steel":
            st.info("ℹ️ Excellent for functional components!")
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Welcome"):
            guidance.advance_step(GuidanceStep.WELCOME)
            st.rerun()
        
        if st.button("➡️ Continue to Design Configuration", type="primary"):
            guidance.advance_step(GuidanceStep.DESIGN_CONFIGURATION)
            st.rerun()

def create_design_configuration_step(guidance: WJPGuidanceSystem):
    """Create design configuration step interface."""
    st.header("📐 Design Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📏 Dimensions Configuration")
        
        # Dimension selection
        dimension_preset = st.selectbox(
            "Choose Dimension Preset:",
            ["Custom", "24×24 inch (Standard Tile)", "36×36 inch (Large Tile)", "48×36 inch (Panel)", "12×12 inch (Small Tile)"],
            help="Select a preset or choose custom dimensions"
        )
        
        if dimension_preset == "Custom":
            col_width, col_height = st.columns(2)
            with col_width:
                width_inch = st.number_input("Width (inches):", 1, 100, 24)
            with col_height:
                height_inch = st.number_input("Height (inches):", 1, 100, 24)
        else:
            # Parse preset dimensions
            if "24×24" in dimension_preset:
                width_inch, height_inch = 24, 24
            elif "36×36" in dimension_preset:
                width_inch, height_inch = 36, 36
            elif "48×36" in dimension_preset:
                width_inch, height_inch = 48, 36
            elif "12×12" in dimension_preset:
                width_inch, height_inch = 12, 12
            
            st.info(f"Selected: {width_inch}×{height_inch} inch")
        
        # Design category
        st.markdown("### 🎨 Design Category")
        
        category = st.selectbox(
            "Design Category:",
            ["Inlay Tile", "Medallion", "Border", "Jali Panel", "Drainage Cover", "Nameplate", "Custom Pattern"],
            help="Select the type of design you want to create"
        )
        
        # Technical parameters
        st.markdown("### ⚙️ Technical Parameters")
        
        col_spacing, col_radius = st.columns(2)
        
        with col_spacing:
            cut_spacing_mm = st.number_input(
                "Cut Spacing (mm):",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Minimum spacing between cut lines"
            )
        
        with col_radius:
            min_radius_mm = st.number_input(
                "Minimum Radius (mm):",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Minimum radius for corners and curves"
            )
        
        # Design complexity
        st.markdown("### 🧩 Design Complexity")
        
        complexity = st.select_slider(
            "Design Complexity:",
            options=["Simple", "Moderate", "Complex", "Very Complex"],
            value="Moderate",
            help="Select the complexity level of your design"
        )
        
        # Complexity impact
        complexity_impact = {
            "Simple": {"processing_time": "1-2 min", "cost_factor": "1.0x", "success_rate": "95%"},
            "Moderate": {"processing_time": "2-3 min", "cost_factor": "1.2x", "success_rate": "90%"},
            "Complex": {"processing_time": "3-5 min", "cost_factor": "1.5x", "success_rate": "85%"},
            "Very Complex": {"processing_time": "5+ min", "cost_factor": "2.0x", "success_rate": "80%"}
        }
        
        impact = complexity_impact[complexity]
        
        col_time, col_cost, col_success = st.columns(3)
        
        with col_time:
            st.metric("Processing Time", impact["processing_time"])
        
        with col_cost:
            st.metric("Cost Factor", impact["cost_factor"])
        
        with col_success:
            st.metric("Success Rate", impact["success_rate"])
    
    with col2:
        st.markdown("### 💡 Configuration Tips")
        
        tips = guidance.get_contextual_tips("design_configuration")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Design Considerations")
        
        warnings = guidance.get_warnings_for_context("design_configuration")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📊 Configuration Summary")
        
        st.write(f"**Dimensions:** {width_inch}×{height_inch} inch")
        st.write(f"**Category:** {category}")
        st.write(f"**Cut Spacing:** {cut_spacing_mm} mm")
        st.write(f"**Min Radius:** {min_radius_mm} mm")
        st.write(f"**Complexity:** {complexity}")
        
        # Validation
        st.markdown("### ✅ Configuration Validation")
        
        validation_results = []
        
        if width_inch >= 12 and height_inch >= 12:
            validation_results.append(("✅ Dimensions", "Suitable for waterjet cutting"))
        else:
            validation_results.append(("⚠️ Dimensions", "Very small - may be challenging"))
        
        if cut_spacing_mm >= 2.0:
            validation_results.append(("✅ Cut Spacing", "Safe for reliable cutting"))
        else:
            validation_results.append(("⚠️ Cut Spacing", "Very tight - may cause issues"))
        
        if min_radius_mm >= 1.0:
            validation_results.append(("✅ Radius", "Good for clean cuts"))
        else:
            validation_results.append(("⚠️ Radius", "Very sharp - may be difficult"))
        
        for result, message in validation_results:
            st.write(f"{result}: {message}")
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Material Selection"):
            guidance.advance_step(GuidanceStep.MATERIAL_SELECTION)
            st.rerun()
        
        if st.button("➡️ Continue to Prompt Creation", type="primary"):
            guidance.advance_step(GuidanceStep.PROMPT_CREATION)
            st.rerun()

def create_prompt_creation_step(guidance: WJPGuidanceSystem):
    """Create prompt creation step interface."""
    st.header("💭 Design Prompt Creation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Create Your Design Prompt")
        
        # Prompt builder
        st.markdown("#### 🏗️ Material & Dimensions")
        material_desc = st.text_input(
            "Material Description:",
            value="Tan Brown granite",
            help="Describe the material and its appearance"
        )
        
        dimension_desc = st.text_input(
            "Dimension Description:",
            value="24×24 inch",
            help="Specify the dimensions"
        )
        
        st.markdown("#### 🎨 Design Style")
        design_style = st.selectbox(
            "Design Style:",
            ["Geometric", "Floral", "Traditional", "Modern", "Minimalist", "Ornate", "Custom"],
            help="Choose the overall design style"
        )
        
        if design_style == "Custom":
            custom_style = st.text_input("Custom Style Description:", help="Describe your custom style")
        else:
            custom_style = design_style.lower()
        
        st.markdown("#### 🧩 Design Elements")
        
        # Design elements
        elements = st.multiselect(
            "Design Elements:",
            ["Border", "Central Medallion", "Corner Patterns", "Inlay Lines", "Perforations", "Text", "Logo", "Symmetrical Pattern"],
            help="Select the design elements you want"
        )
        
        # Additional requirements
        st.markdown("#### 📋 Additional Requirements")
        
        additional_reqs = st.text_area(
            "Additional Requirements:",
            placeholder="e.g., Waterjet-safe design, minimum 3mm spacing, rounded corners",
            help="Add any specific requirements or constraints"
        )
        
        # Generate prompt
        st.markdown("#### 🎯 Generated Prompt")
        
        # Build prompt automatically
        prompt_parts = [
            f"Waterjet-safe {material_desc} {dimension_desc}",
            f"{custom_style} design",
        ]
        
        if elements:
            prompt_parts.append(f"with {', '.join(elements)}")
        
        if additional_reqs:
            prompt_parts.append(f", {additional_reqs}")
        
        generated_prompt = " ".join(prompt_parts)
        
        st.text_area(
            "Generated Prompt:",
            value=generated_prompt,
            height=100,
            help="This is your complete design prompt"
        )
        
        # Manual prompt editing
        st.markdown("#### ✏️ Custom Prompt (Optional)")
        
        manual_prompt = st.text_area(
            "Edit Prompt Manually:",
            value=generated_prompt,
            height=100,
            help="You can modify the generated prompt or write your own"
        )
        
        # Prompt validation
        st.markdown("#### ✅ Prompt Validation")
        
        validation_score = 0
        validation_messages = []
        
        if len(manual_prompt) > 20:
            validation_score += 1
            validation_messages.append("✅ Prompt length is adequate")
        else:
            validation_messages.append("⚠️ Prompt is quite short")
        
        if "waterjet" in manual_prompt.lower():
            validation_score += 1
            validation_messages.append("✅ Mentions waterjet cutting")
        else:
            validation_messages.append("⚠️ Consider mentioning waterjet cutting")
        
        if any(word in manual_prompt.lower() for word in ["mm", "inch", "spacing", "radius"]):
            validation_score += 1
            validation_messages.append("✅ Includes technical specifications")
        else:
            validation_messages.append("💡 Consider adding technical specifications")
        
        for message in validation_messages:
            st.write(message)
        
        st.metric("Prompt Quality Score", f"{validation_score}/3")
    
    with col2:
        st.markdown("### 💡 Prompt Creation Tips")
        
        tips = guidance.get_contextual_tips("prompt_creation")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### 📝 Prompt Examples")
        
        example_prompts = [
            "Waterjet-safe Tan Brown granite tile with white marble inlay, 24×24 inch, geometric pattern with border",
            "Circular medallion design for marble flooring, 36×36 inch, traditional floral pattern, minimum 3mm spacing",
            "Stainless steel drainage cover, 300×200mm, elongated slots with 5mm rounded ends, functional design"
        ]
        
        for i, example in enumerate(example_prompts, 1):
            with st.expander(f"Example {i}"):
                st.write(example)
        
        st.markdown("### ⚠️ Common Mistakes")
        
        mistakes = [
            "Too vague: 'Make it pretty'",
            "Too complex: 'Create a masterpiece with 50 different patterns'",
            "Missing specifications: 'Design a tile'",
            "Unrealistic requirements: 'Cut lines 0.1mm apart'"
        ]
        
        for mistake in mistakes:
            st.warning(f"❌ {mistake}")
        
        st.markdown("### 🎯 Prompt Quality Checklist")
        
        checklist_items = [
            "✅ Specifies material type",
            "✅ Includes dimensions",
            "✅ Describes design style",
            "✅ Mentions waterjet cutting",
            "✅ Includes technical requirements",
            "✅ Clear and specific language"
        ]
        
        for item in checklist_items:
            st.write(item)
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Design Configuration"):
            guidance.advance_step(GuidanceStep.DESIGN_CONFIGURATION)
            st.rerun()
        
        if st.button("➡️ Continue to Job Submission", type="primary"):
            guidance.advance_step(GuidanceStep.JOB_SUBMISSION)
            st.rerun()

def create_job_submission_step(guidance: WJPGuidanceSystem):
    """Create job submission step interface."""
    st.header("🚀 Job Submission")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Job Configuration Summary")
        
        # Display configuration summary
        st.markdown("#### 🏗️ Material & Dimensions")
        st.write("• **Material:** Tan Brown Granite")
        st.write("• **Thickness:** 25 mm")
        st.write("• **Dimensions:** 24×24 inch")
        
        st.markdown("#### ⚙️ Technical Parameters")
        st.write("• **Cut Spacing:** 3.0 mm")
        st.write("• **Minimum Radius:** 2.0 mm")
        st.write("• **Design Category:** Inlay Tile")
        
        st.markdown("#### 💭 Design Prompt")
        st.write("Waterjet-safe Tan Brown granite tile with white marble inlay, 24×24 inch, geometric pattern with border")
        
        # Job ID generation
        st.markdown("### 🆔 Job Identification")
        
        job_id = st.text_input(
            "Job ID:",
            value="SR06",
            help="Unique identifier for your job"
        )
        
        # Job priority
        priority = st.selectbox(
            "Job Priority:",
            ["Normal", "High", "Low"],
            help="Set the processing priority"
        )
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            enable_optimization = st.checkbox("Enable Optimization", True, help="Automatically optimize parameters")
            enable_learning = st.checkbox("Enable Learning", True, help="Learn from this job for future improvements")
        
        with col_opt2:
            generate_preview = st.checkbox("Generate Preview", True, help="Create visual preview images")
            detailed_analysis = st.checkbox("Detailed Analysis", True, help="Perform comprehensive analysis")
        
        # Estimated processing time
        st.markdown("### ⏱️ Processing Estimate")
        
        estimated_time = st.slider(
            "Estimated Processing Time:",
            min_value=1,
            max_value=10,
            value=3,
            help="Estimated time in minutes"
        )
        
        # Cost estimate
        st.markdown("### 💰 Cost Estimate")
        
        estimated_cost = st.number_input(
            "Estimated Cost (₹):",
            min_value=100,
            max_value=10000,
            value=2500,
            help="Estimated total cost for the project"
        )
        
        # Submit button
        st.markdown("### 🚀 Submit Job")
        
        if st.button("🚀 Submit Job for Processing", type="primary", use_container_width=True):
            with st.spinner("Submitting job..."):
                # Simulate job submission
                import time
                time.sleep(2)
                
                st.success("✅ Job submitted successfully!")
                st.info("Your job has been added to the processing queue.")
                
                # Advance to monitoring step
                guidance.advance_step(GuidanceStep.PROCESSING_MONITORING)
                st.rerun()
    
    with col2:
        st.markdown("### 💡 Submission Tips")
        
        tips = guidance.get_contextual_tips("job_submission")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Before You Submit")
        
        warnings = guidance.get_warnings_for_context("job_submission")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📊 Job Summary")
        
        summary_data = {
            "Job ID": job_id,
            "Priority": priority,
            "Estimated Time": f"{estimated_time} min",
            "Estimated Cost": f"₹{estimated_cost:,}",
            "Optimization": "Enabled" if enable_optimization else "Disabled",
            "Learning": "Enabled" if enable_learning else "Disabled"
        }
        
        for key, value in summary_data.items():
            st.write(f"**{key}:** {value}")
        
        st.markdown("### 🔄 What Happens Next?")
        
        next_steps = [
            "1. Job added to processing queue",
            "2. Designer Agent creates your design image",
            "3. Image to DXF Agent converts to cutting file",
            "4. DXF Analyzer calculates costs and validates geometry",
            "5. Report Generator creates professional PDF report"
        ]
        
        for step in next_steps:
            st.write(step)
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Prompt Creation"):
            guidance.advance_step(GuidanceStep.PROMPT_CREATION)
            st.rerun()
        
        if st.button("📊 View Queue Status"):
            st.info("Queue status will be displayed")

def create_processing_monitoring_step(guidance: WJPGuidanceSystem):
    """Create processing monitoring step interface."""
    st.header("📊 Processing Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔄 Job Processing Status")
        
        # Simulate processing stages
        stages = [
            ("🎨 Designer Agent", "Creating design image from prompt", "completed"),
            ("🔄 Image to DXF Agent", "Converting image to DXF format", "completed"),
            ("📊 DXF Analyzer Agent", "Analyzing geometry and calculating costs", "in_progress"),
            ("📄 Report Generator Agent", "Generating professional PDF report", "pending")
        ]
        
        for stage_name, description, status in stages:
            if status == "completed":
                st.success(f"✅ {stage_name}: {description}")
            elif status == "in_progress":
                st.info(f"🔄 {stage_name}: {description}")
                # Progress bar for current stage
                progress = st.progress(0.6)
                st.caption("Processing... 60% complete")
            else:
                st.write(f"⏳ {stage_name}: {description}")
        
        # Real-time updates
        st.markdown("### 📈 Real-time Updates")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        # Processing details
        st.markdown("### 📋 Processing Details")
        
        processing_data = {
            "Job ID": "SR06",
            "Start Time": "12:30:15",
            "Elapsed Time": "2m 30s",
            "Current Stage": "DXF Analysis",
            "Progress": "60%",
            "Estimated Completion": "12:35:00"
        }
        
        for key, value in processing_data.items():
            col_key, col_value = st.columns([1, 1])
            with col_key:
                st.write(f"**{key}:**")
            with col_value:
                st.write(value)
        
        # Live processing log
        st.markdown("### 📝 Processing Log")
        
        log_entries = [
            ("12:30:15", "Job SR06 submitted to queue"),
            ("12:30:16", "Designer Agent started processing"),
            ("12:31:45", "Design image generated successfully"),
            ("12:31:46", "Image to DXF Agent started conversion"),
            ("12:32:20", "DXF file created with 15 entities"),
            ("12:32:21", "DXF Analyzer Agent started analysis"),
            ("12:33:45", "Geometry validation completed"),
            ("12:33:46", "Cost calculation in progress...")
        ]
        
        for timestamp, message in log_entries:
            st.write(f"`{timestamp}` {message}")
    
    with col2:
        st.markdown("### 💡 Monitoring Tips")
        
        tips = guidance.get_contextual_tips("processing_monitoring")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Important Notes")
        
        warnings = guidance.get_warnings_for_context("processing_monitoring")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📊 System Performance")
        
        performance_metrics = {
            "Queue Size": "2 jobs",
            "Active Jobs": "1 job",
            "Completed Today": "12 jobs",
            "Success Rate": "94.2%",
            "Avg Processing Time": "3.2 min"
        }
        
        for metric, value in performance_metrics.items():
            st.metric(metric, value)
        
        st.markdown("### 🎯 What's Happening Now?")
        
        current_activity = [
            "✅ Design image created",
            "✅ DXF conversion completed", 
            "🔄 Analyzing geometry and calculating costs",
            "⏳ Preparing to generate reports"
        ]
        
        for activity in current_activity:
            st.write(activity)
        
        # Auto-advance when complete
        if st.button("➡️ Continue to Results Review", type="primary"):
            guidance.advance_step(GuidanceStep.RESULTS_REVIEW)
            st.rerun()
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Job Submission"):
            guidance.advance_step(GuidanceStep.JOB_SUBMISSION)
            st.rerun()
        
        if st.button("📊 View All Jobs"):
            st.info("Job list will be displayed")

def create_results_review_step(guidance: WJPGuidanceSystem):
    """Create results review step interface."""
    st.header("📈 Results Review")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✅ Processing Complete!")
        
        st.success("🎉 Your waterjet project has been successfully processed!")
        
        # Results summary
        st.markdown("### 📊 Analysis Results")
        
        col_objects, col_length, col_cost = st.columns(3)
        
        with col_objects:
            st.metric("Total Objects", "15")
        
        with col_length:
            st.metric("Cut Length", "6.4 m")
        
        with col_cost:
            st.metric("Total Cost", "₹3,400")
        
        # Detailed metrics
        st.markdown("### 📋 Detailed Metrics")
        
        metrics_data = {
            "Design Code": "SR06",
            "Material": "Tan Brown Granite",
            "Thickness": "25 mm",
            "Total Area": "125,000 mm²",
            "Cutting Time": "24.3 min",
            "Geometry Violations": "0",
            "Complexity Rating": "Low",
            "Quality Score": "8.5/10"
        }
        
        for key, value in metrics_data.items():
            col_key, col_value = st.columns([1, 1])
            with col_key:
                st.write(f"**{key}:**")
            with col_value:
                st.write(value)
        
        # Layer breakdown
        st.markdown("### 🏗️ Layer Breakdown")
        
        layer_data = {
            "OUTER": 2,
            "COMPLEX": 6,
            "DECOR": 7
        }
        
        for layer, count in layer_data.items():
            st.write(f"**{layer}:** {count} objects")
        
        # Quality assessment
        st.markdown("### 🎯 Quality Assessment")
        
        quality_items = [
            "✅ All polylines are properly closed",
            "✅ Cut spacing meets minimum requirements",
            "✅ Corner radii are adequate",
            "✅ No geometry violations detected",
            "✅ Design complexity is manageable"
        ]
        
        for item in quality_items:
            st.write(item)
        
        # Cost breakdown
        st.markdown("### 💰 Cost Breakdown")
        
        cost_breakdown = {
            "Material Cost": "₹2,550",
            "Cutting Cost": "₹850",
            "Total Cost": "₹3,400"
        }
        
        for item, cost in cost_breakdown.items():
            st.write(f"**{item}:** {cost}")
    
    with col2:
        st.markdown("### 💡 Results Tips")
        
        tips = guidance.get_contextual_tips("results_review")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Review Checklist")
        
        warnings = guidance.get_warnings_for_context("results_review")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📈 Performance Summary")
        
        performance_summary = [
            "✅ Processing completed successfully",
            "✅ No geometry violations found",
            "✅ Cost within expected range",
            "✅ Quality score above threshold",
            "✅ Ready for production"
        ]
        
        for item in performance_summary:
            st.write(item)
        
        st.markdown("### 🎯 Next Steps")
        
        next_steps = [
            "1. Review all analysis results",
            "2. Download professional reports",
            "3. Save files to secure location",
            "4. Consider submitting additional projects"
        ]
        
        for step in next_steps:
            st.write(step)
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Processing Monitoring"):
            guidance.advance_step(GuidanceStep.PROCESSING_MONITORING)
            st.rerun()
        
        if st.button("➡️ Continue to Report Download", type="primary"):
            guidance.advance_step(GuidanceStep.REPORT_DOWNLOAD)
            st.rerun()

def create_report_download_step(guidance: WJPGuidanceSystem):
    """Create report download step interface."""
    st.header("📥 Report Download")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Available Reports")
        
        # Report types
        reports = [
            {
                "name": "Professional PDF Report",
                "description": "Complete project report with all visualizations and analysis",
                "file": "WJP_SR06_TANB_25_REPORT_V1_20251008.pdf",
                "size": "2.3 MB",
                "format": "PDF"
            },
            {
                "name": "Analysis Data (JSON)",
                "description": "Structured data for system integration and automation",
                "file": "WJP_SR06_TANB_25_ANALYSIS_V1_20251008.json",
                "size": "15 KB",
                "format": "JSON"
            },
            {
                "name": "Cutting Report (CSV)",
                "description": "Detailed metrics and data for analysis and integration",
                "file": "WJP_SR06_TANB_25_ANALYSIS_V1_20251008.csv",
                "size": "8 KB",
                "format": "CSV"
            },
            {
                "name": "Design Image (PNG)",
                "description": "Original design visualization",
                "file": "WJP_SR06_TANB_25_DESIGN_V1_20251008.png",
                "size": "1.2 MB",
                "format": "PNG"
            },
            {
                "name": "Analysis Visualization (PNG)",
                "description": "Analysis results with color-coded layers",
                "file": "WJP_SR06_TANB_25_ANALYSIS_V1_20251008.png",
                "size": "1.8 MB",
                "format": "PNG"
            },
            {
                "name": "DXF File",
                "description": "Cutting-ready DXF file for production",
                "file": "WJP_SR06_TANB_25_RAW_V1_20251008.dxf",
                "size": "45 KB",
                "format": "DXF"
            }
        ]
        
        # Download buttons
        for report in reports:
            with st.expander(f"📄 {report['name']} ({report['format']})"):
                st.write(f"**Description:** {report['description']}")
                st.write(f"**File:** {report['file']}")
                st.write(f"**Size:** {report['size']}")
                
                # Mock download button
                if st.button(f"📥 Download {report['format']}", key=f"download_{report['format']}"):
                    st.success(f"✅ {report['name']} downloaded successfully!")
        
        # Batch download
        st.markdown("### 📦 Batch Download")
        
        if st.button("📦 Download All Files", type="primary", use_container_width=True):
            st.success("✅ All files downloaded successfully!")
            st.info("Files saved to your Downloads folder")
        
        # File preview
        st.markdown("### 👁️ File Preview")
        
        preview_type = st.selectbox(
            "Preview File:",
            ["Design Image", "Analysis Visualization", "PDF Report"],
            help="Select a file to preview"
        )
        
        if preview_type == "Design Image":
            st.image("https://via.placeholder.com/400x300/4CAF50/FFFFFF?text=Design+Image", caption="Design Image Preview")
        elif preview_type == "Analysis Visualization":
            st.image("https://via.placeholder.com/400x300/2196F3/FFFFFF?text=Analysis+Viz", caption="Analysis Visualization Preview")
        elif preview_type == "PDF Report":
            st.info("PDF preview will be displayed here")
    
    with col2:
        st.markdown("### 💡 Download Tips")
        
        tips = guidance.get_contextual_tips("report_download")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Important Notes")
        
        warnings = guidance.get_warnings_for_context("report_download")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📁 File Organization")
        
        file_organization = [
            "📄 PDF Report: Professional presentation",
            "📊 CSV Report: Data analysis",
            "🔧 JSON Report: System integration",
            "🎨 PNG Images: Visual reference",
            "⚙️ DXF File: Production use"
        ]
        
        for item in file_organization:
            st.write(item)
        
        st.markdown("### 🎯 Recommended Actions")
        
        recommended_actions = [
            "1. Download PDF report for client presentation",
            "2. Save DXF file for production use",
            "3. Keep JSON/CSV for future reference",
            "4. Archive all files in project folder"
        ]
        
        for action in recommended_actions:
            st.write(action)
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Results Review"):
            guidance.advance_step(GuidanceStep.RESULTS_REVIEW)
            st.rerun()
        
        if st.button("➡️ Complete Process", type="primary"):
            guidance.advance_step(GuidanceStep.COMPLETION)
            st.rerun()

def create_completion_step(guidance: WJPGuidanceSystem):
    """Create completion step interface."""
    st.header("🎉 Project Complete")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎊 Congratulations!")
        
        st.success("🎉 Your waterjet project has been successfully completed!")
        
        # Completion summary
        st.markdown("### 📊 Project Summary")
        
        summary_data = {
            "Job ID": "SR06",
            "Material": "Tan Brown Granite",
            "Dimensions": "24×24 inch",
            "Total Cost": "₹3,400",
            "Processing Time": "3.2 minutes",
            "Quality Score": "8.5/10",
            "Files Generated": "6 files",
            "Status": "✅ Complete"
        }
        
        for key, value in summary_data.items():
            col_key, col_value = st.columns([1, 1])
            with col_key:
                st.write(f"**{key}:**")
            with col_value:
                st.write(value)
        
        # What was accomplished
        st.markdown("### ✅ What Was Accomplished")
        
        accomplishments = [
            "✅ Design image generated from your prompt",
            "✅ Image converted to cutting-ready DXF file",
            "✅ Geometry analyzed and validated",
            "✅ Cost and time calculations completed",
            "✅ Quality assessment performed",
            "✅ Professional PDF report generated",
            "✅ All files organized with professional naming"
        ]
        
        for accomplishment in accomplishments:
            st.write(accomplishment)
        
        # Next steps
        st.markdown("### 🚀 What's Next?")
        
        st.markdown("#### Option 1: Create Another Project")
        if st.button("🎨 Create Another Project", type="primary"):
            guidance.advance_step(GuidanceStep.MATERIAL_SELECTION)
            st.rerun()
        
        st.markdown("#### Option 2: Batch Processing")
        if st.button("📦 Start Batch Processing"):
            st.info("Redirecting to batch processing interface...")
        
        st.markdown("#### Option 3: System Management")
        if st.button("⚙️ View System Status"):
            st.info("Redirecting to system status...")
    
    with col2:
        st.markdown("### 💡 Completion Tips")
        
        tips = guidance.get_contextual_tips("completion")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### 📈 Performance Metrics")
        
        performance_metrics = {
            "Total Jobs Today": "13",
            "Success Rate": "94.2%",
            "Average Processing Time": "3.2 min",
            "Total Cost Processed": "₹42,500",
            "Files Generated": "78 files"
        }
        
        for metric, value in performance_metrics.items():
            st.metric(metric, value)
        
        st.markdown("### 🎯 System Capabilities")
        
        capabilities = [
            "✅ Individual project processing",
            "✅ Batch processing for multiple projects",
            "✅ Real-time progress monitoring",
            "✅ Professional reporting",
            "✅ Material-specific cost calculations",
            "✅ Quality assessment and validation",
            "✅ Learning system integration"
        ]
        
        for capability in capabilities:
            st.write(capability)
        
        st.markdown("### 🏆 Achievement Unlocked!")
        
        st.success("🎯 **First Project Complete**")
        st.write("You've successfully completed your first waterjet project using the WJP Automation Pipeline!")
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Report Download"):
            guidance.advance_step(GuidanceStep.REPORT_DOWNLOAD)
            st.rerun()
        
        if st.button("🏠 Back to Welcome"):
            guidance.advance_step(GuidanceStep.WELCOME)
            st.rerun()

if __name__ == "__main__":
    create_guided_interface()
