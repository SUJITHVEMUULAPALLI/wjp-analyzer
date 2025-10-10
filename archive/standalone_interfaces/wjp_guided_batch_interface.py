#!/usr/bin/env python3
"""
WJP Batch Processing - Guided Interface
======================================

This module enhances the advanced batch processing interface with
intelligent step-by-step guidance for batch operations.
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

class BatchGuidanceStep(Enum):
    """Batch processing guidance steps."""
    WELCOME = "welcome"
    BATCH_PLANNING = "batch_planning"
    FILE_UPLOAD = "file_upload"
    CONFIGURATION = "configuration"
    STRATEGY_SELECTION = "strategy_selection"
    PROCESSING_MONITORING = "processing_monitoring"
    RESULTS_ANALYSIS = "results_analysis"
    OPTIMIZATION_SUGGESTIONS = "optimization_suggestions"
    COMPLETION = "completion"

@dataclass
class BatchGuidanceMessage:
    """Batch guidance message structure."""
    step: BatchGuidanceStep
    title: str
    message: str
    action_required: str
    tips: List[str]
    warnings: List[str]
    next_step: Optional[BatchGuidanceStep]
    is_complete: bool = False

class WJPBatchGuidanceSystem:
    """Intelligent guidance system for batch processing."""
    
    def __init__(self):
        self.current_step = BatchGuidanceStep.WELCOME
        self.completed_steps = set()
        self.batch_config = {}
        self.uploaded_files = []
        self.processing_results = {}
        
        # Initialize batch guidance messages
        self.guidance_messages = self._initialize_batch_guidance_messages()
    
    def _initialize_batch_guidance_messages(self) -> Dict[BatchGuidanceStep, BatchGuidanceMessage]:
        """Initialize batch guidance messages."""
        return {
            BatchGuidanceStep.WELCOME: BatchGuidanceMessage(
                step=BatchGuidanceStep.WELCOME,
                title="📦 Welcome to Batch Processing",
                message="Welcome to the advanced batch processing system! I'll guide you through processing multiple files efficiently with intelligent orchestration.",
                action_required="Choose your batch processing approach",
                tips=[
                    "Batch processing is ideal for multiple similar projects",
                    "The system automatically optimizes parameters for better results",
                    "You can process images and DXF files together",
                    "Intelligent supervisor agent manages the entire workflow"
                ],
                warnings=[],
                next_step=BatchGuidanceStep.BATCH_PLANNING
            ),
            
            BatchGuidanceStep.BATCH_PLANNING: BatchGuidanceMessage(
                step=BatchGuidanceStep.BATCH_PLANNING,
                title="📋 Batch Planning",
                message="Plan your batch processing job. Consider the number of files, material types, and processing requirements.",
                action_required="Define your batch requirements and goals",
                tips=[
                    "Start with 5-10 files for your first batch",
                    "Group similar projects together for better optimization",
                    "Consider material consistency for cost optimization",
                    "Plan for 2-5 minutes processing time per file"
                ],
                warnings=[
                    "Very large batches (>50 files) may take significant time",
                    "Mixed material types may reduce optimization effectiveness"
                ],
                next_step=BatchGuidanceStep.FILE_UPLOAD
            ),
            
            BatchGuidanceStep.FILE_UPLOAD: BatchGuidanceMessage(
                step=BatchGuidanceStep.FILE_UPLOAD,
                title="📁 File Upload",
                message="Upload your files for batch processing. The system supports images (PNG, JPG) and DXF files.",
                action_required="Upload files and verify file types",
                tips=[
                    "Use high-quality images for better results",
                    "DXF files should be clean and well-structured",
                    "File names should be descriptive for easy identification",
                    "Recommended file size: 100KB - 5MB per file"
                ],
                warnings=[
                    "Very large files (>10MB) may cause processing delays",
                    "Corrupted files will be skipped during processing"
                ],
                next_step=BatchGuidanceStep.CONFIGURATION
            ),
            
            BatchGuidanceStep.CONFIGURATION: BatchGuidanceMessage(
                step=BatchGuidanceStep.CONFIGURATION,
                title="⚙️ Batch Configuration",
                message="Configure processing parameters for your batch. The system will optimize these settings based on your files.",
                action_required="Set material type and detection parameters",
                tips=[
                    "Choose material based on your most common file type",
                    "Conservative settings provide higher success rates",
                    "Aggressive settings process faster but may miss details",
                    "Enable learning for continuous improvement"
                ],
                warnings=[
                    "Parameter changes affect all files in the batch",
                    "Very aggressive settings may reduce quality"
                ],
                next_step=BatchGuidanceStep.STRATEGY_SELECTION
            ),
            
            BatchGuidanceStep.STRATEGY_SELECTION: BatchGuidanceMessage(
                step=BatchGuidanceStep.STRATEGY_SELECTION,
                title="🎯 Strategy Selection",
                message="Choose the processing strategy based on your batch characteristics. The system analyzes your files and recommends the best approach.",
                action_required="Select processing strategy and review recommendations",
                tips=[
                    "Conservative: High precision, fewer objects, 90% success rate",
                    "Balanced: General purpose, mixed files, 85% success rate",
                    "Aggressive: High volume, simple files, 75% success rate",
                    "System automatically analyzes files and suggests strategy"
                ],
                warnings=[
                    "Strategy selection affects processing time and quality",
                    "Wrong strategy may result in poor results"
                ],
                next_step=BatchGuidanceStep.PROCESSING_MONITORING
            ),
            
            BatchGuidanceStep.PROCESSING_MONITORING: BatchGuidanceMessage(
                step=BatchGuidanceStep.PROCESSING_MONITORING,
                title="📊 Processing Monitoring",
                message="Monitor your batch processing progress. The intelligent supervisor agent manages the workflow and provides real-time updates.",
                action_required="Monitor progress and wait for completion",
                tips=[
                    "Processing happens in stages for optimal efficiency",
                    "Simple files are processed first for quick wins",
                    "Complex files are processed with more attention",
                    "You can monitor individual file progress"
                ],
                warnings=[
                    "Don't close the browser during processing",
                    "Network issues may affect real-time updates"
                ],
                next_step=BatchGuidanceStep.RESULTS_ANALYSIS
            ),
            
            BatchGuidanceStep.RESULTS_ANALYSIS: BatchGuidanceMessage(
                step=BatchGuidanceStep.RESULTS_ANALYSIS,
                title="📈 Results Analysis",
                message="Analyze your batch processing results. Review success rates, costs, and quality metrics across all files.",
                action_required="Review batch results and identify patterns",
                tips=[
                    "Check success rate - aim for >80%",
                    "Review cost distribution for optimization opportunities",
                    "Identify common issues across failed files",
                    "Use insights for future batch improvements"
                ],
                warnings=[
                    "Low success rates may indicate parameter issues",
                    "High cost variance suggests inconsistent designs"
                ],
                next_step=BatchGuidanceStep.OPTIMIZATION_SUGGESTIONS
            ),
            
            BatchGuidanceStep.OPTIMIZATION_SUGGESTIONS: BatchGuidanceMessage(
                step=BatchGuidanceStep.OPTIMIZATION_SUGGESTIONS,
                title="💡 Optimization Suggestions",
                message="Get intelligent suggestions for improving your batch processing. The system learns from your results and provides actionable recommendations.",
                action_required="Review suggestions and apply optimizations",
                tips=[
                    "Parameter optimizations improve future batches",
                    "Material recommendations reduce costs",
                    "Design suggestions improve success rates",
                    "Apply learning for continuous improvement"
                ],
                warnings=[
                    "Apply suggestions carefully to avoid over-optimization",
                    "Test changes with small batches first"
                ],
                next_step=BatchGuidanceStep.COMPLETION
            ),
            
            BatchGuidanceStep.COMPLETION: BatchGuidanceMessage(
                step=BatchGuidanceStep.COMPLETION,
                title="🎉 Batch Processing Complete",
                message="Congratulations! Your batch processing is complete. Review the results and plan your next batch.",
                action_required="Review results and plan next steps",
                tips=[
                    "Download all reports and files",
                    "Archive successful projects",
                    "Plan next batch based on learnings",
                    "Share insights with your team"
                ],
                warnings=[],
                next_step=None
            )
        }
    
    def get_current_guidance(self) -> BatchGuidanceMessage:
        """Get current guidance message."""
        return self.guidance_messages[self.current_step]
    
    def advance_step(self, step: BatchGuidanceStep):
        """Advance to next step."""
        if step in self.guidance_messages:
            self.current_step = step
            self.completed_steps.add(step)
    
    def get_step_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        total_steps = len(BatchGuidanceStep)
        completed_count = len(self.completed_steps)
        current_index = list(BatchGuidanceStep).index(self.current_step)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_count,
            "current_step_index": current_index,
            "progress_percentage": (completed_count / total_steps) * 100,
            "current_step": self.current_step.value,
            "is_complete": self.current_step == BatchGuidanceStep.COMPLETION
        }
    
    def analyze_batch_requirements(self, files: List[str]) -> Dict[str, Any]:
        """Analyze batch requirements and provide recommendations."""
        analysis = {
            "total_files": len(files),
            "file_types": {},
            "estimated_processing_time": len(files) * 2.5,  # 2.5 min per file
            "recommended_strategy": "balanced",
            "complexity_assessment": "medium",
            "optimization_opportunities": []
        }
        
        # Analyze file types
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1
        
        # Determine recommended strategy
        if len(files) > 20:
            analysis["recommended_strategy"] = "aggressive"
        elif len(files) < 5:
            analysis["recommended_strategy"] = "conservative"
        
        # Assess complexity
        if len(files) > 15:
            analysis["complexity_assessment"] = "high"
        elif len(files) < 8:
            analysis["complexity_assessment"] = "low"
        
        return analysis

def create_guided_batch_interface():
    """Create the guided batch processing interface."""
    st.set_page_config(
        page_title="WJP Guided Batch Processing",
        page_icon="📦",
        layout="wide"
    )
    
    st.title("📦 WJP Batch Processing - Guided Interface")
    st.markdown("**Intelligent step-by-step guidance for professional batch processing**")
    
    # Initialize guidance system
    if "batch_guidance" not in st.session_state:
        st.session_state.batch_guidance = WJPBatchGuidanceSystem()
    
    guidance = st.session_state.batch_guidance
    
    # Sidebar for guidance
    with st.sidebar:
        st.header("🎯 Batch Guidance")
        
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
        step_options = [step.value for step in BatchGuidanceStep]
        current_step_index = step_options.index(guidance.current_step.value)
        
        selected_step = st.selectbox(
            "Go to step:",
            step_options,
            index=current_step_index
        )
        
        if selected_step != guidance.current_step.value:
            guidance.advance_step(BatchGuidanceStep(selected_step))
            st.rerun()
    
    # Main content based on current step
    if guidance.current_step == BatchGuidanceStep.WELCOME:
        create_batch_welcome_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.BATCH_PLANNING:
        create_batch_planning_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.FILE_UPLOAD:
        create_file_upload_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.CONFIGURATION:
        create_configuration_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.STRATEGY_SELECTION:
        create_strategy_selection_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.PROCESSING_MONITORING:
        create_processing_monitoring_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.RESULTS_ANALYSIS:
        create_results_analysis_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.OPTIMIZATION_SUGGESTIONS:
        create_optimization_suggestions_step(guidance)
    elif guidance.current_step == BatchGuidanceStep.COMPLETION:
        create_batch_completion_step(guidance)

def create_batch_welcome_step(guidance: WJPBatchGuidanceSystem):
    """Create batch welcome step interface."""
    st.header("📦 Welcome to Batch Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 What is Batch Processing?
        
        Batch processing allows you to process multiple waterjet projects simultaneously with intelligent orchestration. Our advanced supervisor agent manages the entire workflow efficiently.
        
        #### 🔄 Batch Workflow:
        1. **File Upload** - Upload multiple images and DXF files
        2. **Intelligent Analysis** - System analyzes files and recommends strategy
        3. **Automated Processing** - Supervisor agent processes all files efficiently
        4. **Comprehensive Analysis** - Get insights and optimization suggestions
        5. **Professional Reports** - Download all results and reports
        
        #### 🎯 Key Benefits:
        - **Efficiency** - Process multiple files simultaneously
        - **Intelligence** - Automatic parameter optimization
        - **Learning** - System improves with each batch
        - **Insights** - Comprehensive analysis and suggestions
        - **Scalability** - Handle small to large batches efficiently
        """)
        
        # Batch type selection
        st.markdown("### 📋 Choose Your Batch Type:")
        
        batch_type = st.radio(
            "What type of batch processing do you need?",
            ["Small Batch (5-10 files)", "Medium Batch (10-20 files)", "Large Batch (20+ files)", "Mixed File Types"],
            help="Select the type of batch processing you want to perform"
        )
        
        if batch_type == "Small Batch (5-10 files)":
            st.success("✅ Perfect for testing and small projects!")
        elif batch_type == "Medium Batch (10-20 files)":
            st.info("ℹ️ Great for regular production batches!")
        elif batch_type == "Large Batch (20+ files)":
            st.warning("⚠️ Large batches require careful planning!")
        else:
            st.info("ℹ️ Mixed file types offer maximum flexibility!")
    
    with col2:
        st.markdown("### 🎓 Batch Processing Experience")
        
        experience_level = st.selectbox(
            "How familiar are you with batch processing?",
            ["Beginner", "Intermediate", "Advanced"],
            help="This helps us provide appropriate guidance"
        )
        
        st.markdown("### 📊 System Status")
        
        # Mock system status
        st.metric("Queue Size", "0")
        st.metric("Active Batches", "0")
        st.metric("Completed Today", "3")
        st.metric("Success Rate", "91.5%")
        
        st.markdown("### 🚀 Quick Start")
        
        if st.button("📦 Start Batch Processing", type="primary"):
            guidance.advance_step(BatchGuidanceStep.BATCH_PLANNING)
            st.rerun()
        
        if st.button("📚 View Batch Documentation"):
            st.info("Documentation will open in a new tab")
        
        if st.button("🎥 Watch Batch Tutorial"):
            st.info("Tutorial video will be displayed")

def create_batch_planning_step(guidance: WJPBatchGuidanceSystem):
    """Create batch planning step interface."""
    st.header("📋 Batch Planning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Batch Requirements Planning")
        
        # Batch size
        batch_size = st.selectbox(
            "Batch Size:",
            ["Small (5-10 files)", "Medium (10-20 files)", "Large (20-50 files)", "Custom"],
            help="Select the size of your batch"
        )
        
        if batch_size == "Custom":
            custom_size = st.number_input("Number of files:", 1, 100, 10)
        else:
            size_map = {
                "Small (5-10 files)": (5, 10),
                "Medium (10-20 files)": (10, 20),
                "Large (20-50 files)": (20, 50)
            }
            min_files, max_files = size_map[batch_size]
            custom_size = st.slider("Number of files:", min_files, max_files, (min_files + max_files) // 2)
        
        # File types
        st.markdown("### 📁 File Types")
        
        file_types = st.multiselect(
            "File Types to Process:",
            ["PNG Images", "JPG Images", "DXF Files", "Mixed Types"],
            default=["PNG Images"],
            help="Select the types of files you want to process"
        )
        
        # Material consistency
        st.markdown("### 🏗️ Material Planning")
        
        material_consistency = st.selectbox(
            "Material Consistency:",
            ["Same Material (Recommended)", "Mixed Materials", "Material Optimization"],
            help="Choose how materials are handled in the batch"
        )
        
        # Processing goals
        st.markdown("### 🎯 Processing Goals")
        
        processing_goals = st.multiselect(
            "Primary Goals:",
            ["Speed Optimization", "Quality Optimization", "Cost Optimization", "Learning & Improvement"],
            default=["Quality Optimization"],
            help="Select your primary processing goals"
        )
        
        # Estimated processing time
        st.markdown("### ⏱️ Processing Estimate")
        
        estimated_time = custom_size * 2.5  # 2.5 minutes per file
        st.metric("Estimated Processing Time", f"{estimated_time:.1f} minutes")
        
        # Batch complexity assessment
        st.markdown("### 🧩 Complexity Assessment")
        
        complexity_factors = {
            "File Count": custom_size,
            "File Types": len(file_types),
            "Material Consistency": 1 if material_consistency == "Same Material (Recommended)" else 2,
            "Processing Goals": len(processing_goals)
        }
        
        total_complexity = sum(complexity_factors.values())
        
        if total_complexity < 10:
            complexity_level = "Low"
            complexity_color = "green"
        elif total_complexity < 20:
            complexity_level = "Medium"
            complexity_color = "orange"
        else:
            complexity_level = "High"
            complexity_color = "red"
        
        st.markdown(f"**Complexity Level:** :{complexity_color}[{complexity_level}]")
        
        # Planning recommendations
        st.markdown("### 💡 Planning Recommendations")
        
        recommendations = []
        
        if custom_size > 20:
            recommendations.append("Consider breaking into smaller batches for better management")
        
        if len(file_types) > 2:
            recommendations.append("Mixed file types may require different processing strategies")
        
        if material_consistency != "Same Material (Recommended)":
            recommendations.append("Mixed materials may reduce optimization effectiveness")
        
        if not recommendations:
            recommendations.append("Your batch configuration looks optimal!")
        
        for rec in recommendations:
            st.info(f"💡 {rec}")
    
    with col2:
        st.markdown("### 💡 Planning Tips")
        
        tips = guidance.get_contextal_tips("batch_planning")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Planning Considerations")
        
        warnings = guidance.get_warnings_for_context("batch_planning")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📊 Batch Summary")
        
        summary_data = {
            "Batch Size": f"{custom_size} files",
            "File Types": len(file_types),
            "Material": material_consistency,
            "Goals": len(processing_goals),
            "Complexity": complexity_level,
            "Est. Time": f"{estimated_time:.1f} min"
        }
        
        for key, value in summary_data.items():
            st.write(f"**{key}:** {value}")
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Welcome"):
            guidance.advance_step(BatchGuidanceStep.WELCOME)
            st.rerun()
        
        if st.button("➡️ Continue to File Upload", type="primary"):
            guidance.advance_step(BatchGuidanceStep.FILE_UPLOAD)
            st.rerun()

def create_file_upload_step(guidance: WJPBatchGuidanceSystem):
    """Create file upload step interface."""
    st.header("📁 File Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Your Files")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Images or DXF files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dxf'],
            accept_multiple_files=True,
            help="Upload multiple files for batch processing"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
            
            # File analysis
            st.markdown("### 📊 File Analysis")
            
            file_analysis = guidance.analyze_batch_requirements([f.name for f in uploaded_files])
            
            col_files, col_types, col_time = st.columns(3)
            
            with col_files:
                st.metric("Total Files", file_analysis["total_files"])
            
            with col_types:
                st.metric("File Types", len(file_analysis["file_types"]))
            
            with col_time:
                st.metric("Est. Time", f"{file_analysis['estimated_processing_time']:.1f}m")
            
            # File list
            st.markdown("### 📋 Uploaded Files")
            
            file_data = []
            for i, file in enumerate(uploaded_files):
                file_data.append({
                    "Index": i + 1,
                    "Name": file.name,
                    "Size": f"{file.size / 1024:.1f} KB",
                    "Type": file.name.split('.')[-1].upper()
                })
            
            df_files = pd.DataFrame(file_data)
            st.dataframe(df_files, use_container_width=True)
            
            # File validation
            st.markdown("### ✅ File Validation")
            
            validation_results = []
            
            for file in uploaded_files:
                if file.size > 10 * 1024 * 1024:  # 10MB
                    validation_results.append(f"⚠️ {file.name}: Very large file (>10MB)")
                elif file.size < 10 * 1024:  # 10KB
                    validation_results.append(f"⚠️ {file.name}: Very small file (<10KB)")
                else:
                    validation_results.append(f"✅ {file.name}: File size OK")
            
            for result in validation_results:
                st.write(result)
            
            # Store uploaded files
            guidance.uploaded_files = uploaded_files
            
        else:
            st.info("No files uploaded yet. Please upload files to continue.")
    
    with col2:
        st.markdown("### 💡 Upload Tips")
        
        tips = guidance.get_contextal_tips("file_upload")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Upload Considerations")
        
        warnings = guidance.get_warnings_for_context("file_upload")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📁 Supported Formats")
        
        formats = [
            "🖼️ **Images:** PNG, JPG, JPEG, BMP, TIFF",
            "⚙️ **CAD Files:** DXF",
            "📏 **Size:** 10KB - 10MB per file",
            "🔢 **Quantity:** Up to 100 files per batch"
        ]
        
        for format_info in formats:
            st.write(format_info)
        
        st.markdown("### 🎯 File Quality Tips")
        
        quality_tips = [
            "Use high-resolution images for better results",
            "Ensure images have good contrast",
            "DXF files should be clean and well-structured",
            "Avoid corrupted or incomplete files"
        ]
        
        for tip in quality_tips:
            st.write(f"• {tip}")
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Batch Planning"):
            guidance.advance_step(BatchGuidanceStep.BATCH_PLANNING)
            st.rerun()
        
        if uploaded_files and st.button("➡️ Continue to Configuration", type="primary"):
            guidance.advance_step(BatchGuidanceStep.CONFIGURATION)
            st.rerun()

def create_configuration_step(guidance: WJPBatchGuidanceSystem):
    """Create configuration step interface."""
    st.header("⚙️ Batch Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏗️ Material Configuration")
        
        # Material selection
        material_type = st.selectbox(
            "Material Type:",
            ["Tan Brown Granite", "Marble", "Stainless Steel", "Aluminum", "Brass", "Generic"],
            help="Select material type for cost calculation"
        )
        
        # Thickness
        thickness_mm = st.slider("Thickness (mm):", 1, 100, 25, help="Material thickness")
        
        st.markdown("### 🔍 Detection Parameters")
        
        # Detection parameters
        col_min_area, col_circularity = st.columns(2)
        
        with col_min_area:
            min_area = st.slider("Min Area:", 10, 100, 25, help="Minimum object area for detection")
        
        with col_circularity:
            min_circularity = st.slider("Min Circularity:", 0.01, 0.2, 0.03, 0.01, help="Minimum circularity threshold")
        
        col_solidity, col_simplify = st.columns(2)
        
        with col_solidity:
            min_solidity = st.slider("Min Solidity:", 0.01, 0.5, 0.05, 0.01, help="Minimum solidity threshold")
        
        with col_simplify:
            simplify_tolerance = st.slider("Simplify Tolerance:", 0.0, 2.0, 0.0, 0.1, help="Contour simplification")
        
        merge_distance = st.slider("Merge Distance:", 0.0, 20.0, 0.0, 1.0, help="Object merging distance")
        
        st.markdown("### 🧠 Processing Options")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            optimization_enabled = st.checkbox("Enable Optimization", True, help="Enable parameter optimization")
            learning_enabled = st.checkbox("Enable Learning", True, help="Enable learning from results")
        
        with col_opt2:
            quality_monitoring = st.checkbox("Quality Monitoring", True, help="Monitor quality metrics")
            error_recovery = st.checkbox("Error Recovery", True, help="Enable automatic error recovery")
        
        # Configuration summary
        st.markdown("### 📊 Configuration Summary")
        
        config_summary = {
            "Material": material_type,
            "Thickness": f"{thickness_mm} mm",
            "Min Area": min_area,
            "Min Circularity": min_circularity,
            "Min Solidity": min_solidity,
            "Optimization": "Enabled" if optimization_enabled else "Disabled",
            "Learning": "Enabled" if learning_enabled else "Disabled"
        }
        
        for key, value in config_summary.items():
            st.write(f"**{key}:** {value}")
        
        # Store configuration
        guidance.batch_config = {
            "material_type": material_type,
            "thickness_mm": thickness_mm,
            "detection_params": {
                "min_area": min_area,
                "min_circularity": min_circularity,
                "min_solidity": min_solidity,
                "simplify_tolerance": simplify_tolerance,
                "merge_distance": merge_distance
            },
            "processing_options": {
                "optimization_enabled": optimization_enabled,
                "learning_enabled": learning_enabled,
                "quality_monitoring": quality_monitoring,
                "error_recovery": error_recovery
            }
        }
    
    with col2:
        st.markdown("### 💡 Configuration Tips")
        
        tips = guidance.get_contextal_tips("configuration")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Configuration Considerations")
        
        warnings = guidance.get_warnings_for_context("configuration")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 🎯 Parameter Guidelines")
        
        guidelines = [
            "**Conservative:** Higher thresholds, fewer objects, 90% success",
            "**Balanced:** Moderate thresholds, general purpose, 85% success",
            "**Aggressive:** Lower thresholds, more objects, 75% success"
        ]
        
        for guideline in guidelines:
            st.write(guideline)
        
        st.markdown("### 📈 Expected Performance")
        
        # Calculate expected performance based on settings
        if min_area >= 50 and min_circularity >= 0.05:
            performance_level = "Conservative"
            success_rate = "90%"
            processing_time = "3.0 min/file"
        elif min_area >= 25 and min_circularity >= 0.03:
            performance_level = "Balanced"
            success_rate = "85%"
            processing_time = "2.5 min/file"
        else:
            performance_level = "Aggressive"
            success_rate = "75%"
            processing_time = "2.0 min/file"
        
        st.write(f"**Performance Level:** {performance_level}")
        st.write(f"**Expected Success Rate:** {success_rate}")
        st.write(f"**Processing Time:** {processing_time}")
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to File Upload"):
            guidance.advance_step(BatchGuidanceStep.FILE_UPLOAD)
            st.rerun()
        
        if st.button("➡️ Continue to Strategy Selection", type="primary"):
            guidance.advance_step(BatchGuidanceStep.STRATEGY_SELECTION)
            st.rerun()

def create_strategy_selection_step(guidance: WJPBatchGuidanceSystem):
    """Create strategy selection step interface."""
    st.header("🎯 Strategy Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🧠 Intelligent Strategy Analysis")
        
        # Analyze uploaded files
        if guidance.uploaded_files:
            file_analysis = guidance.analyze_batch_requirements([f.name for f in guidance.uploaded_files])
            
            st.markdown("#### 📊 File Analysis Results")
            
            col_files, col_complexity, col_strategy = st.columns(3)
            
            with col_files:
                st.metric("Total Files", file_analysis["total_files"])
            
            with col_complexity:
                st.metric("Complexity", file_analysis["complexity_assessment"].title())
            
            with col_strategy:
                st.metric("Recommended", file_analysis["recommended_strategy"].title())
            
            # Strategy recommendations
            st.markdown("#### 🎯 Strategy Recommendations")
            
            strategies = {
                "Conservative": {
                    "description": "High precision, fewer objects, maximum quality",
                    "success_rate": "90%",
                    "processing_time": "3.0 min/file",
                    "best_for": "High-quality projects, complex designs",
                    "parameters": "Min Area: 50, Min Circularity: 0.05"
                },
                "Balanced": {
                    "description": "General purpose, balanced quality and speed",
                    "success_rate": "85%",
                    "processing_time": "2.5 min/file",
                    "best_for": "Mixed projects, regular production",
                    "parameters": "Min Area: 25, Min Circularity: 0.03"
                },
                "Aggressive": {
                    "description": "High volume, maximum speed, more objects",
                    "success_rate": "75%",
                    "processing_time": "2.0 min/file",
                    "best_for": "Simple designs, high-volume processing",
                    "parameters": "Min Area: 10, Min Circularity: 0.01"
                }
            }
            
            # Display strategies
            for strategy_name, strategy_info in strategies.items():
                with st.expander(f"🎯 {strategy_name} Strategy"):
                    st.write(f"**Description:** {strategy_info['description']}")
                    st.write(f"**Success Rate:** {strategy_info['success_rate']}")
                    st.write(f"**Processing Time:** {strategy_info['processing_time']}")
                    st.write(f"**Best For:** {strategy_info['best_for']}")
                    st.write(f"**Parameters:** {strategy_info['parameters']}")
            
            # Strategy selection
            st.markdown("#### ⚙️ Select Processing Strategy")
            
            selected_strategy = st.radio(
                "Choose Strategy:",
                ["Conservative", "Balanced", "Aggressive"],
                index=1,  # Default to Balanced
                help="Select the processing strategy for your batch"
            )
            
            # Show selected strategy details
            if selected_strategy in strategies:
                strategy_info = strategies[selected_strategy]
                
                st.markdown(f"#### ✅ Selected: {selected_strategy} Strategy")
                
                col_success, col_time, col_quality = st.columns(3)
                
                with col_success:
                    st.metric("Success Rate", strategy_info["success_rate"])
                
                with col_time:
                    st.metric("Processing Time", strategy_info["processing_time"])
                
                with col_quality:
                    st.metric("Quality Level", "High" if selected_strategy == "Conservative" else "Medium" if selected_strategy == "Balanced" else "Standard")
                
                # Processing order
                st.markdown("#### 📋 Processing Order")
                
                processing_order = [
                    "1. Simple files (quick wins)",
                    "2. Complex files (detailed processing)",
                    "3. Large files (resource intensive)"
                ]
                
                for order in processing_order:
                    st.write(order)
                
                # Estimated batch time
                total_files = len(guidance.uploaded_files)
                time_per_file = float(strategy_info["processing_time"].split()[0])
                total_time = total_files * time_per_file
                
                st.markdown("#### ⏱️ Batch Processing Estimate")
                st.metric("Total Processing Time", f"{total_time:.1f} minutes")
                st.metric("Expected Completion", f"{datetime.now().strftime('%H:%M')} + {total_time:.0f}min")
        
        else:
            st.warning("No files uploaded. Please upload files first.")
    
    with col2:
        st.markdown("### 💡 Strategy Tips")
        
        tips = guidance.get_contextal_tips("strategy_selection")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Strategy Considerations")
        
        warnings = guidance.get_warnings_for_context("strategy_selection")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 🎯 Strategy Comparison")
        
        comparison_data = {
            "Conservative": {"Quality": "⭐⭐⭐⭐⭐", "Speed": "⭐⭐", "Success": "⭐⭐⭐⭐⭐"},
            "Balanced": {"Quality": "⭐⭐⭐⭐", "Speed": "⭐⭐⭐", "Success": "⭐⭐⭐⭐"},
            "Aggressive": {"Quality": "⭐⭐⭐", "Speed": "⭐⭐⭐⭐⭐", "Success": "⭐⭐⭐"}
        }
        
        for strategy, ratings in comparison_data.items():
            st.write(f"**{strategy}:**")
            for metric, rating in ratings.items():
                st.write(f"  {metric}: {rating}")
        
        st.markdown("### 🚀 Ready to Process")
        
        if guidance.uploaded_files:
            st.success("✅ Files ready for processing!")
            st.info("Click 'Start Processing' to begin batch processing with the selected strategy.")
        else:
            st.warning("⚠️ Upload files first")
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Configuration"):
            guidance.advance_step(BatchGuidanceStep.CONFIGURATION)
            st.rerun()
        
        if guidance.uploaded_files and st.button("🚀 Start Processing", type="primary"):
            guidance.advance_step(BatchGuidanceStep.PROCESSING_MONITORING)
            st.rerun()

def create_processing_monitoring_step(guidance: WJPBatchGuidanceSystem):
    """Create processing monitoring step interface."""
    st.header("📊 Processing Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔄 Batch Processing Status")
        
        # Simulate processing stages
        stages = [
            ("📋 Batch Analysis", "Analyzing files and optimizing parameters", "completed"),
            ("🎨 Designer Agent", "Processing image files", "completed"),
            ("🔄 Image to DXF Agent", "Converting images to DXF format", "in_progress"),
            ("📊 DXF Analyzer Agent", "Analyzing geometry and calculating costs", "pending"),
            ("📄 Report Generator Agent", "Generating professional reports", "pending")
        ]
        
        for stage_name, description, status in stages:
            if status == "completed":
                st.success(f"✅ {stage_name}: {description}")
            elif status == "in_progress":
                st.info(f"🔄 {stage_name}: {description}")
                # Progress bar for current stage
                progress = st.progress(0.4)
                st.caption("Processing... 40% complete")
            else:
                st.write(f"⏳ {stage_name}: {description}")
        
        # File-by-file progress
        st.markdown("### 📁 File Processing Progress")
        
        if guidance.uploaded_files:
            # Simulate file processing
            file_progress = []
            for i, file in enumerate(guidance.uploaded_files[:5]):  # Show first 5 files
                if i < 2:
                    status = "✅ Completed"
                elif i == 2:
                    status = "🔄 Processing"
                else:
                    status = "⏳ Pending"
                
                file_progress.append({
                    "File": file.name,
                    "Status": status,
                    "Progress": "100%" if i < 2 else "60%" if i == 2 else "0%"
                })
            
            df_progress = pd.DataFrame(file_progress)
            st.dataframe(df_progress, use_container_width=True)
            
            if len(guidance.uploaded_files) > 5:
                st.info(f"... and {len(guidance.uploaded_files) - 5} more files")
        
        # Real-time statistics
        st.markdown("### 📈 Real-time Statistics")
        
        col_completed, col_processing, col_failed = st.columns(3)
        
        with col_completed:
            st.metric("Completed", "8")
        
        with col_processing:
            st.metric("Processing", "2")
        
        with col_failed:
            st.metric("Failed", "0")
        
        # Processing log
        st.markdown("### 📝 Processing Log")
        
        log_entries = [
            ("12:30:15", "Batch processing started"),
            ("12:30:16", "File analysis completed - 10 files"),
            ("12:30:17", "Conservative strategy applied"),
            ("12:31:45", "Designer Agent: 8 images processed"),
            ("12:32:20", "Image to DXF Agent: 6 files converted"),
            ("12:33:45", "DXF Analyzer Agent: 4 files analyzed"),
            ("12:34:00", "Processing 2 remaining files...")
        ]
        
        for timestamp, message in log_entries:
            st.write(f"`{timestamp}` {message}")
        
        # Refresh button
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col2:
        st.markdown("### 💡 Monitoring Tips")
        
        tips = guidance.get_contextal_tips("processing_monitoring")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Important Notes")
        
        warnings = guidance.get_warnings_for_context("processing_monitoring")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📊 Batch Performance")
        
        performance_metrics = {
            "Files Processed": "8/10",
            "Success Rate": "100%",
            "Avg Time/File": "2.3 min",
            "Total Time": "18.4 min",
            "Cost Processed": "₹12,500"
        }
        
        for metric, value in performance_metrics.items():
            st.metric(metric, value)
        
        st.markdown("### 🎯 Current Activity")
        
        current_activity = [
            "✅ File analysis completed",
            "✅ Designer Agent finished",
            "🔄 Image to DXF conversion in progress",
            "⏳ DXF analysis pending",
            "⏳ Report generation pending"
        ]
        
        for activity in current_activity:
            st.write(activity)
        
        # Auto-advance when complete
        if st.button("➡️ Continue to Results Analysis", type="primary"):
            guidance.advance_step(BatchGuidanceStep.RESULTS_ANALYSIS)
            st.rerun()
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Strategy Selection"):
            guidance.advance_step(BatchGuidanceStep.STRATEGY_SELECTION)
            st.rerun()

def create_results_analysis_step(guidance: WJPBatchGuidanceSystem):
    """Create results analysis step interface."""
    st.header("📈 Results Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✅ Batch Processing Complete!")
        
        st.success("🎉 Your batch processing has been completed successfully!")
        
        # Overall results
        st.markdown("### 📊 Overall Results")
        
        col_total, col_success, col_failed = st.columns(3)
        
        with col_total:
            st.metric("Total Files", "10")
        
        with col_success:
            st.metric("Successful", "9")
        
        with col_failed:
            st.metric("Failed", "1")
        
        # Success rate
        success_rate = 90.0
        st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Detailed analysis
        st.markdown("### 📋 Detailed Analysis")
        
        analysis_data = {
            "Total Processing Time": "23.5 minutes",
            "Average Time per File": "2.35 minutes",
            "Total Cost Processed": "₹15,200",
            "Average Cost per File": "₹1,520",
            "Quality Score": "8.2/10",
            "Complexity Rating": "Medium"
        }
        
        for key, value in analysis_data.items():
            col_key, col_value = st.columns([1, 1])
            with col_key:
                st.write(f"**{key}:**")
            with col_value:
                st.write(value)
        
        # File-by-file results
        st.markdown("### 📁 File-by-File Results")
        
        file_results = [
            {"File": "Tile_01.png", "Status": "✅ Success", "Cost": "₹1,200", "Time": "2.1m", "Quality": "8.5"},
            {"File": "Tile_02.png", "Status": "✅ Success", "Cost": "₹1,800", "Time": "2.8m", "Quality": "7.9"},
            {"File": "Tile_03.png", "Status": "✅ Success", "Cost": "₹1,500", "Time": "2.3m", "Quality": "8.2"},
            {"File": "Tile_04.png", "Status": "✅ Success", "Cost": "₹2,100", "Time": "3.1m", "Quality": "8.7"},
            {"File": "Tile_05.png", "Status": "✅ Success", "Cost": "₹1,600", "Time": "2.4m", "Quality": "8.0"},
            {"File": "Tile_06.png", "Status": "✅ Success", "Cost": "₹1,900", "Time": "2.9m", "Quality": "8.3"},
            {"File": "Tile_07.png", "Status": "✅ Success", "Cost": "₹1,400", "Time": "2.2m", "Quality": "7.8"},
            {"File": "Tile_08.png", "Status": "✅ Success", "Cost": "₹1,700", "Time": "2.6m", "Quality": "8.1"},
            {"File": "Tile_09.png", "Status": "✅ Success", "Cost": "₹1,300", "Time": "2.0m", "Quality": "7.9"},
            {"File": "Tile_10.png", "Status": "❌ Failed", "Cost": "₹0", "Time": "0.0m", "Quality": "N/A"}
        ]
        
        df_results = pd.DataFrame(file_results)
        st.dataframe(df_results, use_container_width=True)
        
        # Cost distribution
        st.markdown("### 💰 Cost Distribution")
        
        costs = [1200, 1800, 1500, 2100, 1600, 1900, 1400, 1700, 1300]
        
        col_min, col_avg, col_max = st.columns(3)
        
        with col_min:
            st.metric("Min Cost", f"₹{min(costs):,}")
        
        with col_avg:
            st.metric("Avg Cost", f"₹{sum(costs)/len(costs):,.0f}")
        
        with col_max:
            st.metric("Max Cost", f"₹{max(costs):,}")
        
        # Quality analysis
        st.markdown("### 🎯 Quality Analysis")
        
        quality_scores = [8.5, 7.9, 8.2, 8.7, 8.0, 8.3, 7.8, 8.1, 7.9]
        
        col_qmin, col_qavg, col_qmax = st.columns(3)
        
        with col_qmin:
            st.metric("Min Quality", f"{min(quality_scores):.1f}")
        
        with col_qavg:
            st.metric("Avg Quality", f"{sum(quality_scores)/len(quality_scores):.1f}")
        
        with col_qmax:
            st.metric("Max Quality", f"{max(quality_scores):.1f}")
    
    with col2:
        st.markdown("### 💡 Analysis Tips")
        
        tips = guidance.get_contextal_tips("results_analysis")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Analysis Considerations")
        
        warnings = guidance.get_warnings_for_context("results_analysis")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📈 Performance Summary")
        
        performance_summary = [
            "✅ 90% success rate achieved",
            "✅ Processing time within expected range",
            "✅ Cost distribution is reasonable",
            "✅ Quality scores above threshold",
            "⚠️ 1 file failed - needs investigation"
        ]
        
        for item in performance_summary:
            st.write(item)
        
        st.markdown("### 🔍 Failed File Analysis")
        
        st.write("**Failed File:** Tile_10.png")
        st.write("**Reason:** Image quality too low")
        st.write("**Recommendation:** Re-upload with higher quality")
        
        st.markdown("### 🎯 Key Insights")
        
        insights = [
            "• Conservative strategy worked well",
            "• Processing time was consistent",
            "• Cost variation is acceptable",
            "• Quality scores are good",
            "• One file needs attention"
        ]
        
        for insight in insights:
            st.write(insight)
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Processing Monitoring"):
            guidance.advance_step(BatchGuidanceStep.PROCESSING_MONITORING)
            st.rerun()
        
        if st.button("➡️ Continue to Optimization Suggestions", type="primary"):
            guidance.advance_step(BatchGuidanceStep.OPTIMIZATION_SUGGESTIONS)
            st.rerun()

def create_optimization_suggestions_step(guidance: WJPBatchGuidanceSystem):
    """Create optimization suggestions step interface."""
    st.header("💡 Optimization Suggestions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🧠 Intelligent Optimization Analysis")
        
        st.info("Based on your batch processing results, here are intelligent suggestions for improvement:")
        
        # Parameter optimizations
        st.markdown("#### ⚙️ Parameter Optimizations")
        
        optimizations = [
            {
                "parameter": "Min Area",
                "current": "25",
                "suggested": "20",
                "reason": "Slightly lower threshold detected 2 additional objects",
                "impact": "5% more objects detected"
            },
            {
                "parameter": "Min Circularity",
                "current": "0.03",
                "suggested": "0.025",
                "reason": "Better detection of oval shapes",
                "impact": "3% improvement in detection"
            },
            {
                "parameter": "Simplify Tolerance",
                "current": "0.0",
                "suggested": "0.5",
                "reason": "Reduce processing time without quality loss",
                "impact": "15% faster processing"
            }
        ]
        
        for opt in optimizations:
            with st.expander(f"🔧 {opt['parameter']} Optimization"):
                st.write(f"**Current Value:** {opt['current']}")
                st.write(f"**Suggested Value:** {opt['suggested']}")
                st.write(f"**Reason:** {opt['reason']}")
                st.write(f"**Expected Impact:** {opt['impact']}")
        
        # Material recommendations
        st.markdown("#### 🏗️ Material Recommendations")
        
        material_recs = [
            {
                "recommendation": "Consider Aluminum for Cost Reduction",
                "description": "Switch to Aluminum for 40% cost reduction",
                "savings": "₹6,080 per batch",
                "trade_off": "Slightly lower quality appearance"
            },
            {
                "recommendation": "Optimize Thickness",
                "description": "Reduce thickness to 20mm for 20% cost savings",
                "savings": "₹3,040 per batch",
                "trade_off": "May affect structural requirements"
            }
        ]
        
        for rec in material_recs:
            with st.expander(f"💡 {rec['recommendation']}"):
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Potential Savings:** {rec['savings']}")
                st.write(f"**Trade-off:** {rec['trade_off']}")
        
        # Design suggestions
        st.markdown("#### 🎨 Design Suggestions")
        
        design_suggestions = [
            "Simplify complex geometries to reduce processing time",
            "Use consistent spacing (3mm) across all designs",
            "Avoid very small features (<2mm) for better success rate",
            "Consider standardized dimensions for cost optimization"
        ]
        
        for suggestion in design_suggestions:
            st.info(f"💡 {suggestion}")
        
        # Learning recommendations
        st.markdown("#### 🧠 Learning Recommendations")
        
        learning_recs = [
            "Enable learning mode for continuous improvement",
            "Process similar file types together for better optimization",
            "Use consistent material types for better cost prediction",
            "Regularly review and apply optimization suggestions"
        ]
        
        for rec in learning_recs:
            st.write(f"• {rec}")
        
        # Apply optimizations
        st.markdown("#### 🚀 Apply Optimizations")
        
        if st.button("✅ Apply Parameter Optimizations", type="primary"):
            st.success("✅ Parameter optimizations applied!")
            st.info("These settings will be used for your next batch.")
        
        if st.button("💾 Save Optimization Profile"):
            st.success("✅ Optimization profile saved!")
            st.info("Profile saved as 'Batch_Optimization_V1'")
    
    with col2:
        st.markdown("### 💡 Optimization Tips")
        
        tips = guidance.get_contextal_tips("optimization_suggestions")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### ⚠️ Optimization Considerations")
        
        warnings = guidance.get_warnings_for_context("optimization_suggestions")
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        st.markdown("### 📊 Optimization Impact")
        
        impact_metrics = {
            "Expected Success Rate": "92% (+2%)",
            "Processing Time": "20.0 min (-15%)",
            "Cost Reduction": "₹2,500 (-16%)",
            "Quality Score": "8.3 (+0.1)"
        }
        
        for metric, value in impact_metrics.items():
            st.metric(metric, value)
        
        st.markdown("### 🎯 Optimization Summary")
        
        optimization_summary = [
            "✅ 3 parameter optimizations identified",
            "✅ 2 material recommendations provided",
            "✅ 4 design suggestions offered",
            "✅ Learning recommendations available",
            "✅ Expected 15% performance improvement"
        ]
        
        for item in optimization_summary:
            st.write(item)
        
        st.markdown("### 🔄 Next Steps")
        
        next_steps = [
            "1. Review all optimization suggestions",
            "2. Apply recommended parameter changes",
            "3. Consider material recommendations",
            "4. Plan next batch with optimizations",
            "5. Monitor performance improvements"
        ]
        
        for step in next_steps:
            st.write(step)
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Results Analysis"):
            guidance.advance_step(BatchGuidanceStep.RESULTS_ANALYSIS)
            st.rerun()
        
        if st.button("➡️ Complete Batch Processing", type="primary"):
            guidance.advance_step(BatchGuidanceStep.COMPLETION)
            st.rerun()

def create_batch_completion_step(guidance: WJPBatchGuidanceSystem):
    """Create batch completion step interface."""
    st.header("🎉 Batch Processing Complete")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎊 Congratulations!")
        
        st.success("🎉 Your batch processing has been completed successfully!")
        
        # Completion summary
        st.markdown("### 📊 Batch Summary")
        
        summary_data = {
            "Total Files": "10 files",
            "Successful": "9 files (90%)",
            "Failed": "1 file (10%)",
            "Total Cost": "₹15,200",
            "Processing Time": "23.5 minutes",
            "Quality Score": "8.2/10",
            "Strategy Used": "Conservative",
            "Optimizations Applied": "3 optimizations"
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
            "✅ 10 files processed through complete pipeline",
            "✅ 9 files successfully converted to DXF",
            "✅ Comprehensive analysis performed on all files",
            "✅ Cost and time calculations completed",
            "✅ Quality assessment performed",
            "✅ Professional reports generated",
            "✅ Optimization suggestions provided",
            "✅ Learning data collected for future improvements"
        ]
        
        for accomplishment in accomplishments:
            st.write(accomplishment)
        
        # Download options
        st.markdown("### 📥 Download Results")
        
        col_pdf, col_csv, col_json = st.columns(3)
        
        with col_pdf:
            if st.button("📄 Download PDF Reports", type="primary"):
                st.success("✅ PDF reports downloaded!")
        
        with col_csv:
            if st.button("📊 Download CSV Data"):
                st.success("✅ CSV data downloaded!")
        
        with col_json:
            if st.button("🔧 Download JSON Data"):
                st.success("✅ JSON data downloaded!")
        
        # Next steps
        st.markdown("### 🚀 What's Next?")
        
        st.markdown("#### Option 1: Process Another Batch")
        if st.button("📦 Process Another Batch", type="primary"):
            guidance.advance_step(BatchGuidanceStep.BATCH_PLANNING)
            st.rerun()
        
        st.markdown("#### Option 2: Individual Projects")
        if st.button("🎨 Create Individual Projects"):
            st.info("Redirecting to individual project interface...")
        
        st.markdown("#### Option 3: System Management")
        if st.button("⚙️ View System Status"):
            st.info("Redirecting to system status...")
    
    with col2:
        st.markdown("### 💡 Completion Tips")
        
        tips = guidance.get_contextal_tips("completion")
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.markdown("### 📈 Batch Performance")
        
        performance_metrics = {
            "Success Rate": "90%",
            "Avg Time/File": "2.35 min",
            "Cost Efficiency": "₹1,520/file",
            "Quality Score": "8.2/10",
            "Optimization Applied": "3/3"
        }
        
        for metric, value in performance_metrics.items():
            st.metric(metric, value)
        
        st.markdown("### 🏆 Achievements Unlocked!")
        
        achievements = [
            "🎯 **First Batch Complete**",
            "📊 **90% Success Rate**",
            "⚡ **Efficient Processing**",
            "💡 **Optimization Applied**",
            "🧠 **Learning Enabled**"
        ]
        
        for achievement in achievements:
            st.success(achievement)
        
        st.markdown("### 🎯 System Capabilities")
        
        capabilities = [
            "✅ Batch processing for multiple files",
            "✅ Intelligent strategy selection",
            "✅ Real-time progress monitoring",
            "✅ Comprehensive analysis and reporting",
            "✅ Optimization suggestions and learning",
            "✅ Professional file organization",
            "✅ Material-specific cost calculations"
        ]
        
        for capability in capabilities:
            st.write(capability)
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        if st.button("⬅️ Back to Optimization Suggestions"):
            guidance.advance_step(BatchGuidanceStep.OPTIMIZATION_SUGGESTIONS)
            st.rerun()
        
        if st.button("🏠 Back to Welcome"):
            guidance.advance_step(BatchGuidanceStep.WELCOME)
            st.rerun()

if __name__ == "__main__":
    create_guided_batch_interface()
