#!/usr/bin/env python3
"""
WJP Automation Pipeline - Streamlit Interface
==============================================

This module creates a comprehensive Streamlit interface for the WJP automation pipeline
with intelligent supervisor agent orchestration.
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import time
import threading
import queue

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

# Import WJP agents
try:
    from wjp_supervisor_agent import SupervisorAgent, JobStatus
    from wjp_file_manager import WJPFileManager
    AGENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"WJP Agent system not available: {e}")
    AGENTS_AVAILABLE = False

def create_wjp_automation_interface():
    """Create the WJP automation Streamlit interface."""
    st.set_page_config(
        page_title="WJP Automation Pipeline",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 WJP Automation Pipeline")
    st.markdown("**Complete automation from Prompt → Image → DXF → Analysis → PDF Report**")
    
    if not AGENTS_AVAILABLE:
        st.error("❌ WJP Agent system not available. Please check the installation.")
        return
    
    # Initialize session state
    if "supervisor" not in st.session_state:
        st.session_state.supervisor = SupervisorAgent()
    if "file_manager" not in st.session_state:
        st.session_state.file_manager = WJPFileManager()
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("🎯 Navigation")
        
        page = st.selectbox(
            "Select Page",
            ["Job Submission", "Job Monitoring", "Results & Reports", "System Status", "Batch Processing"]
        )
    
    # Main content based on selected page
    if page == "Job Submission":
        create_job_submission_page()
    elif page == "Job Monitoring":
        create_job_monitoring_page()
    elif page == "Results & Reports":
        create_results_page()
    elif page == "System Status":
        create_system_status_page()
    elif page == "Batch Processing":
        create_batch_processing_page()

def create_job_submission_page():
    """Create job submission page."""
    st.header("📋 Job Submission")
    st.markdown("Submit new jobs for automated processing through the WJP pipeline.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 Job Configuration")
        
        # Job details
        job_id = st.text_input("Job ID", value="SR06", help="Unique identifier for the job")
        
        # Material selection
        material = st.selectbox(
            "Material Type",
            ["Tan Brown Granite", "Marble", "Stainless Steel", "Aluminum", "Brass", "Generic"],
            help="Select material type for cost calculation"
        )
        
        # Thickness
        thickness_mm = st.slider("Thickness (mm)", 1, 100, 25, help="Material thickness in millimeters")
        
        # Category
        category = st.selectbox(
            "Design Category",
            ["Inlay Tile", "Medallion", "Border", "Jali Panel", "Drainage Cover", "Nameplate"],
            help="Select design category"
        )
        
        # Dimensions
        col_width, col_height = st.columns(2)
        with col_width:
            width_inch = st.number_input("Width (inches)", 1, 100, 24)
        with col_height:
            height_inch = st.number_input("Height (inches)", 1, 100, 24)
        
        # Design parameters
        st.subheader("🔧 Design Parameters")
        
        col_spacing, col_radius = st.columns(2)
        with col_spacing:
            cut_spacing_mm = st.number_input("Cut Spacing (mm)", 0.1, 10.0, 3.0, 0.1)
        with col_radius:
            min_radius_mm = st.number_input("Min Radius (mm)", 0.1, 10.0, 2.0, 0.1)
        
        # Prompt
        st.subheader("💭 Design Prompt")
        prompt = st.text_area(
            "Design Prompt",
            value=f"Waterjet-safe {material.lower()} {category.lower()} with clean geometry, {width_inch}x{height_inch} inch",
            height=100,
            help="Describe the design requirements"
        )
    
    with col2:
        st.subheader("📊 Job Preview")
        
        # Preview job details
        preview_data = {
            "Job ID": job_id,
            "Material": material,
            "Thickness": f"{thickness_mm} mm",
            "Category": category,
            "Dimensions": f"{width_inch}×{height_inch} inch",
            "Cut Spacing": f"{cut_spacing_mm} mm",
            "Min Radius": f"{min_radius_mm} mm"
        }
        
        for key, value in preview_data.items():
            st.metric(key, value)
        
        # Submit button
        if st.button("🚀 Submit Job", type="primary"):
            with st.spinner("Submitting job..."):
                try:
                    result = st.session_state.supervisor.submit_job(
                        job_id=job_id,
                        prompt=prompt,
                        material=material,
                        thickness_mm=thickness_mm,
                        category=category,
                        dimensions_inch=[width_inch, height_inch],
                        cut_spacing_mm=cut_spacing_mm,
                        min_radius_mm=min_radius_mm
                    )
                    
                    st.success(f"✅ {result}")
                    st.info("Job submitted successfully! Check the Job Monitoring page to track progress.")
                    
                except Exception as e:
                    st.error(f"❌ Error submitting job: {e}")
        
        # Quick job templates
        st.subheader("📋 Quick Templates")
        
        templates = [
            {
                "name": "Standard Tile",
                "job_id": "ST01",
                "material": "Tan Brown Granite",
                "thickness_mm": 25,
                "category": "Inlay Tile",
                "dimensions_inch": [24, 24],
                "prompt": "Standard granite tile with marble inlay"
            },
            {
                "name": "Medallion",
                "job_id": "MD01",
                "material": "Marble",
                "thickness_mm": 20,
                "category": "Medallion",
                "dimensions_inch": [36, 36],
                "prompt": "Circular medallion design"
            },
            {
                "name": "Jali Panel",
                "job_id": "JL01",
                "material": "Stainless Steel",
                "thickness_mm": 3,
                "category": "Jali Panel",
                "dimensions_inch": [48, 36],
                "prompt": "Geometric jali panel with perforations"
            }
        ]
        
        for template in templates:
            if st.button(f"📋 {template['name']}", key=f"template_{template['job_id']}"):
                st.session_state.template = template
                st.rerun()

def create_job_monitoring_page():
    """Create job monitoring page."""
    st.header("📊 Job Monitoring")
    st.markdown("Monitor the status of submitted jobs and track processing progress.")
    
    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Get queue status
    queue_status = st.session_state.supervisor.get_queue_status()
    
    # Display queue statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Queue Size", queue_status["queue_size"])
    
    with col2:
        st.metric("Active Jobs", queue_status["active_jobs"])
    
    with col3:
        st.metric("Completed", queue_status["completed_jobs"])
    
    with col4:
        st.metric("Failed", queue_status["failed_jobs"])
    
    # Get all jobs
    all_jobs = st.session_state.supervisor.get_all_jobs()
    
    if all_jobs:
        st.subheader("📋 Job Status Details")
        
        # Create jobs DataFrame
        jobs_data = []
        for job_id, job_info in all_jobs.items():
            jobs_data.append({
                "Job ID": job_id,
                "Status": job_info["status"],
                "Start Time": job_info["start_time"],
                "End Time": job_info.get("end_time", "N/A"),
                "Duration (s)": f"{job_info['duration_seconds']:.2f}",
                "Output Files": len(job_info.get("output_files", {})),
                "Errors": len(job_info.get("errors", []))
            })
        
        df_jobs = pd.DataFrame(jobs_data)
        
        # Display jobs table
        st.dataframe(df_jobs, use_container_width=True)
        
        # Job details
        st.subheader("🔍 Job Details")
        
        selected_job = st.selectbox("Select Job", list(all_jobs.keys()))
        
        if selected_job:
            job_info = all_jobs[selected_job]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Information")
                st.write(f"**Job ID:** {job_info['job_id']}")
                st.write(f"**Status:** {job_info['status']}")
                st.write(f"**Start Time:** {job_info['start_time']}")
                if job_info.get('end_time'):
                    st.write(f"**End Time:** {job_info['end_time']}")
                st.write(f"**Duration:** {job_info['duration_seconds']:.2f} seconds")
            
            with col2:
                st.markdown("#### Output Files")
                if job_info.get('output_files'):
                    for file_type, file_path in job_info['output_files'].items():
                        st.write(f"**{file_type}:** {os.path.basename(file_path)}")
                else:
                    st.write("No output files yet")
            
            # Errors
            if job_info.get('errors'):
                st.markdown("#### Errors")
                for error in job_info['errors']:
                    st.error(f"❌ {error}")
    
    else:
        st.info("No jobs found. Submit a job to see it here.")

def create_results_page():
    """Create results and reports page."""
    st.header("📈 Results & Reports")
    st.markdown("View and download completed job results and reports.")
    
    # Get completed jobs
    all_jobs = st.session_state.supervisor.get_all_jobs()
    completed_jobs = {k: v for k, v in all_jobs.items() if v["status"] == "completed"}
    
    if completed_jobs:
        st.subheader("✅ Completed Jobs")
        
        # Job selection
        selected_job = st.selectbox("Select Completed Job", list(completed_jobs.keys()))
        
        if selected_job:
            job_info = completed_jobs[selected_job]
            output_files = job_info.get("output_files", {})
            
            st.markdown(f"#### Job: {selected_job}")
            
            # Display output files
            if output_files:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 📁 Output Files")
                    for file_type, file_path in output_files.items():
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            st.write(f"**{file_type}:** {os.path.basename(file_path)} ({file_size} bytes)")
                        else:
                            st.write(f"**{file_type}:** {os.path.basename(file_path)} (File not found)")
                
                with col2:
                    st.markdown("##### 📊 Job Statistics")
                    st.write(f"**Duration:** {job_info['duration_seconds']:.2f} seconds")
                    st.write(f"**Start Time:** {job_info['start_time']}")
                    st.write(f"**End Time:** {job_info['end_time']}")
                    st.write(f"**Output Files:** {len(output_files)}")
                
                # File downloads
                st.markdown("##### 📥 Download Files")
                
                download_cols = st.columns(min(len(output_files), 3))
                
                for i, (file_type, file_path) in enumerate(output_files.items()):
                    if os.path.exists(file_path):
                        with download_cols[i % 3]:
                            with open(file_path, "rb") as f:
                                file_data = f.read()
                            
                            file_ext = os.path.splitext(file_path)[1]
                            mime_type = {
                                '.png': 'image/png',
                                '.jpg': 'image/jpeg',
                                '.dxf': 'application/dxf',
                                '.json': 'application/json',
                                '.csv': 'text/csv',
                                '.pdf': 'application/pdf'
                            }.get(file_ext, 'application/octet-stream')
                            
                            st.download_button(
                                label=f"📥 {file_type}",
                                data=file_data,
                                file_name=os.path.basename(file_path),
                                mime=mime_type
                            )
                
                # Display images if available
                st.markdown("##### 🖼️ Visualizations")
                
                image_files = {k: v for k, v in output_files.items() 
                              if k.endswith('_image') and os.path.exists(v)}
                
                if image_files:
                    for image_type, image_path in image_files.items():
                        st.markdown(f"**{image_type.replace('_', ' ').title()}:**")
                        st.image(image_path, use_column_width=True)
                
                # Display JSON data if available
                json_files = {k: v for k, v in output_files.items() 
                             if k.endswith('_json') and os.path.exists(v)}
                
                if json_files:
                    st.markdown("##### 📋 Analysis Data")
                    
                    for json_type, json_path in json_files.items():
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        
                        st.markdown(f"**{json_type.replace('_', ' ').title()}:**")
                        st.json(json_data)
    
    else:
        st.info("No completed jobs found. Complete a job to see results here.")

def create_system_status_page():
    """Create system status page."""
    st.header("⚙️ System Status")
    st.markdown("Monitor system performance and processing statistics.")
    
    # Get processing statistics
    stats = st.session_state.supervisor.get_processing_statistics()
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Processing Statistics")
        
        metrics_data = [
            ("Total Jobs", stats["total_jobs"]),
            ("Completed Jobs", stats["completed_jobs"]),
            ("Failed Jobs", stats["failed_jobs"]),
            ("Success Rate", f"{stats['success_rate']:.1%}"),
            ("Total Processing Time", f"{stats['total_processing_time']:.2f}s"),
            ("Average Processing Time", f"{stats['average_processing_time']:.2f}s"),
            ("Current Queue Size", stats["current_queue_size"]),
            ("Active Jobs", stats["active_jobs_count"])
        ]
        
        for metric_name, metric_value in metrics_data:
            st.metric(metric_name, metric_value)
    
    with col2:
        st.subheader("📈 Performance Trends")
        
        # Create a simple performance chart
        if stats["total_jobs"] > 0:
            success_rate = stats["success_rate"]
            
            # Create a simple gauge chart
            st.markdown("#### Success Rate")
            
            # Color based on success rate
            if success_rate >= 0.8:
                color = "green"
            elif success_rate >= 0.6:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{success_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Processing Time Distribution")
            
            # Simple processing time visualization
            if stats["average_processing_time"] > 0:
                avg_time = stats["average_processing_time"]
                
                if avg_time < 60:
                    time_category = "Fast"
                    time_color = "green"
                elif avg_time < 180:
                    time_category = "Moderate"
                    time_color = "orange"
                else:
                    time_category = "Slow"
                    time_color = "red"
                
                st.markdown(f"""
                <div style="background-color: {time_color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: white; margin: 0;">{time_category}</h2>
                    <p style="color: white; margin: 5px 0 0 0;">{avg_time:.1f}s average</p>
                </div>
                """, unsafe_allow_html=True)
    
    # System health
    st.subheader("🏥 System Health")
    
    health_metrics = []
    
    # Check queue health
    if stats["current_queue_size"] < 10:
        health_metrics.append(("Queue Status", "✅ Healthy", "green"))
    else:
        health_metrics.append(("Queue Status", "⚠️ High Load", "orange"))
    
    # Check success rate
    if stats["success_rate"] >= 0.8:
        health_metrics.append(("Success Rate", "✅ Good", "green"))
    elif stats["success_rate"] >= 0.6:
        health_metrics.append(("Success Rate", "⚠️ Moderate", "orange"))
    else:
        health_metrics.append(("Success Rate", "❌ Poor", "red"))
    
    # Check processing time
    if stats["average_processing_time"] < 120:
        health_metrics.append(("Processing Time", "✅ Fast", "green"))
    elif stats["average_processing_time"] < 300:
        health_metrics.append(("Processing Time", "⚠️ Moderate", "orange"))
    else:
        health_metrics.append(("Processing Time", "❌ Slow", "red"))
    
    # Display health metrics
    health_cols = st.columns(len(health_metrics))
    
    for i, (metric_name, status, color) in enumerate(health_metrics):
        with health_cols[i]:
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 5px; text-align: center;">
                <h4 style="color: white; margin: 0;">{metric_name}</h4>
                <p style="color: white; margin: 5px 0 0 0;">{status}</p>
            </div>
            """, unsafe_allow_html=True)

def create_batch_processing_page():
    """Create batch processing page."""
    st.header("📦 Batch Processing")
    st.markdown("Process multiple jobs simultaneously with intelligent orchestration.")
    
    # Batch job configuration
    st.subheader("⚙️ Batch Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Batch job details
        batch_name = st.text_input("Batch Name", value="Batch_001")
        
        # Material selection
        batch_material = st.selectbox(
            "Material Type",
            ["Tan Brown Granite", "Marble", "Stainless Steel", "Aluminum", "Brass", "Generic"]
        )
        
        # Thickness
        batch_thickness = st.slider("Thickness (mm)", 1, 100, 25)
        
        # Category
        batch_category = st.selectbox(
            "Design Category",
            ["Inlay Tile", "Medallion", "Border", "Jali Panel", "Drainage Cover", "Nameplate"]
        )
    
    with col2:
        # Batch parameters
        st.markdown("#### Batch Parameters")
        
        num_jobs = st.number_input("Number of Jobs", 1, 20, 5)
        
        # Job ID prefix
        job_prefix = st.text_input("Job ID Prefix", value="BT")
        
        # Dimensions
        batch_width = st.number_input("Width (inches)", 1, 100, 24)
        batch_height = st.number_input("Height (inches)", 1, 100, 24)
    
    # Generate batch jobs
    if st.button("🔄 Generate Batch Jobs"):
        st.subheader("📋 Generated Batch Jobs")
        
        batch_jobs = []
        
        for i in range(num_jobs):
            job_id = f"{job_prefix}{i+1:03d}"
            prompt = f"Batch {batch_name} - {batch_category.lower()} design {i+1}"
            
            batch_jobs.append({
                "job_id": job_id,
                "prompt": prompt,
                "material": batch_material,
                "thickness_mm": batch_thickness,
                "category": batch_category,
                "dimensions_inch": [batch_width, batch_height],
                "cut_spacing_mm": 3.0,
                "min_radius_mm": 2.0
            })
        
        # Display batch jobs
        batch_df = pd.DataFrame(batch_jobs)
        st.dataframe(batch_df, use_container_width=True)
        
        # Submit batch
        if st.button("🚀 Submit Batch Jobs", type="primary"):
            with st.spinner("Submitting batch jobs..."):
                submitted_count = 0
                
                for job_data in batch_jobs:
                    try:
                        result = st.session_state.supervisor.submit_job(**job_data)
                        submitted_count += 1
                    except Exception as e:
                        st.error(f"Error submitting {job_data['job_id']}: {e}")
                
                st.success(f"✅ {submitted_count}/{len(batch_jobs)} jobs submitted successfully!")
                st.info("Check the Job Monitoring page to track batch progress.")
    
    # Batch processing statistics
    st.subheader("📊 Batch Processing Statistics")
    
    # Get current queue status
    queue_status = st.session_state.supervisor.get_queue_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Queue Size", queue_status["queue_size"])
    
    with col2:
        st.metric("Active Jobs", queue_status["active_jobs"])
    
    with col3:
        st.metric("Completed Jobs", queue_status["completed_jobs"])

if __name__ == "__main__":
    create_wjp_automation_interface()
