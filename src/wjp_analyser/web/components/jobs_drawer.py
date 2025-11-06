"""
Jobs Drawer Component
=====================

Sidebar drawer component for displaying job status and results.
Provides real-time updates and artifact downloads.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime
import streamlit as st


def render_jobs_drawer(
    jobs: List[Dict[str, Any]],
    session_key: str = "jobs_drawer",
    poll_interval: int = 5,
):
    """
    Render jobs drawer in sidebar.
    
    Args:
        jobs: List of job dictionaries with keys: id, type, status, created_at, result
        session_key: Session key for drawer state
        poll_interval: Polling interval in seconds for status updates
    """
    with st.sidebar.expander("📋 Jobs", expanded=False):
        if not jobs:
            st.caption("No jobs yet")
            return
        
        # Filter buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            show_all = st.button("All", key=f"{session_key}_all", use_container_width=True)
        with col2:
            show_running = st.button("Running", key=f"{session_key}_running", use_container_width=True)
        with col3:
            show_completed = st.button("Done", key=f"{session_key}_done", use_container_width=True)
        
        # Determine filter
        if show_all:
            st.session_state[f"{session_key}_filter"] = "all"
        elif show_running:
            st.session_state[f"{session_key}_filter"] = "running"
        elif show_completed:
            st.session_state[f"{session_key}_filter"] = "completed"
        
        filter_type = st.session_state.get(f"{session_key}_filter", "all")
        
        # Filter jobs
        filtered_jobs = jobs
        if filter_type == "running":
            filtered_jobs = [j for j in jobs if j.get("status") in ["queued", "running"]]
        elif filter_type == "completed":
            filtered_jobs = [j for j in jobs if j.get("status") == "completed"]
        
        # Display jobs
        for job in filtered_jobs[:10]:  # Limit to 10 most recent
            render_job_item(job, session_key)


def render_job_item(job: Dict[str, Any], session_key: str):
    """
    Render a single job item.
    
    Args:
        job: Job dictionary
        session_key: Session key prefix
    """
    job_id = job.get("id", "unknown")
    job_type = job.get("type", "unknown")
    status = job.get("status", "unknown")
    created_at = job.get("created_at")
    result = job.get("result", {})
    
    # Status icon and color
    status_config = {
        "queued": ("⏳", "gray"),
        "running": ("🔄", "blue"),
        "completed": ("✅", "green"),
        "failed": ("❌", "red"),
    }
    icon, color = status_config.get(status, ("○", "gray"))
    
    # Job header
    st.markdown(
        f'<div style="border-left: 3px solid {color}; padding-left: 8px; margin: 5px 0;">'
        f'<strong>{icon} {job_type}</strong><br>'
        f'<small style="color: gray;">{status.upper()}</small>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    # Timestamp
    if created_at:
        if isinstance(created_at, str):
            st.caption(f"Started: {created_at}")
        else:
            st.caption(f"Started: {created_at.strftime('%H:%M:%S')}")
    
    # Progress for running jobs
    if status == "running":
        progress = job.get("progress", 0)
        if isinstance(progress, (int, float)):
            st.progress(progress / 100.0, text=f"{progress}%")
    
    # Error message for failed jobs
    if status == "failed":
        error = job.get("error", "Unknown error")
        st.error(f"Error: {error[:50]}...")
    
    # Result preview for completed jobs
    if status == "completed" and result:
        render_job_result(job_id, result, session_key)
    
    st.divider()


def render_job_result(job_id: str, result: Dict[str, Any], session_key: str):
    """
    Render job result with download buttons.
    
    Args:
        job_id: Job ID
        result: Result dictionary
        session_key: Session key prefix
    """
    # Summary metrics
    if "metrics" in result:
        metrics = result["metrics"]
        col1, col2 = st.columns(2)
        with col1:
            if "cutting_length_mm" in metrics:
                st.metric("Length", f"{metrics['cutting_length_mm']:.0f} mm")
        with col2:
            if "pierce_count" in metrics:
                st.metric("Pierces", metrics["pierce_count"])
    
    # Artifact downloads
    artifacts = result.get("artifacts", {})
    if artifacts:
        st.caption("Downloads:")
        
        if "layered_dxf" in artifacts:
            dxf_path = artifacts["layered_dxf"]
            if isinstance(dxf_path, str) and Path(dxf_path).exists():
                with open(dxf_path, "rb") as f:
                    st.download_button(
                        "📄 DXF",
                        data=f.read(),
                        file_name=Path(dxf_path).name,
                        key=f"{session_key}_dxf_{job_id}",
                        use_container_width=True,
                    )
        
        if "gcode" in artifacts:
            gcode_path = artifacts["gcode"]
            if isinstance(gcode_path, str) and Path(gcode_path).exists():
                with open(gcode_path, "rb") as f:
                    st.download_button(
                        "⚙️ G-Code",
                        data=f.read(),
                        file_name=Path(gcode_path).name,
                        key=f"{session_key}_gcode_{job_id}",
                        use_container_width=True,
                    )
        
        if "csv" in artifacts:
            csv_path = artifacts["csv"]
            if isinstance(csv_path, str) and Path(csv_path).exists():
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "📊 CSV",
                        data=f.read(),
                        file_name=Path(csv_path).name,
                        key=f"{session_key}_csv_{job_id}",
                        use_container_width=True,
                    )


def get_job_status_from_api(job_id: str, api_client=None) -> Optional[Dict[str, Any]]:
    """
    Fetch job status from API.
    
    Args:
        job_id: Job ID
        api_client: Optional API client instance
        
    Returns:
        Job status dictionary or None
    """
    try:
        if api_client is None:
            from wjp_analyser.web.api_client import get_api_client
            api_client = get_api_client()
        
        if api_client:
            return api_client.get_job_status(job_id)
    except Exception:
        pass
    
    return None


def poll_job_statuses(
    job_ids: List[str],
    session_key: str = "job_statuses",
    api_client=None,
) -> List[Dict[str, Any]]:
    """
    Poll multiple job statuses from API.
    
    Args:
        job_ids: List of job IDs
        session_key: Session key for caching
        api_client: Optional API client instance
        
    Returns:
        List of job status dictionaries
    """
    jobs = []
    
    for job_id in job_ids:
        # Check cache first
        cache_key = f"{session_key}_{job_id}"
        cached = st.session_state.get(cache_key)
        
        if cached and cached.get("status") in ["completed", "failed"]:
            # Use cached result for completed/failed jobs
            jobs.append(cached)
        else:
            # Fetch from API
            status = get_job_status_from_api(job_id, api_client)
            if status:
                jobs.append(status)
                # Cache result
                st.session_state[cache_key] = status
    
    return jobs


# Import Path at module level
from pathlib import Path





