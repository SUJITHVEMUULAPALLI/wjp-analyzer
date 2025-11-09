"""
Auto Re-Analyze Module

Triggers re-analysis when edit count exceeds threshold.
"""
from __future__ import annotations

from typing import Optional, Dict
import streamlit as st

from .dxf_utils import SESSION_EDIT_COUNT, SESSION_EDIT_LOG, SESSION_PATH_KEY


def check_reanalyze_threshold(st, threshold: int = 10) -> bool:
    """
    Check if edit count exceeds threshold for auto re-analyze.
    
    Args:
        st: Streamlit session state
        threshold: Number of edits before triggering re-analyze
    
    Returns:
        True if threshold exceeded, False otherwise
    """
    edit_count = st.session_state.get(SESSION_EDIT_COUNT, 0)
    return edit_count >= threshold


def increment_edit_count(st) -> int:
    """
    Increment edit count and return new count.
    
    Args:
        st: Streamlit session state
    
    Returns:
        New edit count
    """
    current = st.session_state.get(SESSION_EDIT_COUNT, 0)
    new_count = current + 1
    st.session_state[SESSION_EDIT_COUNT] = new_count
    return new_count


def reset_edit_count(st) -> None:
    """Reset edit count to zero."""
    st.session_state[SESSION_EDIT_COUNT] = 0


def get_edit_count(st) -> int:
    """Get current edit count."""
    return st.session_state.get(SESSION_EDIT_COUNT, 0)


def should_reanalyze(st, threshold: int = 10, show_warning: bool = True) -> bool:
    """
    Check if re-analysis should be triggered and optionally show warning.
    
    Args:
        st: Streamlit session state
        threshold: Number of edits before triggering re-analyze
        show_warning: Whether to show Streamlit warning if threshold exceeded
    
    Returns:
        True if re-analysis should be triggered
    """
    if check_reanalyze_threshold(st, threshold):
        if show_warning:
            st.warning(
                f"⚠️ {get_edit_count(st)} edits made. Consider re-analyzing the DXF "
                f"to update cutting parameters and recommendations."
            )
        return True
    return False


def create_reanalyze_button(st, threshold: int = 10) -> bool:
    """
    Create a re-analyze button that appears when threshold is exceeded.
    
    Args:
        st: Streamlit session state
        threshold: Number of edits before showing button
    
    Returns:
        True if button was clicked, False otherwise
    """
    if check_reanalyze_threshold(st, threshold):
        edit_count = get_edit_count(st)
        if st.button(f"🔄 Re-Analyze DXF ({edit_count} edits)", use_container_width=True):
            # Reset edit count after re-analyze
            reset_edit_count(st)
            # Store path for re-analysis
            path = st.session_state.get(SESSION_PATH_KEY)
            if path:
                st.session_state["analyzed_dxf_path"] = path
                st.info("DXF marked for re-analysis. Navigate to Analyzer page to run analysis.")
                return True
    return False

