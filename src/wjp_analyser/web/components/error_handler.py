"""
Error Handler Component
=======================

Provides actionable error messages with fix suggestions and auto-fix buttons.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable, List
import streamlit as st
import traceback


class ErrorAction:
    """Represents an action that can be taken to fix an error."""
    
    def __init__(
        self,
        label: str,
        action: Callable[[], None],
        description: str = "",
        button_type: str = "primary",
    ):
        """
        Initialize error action.
        
        Args:
            label: Button label
            action: Callable to execute
            description: Optional description
            button_type: Streamlit button type
        """
        self.label = label
        self.action = action
        self.description = description
        self.button_type = button_type


def render_error(
    error: Exception,
    user_message: str = None,
    actions: List[ErrorAction] = None,
    show_traceback: bool = False,
    expander_title: str = "Technical Details",
):
    """
    Render an error with user-friendly message and actions.
    
    Args:
        error: Exception object
        user_message: User-friendly error message (if None, auto-generated)
        actions: List of ErrorAction objects for fixes
        show_traceback: Whether to show full traceback in expander
        expander_title: Title for traceback expander
    """
    # Determine error type and user message
    if user_message is None:
        user_message = get_user_friendly_message(error)
    
    # Display error
    st.error(user_message)
    
    # Display actions
    if actions:
        st.markdown("**Try these fixes:**")
        for i, action in enumerate(actions):
            col1, col2 = st.columns([3, 1])
            with col1:
                if action.description:
                    st.caption(action.description)
            with col2:
                if st.button(
                    action.label,
                    key=f"error_action_{i}",
                    type=action.button_type,
                    use_container_width=True,
                ):
                    try:
                        action.action()
                        st.success(f"Applied: {action.label}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Action failed: {e}")
    
    # Show traceback in expander
    if show_traceback:
        with st.expander(expander_title, expanded=False):
            st.code(traceback.format_exc())


def get_user_friendly_message(error: Exception) -> str:
    """
    Generate user-friendly error message from exception.
    
    Args:
        error: Exception object
        
    Returns:
        User-friendly message string
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # File not found errors
    if "FileNotFoundError" in error_type or "No such file" in error_msg:
        return f"File not found: {error_msg}. Please check the file path and try again."
    
    # Permission errors
    if "PermissionError" in error_type or "permission denied" in error_msg.lower():
        return f"Permission denied: {error_msg}. Please check file permissions."
    
    # Import errors
    if "ImportError" in error_type or "ModuleNotFoundError" in error_type:
        return f"Missing dependency: {error_msg}. Please install required packages."
    
    # DXF parsing errors
    if "DXF" in error_msg or "ezdxf" in error_msg.lower():
        return f"DXF file error: {error_msg}. The file may be corrupted or in an unsupported format."
    
    # Validation errors
    if "ValidationError" in error_type or "invalid" in error_msg.lower():
        return f"Validation error: {error_msg}. Please check your input and try again."
    
    # Network/API errors
    if "ConnectionError" in error_type or "timeout" in error_msg.lower():
        return f"Connection error: {error_msg}. Please check your network connection and API status."
    
    # Default: return formatted error message
    return f"An error occurred: {error_msg}"


def create_file_not_found_actions(file_path: str) -> List[ErrorAction]:
    """Create actions for file not found errors."""
    actions = []
    
    def check_file():
        from pathlib import Path
        path = Path(file_path)
        if path.exists():
            st.success(f"File exists: {file_path}")
        else:
            st.error(f"File still not found: {file_path}")
            st.info("Try uploading the file again or check the path.")
    
    actions.append(ErrorAction(
        label="Check File Path",
        action=check_file,
        description="Verify the file exists at the specified path",
    ))
    
    return actions


def create_dxf_error_actions(file_path: str) -> List[ErrorAction]:
    """Create actions for DXF parsing errors."""
    actions = []
    
    def try_repair():
        st.info("Attempting to repair DXF file...")
        # This would call a repair function
        st.warning("DXF repair not yet implemented. Please try a different file.")
    
    def validate_format():
        from pathlib import Path
        path = Path(file_path)
        if path.suffix.lower() != ".dxf":
            st.error("File is not a DXF file. Please upload a .dxf file.")
        else:
            st.info("File appears to be a DXF file. The issue may be with the file content.")
    
    actions.append(ErrorAction(
        label="Validate Format",
        action=validate_format,
        description="Check if file is a valid DXF",
    ))
    
    actions.append(ErrorAction(
        label="Try Repair",
        action=try_repair,
        description="Attempt to repair corrupted DXF",
    ))
    
    return actions


def create_import_error_actions(module_name: str) -> List[ErrorAction]:
    """Create actions for import errors."""
    actions = []
    
    def show_install_command():
        st.code(f"pip install {module_name}", language="bash")
        st.info("Run this command in your terminal to install the missing package.")
    
    actions.append(ErrorAction(
        label="Show Install Command",
        action=show_install_command,
        description=f"Display pip install command for {module_name}",
    ))
    
    return actions


def create_network_error_actions() -> List[ErrorAction]:
    """Create actions for network/API errors."""
    actions = []
    
    def check_api_status():
        from wjp_analyser.web.api_client import is_api_available
        if is_api_available():
            st.success("API is available and responding.")
        else:
            st.error("API is not available. Please start the API server or check your connection.")
            st.info("Start API with: wjp api")
    
    def retry_connection():
        st.info("Retrying connection...")
        st.rerun()
    
    actions.append(ErrorAction(
        label="Check API Status",
        action=check_api_status,
        description="Verify API server is running",
    ))
    
    actions.append(ErrorAction(
        label="Retry Connection",
        action=retry_connection,
        description="Attempt to reconnect",
    ))
    
    return actions








