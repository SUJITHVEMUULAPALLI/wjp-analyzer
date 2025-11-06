"""
UI Components Package
=====================

Reusable UI components for Streamlit pages.
"""

from .wizard import (
    WizardStep,
    render_wizard,
    render_progress_indicator,
    reset_wizard,
    get_wizard_data,
    set_wizard_data,
)

from .jobs_drawer import (
    render_jobs_drawer,
    render_job_item,
    render_job_result,
    get_job_status_from_api,
    poll_job_statuses,
)

from .error_handler import (
    ErrorAction,
    render_error,
    get_user_friendly_message,
    create_file_not_found_actions,
    create_dxf_error_actions,
    create_import_error_actions,
    create_network_error_actions,
)

from .terminology import (
    standardize_term,
    standardize_label,
    get_label,
    TERMINOLOGY,
    LABELS,
)

__all__ = [
    # Wizard
    "WizardStep",
    "render_wizard",
    "render_progress_indicator",
    "reset_wizard",
    "get_wizard_data",
    "set_wizard_data",
    # Jobs Drawer
    "render_jobs_drawer",
    "render_job_item",
    "render_job_result",
    "get_job_status_from_api",
    "poll_job_statuses",
    # Error Handler
    "ErrorAction",
    "render_error",
    "get_user_friendly_message",
    "create_file_not_found_actions",
    "create_dxf_error_actions",
    "create_import_error_actions",
    "create_network_error_actions",
    # Terminology
    "standardize_term",
    "standardize_label",
    "get_label",
    "TERMINOLOGY",
    "LABELS",
]

