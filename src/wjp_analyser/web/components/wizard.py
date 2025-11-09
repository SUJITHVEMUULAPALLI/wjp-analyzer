"""
Wizard Components for Step-by-Step Workflows
=============================================

Reusable wizard components for Streamlit pages that provide:
- Left-to-right step navigation
- Progress indicators
- Step validation
- Back/Next navigation
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable
import streamlit as st


class WizardStep:
    """Represents a single step in a wizard workflow."""
    
    def __init__(
        self,
        key: str,
        title: str,
        description: str = "",
        validator: Optional[Callable[[], tuple[bool, str]]] = None,
        content_renderer: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize a wizard step.
        
        Args:
            key: Unique identifier for the step
            title: Display title
            description: Optional description text
            validator: Function that returns (is_valid, error_message)
            content_renderer: Function to render step content
        """
        self.key = key
        self.title = title
        self.description = description
        self.validator = validator
        self.content_renderer = content_renderer
    
    def validate(self) -> tuple[bool, str]:
        """Validate this step."""
        if self.validator:
            return self.validator()
        return True, ""
    
    def render(self):
        """Render this step's content."""
        if self.content_renderer:
            self.content_renderer()


def render_wizard(
    steps: List[WizardStep],
    session_key: str = "wizard_state",
    allow_back: bool = True,
    allow_next: bool = True,
    show_progress: bool = True,
) -> Optional[str]:
    """
    Render a wizard interface with steps, progress, and navigation.
    
    Args:
        steps: List of WizardStep objects
        session_key: Key for storing wizard state in session
        allow_back: Whether to show back button
        allow_next: Whether to show next button
        show_progress: Whether to show progress indicator
        
    Returns:
        Current step key, or None if wizard not initialized
    """
    # Initialize wizard state
    if session_key not in st.session_state:
        st.session_state[session_key] = {
            "current_step": 0,
            "completed_steps": set(),
            "step_data": {},
        }
    
    state = st.session_state[session_key]
    current_idx = state["current_step"]
    
    # Validate current step index
    if current_idx < 0:
        current_idx = 0
    if current_idx >= len(steps):
        current_idx = len(steps) - 1
    state["current_step"] = current_idx
    
    current_step = steps[current_idx]
    
    # Progress indicator
    if show_progress:
        render_progress_indicator(steps, current_idx, state["completed_steps"])
    
    # Step title and description
    st.markdown(f"## {current_step.title}")
    if current_step.description:
        st.caption(current_step.description)
    
    st.divider()
    
    # Render current step content
    current_step.render()
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if allow_back and current_idx > 0:
            if st.button("◀ Back", key=f"{session_key}_back", use_container_width=True):
                state["current_step"] = current_idx - 1
                st.rerun()
    
    with col2:
        if allow_next:
            if current_idx < len(steps) - 1:
                button_text = "Next ▶"
                button_key = f"{session_key}_next"
            else:
                button_text = "✓ Complete"
                button_key = f"{session_key}_complete"
            
            if st.button(button_text, key=button_key, use_container_width=True, type="primary"):
                # Validate current step
                is_valid, error_msg = current_step.validate()
                if is_valid:
                    state["completed_steps"].add(current_idx)
                    if current_idx < len(steps) - 1:
                        state["current_step"] = current_idx + 1
                        st.rerun()
                    else:
                        # Wizard complete
                        return "complete"
                else:
                    st.error(f"Validation failed: {error_msg}")
    
    # Step navigation (jump to any step)
    if len(steps) > 1:
        with st.expander("Jump to Step", expanded=False):
            step_options = [f"{i+1}. {step.title}" for i, step in enumerate(steps)]
            selected = st.selectbox(
                "Select step",
                step_options,
                index=current_idx,
                key=f"{session_key}_jump",
            )
            selected_idx = step_options.index(selected)
            if selected_idx != current_idx:
                state["current_step"] = selected_idx
                st.rerun()
    
    return current_step.key


def render_progress_indicator(
    steps: List[WizardStep],
    current_idx: int,
    completed_steps: set[int],
):
    """
    Render a progress indicator showing all steps.
    
    Args:
        steps: List of wizard steps
        current_idx: Index of current step
        completed_steps: Set of completed step indices
    """
    num_steps = len(steps)
    
    # Create progress bar
    progress = (current_idx + 1) / num_steps if num_steps > 0 else 0.0
    st.progress(progress, text=f"Step {current_idx + 1} of {num_steps}")
    
    # Create step indicators
    step_cols = st.columns(num_steps)
    for i, (col, step) in enumerate(zip(step_cols, steps)):
        with col:
            # Determine status icon
            if i in completed_steps:
                status_icon = "✓"
                status_color = "green"
            elif i == current_idx:
                status_icon = "●"
                status_color = "blue"
            else:
                status_icon = "○"
                status_color = "gray"
            
            st.markdown(
                f'<div style="text-align: center;">'
                f'<span style="color: {status_color}; font-size: 1.5em;">{status_icon}</span><br>'
                f'<span style="font-size: 0.9em; color: {"blue" if i == current_idx else "gray"}">{step.title}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def reset_wizard(session_key: str = "wizard_state"):
    """Reset wizard state."""
    if session_key in st.session_state:
        del st.session_state[session_key]


def get_wizard_data(session_key: str = "wizard_state", step_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get data stored in wizard state.
    
    Args:
        session_key: Wizard session key
        step_key: Optional specific step key
        
    Returns:
        Wizard data dictionary
    """
    if session_key not in st.session_state:
        return {}
    
    state = st.session_state[session_key]
    if step_key:
        return state.get("step_data", {}).get(step_key, {})
    return state.get("step_data", {})


def set_wizard_data(session_key: str = "wizard_state", step_key: str = None, data: Dict[str, Any] = None):
    """
    Store data in wizard state.
    
    Args:
        session_key: Wizard session key
        step_key: Step key to store data for
        data: Data to store
    """
    if session_key not in st.session_state:
        st.session_state[session_key] = {
            "current_step": 0,
            "completed_steps": set(),
            "step_data": {},
        }
    
    if step_key and data:
        st.session_state[session_key]["step_data"][step_key] = data








