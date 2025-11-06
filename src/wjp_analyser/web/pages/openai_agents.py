"""
OpenAI Agents Management Interface for Streamlit

This module provides a Streamlit interface for managing and interacting with
OpenAI agents in the WJP ANALYSER system.
"""

import streamlit as st
import json
import time
import sys
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from wjp_analyser.ai.openai_agents_manager import get_agents_manager, is_agents_sdk_available


def render_agents_interface():
    """Render the OpenAI Agents management interface."""
    
    st.title("🤖 OpenAI Agents Management")
    
    # Check if Agents SDK is available
    if not is_agents_sdk_available():
        st.error("""
        **OpenAI Agents SDK not available!**
        
        Please install it with:
        ```bash
        pip install openai-agents
        ```
        """)
        return
    
    # Get agents manager
    agents_manager = get_agents_manager()
    if not agents_manager:
        st.error("Failed to initialize agents manager. Check your OpenAI API key configuration.")
        return
    
    # Create tabs for different functionalities
    tab_overview, tab_agents, tab_workflows, tab_interactive = st.tabs([
        "📊 Overview", 
        "🤖 Agents", 
        "🔄 Workflows", 
        "💬 Interactive"
    ])
    
    with tab_overview:
        render_overview_tab(agents_manager)
    
    with tab_agents:
        render_agents_tab(agents_manager)
    
    with tab_workflows:
        render_workflows_tab(agents_manager)
    
    with tab_interactive:
        render_interactive_tab(agents_manager)


def render_overview_tab(agents_manager):
    """Render the overview tab."""
    
    st.header("📊 System Overview")
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Available Agents",
            len(agents_manager.get_available_agents()),
            delta=None
        )
    
    with col2:
        st.metric(
            "Available Workflows",
            len(agents_manager.get_available_workflows()),
            delta=None
        )
    
    with col3:
        st.metric(
            "Active Sessions",
            len(agents_manager.sessions),
            delta=None
        )
    
    # Agent capabilities overview
    st.subheader("🎯 Agent Capabilities")
    
    capabilities = {
        "DXF Analysis": "Analyze DXF files for manufacturing feasibility",
        "Image Processing": "Convert images to DXF with optimal parameters",
        "Design Optimization": "Optimize designs for waterjet cutting",
        "Quality Assurance": "Validate and approve final designs"
    }
    
    for capability, description in capabilities.items():
        with st.expander(f"🔧 {capability}"):
            st.write(description)
    
    # Quick start guide
    st.subheader("🚀 Quick Start")
    
    st.info("""
    **Getting Started with OpenAI Agents:**
    
    1. **Choose a Workflow**: Select from available workflows in the Workflows tab
    2. **Upload Your Data**: Provide images or DXF files for analysis
    3. **Run Analysis**: Execute the workflow to get AI-powered insights
    4. **Review Results**: Get detailed recommendations and optimizations
    
    **Available Workflows:**
    - **Image to DXF Complete**: Full pipeline from image to production-ready DXF
    - **DXF Analysis Only**: Analyze existing DXF files
    - **Design Review**: Review and optimize designs
    """)


def render_agents_tab(agents_manager):
    """Render the agents management tab."""
    
    st.header("🤖 Agent Management")
    
    available_agents = agents_manager.get_available_agents()
    
    if not available_agents:
        st.warning("No agents available.")
        return
    
    # Agent selection
    selected_agent = st.selectbox(
        "Select Agent to View Details",
        available_agents,
        key="agent_selector"
    )
    
    if selected_agent:
        agent_info = agents_manager.get_agent_info(selected_agent)
        
        if agent_info:
            # Display agent information
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"🔧 {agent_info['name'].title()} Agent")
                
                # Agent details
                st.write(f"**Model:** {agent_info['model']}")
                st.write(f"**Temperature:** {agent_info['temperature']}")
                st.write(f"**Max Tokens:** {agent_info['max_tokens']}")
                
                if agent_info['tools']:
                    st.write("**Available Tools:**")
                    for tool in agent_info['tools']:
                        st.write(f"- {tool}")
            
            with col2:
                # Agent status
                st.metric("Status", "Active", delta="Online")
                
                # Quick test
                if st.button("🧪 Test Agent", key=f"test_{selected_agent}"):
                    test_agent(agents_manager, selected_agent)


def render_workflows_tab(agents_manager):
    """Render the workflows management tab."""
    
    st.header("🔄 Workflow Management")
    
    available_workflows = agents_manager.get_available_workflows()
    
    if not available_workflows:
        st.warning("No workflows available.")
        return
    
    # Workflow selection
    selected_workflow = st.selectbox(
        "Select Workflow to View Details",
        available_workflows,
        key="workflow_selector"
    )
    
    if selected_workflow:
        workflow_info = agents_manager.get_workflow_info(selected_workflow)
        
        if workflow_info:
            # Display workflow information
            st.subheader(f"🔄 {workflow_info['name'].replace('_', ' ').title()}")
            st.write(f"**Description:** {workflow_info['description']}")
            
            # Workflow steps
            st.write("**Workflow Steps:**")
            for i, agent in enumerate(workflow_info['agents'], 1):
                st.write(f"{i}. **{agent.replace('_', ' ').title()}**")
            
            # Handoffs
            if workflow_info['handoffs']:
                st.write("**Handoffs:**")
                for handoff in workflow_info['handoffs']:
                    st.write(f"- {handoff['from']} → {handoff['to']}")
            
            # Run workflow section
            st.subheader("🚀 Run Workflow")
            
            # Input data collection
            input_data = {}
            
            if selected_workflow == "image_to_dxf_complete":
                input_data = collect_image_to_dxf_input()
            elif selected_workflow == "dxf_analysis_only":
                input_data = collect_dxf_analysis_input()
            elif selected_workflow == "design_review":
                input_data = collect_design_review_input()
            
            # Run button
            if st.button("▶️ Run Workflow", key=f"run_{selected_workflow}"):
                run_workflow(agents_manager, selected_workflow, input_data)


def render_interactive_tab(agents_manager):
    """Render the interactive chat tab."""
    
    st.header("💬 Interactive Agent Chat")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Agent selection for chat
    available_agents = agents_manager.get_available_agents()
    chat_agent = st.selectbox(
        "Select Agent for Chat",
        available_agents,
        key="chat_agent_selector"
    )
    
    # Chat interface
    st.subheader(f"💬 Chat with {chat_agent.replace('_', ' ').title()}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask the agent anything..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                try:
                    agent = agents_manager.agents[chat_agent]
                    result = agents_manager.run_workflow("single_agent", {
                        "agent": chat_agent,
                        "prompt": prompt
                    })
                    
                    response = result.get("results", {}).get(chat_agent, {}).get("output", "No response")
                    
                    st.write(response)
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


def collect_image_to_dxf_input() -> Dict[str, Any]:
    """Collect input data for image-to-DXF workflow."""
    
    st.write("**📸 Image to DXF Workflow Input**")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        key="workflow_image_upload"
    )
    
    input_data = {}
    
    if uploaded_file:
        # Save uploaded file
        upload_path = Path("uploads") / uploaded_file.name
        upload_path.parent.mkdir(exist_ok=True)
        
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        input_data["image_path"] = str(upload_path)
        
        # Display image
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        # Additional parameters
        col1, col2 = st.columns(2)
        
        with col1:
            input_data["canvas_size"] = st.text_input(
                "Canvas Size (mm)",
                value="300x300",
                help="Target canvas size in millimeters"
            )
            
            input_data["quality"] = st.selectbox(
                "Quality Level",
                ["High", "Standard", "Fast"],
                index=1
            )
        
        with col2:
            input_data["material"] = st.selectbox(
                "Target Material",
                ["Steel", "Aluminum", "Stainless Steel", "Other"],
                index=0
            )
            
            input_data["thickness"] = st.number_input(
                "Material Thickness (mm)",
                min_value=0.1,
                max_value=50.0,
                value=3.0,
                step=0.1
            )
    
    return input_data


def collect_dxf_analysis_input() -> Dict[str, Any]:
    """Collect input data for DXF analysis workflow."""
    
    st.write("**📐 DXF Analysis Workflow Input**")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload DXF File",
        type=['dxf'],
        key="workflow_dxf_upload"
    )
    
    input_data = {}
    
    if uploaded_file:
        # Save uploaded file
        upload_path = Path("uploads") / uploaded_file.name
        upload_path.parent.mkdir(exist_ok=True)
        
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        input_data["dxf_path"] = str(upload_path)
        
        # Additional parameters
        col1, col2 = st.columns(2)
        
        with col1:
            input_data["material"] = st.selectbox(
                "Material",
                ["Steel", "Aluminum", "Stainless Steel", "Other"],
                index=0,
                key="dxf_material"
            )
            
            input_data["thickness"] = st.number_input(
                "Thickness (mm)",
                min_value=0.1,
                max_value=50.0,
                value=3.0,
                step=0.1,
                key="dxf_thickness"
            )
        
        with col2:
            input_data["cutting_speed"] = st.number_input(
                "Cutting Speed (mm/min)",
                min_value=10,
                max_value=1000,
                value=200,
                step=10,
                key="dxf_speed"
            )
            
            input_data["quality_requirements"] = st.selectbox(
                "Quality Requirements",
                ["High Precision", "Standard", "Rough Cut"],
                index=1,
                key="dxf_quality"
            )
    
    return input_data


def collect_design_review_input() -> Dict[str, Any]:
    """Collect input data for design review workflow."""
    
    st.write("**🎨 Design Review Workflow Input**")
    
    # Design data input
    design_type = st.radio(
        "Design Input Type",
        ["Upload DXF", "Upload Image", "Text Description"],
        key="design_input_type"
    )
    
    input_data = {}
    
    if design_type == "Upload DXF":
        uploaded_file = st.file_uploader(
            "Upload DXF File",
            type=['dxf'],
            key="design_dxf_upload"
        )
        
        if uploaded_file:
            upload_path = Path("uploads") / uploaded_file.name
            upload_path.parent.mkdir(exist_ok=True)
            
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            input_data["design_data"] = str(upload_path)
            input_data["design_type"] = "dxf"
    
    elif design_type == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            key="design_image_upload"
        )
        
        if uploaded_file:
            upload_path = Path("uploads") / uploaded_file.name
            upload_path.parent.mkdir(exist_ok=True)
            
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            input_data["design_data"] = str(upload_path)
            input_data["design_type"] = "image"
            
            st.image(uploaded_file, caption="Design Image", width=300)
    
    elif design_type == "Text Description":
        input_data["design_description"] = st.text_area(
            "Describe your design",
            placeholder="Describe the design you want to review...",
            height=100
        )
        input_data["design_type"] = "description"
    
    # Additional parameters
    if input_data:
        col1, col2 = st.columns(2)
        
        with col1:
            input_data["constraints"] = st.text_input(
                "Manufacturing Constraints",
                placeholder="e.g., Maximum size 500x500mm, Material: Steel",
                help="Describe any manufacturing constraints"
            )
            
            input_data["priority"] = st.selectbox(
                "Optimization Priority",
                ["Cost", "Quality", "Speed", "Balanced"],
                index=3
            )
        
        with col2:
            input_data["target_application"] = st.text_input(
                "Target Application",
                placeholder="e.g., Decorative panel, Functional part",
                help="Describe the intended use"
            )
            
            input_data["budget"] = st.number_input(
                "Budget Range (USD)",
                min_value=0,
                max_value=10000,
                value=1000,
                step=100
            )
    
    return input_data


def test_agent(agents_manager, agent_name: str):
    """Test a specific agent with a sample input."""
    
    with st.spinner(f"Testing {agent_name} agent..."):
        try:
            # Create a simple test input
            test_input = {
                "test": True,
                "agent": agent_name,
                "prompt": f"Please provide a brief analysis of your capabilities as the {agent_name} agent."
            }
            
            result = agents_manager.run_workflow("single_agent", test_input)
            
            if result["status"] == "success":
                st.success(f"✅ {agent_name} agent test successful!")
                
                # Display result
                agent_result = result.get("results", {}).get(agent_name, {})
                if agent_result.get("output"):
                    st.write("**Agent Response:**")
                    st.write(agent_result["output"])
            else:
                st.error(f"❌ {agent_name} agent test failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")


def run_workflow(agents_manager, workflow_name: str, input_data: Dict[str, Any]):
    """Run a workflow with the given input data."""
    
    if not input_data:
        st.warning("Please provide input data for the workflow.")
        return
    
    with st.spinner(f"Running {workflow_name} workflow..."):
        try:
            # Add progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run workflow
            result = agents_manager.run_workflow(workflow_name, input_data)
            
            progress_bar.progress(100)
            status_text.text("Workflow completed!")
            
            if result["status"] == "success":
                st.success(f"✅ {workflow_name} workflow completed successfully!")
                
                # Display results
                st.subheader("📊 Workflow Results")
                
                for agent_name, agent_result in result["results"].items():
                    with st.expander(f"🤖 {agent_name.replace('_', ' ').title()} Results"):
                        if agent_result.get("output"):
                            st.write(agent_result["output"])
                        else:
                            st.write("No output available")
                
                # Download results
                results_json = json.dumps(result, indent=2)
                st.download_button(
                    label="📥 Download Results",
                    data=results_json,
                    file_name=f"{workflow_name}_results.json",
                    mime="application/json"
                )
                
            else:
                st.error(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Workflow execution failed: {str(e)}")
        
        finally:
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
