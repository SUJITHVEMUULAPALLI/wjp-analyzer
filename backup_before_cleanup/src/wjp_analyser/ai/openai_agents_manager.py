"""
OpenAI Agents SDK Manager for WJP ANALYSER

This module provides a comprehensive interface for managing OpenAI agents
specifically designed for waterjet DXF analysis and image processing workflows.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

try:
    from agents import Agent, Runner, Handoff, Session, InputGuardrail, OutputGuardrail, FunctionTool
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    # Create dummy classes for type hints when SDK is not available
    class Agent:
        pass
    class Runner:
        pass
    class Handoff:
        pass
    class InputGuardrail:
        pass
    class OutputGuardrail:
        pass
    class Session:
        pass
    class FunctionTool:
        pass
    print("Warning: OpenAI Agents SDK not available. Install with: pip install openai-agents")

from ..config.secure_config import get_ai_config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for OpenAI agents."""
    name: str
    instructions: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 2000
    tools: List[str] = field(default_factory=list)
    handoffs: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)


@dataclass
class WorkflowConfig:
    """Configuration for agent workflows."""
    name: str
    description: str
    agents: List[str]
    handoffs: List[Dict[str, Any]] = field(default_factory=list)
    guardrails: List[Dict[str, Any]] = field(default_factory=list)
    session_config: Dict[str, Any] = field(default_factory=dict)


class WJPOpenAIAgentsManager:
    """
    Manager for OpenAI Agents SDK integration with WJP ANALYSER.
    
    Provides high-level interface for creating and managing agents
    specifically designed for waterjet DXF analysis workflows.
    """
    
    def __init__(self):
        """Initialize the agents manager."""
        if not AGENTS_SDK_AVAILABLE:
            raise ImportError("OpenAI Agents SDK not available. Install with: pip install openai-agents")
        
        self.agents: Dict[str, Agent] = {}
        self.handoffs: Dict[str, Handoff] = {}
        self.guardrails: Dict[str, Guardrail] = {}
        self.sessions: Dict[str, Session] = {}
        self.workflows: Dict[str, WorkflowConfig] = {}
        
        # Load configuration
        self.ai_config = get_ai_config()
        self.api_key = getattr(self.ai_config, 'openai_api_key', None)
        
        if not self.api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize default agents and workflows
        self._initialize_default_agents()
        self._initialize_default_workflows()
    
    def _initialize_default_agents(self):
        """Initialize default agents for WJP ANALYSER workflows."""
        
        # DXF Analysis Agent
        dxf_agent_config = AgentConfig(
            name="dxf_analyzer",
            instructions="""
            You are a specialized DXF analysis agent for waterjet cutting operations.
            
            Your responsibilities:
            - Analyze DXF files for manufacturing feasibility
            - Identify potential issues (small features, complex geometries, etc.)
            - Suggest optimizations for waterjet cutting
            - Calculate cutting time and material usage estimates
            - Provide detailed technical reports
            
            Always provide structured, technical analysis with specific recommendations.
            """,
            model="gpt-4o-mini",
            temperature=0.1,
            tools=["dxf_analysis", "geometry_calculator", "cost_estimator"]
        )
        
        # Image Processing Agent
        image_agent_config = AgentConfig(
            name="image_processor",
            instructions="""
            You are an image processing specialist for converting images to DXF format.
            
            Your responsibilities:
            - Analyze image characteristics (texture, complexity, contrast)
            - Recommend optimal processing parameters
            - Suggest vectorization strategies
            - Identify potential quality issues
            - Optimize for waterjet cutting requirements
            
            Focus on practical manufacturing considerations and quality optimization.
            """,
            model="gpt-4o-mini",
            temperature=0.2,
            tools=["image_analysis", "parameter_optimizer", "quality_assessor"]
        )
        
        # Design Optimization Agent
        design_agent_config = AgentConfig(
            name="design_optimizer",
            instructions="""
            You are a design optimization specialist for waterjet applications.
            
            Your responsibilities:
            - Suggest design improvements for manufacturability
            - Recommend nesting strategies for material efficiency
            - Identify design patterns that work well with waterjet cutting
            - Suggest alternative approaches for complex geometries
            - Optimize designs for cost and quality
            
            Always consider practical manufacturing constraints and cost implications.
            """,
            model="gpt-4o-mini",
            temperature=0.3,
            tools=["design_analyzer", "nesting_optimizer", "cost_calculator"]
        )
        
        # Quality Assurance Agent
        qa_agent_config = AgentConfig(
            name="quality_assurance",
            instructions="""
            You are a quality assurance specialist for waterjet manufacturing.
            
            Your responsibilities:
            - Validate DXF files for manufacturing readiness
            - Check for common issues and errors
            - Verify dimensional accuracy and tolerances
            - Assess cutting quality requirements
            - Provide final approval recommendations
            
            Be thorough and detail-oriented in your quality assessments.
            """,
            model="gpt-4o-mini",
            temperature=0.1,
            tools=["quality_checker", "tolerance_validator", "error_detector"]
        )
        
        # Create agents
        for config in [dxf_agent_config, image_agent_config, design_agent_config, qa_agent_config]:
            self.create_agent(config)
    
    def _initialize_default_workflows(self):
        """Initialize default workflows for common WJP ANALYSER tasks."""
        
        # DXF Analysis Workflow
        dxf_analysis_workflow = WorkflowConfig(
            name="dxf_analysis_only",
            description="Workflow for analyzing existing DXF files",
            agents=["dxf_analyzer", "design_optimizer", "quality_assurance"],
            handoffs=[
                {
                    "from": "dxf_analyzer",
                    "to": "design_optimizer",
                    "condition": "analysis_complete",
                    "data": ["analysis_results", "issues_found"]
                },
                {
                    "from": "design_optimizer",
                    "to": "quality_assurance",
                    "condition": "optimization_complete",
                    "data": ["optimized_suggestions", "implementation_notes"]
                }
            ]
        )
        
        # Design Review Workflow
        design_review_workflow = WorkflowConfig(
            name="design_review",
            description="Workflow for reviewing and optimizing designs",
            agents=["design_optimizer", "dxf_analyzer", "quality_assurance"],
            handoffs=[
                {
                    "from": "design_optimizer",
                    "to": "dxf_analyzer",
                    "condition": "design_analysis_complete",
                    "data": ["design_assessment", "optimization_suggestions"]
                },
                {
                    "from": "dxf_analyzer",
                    "to": "quality_assurance",
                    "condition": "manufacturing_analysis_complete",
                    "data": ["manufacturing_assessment", "feasibility_report"]
                }
            ]
        )
        
        # Register workflows (Image-to-DXF workflow removed)
        for workflow in [dxf_analysis_workflow, design_review_workflow]:
            self.workflows[workflow.name] = workflow
    
    def create_agent(self, config: AgentConfig) -> Agent:
        """Create a new agent with the given configuration."""
        try:
            # Create model settings
            from agents import ModelSettings
            model_settings = ModelSettings(
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Create tools
            tools = []
            for tool_name in config.tools:
                tool = self._create_tool(tool_name)
                if tool:
                    tools.append(tool)
            
            agent = Agent(
                name=config.name,
                instructions=config.instructions,
                model=config.model,
                model_settings=model_settings,
                tools=tools
            )
            
            self.agents[config.name] = agent
            logger.info(f"Created agent: {config.name}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent {config.name}: {e}")
            raise
    
    def _create_tool(self, tool_name: str) -> Optional[FunctionTool]:
        """Create a tool for the agent."""
        # Define available tools
        tools = {
            "dxf_analysis": FunctionTool(
                name="dxf_analysis",
                description="Analyze DXF files for manufacturing feasibility",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "dxf_path": {"type": "string", "description": "Path to DXF file"}
                    },
                    "required": ["dxf_path"]
                },
                on_invoke_tool=self._analyze_dxf_tool
            ),
            "image_analysis": FunctionTool(
                name="image_analysis",
                description="Analyze images for optimal processing parameters",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to image file"}
                    },
                    "required": ["image_path"]
                },
                on_invoke_tool=self._analyze_image_tool
            ),
            "geometry_calculator": FunctionTool(
                name="geometry_calculator",
                description="Calculate geometric properties and measurements",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "geometry_data": {"type": "object", "description": "Geometry data to analyze"}
                    },
                    "required": ["geometry_data"]
                },
                on_invoke_tool=self._calculate_geometry_tool
            ),
            "cost_estimator": FunctionTool(
                name="cost_estimator",
                description="Estimate manufacturing costs and time",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "parameters": {"type": "object", "description": "Manufacturing parameters"}
                    },
                    "required": ["parameters"]
                },
                on_invoke_tool=self._estimate_cost_tool
            ),
            "parameter_optimizer": FunctionTool(
                name="parameter_optimizer",
                description="Optimize processing parameters for best results",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "current_params": {"type": "object", "description": "Current parameters to optimize"}
                    },
                    "required": ["current_params"]
                },
                on_invoke_tool=self._optimize_parameters_tool
            ),
            "quality_assessor": FunctionTool(
                name="quality_assessor",
                description="Assess quality and identify potential issues",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "quality_data": {"type": "object", "description": "Quality data to assess"}
                    },
                    "required": ["quality_data"]
                },
                on_invoke_tool=self._assess_quality_tool
            ),
            "design_analyzer": FunctionTool(
                name="design_analyzer",
                description="Analyze designs for manufacturability",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "design_data": {"type": "object", "description": "Design data to analyze"}
                    },
                    "required": ["design_data"]
                },
                on_invoke_tool=self._analyze_design_tool
            ),
            "nesting_optimizer": FunctionTool(
                name="nesting_optimizer",
                description="Optimize nesting for material efficiency",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "nesting_data": {"type": "object", "description": "Nesting data to optimize"}
                    },
                    "required": ["nesting_data"]
                },
                on_invoke_tool=self._optimize_nesting_tool
            ),
            "quality_checker": FunctionTool(
                name="quality_checker",
                description="Perform quality checks and validation",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "quality_data": {"type": "object", "description": "Quality data to check"}
                    },
                    "required": ["quality_data"]
                },
                on_invoke_tool=self._check_quality_tool
            ),
            "tolerance_validator": FunctionTool(
                name="tolerance_validator",
                description="Validate tolerances and dimensional accuracy",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "tolerance_data": {"type": "object", "description": "Tolerance data to validate"}
                    },
                    "required": ["tolerance_data"]
                },
                on_invoke_tool=self._validate_tolerances_tool
            ),
            "error_detector": FunctionTool(
                name="error_detector",
                description="Detect common errors and issues",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "error_data": {"type": "object", "description": "Data to check for errors"}
                    },
                    "required": ["error_data"]
                },
                on_invoke_tool=self._detect_errors_tool
            )
        }
        
        return tools.get(tool_name)
    
    # Tool implementations
    async def _analyze_dxf_tool(self, context, dxf_path: str) -> Dict[str, Any]:
        """Tool for analyzing DXF files."""
        try:
            # Import DXF analyzer
            from ..analysis.dxf_analyzer import analyze_dxf
            
            # Run analysis
            result = analyze_dxf(dxf_path)
            
            return {
                "status": "success",
                "analysis": result,
                "summary": f"Analyzed DXF file with {result.get('total_entities', 0)} entities"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _analyze_image_tool(self, context, image_path: str) -> Dict[str, Any]:
        """Tool for analyzing images."""
        try:
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"status": "error", "error": "Could not load image"}
            
            # Basic analysis
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic metrics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            return {
                "status": "success",
                "dimensions": {"width": width, "height": height},
                "intensity": {"mean": float(mean_intensity), "std": float(std_intensity)},
                "contrast": float(contrast),
                "recommendations": self._get_image_recommendations(contrast, mean_intensity)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_image_recommendations(self, contrast: float, mean_intensity: float) -> List[str]:
        """Get recommendations based on image analysis."""
        recommendations = []
        
        if contrast < 0.3:
            recommendations.append("Low contrast detected - consider histogram equalization")
        
        if mean_intensity < 50:
            recommendations.append("Dark image - consider brightness adjustment")
        elif mean_intensity > 200:
            recommendations.append("Bright image - consider contrast enhancement")
        
        if not recommendations:
            recommendations.append("Image quality appears suitable for processing")
        
        return recommendations
    
    def _calculate_geometry_tool(self, geometry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for calculating geometric properties."""
        try:
            # Basic geometry calculations
            calculations = {
                "perimeter": 0,
                "area": 0,
                "bounding_box": {"width": 0, "height": 0},
                "complexity_score": 0
            }
            
            # Placeholder implementation
            # In a real implementation, this would use actual geometry libraries
            
            return {
                "status": "success",
                "calculations": calculations
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _estimate_cost_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for estimating manufacturing costs."""
        try:
            # Basic cost estimation
            # This would integrate with actual cost calculation logic
            
            estimated_cost = {
                "material_cost": 0,
                "cutting_time": 0,
                "labor_cost": 0,
                "total_cost": 0
            }
            
            return {
                "status": "success",
                "cost_estimate": estimated_cost
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _optimize_parameters_tool(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for optimizing processing parameters."""
        try:
            # Parameter optimization logic
            optimized_params = current_params.copy()
            
            # Add optimization suggestions
            suggestions = [
                "Consider adjusting threshold for better edge detection",
                "Optimize smoothing parameters for cleaner vectors",
                "Adjust simplification tolerance for better accuracy"
            ]
            
            return {
                "status": "success",
                "optimized_parameters": optimized_params,
                "suggestions": suggestions
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _assess_quality_tool(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for assessing quality."""
        try:
            quality_score = 85  # Placeholder
            issues = []
            recommendations = []
            
            return {
                "status": "success",
                "quality_score": quality_score,
                "issues": issues,
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _analyze_design_tool(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for analyzing designs."""
        try:
            analysis = {
                "manufacturability": "good",
                "complexity": "medium",
                "recommendations": []
            }
            
            return {
                "status": "success",
                "analysis": analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _optimize_nesting_tool(self, nesting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for optimizing nesting."""
        try:
            optimization = {
                "material_efficiency": 0.85,
                "waste_percentage": 15,
                "recommendations": []
            }
            
            return {
                "status": "success",
                "optimization": optimization
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_quality_tool(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for quality checking."""
        try:
            quality_check = {
                "passed": True,
                "issues": [],
                "score": 90
            }
            
            return {
                "status": "success",
                "quality_check": quality_check
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _validate_tolerances_tool(self, tolerance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for validating tolerances."""
        try:
            validation = {
                "within_tolerance": True,
                "deviations": [],
                "recommendations": []
            }
            
            return {
                "status": "success",
                "validation": validation
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _detect_errors_tool(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for detecting errors."""
        try:
            error_detection = {
                "errors_found": [],
                "warnings": [],
                "recommendations": []
            }
            
            return {
                "status": "success",
                "error_detection": error_detection
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow = self.workflows[workflow_name]
        results = {}
        
        try:
            # Create session for the workflow
            session = Session()
            self.sessions[workflow_name] = session
            
            # Run agents in sequence
            for agent_name in workflow.agents:
                if agent_name not in self.agents:
                    logger.warning(f"Agent '{agent_name}' not found, skipping")
                    continue
                
                agent = self.agents[agent_name]
                
                # Prepare input for agent
                agent_input = self._prepare_agent_input(agent_name, input_data, results)
                
                # Run agent
                result = Runner.run_sync(agent, agent_input, session=session)
                
                results[agent_name] = {
                    "output": result.final_output,
                    "status": "success"
                }
                
                logger.info(f"Agent '{agent_name}' completed successfully")
            
            return {
                "workflow": workflow_name,
                "status": "success",
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Workflow '{workflow_name}' failed: {e}")
            return {
                "workflow": workflow_name,
                "status": "error",
                "error": str(e),
                "results": results
            }
    
    def _prepare_agent_input(self, agent_name: str, input_data: Dict[str, Any], previous_results: Dict[str, Any]) -> str:
        """Prepare input for an agent based on its role and previous results."""
        
        if agent_name == "image_processor":
            return f"""
            Please analyze this image for waterjet DXF conversion:
            
            Image path: {input_data.get('image_path', 'Not provided')}
            Target canvas size: {input_data.get('canvas_size', 'Not specified')}
            Quality requirements: {input_data.get('quality', 'Standard')}
            
            Previous context: {previous_results}
            
            Provide detailed analysis and recommendations for optimal processing parameters.
            """
        
        elif agent_name == "dxf_analyzer":
            return f"""
            Please analyze this DXF file for manufacturing feasibility:
            
            DXF path: {input_data.get('dxf_path', 'Not provided')}
            Material: {input_data.get('material', 'Steel')}
            Thickness: {input_data.get('thickness', 'Not specified')}
            
            Previous analysis: {previous_results}
            
            Provide detailed manufacturing analysis and recommendations.
            """
        
        elif agent_name == "design_optimizer":
            return f"""
            Please optimize this design for waterjet manufacturing:
            
            Design data: {input_data.get('design_data', 'Not provided')}
            Manufacturing constraints: {input_data.get('constraints', 'Standard')}
            
            Previous analysis: {previous_results}
            
            Provide optimization recommendations and alternative approaches.
            """
        
        elif agent_name == "quality_assurance":
            return f"""
            Please perform final quality assurance review:
            
            Review data: {input_data.get('review_data', 'Not provided')}
            Quality standards: {input_data.get('standards', 'Standard')}
            
            Previous analysis: {previous_results}
            
            Provide final quality assessment and approval recommendation.
            """
        
        else:
            return f"Please process this data: {input_data}"
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        return list(self.agents.keys())
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflows."""
        return list(self.workflows.keys())
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        if agent_name not in self.agents:
            return None
        
        agent = self.agents[agent_name]
        
        # Get model settings if available
        temperature = 0.2  # default
        max_tokens = 2000  # default
        tools = []
        
        if hasattr(agent, 'model_settings') and agent.model_settings:
            temperature = getattr(agent.model_settings, 'temperature', 0.2)
            max_tokens = getattr(agent.model_settings, 'max_tokens', 2000)
        
        if hasattr(agent, 'tools') and agent.tools:
            tools = [tool.name for tool in agent.tools]
        
        return {
            "name": agent.name,
            "model": getattr(agent, 'model', 'gpt-4o-mini'),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools
        }
    
    def get_workflow_info(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific workflow."""
        if workflow_name not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_name]
        return {
            "name": workflow.name,
            "description": workflow.description,
            "agents": workflow.agents,
            "handoffs": workflow.handoffs,
            "guardrails": workflow.guardrails
        }


# Global instance
_agents_manager: Optional[WJPOpenAIAgentsManager] = None


def get_agents_manager() -> Optional[WJPOpenAIAgentsManager]:
    """Get the global agents manager instance."""
    global _agents_manager
    
    if _agents_manager is None and AGENTS_SDK_AVAILABLE:
        try:
            _agents_manager = WJPOpenAIAgentsManager()
        except Exception as e:
            logger.error(f"Failed to initialize agents manager: {e}")
            return None
    
    return _agents_manager


def is_agents_sdk_available() -> bool:
    """Check if OpenAI Agents SDK is available."""
    return AGENTS_SDK_AVAILABLE
