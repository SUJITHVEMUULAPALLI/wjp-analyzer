#!/usr/bin/env python3
"""
Unified Agent Manager
====================

This module provides a unified approach to managing all agents in the WJP ANALYSER system.
It consolidates the different agent approaches and provides consistent parameter handling,
logging, and integration with the unified converter system.

Key Features:
- Unified parameter management
- Consistent logging across all agents
- Integration with unified converter
- Proper error handling and fallbacks
- Standardized agent interfaces
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agents."""
    log_level: str = "INFO"
    output_dir: str = "output"
    max_iterations: int = 20
    timeout: int = 300  # 5 minutes
    enable_optimization: bool = True
    enable_learning: bool = True


class UnifiedAgentManager:
    """
    Unified manager for all agents in the WJP ANALYSER system.
    
    This class provides consistent access to:
    - Agent initialization and configuration
    - Parameter management
    - Logging and error handling
    - Integration with unified systems
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the unified agent manager."""
        self.config = config or AgentConfig()
        self._agents: Dict[str, Any] = {}
        self._setup_paths()
        
    def _setup_paths(self):
        """Setup import paths for the agents."""
        # Add src directory to path
        src_dir = project_root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            
        # Add wjp_agents to path
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
    
    def get_unified_converter(self):
        """Get the unified image to DXF converter."""
        try:
            from src.wjp_analyser.image_processing.converters.unified_converter import UnifiedImageToDXFConverter, ConversionParams
            return UnifiedImageToDXFConverter, ConversionParams
        except ImportError as e:
            logger.warning(f"Failed to import unified converter: {e}")
            return None, None
    
    def get_dxf_analyzer(self):
        """Get the DXF analyzer."""
        try:
            from src.wjp_analyser.analysis.dxf_analyzer import analyze_dxf, AnalyzeArgs
            return analyze_dxf, AnalyzeArgs
        except ImportError as e:
            logger.warning(f"Failed to import DXF analyzer: {e}")
            return None, None
    
    def create_image_to_dxf_agent(self):
        """Create a unified image to DXF agent."""
        try:
            from .image_to_dxf_agent import ImageToDXFAgent
            agent = ImageToDXFAgent()
            logger.info("Created unified ImageToDXFAgent")
            return agent
        except ImportError as e:
            logger.warning(f"Failed to create ImageToDXFAgent: {e}")
            return None
    
    def create_analyze_dxf_agent(self):
        """Create a unified DXF analyzer agent."""
        try:
            from .analyze_dxf_agent import AnalyzeDXFAgent
            agent = AnalyzeDXFAgent()
            logger.info("Created unified AnalyzeDXFAgent")
            return agent
        except ImportError as e:
            logger.warning(f"Failed to create AnalyzeDXFAgent: {e}")
            return None
    
    def create_designer_agent(self):
        """Create a unified designer agent."""
        try:
            from .designer_agent import DesignerAgent
            agent = DesignerAgent()
            logger.info("Created unified DesignerAgent")
            return agent
        except ImportError as e:
            logger.warning(f"Failed to create DesignerAgent: {e}")
            return None
    
    def create_learning_agent(self):
        """Create a unified learning agent."""
        try:
            from .learning_agent import LearningAgent
            agent = LearningAgent()
            logger.info("Created unified LearningAgent")
            return agent
        except ImportError as e:
            logger.warning(f"Failed to create LearningAgent: {e}")
            return None
    
    def create_report_agent(self):
        """Create a unified report agent."""
        try:
            from .report_agent import ReportAgent
            agent = ReportAgent()
            logger.info("Created unified ReportAgent")
            return agent
        except ImportError as e:
            logger.warning(f"Failed to create ReportAgent: {e}")
            return None
    
    def create_supervisor_agent(self):
        """Create a unified supervisor agent."""
        try:
            from .supervisor_agent import SupervisorAgent
            agent = SupervisorAgent()
            logger.info("Created unified SupervisorAgent")
            return agent
        except ImportError as e:
            logger.warning(f"Failed to create SupervisorAgent: {e}")
            return None
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get all available agents."""
        if not self._agents:
            self._agents = {
                'image_to_dxf': self.create_image_to_dxf_agent(),
                'analyze_dxf': self.create_analyze_dxf_agent(),
                'designer': self.create_designer_agent(),
                'learning': self.create_learning_agent(),
                'report': self.create_report_agent(),
                'supervisor': self.create_supervisor_agent()
            }
        return self._agents
    
    def create_conversion_params(self, **kwargs) -> Any:
        """Create conversion parameters with defaults."""
        _, ConversionParams = self.get_unified_converter()
        if ConversionParams is None:
            # Fallback to basic dict
            defaults = {
                "binary_threshold": 180,
                "min_area": 500,
                "dxf_size": 1000.0,
                "use_border_removal": True,
                "simplify_tolerance": 1.0,
                "gaussian_blur": 5,
                "morph_operations": True,
                "multi_threshold": True
            }
            defaults.update(kwargs)
            return defaults
        
        return ConversionParams(**kwargs)
    
    def create_analyze_args(self, **kwargs) -> Any:
        """Create analyze arguments with defaults."""
        _, AnalyzeArgs = self.get_dxf_analyzer()
        if AnalyzeArgs is None:
            # Fallback to basic dict
            defaults = {
                "material": "Generic Material",
                "thickness": 10.0,
                "kerf": 1.0,
                "rate_per_m": 800.0,
                "pierce_cost": 5.0,
                "out": "out"
            }
            defaults.update(kwargs)
            return defaults
        
        return AnalyzeArgs(**kwargs)
    
    def log_agent_event(self, agent_name: str, event_type: str, details: Dict[str, Any]):
        """Log agent events consistently."""
        logger.info(f"Agent Event - {agent_name} - {event_type}: {details}")
    
    def handle_agent_error(self, agent_name: str, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle agent errors consistently."""
        error_details = {
            "agent": agent_name,
            "error": str(error),
            "context": context,
            "type": type(error).__name__
        }
        
        logger.error(f"Agent Error - {agent_name} - {context}: {error}")
        return error_details
    
    def run_agent_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete agent pipeline."""
        try:
            agents = self.get_all_agents()
            results = {}
            
            # Run each agent in the pipeline
            for step_name, step_config in pipeline_config.items():
                agent_name = step_config.get('agent')
                if agent_name in agents and agents[agent_name]:
                    agent = agents[agent_name]
                    
                    # Run the agent
                    if hasattr(agent, 'run'):
                        result = agent.run(**step_config.get('params', {}))
                        results[step_name] = result
                        self.log_agent_event(agent_name, 'completed', {'step': step_name})
                    else:
                        logger.warning(f"Agent {agent_name} does not have a run method")
                        results[step_name] = {'error': 'Agent does not have run method'}
                else:
                    logger.warning(f"Agent {agent_name} not available for step {step_name}")
                    results[step_name] = {'error': 'Agent not available'}
            
            return results
            
        except Exception as e:
            return self.handle_agent_error('pipeline', e, 'pipeline execution')
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        agents = self.get_all_agents()
        status = {}
        
        for name, agent in agents.items():
            status[name] = {
                'available': agent is not None,
                'type': type(agent).__name__ if agent else None,
                'has_run_method': hasattr(agent, 'run') if agent else False
            }
        
        return status


# Global agent manager instance
agent_manager = UnifiedAgentManager()


# Convenience functions for backward compatibility
def get_all_agents():
    """Get all available agents."""
    return agent_manager.get_all_agents()


def create_conversion_params(**kwargs):
    """Create conversion parameters with defaults."""
    return agent_manager.create_conversion_params(**kwargs)


def create_analyze_args(**kwargs):
    """Create analyze arguments with defaults."""
    return agent_manager.create_analyze_args(**kwargs)


def run_agent_pipeline(pipeline_config: Dict[str, Any]):
    """Run a complete agent pipeline."""
    return agent_manager.run_agent_pipeline(pipeline_config)


def get_agent_status():
    """Get status of all agents."""
    return agent_manager.get_agent_status()


if __name__ == "__main__":
    # Test the unified agent manager
    print("Testing Unified Agent Manager...")
    
    # Test agent creation
    agents = get_all_agents()
    print(f"Available agents: {list(agents.keys())}")
    
    # Test parameter creation
    conv_params = create_conversion_params(binary_threshold=150)
    print(f"Conversion Params Created: {conv_params is not None}")
    
    # Test agent status
    status = get_agent_status()
    print(f"Agent Status: {status}")
    
    print("Unified Agent Manager test completed!")
