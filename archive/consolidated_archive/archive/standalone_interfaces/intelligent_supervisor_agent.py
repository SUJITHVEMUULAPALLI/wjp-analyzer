#!/usr/bin/env python3
"""
Intelligent Supervisor Agent for Batch Processing
================================================

This module implements an intelligent supervisor agent that can orchestrate
batch processing, analyze results, and provide insights and suggestions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

class ProcessingStrategy(Enum):
    """Processing strategies for different file types."""
    CONSERVATIVE = "conservative"  # High precision, fewer objects
    BALANCED = "balanced"         # Balanced precision and recall
    AGGRESSIVE = "aggressive"     # High recall, more objects
    CUSTOM = "custom"            # User-defined parameters

class OptimizationGoal(Enum):
    """Optimization goals for batch processing."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCED = "balanced"

@dataclass
class ProcessingInsight:
    """Processing insight with actionable recommendations."""
    insight_type: str
    description: str
    impact: str  # "high", "medium", "low"
    recommendation: str
    confidence: float  # 0-1
    affected_files: List[str]

@dataclass
class ParameterOptimization:
    """Parameter optimization suggestion."""
    parameter: str
    current_value: float
    suggested_value: float
    reason: str
    expected_improvement: str
    confidence: float

@dataclass
class BatchStrategy:
    """Batch processing strategy."""
    strategy_name: str
    detection_params: Dict[str, Any]
    material_recommendation: str
    processing_order: List[str]
    optimization_goals: List[str]
    expected_outcomes: Dict[str, Any]

class IntelligentSupervisorAgent:
    """Intelligent supervisor agent for batch processing orchestration."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.processing_history = []
        self.learning_data = []
        
        # Initialize processing strategies
        self.strategies = self._initialize_strategies()
        
        # Performance thresholds
        self.thresholds = {
            "min_success_rate": 0.8,
            "max_processing_time": 60,  # seconds per file
            "max_cost_per_file": 5000,  # ₹
            "min_quality_score": 0.6
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the supervisor agent."""
        logger = logging.getLogger("IntelligentSupervisorAgent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_strategies(self) -> Dict[str, BatchStrategy]:
        """Initialize processing strategies."""
        return {
            ProcessingStrategy.CONSERVATIVE.value: BatchStrategy(
                strategy_name="Conservative Processing",
                detection_params={
                    "min_area": 50,
                    "min_circularity": 0.05,
                    "min_solidity": 0.1,
                    "simplify_tolerance": 0.5,
                    "merge_distance": 5.0
                },
                material_recommendation="GRANITE",
                processing_order=["large_files", "simple_files", "complex_files"],
                optimization_goals=["MAXIMIZE_QUALITY"],
                expected_outcomes={
                    "success_rate": 0.9,
                    "avg_objects_per_file": 5,
                    "avg_processing_time": 30
                }
            ),
            ProcessingStrategy.BALANCED.value: BatchStrategy(
                strategy_name="Balanced Processing",
                detection_params={
                    "min_area": 25,
                    "min_circularity": 0.03,
                    "min_solidity": 0.05,
                    "simplify_tolerance": 0.0,
                    "merge_distance": 0.0
                },
                material_recommendation="GENERIC",
                processing_order=["simple_files", "complex_files", "large_files"],
                optimization_goals=["BALANCED"],
                expected_outcomes={
                    "success_rate": 0.85,
                    "avg_objects_per_file": 10,
                    "avg_processing_time": 45
                }
            ),
            ProcessingStrategy.AGGRESSIVE.value: BatchStrategy(
                strategy_name="Aggressive Processing",
                detection_params={
                    "min_area": 10,
                    "min_circularity": 0.01,
                    "min_solidity": 0.02,
                    "simplify_tolerance": 0.0,
                    "merge_distance": 0.0
                },
                material_recommendation="ALUMINUM",
                processing_order=["complex_files", "simple_files", "large_files"],
                optimization_goals=["MINIMIZE_COST", "MINIMIZE_TIME"],
                expected_outcomes={
                    "success_rate": 0.75,
                    "avg_objects_per_file": 20,
                    "avg_processing_time": 60
                }
            )
        }
    
    def analyze_batch_requirements(self, files: List[str]) -> Dict[str, Any]:
        """Analyze batch requirements and recommend strategy."""
        self.logger.info(f"Analyzing batch requirements for {len(files)} files")
        
        # Analyze file characteristics
        file_analysis = self._analyze_file_characteristics(files)
        
        # Determine optimal strategy
        recommended_strategy = self._recommend_strategy(file_analysis)
        
        # Generate processing plan
        processing_plan = self._generate_processing_plan(files, recommended_strategy)
        
        return {
            "file_analysis": file_analysis,
            "recommended_strategy": recommended_strategy,
            "processing_plan": processing_plan,
            "estimated_duration": self._estimate_processing_duration(files, recommended_strategy),
            "resource_requirements": self._estimate_resource_requirements(files)
        }
    
    def _analyze_file_characteristics(self, files: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of files in the batch."""
        analysis = {
            "total_files": len(files),
            "file_types": {},
            "file_sizes": [],
            "complexity_indicators": [],
            "estimated_processing_difficulty": "medium"
        }
        
        for file_path in files:
            # Analyze file type
            file_ext = Path(file_path).suffix.lower()
            analysis["file_types"][file_ext] = analysis["file_types"].get(file_ext, 0) + 1
            
            # Analyze file size
            try:
                file_size = os.path.getsize(file_path)
                analysis["file_sizes"].append(file_size)
            except:
                analysis["file_sizes"].append(0)
            
            # Analyze complexity indicators
            complexity = self._estimate_file_complexity(file_path)
            analysis["complexity_indicators"].append(complexity)
        
        # Calculate statistics
        if analysis["file_sizes"]:
            analysis["avg_file_size"] = np.mean(analysis["file_sizes"])
            analysis["max_file_size"] = max(analysis["file_sizes"])
            analysis["min_file_size"] = min(analysis["file_sizes"])
        
        if analysis["complexity_indicators"]:
            analysis["avg_complexity"] = np.mean(analysis["complexity_indicators"])
            analysis["max_complexity"] = max(analysis["complexity_indicators"])
        
        # Determine processing difficulty
        if analysis["avg_complexity"] > 0.7:
            analysis["estimated_processing_difficulty"] = "high"
        elif analysis["avg_complexity"] < 0.3:
            analysis["estimated_processing_difficulty"] = "low"
        
        return analysis
    
    def _estimate_file_complexity(self, file_path: str) -> float:
        """Estimate file complexity based on various factors."""
        complexity = 0.5  # Default complexity
        
        try:
            # File size factor
            file_size = os.path.getsize(file_path)
            if file_size > 5 * 1024 * 1024:  # > 5MB
                complexity += 0.2
            elif file_size < 100 * 1024:  # < 100KB
                complexity -= 0.1
            
            # File type factor
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.dxf']:
                complexity += 0.1  # DXF files are typically more complex
            elif file_ext in ['.png', '.jpg']:
                complexity += 0.05  # Images may have noise
            
            # File name patterns
            file_name = Path(file_path).stem.lower()
            if any(keyword in file_name for keyword in ['complex', 'detailed', 'intricate']):
                complexity += 0.2
            elif any(keyword in file_name for keyword in ['simple', 'basic', 'clean']):
                complexity -= 0.1
            
        except Exception as e:
            self.logger.warning(f"Error analyzing file complexity for {file_path}: {e}")
        
        return max(0.0, min(1.0, complexity))  # Clamp between 0 and 1
    
    def _recommend_strategy(self, file_analysis: Dict[str, Any]) -> str:
        """Recommend processing strategy based on file analysis."""
        # Decision logic based on file characteristics
        total_files = file_analysis["total_files"]
        avg_complexity = file_analysis.get("avg_complexity", 0.5)
        difficulty = file_analysis.get("estimated_processing_difficulty", "medium")
        
        # Large batch with high complexity -> Conservative
        if total_files > 20 and avg_complexity > 0.6:
            return ProcessingStrategy.CONSERVATIVE.value
        
        # Small batch with low complexity -> Aggressive
        elif total_files < 10 and avg_complexity < 0.4:
            return ProcessingStrategy.AGGRESSIVE.value
        
        # Default to balanced
        else:
            return ProcessingStrategy.BALANCED.value
    
    def _generate_processing_plan(self, files: List[str], strategy: str) -> Dict[str, Any]:
        """Generate detailed processing plan."""
        strategy_config = self.strategies[strategy]
        
        # Sort files by processing order
        sorted_files = self._sort_files_by_processing_order(files, strategy_config.processing_order)
        
        # Group files by characteristics
        file_groups = self._group_files_by_characteristics(sorted_files)
        
        return {
            "strategy": strategy,
            "sorted_files": sorted_files,
            "file_groups": file_groups,
            "processing_stages": self._define_processing_stages(file_groups),
            "quality_checkpoints": self._define_quality_checkpoints(),
            "fallback_strategy": self._get_fallback_strategy(strategy)
        }
    
    def _sort_files_by_processing_order(self, files: List[str], processing_order: List[str]) -> List[str]:
        """Sort files by processing order priority."""
        # Simple implementation - can be enhanced with more sophisticated sorting
        return files  # For now, return as-is
    
    def _group_files_by_characteristics(self, files: List[str]) -> Dict[str, List[str]]:
        """Group files by characteristics for batch processing."""
        groups = {
            "large_files": [],
            "simple_files": [],
            "complex_files": []
        }
        
        for file_path in files:
            complexity = self._estimate_file_complexity(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            if file_size > 2 * 1024 * 1024:  # > 2MB
                groups["large_files"].append(file_path)
            elif complexity > 0.6:
                groups["complex_files"].append(file_path)
            else:
                groups["simple_files"].append(file_path)
        
        return groups
    
    def _define_processing_stages(self, file_groups: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Define processing stages for the batch."""
        stages = []
        
        # Stage 1: Simple files (quick wins)
        if file_groups["simple_files"]:
            stages.append({
                "stage": 1,
                "name": "Simple Files Processing",
                "files": file_groups["simple_files"],
                "expected_duration": len(file_groups["simple_files"]) * 30,
                "success_threshold": 0.9
            })
        
        # Stage 2: Complex files (challenging)
        if file_groups["complex_files"]:
            stages.append({
                "stage": 2,
                "name": "Complex Files Processing",
                "files": file_groups["complex_files"],
                "expected_duration": len(file_groups["complex_files"]) * 60,
                "success_threshold": 0.7
            })
        
        # Stage 3: Large files (resource intensive)
        if file_groups["large_files"]:
            stages.append({
                "stage": 3,
                "name": "Large Files Processing",
                "files": file_groups["large_files"],
                "expected_duration": len(file_groups["large_files"]) * 90,
                "success_threshold": 0.8
            })
        
        return stages
    
    def _define_quality_checkpoints(self) -> List[Dict[str, Any]]:
        """Define quality checkpoints during processing."""
        return [
            {
                "checkpoint": "after_stage_1",
                "metrics": ["success_rate", "avg_processing_time"],
                "thresholds": {"success_rate": 0.8, "avg_processing_time": 60}
            },
            {
                "checkpoint": "after_stage_2",
                "metrics": ["success_rate", "avg_quality_score"],
                "thresholds": {"success_rate": 0.7, "avg_quality_score": 0.6}
            },
            {
                "checkpoint": "final",
                "metrics": ["overall_success_rate", "total_cost", "total_time"],
                "thresholds": {"overall_success_rate": 0.8}
            }
        ]
    
    def _get_fallback_strategy(self, current_strategy: str) -> str:
        """Get fallback strategy if current strategy fails."""
        fallback_map = {
            ProcessingStrategy.AGGRESSIVE.value: ProcessingStrategy.BALANCED.value,
            ProcessingStrategy.BALANCED.value: ProcessingStrategy.CONSERVATIVE.value,
            ProcessingStrategy.CONSERVATIVE.value: ProcessingStrategy.BALANCED.value
        }
        return fallback_map.get(current_strategy, ProcessingStrategy.BALANCED.value)
    
    def _estimate_processing_duration(self, files: List[str], strategy: str) -> Dict[str, Any]:
        """Estimate processing duration for the batch."""
        strategy_config = self.strategies[strategy]
        expected_time_per_file = strategy_config.expected_outcomes["avg_processing_time"]
        
        total_files = len(files)
        estimated_total_time = total_files * expected_time_per_file
        
        return {
            "estimated_total_time_seconds": estimated_total_time,
            "estimated_total_time_minutes": estimated_total_time / 60,
            "estimated_time_per_file": expected_time_per_file,
            "confidence": 0.7  # 70% confidence in estimate
        }
    
    def _estimate_resource_requirements(self, files: List[str]) -> Dict[str, Any]:
        """Estimate resource requirements for processing."""
        total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        
        return {
            "estimated_memory_mb": total_size / (1024 * 1024) * 2,  # 2x file size
            "estimated_disk_space_mb": total_size / (1024 * 1024) * 3,  # 3x file size
            "estimated_cpu_cores": min(4, len(files) // 5 + 1),  # Scale with file count
            "estimated_processing_power": "medium" if len(files) < 20 else "high"
        }
    
    def analyze_batch_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch processing results and generate insights."""
        self.logger.info(f"Analyzing batch results for {len(results)} files")
        
        # Extract metrics
        metrics = self._extract_batch_metrics(results)
        
        # Generate insights
        insights = self._generate_processing_insights(results, metrics)
        
        # Generate parameter optimizations
        optimizations = self._generate_parameter_optimizations(results, metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, metrics, insights)
        
        # Update learning data
        self._update_learning_data(results, metrics)
        
        return {
            "metrics": metrics,
            "insights": insights,
            "optimizations": optimizations,
            "recommendations": recommendations,
            "learning_applied": True
        }
    
    def _extract_batch_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key metrics from batch results."""
        successful_results = [r for r in results if r.get("success", False)]
        
        metrics = {
            "total_files": len(results),
            "successful_files": len(successful_results),
            "failed_files": len(results) - len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0
        }
        
        if successful_results:
            # Processing time metrics
            processing_times = [r.get("processing_time", 0) for r in successful_results]
            metrics["avg_processing_time"] = np.mean(processing_times)
            metrics["max_processing_time"] = max(processing_times)
            metrics["min_processing_time"] = min(processing_times)
            
            # Cost metrics
            costs = [r.get("cost_analysis", {}).get("totals", {}).get("grand_total", 0) for r in successful_results]
            metrics["total_cost"] = sum(costs)
            metrics["avg_cost_per_file"] = np.mean(costs)
            metrics["max_cost"] = max(costs)
            metrics["min_cost"] = min(costs)
            
            # Quality metrics
            quality_scores = [r.get("quality_metrics", {}).get("average_quality", 0) for r in successful_results]
            metrics["avg_quality_score"] = np.mean(quality_scores)
            metrics["min_quality_score"] = min(quality_scores)
            
            # Object detection metrics
            object_counts = [r.get("objects_detected", 0) for r in successful_results]
            metrics["avg_objects_per_file"] = np.mean(object_counts)
            metrics["total_objects"] = sum(object_counts)
        
        return metrics
    
    def _generate_processing_insights(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[ProcessingInsight]:
        """Generate processing insights from results."""
        insights = []
        
        # Success rate insight
        if metrics["success_rate"] < self.thresholds["min_success_rate"]:
            insights.append(ProcessingInsight(
                insight_type="performance",
                description=f"Low success rate: {metrics['success_rate']:.1%}",
                impact="high",
                recommendation="Consider adjusting detection parameters or using a more conservative strategy",
                confidence=0.9,
                affected_files=[r["file_path"] for r in results if not r.get("success", False)]
            ))
        
        # Processing time insight
        if metrics.get("avg_processing_time", 0) > self.thresholds["max_processing_time"]:
            insights.append(ProcessingInsight(
                insight_type="performance",
                description=f"Long processing times: {metrics['avg_processing_time']:.1f}s average",
                impact="medium",
                recommendation="Consider optimizing detection parameters or processing smaller batches",
                confidence=0.8,
                affected_files=[r["file_path"] for r in results if r.get("processing_time", 0) > self.thresholds["max_processing_time"]]
            ))
        
        # Cost insight
        if metrics.get("avg_cost_per_file", 0) > self.thresholds["max_cost_per_file"]:
            insights.append(ProcessingInsight(
                insight_type="cost",
                description=f"High cost per file: ₹{metrics['avg_cost_per_file']:.2f} average",
                impact="high",
                recommendation="Consider material optimization or design simplification",
                confidence=0.85,
                affected_files=[r["file_path"] for r in results if r.get("cost_analysis", {}).get("totals", {}).get("grand_total", 0) > self.thresholds["max_cost_per_file"]]
            ))
        
        # Quality insight
        if metrics.get("avg_quality_score", 0) < self.thresholds["min_quality_score"]:
            insights.append(ProcessingInsight(
                insight_type="quality",
                description=f"Low quality scores: {metrics['avg_quality_score']:.2f} average",
                impact="medium",
                recommendation="Review geometry complexity and consider design improvements",
                confidence=0.75,
                affected_files=[r["file_path"] for r in results if r.get("quality_metrics", {}).get("average_quality", 0) < self.thresholds["min_quality_score"]]
            ))
        
        return insights
    
    def _generate_parameter_optimizations(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[ParameterOptimization]:
        """Generate parameter optimization suggestions."""
        optimizations = []
        
        # Analyze object detection
        if metrics.get("avg_objects_per_file", 0) < 5:
            optimizations.append(ParameterOptimization(
                parameter="min_area",
                current_value=25,
                suggested_value=15,
                reason="Low object detection rate",
                expected_improvement="Increase object detection by 20-30%",
                confidence=0.8
            ))
        
        # Analyze processing time
        if metrics.get("avg_processing_time", 0) > 45:
            optimizations.append(ParameterOptimization(
                parameter="simplify_tolerance",
                current_value=0.0,
                suggested_value=0.5,
                reason="Long processing times",
                expected_improvement="Reduce processing time by 15-25%",
                confidence=0.7
            ))
        
        return optimizations
    
    def _generate_recommendations(self, results: List[Dict[str, Any]], metrics: Dict[str, Any], insights: List[ProcessingInsight]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Strategy recommendations
        if metrics["success_rate"] < 0.8:
            recommendations.append("Switch to Conservative processing strategy for better success rate")
        
        if metrics.get("avg_cost_per_file", 0) > 3000:
            recommendations.append("Consider using Aluminum material for cost reduction")
        
        if metrics.get("avg_processing_time", 0) > 60:
            recommendations.append("Process files in smaller batches for better performance")
        
        # Quality recommendations
        if metrics.get("avg_quality_score", 0) < 0.7:
            recommendations.append("Review and simplify complex geometries")
        
        # Learning recommendations
        if len(self.learning_data) > 10:
            recommendations.append("Sufficient data available for advanced learning optimization")
        
        return recommendations
    
    def _update_learning_data(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Update learning data for future optimization."""
        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "metrics": metrics,
            "strategy_used": "balanced",  # This should be passed from the processing
            "success_rate": metrics["success_rate"],
            "avg_processing_time": metrics.get("avg_processing_time", 0),
            "avg_cost": metrics.get("avg_cost_per_file", 0)
        }
        
        self.learning_data.append(learning_entry)
        
        # Keep only last 100 entries
        if len(self.learning_data) > 100:
            self.learning_data = self.learning_data[-100:]
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning data."""
        if not self.learning_data:
            return {"message": "No learning data available"}
        
        # Calculate trends
        recent_data = self.learning_data[-10:] if len(self.learning_data) >= 10 else self.learning_data
        
        success_rates = [entry["success_rate"] for entry in recent_data]
        processing_times = [entry["avg_processing_time"] for entry in recent_data]
        costs = [entry["avg_cost"] for entry in recent_data]
        
        return {
            "total_batches_processed": len(self.learning_data),
            "recent_success_rate": np.mean(success_rates),
            "recent_avg_processing_time": np.mean(processing_times),
            "recent_avg_cost": np.mean(costs),
            "improvement_trend": "positive" if len(success_rates) > 1 and success_rates[-1] > success_rates[0] else "stable",
            "learning_confidence": min(1.0, len(self.learning_data) / 50)  # Confidence based on data volume
        }

def test_intelligent_supervisor():
    """Test the intelligent supervisor agent."""
    print("🧠 **Testing Intelligent Supervisor Agent**")
    print("=" * 60)
    
    # Create supervisor
    supervisor = IntelligentSupervisorAgent()
    
    # Test batch analysis
    test_files = [
        "C:\\Users\\vemul\\Downloads\\samples\\Tile 21.png",
        "C:\\Users\\vemul\\Downloads\\samples\\Tile 22.png",
        "C:\\Users\\vemul\\Downloads\\samples\\Tile 14.png"
    ]
    
    # Filter existing files
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print(f"📁 Analyzing {len(existing_files)} files...")
        
        # Analyze batch requirements
        analysis = supervisor.analyze_batch_requirements(existing_files)
        
        print(f"✅ Recommended Strategy: {analysis['recommended_strategy']}")
        print(f"⏱️ Estimated Duration: {analysis['estimated_duration']['estimated_total_time_minutes']:.1f} minutes")
        print(f"💾 Resource Requirements: {analysis['resource_requirements']['estimated_processing_power']}")
        
        # Show processing plan
        plan = analysis['processing_plan']
        print(f"\n📋 Processing Plan:")
        print(f"   Strategy: {plan['strategy']}")
        print(f"   Stages: {len(plan['processing_stages'])}")
        
        # Test learning summary
        learning_summary = supervisor.get_learning_summary()
        print(f"\n🧠 Learning Summary:")
        print(f"   Batches Processed: {learning_summary.get('total_batches_processed', 0)}")
        print(f"   Learning Confidence: {learning_summary.get('learning_confidence', 0):.1%}")
        
        print("\n🎉 **Intelligent Supervisor Test Completed Successfully!**")
        
        return analysis
    else:
        print("❌ No test files found")
        return None

if __name__ == "__main__":
    test_intelligent_supervisor()
