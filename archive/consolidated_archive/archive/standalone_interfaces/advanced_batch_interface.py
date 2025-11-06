#!/usr/bin/env python3
"""
Advanced Batch Processing Interface
==================================

This module creates an advanced Streamlit interface for batch processing
with intelligent supervisor agent orchestration, parameter optimization,
and comprehensive reporting with insights and suggestions.
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import zipfile
import tempfile
import shutil
from dataclasses import dataclass, asdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import our agents and enhancement system
try:
    from wjp_agents.supervisor_agent import SupervisorAgent
    from wjp_agents.image_to_dxf_agent import ImageToDXFAgent
    from wjp_agents.analyze_dxf_agent import AnalyzeDXFAgent
    from wjp_agents.learning_agent import LearningAgent
    from wjp_agents.report_agent import ReportAgent
    from src.wjp_analyser.image_processing.object_detector import DetectionParams
    from practical_enhancement_system import PracticalEnhancementSystem, MaterialType
    AGENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Agent system not available: {e}")
    AGENTS_AVAILABLE = False

@dataclass
class BatchJob:
    """Batch processing job configuration."""
    job_id: str
    files: List[str]
    material_type: str
    detection_params: Dict[str, Any]
    optimization_enabled: bool
    learning_enabled: bool
    created_at: datetime
    status: str = "pending"

@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    file_path: str
    file_type: str
    success: bool
    objects_detected: int
    layer_breakdown: Dict[str, int]
    cost_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time: float
    errors: List[str]
    suggestions: List[str]

@dataclass
class BatchInsights:
    """Insights and suggestions from batch processing."""
    total_files: int
    successful_files: int
    failed_files: int
    success_rate: float
    total_cost: float
    total_cutting_time: float
    average_quality: float
    common_issues: List[str]
    optimization_suggestions: List[str]
    material_recommendations: List[str]
    parameter_optimizations: Dict[str, Any]

class AdvancedBatchProcessor:
    """Advanced batch processor with intelligent orchestration."""
    
    def __init__(self):
        self.supervisor = SupervisorAgent() if AGENTS_AVAILABLE else None
        self.image_agent = ImageToDXFAgent() if AGENTS_AVAILABLE else None
        self.analyze_agent = AnalyzeDXFAgent() if AGENTS_AVAILABLE else None
        self.learning_agent = LearningAgent() if AGENTS_AVAILABLE else None
        self.report_agent = ReportAgent() if AGENTS_AVAILABLE else None
        self.enhancement_system = PracticalEnhancementSystem()
        
        # Create output directories
        self.output_dir = Path("output/batch_processing")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dxf_dir = self.output_dir / "dxf"
        self.reports_dir = self.output_dir / "reports"
        self.analysis_dir = self.output_dir / "analysis"
        
        for dir_path in [self.dxf_dir, self.reports_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def process_batch(self, job: BatchJob) -> Tuple[List[ProcessingResult], BatchInsights]:
        """Process a batch of files with intelligent orchestration."""
        results = []
        
        # Initialize learning agent for optimization
        if job.learning_enabled and self.learning_agent:
            self.learning_agent.start_learning_session()
        
        # Process each file
        for file_path in job.files:
            result = self._process_single_file(file_path, job)
            results.append(result)
            
            # Apply learning if enabled
            if job.learning_enabled and self.learning_agent and result.success:
                self.learning_agent.learn_from_result(result)
        
        # Generate batch insights
        insights = self._generate_batch_insights(results, job)
        
        # Save batch report
        self._save_batch_report(results, insights, job)
        
        return results, insights
    
    def _process_single_file(self, file_path: str, job: BatchJob) -> ProcessingResult:
        """Process a single file with intelligent parameter adjustment."""
        start_time = datetime.now()
        errors = []
        suggestions = []
        
        try:
            # Determine file type
            file_type = self._get_file_type(file_path)
            
            if file_type == "image":
                # Process image to DXF
                dxf_path = self._process_image_to_dxf(file_path, job.detection_params)
                if not dxf_path:
                    errors.append("Failed to convert image to DXF")
                    return ProcessingResult(
                        file_path=file_path, file_type=file_type, success=False,
                        objects_detected=0, layer_breakdown={}, cost_analysis={},
                        quality_metrics={}, processing_time=0, errors=errors, suggestions=[]
                    )
            else:
                # Use existing DXF
                dxf_path = file_path
            
            # Analyze DXF with enhancement system
            enhancement_result = self.enhancement_system.enhance_existing_analysis(
                dxf_path, MaterialType(job.material_type)
            )
            
            if enhancement_result["success"]:
                objects = enhancement_result["objects"]
                cost_data = enhancement_result["cost_data"]
                
                # Calculate layer breakdown
                layer_breakdown = {}
                for obj in objects:
                    layer = obj.layer_type.value
                    layer_breakdown[layer] = layer_breakdown.get(layer, 0) + 1
                
                # Generate suggestions
                suggestions = self._generate_file_suggestions(objects, cost_data)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return ProcessingResult(
                    file_path=file_path,
                    file_type=file_type,
                    success=True,
                    objects_detected=len(objects),
                    layer_breakdown=layer_breakdown,
                    cost_analysis=cost_data,
                    quality_metrics={
                        "average_complexity": np.mean([obj.complexity_score for obj in objects]),
                        "average_quality": np.mean([obj.quality_score for obj in objects]),
                        "total_area": cost_data["totals"]["total_area_mm2"],
                        "total_length": cost_data["totals"]["total_length_mm"]
                    },
                    processing_time=processing_time,
                    errors=errors,
                    suggestions=suggestions
                )
            else:
                errors.append(f"Enhancement failed: {enhancement_result['error']}")
                
        except Exception as e:
            errors.append(f"Processing error: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            file_path=file_path, file_type=file_type, success=False,
            objects_detected=0, layer_breakdown={}, cost_analysis={},
            quality_metrics={}, processing_time=processing_time, errors=errors, suggestions=[]
        )
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension."""
        ext = Path(file_path).suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return "image"
        elif ext == '.dxf':
            return "dxf"
        else:
            return "unknown"
    
    def _process_image_to_dxf(self, image_path: str, detection_params: Dict[str, Any]) -> Optional[str]:
        """Process image to DXF using ImageToDXFAgent."""
        try:
            if not self.image_agent:
                return None
            
            # Create detection parameters
            params = DetectionParams(
                min_area=detection_params.get("min_area", 25),
                min_circularity=detection_params.get("min_circularity", 0.03),
                min_solidity=detection_params.get("min_solidity", 0.05),
                simplify_tolerance=detection_params.get("simplify_tolerance", 0.0),
                merge_distance=detection_params.get("merge_distance", 0.0)
            )
            
            # Convert image to DXF
            dxf_path = self.image_agent.convert_image_to_dxf(image_path, params)
            
            # Fix open polylines
            if dxf_path and os.path.exists(dxf_path):
                import ezdxf
                doc = ezdxf.readfile(dxf_path)
                
                open_polylines = 0
                for entity in doc.modelspace():
                    if entity.dxftype() == 'LWPOLYLINE' and not entity.closed:
                        open_polylines += 1
                        points = list(entity.get_points())
                        if len(points) > 2:
                            points.append(points[0])
                            entity.set_points(points)
                            entity.closed = True
                
                if open_polylines > 0:
                    fixed_dxf = dxf_path.replace('.dxf', '_fixed.dxf')
                    doc.saveas(fixed_dxf)
                    return fixed_dxf
                
                return dxf_path
            
            return None
            
        except Exception as e:
            print(f"Error processing image to DXF: {e}")
            return None
    
    def _generate_file_suggestions(self, objects: List, cost_data: Dict[str, Any]) -> List[str]:
        """Generate suggestions for a single file."""
        suggestions = []
        
        # Analyze objects
        total_objects = len(objects)
        complex_objects = len([obj for obj in objects if obj.layer_type.value == "COMPLEX"])
        decor_objects = len([obj for obj in objects if obj.layer_type.value == "DECOR"])
        
        # Cost analysis
        total_cost = cost_data["totals"]["grand_total"]
        cutting_time = cost_data["totals"]["cutting_time_min"]
        
        # Generate suggestions based on analysis
        if complex_objects > total_objects * 0.7:
            suggestions.append("High complexity detected - consider simplifying geometry")
        
        if decor_objects > total_objects * 0.5:
            suggestions.append("Many small decorative elements - optimize for efficiency")
        
        if total_cost > 10000:
            suggestions.append("High cost detected - consider material optimization")
        
        if cutting_time > 60:
            suggestions.append("Long cutting time - consider breaking into smaller jobs")
        
        # Quality suggestions
        avg_complexity = np.mean([obj.complexity_score for obj in objects])
        if avg_complexity > 0.5:
            suggestions.append("High complexity score - review geometry smoothness")
        
        return suggestions
    
    def _generate_batch_insights(self, results: List[ProcessingResult], job: BatchJob) -> BatchInsights:
        """Generate comprehensive batch insights."""
        total_files = len(results)
        successful_files = len([r for r in results if r.success])
        failed_files = total_files - successful_files
        success_rate = successful_files / total_files if total_files > 0 else 0
        
        # Calculate totals
        total_cost = sum(r.cost_analysis.get("totals", {}).get("grand_total", 0) for r in results if r.success)
        total_cutting_time = sum(r.cost_analysis.get("totals", {}).get("cutting_time_min", 0) for r in results if r.success)
        
        # Calculate average quality
        quality_scores = [r.quality_metrics.get("average_quality", 0) for r in results if r.success]
        average_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Identify common issues
        common_issues = self._identify_common_issues(results)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(results, job)
        
        # Material recommendations
        material_recommendations = self._generate_material_recommendations(results, job)
        
        # Parameter optimizations
        parameter_optimizations = self._generate_parameter_optimizations(results, job)
        
        return BatchInsights(
            total_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            success_rate=success_rate,
            total_cost=total_cost,
            total_cutting_time=total_cutting_time,
            average_quality=average_quality,
            common_issues=common_issues,
            optimization_suggestions=optimization_suggestions,
            material_recommendations=material_recommendations,
            parameter_optimizations=parameter_optimizations
        )
    
    def _identify_common_issues(self, results: List[ProcessingResult]) -> List[str]:
        """Identify common issues across files."""
        issues = []
        
        # Count error types
        error_counts = {}
        for result in results:
            for error in result.errors:
                error_type = error.split(':')[0] if ':' in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Identify common issues
        total_files = len(results)
        for error_type, count in error_counts.items():
            if count > total_files * 0.3:  # More than 30% of files
                issues.append(f"{error_type} affects {count}/{total_files} files")
        
        return issues
    
    def _generate_optimization_suggestions(self, results: List[ProcessingResult], job: BatchJob) -> List[str]:
        """Generate optimization suggestions based on batch results."""
        suggestions = []
        
        # Analyze success rate
        if job.success_rate < 0.8:
            suggestions.append("Low success rate - consider adjusting detection parameters")
        
        # Analyze cost distribution
        costs = [r.cost_analysis.get("totals", {}).get("grand_total", 0) for r in results if r.success]
        if costs:
            cost_variance = np.var(costs)
            if cost_variance > np.mean(costs) * 0.5:
                suggestions.append("High cost variance - standardize design complexity")
        
        # Analyze processing time
        times = [r.processing_time for r in results]
        if times:
            avg_time = np.mean(times)
            if avg_time > 30:
                suggestions.append("Long processing times - optimize batch size")
        
        return suggestions
    
    def _generate_material_recommendations(self, results: List[ProcessingResult], job: BatchJob) -> List[str]:
        """Generate material recommendations."""
        recommendations = []
        
        # Analyze total cost
        total_cost = sum(r.cost_analysis.get("totals", {}).get("grand_total", 0) for r in results if r.success)
        
        if total_cost > 50000:
            recommendations.append("High total cost - consider aluminum for cost reduction")
        elif total_cost < 10000:
            recommendations.append("Low cost project - granite provides good value")
        
        # Analyze cutting time
        total_time = sum(r.cost_analysis.get("totals", {}).get("cutting_time_min", 0) for r in results if r.success)
        
        if total_time > 300:
            recommendations.append("Long cutting time - consider faster cutting materials")
        
        return recommendations
    
    def _generate_parameter_optimizations(self, results: List[ProcessingResult], job: BatchJob) -> Dict[str, Any]:
        """Generate parameter optimization suggestions."""
        optimizations = {}
        
        # Analyze object detection
        successful_results = [r for r in results if r.success]
        if successful_results:
            avg_objects = np.mean([r.objects_detected for r in successful_results])
            
            if avg_objects < 5:
                optimizations["min_area"] = max(10, job.detection_params.get("min_area", 25) - 10)
                optimizations["min_circularity"] = max(0.01, job.detection_params.get("min_circularity", 0.03) - 0.01)
            elif avg_objects > 20:
                optimizations["min_area"] = job.detection_params.get("min_area", 25) + 10
                optimizations["min_circularity"] = min(0.1, job.detection_params.get("min_circularity", 0.03) + 0.01)
        
        return optimizations
    
    def _save_batch_report(self, results: List[ProcessingResult], insights: BatchInsights, job: BatchJob):
        """Save comprehensive batch report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create batch report
        batch_report = {
            "job_id": job.job_id,
            "timestamp": timestamp,
            "job_config": asdict(job),
            "results": [asdict(result) for result in results],
            "insights": asdict(insights)
        }
        
        # Save JSON report
        json_path = self.reports_dir / f"batch_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(batch_report, f, indent=2, default=str)
        
        # Save CSV summary
        csv_data = []
        for result in results:
            csv_data.append({
                "file_path": result.file_path,
                "file_type": result.file_type,
                "success": result.success,
                "objects_detected": result.objects_detected,
                "processing_time": result.processing_time,
                "total_cost": result.cost_analysis.get("totals", {}).get("grand_total", 0),
                "cutting_time": result.cost_analysis.get("totals", {}).get("cutting_time_min", 0),
                "errors": "; ".join(result.errors),
                "suggestions": "; ".join(result.suggestions)
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.reports_dir / f"batch_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

def create_batch_processing_interface():
    """Create the advanced batch processing Streamlit interface."""
    st.set_page_config(
        page_title="Advanced Batch Processing",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Advanced Batch Processing Interface")
    st.markdown("**Professional batch processing with intelligent agent orchestration**")
    
    if not AGENTS_AVAILABLE:
        st.error("❌ Agent system not available. Please check the installation.")
        return
    
    # Initialize session state
    if "batch_job" not in st.session_state:
        st.session_state.batch_job = None
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = None
    if "batch_insights" not in st.session_state:
        st.session_state.batch_insights = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Batch Configuration")
        
        # Material selection
        material_type = st.selectbox(
            "Material Type",
            ["GENERIC", "GRANITE", "MARBLE", "STAINLESS_STEEL", "ALUMINUM", "BRASS"],
            help="Select material type for cost calculation"
        )
        
        # Detection parameters
        st.subheader("🔍 Detection Parameters")
        min_area = st.slider("Min Area", 10, 100, 25, help="Minimum object area for detection")
        min_circularity = st.slider("Min Circularity", 0.01, 0.2, 0.03, 0.01, help="Minimum circularity threshold")
        min_solidity = st.slider("Min Solidity", 0.01, 0.5, 0.05, 0.01, help="Minimum solidity threshold")
        simplify_tolerance = st.slider("Simplify Tolerance", 0.0, 2.0, 0.0, 0.1, help="Contour simplification tolerance")
        merge_distance = st.slider("Merge Distance", 0.0, 20.0, 0.0, 1.0, help="Object merging distance")
        
        # Processing options
        st.subheader("🧠 Processing Options")
        optimization_enabled = st.checkbox("Enable Optimization", True, help="Enable parameter optimization")
        learning_enabled = st.checkbox("Enable Learning", True, help="Enable learning from results")
        
        # Detection parameters dictionary
        detection_params = {
            "min_area": min_area,
            "min_circularity": min_circularity,
            "min_solidity": min_solidity,
            "simplify_tolerance": simplify_tolerance,
            "merge_distance": merge_distance
        }
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 File Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Images or DXF files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dxf'],
            accept_multiple_files=True,
            help="Upload multiple files for batch processing"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            # Show file list
            with st.expander("📋 Uploaded Files"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"{i+1}. {file.name} ({file.size} bytes)")
        
        # Process button
        if uploaded_files and st.button("🚀 Process Batch", type="primary"):
            with st.spinner("Processing batch..."):
                # Create batch job
                job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                file_paths = []
                
                # Save uploaded files
                temp_dir = tempfile.mkdtemp()
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file_path)
                
                # Create batch job
                batch_job = BatchJob(
                    job_id=job_id,
                    files=file_paths,
                    material_type=material_type,
                    detection_params=detection_params,
                    optimization_enabled=optimization_enabled,
                    learning_enabled=learning_enabled,
                    created_at=datetime.now()
                )
                
                # Process batch
                processor = AdvancedBatchProcessor()
                results, insights = processor.process_batch(batch_job)
                
                # Store results
                st.session_state.batch_job = batch_job
                st.session_state.processing_results = results
                st.session_state.batch_insights = insights
                
                st.success("✅ Batch processing completed!")
    
    with col2:
        st.header("📊 Quick Stats")
        
        if st.session_state.batch_insights:
            insights = st.session_state.batch_insights
            
            # Success rate
            st.metric("Success Rate", f"{insights.success_rate:.1%}")
            
            # Total cost
            st.metric("Total Cost", f"₹{insights.total_cost:,.2f}")
            
            # Total time
            st.metric("Cutting Time", f"{insights.total_cutting_time:.1f} min")
            
            # Average quality
            st.metric("Avg Quality", f"{insights.average_quality:.2f}")
    
    # Results section
    if st.session_state.processing_results:
        st.header("📈 Processing Results")
        
        results = st.session_state.processing_results
        insights = st.session_state.batch_insights
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", insights.total_files)
        
        with col2:
            st.metric("Successful", insights.successful_files)
        
        with col3:
            st.metric("Failed", insights.failed_files)
        
        with col4:
            st.metric("Success Rate", f"{insights.success_rate:.1%}")
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        # Create results DataFrame
        results_data = []
        for result in results:
            results_data.append({
                "File": os.path.basename(result.file_path),
                "Type": result.file_type,
                "Status": "✅ Success" if result.success else "❌ Failed",
                "Objects": result.objects_detected,
                "Cost": f"₹{result.cost_analysis.get('totals', {}).get('grand_total', 0):,.2f}",
                "Time": f"{result.processing_time:.1f}s",
                "Errors": len(result.errors),
                "Suggestions": len(result.suggestions)
            })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate pie chart
            success_data = [insights.successful_files, insights.failed_files]
            success_labels = ["Successful", "Failed"]
            
            fig_pie = px.pie(
                values=success_data,
                names=success_labels,
                title="Processing Success Rate",
                color_discrete_map={"Successful": "#00ff00", "Failed": "#ff0000"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Cost distribution
            successful_results = [r for r in results if r.success]
            if successful_results:
                costs = [r.cost_analysis.get("totals", {}).get("grand_total", 0) for r in successful_results]
                files = [os.path.basename(r.file_path) for r in successful_results]
                
                fig_bar = px.bar(
                    x=files,
                    y=costs,
                    title="Cost Distribution by File",
                    labels={"x": "File", "y": "Cost (₹)"}
                )
                fig_bar.update_xaxis(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Insights and suggestions
        st.subheader("🧠 Intelligent Insights & Suggestions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Common Issues")
            if insights.common_issues:
                for issue in insights.common_issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ No common issues detected")
        
        with col2:
            st.markdown("#### 💡 Optimization Suggestions")
            if insights.optimization_suggestions:
                for suggestion in insights.optimization_suggestions:
                    st.info(f"💡 {suggestion}")
            else:
                st.success("✅ No optimization suggestions")
        
        # Material recommendations
        st.markdown("#### 🏗️ Material Recommendations")
        if insights.material_recommendations:
            for rec in insights.material_recommendations:
                st.info(f"🏗️ {rec}")
        else:
            st.success("✅ Current material selection is optimal")
        
        # Parameter optimizations
        if insights.parameter_optimizations:
            st.markdown("#### ⚙️ Parameter Optimizations")
            st.json(insights.parameter_optimizations)
        
        # Download results
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download CSV Report"):
                # Create CSV data
                csv_data = []
                for result in results:
                    csv_data.append({
                        "file_path": result.file_path,
                        "file_type": result.file_type,
                        "success": result.success,
                        "objects_detected": result.objects_detected,
                        "processing_time": result.processing_time,
                        "total_cost": result.cost_analysis.get("totals", {}).get("grand_total", 0),
                        "cutting_time": result.cost_analysis.get("totals", {}).get("cutting_time_min", 0),
                        "errors": "; ".join(result.errors),
                        "suggestions": "; ".join(result.suggestions)
                    })
                
                df_csv = pd.DataFrame(csv_data)
                csv = df_csv.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Download JSON Report"):
                # Create JSON data
                json_data = {
                    "job_id": st.session_state.batch_job.job_id,
                    "timestamp": datetime.now().isoformat(),
                    "results": [asdict(result) for result in results],
                    "insights": asdict(insights)
                }
                
                json_str = json.dumps(json_data, indent=2, default=str)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_str,
                    file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📝 Download Summary"):
                # Create summary
                summary = f"""# Batch Processing Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Files**: {insights.total_files}
- **Successful**: {insights.successful_files}
- **Failed**: {insights.failed_files}
- **Success Rate**: {insights.success_rate:.1%}

## Cost Analysis
- **Total Cost**: ₹{insights.total_cost:,.2f}
- **Total Cutting Time**: {insights.total_cutting_time:.1f} minutes
- **Average Quality**: {insights.average_quality:.2f}

## Common Issues
{chr(10).join(f"- {issue}" for issue in insights.common_issues)}

## Optimization Suggestions
{chr(10).join(f"- {suggestion}" for suggestion in insights.optimization_suggestions)}

## Material Recommendations
{chr(10).join(f"- {rec}" for rec in insights.material_recommendations)}
"""
                
                st.download_button(
                    label="📝 Download Summary",
                    data=summary,
                    file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    create_batch_processing_interface()
