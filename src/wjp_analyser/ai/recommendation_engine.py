"""
AI+Rules Recommendation Engine
==============================

Hybrid rule-based and AI-powered recommendation system that produces
executable operations for fixing DXF issues.

This combines:
- Rules engine for must-fix issues (deterministic)
- AI/LLM for explaining fixes and suggesting strategies
- Executable operations that can be applied automatically
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class OperationType(str, Enum):
    """Types of operations that can be executed."""
    REMOVE_ZERO_AREA = "remove_zero_area"
    SIMPLIFY_EPS = "simplify_eps"
    FILLET_MIN_RADIUS = "fillet_min_radius"
    FILTER_TINY = "filter_tiny"
    CLOSE_OPEN_CONTOUR = "close_open_contour"
    FIX_MIN_SPACING = "fix_min_spacing"
    REMOVE_DUPLICATE = "remove_duplicate"
    ASSIGN_LAYER = "assign_layer"
    GROUP_SIMILAR = "group_similar"


@dataclass
class Operation:
    """An executable operation with parameters and metadata."""
    operation: OperationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    estimated_impact: Dict[str, Any] = field(default_factory=dict)
    auto_apply: bool = False
    affected_count: int = 0
    severity: str = "info"  # info, warning, error, critical


class RecommendationEngine:
    """Hybrid rule+AI recommendation engine."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.rules_enabled = True
        self.ai_enabled = True
    
    def analyze_report(
        self,
        report: Dict[str, Any],
        csv_stats: Optional[Dict[str, Any]] = None,
    ) -> List[Operation]:
        """
        Analyze a DXF analysis report and generate executable operations.
        
        Args:
            report: DXF analysis report
            csv_stats: Optional CSV analysis statistics
            
        Returns:
            List of executable operations
        """
        operations = []
        
        # Extract components and metrics
        components = report.get("components", [])
        metrics = report.get("metrics", {})
        
        if not components:
            return operations
        
        # Rule-based analysis (must-fix issues)
        if self.rules_enabled:
            operations.extend(self._apply_rules(components, metrics))
        
        # AI-powered analysis (suggestions and explanations)
        if self.ai_enabled:
            ai_ops = self._apply_ai_analysis(components, metrics, csv_stats)
            operations.extend(ai_ops)
        
        # Sort by severity and auto_apply
        operations.sort(key=lambda op: (
            0 if op.auto_apply else 1,
            {"critical": 0, "error": 1, "warning": 2, "info": 3}.get(op.severity, 4)
        ))
        
        return operations
    
    def _apply_rules(
        self,
        components: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> List[Operation]:
        """Apply rule-based analysis for must-fix issues."""
        operations = []
        
        # Rule 1: Zero-area objects (critical - must remove)
        zero_area_count = sum(1 for c in components if c.get("area", 0) == 0)
        if zero_area_count > 0:
            operations.append(Operation(
                operation=OperationType.REMOVE_ZERO_AREA,
                parameters={"threshold": 0.0},
                rationale=f"Found {zero_area_count} zero-area objects that will cause processing errors",
                estimated_impact={
                    "objects_removed": zero_area_count,
                    "processing_time_delta": 0,
                },
                auto_apply=True,
                affected_count=zero_area_count,
                severity="critical",
            ))
        
        # Rule 2: Tiny objects (< 1 mm²) - warning
        tiny_objects = [c for c in components if 0 < c.get("area", 0) < 1.0]
        if len(tiny_objects) > 10:  # Only flag if many tiny objects
            operations.append(Operation(
                operation=OperationType.FILTER_TINY,
                parameters={"min_area_mm2": 1.0},
                rationale=f"Found {len(tiny_objects)} very small objects (<1 mm²) that may be noise",
                estimated_impact={
                    "objects_removed": len(tiny_objects),
                    "area_removed_mm2": sum(c.get("area", 0) for c in tiny_objects),
                },
                auto_apply=False,  # Let user decide
                affected_count=len(tiny_objects),
                severity="warning",
            ))
        
        # Rule 3: Open contours (critical - must fix)
        open_contours = metrics.get("open_contours", 0)
        if open_contours > 0:
            operations.append(Operation(
                operation=OperationType.CLOSE_OPEN_CONTOUR,
                parameters={"tolerance_mm": 0.1},
                rationale=f"Found {open_contours} open contours that must be closed for waterjet cutting",
                estimated_impact={
                    "contours_fixed": open_contours,
                    "length_added_mm": open_contours * 0.5,  # Estimate
                },
                auto_apply=True,
                affected_count=open_contours,
                severity="critical",
            ))
        
        # Rule 4: Minimum radius violations (error - should fix)
        min_radius_violations = metrics.get("min_radius_violations", 0)
        if min_radius_violations > 0:
            min_radius = metrics.get("min_radius_mm", 2.0)
            operations.append(Operation(
                operation=OperationType.FILLET_MIN_RADIUS,
                parameters={"min_radius_mm": max(2.0, min_radius)},
                rationale=f"Found {min_radius_violations} corners with radius < {min_radius} mm",
                estimated_impact={
                    "corners_fixed": min_radius_violations,
                    "length_added_mm": min_radius_violations * 0.5,  # Estimate
                },
                auto_apply=False,  # User decision - may change design intent
                affected_count=min_radius_violations,
                severity="error",
            ))
        
        # Rule 5: Minimum spacing violations (error - should fix)
        min_spacing_violations = metrics.get("min_spacing_violations", 0)
        if min_spacing_violations > 0:
            min_spacing = metrics.get("min_spacing_mm", 3.0)
            operations.append(Operation(
                operation=OperationType.FIX_MIN_SPACING,
                parameters={"min_spacing_mm": max(3.0, min_spacing), "kerf_mm": 1.1},
                rationale=f"Found {min_spacing_violations} spacing violations (< {min_spacing} mm)",
                estimated_impact={
                    "violations_fixed": min_spacing_violations,
                    "geometry_modified": True,
                },
                auto_apply=False,  # Complex operation, user should review
                affected_count=min_spacing_violations,
                severity="error",
            ))
        
        return operations
    
    def _apply_ai_analysis(
        self,
        components: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        csv_stats: Optional[Dict[str, Any]] = None,
    ) -> List[Operation]:
        """Apply AI-powered analysis for suggestions and explanations."""
        operations = []
        
        # For now, use heuristic-based suggestions
        # TODO: Integrate with LLM for more sophisticated analysis
        
        total_objects = len(components)
        
        # Suggestion: Simplify if too many segments
        total_segments = sum(len(c.get("points", [])) for c in components)
        avg_segments = total_segments / max(total_objects, 1)
        if avg_segments > 100:  # High complexity
            operations.append(Operation(
                operation=OperationType.SIMPLIFY_EPS,
                parameters={"tolerance_mm": 0.05},
                rationale=f"High complexity detected (avg {avg_segments:.0f} points per object). Simplification may improve processing speed.",
                estimated_impact={
                    "segments_reduced": int(total_segments * 0.1),  # Estimate 10% reduction
                    "processing_time_delta": -0.05,  # Faster processing
                },
                auto_apply=False,
                affected_count=total_objects,
                severity="info",
            ))
        
        # Suggestion: Group similar objects for efficiency
        if total_objects > 50:
            operations.append(Operation(
                operation=OperationType.GROUP_SIMILAR,
                parameters={"similarity_threshold": 0.95},
                rationale=f"Large number of objects ({total_objects}). Grouping similar objects may improve processing efficiency.",
                estimated_impact={
                    "groups_created": max(5, total_objects // 10),  # Estimate
                    "processing_efficiency": 1.1,  # 10% improvement estimate
                },
                auto_apply=False,
                affected_count=total_objects,
                severity="info",
            ))
        
        # Use CSV stats if available for more insights
        if csv_stats:
            stats = csv_stats.get("statistics", {})
            area_dist = stats.get("area_distribution", {})
            
            # Large objects might benefit from layer assignment
            large_objects = area_dist.get("large_ge100", 0)
            if large_objects > 0 and total_objects > 20:
                operations.append(Operation(
                    operation=OperationType.ASSIGN_LAYER,
                    parameters={"layer": "OUTER", "min_area_mm2": 100.0},
                    rationale=f"Found {large_objects} large objects. Assigning to OUTER layer may optimize cutting order.",
                    estimated_impact={
                        "objects_assigned": large_objects,
                        "cutting_efficiency": 1.05,  # 5% improvement estimate
                    },
                    auto_apply=False,
                    affected_count=large_objects,
                    severity="info",
                ))
        
        return operations
    
    def to_dict(self, operations: List[Operation]) -> List[Dict[str, Any]]:
        """Convert operations to dictionary format for JSON/CSV export."""
        return [
            {
                "operation": op.operation.value,
                "parameters": op.parameters,
                "rationale": op.rationale,
                "estimated_impact": op.estimated_impact,
                "auto_apply": op.auto_apply,
                "affected_count": op.affected_count,
                "severity": op.severity,
            }
            for op in operations
        ]
    
    def filter_operations(
        self,
        operations: List[Operation],
        auto_apply_only: bool = False,
        severity_filter: Optional[List[str]] = None,
    ) -> List[Operation]:
        """Filter operations by criteria."""
        filtered = operations
        
        if auto_apply_only:
            filtered = [op for op in filtered if op.auto_apply]
        
        if severity_filter:
            filtered = [op for op in filtered if op.severity in severity_filter]
        
        return filtered


def analyze_and_recommend(
    report: Dict[str, Any],
    csv_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to analyze and get recommendations.
    
    Args:
        report: DXF analysis report
        csv_stats: Optional CSV analysis statistics
        
    Returns:
        Dict with operations, summary, and recommendations
    """
    engine = RecommendationEngine()
    operations = engine.analyze_report(report, csv_stats)
    
    # Generate summary
    auto_apply_count = sum(1 for op in operations if op.auto_apply)
    critical_count = sum(1 for op in operations if op.severity == "critical")
    error_count = sum(1 for op in operations if op.severity == "error")
    warning_count = sum(1 for op in operations if op.severity == "warning")
    
    # Calculate readiness score
    total_issues = critical_count + error_count + warning_count
    readiness_score = max(0, 100 - (critical_count * 30) - (error_count * 10) - (warning_count * 5))
    readiness_score = min(100, readiness_score)
    
    if readiness_score >= 80:
        level = "excellent"
    elif readiness_score >= 60:
        level = "good"
    elif readiness_score >= 40:
        level = "fair"
    else:
        level = "poor"
    
    return {
        "success": True,
        "operations": engine.to_dict(operations),
        "summary": {
            "total_operations": len(operations),
            "auto_apply_count": auto_apply_count,
            "critical_count": critical_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": len(operations) - critical_count - error_count - warning_count,
        },
        "readiness_score": {
            "score": readiness_score,
            "level": level,
        },
        "recommendations": {
            "must_fix": engine.to_dict([op for op in operations if op.severity == "critical"]),
            "should_fix": engine.to_dict([op for op in operations if op.severity == "error"]),
            "consider": engine.to_dict([op for op in operations if op.severity in ["warning", "info"]]),
        },
    }


# Backward compatibility - integrate with existing csv_analysis_service
def enhance_csv_analysis(csv_analysis: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance CSV analysis results with executable operations.
    
    Args:
        csv_analysis: Results from csv_analysis_service
        report: DXF analysis report
        
    Returns:
        Enhanced analysis with operations
    """
    recommendations = analyze_and_recommend(report, csv_analysis)
    
    # Merge with existing CSV analysis
    enhanced = csv_analysis.copy()
    enhanced["operations"] = recommendations["operations"]
    enhanced["readiness_score"] = recommendations["readiness_score"]
    
    return enhanced





