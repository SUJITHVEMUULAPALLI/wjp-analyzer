"""
Service for analyzing DXF component CSV exports and providing AI-powered recommendations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd


def analyze_csv(csv_path: str) -> Dict[str, Any]:
    """
    Analyze a DXF component CSV export and provide insights/recommendations.
    
    Args:
        csv_path: Path to the CSV file with columns: ID, Layer, Group, Area_mm2, Perimeter_mm, Selected
        
    Returns:
        Dictionary containing statistics and recommendations
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "error": f"Failed to read CSV: {e}",
            "success": False
        }
    
    # Basic Statistics
    total_objects = len(df)
    
    # Layer Distribution
    layer_dist = df['Layer'].value_counts().to_dict() if 'Layer' in df.columns else {}
    
    # Selection Status
    selected_count = df['Selected'].sum() if 'Selected' in df.columns else 0
    
    # Area Statistics
    area_col = 'Area_mm2' if 'Area_mm2' in df.columns else 'Area_mm2'
    if area_col not in df.columns:
        return {"error": "CSV missing Area_mm2 column", "success": False}
    
    total_area = float(df[area_col].sum())
    mean_area = float(df[area_col].mean()) if total_objects > 0 else 0.0
    median_area = float(df[area_col].median()) if total_objects > 0 else 0.0
    min_area = float(df[area_col].min())
    max_area = float(df[area_col].max())
    
    # Perimeter Statistics
    perim_col = 'Perimeter_mm' if 'Perimeter_mm' in df.columns else 'Perimeter_mm'
    if perim_col not in df.columns:
        return {"error": "CSV missing Perimeter_mm column", "success": False}
    
    total_perimeter = float(df[perim_col].sum())
    mean_perimeter = float(df[perim_col].mean()) if total_objects > 0 else 0.0
    median_perimeter = float(df[perim_col].median()) if total_objects > 0 else 0.0
    min_perimeter = float(df[perim_col].min())
    max_perimeter = float(df[perim_col].max())
    
    # Issue Detection
    zero_area = int((df[area_col] == 0).sum())
    tiny_area = int((df[area_col] < 1).sum())
    small_area = int(((df[area_col] >= 1) & (df[area_col] < 10)).sum())
    medium_area = int(((df[area_col] >= 10) & (df[area_col] < 100)).sum())
    large_area = int((df[area_col] >= 100).sum())
    
    tiny_perimeter = int((df[perim_col] < 5).sum())
    small_perimeter = int(((df[perim_col] >= 5) & (df[perim_col] < 50)).sum())
    medium_perimeter = int(((df[perim_col] >= 50) & (df[perim_col] < 100)).sum())
    large_perimeter = int((df[perim_col] >= 100).sum())
    
    # Layer-specific statistics
    layer_stats = {}
    if 'Layer' in df.columns:
        for layer in df['Layer'].unique():
            layer_df = df[df['Layer'] == layer]
            layer_stats[layer] = {
                "count": int(len(layer_df)),
                "total_area": float(layer_df[area_col].sum()),
                "total_perimeter": float(layer_df[perim_col].sum()),
                "avg_area": float(layer_df[area_col].mean()),
                "avg_perimeter": float(layer_df[perim_col].mean()),
                "zero_area": int((layer_df[area_col] == 0).sum())
            }
    
    # Top objects
    top_largest = df.nlargest(5, area_col)[['ID', 'Layer', area_col, perim_col, 'Selected']].to_dict('records') if total_objects > 0 else []
    smallest_non_zero = df[df[area_col] > 0].nsmallest(10, area_col)[['ID', 'Layer', area_col, perim_col, 'Selected']].to_dict('records') if len(df[df[area_col] > 0]) > 0 else []
    
    # Generate Recommendations
    recommendations = []
    warnings = []
    info = []
    
    if zero_area > 0:
        warnings.append({
            "type": "zero_area",
            "count": zero_area,
            "message": f"{zero_area} objects have zero area - likely open contours or invalid geometry",
            "action": "Remove zero-area objects using DXF Editor cleanup tools"
        })
    
    if tiny_area > total_objects * 0.5:
        warnings.append({
            "type": "too_many_tiny",
            "count": tiny_area,
            "message": f"{tiny_area} objects have area < 1 mm² - may be too small for waterjet cutting",
            "action": "Filter objects with area < 1 mm² or perimeter < 5 mm"
        })
    
    if tiny_perimeter > total_objects * 0.5:
        warnings.append({
            "type": "tiny_perimeter",
            "count": tiny_perimeter,
            "message": f"{tiny_perimeter} objects have perimeter < 5 mm - very small features",
            "action": "Review if these features are manufacturable on waterjet"
        })
    
    # Check layer distribution
    if len(layer_stats) == 1:
        info.append({
            "type": "single_layer",
            "message": "All objects are on a single layer - consider organizing into OUTER/INNER/HOLE layers",
            "action": "Use DXF Editor to assign appropriate layers"
        })
    elif 'OUTER' in layer_dist and layer_dist.get('OUTER', 0) == 1:
        info.append({
            "type": "single_outer",
            "message": "Only 1 object on OUTER layer - ensure proper layer assignment",
            "action": "Review layer assignments in DXF Editor"
        })
    
    # Selection status
    if selected_count == total_objects:
        info.append({
            "type": "all_selected",
            "message": "All objects are selected - consider selective processing",
            "action": "Deselect objects you don't want to process"
        })
    elif selected_count == 0:
        info.append({
            "type": "none_selected",
            "message": "No objects are selected - select objects to process",
            "action": "Use the object table to select which objects to process"
        })
    
    # Waterjet viability assessment
    operable_area = float(df[df[area_col] > 0][area_col].sum())
    operable_objects = int(len(df[df[area_col] > 0]))
    
    viability_score = "good"
    if zero_area > total_objects * 0.5:
        viability_score = "poor"
    elif zero_area > total_objects * 0.3:
        viability_score = "fair"
    
    return {
        "success": True,
        "statistics": {
            "total_objects": total_objects,
            "selected_count": selected_count,
            "not_selected_count": total_objects - selected_count,
            "total_area_mm2": total_area,
            "total_perimeter_mm": total_perimeter,
            "operable_area_mm2": operable_area,
            "operable_objects": operable_objects,
            "area_stats": {
                "mean": mean_area,
                "median": median_area,
                "min": min_area,
                "max": max_area,
            },
            "perimeter_stats": {
                "mean": mean_perimeter,
                "median": median_perimeter,
                "min": min_perimeter,
                "max": max_perimeter,
            },
            "area_distribution": {
                "zero": zero_area,
                "tiny_lt1": tiny_area,
                "small_1to10": small_area,
                "medium_10to100": medium_area,
                "large_ge100": large_area,
            },
            "perimeter_distribution": {
                "tiny_lt5": tiny_perimeter,
                "small_5to50": small_perimeter,
                "medium_50to100": medium_perimeter,
                "large_ge100": large_perimeter,
            },
        },
        "layer_distribution": layer_dist,
        "layer_statistics": layer_stats,
        "top_objects": {
            "largest_by_area": top_largest,
            "smallest_non_zero": smallest_non_zero,
        },
        "recommendations": {
            "warnings": warnings,
            "info": info,
            "viability_score": viability_score,
        },
        "summary": {
            "cutting_length_m": total_perimeter / 1000.0,
            "pierces_estimate": operable_objects,
            "net_operable_area_mm2": operable_area,
        }
    }


def analyze_with_recommendations(
    csv_path: str,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze CSV and enhance with executable recommendations.
    
    This combines CSV analysis with the recommendation engine to produce
    actionable operations.
    
    Args:
        csv_path: Path to CSV file
        report: Optional DXF analysis report (for better recommendations)
        
    Returns:
        Enhanced analysis with executable operations
    """
    from wjp_analyser.ai.recommendation_engine import enhance_csv_analysis
    
    # Get base CSV analysis
    csv_analysis = analyze_csv(csv_path)
    
    # If report provided, enhance with recommendations
    if report and csv_analysis.get("success"):
        return enhance_csv_analysis(csv_analysis, report)
    
    return csv_analysis

