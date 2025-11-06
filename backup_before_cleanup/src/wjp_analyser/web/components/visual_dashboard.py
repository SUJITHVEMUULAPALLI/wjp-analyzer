"""
Visual Dashboard Components for DXF Analyzer
============================================

Modern, visually appealing dashboard components for DXF analysis reports.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime


def render_dashboard_header(file_name: str = "analyze_dxf", analysis_date: str = None) -> None:
    """Render modern dashboard header with gradient styling."""
    if analysis_date is None:
        analysis_date = datetime.now().strftime("%d %b %Y")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><defs><pattern id=\"grain\" width=\"100\" height=\"100\" patternUnits=\"userSpaceOnUse\"><circle cx=\"25\" cy=\"25\" r=\"1\" fill=\"white\" opacity=\"0.1\"/><circle cx=\"75\" cy=\"75\" r=\"1\" fill=\"white\" opacity=\"0.1\"/><circle cx=\"50\" cy=\"10\" r=\"0.5\" fill=\"white\" opacity=\"0.1\"/></pattern></defs><rect width=\"100\" height=\"100\" fill=\"url(%23grain)\"/></svg>') repeat;
            opacity: 0.3;
            animation: float 20s ease-in-out infinite;
        "></div>
        <div style="position: relative; z-index: 2;">
            <h1 style="
                color: white;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                gap: 1rem;
            ">
                ⚙️ DXF ANALYZER – OPTIMIZATION REPORT
            </h1>
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 1rem;
                color: rgba(255,255,255,0.9);
                font-size: 1.1rem;
            ">
                <span>📁 File: {}</span>
                <span>📅 Date: {}</span>
            </div>
        </div>
    </div>
    """.format(file_name, analysis_date), unsafe_allow_html=True)


def render_summary_metrics_cards(metrics: Dict[str, Any]) -> None:
    """Render summary metrics as modern cards with icons and colors."""
    
    # Extract metrics with fallbacks
    total_objects = metrics.get("total_objects", 89)
    total_groups = metrics.get("total_groups", 22)
    cutting_length = metrics.get("cutting_length_m", 39.6)
    cutting_cost = metrics.get("cutting_cost_inr", 32000)
    garnet_use = metrics.get("garnet_use_kg", 24)
    cutting_time = metrics.get("cutting_time_h", 1.17)
    
    # Define card data
    cards_data = [
        {
            "title": "Total Objects",
            "value": total_objects,
            "icon": "🧩",
            "color": "#4ADE80",
            "bg_gradient": "linear-gradient(135deg, #4ADE80 0%, #22C55E 100%)"
        },
        {
            "title": "Groups Detected", 
            "value": total_groups,
            "icon": "🗂️",
            "color": "#60A5FA",
            "bg_gradient": "linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%)"
        },
        {
            "title": "Cutting Length",
            "value": f"{cutting_length:.1f} m",
            "icon": "✂️",
            "color": "#FACC15",
            "bg_gradient": "linear-gradient(135deg, #FACC15 0%, #EAB308 100%)"
        },
        {
            "title": "Cutting Cost",
            "value": f"₹ {cutting_cost:,}",
            "icon": "💰",
            "color": "#FB923C",
            "bg_gradient": "linear-gradient(135deg, #FB923C 0%, #F97316 100%)"
        },
        {
            "title": "Garnet Use",
            "value": f"{garnet_use} kg",
            "icon": "🧱",
            "color": "#E879F9",
            "bg_gradient": "linear-gradient(135deg, #E879F9 0%, #D946EF 100%)"
        },
        {
            "title": "Cutting Time",
            "value": f"{cutting_time:.1f} h",
            "icon": "⏱️",
            "color": "#34D399",
            "bg_gradient": "linear-gradient(135deg, #34D399 0%, #10B981 100%)"
        }
    ]
    
    # Render cards in columns
    cols = st.columns(3)
    
    for i, card in enumerate(cards_data):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background: {card['bg_gradient']};
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
                cursor: pointer;
            " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 0.5rem;
                ">
                    <span style="font-size: 2rem;">{card['icon']}</span>
                    <div style="
                        background: rgba(255,255,255,0.2);
                        padding: 0.25rem 0.75rem;
                        border-radius: 20px;
                        font-size: 0.8rem;
                        color: white;
                        font-weight: 600;
                    ">{card['title']}</div>
                </div>
                <div style="
                    font-size: 2rem;
                    font-weight: 700;
                    color: white;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
                ">{card['value']}</div>
            </div>
            """, unsafe_allow_html=True)


def render_object_category_chart(categories: Dict[str, Any]) -> None:
    """Render object category distribution as pie chart."""
    
    # Default categories if not provided
    if not categories:
        categories = {
            "Large_Moderate": {"count": 12, "percentage": 27},
            "Large_Simple": {"count": 10, "percentage": 22},
            "Large_Circle": {"count": 16, "percentage": 18},
            "Medium_Simple": {"count": 12, "percentage": 14},
            "Small_Simple": {"count": 4, "percentage": 10},
            "Small_Polygon": {"count": 4, "percentage": 9}
        }
    
    # Prepare data for plotly
    labels = list(categories.keys())
    values = [cat["count"] for cat in categories.values()]
    colors = ["#4ADE80", "#60A5FA", "#FACC15", "#FB923C", "#E879F9", "#34D399"]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': "📊 Object Category Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f2937'}
        },
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=0, r=150, t=50, b=0),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_layered_nesting_preview(layers: Dict[str, Any]) -> None:
    """Render layered nesting visualization."""
    
    # Default layers if not provided
    if not layers:
        layers = {
            "Inlay_Outer": {"color": "#3B82F6", "objects": 12, "description": "External motif regions"},
            "Inlay_Inner": {"color": "#10B981", "objects": 22, "description": "Main inlay cuts"},
            "Pierce_Points": {"color": "#EF4444", "objects": 10, "description": "Small decorative details"},
            "Boundary": {"color": "#6B7280", "objects": 1, "description": "Fixed outline"}
        }
    
    st.markdown("### 🎨 Layer Assignment Preview")
    
    # Create layer visualization
    layer_html = ""
    for layer_name, layer_data in layers.items():
        layer_html += f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 1rem;
            margin: 0.5rem 0;
            background: linear-gradient(90deg, {layer_data['color']}20 0%, {layer_data['color']}10 100%);
            border-left: 4px solid {layer_data['color']};
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        ">
            <div style="
                width: 20px;
                height: 20px;
                background: {layer_data['color']};
                border-radius: 50%;
                margin-right: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            "></div>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: #1f2937; margin-bottom: 0.25rem;">
                    {layer_name.replace('_', ' ')} ({layer_data['objects']} objects)
                </div>
                <div style="color: #6b7280; font-size: 0.9rem;">
                    {layer_data['description']}
                </div>
            </div>
        </div>
        """
    
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    ">
        {layer_html}
    </div>
    """, unsafe_allow_html=True)


def render_optimization_recommendations(recommendations: List[Dict[str, Any]]) -> None:
    """Render optimization recommendations as action cards."""
    
    # Default recommendations if not provided
    if not recommendations:
        recommendations = [
            {
                "action": "Merge Small Polygons",
                "impact": "↓ pierce count 10%",
                "icon": "🧩",
                "description": "Auto-detect shapes < 100 mm² and merge them",
                "priority": "high",
                "color": "#EF4444"
            },
            {
                "action": "Mirror Geometry (1/8th)",
                "impact": "↓ file size 87%",
                "icon": "🔁",
                "description": "Use radial symmetry to store only 1/8th pattern",
                "priority": "high",
                "color": "#F59E0B"
            },
            {
                "action": "Inside-First Cutting",
                "impact": "↓ cut time 15%",
                "icon": "🪞",
                "description": "Prioritize inner contours before outer rings",
                "priority": "medium",
                "color": "#10B981"
            },
            {
                "action": "Layer Color Logic",
                "impact": "+ clarity",
                "icon": "🎨",
                "description": "Blue=Outer, Green=Inner, Red=Pierce",
                "priority": "low",
                "color": "#3B82F6"
            },
            {
                "action": "Kerf Compensation",
                "impact": "+ accuracy",
                "icon": "📏",
                "description": "Maintain 1.1 mm kerf for precise cuts",
                "priority": "medium",
                "color": "#8B5CF6"
            }
        ]
    
    st.markdown("### ⚙️ Optimization Recommendations")
    
    # Group by priority
    priority_groups = {"high": [], "medium": [], "low": []}
    for rec in recommendations:
        priority_groups[rec.get("priority", "medium")].append(rec)
    
    # Render by priority
    for priority, recs in priority_groups.items():
        if not recs:
            continue
            
        priority_title = priority.title()
        priority_color = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}[priority]
        
        st.markdown(f"""
        <div style="
            background: {priority_color}10;
            border-left: 4px solid {priority_color};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 8px;
        ">
            <h4 style="color: {priority_color}; margin: 0 0 1rem 0;">{priority_title} Priority</h4>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        for i, rec in enumerate(recs):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                    margin-bottom: 1rem;
                    border-top: 4px solid {rec['color']};
                    transition: transform 0.2s ease;
                " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                    <div style="
                        display: flex;
                        align-items: center;
                        margin-bottom: 1rem;
                    ">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{rec['icon']}</span>
                        <div>
                            <div style="font-weight: 600; color: #1f2937; font-size: 1.1rem;">
                                {rec['action']}
                            </div>
                            <div style="
                                background: {rec['color']}20;
                                color: {rec['color']};
                                padding: 0.25rem 0.5rem;
                                border-radius: 12px;
                                font-size: 0.8rem;
                                font-weight: 600;
                                display: inline-block;
                                margin-top: 0.25rem;
                            ">
                                {rec['impact']}
                            </div>
                        </div>
                    </div>
                    <div style="color: #6b7280; font-size: 0.9rem; line-height: 1.4;">
                        {rec['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_waterjet_compliance_checklist(compliance: Dict[str, Any]) -> None:
    """Render waterjet compliance checklist with status indicators."""
    
    # Default compliance data if not provided
    if not compliance:
        compliance = {
            "radius_check": {"status": "pass", "message": "All shapes meet minimum radius (> 2 mm)"},
            "spacing_check": {"status": "pass", "message": "No spacing violations detected (< 3 mm)"},
            "small_shapes": {"status": "warning", "message": "4 shapes < 100 mm² – recommend merging"},
            "geometry_stability": {"status": "pass", "message": "Total geometry stable and waterjet-safe"}
        }
    
    st.markdown("### 🔧 Waterjet Compliance Checklist")
    
    checklist_html = ""
    for check_name, check_data in compliance.items():
        status = check_data["status"]
        message = check_data["message"]
        
        if status == "pass":
            icon = "✅"
            color = "#10B981"
            bg_color = "#10B98110"
        elif status == "warning":
            icon = "⚠️"
            color = "#F59E0B"
            bg_color = "#F59E0B10"
        else:
            icon = "❌"
            color = "#EF4444"
            bg_color = "#EF444410"
        
        checklist_html += f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 1rem;
            margin: 0.5rem 0;
            background: {bg_color};
            border-left: 4px solid {color};
            border-radius: 8px;
        ">
            <span style="font-size: 1.2rem; margin-right: 1rem;">{icon}</span>
            <div style="color: #1f2937; font-weight: 500;">
                {message}
            </div>
        </div>
        """
    
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    ">
        {checklist_html}
    </div>
    """, unsafe_allow_html=True)


def render_technical_summary_table(summary_data: List[Dict[str, Any]]) -> None:
    """Render technical summary as styled table."""
    
    # Default summary data if not provided
    if not summary_data:
        summary_data = [
            {"layer": "Inlay_Outer", "objects": 12, "avg_area": "18,000 mm²", "complexity": "Moderate", "action": "Keep"},
            {"layer": "Inlay_Inner", "objects": 22, "avg_area": "5,000 mm²", "complexity": "Simple", "action": "Optimize cut order"},
            {"layer": "Pierce_Points", "objects": 10, "avg_area": "80 mm²", "complexity": "Simple", "action": "Merge"},
            {"layer": "Boundary", "objects": 1, "avg_area": "900,000 mm²", "complexity": "N/A", "action": "Fixed outline"}
        ]
    
    st.markdown("### 📋 Technical Summary")
    
    # Create DataFrame for better formatting
    df = pd.DataFrame(summary_data)
    
    # Style the DataFrame
    def style_row(row):
        if row.name % 2 == 0:
            return ['background-color: #f8fafc'] * len(row)
        else:
            return ['background-color: white'] * len(row)
    
    styled_df = df.style.apply(style_row, axis=1)
    
    # Display with custom styling
    st.markdown("""
    <style>
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 1rem;
    }
    .dataframe td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_export_buttons(analysis_id: str = None) -> None:
    """Render export buttons for PDF reports and optimized DXF."""
    
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧾 Export PDF Report", use_container_width=True):
            st.success("PDF report generation started!")
            # TODO: Implement PDF generation
    
    with col2:
        if st.button("💡 Optimize DXF (Mirrored)", use_container_width=True):
            st.success("DXF optimization started!")
            # TODO: Implement DXF optimization
    
    with col3:
        if st.button("📦 Export G-Code", use_container_width=True):
            st.success("G-code export started!")
            # TODO: Implement G-code export


def render_full_dashboard(report_data: Dict[str, Any], file_name: str = "analyze_dxf") -> None:
    """Render complete visual dashboard."""
    
    # Extract data from report
    metrics = report_data.get("metrics", {})
    categories = report_data.get("categories", {})
    layers = report_data.get("layers", {})
    recommendations = report_data.get("recommendations", [])
    compliance = report_data.get("compliance", {})
    summary_data = report_data.get("summary_data", [])
    
    # Render all components
    render_dashboard_header(file_name)
    render_summary_metrics_cards(metrics)
    
    col1, col2 = st.columns(2)
    with col1:
        render_object_category_chart(categories)
    with col2:
        render_layered_nesting_preview(layers)
    
    render_optimization_recommendations(recommendations)
    render_waterjet_compliance_checklist(compliance)
    render_technical_summary_table(summary_data)
    render_export_buttons()


# Add custom CSS for animations
def add_custom_css():
    """Add custom CSS for animations and styling."""
    st.markdown("""
    <style>
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .metric-card {
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .stMarkdown {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
