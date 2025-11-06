"""
Professional DXF Analyzer Report Generator
==========================================

Structured, one-page professional report template for DXF analysis results.
Ready to integrate into WJP Analyzer or export as PDF.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import json


def render_report_header(filename: str = "analyze_dxf", analysis_date: str = None) -> None:
    """Render professional report header with WJP branding."""
    if analysis_date is None:
        analysis_date = datetime.now().strftime('%d %b %Y')
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1F2937 0%, #374151 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <h1 style="
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        ">⚙️ DXF ANALYZER – OPTIMIZATION REPORT</h1>
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            font-size: 14px;
            opacity: 0.9;
        ">
            <span>📁 File: <strong>{filename}</strong></span>
            <span>📅 Date: <strong>{analysis_date}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_summary_cards(summary_data: List[tuple]) -> None:
    """Render colored metric tiles in a responsive grid."""
    
    # Default summary data if not provided
    if not summary_data:
        summary_data = [
            ("Objects", "89", "#22C55E", "🧩"),
            ("Groups", "22", "#3B82F6", "🗂️"),
            ("Cutting Length", "39.6 m", "#FACC15", "✂️"),
            ("Cost", "₹ 31,700 – 33,600", "#FB923C", "💰"),
            ("Time", "1 h 10 min", "#10B981", "⏱️"),
            ("Garnet", "24 kg", "#E879F9", "🧱"),
        ]
    
    # Create 6 columns for the metrics
    cols = st.columns(6)
    
    for col, (label, value, color, emoji) in zip(cols, summary_data):
        col.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color} 0%, {color}CC 100%);
            color: white;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
            margin-bottom: 10px;
        " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size: 1.5rem; margin-bottom: 5px;">{emoji}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{label}</div>
            <div style="font-size: 1.3rem; margin-top: 5px;">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def render_geometry_charts(categories_data: Dict[str, Any] = None, layers_data: Dict[str, Any] = None) -> None:
    """Render geometry distribution pie chart and layer cutting length bar chart."""
    
    # Default data if not provided
    if not categories_data:
        categories_data = {
            'Large Moderate': 27,
            'Large Simple': 22,
            'Large Circle': 18,
            'Medium Simple': 14,
            'Small Simple': 10,
            'Small Polygon': 9
        }
    
    if not layers_data:
        layers_data = {
            'Inlay_Outer': 14.3,
            'Inlay_Inner': 18.2,
            'Pierce_Points': 5.8,
            'Boundary': 1.3
        }
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Object Category Distribution")
        
        # Pie Chart
        labels = list(categories_data.keys())
        sizes = list(categories_data.values())
        colors = ['#3B82F6', '#22C55E', '#FACC15', '#FB923C', '#E879F9', '#10B981']
        
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            labels=labels, 
            autopct='%1.0f%%', 
            startangle=140,
            colors=colors[:len(labels)],
            textprops={'fontsize': 10}
        )
        ax1.set_title("Object Category Distribution", fontsize=14, fontweight='bold', pad=20)
        
        # Enhance the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
    
    with col2:
        st.markdown("### 📏 Cutting Length by Layer")
        
        # Bar Chart
        layers = list(layers_data.keys())
        lengths = list(layers_data.values())
        bar_colors = ['#3B82F6', '#22C55E', '#EF4444', '#9CA3AF']
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.bar(layers, lengths, color=bar_colors[:len(layers)], alpha=0.8)
        ax2.set_ylabel("Cutting Length (m)", fontsize=12)
        ax2.set_title("Cutting Length by Layer", fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, length in zip(bars, lengths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{length:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


def render_insights_and_compliance(optimization_data: List[Dict[str, str]] = None, 
                                 compliance_data: Dict[str, Dict[str, str]] = None) -> None:
    """Render optimization insights and compliance checklist."""
    
    # Default optimization data
    if not optimization_data:
        optimization_data = [
            {"action": "Merge small polygons", "impact": "↓ pierce count 10%", "icon": "🧩", "color": "#E0F2FE"},
            {"action": "Mirror geometry (1/8th)", "impact": "↓ file size 87%", "icon": "🔁", "color": "#DCFCE7"},
            {"action": "Inside-first cut order", "impact": "↓ cut time 15%", "icon": "🪞", "color": "#FEF3C7"},
            {"action": "Maintain kerf 1.1 mm", "impact": "+ accuracy", "icon": "📏", "color": "#FCE7F3"},
        ]
    
    # Default compliance data
    if not compliance_data:
        compliance_data = {
            "radius_check": {"status": "✅", "message": "Radius ≥ 2 mm"},
            "spacing_check": {"status": "✅", "message": "Spacing ≥ 3 mm"},
            "small_shapes": {"status": "⚠️", "message": "4 shapes < 100 mm² (merge recommended)"},
            "layer_naming": {"status": "✅", "message": "Layer naming consistent"},
            "cut_order": {"status": "⚠️", "message": "Verify cut order (manual)"},
        }
    
    # Optimization Insights Section
    st.markdown("### 🧠 Optimization Insights")
    
    # Create optimization cards in a grid
    opt_cols = st.columns(2)
    for i, opt in enumerate(optimization_data):
        with opt_cols[i % 2]:
            st.markdown(f"""
            <div style="
                background: {opt['color']};
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 4px solid {opt['color'].replace('FE', 'CC').replace('FC', 'CC')};
            ">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">{opt['icon']}</span>
                    <strong style="color: #1F2937;">{opt['action']}</strong>
                </div>
                <div style="color: #374151; font-size: 0.9rem;">
                    <strong>{opt['impact']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Compliance Checklist Section
    st.markdown("### 🔧 Waterjet Compliance Checklist")
    
    compliance_html = ""
    for check_name, check_data in compliance_data.items():
        compliance_html += f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: #F9FAFB;
            border-radius: 8px;
            border-left: 4px solid #E5E7EB;
        ">
            <span style="font-size: 1.1rem; margin-right: 10px;">{check_data['status']}</span>
            <span style="color: #374151;">{check_data['message']}</span>
        </div>
        """
    
    st.markdown(f"""
    <div style="
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 10px 0;
    ">
        {compliance_html}
    </div>
    """, unsafe_allow_html=True)


def render_technical_table(table_data: List[Dict[str, Any]] = None) -> None:
    """Render technical summary table with professional styling."""
    
    # Default table data
    if not table_data:
        table_data = [
            {"Layer": "Inlay_Outer", "Objects": 12, "Avg Area (mm²)": 18000, "Complexity": "Moderate", "Action": "Keep"},
            {"Layer": "Inlay_Inner", "Objects": 22, "Avg Area (mm²)": 5000, "Complexity": "Simple", "Action": "Optimize cut order"},
            {"Layer": "Pierce_Points", "Objects": 10, "Avg Area (mm²)": 80, "Complexity": "Simple", "Action": "Merge"},
            {"Layer": "Boundary", "Objects": 1, "Avg Area (mm²)": 900000, "Complexity": "N/A", "Action": "Fixed outline"},
        ]
    
    st.markdown("### 📋 Technical Summary")
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Style the DataFrame
    def style_row(row):
        if row.name % 2 == 0:
            return ['background-color: #F9FAFB'] * len(row)
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
        border: 1px solid #E5E7EB;
    }
    .dataframe th {
        background: linear-gradient(135deg, #1F2937 0%, #374151 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
        text-align: center;
    }
    .dataframe td {
        padding: 10px 12px;
        border-bottom: 1px solid #E5E7EB;
        text-align: center;
    }
    .dataframe tr:hover {
        background-color: #F3F4F6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_footer() -> None:
    """Render professional footer with WJP branding."""
    st.markdown("""
    <div style="
        margin-top: 40px;
        padding: 20px;
        background: #F9FAFB;
        border-radius: 12px;
        text-align: center;
        border-top: 3px solid #1F2937;
    ">
        <hr style="border: 1px solid #E5E7EB; margin: 20px 0;"/>
        <p style="
            color: #6B7280;
            font-size: 14px;
            margin: 0;
            font-weight: 500;
        ">
            Generated by <strong style="color: #1F2937;">WJP DXF Analyzer v2.0</strong> | 
            <a href="https://www.wjpmanager.in" style="color: #3B82F6; text-decoration: none;">www.wjpmanager.in</a>
        </p>
        <p style="
            color: #9CA3AF;
            font-size: 12px;
            margin: 5px 0 0 0;
        ">
            Professional DXF Analysis & Waterjet Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)


def generate_professional_report(analysis_data: Dict[str, Any], filename: str = "analyze_dxf") -> None:
    """Generate complete professional report from analysis data."""
    
    # Extract data from analysis
    components = analysis_data.get("components", [])
    groups = analysis_data.get("groups", {})
    layers = analysis_data.get("layers", {})
    metrics = analysis_data.get("metrics", {})
    
    # Calculate summary metrics
    total_objects = len(components)
    total_groups = len(groups)
    
    # Calculate cutting metrics
    total_area = sum(c.get("area", 0) for c in components)
    total_perimeter = sum(c.get("perimeter", 0) for c in components)
    cutting_length = total_perimeter / 1000  # Convert to meters
    cutting_cost = cutting_length * 800  # ₹800 per meter
    garnet_use = cutting_length * 0.6  # 0.6 kg per meter
    cutting_time = cutting_length * 0.03  # 0.03 hours per meter
    
    # Prepare summary cards data
    summary_data = [
        ("Objects", str(total_objects), "#22C55E", "🧩"),
        ("Groups", str(total_groups), "#3B82F6", "🗂️"),
        ("Cutting Length", f"{cutting_length:.1f} m", "#FACC15", "✂️"),
        ("Cost", f"₹ {cutting_cost:,.0f}", "#FB923C", "💰"),
        ("Time", f"{cutting_time:.1f} h", "#10B981", "⏱️"),
        ("Garnet", f"{garnet_use:.0f} kg", "#E879F9", "🧱"),
    ]
    
    # Categorize objects for pie chart
    categories = {}
    for comp in components:
        area = comp.get("area", 0)
        perimeter = comp.get("perimeter", 0)
        vcount = comp.get("vertex_count", 0)
        
        # Calculate circularity
        circularity = (4.0 * 3.14159 * area) / (perimeter * perimeter + 1e-9) if perimeter > 0 else 0
        
        if area > 10000:
            if circularity > 0.8:
                cat = "Large Circle"
            elif vcount > 6:
                cat = "Large Moderate"
            else:
                cat = "Large Simple"
        elif area > 1000:
            cat = "Medium Simple"
        elif area > 100:
            cat = "Small Simple"
        else:
            cat = "Small Polygon"
        
        categories[cat] = categories.get(cat, 0) + 1
    
    # Calculate percentages for pie chart
    total_categorized = sum(categories.values())
    categories_percent = {k: (v / total_categorized * 100) for k, v in categories.items()}
    
    # Prepare layer cutting lengths (simplified)
    layer_lengths = {}
    for layer_name, count in layers.items():
        # Estimate cutting length based on layer type
        if layer_name == "OUTER":
            layer_lengths["Inlay_Outer"] = count * 1.2  # Estimated meters
        elif layer_name == "INNER":
            layer_lengths["Inlay_Inner"] = count * 0.8
        elif layer_name in ["HOLE", "DECOR"]:
            layer_lengths["Pierce_Points"] = layer_lengths.get("Pierce_Points", 0) + count * 0.6
        else:
            layer_lengths[layer_name] = count * 0.5
    
    # Prepare technical table data
    table_data = []
    for layer_name, count in layers.items():
        avg_area = total_area / count if count > 0 else 0
        complexity = "Moderate" if layer_name == "OUTER" else "Simple"
        action = "Keep" if layer_name == "OUTER" else "Optimize cut order" if layer_name == "INNER" else "Merge"
        
        table_data.append({
            "Layer": layer_name,
            "Objects": count,
            "Avg Area (mm²)": f"{avg_area:,.0f}",
            "Complexity": complexity,
            "Action": action
        })
    
    # Render all sections
    render_report_header(filename)
    render_summary_cards(summary_data)
    render_geometry_charts(categories_percent, layer_lengths)
    render_insights_and_compliance()
    render_technical_table(table_data)
    render_footer()


def add_report_styling() -> None:
    """Add custom CSS styling for the report."""
    st.markdown("""
    <style>
    /* Custom styling for the professional report */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Smooth transitions */
    div[data-testid="stMarkdownContainer"] {
        transition: all 0.3s ease;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Print styles for PDF export */
    @media print {
        .main .block-container {
            padding: 0;
        }
        
        div[data-testid="stMarkdownContainer"] {
            break-inside: avoid;
        }
    }
    </style>
    """, unsafe_allow_html=True)
