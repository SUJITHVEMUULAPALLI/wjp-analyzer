# Visual Dashboard Components

## Overview

The Visual Dashboard Components provide a modern, visually appealing interface for DXF analysis reports. These components transform raw analysis data into professional, interactive dashboards with:

- **Modern Design**: Gradient backgrounds, color-coded metrics, and professional styling
- **Interactive Visualizations**: Pie charts, layered previews, and hover effects
- **Comprehensive Insights**: Optimization recommendations, compliance checks, and technical summaries
- **Export Functionality**: PDF reports, optimized DXF files, and G-code generation

## Components

### Core Components

#### `render_dashboard_header(file_name, analysis_date)`
Creates a modern header with gradient styling and WJP branding.

#### `render_summary_metrics_cards(metrics)`
Displays key metrics as colorful, animated cards with icons:
- Total Objects (🧩)
- Groups Detected (🗂️)
- Cutting Length (✂️)
- Cutting Cost (💰)
- Garnet Use (🧱)
- Cutting Time (⏱️)

#### `render_object_category_chart(categories)`
Interactive pie chart showing object distribution by category with hover tooltips.

#### `render_layered_nesting_preview(layers)`
Color-coded layer visualization with descriptions and object counts.

#### `render_optimization_recommendations(recommendations)`
Action cards with priority-based styling and impact indicators.

#### `render_waterjet_compliance_checklist(compliance)`
Status indicators (✅ ⚠️ ❌) for waterjet manufacturing requirements.

#### `render_technical_summary_table(summary_data)`
Styled table with alternating row colors and professional formatting.

#### `render_export_buttons(analysis_id)`
Export options for PDF reports, optimized DXF, and G-code files.

### Utility Functions

#### `render_full_dashboard(report_data, file_name)`
Renders the complete dashboard with all components.

#### `add_custom_css()`
Adds custom CSS for animations and styling effects.

## Usage

### Basic Integration

```python
from wjp_analyser.web.components.visual_dashboard import render_full_dashboard

# Prepare your analysis data
dashboard_data = {
    "metrics": {
        "total_objects": 89,
        "total_groups": 22,
        "cutting_length_m": 39.6,
        "cutting_cost_inr": 32000,
        "garnet_use_kg": 24,
        "cutting_time_h": 1.17
    },
    "categories": {
        "Large_Moderate": {"count": 12, "percentage": 27},
        "Large_Simple": {"count": 10, "percentage": 22},
        # ... more categories
    },
    "layers": {
        "Inlay_Outer": {"color": "#3B82F6", "objects": 12, "description": "External motif regions"},
        # ... more layers
    },
    "compliance": {
        "radius_check": {"status": "pass", "message": "All shapes meet minimum radius (> 2 mm)"},
        # ... more compliance checks
    },
    "summary_data": [
        {"layer": "Inlay_Outer", "objects": 12, "avg_area": "18,000 mm²", "complexity": "Moderate", "action": "Keep"},
        # ... more summary data
    ],
    "recommendations": [
        {
            "action": "Merge Small Polygons",
            "impact": "↓ pierce count 10%",
            "icon": "🧩",
            "description": "Auto-detect shapes < 100 mm² and merge them",
            "priority": "high",
            "color": "#EF4444"
        },
        # ... more recommendations
    ]
}

# Render the dashboard
render_full_dashboard(dashboard_data, "analyze_dxf")
```

### Individual Components

```python
from wjp_analyser.web.components.visual_dashboard import (
    render_dashboard_header,
    render_summary_metrics_cards,
    render_object_category_chart,
    # ... other components
)

# Add custom CSS
add_custom_css()

# Render individual components
render_dashboard_header("my_file.dxf")
render_summary_metrics_cards(metrics_data)
render_object_category_chart(categories_data)
```

## Data Format

### Metrics Data
```python
metrics = {
    "total_objects": int,           # Total number of objects
    "total_groups": int,            # Number of similarity groups
    "cutting_length_m": float,      # Total cutting length in meters
    "cutting_cost_inr": float,      # Estimated cutting cost in INR
    "garnet_use_kg": float,         # Garnet consumption in kg
    "cutting_time_h": float         # Estimated cutting time in hours
}
```

### Categories Data
```python
categories = {
    "category_name": {
        "count": int,               # Number of objects in category
        "percentage": float         # Percentage of total objects
    }
}
```

### Layers Data
```python
layers = {
    "layer_name": {
        "color": str,               # Hex color code (e.g., "#3B82F6")
        "objects": int,             # Number of objects in layer
        "description": str          # Human-readable description
    }
}
```

### Compliance Data
```python
compliance = {
    "check_name": {
        "status": str,              # "pass", "warning", or "fail"
        "message": str              # Status message
    }
}
```

### Recommendations Data
```python
recommendations = [
    {
        "action": str,              # Action name
        "impact": str,              # Impact description
        "icon": str,                # Emoji icon
        "description": str,         # Detailed description
        "priority": str,            # "high", "medium", or "low"
        "color": str                # Hex color code
    }
]
```

## Styling

The dashboard uses a modern design system with:

- **Color Palette**: Professional blues, greens, and accent colors
- **Typography**: Clean, readable fonts with proper hierarchy
- **Spacing**: Consistent margins and padding
- **Animations**: Subtle hover effects and transitions
- **Responsive Design**: Adapts to different screen sizes

## Dependencies

- **Streamlit**: Web framework for the dashboard interface
- **Plotly**: Interactive charts and visualizations
- **Pandas**: Data manipulation for tables
- **NumPy**: Numerical computations

## Demo

A complete demo is available at `src/wjp_analyser/web/pages/demo_dashboard.py` showing all components with sample data from the analysis report.

## Integration with DXF Analyzer

The visual dashboard is integrated into the main DXF analyzer page (`src/wjp_analyser/web/pages/analyze_dxf.py`) with a toggle between "Modern Dashboard" and "Classic View" modes.

## Future Enhancements

- **PDF Export**: Generate professional PDF reports
- **DXF Optimization**: Export optimized DXF files with mirroring
- **G-code Generation**: Direct G-code export functionality
- **Interactive Editing**: Allow users to modify recommendations
- **Real-time Updates**: Live updates as analysis progresses
