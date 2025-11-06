# Interactive DXF Preview System

## 🎯 Overview

The Interactive DXF Preview System provides bidirectional object selection between the DXF preview and object management interface. Users can click objects in the preview to select them, or select objects from a list to highlight them in the preview - creating a mini CAD viewer experience.

## 🏗️ Architecture

### Core Components

1. **DXF to SVG Converter** (`dxf_to_svg.py`)
   - Converts DXF files to interactive SVG with object IDs
   - Adds click detection and hover effects
   - Supports both ezdxf and svgwrite backends

2. **Interactive SVG Viewer** (`interactive_svg_viewer.py`)
   - Streamlit component for displaying interactive SVG
   - Handles click detection and selection
   - Provides tooltips and highlighting

3. **Integration** (`analyze_dxf.py`)
   - New "Interactive Preview" display mode
   - Bidirectional selection synchronization
   - Session state management

## 🚀 Features

### Level 1: Basic Interactive Selection
- ✅ **Object List Selection** → Highlights in preview
- ✅ **Click Preview** → Updates selection in list
- ✅ **Hover Tooltips** → Object information on hover
- ✅ **Visual Highlighting** → Selected objects highlighted in red

### Level 2: Advanced Interactive Features
- ✅ **Layer Filtering** → Toggle layer visibility
- ✅ **Preview Controls** → Zoom, grid, styling options
- ✅ **Object Details** → Real-time object information
- ✅ **Bidirectional Sync** → Selection updates both ways

### Level 3: CAD-like Experience
- ✅ **Interactive SVG** → Clickable objects with JavaScript
- ✅ **Professional Styling** → Modern, clean interface
- ✅ **Error Handling** → Graceful fallbacks
- ✅ **Performance Optimized** → Efficient rendering

## 📊 How It Works

### 1. DXF to SVG Conversion
```python
# Convert DXF to interactive SVG
svg_code, object_mapping = convert_dxf_to_interactive_svg(dxf_path, components)

# Or create simple SVG from component data
svg_code = create_simple_interactive_svg(components)
```

### 2. Interactive SVG Display
```python
# Display interactive SVG with click detection
clicked_object = interactive_svg_viewer(
    svg_code,
    object_mapping,
    selected_object_id,
    width=800,
    height=600
)
```

### 3. Bidirectional Selection
```python
# Object selection panel
selected_object, mapping = create_object_selection_panel(
    components,
    selected_object_id,
    object_mapping
)

# Selection syncs automatically between preview and list
```

## 🎨 User Experience

### Interactive Preview Mode
1. **Upload DXF** → File is analyzed and converted to interactive SVG
2. **View Preview** → Interactive SVG displayed with clickable objects
3. **Click Objects** → Objects are highlighted and selected
4. **Select from List** → Preview highlights selected object
5. **View Details** → Object information displayed in real-time

### Visual Feedback
- **Hover Effects**: Objects highlight on mouse over
- **Selection Highlighting**: Selected objects shown in red
- **Tooltips**: Object information on hover
- **Smooth Transitions**: CSS animations for better UX

## 🔧 Technical Implementation

### SVG Enhancement
```html
<path d="M 100,100 L 200,100 L 200,200 L 100,200 Z" 
      class="dxf-object" 
      data-object-id="OBJ-001"
      stroke="#1F2937" 
      stroke-width="1" 
      fill="none"
      onclick="handleClick(event)" />
```

### JavaScript Interaction
```javascript
function handleObjectClick(event) {
    const objectId = event.target.getAttribute('data-object-id');
    
    // Remove previous selection
    document.querySelectorAll('.dxf-object.selected').forEach(el => {
        el.classList.remove('selected');
    });
    
    // Add selection to clicked object
    event.target.classList.add('selected');
    
    // Send to Streamlit
    window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: objectId
    }, '*');
}
```

### Streamlit Integration
```python
# Display interactive SVG
clicked_object = interactive_svg_viewer(svg_code, object_mapping)

# Update selection if clicked
if clicked_object:
    st.session_state["selected_object"] = clicked_object
    st.rerun()
```

## 📋 Usage Examples

### Basic Integration
```python
from wjp_analyser.web.components.interactive_svg_viewer import interactive_svg_viewer
from wjp_analyser.web.components.dxf_to_svg import create_simple_interactive_svg

# Create interactive SVG
svg_code = create_simple_interactive_svg(components)

# Display with click detection
clicked_object = interactive_svg_viewer(svg_code, object_mapping)
```

### Full Integration
```python
# In analyze_dxf.py
if display_mode == "Interactive Preview":
    # Create interactive SVG
    svg_code, object_mapping = convert_dxf_to_interactive_svg(dxf_path, components)
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive preview
        clicked_object = interactive_svg_viewer(svg_code, object_mapping)
    
    with col2:
        # Object selection panel
        selected_object, mapping = create_object_selection_panel(components)
    
    # Bidirectional sync
    if clicked_object != selected_object:
        st.session_state["selected_object"] = clicked_object
        st.rerun()
```

## 🎮 Interactive Features

### Click Selection
- Click any object in the preview
- Object is highlighted in red
- Selection updates in object list
- Object details displayed below

### Dropdown Selection
- Select object from dropdown
- Preview highlights selected object
- Bidirectional synchronization
- Real-time updates

### Hover Effects
- Mouse over objects for tooltips
- Object information displayed
- Smooth hover animations
- Professional styling

### Layer Controls
- Toggle layer visibility
- Filter objects by layer
- Customize preview appearance
- Real-time layer updates

## 🔧 Configuration

### Dependencies
```bash
# Required for DXF to SVG conversion
pip install ezdxf svgwrite

# Optional for advanced features
pip install streamlit-components-v1
```

### Environment Setup
```python
# Check component availability
try:
    from wjp_analyser.web.components.dxf_to_svg import convert_dxf_to_interactive_svg
    INTERACTIVE_SVG_AVAILABLE = True
except ImportError:
    INTERACTIVE_SVG_AVAILABLE = False
```

## 📱 Responsive Design

### Screen Sizes
- **Desktop**: Full interactive experience
- **Tablet**: Optimized layout and controls
- **Mobile**: Simplified interface with touch support

### Performance
- **Lazy Loading**: Components load only when needed
- **Efficient Rendering**: Optimized SVG generation
- **Memory Management**: Proper cleanup of resources
- **Error Handling**: Graceful fallbacks

## 🚀 Future Enhancements

### Planned Features
1. **Pan and Zoom**: Advanced navigation controls
2. **Multi-selection**: Select multiple objects
3. **Object Editing**: In-place object modification
4. **Layer Management**: Advanced layer operations
5. **Export Options**: Save interactive SVG

### Extension Points
- **Custom Styling**: User-defined color schemes
- **Plugin System**: Extensible functionality
- **API Integration**: Connect with external systems
- **Real-time Collaboration**: Multi-user editing

## 📞 Support

### Troubleshooting
- **SVG Not Displaying**: Check ezdxf installation
- **Click Detection Not Working**: Verify JavaScript enabled
- **Performance Issues**: Reduce object count or simplify geometry
- **Styling Problems**: Check CSS compatibility

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎉 Result

The Interactive DXF Preview System transforms your DXF Analyzer into a **mini CAD viewer** where users can:

- **Click objects** in the preview to select them
- **See selections** reflected in the object list
- **Hover for details** with professional tooltips
- **Filter by layers** with real-time updates
- **Control preview** with zoom, grid, and styling options

This creates a **professional, interactive experience** that makes DXF analysis intuitive and engaging! 🎯

---

**Generated by WJP DXF Analyzer v2.0** | www.wjpmanager.in
