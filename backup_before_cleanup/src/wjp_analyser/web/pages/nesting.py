import uuid
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt


st.title("Nesting")

st.info(
    "If most parts are rectangular, installing 'rectpack' speeds up packing. "
    "This UI will automatically fall back to polygon-based packing if it's missing."
)


def do_nest(input_dxf_path: str, width: float, height: float, gap: float, polygon_mode: bool = True):
    # Import from the current codebase structure
    import sys
    import os
    from pathlib import Path
    
    # Add the src directory to Python path
    repo_root = Path(__file__).resolve().parents[4]  # pages -> web -> wjp_analyser -> src -> repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Import from the current codebase
    import ezdxf
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    
    # Add src to path for absolute imports
    import sys
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from wjp_analyser.nesting.nesting_engine import NestingEngine, NestingAlgorithm
    
    # Read DXF file
    doc = ezdxf.readfile(input_dxf_path)
    msp = doc.modelspace()
    
    # Extract parts from LWPOLYLINE entities
    parts = []
    for e in msp.query("LWPOLYLINE"):
        pts = [(v[0], v[1]) for v in e.get_points("xy")]
        if len(pts) < 3:
            continue
        try:
            poly = Polygon(pts)
            if poly.is_empty or not poly.is_valid:
                continue
            parts.append({
                "id": e.dxf.handle,
                "area": float(poly.area),
                "bbox": tuple(poly.bounds),
                "poly": poly,
                "points": pts,
            })
        except Exception:
            continue
    
    if not parts:
        raise ValueError("No valid parts found in DXF file")
    
    # Use the current nesting engine
    engine = NestingEngine()
    
    # Convert parts to DXFObject format expected by the engine
    from wjp_analyser.object_management.dxf_object_manager import DXFObject, ObjectType, ObjectComplexity, ObjectGeometry, ObjectMetadata
    
    dxf_objects = []
    for part in parts:
        # Create geometry info
        geom = ObjectGeometry(
            bounding_box=part["bbox"],
            area=part["area"],
            perimeter=part["poly"].length,
            centroid=(part["poly"].centroid.x, part["poly"].centroid.y),
            width=part["bbox"][2] - part["bbox"][0],
            height=part["bbox"][3] - part["bbox"][1],
            aspect_ratio=(part["bbox"][2] - part["bbox"][0]) / max(part["bbox"][3] - part["bbox"][1], 0.001),
            complexity_score=len(part["points"]),
            vertex_count=len(part["points"]),
            is_closed=True,
            is_convex=part["poly"].is_valid and part["poly"].convex_hull.equals(part["poly"]),
            has_holes=len(part["poly"].interiors) > 0,
            hole_count=len(part["poly"].interiors)
        )
        
        # Create metadata
        metadata = ObjectMetadata(
            layer_name="0",
            color=1,
            line_type="CONTINUOUS"
        )
        
        obj = DXFObject(
            object_id=part["id"],
            entity=None,  # We don't have the original entity
            object_type=ObjectType.POLYGON,
            complexity=ObjectComplexity.SIMPLE if geom.vertex_count < 10 else ObjectComplexity.MODERATE,
            geometry=geom,
            metadata=metadata
        )
        dxf_objects.append(obj)
    
    # Create a CuttingLayer for nesting
    from wjp_analyser.object_management.layer_manager import CuttingLayer, LayerType, MaterialSettings, NestingSettings
    
    # Create material and nesting settings
    material_settings = MaterialSettings(
        material_name="Steel",
        thickness=1.0,
        width=width,
        height=height,
        density=7.85,
        cost_per_kg=2.0
    )
    
    nesting_settings = NestingSettings(
        algorithm="bottom_left",
        rotation_enabled=True,
        allowed_rotations=[0, 90, 180, 270]
    )
    
    # Create cutting layer
    layer = CuttingLayer(
        layer_id="nesting_layer",
        name="Nesting Layer",
        layer_type=LayerType.NESTED,
        description="Layer for nesting optimization",
        objects=dxf_objects,
        material_settings=material_settings,
        nesting_settings=nesting_settings
    )
    
    # Perform nesting optimization
    result = engine.optimize_nesting(layer)
    
    # Extract positioned objects
    out_entities = []
    placed_ids = set()
    for pos_obj in result.positioned_objects:
        if pos_obj.is_positioned:
            placed_ids.add(pos_obj.object.id)
            # Get the geometry coordinates
            coords = list(pos_obj.geometry.exterior.coords[:-1])
            out_entities.append(coords)

    # Use absolute path for output
    input_path = Path(input_dxf_path)
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "output" / "nesting"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}_nested.dxf"
    
    # Write new DXF file
    new_doc = ezdxf.new('R2010')
    msp_new = new_doc.modelspace()
    
    for coords in out_entities:
        if len(coords) >= 3:
            msp_new.add_lwpolyline(coords)
    
    new_doc.saveas(str(out_path))

    # Calculate utilization
    total_area = sum(p["area"] for p in parts if p["id"] in placed_ids)
    sheet_area = width * height
    util = total_area / sheet_area if sheet_area > 0 else 0
    
    # Calculate total cut length
    total_len = sum(p["poly"].length for p in parts if p["id"] in placed_ids)
    
    return out_path, util, total_len, result.algorithm_used, out_entities, len(parts)


def _render_nest_preview(out_entities, sheet_w: float, sheet_h: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    # sheet boundary
    ax.plot([0, sheet_w, sheet_w, 0, 0], [0, 0, sheet_h, sheet_h, 0], 'k--', linewidth=1.0)
    # parts
    for poly in out_entities:
        if not poly:
            continue
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, 'b-', linewidth=1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, sheet_w])
    ax.set_ylim([0, sheet_h])
    ax.set_title('Nested Layout Preview')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


tab_upload, tab_path = st.tabs(["Upload file", "Use local path"])

dxf_path = None
with tab_upload:
    up = st.file_uploader("Upload DXF", type=["dxf"], key="nest_upload")
    if up is not None:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        dxf_path = str(upload_dir / up.name)
        Path(dxf_path).write_bytes(up.getvalue())
        st.success(f"Saved to {dxf_path}")

with tab_path:
    p = st.text_input("DXF file path", value=dxf_path or "")
    if p:
        dxf_path = p

with st.sidebar.expander("Sheet", expanded=True):
    width = st.number_input("Width (mm)", min_value=10.0, value=600.0)
    height = st.number_input("Height (mm)", min_value=10.0, value=600.0)

with st.sidebar.expander("Parameters", expanded=False):
    kerf = st.number_input("Kerf (mm)", min_value=0.0, value=1.1, step=0.1)
    gap = st.number_input("Min gap (mm)", min_value=0.0, value=3.0, step=0.5)
    polygon_mode = st.checkbox("Polygon nesting", True)

if st.button("Nest", type="primary", disabled=not bool(dxf_path)):
    try:
        out_dxf, util, total_len, strategy, out_entities, parts_count = do_nest(dxf_path, width, height, gap, polygon_mode=polygon_mode)
        st.success("Nesting complete")
        st.caption(f"Parts detected: {parts_count}")
        st.metric("Utilization", f"{util*100:.1f}%")
        st.metric("Total cut length (mm)", f"{total_len:.0f}")
        st.caption(f"Strategy: {strategy}")
        st.write(f"Output DXF: {out_dxf}")
        # Inline preview
        try:
            fig = _render_nest_preview(out_entities, width, height)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        except Exception as pe:
            st.info(f"Preview not available: {pe}")
        # Quick handoff to Analyze DXF
        c_prev, c_link = st.columns([1, 1])
        if c_prev.button("Preview in Analyzer"):
            st.session_state["last_output_dxf"] = str(out_dxf)
            st.success("Nested DXF set. Open Analyze DXF page.")
        try:
            c_link.page_link("src/wjp_analyser/web/pages/analyze_dxf.py", label="Open Analyze DXF", page_icon="🔎")
        except Exception:
            pass
        try:
            out_bytes = Path(out_dxf).read_bytes()
            st.download_button("Download nested.dxf", data=out_bytes, file_name=Path(out_dxf).name)
        except Exception:
            st.info("Unable to read nested DXF for download; open from disk.")
    except Exception as e:
        st.error(f"Nesting failed: {e}")
