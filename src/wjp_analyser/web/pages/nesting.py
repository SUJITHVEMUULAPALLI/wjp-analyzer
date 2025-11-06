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
    # Import nesting utilities from the v2 scaffold
    import sys, os
    repo_root = Path(__file__).resolve().parents[4]  # pages -> web -> wjp_analyser -> src -> repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from wjpanalyser.app.services import dxf_io, geometry, metrics
    from wjpanalyser.app.services.nesting.arrange import arrange

    doc, msp = dxf_io.read_modelspace(input_dxf_path)
    parts = []
    for e in msp.query("LWPOLYLINE"):
        pts = [(v[0], v[1]) for v in e.get_points("xy")]
        poly = geometry.polyline_to_polygon(pts)
        if poly.is_empty:
            continue
        parts.append({
            "id": e.dxf.handle,
            "area": float(poly.area),
            "bbox": tuple(poly.bounds),
            "poly": poly,
            "points": pts,
        })
    placed, strategy = arrange(parts, width, height, gap, polygon_mode=polygon_mode)
    out_entities = []
    placed_ids = set()
    for pid, ang, pose in placed:
        if pose is None:
            continue
        placed_ids.add(pid)
        xs, ys = pose.exterior.xy
        out_entities.append(list(zip(xs, ys)))

    # Use absolute path for output
    input_path = Path(input_dxf_path)
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "output" / "nesting"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}_nested.dxf"
    dxf_io.write_new_dxf(out_entities, str(out_path))

    util = metrics.utilization([p for p in parts if p["id"] in placed_ids], width, height)
    total_len = sum(geometry.length_of_polygon_edges(p["poly"]) for p in parts if p["id"] in placed_ids)
    return out_path, util, total_len, strategy, out_entities, len(parts)


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
            st.pyplot(fig, clear_figure=True, width="stretch")
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
