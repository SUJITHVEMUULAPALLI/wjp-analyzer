import streamlit as st
from dxf_tools.io_utils import load_dxf, save_dxf
from dxf_tools.visualize import plot_entities
from dxf_tools.transform_utils import translate, scale, rotate
from dxf_tools.selection import pick_entity
from dxf_tools.layers import get_layers, ensure_layer, rename_layer, recolor_layer, move_entities_to_layer
from dxf_tools.groups import create_group, list_groups, get_group
from dxf_tools.draw import add_line, add_circle, add_rect, add_polyline
from dxf_tools.measure import distance, bbox_of_entity, bbox_size, polyline_length
from dxf_tools.validate import check_min_radius, kerf_preview_value
from dxf_tools.repair import close_small_gaps, remove_duplicates
from dxf_tools.session import load_session, save_session

st.set_page_config(page_title="WJP DXF Editor v2", layout="wide")
st.title("🧭 WJP DXF Editor — v2 (Selection • Layers • Groups • Draw • Measure)")

uploaded = st.file_uploader("Upload DXF File", type=["dxf"])

if "state" not in st.session_state:
    st.session_state["state"] = {
        "selected": [],  # list of handles
        "hidden_layers": [],
        "session_path": "session.json"
    }

if uploaded:
    doc = load_dxf(uploaded)
    msp = doc.modelspace()
    entities = [e for e in msp.query("LINE CIRCLE LWPOLYLINE")]
    layers = get_layers(doc)

    # Sidebar — Layers / Groups / Selection
    with st.sidebar:
        st.header("Layers")
        layer_names = list(layers.keys())
        st.write(f"Layers: {layer_names}")
        new_layer = st.text_input("Create Layer", value="WJP_NEW")
        if st.button("Add Layer"):
            ensure_layer(doc, new_layer, color=7)
            st.success(f"Layer '{new_layer}' ensured.")
        layer_pick = st.selectbox("Pick Layer", options=layer_names if layer_names else ["0"])
        new_name = st.text_input("Rename Layer To", value=layer_pick)
        if st.button("Rename Layer"):
            rename_layer(doc, layer_pick, new_name)
            st.success(f"Renamed layer {layer_pick} → {new_name}")
        new_color = st.number_input("ACI Color (1-255)", value=7, min_value=1, max_value=255, step=1)
        if st.button("Recolor Layer"):
            recolor_layer(doc, layer_pick, new_color)
            st.success(f"Layer {layer_pick} recolored to {new_color}")

        st.subheader("Layer Visibility")
        vis_multi = st.multiselect("Hidden Layers", options=list(layers.keys()), default=st.session_state["state"]["hidden_layers"])
        st.session_state["state"]["hidden_layers"] = vis_multi

        st.header("Groups")
        st.write("Existing:", list_groups(doc))
        grp_name = st.text_input("New Group Name", value="group_1")
        if st.button("Group Selection"):
            sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
            create_group(doc, grp_name, [e.dxf.handle for e in sel])
            st.success(f"Created group '{grp_name}' with {len(sel)} entities")

        st.header("Selection")
        st.write("Selected handles:", st.session_state["state"]["selected"])
        x = st.number_input("Pick X", value=0.0)
        y = st.number_input("Pick Y", value=0.0)
        tol = st.number_input("Pick Tolerance", value=2.0)
        if st.button("Pick @ (X,Y)"):
            picked = pick_entity(entities, x, y, tol)
            if picked:
                if picked.dxf.handle not in st.session_state["state"]["selected"]:
                    st.session_state["state"]["selected"].append(picked.dxf.handle)
                st.success(f"Selected {picked.dxftype()} on layer {picked.dxf.layer} handle {picked.dxf.handle}")
            else:
                st.warning("No entity near that point.")

        if st.button("Clear Selection"):
            st.session_state["state"]["selected"] = []

        st.subheader("Move Selected to Layer")
        move_to = st.text_input("Target Layer", value="OUTER")
        if st.button("Move to Layer"):
            target = ensure_layer(doc, move_to, color=7)
            sel = [e for e in entities if e.dxf.handle in st.session_state["state"]["selected"]]
            move_entities_to_layer(sel, move_to)
            st.success(f"Moved {len(sel)} entities to layer '{move_to}'")

        st.header("Session")
        if st.button("Save Session State"):
            save_session(st.session_state["state"]["session_path"], st.session_state["state"])
            st.success("Session saved.")
        if st.button("Load Session State"):
            st.session_state["state"] = load_session(st.session_state["state"]["session_path"])
            st.success("Session loaded.")

    # Canvas
    fig = plot_entities(entities,
                        selected_handles=st.session_state["state"]["selected"],
                        hidden_layers=st.session_state["state"]["hidden_layers"],
                        color_by_layer=True)
    st.pyplot(fig, use_container_width=True)

    # Transform panel
    st.subheader("Transforms")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dx = st.number_input("ΔX (mm)", value=0.0)
        dy = st.number_input("ΔY (mm)", value=0.0)
        if st.button("Translate Selected"):
            for e in entities:
                if e.dxf.handle in st.session_state["state"]["selected"]:
                    translate(e, dx, dy)
            st.success("Translated.")
    with c2:
        s = st.number_input("Scale Factor", value=1.0)
        if st.button("Scale Selected"):
            for e in entities:
                if e.dxf.handle in st.session_state["state"]["selected"]:
                    scale(e, s)
            st.success("Scaled.")
    with c3:
        ang = st.number_input("Rotate (deg)", value=0.0)
        if st.button("Rotate Selected"):
            for e in entities:
                if e.dxf.handle in st.session_state["state"]["selected"]:
                    rotate(e, ang)
            st.success("Rotated.")
    with c4:
        if st.button("Delete Selected"):
            count = 0
            for e in list(entities):
                if e.dxf.handle in st.session_state["state"]["selected"]:
                    try:
                        msp.delete_entity(e)
                        count += 1
                    except Exception:
                        pass
            st.session_state["state"]["selected"] = []
            st.success(f"Deleted {count} entities.")

    # Draw tools
    st.subheader("Draw")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        x1 = st.number_input("Line x1", value=0.0); y1 = st.number_input("Line y1", value=0.0)
        x2 = st.number_input("Line x2", value=10.0); y2 = st.number_input("Line y2", value=0.0)
        if st.button("Add Line"):
            add_line(msp, x1, y1, x2, y2, layer="WJP_DRAW")
            st.success("Line added.")
    with d2:
        cx = st.number_input("Circle cx", value=0.0); cy = st.number_input("Circle cy", value=0.0)
        r = st.number_input("Circle r", value=5.0)
        if st.button("Add Circle"):
            add_circle(msp, cx, cy, r, layer="WJP_DRAW")
            st.success("Circle added.")
    with d3:
        rx = st.number_input("Rect x", value=0.0); ry = st.number_input("Rect y", value=0.0)
        rw = st.number_input("Rect w", value=10.0); rh = st.number_input("Rect h", value=6.0)
        if st.button("Add Rect"):
            add_rect(msp, rx, ry, rw, rh, layer="WJP_DRAW")
            st.success("Rect added.")
    with d4:
        poly = st.text_area("Polyline points (x,y per line)", value="0,0\n20,0\n20,10\n0,10")
        closed = st.checkbox("Closed", value=True)
        if st.button("Add Polyline"):
            pts = []
            for line in poly.strip().splitlines():
                try:
                    x,y = line.split(",")
                    pts.append((float(x), float(y)))
                except Exception:
                    pass
            add_polyline(msp, pts, closed=closed, layer="WJP_DRAW")
            st.success(f"Polyline added with {len(pts)} points.")

    # Measure/Validate
    st.subheader("Measure / Validate")
    m1, m2, m3 = st.columns(3)
    with m1:
        a = st.text_input("Point A (x,y)", value="0,0")
        b = st.text_input("Point B (x,y)", value="10,0")
        if st.button("Distance A→B"):
            try:
                ax, ay = [float(v) for v in a.split(",")]
                bx, by = [float(v) for v in b.split(",")]
                st.info(f"Distance: {distance((ax,ay),(bx,by)):.3f} mm")
            except Exception:
                st.error("Invalid input")
    with m2:
        kerf = st.number_input("Kerf preview (mm)", value=1.1)
        st.write(f"Kerf value: {kerf_preview_value(kerf)} mm")
    with m3:
        # Minimal radius check for selected circle entities
        if st.button("Check Min Radius (selected)"):
            issues = []
            for e in entities:
                if e.dxf.handle in st.session_state["state"]["selected"]:
                    ok, meta = check_min_radius(e, min_r=2.0)
                    if not ok:
                        issues.append((e.dxf.handle, meta))
            if issues:
                st.warning(f"Violations: {issues}")
            else:
                st.success("No radius violations (or not applicable).")

    # Save / Export
    st.subheader("Save / Export")
    out_path = st.text_input("Save as", value="edited_output_v2.dxf")
    if st.button("💾 Save DXF"):
        save_dxf(doc, out_path)
        with open(out_path, "rb") as f:
            st.download_button("Download Edited DXF", data=f, file_name=out_path)

    st.subheader("Analyzer Handoff (stub)")
    if st.button("Send to Analyzer"):
        payload = {
            "filename": getattr(uploaded, "name", "uploaded.dxf"),
            "selected": st.session_state["state"]["selected"],
            "hidden_layers": st.session_state["state"]["hidden_layers"],
            "kerf": kerf if "kerf" in locals() else 1.1
        }
        st.json(payload)
        st.success("Payload prepared (hook to your Analyzer function or HTTP endpoint).")
