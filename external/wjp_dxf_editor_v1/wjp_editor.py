import streamlit as st
from dxf_tools.io_utils import load_dxf, save_dxf
from dxf_tools.transform_utils import translate, scale, rotate
from dxf_tools.visualize import plot_entities

st.set_page_config(page_title="WJP DXF Editor", layout="wide")
st.title("🧭 WJP DXF Editor (Phase 1 Prototype)")

uploaded = st.file_uploader("Upload DXF File", type=["dxf"])

if uploaded:
    doc = load_dxf(uploaded)
    msp = doc.modelspace()
    entities = [e for e in msp.query("LINE CIRCLE LWPOLYLINE")]

    st.write(f"Total Entities: {len(entities)}")
    fig = plot_entities(entities)
    st.pyplot(fig)

    st.subheader("Edit Transformations")
    col1, col2, col3 = st.columns(3)
    with col1:
        dx = st.number_input("Move X (mm)", value=0.0)
        dy = st.number_input("Move Y (mm)", value=0.0)
        if st.button("Translate"):
            for e in entities: translate(e, dx, dy)
            st.success("Translation applied.")

    with col2:
        factor = st.number_input("Scale Factor", value=1.0)
        if st.button("Scale"):
            for e in entities: scale(e, factor)
            st.success("Scaling applied.")

    with col3:
        angle = st.number_input("Rotate (°)", value=0.0)
        if st.button("Rotate"):
            for e in entities: rotate(e, angle)
            st.success("Rotation applied.")

    if st.button("💾 Save Edited DXF"):
        out_path = "edited_output.dxf"
        save_dxf(doc, out_path)
        st.download_button("Download Edited DXF", open(out_path, "rb"), file_name="edited_output.dxf")
