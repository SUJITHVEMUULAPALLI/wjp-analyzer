from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st


st.set_page_config(page_title="WJP Supervisor Dashboard", layout="wide")
st.title("Waterjet Supervisor Dashboard")
st.caption("Designer → Image2DXF → Analyze (single-screen pipeline)")
st.markdown("---")


# ------------------------------
# Sidebar: Input + Params
# ------------------------------
st.sidebar.header("Pipeline Control")
user_input = st.sidebar.text_area(
    "Design requirements (optional)",
    placeholder="e.g., Tan Brown granite medallion with Jaisalmer yellow inlay",
)

base_out = Path(st.sidebar.text_input("Output base folder", value="output/supervisor")).resolve()
base_out.mkdir(parents=True, exist_ok=True)

img_file = st.sidebar.file_uploader(
    "Upload image (optional)", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="sup_img"
)
img_path_text = st.sidebar.text_input("Or image file path", value="")

st.sidebar.subheader("Texture Vectorizer")
mode = st.sidebar.selectbox("Mode", ["Auto Mix", "Edges", "Stipple", "Hatch", "Contour"], index=1)
kerf_width = st.sidebar.number_input("Kerf width (mm)", min_value=0.0, max_value=5.0, value=1.1, step=0.1)
kerf_mode = st.sidebar.selectbox(
    "Kerf compensation",
    ["None", "Outward (+kerf/2)", "Inward (-kerf/2)", "Inside/Outside (+/- kerf/2)"],
    index=3,
)
simplify_tol = st.sidebar.number_input("Simplify tolerance (mm)", min_value=0.0, max_value=5.0, value=0.2, step=0.05)
min_feat_size = st.sidebar.number_input("Min feature size (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
min_feat_area = st.sidebar.number_input("Min feature area (mm^2)", min_value=0.1, max_value=200.0, value=4.0, step=0.1)

run_pipeline = st.sidebar.button("Run Full Pipeline", type="primary")


# ------------------------------
# Helpers
# ------------------------------
def _save_uploaded_image(file, out_dir: Path) -> Optional[Path]:
    if not file:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / file.name
    p.write_bytes(file.getvalue())
    return p


def _build_stub_image(out_dir: Path) -> Path:
    from base64 import b64decode

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "designer_stub.png"
    if not p.exists():
        p.write_bytes(
            b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII=")
        )
    return p


def _run_texture_pipeline(image_path: str, out_dir: Path):
    from wjp_analyser.image_processing.texture_pipeline import (
        generate_texture_dxf,
        PreprocessParams,
        TextureClassifyParams,
        TextureVectorizeParams,
    )

    m = mode.lower().split()[0]
    vec = TextureVectorizeParams(
        mode=m if m in ("edges", "stipple", "hatch", "contour") else "auto",
        dxf_size_mm=1000.0,
        min_feature_size_mm=float(min_feat_size),
        min_feature_area_mm2=float(min_feat_area),
        simplify_tol_mm=float(simplify_tol),
        kerf_mm=float(kerf_width),
        kerf_offset_mm=(
            0.5 * float(kerf_width)
            if kerf_mode.startswith("Out")
            else (-0.5 * float(kerf_width) if kerf_mode.startswith("Inw") else 0.0)
        ),
        kerf_inout=bool(kerf_mode.startswith("Inside/Outside")),
    )
    dxf_path, preview_png = generate_texture_dxf(
        image_path=str(image_path),
        out_dir=str(out_dir),
        preprocess_params=PreprocessParams(working_px=1000),
        classify_params=TextureClassifyParams(tile=32, clusters=4),
        vec_params=vec,
    )
    return Path(dxf_path), Path(preview_png)


def _analyze_dxf(dxf_path: Path, out_dir: Path) -> dict:
    from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf

    args = AnalyzeArgs(out=str(out_dir), sheet_width=1000.0, sheet_height=1000.0)
    report = analyze_dxf(str(dxf_path), args)
    return report


def _quick_plot_dxf(dxf_path: Path):
    try:
        import ezdxf  # type: ignore
        import numpy as np
        import matplotlib.pyplot as plt

        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        fig, ax = plt.subplots(figsize=(6, 6))
        for e in msp:
            try:
                t = e.dxftype()
                if t == "LWPOLYLINE":
                    pts = [(v[0], v[1]) for v in e.get_points("xy")]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    ax.plot(xs + xs[:1], ys + ys[:1], "k-", linewidth=0.6, alpha=0.7)
                elif t == "LINE":
                    ax.plot([e.dxf.start.x, e.dxf.end.x], [e.dxf.start.y, e.dxf.end.y], "k-", linewidth=0.6, alpha=0.7)
                elif t == "CIRCLE":
                    cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
                    t = np.linspace(0, 2 * np.pi, 100)
                    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), "k-", linewidth=0.6, alpha=0.7)
                elif t == "ARC":
                    cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
                    a1, a2 = np.radians(e.dxf.start_angle), np.radians(e.dxf.end_angle)
                    t = np.linspace(a1, a2, 60)
                    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), "k-", linewidth=0.6, alpha=0.7)
            except Exception:
                continue
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(os.path.basename(str(dxf_path)))
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        st.pyplot(fig, clear_figure=True, width="stretch")
    except Exception as e:
        st.info(f"Preview skipped: {e}")


# ------------------------------
# Pipeline execution
# ------------------------------
image_source_path: Optional[Path] = None
if run_pipeline:
    st.info("Running pipeline...")
    try:
        img_dir = base_out / "images"
        if img_file:
            image_source_path = _save_uploaded_image(img_file, img_dir)
        elif img_path_text:
            image_source_path = Path(img_path_text)
        else:
            # Use stub if nothing provided
            image_source_path = _build_stub_image(img_dir)

        if not image_source_path or not image_source_path.exists():
            st.error("No image available. Upload one or provide a valid path.")
        else:
            tex_out = base_out / "texture"
            dxf_path, preview_png = _run_texture_pipeline(str(image_source_path), tex_out)
            st.success("Texture vectorization complete")
            st.image(str(preview_png), caption="Texture Preview", width="stretch")

            ana_out = base_out / "analysis"
            report = _analyze_dxf(dxf_path, ana_out)
            st.success("DXF analysis complete")

            # Quick stats
            cols = st.columns(3)
            m = report.get("metrics", {})
            cols[0].metric("Cut length (mm)", f"{m.get('length_internal_mm', 0):.0f}")
            cols[1].metric("Pierces", m.get("pierces", 0))
            cols[2].metric("Est. cost", f"{m.get('estimated_cutting_cost_inr', 0):.0f}")

            st.subheader("DXF Preview")
            _quick_plot_dxf(dxf_path)

            st.subheader("Downloads")
            with open(dxf_path, "rb") as fh:
                st.download_button("Download DXF", data=fh.read(), file_name=dxf_path.name)
            rep_json = Path(report.get("artifacts", {}).get("report_json", ""))
            if rep_json and rep_json.exists():
                with open(rep_json, "rb") as fh:
                    st.download_button("Download Report JSON", data=fh.read(), file_name=rep_json.name)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")

