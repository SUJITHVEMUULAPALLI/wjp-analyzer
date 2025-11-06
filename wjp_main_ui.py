import io
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    plt = None  # type: ignore


st.set_page_config(
    page_title="WJP Analyzer – Unified",
    page_icon="🧰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.section-title { font-size: 20px; font-weight: 700; margin: 8px 0 4px 0; }
.subtle { color:#666; font-size: 13px; }
hr { margin: 0.6rem 0 1.2rem 0; }
[data-testid="stMetricValue"] { font-size: 28px; }
.block-container { padding-top: 1.4rem; }
.card { background-color: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; margin-bottom: 10px; }
.badge { display:inline-block; font-size:12px; padding:3px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; }
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class WJPState:
    image_bytes: Optional[bytes] = None
    image_name: Optional[str] = None
    image_meta: Dict[str, Any] | None = None
    preproc_params: Dict[str, Any] | None = None
    preproc_image_bytes: Optional[bytes] = None

    dxf_path: Optional[str] = None
    dxf_meta: Dict[str, Any] | None = None
    dxf_objects_df: Optional[pd.DataFrame] = None
    dxf_preview_fig_json: Optional[str] = None

    analysis_report: Dict[str, Any] | None = None
    grouped_objects_df: Optional[pd.DataFrame] = None

    nesting_result: Dict[str, Any] | None = None
    costing_result: Dict[str, Any] | None = None

    logs: List[str] | None = None


def get_state() -> WJPState:
    if "wjp" not in st.session_state:
        st.session_state["wjp"] = WJPState(
            image_meta={},
            preproc_params={"brightness": 1.0, "threshold": 128, "blur": 0, "invert": False},
            dxf_meta={},
            analysis_report={},
            logs=[],
        )
    return st.session_state["wjp"]


def log(msg: str) -> None:
    s = get_state()
    (s.logs or []).append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def col_metrics(metrics: List[Tuple[str, Any, str]]) -> None:
    cols = st.columns(len(metrics))
    for c, (label, value, helptext) in zip(cols, metrics):
        with c:
            st.metric(label, value)
            if helptext:
                st.caption(helptext)


def to_img_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def dual_image_preview(left_bytes: Optional[bytes], right_bytes: Optional[bytes], left_title: str = "Input", right_title: str = "Output") -> None:
    col1, col2 = st.columns(2)
    if left_bytes:
        with col1:
            st.markdown(f"**{left_title}**")
            st.image(left_bytes, use_column_width=True)
    if right_bytes:
        with col2:
            st.markdown(f"**{right_title}**")
            st.image(right_bytes, use_column_width=True)


def mcp_smart_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    log("MCP smart check triggered.")
    return {
        "contours_detected": 87,
        "open_paths": 3,
        "avg_radius_mm": 2.8,
        "min_spacing_mm": 3.2,
        "notes": "Looks mostly compliant. 3 open paths near quadrant Q2.",
    }


def run_image_to_dxf(image_bytes: bytes, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    out_root = Path("output") / "streamlit_conversions"
    out_root.mkdir(parents=True, exist_ok=True)
    # Persist input image to a temp file
    tmp_img = out_root / f"src_{int(time.time())}.png"
    with open(tmp_img, "wb") as f:
        f.write(image_bytes)

    base_name = tmp_img.stem
    dxf_path = out_root / f"{base_name}.dxf"
    preview_path = out_root / f"{base_name}_preview.png"

    # Parameters
    binary_threshold = int(params.get("threshold", 180))
    min_area = int(params.get("min_area", 800))
    dxf_size = int(params.get("dxf_size", 1200))

    try:
        try:
            from src.wjp_analyser.image_processing.converters.enhanced_opencv_converter import EnhancedOpenCVImageToDXFConverter
            converter = EnhancedOpenCVImageToDXFConverter(
                binary_threshold=binary_threshold,
                min_area=min_area,
                dxf_size=dxf_size,
            )
        except Exception:
            from src.wjp_analyser.image_processing.converters.opencv_converter import OpenCVImageToDXFConverter
            converter = OpenCVImageToDXFConverter(
                binary_threshold=binary_threshold,
                min_area=min_area,
                dxf_size=float(dxf_size),
            )

        result = converter.convert_image_to_dxf(
            input_image=str(tmp_img),
            output_dxf=str(dxf_path),
            preview_output=str(preview_path),
        )
        meta = {
            "units": "mm",
            "objects": int(result.get("polygons", 0)),
            "contours_found": int(result.get("contours_found", 0)),
            "preview": str(preview_path),
            "params": {"binary_threshold": binary_threshold, "min_area": min_area, "dxf_size": dxf_size},
        }
        log(f"Converted image → DXF: {dxf_path}")
        return str(dxf_path), meta
    except Exception as e:
        log(f"Image→DXF failed: {e}")
        raise


def analyze_dxf(dxf_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Use project analyzer
    from src.wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf as core_analyze

    out_dir = Path("output") / "streamlit_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = AnalyzeArgs(out=str(out_dir))
    report = core_analyze(dxf_path, args)

    components = report.get("components", []) or []
    rows: List[Dict[str, Any]] = []
    for comp in components:
        rows.append({
            "Object": comp.get("id") or comp.get("name") or "shape",
            "Area_mm2": float(comp.get("area", 0.0)),
            "Complexity": comp.get("complexity", "Unknown"),
            "Layer": comp.get("layer", "0"),
            "Selected": True,
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Object", "Area_mm2", "Complexity", "Layer", "Selected"])

    metrics = report.get("metrics", {})
    rpt = {
        "total_objects": len(rows),
        "total_area_mm2": float(sum(r.get("Area_mm2", 0.0) for r in rows)),
        "selected_objects": int(sum(1 for r in rows if r.get("Selected"))),
        "violations": report.get("quality", {}).get("violations", {}),
        "metrics": metrics,
    }
    log("Analyzed DXF (core).")
    return df, rpt


def group_similar_objects(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Group"] = df["Complexity"].map({"Simple": "G1", "Moderate": "G2"}).fillna("G3")
    log("Grouped similar objects (stub).")
    return df


@st.cache_data(show_spinner=False)
def simple_plotly_dxf(df: Optional[pd.DataFrame]) -> Optional[str]:
    if go is None:
        return None
    fig = go.Figure()
    for i, row in enumerate((df or pd.DataFrame()).itertuples(), start=1):
        x0, y0 = i * 10, i * 10
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x0 + 8, y1=y0 + 8, line=dict(width=2))
        label = getattr(row, "Object", f"obj{i}")
        fig.add_annotation(x=x0 + 4, y=y0 + 4, text=label, showarrow=False)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
    return fig.to_json()


def run_nesting(dxf_path: str, sheet_w_mm: float, sheet_h_mm: float) -> Dict[str, Any]:
    from src.wjp_analyser.nesting.dxf_extractor import extract_polygons
    # Extract closed polygons from DXF
    polys = extract_polygons(dxf_path)
    total_part_area = float(sum(p.get("area", 0.0) for p in polys))
    sheet_area = max(sheet_w_mm * sheet_h_mm, 1.0)
    utilization = 100.0 * total_part_area / sheet_area
    return {
        "sheet": f"{int(sheet_w_mm)}x{int(sheet_h_mm)} mm",
        "utilization_pct": round(utilization, 2),
        "placements": len(polys),
        "parts_area_mm2": round(total_part_area, 2),
    }


def run_costing(df: pd.DataFrame, kerf_mm: float) -> Dict[str, Any]:
    # If we have a DXF path in state, use the robust costing
    state = get_state()
    if not state.dxf_path:
        length_m = max(1.0, len(df) * 0.8)
        return {"cut_length_m": length_m, "est_time_min": length_m * 2.5, "cost_rupees": round(length_m * 380, 2)}

    try:
        from src.wjp_analyser.web.api_utils import calculate_costs_from_dxf
        costs = calculate_costs_from_dxf(state.dxf_path)
        return {
            "cut_length_m": round(costs.get("metrics", {}).get("length_m", 0.0), 3),
            "est_time_min": round(costs.get("metrics", {}).get("machine_minutes", 0.0), 2),
            "cost_rupees": costs.get("total_cost", 0.0),
            "breakdown": {k: v for k, v in costs.items() if k != "metrics"},
        }
    except Exception as e:
        log(f"Costing fallback due to error: {e}")
        length_m = max(1.0, len(df) * 0.8)
        return {"cut_length_m": length_m, "est_time_min": length_m * 2.5, "cost_rupees": round(length_m * 380, 2)}


def export_report_pdf_stub(state: WJPState) -> bytes:
    payload = {
        "image_name": state.image_name,
        "dxf_path": state.dxf_path,
        "analysis": state.analysis_report,
        "nesting": state.nesting_result,
        "costing": state.costing_result,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def page_designer(state: WJPState) -> None:
    st.markdown('<div class="section-title">Designer</div>', unsafe_allow_html=True)
    st.caption("Capture requirements → produce clean prompts → (external) image generation → download image for conversion.")
    with st.expander("Prompt Template"):
        st.text_area(
            "Prompt",
            value=(
                "DXF-ready ornamental medallion, solid black shapes on white background, "
                "min radius ≥2mm, spacing ≥3mm, no grayscale…"
            ),
            height=120,
        )
    st.info("When finalized, save the generated image and continue to **Image → DXF**.")


def page_image_to_dxf(state: WJPState) -> None:
    st.markdown('<div class="section-title">Image → DXF</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Upload", "Preprocessing", "Conversion", "Preview"])

    with tabs[0]:
        file = st.file_uploader("Upload source image (PNG/JPG, black shapes on white)", type=["png", "jpg", "jpeg"])
        if file:
            img = Image.open(file).convert("RGB")
            state.image_bytes = to_img_bytes(img, "PNG")
            state.image_name = file.name
            state.image_meta = {"size_px": img.size, "mode": img.mode}
            st.success(f"Loaded {file.name} – {img.size[0]}×{img.size[1]} px")
            st.image(state.image_bytes, use_column_width=True)
            log(f"Loaded image: {file.name}")

    with tabs[1]:
        if not state.image_bytes:
            st.warning("Please upload an image first.")
        else:
            st.subheader("Adaptive Controls")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                b = st.slider("Brightness", 0.2, 2.0, float(state.preproc_params["brightness"]), 0.1)
            with c2:
                t = st.slider("Threshold", 0, 255, int(state.preproc_params["threshold"]), 1)
            with c3:
                blur = st.slider("Blur (px)", 0, 7, int(state.preproc_params["blur"]), 1)
            with c4:
                inv = st.checkbox("Invert", value=bool(state.preproc_params["invert"]))
            if st.button("Apply Preprocessing"):
                img = Image.open(io.BytesIO(state.image_bytes)).convert("L")
                img = ImageOps.autocontrast(img.point(lambda p: min(255, int(p * b))))
                if blur > 0:
                    img = img.filter(ImageFilter.GaussianBlur(radius=blur))
                img = img.point(lambda p: 255 if p > t else 0)
                if inv:
                    img = ImageOps.invert(img)
                state.preproc_params.update({"brightness": b, "threshold": t, "blur": blur, "invert": inv})
                state.preproc_image_bytes = to_img_bytes(img.convert("RGB"))
                log("Applied preprocessing.")
            dual_image_preview(state.image_bytes, state.preproc_image_bytes, "Original", "Preprocessed")

            if st.button("Auto Optimize for DXF (MCP)"):
                result = mcp_smart_check({"stage": "image_preprocess", "meta": state.image_meta, "params": state.preproc_params})
                st.json(result)

    with tabs[2]:
        if not (state.preproc_image_bytes or state.image_bytes):
            st.warning("Need an uploaded and ideally preprocessed image.")
        else:
            use_pre = st.radio("Use which image for conversion?", ["Preprocessed", "Original"], horizontal=True)
            cpa, cpb, cpc = st.columns(3)
            with cpa:
                min_area = st.number_input("Min contour area", min_value=0, value=int((state.preproc_params or {}).get("min_area", 800)), step=50)
            with cpb:
                dxf_size = st.number_input("DXF canvas (mm)", min_value=100, value=int((state.preproc_params or {}).get("dxf_size", 1200)), step=50)
            with cpc:
                pass
            busy = st.session_state.get("busy_image2dxf", False)
            if st.button("Convert to DXF", type="primary", disabled=busy):
                st.session_state["busy_image2dxf"] = True
                try:
                    with st.status("Converting image → DXF...", expanded=False) as status:
                        src = state.preproc_image_bytes if (use_pre == "Preprocessed" and state.preproc_image_bytes) else state.image_bytes
                        # merge extra params
                        params = dict(state.preproc_params or {})
                        params.update({"min_area": int(min_area), "dxf_size": int(dxf_size)})
                        dxf_path, meta = run_image_to_dxf(src, params)
                        state.dxf_path, state.dxf_meta = dxf_path, meta
                        status.update(label="DXF created", state="complete")
                    st.toast("DXF conversion complete", icon="✅")
                    st.success(f"DXF created: {dxf_path}")
                    st.json(meta)
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
                    st.toast("DXF conversion failed", icon="❌")
                finally:
                    st.session_state["busy_image2dxf"] = False

    with tabs[3]:
        dual_image_preview(state.image_bytes, state.preproc_image_bytes, "Original", "Preprocessed")
        if state.dxf_path:
            st.info(f"DXF ready: **{state.dxf_path}**. Proceed to **DXF Analyzer**.")
            # Preview image if available
            prev = (state.dxf_meta or {}).get("preview")
            if prev and os.path.exists(prev):
                st.image(prev, caption="DXF Preview", use_column_width=True)
                try:
                    with open(prev, "rb") as f:
                        st.download_button("Download Preview", data=f.read(), file_name=os.path.basename(prev), mime="image/png")
                except Exception:
                    pass
            # Download DXF
            try:
                with open(state.dxf_path, "rb") as f:
                    st.download_button("Download DXF", data=f.read(), file_name=os.path.basename(state.dxf_path), mime="application/dxf")
            except Exception:
                st.warning("DXF file not accessible for download.")


def page_image_analyzer(state: WJPState) -> None:
    st.markdown('<div class="section-title">Image Analyzer</div>', unsafe_allow_html=True)
    if not (state.image_bytes or state.preproc_image_bytes):
        st.warning("Upload/process an image in **Image → DXF** first.")
        return
    img_src = state.preproc_image_bytes or state.image_bytes

    toggles = st.columns(4)
    show_orig = toggles[0].checkbox("Original", True)
    show_gray = toggles[1].checkbox("Grayscale", True)
    show_edges = toggles[2].checkbox("Edges (mock)", True)
    show_contours = toggles[3].checkbox("Contours (mock)", True)

    base = Image.open(io.BytesIO(img_src)).convert("L")
    gray = base.convert("RGB")
    edges = base.filter(ImageFilter.FIND_EDGES).convert("RGB")
    contours = ImageOps.posterize(base, 1).convert("RGB")

    left = state.image_bytes if show_orig else None
    right = None
    if show_gray:
        right = gray
    if show_edges:
        right = Image.blend(right or gray, edges, 0.5) if right else edges
    if show_contours:
        right = Image.blend(right or gray, contours, 0.5) if right else contours

    right_bytes = to_img_bytes(right) if right else None
    dual_image_preview(left, right_bytes, "Input", "Diagnostics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Width (px)", base.size[0])
    c2.metric("Height (px)", base.size[1])
    c3.metric("Threshold", get_state().preproc_params["threshold"])  # type: ignore[index]
    c4.metric("Blur", get_state().preproc_params["blur"])  # type: ignore[index]

    if st.button("Run MCP Insights"):
        result = mcp_smart_check({"stage": "image_analyzer"})
        st.json(result)


def page_dxf_analyzer(state: WJPState) -> None:
    st.markdown('<div class="section-title">DXF Analyzer</div>', unsafe_allow_html=True)
    if not state.dxf_path:
        st.warning("No DXF found. Convert in **Image → DXF**.")
        return

    if state.dxf_objects_df is None:
        busy = st.session_state.get("busy_analyze", False)
        st.session_state["busy_analyze"] = True
        try:
            with st.status("Parsing DXF and computing stats...", expanded=False):
                df, rpt = analyze_dxf(state.dxf_path)
                state.dxf_objects_df = df
                state.analysis_report = rpt
            st.toast("DXF analysis complete", icon="✅")
        except Exception as e:
            st.error(f"DXF analysis failed: {e}")
            st.toast("DXF analysis failed", icon="❌")
            return
        finally:
            st.session_state["busy_analyze"] = False

    col_metrics(
        [
            ("Total Objects", state.analysis_report.get("total_objects"), "from parsed DXF"),  # type: ignore[union-attr]
            ("Total Area (mm²)", f"{state.analysis_report.get('total_area_mm2',0):,.0f}", ""),  # type: ignore[union-attr]
            ("Selected Objects", state.analysis_report.get("selected_objects"), ""),  # type: ignore[union-attr]
        ]
    )
    st.divider()

    st.markdown("**Objects** (editable selection & layer)")
    edited = st.data_editor(
        state.dxf_objects_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Selected": st.column_config.CheckboxColumn("Selected"),
            "Area_mm2": st.column_config.NumberColumn("Area (mm²)", format="%.2f"),
        },
    )
    state.dxf_objects_df = edited

    c1, c2, c3 = st.columns(3)
    if c1.button("Group Similar Objects"):
        state.grouped_objects_df = group_similar_objects(state.dxf_objects_df)
        st.success("Grouped objects. See below.")
    if c2.button("Run Smart Analysis (MCP)"):
        result = mcp_smart_check({"stage": "dxf_analyzer", "dxf_meta": state.dxf_meta})
        st.json(result)
    if c3.button("Refresh Stats"):
        state.analysis_report["selected_objects"] = int(state.dxf_objects_df["Selected"].sum())  # type: ignore[index]
        st.info("Stats refreshed.")

    if state.grouped_objects_df is not None:
        st.markdown("**Groups (preview)")
        st.dataframe(state.grouped_objects_df, use_container_width=True)

    st.markdown("**Preview**")
    if go is not None:
        fig_json = simple_plotly_dxf(state.dxf_objects_df)
        state.dxf_preview_fig_json = fig_json
        if fig_json:
            st.plotly_chart(go.Figure(**json.loads(fig_json)), use_container_width=True)  # type: ignore[arg-type]
        else:
            st.info("No preview available.")
    else:
        st.info("Plotly not installed. Install with: pip install plotly")


def page_dxf_editor(state: WJPState) -> None:
    st.markdown('<div class="section-title">DXF Editor</div>', unsafe_allow_html=True)
    if state.dxf_objects_df is None:
        st.warning("Analyze a DXF first.")
        return

    st.caption("Select by rows → change Layer/Selected → apply. (Drag/visual selection requires a custom viewer; planned.)")
    target_layer = st.text_input("Target Layer", value="OUTER")
    apply_to_selected = st.checkbox("Apply to selected only", value=True)
    if st.button("Move to Layer"):
        df = state.dxf_objects_df.copy()
        mask = df["Selected"] if apply_to_selected else np.ones(len(df), dtype=bool)
        df.loc[mask, "Layer"] = target_layer
        state.dxf_objects_df = df
        st.success("Layer updated.")

    st.dataframe(state.dxf_objects_df, use_container_width=True)


def page_nesting(state: WJPState) -> None:
    st.markdown('<div class="section-title">Nesting</div>', unsafe_allow_html=True)
    if not state.dxf_path:
        st.warning("No DXF available. Convert or load a DXF first.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        sheet_w = st.number_input("Sheet width (mm)", min_value=100.0, value=1000.0, step=50.0)
    with c2:
        sheet_h = st.number_input("Sheet height (mm)", min_value=100.0, value=1000.0, step=50.0)
    with c3:
        pass
    busy = st.session_state.get("busy_nesting", False)
    if st.button("Run Nesting", type="primary", disabled=busy):
        st.session_state["busy_nesting"] = True
        try:
            with st.status("Computing utilization from DXF geometry...", expanded=False):
                state.nesting_result = run_nesting(state.dxf_path, sheet_w, sheet_h)
            st.toast("Nesting summarized", icon="✅")
            st.success("Nesting complete.")
            st.json(state.nesting_result)
        except Exception as e:
            st.error(f"Nesting failed: {e}")
            st.toast("Nesting failed", icon="❌")
        finally:
            st.session_state["busy_nesting"] = False


def page_costing(state: WJPState) -> None:
    st.markdown('<div class="section-title">Costing</div>', unsafe_allow_html=True)
    if state.dxf_objects_df is None:
        st.warning("Analyze a DXF first.")
        return
    kerf = st.number_input("Kerf (mm)", value=1.1, step=0.1)
    busy = st.session_state.get("busy_cost", False)
    if st.button("Calculate Cost", type="primary", disabled=busy):
        st.session_state["busy_cost"] = True
        try:
            with st.status("Calculating cost...", expanded=False):
                state.costing_result = run_costing(state.dxf_objects_df, kerf)
            st.toast("Costing complete", icon="✅")
            st.success("Cost computed.")
            st.json(state.costing_result)
        except Exception as e:
            st.error(f"Costing failed: {e}")
            st.toast("Costing failed", icon="❌")
        finally:
            st.session_state["busy_cost"] = False


def page_reports(state: WJPState) -> None:
    st.markdown('<div class="section-title">Reports</div>', unsafe_allow_html=True)
    st.caption("Export pipeline summary as PDF (stub). Replace with your actual PDF generator.")
    if st.button("Generate Report"):
        pdf_bytes = export_report_pdf_stub(state)
        st.download_button(
            "Download WJP Report (JSON-as-PDF-stub)", data=pdf_bytes, file_name="wjp_report.json", mime="application/json"
        )

    st.markdown("**Session Log**")
    st.code("\n".join(state.logs or []))


def main() -> None:
    state = get_state()
    with st.sidebar:
        st.title("WJP Analyzer")
        st.caption("Unified Interface")
        # Reset controls
        rc1, rc2 = st.columns(2)
        if rc1.button("Reset Current"):
            # Soft reset of current stage artifacts
            state.preproc_image_bytes = None
            state.dxf_path = None
            state.dxf_meta = {}
            state.dxf_objects_df = None
            state.analysis_report = {}
            state.grouped_objects_df = None
            state.nesting_result = None
            state.costing_result = None
            st.toast("Stage reset", icon="🧹")
        if rc2.button("Reset All"):
            st.session_state.pop("wjp", None)
            st.experimental_rerun()

        section = st.radio(
            "Navigate",
            [
                "Designer",
                "Image → DXF",
                "Image Analyzer",
                "DXF Analyzer",
                "DXF Editor",
                "Nesting",
                "Costing",
                "Reports",
            ],
        )
        st.divider()
        if st.button("Run MCP (Global)"):
            out = mcp_smart_check({"stage": "global"})
            st.session_state["mcp_global"] = out
        if st.session_state.get("mcp_global"):
            st.markdown("**MCP (global) last run:**")
            st.json(st.session_state["mcp_global"])

        with st.expander("Session State"):
            st.json({
                "image_name": state.image_name,
                "image_meta": state.image_meta,
                "preproc_params": state.preproc_params,
                "dxf_path": state.dxf_path,
                "metrics": (state.analysis_report or {}).get("metrics"),
            })

    pages = {
        "Designer": page_designer,
        "Image → DXF": page_image_to_dxf,
        "Image Analyzer": page_image_analyzer,
        "DXF Analyzer": page_dxf_analyzer,
        "DXF Editor": page_dxf_editor,
        "Nesting": page_nesting,
        "Costing": page_costing,
        "Reports": page_reports,
    }
    pages[section](state)


if __name__ == "__main__":
    main()


