import os
import io
import sys
import streamlit as st

# Ensure project src is on sys.path when running this page directly
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..', '..', '..'))
if _SRC_DIR not in sys.path:
	sys.path.insert(0, _SRC_DIR)

from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs
from wjp_analyser.gcode.gcode_workflow import run_gcode_workflow
# Use API wrapper for analysis
from wjp_analyser.web.api_client_wrapper import analyze_dxf


def _save_uploaded_file(uploaded_file, save_dir: str) -> str:
	os.makedirs(save_dir, exist_ok=True)
	path = os.path.join(save_dir, uploaded_file.name)
	with open(path, "wb") as f:
		f.write(uploaded_file.getbuffer())
	return path


def main():
	st.set_page_config(page_title="G-code Workflow", page_icon="🧰", layout="wide")
	st.title("G-code Workflow")
	st.caption("Generate manufacturing artifacts from analyzed DXF components.")

	with st.sidebar:
		st.header("Input & Settings")
		upload = st.file_uploader("Upload DXF", type=["dxf"]) 
		out_dir = st.text_input("Output directory", value="out")
		material = st.text_input("Material", value="Generic Material")
		thickness = st.number_input("Thickness (mm)", value=10.0)
		kerf = st.number_input("Kerf (mm)", value=1.0)
		rate_per_m = st.number_input("Rate per meter", value=800.0)
		pierce_cost = st.number_input("Pierce cost", value=5.0)
		go = st.button("Run Workflow", type="primary", use_container_width=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.subheader("Analysis → Components")
		analysis_placeholder = st.empty()

	with col2:
		st.subheader("Artifacts")
		downloads_placeholder = st.empty()

	if go:
		if not upload:
			st.error("Please upload a DXF file.")
			st.stop()

		# Save uploaded DXF to output folder for reproducibility
		dxf_path = _save_uploaded_file(upload, out_dir)

		# Analyze to get components using API wrapper
		# Note: API wrapper uses args_overrides, but we need AnalyzeArgs for gcode_workflow
		# So we'll call analyze_dxf directly and convert result
		analysis = analyze_dxf(
			dxf_path,
			out_dir=out_dir,
			args_overrides={
				"material": material,
				"thickness": float(thickness),
				"kerf": float(kerf),
				"rate_per_m": float(rate_per_m),
				"pierce_cost": float(pierce_cost),
			}
		)
		
		# Create AnalyzeArgs for gcode_workflow (still needed for run_gcode_workflow)
		args = AnalyzeArgs(
			material=material,
			thickness=float(thickness),
			kerf=float(kerf),
			rate_per_m=float(rate_per_m),
			pierce_cost=float(pierce_cost),
			out=out_dir,
		)
		components = analysis.get("components", [])
		if not components:
			analysis_placeholder.warning("No components detected in DXF.")
			st.stop()

		analysis_placeholder.success(
			f"Detected {len(components)} components. Proceeding to manufacturing artifacts..."
		)

		# Run G-code workflow to produce artifacts and metrics
		manufacturing = run_gcode_workflow(components, args)

		art = manufacturing.get("artifacts", {})
		metrics = manufacturing.get("metrics", {})

		with downloads_placeholder.container():
			st.write("Metrics:")
			st.json(metrics)

			def _make_download(label, path):
				if os.path.exists(path):
					with open(path, "rb") as fh:
						st.download_button(label=label, data=fh.read(), file_name=os.path.basename(path))

			_make_download("Download layered DXF", art.get("layered_dxf", ""))
			_make_download("Download lengths CSV", art.get("lengths_csv", ""))
			_make_download("Download report JSON", art.get("report_json", ""))


if __name__ == "__main__":
	main()
