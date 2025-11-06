import argparse
import os
import sys

from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf
from wjp_analyser.gcode.gcode_workflow import run_gcode_workflow


def main():
	parser = argparse.ArgumentParser(description="G-code workflow: generate artifacts from analyzed DXF components")
	parser.add_argument("dxf", help="Path to DXF file")
	parser.add_argument("--out", default="out", help="Output directory")
	parser.add_argument("--material", default="Generic Material")
	parser.add_argument("--thickness", type=float, default=10.0)
	parser.add_argument("--kerf", type=float, default=1.0)
	parser.add_argument("--rate-per-m", type=float, default=800.0)
	parser.add_argument("--pierce-cost", type=float, default=5.0)
	args_ns = parser.parse_args()

	dxf_path = os.path.abspath(args_ns.dxf)
	if not os.path.exists(dxf_path):
		print(f"DXF not found: {dxf_path}", file=sys.stderr)
		sys.exit(2)

	args = AnalyzeArgs(
		material=args_ns.material,
		thickness=args_ns.thickness,
		kerf=args_ns.kerf,
		rate_per_m=args_ns["rate_per_m"] if isinstance(args_ns, dict) else args_ns.rate_per_m,
		pierce_cost=args_ns.pierce_cost,
		out=args_ns.out,
	)

	analysis = analyze_dxf(dxf_path, args)
	components = analysis.get("components", [])
	if not components:
		print("No components found in DXF.", file=sys.stderr)
		sys.exit(1)

	result = run_gcode_workflow(components, args)
	art = result.get("artifacts", {})
	print("Artifacts:")
	for k in ("layered_dxf", "lengths_csv", "report_json"):
		if art.get(k):
			print(f"  {k}: {art[k]}")

	print("Metrics:")
	for k, v in (result.get("metrics") or {}).items():
		print(f"  {k}: {v}")

	return 0


if __name__ == "__main__":
	sys.exit(main())


