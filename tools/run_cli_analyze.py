import os
import sys
import json
from pathlib import Path
import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description="Run DXF analyzer with optional softening/fillet/normalization")
    ap.add_argument("input", help="Input DXF path")
    ap.add_argument("output", nargs="?", help="Output directory (default: output/<name>)")
    ap.add_argument("--soften", choices=["none", "simplify", "chaikin"], default="none")
    ap.add_argument("--tol", type=float, default=0.1, help="Simplify tolerance (mm) for soften=simplify")
    ap.add_argument("--iterations", type=int, default=1, help="Chaikin iterations for soften=chaikin")
    ap.add_argument("--fillet-radius", type=float, default=0.5, help="Fillet radius (mm)")
    ap.add_argument("--fillet-min-angle", type=float, default=135.0, help="Min interior angle to fillet (deg)")
    ap.add_argument("--fit", action="store_true", help="Normalize to frame before nesting")
    ap.add_argument("--frame", nargs=2, type=float, default=[1000.0, 1000.0], metavar=("W","H"))
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()

    input_dxf = Path(args.input).resolve()
    if not input_dxf.exists():
        print(f"Input not found: {input_dxf}")
        return 3

    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = Path.cwd() / "output" / input_dxf.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure project src is on path
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

    from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf

    cfg = AnalyzeArgs(out=str(out_dir))
    # Soften
    if args.soften != "none":
        cfg.soften_method = args.soften
        if args.soften == "simplify":
            cfg.soften_tolerance = float(args.tol)
        elif args.soften == "chaikin":
            cfg.soften_iterations = int(args.iterations)
    # Fillet
    if float(args.fillet_radius) > 0:
        cfg.fillet_radius_mm = float(args.fillet_radius)
        cfg.fillet_min_angle_deg = float(args.fillet_min_angle)
    # Fit to frame
    if args.fit:
        cfg.normalize_mode = "fit"
        cfg.target_frame_w_mm = float(args.frame[0])
        cfg.target_frame_h_mm = float(args.frame[1])
        cfg.frame_margin_mm = float(args.margin)
        cfg.normalize_origin = True
        cfg.require_fit_within_frame = True

    report = analyze_dxf(str(input_dxf), cfg)

    report_path = out_dir / "report.json"
    print("Report:", report_path)
    try:
        data = json.loads(Path(report_path).read_text(encoding="utf-8"))
        mc = data.get("mastery_checklist") or {}
        print("Checklist summary:")
        print(json.dumps({
            "Entities": mc.get("Entities"),
            "OpenPolylines": mc.get("OpenPolylines"),
            "TinySegments": mc.get("TinySegments"),
            "Duplicates": mc.get("Duplicates"),
            "TotalLength_mm": mc.get("TotalLength_mm"),
            "Pierces": mc.get("Pierces"),
            "Warnings": mc.get("Warnings"),
        }, indent=2))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
