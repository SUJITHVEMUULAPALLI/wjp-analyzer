import argparse, os
from rich import print
from pydantic import BaseModel
from waterjet_dxf.io_dxf import load_dxf_as_lines
from waterjet_dxf.geom_clean import merge_and_polygonize
from waterjet_dxf.topology import containment_depth
from waterjet_dxf.classify import classify_by_depth
from waterjet_dxf.checks import check_open_contours, check_min_spacing, check_acute_vertices
from waterjet_dxf.metrics import compute_lengths
from waterjet_dxf.toolpath import plan_order, kerf_preview
from waterjet_dxf.gcode import write_gcode
from waterjet_dxf.report import save_json, save_lengths_csv
from waterjet_dxf.viz import preview_png
from waterjet_dxf.image_processor import ImageProcessor

class Args(BaseModel):
    material: str = "Granite"
    thickness: float = 25.0
    kerf: float = 1.1
    rate_per_m: float = 825.0
    out: str

def analyze(dxf_path: str, a: Args):
    os.makedirs(a.out, exist_ok=True)
    lines = load_dxf_as_lines(dxf_path)
    merged_lines, polys = merge_and_polygonize(lines)
    depths = containment_depth(polys)
    classes = classify_by_depth(depths)

    # checks
    issues = []
    issues += check_open_contours(merged_lines)
    issues += check_min_spacing(polys, limit_mm=3.0)
    issues += check_acute_vertices(polys, limit_deg=30.0)

    # metrics
    m = compute_lengths(polys, classes)
    est_time_min = (m["length_internal_mm"]/1200.0) + (0.5 * m["pierces"])  # crude
    cost_inr = (m["length_internal_mm"]/1000.0) * a.rate_per_m

    order = plan_order(polys, classes)
    kerf_polys = kerf_preview(polys, kerf_mm=a.kerf)

    # report
    rep = {
        "file": os.path.basename(dxf_path),
        "material": {"name": a.material, "thickness_mm": a.thickness},
        "kerf_mm": a.kerf,
        "metrics": {
            **m,
            "est_time_min": round(est_time_min,2),
            "cost_inr": round(cost_inr,0),
        },
        "violations": [x for x in issues if x["type"]!="acute_angle"],
        "warnings": [x for x in issues if x["type"]=="acute_angle"],
        "toolpath": {"order": order},
    }
    save_json(os.path.join(a.out, "report.json"), rep)

    # lengths csv
    rows = []
    for i,p in enumerate(polys):
        rows.append({"id": i, "class": classes.get(i), "perimeter_mm": round(p.exterior.length,2), "area_mm2": round(p.area,2)})
    save_lengths_csv(os.path.join(a.out, "lengths.csv"), rows)

    # viz
    preview_png(os.path.join(a.out, "preview.png"), polys, classes, order, kerf_polys)

    print("[green]Analyze complete[/green] ->", a.out)

def command_analyze(args):
    a = Args(material=args.material, thickness=args.thickness, kerf=args.kerf, rate_per_m=args.rate_per_m, out=args.out)
    analyze(args.dxf, a)

def command_gcode(args):
    aout = args.out
    os.makedirs(aout, exist_ok=True)
    from waterjet_dxf.io_dxf import load_dxf_as_lines
    from waterjet_dxf.geom_clean import merge_and_polygonize
    from waterjet_dxf.topology import containment_depth
    from waterjet_dxf.classify import classify_by_depth
    from waterjet_dxf.toolpath import plan_order
    lines = load_dxf_as_lines(args.dxf)
    _, polys = merge_and_polygonize(lines)
    depths = containment_depth(polys)
    classes = classify_by_depth(depths)
    order = plan_order(polys, classes)
    write_gcode(os.path.join(aout, "program.nc"), polys, order, feed=args.feed, m_on=args.m_on, m_off=args.m_off, pierce_ms=args.pierce_ms)
    print("[green]G-code written[/green] ->", os.path.join(aout, "program.nc"))

def command_image(args):
    """Convert image to DXF"""
    aout = args.out
    os.makedirs(aout, exist_ok=True)
    
    # Create image processor with custom parameters
    processor = ImageProcessor(
        edge_threshold=args.edge_threshold,
        min_contour_area=args.min_area,
        simplify_tolerance=args.simplify,
        blur_kernel_size=args.blur,
    )
    
    # Generate output DXF path
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    output_dxf = os.path.join(aout, f"{image_name}_converted.dxf")
    
    # Process image to DXF
    contour_count = processor.process_image_to_dxf(
        args.image,
        output_dxf,
        scale_factor=args.scale,
        debug_dir=aout,
    )
    
    print(f"[green]Image converted[/green] -> {output_dxf}")
    print(f"[blue]Contours extracted:[/blue] {contour_count}")
    
    # Optionally analyze the generated DXF
    if contour_count > 0:
        print("[yellow]Running analysis on generated DXF...[/yellow]")
        try:
            a = Args(material="Granite", thickness=25.0, kerf=1.1, rate_per_m=825.0, out=aout)
            analyze(output_dxf, a)
            print("[green]Analysis complete[/green] ->", aout)
        except Exception as e:
            print(f"[red]Analysis failed:[/red] {str(e)}")

def main():
    p = argparse.ArgumentParser(prog="wjdx", description="Waterjet DXF analyzer")
    sub = p.add_subparsers()

    pa = sub.add_parser("analyze", help="Validate, analyze, visualize, report")
    pa.add_argument("dxf")
    pa.add_argument("--material", default="Granite")
    pa.add_argument("--thickness", type=float, default=25.0)
    pa.add_argument("--kerf", type=float, default=1.1)
    pa.add_argument("--rate-per-m", type=float, default=825.0)
    pa.add_argument("--out", default="out")
    pa.set_defaults(func=command_analyze)

    pg = sub.add_parser("gcode", help="Generate toy G-code")
    pg.add_argument("dxf")
    pg.add_argument("--feed", type=float, default=1200.0)
    pg.add_argument("--m-on", default="M62")
    pg.add_argument("--m-off", default="M63")
    pg.add_argument("--pierce-ms", type=int, default=500)
    pg.add_argument("--out", default="out")
    pg.set_defaults(func=command_gcode)

    # Image processing commands
    pi = sub.add_parser("image", help="Convert image (JPG/JPEG) to DXF")
    pi.add_argument("image")
    pi.add_argument("--out", default="out")
    pi.add_argument("--scale", type=float, default=1.0, help="Scale factor for output DXF")
    pi.add_argument("--edge-threshold", type=float, default=0.33, help="Sigma factor (0-1) for adaptive Canny thresholds")
    pi.add_argument("--min-area", type=int, default=100, help="Minimum contour area in pixels")
    pi.add_argument("--simplify", type=float, default=0.02, help="Douglas-Peucker simplification tolerance (ratio <1 or pixels)")
    pi.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (odd integer)")
    pi.set_defaults(func=command_image)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
