import argparse
import os
import sys
from rich import print
from pydantic import BaseModel

# Ensure local package is importable when running from source tree
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import from reorganized modules
from wjp_analyser.io.dxf_io import load_dxf_as_lines, load_dxf_lines_with_layers
from wjp_analyser.analysis.geometry_cleaner import merge_and_polygonize
from wjp_analyser.analysis.topology import containment_depth
from wjp_analyser.analysis.classification import classify_by_depth, classify_by_depth_and_layers
from wjp_analyser.analysis.quality_checks import check_open_contours, check_min_spacing, check_acute_vertices
from wjp_analyser.manufacturing.cost_calculator import compute_lengths
from wjp_analyser.manufacturing.toolpath import (
    plan_order, kerf_preview, 
    plan_advanced_toolpath, generate_optimized_gcode,
    ToolpathOptimization, CuttingPath
)
from wjp_analyser.manufacturing.cam_processor import AdvancedCAMProcessor, CAMSettings
from wjp_analyser.manufacturing.path_optimizer import WaterjetPathOptimizer, optimize_dxf_with_visualization
from wjp_analyser.manufacturing.gcode_generator import write_gcode
from wjp_analyser.io.report_generator import save_json, save_lengths_csv
from wjp_analyser.io.visualization import preview_png
from wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs as APIArgs, analyze_dxf as api_analyze_dxf, load_polys_and_classes
from wjp_analyser.image_processing.image_processor import ImageProcessor
from wjp_analyser.ai.openai_client import OpenAIAnalyzer, OpenAIConfig, ManufacturingAnalysis, load_analysis_data_from_report, save_ai_analysis
from wjp_analyser.ai.ollama_client import OllamaAnalyzer, OllamaConfig
from wjp_analyser.config.preset_loader import PresetLoader, list_presets

class Args(BaseModel):
    material: str = "Granite"
    thickness: float = 25.0
    kerf: float = 1.1
    rate_per_m: float = 825.0
    out: str
    # Advanced toolpath options
    use_advanced_toolpath: bool = True
    rapid_speed: float = 10000.0
    cutting_speed: float = 1200.0
    pierce_time: float = 0.5
    optimize_rapids: bool = True
    optimize_direction: bool = True
    entry_strategy: str = "tangent"

def analyze(dxf_path: str, a: Args, selected_groups=None):
    api_args = APIArgs(
        material=a.material, 
        thickness=a.thickness, 
        kerf=a.kerf, 
        rate_per_m=a.rate_per_m, 
        out=a.out,
        use_advanced_toolpath=a.use_advanced_toolpath,
        rapid_speed=a.rapid_speed,
        cutting_speed=a.cutting_speed,
        pierce_time=a.pierce_time,
        optimize_rapids=a.optimize_rapids,
        optimize_direction=a.optimize_direction,
        entry_strategy=a.entry_strategy
    )
    report = api_analyze_dxf(dxf_path, api_args, selected_groups=selected_groups)
    print("[green]Analyze complete[/green] ->", a.out)
    if report.get("groups"):
        print("[cyan]Similarity groups:[/cyan]")
        for name, meta in report["groups"].items():
            count = meta.get("count", 0)
            vcount = meta.get("vcount")
            avg_area = meta.get("avg_area", 0.0)
            avg_circ = meta.get("avg_circ", 0.0)
            complexity = meta.get("complexity", "")
            print(
                f"  • {name}: count={count}, vertices={vcount}, "
                f"avg_area={avg_area:.2f} mm², avg_circ={avg_circ:.3f}, complexity={complexity}"
            )
    return report

def command_analyze(args):
    a = Args(
        material=args.material, 
        thickness=args.thickness, 
        kerf=args.kerf, 
        rate_per_m=args.rate_per_m, 
        out=args.out,
        use_advanced_toolpath=args.use_advanced_toolpath,
        rapid_speed=args.rapid_speed,
        cutting_speed=args.cutting_speed,
        pierce_time=args.pierce_time,
        optimize_rapids=args.optimize_rapids,
        optimize_direction=args.optimize_direction,
        entry_strategy=args.entry_strategy
    )
    selected = None
    if getattr(args, "select_groups", None):
        selected = [s.strip() for s in str(args.select_groups).split(",") if s.strip()]
    analyze(args.dxf, a, selected_groups=selected)

def command_gcode(args):
    aout = args.out
    os.makedirs(aout, exist_ok=True)
    polys, classes, order = load_polys_and_classes(args.dxf)
    write_gcode(os.path.join(aout, "program.nc"), polys, order, feed=args.feed, m_on=args.m_on, m_off=args.m_off, pierce_ms=args.pierce_ms)
    print("[green]G-code written[/green] ->", os.path.join(aout, "program.nc"))

def command_image(args):
    print("[red]Image-to-DXF CLI has been removed.")

def command_opencv(args):
    print("[red]Image-to-DXF CLI has been removed.")

def command_inkscape(args):
    print("[red]Image-to-DXF CLI has been removed.")

def command_openai_analyze(args):
    """Analyze DXF with OpenAI for manufacturing insights"""
    try:
        # Load OpenAI configuration
        config = OpenAIConfig(
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Initialize analyzer
        analyzer = OpenAIAnalyzer(config)
        
        # Load analysis data from report if available
        report_path = os.path.join(args.out, "report.json")
        analysis_data = {}
        
        if os.path.exists(report_path):
            analysis_data = load_analysis_data_from_report(report_path)
            print(f"[blue]Loaded existing analysis data from:[/blue] {report_path}")
        else:
            print("[yellow]No existing report found. Running basic analysis first...[/yellow]")
            # Run basic analysis first
            a = Args(material=args.material, thickness=args.thickness, kerf=args.kerf, rate_per_m=args.rate_per_m, out=args.out)
            analyze(args.dxf, a)
            analysis_data = load_analysis_data_from_report(report_path)
        
        # Run OpenAI analysis
        print("[blue]Running OpenAI manufacturing analysis...[/blue]")
        ai_analysis = analyzer.analyze_dxf_manufacturing(args.dxf, analysis_data)
        
        if ai_analysis:
            # Save AI analysis
            ai_output_path = os.path.join(args.out, "ai_analysis.json")
            save_ai_analysis(ai_analysis, ai_output_path)
            
            # Display results
            print("\n[bold green]AI Manufacturing Analysis Results:[/bold green]")
            print(f"[bold]Feasibility Score:[/bold] {ai_analysis.feasibility_score}/100")
            print(f"[bold]Complexity Level:[/bold] {ai_analysis.complexity_level}")
            print(f"[bold]Estimated Time:[/bold] {ai_analysis.estimated_time}")
            
            print(f"\n[bold]Material Recommendations:[/bold]")
            for rec in ai_analysis.material_recommendations:
                print(f"  - {rec}")
            
            print(f"\n[bold]Toolpath Suggestions:[/bold]")
            for suggestion in ai_analysis.toolpath_suggestions:
                print(f"  - {suggestion}")
            
            print(f"\n[bold]Potential Issues:[/bold]")
            for issue in ai_analysis.potential_issues:
                print(f"  WARNING: {issue}")
            
            print(f"\n[bold]Optimization Tips:[/bold]")
            for tip in ai_analysis.optimization_tips:
                print(f"  TIP: {tip}")
            
            print(f"\n[bold]Cost Considerations:[/bold]")
            for consideration in ai_analysis.cost_considerations:
                print(f"  COST: {consideration}")
                
        else:
            print("[red]Failed to get AI analysis results[/red]")
            
    except Exception as e:
        print(f"[red]OpenAI analysis failed:[/red] {str(e)}")

def command_openai_design(args):
    """Generate design suggestions using OpenAI"""
    try:
        # Load OpenAI configuration
        config = OpenAIConfig(
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Initialize analyzer
        analyzer = OpenAIAnalyzer(config)
        
        # Generate design suggestions
        print("[blue]Generating design suggestions with OpenAI...[/blue]")
        suggestions = analyzer.generate_design_suggestions(args.description)
        
        if suggestions:
            print("\n[bold green]AI Design Suggestions:[/bold green]")
            print(suggestions)
            
            # Save suggestions to file
            suggestions_path = os.path.join(args.out, "design_suggestions.txt")
            os.makedirs(args.out, exist_ok=True)
            with open(suggestions_path, 'w') as f:
                f.write(f"Design Description: {args.description}\n\n")
                f.write("AI Design Suggestions:\n")
                f.write("=" * 50 + "\n\n")
                f.write(suggestions)
            
            print(f"\n[green]Suggestions saved to:[/green] {suggestions_path}")
        else:
            print("[red]Failed to generate design suggestions[/red]")
            
    except Exception as e:
        print(f"[red]Design suggestion generation failed:[/red] {str(e)}")

def command_ollama_analyze(args):
    """Analyze DXF with Ollama for manufacturing insights"""
    try:
        # Load Ollama configuration
        config = OllamaConfig(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout
        )
        
        # Initialize analyzer
        analyzer = OllamaAnalyzer(config)
        
        # Load analysis data from report if available
        report_path = os.path.join(args.out, "report.json")
        analysis_data = {}
        
        if os.path.exists(report_path):
            analysis_data = load_analysis_data_from_report(report_path)
            print(f"[blue]Loaded existing analysis data from:[/blue] {report_path}")
        else:
            print("[yellow]No existing report found. Running basic analysis first...[/yellow]")
            # Run basic analysis first
            a = Args(material=args.material, thickness=args.thickness, kerf=args.kerf, rate_per_m=args.rate_per_m, out=args.out)
            analyze(args.dxf, a)
            analysis_data = load_analysis_data_from_report(report_path)
        
        # Run Ollama analysis
        print("[blue]Running Ollama manufacturing analysis...[/blue]")
        ai_analysis = analyzer.analyze_dxf_manufacturing(args.dxf, analysis_data)
        
        if ai_analysis:
            # Save AI analysis
            ai_output_path = os.path.join(args.out, "ollama_analysis.json")
            save_ai_analysis(ai_analysis, ai_output_path)
            
            # Display results
            print("\n[bold green]Ollama Manufacturing Analysis Results:[/bold green]")
            print(f"[bold]Feasibility Score:[/bold] {ai_analysis.feasibility_score}/100")
            print(f"[bold]Complexity Level:[/bold] {ai_analysis.complexity_level}")
            print(f"[bold]Estimated Time:[/bold] {ai_analysis.estimated_time}")
            
            print(f"\n[bold]Material Recommendations:[/bold]")
            for rec in ai_analysis.material_recommendations:
                print(f"  - {rec}")
            
            print(f"\n[bold]Toolpath Suggestions:[/bold]")
            for suggestion in ai_analysis.toolpath_suggestions:
                print(f"  - {suggestion}")
            
            print(f"\n[bold]Potential Issues:[/bold]")
            for issue in ai_analysis.potential_issues:
                print(f"  WARNING: {issue}")
            
            print(f"\n[bold]Optimization Tips:[/bold]")
            for tip in ai_analysis.optimization_tips:
                print(f"  TIP: {tip}")
            
            print(f"\n[bold]Cost Considerations:[/bold]")
            for consideration in ai_analysis.cost_considerations:
                print(f"  COST: {consideration}")
                
        else:
            print("[red]Failed to get Ollama analysis results[/red]")
            
    except Exception as e:
        print(f"[red]Ollama analysis failed:[/red] {str(e)}")

def command_ollama_design(args):
    """Generate design suggestions using Ollama"""
    try:
        # Load Ollama configuration
        config = OllamaConfig(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout
        )
        
        # Initialize analyzer
        analyzer = OllamaAnalyzer(config)
        
        # Generate design suggestions
        print("[blue]Generating design suggestions with Ollama...[/blue]")
        suggestions = analyzer.generate_design_suggestions(args.description)
        
        if suggestions:
            print("\n[bold green]Ollama Design Suggestions:[/bold green]")
            print(suggestions)
            
            # Save suggestions to file
            suggestions_path = os.path.join(args.out, "ollama_design_suggestions.txt")
            os.makedirs(args.out, exist_ok=True)
            with open(suggestions_path, 'w') as f:
                f.write(f"Design Description: {args.description}\n\n")
                f.write("Ollama Design Suggestions:\n")
                f.write("=" * 50 + "\n\n")
                f.write(suggestions)
            
            print(f"\n[green]Suggestions saved to:[/green] {suggestions_path}")
        else:
            print("[red]Failed to generate design suggestions[/red]")
            
    except Exception as e:
        print(f"[red]Design suggestion generation failed:[/red] {str(e)}")

def command_advanced_toolpath(args):
    """Generate advanced optimized toolpath"""
    aout = args.out
    os.makedirs(aout, exist_ok=True)
    
    # Load polygons and classifications
    polys, classes, _ = load_polys_and_classes(args.dxf)
    
    # Create optimization parameters
    optimization = ToolpathOptimization(
        kerf_compensation=args.kerf,
        rapid_speed=args.rapid_speed,
        cutting_speed=args.cutting_speed,
        pierce_time=args.pierce_time,
        min_rapid_distance=args.min_rapid_distance,
        optimize_rapids=args.optimize_rapids,
        optimize_direction=args.optimize_direction,
        entry_strategy=args.entry_strategy
    )
    
    # Generate optimized cutting paths
    cutting_paths = plan_advanced_toolpath(polys, classes, optimization)
    
    # Generate optimized G-code
    gcode_lines = generate_optimized_gcode(polys, classes, optimization)
    
    # Save G-code
    gcode_path = os.path.join(aout, "optimized_program.nc")
    with open(gcode_path, 'w') as f:
        f.write('\n'.join(gcode_lines))
    
    # Save toolpath analysis
    analysis_path = os.path.join(aout, "toolpath_analysis.json")
    analysis_data = {
        "total_paths": len(cutting_paths),
        "total_cutting_length": sum(path.cutting_length for path in cutting_paths),
        "total_rapid_distance": sum(path.rapid_distance for path in cutting_paths),
        "optimization_settings": {
            "kerf_compensation": optimization.kerf_compensation,
            "rapid_speed": optimization.rapid_speed,
            "cutting_speed": optimization.cutting_speed,
            "pierce_time": optimization.pierce_time,
            "min_rapid_distance": optimization.min_rapid_distance,
            "optimize_rapids": optimization.optimize_rapids,
            "optimize_direction": optimization.optimize_direction,
            "entry_strategy": optimization.entry_strategy
        },
        "cutting_paths": [
            {
                "polygon_index": path.polygon_index,
                "start_point": path.start_point,
                "end_point": path.end_point,
                "cutting_direction": path.cutting_direction,
                "entry_angle": path.entry_angle,
                "exit_angle": path.exit_angle,
                "rapid_distance": path.rapid_distance,
                "cutting_length": path.cutting_length
            }
            for path in cutting_paths
        ]
    }
    
    import json
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print("[green]Advanced toolpath generated[/green]")
    print(f"  G-code: {gcode_path}")
    print(f"  Analysis: {analysis_path}")
    print(f"  Total paths: {len(cutting_paths)}")
    print(f"  Total cutting length: {sum(path.cutting_length for path in cutting_paths):.1f} mm")
    print(f"  Total rapid distance: {sum(path.rapid_distance for path in cutting_paths):.1f} mm")

def command_cam_process(args):
    """Process DXF with advanced CAM features"""
    aout = args.out
    os.makedirs(aout, exist_ok=True)
    
    # Create CAM settings
    cam_settings = CAMSettings(
        simplify_tolerance=args.simplify_tolerance,
        join_curves=args.join_curves,
        use_arc_interpolation=args.use_arc_interpolation,
        arc_tolerance=args.arc_tolerance,
        min_arc_radius=args.min_arc_radius,
        max_arc_radius=args.max_arc_radius,
        holes_first=args.holes_first,
        reduce_pierces=args.reduce_pierces,
        join_tolerance=args.join_tolerance,
        rapid_speed=args.rapid_speed,
        cutting_speed=args.cutting_speed,
        pierce_time=args.pierce_time
    )
    
    # Process DXF with advanced CAM
    processor = AdvancedCAMProcessor(cam_settings)
    contours = processor.process_dxf(args.dxf)
    
    # Generate optimized G-code
    gcode_lines = processor.generate_optimized_gcode(contours)
    
    # Save G-code
    gcode_path = os.path.join(aout, "cam_optimized_program.nc")
    with open(gcode_path, 'w') as f:
        f.write('\n'.join(gcode_lines))
    
    # Save CAM analysis
    analysis_path = os.path.join(aout, "cam_analysis.json")
    analysis_data = {
        "dxf_file": os.path.basename(args.dxf),
        "cam_settings": {
            "simplify_tolerance": cam_settings.simplify_tolerance,
            "join_curves": cam_settings.join_curves,
            "use_arc_interpolation": cam_settings.use_arc_interpolation,
            "arc_tolerance": cam_settings.arc_tolerance,
            "min_arc_radius": cam_settings.min_arc_radius,
            "max_arc_radius": cam_settings.max_arc_radius,
            "holes_first": cam_settings.holes_first,
            "reduce_pierces": cam_settings.reduce_pierces,
            "join_tolerance": cam_settings.join_tolerance,
            "rapid_speed": cam_settings.rapid_speed,
            "cutting_speed": cam_settings.cutting_speed,
            "pierce_time": cam_settings.pierce_time
        },
        "processing_results": {
            "total_contours": len(contours),
            "inner_contours": len([c for c in contours if c.contour_type == 'inner']),
            "outer_contours": len([c for c in contours if c.contour_type == 'outer']),
            "open_contours": len([c for c in contours if c.contour_type == 'open']),
            "total_arc_segments": sum(len(c.arc_segments) for c in contours),
            "total_pierce_points": sum(len(c.pierce_points) for c in contours)
        },
        "contours": [
            {
                "contour_type": contour.contour_type,
                "start_point": contour.start_point,
                "end_point": contour.end_point,
                "is_closed": contour.is_closed,
                "arc_segments_count": len(contour.arc_segments),
                "pierce_points_count": len(contour.pierce_points)
            }
            for contour in contours
        ]
    }
    
    import json
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print("[green]Advanced CAM processing complete[/green]")
    print(f"  G-code: {gcode_path}")
    print(f"  Analysis: {analysis_path}")
    print(f"  Total contours: {len(contours)}")
    print(f"  Arc segments: {sum(len(c.arc_segments) for c in contours)}")
    print(f"  Pierce points: {sum(len(c.pierce_points) for c in contours)}")

def command_path_optimizer(args):
    """Optimize DXF paths with visualization"""
    aout = args.out
    os.makedirs(aout, exist_ok=True)
    
    # Initialize optimizer
    optimizer = WaterjetPathOptimizer(
        simplify_tolerance=args.simplify_tolerance,
        join_tolerance=args.join_tolerance,
        use_arc_interpolation=args.use_arc_interpolation,
        arc_tolerance=args.arc_tolerance,
        min_arc_radius=args.min_arc_radius,
        max_arc_radius=args.max_arc_radius
    )
    
    # Load and optimize
    print(f"[blue]Loading DXF: {args.dxf}[/blue]")
    paths = optimizer.load_dxf(args.dxf)
    print(f"[green]Loaded {len(paths)} paths[/green]")
    
    print("[blue]Optimizing paths...[/blue]")
    optimized_paths = optimizer.optimize_paths(paths)
    print(f"[green]Optimized to {len(optimized_paths)} paths[/green]")
    
    # Generate visualization
    viz_path = os.path.join(aout, "optimization_comparison.png")
    print("[blue]Generating visualization...[/blue]")
    optimizer.plot_optimization(save_path=viz_path, show=False)
    
    # Export G-code
    gcode_path = os.path.join(aout, "optimized_program.nc")
    print("[blue]Exporting optimized G-code...[/blue]")
    optimizer.export_gcode(gcode_path, 
                          feed_rate=args.feed_rate,
                          rapid_rate=args.rapid_rate,
                          pierce_time=args.pierce_time)
    
    # Generate report
    report = optimizer.get_optimization_report()
    report_path = os.path.join(aout, "optimization_report.json")
    
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    metrics = optimizer.metrics
    print("\n[bold green]Path Optimization Complete![/bold green]")
    print(f"  Visualization: {viz_path}")
    print(f"  📄 G-code: {gcode_path}")
    print(f"  📋 Report: {report_path}")
    print(f"\n[bold blue]Optimization Summary:[/bold blue]")
    print(f"  - Total contours: {metrics.contour_count}")
    print(f"  - Inner contours: {metrics.inner_contours}")
    print(f"  - Outer contours: {metrics.outer_contours}")
    print(f"  - Total cutting length: {metrics.total_length:.1f} mm")
    print(f"  - Total rapid distance: {metrics.rapid_distance:.1f} mm")
    print(f"  - Pierce count: {metrics.pierce_count}")
    print(f"  - Optimization time: {metrics.optimization_time:.3f}s")

def command_presets(args):
    """List available toolpath presets"""
    presets = list_presets()
    
    if args.preset:
        # Show details for specific preset
        loader = PresetLoader()
        preset_data = loader.load_preset(args.preset)
        
        print(f"[bold blue]Preset: {args.preset}[/bold blue]")
        print(f"Kerf compensation: {preset_data.get('kerf_compensation', 'N/A')} mm")
        print(f"Rapid speed: {preset_data.get('rapid_speed', 'N/A')} mm/min")
        print(f"Cutting speed: {preset_data.get('cutting_speed', 'N/A')} mm/min")
        print(f"Pierce time: {preset_data.get('pierce_time', 'N/A')} seconds")
        print(f"Optimize rapids: {preset_data.get('optimize_rapids', 'N/A')}")
        print(f"Optimize direction: {preset_data.get('optimize_direction', 'N/A')}")
        print(f"Entry strategy: {preset_data.get('entry_strategy', 'N/A')}")
    else:
        # List all presets
        print("[bold blue]Available Toolpath Presets:[/bold blue]")
        for preset in presets:
            print(f"  - {preset}")
        print(f"\n[green]Total: {len(presets)} presets available[/green]")
        print(f"[blue]Use --preset <name> to see details[/blue]")

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
    # Advanced toolpath options
    pa.add_argument("--use-advanced-toolpath", action="store_true", default=True, help="Use advanced toolpath optimization")
    pa.add_argument("--no-advanced-toolpath", action="store_false", dest="use_advanced_toolpath", help="Disable advanced toolpath optimization")
    pa.add_argument("--rapid-speed", type=float, default=10000.0, help="Rapid speed in mm/min")
    pa.add_argument("--cutting-speed", type=float, default=1200.0, help="Cutting speed in mm/min")
    pa.add_argument("--pierce-time", type=float, default=0.5, help="Pierce time in seconds")
    pa.add_argument("--optimize-rapids", action="store_true", default=True, help="Optimize rapid moves")
    pa.add_argument("--no-optimize-rapids", action="store_false", dest="optimize_rapids", help="Disable rapid move optimization")
    pa.add_argument("--optimize-direction", action="store_true", default=True, help="Optimize cutting direction")
    pa.add_argument("--no-optimize-direction", action="store_false", dest="optimize_direction", help="Disable cutting direction optimization")
    pa.add_argument("--entry-strategy", default="tangent", choices=["tangent", "perpendicular", "angle"], help="Entry strategy")
    pa.add_argument("--select-groups", help="Comma-separated groups to include (e.g. Group1,Group3)")
    pa.set_defaults(func=command_analyze)

    pg = sub.add_parser("gcode", help="Generate toy G-code")
    pg.add_argument("dxf")
    pg.add_argument("--feed", type=float, default=1200.0)
    pg.add_argument("--m-on", default="M62")
    pg.add_argument("--m-off", default="M63")
    pg.add_argument("--pierce-ms", type=int, default=500)
    pg.add_argument("--out", default="out")
    pg.set_defaults(func=command_gcode)

    # OpenCV converter command (unified implementation)
    # removed opencv image-to-dxf command

    # Inkscape converter command
    # removed inkscape image-to-dxf command

    # OpenAI analysis command
    pai = sub.add_parser("ai-analyze", help="Analyze DXF with OpenAI for manufacturing insights")
    pai.add_argument("dxf")
    pai.add_argument("--out", default="output")
    pai.add_argument("--material", default="Granite")
    pai.add_argument("--thickness", type=float, default=25.0)
    pai.add_argument("--kerf", type=float, default=1.1)
    pai.add_argument("--rate-per-m", type=float, default=825.0)
    pai.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    pai.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    pai.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens for response")
    pai.add_argument("--temperature", type=float, default=0.7, help="Response creativity (0.0-1.0)")
    pai.set_defaults(func=command_openai_analyze)

    # OpenAI design suggestions command
    pdes = sub.add_parser("ai-design", help="Generate design suggestions using OpenAI")
    pdes.add_argument("description", help="Description of the desired design")
    pdes.add_argument("--out", default="output")
    pdes.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    pdes.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    pdes.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens for response")
    pdes.add_argument("--temperature", type=float, default=0.7, help="Response creativity (0.0-1.0)")
    pdes.set_defaults(func=command_openai_design)

    # Ollama analysis command
    pollama = sub.add_parser("ollama-analyze", help="Analyze DXF with Ollama for manufacturing insights")
    pollama.add_argument("dxf")
    pollama.add_argument("--out", default="output")
    pollama.add_argument("--material", default="Granite")
    pollama.add_argument("--thickness", type=float, default=25.0)
    pollama.add_argument("--kerf", type=float, default=1.1)
    pollama.add_argument("--rate-per-m", type=float, default=825.0)
    pollama.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL")
    pollama.add_argument("--model", default="gpt-oss-20b", help="Ollama model to use")
    pollama.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    pollama.set_defaults(func=command_ollama_analyze)

    # Ollama design suggestions command
    polldes = sub.add_parser("ollama-design", help="Generate design suggestions using Ollama")
    polldes.add_argument("description", help="Description of the desired design")
    polldes.add_argument("--out", default="output")
    polldes.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL")
    polldes.add_argument("--model", default="gpt-oss-20b", help="Ollama model to use")
    polldes.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    polldes.set_defaults(func=command_ollama_design)

    # Advanced toolpath command
    ptoolpath = sub.add_parser("advanced-toolpath", help="Generate advanced optimized toolpath")
    ptoolpath.add_argument("dxf")
    ptoolpath.add_argument("--out", default="output")
    ptoolpath.add_argument("--kerf", type=float, default=1.1, help="Kerf compensation in mm")
    ptoolpath.add_argument("--rapid-speed", type=float, default=10000.0, help="Rapid speed in mm/min")
    ptoolpath.add_argument("--cutting-speed", type=float, default=1200.0, help="Cutting speed in mm/min")
    ptoolpath.add_argument("--pierce-time", type=float, default=0.5, help="Pierce time in seconds")
    ptoolpath.add_argument("--min-rapid-distance", type=float, default=5.0, help="Minimum rapid distance in mm")
    ptoolpath.add_argument("--optimize-rapids", action="store_true", default=True, help="Optimize rapid moves")
    ptoolpath.add_argument("--optimize-direction", action="store_true", default=True, help="Optimize cutting direction")
    ptoolpath.add_argument("--entry-strategy", default="tangent", choices=["tangent", "perpendicular", "angle"], help="Entry strategy")
    ptoolpath.set_defaults(func=command_advanced_toolpath)

    # Advanced CAM command
    pcam = sub.add_parser("cam-process", help="Process DXF with advanced CAM features")
    pcam.add_argument("dxf")
    pcam.add_argument("--out", default="output")
    # DXF Preprocessing
    pcam.add_argument("--simplify-tolerance", type=float, default=0.1, help="Curve simplification tolerance (mm)")
    pcam.add_argument("--join-curves", action="store_true", default=True, help="Join connected curves")
    pcam.add_argument("--no-join-curves", action="store_false", dest="join_curves", help="Disable curve joining")
    # Arc Interpolation
    pcam.add_argument("--use-arc-interpolation", action="store_true", default=True, help="Use G2/G3 arc interpolation")
    pcam.add_argument("--no-arc-interpolation", action="store_false", dest="use_arc_interpolation", help="Disable arc interpolation")
    pcam.add_argument("--arc-tolerance", type=float, default=0.01, help="Arc fitting tolerance (mm)")
    pcam.add_argument("--min-arc-radius", type=float, default=0.5, help="Minimum arc radius (mm)")
    pcam.add_argument("--max-arc-radius", type=float, default=1000.0, help="Maximum arc radius (mm)")
    # Contour Sorting
    pcam.add_argument("--holes-first", action="store_true", default=True, help="Cut holes first")
    pcam.add_argument("--no-holes-first", action="store_false", dest="holes_first", help="Disable holes-first ordering")
    # Path Optimization
    pcam.add_argument("--reduce-pierces", action="store_true", default=True, help="Reduce pierce points")
    pcam.add_argument("--no-reduce-pierces", action="store_false", dest="reduce_pierces", help="Disable pierce reduction")
    pcam.add_argument("--join-tolerance", type=float, default=0.1, help="Path joining tolerance (mm)")
    # G-code Settings
    pcam.add_argument("--rapid-speed", type=float, default=10000.0, help="Rapid speed (mm/min)")
    pcam.add_argument("--cutting-speed", type=float, default=1200.0, help="Cutting speed (mm/min)")
    pcam.add_argument("--pierce-time", type=float, default=0.5, help="Pierce time (seconds)")
    pcam.set_defaults(func=command_cam_process)

    # Path Optimizer command
    popt = sub.add_parser("path-optimizer", help="Optimize DXF paths with visualization")
    popt.add_argument("dxf")
    popt.add_argument("--out", default="optimized_output")
    # Optimization settings
    popt.add_argument("--simplify-tolerance", type=float, default=0.1, help="Geometry simplification tolerance (mm)")
    popt.add_argument("--join-tolerance", type=float, default=0.1, help="Path joining tolerance (mm)")
    popt.add_argument("--use-arc-interpolation", action="store_true", default=True, help="Use arc interpolation")
    popt.add_argument("--no-arc-interpolation", action="store_false", dest="use_arc_interpolation", help="Disable arc interpolation")
    popt.add_argument("--arc-tolerance", type=float, default=0.01, help="Arc fitting tolerance (mm)")
    popt.add_argument("--min-arc-radius", type=float, default=0.5, help="Minimum arc radius (mm)")
    popt.add_argument("--max-arc-radius", type=float, default=1000.0, help="Maximum arc radius (mm)")
    # G-code settings
    popt.add_argument("--feed-rate", type=float, default=1200.0, help="Cutting feed rate (mm/min)")
    popt.add_argument("--rapid-rate", type=float, default=10000.0, help="Rapid feed rate (mm/min)")
    popt.add_argument("--pierce-time", type=float, default=0.5, help="Pierce delay time (seconds)")
    popt.set_defaults(func=command_path_optimizer)

    # Presets command
    ppresets = sub.add_parser("presets", help="List and view toolpath presets")
    ppresets.add_argument("--preset", help="Show details for specific preset")
    ppresets.set_defaults(func=command_presets)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
