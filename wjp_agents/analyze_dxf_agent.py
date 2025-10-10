import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils.io_helpers import ensure_dirs, append_log, timestamp, save_json
from src.wjp_analyser.analysis import dxf_analyzer


class AnalyzeDXFAgent:
    """
    Runs the Waterjet DXF Analyzer on the provided DXF file.
    Handles parameter tuning (kerf, scaling, etc.) and report saving.
    """

    def __init__(self):
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.reports_dir = os.path.join(project_root, "output", "reports")
        ensure_dirs([self.reports_dir])

    def run(self, dxf_path):
        print(f"[AnalyzeDXFAgent] Running DXF analysis on {dxf_path}...")

        try:
            # Use built-in analyzer API
            # Write artifacts under 'out' to align with existing project layout
            result = dxf_analyzer.analyze_dxf(dxf_path)
        except Exception as e:
            result = {"error": str(e)}
            print(f"[AnalyzeDXFAgent] ERROR: {e}")

        report_filename = f"report_{timestamp()}.json"
        report_path = os.path.join(self.reports_dir, report_filename)
        save_json(result, report_path)

        append_log({
            "agent": "AnalyzeDXFAgent",
            "dxf_path": dxf_path,
            "report_path": report_path,
            "result": result,
        })

        print(f"[AnalyzeDXFAgent] Analysis report saved -> {report_path}")
        return {"dxf_path": dxf_path, "report_path": report_path, "analysis": result}


