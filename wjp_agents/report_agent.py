from .utils.io_helpers import append_log, timestamp, save_json
import os


class ReportAgent:
    """
    Compiles pipeline results into a consolidated report.
    """

    def __init__(self):
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.reports_dir = os.path.join(project_root, "output", "reports")

    def compile(self, design, conversion, analysis):
        final_report = {
            "job_id": f"WJP_{timestamp()}",
            "designer": design,
            "image_to_dxf": conversion,
            "analysis": analysis,
        }

        report_path = os.path.join(self.reports_dir, f"final_summary_{timestamp()}.json")
        save_json(final_report, report_path)
        append_log({"agent": "ReportAgent", "summary_path": report_path})
        print(f"[ReportAgent] Final report saved -> {report_path}")
        return report_path


