from .designer_agent import DesignerAgent
from .image_to_dxf_agent import ImageToDXFAgent
from .analyze_dxf_agent import AnalyzeDXFAgent
from .report_agent import ReportAgent
from .learning_agent import LearningAgent
from .utils.io_helpers import append_log


class SupervisorAgent:
    """
    Controls all agent operations in sequence and logs the workflow.
    """

    def __init__(self):
        self.designer = DesignerAgent()
        self.image2dxf = ImageToDXFAgent()
        self.analyzer = AnalyzeDXFAgent()
        self.reporter = ReportAgent()
        self.learner = LearningAgent()

    def run_pipeline(self, user_input):
        print("\nStarting Full Waterjet Agent Pipeline (with Learning)...\n")

        try:
            design = self.designer.run(user_input)

            optimized = self.learner.run(design["image_path"])

            final_dxf = self.image2dxf.convert_image_to_dxf(design["image_path"])
            final_analysis = self.analyzer.run(final_dxf)

            # Convert DetectionParams to dictionary for JSON serialization
            params_dict = {
                "min_area": optimized["best_params"].min_area,
                "max_area": optimized["best_params"].max_area,
                "min_circularity": optimized["best_params"].min_circularity,
                "min_solidity": optimized["best_params"].min_solidity,
                "merge_distance": optimized["best_params"].merge_distance
            }
            
            report_path = self.reporter.compile(
                design,
                {"dxf_path": final_dxf, "params": params_dict},
                final_analysis,
            )

            append_log(
                {
                    "agent": "SupervisorAgent",
                    "status": "Pipeline completed successfully with optimization",
                    "final_report": report_path,
                    "best_params": params_dict,
                }
            )
            print(f"\nFull Pipeline Completed. Optimized Report -> {report_path}\n")

        except Exception as e:
            append_log({"agent": "SupervisorAgent", "status": "Error", "error": str(e)})
            print(f"\nPipeline Error: {e}\n")


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Sample medallion with geometric border"
    SupervisorAgent().run_pipeline(prompt)


