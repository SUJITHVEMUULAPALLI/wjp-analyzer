import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils.io_helpers import append_log
from .image_to_dxf_agent import ImageToDXFAgent
from .analyze_dxf_agent import AnalyzeDXFAgent
from src.wjp_analyser.image_processing.object_detector import DetectionParams


class LearningAgent:
    """
    Enhanced Learning Agent that optimizes parameters for cleaner DXF conversion and analysis.
    Uses heuristic feedback from AnalyzeDXFAgent to minimize open contours
    and improve path clarity for waterjet cutting.
    Now integrates with the new interactive editing system.
    """

    def __init__(self):
        self.image2dxf = ImageToDXFAgent()
        self.analyzer = AnalyzeDXFAgent()
        
        # Expanded parameter space for better optimization
        self.param_space = {
            "min_area": [50, 100, 150, 200],
            "max_area": [500000, 1000000, 2000000],
            "min_circularity": [0.05, 0.1, 0.15, 0.2],
            "min_solidity": [0.2, 0.3, 0.4, 0.5],
            "merge_distance": [5.0, 10.0, 15.0, 20.0],
            "threshold": [120, 140, 160, 180, 200]
        }

    def score_result(self, result):
        """
        Enhanced scoring function that considers multiple quality metrics.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            Score (lower is better)
        """
        if "error" in result:
            return 99999
        
        # Extract metrics
        metrics = result.get("metrics", {})
        quality = result.get("quality", {})
        
        # Scoring factors
        open_contours = quality.get("Open Polylines", 0)
        total_length = metrics.get("length_internal_mm", 0)
        pierce_count = metrics.get("pierces", 0)
        shaky_polylines = len(quality.get("Shaky Polylines", []))
        tiny_segments = quality.get("Tiny Segments", 0)
        
        # Calculate composite score
        score = (
            open_contours * 100 +           # Penalize open contours heavily
            shaky_polylines * 50 +          # Penalize shaky polylines
            tiny_segments * 25 +            # Penalize tiny segments
            abs(total_length - 30000) / 100 +  # Prefer reasonable cutting length
            pierce_count * 10               # Penalize excessive pierce points
        )
        
        return score

    def optimize_parameters(self, image_path, max_iterations=20):
        """
        Optimize parameters using a more sophisticated approach.
        
        Args:
            image_path: Path to input image
            max_iterations: Maximum number of optimization iterations
            
        Returns:
            Best parameters and result
        """
        print(f"\n[LearningAgent] Starting parameter optimization for {image_path}...\n")
        
        best_score = 99999
        best_result = None
        best_params = None
        iteration = 0
        
        # Start with default parameters
        current_params = DetectionParams(
            min_area=100,
            max_area=1000000,
            min_circularity=0.1,
            min_solidity=0.3,
            merge_distance=10.0
        )
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Iteration {iteration}/{max_iterations}")
            
            # Test current parameters
            try:
                dxf_path = self.image2dxf.convert_image_to_dxf(image_path, current_params)
                result = self.analyzer.run(dxf_path)["analysis"]
                score = self.score_result(result)
                
                print(f"  Score: {score:.2f}")
                print(f"  Params: min_area={current_params.min_area}, "
                      f"min_circularity={current_params.min_circularity:.2f}, "
                      f"merge_distance={current_params.merge_distance}")
                
                if score < best_score:
                    best_score = score
                    best_result = result
                    best_params = DetectionParams(
                        min_area=current_params.min_area,
                        max_area=current_params.max_area,
                        min_circularity=current_params.min_circularity,
                        min_solidity=current_params.min_solidity,
                        merge_distance=current_params.merge_distance
                    )
                print(f"  [OK] New best score!")
                
            except Exception as e:
                print(f"  [ERROR] Error: {e}")
                score = 99999
            
            # Generate next parameters using adaptive search
            if iteration < max_iterations:
                current_params = self._generate_next_params(current_params, best_score, score)
        
        return best_params, best_result, best_score

    def _generate_next_params(self, current_params, best_score, current_score):
        """
        Generate next parameters using adaptive search strategy.
        
        Args:
            current_params: Current parameter set
            best_score: Best score so far
            current_score: Current iteration score
            
        Returns:
            Next parameter set to try
        """
        import random
        
        # If current score is worse than best, try to move back towards best
        if current_score > best_score:
            # Randomly adjust parameters
            new_params = DetectionParams(
                min_area=max(50, min(200, current_params.min_area + random.choice([-25, 0, 25]))),
                max_area=max(500000, min(2000000, current_params.max_area + random.choice([-100000, 0, 100000]))),
                min_circularity=max(0.05, min(0.2, current_params.min_circularity + random.choice([-0.02, 0, 0.02]))),
                min_solidity=max(0.2, min(0.5, current_params.min_solidity + random.choice([-0.05, 0, 0.05]))),
                merge_distance=max(5.0, min(20.0, current_params.merge_distance + random.choice([-2.5, 0, 2.5])))
            )
        else:
            # If improving, try more aggressive changes
            new_params = DetectionParams(
                min_area=random.choice(self.param_space["min_area"]),
                max_area=random.choice(self.param_space["max_area"]),
                min_circularity=random.choice(self.param_space["min_circularity"]),
                min_solidity=random.choice(self.param_space["min_solidity"]),
                merge_distance=random.choice(self.param_space["merge_distance"])
            )
        
        return new_params

    def run(self, image_path, max_iterations=20):
        """
        Main run method for the learning agent.
        
        Args:
            image_path: Path to input image
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        try:
            best_params, best_result, best_score = self.optimize_parameters(image_path, max_iterations)
            
            # Log results
            append_log({
                "agent": "LearningAgent",
                "image_path": image_path,
                "best_params": {
                    "min_area": best_params.min_area,
                    "max_area": best_params.max_area,
                    "min_circularity": best_params.min_circularity,
                    "min_solidity": best_params.min_solidity,
                    "merge_distance": best_params.merge_distance
                },
                "best_score": best_score,
                "iterations": max_iterations,
                "optimization_successful": True
            })
            
            print(f"\n[LearningAgent] Optimization Complete")
            print(f"   Best Score: {best_score:.2f}")
            print(f"   Best Parameters:")
            print(f"     - Min Area: {best_params.min_area}")
            print(f"     - Max Area: {best_params.max_area}")
            print(f"     - Min Circularity: {best_params.min_circularity:.2f}")
            print(f"     - Min Solidity: {best_params.min_solidity:.2f}")
            print(f"     - Merge Distance: {best_params.merge_distance}")
            
            return {
                "best_params": best_params,
                "best_result": best_result,
                "best_score": best_score,
                "optimization_successful": True
            }
            
        except Exception as e:
            append_log({
                "agent": "LearningAgent",
                "error": str(e),
                "image_path": image_path,
                "optimization_successful": False
            })
            print(f"\n[LearningAgent] Optimization failed: {e}")
            
            return {
                "best_params": None,
                "best_result": None,
                "best_score": 99999,
                "optimization_successful": False,
                "error": str(e)
            }


