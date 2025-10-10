import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils.io_helpers import ensure_dirs, append_log, timestamp
from src.wjp_analyser.image_processing.object_detector import ObjectDetector, DetectionParams
from src.wjp_analyser.image_processing.interactive_editor import InteractiveEditor
from src.wjp_analyser.image_processing.preview_renderer import PreviewRenderer


class ImageToDXFAgent:
    """
    Enhanced Image to DXF conversion agent with interactive editing capabilities.
    Integrates with the new object detection and interactive editing system.
    """

    def __init__(self):
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.output_dir = os.path.join(project_root, "output")
        self.dxf_dir = os.path.join(self.output_dir, "dxf")
        self.previews_dir = os.path.join(self.output_dir, "previews")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        
        ensure_dirs([self.dxf_dir, self.previews_dir, self.reports_dir])
        self.editor = InteractiveEditor()
        self.renderer = PreviewRenderer()

    def convert_image_to_dxf(self, image_path, detection_params=None, interactive_mode=False):
        """
        Convert image to DXF with enhanced object detection and editing.
        
        Args:
            image_path: Path to input image
            detection_params: Detection parameters for object detection
            interactive_mode: Whether to enable interactive editing features
            
        Returns:
            Path to generated DXF file
        """
        try:
            # Load image into editor
            if not self.editor.load_image(image_path):
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Generate binary image for object detection
            import cv2
            from PIL import Image
            
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply threshold - use INV because DesignerAgent creates black lines on white background
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Set binary image in editor
            self.editor.set_binary_image(binary)
            
            # Use default parameters if not provided
            if detection_params is None:
                detection_params = DetectionParams(
                    min_area=100,
                    max_area=1000000,
                    min_perimeter=20,
                    min_circularity=0.1,
                    min_solidity=0.3,
                    merge_distance=10.0
                )
            
            # Detect objects
            objects = self.editor.detect_objects(detection_params)
            
            # Generate DXF filename
            dxf_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_converted.dxf"
            dxf_path = os.path.join(self.dxf_dir, dxf_filename)
            
            # Export to DXF - ensure all objects are visible and selected
            for obj in objects:
                obj.visible = True
                obj.selected = True
            
            if self.editor.export_all_objects(dxf_path):
                # Generate preview
                preview_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_preview.png"
                preview_path = os.path.join(self.previews_dir, preview_filename)
                
                # Set up preview renderer
                self.renderer.set_images(self.editor.current_image, binary)
                self.renderer.set_objects(objects)
                
                # Generate preview
                preview_image = self.renderer.render_vector_overlay(alpha=0.8, line_width=2.0)
                
                # Save preview
                from PIL import Image
                preview_pil = Image.fromarray(preview_image)
                preview_pil.save(preview_path)
                
                # Log results
                append_log({
                    "agent": "ImageToDXFAgent",
                    "image_path": image_path,
                    "dxf_path": dxf_path,
                    "preview_path": preview_path,
                    "objects_detected": len(objects),
                    "interactive_mode": interactive_mode,
                    "detection_params": {
                        "min_area": detection_params.min_area,
                        "max_area": detection_params.max_area,
                        "min_circularity": detection_params.min_circularity
                    }
                })
                
                print(f"[ImageToDXFAgent] Converted image -> DXF at {dxf_path}")
                print(f"[ImageToDXFAgent] Detected {len(objects)} objects")
                print(f"[ImageToDXFAgent] Preview saved at {preview_path}")
                
                return dxf_path
            else:
                raise RuntimeError("Failed to export objects to DXF")
                
        except Exception as e:
            append_log({
                "agent": "ImageToDXFAgent",
                "error": str(e),
                "image_path": image_path
            })
            raise

    def run(self, image_path, detection_params=None, interactive_mode=False):
        """
        Main run method for the agent.
        
        Args:
            image_path: Path to input image
            detection_params: Detection parameters
            interactive_mode: Enable interactive features
            
        Returns:
            Dictionary with conversion results
        """
        dxf_path = self.convert_image_to_dxf(image_path, detection_params, interactive_mode)
        
        # Get object statistics
        if self.editor.detector:
            stats = self.editor.detector.get_statistics()
        else:
            stats = {}
        
        return {
            "image_path": image_path,
            "dxf_path": dxf_path,
            "objects_detected": stats.get("total_objects", 0),
            "object_types": stats.get("type_counts", {}),
            "interactive_mode": interactive_mode
        }


