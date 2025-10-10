"""
Updated tests for the WJP ANALYSER system.
These tests work with the current src/wjp_analyser structure.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Test the new object detection functionality
def test_object_detector():
    """Test the ObjectDetector class."""
    try:
        from src.wjp_analyser.image_processing.object_detector import ObjectDetector, DetectionParams
        
        # Create a simple test image with geometric shapes
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw some test shapes
        import cv2
        cv2.circle(test_image, (100, 100), 50, (0, 0, 0), -1)  # Circle
        cv2.rectangle(test_image, (200, 200), (300, 300), (0, 0, 0), -1)  # Rectangle
        
        # Convert to binary
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Test object detection
        detector = ObjectDetector(DetectionParams(min_area=100))
        objects = detector.detect_objects(binary, test_image)
        
        assert len(objects) >= 2  # Should detect at least 2 objects
        assert all(obj.area > 100 for obj in objects)  # All objects should meet min area
        assert all(obj.circularity >= 0 for obj in objects)  # Circularity should be valid
        
        print(f"✓ Detected {len(objects)} objects")
        for obj in objects:
            print(f"  - Object {obj.id}: {obj.layer_type}, area={obj.area:.1f}, circularity={obj.circularity:.2f}")
            
    except ImportError as e:
        pytest.skip(f"ObjectDetector not available: {e}")


def test_interactive_editor():
    """Test the InteractiveEditor class."""
    try:
        from src.wjp_analyser.image_processing.interactive_editor import InteractiveEditor
        
        editor = InteractiveEditor()
        
        # Create test image
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        import cv2
        cv2.circle(test_image, (100, 100), 30, (0, 0, 0), -1)
        
        # Test binary image setting
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        editor.set_binary_image(binary)
        
        # Test object detection
        objects = editor.detect_objects()
        assert len(objects) >= 1  # Should detect at least 1 object
        
        # Test preview generation
        preview = editor.generate_preview()
        assert preview.shape == (200, 200, 3)  # Should match input image shape
        
        print(f"✓ Editor detected {len(objects)} objects")
        print(f"✓ Preview generated: {preview.shape}")
        
    except ImportError as e:
        pytest.skip(f"InteractiveEditor not available: {e}")


def test_preview_renderer():
    """Test the PreviewRenderer class."""
    try:
        from src.wjp_analyser.image_processing.preview_renderer import PreviewRenderer
        
        renderer = PreviewRenderer()
        
        # Create test data
        test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        binary_image = np.zeros((300, 300), dtype=np.uint8)
        import cv2
        cv2.circle(binary_image, (150, 150), 50, 255, -1)
        
        # Set test data
        renderer.set_images(test_image, binary_image)
        
        # Test vector overlay
        overlay = renderer.render_vector_overlay()
        assert overlay.shape == (300, 300, 3)
        
        # Test multi-layer preview
        layer_previews = renderer.render_multi_layer_preview()
        assert isinstance(layer_previews, dict)
        
        # Test export preview
        export_preview = renderer.generate_export_preview()
        assert export_preview.shape == (800, 800, 3)
        
        print(f"✓ Vector overlay generated: {overlay.shape}")
        print(f"✓ Multi-layer preview generated: {len(layer_previews)} layers")
        print(f"✓ Export preview generated: {export_preview.shape}")
        
    except ImportError as e:
        pytest.skip(f"PreviewRenderer not available: {e}")


def test_agent_system():
    """Test the agent system."""
    try:
        from wjp_agents.analyze_dxf_agent import AnalyzeDXFAgent
        from wjp_agents.designer_agent import DesignerAgent
        from wjp_agents.image_to_dxf_agent import ImageToDXFAgent
        from wjp_agents.learning_agent import LearningAgent
        from wjp_agents.report_agent import ReportAgent
        from wjp_agents.supervisor_agent import SupervisorAgent
        
        # Test individual agents
        designer = DesignerAgent()
        assert designer is not None
        
        image2dxf = ImageToDXFAgent()
        assert image2dxf is not None
        
        analyzer = AnalyzeDXFAgent()
        assert analyzer is not None
        
        learner = LearningAgent()
        assert learner is not None
        
        reporter = ReportAgent()
        assert reporter is not None
        
        supervisor = SupervisorAgent()
        assert supervisor is not None
        
        print("✓ All agents initialized successfully")
        
    except ImportError as e:
        pytest.skip(f"Agent system not available: {e}")


def test_agent_utils():
    """Test agent utility functions."""
    try:
        from wjp_agents.utils.io_helpers import ensure_dirs, timestamp, save_json, append_log
        
        # Test directory creation
        test_dir = "test_output"
        ensure_dirs([test_dir])
        assert os.path.exists(test_dir)
        
        # Test timestamp generation
        ts = timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0
        
        # Test JSON saving
        test_data = {"test": "data", "number": 42}
        test_file = os.path.join(test_dir, "test.json")
        save_json(test_data, test_file)
        assert os.path.exists(test_file)
        
        # Cleanup
        os.remove(test_file)
        os.rmdir(test_dir)
        
        print("✓ Agent utilities working correctly")
        
    except ImportError as e:
        pytest.skip(f"Agent utilities not available: {e}")


def test_dxf_analysis():
    """Test DXF analysis functionality."""
    try:
        from src.wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf
        
        # Create a simple test DXF file
        import ezdxf
        
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        
        # Add some test entities
        msp.add_circle((0, 0), 10)
        msp.add_line((0, 0), (20, 20))
        
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            doc.saveas(tmp.name)
            dxf_path = tmp.name
        
        try:
            # Test analysis
            args = AnalyzeArgs(out="test_output")
            result = analyze_dxf(dxf_path, args)
            
            assert "metrics" in result
            assert "components" in result
            assert "groups" in result
            
            print("✓ DXF analysis completed successfully")
            
        finally:
            # Cleanup
            os.unlink(dxf_path)
            if os.path.exists("test_output"):
                import shutil
                shutil.rmtree("test_output")
        
    except ImportError as e:
        pytest.skip(f"DXF analysis not available: {e}")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_object_detector()
    test_interactive_editor()
    test_preview_renderer()
    test_agent_system()
    test_agent_utils()
    test_dxf_analysis()
    print("\n✓ All tests completed successfully!")

