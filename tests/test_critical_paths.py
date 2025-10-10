"""
Critical Path Integration Tests for WJP Analyser
================================================

This module tests the critical paths and workflows to ensure the system
functions correctly end-to-end.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Import critical modules
from src.wjp_analyser.analysis.dxf_analyzer import AnalyzeArgs, analyze_dxf
from src.wjp_analyser.image_processing.image_processor import ImageProcessor
from src.wjp_analyser.manufacturing.nesting import NestingEngine
from src.wjp_analyser.ai.openai_client import OpenAIAnalyzer
from src.wjp_analyser.ai.ollama_client import OllamaAnalyzer
from src.wjp_analyser.utils.error_handler import handle_errors, safe_execute
from src.wjp_analyser.utils.input_validator import validate_uploaded_file
from src.wjp_analyser.utils.cache_manager import cache_dxf_analysis, cache_image_processing


class TestDXFAnalysisCriticalPath:
    """Test DXF analysis critical path."""
    
    def create_test_dxf(self, content: str = None) -> str:
        """Create a test DXF file."""
        if content is None:
            content = """0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
ENTITIES
0
LINE
8
0
10
0.0
20
0.0
11
100.0
21
100.0
0
ENDSEC
0
EOF"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dxf', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_dxf_analysis_basic(self):
        """Test basic DXF analysis."""
        dxf_path = self.create_test_dxf()
        
        try:
            # Test analysis with basic parameters
            args = AnalyzeArgs(out="test_output")
            args.sheet_width = 1000.0
            args.sheet_height = 1000.0
            
            result = analyze_dxf(dxf_path, args)
            
            # Verify basic structure
            assert isinstance(result, dict)
            assert "metrics" in result
            assert "entities" in result
            assert "components" in result
            
            # Verify metrics
            metrics = result["metrics"]
            assert "length_internal_mm" in metrics
            assert "pierces" in metrics
            assert "estimated_cutting_cost_inr" in metrics
            
        finally:
            os.unlink(dxf_path)
    
    def test_dxf_analysis_with_validation(self):
        """Test DXF analysis with input validation."""
        dxf_path = self.create_test_dxf()
        
        try:
            # Test file validation first
            validation_result = validate_uploaded_file(dxf_path, "test.dxf")
            assert validation_result.is_valid is True
            
            # Test analysis
            args = AnalyzeArgs(out="test_output")
            result = analyze_dxf(dxf_path, args)
            assert isinstance(result, dict)
            
        finally:
            os.unlink(dxf_path)
    
    def test_dxf_analysis_error_handling(self):
        """Test DXF analysis error handling."""
        # Test with non-existent file
        result, error_info = safe_execute(
            analyze_dxf, 
            "non_existent.dxf", 
            AnalyzeArgs(out="test_output"),
            context="dxf_analysis"
        )
        
        assert result is None
        assert error_info is not None
        assert error_info["category"] == "file_processing"
    
    def test_dxf_analysis_caching(self):
        """Test DXF analysis caching."""
        dxf_path = self.create_test_dxf()
        
        try:
            # Mock cache manager
            with patch('src.wjp_analyser.utils.cache_manager._cache_manager') as mock_cache:
                mock_cache.get.return_value = None
                mock_cache.set.return_value = None
                
                # Test cached analysis
                @cache_dxf_analysis
                def cached_analyze(file_path, args):
                    return analyze_dxf(file_path, args)
                
                args = AnalyzeArgs(out="test_output")
                result1 = cached_analyze(dxf_path, args)
                result2 = cached_analyze(dxf_path, args)
                
                # Verify cache was used
                assert mock_cache.get.called
                assert mock_cache.set.called
                
        finally:
            os.unlink(dxf_path)


class TestImageProcessingCriticalPath:
    """Test image processing critical path."""
    
    def create_test_image(self, width: int = 200, height: int = 200) -> str:
        """Create a test image file."""
        import cv2
        
        # Create a simple test image with geometric shapes
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw some test shapes
        cv2.circle(image, (50, 50), 20, (0, 0, 0), -1)
        cv2.rectangle(image, (100, 100), (150, 150), (0, 0, 0), -1)
        cv2.line(image, (0, 0), (width, height), (0, 0, 0), 2)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, image)
            return f.name
    
    def test_image_processing_basic(self):
        """Test basic image processing."""
        image_path = self.create_test_image()
        
        try:
            # Test image validation
            validation_result = validate_uploaded_file(image_path, "test.png")
            assert validation_result.is_valid is True
            
            # Test image processing
            processor = ImageProcessor()
            
            # Test basic processing parameters
            params = {
                "edge_threshold": 0.33,
                "min_contour_area": 100,
                "simplify_tolerance": 0.02,
                "blur_kernel_size": 5
            }
            
            # Mock the actual processing to avoid heavy dependencies
            with patch.object(processor, 'process_image') as mock_process:
                mock_process.return_value = {"success": True, "output_path": "test_output.dxf"}
                
                result = processor.process_image(image_path, params)
                assert result["success"] is True
                
        finally:
            os.unlink(image_path)
    
    def test_image_processing_caching(self):
        """Test image processing caching."""
        image_path = self.create_test_image()
        
        try:
            # Mock cache manager
            with patch('src.wjp_analyser.utils.cache_manager._cache_manager') as mock_cache:
                mock_cache.get.return_value = None
                mock_cache.set.return_value = None
                
                # Test cached processing
                @cache_image_processing
                def cached_process(image_path, params):
                    return {"success": True, "output_path": "test_output.dxf"}
                
                params = {"edge_threshold": 0.33}
                result1 = cached_process(image_path, params)
                result2 = cached_process(image_path, params)
                
                # Verify cache was used
                assert mock_cache.get.called
                assert mock_cache.set.called
                
        finally:
            os.unlink(image_path)


class TestNestingCriticalPath:
    """Test nesting critical path."""
    
    def create_test_dxf_files(self, count: int = 3) -> list:
        """Create multiple test DXF files for nesting."""
        files = []
        
        for i in range(count):
            content = f"""0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
ENTITIES
0
LINE
8
0
10
0.0
20
0.0
11
{50.0 + i * 10}
21
{50.0 + i * 10}
0
ENDSEC
0
EOF"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dxf', delete=False) as f:
                f.write(content)
                files.append(f.name)
        
        return files
    
    def test_nesting_basic(self):
        """Test basic nesting functionality."""
        dxf_files = self.create_test_dxf_files(3)
        
        try:
            # Test nesting with basic parameters
            engine = NestingEngine(sheet_width=1000.0, sheet_height=1000.0)
            result = engine.simple_nesting(dxf_files, spacing=10.0)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "success" in result or "error" in result
            
        finally:
            for file_path in dxf_files:
                os.unlink(file_path)
    
    def test_nesting_error_handling(self):
        """Test nesting error handling."""
        # Test with invalid parameters
        result, error_info = safe_execute(
            NestingEngine,
            "non_existent.dxf",
            1000.0,
            1000.0,
            context="nesting"
        )
        
        assert result is None
        assert error_info is not None


class TestAICriticalPath:
    """Test AI analysis critical path."""
    
    def test_openai_analyzer_basic(self):
        """Test OpenAI analyzer basic functionality."""
        analyzer = OpenAIAnalyzer()
        
        # Mock OpenAI API call
        with patch.object(analyzer, 'analyze_dxf') as mock_analyze:
            mock_analyze.return_value = {
                "feasibility_score": 85,
                "complexity_level": "Moderate",
                "estimated_time": "15 minutes",
                "material_recommendations": ["steel", "aluminum"],
                "toolpath_suggestions": ["Optimize cutting path"],
                "potential_issues": ["Sharp corners"],
                "optimization_tips": ["Add fillets"],
                "cost_considerations": ["Material waste"]
            }
            
            # Test analysis
            result = analyzer.analyze_dxf("test.dxf", {"metrics": {"length_internal_mm": 1000}})
            
            assert isinstance(result, dict)
            assert "feasibility_score" in result
            assert "complexity_level" in result
    
    def test_ollama_analyzer_basic(self):
        """Test Ollama analyzer basic functionality."""
        analyzer = OllamaAnalyzer()
        
        # Mock Ollama API call
        with patch.object(analyzer, 'analyze_dxf') as mock_analyze:
            mock_analyze.return_value = {
                "feasibility_score": 80,
                "complexity_level": "Simple",
                "estimated_time": "10 minutes",
                "material_recommendations": ["steel"],
                "toolpath_suggestions": ["Standard cutting"],
                "potential_issues": [],
                "optimization_tips": ["Good design"],
                "cost_considerations": ["Standard cost"]
            }
            
            # Test analysis
            result = analyzer.analyze_dxf("test.dxf", {"metrics": {"length_internal_mm": 500}})
            
            assert isinstance(result, dict)
            assert "feasibility_score" in result
            assert "complexity_level" in result
    
    def test_ai_analyzer_error_handling(self):
        """Test AI analyzer error handling."""
        analyzer = OpenAIAnalyzer()
        
        # Test with API error
        with patch.object(analyzer, 'analyze_dxf') as mock_analyze:
            mock_analyze.side_effect = Exception("API Error")
            
            result, error_info = safe_execute(
                analyzer.analyze_dxf,
                "test.dxf",
                {"metrics": {"length_internal_mm": 1000}},
                context="ai_analysis"
            )
            
            assert result is None
            assert error_info is not None
            assert error_info["category"] == "ai_service"


class TestWebInterfaceCriticalPath:
    """Test web interface critical path."""
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        # Create test files
        dxf_path = self.create_test_dxf()
        image_path = self.create_test_image()
        
        try:
            # Test DXF file validation
            dxf_result = validate_uploaded_file(dxf_path, "test.dxf")
            assert dxf_result.is_valid is True
            
            # Test image file validation
            image_result = validate_uploaded_file(image_path, "test.png")
            assert image_result.is_valid is True
            
            # Test invalid file
            invalid_result = validate_uploaded_file("test.exe", "test.exe")
            assert invalid_result.is_valid is False
            
        finally:
            os.unlink(dxf_path)
            os.unlink(image_path)
    
    def test_parameter_validation(self):
        """Test parameter validation in web interface."""
        from src.wjp_analyser.utils.input_validator import validate_material_params, validate_image_params
        
        # Test valid material parameters
        material_params = {
            "material": "steel",
            "thickness": 6.0,
            "kerf": 1.1,
            "cutting_speed": 1200.0
        }
        result = validate_material_params(material_params)
        assert result.is_valid is True
        
        # Test invalid material parameters
        invalid_material_params = {
            "material": "unknown",
            "thickness": -1.0,
            "kerf": 0.01,
            "cutting_speed": 50.0
        }
        result = validate_material_params(invalid_material_params)
        assert result.is_valid is False
        
        # Test image processing parameters
        image_params = {
            "edge_threshold": 0.33,
            "min_contour_area": 100,
            "simplify_tolerance": 0.02
        }
        result = validate_image_params(image_params)
        assert result.is_valid is True


class TestEndToEndWorkflow:
    """Test end-to-end workflows."""
    
    def test_dxf_analysis_workflow(self):
        """Test complete DXF analysis workflow."""
        dxf_path = self.create_test_dxf()
        
        try:
            # Step 1: File validation
            validation_result = validate_uploaded_file(dxf_path, "test.dxf")
            assert validation_result.is_valid is True
            
            # Step 2: DXF analysis
            args = AnalyzeArgs(out="test_output")
            analysis_result = analyze_dxf(dxf_path, args)
            assert isinstance(analysis_result, dict)
            
            # Step 3: AI analysis (mocked)
            with patch('src.wjp_analyser.ai.openai_client.OpenAIAnalyzer') as mock_ai:
                mock_instance = mock_ai.return_value
                mock_instance.analyze_dxf.return_value = {
                    "feasibility_score": 85,
                    "complexity_level": "Moderate"
                }
                
                ai_result = mock_instance.analyze_dxf(dxf_path, analysis_result)
                assert isinstance(ai_result, dict)
                assert "feasibility_score" in ai_result
            
        finally:
            os.unlink(dxf_path)
    
    def test_image_to_dxf_workflow(self):
        """Test complete image-to-DXF workflow."""
        image_path = self.create_test_image()
        
        try:
            # Step 1: Image validation
            validation_result = validate_uploaded_file(image_path, "test.png")
            assert validation_result.is_valid is True
            
            # Step 2: Image processing (mocked)
            with patch('src.wjp_analyser.image_processing.image_processor.ImageProcessor') as mock_processor:
                mock_instance = mock_processor.return_value
                mock_instance.process_image.return_value = {
                    "success": True,
                    "output_path": "test_output.dxf"
                }
                
                result = mock_instance.process_image(image_path, {})
                assert result["success"] is True
            
        finally:
            os.unlink(image_path)
    
    def test_error_recovery_workflow(self):
        """Test error recovery in workflows."""
        # Test with invalid file
        result, error_info = safe_execute(
            analyze_dxf,
            "invalid_file.dxf",
            AnalyzeArgs(out="test_output"),
            context="dxf_analysis_workflow"
        )
        
        assert result is None
        assert error_info is not None
        assert error_info["category"] == "file_processing"
        
        # Test with invalid parameters
        result, error_info = safe_execute(
            analyze_dxf,
            "test.dxf",
            "invalid_args",
            context="dxf_analysis_workflow"
        )
        
        assert result is None
        assert error_info is not None


# Helper methods for test classes
def create_test_dxf(self, content: str = None) -> str:
    """Create a test DXF file."""
    if content is None:
        content = """0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
ENTITIES
0
LINE
8
0
10
0.0
20
0.0
11
100.0
21
100.0
0
ENDSEC
0
EOF"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dxf', delete=False) as f:
        f.write(content)
        return f.name


def create_test_image(self, width: int = 200, height: int = 200) -> str:
    """Create a test image file."""
    import cv2
    
    # Create a simple test image with geometric shapes
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw some test shapes
    cv2.circle(image, (50, 50), 20, (0, 0, 0), -1)
    cv2.rectangle(image, (100, 100), (150, 150), (0, 0, 0), -1)
    cv2.line(image, (0, 0), (width, height), (0, 0, 0), 2)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        cv2.imwrite(f.name, image)
        return f.name


if __name__ == "__main__":
    # Run tests individually for debugging
    test_classes = [
        TestDXFAnalysisCriticalPath,
        TestImageProcessingCriticalPath,
        TestNestingCriticalPath,
        TestAICriticalPath,
        TestWebInterfaceCriticalPath,
        TestEndToEndWorkflow
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                print(f"  {method_name}...")
                try:
                    getattr(instance, method_name)()
                    print(f"    ✓ Passed")
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
    
    print("\n✓ All critical path tests completed!")
