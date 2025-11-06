#!/usr/bin/env python3
"""
WJP Image Analyzer Integration Example

This script demonstrates how to integrate the image analyzer into your existing
image-to-DXF conversion workflow.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wjp_analyser.image_analyzer import ImageAnalyzerGate, create_analyzer_gate


def example_integration():
    """Example of how to integrate the analyzer into your workflow."""
    
    print("WJP Image Analyzer Integration Example")
    print("=" * 50)
    
    # Example 1: Basic integration with default settings
    print("\n1. Basic Integration (Default Settings)")
    gate = create_analyzer_gate(min_score=75.0)
    
    # Analyze test images
    test_images = ['test_image.png', 'test_images/simple_shapes.png']
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\nAnalyzing: {image_path}")
            should_proceed, report = gate.analyze_and_decide(image_path, output_dir="output/image_analyzer")
            
            print(f"  Score: {report['score']}/100")
            print(f"  Should proceed: {should_proceed}")
            print(f"  Reason: {report['gate_decision']['reason']}")
            
            if report['suggestions']:
                print(f"  Suggestions: {len(report['suggestions'])}")
                for i, suggestion in enumerate(report['suggestions'][:3], 1):  # Show first 3
                    print(f"    {i}. {suggestion}")
    
    # Example 2: Custom configuration
    print("\n\n2. Custom Configuration")
    custom_config = {
        'px_to_unit': 0.1,  # 0.1 mm per pixel
        'min_spacing_unit': 2.0,  # 2mm minimum spacing
        'min_radius_unit': 1.0,   # 1mm minimum radius
        'max_texture_fft_energy': 0.25,  # More lenient texture detection
    }
    
    custom_gate = create_analyzer_gate(config_dict=custom_config, min_score=60.0)
    
    if Path('test_image.png').exists():
        print(f"\nAnalyzing with custom config: test_image.png")
        should_proceed, report = custom_gate.analyze_and_decide('test_image.png')
        
        print(f"  Score: {report['score']}/100")
        print(f"  Should proceed: {should_proceed}")
        print(f"  Min spacing: {report['manufacturability']['min_spacing_unit']:.2f} mm")
        print(f"  Min radius: {report['manufacturability']['min_radius_unit']:.2f} mm")
    
    # Example 3: Batch processing
    print("\n\n3. Batch Processing Example")
    
    def process_image_batch(image_paths, min_score=75.0):
        """Process a batch of images and categorize results."""
        gate = create_analyzer_gate(min_score=min_score)
        
        results = {
            'ready': [],
            'needs_work': [],
            'failed': []
        }
        
        for image_path in image_paths:
            if not Path(image_path).exists():
                continue
                
            should_proceed, report = gate.analyze_and_decide(image_path)
            
            if should_proceed:
                results['ready'].append((image_path, report['score']))
            elif report['score'] >= 50:
                results['needs_work'].append((image_path, report['score']))
            else:
                results['failed'].append((image_path, report['score']))
        
        return results
    
    batch_results = process_image_batch(test_images, min_score=75.0)
    
    print(f"Ready for conversion: {len(batch_results['ready'])}")
    for path, score in batch_results['ready']:
        print(f"  [OK] {path} (score: {score})")
    
    print(f"Needs work: {len(batch_results['needs_work'])}")
    for path, score in batch_results['needs_work']:
        print(f"  [WARN] {path} (score: {score})")
    
    print(f"Failed analysis: {len(batch_results['failed'])}")
    for path, score in batch_results['failed']:
        print(f"  [FAIL] {path} (score: {score})")
    
    print("\n" + "=" * 50)
    print("Integration complete! Check 'output/image_analyzer/' for saved reports.")


if __name__ == "__main__":
    example_integration()
