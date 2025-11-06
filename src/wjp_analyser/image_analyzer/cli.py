#!/usr/bin/env python3
"""
WJP Image Analyzer CLI Tool

Simple command-line interface for testing the image analyzer.
Usage: python wjp_image_analyzer_cli.py <image_path> [--config-file config.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Support running as a module and directly from source tree without relying on 'src' prefix
try:
    from wjp_analyser.image_analyzer.core import analyze_image_for_wjp, AnalyzerConfig
except ImportError:  # fallback if executed with old PYTHONPATH layout
    from .core import analyze_image_for_wjp, AnalyzerConfig


def main():
    parser = argparse.ArgumentParser(description='WJP Image Analyzer CLI')
    parser.add_argument('image_path', help='Path to image file to analyze')
    parser.add_argument('--config-file', help='JSON config file (optional)')
    parser.add_argument('--output', '-o', help='Output JSON file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file '{args.image_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Load config if provided
    config = AnalyzerConfig()
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config_dict = json.load(f)
                config = AnalyzerConfig(**config_dict)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    
    try:
        # Analyze the image
        result = analyze_image_for_wjp(args.image_path, config)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Analysis saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
        # Verbose summary
        if args.verbose:
            print("\n" + "="*50)
            print("ANALYSIS SUMMARY")
            print("="*50)
            print(f"File: {result['file']}")
            print(f"Score: {result['score']}/100")
            print(f"Size: {result['image_original_size']['w']}x{result['image_original_size']['h']}")
            print(f"Contours: {result['topology_preview']['total_contours']} total, {result['topology_preview']['closed_contours']} closed")
            print(f"Skew: {result['orientation']['skew_angle_deg']:.1f}°")
            
            if result['suggestions']:
                print("\nSuggestions:")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    print(f"  {i}. {suggestion}")
            
            # Determine readiness
            if result['score'] >= 75:
                print(f"\n✅ READY for DXF conversion (score: {result['score']})")
            elif result['score'] >= 50:
                print(f"\n⚠️  MODERATE suitability (score: {result['score']}) - review suggestions")
            else:
                print(f"\n❌ POOR suitability (score: {result['score']}) - significant issues")
                
    except Exception as e:
        print(f"Error analyzing image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

