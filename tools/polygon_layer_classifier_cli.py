from __future__ import annotations

import argparse
import os
import sys

from wjp_analyser.object_management.polygon_layer_classifier import create_layers


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify LWPOLYLINEs into layers and export a new DXF")
    parser.add_argument("input", help="Path to input DXF file")
    parser.add_argument("output", nargs="?", help="Path to output layered DXF file")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return 2

    output_path = os.path.abspath(args.output) if args.output else os.path.splitext(input_path)[0] + "_layered.dxf"

    try:
        create_layers(input_path, output_path)
        print(f"Saved layered DXF: {output_path}")
        return 0
    except Exception as exc:
        print(f"Failed to create layered DXF: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


