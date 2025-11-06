from __future__ import annotations

import argparse
import os
from wjp_analyser.object_management import export_layered_dxf


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify LWPOLYLINEs into layers and export a new DXF")
    parser.add_argument("input", help="Path to input DXF file")
    parser.add_argument("output", nargs="?", help="Path to output layered DXF file")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else os.path.splitext(input_path)[0] + "_layered.dxf"
    export_layered_dxf(input_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


