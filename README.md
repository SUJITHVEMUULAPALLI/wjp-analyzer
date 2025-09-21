 waterjet-dxf (starter)
DXF validator -> analyzer -> report -> cutting visualization -> G-code generator for waterjet.

This is a **Cursor-ready** starter you can open as a folder and start hacking.

## Quick start
1) Create a fresh Python 3.11+ venv.
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2) Generate a sample DXF and run analysis:
```
python scripts/make_sample_dxf.py samples/medallion_sample.dxf
python -m cli.wjdx analyze samples/medallion_sample.dxf --out out --material "Tan Brown Granite" --thickness 25 --kerf 1.1 --rate-per-m 825
```
3) Generate example G-code (toy post):
```
python -m cli.wjdx gcode samples/medallion_sample.dxf --out out --post generic
```

Outputs:
- `out/report.json`, `out/lengths.csv`
- `out/preview.png`
- `out/program.nc` (toy G-code)

## One-click launcher
- Run `python run_one_click.py` (or double-click `run_one_click.bat` on Windows) to install dependencies, start the Flask UI, and open it in your browser.
- Add `--mode demo` to run the CLI sample pipeline instead; `--skip-install` skips pip install and `--open-preview` opens the generated preview image after the demo.

## What's implemented
- DXF import (LINE, LWPOLYLINE, POLYLINE, CIRCLE, ARC, SPLINE->polyline approx)
- Basic cleanup + polygonization
- Containment-based **outer vs inner** classification (+ layer-name hints)
- Minimal checks (open contours, spacing via buffer overlap, acute angle warning)
- Metrics (internal length, outer length, pierces approx. polygons)
- Simple order (small internals -> larger -> outer last)
- Kerf offset preview (buffer-based)
- JSON/CSV report + PNG viz
- Toy post that emits **G-code** with configurable ON/OFF M-codes

## Roadmap (fill next in Cursor)
- Robust inner-radius check, web/bridge thickness
- Tabs/micro-joints planner
- Per-material feed table and accurate time model
- Real post processors (Flow/OMAX/PathPilot/etc.)

## Cursor tips
- Use "New Task" -> "Refactor" on any module. Keep functions small and add unit tests.
- Add your shop presets in `presets/defaults.yaml`.
- Create a `tests/` folder and run `pytest` as you iterate.
