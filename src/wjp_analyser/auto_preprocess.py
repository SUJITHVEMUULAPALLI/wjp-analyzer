#!/usr/bin/env python3
"""
Auto Image Pre-Processor for Image to DXF Pipelines
--------------------------------------------------

This module analyzes a black-on-white design image and automatically
chooses the right pre-processing path to yield a clean, vector-friendly
bitmap. It then (optionally) runs Potrace to SVG and Inkscape to DXF,
and can post-simplify the DXF geometry.

Designed for WJP Analyzer with a typical scale like 10 px = 1 mm.

Dependencies (pip):
    pip install opencv-python scikit-image numpy ezdxf shapely

Optional CLIs (installed on PATH if you want auto vectorization):
    - potrace
    - inkscape

Usage (CLI):
    python auto_preprocess.py --input input.png --outdir ./out --scale-px-per-mm 10 \
        --run-potrace --export-dxf --dxf-simplify-mm 0.5

In code:
    from auto_preprocess import auto_process
    result = auto_process("input.png", outdir="out", scale_px_per_mm=10, run_potrace_flag=True)
"""
import os
import sys
import json
import argparse
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

# Optional imports guarded (not strictly required unless you enable certain paths)
try:
    from skimage.morphology import skeletonize
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

try:
    import ezdxf  # noqa: F401
    HAVE_EZDXF = True
except Exception:
    HAVE_EZDXF = False

try:
    from shapely.geometry import LineString  # noqa: F401
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False


@dataclass
class AutoConfig:
    # Thresholds in pixels (before scaling)
    # If mean stroke width > skeletonize_threshold_px -> use skeletonization
    skeletonize_threshold_px: float = 6.0
    # If 3 < width <= 6 -> light simplify (blur + threshold)
    light_simplify_min_px: float = 3.0

    # Morphological clean kernel size (pixels)
    morph_kernel: int = 3

    # Simplifications / tolerances
    # DXF post-simplification tolerance in mm
    dxf_simplify_tolerance_mm: float = 0.5

    # Default Potrace args (overridden adaptively)
    potrace_args_skeleton: str = "--turdsize 25 --opttolerance 1.5 --longcurve"
    potrace_args_light: str = "--turdsize 10 --opttolerance 1.0"
    potrace_args_none: str = "--turdsize 5 --opttolerance 0.5"

    # Binarization threshold (0-255). If None, use Otsu.
    binarize_threshold: Optional[int] = 200

    # Debug artifacts
    save_debug: bool = True


@dataclass
class AutoResult:
    input_path: str
    outdir: str
    cleaned_png: Optional[str]
    mode: str
    mean_thickness_px: float
    median_thickness_px: float
    p95_thickness_px: float
    potrace_svg: Optional[str] = None
    dxf_path: Optional[str] = None
    dxf_simplified_path: Optional[str] = None
    logs: Optional[Dict] = None


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _to_binary_inv(gray: np.ndarray, threshold: Optional[int]) -> np.ndarray:
    """Return binary with foreground=255 (black strokes become white foreground) via inversion."""
    if threshold is None:
        # Otsu threshold
        _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bin_ = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # bin_ now is 0/255 with white for background; invert to get strokes=255
    bin_inv = cv2.bitwise_not(bin_)
    return bin_inv


def estimate_line_thickness(binary_inv: np.ndarray) -> Tuple[float, float, float]:
    """Estimate mean/median/95th percentile stroke thickness (in pixels).
    Input: binary_inv with strokes=255, background=0.
    """
    # Distance transform inside strokes (white regions)
    dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 3)
    # Only consider positive stroke pixels
    vals = dist[dist > 0] * 2.0  # radius -> diameter (thickness in px)
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    mean_t = float(vals.mean())
    med_t = float(np.median(vals))
    p95_t = float(np.percentile(vals, 95))
    return mean_t, med_t, p95_t


def decide_mode(mean_thickness_px: float, cfg: AutoConfig) -> str:
    if mean_thickness_px > cfg.skeletonize_threshold_px:
        return "skeletonize"
    if mean_thickness_px > cfg.light_simplify_min_px:
        return "light_simplify"
    return "none"


def skeletonize_image(binary_inv: np.ndarray) -> np.ndarray:
    """Skeletonize strokes to 1-pixel centerlines. Returns inverted binary (white background)."""
    if HAVE_SKIMAGE:
        skel = skeletonize(binary_inv > 0)
        skel_u8 = (skel.astype(np.uint8) * 255)
        out = cv2.bitwise_not(skel_u8)  # back to black lines on white background
        return out
    # Fallback using OpenCV thinning (requires ximgproc)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        # cv2.ximgproc expects white foreground, so use binary_inv directly
        thinned = np.zeros_like(binary_inv)
        cv2.ximgproc.thinning(binary_inv, thinned, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        out = cv2.bitwise_not(thinned)
        return out
    raise RuntimeError("Skeletonization requested but neither scikit-image nor cv2.ximgproc.thinning is available.")


def light_simplify(binary_inv: np.ndarray) -> np.ndarray:
    """Light blur + threshold to smooth edges; returns inverted binary (white background)."""
    blurred = cv2.GaussianBlur(binary_inv, (3, 3), 0)
    _, rebin = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    out = cv2.bitwise_not(rebin)
    return out


def clean_binary(binary_inv: np.ndarray, kernel_size: int) -> np.ndarray:
    """Remove small specks with morphological opening; keep strokes robust."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    return opened


def adaptive_potrace_args(mode: str, cfg: AutoConfig) -> str:
    if mode == "skeletonize":
        return cfg.potrace_args_skeleton
    if mode == "light_simplify":
        return cfg.potrace_args_light
    return cfg.potrace_args_none


def run_cmd(cmd: str) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def run_potrace(png_path: str, svg_path: str, mode: str, cfg: AutoConfig, potrace_exe: Optional[str] = None) -> Tuple[bool, str]:
    """Run potrace natively or via WSL if available."""
    args = adaptive_potrace_args(mode, cfg)
    try:
        from pathlib import Path
        from wjp_analyser.image_processing.potrace_pipeline import ensure_potrace  # reuse detection
        potrace_bin = ensure_potrace(potrace_exe)
    except Exception:
        potrace_bin = None

    if not potrace_bin:
        return False, "Potrace not found in PATH. Install Potrace or enable WSL."

    if potrace_bin == "wsl":
        # Convert Windows paths to WSL paths
        def to_wsl(p: str) -> str:
            return str(Path(p)).replace("\\", "/").replace("C:", "/mnt/c")

        cmd = f'wsl potrace -s -o "{to_wsl(svg_path)}" {args} "{to_wsl(png_path)}"'
    else:
        cmd = f'"{potrace_bin}" -s -o "{svg_path}" {args} "{png_path}"'

    rc, out, err = run_cmd(cmd)
    ok = rc == 0 and os.path.exists(svg_path)
    log = out + ("\n" + err if err else "")
    return ok, log


def export_dxf_with_inkscape(svg_path: str, dxf_path: str) -> Tuple[bool, str]:
    """Export DXF using Inkscape CLI if available."""
    import shutil
    inkscape_bin = shutil.which("inkscape")
    if not inkscape_bin:
        return False, "Inkscape CLI not found in PATH; skip DXF export."
    cmd = f'"{inkscape_bin}" "{svg_path}" --export-type="dxf" --export-filename="{dxf_path}"'
    rc, out, err = run_cmd(cmd)
    ok = rc == 0 and os.path.exists(dxf_path)
    log = out + ("\n" + err if err else "")
    return ok, log


def simplify_dxf(input_dxf: str, output_dxf: str, tolerance_mm: float) -> Tuple[bool, str]:
    if not (HAVE_EZDXF and HAVE_SHAPELY):
        return False, "ezdxf or shapely not available; skipping DXF simplification."
    import ezdxf
    from shapely.geometry import LineString

    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()

    changed = 0
    for pl in list(msp.query("LWPOLYLINE")):
        pts = [(p[0], p[1]) for p in pl]
        if len(pts) < 3:
            continue
        ls = LineString(pts).simplify(tolerance=tolerance_mm, preserve_topology=True)
        coords = list(ls.coords)
        if len(coords) < len(pts):
            try:
                pl.set_points(coords)
                changed += 1
            except Exception:
                pass

    doc.saveas(output_dxf)
    return True, f"Simplified polylines: {changed}"


def auto_process(
    input_path: str,
    outdir: str = "out",
    scale_px_per_mm: float = 10.0,
    cfg: Optional[AutoConfig] = None,
    run_potrace_flag: bool = False,
    export_dxf_flag: bool = False,
    dxf_simplify_mm: Optional[float] = None,
    potrace_exe: Optional[str] = None,
) -> AutoResult:
    """
    End-to-end automatic pre-processing.
    Returns AutoResult with file paths and metrics.
    """
    cfg = cfg or AutoConfig()
    os.makedirs(outdir, exist_ok=True)

    gray = _imread_gray(input_path)
    bin_inv = _to_binary_inv(gray, cfg.binarize_threshold)
    bin_inv = clean_binary(bin_inv, cfg.morph_kernel)

    mean_t, med_t, p95_t = estimate_line_thickness(bin_inv)
    mode = decide_mode(mean_t, cfg)

    if cfg.save_debug:
        debug_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(debug_vis, f"mean_thickness_px={mean_t:.2f}, mode={mode}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(outdir, "debug_input_gray.png"), debug_vis)
        cv2.imwrite(os.path.join(outdir, "debug_binary_inv.png"), bin_inv)

    # Decide processing branch
    if mode == "skeletonize":
        processed = skeletonize_image(bin_inv)
    elif mode == "light_simplify":
        processed = light_simplify(bin_inv)
    else:
        # Already fairly thin; invert back to black lines on white
        processed = cv2.bitwise_not(bin_inv)

    cleaned_png = os.path.join(outdir, "auto_cleaned.png")
    cv2.imwrite(cleaned_png, processed)

    result = AutoResult(
        input_path=input_path,
        outdir=outdir,
        cleaned_png=cleaned_png,
        mode=mode,
        mean_thickness_px=mean_t,
        median_thickness_px=med_t,
        p95_thickness_px=p95_t,
        logs={},
    )

    # Optional: vectorize via Potrace
    if run_potrace_flag:
        svg_path = os.path.join(outdir, "vectorized.svg")
        ok, log = run_potrace(cleaned_png, svg_path, mode, cfg, potrace_exe=potrace_exe)
        result.logs["potrace"] = log
        result.potrace_svg = svg_path if ok else None

        # Optional: export to DXF with Inkscape
        if ok and export_dxf_flag:
            dxf_path = os.path.join(outdir, "vectorized.dxf")
            ok2, log2 = export_dxf_with_inkscape(svg_path, dxf_path)
            result.logs["inkscape"] = log2
            result.dxf_path = dxf_path if ok2 else None

            # Optional: post-simplify DXF
            if ok2 and dxf_simplify_mm:
                simp_path = os.path.join(outdir, "vectorized_simplified.dxf")
                ok3, log3 = simplify_dxf(dxf_path, simp_path, dxf_simplify_mm)
                result.logs["dxf_simplify"] = log3
                result.dxf_simplified_path = simp_path if ok3 else None

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto pre-processor for Image → DXF pipelines.")
    p.add_argument("--input", required=True, help="Path to input image (black lines on white background).")
    p.add_argument("--outdir", default="out", help="Output directory.")
    p.add_argument("--scale-px-per-mm", type=float, default=10.0, help="Image scale (pixels per mm).")
    p.add_argument("--run-potrace", action="store_true", help="Run Potrace to generate SVG.")
    p.add_argument("--export-dxf", action="store_true", help="Export DXF via Inkscape (requires --run-potrace).")
    p.add_argument("--dxf-simplify-mm", type=float, default=None, help="Simplify DXF with this tolerance (mm).")
    p.add_argument("--no-debug", action="store_true", help="Do not save debug images.")
    p.add_argument("--skeletonize-threshold-px", type=float, default=6.0, help="Mean stroke width above which skeletonization is applied.")
    p.add_argument("--light-simplify-min-px", type=float, default=3.0, help="Lower bound for light simplification mode.")
    p.add_argument("--binarize-threshold", type=int, default=200, help="Fixed threshold (0-255). Use -1 for Otsu.")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = AutoConfig(
        skeletonize_threshold_px=args.skeletonize_threshold_px,
        light_simplify_min_px=args.light_simplify_min_px,
        binarize_threshold=None if args.binarize_threshold < 0 else args.binarize_threshold,
        save_debug=not args.no_debug,
    )

    res = auto_process(
        input_path=args.input,
        outdir=args.outdir,
        scale_px_per_mm=args.scale_px_per_mm,
        cfg=cfg,
        run_potrace_flag=args.run_potrace,
        export_dxf_flag=args.export_dxf,
        dxf_simplify_mm=args.dxf_simplify_mm,
    )

    print(json.dumps(asdict(res), indent=2))


if __name__ == "__main__":
    main()


