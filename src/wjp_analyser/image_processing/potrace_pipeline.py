from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import ezdxf  # type: ignore
except Exception:  # pragma: no cover
    ezdxf = None  # type: ignore


@dataclass
class PreprocessOptions:
    threshold_type: str = "global"  # "global" or "adaptive"
    threshold_value: int = 180
    adaptive_block_size: int = 21
    adaptive_C: int = 5
    gaussian_blur_ksize: int = 5  # 0 to disable
    use_canny: bool = False
    morph_op: str = "open"  # "open" | "close" | "none"
    morph_ksize: int = 3
    morph_iters: int = 1
    invert: bool = False  # invert binary prior to PBM for Potrace black foreground


def ensure_potrace(potrace_exe: Optional[str] = None) -> Optional[str]:
    """Resolve potrace executable path.

    Precedence:
    1) Explicit potrace_exe argument if provided and exists
    2) PATH lookup
    3) On Windows, WSL 'potrace' if available
    """
    if potrace_exe and os.path.exists(potrace_exe):
        return potrace_exe

    exe = "potrace.exe" if os.name == "nt" else "potrace"
    path = shutil.which(exe)

    # If not found on Windows, try WSL
    if os.name == "nt" and path is None:
        try:
            result = subprocess.run(["wsl", "which", "potrace"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return "wsl"  # Return special marker for WSL
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return path


def preprocess_image(image_path: str, out_dir: Path, opts: PreprocessOptions) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    img = gray.copy()
    if opts.gaussian_blur_ksize and opts.gaussian_blur_ksize >= 3:
        k = int(opts.gaussian_blur_ksize)
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    if opts.threshold_type == "adaptive":
        block = max(3, int(opts.adaptive_block_size) // 2 * 2 + 1)
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(opts.adaptive_C)
        )
    else:
        _, binary = cv2.threshold(img, int(opts.threshold_value), 255, cv2.THRESH_BINARY)

    if opts.use_canny:
        edges = cv2.Canny(img, 50, 150)
        binary = cv2.bitwise_or(binary, edges)

    # Morphology
    if opts.morph_op in ("open", "close") and opts.morph_ksize > 0 and opts.morph_iters > 0:
        k = max(1, int(opts.morph_ksize))
        kernel = np.ones((k, k), np.uint8)
        if opts.morph_op == "open":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=int(opts.morph_iters))
        else:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=int(opts.morph_iters))

    # Optional invert so Potrace treats design as foreground
    if opts.invert:
        binary = cv2.bitwise_not(binary)

    # Save preview and PBM
    preview_png = out_dir / "preprocessed_binary.png"
    cv2.imwrite(str(preview_png), binary)
    pbm_path = out_dir / "temp.pbm"
    cv2.imwrite(str(pbm_path), binary)
    return pbm_path, preview_png


def potrace_vectorize(
    pbm_path: Path,
    out_dir: Path,
    want_svg: bool = True,
    want_dxf: bool = True,
    turdsize: int = 2,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
) -> Tuple[Optional[Path], Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    potrace_bin = ensure_potrace()
    if not potrace_bin:
        raise RuntimeError("Potrace is not installed or not found in PATH. Install it and retry.")

    svg_path = out_dir / "potrace.svg" if want_svg else None
    dxf_path = out_dir / "potrace.dxf" if want_dxf else None

    # Prepare command arguments
    common_args = ["--turdsize", str(int(max(0, turdsize))), "--alphamax", str(float(alphamax)), "--opttolerance", str(float(opttolerance))]
    
    # Handle WSL vs native execution
    if potrace_bin == "wsl":
        # Convert Windows paths to WSL paths
        pbm_wsl_path = str(pbm_path).replace("\\", "/").replace("C:", "/mnt/c")
        svg_wsl_path = str(svg_path).replace("\\", "/").replace("C:", "/mnt/c") if svg_path else None
        dxf_wsl_path = str(dxf_path).replace("\\", "/").replace("C:", "/mnt/c") if dxf_path else None
        
        if svg_path is not None:
            subprocess.run(["wsl", "potrace", "-s", "-o", svg_wsl_path, *common_args, pbm_wsl_path], check=True)
        if dxf_path is not None:
            subprocess.run(["wsl", "potrace", "-b", "dxf", "-o", dxf_wsl_path, *common_args, pbm_wsl_path], check=True)
    else:
        # Native execution
        if svg_path is not None:
            subprocess.run([potrace_bin, "-s", "-o", str(svg_path), *common_args, str(pbm_path)], check=True)
        if dxf_path is not None:
            subprocess.run([potrace_bin, "-b", "dxf", "-o", str(dxf_path), *common_args, str(pbm_path)], check=True)

    return svg_path, dxf_path


def scale_dxf_to_target(dxf_path: Path, target_size_mm: float) -> None:
    if ezdxf is None:
        return
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    # compute bbox
    try:
        import math

        minx = miny = 1e18
        maxx = maxy = -1e18
        for e in msp:
            try:
                box = e.bbox()
                if box:
                    (bxmin, bymin, _), (bxmax, bymax, _) = box.extmin, box.extmax
                else:
                    continue
            except Exception:
                # Fallback approximate from entity type
                try:
                    verts = list(e.get_points("xy"))  # type: ignore
                    xs = [v[0] for v in verts]
                    ys = [v[1] for v in verts]
                    bxmin, bymin, bxmax, bymax = min(xs), min(ys), max(xs), max(ys)
                except Exception:
                    continue
            minx = min(minx, bxmin)
            miny = min(miny, bymin)
            maxx = max(maxx, bxmax)
            maxy = max(maxy, bymax)
        w = max(1e-6, maxx - minx)
        h = max(1e-6, maxy - miny)
        scale = float(target_size_mm) / max(w, h)
        if not math.isfinite(scale) or scale <= 0:
            return
        sx, sy = scale, scale
        # translate to origin then scale
        dx, dy = -minx, -miny
        for e in list(msp):
            try:
                e.transform(ezdxf.math.Matrix44.translate(dx, dy))
                e.transform(ezdxf.math.Matrix44.scale(sx, sy, 1.0))
            except Exception:
                pass
        doc.saveas(str(dxf_path))
    except Exception:
        # if bbox fails, skip scaling
        return


def preprocess_and_vectorize(
    image_path: str,
    out_dir: Path,
    target_size_mm: float = 1000.0,
    threshold_type: str = "global",
    threshold_value: int = 180,
    adaptive_block_size: int = 21,
    adaptive_C: int = 5,
    gaussian_blur_ksize: int = 5,
    use_canny: bool = False,
    morph_op: str = "open",
    morph_ksize: int = 3,
    morph_iters: int = 1,
    output_route: str = "potrace_dxf",  # "potrace_dxf" | "svg_then_dxf"
    simplify_tolerance: float = 0.0,
    invert: bool = False,
    potrace_turdsize: int = 2,
    potrace_alphamax: float = 1.0,
    potrace_opttolerance: float = 0.2,
    potrace_exe: Optional[str] = None,
) -> Tuple[Optional[Path], Path, Optional[Path]]:
    opts = PreprocessOptions(
        threshold_type=threshold_type,
        threshold_value=int(threshold_value),
        adaptive_block_size=int(adaptive_block_size),
        adaptive_C=int(adaptive_C),
        gaussian_blur_ksize=int(gaussian_blur_ksize),
        use_canny=bool(use_canny),
        morph_op=morph_op,
        morph_ksize=int(morph_ksize),
        morph_iters=int(morph_iters),
        invert=bool(invert),
    )
    pbm_path, preview_png = preprocess_image(image_path, out_dir, opts)
    want_svg = output_route != "potrace_dxf"
    want_dxf = True
    # If an explicit potrace exe is provided, ensure_potrace will be called earlier
    # by temporarily injecting into PATH for this call when needed.
    if potrace_exe and os.path.exists(potrace_exe):
        # Prepend directory to PATH for subprocess calls
        prev_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.path.dirname(potrace_exe) + os.pathsep + prev_path
    try:
        svg_path, dxf_path = potrace_vectorize(
        pbm_path,
        out_dir,
        want_svg=want_svg,
        want_dxf=want_dxf,
        turdsize=int(potrace_turdsize),
        alphamax=float(potrace_alphamax),
        opttolerance=float(potrace_opttolerance),
        )
    finally:
        if potrace_exe and os.path.exists(potrace_exe):
            os.environ["PATH"] = prev_path

    # SVG -> DXF route with optional simplification
    if output_route == "svg_then_dxf":
        try:
            out_dxf = out_dir / "svg_simplified.dxf"
            built = svg_to_dxf_with_simplify(
                svg_path=svg_path,
                out_dxf=out_dxf,
                simplify_tolerance=float(simplify_tolerance),
            )
            if built:
                dxf_path = out_dxf
        except Exception:
            # leave dxf_path as None if building fails
            pass

    if dxf_path is not None:
        scale_dxf_to_target(dxf_path, target_size_mm)

    return dxf_path, preview_png, svg_path


def _collect_dxf_polylines(doc) -> list[list[tuple[float, float]]]:
    """Extract polyline-like sequences from DXF document modelspace."""
    msp = doc.modelspace()
    polys: list[list[tuple[float, float]]] = []
    # LWPOLYLINE
    for e in msp.query("LWPOLYLINE"):
        try:
            pts = [(v[0], v[1]) for v in e.get_points("xy")]
            if len(pts) >= 2:
                polys.append(pts)
        except Exception:
            continue
    # LINE
    for e in msp.query("LINE"):
        try:
            pts = [
                (float(e.dxf.start.x), float(e.dxf.start.y)),
                (float(e.dxf.end.x), float(e.dxf.end.y)),
            ]
            polys.append(pts)
        except Exception:
            continue
    # ARC -> sample to polyline
    for e in msp.query("ARC"):
        try:
            import math

            cx, cy = float(e.dxf.center.x), float(e.dxf.center.y)
            r = float(e.dxf.radius)
            a1 = math.radians(float(e.dxf.start_angle))
            a2 = math.radians(float(e.dxf.end_angle))
            if a2 < a1:
                a2 += 2 * math.pi
            steps = 64
            ts = np.linspace(a1, a2, steps)
            pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in ts]
            if len(pts) >= 2:
                polys.append(pts)
        except Exception:
            continue
    return polys


def simplify_dxf_inplace(dxf_path: Path, tolerance_mm: float) -> bool:
    """Simplify DXF polylines in-place using Shapely if available, else RDP fallback.

    Returns True if successful, False otherwise.
    """
    if tolerance_mm <= 0:
        return False
    if ezdxf is None:
        return False
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception:
        return False

    msp = doc.modelspace()
    polys = _collect_dxf_polylines(doc)

    used_shapely = False
    simplified: list[list[tuple[float, float]]] = []
    try:
        from shapely.geometry import LineString  # type: ignore

        used_shapely = True
        for pts in polys:
            ls = LineString(pts)
            ls2 = ls.simplify(tolerance_mm, preserve_topology=True)
            simplified.append(list(ls2.coords))
    except Exception:
        used_shapely = False

    if not used_shapely:
        # Fallback to RDP per polyline
        for pts in polys:
            simplified.append(_rdp(pts, tolerance_mm))

    # Clear and rewrite as LWPOLYLINEs
    try:
        # Remove existing entities that we simplified
        for e in list(msp.query("LWPOLYLINE LINE ARC")):
            try:
                msp.delete_entity(e)
            except Exception:
                pass
        for pts in simplified:
            if len(pts) < 2:
                continue
            closed = False
            if len(pts) >= 3:
                (x1, y1) = pts[0]
                (x2, y2) = pts[-1]
                if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
                    closed = True
            msp.add_lwpolyline(pts, format="xy", close=closed)
        doc.saveas(str(dxf_path))
        return True
    except Exception:
        return False


def compute_dxf_complexity(dxf_path: Path) -> tuple[int, int, float]:
    """Return (num_entities, total_nodes, total_length_mm) for a DXF file."""
    if ezdxf is None:
        return (0, 0, 0.0)
    try:
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
    except Exception:
        return (0, 0, 0.0)

    import math
    ent_count = 0
    node_count = 0
    total_len = 0.0

    for e in msp:
        try:
            if e.dxftype() == "LWPOLYLINE":
                pts = [(v[0], v[1]) for v in e.get_points("xy")]
                ent_count += 1
                node_count += len(pts)
                for i in range(1, len(pts)):
                    x1, y1 = pts[i - 1]
                    x2, y2 = pts[i]
                    total_len += math.hypot(x2 - x1, y2 - y1)
            elif e.dxftype() == "LINE":
                ent_count += 1
                node_count += 2
                x1, y1 = float(e.dxf.start.x), float(e.dxf.start.y)
                x2, y2 = float(e.dxf.end.x), float(e.dxf.end.y)
                total_len += math.hypot(x2 - x1, y2 - y1)
            elif e.dxftype() == "ARC":
                ent_count += 1
                node_count += 64  # sampled equivalent
                r = float(e.dxf.radius)
                a1 = math.radians(float(e.dxf.start_angle))
                a2 = math.radians(float(e.dxf.end_angle))
                if a2 < a1:
                    a2 += 2 * math.pi
                total_len += abs(a2 - a1) * r
        except Exception:
            continue

    return (ent_count, node_count, total_len)


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3 or epsilon <= 0:
        return points
    import math

    def perp_distance(pt, a, b):
        (x, y) = pt
        (x1, y1) = a
        (x2, y2) = b
        if (x1, y1) == (x2, y2):
            return math.hypot(x - x1, y - y1)
        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        den = math.hypot(y2 - y1, x2 - x1)
        return num / den

    # Find the point with the maximum distance
    dmax = 0.0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = perp_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        rec1 = _rdp(points[: index + 1], epsilon)
        rec2 = _rdp(points[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [points[0], points[-1]]


def svg_to_dxf_with_simplify(svg_path: Optional[Path], out_dxf: Path, simplify_tolerance: float = 0.0) -> bool:
    if svg_path is None or not svg_path.exists():
        return False
    try:
        from svgpathtools import svg2paths  # type: ignore
    except Exception:
        return False

    try:
        paths, attrs = svg2paths(str(svg_path))
    except Exception:
        return False

    # Sample SVG paths to polylines
    polylines: list[list[tuple[float, float]]] = []
    import numpy as np

    for path in paths:
        pts: list[tuple[float, float]] = []
        # choose sampling density based on path length
        try:
            total_len = float(path.length(error=1e-2))
        except Exception:
            total_len = 100.0
        segments = max(50, int(total_len / 5.0))
        for s in path:
            # adaptive samples per segment
            n = max(10, min(segments, 200))
            for t in np.linspace(0.0, 1.0, n, endpoint=True):
                z = s.point(t)
                pts.append((float(z.real), float(z.imag)))
        if len(pts) >= 2:
            polylines.append(pts)

    # Simplify polylines
    simplified: list[list[tuple[float, float]]] = []
    used_shapely = False
    if simplify_tolerance > 0:
        try:
            from shapely.geometry import LineString  # type: ignore

            used_shapely = True
            for pts in polylines:
                ls = LineString(pts)
                ls2 = ls.simplify(simplify_tolerance, preserve_topology=True)
                simplified.append(list(ls2.coords))
        except Exception:
            used_shapely = False
    if simplify_tolerance > 0 and not used_shapely:
        for pts in polylines:
            simplified.append(_rdp(pts, simplify_tolerance))
    if simplify_tolerance <= 0:
        simplified = polylines

    # Write DXF
    if ezdxf is None:
        return False
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for pts in simplified:
        if len(pts) < 2:
            continue
        # Close if near equal
        closed = False
        if len(pts) >= 3:
            (x1, y1) = pts[0]
            (x2, y2) = pts[-1]
            if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
                closed = True
        msp.add_lwpolyline(pts, format="xy", close=closed)
    doc.saveas(str(out_dxf))
    return True
