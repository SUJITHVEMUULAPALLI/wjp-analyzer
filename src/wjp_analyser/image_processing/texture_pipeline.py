from __future__ import annotations

"""
Texture-aware Image → DXF pipeline
=================================

Implements a multi-stage process:
- Phase 1: Preprocessing (grayscale, resize to working area, normalization, adaptive threshold, morphology)
- Phase 2: Texture features and tile-wise classification (edges/stipple/hatch/contour)
- Phase 3: Vectorization per texture type
- Phase 4: DXF assembly with layers

Notes:
- Potrace is used when available for robust contour vectorization. Falls back to OpenCV findContours otherwise.
- Stippling uses Poisson-disk sampling with density driven by local intensity.
- Hatching generates a rotated line-stripe bitmap masked to the region, then vectorizes.
- Contour bands are derived from multi-level thresholds.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import ezdxf  # type: ignore
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon

try:
    from skimage.filters import gabor
except Exception:  # pragma: no cover
    gabor = None  # type: ignore

from .potrace_pipeline import (
    ensure_potrace,
    preprocess_image as potrace_preprocess_image,  # reuse if useful
)


# -----------------------------
# Phase 1 — Preprocessing
# -----------------------------

@dataclass
class PreprocessParams:
    working_px: int = 1000            # maps to 1000 mm canvas
    adaptive_block: int = 35
    adaptive_C: int = 2
    morph_kernel: int = 3
    morph_iters: int = 1


def _to_grayscale_and_resize(image_path: str, working_px: int) -> Tuple[np.ndarray, float]:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not load image: {image_path}")
    h, w = gray.shape[:2]
    scale = working_px / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # pad to square canvas
    pad_top = (working_px - new_h) // 2
    pad_bottom = working_px - new_h - pad_top
    pad_left = (working_px - new_w) // 2
    pad_right = working_px - new_w - pad_left
    canvas = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
    return canvas, scale


def preprocess(image_path: str, pp: PreprocessParams) -> Dict[str, np.ndarray]:
    gray, _ = _to_grayscale_and_resize(image_path, pp.working_px)
    norm = cv2.equalizeHist(gray)
    block = max(3, int(pp.adaptive_block) // 2 * 2 + 1)
    thresh = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(pp.adaptive_C)
    )
    kernel = np.ones((max(1, pp.morph_kernel), max(1, pp.morph_kernel)), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=int(pp.morph_iters))

    # Texture decomposition
    edges = cv2.Canny(norm, 60, 180)
    sobelx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lap = cv2.Laplacian(norm, cv2.CV_32F, ksize=3)
    hf = cv2.convertScaleAbs(lap)
    lf = cv2.GaussianBlur(norm, (11, 11), 0)

    return {
        "gray": gray,
        "norm": norm,
        "thresh": thresh,
        "clean": clean,
        "edges": edges,
        "grad_mag": grad_mag,
        "hf": hf,
        "lf": lf,
    }


# -----------------------------
# Phase 2 — Texture Classification
# -----------------------------

@dataclass
class TextureClassifyParams:
    tile: int = 32
    clusters: int = 4  # expecting Edge/Stipple/Hatch/Contour


def _tile_stats(imgs: Dict[str, np.ndarray], tile: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = imgs["gray"].shape
    feats: List[List[float]] = []
    for iy in range(0, h, tile):
        for ix in range(0, w, tile):
            y2 = min(h, iy + tile)
            x2 = min(w, ix + tile)
            sl = slice(iy, y2), slice(ix, x2)
            # features
            edge_density = float(np.mean(imgs["edges"][sl] > 0))
            hf_mean = float(np.mean(imgs["hf"][sl]))
            grad_mean = float(np.mean(imgs["grad_mag"][sl]))
            var = float(np.var(imgs["norm"][sl]))
            directionality = 0.0
            if gabor is not None:
                # a simple gabor at a fixed frequency and best of a few angles
                angles = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
                responses = []
                for th in angles:
                    real, imag = gabor(imgs["norm"][sl], frequency=0.2, theta=th)
                    responses.append(float(np.mean(real ** 2 + imag ** 2)))
                # directionality as peak - mean
                directionality = max(responses) - float(np.mean(responses))
            feats.append([edge_density, hf_mean, grad_mean, var, directionality])
    feat_arr = np.array(feats, dtype=np.float32)
    return feat_arr, (h, w)


def _kmeans_labels(feat_arr: np.ndarray, k: int):
    # Use OpenCV kmeans to avoid sklearn dependency
    if k <= 1:
        centers = np.mean(feat_arr, axis=0, keepdims=True)
        return np.zeros((feat_arr.shape[0],), dtype=np.int32), centers
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    attempts = 5
    compactness, labels, centers = cv2.kmeans(feat_arr, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(-1), centers


def _map_clusters_to_texture(centers: np.ndarray) -> Dict[int, str]:
    # Heuristic mapping: edge_density high → edges; directionality high → hatch;
    # hf_mean high with var moderate → stipple; everything smooth → contour
    mapping: Dict[int, str] = {}
    if centers is None or len(centers) == 0:
        return mapping
    # normalize features per dimension to compare
    c = centers.copy()
    c = (c - c.min(axis=0)) / np.maximum(1e-9, (c.max(axis=0) - c.min(axis=0)))
    # indices per heuristic
    edge_idx = int(np.argmax(c[:, 0]))
    hatch_idx = int(np.argmax(c[:, 4]))
    # For stipple, look for high HF and grad, moderate var
    stipple_idx = int(np.argmax(0.6 * c[:, 1] + 0.4 * c[:, 2] - 0.2 * c[:, 3]))
    # Contour: lowest combined energy
    contour_idx = int(np.argmin(0.5 * c[:, 0] + 0.5 * c[:, 1] + 0.5 * c[:, 2] + 0.2 * c[:, 4]))

    # ensure uniqueness by priority resolution
    chosen = []
    for idx, name in [(edge_idx, "edges"), (hatch_idx, "hatch"), (stipple_idx, "stipple"), (contour_idx, "contour")]:
        if idx in chosen:
            # pick next best not used
            scores = {
                "edges": c[:, 0],
                "hatch": c[:, 4],
                "stipple": 0.6 * c[:, 1] + 0.4 * c[:, 2] - 0.2 * c[:, 3],
                "contour": -(0.5 * c[:, 0] + 0.5 * c[:, 1] + 0.5 * c[:, 2] + 0.2 * c[:, 4]),
            }[name]
            order = np.argsort(-scores)  # descending
            for j in order:
                if j not in chosen:
                    idx = int(j)
                    break
        mapping[idx] = name
        chosen.append(idx)
    # fallback for any remaining clusters
    for i in range(len(c)):
        if i not in mapping:
            mapping[i] = "contour"
    return mapping


def classify_textures(imgs: Dict[str, np.ndarray], tp: TextureClassifyParams) -> Dict[str, np.ndarray]:
    """Return binary masks per texture type keyed by name."""
    feat_arr, (h, w) = _tile_stats(imgs, tp.tile)
    labels, centers = _kmeans_labels(feat_arr, tp.clusters)
    mapping = _map_clusters_to_texture(centers)

    # build masks per tile
    masks = {name: np.zeros((h, w), dtype=np.uint8) for name in ["edges", "stipple", "hatch", "contour"]}
    ti = 0
    for iy in range(0, h, tp.tile):
        for ix in range(0, w, tp.tile):
            y2 = min(h, iy + tp.tile)
            x2 = min(w, ix + tp.tile)
            label = int(labels[ti]) if ti < len(labels) else 0
            name = mapping.get(label, "contour")
            masks[name][iy:y2, ix:x2] = 255
            ti += 1
    return masks


# -----------------------------
# Phase 3 — Vectorization primitives
# -----------------------------

def _vectorize_binary_mask(mask: np.ndarray, out_dir: Path, basename: str) -> Optional[Path]:
    """Vectorize a binary mask using potrace if available, else OpenCV contours.

    Returns path to DXF file with polylines.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    potrace_bin = ensure_potrace()
    dxf_path = out_dir / f"{basename}.dxf"
    if potrace_bin:
        # write PBM and call potrace
        pbm = out_dir / f"{basename}.pbm"
        cv2.imwrite(str(pbm), mask)
        import subprocess

        subprocess.run([potrace_bin, "-b", "dxf", "-o", str(dxf_path), str(pbm)], check=True)
        return dxf_path
    # fallback: OpenCV contours → DXF
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) >= 2:
            msp.add_lwpolyline([(float(x), float(y)) for (x, y) in pts], close=True)
    doc.saveas(str(dxf_path))
    return dxf_path


def _draw_stripe_pattern(size: int, spacing_px: int, angle_deg: float) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    # draw white stripes on black background
    rad = math.radians(angle_deg % 180.0)
    # compute a direction vector perpendicular to stripes
    dx = math.cos(rad)
    dy = math.sin(rad)
    # For each offset along perpendicular, draw a line along (dx, dy)
    max_len = int(math.hypot(size, size)) + 2
    for k in range(-size * 2, size * 2, max(1, spacing_px)):
        # a point on the perpendicular offset
        cx = size / 2 + k * (-dy)
        cy = size / 2 + k * (dx)
        # line direction along stripes is (dx, dy)
        x1 = int(round(cx - dx * max_len))
        y1 = int(round(cy - dy * max_len))
        x2 = int(round(cx + dx * max_len))
        y2 = int(round(cy + dy * max_len))
        cv2.line(img, (x1, y1), (x2, y2), color=255, thickness=1)
    return img


def _vectorize_stripes_masked(mask: np.ndarray, spacing_px: int, angle_deg: float, out_dir: Path, basename: str) -> Optional[Path]:
    stripes = _draw_stripe_pattern(mask.shape[0], max(1, int(spacing_px)), angle_deg)
    hatched = cv2.bitwise_and(stripes, mask)
    return _vectorize_binary_mask(hatched, out_dir, basename)


def _poisson_disk_points(mask: np.ndarray, min_dist_px: float, max_points: int = 50000) -> List[Tuple[float, float]]:
    """Simple Poisson-disk sampling using grid acceleration (Bridson).
    Returns a list of (x, y) in image coords where mask>0.
    """
    H, W = mask.shape
    if min_dist_px <= 1:
        min_dist_px = 1.0
    cell = min_dist_px / math.sqrt(2)
    grid_w = int(math.ceil(W / cell))
    grid_h = int(math.ceil(H / cell))
    grid = -np.ones((grid_h, grid_w), dtype=np.int32)
    pts: List[Tuple[float, float]] = []
    active: List[int] = []

    def fits(x: float, y: float) -> bool:
        if x < 0 or y < 0 or x >= W or y >= H:
            return False
        if mask[int(y), int(x)] == 0:
            return False
        gx = int(x / cell)
        gy = int(y / cell)
        r2 = min_dist_px * min_dist_px
        for yy in range(max(0, gy - 2), min(grid_h, gy + 3)):
            for xx in range(max(0, gx - 2), min(grid_w, gx + 3)):
                idx = grid[yy, xx]
                if idx >= 0:
                    px, py = pts[idx]
                    if (px - x) * (px - x) + (py - y) * (py - y) < r2:
                        return False
        return True

    # seed center-ish within mask
    for _ in range(1000):
        sx = np.random.uniform(0, W)
        sy = np.random.uniform(0, H)
        if fits(sx, sy):
            pts.append((sx, sy))
            gx = int(sx / cell)
            gy = int(sy / cell)
            grid[gy, gx] = 0
            active.append(0)
            break
    if not active:
        return pts

    k = 30
    while active and len(pts) < max_points:
        i = int(np.random.choice(active))
        ox, oy = pts[i]
        found = False
        for _ in range(k):
            r = np.random.uniform(min_dist_px, 2 * min_dist_px)
            a = np.random.uniform(0, 2 * math.pi)
            nx = ox + r * math.cos(a)
            ny = oy + r * math.sin(a)
            if fits(nx, ny):
                pts.append((nx, ny))
                gx = int(nx / cell)
                gy = int(ny / cell)
                grid[gy, gx] = len(pts) - 1
                active.append(len(pts) - 1)
                found = True
                break
        if not found:
            active.remove(i)
    return pts


def _add_circles(msp, points: List[Tuple[float, float]], radius_px: float, layer: str = "LAYER_STIPPLE") -> int:
    count = 0
    r = float(max(0.1, radius_px))
    for (x, y) in points:
        try:
            e = msp.add_circle((float(x), float(y), 0.0), radius=r)
            try:
                e.dxf.layer = layer
            except Exception:
                pass
            count += 1
        except Exception:
            continue
    return count


# -----------------------------
# Phase 4 — Assembly
# -----------------------------

@dataclass
class TextureVectorizeParams:
    mode: str = "auto"  # 'edges' | 'stipple' | 'hatch' | 'contour' | 'auto'
    dxf_size_mm: float = 1000.0
    kerf_mm: float = 1.1
    # geometry cleanup
    min_feature_size_mm: float = 1.0
    simplify_tol_mm: float = 0.2
    # kerf compensation (basic)
    kerf_offset_mm: float = 0.0  # +outward, -inward; 0 disables (used when kerf_inout=False)
    kerf_inout: bool = False     # inside/outside-aware kerf (±kerf/2 based on containment)
    # area-based micro filtering and colinear merge
    min_feature_area_mm2: float = 1.0
    merge_angle_deg: float = 3.0
    # arc handling
    preserve_arcs: bool = True
    # per-layer overrides (0 or None = use global)
    simplify_tol_edges_mm: Optional[float] = None
    simplify_tol_hatch_mm: Optional[float] = None
    simplify_tol_contour_mm: Optional[float] = None
    min_area_edges_mm2: Optional[float] = None
    min_area_hatch_mm2: Optional[float] = None
    min_area_contour_mm2: Optional[float] = None
    # stipple
    dot_spacing_mm: float = 1.5
    dot_radius_mm: float = 0.5
    # hatch
    hatch_spacing_mm: float = 2.0
    hatch_angle_deg: float = 45.0
    cross_hatch: bool = False
    # contour
    contour_bands: int = 6


def _scale_doc_to_mm(doc: ezdxf.EzDxf, size_px: int, size_mm: float) -> None:
    s = size_mm / float(size_px)
    msp = doc.modelspace()
    for e in list(msp):
        try:
            e.translate(0, 0, 0)
            e.scale_uniform(s)
        except Exception:
            # fallback: recreate scaled entity where possible
            try:
                if e.dxftype() == "LWPOLYLINE":
                    pts = [(float(x) * s, float(y) * s) for (x, y) in e.get_points("xy")]
                    msp.add_lwpolyline(pts, close=bool(e.closed))
                    msp.delete_entity(e)
                elif e.dxftype() == "CIRCLE":
                    c = e.dxf.center
                    r = float(e.dxf.radius) * s
                    msp.add_circle((float(c.x) * s, float(c.y) * s), radius=r)
                    msp.delete_entity(e)
            except Exception:
                continue


def _add_layer(doc: ezdxf.EzDxf, name: str, color: int) -> None:
    if name not in doc.layers:
        doc.layers.add(name, color=color)


def generate_texture_dxf(
    image_path: str,
    out_dir: str | Path,
    preprocess_params: Optional[PreprocessParams] = None,
    classify_params: Optional[TextureClassifyParams] = None,
    vec_params: Optional[TextureVectorizeParams] = None,
) -> Tuple[Path, Path]:
    """Run the texture-aware pipeline and return (dxf_path, preview_png)."""
    preprocess_params = preprocess_params or PreprocessParams()
    classify_params = classify_params or TextureClassifyParams()
    vec_params = vec_params or TextureVectorizeParams()
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    imgs = preprocess(image_path, preprocess_params)
    H, W = imgs["gray"].shape
    assert H == W == preprocess_params.working_px, "Working canvas must be square"

    # classify if auto, else derive mask from respective layer heuristics
    mode = (vec_params.mode or "auto").lower()
    if mode == "auto":
        masks = classify_textures(imgs, classify_params)
    else:
        z = np.zeros_like(imgs["gray"], dtype=np.uint8)
        masks = {"edges": z.copy(), "stipple": z.copy(), "hatch": z.copy(), "contour": z.copy()}
        if mode == "edges":
            masks["edges"] = np.where(imgs["edges"] > 0, 255, 0).astype(np.uint8)
        elif mode == "stipple":
            # use high-frequency layer as density mask
            th = max(10, int(np.mean(imgs["hf"]) + np.std(imgs["hf"])) )
            masks["stipple"] = np.where(imgs["hf"] >= th, 255, 0).astype(np.uint8)
        elif mode == "hatch":
            # use gradient magnitude to indicate directional texture
            th = max(10, int(np.mean(imgs["grad_mag"]) + 0.5*np.std(imgs["grad_mag"])) )
            masks["hatch"] = np.where(imgs["grad_mag"] >= th, 255, 0).astype(np.uint8)
        elif mode == "contour":
            # whole area eligible for contour banding
            masks["contour"] = np.full_like(z, 255)

    # Build DXF
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    _add_layer(doc, "LAYER_EDGES", color=1)
    _add_layer(doc, "LAYER_STIPPLE", color=3)
    _add_layer(doc, "LAYER_HATCH", color=5)
    _add_layer(doc, "LAYER_CONTOUR", color=7)

    tmp = out_base / "texture_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # Edges → polylines
    if np.any(masks["edges"]):
        dxf_e = _vectorize_binary_mask(masks["edges"], tmp, "edges")
        if dxf_e and Path(dxf_e).exists():
            try:
                sub = ezdxf.readfile(str(dxf_e))
                for e in sub.modelspace():
                    ne = msp.add_entity(e)
                    ne.dxf.layer = "LAYER_EDGES"
            except Exception:
                pass

    # Stipple → circles
    if np.any(masks["stipple"]):
        # compute pixel-per-mm scale
        px_per_mm = float(preprocess_params.working_px) / float(vec_params.dxf_size_mm)
        spacing_px = max(1.0, vec_params.dot_spacing_mm * px_per_mm)
        radius_px = max(0.2, vec_params.dot_radius_mm * px_per_mm)
        pts = _poisson_disk_points(masks["stipple"], spacing_px)
        _ = _add_circles(msp, pts, radius_px, layer="LAYER_STIPPLE")

    # Hatch → stripes masked
    if np.any(masks["hatch"]):
        px_per_mm = float(preprocess_params.working_px) / float(vec_params.dxf_size_mm)
        spacing_px = max(1, int(round(vec_params.hatch_spacing_mm * px_per_mm)))
        dxf_h = _vectorize_stripes_masked(masks["hatch"], spacing_px, vec_params.hatch_angle_deg, tmp, "hatch")
        if dxf_h and Path(dxf_h).exists():
            try:
                sub = ezdxf.readfile(str(dxf_h))
                for e in sub.modelspace():
                    ne = msp.add_entity(e)
                    ne.dxf.layer = "LAYER_HATCH"
            except Exception:
                pass
        if vec_params.cross_hatch:
            dxf_h2 = _vectorize_stripes_masked(masks["hatch"], spacing_px, vec_params.hatch_angle_deg + 90.0, tmp, "hatch2")
            if dxf_h2 and Path(dxf_h2).exists():
                try:
                    sub = ezdxf.readfile(str(dxf_h2))
                    for e in sub.modelspace():
                        ne = msp.add_entity(e)
                        ne.dxf.layer = "LAYER_HATCH"
                except Exception:
                    pass

    # Contour bands → multi-threshold contours
    if np.any(masks["contour"]):
        levels = max(2, int(vec_params.contour_bands))
        hist_levels = np.linspace(20, 230, levels, dtype=np.uint8)
        cnt_i = 1
        for t in hist_levels:
            if t <= 0 or t >= 255:
                continue
            _, bin_img = cv2.threshold(imgs["norm"], int(t), 255, cv2.THRESH_BINARY)
            bin_img = cv2.bitwise_and(bin_img, masks["contour"])
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            layer_name = f"LAYER_CONTOUR_{cnt_i}"
            _add_layer(doc, layer_name, color=7)
            for c in contours:
                pts = c.reshape(-1, 2)
                if len(pts) >= 2:
                    e = msp.add_lwpolyline([(float(x), float(y)) for (x, y) in pts], close=True)
                    e.dxf.layer = layer_name
            cnt_i += 1

    # Scale to mm before cleanup
    _scale_doc_to_mm(doc, preprocess_params.working_px, vec_params.dxf_size_mm)

    # Kerf compensation
    if bool(getattr(vec_params, "kerf_inout", False)) and float(getattr(vec_params, "kerf_mm", 0.0)) > 0:
        _apply_kerf_offset_inout(doc, float(vec_params.kerf_mm) * 0.5)
    elif abs(float(getattr(vec_params, "kerf_offset_mm", 0.0))) > 1e-9:
        _apply_kerf_offset(doc, float(vec_params.kerf_offset_mm))

    # Simplify and remove micro-features
    _simplify_and_filter_doc(
        doc,
        min_size_mm=float(getattr(vec_params, "min_feature_size_mm", 1.0)),
        tol_mm=float(getattr(vec_params, "simplify_tol_mm", 0.0)),
        min_area_mm2=float(getattr(vec_params, "min_feature_area_mm2", 1.0)),
        merge_angle_deg=float(getattr(vec_params, "merge_angle_deg", 3.0)),
        vec_params=vec_params,
    )

    # Save
    out_dxf = out_base / f"{Path(image_path).stem}_texture.dxf"
    doc.saveas(str(out_dxf))

    # Preview: render layer overlay on grayscale
    prev = _render_preview(imgs["gray"], doc)
    preview_png = out_base / f"{Path(image_path).stem}_texture_preview.png"
    cv2.imwrite(str(preview_png), prev)

    return out_dxf, preview_png


def _render_preview(gray: np.ndarray, doc: ezdxf.EzDxf) -> np.ndarray:
    # composite polylines over gray
    h, w = gray.shape
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    try:
        msp = doc.modelspace()
        # get extents
        minx = miny = 1e18
        maxx = maxy = -1e18
        for e in msp:
            try:
                box = e.bbox()
                if box:
                    (x1, y1, _), (x2, y2, _) = box.extmin, box.extmax
                    minx = min(minx, float(x1))
                    miny = min(miny, float(y1))
                    maxx = max(maxx, float(x2))
                    maxy = max(maxy, float(y2))
            except Exception:
                continue
        spanx = max(1e-6, maxx - minx)
        spany = max(1e-6, maxy - miny)
        sx = (w - 1) / spanx
        sy = (h - 1) / spany
        s = min(sx, sy)

        def map_pt(x, y):
            xx = int(round((x - minx) * s))
            yy = int(round((y - miny) * s))
            yy = h - 1 - yy  # flip Y for display
            return xx, yy

        for e in msp:
            try:
                if e.dxftype() == "LWPOLYLINE":
                    pts = [(v[0], v[1]) for v in e.get_points("xy")]
                    for i in range(1, len(pts)):
                        p1 = map_pt(*pts[i - 1])
                        p2 = map_pt(*pts[i])
                        cv2.line(rgb, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
                elif e.dxftype() == "LINE":
                    x1, y1 = float(e.dxf.start.x), float(e.dxf.start.y)
                    x2, y2 = float(e.dxf.end.x), float(e.dxf.end.y)
                    cv2.line(rgb, map_pt(x1, y1), map_pt(x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                elif e.dxftype() == "CIRCLE":
                    c = e.dxf.center
                    r = float(e.dxf.radius)
                    cx, cy = map_pt(float(c.x), float(c.y))
                    rr = max(1, int(round(r * s)))
                    cv2.circle(rgb, (cx, cy), rr, (0, 255, 0), 1, cv2.LINE_AA)
            except Exception:
                continue
    except Exception:
        pass
    return rgb


def _apply_kerf_offset(doc: ezdxf.EzDxf, offset_mm: float) -> None:
    """Apply a uniform kerf offset to closed polylines using shapely buffers.

    Positive = outward, Negative = inward. Open polylines are left unchanged.
    """
    msp = doc.modelspace()
    to_process = list(msp.query("LWPOLYLINE"))
    for e in to_process:
        try:
            closed = bool(e.closed)
            if not closed:
                continue
            layer = e.dxf.layer
            pts = [(float(v[0]), float(v[1])) for v in e.get_points("xy")]
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if not poly.is_valid or poly.is_empty:
                continue
            buff = poly.buffer(offset_mm, join_style=2)
            # Remove original
            msp.delete_entity(e)
            # Write new geometry (can be multipolygon)
            def write_poly(p: Polygon):
                ext = list(p.exterior.coords)
                if len(ext) >= 2:
                    ne = msp.add_lwpolyline(ext, format="xy", close=True)
                    ne.dxf.layer = layer
                for ring in p.interiors:
                    coords = list(ring.coords)
                    if len(coords) >= 2:
                        ne2 = msp.add_lwpolyline(coords, format="xy", close=True)
                        ne2.dxf.layer = layer
            if isinstance(buff, (MultiPolygon,)):
                for geom in buff.geoms:
                    write_poly(geom)
            elif isinstance(buff, Polygon):
                write_poly(buff)
        except Exception:
            # Best-effort: skip entity on error
            continue


def _simplify_and_filter_doc(doc: ezdxf.EzDxf, min_size_mm: float, tol_mm: float, min_area_mm2: float, merge_angle_deg: float, vec_params: Optional[TextureVectorizeParams] = None) -> None:
    """Simplify polylines and remove micro features.

    - For LWPOLYLINE (closed): simplify (if tol>0), remove if bbox min-dimension < min_size
    - For LWPOLYLINE (open): simplify (if tol>0), remove if total length < min_size
    - For CIRCLE: remove if diameter < min_size
    """
    msp = doc.modelspace()
    # CIRCLE filtering
    for e in list(msp.query("CIRCLE")):
        try:
            if float(e.dxf.radius) * 2.0 < float(min_size_mm):
                msp.delete_entity(e)
        except Exception:
            continue

    def _resolve_overrides(layer: str):
        lt = layer.upper() if isinstance(layer, str) else ""
        tol = tol_mm
        min_area = min_area_mm2
        # map layer groups
        is_edges = lt.startswith("LAYER_EDGES")
        is_hatch = lt.startswith("LAYER_HATCH")
        is_contour = lt.startswith("LAYER_CONTOUR")
        if vec_params is not None:
            te = vec_params.simplify_tol_edges_mm if vec_params.simplify_tol_edges_mm not in (None, 0.0) else None
            th = vec_params.simplify_tol_hatch_mm if vec_params.simplify_tol_hatch_mm not in (None, 0.0) else None
            tc = vec_params.simplify_tol_contour_mm if vec_params.simplify_tol_contour_mm not in (None, 0.0) else None
            ae = vec_params.min_area_edges_mm2 if vec_params.min_area_edges_mm2 not in (None, 0.0) else None
            ah = vec_params.min_area_hatch_mm2 if vec_params.min_area_hatch_mm2 not in (None, 0.0) else None
            ac = vec_params.min_area_contour_mm2 if vec_params.min_area_contour_mm2 not in (None, 0.0) else None
            if is_edges and te is not None:
                tol = te
            if is_hatch and th is not None:
                tol = th
            if is_contour and tc is not None:
                tol = tc
            if is_edges and ae is not None:
                min_area = ae
            if is_hatch and ah is not None:
                min_area = ah
            if is_contour and ac is not None:
                min_area = ac
        return float(tol), float(min_area)

    # LWPOLYLINE simplify + filter
    for e in list(msp.query("LWPOLYLINE")):
        try:
            layer = e.dxf.layer
            pts = [(float(v[0]), float(v[1])) for v in e.get_points("xy")]
            if len(pts) < 2:
                msp.delete_entity(e)
                continue
            closed = bool(e.closed)
            tol_use, min_area_use = _resolve_overrides(layer)
            # Simplify
            if tol_use > 0 and len(pts) >= 3:
                if closed:
                    geom = Polygon(pts)
                    if geom.is_valid and not geom.is_empty:
                        geom2 = geom.simplify(tol_use, preserve_topology=True)
                        if isinstance(geom2, Polygon) and not geom2.is_empty and len(list(geom2.exterior.coords)) >= 3:
                            pts = [(float(x), float(y)) for (x, y) in list(geom2.exterior.coords)]
                else:
                    geom = LineString(pts)
                    if geom.is_valid and not geom.is_empty:
                        geom2 = geom.simplify(tol_use, preserve_topology=True)
                        if isinstance(geom2, LineString) and not geom2.is_empty and len(list(geom2.coords)) >= 2:
                            pts = [(float(x), float(y)) for (x, y) in list(geom2.coords)]

            # Merge near-colinear vertices
            if len(pts) >= 3:
                pts = _merge_colinear_vertices(pts, float(merge_angle_deg), closed)

            # Filter micro features
            if closed and len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid or poly.is_empty:
                    msp.delete_entity(e)
                    continue
                area = float(poly.area)
                if area < float(min_area_use):
                    msp.delete_entity(e)
                    continue
            else:
                # open: length threshold
                length = 0.0
                for i in range(1, len(pts)):
                    dx = pts[i][0] - pts[i - 1][0]
                    dy = pts[i][1] - pts[i - 1][1]
                    length += math.hypot(dx, dy)
                if length < float(min_size_mm):
                    msp.delete_entity(e)
                    continue

            # Rewrite entity with simplified points if changed
            # Compare counts as a proxy
            old_pts = [(float(v[0]), float(v[1])) for v in e.get_points("xy")]
            if len(pts) != len(old_pts):
                try:
                    ne = msp.add_lwpolyline(pts, format="xy", close=closed)
                    ne.dxf.layer = layer
                    msp.delete_entity(e)
                except Exception:
                    # if rewrite fails, leave original
                    pass
        except Exception:
            continue


def _merge_colinear_vertices(pts: List[Tuple[float, float]], angle_deg: float, closed: bool) -> List[Tuple[float, float]]:
    """Remove vertices where direction change is below angle threshold.

    angle_deg in degrees; works for open/closed polylines. Keeps endpoints.
    """
    if len(pts) < 3:
        return pts
    thresh = max(0.0, float(angle_deg))
    # build list of points; ensure closed ring if closed
    ring = pts[:]
    if closed:
        # ensure first != last
        if ring[0] == ring[-1]:
            ring = ring[:-1]
    def seg_angle(a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        ang = math.degrees(math.atan2(dy, dx))
        return ang
    keep = [True] * len(ring)
    for i in range(len(ring)):
        if i == 0 and not closed:
            continue
        if i == len(ring) - 1 and not closed:
            continue
        i0 = (i - 1) % len(ring)
        i1 = i
        i2 = (i + 1) % len(ring)
        a1 = seg_angle(ring[i0], ring[i1])
        a2 = seg_angle(ring[i1], ring[i2])
        d = abs((a2 - a1 + 180.0) % 360.0 - 180.0)
        if d <= thresh:
            keep[i1] = False
    new_ring = [p for p, k in zip(ring, keep) if k]
    if closed:
        if not new_ring:
            return ring
        if new_ring[0] != new_ring[-1]:
            new_ring.append(new_ring[0])
    return new_ring


def _apply_kerf_offset_inout(doc: ezdxf.EzDxf, half_kerf_mm: float) -> None:
    """Apply ±kerf/2 by containment depth: even depth → outward, odd → inward.

    Depth is number of enclosing polygons among closed LWPOLYLINEs in the doc.
    """
    msp = doc.modelspace()
    # collect closed polygons
    polys = []  # list of (entity, polygon, layer)
    for e in list(msp.query("LWPOLYLINE")):
        try:
            if not bool(e.closed):
                continue
            layer = e.dxf.layer
            pts = [(float(v[0]), float(v[1])) for v in e.get_points("xy")]
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if not poly.is_valid or poly.is_empty:
                continue
            polys.append((e, poly, layer))
        except Exception:
            continue
    if not polys:
        return
    # compute containment depth for each polygon
    depths = [0] * len(polys)
    for i, (_, pi, _) in enumerate(polys):
        d = 0
        for j, (_, pj, _) in enumerate(polys):
            if i == j:
                continue
            try:
                if pj.contains(pi):
                    d += 1
            except Exception:
                continue
        depths[i] = d
    # apply buffer based on depth parity
    for (e, poly, layer), depth in zip(polys, depths):
        try:
            offset = half_kerf_mm if (depth % 2 == 0) else -half_kerf_mm
            buff = poly.buffer(offset, join_style=2)
            msp.delete_entity(e)
            def write_poly(p: Polygon):
                ext = list(p.exterior.coords)
                if len(ext) >= 2:
                    ne = msp.add_lwpolyline(ext, format="xy", close=True)
                    ne.dxf.layer = layer
                for ring in p.interiors:
                    coords = list(ring.coords)
                    if len(coords) >= 2:
                        ne2 = msp.add_lwpolyline(coords, format="xy", close=True)
                        ne2.dxf.layer = layer
            if isinstance(buff, (MultiPolygon,)):
                for geom in buff.geoms:
                    write_poly(geom)
            elif isinstance(buff, Polygon):
                write_poly(buff)
        except Exception:
            # skip on error
            continue


__all__ = [
    "PreprocessParams",
    "TextureClassifyParams",
    "TextureVectorizeParams",
    "generate_texture_dxf",
]
