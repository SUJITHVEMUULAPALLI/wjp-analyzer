"""
WJP Image Analyzer – Phase 1 (Diagnostic Core)
Evaluates an input image for DXF-conversion suitability (waterjet context).

Metrics:
- basic: size, aspect, grayscale stats
- contrast/edges: edge density, edge contrast ratio
- texture/noise: entropy, FFT high-frequency energy
- orientation: skew angle estimate
- topology preview: contour counts, closed %, small features count
- manufacturability checks: min spacing (approx), min radius (approx)
- composite readiness score + flags

Dependencies: opencv-python, numpy, pillow (optional)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import math
import json
import cv2
import numpy as np


# --------------------------
# Config
# --------------------------

@dataclass
class AnalyzerConfig:
    # preprocessing
    max_size_px: int = 2000              # resize longest edge if larger
    gaussian_blur_ksize: int = 3         # 0 to disable
    invert: bool = False                 # set True if white shape on black
    # binarization
    binarize_mode: str = "adaptive"      # adaptive | otsu | fixed
    adaptive_block: int = 31
    adaptive_C: int = 5
    fixed_thresh: int = 180
    # deskew
    deskew: bool = True
    hough_min_line_len: int = 80
    hough_max_line_gap: int = 10
    # feature scales
    px_to_unit: float = 1.0              # set mm/px if known; else 1.0
    # topology
    min_contour_perimeter_px: float = 30.0
    contour_mode_tree: bool = True       # RETR_TREE vs RETR_EXTERNAL
    # manufacturability heuristics (units = your px_to_unit)
    min_spacing_unit: float = 3.0        # waterjet safe gap ~≥3 mm
    min_radius_unit: float = 1.5
    small_feature_perimeter_unit: float = 3.0
    # scoring thresholds
    good_edge_density: Tuple[float, float] = (0.05, 0.25)  # ok range
    max_texture_fft_energy: float = 0.15
    entropy_bounds: Tuple[float, float] = (3.0, 6.0)
    min_closed_ratio_good: float = 0.80
    min_closed_ratio_warn: float = 0.60
    # output
    return_debug_masks: bool = False


# --------------------------
# Utility
# --------------------------

def _resize_max(img: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return img, 1.0
    scale = max_size / float(m)
    out = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return out, scale

def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def _edge_density(gray: np.ndarray) -> Tuple[float, np.ndarray]:
    edges = cv2.Canny(gray, 60, 180)
    return float((edges > 0).sum()) / float(gray.size), edges

def _edge_contrast_ratio(gray: np.ndarray, edges: np.ndarray) -> float:
    # mean gradient magnitude along edges / global stddev proxy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    e_mag = mag[edges > 0]
    if e_mag.size == 0:
        return 0.0
    return float(np.mean(e_mag) / (np.std(gray) + 1e-6))

def _fft_highfreq_energy(gray: np.ndarray, frac=0.2) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h//2, w//2
    ry, rx = int(h*frac/2), int(w*frac/2)
    low = mag[cy-ry:cy+ry, cx-rx:cx+rx].sum()
    total = mag.sum() + 1e-9
    return float(1.0 - low/total)

def _estimate_skew_angle_deg(gray: np.ndarray, min_len=80, max_gap=10) -> float:
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=min_len, maxLineGap=max_gap)
    angs = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            ang = math.degrees(math.atan2(y2-y1, x2-x1))
            if ang < -90: ang += 180
            if ang > 90:  ang -= 180
            angs.append(ang)
    if len(angs) >= 4:
        angs.sort()
        return float(np.median(angs))
    # fallback: minAreaRect orientation
    ys, xs = np.where(edges > 0)
    if len(xs) < 50:
        return 0.0
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]  # [-90, 0)
    return float(angle)

def _binarize(gray: np.ndarray, cfg: AnalyzerConfig) -> np.ndarray:
    g = gray.copy()
    if cfg.invert:
        g = cv2.bitwise_not(g)
    if cfg.gaussian_blur_ksize and cfg.gaussian_blur_ksize >= 3:
        k = cfg.gaussian_blur_ksize | 1
        g = cv2.GaussianBlur(g, (k, k), 0)

    if cfg.binarize_mode == "adaptive":
        bs = cfg.adaptive_block if (cfg.adaptive_block % 2 == 1) else (cfg.adaptive_block + 1)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, cfg.adaptive_C)
    elif cfg.binarize_mode == "otsu":
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(g, cfg.fixed_thresh, 255, cv2.THRESH_BINARY)
    return bw

def _contours_with_hierarchy(bw: np.ndarray, tree: bool) -> Tuple[List[np.ndarray], np.ndarray]:
    mode = cv2.RETR_TREE if tree else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(bw, mode, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        hierarchy = np.zeros((1, len(contours), 4), dtype=np.int32)
    hierarchy = hierarchy[0] if len(hierarchy.shape) == 3 else hierarchy
    return contours, hierarchy


# --------------------------
# Manufacturability heuristics
# --------------------------

def _approx_min_spacing(bw: np.ndarray, px_to_unit: float) -> float:
    # very cheap proxy: erode then dilate and see when features merge
    # We estimate minimal separations by checking skeleton collisions lightly.
    # Keep it cheap: use distance transform to get minimal gap.
    inv = (255 - bw)  # shapes as 0, background 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)  # px
    # minimal nonzero distance near boundaries
    d = dist[(dist > 0) & (dist < 50)]
    if d.size == 0:
        return float("inf")
    # distance transform gives distance to nearest background; min spacing ~ 2*min(dist)
    min_px = float(np.percentile(d, 1)) * 2.0  # robust
    return min_px * px_to_unit

def _approx_min_radius(contour: np.ndarray, px_to_unit: float) -> float:
    # crude local radius estimate via three-point circle over a sliding window
    pts = contour[:,0,:].astype(np.float32)  # (N,2)
    n = len(pts)
    if n < 6:
        return float("inf")
    def circ_radius(a,b,c):
        # radius = |AB|*|BC|*|CA| / (4*Area)
        AB = np.linalg.norm(b-a); BC = np.linalg.norm(c-b); CA = np.linalg.norm(a-c)
        area = abs(0.5*((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])))
        if area < 1e-6: return float("inf")
        return (AB*BC*CA)/(4.0*area)
    window = max(3, min(9, n//20))
    radii = []
    for i in range(n - window):
        a = pts[i]
        b = pts[i + window//2]
        c = pts[i + window - 1]
        r = circ_radius(a,b,c)
        if np.isfinite(r) and r < 1e6:
            radii.append(r)
    if not radii:
        return float("inf")
    return float(np.percentile(radii, 5)) * px_to_unit  # robust small-radius estimate


# --------------------------
# Main API
# --------------------------

def analyze_image_for_wjp(image_path: str, cfg: Optional[AnalyzerConfig] = None) -> Dict[str, Any]:
    """
    Returns a JSON-able dict with all metrics/flags/scores.
    No DXF writing here – this is the precheck gate.
    """
    cfg = cfg or AnalyzerConfig()

    # load
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    h0, w0 = img.shape[:2]

    # resize for analysis speed
    img_r, s = _resize_max(img, cfg.max_size_px)
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # optional blur
    if cfg.gaussian_blur_ksize and cfg.gaussian_blur_ksize >= 3:
        k = cfg.gaussian_blur_ksize | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # orientation
    angle_deg = _estimate_skew_angle_deg(gray, cfg.hough_min_line_len, cfg.hough_max_line_gap) if cfg.deskew else 0.0

    # edge/texture stats
    edens, edges = _edge_density(gray)
    econ = _edge_contrast_ratio(gray, edges)
    ent = _entropy(gray)
    fft_hf = _fft_highfreq_energy(gray)

    # binarize
    bw = _binarize(gray, cfg)

    # contours
    contours, hierarchy = _contours_with_hierarchy(bw, cfg.contour_mode_tree)
    total_ct = len(contours)

    # classify closed/open + small features
    closed_ct = 0
    small_features = 0
    min_radius_unit_vals: List[float] = []

    px_to_unit = cfg.px_to_unit / s  # account for resize
    for c in contours:
        per_px = cv2.arcLength(c, True)
        if per_px < cfg.min_contour_perimeter_px:
            continue
        # closed? (OpenCV contours are by default closed; but check area sanity)
        area = abs(cv2.contourArea(c))
        if area > 0.5:  # px^2
            closed_ct += 1
        # small features (by perimeter)
        per_unit = per_px * px_to_unit
        if per_unit < cfg.small_feature_perimeter_unit:
            small_features += 1
        # local min radius estimate
        min_r = _approx_min_radius(c, px_to_unit)
        min_radius_unit_vals.append(min_r)

    closed_ratio = (closed_ct / total_ct) if total_ct else 0.0

    # manufacturability: min spacing (approx)
    min_spacing_unit = _approx_min_spacing(bw, px_to_unit)
    min_radius_unit = float(np.min(min_radius_unit_vals)) if min_radius_unit_vals else float("inf")

    # --- scoring ---
    score = 100.0
    # edge density score (prefer within band)
    lo, hi = cfg.good_edge_density
    if edens < lo: score -= 15
    if edens > hi: score -= 15
    # texture penalty
    if fft_hf > cfg.max_texture_fft_energy: score -= 20
    # entropy penalty for too low/high
    elo, ehi = cfg.entropy_bounds
    if ent < elo: score -= 10
    if ent > ehi: score -= 10
    # closed ratio
    if closed_ratio < cfg.min_closed_ratio_good:
        if closed_ratio >= cfg.min_closed_ratio_warn:
            score -= 10
        else:
            score -= 25
    # spacing
    if min_spacing_unit < cfg.min_spacing_unit:
        score -= 20
    # radius
    if min_radius_unit < cfg.min_radius_unit:
        score -= 10

    score = max(0.0, min(100.0, score))

    # flags
    flags = {
        "low_edge_density": edens < lo,
        "high_edge_density_noise": edens > hi,
        "high_texture_noise": fft_hf > cfg.max_texture_fft_energy,
        "entropy_low": ent < elo,
        "entropy_high": ent > ehi,
        "closed_ratio_low": closed_ratio < cfg.min_closed_ratio_good,
        "tight_spacing": min_spacing_unit < cfg.min_spacing_unit,
        "tight_radius": min_radius_unit < cfg.min_radius_unit,
        "skew_detected_deg": angle_deg,
    }

    # suggestions (direct, no fluff)
    suggestions = []
    if flags["low_edge_density"]:
        suggestions.append("Increase contrast or sharpen edges before tracing.")
    if flags["high_edge_density_noise"]:
        suggestions.append("Reduce noise; use smoothing or remove textured background.")
    if flags["high_texture_noise"]:
        suggestions.append("This looks like texture/photo. Limit to outlines or simplify background.")
    if flags["entropy_low"]:
        suggestions.append("Image is too flat; boost contrast.")
    if flags["entropy_high"]:
        suggestions.append("Image is too busy; mask or simplify non-cut areas.")
    if flags["closed_ratio_low"]:
        suggestions.append("Fix open contours; ensure shapes are watertight.")
    if flags["tight_spacing"]:
        suggestions.append(f"Increase minimum gap to >= {cfg.min_spacing_unit:.2f} units.")
    if flags["tight_radius"]:
        suggestions.append(f"Avoid fillets below {cfg.min_radius_unit:.2f} units.")
    if abs(angle_deg) > 1.0:
        suggestions.append(f"Rotate by {angle_deg:.1f} degrees to deskew before conversion.")

    report = {
        "file": image_path,
        "config": asdict(cfg),
        "image_original_size": {"w": w0, "h": h0},
        "image_used_scale": 1.0 / s,
        "basic_stats": {
            "mean_gray": float(np.mean(gray)),
            "std_gray": float(np.std(gray)),
            "aspect_ratio": float(w0 / max(1, h0)),
        },
        "orientation": {
            "skew_angle_deg": float(angle_deg),
        },
        "texture_metrics": {
            "edge_density": float(edens),
            "edge_contrast_ratio": float(econ),
            "entropy": float(ent),
            "fft_highfreq_energy": float(fft_hf),
        },
        "topology_preview": {
            "total_contours": int(total_ct),
            "closed_contours": int(closed_ct),
            "closed_ratio": float(round(closed_ratio, 3)),
            "small_features_count": int(small_features),
        },
        "manufacturability": {
            "min_spacing_unit": float(min_spacing_unit),
            "min_radius_unit": float(min_radius_unit),
        },
        "score": float(round(score, 1)),
        "flags": flags,
        "suggestions": suggestions,
    }

    if cfg.return_debug_masks:
        report["_debug_shapes"] = {
            "edges_sum": int((edges > 0).sum()),
            "bw_white_px": int((cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] > 0).sum()),
        }

    return report


# --------------------------
# CLI helper
# --------------------------

if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) < 2:
        print("Usage: python wjp_image_analyzer_core.py <image_path>")
        sys.exit(1)
    cfg = AnalyzerConfig()
    rep = analyze_image_for_wjp(sys.argv[1], cfg)
    print(json.dumps(rep, indent=2))
