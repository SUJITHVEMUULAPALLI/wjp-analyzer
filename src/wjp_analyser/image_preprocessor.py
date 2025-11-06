"""
WJP Image Preprocessor – Lighting, Perspective & View Correction
---------------------------------------------------------------
Purpose:
  • Normalize exposure & contrast
  • Remove flash hotspots (specular reflection masking)
  • Estimate camera tilt & deskew angle
  • Detect oblique captures (non-top-down)
  • Output a 'corrected' image for WJP Analyzer

Dependencies: opencv-python, numpy
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any


@dataclass
class PreprocConfig:
    deskew: bool = True
    max_size_px: int = 2000
    glare_threshold: int = 225       # pixels brighter than this → glare (improved from 240)
    inpaint_radius: int = 5
    contrast_clip_limit: float = 2.0 # CLAHE contrast enhancement
    tile_grid_size: tuple = (8, 8)
    hough_min_line_len: int = 120
    hough_max_line_gap: int = 10
    skew_conf_threshold: float = 0.6 # minimal confidence to apply rotation (improved from 0.5)
    perspective_warn_angle: float = 8.0  # degrees tilt tolerance
    auto_rotation_threshold: float = 0.6  # confidence threshold for auto-rotation
    use_contour_skew: bool = True    # enable secondary skew detection via contour bounding box
    color_restoration: bool = True   # overlay corrected grayscale on original color
    return_debug: bool = False


def detect_shadow_zones(gray: np.ndarray) -> np.ndarray:
    """
    Detect shadow zones using illumination gradient analysis.
    
    Args:
        gray: Input grayscale image
        
    Returns:
        Binary mask where shadows are detected (255 = shadow, 0 = normal)
    """
    # 1. Estimate illumination gradient using large Gaussian blur
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    
    # 2. Calculate ratio to detect darker-than-average regions
    # Avoid division by zero
    blur_safe = np.maximum(blur, 1)
    ratio = cv2.divide(gray, blur_safe, scale=255)
    
    # 3. Threshold darker-than-average regions (shadows)
    # Lower ratio values indicate darker regions relative to local illumination
    shadow_mask = cv2.inRange(ratio, 0, 120)
    
    # 4. Clean up the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. Remove small noise regions
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    return shadow_mask


def compensate_shadows(gray: np.ndarray, shadow_mask: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """
    Compensate for detected shadow zones using inpainting and CLAHE.
    
    Args:
        gray: Input grayscale image
        shadow_mask: Binary mask of shadow zones
        cfg: Preprocessing configuration
        
    Returns:
        Shadow-compensated grayscale image
    """
    if shadow_mask.sum() == 0:
        # No shadows detected, return original
        return gray
    
    # 1. Inpaint shadow regions to remove them
    shadow_free = cv2.inpaint(gray, shadow_mask, cfg.inpaint_radius, cv2.INPAINT_TELEA)
    
    # 2. Apply CLAHE to restore local contrast
    clahe = cv2.createCLAHE(clipLimit=cfg.contrast_clip_limit, tileGridSize=cfg.tile_grid_size)
    compensated = clahe.apply(shadow_free)
    
    return compensated


def correct_lighting(img: np.ndarray, cfg: PreprocConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equalize illumination, inpaint flash reflections, and compensate shadows."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- Step 1: Detect glare hotspots ---
    glare_mask = (gray > cfg.glare_threshold).astype(np.uint8) * 255
    if glare_mask.sum() > 0:
        img = cv2.inpaint(img, glare_mask, cfg.inpaint_radius, cv2.INPAINT_TELEA)
    
    # --- Step 2: Detect shadow zones ---
    shadow_mask = detect_shadow_zones(gray)
    
    # --- Step 3: Compensate shadows in grayscale ---
    gray_compensated = compensate_shadows(gray, shadow_mask, cfg)
    
    # --- Step 4: Apply shadow compensation to color image ---
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply shadow compensation to L channel
    l_compensated = compensate_shadows(l, shadow_mask, cfg)
    
    # Merge channels and convert back to BGR
    merged = cv2.merge((l_compensated, a, b))
    eq = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return eq, glare_mask, shadow_mask


def estimate_skew_angle(gray: np.ndarray, cfg: PreprocConfig) -> Tuple[float, float]:
    """Estimate skew angle using Hough line detection."""
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=cfg.hough_min_line_len,
                            maxLineGap=cfg.hough_max_line_gap)
    
    if lines is None or len(lines) < 4:
        return 0.0, 0.0
    
    angs = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        ang = (ang + 180) % 180
        if ang > 90: 
            ang -= 180
        angs.append(ang)
    
    angs = np.array(angs)
    median = np.median(angs)
    spread = np.std(angs)
    confidence = max(0.0, 1.0 - (spread / 45.0))
    return float(median), float(confidence)


def estimate_skew_from_contour(gray: np.ndarray, cfg: PreprocConfig) -> Tuple[float, float]:
    """Secondary skew detection using largest contour bounding box."""
    if not cfg.use_contour_skew:
        return 0.0, 0.0
    
    # Threshold and find contours
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return 0.0, 0.0
    
    # Find largest contour
    largest_contour = max(cnts, key=cv2.contourArea)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Normalize angle to [-45, 45] range
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    
    # Estimate confidence based on contour area and aspect ratio
    area = cv2.contourArea(largest_contour)
    total_area = gray.shape[0] * gray.shape[1]
    area_ratio = area / total_area
    
    # Higher confidence for larger, more rectangular contours
    confidence = min(0.8, area_ratio * 10)  # Cap at 0.8 for contour-based detection
    
    return float(angle), float(confidence)


def combine_skew_estimates(hough_angle: float, hough_conf: float, 
                          contour_angle: float, contour_conf: float) -> Tuple[float, float]:
    """Combine Hough and contour-based skew estimates with weighted average."""
    if hough_conf < 0.3 and contour_conf < 0.3:
        return 0.0, 0.0
    
    # Use weighted average based on confidence
    total_weight = hough_conf + contour_conf
    if total_weight == 0:
        return 0.0, 0.0
    
    combined_angle = (hough_angle * hough_conf + contour_angle * contour_conf) / total_weight
    combined_confidence = min(0.9, total_weight / 2)  # Cap combined confidence
    
    return float(combined_angle), float(combined_confidence)


def restore_color_texture(original_img: np.ndarray, corrected_img: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """Overlay corrected grayscale on original color texture for natural preview."""
    if not cfg.color_restoration:
        return corrected_img
    
    # Convert corrected image to grayscale
    corrected_gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    
    # Convert original to LAB color space
    original_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(original_lab)
    
    # Replace L channel with corrected grayscale
    restored_lab = cv2.merge((corrected_gray, a_channel, b_channel))
    restored_bgr = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2BGR)
    
    return restored_bgr


def correct_perspective(img: np.ndarray, cfg: PreprocConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rough perspective flattening via contour bounding box."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return img, np.array([]), 0.0
    
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    w, h = int(rect[1][0]), int(rect[1][1])
    
    if w == 0 or h == 0: 
        return img, box, 0.0
    
    aspect = max(w, h) / max(1, min(w, h))
    tilt_angle = abs(90 - abs(rect[2])) if rect[2] != 0 else 0
    
    # no warp, just report tilt severity
    return img, box, tilt_angle


def preprocess_image_for_analyzer(image_path: str, cfg: PreprocConfig = PreprocConfig()) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enhanced preprocessing pipeline:
    glare removal, shadow detection & compensation, contrast equalization,
    dual skew detection, auto-rotation, color restoration.
    Returns corrected image + JSON metrics.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Store original for color restoration
    original_img = img.copy()

    # resize
    h, w = img.shape[:2]
    m = max(h, w)
    if m > cfg.max_size_px:
        scale = cfg.max_size_px / m
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        original_img = cv2.resize(original_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Step 1: Lighting correction (including shadow detection and compensation)
    corrected, glare_mask, shadow_mask = correct_lighting(img, cfg)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    # Step 2: Dual skew detection
    hough_angle, hough_conf = estimate_skew_angle(gray, cfg)
    contour_angle, contour_conf = estimate_skew_from_contour(gray, cfg)
    skew_deg, skew_conf = combine_skew_estimates(hough_angle, hough_conf, contour_angle, contour_conf)

    # Step 3: Auto-rotation if confidence is high enough
    rotation_applied = False
    if cfg.deskew and abs(skew_deg) > 0.5 and skew_conf > cfg.auto_rotation_threshold:
        M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), -skew_deg, 1.0)
        corrected = cv2.warpAffine(corrected, M, (gray.shape[1], gray.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rotation_applied = True

    # Step 4: Perspective check
    corr2, box, tilt = correct_perspective(corrected, cfg)
    perspective_flag = tilt > cfg.perspective_warn_angle

    # Step 5: Color restoration
    final_img = restore_color_texture(original_img, corr2, cfg)

    # Enhanced metrics including shadow detection
    metrics = {
        "glare_pixels": int((glare_mask > 0).sum()),
        "shadow_pixels": int((shadow_mask > 0).sum()),
        "shadow_flagged": int((shadow_mask > 0).sum()) > 5000,  # Flag if significant shadows detected
        "skew_angle_deg": round(float(skew_deg), 2),
        "skew_confidence": round(float(skew_conf), 2),
        "hough_angle": round(float(hough_angle), 2),
        "hough_confidence": round(float(hough_conf), 2),
        "contour_angle": round(float(contour_angle), 2),
        "contour_confidence": round(float(contour_conf), 2),
        "rotation_applied": bool(rotation_applied),
        "perspective_tilt_deg": round(float(tilt), 2),
        "perspective_flagged": bool(perspective_flag),
        "color_restoration_applied": bool(cfg.color_restoration),
        "shadow_compensation_applied": int((shadow_mask > 0).sum()) > 0,
        "config": asdict(cfg),
    }

    return final_img, metrics
