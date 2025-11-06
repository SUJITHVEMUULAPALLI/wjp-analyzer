from __future__ import annotations

"""
Skeletonization Preprocessor
Converts thick, filled raster strokes to single-pixel centerlines to avoid
double-contour vectorization. Designed to be used before potrace/Inkscape.

Dependencies (optional fallbacks):
- scikit-image (preferred) for skeletonize
- opencv-contrib-python (ximgproc.thinning) as alternative

API:
- skeletonize_image(input_path, output_path=None, invert=True) -> str
  Returns path to skeletonized PNG. If output_path is None, writes next to input.

- skeletonize_array(img: np.ndarray, invert=True) -> np.ndarray

The module avoids heavy dependencies at import time by optional imports.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _ensure_binary(gray: np.ndarray, invert: bool = True, thresh: int = 200) -> np.ndarray:
    """Binarize grayscale image to 0/255. Optionally invert so foreground is white.

    Many scanned drawings are black-on-white. For skeletonization, having
    foreground as 255 (True) is convenient.
    """
    if len(gray.shape) != 2:
        raise ValueError("_ensure_binary expects a single-channel grayscale image")
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    if invert:
        bw = cv2.bitwise_not(bw)
    return bw


def _skeletonize_bool(bool_img: np.ndarray) -> np.ndarray:
    """Skeletonize a boolean image using available backends.

    Tries scikit-image first, then OpenCV ximgproc thinning. Returns boolean array.
    """
    # Try scikit-image
    try:
        from skimage.morphology import skeletonize as sk_skeletonize  # type: ignore
        skel = sk_skeletonize(bool_img)
        return skel
    except Exception:
        pass

    # Try OpenCV ximgproc thinning
    try:
        import cv2.ximgproc as xip  # type: ignore
        # ximgproc.thinning expects 8-bit single-channel, 0/255
        tmp = (bool_img.astype(np.uint8) * 255)
        thinned = xip.thinning(tmp, thinningType=xip.THINNING_GUOHALL)
        return thinned > 0
    except Exception:
        pass

    # Fallback: morphological thinning approximation via hit-or-miss iterations
    # This is slower and less accurate but avoids hard dependency failures.
    kernel = np.array([[0, 0, 0],
                       [ -1, 1, -1],
                       [1, 1, 1]], dtype=np.int8)
    work = bool_img.copy().astype(np.uint8)
    changed = True
    iterations = 0
    while changed and iterations < 100:
        eroded = cv2.erode(work, np.ones((3, 3), np.uint8))
        opened = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
        diff = cv2.subtract(work, opened)
        changed = diff.any()
        work = opened
        iterations += 1
    return work.astype(bool)


def skeletonize_array(img: np.ndarray, invert: bool = True, thresh: int = 200) -> np.ndarray:
    """Skeletonize an image array. Returns 8-bit single-channel 0/255 image.

    Parameters:
    - img: BGR or grayscale numpy array
    - invert: If True, treat dark strokes on light background as foreground
    - thresh: Binarization threshold (0-255)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    bw = _ensure_binary(gray, invert=invert, thresh=thresh)
    bool_img = bw > 0
    skel_bool = _skeletonize_bool(bool_img)
    skel_u8 = (skel_bool.astype(np.uint8) * 255)
    # Return as black-on-white (invert back so background is 255)
    result = cv2.bitwise_not(skel_u8)
    return result


def skeletonize_image(input_path: str, output_path: Optional[str] = None, invert: bool = True, thresh: int = 200) -> str:
    """Skeletonize an input image and write a PNG.

    Returns output path. If output_path is not provided, writes alongside input
    as <stem>_skeleton.png.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(str(src))
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    skel = skeletonize_array(img, invert=invert, thresh=thresh)

    if output_path is None:
        out = src.with_name(src.stem + "_skeleton.png")
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out), skel)
    return str(out)


def centerline_preprocess_for_vectorization(
    input_path: str,
    out_dir: str | Path,
    invert: bool = True,
    thresh: int = 200,
    open_kernel: Tuple[int, int] = (3, 3),
) -> str:
    """Convenience pipeline used before potrace:
    1) Threshold + optional invert
    2) Morphological open to remove specks
    3) Skeletonize to one-pixel centerlines
    4) Return path to cleaned PNG stored in out_dir
    """
    src = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    bw = _ensure_binary(img, invert=invert, thresh=thresh)
    kernel = np.ones(open_kernel, np.uint8)
    cleaned = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    skel = skeletonize_array(cleaned, invert=False, thresh=128)  # already binary

    out_path = out_dir / f"{src.stem}_centerline.png"
    cv2.imwrite(str(out_path), skel)
    return str(out_path)


__all__ = [
    "skeletonize_array",
    "skeletonize_image",
    "centerline_preprocess_for_vectorization",
]






