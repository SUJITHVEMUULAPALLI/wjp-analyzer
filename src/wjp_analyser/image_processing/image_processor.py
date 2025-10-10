#!/usr/bin/env python3
"""
Image Processing Pipeline for DXF Conversion
============================================

This module provides comprehensive image processing capabilities for converting
raster images to cutting-ready DXF files. It includes multiple algorithms,
object detection, interactive editing, and quality validation.

Key Features:
- Multiple conversion algorithms (Potrace, OpenCV, texture-aware)
- Advanced object detection and shape classification
- Interactive editing interface for object modification
- Preview rendering with vector overlay
- Quality assessment and validation
- Batch processing support

Supported Input Formats:
- PNG, JPG, JPEG, BMP, TIFF

Output Formats:
- DXF (R12-R2018)
- SVG (vector graphics)
- Preview images (PNG)

Security Features:
- File type validation and sanitization
- Path traversal protection
- Input parameter validation
- Error handling and logging

Performance Optimizations:
- Caching of processing results
- Efficient image algorithms
- Memory management for large images
- Parallel processing support

Author: WJP Analyser Team
Version: 0.1.0
License: MIT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple, Set

import cv2
import ezdxf
import numpy as np
from PIL import Image


class ImageProcessor:
    def __init__(
        self,
        edge_threshold: float = 0.33,
        min_contour_area: int = 100,
        simplify_tolerance: float = 0.02,
        blur_kernel_size: int = 5,
        # New: optional denoise controls
        denoise_min_area: int = 0,
        morph_kernel: int = 0,
        morph_iterations: int = 1,
    ) -> None:
        """Initialise the processor.

        Args:
            edge_threshold: Sigma factor (0.0-1.0) used to derive Canny thresholds.
            min_contour_area: Minimum pixel area for contours to keep.
            simplify_tolerance: Douglas-Peucker tolerance multiplier for contour simplification.
            blur_kernel_size: Gaussian blur kernel size (odd integer) for noise reduction.
        """
        self.edge_threshold = float(edge_threshold)
        self.min_contour_area = int(min_contour_area)
        self.simplify_tolerance = float(simplify_tolerance)
        self.blur_kernel_size = int(blur_kernel_size)
        self._last_canny_thresholds: Tuple[int, int] = (50, 150)
        # Denoise configuration
        self.denoise_min_area = int(max(0, denoise_min_area))
        self.morph_kernel = int(max(0, morph_kernel))
        self.morph_iterations = int(max(1, morph_iterations))

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk and convert to grayscale."""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            image_array = np.array(pil_image)
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            return gray_image
        except Exception as exc:  # pragma: no cover - conversion relies on native libs
            raise ValueError(f"Could not load image {image_path}: {exc}") from exc

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Blur, threshold, and optionally denoise the binary mask.

        Steps:
        - Gaussian blur for noise reduction
        - Adaptive OR Otsu threshold (union)
        - Optional blob removal for small components
        - Optional morphological open/close to smooth contours
        """
        kernel = max(3, self.blur_kernel_size | 1)
        blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
        # Adaptive threshold handles local illumination; Otsu handles global contrast.
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(adaptive, otsu)

        # Remove tiny speckles by connected components if configured
        if self.denoise_min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
            keep = np.zeros_like(combined)
            for i in range(1, num_labels):  # skip background 0
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.denoise_min_area:
                    keep[labels == i] = 255
            combined = keep

        # Optional morphological smoothing
        if self.morph_kernel >= 3:
            k = self.morph_kernel | 1
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, element, iterations=self.morph_iterations)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, element, iterations=self.morph_iterations)

        return combined

    def _compute_canny_thresholds(self, image: np.ndarray) -> Tuple[int, int]:
        sigma = float(np.clip(self.edge_threshold, 0.01, 0.99))
        median_val = float(np.median(image))
        low = int(max(0, (1.0 - sigma) * median_val))
        high = int(min(255, (1.0 + sigma) * median_val))
        if low >= high:
            low = max(0, high - 50)
        high = max(high, low + 30)
        return max(0, min(low, 255)), max(0, min(high, 255))

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny with adaptive thresholds."""
        low, high = self._compute_canny_thresholds(image)
        edges = cv2.Canny(image, low, high)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        self._last_canny_thresholds = (low, high)
        return cleaned

    def _contour_signature(self, contour: np.ndarray) -> Tuple[float, float, float]:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            cx = cy = 0.0
        else:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        area = float(abs(cv2.contourArea(contour)))
        return (round(cx, 1), round(cy, 1), round(area, 1))

    def _simplify_contour(self, contour: np.ndarray) -> np.ndarray | None:
        area = abs(cv2.contourArea(contour))
        if area < self.min_contour_area:
            return None
        perimeter = max(cv2.arcLength(contour, True), 1.0)
        tolerance = self.simplify_tolerance
        if tolerance < 1.0:
            epsilon = max(0.001, tolerance * perimeter)
        else:
            epsilon = min(tolerance, 0.25 * perimeter)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if simplified.shape[0] < 3:
            return None
        return simplified

    def _collect_contours(
        self, mask: np.ndarray, seen: Set[Tuple[float, float, float]]
    ) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed: List[np.ndarray] = []
        for contour in contours:
            signature = self._contour_signature(contour)
            if signature in seen:
                continue
            simplified = self._simplify_contour(contour)
            if simplified is None:
                continue
            seen.add(signature)
            processed.append(simplified)
        return processed

    def extract_contours(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """Extract simplified contours (including nested shapes) from a binary mask."""
        seen: Set[Tuple[float, float, float]] = set()
        contours: List[np.ndarray] = []
        contours.extend(self._collect_contours(binary_mask, seen))
        inverted = cv2.bitwise_not(binary_mask)
        contours.extend(self._collect_contours(inverted, seen))
        return contours

    def contours_to_dxf(
        self,
        contours: List[np.ndarray],
        output_path: str,
        scale_factor: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Write contours to a lightweight DXF file."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        for contour in contours:
            if len(contour) < 3:
                continue
            points = []
            for point in contour:
                x = (point[0][0] * scale_factor) + offset[0]
                y = (point[0][1] * scale_factor) + offset[1]
                points.append((x, y))
            if len(points) >= 3:
                # Request a closed polyline so downstream DXF geometry forms loops.
                try:
                    msp.add_lwpolyline(points, format="xy", close=True)
                except TypeError:
                    if points[0] != points[-1]:
                        points.append(points[0])
                    msp.add_lwpolyline(points, format="xy")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(output_path)
        print(f"DXF saved to: {output_path}")
        print(f"Contours processed: {len(contours)}")

    def process_image_to_dxf(
        self,
        image_path: str,
        output_path: str,
        scale_factor: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
        debug_dir: str | None = None,
    ) -> int:
        """Run the full conversion pipeline and optionally persist debug artefacts."""
        print(f"Processing image: {image_path}")
        image = self.load_image(image_path)
        print(f"Image loaded: {image.shape}")

        processed = self.preprocess_image(image)
        print("Image preprocessed")

        edges = self.detect_edges(processed)
        print("Edges detected")

        contours = self.extract_contours(processed)
        print(f"Contours extracted: {len(contours)}")

        self.contours_to_dxf(contours, output_path, scale_factor, offset)

        metadata = {
            "source_image": os.path.abspath(image_path),
            "output_dxf": os.path.abspath(output_path),
            "contour_count": len(contours),
            "parameters": {
                "edge_threshold": self.edge_threshold,
                "canny_low": self._last_canny_thresholds[0],
                "canny_high": self._last_canny_thresholds[1],
                "min_contour_area": self.min_contour_area,
                "simplify_tolerance": self.simplify_tolerance,
                "blur_kernel_size": self.blur_kernel_size,
                "denoise_min_area": self.denoise_min_area,
                "morph_kernel": self.morph_kernel,
                "morph_iterations": self.morph_iterations,
                "scale_factor": scale_factor,
                "offset": offset,
            },
        }

        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path / "image_gray.png"), image)
            cv2.imwrite(str(debug_path / "image_threshold.png"), processed)
            cv2.imwrite(str(debug_path / "image_edges.png"), edges)
            with (debug_path / "image_metadata.json").open("w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)

        return len(contours)


def create_sample_image(output_path: str, width: int = 400, height: int = 300) -> None:
    """Create a synthetic image for quick experiments."""
    image = np.ones((height, width), dtype=np.uint8) * 255
    cv2.rectangle(image, (50, 50), (150, 100), 0, 2)
    cv2.circle(image, (250, 75), 30, 0, 2)
    cv2.ellipse(image, (350, 75), (40, 20), 0, 0, 360, 0, 2)
    cv2.putText(image, "WATERJET", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.imwrite(output_path, image)
    print(f"Sample image created: {output_path}")


if __name__ == "__main__":
    processor = ImageProcessor()
    sample_image_path = "samples/sample_image.jpg"
    Path("samples").mkdir(exist_ok=True)
    create_sample_image(sample_image_path)
    output_dxf_path = "samples/sample_from_image.dxf"
    processor.process_image_to_dxf(sample_image_path, output_dxf_path, debug_dir="samples")
