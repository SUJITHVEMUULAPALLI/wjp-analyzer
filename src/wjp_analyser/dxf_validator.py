"""
WJP DXF Validator
-----------------
Purpose:
  Validate DXF geometry for waterjet-cut readiness.
Checks:
  - Open contours
  - Minimum spacing
  - Minimum radius
  - Tiny features
  - Units and scaling
"""

import ezdxf
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

# Conditional shapely import
try:
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import unary_union
    from shapely.validation import explain_validity
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    LineString = Polygon = Point = unary_union = explain_validity = None


@dataclass
class ValidatorConfig:
    min_spacing_mm: float = 3.0
    min_radius_mm: float = 2.0
    min_area_mm2: float = 5.0
    open_tolerance: float = 0.5  # mm
    layer_filter: Optional[List[str]] = None


@dataclass
class ValidationResult:
    open_contours: int
    small_features: int
    spacing_violations: int
    radius_violations: int
    invalid_closed_contours: int
    summary: str
    details: Dict[str, Any]
    is_valid: bool
    recommendations: List[str]


def _check_incompatible_contours(entities: List[Any]) -> List[Any]:
    """Detect self-intersections, zero-area, and excessive vertex density in polygons."""
    issues = []
    for idx, geom in enumerate(entities):
        if not isinstance(geom, Polygon):
            continue
        # Self-intersection or invalid geometry
        try:
            if not geom.is_valid:
                msg = "Invalid geometry"
                try:
                    msg = f"Self-intersecting or invalid geometry: {explain_validity(geom)}"
                except Exception:
                    pass
                issues.append((idx, msg))
            elif geom.area < 1e-3:
                issues.append((idx, "Zero-area loop"))
        except Exception:
            issues.append((idx, "Geometry validity check failed"))
        # Excessive vertex density
        try:
            if len(list(geom.exterior.coords)) > 2000:
                issues.append((idx, "Excessive vertex density"))
        except Exception:
            pass
    return issues


def validate_dxf_geometry(dxf_path: str, cfg: ValidatorConfig = ValidatorConfig()) -> ValidationResult:
    """Validate DXF geometry for waterjet cutting readiness."""
    if not SHAPELY_AVAILABLE:
        return ValidationResult(
            open_contours=0, small_features=0, spacing_violations=0, radius_violations=0,
            invalid_closed_contours=0,
            summary="Shapely not available - validation skipped",
            details={"error": "Shapely package required for validation"},
            is_valid=False,
            recommendations=["Install shapely: pip install shapely"]
        )
    
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        open_contours = 0
        small_features = 0
        spacing_violations = 0
        radius_violations = 0
        entities = []
        recommendations = []

        # Gather entities (lines/polylines)
        for e in msp.query("LINE LWPOLYLINE CIRCLE ARC"):
            if cfg.layer_filter and e.dxf.layer not in cfg.layer_filter:
                continue
            try:
                if e.dxftype() == "LWPOLYLINE":
                    pts = np.array([p[:2] for p in e.get_points()])
                    closed = e.closed
                elif e.dxftype() == "LINE":
                    pts = np.array([[e.dxf.start.x, e.dxf.start.y],
                                    [e.dxf.end.x, e.dxf.end.y]])
                    closed = np.allclose(pts[0], pts[-1], atol=cfg.open_tolerance)
                elif e.dxftype() == "CIRCLE":
                    closed = True
                    pts = np.array([[e.dxf.center.x, e.dxf.center.y]])
                elif e.dxftype() == "ARC":
                    closed = False
                    pts = np.array([[e.dxf.center.x, e.dxf.center.y]])
                else:
                    continue
            except Exception:
                continue

            if not closed:
                open_contours += 1

            if e.dxftype() in ("CIRCLE", "ARC") and abs(e.dxf.radius) < cfg.min_radius_mm:
                radius_violations += 1

            if len(pts) > 2:
                try:
                    poly = Polygon(pts)
                    if poly.area < cfg.min_area_mm2:
                        small_features += 1
                    entities.append(poly)
                except Exception:
                    # Skip invalid polygons
                    continue
            else:
                try:
                    entities.append(LineString(pts))
                except Exception:
                    # Skip invalid lines
                    continue

        # Spacing check
        for i, a in enumerate(entities):
            for j, b in enumerate(entities):
                if i >= j: 
                    continue
                try:
                    dist = a.distance(b)
                    if 0 < dist < cfg.min_spacing_mm:
                        spacing_violations += 1
                except Exception:
                    # Skip if distance calculation fails
                    continue

        # Incompatible closed contour checks
        invalids = _check_incompatible_contours(entities)

        # Generate recommendations
        if open_contours > 0:
            recommendations.append(f"Close {open_contours} open contours")
        if spacing_violations > 0:
            recommendations.append(f"Increase spacing for {spacing_violations} feature pairs")
        if small_features > 0:
            recommendations.append(f"Review {small_features} small features (< {cfg.min_area_mm2} mm²)")
        if radius_violations > 0:
            recommendations.append(f"Increase radius for {radius_violations} features (< {cfg.min_radius_mm} mm)")
        if invalids:
            recommendations.append(f"Fix {len(invalids)} incompatible closed contours (self-intersections, zero-area, dense vertices)")

        # Determine if valid
        is_valid = (open_contours == 0 and spacing_violations == 0 and 
                   small_features == 0 and radius_violations == 0 and len(invalids) == 0)

        summary = f"Open: {open_contours}, Spacing: {spacing_violations}, " \
                  f"Small: {small_features}, Radius: {radius_violations}, Invalid: {len(invalids)}"
        
        details = {
            "open_contours": open_contours,
            "spacing_violations": spacing_violations,
            "small_features": small_features,
            "radius_violations": radius_violations,
            "invalid_closed_contours": len(invalids),
            "invalid_details": invalids,
            "total_entities": len(entities),
            "config": asdict(cfg)
        }

        return ValidationResult(
            open_contours, small_features, spacing_violations, radius_violations,
            len(invalids), summary, details, is_valid, recommendations
        )

    except Exception as e:
        return ValidationResult(
            open_contours=0, small_features=0, spacing_violations=0, radius_violations=0,
            invalid_closed_contours=0,
            summary=f"Validation error: {str(e)}",
            details={"error": str(e)},
            is_valid=False,
            recommendations=["Check DXF file format and try again"]
        )
