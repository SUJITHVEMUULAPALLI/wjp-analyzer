"""
API Client Wrapper with Fallback
=================================

Wrapper that uses API client if available, otherwise falls back to direct service calls.
This provides a smooth migration path while maintaining backward compatibility.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# Try to use API client, fallback to services
try:
    from .api_client import get_api_client, is_api_available, WJPAPIClient
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False

# Direct service imports (fallback)
from ..services.analysis_service import run_analysis as svc_run_analysis, summarize_for_quote
from ..services.costing_service import estimate_cost as svc_estimate_cost, estimate_cost_from_toolpath
from ..services.layered_dxf_service import write_layered_dxf_from_report as svc_write_layered_dxf
from ..services.editor_service import export_components_csv as svc_export_components_csv
from ..services.csv_analysis_service import analyze_csv as svc_analyze_csv, analyze_with_recommendations


# Global setting: use API or direct services
USE_API = os.getenv("WJP_USE_API", "true").lower() == "true"


def _get_client_or_fallback():
    """Get API client if available, otherwise None."""
    if not USE_API or not API_CLIENT_AVAILABLE:
        return None
    
    try:
        if is_api_available():
            return get_api_client()
    except Exception:
        pass
    
    return None


def analyze_dxf(
    dxf_path: str,
    out_dir: Optional[str] = None,
    args_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze DXF file - uses API if available, otherwise direct service.
    
    Args:
        dxf_path: Path to DXF file
        out_dir: Output directory (ignored if using API)
        args_overrides: Analysis arguments override
        **kwargs: Additional arguments
        
    Returns:
        Analysis report
    """
    client = _get_client_or_fallback()
    
    if client:
        # Use API
        try:
            api_args = args_overrides or {}
            result = client.analyze_dxf(
                dxf_path=dxf_path,
                material=api_args.get("material"),
                thickness=api_args.get("thickness"),
                kerf=api_args.get("kerf"),
                normalize=api_args.get("normalize_mode") == "fit",
                target_frame_w=api_args.get("target_frame_w_mm"),
                target_frame_h=api_args.get("target_frame_h_mm"),
            )
            if result.get("success") and result.get("report"):
                return result["report"]
            elif result.get("error"):
                raise Exception(f"API error: {result['error']}")
        except Exception as e:
            # Fallback to service on API error
            pass
    
    # Fallback to direct service call
    return svc_run_analysis(dxf_path, out_dir=out_dir, args_overrides=args_overrides)


def estimate_cost(
    dxf_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate cost - uses API if available, otherwise direct service.
    """
    client = _get_client_or_fallback()
    
    if client:
        try:
            result = client.estimate_cost(
                dxf_path=dxf_path,
                material=overrides.get("material") if overrides else None,
                thickness=overrides.get("thickness") if overrides else None,
                kerf=overrides.get("kerf") if overrides else None,
                rate_per_m=overrides.get("rate_per_m") if overrides else None,
                pierce_cost=overrides.get("pierce_cost") if overrides else None,
                setup_cost=overrides.get("setup_cost") if overrides else None,
            )
            if result.get("success") and result.get("cost"):
                return result["cost"]
            elif result.get("error"):
                raise Exception(f"API error: {result['error']}")
        except Exception:
            pass
    
    # Fallback to direct service
    return svc_estimate_cost(dxf_path, overrides=overrides)


def write_layered_dxf_from_report(
    report: Dict[str, Any],
    output_path: str,
    selected_only: bool = False,
) -> str:
    """
    Write layered DXF - uses API if available, otherwise direct service.
    """
    client = _get_client_or_fallback()
    
    if client:
        try:
            # API returns bytes, need to save
            dxf_bytes = client.export_layered_dxf(report, output_filename=os.path.basename(output_path))
            with open(output_path, "wb") as f:
                f.write(dxf_bytes)
            return output_path
        except Exception:
            pass
    
    # Fallback to direct service
    return svc_write_layered_dxf(report, output_path, selected_only=selected_only)


def export_components_csv(
    report: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Export components CSV - uses API if available, otherwise direct service.
    """
    client = _get_client_or_fallback()
    
    if client:
        try:
            csv_bytes = client.export_components_csv(report, output_filename=os.path.basename(output_path))
            with open(output_path, "wb") as f:
                f.write(csv_bytes)
            return output_path
        except Exception:
            pass
    
    # Fallback to direct service
    return svc_export_components_csv(report, output_path)


def analyze_csv(
    csv_path: str,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze CSV - uses API if available, otherwise direct service.
    """
    client = _get_client_or_fallback()
    
    if client:
        try:
            return client.analyze_csv(csv_path, report=report)
        except Exception:
            pass
    
    # Fallback to direct service
    if report:
        return analyze_with_recommendations(csv_path, report)
    return svc_analyze_csv(csv_path)


def convert_image(
    image_path: str,
    min_area: Optional[int] = None,
    threshold: Optional[int] = None,
) -> str:
    """
    Convert image to DXF - uses API if available, otherwise direct service.
    """
    client = _get_client_or_fallback()
    
    if client:
        try:
            return client.convert_image(image_path, min_area=min_area, threshold=threshold)
        except Exception:
            pass
    
    # Fallback to direct service
    from ..image_processing.converters.enhanced_opencv_converter import EnhancedOpenCVImageToDXFConverter
    converter = EnhancedOpenCVImageToDXFConverter()
    converter_params = {}
    if min_area:
        converter_params["min_area"] = min_area
    if threshold:
        converter_params["threshold"] = threshold
    return converter.convert(image_path, **converter_params)


# Re-export summarize_for_quote (no API equivalent needed)
summarize_for_quote = summarize_for_quote





