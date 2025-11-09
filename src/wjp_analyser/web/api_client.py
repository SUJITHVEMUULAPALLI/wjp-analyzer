"""
API Client for Streamlit Pages
===============================

Lightweight HTTP client for Streamlit pages to interact with FastAPI backend.

This client provides:
- Easy-to-use functions for each API endpoint
- Automatic error handling
- Session management
- File upload/download helpers
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Union
from pathlib import Path
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Default API base URL (can be overridden)
DEFAULT_API_URL = os.getenv("WJP_API_URL", "http://127.0.0.1:8000")


class WJPAPIClient:
    """Client for WJP ANALYSER API."""
    
    def __init__(self, base_url: str = DEFAULT_API_URL):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required. Install with: pip install requests")
        
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def analyze_dxf(
        self,
        dxf_path: str,
        material: Optional[str] = None,
        thickness: Optional[float] = None,
        kerf: Optional[float] = None,
        normalize: bool = False,
        target_frame_w: Optional[float] = None,
        target_frame_h: Optional[float] = None,
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze a DXF file.
        
        Args:
            dxf_path: Path to DXF file
            material: Material type
            thickness: Material thickness (mm)
            kerf: Kerf width (mm)
            normalize: Normalize to origin
            target_frame_w: Target frame width (mm)
            target_frame_h: Target frame height (mm)
            async_mode: If True, return job_id for async processing
            
        Returns:
            Analysis report or job_id if async_mode=True
        """
        payload = {
            "dxf_path": dxf_path,
        }
        if material:
            payload["material"] = material
        if thickness is not None:
            payload["thickness"] = thickness
        if kerf is not None:
            payload["kerf"] = kerf
        if normalize:
            payload["normalize"] = True
            if target_frame_w and target_frame_h:
                payload["target_frame_w"] = target_frame_w
                payload["target_frame_h"] = target_frame_h
        
        params = {}
        if async_mode:
            params["async_mode"] = "true"
        
        response = self.session.post(
            f"{self.base_url}/analyze-dxf",
            json=payload,
            params=params,
        )
        response.raise_for_status()
        return response.json()
    
    def estimate_cost(
        self,
        dxf_path: str,
        material: Optional[str] = None,
        thickness: Optional[float] = None,
        kerf: Optional[float] = None,
        rate_per_m: Optional[float] = None,
        pierce_cost: Optional[float] = None,
        setup_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate cost for DXF file.
        
        Args:
            dxf_path: Path to DXF file
            material: Material type
            thickness: Material thickness (mm)
            kerf: Kerf width (mm)
            rate_per_m: Rate per meter
            pierce_cost: Cost per pierce
            setup_cost: Setup cost
            
        Returns:
            Cost estimate
        """
        payload = {
            "dxf_path": dxf_path,
        }
        if material:
            payload["material"] = material
        if thickness is not None:
            payload["thickness"] = thickness
        if kerf is not None:
            payload["kerf"] = kerf
        if rate_per_m is not None:
            payload["rate_per_m"] = rate_per_m
        if pierce_cost is not None:
            payload["pierce_cost"] = pierce_cost
        if setup_cost is not None:
            payload["setup_cost"] = setup_cost
        
        response = self.session.post(
            f"{self.base_url}/cost",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def convert_image(
        self,
        image_path: str,
        min_area: Optional[int] = None,
        threshold: Optional[int] = None,
    ) -> str:
        """
        Convert image to DXF.
        
        Args:
            image_path: Path to image file
            min_area: Minimum area for objects
            threshold: Threshold value
            
        Returns:
            Path to generated DXF file
        """
        params = {}
        if min_area:
            params["min_area"] = min_area
        if threshold:
            params["threshold"] = threshold
        
        with open(image_path, "rb") as f:
            files = {"image_file": (Path(image_path).name, f, "image/png")}
            response = self.session.post(
                f"{self.base_url}/convert-image",
                files=files,
                params=params,
            )
            response.raise_for_status()
            
            # Save DXF file
            output_path = str(Path(image_path).with_suffix(".dxf"))
            with open(output_path, "wb") as out_f:
                out_f.write(response.content)
            
            return output_path
    
    def analyze_csv(
        self,
        csv_path: str,
        report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze CSV and get recommendations.
        
        Args:
            csv_path: Path to CSV file
            report: Optional DXF analysis report
            
        Returns:
            CSV analysis with recommendations
        """
        params = {}
        if report:
            params["report"] = json.dumps(report)
        
        with open(csv_path, "rb") as f:
            files = {"csv_file": (Path(csv_path).name, f, "text/csv")}
            response = self.session.post(
                f"{self.base_url}/csv/ai-analysis",
                files=files,
                params=params,
            )
            response.raise_for_status()
            return response.json()
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload a file to the API.
        
        Args:
            file_path: Path to file to upload
            
        Returns:
            Upload result with file_path
        """
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f)}
            response = self.session.post(
                f"{self.base_url}/upload",
                files=files,
            )
            response.raise_for_status()
            return response.json()
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of an async job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status and result
        """
        response = self.session.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def export_components_csv(
        self,
        report: Dict[str, Any],
        output_filename: Optional[str] = None,
    ) -> bytes:
        """
        Export components to CSV.
        
        Args:
            report: Analysis report
            output_filename: Optional output filename
            
        Returns:
            CSV file bytes
        """
        params = {}
        if output_filename:
            params["output_filename"] = output_filename
        
        response = self.session.post(
            f"{self.base_url}/export/components-csv",
            json=report,
            params=params,
        )
        response.raise_for_status()
        return response.content
    
    def export_layered_dxf(
        self,
        report: Dict[str, Any],
        output_filename: Optional[str] = None,
    ) -> bytes:
        """
        Export layered DXF.
        
        Args:
            report: Analysis report
            output_filename: Optional output filename
            
        Returns:
            DXF file bytes
        """
        params = {}
        if output_filename:
            params["output_filename"] = output_filename
        
        response = self.session.post(
            f"{self.base_url}/export/layered-dxf",
            json=report,
            params=params,
        )
        response.raise_for_status()
        return response.content
    
    def nest(
        self,
        dxf_path: str,
        sheet_width: float,
        sheet_height: float,
        gap: float = 3.0,
        kerf: float = 1.1,
    ) -> Dict[str, Any]:
        """
        Optimize nesting.
        
        Args:
            dxf_path: Path to DXF file
            sheet_width: Sheet width (mm)
            sheet_height: Sheet height (mm)
            gap: Minimum gap (mm)
            kerf: Kerf width (mm)
            
        Returns:
            Nesting result
        """
        params = {
            "dxf_path": dxf_path,
            "sheet_width": sheet_width,
            "sheet_height": sheet_height,
            "gap": gap,
            "kerf": kerf,
        }
        response = self.session.post(
            f"{self.base_url}/nest",
            params=params,
        )
        response.raise_for_status()
        return response.json()
    
    def generate_gcode(
        self,
        dxf_path: str,
        material: str = "steel",
        thickness: float = 6.0,
        kerf: float = 1.1,
    ) -> Dict[str, Any]:
        """
        Generate G-code.
        
        Args:
            dxf_path: Path to DXF file
            material: Material type
            thickness: Material thickness (mm)
            kerf: Kerf width (mm)
            
        Returns:
            G-code generation result
        """
        params = {
            "dxf_path": dxf_path,
            "material": material,
            "thickness": thickness,
            "kerf": kerf,
        }
        response = self.session.post(
            f"{self.base_url}/gcode",
            params=params,
        )
        response.raise_for_status()
        return response.json()


# Global client instance (singleton pattern)
_client: Optional[WJPAPIClient] = None


def get_api_client(base_url: Optional[str] = None) -> WJPAPIClient:
    """
    Get or create API client instance.
    
    Args:
        base_url: Optional base URL override
        
    Returns:
        API client instance
    """
    global _client
    
    if _client is None or base_url:
        _client = WJPAPIClient(base_url or DEFAULT_API_URL)
    
    return _client


def is_api_available(base_url: Optional[str] = None) -> bool:
    """
    Check if API is available.
    
    Args:
        base_url: Optional base URL override
        
    Returns:
        True if API is available, False otherwise
    """
    try:
        client = get_api_client(base_url)
        client.health_check()
        return True
    except Exception:
        return False








