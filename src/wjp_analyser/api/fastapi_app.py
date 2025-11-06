"""
FastAPI Core Application
========================

FastAPI-based REST API for WJP ANALYSER.
This is the canonical API that Streamlit and other clients should use.

All business logic is in services; this API layer only handles HTTP concerns.
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: fastapi is required. Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Add src to path
_THIS_DIR = Path(__file__).parent
_SRC_DIR = _THIS_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

PROJECT_ROOT = _SRC_DIR.parent.resolve()
UPLOAD_ROOT = (PROJECT_ROOT / "data" / "uploads").resolve()
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
OUTPUT_ROOT = (PROJECT_ROOT / "output").resolve()
ALLOWED_BASES = {UPLOAD_ROOT, DATA_ROOT, OUTPUT_ROOT}
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(filename: str, *, add_suffix: bool = True) -> str:
    """Return a filesystem-safe filename with optional collision protection."""
    name = Path(filename or "upload").name
    stem, ext = os.path.splitext(name)
    safe_stem = SAFE_FILENAME_RE.sub("_", stem) or "upload"
    safe_ext = SAFE_FILENAME_RE.sub("", ext)
    safe_stem = safe_stem[:48]
    ext_part = f".{safe_ext.lstrip('.')}" if safe_ext else ""
    if add_suffix:
        unique_suffix = uuid.uuid4().hex
        return f"{safe_stem}_{unique_suffix}{ext_part.lower()}"
    return f"{safe_stem}{ext_part.lower()}"


def _resolve_managed_path(path_value: str) -> Path:
    """
    Resolve a user-supplied path into managed storage directories.

    Raises HTTPException if the path attempts to escape managed roots.
    """
    if not path_value:
        raise HTTPException(status_code=400, detail="Path value is required")

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = UPLOAD_ROOT / candidate

    resolved = candidate.resolve()
    for base in ALLOWED_BASES:
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            continue

    raise HTTPException(
        status_code=400,
        detail="Access to files outside managed directories is not permitted",
    )


def _project_relative_path(path_value: Path) -> str:
    """Return the project-relative representation of a managed path."""
    try:
        return str(path_value.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path_value.resolve())

# Import services
from wjp_analyser.services.analysis_service import run_analysis
from wjp_analyser.services.costing_service import estimate_cost, estimate_cost_from_toolpath
from wjp_analyser.services.csv_analysis_service import analyze_csv, analyze_with_recommendations
from wjp_analyser.services.layered_dxf_service import write_layered_dxf_from_report
from wjp_analyser.services.editor_service import export_components_csv

# Initialize FastAPI app
app = FastAPI(
    title="WJP ANALYSER API",
    description="Waterjet DXF Analysis and Manufacturing Optimization API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Streamlit client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],  # Configure appropriately
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AnalyzeDXFRequest(BaseModel):
    """Request model for DXF analysis."""
    dxf_path: str = Field(..., description="Path to DXF file")
    material: Optional[str] = Field(None, description="Material type")
    thickness: Optional[float] = Field(None, description="Material thickness (mm)")
    kerf: Optional[float] = Field(None, description="Kerf width (mm)")
    normalize: Optional[bool] = Field(False, description="Normalize to origin")
    target_frame_w: Optional[float] = Field(None, description="Target frame width (mm)")
    target_frame_h: Optional[float] = Field(None, description="Target frame height (mm)")


class AnalyzeDXFResponse(BaseModel):
    """Response model for DXF analysis."""
    success: bool
    report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    job_id: Optional[str] = None  # For async jobs


class CostEstimateRequest(BaseModel):
    """Request model for cost estimation."""
    dxf_path: str
    material: Optional[str] = None
    thickness: Optional[float] = None
    kerf: Optional[float] = None
    rate_per_m: Optional[float] = None
    pierce_cost: Optional[float] = None
    setup_cost: Optional[float] = None


class CostEstimateResponse(BaseModel):
    """Response model for cost estimation."""
    success: bool
    cost: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str  # queued, running, completed, failed
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "WJP ANALYSER API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
    }


# DXF Analysis Endpoint
@app.post("/analyze-dxf", response_model=AnalyzeDXFResponse)
async def analyze_dxf_endpoint(
    request: AnalyzeDXFRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = False,
):
    """
    Analyze a DXF file and return analysis report.
    
    This endpoint processes DXF files and extracts:
    - Geometric metrics (length, area, perimeter)
    - Quality checks (open contours, min radius, min spacing)
    - Component extraction and layer classification
    - Grouping of similar objects
    
    Args:
        async_mode: If True, enqueue job and return job_id immediately
    """
    try:
        dxf_path = _resolve_managed_path(request.dxf_path)
        if not dxf_path.exists():
            raise HTTPException(status_code=404, detail=f"DXF file not found: {request.dxf_path}")
        
        # Prepare analysis arguments
        args_overrides = {}
        if request.material:
            args_overrides["material"] = request.material
        if request.thickness is not None:
            args_overrides["thickness"] = request.thickness
        if request.kerf is not None:
            args_overrides["kerf"] = request.kerf
        if request.normalize:
            args_overrides["normalize_mode"] = "fit" if (request.target_frame_w and request.target_frame_h) else "scale"
            if request.target_frame_w and request.target_frame_h:
                args_overrides["target_frame_w_mm"] = request.target_frame_w
                args_overrides["target_frame_h_mm"] = request.target_frame_h
                args_overrides["normalize_origin"] = True
                args_overrides["require_fit_within_frame"] = True
        
        # Check if async mode and queue available
        if async_mode:
            try:
                from wjp_analyser.api.queue_manager import enqueue_job, QUEUE_ANALYSIS, worker_analyze_dxf
                job_id = enqueue_job(
                    QUEUE_ANALYSIS,
                    worker_analyze_dxf,
                    str(dxf_path),
                    args_overrides=args_overrides if args_overrides else None,
                    job_timeout=600,  # 10 minutes for analysis
                )
                if job_id:
                    return AnalyzeDXFResponse(
                        success=True,
                        job_id=job_id,
                    )
                # Fall through to synchronous if queue unavailable
            except Exception:
                # Fall through to synchronous if queue unavailable
                pass
        
        # Synchronous execution
        report = run_analysis(
            str(dxf_path),
            out_dir=None,  # API can manage output directory
            args_overrides=args_overrides if args_overrides else None,
        )
        
        return AnalyzeDXFResponse(
            success=True,
            report=report,
        )
        
    except Exception as e:
        return AnalyzeDXFResponse(
            success=False,
            error=str(e),
        )


# Cost Estimation Endpoint
@app.post("/cost", response_model=CostEstimateResponse)
async def estimate_cost_endpoint(request: CostEstimateRequest):
    """
    Estimate manufacturing cost from DXF file.
    
    Calculates cost based on:
    - Cutting length
    - Pierce count
    - Material parameters
    - Machine time
    """
    try:
        dxf_path = _resolve_managed_path(request.dxf_path)
        if not dxf_path.exists():
            raise HTTPException(status_code=404, detail=f"DXF file not found: {request.dxf_path}")
        
        # Prepare overrides
        overrides = {}
        if request.material:
            overrides["material"] = request.material
        if request.thickness is not None:
            overrides["thickness"] = request.thickness
        if request.kerf is not None:
            overrides["kerf"] = request.kerf
        if request.rate_per_m is not None:
            overrides["rate_per_m"] = request.rate_per_m
        if request.pierce_cost is not None:
            overrides["pierce_cost"] = request.pierce_cost
        if request.setup_cost is not None:
            overrides["setup_cost"] = request.setup_cost
        
        # Calculate cost using service
        cost_result = estimate_cost(
            str(dxf_path),
            overrides=overrides if overrides else None,
        )
        
        return CostEstimateResponse(
            success=True,
            cost=cost_result,
        )
        
    except Exception as e:
        return CostEstimateResponse(
            success=False,
            error=str(e),
        )


# Image to DXF Conversion Endpoint
@app.post("/convert-image")
async def convert_image_endpoint(
    image_file: UploadFile = File(...),
    min_area: Optional[int] = None,
    threshold: Optional[int] = None,
):
    """
    Convert an image to DXF format.
    
    Supports PNG, JPG, JPEG formats.
    Uses enhanced OpenCV converter by default.
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.filename).suffix) as tmp:
            tmp.write(await image_file.read())
            tmp_path = tmp.name
        
        # Convert using enhanced converter
        from wjp_analyser.image_processing.converters.enhanced_opencv_converter import EnhancedOpenCVImageToDXFConverter
        
        converter = EnhancedOpenCVImageToDXFConverter()
        converter_params = {}
        if min_area:
            converter_params["min_area"] = min_area
        if threshold:
            converter_params["threshold"] = threshold
        
        dxf_path = converter.convert(tmp_path, **converter_params)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Return DXF file
        return FileResponse(
            dxf_path,
            media_type="application/dxf",
            filename=Path(dxf_path).name,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")


# CSV Analysis Endpoint
@app.post("/csv/ai-analysis")
async def analyze_csv_endpoint(
    csv_file: UploadFile = File(...),
    report: Optional[str] = None,  # JSON string of report
):
    """
    Analyze DXF component CSV and provide AI-powered recommendations.
    
    Returns:
    - Statistics about components
    - Warnings and recommendations
    - Executable operations (if report provided)
    """
    try:
        # Save uploaded CSV temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
            tmp.write(await csv_file.read())
            csv_path = tmp.name
        
        # Parse report if provided
        report_dict = None
        if report:
            import json
            report_dict = json.loads(report)
        
        # Analyze CSV
        if report_dict:
            analysis = analyze_with_recommendations(csv_path, report_dict)
        else:
            analysis = analyze_csv(csv_path)
        
        # Clean up temp file
        os.unlink(csv_path)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV analysis failed: {str(e)}")


# Nesting Endpoint (stub for now)
@app.post("/nest")
async def nest_endpoint(
    dxf_path: str,
    sheet_width: float,
    sheet_height: float,
    gap: float = 3.0,
    kerf: float = 1.1,
):
    """
    Optimize nesting of DXF components on a sheet.
    
    Returns nesting layout and utilization metrics.
    """
    try:
        dxf_file = _resolve_managed_path(dxf_path)
        if not dxf_file.exists():
            raise HTTPException(status_code=404, detail=f"DXF file not found: {dxf_path}")
        # TODO: Implement nesting service
        return {
            "success": True,
            "message": "Nesting endpoint - implementation pending",
            "sheet": {"width": sheet_width, "height": sheet_height},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nesting failed: {str(e)}")


# G-code Generation Endpoint (stub for now)
@app.post("/gcode")
async def generate_gcode_endpoint(
    dxf_path: str,
    material: str = "steel",
    thickness: float = 6.0,
    kerf: float = 1.1,
):
    """
    Generate G-code from DXF file.
    
    Returns NC file and toolpath information.
    """
    try:
        dxf_file = _resolve_managed_path(dxf_path)
        if not dxf_file.exists():
            raise HTTPException(status_code=404, detail=f"DXF file not found: {dxf_path}")
        # TODO: Implement G-code generation service
        return {
            "success": True,
            "message": "G-code generation endpoint - implementation pending",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"G-code generation failed: {str(e)}")


# Job Status Endpoint
@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of an async job.
    
    Returns job status, progress, and result (if completed).
    """
    try:
        from wjp_analyser.api.queue_manager import get_job_status as get_rq_job_status
        
        job_status = get_rq_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return JobStatusResponse(
            job_id=job_status["job_id"],
            status=job_status["status"],
            progress=job_status.get("progress"),
            result=job_status.get("result"),
            error=job_status.get("error"),
            created_at=datetime.fromisoformat(job_status["created_at"]) if job_status.get("created_at") else None,
            completed_at=datetime.fromisoformat(job_status["ended_at"]) if job_status.get("ended_at") else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


# File Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (DXF, image, etc.) and return its path.
    
    Files are stored in uploads/ directory.
    """
    try:
        original_name = file.filename or "upload"
        safe_name = _sanitize_filename(original_name)
        file_path = UPLOAD_ROOT / safe_name
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "success": True,
            "file_id": safe_name,
            "file_path": _project_relative_path(file_path),
            "relative_path": safe_name,
            "filename": original_name,
            "size": len(content),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Export Components CSV Endpoint
@app.post("/export/components-csv")
async def export_components_csv_endpoint(
    report: Dict[str, Any],
    output_filename: Optional[str] = None,
):
    """
    Export DXF analysis report components to CSV.
    
    Returns CSV file with component data.
    """
    try:
        import tempfile
        output_dir = Path("output/exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name_source = output_filename or f"components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        safe_filename = _sanitize_filename(safe_name_source, add_suffix=False)
        csv_path = str(output_dir / safe_filename)
        export_components_csv(report, csv_path)
        
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename=safe_filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")


# Export Layered DXF Endpoint
@app.post("/export/layered-dxf")
async def export_layered_dxf_endpoint(
    report: Dict[str, Any],
    output_filename: Optional[str] = None,
):
    """
    Export DXF analysis report components as layered DXF file.
    """
    try:
        output_dir = Path("output/exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name_source = output_filename or f"layered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dxf"
        safe_filename = _sanitize_filename(safe_name_source, add_suffix=False)
        dxf_path = str(output_dir / safe_filename)
        write_layered_dxf_from_report(report, dxf_path)
        
        return FileResponse(
            dxf_path,
            media_type="application/dxf",
            filename=safe_filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DXF export failed: {str(e)}")


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
