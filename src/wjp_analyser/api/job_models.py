"""
Job Models for Async Processing
================================

Models for job queue system to handle long-running tasks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type enumeration."""
    ANALYZE_DXF = "analyze_dxf"
    CONVERT_IMAGE = "convert_image"
    NEST = "nest"
    GENERATE_GCODE = "generate_gcode"
    CALCULATE_COST = "calculate_cost"
    ANALYZE_CSV = "analyze_csv"


@dataclass
class Job:
    """Job model for async processing."""
    job_id: str
    job_type: JobType
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 to 1.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parameters": self.parameters,
        }


class JobResponse(BaseModel):
    """Pydantic model for job API responses."""
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# In-memory job store (will be replaced with Redis/database)
_job_store: Dict[str, Job] = {}


def create_job(job_type: JobType, parameters: Dict[str, Any]) -> Job:
    """Create a new job."""
    import uuid
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        job_type=job_type,
        parameters=parameters,
    )
    _job_store[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    """Get a job by ID."""
    return _job_store.get(job_id)


def update_job_status(
    job_id: str,
    status: JobStatus,
    progress: Optional[float] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Optional[Job]:
    """Update job status."""
    job = _job_store.get(job_id)
    if not job:
        return None
    
    job.status = status
    if progress is not None:
        job.progress = progress
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    
    if status == JobStatus.RUNNING and not job.started_at:
        job.started_at = datetime.now()
    elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        job.completed_at = datetime.now()
        job.progress = 1.0
    
    return job





