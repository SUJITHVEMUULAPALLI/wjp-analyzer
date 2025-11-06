"""
Job Queue Manager
=================

Manages background job processing using RQ (Redis Queue).

This module provides:
- Job enqueueing
- Worker management
- Job status tracking
- Result retrieval
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

try:
    from rq import Queue, Worker, Connection
    from rq.job import Job as RQJob
    import redis
    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False
    redis = None
    Queue = None
    Worker = None


# Redis connection configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

# Queue names
QUEUE_DEFAULT = "default"
QUEUE_ANALYSIS = "analysis"
QUEUE_CONVERSION = "conversion"
QUEUE_NESTING = "nesting"
QUEUE_GCODE = "gcode"

# All queues
QUEUES = {
    QUEUE_DEFAULT,
    QUEUE_ANALYSIS,
    QUEUE_CONVERSION,
    QUEUE_NESTING,
    QUEUE_GCODE,
}


def get_redis_connection():
    """Get Redis connection."""
    if not RQ_AVAILABLE:
        raise ImportError(
            "RQ and redis are required. Install with: pip install rq redis"
        )
    
    try:
        if REDIS_URL.startswith("redis://"):
            conn = redis.from_url(REDIS_URL)
        else:
            conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        # Test connection
        conn.ping()
        return conn
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Redis: {e}")


def get_queue(queue_name: str = QUEUE_DEFAULT) -> Optional[Queue]:
    """
    Get a job queue.
    
    Args:
        queue_name: Name of the queue
        
    Returns:
        Queue instance or None if RQ unavailable
    """
    if not RQ_AVAILABLE:
        return None
    
    try:
        conn = get_redis_connection()
        return Queue(queue_name, connection=conn)
    except Exception:
        # Redis not available - return None for graceful degradation
        return None


def enqueue_job(
    queue_name: str,
    job_func,
    *args,
    job_timeout: int = 300,  # 5 minutes default
    result_ttl: int = 3600,  # 1 hour
    job_id: Optional[str] = None,
    check_existing: bool = True,
    **kwargs,
) -> Optional[str]:
    """
    Enqueue a job for background processing with optional idempotency.
    
    Args:
        queue_name: Name of the queue
        job_func: Function to execute
        *args: Positional arguments for job function
        job_timeout: Job timeout in seconds
        result_ttl: Result time-to-live in seconds
        job_id: Optional explicit job ID (for idempotency)
        check_existing: If True, check for existing job with same ID
        **kwargs: Keyword arguments for job function
        
    Returns:
        Job ID if enqueued successfully, None if queue unavailable
    """
    queue = get_queue(queue_name)
    if not queue:
        return None
    
    # Check for existing job if idempotency enabled
    if check_existing and job_id:
        try:
            conn = get_redis_connection()
            existing_job = RQJob.fetch(job_id, connection=conn)
            if existing_job and existing_job.get_status() in ('queued', 'started'):
                # Job already exists and is running
                return job_id
            elif existing_job and existing_job.get_status() == 'finished':
                # Job already completed - return existing result
                return job_id
        except Exception:
            # Job doesn't exist yet - proceed to create
            pass
    
    try:
        job = queue.enqueue(
            job_func,
            *args,
            job_timeout=job_timeout,
            result_ttl=result_ttl,
            job_id=job_id,
            **kwargs,
        )
        return job.id
    except Exception as e:
        raise RuntimeError(f"Failed to enqueue job: {e}")


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job status from Redis.
    
    Args:
        job_id: Job ID
        
    Returns:
        Dict with job status or None if not found
    """
    if not RQ_AVAILABLE:
        return None
    
    try:
        conn = get_redis_connection()
        job = RQJob.fetch(job_id, connection=conn)
        
        # Map RQ status to our status
        status_map = {
            "queued": "queued",
            "started": "running",
            "finished": "completed",
            "failed": "failed",
        }
        
        status = status_map.get(job.get_status(), "unknown")
        
        result = {
            "job_id": job_id,
            "status": status,
            "progress": None,  # RQ doesn't have built-in progress
            "result": job.result if status == "completed" else None,
            "error": str(job.exc_info) if status == "failed" else None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        }
        
        return result
    except Exception as e:
        # Job not found or other error
        return None


def cancel_job(job_id: str) -> bool:
    """
    Cancel a job.
    
    Args:
        job_id: Job ID
        
    Returns:
        True if cancelled, False otherwise
    """
    if not RQ_AVAILABLE:
        return False
    
    try:
        conn = get_redis_connection()
        job = RQJob.fetch(job_id, connection=conn)
        job.cancel()
        return True
    except Exception:
        return False


def get_queue_length(queue_name: str = QUEUE_DEFAULT) -> int:
    """Get the length of a queue."""
    queue = get_queue(queue_name)
    if not queue:
        return 0
    return len(queue)


def get_failed_jobs(queue_name: str = QUEUE_DEFAULT, limit: int = 10) -> list:
    """Get failed jobs from a queue."""
    queue = get_queue(queue_name)
    if not queue:
        return []
    
    try:
        failed = queue.failed_job_registry.get_job_ids(0, limit - 1)
        return list(failed)
    except Exception:
        return []


# Worker functions for different job types
def worker_analyze_dxf(dxf_path: str, **kwargs) -> Dict[str, Any]:
    """Worker function for DXF analysis."""
    from wjp_analyser.services.analysis_service import run_analysis
    return run_analysis(dxf_path, **kwargs)


def worker_convert_image(image_path: str, **kwargs) -> str:
    """Worker function for image to DXF conversion."""
    from wjp_analyser.image_processing.converters.enhanced_opencv_converter import (
        EnhancedOpenCVImageToDXFConverter,
    )
    converter = EnhancedOpenCVImageToDXFConverter()
    return converter.convert(image_path, **kwargs)


def worker_calculate_cost(dxf_path: str, **kwargs) -> Dict[str, Any]:
    """Worker function for cost calculation."""
    from wjp_analyser.services.costing_service import estimate_cost
    return estimate_cost(dxf_path, overrides=kwargs)


def worker_analyze_csv(csv_path: str, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Worker function for CSV analysis."""
    from wjp_analyser.services.csv_analysis_service import analyze_with_recommendations, analyze_csv
    if report:
        return analyze_with_recommendations(csv_path, report)
    return analyze_csv(csv_path)


# Check if Redis is available
def is_redis_available() -> bool:
    """Check if Redis connection is available."""
    if not RQ_AVAILABLE:
        return False
    try:
        conn = get_redis_connection()
        conn.ping()
        return True
    except Exception:
        return False

