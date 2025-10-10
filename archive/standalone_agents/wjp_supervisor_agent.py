#!/usr/bin/env python3
"""
WJP Supervisor Agent - Automation Controller
============================================

This agent orchestrates the entire WJP automation pipeline from prompt to PDF report.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import threading

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from wjp_file_manager import WJPFileManager, JobMetadata, ProcessStage, MaterialCode
from wjp_designer_agent import DesignerAgent
from wjp_image_to_dxf_agent import ImageToDXFAgent
from wjp_dxf_analyzer_agent import DXFAnalyzerAgent
from wjp_report_generator_agent import ReportGeneratorAgent

class JobStatus(Enum):
    """Job processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobResult:
    """Result of a job processing."""
    job_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    output_files: Dict[str, str]
    errors: List[str]
    metadata: Dict[str, Any]

class SupervisorAgent:
    """Supervisor Agent that orchestrates the entire WJP automation pipeline."""
    
    def __init__(self):
        self.file_manager = WJPFileManager()
        self.job_queue = queue.Queue()
        self.active_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # Initialize agents
        self.designer_agent = DesignerAgent()
        self.image_to_dxf_agent = ImageToDXFAgent()
        self.dxf_analyzer_agent = DXFAnalyzerAgent()
        self.report_generator_agent = ReportGeneratorAgent()
        
        # Processing statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._process_job_queue, daemon=True)
        self.processing_thread.start()
    
    def submit_job(self, job_id: str, prompt: str, material: str, thickness_mm: int,
                   category: str = "Inlay Tile", dimensions_inch: list = [24, 24],
                   cut_spacing_mm: float = 3.0, min_radius_mm: float = 2.0) -> str:
        """
        Submit a new job for processing.
        
        Args:
            job_id: Unique job identifier
            prompt: Design prompt
            material: Material type
            thickness_mm: Material thickness
            category: Design category
            dimensions_inch: Design dimensions [width, height]
            cut_spacing_mm: Minimum cut spacing
            min_radius_mm: Minimum radius
            
        Returns:
            Job submission confirmation
        """
        print(f"📋 **Supervisor Agent - Submitting Job: {job_id}**")
        
        # Create job data
        job_data = {
            "job_id": job_id,
            "prompt": prompt,
            "material": material,
            "thickness_mm": thickness_mm,
            "category": category,
            "dimensions_inch": dimensions_inch,
            "cut_spacing_mm": cut_spacing_mm,
            "min_radius_mm": min_radius_mm,
            "submitted_at": datetime.now().isoformat()
        }
        
        # Add to queue
        self.job_queue.put(job_data)
        self.stats["total_jobs"] += 1
        
        print(f"✅ Job {job_id} submitted to queue")
        print(f"   Queue size: {self.job_queue.qsize()}")
        
        return f"Job {job_id} submitted successfully"
    
    def _process_job_queue(self):
        """Background thread to process job queue."""
        while True:
            try:
                # Get job from queue
                job_data = self.job_queue.get(timeout=1)
                
                # Process job
                self._process_single_job(job_data)
                
                # Mark task as done
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in job processing thread: {e}")
    
    def _process_single_job(self, job_data: Dict[str, Any]):
        """Process a single job through the entire pipeline."""
        job_id = job_data["job_id"]
        start_time = datetime.now()
        
        print(f"\n🚀 **Processing Job: {job_id}**")
        print(f"   Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize job result
        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.IN_PROGRESS,
            start_time=start_time,
            end_time=None,
            duration_seconds=0.0,
            output_files={},
            errors=[],
            metadata=job_data
        )
        
        # Add to active jobs
        self.active_jobs[job_id] = job_result
        
        try:
            # Stage 1: Designer Agent (Prompt → Image)
            print(f"   🎨 Stage 1: Designer Agent")
            image_path, designer_metadata_path = self.designer_agent.run(
                job_id=job_id,
                prompt=job_data["prompt"],
                material=job_data["material"],
                thickness_mm=job_data["thickness_mm"],
                category=job_data["category"],
                dimensions_inch=job_data["dimensions_inch"],
                cut_spacing_mm=job_data["cut_spacing_mm"],
                min_radius_mm=job_data["min_radius_mm"]
            )
            
            job_result.output_files["design_image"] = image_path
            job_result.output_files["designer_metadata"] = designer_metadata_path
            
            print(f"   ✅ Stage 1 Complete: {os.path.basename(image_path)}")
            
            # Stage 2: Image to DXF Agent
            print(f"   🔄 Stage 2: Image to DXF Agent")
            dxf_path, conversion_metadata_path = self.image_to_dxf_agent.run(designer_metadata_path)
            
            job_result.output_files["dxf_file"] = dxf_path
            job_result.output_files["conversion_metadata"] = conversion_metadata_path
            
            print(f"   ✅ Stage 2 Complete: {os.path.basename(dxf_path)}")
            
            # Stage 3: DXF Analyzer Agent
            print(f"   📊 Stage 3: DXF Analyzer Agent")
            analysis_dxf_path, analysis_json_path, analysis_image_path, csv_path = self.dxf_analyzer_agent.run(conversion_metadata_path)
            
            job_result.output_files["analysis_dxf"] = analysis_dxf_path
            job_result.output_files["analysis_json"] = analysis_json_path
            job_result.output_files["analysis_image"] = analysis_image_path
            job_result.output_files["analysis_csv"] = csv_path
            
            print(f"   ✅ Stage 3 Complete: {os.path.basename(analysis_json_path)}")
            
            # Stage 4: Report Generator Agent
            print(f"   📄 Stage 4: Report Generator Agent")
            pdf_path = self.report_generator_agent.run(analysis_json_path)
            
            job_result.output_files["pdf_report"] = pdf_path
            
            print(f"   ✅ Stage 4 Complete: {os.path.basename(pdf_path)}")
            
            # Job completed successfully
            end_time = datetime.now()
            job_result.status = JobStatus.COMPLETED
            job_result.end_time = end_time
            job_result.duration_seconds = (end_time - start_time).total_seconds()
            
            # Update statistics
            self.stats["completed_jobs"] += 1
            self.stats["total_processing_time"] += job_result.duration_seconds
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["completed_jobs"]
            )
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_result
            del self.active_jobs[job_id]
            
            print(f"🎉 **Job {job_id} Completed Successfully!**")
            print(f"   Duration: {job_result.duration_seconds:.2f} seconds")
            print(f"   Output Files: {len(job_result.output_files)}")
            
        except Exception as e:
            # Job failed
            end_time = datetime.now()
            job_result.status = JobStatus.FAILED
            job_result.end_time = end_time
            job_result.duration_seconds = (end_time - start_time).total_seconds()
            job_result.errors.append(str(e))
            
            # Update statistics
            self.stats["failed_jobs"] += 1
            
            # Move to failed jobs
            self.failed_jobs[job_id] = job_result
            del self.active_jobs[job_id]
            
            print(f"❌ **Job {job_id} Failed!**")
            print(f"   Error: {e}")
            print(f"   Duration: {job_result.duration_seconds:.2f} seconds")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        # Check active jobs
        if job_id in self.active_jobs:
            job_result = self.active_jobs[job_id]
            return {
                "job_id": job_result.job_id,
                "status": job_result.status.value,
                "start_time": job_result.start_time.isoformat(),
                "duration_seconds": job_result.duration_seconds,
                "output_files": len(job_result.output_files),
                "errors": job_result.errors
            }
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job_result = self.completed_jobs[job_id]
            return {
                "job_id": job_result.job_id,
                "status": job_result.status.value,
                "start_time": job_result.start_time.isoformat(),
                "end_time": job_result.end_time.isoformat() if job_result.end_time else None,
                "duration_seconds": job_result.duration_seconds,
                "output_files": job_result.output_files,
                "errors": job_result.errors
            }
        
        # Check failed jobs
        if job_id in self.failed_jobs:
            job_result = self.failed_jobs[job_id]
            return {
                "job_id": job_result.job_id,
                "status": job_result.status.value,
                "start_time": job_result.start_time.isoformat(),
                "end_time": job_result.end_time.isoformat() if job_result.end_time else None,
                "duration_seconds": job_result.duration_seconds,
                "output_files": job_result.output_files,
                "errors": job_result.errors
            }
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "statistics": self.stats
        }
    
    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all jobs (active, completed, failed)."""
        all_jobs = {}
        
        # Add active jobs
        for job_id, job_result in self.active_jobs.items():
            all_jobs[job_id] = {
                "status": job_result.status.value,
                "start_time": job_result.start_time.isoformat(),
                "duration_seconds": job_result.duration_seconds,
                "output_files": len(job_result.output_files),
                "errors": job_result.errors
            }
        
        # Add completed jobs
        for job_id, job_result in self.completed_jobs.items():
            all_jobs[job_id] = {
                "status": job_result.status.value,
                "start_time": job_result.start_time.isoformat(),
                "end_time": job_result.end_time.isoformat() if job_result.end_time else None,
                "duration_seconds": job_result.duration_seconds,
                "output_files": job_result.output_files,
                "errors": job_result.errors
            }
        
        # Add failed jobs
        for job_id, job_result in self.failed_jobs.items():
            all_jobs[job_id] = {
                "status": job_result.status.value,
                "start_time": job_result.start_time.isoformat(),
                "end_time": job_result.end_time.isoformat() if job_result.end_time else None,
                "duration_seconds": job_result.duration_seconds,
                "output_files": job_result.output_files,
                "errors": job_result.errors
            }
        
        return all_jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job (only if it's pending)."""
        # Note: This is a simplified implementation
        # In a real system, you'd need more sophisticated job cancellation
        if job_id in self.active_jobs:
            job_result = self.active_jobs[job_id]
            job_result.status = JobStatus.CANCELLED
            job_result.end_time = datetime.now()
            
            # Move to failed jobs
            self.failed_jobs[job_id] = job_result
            del self.active_jobs[job_id]
            
            return True
        
        return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_jobs": self.stats["total_jobs"],
            "completed_jobs": self.stats["completed_jobs"],
            "failed_jobs": self.stats["failed_jobs"],
            "success_rate": (
                self.stats["completed_jobs"] / self.stats["total_jobs"] 
                if self.stats["total_jobs"] > 0 else 0
            ),
            "total_processing_time": self.stats["total_processing_time"],
            "average_processing_time": self.stats["average_processing_time"],
            "current_queue_size": self.job_queue.qsize(),
            "active_jobs_count": len(self.active_jobs)
        }

def test_supervisor_agent():
    """Test the Supervisor Agent."""
    print("🎯 **Testing Supervisor Agent**")
    print("=" * 50)
    
    # Create supervisor agent
    supervisor = SupervisorAgent()
    
    # Test job submission
    print("📋 **Testing Job Submission**")
    
    test_jobs = [
        {
            "job_id": "SR06",
            "prompt": "Waterjet-safe Tan Brown granite tile with white marble inlay, 24x24 inch",
            "material": "Tan Brown Granite",
            "thickness_mm": 25,
            "category": "Inlay Tile",
            "dimensions_inch": [24, 24]
        },
        {
            "job_id": "MD01",
            "prompt": "Circular medallion design for granite flooring",
            "material": "Marble",
            "thickness_mm": 20,
            "category": "Medallion",
            "dimensions_inch": [36, 36]
        }
    ]
    
    # Submit jobs
    for job_data in test_jobs:
        result = supervisor.submit_job(**job_data)
        print(f"   ✅ {result}")
    
    # Wait for processing
    print(f"\n⏳ **Waiting for job processing...**")
    
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        queue_status = supervisor.get_queue_status()
        
        if queue_status["active_jobs"] == 0 and queue_status["queue_size"] == 0:
            break
        
        print(f"   Queue: {queue_status['queue_size']}, Active: {queue_status['active_jobs']}")
        time.sleep(5)
    
    # Get final status
    print(f"\n📊 **Final Processing Results**")
    
    statistics = supervisor.get_processing_statistics()
    print(f"   Total Jobs: {statistics['total_jobs']}")
    print(f"   Completed: {statistics['completed_jobs']}")
    print(f"   Failed: {statistics['failed_jobs']}")
    print(f"   Success Rate: {statistics['success_rate']:.1%}")
    print(f"   Average Processing Time: {statistics['average_processing_time']:.2f}s")
    
    # Get individual job statuses
    all_jobs = supervisor.get_all_jobs()
    for job_id, job_info in all_jobs.items():
        print(f"\n   Job {job_id}:")
        print(f"     Status: {job_info['status']}")
        print(f"     Duration: {job_info['duration_seconds']:.2f}s")
        if job_info['status'] == 'completed':
            print(f"     Output Files: {len(job_info['output_files'])}")
        elif job_info['status'] == 'failed':
            print(f"     Errors: {len(job_info['errors'])}")
    
    print("\n🎉 **Supervisor Agent Test Completed!**")
    
    return supervisor

if __name__ == "__main__":
    test_supervisor_agent()
