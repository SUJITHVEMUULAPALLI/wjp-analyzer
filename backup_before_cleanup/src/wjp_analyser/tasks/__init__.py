"""
Celery Task Definitions
=======================

Background task definitions for async processing of long-running operations.
"""

import os
import time
from typing import Dict, Any, Optional
from celery import current_task
from datetime import datetime
import logging

from ..workers.celery_app import celery_app
from ..database import get_db_session
from ..database.models import Analysis, Conversion, Nesting
from ..analysis.dxf_analyzer import analyze_dxf, AnalyzeArgs
from ..utils.error_handler import WJPAnalyserError, ErrorContext, safe_execute
from ..monitoring.metrics import track_celery_task

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='dxf_analysis')
def analyze_dxf_task(self, analysis_id: str, dxf_path: str, user_id: str,
                    analysis_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task for DXF analysis.
    
    Args:
        analysis_id: Database ID of the analysis record
        dxf_path: Path to the DXF file
        user_id: ID of the user requesting the analysis
        analysis_params: Analysis parameters
    
    Returns:
        Analysis results
    """
    start_time = time.time()
    task_id = self.request.id
    
    logger.info(f"Starting DXF analysis task {task_id} for analysis {analysis_id}")
    
    try:
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting analysis', 'progress': 0}
        )
        
        # Get database session
        session = get_db_session()
        
        try:
            # Update analysis status to processing
            analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
            if not analysis:
                raise WJPAnalyserError(f"Analysis {analysis_id} not found")
            
            analysis.status = 'processing'
            session.commit()
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Loading DXF file', 'progress': 20}
            )
            
            # Create analysis arguments
            analyze_args = AnalyzeArgs(
                dxf_path=dxf_path,
                material=analysis_params.get('material', 'steel'),
                thickness=analysis_params.get('thickness', 6.0),
                kerf=analysis_params.get('kerf', 1.1),
                cutting_speed=analysis_params.get('cutting_speed', 1200.0),
                cost_per_meter=analysis_params.get('cost_per_meter', 50.0),
                sheet_width=analysis_params.get('sheet_width', 3000.0),
                sheet_height=analysis_params.get('sheet_height', 1500.0),
                spacing=analysis_params.get('spacing', 10.0)
            )
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Performing analysis', 'progress': 40}
            )
            
            # Perform analysis
            results = analyze_dxf(analyze_args)
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Saving results', 'progress': 80}
            )
            
            # Update analysis record
            processing_time = time.time() - start_time
            analysis.status = 'completed'
            analysis.results = results
            analysis.processing_time = processing_time
            analysis.completed_at = datetime.utcnow()
            session.commit()
            
            # Track metrics
            track_celery_task('dxf_analysis', 'completed', processing_time)
            
            logger.info(f"DXF analysis task {task_id} completed successfully")
            
            return {
                'success': True,
                'analysis_id': analysis_id,
                'results': results,
                'processing_time': processing_time
            }
            
        finally:
            session.close()
            
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Update analysis status to failed
        session = get_db_session()
        try:
            analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
            if analysis:
                analysis.status = 'failed'
                analysis.error_message = str(e)
                analysis.processing_time = processing_time
                session.commit()
        finally:
            session.close()
        
        # Track metrics
        track_celery_task('dxf_analysis', 'failed', processing_time)
        
        logger.error(f"DXF analysis task {task_id} failed: {e}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'processing_time': processing_time}
        )
        
        raise




@celery_app.task(bind=True, name='nesting_optimization')
def optimize_nesting_task(self, nesting_id: str, input_dxf_path: str, user_id: str,
                         nesting_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task for nesting optimization.
    
    Args:
        nesting_id: Database ID of the nesting record
        input_dxf_path: Path to the input DXF file
        user_id: ID of the user requesting the nesting
        nesting_params: Nesting parameters
    
    Returns:
        Nesting results
    """
    start_time = time.time()
    task_id = self.request.id
    
    logger.info(f"Starting nesting optimization task {task_id} for nesting {nesting_id}")
    
    try:
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting nesting', 'progress': 0}
        )
        
        # Get database session
        session = get_db_session()
        
        try:
            # Update nesting status to processing
            nesting = session.query(Nesting).filter(Nesting.id == nesting_id).first()
            if not nesting:
                raise WJPAnalyserError(f"Nesting {nesting_id} not found")
            
            nesting.status = 'processing'
            session.commit()
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Loading DXF file', 'progress': 20}
            )
            
            # Import nesting modules
            from ..nesting.nesting_optimizer import NestingOptimizer
            
            # Create nesting optimizer
            optimizer = NestingOptimizer()
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Optimizing layout', 'progress': 40}
            )
            
            # Perform nesting optimization
            results = optimizer.optimize_nesting(
                input_dxf_path=input_dxf_path,
                sheet_width=nesting_params.get('sheet_width', 3000.0),
                sheet_height=nesting_params.get('sheet_height', 1500.0),
                spacing=nesting_params.get('spacing', 10.0),
                algorithm=nesting_params.get('algorithm', 'rectangular')
            )
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Saving results', 'progress': 80}
            )
            
            # Update nesting record
            processing_time = time.time() - start_time
            nesting.status = 'completed'
            nesting.results = results
            nesting.processing_time = processing_time
            nesting.completed_at = datetime.utcnow()
            session.commit()
            
            # Track metrics
            track_celery_task('nesting_optimization', 'completed', processing_time)
            
            logger.info(f"Nesting optimization task {task_id} completed successfully")
            
            return {
                'success': True,
                'nesting_id': nesting_id,
                'results': results,
                'processing_time': processing_time
            }
            
        finally:
            session.close()
            
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Update nesting status to failed
        session = get_db_session()
        try:
            nesting = session.query(Nesting).filter(Nesting.id == nesting_id).first()
            if nesting:
                nesting.status = 'failed'
                nesting.error_message = str(e)
                nesting.processing_time = processing_time
                session.commit()
        finally:
            session.close()
        
        # Track metrics
        track_celery_task('nesting_optimization', 'failed', processing_time)
        
        logger.error(f"Nesting optimization task {task_id} failed: {e}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'processing_time': processing_time}
        )
        
        raise


@celery_app.task(bind=True, name='batch_processing')
def batch_process_task(self, task_list: list, user_id: str) -> Dict[str, Any]:
    """
    Background task for batch processing multiple operations.
    
    Args:
        task_list: List of tasks to process
        user_id: ID of the user requesting the batch processing
    
    Returns:
        Batch processing results
    """
    start_time = time.time()
    task_id = self.request.id
    
    logger.info(f"Starting batch processing task {task_id} with {len(task_list)} tasks")
    
    try:
        results = []
        completed = 0
        failed = 0
        
        for i, task_info in enumerate(task_list):
            try:
                # Update progress
                progress = int((i / len(task_list)) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'status': f'Processing task {i+1}/{len(task_list)}',
                        'progress': progress,
                        'completed': completed,
                        'failed': failed
                    }
                )
                
                # Process individual task
                task_type = task_info.get('type')
                task_data = task_info.get('data', {})
                
                if task_type == 'dxf_analysis':
                    result = analyze_dxf_task.delay(
                        task_data['analysis_id'],
                        task_data['dxf_path'],
                        user_id,
                        task_data['analysis_params']
                    )
                elif task_type == 'nesting_optimization':
                    result = optimize_nesting_task.delay(
                        task_data['nesting_id'],
                        task_data['input_dxf_path'],
                        user_id,
                        task_data['nesting_params']
                    )
                else:
                    raise WJPAnalyserError(f"Unknown task type: {task_type}")
                
                # Wait for result
                task_result = result.get(timeout=300)  # 5 minute timeout per task
                
                if task_result.get('success'):
                    completed += 1
                else:
                    failed += 1
                
                results.append({
                    'task_id': task_info.get('id'),
                    'type': task_type,
                    'result': task_result
                })
                
            except Exception as e:
                failed += 1
                logger.error(f"Batch task {i+1} failed: {e}")
                results.append({
                    'task_id': task_info.get('id'),
                    'type': task_type,
                    'error': str(e)
                })
        
        processing_time = time.time() - start_time
        
        # Track metrics
        track_celery_task('batch_processing', 'completed', processing_time)
        
        logger.info(f"Batch processing task {task_id} completed: {completed} successful, {failed} failed")
        
        return {
            'success': True,
            'total_tasks': len(task_list),
            'completed': completed,
            'failed': failed,
            'results': results,
            'processing_time': processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Track metrics
        track_celery_task('batch_processing', 'failed', processing_time)
        
        logger.error(f"Batch processing task {task_id} failed: {e}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'processing_time': processing_time}
        )
        
        raise


@celery_app.task(bind=True, name='cleanup_task')
def cleanup_task(self) -> Dict[str, Any]:
    """
    Background task for cleanup operations.
    
    Returns:
        Cleanup results
    """
    start_time = time.time()
    task_id = self.request.id
    
    logger.info(f"Starting cleanup task {task_id}")
    
    try:
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting cleanup', 'progress': 0}
        )
        
        # Get database session
        session = get_db_session()
        
        try:
            # Clean up expired sessions
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Cleaning expired sessions', 'progress': 25}
            )
            
            from datetime import datetime, timedelta
            expired_sessions = session.query(UserSession).filter(
                UserSession.expires_at < datetime.utcnow()
            ).all()
            
            for session_obj in expired_sessions:
                session.delete(session_obj)
            
            session.commit()
            
            # Clean up old audit logs (keep last 90 days)
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Cleaning old audit logs', 'progress': 50}
            )
            
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            old_logs = session.query(AuditLog).filter(
                AuditLog.created_at < cutoff_date
            ).all()
            
            for log in old_logs:
                session.delete(log)
            
            session.commit()
            
            # Clean up temporary files
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Cleaning temporary files', 'progress': 75}
            )
            
            import os
            import glob
            temp_dir = "temp"
            if os.path.exists(temp_dir):
                temp_files = glob.glob(os.path.join(temp_dir, "*"))
                for temp_file in temp_files:
                    try:
                        if os.path.isfile(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {temp_file}: {e}")
            
            processing_time = time.time() - start_time
            
            # Track metrics
            track_celery_task('cleanup', 'completed', processing_time)
            
            logger.info(f"Cleanup task {task_id} completed successfully")
            
            return {
                'success': True,
                'expired_sessions_cleaned': len(expired_sessions),
                'old_logs_cleaned': len(old_logs),
                'processing_time': processing_time
            }
            
        finally:
            session.close()
            
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Track metrics
        track_celery_task('cleanup', 'failed', processing_time)
        
        logger.error(f"Cleanup task {task_id} failed: {e}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'processing_time': processing_time}
        )
        
        raise


# Periodic tasks
@celery_app.task(name='periodic_cleanup')
def periodic_cleanup():
    """Periodic cleanup task."""
    return cleanup_task.delay()


@celery_app.task(name='health_check')
def health_check():
    """Health check task."""
    try:
        # Check database connection
        session = get_db_session()
        session.execute("SELECT 1")
        session.close()
        
        # Check Redis connection
        from ..workers.celery_app import celery_app
        celery_app.control.inspect().stats()
        
        return {'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {'status': 'unhealthy', 'error': str(e), 'timestamp': datetime.utcnow().isoformat()}