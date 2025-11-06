"""
Celery Worker Configuration
===========================

Celery application configuration for background task processing.
"""

from celery import Celery
from kombu import Queue
import os
import logging

from ..config.unified_config_manager import get_config_manager

logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    config_manager = get_config_manager()
    config = config_manager.get_config()
    
    # Create Celery app
    celery_app = Celery('wjp_analyser')
    
    # Configure broker and result backend
    celery_app.conf.update(
        broker_url=config_manager.get_redis_url(),
        result_backend=config_manager.get_redis_url(),
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,
        task_soft_time_limit=240,
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
        task_routes={
            'dxf_analysis': {'queue': 'analysis'},
            'nesting_optimization': {'queue': 'nesting'},
            'batch_processing': {'queue': 'batch'},
            'cleanup_task': {'queue': 'maintenance'},
        },
        task_queues=(
            Queue('analysis', routing_key='analysis'),
            Queue('nesting', routing_key='nesting'),
            Queue('batch', routing_key='batch'),
            Queue('maintenance', routing_key='maintenance'),
        ),
        beat_schedule={
            'periodic-cleanup': {
                'task': 'periodic_cleanup',
                'schedule': 3600.0,  # Every hour
            },
            'health-check': {
                'task': 'health_check',
                'schedule': 300.0,  # Every 5 minutes
            },
        },
    )
    
    logger.info("Celery application created and configured")
    return celery_app


# Create global Celery app instance
celery_app = create_celery_app()