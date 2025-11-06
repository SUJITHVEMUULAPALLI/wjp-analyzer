"""
Prometheus Metrics Collection
=============================

Comprehensive metrics collection for WJP ANALYSER using Prometheus client.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, Info, CollectorRegistry, generate_latest
import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
import threading

logger = logging.getLogger(__name__)

# Create custom registry for WJP ANALYSER metrics
registry = CollectorRegistry()

# Authentication Metrics
auth_attempts_total = Counter(
    'wjp_auth_attempts_total',
    'Total number of authentication attempts',
    ['result', 'method'],
    registry=registry
)

auth_failures_total = Counter(
    'wjp_auth_failures_total',
    'Total number of authentication failures',
    ['reason'],
    registry=registry
)

active_sessions = Gauge(
    'wjp_active_sessions',
    'Number of active user sessions',
    registry=registry
)

# API Metrics
api_requests_total = Counter(
    'wjp_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

api_request_duration = Histogram(
    'wjp_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

api_response_size = Histogram(
    'wjp_api_response_size_bytes',
    'API response size in bytes',
    ['endpoint'],
    buckets=[1024, 10240, 102400, 1048576, 10485760],
    registry=registry
)

# Task Queue Metrics
celery_tasks_total = Counter(
    'wjp_celery_tasks_total',
    'Total number of Celery tasks',
    ['task_name', 'status'],
    registry=registry
)

celery_task_duration = Histogram(
    'wjp_celery_task_duration_seconds',
    'Celery task duration in seconds',
    ['task_name'],
    buckets=[1, 5, 10, 30, 60, 300, 600],
    registry=registry
)

celery_queue_size = Gauge(
    'wjp_celery_queue_size',
    'Number of tasks in Celery queue',
    ['queue_name'],
    registry=registry
)

# DXF Analysis Metrics
dxf_analysis_total = Counter(
    'wjp_dxf_analysis_total',
    'Total number of DXF analyses',
    ['status'],
    registry=registry
)

dxf_analysis_duration = Histogram(
    'wjp_dxf_analysis_duration_seconds',
    'DXF analysis duration in seconds',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry
)

dxf_file_size = Histogram(
    'wjp_dxf_file_size_bytes',
    'DXF file size in bytes',
    buckets=[1024, 10240, 102400, 1048576, 10485760],
    registry=registry
)

dxf_polygons_count = Histogram(
    'wjp_dxf_polygons_count',
    'Number of polygons in DXF files',
    buckets=[10, 50, 100, 500, 1000, 5000],
    registry=registry
)


# Nesting Metrics
nesting_optimization_total = Counter(
    'wjp_nesting_optimization_total',
    'Total number of nesting optimizations',
    ['status'],
    registry=registry
)

nesting_utilization_rate = Histogram(
    'wjp_nesting_utilization_rate',
    'Material utilization rate in nesting',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

nesting_parts_count = Histogram(
    'wjp_nesting_parts_count',
    'Number of parts in nesting',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
    registry=registry
)

# Database Metrics
db_connections_active = Gauge(
    'wjp_db_connections_active',
    'Number of active database connections',
    registry=registry
)

db_connections_idle = Gauge(
    'wjp_db_connections_idle',
    'Number of idle database connections',
    registry=registry
)

db_query_duration = Histogram(
    'wjp_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

db_transactions_total = Counter(
    'wjp_db_transactions_total',
    'Total number of database transactions',
    ['operation', 'status'],
    registry=registry
)

# Cache Metrics
cache_hits_total = Counter(
    'wjp_cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses_total = Counter(
    'wjp_cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

cache_size_bytes = Gauge(
    'wjp_cache_size_bytes',
    'Cache size in bytes',
    ['cache_type'],
    registry=registry
)

# System Metrics
memory_usage_bytes = Gauge(
    'wjp_memory_usage_bytes',
    'Memory usage in bytes',
    ['component'],
    registry=registry
)

cpu_usage_percent = Gauge(
    'wjp_cpu_usage_percent',
    'CPU usage percentage',
    ['component'],
    registry=registry
)

disk_usage_bytes = Gauge(
    'wjp_disk_usage_bytes',
    'Disk usage in bytes',
    ['path'],
    registry=registry
)

# Business Metrics
users_total = Gauge(
    'wjp_users_total',
    'Total number of users',
    ['status'],
    registry=registry
)

projects_total = Gauge(
    'wjp_projects_total',
    'Total number of projects',
    ['status'],
    registry=registry
)

analyses_total = Gauge(
    'wjp_analyses_total',
    'Total number of analyses',
    ['status'],
    registry=registry
)

conversions_total = Gauge(
    'wjp_conversions_total',
    'Total number of conversions',
    ['status'],
    registry=registry
)

# Error Metrics
errors_total = Counter(
    'wjp_errors_total',
    'Total number of errors',
    ['component', 'error_type'],
    registry=registry
)

# Rate Limiting Metrics
rate_limit_hits_total = Counter(
    'wjp_rate_limit_hits_total',
    'Total number of rate limit hits',
    ['endpoint', 'limit_type'],
    registry=registry
)

# Application Info
app_info = Info(
    'wjp_app_info',
    'Application information',
    registry=registry
)

# Set application info
app_info.info({
    'version': '1.0.0',
    'build_date': '2024-01-01',
    'python_version': '3.11',
    'environment': 'production'
})


class MetricsCollector:
    """Centralized metrics collection manager."""
    
    def __init__(self):
        self.registry = registry
        self._lock = threading.Lock()
    
    def record_auth_attempt(self, result: str, method: str = 'password'):
        """Record authentication attempt."""
        auth_attempts_total.labels(result=result, method=method).inc()
    
    def record_auth_failure(self, reason: str):
        """Record authentication failure."""
        auth_failures_total.labels(reason=reason).inc()
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, response_size: int):
        """Record API request metrics."""
        api_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        api_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        api_response_size.labels(endpoint=endpoint).observe(response_size)
    
    def record_celery_task(self, task_name: str, status: str, duration: float):
        """Record Celery task metrics."""
        celery_tasks_total.labels(task_name=task_name, status=status).inc()
        celery_task_duration.labels(task_name=task_name).observe(duration)
    
    def record_dxf_analysis(self, status: str, duration: float, file_size: int, 
                           polygons_count: int):
        """Record DXF analysis metrics."""
        dxf_analysis_total.labels(status=status).inc()
        dxf_analysis_duration.observe(duration)
        dxf_file_size.observe(file_size)
        dxf_polygons_count.observe(polygons_count)
    
    
    def record_nesting_optimization(self, status: str, utilization_rate: float, 
                                   parts_count: int):
        """Record nesting optimization metrics."""
        nesting_optimization_total.labels(status=status).inc()
        nesting_utilization_rate.observe(utilization_rate)
        nesting_parts_count.observe(parts_count)
    
    def record_db_operation(self, operation: str, duration: float, status: str):
        """Record database operation metrics."""
        db_query_duration.labels(operation=operation).observe(duration)
        db_transactions_total.labels(operation=operation, status=status).inc()
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics."""
        if hit:
            cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            cache_misses_total.labels(cache_type=cache_type).inc()
    
    def record_error(self, component: str, error_type: str):
        """Record error metrics."""
        errors_total.labels(component=component, error_type=error_type).inc()
    
    def record_rate_limit_hit(self, endpoint: str, limit_type: str):
        """Record rate limit hit."""
        rate_limit_hits_total.labels(endpoint=endpoint, limit_type=limit_type).inc()
    
    def update_system_metrics(self, memory_usage: Dict[str, int], 
                             cpu_usage: Dict[str, float], 
                             disk_usage: Dict[str, int]):
        """Update system resource metrics."""
        for component, usage in memory_usage.items():
            memory_usage_bytes.labels(component=component).set(usage)
        
        for component, usage in cpu_usage.items():
            cpu_usage_percent.labels(component=component).set(usage)
        
        for path, usage in disk_usage.items():
            disk_usage_bytes.labels(path=path).set(usage)
    
    def update_business_metrics(self, users: Dict[str, int], 
                               projects: Dict[str, int],
                               analyses: Dict[str, int],
                               conversions: Dict[str, int]):
        """Update business metrics."""
        for status, count in users.items():
            users_total.labels(status=status).set(count)
        
        for status, count in projects.items():
            projects_total.labels(status=status).set(count)
        
        for status, count in analyses.items():
            analyses_total.labels(status=status).set(count)
        
        for status, count in conversions.items():
            conversions_total.labels(status=status).set(count)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_api_request(func):
    """Decorator to track API request metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract request info from FastAPI request object
            if hasattr(args[0], 'method') and hasattr(args[0], 'url'):
                method = args[0].method
                endpoint = str(args[0].url.path)
                status_code = 200
                response_size = len(str(result)) if result else 0
                
                metrics_collector.record_api_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    duration=duration,
                    response_size=response_size
                )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            metrics_collector.record_error('api', type(e).__name__)
            raise
    return wrapper


def track_celery_task(func):
    """Decorator to track Celery task metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        task_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            metrics_collector.record_celery_task(
                task_name=task_name,
                status='success',
                duration=duration
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            metrics_collector.record_celery_task(
                task_name=task_name,
                status='failure',
                duration=duration
            )
            
            metrics_collector.record_error('celery', type(e).__name__)
            raise
    return wrapper


def track_dxf_analysis(func):
    """Decorator to track DXF analysis metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract metrics from result
            file_size = kwargs.get('file_size', 0)
            polygons_count = result.get('polygons_count', 0) if isinstance(result, dict) else 0
            
            metrics_collector.record_dxf_analysis(
                status='success',
                duration=duration,
                file_size=file_size,
                polygons_count=polygons_count
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            metrics_collector.record_dxf_analysis(
                status='failure',
                duration=duration,
                file_size=0,
                polygons_count=0
            )
            
            metrics_collector.record_error('dxf_analysis', type(e).__name__)
            raise
    return wrapper




def track_nesting_optimization(func):
    """Decorator to track nesting optimization metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract metrics from result
            utilization_rate = result.get('utilization_rate', 0) if isinstance(result, dict) else 0
            parts_count = result.get('parts_count', 0) if isinstance(result, dict) else 0
            
            metrics_collector.record_nesting_optimization(
                status='success',
                utilization_rate=utilization_rate,
                parts_count=parts_count
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            metrics_collector.record_nesting_optimization(
                status='failure',
                utilization_rate=0,
                parts_count=0
            )
            
            metrics_collector.record_error('nesting', type(e).__name__)
            raise
    return wrapper


# System metrics updater
def update_system_metrics():
    """Update system resource metrics."""
    import psutil
    import os
    
    # Memory usage
    memory_usage = {
        'total': psutil.virtual_memory().total,
        'available': psutil.virtual_memory().available,
        'used': psutil.virtual_memory().used,
        'process': psutil.Process().memory_info().rss
    }
    
    # CPU usage
    cpu_usage = {
        'system': psutil.cpu_percent(interval=1),
        'process': psutil.Process().cpu_percent()
    }
    
    # Disk usage
    disk_usage = {}
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_usage[partition.mountpoint] = usage.used
        except PermissionError:
            continue
    
    metrics_collector.update_system_metrics(memory_usage, cpu_usage, disk_usage)


# Business metrics updater
def update_business_metrics():
    """Update business metrics from database."""
    try:
        from ..database import get_db_session
        from ..database.models import User, Project, Analysis, Conversion
        
        with get_db_session() as session:
            # Users
            users = {
                'active': session.query(User).filter(User.is_active == True).count(),
                'total': session.query(User).count()
            }
            
            # Projects
            projects = {
                'active': session.query(Project).filter(Project.status == 'active').count(),
                'total': session.query(Project).count()
            }
            
            # Analyses
            analyses = {
                'completed': session.query(Analysis).filter(Analysis.status == 'completed').count(),
                'total': session.query(Analysis).count()
            }
            
            # Conversions
            conversions = {
                'completed': session.query(Conversion).filter(Conversion.status == 'completed').count(),
                'total': session.query(Conversion).count()
            }
            
            metrics_collector.update_business_metrics(users, projects, analyses, conversions)
            
    except Exception as e:
        logger.error(f"Failed to update business metrics: {e}")


# Periodic metrics updater
def start_metrics_updater():
    """Start periodic metrics updater."""
    import threading
    import time
    
    def updater():
        while True:
            try:
                update_system_metrics()
                update_business_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Metrics updater error: {e}")
                time.sleep(60)  # Wait longer on error
    
    thread = threading.Thread(target=updater, daemon=True)
    thread.start()
    logger.info("Metrics updater started")


# Export metrics endpoint for Prometheus
def get_metrics_endpoint():
    """Get metrics endpoint for Prometheus scraping."""
    return metrics_collector.get_metrics()
