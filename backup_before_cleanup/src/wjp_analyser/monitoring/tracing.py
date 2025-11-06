"""
Distributed Tracing for WJP ANALYSER
====================================

Jaeger integration for distributed tracing across all components.
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time
import threading
from dataclasses import dataclass

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry not available. Tracing will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class TracingConfig:
    """Tracing configuration."""
    enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    service_name: str = "wjp-analyser"
    environment: str = "production"
    sample_rate: float = 0.1  # 10% sampling


class WJPTracer:
    """WJP ANALYSER distributed tracer."""
    
    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self.tracer = None
        self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        if not TRACING_AVAILABLE or not self.config.enabled:
            logger.info("Tracing disabled or not available")
            return
        
        try:
            # Create tracer provider
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()
            
            # Create Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            # Create span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(self.config.service_name)
            
            logger.info("Tracing initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.tracer = None
    
    def get_tracer(self):
        """Get the tracer instance."""
        return self.tracer
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a new span."""
        if not self.tracer:
            return self._dummy_span()
        
        try:
            span = self.tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span
        except Exception as e:
            logger.error(f"Failed to start span {name}: {e}")
            return self._dummy_span()
    
    def _dummy_span(self):
        """Return a dummy span when tracing is disabled."""
        return DummySpan()


class DummySpan:
    """Dummy span for when tracing is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set_attribute(self, key: str, value: Any):
        pass
    
    def set_status(self, status):
        pass
    
    def record_exception(self, exception):
        pass
    
    def finish(self):
        pass


# Global tracer instance
wjp_tracer = WJPTracer()


def trace_function(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = wjp_tracer.get_tracer()
            if not tracer:
                return func(*args, **kwargs)
            
            with tracer.start_span(operation_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_async_function(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = wjp_tracer.get_tracer()
            if not tracer:
                return await func(*args, **kwargs)
            
            with tracer.start_span(operation_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


@contextmanager
def trace_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing operations."""
    tracer = wjp_tracer.get_tracer()
    if not tracer:
        yield
        return
    
    with tracer.start_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
            span.set_attribute("success", True)
        except Exception as e:
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            span.record_exception(e)
            raise


def trace_api_request(request_id: str, method: str, endpoint: str, user_id: Optional[str] = None):
    """Trace API request."""
    attributes = {
        "request_id": request_id,
        "http.method": method,
        "http.url": endpoint,
        "component": "api"
    }
    
    if user_id:
        attributes["user.id"] = user_id
    
    return trace_operation("api_request", attributes)


def trace_dxf_analysis(file_path: str, file_size: int, user_id: Optional[str] = None):
    """Trace DXF analysis operation."""
    attributes = {
        "file.path": file_path,
        "file.size": file_size,
        "operation.type": "dxf_analysis",
        "component": "analysis"
    }
    
    if user_id:
        attributes["user.id"] = user_id
    
    return trace_operation("dxf_analysis", attributes)




def trace_nesting_optimization(parts_count: int, sheet_size: tuple, user_id: Optional[str] = None):
    """Trace nesting optimization operation."""
    attributes = {
        "nesting.parts_count": parts_count,
        "nesting.sheet_width": sheet_size[0],
        "nesting.sheet_height": sheet_size[1],
        "operation.type": "nesting_optimization",
        "component": "nesting"
    }
    
    if user_id:
        attributes["user.id"] = user_id
    
    return trace_operation("nesting_optimization", attributes)


def trace_database_operation(operation: str, table: str, duration: float):
    """Trace database operation."""
    attributes = {
        "db.operation": operation,
        "db.table": table,
        "db.duration": duration,
        "component": "database"
    }
    
    return trace_operation("database_operation", attributes)


def trace_cache_operation(operation: str, cache_key: str, hit: bool):
    """Trace cache operation."""
    attributes = {
        "cache.operation": operation,
        "cache.key": cache_key,
        "cache.hit": hit,
        "component": "cache"
    }
    
    return trace_operation("cache_operation", attributes)


def trace_celery_task(task_name: str, task_id: str, duration: float):
    """Trace Celery task execution."""
    attributes = {
        "celery.task_name": task_name,
        "celery.task_id": task_id,
        "celery.duration": duration,
        "component": "celery"
    }
    
    return trace_operation("celery_task", attributes)


def setup_instrumentation():
    """Setup OpenTelemetry instrumentation for all components."""
    if not TRACING_AVAILABLE:
        logger.warning("OpenTelemetry not available. Instrumentation skipped.")
        return
    
    try:
        # Instrument FastAPI
        FastAPIInstrumentor.instrument()
        
        # Instrument Celery
        CeleryInstrumentor().instrument()
        
        # Instrument SQLAlchemy
        SQLAlchemyInstrumentor().instrument()
        
        # Instrument Redis
        RedisInstrumentor().instrument()
        
        # Instrument Requests
        RequestsInstrumentor().instrument()
        
        logger.info("OpenTelemetry instrumentation setup complete")
        
    except Exception as e:
        logger.error(f"Failed to setup instrumentation: {e}")


def get_trace_context():
    """Get current trace context."""
    if not TRACING_AVAILABLE:
        return None
    
    try:
        span = trace.get_current_span()
        if span:
            return {
                "trace_id": format(span.get_span_context().trace_id, '032x'),
                "span_id": format(span.get_span_context().span_id, '016x')
            }
    except Exception as e:
        logger.error(f"Failed to get trace context: {e}")
    
    return None


def inject_trace_context(headers: Dict[str, str]):
    """Inject trace context into headers."""
    if not TRACING_AVAILABLE:
        return headers
    
    try:
        span = trace.get_current_span()
        if span:
            trace_id = format(span.get_span_context().trace_id, '032x')
            span_id = format(span.get_span_context().span_id, '016x')
            
            headers["X-Trace-ID"] = trace_id
            headers["X-Span-ID"] = span_id
    except Exception as e:
        logger.error(f"Failed to inject trace context: {e}")
    
    return headers
