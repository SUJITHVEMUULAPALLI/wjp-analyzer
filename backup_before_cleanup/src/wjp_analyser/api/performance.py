"""
API Performance Optimization
=============================

High-performance API optimizations for <500ms response times.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from functools import wraps
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import gzip
from dataclasses import dataclass

from ..monitoring.metrics import track_api_request, metrics_collector
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class APIPerformanceConfig:
    """API performance configuration."""
    enable_response_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    enable_response_compression: bool = True
    max_response_size_for_compression: int = 1024  # 1KB
    enable_async_processing: bool = True
    max_concurrent_requests: int = 100
    response_timeout: float = 30.0


class APIPerformanceOptimizer:
    """API performance optimizer with caching and async processing."""
    
    def __init__(self, config: Optional[APIPerformanceConfig] = None):
        self.config = config or APIPerformanceConfig()
        self.cache_manager = CacheManager()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        self._response_cache = {}
        self._cache_lock = threading.Lock()
    
    def optimize_response(self, endpoint: str, response_data: Any, 
                         user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize API response with caching and compression.
        
        Args:
            endpoint: API endpoint
            response_data: Response data to optimize
            user_id: User ID for cache key
            
        Returns:
            Optimized response
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(endpoint, response_data, user_id)
            
            # Check cache
            if self.config.enable_response_caching:
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    logger.debug(f"Cache hit for endpoint {endpoint}")
                    return cached_response
            
            # Optimize response
            optimized_data = self._optimize_response_data(response_data)
            
            # Compress if needed
            if self.config.enable_response_compression:
                optimized_data = self._compress_response(optimized_data)
            
            # Create response
            response = {
                'data': optimized_data,
                'metadata': {
                    'cached': False,
                    'compressed': self.config.enable_response_compression,
                    'processing_time': time.time() - start_time
                }
            }
            
            # Cache response
            if self.config.enable_response_caching:
                self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error optimizing response for {endpoint}: {e}")
            return {
                'data': response_data,
                'metadata': {
                    'cached': False,
                    'compressed': False,
                    'processing_time': time.time() - start_time,
                    'error': str(e)
                }
            }
    
    def _generate_cache_key(self, endpoint: str, response_data: Any, 
                           user_id: Optional[str] = None) -> str:
        """Generate cache key for response."""
        data_hash = hash(str(response_data))
        user_part = f":{user_id}" if user_id else ""
        return f"api_response:{endpoint}{user_part}:{data_hash}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        try:
            with self._cache_lock:
                if cache_key in self._response_cache:
                    cached_data = self._response_cache[cache_key]
                    if time.time() - cached_data['timestamp'] < self.config.cache_ttl:
                        cached_data['metadata']['cached'] = True
                        return cached_data
                    else:
                        # Remove expired cache
                        del self._response_cache[cache_key]
        except Exception as e:
            logger.error(f"Error getting cached response: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response."""
        try:
            with self._cache_lock:
                response['timestamp'] = time.time()
                self._response_cache[cache_key] = response
                
                # Limit cache size
                if len(self._response_cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = sorted(
                        self._response_cache.keys(),
                        key=lambda k: self._response_cache[k]['timestamp']
                    )[:100]
                    for key in oldest_keys:
                        del self._response_cache[key]
                        
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    def _optimize_response_data(self, data: Any) -> Any:
        """Optimize response data structure."""
        try:
            if isinstance(data, dict):
                return self._optimize_dict(data)
            elif isinstance(data, list):
                return self._optimize_list(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Error optimizing response data: {e}")
            return data
    
    def _optimize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize dictionary data."""
        optimized = {}
        
        for key, value in data.items():
            # Skip None values
            if value is None:
                continue
            
            # Optimize nested structures
            if isinstance(value, dict):
                optimized[key] = self._optimize_dict(value)
            elif isinstance(value, list):
                optimized[key] = self._optimize_list(value)
            else:
                optimized[key] = value
        
        return optimized
    
    def _optimize_list(self, data: List[Any]) -> List[Any]:
        """Optimize list data."""
        optimized = []
        
        for item in data:
            if isinstance(item, dict):
                optimized.append(self._optimize_dict(item))
            elif isinstance(item, list):
                optimized.append(self._optimize_list(item))
            elif item is not None:
                optimized.append(item)
        
        return optimized
    
    def _compress_response(self, data: Any) -> Any:
        """Compress response data if beneficial."""
        try:
            # Serialize to JSON to check size
            json_data = json.dumps(data)
            
            if len(json_data) > self.config.max_response_size_for_compression:
                # Compress the data
                compressed_data = gzip.compress(json_data.encode('utf-8'))
                
                # Only use compression if it's beneficial
                if len(compressed_data) < len(json_data) * 0.8:
                    return {
                        'compressed': True,
                        'data': compressed_data.hex(),  # Convert to hex for JSON
                        'original_size': len(json_data),
                        'compressed_size': len(compressed_data)
                    }
            
            return data
            
        except Exception as e:
            logger.error(f"Error compressing response: {e}")
            return data
    
    def decompress_response(self, data: Any) -> Any:
        """Decompress response data if needed."""
        try:
            if isinstance(data, dict) and data.get('compressed'):
                # Decompress the data
                compressed_bytes = bytes.fromhex(data['data'])
                decompressed_data = gzip.decompress(compressed_bytes)
                return json.loads(decompressed_data.decode('utf-8'))
            
            return data
            
        except Exception as e:
            logger.error(f"Error decompressing response: {e}")
            return data


# Global API optimizer instance
api_optimizer = APIPerformanceOptimizer()


def optimize_api_response(endpoint: str, response_data: Any, 
                        user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Optimize API response for performance.
    
    Args:
        endpoint: API endpoint
        response_data: Response data
        user_id: User ID for caching
        
    Returns:
        Optimized response
    """
    return api_optimizer.optimize_response(endpoint, response_data, user_id)


def fast_api_decorator(endpoint: str):
    """
    Decorator for FastAPI endpoints to add performance optimization.
    
    Args:
        endpoint: API endpoint name
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Record metrics
                metrics_collector.record_api_request(
                    method='GET',  # Default, should be extracted from request
                    endpoint=endpoint,
                    status_code=200,
                    duration=response_time,
                    response_size=len(str(result))
                )
                
                # Optimize response
                optimized_result = optimize_api_response(endpoint, result)
                
                return optimized_result
                
            except Exception as e:
                response_time = time.time() - start_time
                
                # Record error metrics
                metrics_collector.record_api_request(
                    method='GET',
                    endpoint=endpoint,
                    status_code=500,
                    duration=response_time,
                    response_size=0
                )
                
                metrics_collector.record_error('api', type(e).__name__)
                raise
                
        return wrapper
    return decorator


class AsyncTaskManager:
    """Manager for async API tasks."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=50)
        self.active_tasks = {}
        self._lock = threading.Lock()
    
    async def execute_async_task(self, task_id: str, task_func, *args, **kwargs):
        """
        Execute task asynchronously.
        
        Args:
            task_id: Unique task ID
            task_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task result
        """
        try:
            # Run task in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                task_func, 
                *args, 
                **kwargs
            )
            
            # Clean up task
            with self._lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            return result
            
        except Exception as e:
            # Clean up task on error
            with self._lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            logger.error(f"Async task {task_id} failed: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        with self._lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            else:
                return {'status': 'not_found'}


# Global async task manager
async_task_manager = AsyncTaskManager()


def create_async_response(task_id: str, task_func, *args, **kwargs) -> Dict[str, Any]:
    """
    Create async response for long-running tasks.
    
    Args:
        task_id: Unique task ID
        task_func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Async response with task ID
    """
    try:
        # Start async task
        asyncio.create_task(
            async_task_manager.execute_async_task(task_id, task_func, *args, **kwargs)
        )
        
        return {
            'task_id': task_id,
            'status': 'started',
            'message': 'Task started successfully'
        }
        
    except Exception as e:
        logger.error(f"Failed to start async task {task_id}: {e}")
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(e)
        }


# Database query optimization
class DatabaseQueryOptimizer:
    """Optimizer for database queries."""
    
    def __init__(self):
        self.query_cache = {}
        self._lock = threading.Lock()
    
    def optimize_query(self, query_func, cache_key: str, ttl: int = 300):
        """
        Optimize database query with caching.
        
        Args:
            query_func: Function that executes the query
            cache_key: Cache key for the query
            ttl: Time to live in seconds
            
        Returns:
            Query result
        """
        try:
            # Check cache
            with self._lock:
                if cache_key in self.query_cache:
                    cached_data = self.query_cache[cache_key]
                    if time.time() - cached_data['timestamp'] < ttl:
                        logger.debug(f"Cache hit for query {cache_key}")
                        return cached_data['result']
                    else:
                        # Remove expired cache
                        del self.query_cache[cache_key]
            
            # Execute query
            start_time = time.time()
            result = query_func()
            query_time = time.time() - start_time
            
            # Cache result
            with self._lock:
                self.query_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
            
            # Record metrics
            metrics_collector.record_db_operation(
                operation='query',
                duration=query_time,
                status='success'
            )
            
            logger.debug(f"Query {cache_key} executed in {query_time:.3f}s")
            return result
            
        except Exception as e:
            query_time = time.time() - start_time
            
            # Record error metrics
            metrics_collector.record_db_operation(
                operation='query',
                duration=query_time,
                status='error'
            )
            
            logger.error(f"Query {cache_key} failed: {e}")
            raise


# Global database optimizer
db_optimizer = DatabaseQueryOptimizer()


def optimize_db_query(query_func, cache_key: str, ttl: int = 300):
    """
    Optimize database query with caching.
    
    Args:
        query_func: Function that executes the query
        cache_key: Cache key for the query
        ttl: Time to live in seconds
        
    Returns:
        Query result
    """
    return db_optimizer.optimize_query(query_func, cache_key, ttl)
