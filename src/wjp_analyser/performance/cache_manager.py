"""
Enhanced Cache Manager
======================

Function-level memoization and artifact caching for performance optimization.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union
from functools import wraps
import time


class CacheManager:
    """Enhanced cache manager with job hashing and artifact caching."""
    
    def __init__(self, cache_dir: str = ".cache", ttl_seconds: int = 3600 * 24):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Cache directory path
            ttl_seconds: Time-to-live for cache entries (seconds)
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic key from args and kwargs
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {},
        }
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_json.encode()).hexdigest()
    
    def _get_cache_path(self, key: str, suffix: str = ".cache") -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}{suffix}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check TTL
        if self.ttl_seconds > 0:
            age = time.time() - cache_path.stat().st_mtime
            if age > self.ttl_seconds:
                cache_path.unlink()
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, key: str, value: Any):
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass  # Silently fail on cache write errors
    
    def delete(self, key: str):
        """Delete cached value."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self):
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
    
    def get_job_hash(
        self,
        file_path: str,
        parameters: Dict[str, Any],
    ) -> str:
        """
        Generate job hash for idempotent job processing.
        
        Args:
            file_path: Path to input file
            parameters: Job parameters
            
        Returns:
            Job hash string
        """
        from .streaming_parser import compute_file_hash
        
        # Compute file hash
        file_hash = compute_file_hash(file_path)
        
        # Combine with parameters
        param_json = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_json.encode()).hexdigest()
        
        # Combined hash
        combined = f"{file_hash}_{param_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def cache_artifact(
        self,
        job_hash: str,
        artifact_name: str,
        artifact_path: str,
    ):
        """
        Cache artifact path for a job.
        
        Args:
            job_hash: Job hash
            artifact_name: Name of artifact (e.g., 'layered_dxf', 'gcode')
            artifact_path: Path to artifact file
        """
        metadata_path = self._get_cache_path(job_hash, suffix=".meta")
        
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        metadata['artifacts'] = metadata.get('artifacts', {})
        metadata['artifacts'][artifact_name] = artifact_path
        metadata['updated'] = time.time()
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        except Exception:
            pass
    
    def get_cached_artifacts(self, job_hash: str) -> Dict[str, str]:
        """
        Get cached artifacts for a job.
        
        Args:
            job_hash: Job hash
            
        Returns:
            Dictionary of artifact names to paths
        """
        metadata_path = self._get_cache_path(job_hash, suffix=".meta")
        
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get('artifacts', {})
        except Exception:
            return {}


def memoize(cache_manager: CacheManager, ttl_seconds: int = None):
    """
    Decorator for function-level memoization.
    
    Args:
        cache_manager: CacheManager instance
        ttl_seconds: Optional TTL override
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._get_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached = cache_manager.get(cache_key)
            if cached is not None:
                return cached
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


_CACHE_INSTANCES: Dict[Path, CacheManager] = {}


def _resolve_cache_dir(cache_dir: Union[str, Path]) -> Path:
    """Return canonical absolute cache directory path."""
    return Path(cache_dir).resolve()


def get_cache_manager(cache_dir: str = ".cache") -> CacheManager:
    """Get or create a cache manager scoped to the provided directory."""
    resolved_dir = _resolve_cache_dir(cache_dir)
    manager = _CACHE_INSTANCES.get(resolved_dir)
    if manager is None:
        manager = CacheManager(str(resolved_dir))
        _CACHE_INSTANCES[resolved_dir] = manager
    return manager


def clear_cache(cache_dir: str = ".cache"):
    """Clear cache contents for the provided directory."""
    resolved_dir = _resolve_cache_dir(cache_dir)
    manager = _CACHE_INSTANCES.get(resolved_dir)
    if manager is None:
        manager = CacheManager(str(resolved_dir))
        _CACHE_INSTANCES[resolved_dir] = manager
    manager.clear()








