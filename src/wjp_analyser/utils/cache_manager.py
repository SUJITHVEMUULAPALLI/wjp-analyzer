"""
Cache Manager for WJP Analyser
==============================

This module provides intelligent caching for analysis results, file processing,
and other expensive operations to improve performance.
"""

import os
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Dict, Union, Callable
from dataclasses import dataclass
from functools import wraps
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_access = time.time()


class FileBasedCache:
    """File-based cache for persistent storage."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _cleanup_old_files(self):
        """Remove old cache files to stay under size limit."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            if not cache_files:
                return
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Remove oldest files until under limit
            for cache_file in cache_files:
                if total_size <= self.max_size_bytes:
                    break
                
                total_size -= cache_file.stat().st_size
                cache_file.unlink()
                logger.debug(f"Removed old cache file: {cache_file}")
                
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    entry: CacheEntry = pickle.load(f)
                
                if entry.is_expired():
                    cache_path.unlink()
                    return None
                
                entry.touch()
                
                # Update file with new access info
                with open(cache_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                return entry.data
                
            except Exception as e:
                logger.warning(f"Error reading cache for key {key}: {e}")
                # Remove corrupted cache file
                try:
                    cache_path.unlink()
                except:
                    pass
                return None
    
    def set(self, key: str, value: Any, ttl: float = 3600.0):
        """Set value in cache."""
        with self._lock:
            try:
                entry = CacheEntry(
                    data=value,
                    timestamp=time.time(),
                    ttl=ttl
                )
                
                cache_path = self._get_cache_path(key)
                
                with open(cache_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                # Periodic cleanup
                if len(list(self.cache_dir.glob("*.cache"))) % 10 == 0:
                    self._cleanup_old_files()
                
            except Exception as e:
                logger.warning(f"Error writing cache for key {key}: {e}")
    
    def delete(self, key: str):
        """Delete cache entry."""
        with self._lock:
            cache_path = self._get_cache_path(key)
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Error deleting cache for key {key}: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                logger.info("Cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "file_count": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(self.cache_dir)
            }


class MemoryCache:
    """In-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.touch()
            return entry.data
    
    def set(self, key: str, value: Any, ttl: float = 3600.0):
        """Set value in cache."""
        with self._lock:
            # Remove oldest entries if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            self._cache[key] = entry
    
    def _evict_oldest(self):
        """Remove oldest cache entry."""
        if not self._cache:
            return
        
        # Find the entry with the oldest timestamp (not last_access)
        oldest_key = min(self._cache.keys(), 
                        key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
    
    def delete(self, key: str):
        """Delete cache entry."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_access = sum(entry.access_count for entry in self._cache.values())
            avg_access = total_access / len(self._cache) if self._cache else 0
            
            return {
                "entry_count": len(self._cache),
                "max_size": self.max_size,
                "total_access_count": total_access,
                "average_access_count": avg_access
            }


class CacheManager:
    """Unified cache manager with multiple cache backends."""
    
    def __init__(self, cache_dir: str = "cache", memory_size: int = 1000, file_size_mb: int = 100):
        self.memory_cache = MemoryCache(memory_size)
        self.file_cache = FileBasedCache(cache_dir, file_size_mb)
        self._lock = threading.RLock()
    
    def get(self, key: str, use_file_cache: bool = True) -> Optional[Any]:
        """Get value from cache (memory first, then file)."""
        with self._lock:
            # Try memory cache first
            value = self.memory_cache.get(key)
            if value is not None:
                return value
            
            # Try file cache if enabled
            if use_file_cache:
                value = self.file_cache.get(key)
                if value is not None:
                    # Store in memory cache for faster access
                    self.memory_cache.set(key, value)
                    return value
            
            return None
    
    def set(self, key: str, value: Any, ttl: float = 3600.0, use_file_cache: bool = True):
        """Set value in cache."""
        with self._lock:
            # Always store in memory cache
            self.memory_cache.set(key, value, ttl)
            
            # Store in file cache if enabled and value is serializable
            if use_file_cache:
                try:
                    self.file_cache.set(key, value, ttl)
                except Exception as e:
                    logger.warning(f"Failed to store in file cache: {e}")
    
    def delete(self, key: str):
        """Delete cache entry from all backends."""
        with self._lock:
            self.memory_cache.delete(key)
            self.file_cache.delete(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.memory_cache.clear()
            self.file_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            memory_stats = self.memory_cache.get_stats()
            file_stats = self.file_cache.get_stats()
            
            return {
                "memory_cache": memory_stats,
                "file_cache": file_stats
            }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def initialize_cache(cache_dir: str = "cache", memory_size: int = 1000, file_size_mb: int = 100) -> CacheManager:
    """Initialize cache system."""
    global _cache_manager
    _cache_manager = CacheManager(cache_dir, memory_size, file_size_mb)
    return _cache_manager


def get_cache_manager() -> Optional[CacheManager]:
    """Get cache manager instance."""
    return _cache_manager


def cached(ttl: float = 3600.0, use_file_cache: bool = True, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _cache_manager is None:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = _cache_manager.get(cache_key, use_file_cache)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            _cache_manager.set(cache_key, result, ttl, use_file_cache)
            
            return result
        
        return wrapper
    return decorator


def cache_dxf_analysis(func):
    """Specialized cache decorator for DXF analysis."""
    def key_func(file_path: str, *args, **kwargs):
        # Use file path and modification time for cache key
        try:
            mtime = os.path.getmtime(file_path)
            key_data = f"dxf_analysis:{file_path}:{mtime}:{str(args)}:{str(sorted(kwargs.items()))}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except OSError:
            # Fallback if file doesn't exist
            key_data = f"dxf_analysis:{file_path}:{str(args)}:{str(sorted(kwargs.items()))}"
            return hashlib.md5(key_data.encode()).hexdigest()
    
    return cached(ttl=7200.0, use_file_cache=True, key_func=key_func)(func)


def cache_image_processing(func):
    """Specialized cache decorator for image processing."""
    def key_func(image_path: str, *args, **kwargs):
        # Use image path and modification time for cache key
        try:
            mtime = os.path.getmtime(image_path)
            key_data = f"image_processing:{image_path}:{mtime}:{str(args)}:{str(sorted(kwargs.items()))}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except OSError:
            # Fallback if file doesn't exist
            key_data = f"image_processing:{image_path}:{str(args)}:{str(sorted(kwargs.items()))}"
            return hashlib.md5(key_data.encode()).hexdigest()
    
    return cached(ttl=3600.0, use_file_cache=True, key_func=key_func)(func)


def cache_ai_response(func):
    """Specialized cache decorator for AI responses."""
    def key_func(prompt: str, *args, **kwargs):
        # Use prompt hash for cache key
        key_data = f"ai_response:{hashlib.md5(prompt.encode()).hexdigest()}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    return cached(ttl=86400.0, use_file_cache=False, key_func=key_func)(func)


# Convenience functions
def get_cached(key: str) -> Optional[Any]:
    """Get value from cache."""
    if _cache_manager:
        return _cache_manager.get(key)
    return None


def set_cached(key: str, value: Any, ttl: float = 3600.0):
    """Set value in cache."""
    if _cache_manager:
        _cache_manager.set(key, value, ttl)


def delete_cached(key: str):
    """Delete value from cache."""
    if _cache_manager:
        _cache_manager.delete(key)


def clear_cache():
    """Clear all cache entries."""
    if _cache_manager:
        _cache_manager.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    if _cache_manager:
        return _cache_manager.get_stats()
    return {}
