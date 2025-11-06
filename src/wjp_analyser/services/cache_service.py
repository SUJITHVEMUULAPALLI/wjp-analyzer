import os
from pathlib import Path
import json
import time
from typing import Any, Optional, Dict

class CacheService:
    """Service for handling caching operations with file-based and in-memory caching."""
    
    def __init__(self, cache_dir: str = "cache", max_age: int = 3600):
        """
        Initialize the cache service.
        
        Args:
            cache_dir: Directory to store cache files
            max_age: Maximum age of cache entries in seconds (default 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.max_age = max_age
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.cache"
        
    def get(self, key: str, use_memory: bool = True) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            use_memory: Whether to check memory cache first
            
        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first if enabled
        if use_memory and key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry["timestamp"] <= self.max_age:
                return entry["value"]
            del self.memory_cache[key]
            
        # Check file cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    entry = json.load(f)
                if time.time() - entry["timestamp"] <= self.max_age:
                    # Update memory cache if enabled
                    if use_memory:
                        self.memory_cache[key] = entry
                    return entry["value"]
                # Clean up expired cache file
                os.remove(cache_path)
            except (json.JSONDecodeError, KeyError, OSError):
                # Handle corrupted cache files
                if cache_path.exists():
                    os.remove(cache_path)
        return None
        
    def set(self, key: str, value: Any, use_memory: bool = True) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            use_memory: Whether to also store in memory cache
        """
        entry = {
            "timestamp": time.time(),
            "value": value
        }
        
        # Update memory cache if enabled
        if use_memory:
            self.memory_cache[key] = entry
            
        # Update file cache
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w") as f:
                json.dump(entry, f)
        except OSError:
            # Handle file write errors gracefully
            pass
            
    def invalidate(self, key: str) -> None:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            
        # Remove cache file
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                os.remove(cache_path)
            except OSError:
                pass
                
    def clear_all(self) -> None:
        """Clear all cache entries (both memory and file)."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear file cache
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                os.remove(cache_file)
        except OSError:
            pass