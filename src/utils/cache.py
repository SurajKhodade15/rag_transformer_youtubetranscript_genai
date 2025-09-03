"""
Caching utilities for the RAG application.
Provides memory and Redis-based caching with TTL support.
"""

import json
import hashlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import threading

from config.settings import settings
from src.utils.logging_config import get_logger
from src.core.exceptions import CacheError

logger = get_logger("cache")


class CacheInterface(ABC):
    """Abstract base class for cache implementations"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass


class MemoryCache(CacheInterface):
    """In-memory cache implementation with TTL support"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        if entry.get("expires_at") is None:
            return False
        return datetime.utcnow() > entry["expires_at"]
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            with self._lock:
                if key not in self._cache:
                    return None
                
                entry = self._cache[key]
                if self._is_expired(entry):
                    del self._cache[key]
                    return None
                
                return entry["value"]
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            with self._lock:
                entry = {
                    "value": value,
                    "created_at": datetime.utcnow(),
                    "expires_at": None
                }
                
                if ttl:
                    entry["expires_at"] = datetime.utcnow() + timedelta(seconds=ttl)
                
                self._cache[key] = entry
                
                # Cleanup expired entries periodically
                if len(self._cache) % 100 == 0:
                    self._cleanup_expired()
                
                return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            with self._lock:
                self._cache.clear()
                return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            with self._lock:
                if key not in self._cache:
                    return False
                
                entry = self._cache[key]
                if self._is_expired(entry):
                    del self._cache[key]
                    return False
                
                return True
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values()
                if self._is_expired(entry)
            )
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries
            }


class RedisCache(CacheInterface):
    """Redis cache implementation"""
    
    def __init__(self):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=settings.database.redis_host,
                port=settings.database.redis_port,
                db=settings.database.redis_db,
                password=settings.database.redis_password,
                decode_responses=False  # We'll handle encoding/decoding
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except ImportError:
            raise CacheError("Redis library not installed. Install with: pip install redis")
        except Exception as e:
            raise CacheError(f"Failed to connect to Redis: {e}")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            data = self._serialize(value)
            if ttl:
                return self.redis_client.setex(key, ttl, data)
            else:
                return self.redis_client.set(key, data)
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            return self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False


class CacheManager:
    """Main cache manager that handles cache operations"""
    
    def __init__(self):
        self.enabled = settings.app.cache_enabled
        self.default_ttl = settings.app.cache_ttl
        
        if not self.enabled:
            self.cache = None
            logger.info("Cache disabled")
            return
        
        # Initialize cache based on configuration
        if settings.app.cache_type.lower() == "redis":
            try:
                self.cache = RedisCache()
                logger.info("Using Redis cache")
            except CacheError as e:
                logger.warning(f"Redis cache initialization failed: {e}. Falling back to memory cache.")
                self.cache = MemoryCache()
        else:
            self.cache = MemoryCache()
            logger.info("Using memory cache")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and arguments"""
        # Create a hash of all arguments for consistent keys
        content = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.cache:
            return None
        
        result = self.cache.get(key)
        from src.utils.logging_config import performance_logger
        performance_logger.log_cache_hit(key, result is not None)
        return result
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.enabled or not self.cache:
            return False
        
        ttl = ttl or self.default_ttl
        return self.cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.enabled or not self.cache:
            return False
        
        return self.cache.delete(key)
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        if not self.enabled or not self.cache:
            return False
        
        return self.cache.clear()
    
    def cache_result(self, prefix: str, ttl: Optional[int] = None):
        """Decorator to cache function results"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Generate cache key
                cache_key = self._generate_key(prefix, func.__name__, *args, **kwargs)
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
cache_manager = CacheManager()
