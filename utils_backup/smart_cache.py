"""
Advanced caching system for improved performance.
"""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, Optional


class SmartCache:
    """
    Intelligent caching system with TTL and smart eviction.
    """

    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        """
        Initialize smart cache.

        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}

    def _create_key(self, text: str) -> str:
        """Create consistent cache key from text."""
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.cache:
            return True

        entry = self.cache[key]
        return time.time() > entry["expires_at"]

    def _evict_oldest(self):
        """Evict least recently used items."""
        if len(self.cache) <= self.max_size:
            return

        # Sort by access time and remove oldest
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_keys[: len(self.cache) - self.max_size + 1]]

        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for text.

        Args:
            text: Input text to lookup

        Returns:
            Cached response or None if not found/expired
        """
        key = self._create_key(text)

        if key not in self.cache or self._is_expired(key):
            return None

        # Update access time
        self.access_times[key] = time.time()

        return self.cache[key]["data"]

    def set(self, text: str, response: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Cache response for text.

        Args:
            text: Input text
            response: Response to cache
            ttl: Time-to-live override
        """
        key = self._create_key(text)
        expires_at = time.time() + (ttl or self.default_ttl)

        self.cache[key] = {"data": response, "expires_at": expires_at, "created_at": time.time()}
        self.access_times[key] = time.time()

        # Evict old entries if needed
        self._evict_oldest()

    def clear_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key in self.cache.keys() if self._is_expired(key)]

        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "expired_entries": sum(1 for key in self.cache.keys() if self._is_expired(key)),
            "memory_usage_mb": len(pickle.dumps(self.cache)) / (1024 * 1024),
            "oldest_entry_age": (
                time.time() - min(self.access_times.values()) if self.access_times else 0
            ),
        }


class EmbeddingCache:
    """
    Specialized cache for embeddings to avoid recomputation.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize embedding cache."""
        self.max_size = max_size
        self.embeddings: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}

    def get_embedding(self, text: str) -> Optional[Any]:
        """Get cached embedding for text."""
        key = hashlib.md5(text.encode("utf-8")).hexdigest()

        if key in self.embeddings:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.embeddings[key]

        return None

    def cache_embedding(self, text: str, embedding: Any) -> None:
        """Cache embedding for text."""
        key = hashlib.md5(text.encode("utf-8")).hexdigest()

        # Evict least used if at capacity
        if len(self.embeddings) >= self.max_size:
            least_used = min(self.access_count.items(), key=lambda x: x[1])
            least_used_key = least_used[0]
            self.embeddings.pop(least_used_key, None)
            self.access_count.pop(least_used_key, None)

        self.embeddings[key] = embedding
        self.access_count[key] = 1

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "cached_embeddings": len(self.embeddings),
            "max_size": self.max_size,
            "total_accesses": sum(self.access_count.values()),
            "average_accesses": (
                sum(self.access_count.values()) / len(self.access_count) if self.access_count else 0
            ),
        }
