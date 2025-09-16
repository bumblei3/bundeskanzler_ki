"""
Tests für advanced_cache.py - Erweitertes Caching-System
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
from advanced_cache import (
    CacheBackend,
    CacheEntry,
    CacheManager,
    CacheStats,
    FileSystemCache,
    MemoryCache,
    MultiLevelCache,
    RedisCache,
)


class TestCacheStats:
    """Tests für CacheStats Klasse"""

    def test_init(self):
        """Test CacheStats Initialisierung"""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.errors == 0
        assert hasattr(stats, "lock")

    def test_hit(self):
        """Test hit Methode"""
        stats = CacheStats()
        stats.hit()
        assert stats.hits == 1

    def test_miss(self):
        """Test miss Methode"""
        stats = CacheStats()
        stats.miss()
        assert stats.misses == 1

    def test_set(self):
        """Test set Methode"""
        stats = CacheStats()
        stats.set()
        assert stats.sets == 1

    def test_delete(self):
        """Test delete Methode"""
        stats = CacheStats()
        stats.delete()
        assert stats.deletes == 1

    def test_error(self):
        """Test error Methode"""
        stats = CacheStats()
        stats.error()
        assert stats.errors == 1

    def test_thread_safety(self):
        """Test Thread-Sicherheit"""
        stats = CacheStats()
        import concurrent.futures
        import threading

        def increment_hits():
            for _ in range(100):
                stats.hit()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(increment_hits) for _ in range(4)]
            for future in futures:
                future.result()

        assert stats.hits == 400


class TestCacheEntry:
    """Tests für CacheEntry Klasse"""

    def test_init(self):
        """Test CacheEntry Initialisierung"""
        entry = CacheEntry("test_value", ttl=60)
        assert entry.value == "test_value"
        assert entry.ttl == 60
        assert entry.created_at is not None
        assert not entry.is_expired()

    def test_init_without_ttl(self):
        """Test CacheEntry ohne TTL"""
        entry = CacheEntry("test_value")
        assert entry.value == "test_value"
        assert entry.ttl is None
        assert not entry.is_expired()

    def test_is_expired_with_ttl(self):
        """Test is_expired mit TTL"""
        entry = CacheEntry("test_value", ttl=1)
        assert not entry.is_expired()

        # Simulate expiration
        entry.created_at = time.time() - 2
        assert entry.is_expired()

    def test_is_expired_without_ttl(self):
        """Test is_expired ohne TTL"""
        entry = CacheEntry("test_value")
        # Should never expire
        entry.created_at = time.time() - 365 * 24 * 60 * 60  # 1 year ago
        assert not entry.is_expired()

    def test_access(self):
        """Test access Methode"""
        entry = CacheEntry("test_value")
        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed

        time.sleep(0.01)  # Small delay
        entry.access()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed > initial_last_accessed


class TestMemoryCache:
    """Tests für MemoryCache Klasse"""

    def test_init(self):
        """Test MemoryCache Initialisierung"""
        cache = MemoryCache(max_size=100)
        assert cache.max_size == 100
        assert isinstance(cache.stats, CacheStats)
        assert cache.cache == {}

    def test_set_and_get(self):
        """Test set und get Methoden"""
        cache = MemoryCache()

        # Set a value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.stats.sets == 1

        # Get existing value (should be hit)
        result = cache.get("key1")
        assert result == "value1"
        assert cache.stats.hits == 2  # First get() and second get() both hit

        # Get non-existing value
        assert cache.get("nonexistent") is None
        assert cache.stats.misses == 1

    def test_set_with_ttl(self):
        """Test set mit TTL"""
        cache = MemoryCache()

        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_delete(self):
        """Test delete Methode"""
        cache = MemoryCache()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.delete("key1")
        assert cache.get("key1") is None
        assert cache.stats.deletes == 1

    def test_clear(self):
        """Test clear Methode"""
        cache = MemoryCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache.cache) == 2

        cache.clear()
        assert len(cache.cache) == 0

    def test_has_key(self):
        """Test has_key Methode"""
        cache = MemoryCache()

        assert not cache.has_key("key1")
        cache.set("key1", "value1")
        assert cache.has_key("key1")

    def test_size_limit(self):
        """Test Größenlimit"""
        cache = MemoryCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should trigger cleanup

        # Should have only 2 items
        assert len(cache.cache) <= 2


class TestFileSystemCache:
    """Tests für FileSystemCache Klasse"""

    def test_init(self):
        """Test FileSystemCache Initialisierung"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileSystemCache(cache_dir=temp_dir)
            assert str(cache.cache_dir) == temp_dir
            assert isinstance(cache.stats, CacheStats)

    def test_set_and_get(self):
        """Test set und get Methoden"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileSystemCache(cache_dir=temp_dir)

            # Set a value
            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"

            # Get non-existing value
            assert cache.get("nonexistent") is None

    def test_set_with_ttl(self):
        """Test set mit TTL"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileSystemCache(cache_dir=temp_dir)

            cache.set("key1", "value1", ttl=1)
            assert cache.get("key1") == "value1"

            # Wait for expiration
            time.sleep(1.1)
            assert cache.get("key1") is None

    def test_delete(self):
        """Test delete Methode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileSystemCache(cache_dir=temp_dir)

            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"

            cache.delete("key1")
            assert cache.get("key1") is None

    def test_clear(self):
        """Test clear Methode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileSystemCache(cache_dir=temp_dir)

            cache.set("key1", "value1")
            cache.set("key2", "value2")

            cache.clear()
            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_has_key(self):
        """Test has_key Methode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileSystemCache(cache_dir=temp_dir)

            assert not cache.has_key("key1")
            cache.set("key1", "value1")
            assert cache.has_key("key1")

    def test_complex_data_types(self):
        """Test komplexe Datentypen"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileSystemCache(cache_dir=temp_dir)

            test_data = {"list": [1, 2, 3], "dict": {"nested": "value"}, "number": 42}

            cache.set("complex_key", test_data)
            result = cache.get("complex_key")

            assert result == test_data


class TestMultiLevelCache:
    """Tests für MultiLevelCache Klasse"""

    def test_init(self):
        """Test MultiLevelCache Initialisierung"""
        l1_cache = MemoryCache()
        l2_cache = MemoryCache()

        cache = MultiLevelCache(l1_cache, l2_cache)
        assert cache.l1_cache == l1_cache
        assert cache.l2_cache == l2_cache

    def test_set_and_get(self):
        """Test set und get Methoden"""
        l1_cache = MemoryCache()
        l2_cache = MemoryCache()

        cache = MultiLevelCache(l1_cache, l2_cache)

        # Set value
        cache.set("key1", "value1")

        # Should be in both caches
        assert l1_cache.get("key1") == "value1"
        assert l2_cache.get("key1") == "value1"

        # Get should check L1 first
        assert cache.get("key1") == "value1"

    def test_get_fallback(self):
        """Test Fallback-Mechanismus"""
        l1_cache = MemoryCache()
        l2_cache = MemoryCache()

        cache = MultiLevelCache(l1_cache, l2_cache)

        # Set only in L2
        l2_cache.set("key1", "value1")

        # Get should find it in L2 and promote to L1
        assert cache.get("key1") == "value1"
        assert l1_cache.get("key1") == "value1"

    def test_delete(self):
        """Test delete Methode"""
        l1_cache = MemoryCache()
        l2_cache = MemoryCache()

        cache = MultiLevelCache(l1_cache, l2_cache)

        cache.set("key1", "value1")
        cache.delete("key1")

        assert l1_cache.get("key1") is None
        assert l2_cache.get("key1") is None

    def test_clear(self):
        """Test clear Methode"""
        l1_cache = MemoryCache()
        l2_cache = MemoryCache()

        cache = MultiLevelCache(l1_cache, l2_cache)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert l1_cache.get("key1") is None
        assert l2_cache.get("key1") is None


class TestCacheManager:
    """Tests für CacheManager Klasse"""

    def test_init(self):
        """Test CacheManager Initialisierung"""
        manager = CacheManager()
        assert hasattr(manager, "caches")
        assert isinstance(manager.caches, dict)

    def test_create_cache(self):
        """Test create_cache Methode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager()

            cache = manager.create_cache("test_cache", l2_config={"cache_dir": temp_dir})
            assert isinstance(cache, MultiLevelCache)
            assert "test_cache" in manager.caches

    def test_get_cache(self):
        """Test get_cache Methode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager()

            # Create cache first
            manager.create_cache("test_cache", l2_config={"cache_dir": temp_dir})

            # Get cache
            cache = manager.get_cache("test_cache")
            assert isinstance(cache, MultiLevelCache)

            # Get non-existing cache
            assert manager.get_cache("nonexistent") is None

    @patch("advanced_cache.FileSystemCache")
    def test_create_cache_filesystem(self, mock_fs_cache):
        """Test create_cache mit filesystem"""
        mock_instance = MagicMock()
        mock_fs_cache.return_value = mock_instance

        manager = CacheManager()
        cache = manager.create_cache("test", l2_type="filesystem")

        mock_fs_cache.assert_called_once()
        assert isinstance(cache, MultiLevelCache)

    @patch("advanced_cache.RedisCache")
    def test_create_cache_redis(self, mock_redis_cache):
        """Test create_cache mit redis"""
        mock_instance = MagicMock()
        mock_redis_cache.return_value = mock_instance

        manager = CacheManager()
        cache = manager.create_cache("test", l2_type="redis")

        mock_redis_cache.assert_called_once()
        assert isinstance(cache, MultiLevelCache)

    def test_get_all_stats(self):
        """Test get_all_stats Methode"""
        manager = CacheManager()
        stats = manager.get_all_stats()

        assert isinstance(stats, dict)
