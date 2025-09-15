"""
Erweitertes Caching-System für Bundeskanzler-KI
Unterstützt Memory, Redis, Filesystem-Cache mit TTL, Statistiken und Monitoring
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


class CacheStats:
    """Cache-Statistiken und Monitoring"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.lock = threading.Lock()

    def hit(self):
        with self.lock:
            self.hits += 1

    def miss(self):
        with self.lock:
            self.misses += 1

    def set(self):
        with self.lock:
            self.sets += 1

    def delete(self):
        with self.lock:
            self.deletes += 1

    def error(self):
        with self.lock:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "errors": self.errors,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "miss_rate": 100 - hit_rate,
            }

    def reset(self):
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.deletes = 0
            self.errors = 0


class CacheEntry:
    """Cache-Eintrag mit TTL und Metadaten"""

    def __init__(
        self, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict] = None
    ):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Prüft ob der Eintrag abgelaufen ist"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def access(self):
        """Markiert Zugriff auf den Eintrag"""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary für Serialisierung"""
        return {
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CacheEntry":
        """Erstellt CacheEntry aus Dictionary"""
        entry = cls(
            value=data["value"], ttl=data["ttl"], metadata=data.get("metadata", {})
        )
        entry.created_at = data["created_at"]
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = data.get("last_accessed", entry.created_at)
        return entry


class CacheBackend(ABC):
    """Abstrakte Basisklasse für Cache-Backends"""

    def __init__(self):
        self.stats = CacheStats()

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Holt Wert aus Cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Speichert Wert im Cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Löscht Wert aus Cache"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Leert gesamten Cache"""
        pass

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Prüft ob Schlüssel existiert"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Cache-Statistiken zurück"""
        return self.stats.get_stats()


class MemoryCache(CacheBackend):
    """In-Memory Cache Backend"""

    def __init__(self, max_size: int = 1000):
        super().__init__()
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    self.stats.miss()
                    return None
                entry.access()
                self.stats.hit()
                return entry.value
            self.stats.miss()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self.lock:
            try:
                # Eviction falls nötig
                if key not in self.cache and len(self.cache) >= self.max_size:
                    self._evict_lru()

                self.cache[key] = CacheEntry(value, ttl)
                self.stats.set()
                return True
            except Exception as e:
                logger.error(f"MemoryCache set error: {e}")
                self.stats.error()
                return False

    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.delete()
                return True
            return False

    def clear(self) -> bool:
        with self.lock:
            self.cache.clear()
            return True

    def has_key(self, key: str) -> bool:
        with self.lock:
            return key in self.cache and not self.cache[key].is_expired()

    def _evict_lru(self):
        """Entfernt den am längsten nicht verwendeten Eintrag"""
        if not self.cache:
            return

        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
        logger.debug(f"Evicted LRU key: {lru_key}")


class FileSystemCache(CacheBackend):
    """Filesystem-basierter Cache"""

    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 100):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.lock = threading.RLock()

    def _get_cache_path(self, key: str) -> Path:
        """Generiert Cache-Dateipfad für Schlüssel"""
        # Erstelle Subdirectory basierend auf ersten 2 Zeichen des Hashes
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = self.cache_dir / key_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            self.stats.miss()
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            entry = CacheEntry.from_dict(data)
            if entry.is_expired():
                cache_path.unlink(missing_ok=True)
                self.stats.miss()
                return None

            entry.access()
            # Speichere aktualisierte Metadaten
            self._save_entry(cache_path, entry)
            self.stats.hit()
            return entry.value

        except Exception as e:
            logger.error(f"FileSystemCache get error: {e}")
            self.stats.error()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        cache_path = self._get_cache_path(key)

        try:
            entry = CacheEntry(value, ttl)
            self._save_entry(cache_path, entry)
            self.stats.set()

            # Prüfe Cache-Größe und räume auf falls nötig
            self._check_size_limit()
            return True

        except Exception as e:
            logger.error(f"FileSystemCache set error: {e}")
            self.stats.error()
            return False

    def delete(self, key: str) -> bool:
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.unlink(missing_ok=True):
                self.stats.delete()
                return True
            return False
        except Exception as e:
            logger.error(f"FileSystemCache delete error: {e}")
            return False

    def clear(self) -> bool:
        try:
            for cache_file in self.cache_dir.rglob("*.cache"):
                cache_file.unlink()
            self.stats.reset()
            return True
        except Exception as e:
            logger.error(f"FileSystemCache clear error: {e}")
            return False

    def has_key(self, key: str) -> bool:
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            entry = CacheEntry.from_dict(data)
            return not entry.is_expired()
        except:
            return False

    def _save_entry(self, path: Path, entry: CacheEntry):
        """Speichert CacheEntry in Datei"""
        with open(path, "wb") as f:
            pickle.dump(entry.to_dict(), f)

    def _check_size_limit(self):
        """Prüft Cache-Größe und räumt auf falls nötig"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*.cache"))
            if total_size > self.max_size_bytes:
                # Entferne älteste Dateien
                cache_files = []
                for f in self.cache_dir.rglob("*.cache"):
                    try:
                        with open(f, "rb") as file:
                            data = pickle.load(file)
                        entry = CacheEntry.from_dict(data)
                        cache_files.append((f, entry.last_accessed))
                    except:
                        f.unlink(missing_ok=True)

                # Sortiere nach letztem Zugriff (älteste zuerst)
                cache_files.sort(key=lambda x: x[1])

                # Entferne Dateien bis unter Limit
                for file_path, _ in cache_files:
                    if total_size <= self.max_size_bytes:
                        break
                    size = file_path.stat().st_size
                    file_path.unlink(missing_ok=True)
                    total_size -= size

        except Exception as e:
            logger.error(f"Size limit check error: {e}")


class RedisCache(CacheBackend):
    """Redis-basierter Cache (optional)"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        super().__init__()
        try:
            import redis

            self.redis = redis.Redis(host=host, port=port, db=db, password=password)
            self.redis.ping()  # Test connection
            self.available = True
        except ImportError:
            logger.warning("Redis nicht verfügbar, verwende Fallback")
            self.available = False
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.available = False

    def get(self, key: str) -> Optional[Any]:
        if not self.available:
            self.stats.miss()
            return None

        try:
            data = self.redis.get(key)
            if data is None:
                self.stats.miss()
                return None

            entry_dict = json.loads(data.decode("utf-8"))
            entry = CacheEntry.from_dict(entry_dict)

            if entry.is_expired():
                self.redis.delete(key)
                self.stats.miss()
                return None

            entry.access()
            # Aktualisiere in Redis
            self.redis.setex(key, entry.ttl or 3600, json.dumps(entry.to_dict()))
            self.stats.hit()
            return entry.value

        except Exception as e:
            logger.error(f"RedisCache get error: {e}")
            self.stats.error()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.available:
            return False

        try:
            entry = CacheEntry(value, ttl)
            data = json.dumps(entry.to_dict())
            if ttl:
                self.redis.setex(key, ttl, data)
            else:
                self.redis.set(key, data)
            self.stats.set()
            return True
        except Exception as e:
            logger.error(f"RedisCache set error: {e}")
            self.stats.error()
            return False

    def delete(self, key: str) -> bool:
        if not self.available:
            return False

        try:
            result = self.redis.delete(key)
            if result:
                self.stats.delete()
            return bool(result)
        except Exception as e:
            logger.error(f"RedisCache delete error: {e}")
            return False

    def clear(self) -> bool:
        if not self.available:
            return False

        try:
            self.redis.flushdb()
            self.stats.reset()
            return True
        except Exception as e:
            logger.error(f"RedisCache clear error: {e}")
            return False

    def has_key(self, key: str) -> bool:
        if not self.available:
            return False
        return bool(self.redis.exists(key))


class MultiLevelCache:
    """Multi-Level Cache mit L1 (Memory) und L2 (Filesystem/Redis)"""

    def __init__(
        self,
        l1_cache: Optional[CacheBackend] = None,
        l2_cache: Optional[CacheBackend] = None,
        enable_l2: bool = True,
    ):
        self.l1_cache = l1_cache or MemoryCache(max_size=500)
        self.l2_cache = l2_cache
        self.enable_l2 = enable_l2 and (l2_cache is not None)

        # Fallback zu Filesystem-Cache falls kein L2 angegeben
        if self.enable_l2 and self.l2_cache is None:
            self.l2_cache = FileSystemCache()

    def get(self, key: str) -> Optional[Any]:
        # Versuche L1 zuerst
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Fallback zu L2
        if self.enable_l2:
            value = self.l2_cache.get(key)
            if value is not None:
                # Schreibe zurück in L1 für schnellere zukünftige Zugriffe
                self.l1_cache.set(key, value, ttl=300)  # 5 Minuten in L1
                return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # Setze in L1
        l1_success = self.l1_cache.set(
            key, value, ttl=min(ttl or 3600, 300)
        )  # Max 5 Min in L1

        # Setze in L2 falls aktiviert
        l2_success = True
        if self.enable_l2:
            l2_success = self.l2_cache.set(key, value, ttl)

        return l1_success and l2_success

    def delete(self, key: str) -> bool:
        l1_deleted = self.l1_cache.delete(key)
        l2_deleted = self.l2_cache.delete(key) if self.enable_l2 else True
        return l1_deleted or l2_deleted

    def clear(self) -> bool:
        l1_cleared = self.l1_cache.clear()
        l2_cleared = self.l2_cache.clear() if self.enable_l2 else True
        return l1_cleared and l2_cleared

    def has_key(self, key: str) -> bool:
        return self.l1_cache.has_key(key) or (
            self.enable_l2 and self.l2_cache.has_key(key)
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = {"l1": self.l1_cache.get_stats()}
        if self.enable_l2:
            stats["l2"] = self.l2_cache.get_stats()
        return stats


class CacheManager:
    """Haupt-Cache-Manager für verschiedene Cache-Typen"""

    def __init__(self):
        self.caches: Dict[str, MultiLevelCache] = {}
        self.lock = threading.RLock()

    def create_cache(
        self,
        name: str,
        l1_size: int = 500,
        l2_type: str = "filesystem",
        l2_config: Optional[Dict] = None,
    ) -> MultiLevelCache:
        """Erstellt einen neuen Cache"""

        l1_cache = MemoryCache(max_size=l1_size)
        l2_cache = None

        if l2_type == "filesystem":
            config = l2_config or {}
            l2_cache = FileSystemCache(
                cache_dir=config.get("cache_dir", f"./cache/{name}"),
                max_size_mb=config.get("max_size_mb", 500),
            )
        elif l2_type == "redis":
            config = l2_config or {}
            l2_cache = RedisCache(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                db=config.get("db", 0),
                password=config.get("password"),
            )

        cache = MultiLevelCache(l1_cache=l1_cache, l2_cache=l2_cache)
        with self.lock:
            self.caches[name] = cache
        return cache

    def get_cache(self, name: str) -> Optional[MultiLevelCache]:
        """Holt Cache nach Namen"""
        with self.lock:
            return self.caches.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken aller Caches zurück"""
        with self.lock:
            return {name: cache.get_stats() for name, cache in self.caches.items()}


# Globale Cache-Manager Instanz
cache_manager = CacheManager()


def cached(
    cache_name: str, ttl: Optional[int] = None, key_func: Optional[Callable] = None
):
    """Decorator für automatische Cache-Funktionalität"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_name)
            if not cache:
                # Fallback: Funktion ohne Cache ausführen
                return func(*args, **kwargs)

            # Generiere Cache-Key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Standard-Key aus Funktionsname und Argumenten
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                key = hashlib.md5(key_data.encode()).hexdigest()

            # Versuche Cache-Hit
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Cache-Miss: Führe Funktion aus
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)

            # Speichere im Cache
            cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator


def initialize_caches():
    """Initialisiert Standard-Caches für die Bundeskanzler-KI"""

    # Model-Cache für vortrainierte Modelle
    cache_manager.create_cache(
        "models",
        l1_size=10,  # Wenige Modelle im Memory
        l2_type="filesystem",
        l2_config={"cache_dir": "./cache/models", "max_size_mb": 2000},
    )

    # Embedding-Cache für häufig verwendete Embeddings
    cache_manager.create_cache(
        "embeddings",
        l1_size=1000,
        l2_type="filesystem",
        l2_config={"cache_dir": "./cache/embeddings", "max_size_mb": 1000},
    )

    # API-Response-Cache für wiederholte Anfragen
    cache_manager.create_cache(
        "api_responses",
        l1_size=500,
        l2_type="filesystem",
        l2_config={"cache_dir": "./cache/api_responses", "max_size_mb": 500},
    )

    # Translation-Cache für Übersetzungen
    cache_manager.create_cache(
        "translations",
        l1_size=2000,
        l2_type="filesystem",
        l2_config={"cache_dir": "./cache/translations", "max_size_mb": 200},
    )

    logger.info("✅ Caching-System initialisiert")


# Cache-Statistiken für Monitoring
def get_cache_stats() -> Dict[str, Any]:
    """Gibt globale Cache-Statistiken zurück"""
    return cache_manager.get_all_stats()
