#!/usr/bin/env python3
"""
Redis-basiertes Advanced Caching-System für Bundeskanzler-KI
Erweitert das bestehende Cache-System mit Redis-Integration und Smart Invalidation
"""

import json
import logging
import pickle
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import redis
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class RedisCacheConfig:
    """Konfiguration für Redis Cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 20
    decode_responses: bool = False  # Für binäre Daten
    retry_on_timeout: bool = True


@dataclass
class CacheInvalidationRule:
    """Smart Invalidation Regel"""
    pattern: str
    condition: Callable[[Dict[str, Any]], bool]
    action: str  # 'invalidate', 'update', 'refresh'
    priority: int = 1


class RedisCacheManager:
    """
    Redis-basierter Cache Manager mit Smart Invalidation
    Integriert sich mit dem bestehenden IntelligentCache
    """

    def __init__(self, config: RedisCacheConfig = None, namespace: str = "bkki"):
        self.config = config or RedisCacheConfig()
        self.namespace = namespace
        self.redis_client = None
        self.invalidation_rules: List[CacheInvalidationRule] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._connect()

    def _connect(self):
        """Stellt Verbindung zu Redis her"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses,
                retry_on_timeout=self.config.retry_on_timeout
            )
            # Test-Verbindung
            self.redis_client.ping()
            logger.info("✅ Redis Cache Manager verbunden")
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Redis nicht verfügbar: {e}. Fallback auf lokalen Cache.")
            self.redis_client = None

    def _make_key(self, key: str) -> str:
        """Erstellt namespaced Redis-Key"""
        return f"{self.namespace}:{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialisiert Wert für Redis (unterstützt numpy arrays)"""
        if isinstance(value, np.ndarray):
            return pickle.dumps({
                'type': 'numpy_array',
                'data': value.tobytes(),
                'shape': value.shape,
                'dtype': str(value.dtype)
            })
        elif isinstance(value, (dict, list, str, int, float, bool)):
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps({
                'type': 'pickle',
                'data': value
            })

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialisiert Wert aus Redis"""
        try:
            # Versuche JSON zuerst
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fallback auf pickle
            try:
                obj = pickle.loads(data)
                if isinstance(obj, dict) and obj.get('type') == 'numpy_array':
                    array = np.frombuffer(obj['data'], dtype=obj['dtype'])
                    return array.reshape(obj['shape'])
                elif isinstance(obj, dict) and obj.get('type') == 'pickle':
                    return obj['data']
                return obj
            except Exception as e:
                logger.error(f"❌ Deserialisierung fehlgeschlagen: {e}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None,
            metadata: Dict[str, Any] = None) -> bool:
        """
        Speichert Wert im Redis Cache

        Args:
            key: Cache-Key
            value: Zu cachender Wert
            ttl: Time-to-live in Sekunden
            metadata: Zusätzliche Metadaten für Smart Invalidation
        """
        if not self.redis_client:
            return False

        try:
            redis_key = self._make_key(key)
            serialized_value = self._serialize_value(value)

            # Speichere Wert
            if ttl:
                self.redis_client.setex(redis_key, ttl, serialized_value)
            else:
                self.redis_client.set(redis_key, serialized_value)

            # Speichere Metadaten separat für Smart Invalidation
            if metadata:
                meta_key = f"{redis_key}:meta"
                self.redis_client.setex(meta_key, ttl or 3600,
                                      json.dumps(metadata).encode('utf-8'))

            # Führe Smart Invalidation durch
            self._apply_invalidation_rules(key, metadata or {})

            return True
        except Exception as e:
            logger.error(f"❌ Redis set fehlgeschlagen: {e}")
            return False

    def get(self, key: str) -> Any:
        """Holt Wert aus Redis Cache"""
        if not self.redis_client:
            return None

        try:
            redis_key = self._make_key(key)
            data = self.redis_client.get(redis_key)

            if data is None:
                return None

            return self._deserialize_value(data)
        except Exception as e:
            logger.error(f"❌ Redis get fehlgeschlagen: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Löscht Wert aus Redis Cache"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._make_key(key)
            meta_key = f"{redis_key}:meta"

            # Lösche beide Keys
            deleted = self.redis_client.delete(redis_key, meta_key)
            return deleted > 0
        except Exception as e:
            logger.error(f"❌ Redis delete fehlgeschlagen: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Überprüft ob Key existiert"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._make_key(key)
            return bool(self.redis_client.exists(redis_key))
        except Exception as e:
            logger.error(f"❌ Redis exists fehlgeschlagen: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """Gibt verbleibende TTL zurück (-1 = kein TTL, -2 = nicht gefunden)"""
        if not self.redis_client:
            return -2

        try:
            redis_key = self._make_key(key)
            return self.redis_client.ttl(redis_key)
        except Exception as e:
            logger.error(f"❌ Redis TTL fehlgeschlagen: {e}")
            return -2

    def add_invalidation_rule(self, rule: CacheInvalidationRule):
        """Fügt Smart Invalidation Regel hinzu"""
        self.invalidation_rules.append(rule)
        # Sortiere nach Priorität (höher = wichtiger)
        self.invalidation_rules.sort(key=lambda r: r.priority, reverse=True)

    def _apply_invalidation_rules(self, key: str, metadata: Dict[str, Any]):
        """Wendet Smart Invalidation Regeln an"""
        for rule in self.invalidation_rules:
            try:
                if rule.condition(metadata):
                    if rule.action == 'invalidate':
                        self._invalidate_pattern(rule.pattern)
                    elif rule.action == 'update':
                        self._update_related_keys(key, metadata)
                    elif rule.action == 'refresh':
                        self._schedule_refresh(key)
            except Exception as e:
                logger.error(f"❌ Invalidation rule fehlgeschlagen: {e}")

    def _invalidate_pattern(self, pattern: str):
        """Invalidiert alle Keys matching einem Pattern"""
        if not self.redis_client:
            return

        try:
            # Verwende SCAN für effiziente Pattern-Matching
            cursor = 0
            pattern_key = self._make_key(pattern)

            while True:
                cursor, keys = self.redis_client.scan(cursor, match=pattern_key)
                if keys:
                    # Lösche auch Meta-Keys
                    meta_keys = [f"{key}:meta" for key in keys]
                    self.redis_client.delete(*(keys + meta_keys))

                if cursor == 0:
                    break

            logger.info(f"🗑️ {len(keys)} Keys invalidiert (Pattern: {pattern})")
        except Exception as e:
            logger.error(f"❌ Pattern invalidation fehlgeschlagen: {e}")

    def _update_related_keys(self, key: str, metadata: Dict[str, Any]):
        """Aktualisiert verwandte Keys basierend auf Metadaten"""
        # Placeholder für komplexe Update-Logik
        logger.info(f"🔄 Update related keys for: {key}")

    def _schedule_refresh(self, key: str):
        """Plant Refresh für Key"""
        # Asynchrone Refresh-Planung
        self.executor.submit(self._async_refresh, key)

    def _async_refresh(self, key: str):
        """Asynchroner Refresh eines Keys"""
        logger.info(f"🔄 Refreshing key: {key}")
        # Hier würde die Refresh-Logik implementiert werden
        # z.B. erneutes Laden aus Datenbank oder Modell-Inferenz

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Cache-Statistiken zurück"""
        if not self.redis_client:
            return {"status": "disconnected"}

        try:
            info = self.redis_client.info()
            keys = len(self.redis_client.keys(f"{self.namespace}:*"))

            return {
                "status": "connected",
                "total_keys": keys,
                "memory_used": info.get('used_memory_human', 'unknown'),
                "connected_clients": info.get('connected_clients', 0),
                "uptime_seconds": info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"❌ Stats fehlgeschlagen: {e}")
            return {"status": "error", "error": str(e)}

    def clear_namespace(self):
        """Leert gesamten Namespace"""
        if not self.redis_client:
            return

        try:
            keys = self.redis_client.keys(f"{self.namespace}:*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info(f"🗑️ {len(keys)} Keys aus Namespace {self.namespace} gelöscht")
        except Exception as e:
            logger.error(f"❌ Clear namespace fehlgeschlagen: {e}")

    def health_check(self) -> bool:
        """Überprüft Redis-Verbindung"""
        if not self.redis_client:
            return False

        try:
            return self.redis_client.ping()
        except:
            return False


class HybridCache:
    """
    Hybrid Cache: Kombiniert lokalen IntelligentCache mit Redis
    Automatische Fallback-Strategie
    """

    def __init__(self, redis_config: RedisCacheConfig = None,
                 local_cache_size_mb: int = 100):
        self.redis_cache = RedisCacheManager(redis_config)
        self.local_cache = None

        # Lazy loading des lokalen Caches
        self._local_cache_size_mb = local_cache_size_mb

    def _get_local_cache(self):
        """Lazy initialization des lokalen Caches"""
        if self.local_cache is None:
            try:
                from .intelligent_cache import IntelligentCache
                self.local_cache = IntelligentCache(
                    name="hybrid_fallback",
                    max_size_mb=self._local_cache_size_mb,
                    enable_compression=True
                )
            except ImportError:
                logger.warning("⚠️ IntelligentCache nicht verfügbar")
                self.local_cache = None
        return self.local_cache

    def set(self, key: str, value: Any, **kwargs) -> bool:
        """Set mit Fallback auf lokalen Cache"""
        # Versuche Redis zuerst
        if self.redis_cache.set(key, value, **kwargs):
            return True

        # Fallback auf lokalen Cache
        local_cache = self._get_local_cache()
        if local_cache:
            try:
                local_cache.set(key, value, **kwargs)
                return True
            except Exception as e:
                logger.error(f"❌ Lokaler Cache Fallback fehlgeschlagen: {e}")

        return False

    def get(self, key: str) -> Any:
        """Get mit Fallback auf lokalen Cache"""
        # Versuche Redis zuerst
        value = self.redis_cache.get(key)
        if value is not None:
            return value

        # Fallback auf lokalen Cache
        local_cache = self._get_local_cache()
        if local_cache:
            try:
                return local_cache.get(key)
            except Exception as e:
                logger.error(f"❌ Lokaler Cache Fallback fehlgeschlagen: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Kombinierte Statistiken"""
        stats = {"hybrid_cache": True}

        # Redis Stats
        redis_stats = self.redis_cache.get_stats()
        stats.update({f"redis_{k}": v for k, v in redis_stats.items()})

        # Lokaler Cache Stats
        local_cache = self._get_local_cache()
        if local_cache:
            try:
                local_stats = local_cache.get_stats()
                stats.update({f"local_{k}": v for k, v in local_stats.items()})
            except Exception as e:
                stats["local_status"] = "error"

        return stats