#!/usr/bin/env python3
"""
Intelligentes Caching-System für Bundeskanzler-KI
Erweiterte Features: Semantische Ähnlichkeit, LRU Eviction, Kompression, Monitoring
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Intelligenter Cache-Eintrag mit Metadaten"""
    key: str
    value: Any
    embedding: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Überprüft ob Eintrag abgelaufen ist"""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)

    def update_access(self):
        """Aktualisiert Zugriffsstatistiken"""
        self.last_accessed = time.time()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """Gibt Alter des Eintrags in Sekunden zurück"""
        return time.time() - self.created_at

    def get_score(self) -> float:
        """Berechnet Score für LRU-Eviction (kombiniert Alter und Zugriffshäufigkeit)"""
        # Höhere Scores = wichtiger (weniger wahrscheinlich evicted)
        recency_score = 1.0 / (1.0 + self.get_age_seconds() / 3600)  # Stündliche Abnahme
        frequency_score = min(1.0, self.access_count / 10.0)  # Max bei 10 Zugriffen
        return (recency_score * 0.7) + (frequency_score * 0.3)


class IntelligentCache:
    """
    Intelligenter Cache mit semantischer Ähnlichkeit und LRU-Eviction
    """

    def __init__(self, name: str, max_size_mb: float = 500, cache_dir: str = "data/cache",
                 enable_compression: bool = True, enable_similarity: bool = True):
        self.name = name
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache_dir = cache_dir
        self.enable_compression = enable_compression
        self.enable_similarity = enable_similarity

        # Cache-Speicher (OrderedDict für LRU)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()

        # Statistiken
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "compressions": 0,
            "similarity_hits": 0,
            "size_bytes": 0,
            "entries": 0
        }

        # Cache-Datei
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"{name}_intelligent.pkl")

        # Lade Cache bei Initialisierung
        self._load_cache()

        logger.info(f"🧠 IntelligentCache '{name}' initialisiert - Max: {max_size_mb}MB")

    def _load_cache(self):
        """Lädt Cache aus Datei"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', OrderedDict())
                    self.stats = data.get('stats', self.stats)
                logger.info(f"✅ Cache '{self.name}' geladen - {len(self.cache)} Einträge")
            except Exception as e:
                logger.warning(f"⚠️ Fehler beim Laden von Cache '{self.name}': {e}")

    def _save_cache(self):
        """Speichert Cache in Datei"""
        try:
            data = {
                'cache': self.cache,
                'stats': self.stats,
                'timestamp': time.time()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"⚠️ Fehler beim Speichern von Cache '{self.name}': {e}")

    def _compress_value(self, value: Any) -> Tuple[Any, bool]:
        """Komprimiert Wert wenn möglich"""
        if not self.enable_compression:
            return value, False

        try:
            # Konvertiere zu JSON-String für Kompression
            if isinstance(value, (dict, list)):
                json_str = json.dumps(value, ensure_ascii=False)
                compressed = zlib.compress(json_str.encode('utf-8'))
                if len(compressed) < len(json_str.encode('utf-8')) * 0.8:  # Nur wenn >20% Einsparung
                    return compressed, True
        except Exception:
            pass

        return value, False

    def _decompress_value(self, value: Any, compressed: bool) -> Any:
        """Dekomprimiert Wert wenn nötig"""
        if not compressed:
            return value

        try:
            decompressed = zlib.decompress(value)
            return json.loads(decompressed.decode('utf-8'))
        except Exception as e:
            logger.warning(f"⚠️ Fehler beim Dekomprimieren: {e}")
            return value

    def _calculate_size(self, value: Any) -> int:
        """Berechnet ungefähre Größe eines Werts in Bytes"""
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, (list, tuple)):
            return sum(self._calculate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(len(str(k)) + self._calculate_size(v) for k, v in value.items())
        elif isinstance(value, np.ndarray):
            return value.nbytes
        else:
            return len(str(value))

    def _evict_lru(self, required_space: int = 0):
        """Entfernt am wenigsten verwendete Einträge (LRU)"""
        target_size = self.max_size_bytes - required_space

        while self.stats['size_bytes'] > target_size and self.cache:
            # Finde Eintrag mit niedrigstem Score
            worst_key = None
            worst_score = float('inf')

            for key, entry in self.cache.items():
                score = entry.get_score()
                if score < worst_score:
                    worst_score = score
                    worst_key = key

            if worst_key:
                entry = self.cache[worst_key]
                self.stats['size_bytes'] -= entry.size_bytes
                self.stats['entries'] -= 1
                self.stats['evictions'] += 1
                del self.cache[worst_key]
                logger.debug(f"🗑️ Evicted '{worst_key}' (score: {worst_score:.3f})")

    def _find_similar(self, query_embedding: np.ndarray, threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Findet semantisch ähnlichen Cache-Eintrag"""
        if not self.enable_similarity or not query_embedding is not None:
            return None

        best_match = None
        best_similarity = 0.0

        for key, entry in self.cache.items():
            if entry.embedding is not None:
                try:
                    # Kosinus-Ähnlichkeit
                    similarity = np.dot(query_embedding, entry.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                    )

                    if similarity > threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = key
                except Exception:
                    continue

        if best_match:
            logger.debug(f"🔍 Similar match found: '{best_match}' (similarity: {best_similarity:.3f})")
            return best_match, best_similarity

        return None

    def get(self, key: str, query_embedding: Optional[np.ndarray] = None,
            similarity_threshold: float = 0.85) -> Any:
        """Holt Wert aus Cache mit optionaler semantischer Suche"""
        with self.lock:
            # Direkter Hit
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    self.delete(key)
                    self.stats['misses'] += 1
                    return None

                entry.update_access()
                self.cache.move_to_end(key)  # Markiere als zuletzt verwendet
                self.stats['hits'] += 1

                value = self._decompress_value(entry.value, entry.compressed)
                return value

            # Semantische Suche falls kein direkter Hit
            similar_match = self._find_similar(query_embedding, similarity_threshold)
            if similar_match:
                similar_key, similarity = similar_match
                entry = self.cache[similar_key]
                entry.update_access()
                self.cache.move_to_end(similar_key)
                self.stats['similarity_hits'] += 1
                self.stats['hits'] += 1

                value = self._decompress_value(entry.value, entry.compressed)
                logger.debug(f"🎯 Similar hit: '{key}' -> '{similar_key}' ({similarity:.3f})")
                return value

            self.stats['misses'] += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            embedding: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None):
        """Setzt Wert in Cache"""
        with self.lock:
            # Komprimiere Wert
            compressed_value, is_compressed = self._compress_value(value)
            if is_compressed:
                self.stats['compressions'] += 1

            # Berechne Größe
            size_bytes = self._calculate_size(compressed_value)

            # Erstelle Eintrag
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                embedding=embedding,
                size_bytes=size_bytes,
                compressed=is_compressed,
                ttl=ttl,
                metadata=metadata or {}
            )

            # Evict wenn nötig
            if key not in self.cache:
                self._evict_lru(size_bytes)

            # Aktualisiere Statistiken
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats['size_bytes'] -= old_entry.size_bytes
            else:
                self.stats['entries'] += 1

            self.stats['size_bytes'] += size_bytes
            self.stats['sets'] += 1

            # Speichere Eintrag
            self.cache[key] = entry
            self.cache.move_to_end(key)  # Markiere als zuletzt verwendet

            # Periodisch speichern (alle 10 Sets)
            if self.stats['sets'] % 10 == 0:
                self._save_cache()

    def delete(self, key: str) -> bool:
        """Löscht Wert aus Cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats['size_bytes'] -= entry.size_bytes
                self.stats['entries'] -= 1
                self.stats['deletes'] += 1
                del self.cache[key]
                self._save_cache()
                return True
            return False

    def clear(self):
        """Leert Cache komplett"""
        with self.lock:
            self.cache.clear()
            self.stats = {k: 0 for k in self.stats}
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            logger.info(f"🧹 Cache '{self.name}' geleert")

    def get_stats(self) -> Dict[str, Any]:
        """Gibt detaillierte Statistiken zurück"""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                "hit_rate": self._calculate_hit_rate(),
                "size_mb": self.stats['size_bytes'] / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_percent": (self.stats['size_bytes'] / self.max_size_bytes) * 100,
                "entries": len(self.cache),
                "oldest_entry_age_hours": self._get_oldest_entry_age_hours(),
                "compression_ratio": self._calculate_compression_ratio()
            })
            return stats

    def _calculate_hit_rate(self) -> float:
        """Berechnet Cache Hit Rate"""
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests == 0:
            return 0.0
        return (self.stats['hits'] / total_requests) * 100

    def _get_oldest_entry_age_hours(self) -> float:
        """Gibt Alter des ältesten Eintrags in Stunden zurück"""
        if not self.cache:
            return 0.0

        oldest = min(entry.created_at for entry in self.cache.values())
        return (time.time() - oldest) / 3600

    def _calculate_compression_ratio(self) -> float:
        """Berechnet durchschnittliches Kompressionsverhältnis"""
        if self.stats['compressions'] == 0:
            return 1.0

        compressed_entries = [e for e in self.cache.values() if e.compressed]
        if not compressed_entries:
            return 1.0

        total_original = sum(e.size_bytes * 2 for e in compressed_entries)  # Schätzung
        total_compressed = sum(e.size_bytes for e in compressed_entries)

        return total_original / total_compressed if total_compressed > 0 else 1.0

    def cleanup_expired(self):
        """Entfernt abgelaufene Einträge"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self.cache[key]
                self.stats['size_bytes'] -= entry.size_bytes
                self.stats['entries'] -= 1
                self.stats['deletes'] += 1
                del self.cache[key]

            if expired_keys:
                logger.info(f"🧹 Entfernt {len(expired_keys)} abgelaufene Einträge aus '{self.name}'")
                self._save_cache()

    def optimize(self):
        """Optimiert Cache (bereinigt und speichert)"""
        with self.lock:
            self.cleanup_expired()
            self._save_cache()
            logger.info(f"⚡ Cache '{self.name}' optimiert")


class IntelligentCacheManager:
    """
    Manager für mehrere intelligente Caches
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.caches: Dict[str, IntelligentCache] = {}
        self.lock = threading.RLock()

        # Standard-Cache-Konfigurationen
        self.default_configs = {
            "embeddings": {"max_size_mb": 1000, "enable_similarity": True},
            "responses": {"max_size_mb": 500, "enable_similarity": True},
            "models": {"max_size_mb": 2000, "enable_similarity": False},
            "api_responses": {"max_size_mb": 200, "enable_similarity": False},
            "search_results": {"max_size_mb": 300, "enable_similarity": True},
        }

        os.makedirs(cache_dir, exist_ok=True)
        logger.info("🎯 IntelligentCacheManager initialisiert")

    def get_cache(self, name: str) -> IntelligentCache:
        """Holt oder erstellt einen intelligenten Cache"""
        with self.lock:
            if name not in self.caches:
                config = self.default_configs.get(name, {"max_size_mb": 500})
                self.caches[name] = IntelligentCache(
                    name=name,
                    cache_dir=self.cache_dir,
                    **config
                )
            return self.caches[name]

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken aller Caches zurück"""
        with self.lock:
            total_stats = {
                "total_caches": len(self.caches),
                "total_size_mb": 0,
                "total_entries": 0,
                "cache_details": {}
            }

            for name, cache in self.caches.items():
                stats = cache.get_stats()
                total_stats["total_size_mb"] += stats.get("size_mb", 0)
                total_stats["total_entries"] += stats.get("entries", 0)
                total_stats["cache_details"][name] = stats

            return total_stats

    def clear_all(self):
        """Leert alle Caches"""
        with self.lock:
            for cache in self.caches.values():
                cache.clear()
            logger.info("🧹 Alle intelligenten Caches geleert")

    def optimize_all(self):
        """Optimiert alle Caches"""
        with self.lock:
            for cache in self.caches.values():
                cache.optimize()
            logger.info("⚡ Alle intelligenten Caches optimiert")

    def cleanup_all(self):
        """Bereinigt alle Caches (entfernt abgelaufene Einträge)"""
        with self.lock:
            for cache in self.caches.values():
                cache.cleanup_expired()


# Globale Instanz
intelligent_cache_manager = IntelligentCacheManager()

def get_intelligent_cache(name: str) -> IntelligentCache:
    """Holt einen intelligenten Cache"""
    return intelligent_cache_manager.get_cache(name)

def get_intelligent_cache_stats() -> Dict[str, Any]:
    """Gibt Statistiken aller intelligenten Caches zurück"""
    return intelligent_cache_manager.get_stats()

def initialize_intelligent_caches():
    """Initialisiert alle intelligenten Caches"""
    # Erstelle Standard-Caches
    for cache_name in ["embeddings", "responses", "models", "api_responses", "search_results"]:
        intelligent_cache_manager.get_cache(cache_name)

    logger.info("✅ Intelligente Caches initialisiert")

# Automatisch initialisieren
initialize_intelligent_caches()