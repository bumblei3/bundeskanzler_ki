"""
Optimierte Memory-Systeme für die Bundeskanzler KI
Implementiert Quantisierung, Caching und Performance-Optimierungen
"""

import json
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class QuantizedEmbedding:
    """Quantisiertes Embedding für Speicher-Effizienz"""

    values: np.ndarray  # int8 oder float16 Werte
    scale: float  # Skalierungsfaktor für Dequantisierung
    offset: float  # Offset für Dequantisierung
    original_dtype: np.dtype
    compression_type: str  # 'int8', 'float16', 'none'

    def dequantize(self) -> np.ndarray:
        """Wandelt quantisiertes Embedding zurück in Originalformat"""
        if self.compression_type == "int8":
            return (self.values.astype(np.float32) * self.scale) + self.offset
        elif self.compression_type == "float16":
            return self.values.astype(np.float32)
        else:
            return self.values


class MemoryPool:
    """Memory Pool für effiziente Wiederverwendung von Arrays"""

    def __init__(self, embedding_dim: int, pool_size: int = 1000):
        self.embedding_dim = embedding_dim
        self.pool_size = pool_size
        self.pool = []
        self.lock = threading.Lock()

        # Pre-allocate pool
        for _ in range(pool_size):
            self.pool.append(np.zeros(embedding_dim, dtype=np.float32))

    def get_array(self) -> np.ndarray:
        """Holt ein Array aus dem Pool oder erstellt neues"""
        with self.lock:
            if self.pool:
                return self.pool.pop()
            else:
                return np.zeros(self.embedding_dim, dtype=np.float32)

    def return_array(self, array: np.ndarray):
        """Gibt Array zurück in den Pool"""
        with self.lock:
            if len(self.pool) < self.pool_size and array.shape == (self.embedding_dim,):
                array.fill(0)  # Reset to zeros
                self.pool.append(array)


class QuantizationEngine:
    """Engine für Embedding-Quantisierung"""

    @staticmethod
    def quantize_int8(embedding: np.ndarray) -> QuantizedEmbedding:
        """Quantisiert float32 Embedding zu int8"""
        # Berechne Skalierungsfaktoren
        min_val = np.min(embedding)
        max_val = np.max(embedding)
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        offset = min_val

        # Quantisiere zu int8
        quantized = np.round((embedding - offset) / scale).astype(np.int8)
        quantized = np.clip(quantized, -127, 127)  # int8 range

        return QuantizedEmbedding(
            values=quantized,
            scale=scale,
            offset=offset,
            original_dtype=embedding.dtype,
            compression_type="int8",
        )

    @staticmethod
    def quantize_float16(embedding: np.ndarray) -> QuantizedEmbedding:
        """Konvertiert zu float16 für reduzierte Präzision"""
        return QuantizedEmbedding(
            values=embedding.astype(np.float16),
            scale=1.0,
            offset=0.0,
            original_dtype=embedding.dtype,
            compression_type="float16",
        )

    @staticmethod
    def auto_quantize(
        embedding: np.ndarray, target_memory_mb: float = 100.0
    ) -> QuantizedEmbedding:
        """Automatische Quantisierung basierend auf Speicherziel"""
        embedding_size_mb = embedding.nbytes / (1024 * 1024)

        if embedding_size_mb > target_memory_mb:
            return QuantizationEngine.quantize_int8(embedding)
        elif embedding_size_mb > target_memory_mb * 0.5:
            return QuantizationEngine.quantize_float16(embedding)
        else:
            return QuantizedEmbedding(
                values=embedding,
                scale=1.0,
                offset=0.0,
                original_dtype=embedding.dtype,
                compression_type="none",
            )


class LRUEmbeddingCache:
    """LRU Cache für häufig verwendete Embeddings"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Holt Embedding aus Cache"""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    return None

                self.access_times[key] = time.time()
                return self.cache[key].copy()
            return None

    def put(self, key: str, embedding: np.ndarray):
        """Speichert Embedding im Cache"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Entferne ältestes Element
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = embedding.copy()
            self.access_times[key] = time.time()

    def clear(self):
        """Leert den Cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """Cache-Statistiken"""
        with self.lock:
            # Berechne Memory-Usage für alle Embeddings im Cache
            memory_usage = sum(arr.nbytes for arr in self.cache.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": 0.0,  # Könnte mit Countern implementiert werden
                "memory_usage_mb": memory_usage / (1024 * 1024),
            }


@dataclass
class OptimizedMemoryItem:
    """Optimiertes Gedächtnis-Element mit Quantisierung"""

    content: str
    embedding: Union[np.ndarray, QuantizedEmbedding]
    timestamp: datetime
    importance: float
    access_count: int = 0
    last_access: Optional[datetime] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    content_hash: Optional[str] = None  # Für Deduplizierung

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.last_access is None:
            self.last_access = self.timestamp
        if self.content_hash is None:
            import hashlib

            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()

    def get_embedding(self) -> np.ndarray:
        """Gibt dequantisiertes Embedding zurück"""
        if isinstance(self.embedding, QuantizedEmbedding):
            return self.embedding.dequantize()
        return self.embedding

    def memory_usage(self) -> int:
        """Gibt Speicherverbrauch in Bytes zurück"""
        if isinstance(self.embedding, QuantizedEmbedding):
            return self.embedding.values.nbytes
        return self.embedding.nbytes


class OptimizedHierarchicalMemory:
    """
    Optimiertes hierarchisches Gedächtnissystem mit Quantisierung und Caching
    """

    def __init__(
        self,
        short_term_capacity: int = 50,
        long_term_capacity: int = 1000,
        embedding_dim: int = 512,
        decay_factor: float = 0.95,
        importance_threshold: float = 0.7,
        persistence_path: Optional[str] = None,
        enable_quantization: bool = True,
        enable_caching: bool = True,
        cache_size: int = 500,
        memory_pool_size: int = 1000,
    ):
        """
        Initialisiert das optimierte hierarchische Gedächtnissystem

        Args:
            short_term_capacity: Kapazität des Kurzzeitgedächtnisses
            long_term_capacity: Kapazität des Langzeitgedächtnisses
            embedding_dim: Dimensionalität der Embeddings
            decay_factor: Faktor für zeitbasierte Bedeutungsabnahme
            importance_threshold: Schwellwert für Übergang ins Langzeitgedächtnis
            persistence_path: Pfad für persistente Speicherung
            enable_quantization: Aktiviert Embedding-Quantisierung
            enable_caching: Aktiviert LRU-Cache für Embeddings
            cache_size: Größe des Embedding-Caches
            memory_pool_size: Größe des Memory-Pools
        """
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.embedding_dim = embedding_dim
        self.decay_factor = decay_factor
        self.importance_threshold = importance_threshold
        self.enable_quantization = enable_quantization
        self.enable_caching = enable_caching

        # Gedächtnisspeicher
        self.short_term_memory: List[OptimizedMemoryItem] = []
        self.long_term_memory: List[OptimizedMemoryItem] = []

        # Optimierungskomponenten
        self.quantization_engine = QuantizationEngine()
        self.embedding_cache = (
            LRUEmbeddingCache(max_size=cache_size) if enable_caching else None
        )
        self.memory_pool = (
            MemoryPool(embedding_dim, memory_pool_size) if enable_caching else None
        )

        # Semantic Index für schnelle Suche
        self.semantic_index = {}

        # Performance-Metriken
        self.metrics = {
            "total_memories": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "quantization_savings_mb": 0.0,
            "search_time_avg": 0.0,
            "memory_usage_mb": 0.0,
        }

        # Persistenz
        self.persistence_path = Path(persistence_path) if persistence_path else None
        if self.persistence_path:
            self.load_memory()

    def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        importance: float = 0.5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        Fügt ein neues Gedächtnis hinzu mit Optimierungen
        """
        # Deduplizierung prüfen
        content_hash = self._hash_content(content)
        if self._is_duplicate(content_hash):
            return

        # Embedding optimieren
        if self.enable_quantization:
            optimized_embedding = self.quantization_engine.auto_quantize(embedding)
            original_size = embedding.nbytes
            optimized_size = optimized_embedding.values.nbytes
            self.metrics["quantization_savings_mb"] += (
                original_size - optimized_size
            ) / (1024 * 1024)
        else:
            optimized_embedding = embedding

        # Cache aktualisieren
        if self.enable_caching and self.embedding_cache:
            self.embedding_cache.put(content_hash, embedding)

        memory_item = OptimizedMemoryItem(
            content=content,
            embedding=optimized_embedding,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            content_hash=content_hash,
        )

        # Zu Kurzzeitgedächtnis hinzufügen
        self.short_term_memory.append(memory_item)
        self.metrics["total_memories"] += 1

        # Kapazität verwalten
        self._manage_capacity()

        # Semantic Index aktualisieren
        self._update_semantic_index(memory_item)

    def search_semantic(
        self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7
    ) -> List[Tuple[OptimizedMemoryItem, float]]:
        """
        Optimierte semantische Suche mit Cache-Unterstützung
        """
        start_time = time.time()

        # Cache prüfen für Query
        query_hash = self._hash_embedding(query_embedding)
        cached_result = None
        if self.enable_caching and self.embedding_cache:
            cached_result = self.embedding_cache.get(f"search_{query_hash}")

        if cached_result is not None:
            self.metrics["cache_hits"] += 1
            search_time = time.time() - start_time
            self._update_search_time(search_time)
            return cached_result

        self.metrics["cache_misses"] += 1

        # Normale Suche durchführen
        results = []
        all_memories = self.short_term_memory + self.long_term_memory

        for memory in all_memories:
            embedding = memory.get_embedding()
            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity >= threshold:
                results.append((memory, similarity))

        # Nach Ähnlichkeit sortieren
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        # Cache für Suchergebnisse könnte hier hinzugefügt werden (optional)
        # if self.enable_caching and self.embedding_cache:
        #     self.embedding_cache.put(f"search_{query_hash}", results)

        search_time = time.time() - start_time
        self._update_search_time(search_time)

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Umfassende Memory-Statistiken"""
        short_term_count = len(self.short_term_memory)
        long_term_count = len(self.long_term_memory)

        # Speicherverbrauch berechnen
        short_term_memory_usage = sum(m.memory_usage() for m in self.short_term_memory)
        long_term_memory_usage = sum(m.memory_usage() for m in self.long_term_memory)

        total_memory_usage = short_term_memory_usage + long_term_memory_usage
        self.metrics["memory_usage_mb"] = total_memory_usage / (1024 * 1024)

        return {
            "short_term": {
                "count": short_term_count,
                "memory_usage_mb": short_term_memory_usage / (1024 * 1024),
                "capacity": self.short_term_capacity,
            },
            "long_term": {
                "count": long_term_count,
                "memory_usage_mb": long_term_memory_usage / (1024 * 1024),
                "capacity": self.long_term_capacity,
            },
            "cache": self.embedding_cache.stats() if self.embedding_cache else None,
            "quantization": {
                "enabled": self.enable_quantization,
                "savings_mb": self.metrics["quantization_savings_mb"],
            },
            "performance": {
                "total_memories": self.metrics["total_memories"],
                "cache_hit_rate": self.metrics["cache_hits"]
                / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"]),
                "avg_search_time_ms": self.metrics["search_time_avg"] * 1000,
                "total_memory_usage_mb": self.metrics["memory_usage_mb"],
            },
        }

    def optimize_memory(self) -> Dict[str, Any]:
        """Führt umfassende Memory-Optimierung durch"""
        optimizations = {
            "quantization_applied": 0,
            "duplicates_removed": 0,
            "old_memories_cleaned": 0,
            "memory_saved_mb": 0.0,
        }

        # Alle Embeddings neu quantisieren falls aktiviert
        if self.enable_quantization:
            for memory in self.short_term_memory + self.long_term_memory:
                if not isinstance(memory.embedding, QuantizedEmbedding):
                    original_size = memory.embedding.nbytes
                    memory.embedding = self.quantization_engine.auto_quantize(
                        memory.embedding
                    )
                    new_size = memory.embedding.values.nbytes
                    optimizations["memory_saved_mb"] += (original_size - new_size) / (
                        1024 * 1024
                    )
                    optimizations["quantization_applied"] += 1

        # Duplikate entfernen
        seen_hashes = set()
        original_short_count = len(self.short_term_memory)
        original_long_count = len(self.long_term_memory)

        self.short_term_memory = [
            m
            for m in self.short_term_memory
            if m.content_hash not in seen_hashes and not seen_hashes.add(m.content_hash)
        ]

        self.long_term_memory = [
            m
            for m in self.long_term_memory
            if m.content_hash not in seen_hashes and not seen_hashes.add(m.content_hash)
        ]

        optimizations["duplicates_removed"] = (
            original_short_count - len(self.short_term_memory)
        ) + (original_long_count - len(self.long_term_memory))

        # Alte Memories bereinigen
        optimizations["old_memories_cleaned"] = self.forget_old_memories()

        # Cache leeren für Konsistenz
        if self.embedding_cache:
            self.embedding_cache.clear()

        return optimizations

    # Hilfsmethoden
    def _hash_content(self, content: str) -> str:
        """Erstellt Hash für Content-Deduplizierung"""
        import hashlib

        return hashlib.md5(content.encode()).hexdigest()

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """Erstellt Hash für Embedding-Cache"""
        import hashlib

        return hashlib.md5(embedding.tobytes()).hexdigest()

    def _is_duplicate(self, content_hash: str) -> bool:
        """Prüft auf Duplikate"""
        all_hashes = {
            m.content_hash for m in self.short_term_memory + self.long_term_memory
        }
        return content_hash in all_hashes

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Berechnet Kosinus-Ähnlichkeit"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0.0

    def _manage_capacity(self):
        """Verwaltet Gedächtniskapazität"""
        # Kurzzeitgedächtnis verwalten
        if len(self.short_term_memory) > self.short_term_capacity:
            # Nach Wichtigkeit sortieren und überschüssige entfernen
            self.short_term_memory.sort(key=lambda x: x.importance, reverse=True)
            removed = self.short_term_memory[self.short_term_capacity :]
            self.short_term_memory = self.short_term_memory[: self.short_term_capacity]

            # Wichtige in Langzeitgedächtnis verschieben
            for memory in removed:
                if memory.importance >= self.importance_threshold:
                    self.long_term_memory.append(memory)

        # Langzeitgedächtnis verwalten
        if len(self.long_term_memory) > self.long_term_capacity:
            self.long_term_memory.sort(key=lambda x: x.importance, reverse=True)
            self.long_term_memory = self.long_term_memory[: self.long_term_capacity]

    def _update_semantic_index(self, memory: OptimizedMemoryItem):
        """Aktualisiert Semantic Index"""
        # Vereinfachte Implementierung - könnte durch FAISS oder ähnliches ersetzt werden
        pass

    def _update_search_time(self, search_time: float):
        """Aktualisiert durchschnittliche Suchzeit"""
        alpha = 0.1  # Exponential moving average
        self.metrics["search_time_avg"] = (
            alpha * search_time + (1 - alpha) * self.metrics["search_time_avg"]
        )

    def save_memory(self):
        """Speichert Gedächtnis persistent"""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Konvertiere zu serialisierbarem Format
            serializable_data = {
                "short_term": [self._memory_to_dict(m) for m in self.short_term_memory],
                "long_term": [self._memory_to_dict(m) for m in self.long_term_memory],
                "metrics": self.metrics,
            }

            with open(self.persistence_path, "wb") as f:
                pickle.dump(serializable_data, f)

        except Exception as e:
            print(f"Fehler beim Speichern des Gedächtnisses: {e}")

    def load_memory(self):
        """Lädt Gedächtnis von persistentem Speicher"""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "rb") as f:
                data = pickle.load(f)

            self.short_term_memory = [
                self._dict_to_memory(m) for m in data.get("short_term", [])
            ]
            self.long_term_memory = [
                self._dict_to_memory(m) for m in data.get("long_term", [])
            ]
            self.metrics.update(data.get("metrics", {}))

        except Exception as e:
            print(f"Fehler beim Laden des Gedächtnisses: {e}")

    def _memory_to_dict(self, memory: OptimizedMemoryItem) -> Dict[str, Any]:
        """Konvertiert MemoryItem zu Dictionary für Serialisierung"""
        return {
            "content": memory.content,
            "embedding": (
                memory.embedding.values
                if isinstance(memory.embedding, QuantizedEmbedding)
                else memory.embedding
            ),
            "embedding_meta": (
                {
                    "scale": (
                        memory.embedding.scale
                        if isinstance(memory.embedding, QuantizedEmbedding)
                        else 1.0
                    ),
                    "offset": (
                        memory.embedding.offset
                        if isinstance(memory.embedding, QuantizedEmbedding)
                        else 0.0
                    ),
                    "original_dtype": (
                        str(memory.embedding.original_dtype)
                        if isinstance(memory.embedding, QuantizedEmbedding)
                        else str(memory.embedding.dtype)
                    ),
                    "compression_type": (
                        memory.embedding.compression_type
                        if isinstance(memory.embedding, QuantizedEmbedding)
                        else "none"
                    ),
                }
                if isinstance(memory.embedding, QuantizedEmbedding)
                else None
            ),
            "timestamp": memory.timestamp.isoformat(),
            "importance": memory.importance,
            "access_count": memory.access_count,
            "last_access": (
                memory.last_access.isoformat() if memory.last_access else None
            ),
            "tags": memory.tags,
            "metadata": memory.metadata,
            "content_hash": memory.content_hash,
        }

    def _dict_to_memory(self, data: Dict[str, Any]) -> OptimizedMemoryItem:
        """Konvertiert Dictionary zurück zu MemoryItem"""
        # Embedding rekonstruieren
        if data.get("embedding_meta"):
            meta = data["embedding_meta"]
            embedding = QuantizedEmbedding(
                values=np.array(data["embedding"]),
                scale=meta["scale"],
                offset=meta["offset"],
                original_dtype=np.dtype(meta["original_dtype"]),
                compression_type=meta["compression_type"],
            )
        else:
            embedding = np.array(data["embedding"])

        return OptimizedMemoryItem(
            content=data["content"],
            embedding=embedding,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data["importance"],
            access_count=data.get("access_count", 0),
            last_access=(
                datetime.fromisoformat(data["last_access"])
                if data.get("last_access")
                else None
            ),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            content_hash=data.get("content_hash"),
        )

    def forget_old_memories(self, max_age_days: int = 30) -> int:
        """
        Vergisst alte, unwichtige Gedächtnisinhalte
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Kurzzeitgedächtnis bereinigen
        before_count = len(self.short_term_memory)
        self.short_term_memory = [
            memory
            for memory in self.short_term_memory
            if memory.timestamp > cutoff_date or memory.importance > 0.8
        ]

        # Langzeitgedächtnis bereinigen (nur sehr alte, unwichtige)
        self.long_term_memory = [
            memory
            for memory in self.long_term_memory
            if memory.timestamp > cutoff_date or memory.importance > 0.9
        ]

        forgotten_count = before_count - len(self.short_term_memory)
        return forgotten_count

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Gibt detaillierte Statistiken über das Gedächtnissystem zurück
        """
        try:
            # Berechne Cache-Statistiken
            cache_hit_rate = 0.0
            if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
                cache_hit_rate = self.metrics["cache_hits"] / (
                    self.metrics["cache_hits"] + self.metrics["cache_misses"]
                )

            # Berechne Memory-Effizienz (basierend auf Quantisierungseinsparungen)
            total_memory_mb = sum(
                m.memory_usage() for m in self.short_term_memory + self.long_term_memory
            ) / (1024 * 1024)
            memory_efficiency = 0.0
            if total_memory_mb > 0:
                memory_efficiency = (
                    self.metrics["quantization_savings_mb"] / total_memory_mb
                ) * 100
                memory_efficiency = min(memory_efficiency, 100.0)  # Cap at 100%

            return {
                "short_term_count": len(self.short_term_memory),
                "long_term_count": len(self.long_term_memory),
                "total_entries": len(self.short_term_memory)
                + len(self.long_term_memory),
                "memory_efficiency": memory_efficiency,
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "cache_hit_rate": cache_hit_rate,
                "quantization_enabled": self.enable_quantization,
                "pool_enabled": self.memory_pool is not None,
                "memory_saved_mb": self.metrics["quantization_savings_mb"],
                "total_memory_mb": total_memory_mb,
                "cache_size": (
                    len(self.embedding_cache.cache) if self.embedding_cache else 0
                ),
                "max_cache_size": (
                    self.embedding_cache.max_size if self.embedding_cache else 0
                ),
            }
        except Exception as e:
            print(f"❌ Fehler in get_memory_stats: {e}")
            import traceback

            traceback.print_exc()
            # Return basic stats on error
            return {
                "short_term_count": len(self.short_term_memory),
                "long_term_count": len(self.long_term_memory),
                "total_entries": len(self.short_term_memory)
                + len(self.long_term_memory),
                "memory_efficiency": 0.0,
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "cache_hit_rate": 0.0,
                "quantization_enabled": self.enable_quantization,
                "pool_enabled": self.memory_pool is not None,
                "memory_saved_mb": self.metrics["quantization_savings_mb"],
                "total_memory_mb": 0.0,
                "cache_size": (
                    len(self.embedding_cache.cache) if self.embedding_cache else 0
                ),
                "max_cache_size": (
                    self.embedding_cache.max_size if self.embedding_cache else 0
                ),
                "error": str(e),
            }

    def forget_old_memories(self, max_age_days: int = 30) -> int:
        """
        Vergisst alte, unwichtige Gedächtnisinhalte
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Kurzzeitgedächtnis bereinigen
        before_count = len(self.short_term_memory)
        self.short_term_memory = [
            memory
            for memory in self.short_term_memory
            if memory.timestamp > cutoff_date or memory.importance > 0.8
        ]

        # Langzeitgedächtnis bereinigen (nur sehr alte, unwichtige)
        self.long_term_memory = [
            memory
            for memory in self.long_term_memory
            if memory.timestamp > cutoff_date or memory.importance > 0.9
        ]

        forgotten_count = before_count - len(self.short_term_memory)
        return forgotten_count
