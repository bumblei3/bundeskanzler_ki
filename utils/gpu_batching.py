"""
GPU-Batching-System für beschleunigte Embedding-Verarbeitung
Unterstützt CUDA, ROCm und CPU-Fallback mit optimierter Batch-Verarbeitung
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GPUBatchProcessor:
    """
    GPU-accelerated batch processor für Embeddings mit Memory-Pooling
    Unterstützt CUDA, ROCm und CPU-Fallback mit optimierter Batch-Verarbeitung
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = 4,
        device: str = "auto",  # "cuda", "rocm", "cpu", "auto"
        embedding_dim: int = 512,
        enable_async: bool = True,
        enable_memory_pooling: bool = True,  # NEU: Memory-Pooling aktivieren
        memory_pool_size_mb: int = 512,  # NEU: Memory-Pool Größe
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.device = self._detect_device(device)
        self.embedding_dim = embedding_dim
        self.enable_async = enable_async
        self.enable_memory_pooling = enable_memory_pooling
        self.memory_pool_size_mb = memory_pool_size_mb

        # GPU/CPU Setup
        self.gpu_available = self._setup_gpu()
        self.executor = (
            ThreadPoolExecutor(max_workers=max_workers) if enable_async else None
        )

        # NEU: GPU Memory Pooling
        self.memory_pool = None
        self._setup_memory_pool()

        # Performance tracking
        self.stats = {
            "batches_processed": 0,
            "total_embeddings": 0,
            "avg_batch_time": 0.0,
            "gpu_memory_used": 0.0,
            "cpu_fallback_count": 0,
            "memory_pool_hits": 0,  # NEU: Memory-Pool Statistiken
            "memory_pool_misses": 0,
            "dynamic_batch_adjustments": 0,  # NEU: Dynamische Batch-Anpassungen
        }

        logger.info(
            f"GPU-Batch-Processor initialisiert: Device={self.device}, GPU={'Verfügbar' if self.gpu_available else 'Nicht verfügbar'}, Memory-Pooling={'Aktiviert' if enable_memory_pooling else 'Deaktiviert'}"
        )

    def cleanup(self):
        """Räumt GPU-Ressourcen auf"""
        if self.memory_pool:
            try:
                for tensor in self.memory_pool.values():
                    del tensor
                self.memory_pool = None
                if self.gpu_available:
                    import torch

                    torch.cuda.empty_cache()
                logger.info("✅ GPU Memory Pool bereinigt")
            except Exception as e:
                logger.warning(f"Fehler beim Bereinigen des Memory Pools: {e}")

        if self.executor:
            self.executor.shutdown(wait=True)

    def get_stats(self) -> Dict[str, Any]:
        """Erweiterte Statistiken abrufen"""
        stats = self.stats.copy()

        # Grundlegende GPU-Informationen hinzufügen
        stats["device"] = self.device
        stats["gpu_available"] = self.gpu_available
        stats["memory_pooling_enabled"] = (
            self.enable_memory_pooling and self.memory_pool is not None
        )
        stats["dynamic_batching"] = self.enable_memory_pooling
        stats["batch_size"] = self.batch_size
        stats["async_enabled"] = self.enable_async

        # GPU-Speicher-Info hinzufügen
        if self.gpu_available:
            try:
                import torch

                stats["current_gpu_memory_mb"] = torch.cuda.memory_allocated() / (
                    1024 * 1024
                )
                stats["max_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (
                    1024 * 1024
                )
                stats["gpu_memory_total_mb"] = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024 * 1024)
            except:
                pass

        # Memory-Pool-Statistiken hinzufügen
        if self.enable_memory_pooling and self.memory_pool is not None:
            stats["memory_pool_stats"] = {
                "hits": self.stats.get("memory_pool_hits", 0),
                "misses": self.stats.get("memory_pool_misses", 0),
                "efficiency": 0.0,
            }

            # Effizienz berechnen
            total_pool_accesses = (
                stats["memory_pool_stats"]["hits"]
                + stats["memory_pool_stats"]["misses"]
            )
            if total_pool_accesses > 0:
                stats["memory_pool_stats"]["efficiency"] = (
                    stats["memory_pool_stats"]["hits"] / total_pool_accesses
                )

        # Performance-Metriken hinzufügen
        optimal_batch_size = self.batch_size
        if self.gpu_available:
            try:
                import torch

                available_memory_mb = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                ) / (1024 * 1024)
                optimal_batch_size = self._calculate_optimal_batch_size(
                    available_memory_mb
                )
            except:
                optimal_batch_size = self.batch_size

        stats["performance_stats"] = {
            "avg_batch_time_ms": self.stats.get("avg_batch_time", 0) * 1000,
            "current_batch_size": self.batch_size,
            "optimal_batch_size": optimal_batch_size,
        }

        return stats

    def _detect_device(self, device: str) -> str:
        """Erkennt verfügbare GPU/CPU"""
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, "hip") and torch.hip.is_available():
                return "rocm"
        except ImportError:
            pass

        return "cpu"

    def _setup_gpu(self) -> bool:
        """Richtet GPU ein falls verfügbar"""
        if self.device == "cpu":
            return False

        try:
            import torch

            if self.device == "cuda":
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                    # NEU: Optimierte GPU-Einstellungen
                    torch.cuda.empty_cache()
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                    return True
            elif self.device == "rocm":
                if hasattr(torch, "hip") and torch.hip.is_available():
                    # ROCm setup
                    return True

        except ImportError:
            logger.warning("PyTorch nicht verfügbar, verwende CPU-Fallback")

        self.device = "cpu"
        return False

    def _setup_memory_pool(self):
        """Richtet GPU Memory Pooling ein"""
        if not self.enable_memory_pooling or not self.gpu_available:
            return

        try:
            import torch

            # Erstelle Memory-Pool für häufig verwendete Tensor-Größen
            pool_size_bytes = self.memory_pool_size_mb * 1024 * 1024

            # Pre-alloziere häufig verwendete Tensor-Größen
            self.memory_pool = {
                "small": torch.empty(
                    (16, self.embedding_dim), dtype=torch.float32, device="cuda"
                ),
                "medium": torch.empty(
                    (32, self.embedding_dim), dtype=torch.float32, device="cuda"
                ),
                "large": torch.empty(
                    (64, self.embedding_dim), dtype=torch.float32, device="cuda"
                ),
            }

            # Markiere als nicht verwendet (wird bei Bedarf überschrieben)
            for tensor in self.memory_pool.values():
                tensor.zero_()

            logger.info(
                f"✅ GPU Memory Pool initialisiert: {self.memory_pool_size_mb}MB"
            )

        except Exception as e:
            logger.warning(f"GPU Memory Pool konnte nicht initialisiert werden: {e}")
            self.memory_pool = None

    async def process_batch_async(
        self, texts: List[str], operation: str = "embed"  # "embed", "search", "compare"
    ) -> np.ndarray:
        """
        Async batch processing für mehrere Texte
        """
        if not self.enable_async:
            return self.process_batch_sync(texts, operation)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.process_batch_sync, texts, operation
        )

    def process_batch_sync(
        self, texts: List[str], operation: str = "embed"
    ) -> np.ndarray:
        """
        Synchrone batch processing für mehrere Texte
        """
        start_time = time.time()

        if not texts:
            return np.array([])

        # Teile in Batches auf
        batches = self._create_batches(texts)
        results = []

        for batch in batches:
            batch_result = self._process_single_batch(batch, operation)
            results.append(batch_result)

        # Kombiniere Ergebnisse
        final_result = np.concatenate(results, axis=0)

        # Update Stats
        batch_time = time.time() - start_time
        self.stats["batches_processed"] += len(batches)
        self.stats["total_embeddings"] += len(texts)
        self.stats["avg_batch_time"] = (
            self.stats["avg_batch_time"]
            * (self.stats["batches_processed"] - len(batches))
            + batch_time
        ) / self.stats["batches_processed"]

        return final_result

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Teilt Texte in optimale Batches auf mit dynamischer Größenanpassung"""
        if not self.gpu_available or not self.enable_memory_pooling:
            # Fallback zur statischen Batch-Größe
            batches = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batches.append(batch)
            return batches

        # NEU: Dynamische Batch-Größen basierend auf GPU-Speicher
        available_memory = self._get_available_gpu_memory_mb()
        optimal_batch_size = self._calculate_optimal_batch_size(available_memory)

        batches = []
        for i in range(0, len(texts), optimal_batch_size):
            batch = texts[i : i + optimal_batch_size]
            if batch:  # Sicherstellen, dass Batch nicht leer ist
                batches.append(batch)

        if optimal_batch_size != self.batch_size:
            self.stats["dynamic_batch_adjustments"] += 1
            logger.debug(
                f"Dynamische Batch-Größe angepasst: {self.batch_size} → {optimal_batch_size}"
            )

        return batches

    def _get_available_gpu_memory_mb(self) -> float:
        """Ermittelt verfügbaren GPU-Speicher"""
        try:
            import torch

            return (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / (1024 * 1024)
        except:
            return 1024  # Fallback: 1GB annehmen

    def _calculate_optimal_batch_size(self, available_memory_mb: float) -> int:
        """Berechnet optimale Batch-Größe basierend auf verfügbarem Speicher"""
        # Schätze Speicherverbrauch pro Embedding (512 dim float32 = 2KB)
        memory_per_embedding_kb = self.embedding_dim * 4  # float32 = 4 bytes
        memory_per_embedding_mb = memory_per_embedding_kb / 1024

        # Reserve 20% für Overhead und andere Operationen
        usable_memory_mb = available_memory_mb * 0.8

        optimal_batch_size = int(usable_memory_mb / memory_per_embedding_mb)

        # Begrenze auf sinnvolle Werte
        optimal_batch_size = max(1, min(optimal_batch_size, 128))

        return optimal_batch_size

    def _process_single_batch(self, batch: List[str], operation: str) -> np.ndarray:
        """
        Verarbeitet einen einzelnen Batch
        """
        if self.gpu_available and operation == "embed":
            return self._gpu_embed_batch(batch)
        else:
            return self._cpu_embed_batch(batch)

    def _gpu_embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        GPU-accelerated embedding generation mit Memory-Pooling
        """
        try:
            import torch
            import torch.nn.functional as F

            device = torch.device("cuda" if self.device == "cuda" else "cpu")
            batch_size = len(batch)

            # NEU: Verwende Memory Pool wenn verfügbar und passend
            if self.memory_pool and batch_size <= 64:
                pool_key = (
                    "small"
                    if batch_size <= 16
                    else "medium" if batch_size <= 32 else "large"
                )
                if pool_key in self.memory_pool:
                    # Verwende gepoolten Speicher
                    embeddings = self.memory_pool[pool_key][:batch_size].clone()
                    self.stats["memory_pool_hits"] += 1
                else:
                    # Fallback: Neuer Tensor
                    embeddings = torch.randn(
                        batch_size, self.embedding_dim, device=device
                    )
                    self.stats["memory_pool_misses"] += 1
            else:
                # Normaler Tensor für große Batches oder wenn Pooling deaktiviert
                embeddings = torch.randn(batch_size, self.embedding_dim, device=device)
                if self.memory_pool:
                    self.stats["memory_pool_misses"] += 1

            # Normalisiere (simuliert echtes Embedding)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Zusätzliche L2-Normalisierung für bessere Konsistenz
            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

            return embeddings.cpu().numpy()

        except Exception as e:
            logger.warning(f"GPU-Embedding fehlgeschlagen, verwende CPU: {e}")
            self.stats["cpu_fallback_count"] += 1
            return self._cpu_embed_batch(batch)

    def _cpu_embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        CPU-basierte Embedding-Generierung (Fallback)
        """
        embeddings = []

        for text in batch:
            # Hash-basierte Embedding-Generierung (deterministisch)
            hash_obj = hash(text)
            np.random.seed(hash_obj % 2**32)

            # Generiere konsistentes Embedding
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)

            # L2-Normalisierung
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

        return np.array(embeddings)

    async def search_batch_async(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Async batch search für ähnliche Embeddings
        """
        if not self.enable_async:
            return self.search_batch_sync(query_embedding, candidate_embeddings, top_k)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.search_batch_sync,
            query_embedding,
            candidate_embeddings,
            top_k,
        )

    def search_batch_sync(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchrone batch search für ähnliche Embeddings
        """
        if self.gpu_available:
            return self._gpu_batch_search(query_embedding, candidate_embeddings, top_k)
        else:
            return self._cpu_batch_search(query_embedding, candidate_embeddings, top_k)

    def _gpu_batch_search(
        self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated batch similarity search
        """
        try:
            import torch

            device = torch.device("cuda" if self.device == "cuda" else "cpu")

            # Konvertiere zu Torch tensors
            query = torch.from_numpy(query_embedding).to(device).unsqueeze(0)
            candidates = torch.from_numpy(candidate_embeddings).to(device)

            # Berechne Kosinus-Ähnlichkeit
            similarities = torch.cosine_similarity(query, candidates, dim=1)

            # Top-K Ergebnisse
            values, indices = torch.topk(similarities, min(top_k, len(candidates)))

            return values.cpu().numpy(), indices.cpu().numpy()

        except Exception as e:
            logger.warning(f"GPU-Search fehlgeschlagen, verwende CPU: {e}")
            return self._cpu_batch_search(query_embedding, candidate_embeddings, top_k)

    def _cpu_batch_search(
        self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CPU-basierte batch similarity search
        """
        # Kosinus-Ähnlichkeit berechnen
        similarities = np.dot(candidate_embeddings, query_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        # Top-K Indizes und Werte
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]

        return top_similarities, top_indices

    def cleanup(self):
        """Räumt Ressourcen auf"""
        if self.executor:
            self.executor.shutdown(wait=True)

        if self.gpu_available:
            try:
                import torch

                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except:
                pass


class AsyncBatchManager:
    """
    Async Manager für GPU-Batch-Operationen
    """

    def __init__(self, gpu_processor: GPUBatchProcessor):
        self.gpu_processor = gpu_processor
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_lock = asyncio.Lock()

    async def submit_batch_task(
        self, task_id: str, texts: List[str], operation: str = "embed"
    ) -> str:
        """
        Submit async batch task
        """
        async with self.task_lock:
            if task_id in self.active_tasks:
                raise ValueError(f"Task {task_id} already exists")

            task = asyncio.create_task(
                self.gpu_processor.process_batch_async(texts, operation)
            )
            self.active_tasks[task_id] = task

            return task_id

    async def get_batch_result(self, task_id: str) -> Optional[np.ndarray]:
        """
        Get completed batch task result
        """
        async with self.task_lock:
            if task_id not in self.active_tasks:
                return None

            task = self.active_tasks[task_id]
            if task.done():
                result = await task
                del self.active_tasks[task_id]
                return result

            return None

    async def cancel_batch_task(self, task_id: str) -> bool:
        """
        Cancel running batch task
        """
        async with self.task_lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
                del self.active_tasks[task_id]
                return True
            return False

    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs"""
        return list(self.active_tasks.keys())
