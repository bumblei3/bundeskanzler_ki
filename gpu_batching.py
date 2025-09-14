"""
GPU-Batching-System für beschleunigte Embedding-Verarbeitung
Unterstützt CUDA, ROCm und CPU-Fallback mit optimierter Batch-Verarbeitung
"""

import numpy as np
import asyncio
import threading
import time
from typing import List, Optional, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class GPUBatchProcessor:
    """
    GPU-accelerated batch processor für Embeddings
    Unterstützt CUDA, ROCm und CPU-Fallback
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = 4,
        device: str = "auto",  # "cuda", "rocm", "cpu", "auto"
        embedding_dim: int = 512,
        enable_async: bool = True
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.device = self._detect_device(device)
        self.embedding_dim = embedding_dim
        self.enable_async = enable_async

        # GPU/CPU Setup
        self.gpu_available = self._setup_gpu()
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_async else None

        # Performance tracking
        self.stats = {
            "batches_processed": 0,
            "total_embeddings": 0,
            "avg_batch_time": 0.0,
            "gpu_memory_used": 0.0,
            "cpu_fallback_count": 0
        }

        logger.info(f"GPU-Batch-Processor initialisiert: Device={self.device}, GPU={'Verfügbar' if self.gpu_available else 'Nicht verfügbar'}")

    def _detect_device(self, device: str) -> str:
        """Erkennt verfügbare GPU/CPU"""
        if device != "auto":
            return device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'hip') and torch.hip.is_available():
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
                    torch.cuda.empty_cache()
                    return True
            elif self.device == "rocm":
                if hasattr(torch, 'hip') and torch.hip.is_available():
                    # ROCm setup
                    return True

        except ImportError:
            logger.warning("PyTorch nicht verfügbar, verwende CPU-Fallback")

        self.device = "cpu"
        return False

    async def process_batch_async(
        self,
        texts: List[str],
        operation: str = "embed"  # "embed", "search", "compare"
    ) -> np.ndarray:
        """
        Async batch processing für mehrere Texte
        """
        if not self.enable_async:
            return self.process_batch_sync(texts, operation)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.process_batch_sync,
            texts,
            operation
        )

    def process_batch_sync(
        self,
        texts: List[str],
        operation: str = "embed"
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
            (self.stats["avg_batch_time"] * (self.stats["batches_processed"] - len(batches)) +
             batch_time) / self.stats["batches_processed"]
        )

        return final_result

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Teilt Texte in optimale Batches auf"""
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def _process_single_batch(
        self,
        batch: List[str],
        operation: str
    ) -> np.ndarray:
        """
        Verarbeitet einen einzelnen Batch
        """
        if self.gpu_available and operation == "embed":
            return self._gpu_embed_batch(batch)
        else:
            return self._cpu_embed_batch(batch)

    def _gpu_embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        GPU-accelerated embedding generation
        """
        try:
            import torch
            import torch.nn.functional as F

            # Simulierte GPU-Embedding-Generierung
            # In der Praxis würde hier ein echtes Modell verwendet

            device = torch.device("cuda" if self.device == "cuda" else "cpu")

            # Batch als Tensor
            batch_size = len(batch)
            noise = torch.randn(batch_size, self.embedding_dim, device=device)

            # Normalisiere (simuliert echtes Embedding)
            embeddings = F.normalize(noise, p=2, dim=1)

            # L2-Normalisierung für bessere Konsistenz
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
        top_k: int = 5
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
            top_k
        )

    def search_batch_sync(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchrone batch search für ähnliche Embeddings
        """
        if self.gpu_available:
            return self._gpu_batch_search(query_embedding, candidate_embeddings, top_k)
        else:
            return self._cpu_batch_search(query_embedding, candidate_embeddings, top_k)

    def _gpu_batch_search(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int
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
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CPU-basierte batch similarity search
        """
        # Kosinus-Ähnlichkeit berechnen
        similarities = np.dot(candidate_embeddings, query_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Top-K Indizes und Werte
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]

        return top_similarities, top_indices

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        gpu_memory = 0.0
        if self.gpu_available:
            try:
                import torch
                if self.device == "cuda":
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            except:
                pass

        return {
            **self.stats,
            "device": self.device,
            "gpu_available": self.gpu_available,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "gpu_memory_used_mb": gpu_memory,
            "async_enabled": self.enable_async
        }

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
        self,
        task_id: str,
        texts: List[str],
        operation: str = "embed"
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