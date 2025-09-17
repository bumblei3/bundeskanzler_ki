#!/usr/bin/env python3
"""
Dynamic Batching System für Bundeskanzler KI
Optimiert GPU-Auslastung durch intelligentes Request-Batching
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Einzelne Request im Batch"""
    id: str
    data: Any
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable[[Any], Awaitable[None]]] = None


@dataclass
class BatchConfig:
    """Konfiguration für Dynamic Batching"""
    max_batch_size: int = 8  # Maximale Batch-Größe für RTX 2070
    min_batch_size: int = 1  # Minimale Batch-Größe
    max_wait_time: float = 0.1  # Maximale Wartezeit in Sekunden
    gpu_memory_threshold: float = 0.8  # GPU Memory Threshold
    enable_adaptive_batching: bool = True  # Adaptives Batching aktivieren


@dataclass
class BatchMetrics:
    """Performance-Metriken für Batching"""
    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0
    gpu_utilization: float = 0.0
    throughput_requests_per_sec: float = 0.0


class DynamicBatchProcessor:
    """
    Intelligenter Batch-Processor für optimale GPU-Auslastung
    Sammelt Requests und verarbeitet sie in optimalen Batches
    """

    def __init__(self, config: Optional[BatchConfig] = None, gpu_memory_gb: float = 8.0):
        self.config = config or BatchConfig()
        self.gpu_memory_gb = gpu_memory_gb

        # RTX 2070 spezifische Optimierungen
        self._optimize_for_rtx2070()

        # Batch-Management
        self.pending_requests: List[BatchRequest] = []
        self.processing_batches: Dict[str, List[BatchRequest]] = {}
        self.batch_metrics = BatchMetrics()

        # Threading und Async
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        self.running = True

        # Performance-Tracking
        self.request_times: List[float] = []
        self.batch_sizes: List[int] = []

        # Starte Background-Processing
        self._start_background_processing()

        logger.info(f"🚀 DynamicBatchProcessor initialisiert - GPU: {gpu_memory_gb:.1f}GB, Max Batch Size: {self.config.max_batch_size}")

    def _start_background_processing(self):
        """Startet Hintergrund-Verarbeitung für Batches"""
        asyncio.create_task(self._background_batch_processor())

    async def _background_batch_processor(self):
        """Hintergrund-Task für kontinuierliche Batch-Verarbeitung"""
        while self.running:
            try:
                await self._check_and_process_batch()
                await asyncio.sleep(0.01)  # Kleine Pause zwischen Checks
            except Exception as e:
                logger.error(f"Fehler im Background-Processor: {e}")
                await asyncio.sleep(0.1)

    def _optimize_for_rtx2070(self):
        """RTX 2070 spezifische Batch-Optimierungen"""
        if self.gpu_memory_gb >= 11.0:  # RTX 3080/3090
            self.config.max_batch_size = 16
            self.config.max_wait_time = 0.03
        elif self.gpu_memory_gb >= 7.5:  # RTX 2070 / ähnliche
            # Optimale Batch-Größen für RTX 2070
            self.config.max_batch_size = 8  # 8 Requests parallel
            self.config.max_wait_time = 0.05  # Schnellere Verarbeitung
            self.config.gpu_memory_threshold = 0.85  # Höherer Memory Threshold
        else:  # Kleinere GPUs
            self.config.max_batch_size = 4
            self.config.max_wait_time = 0.1

    async def submit_request(self, request: BatchRequest) -> str:
        """
        Submitte eine neue Request für Batching

        Args:
            request: Die zu verarbeitende Request

        Returns:
            Batch-ID für Tracking
        """
        with self.lock:
            self.pending_requests.append(request)
            self.batch_metrics.total_requests += 1

        # Trigger Batch-Verarbeitung wenn nötig
        await self._check_and_process_batch()

        return f"batch_{int(time.time() * 1000)}"

    async def _check_and_process_batch(self):
        """Prüft, ob ein Batch verarbeitet werden sollte"""
        current_time = time.time()

        with self.lock:
            if not self.pending_requests:
                return

            # Sammle alle verfügbaren Requests
            all_requests = self.pending_requests.copy()
            self.pending_requests.clear()

            # Prüfe, ob wir genug Requests für einen Batch haben
            if len(all_requests) >= self.config.min_batch_size:
                batch_id = f"batch_{int(current_time * 1000000)}"
                self.processing_batches[batch_id] = all_requests

                # Starte Batch-Verarbeitung
                asyncio.create_task(self._process_batch(batch_id, all_requests))
            else:
                # Zu wenige Requests - zurück in die Queue
                self.pending_requests.extend(all_requests)

    async def _process_batch(self, batch_id: str, requests: List[BatchRequest]):
        """Verarbeitet einen Batch von Requests"""
        start_time = time.time()

        try:
            logger.info(f"🔄 Verarbeite Batch {batch_id} mit {len(requests)} Requests")

            # Simuliere Batch-Verarbeitung (hier würde die eigentliche KI-Verarbeitung stattfinden)
            batch_data = [req.data for req in requests]

            # GPU-optimierte Verarbeitung
            results = await self._execute_batch_on_gpu(batch_data)

            # Ergebnisse zurückgeben
            for req, result in zip(requests, results):
                if req.callback:
                    await req.callback(result)

            processing_time = time.time() - start_time

            # Metriken aktualisieren
            with self.lock:
                self.batch_metrics.total_batches += 1
                self.batch_metrics.avg_batch_size = (
                    (self.batch_metrics.avg_batch_size * (self.batch_metrics.total_batches - 1)) +
                    len(requests)
                ) / self.batch_metrics.total_batches

                self.batch_metrics.avg_processing_time = (
                    (self.batch_metrics.avg_processing_time * (self.batch_metrics.total_batches - 1)) +
                    processing_time
                ) / self.batch_metrics.total_batches

                self.batch_sizes.append(len(requests))
                self.request_times.extend([processing_time / len(requests)] * len(requests))

            logger.info(f"✅ Batch {batch_id} verarbeitet - {len(requests)} Requests in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Fehler bei Batch-Verarbeitung {batch_id}: {e}")
        finally:
            # Batch aus Processing entfernen
            with self.lock:
                if batch_id in self.processing_batches:
                    del self.processing_batches[batch_id]

    async def _execute_batch_on_gpu(self, batch_data: List[Any]) -> List[Any]:
        """
        Führt Batch-Verarbeitung auf GPU aus
        Hier würde die Integration mit dem KI-Modell stattfinden
        """
        # Simuliere GPU-Verarbeitung
        await asyncio.sleep(0.01)  # Simulierte Verarbeitungszeit

        # Mock-Ergebnisse zurückgeben
        return [f"processed_{i}" for i in range(len(batch_data))]

    def get_batch_metrics(self) -> Dict[str, Any]:
        """Gibt aktuelle Batch-Metriken zurück"""
        with self.lock:
            if self.request_times:
                avg_request_time = sum(self.request_times) / len(self.request_times)
                self.batch_metrics.throughput_requests_per_sec = 1.0 / avg_request_time if avg_request_time > 0 else 0.0

            return {
                "total_requests": self.batch_metrics.total_requests,
                "total_batches": self.batch_metrics.total_batches,
                "avg_batch_size": round(self.batch_metrics.avg_batch_size, 2),
                "avg_processing_time": round(self.batch_metrics.avg_processing_time, 3),
                "throughput_requests_per_sec": round(self.batch_metrics.throughput_requests_per_sec, 2),
                "pending_requests": len(self.pending_requests),
                "processing_batches": len(self.processing_batches),
                "gpu_memory_gb": self.gpu_memory_gb,
                "max_batch_size": self.config.max_batch_size,
            }

    async def shutdown(self):
        """Beendet den Batch-Processor sauber"""
        logger.info("🛑 Shutting down DynamicBatchProcessor...")

        self.running = False

        # Warte auf laufende Batches
        while self.processing_batches:
            await asyncio.sleep(0.1)

        # Verarbeite verbleibende Requests
        if self.pending_requests:
            logger.info(f"🔄 Verarbeite {len(self.pending_requests)} verbleibende Requests...")
            batch_id = f"final_batch_{int(time.time() * 1000000)}"
            await self._process_batch(batch_id, self.pending_requests.copy())
            self.pending_requests.clear()

        self.executor.shutdown(wait=True)
        logger.info("✅ DynamicBatchProcessor beendet")


# Globale Instanz für einfachen Zugriff
dynamic_batch_processor = DynamicBatchProcessor()