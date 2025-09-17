#!/usr/bin/env python3
"""
Request Batching System für Bundeskanzler KI
Optimiert gleichzeitige Anfragen durch intelligente Batch-Verarbeitung
"""

import asyncio
import threading
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Repräsentiert eine einzelne Anfrage im Batch"""
    id: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    priority: int = 1
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchResult:
    """Ergebnis einer Batch-Verarbeitung"""
    request_id: str
    result: Any
    processing_time: float
    success: bool
    error: Optional[str] = None

class IntelligentBatchProcessor:
    """
    Intelligenter Batch-Processor für optimierte Anfrage-Verarbeitung
    Unterstützt dynamische Batch-Größen, Prioritäten und adaptive Optimierung
    """

    def __init__(self,
                 max_batch_size: int = 16,
                 max_wait_time: float = 0.1,
                 enable_adaptive_batching: bool = True,
                 enable_priority_queue: bool = True):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.enable_adaptive_batching = enable_adaptive_batching
        self.enable_priority_queue = enable_priority_queue

        # Batch-Warteschlangen
        self.request_queue: List[BatchRequest] = []
        self.processing_queue: List[BatchRequest] = []

        # Statistiken
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0,
            'avg_processing_time': 0,
            'throughput': 0,
            'last_batch_time': time.time()
        }

        # Locks für Thread-Sicherheit
        self.queue_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        # Batch-Verarbeitungs-Task
        self.batch_task: Optional[asyncio.Task] = None
        self.running = True

        # Threading und Async - nur einmal initialisieren
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async_loop, daemon=False)
        self.thread.start()

        logger.info(f"🎯 IntelligentBatchProcessor initialisiert - Max Batch: {max_batch_size}, Max Wait: {max_wait_time}s")

    def _run_async_loop(self):
        """Führt den Async-Event-Loop in einem separaten Thread"""
        try:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._batch_processing_loop())
        except Exception as e:
            logger.error(f"❌ Fehler im Async-Loop: {e}")
        finally:
            # Cleanup beim Beenden
            try:
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception as e:
                logger.warning(f"⚠️ Fehler beim Schließen des Event-Loops: {e}")

    async def _batch_processing_loop(self):
        """Haupt-Loop für Batch-Verarbeitung"""
        while self.running:
            try:
                # Sammle Anfragen für Batch
                batch = await self._collect_batch_requests()

                if batch:
                    # Verarbeite Batch
                    await self._process_batch(batch)

                # Kurze Pause um CPU zu schonen
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                # Graceful shutdown
                logger.info("🛑 Batch-Processing-Loop beendet")
                break
            except Exception as e:
                logger.error(f"❌ Fehler in Batch-Loop: {e}")
                if not self.running:
                    break
                await asyncio.sleep(0.1)

    async def _collect_batch_requests(self) -> List[BatchRequest]:
        """Sammelt Anfragen für den nächsten Batch"""
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            with self.queue_lock:
                if len(self.request_queue) >= self.max_batch_size:
                    # Genug Anfragen für vollen Batch
                    batch = self.request_queue[:self.max_batch_size]
                    self.request_queue = self.request_queue[self.max_batch_size:]
                    return batch
                elif self.request_queue and time.time() - start_time >= self.max_wait_time:
                    # Timeout erreicht, verarbeite verfügbare Anfragen
                    batch = self.request_queue.copy()
                    self.request_queue.clear()
                    return batch

            await asyncio.sleep(0.01)

        # Timeout ohne Anfragen
        return []

    async def _process_batch(self, batch: List[BatchRequest]):
        """Verarbeitet einen Batch von Anfragen"""
        if not batch:
            return

        batch_start = time.time()
        batch_size = len(batch)

        logger.info(f"🔄 Verarbeite Batch mit {batch_size} Anfragen...")

        try:
            # Batch-Verarbeitung (hier würde die eigentliche KI-Verarbeitung stattfinden)
            results = await self._execute_batch_processing(batch)

            # Ergebnisse verteilen
            processing_time = time.time() - batch_start
            await self._distribute_results(batch, results, processing_time)

            # Statistiken aktualisieren
            self._update_stats(batch_size, processing_time)

            logger.info(f"✅ Batch verarbeitet: {batch_size} Anfragen in {processing_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Batch-Verarbeitung fehlgeschlagen: {e}")
            # Fehlerhafte Ergebnisse zurückgeben
            await self._distribute_errors(batch, str(e))

    async def _execute_batch_processing(self, batch: List[BatchRequest]) -> List[Any]:
        """Führt die eigentliche Batch-Verarbeitung aus"""
        # Hier würde die Integration mit dem multimodalen KI-System stattfinden
        # Für jetzt: Simuliere Verarbeitung mit zufälligen Ergebnissen

        results = []
        for request in batch:
            # Simuliere KI-Verarbeitung
            await asyncio.sleep(0.01)  # Simulierte Verarbeitungszeit

            # Mock-Ergebnis basierend auf Anfrage-Typ
            if isinstance(request.data, dict) and 'text' in request.data:
                result = {
                    'response': f"Verarbeitet: {request.data['text'][:50]}...",
                    'confidence': np.random.uniform(0.8, 0.95),
                    'processing_method': 'batch_optimized'
                }
            else:
                result = {
                    'response': f"Batch-Ergebnis für Anfrage {request.id}",
                    'confidence': np.random.uniform(0.7, 0.9),
                    'processing_method': 'batch_optimized'
                }

            results.append(result)

        return results

    async def _distribute_results(self, batch: List[BatchRequest], results: List[Any], processing_time: float):
        """Verteilt Ergebnisse an die entsprechenden Callbacks"""
        for request, result in zip(batch, results):
            batch_result = BatchResult(
                request_id=request.id,
                result=result,
                processing_time=processing_time / len(batch),  # Durchschnittliche Zeit pro Anfrage
                success=True
            )

            # Callback aufrufen falls vorhanden
            if request.callback:
                try:
                    if asyncio.iscoroutinefunction(request.callback):
                        await request.callback(batch_result)
                    else:
                        request.callback(batch_result)
                except Exception as e:
                    logger.error(f"❌ Fehler beim Aufrufen von Callback für {request.id}: {e}")

    async def _distribute_errors(self, batch: List[BatchRequest], error: str):
        """Verteilt Fehler an alle Anfragen im Batch"""
        for request in batch:
            batch_result = BatchResult(
                request_id=request.id,
                result=None,
                processing_time=0,
                success=False,
                error=error
            )

            if request.callback:
                try:
                    if asyncio.iscoroutinefunction(request.callback):
                        await request.callback(batch_result)
                    else:
                        request.callback(batch_result)
                except Exception as e:
                    logger.error(f"❌ Fehler beim Aufrufen von Error-Callback für {request.id}: {e}")

    def _update_stats(self, batch_size: int, processing_time: float):
        """Aktualisiert Performance-Statistiken"""
        with self.stats_lock:
            self.stats['total_requests'] += batch_size
            self.stats['total_batches'] += 1

            # Gleitender Durchschnitt für Batch-Größe
            current_avg = self.stats['avg_batch_size']
            self.stats['avg_batch_size'] = (current_avg * 0.9) + (batch_size * 0.1)

            # Gleitender Durchschnitt für Verarbeitungszeit
            current_time_avg = self.stats['avg_processing_time']
            self.stats['avg_processing_time'] = (current_time_avg * 0.9) + (processing_time * 0.1)

            # Throughput berechnen (Anfragen pro Sekunde)
            time_diff = time.time() - self.stats['last_batch_time']
            if time_diff > 0:
                current_throughput = batch_size / time_diff
                prev_throughput = self.stats['throughput']
                self.stats['throughput'] = (prev_throughput * 0.9) + (current_throughput * 0.1)

            self.stats['last_batch_time'] = time.time()

    def submit_request(self, request_data: Any, priority: int = 1,
                      callback: Optional[Callable] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Fügt eine neue Anfrage zur Batch-Warteschlange hinzu"""
        request_id = f"req_{int(time.time() * 1000000)}_{np.random.randint(1000, 9999)}"

        request = BatchRequest(
            id=request_id,
            data=request_data,
            priority=priority,
            callback=callback,
            metadata=metadata or {}
        )

        with self.queue_lock:
            if self.enable_priority_queue:
                # Einfache Prioritäts-Warteschlange (höhere Priorität = niedrigerer Index)
                insert_pos = 0
                for i, existing_req in enumerate(self.request_queue):
                    if existing_req.priority < priority:
                        insert_pos = i
                        break
                    insert_pos = i + 1
                self.request_queue.insert(insert_pos, request)
            else:
                self.request_queue.append(request)

        logger.info(f"📝 Anfrage {request_id} zur Batch-Warteschlange hinzugefügt (Priorität: {priority})")
        return request_id

    async def submit_request_async(self, request_data: Any, priority: int = 1,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Async-Version von submit_request"""
        return self.submit_request(request_data, priority, None, metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Statistiken zurück"""
        with self.stats_lock:
            stats_copy = self.stats.copy()

        # Zusätzliche berechnete Metriken
        with self.queue_lock:
            queue_size = len(self.request_queue)

        stats_copy.update({
            'queue_size': queue_size,
            'adaptive_batching': self.enable_adaptive_batching,
            'priority_queue': self.enable_priority_queue,
            'max_batch_size': self.max_batch_size,
            'max_wait_time': self.max_wait_time
        })

        return stats_copy

    def optimize_batch_size(self):
        """Optimiert Batch-Größe basierend auf Performance-Daten"""
        if not self.enable_adaptive_batching:
            return

        stats = self.get_stats()

        # Adaptive Batch-Größen-Optimierung
        if stats['throughput'] > 100:  # Hoher Throughput
            self.max_batch_size = min(self.max_batch_size + 2, 32)
        elif stats['throughput'] < 50:  # Niedriger Throughput
            self.max_batch_size = max(self.max_batch_size - 1, 4)

        logger.info(f"⚡ Batch-Größe optimiert auf {self.max_batch_size} (Throughput: {stats['throughput']:.1f} req/s)")

    def shutdown(self):
        """Beendet den Batch-Processor sauber"""
        logger.info("🛑 Beende IntelligentBatchProcessor...")
        self.running = False

        # Beende alle laufenden Tasks
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()
            try:
                # Warte kurz auf das Ende der Task
                if asyncio.iscoroutinefunction(asyncio.wait_for):
                    asyncio.wait_for(self.batch_task, timeout=1.0)
            except:
                pass

        # Beende Thread-Pool
        self.executor.shutdown(wait=True)

        # Warte auf Thread-Ende
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("⚠️ Thread konnte nicht ordnungsgemäß beendet werden")

        # Schließe Event-Loop
        try:
            if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
                self.loop.close()
        except Exception as e:
            logger.warning(f"⚠️ Fehler beim Schließen des Event-Loops: {e}")

        logger.info("✅ IntelligentBatchProcessor beendet")

class RequestBatcher:
    """
    Haupt-Interface für Request Batching
    Integriert mit multimodalem KI-System
    """

    def __init__(self):
        self.batch_processor = IntelligentBatchProcessor(
            max_batch_size=16,
            max_wait_time=0.1,
            enable_adaptive_batching=True,
            enable_priority_queue=True
        )

        # Verschiedene Batch-Processor für verschiedene Anfrage-Typen
        self.text_processor = IntelligentBatchProcessor(max_batch_size=8, max_wait_time=0.05)
        self.embedding_processor = IntelligentBatchProcessor(max_batch_size=32, max_wait_time=0.2)
        self.search_processor = IntelligentBatchProcessor(max_batch_size=12, max_wait_time=0.08)

        logger.info("🚀 RequestBatcher initialisiert mit spezialisierten Prozessoren")

    def submit_text_request(self, text: str, priority: int = 1,
                           callback: Optional[Callable] = None) -> str:
        """Fügt eine Text-Verarbeitungsanfrage hinzu"""
        request_data = {'type': 'text', 'text': text}
        return self.text_processor.submit_request(request_data, priority, callback)

    def submit_embedding_request(self, texts: List[str], priority: int = 1,
                                callback: Optional[Callable] = None) -> str:
        """Fügt eine Embedding-Anfrage hinzu"""
        request_data = {'type': 'embedding', 'texts': texts}
        return self.embedding_processor.submit_request(request_data, priority, callback)

    def submit_search_request(self, query: str, context: Optional[List[str]] = None,
                             priority: int = 1, callback: Optional[Callable] = None) -> str:
        """Fügt eine Suchanfrage hinzu"""
        request_data = {'type': 'search', 'query': query, 'context': context or []}
        return self.search_processor.submit_request(request_data, priority, callback)

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken aller Batch-Processor zurück"""
        return {
            'text_processor': self.text_processor.get_stats(),
            'embedding_processor': self.embedding_processor.get_stats(),
            'search_processor': self.search_processor.get_stats(),
            'overall': {
                'total_requests': sum(p.get_stats()['total_requests'] for p in
                                    [self.text_processor, self.embedding_processor, self.search_processor]),
                'total_batches': sum(p.get_stats()['total_batches'] for p in
                                   [self.text_processor, self.embedding_processor, self.search_processor]),
                'avg_throughput': sum(p.get_stats()['throughput'] for p in
                                     [self.text_processor, self.embedding_processor, self.search_processor]) / 3
            }
        }

    def optimize_all(self):
        """Optimiert alle Batch-Processor"""
        self.text_processor.optimize_batch_size()
        self.embedding_processor.optimize_batch_size()
        self.search_processor.optimize_batch_size()
        logger.info("⚡ Alle Batch-Processor optimiert")

    def shutdown(self):
        """Beendet alle Batch-Processor"""
        logger.info("🛑 Beende RequestBatcher...")
        self.text_processor.shutdown()
        self.embedding_processor.shutdown()
        self.search_processor.shutdown()
        logger.info("✅ RequestBatcher beendet")

import atexit

# ... existing code ...

# Globale Instanz für einfachen Zugriff
request_batcher = RequestBatcher()

def get_request_batcher() -> RequestBatcher:
    """Gibt die globale RequestBatcher-Instanz zurück"""
    return request_batcher

# Registriere Cleanup beim Programmende
def _cleanup_request_batcher():
    """Cleanup für RequestBatcher beim Programmende"""
    try:
        # Prüfe ob bereits Graceful Shutdown durchgeführt wurde
        try:
            from graceful_shutdown import is_graceful_shutdown_completed
            if is_graceful_shutdown_completed():
                logger.debug("🧹 Graceful Shutdown bereits durchgeführt - überspringe RequestBatcher Cleanup")
                return
        except ImportError:
            pass  # Graceful Shutdown nicht verfügbar

        logger.info("🛑 Cleanup RequestBatcher beim Programmende...")
        request_batcher.shutdown()
        logger.info("✅ RequestBatcher Cleanup abgeschlossen")

    except Exception as e:
        logger.warning(f"⚠️ Fehler beim RequestBatcher Cleanup: {e}")

atexit.register(_cleanup_request_batcher)