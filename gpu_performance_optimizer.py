#!/usr/bin/env python3
"""
🚀 GPU Performance Optimizer
===========================

GPU-spezifische Performance-Optimierungen für die Bundeskanzler KI.

Features:
- GPU Memory Caching
- Query Batch Processing
- Memory Pool Management
- Performance Monitoring

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import torch
import time
import threading
import psutil
import GPUtil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class GPUStats:
    """GPU Performance Statistiken"""
    memory_used: float
    memory_total: float
    memory_util: float
    gpu_load: float
    temperature: float
    timestamp: float

@dataclass
class QueryBatch:
    """Batch von Queries für optimierte Verarbeitung"""
    queries: List[str]
    languages: List[str]
    batch_id: str
    created_at: float

class GPUPerformanceOptimizer:
    """GPU Performance Optimizer"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_cache = {}
        self.query_cache = {}
        self.batch_queue = []
        self.stats_history = []

        # Performance Einstellungen
        self.max_batch_size = 4
        self.cache_ttl = 300  # 5 Minuten
        self.memory_threshold = 0.8  # 80% Memory Limit

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.monitoring_active = False

        # Logging
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Starte GPU-Monitoring"""
        self.monitoring_active = True
        self.executor.submit(self._gpu_monitor)

    def stop_monitoring(self):
        """Stoppe GPU-Monitoring"""
        self.monitoring_active = False
        self.executor.shutdown(wait=True)

    def _gpu_monitor(self):
        """GPU-Monitoring Thread"""
        while self.monitoring_active:
            try:
                stats = self.get_gpu_stats()
                self.stats_history.append(stats)

                # Memory Management
                if stats.memory_util > self.memory_threshold:
                    self._cleanup_memory()

                # Keep only last 100 entries
                if len(self.stats_history) > 100:
                    self.stats_history = self.stats_history[-100:]

                time.sleep(5)  # Update every 5 seconds

            except Exception as e:
                self.logger.error(f"GPU Monitoring error: {e}")
                time.sleep(10)

    def get_gpu_stats(self) -> GPUStats:
        """Hole aktuelle GPU-Statistiken"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return GPUStats(
                    memory_used=gpu.memoryUsed,
                    memory_total=gpu.memoryTotal,
                    memory_util=gpu.memoryUtil,
                    gpu_load=gpu.load,
                    temperature=gpu.temperature,
                    timestamp=time.time()
                )
        except Exception as e:
            self.logger.error(f"GPU stats error: {e}")

        # Fallback
        return GPUStats(0, 0, 0, 0, 0, time.time())

    def optimize_query(self, query: str, language: str = "de") -> Dict[str, Any]:
        """Optimiere Query für GPU-Verarbeitung"""
        start_time = time.time()

        # Cache-Check
        cache_key = f"{query}_{language}"
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.cache_ttl:
                return {
                    'query': query,
                    'optimized': True,
                    'cached': True,
                    'processing_time': time.time() - start_time
                }

        # Query-Optimierung
        optimized_query = self._preprocess_query(query, language)

        # GPU Memory vorbereiten
        self._prepare_gpu_memory()

        result = {
            'query': optimized_query,
            'optimized': True,
            'cached': False,
            'processing_time': time.time() - start_time,
            'gpu_memory_prepared': True
        }

        # Cache speichern
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

        return result

    def _preprocess_query(self, query: str, language: str) -> str:
        """Preprocess Query für bessere GPU-Performance"""
        # Entferne unnötige Whitespaces
        query = ' '.join(query.split())

        # Language-spezifische Optimierungen
        if language == "de":
            # Deutsche Stopwords entfernen
            german_stopwords = ['der', 'die', 'das', 'und', 'oder', 'aber', 'weil']
            words = query.split()
            filtered_words = [w for w in words if w.lower() not in german_stopwords]
            query = ' '.join(filtered_words)

        return query

    def _prepare_gpu_memory(self):
        """Bereite GPU Memory für optimale Performance vor"""
        if torch.cuda.is_available():
            try:
                # GPU Memory Cache leeren wenn nötig
                if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.7:
                    torch.cuda.empty_cache()

                # Memory Pool optimieren
                torch.cuda.synchronize()

            except Exception as e:
                self.logger.warning(f"GPU Memory preparation failed: {e}")

    def create_batch(self, queries: List[str], languages: List[str]) -> QueryBatch:
        """Erstelle optimierten Query-Batch"""
        batch_id = f"batch_{int(time.time())}_{len(queries)}"

        batch = QueryBatch(
            queries=queries,
            languages=languages,
            batch_id=batch_id,
            created_at=time.time()
        )

        self.batch_queue.append(batch)
        return batch

    def process_batch(self, batch: QueryBatch) -> List[Dict[str, Any]]:
        """Verarbeite Query-Batch für optimale GPU-Nutzung"""
        results = []

        # Batch-Größe anpassen
        batch_size = min(len(batch.queries), self.max_batch_size)

        for i in range(0, len(batch.queries), batch_size):
            batch_queries = batch.queries[i:i+batch_size]
            batch_languages = batch.languages[i:i+batch_size]

            # Parallele Verarbeitung
            batch_results = []
            for query, language in zip(batch_queries, batch_languages):
                result = self.optimize_query(query, language)
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def _cleanup_memory(self):
        """Memory Cleanup bei hoher Auslastung"""
        try:
            # Alte Cache-Einträge entfernen
            current_time = time.time()
            expired_keys = [
                key for key, value in self.query_cache.items()
                if current_time - value['timestamp'] > self.cache_ttl
            ]

            for key in expired_keys:
                del self.query_cache[key]

            # GPU Memory freigeben
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"Memory cleanup: {len(expired_keys)} cache entries removed")

        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generiere Performance-Report"""
        if not self.stats_history:
            return {"error": "Keine Performance-Daten verfügbar"}

        latest_stats = self.stats_history[-1]

        # Berechne Durchschnittswerte
        avg_memory_util = sum(s.memory_util for s in self.stats_history) / len(self.stats_history)
        avg_gpu_load = sum(s.gpu_load for s in self.stats_history) / len(self.stats_history)
        avg_temp = sum(s.temperature for s in self.stats_history) / len(self.stats_history)

        return {
            "current": {
                "gpu_memory_used": f"{latest_stats.memory_used:.0f}MB",
                "gpu_memory_total": f"{latest_stats.memory_total:.0f}MB",
                "gpu_utilization": f"{latest_stats.memory_util*100:.1f}%",
                "gpu_load": f"{latest_stats.gpu_load*100:.1f}%",
                "temperature": f"{latest_stats.temperature:.1f}°C"
            },
            "average": {
                "memory_utilization": f"{avg_memory_util*100:.1f}%",
                "gpu_load": f"{avg_gpu_load*100:.1f}%",
                "temperature": f"{avg_temp:.1f}°C"
            },
            "cache_stats": {
                "query_cache_size": len(self.query_cache),
                "memory_cache_size": len(self.memory_cache),
                "batch_queue_size": len(self.batch_queue)
            },
            "optimization_active": True
        }

# Globale Instanz
gpu_optimizer = GPUPerformanceOptimizer()

if __name__ == "__main__":
    print("🚀 GPU Performance Optimizer")
    print("=" * 40)

    # Test Performance Optimizer
    optimizer = GPUPerformanceOptimizer()
    optimizer.start_monitoring()

    # Warte kurz für Monitoring
    time.sleep(2)

    # Test Query-Optimierung
    test_queries = [
        "Was ist die aktuelle Klimapolitik Deutschlands?",
        "Wie steht es um die Wirtschaftslage?",
        "Welche Reformen sind geplant?"
    ]

    print("\n🔍 Teste Query-Optimierung:")
    for query in test_queries:
        result = optimizer.optimize_query(query)
        print(f"✅ {query[:30]}... -> {result['processing_time']:.3f}s")

    # Performance Report
    time.sleep(3)
    report = optimizer.get_performance_report()
    print("\n📊 Performance Report:")
    print(f"  GPU Memory: {report['current']['gpu_memory_used']} / {report['current']['gpu_memory_total']}")
    print(f"  GPU Load: {report['current']['gpu_load']}")
    print(f"  Cache Size: {report['cache_stats']['query_cache_size']} queries")

    optimizer.stop_monitoring()
    print("\n✅ GPU Performance Optimizer bereit!")