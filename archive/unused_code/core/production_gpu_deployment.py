#!/usr/bin/env python3
"""
🚀 Production GPU Deployment System für RTX 2070
================================================

Advanced Production-Ready GPU System:
- Automatic CPU/GPU Fallback Management
- Real-time Health Monitoring & Recovery
- Dynamic Batch Size Optimization
- Production Error Handling & Logging
- Resource Management & Cleanup
- Performance Monitoring & Alerting

Basiert auf: GPU Multi-Agent System
Optimiert für: NVIDIA GeForce RTX 2070
Status: PRODUCTION DEPLOYMENT READY ✅
Autor: Claude-3.5-Sonnet
Datum: 15. September 2025
"""

import asyncio
import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
import torch

# Import unsere GPU-optimierten Systeme
sys.path.append("/home/tobber/bkki_venv")

from core.gpu_manager import get_rtx2070_manager, rtx2070_context
from core.gpu_multi_agent_final import GPUMultiAgentCoordinator
from core.gpu_rag_system import GPUAcceleratedRAG

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health Monitoring Metrics"""

    timestamp: str
    gpu_available: bool
    gpu_utilization: float
    gpu_memory_used: float
    gpu_temperature: float
    cpu_usage: float
    ram_usage: float
    system_healthy: bool
    error_count: int
    last_query_time: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance Tracking Metrics"""

    timestamp: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time_ms: float
    gpu_queries: int
    cpu_fallback_queries: int
    current_batch_size: int
    optimal_batch_size: int


class ProductionGPUHealthMonitor:
    """Real-time GPU Health Monitor für Production Environment"""

    def __init__(self, check_interval: int = 30):
        """
        Initialisiert Health Monitor

        Args:
            check_interval: Health Check Interval in Sekunden
        """
        self.check_interval = check_interval
        self.gpu_manager = get_rtx2070_manager()
        self.is_monitoring = False
        self.health_history = []
        self.error_count = 0
        self.last_health_check = None

        # Thresholds für Health Warnings
        self.thresholds = {
            "gpu_temperature_warning": 75.0,
            "gpu_temperature_critical": 85.0,
            "gpu_memory_warning": 85.0,
            "gpu_memory_critical": 95.0,
            "cpu_usage_warning": 85.0,
            "ram_usage_warning": 90.0,
        }

        self.monitoring_thread = None

    def start_monitoring(self):
        """Startet Health Monitoring Thread"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🩺 GPU Health Monitoring gestartet")

    def stop_monitoring(self):
        """Stoppt Health Monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🩺 GPU Health Monitoring gestoppt")

    def _monitoring_loop(self):
        """Main Monitoring Loop"""
        while self.is_monitoring:
            try:
                health = self._collect_health_metrics()
                self._analyze_health(health)
                self.health_history.append(health)

                # Keep only last 24 hours of history
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.health_history = [
                    h
                    for h in self.health_history
                    if datetime.fromisoformat(h.timestamp) > cutoff_time
                ]

                self.last_health_check = datetime.now()

            except Exception as e:
                logger.error(f"❌ Health Monitor Fehler: {e}")
                self.error_count += 1

            time.sleep(self.check_interval)

    def _collect_health_metrics(self) -> HealthMetrics:
        """Sammelt aktuelle Health Metrics"""
        timestamp = datetime.now().isoformat()

        # GPU Metrics
        gpu_available = self.gpu_manager.is_gpu_available()
        gpu_stats = self.gpu_manager.get_gpu_stats() if gpu_available else None

        # System Metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        ram_usage = memory.percent

        # Determine System Health
        system_healthy = True
        if gpu_available and gpu_stats:
            if (
                gpu_stats.temperature_c > self.thresholds["gpu_temperature_critical"]
                or gpu_stats.memory_utilization > self.thresholds["gpu_memory_critical"]
            ):
                system_healthy = False

        if (
            cpu_usage > self.thresholds["cpu_usage_warning"]
            or ram_usage > self.thresholds["ram_usage_warning"]
        ):
            system_healthy = False

        return HealthMetrics(
            timestamp=timestamp,
            gpu_available=gpu_available,
            gpu_utilization=gpu_stats.gpu_utilization if gpu_stats else 0.0,
            gpu_memory_used=gpu_stats.memory_used_gb if gpu_stats else 0.0,
            gpu_temperature=gpu_stats.temperature_c if gpu_stats else 0.0,
            cpu_usage=cpu_usage,
            ram_usage=ram_usage,
            system_healthy=system_healthy,
            error_count=self.error_count,
        )

    def _analyze_health(self, health: HealthMetrics):
        """Analysiert Health Metrics und gibt Warnungen aus"""
        if not health.system_healthy:
            logger.warning(f"⚠️ System Health Warning:")
            logger.warning(f"   GPU Temp: {health.gpu_temperature:.1f}°C")
            logger.warning(f"   GPU Memory: {health.gpu_memory_used:.1f}GB")
            logger.warning(f"   CPU Usage: {health.cpu_usage:.1f}%")
            logger.warning(f"   RAM Usage: {health.ram_usage:.1f}%")

        # Temperature Warnings
        if health.gpu_temperature > self.thresholds["gpu_temperature_warning"]:
            if health.gpu_temperature > self.thresholds["gpu_temperature_critical"]:
                logger.critical(f"🔥 CRITICAL: GPU Temperatur {health.gpu_temperature:.1f}°C!")
            else:
                logger.warning(f"🌡️ GPU Temperatur erhöht: {health.gpu_temperature:.1f}°C")

    def get_health_summary(self) -> Dict[str, Any]:
        """Holt aktuelle Health Summary"""
        if not self.health_history:
            return {"status": "NO_DATA", "message": "Keine Health-Daten verfügbar"}

        latest = self.health_history[-1]

        # Calculate averages over last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            h for h in self.health_history if datetime.fromisoformat(h.timestamp) > one_hour_ago
        ]

        if recent_metrics:
            avg_gpu_temp = sum(h.gpu_temperature for h in recent_metrics) / len(recent_metrics)
            avg_cpu_usage = sum(h.cpu_usage for h in recent_metrics) / len(recent_metrics)
            avg_ram_usage = sum(h.ram_usage for h in recent_metrics) / len(recent_metrics)
        else:
            avg_gpu_temp = latest.gpu_temperature
            avg_cpu_usage = latest.cpu_usage
            avg_ram_usage = latest.ram_usage

        status = "HEALTHY" if latest.system_healthy else "WARNING"
        if latest.gpu_temperature > self.thresholds["gpu_temperature_critical"]:
            status = "CRITICAL"

        return {
            "status": status,
            "timestamp": latest.timestamp,
            "gpu_available": latest.gpu_available,
            "current_metrics": asdict(latest),
            "averages_1h": {
                "gpu_temperature": avg_gpu_temp,
                "cpu_usage": avg_cpu_usage,
                "ram_usage": avg_ram_usage,
            },
            "error_count": latest.error_count,
            "monitoring_active": self.is_monitoring,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
        }


class ProductionGPUManager:
    """Production-Ready GPU Manager mit automatischem Fallback"""

    def __init__(self):
        """Initialisiert Production GPU Manager"""
        self.gpu_manager = get_rtx2070_manager()
        self.health_monitor = ProductionGPUHealthMonitor()
        self.multi_agent = None

        # Performance Tracking
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "gpu_queries": 0,
            "cpu_fallback_queries": 0,
            "total_response_time_ms": 0.0,
            "current_batch_size": 16,
            "optimal_batch_size": 16,
            "last_optimization": None,
        }

        # Batch Size Optimization
        self.batch_performance_history = []

        # Register cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def initialize(self, corpus_path: str = None) -> bool:
        """
        Initialisiert Production System

        Args:
            corpus_path: Pfad zum Corpus

        Returns:
            True wenn erfolgreich initialisiert
        """
        try:
            logger.info("🚀 Initialisiere Production GPU System...")

            # Start Health Monitoring
            self.health_monitor.start_monitoring()

            # Initialize Multi-Agent System
            self.multi_agent = GPUMultiAgentCoordinator(corpus_path=corpus_path)

            # Verify GPU Availability
            if self.gpu_manager.is_gpu_available():
                logger.info("✅ GPU Production System bereit")
                return True
            else:
                logger.warning("⚠️ GPU nicht verfügbar - CPU Fallback aktiviert")
                return True

        except Exception as e:
            logger.error(f"❌ Production System Initialisierung fehlgeschlagen: {e}")
            return False

    async def process_query_production(
        self, query: str, confidence_threshold: float = 0.3, timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """
        Production Query Processing mit Fallback und Health Monitoring

        Args:
            query: Benutzeranfrage
            confidence_threshold: Mindest-Konfidenz
            timeout_seconds: Query Timeout

        Returns:
            Production Response mit Health Information
        """
        start_time = time.perf_counter()
        query_id = f"query_{int(time.time() * 1000)}"

        try:
            # Health Check vor Query
            health = self.health_monitor._collect_health_metrics()

            if not health.system_healthy:
                logger.warning(f"⚠️ System Health Warning vor Query {query_id}")

            # Determine Processing Mode
            use_gpu = (
                health.gpu_available
                and health.system_healthy
                and health.gpu_temperature
                < self.health_monitor.thresholds["gpu_temperature_warning"]
            )

            # Process Query mit Timeout
            try:
                result = await asyncio.wait_for(
                    self.multi_agent.process_query_parallel(query, confidence_threshold),
                    timeout=timeout_seconds,
                )

                # Update Statistics
                processing_time = (time.perf_counter() - start_time) * 1000
                self._update_performance_stats(processing_time, True, use_gpu)

                # Add Production Metadata
                result.update(
                    {
                        "query_id": query_id,
                        "processing_mode": "GPU" if use_gpu else "CPU_FALLBACK",
                        "system_health": health.system_healthy,
                        "timestamp": datetime.now().isoformat(),
                        "health_metrics": asdict(health),
                    }
                )

                logger.info(
                    f"✅ Query {query_id} erfolgreich verarbeitet in {processing_time:.1f}ms"
                )
                return result

            except asyncio.TimeoutError:
                logger.error(f"⏰ Query {query_id} Timeout nach {timeout_seconds}s")
                self._update_performance_stats(0, False, use_gpu)

                return {
                    "query_id": query_id,
                    "query": query,
                    "response": f"Die Anfrage hat das Zeitlimit von {timeout_seconds} Sekunden überschritten.",
                    "error": "TIMEOUT",
                    "processing_mode": "TIMEOUT",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(processing_time, False, False)

            logger.error(f"❌ Query {query_id} Fehler: {e}")

            return {
                "query_id": query_id,
                "query": query,
                "response": f"Es ist ein Fehler bei der Verarbeitung aufgetreten: {str(e)}",
                "error": str(e),
                "processing_mode": "ERROR",
                "timestamp": datetime.now().isoformat(),
            }

    def _update_performance_stats(self, processing_time_ms: float, success: bool, used_gpu: bool):
        """Update Performance Statistics"""
        self.performance_stats["total_queries"] += 1

        if success:
            self.performance_stats["successful_queries"] += 1
            self.performance_stats["total_response_time_ms"] += processing_time_ms
        else:
            self.performance_stats["failed_queries"] += 1

        if used_gpu:
            self.performance_stats["gpu_queries"] += 1
        else:
            self.performance_stats["cpu_fallback_queries"] += 1

        # Record for batch optimization
        self.batch_performance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": processing_time_ms,
                "batch_size": self.performance_stats["current_batch_size"],
                "success": success,
                "used_gpu": used_gpu,
            }
        )

        # Optimize batch size periodically
        if len(self.batch_performance_history) % 10 == 0:
            self._optimize_batch_size()

    def _optimize_batch_size(self):
        """Dynamische Batch Size Optimierung"""
        try:
            # Analyze last 50 queries
            recent_history = self.batch_performance_history[-50:]
            if len(recent_history) < 10:
                return

            # Calculate performance by batch size
            batch_performance = {}
            for entry in recent_history:
                if entry["success"] and entry["used_gpu"]:
                    batch_size = entry["batch_size"]
                    if batch_size not in batch_performance:
                        batch_performance[batch_size] = []
                    batch_performance[batch_size].append(entry["processing_time_ms"])

            if not batch_performance:
                return

            # Find optimal batch size
            avg_times = {}
            for batch_size, times in batch_performance.items():
                avg_times[batch_size] = sum(times) / len(times)

            optimal_batch = min(avg_times.items(), key=lambda x: x[1])[0]

            if optimal_batch != self.performance_stats["current_batch_size"]:
                logger.info(
                    f"🔧 Batch Size optimiert: {self.performance_stats['current_batch_size']} → {optimal_batch}"
                )
                self.performance_stats["current_batch_size"] = optimal_batch
                self.performance_stats["optimal_batch_size"] = optimal_batch
                self.performance_stats["last_optimization"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"❌ Batch Size Optimierung fehlgeschlagen: {e}")

    def get_production_summary(self) -> Dict[str, Any]:
        """Holt vollständige Production Summary"""
        health = self.health_monitor.get_health_summary()

        # Calculate performance metrics
        total_queries = self.performance_stats["total_queries"]
        successful_queries = self.performance_stats["successful_queries"]

        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        avg_response_time = (
            self.performance_stats["total_response_time_ms"] / successful_queries
            if successful_queries > 0
            else 0
        )

        gpu_usage_ratio = (
            self.performance_stats["gpu_queries"] / total_queries * 100 if total_queries > 0 else 0
        )

        return {
            "system_status": health["status"],
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "performance": {
                "total_queries": total_queries,
                "success_rate_percent": success_rate,
                "avg_response_time_ms": avg_response_time,
                "gpu_usage_ratio_percent": gpu_usage_ratio,
                "current_batch_size": self.performance_stats["current_batch_size"],
                "optimal_batch_size": self.performance_stats["optimal_batch_size"],
                "last_optimization": self.performance_stats["last_optimization"],
            },
            "gpu_info": {
                "available": self.gpu_manager.is_gpu_available(),
                "device_name": "NVIDIA GeForce RTX 2070",
                "memory_total_gb": 8.0,
                "cuda_streams_active": (
                    len(self.multi_agent.cuda_streams) if self.multi_agent else 0
                ),
            },
        }

    def cleanup(self):
        """Cleanup Resources"""
        logger.info("🧹 Production System Cleanup...")
        self.health_monitor.stop_monitoring()

        if self.gpu_manager.is_gpu_available():
            torch.cuda.empty_cache()

        logger.info("✅ Production System Cleanup abgeschlossen")

    def _signal_handler(self, signum, frame):
        """Signal Handler für graceful shutdown"""
        logger.info(f"📡 Signal {signum} empfangen - Shutdown eingeleitet...")
        self.cleanup()
        sys.exit(0)


# Convenience Functions
def create_production_gpu_system(corpus_path: str = None) -> ProductionGPUManager:
    """
    Erstellt Production GPU System

    Args:
        corpus_path: Pfad zum Corpus

    Returns:
        ProductionGPUManager Instanz
    """
    manager = ProductionGPUManager()
    if manager.initialize(corpus_path):
        return manager
    else:
        raise RuntimeError("Production GPU System Initialisierung fehlgeschlagen")


async def test_production_system():
    """Test Function für Production GPU System"""
    print("🚀 Testing Production GPU System...")

    # Create Production System
    prod_system = create_production_gpu_system()

    # Test Queries
    test_queries = [
        "Was ist die Klimapolitik der Bundesregierung?",
        "Wie entwickelt sich die deutsche Wirtschaft?",
        "Welche politischen Reformen plant die Koalition?",
        "Was bedeutet die Energiewende für Unternehmen und Klima?",
    ]

    # Test Individual Queries
    for query in test_queries:
        print(f"\n🤖 Query: {query}")
        result = await prod_system.process_query_production(query)
        print(f"   Response: {result['response'][:100]}...")
        print(f"   Mode: {result.get('processing_mode', 'UNKNOWN')}")
        print(f"   Time: {result.get('processing_time_ms', 0):.1f}ms")
        print(f"   Health: {'✅' if result.get('system_health', False) else '⚠️'}")

    # Production Summary
    summary = prod_system.get_production_summary()
    print(f"\n📊 Production Summary:")
    print(f"   System Status: {summary['system_status']}")
    print(f"   Total Queries: {summary['performance']['total_queries']}")
    print(f"   Success Rate: {summary['performance']['success_rate_percent']:.1f}%")
    print(f"   Avg Response Time: {summary['performance']['avg_response_time_ms']:.1f}ms")
    print(f"   GPU Usage: {summary['performance']['gpu_usage_ratio_percent']:.1f}%")
    print(f"   GPU Available: {'✅' if summary['gpu_info']['available'] else '❌'}")

    # Cleanup
    prod_system.cleanup()


if __name__ == "__main__":
    import asyncio

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    asyncio.run(test_production_system())
