#!/usr/bin/env python3
"""
Lokales Monitoring-System für Bundeskanzler-KI
API-FREI: Keine externen Abhängigkeiten, lokale Metriken nur

Features:
- GPU-Nutzung Live-Tracking
- Modell-Performance-Metriken
- System-Health Dashboard
- Lokale Logs & Analytics
- RTX 2070 spezifische Optimierungen

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import json
import logging
import os
import psutil
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU-Metriken für RTX 2070"""
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature_celsius: float
    timestamp: str


@dataclass
class ModelMetrics:
    """Modell-Performance-Metriken"""
    model_name: str
    inference_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    memory_peak_mb: float
    timestamp: str


@dataclass
class SystemMetrics:
    """System-Health-Metriken"""
    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_used_gb: float
    disk_total_gb: float
    timestamp: str


class LocalMonitoringSystem:
    """
    Lokales Monitoring-System ohne externe Abhängigkeiten
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history

        # Metriken-Historie
        self.gpu_history = deque(maxlen=max_history)
        self.model_history = deque(maxlen=max_history)
        self.system_history = deque(maxlen=max_history)

        # Monitoring aktiv?
        self.monitoring_active = False
        self.monitor_thread = None

        # RTX 2070 spezifische Optimierungen
        self.is_rtx2070 = self._detect_rtx2070()

        logger.info(f"✅ Lokales Monitoring-System initialisiert (RTX 2070: {self.is_rtx2070})")

    def _detect_rtx2070(self) -> bool:
        """Erkennt RTX 2070 GPU"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return 'RTX 2070' in gpu_name
        return False

    def start_monitoring(self, interval_seconds: float = 5.0):
        """Startet kontinuierliches Monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring bereits aktiv")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"📊 Monitoring gestartet (Intervall: {interval_seconds}s)")

    def stop_monitoring(self):
        """Stoppt Monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("📊 Monitoring gestoppt")

    def _monitor_loop(self, interval: float):
        """Monitoring-Hauptschleife"""
        while self.monitoring_active:
            try:
                # Sammle alle Metriken
                gpu_metrics = self.get_gpu_metrics()
                system_metrics = self.get_system_metrics()

                # Speichere in Historie
                if gpu_metrics:
                    self.gpu_history.append(gpu_metrics)
                self.system_history.append(system_metrics)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Monitoring-Fehler: {e}")
                time.sleep(interval)

    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Sammelt GPU-Metriken (RTX 2070 optimiert)"""
        try:
            if not torch.cuda.is_available():
                return None

            # GPU-Speicher
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2

            # Vereinfachte Utilisierung (nicht verfügbar ohne nvidia-ml-py)
            utilization = 0.0  # Placeholder

            # Temperatur (nicht verfügbar ohne nvidia-ml-py)
            temperature = 0.0  # Placeholder

            return GPUMetrics(
                memory_used_mb=memory_allocated,
                memory_total_mb=memory_total,
                memory_free_mb=memory_total - memory_allocated,
                utilization_percent=utilization,
                temperature_celsius=temperature,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"GPU-Metriken Fehler: {e}")
            return None

    def get_system_metrics(self) -> SystemMetrics:
        """Sammelt System-Metriken"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # RAM
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / 1024**2
            memory_total_mb = memory.total / 1024**2

            # Disk
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / 1024**3
            disk_total_gb = disk.total / 1024**3

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"System-Metriken Fehler: {e}")
            # Fallback
            return SystemMetrics(
                cpu_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                timestamp=datetime.now().isoformat()
            )

    def record_model_inference(self, model_name: str, inference_time_ms: float,
                             tokens_generated: int, memory_peak_mb: float):
        """Zeichnet Modell-Inferenz-Metriken auf"""
        try:
            tokens_per_second = (tokens_generated / inference_time_ms) * 1000 if inference_time_ms > 0 else 0

            metrics = ModelMetrics(
                model_name=model_name,
                inference_time_ms=inference_time_ms,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                memory_peak_mb=memory_peak_mb,
                timestamp=datetime.now().isoformat()
            )

            self.model_history.append(metrics)

        except Exception as e:
            logger.error(f"Modell-Metriken Fehler: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Gibt Dashboard-Daten für Anzeige zurück"""
        try:
            # Aktuelle Metriken
            current_gpu = self.gpu_history[-1] if self.gpu_history else None
            current_system = self.system_history[-1] if self.system_history else None

            # Performance-Zusammenfassung
            model_performance = self._calculate_model_performance()

            return {
                "timestamp": datetime.now().isoformat(),
                "gpu": asdict(current_gpu) if current_gpu else None,
                "system": asdict(current_system) if current_system else None,
                "model_performance": model_performance,
                "history_counts": {
                    "gpu": len(self.gpu_history),
                    "system": len(self.system_history),
                    "model": len(self.model_history)
                },
                "rtx2070_optimized": self.is_rtx2070
            }

        except Exception as e:
            logger.error(f"Dashboard-Daten Fehler: {e}")
            return {"error": str(e)}

    def _calculate_model_performance(self) -> Dict[str, Any]:
        """Berechnet Modell-Performance-Statistiken"""
        if not self.model_history:
            return {}

        try:
            # Gruppiere nach Modell
            model_stats = {}
            for metric in self.model_history:
                if metric.model_name not in model_stats:
                    model_stats[metric.model_name] = {
                        "count": 0,
                        "total_time": 0,
                        "total_tokens": 0,
                        "peak_memory": 0
                    }

                stats = model_stats[metric.model_name]
                stats["count"] += 1
                stats["total_time"] += metric.inference_time_ms
                stats["total_tokens"] += metric.tokens_generated
                stats["peak_memory"] = max(stats["peak_memory"], metric.memory_peak_mb)

            # Berechne Durchschnitte
            for model_name, stats in model_stats.items():
                stats["avg_time_ms"] = stats["total_time"] / stats["count"]
                stats["avg_tokens_per_sec"] = (stats["total_tokens"] / (stats["total_time"] / 1000)) if stats["total_time"] > 0 else 0
                stats["total_inferences"] = stats["count"]

            return model_stats

        except Exception as e:
            logger.error(f"Performance-Berechnung Fehler: {e}")
            return {}

    def export_metrics(self, filepath: str):
        """Exportiert alle Metriken als JSON"""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "gpu_history": [asdict(m) for m in self.gpu_history],
                "system_history": [asdict(m) for m in self.system_history],
                "model_history": [asdict(m) for m in self.model_history]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Metriken exportiert nach: {filepath}")

        except Exception as e:
            logger.error(f"Export-Fehler: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Gibt System-Health-Status zurück"""
        try:
            gpu_ok = torch.cuda.is_available() if self.is_rtx2070 else True
            gpu_metrics = self.get_gpu_metrics()
            system_metrics = self.get_system_metrics()

            # Health-Kriterien
            health_checks = {
                "gpu_available": gpu_ok,
                "gpu_memory_ok": gpu_metrics.memory_used_mb < gpu_metrics.memory_total_mb * 0.9 if gpu_metrics else False,
                "system_memory_ok": system_metrics.memory_used_mb < system_metrics.memory_total_mb * 0.9,
                "disk_space_ok": system_metrics.disk_used_gb < system_metrics.disk_total_gb * 0.9,
                "cpu_ok": system_metrics.cpu_percent < 90.0
            }

            overall_health = all(health_checks.values())

            return {
                "overall_healthy": overall_health,
                "checks": health_checks,
                "timestamp": datetime.now().isoformat(),
                "rtx2070_mode": self.is_rtx2070
            }

        except Exception as e:
            logger.error(f"Health-Check Fehler: {e}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Globale Instanz
_monitoring_system = None

def get_monitoring_system() -> LocalMonitoringSystem:
    """Gibt globale Monitoring-Instanz zurück"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = LocalMonitoringSystem()
    return _monitoring_system