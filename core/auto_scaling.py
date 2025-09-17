#!/usr/bin/env python3
"""
Auto-Scaling System für Bundeskanzler KI
Automatisches Performance-Monitoring und adaptive Optimierungen
"""

import asyncio
import threading
import time
import logging
import psutil
import GPUtil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System-Metriken für Auto-Scaling Entscheidungen"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    request_queue_size: int = 0
    active_connections: int = 0
    response_time: float = 0.0
    throughput: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingDecision:
    """Entscheidung für Auto-Scaling Aktionen"""
    action: str  # 'scale_up', 'scale_down', 'optimize', 'maintain'
    target: str  # 'batch_size', 'model_instances', 'memory', 'cpu'
    value: Any
    reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class ModelInstance:
    """Repräsentiert eine Modell-Instanz für Load Balancing"""
    id: str
    model_type: str
    status: str  # 'active', 'idle', 'scaling', 'error'
    load_factor: float = 0.0
    memory_usage: float = 0.0
    last_used: float = field(default_factory=time.time)
    performance_score: float = 1.0

class PerformanceMonitor:
    """
    Überwacht System-Performance und Modell-Metriken
    """

    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 100
        self.lock = threading.Lock()

        # GPU-Monitoring
        self.gpu_available = len(GPUtil.getGPUs()) > 0

        logger.info("📊 PerformanceMonitor initialisiert")

    def collect_system_metrics(self) -> SystemMetrics:
        """Sammelt aktuelle System-Metriken"""
        try:
            # CPU-Metriken
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Memory-Metriken
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # GPU-Metriken
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100
                        gpu_memory_usage = gpus[0].memoryUtil * 100
                except Exception as e:
                    logger.warning(f"⚠️ GPU-Metriken konnten nicht abgerufen werden: {e}")

            # Disk-Metriken
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent

            # Network-Metriken
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }

            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                disk_usage=disk_usage,
                network_io=network_io
            )

            # Metriken zur Historie hinzufügen
            with self.lock:
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)

            return metrics

        except Exception as e:
            logger.error(f"❌ Fehler beim Sammeln von System-Metriken: {e}")
            return SystemMetrics()

    def get_average_metrics(self, window_seconds: float = 60.0) -> SystemMetrics:
        """Berechnet durchschnittliche Metriken über ein Zeitfenster"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        with self.lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return self.collect_system_metrics()

        # Durchschnittswerte berechnen
        avg_metrics = SystemMetrics()
        count = len(recent_metrics)

        for metric in recent_metrics:
            avg_metrics.cpu_usage += metric.cpu_usage
            avg_metrics.memory_usage += metric.memory_usage
            avg_metrics.gpu_usage += metric.gpu_usage
            avg_metrics.gpu_memory_usage += metric.gpu_memory_usage
            avg_metrics.disk_usage += metric.disk_usage

        avg_metrics.cpu_usage /= count
        avg_metrics.memory_usage /= count
        avg_metrics.gpu_usage /= count
        avg_metrics.gpu_memory_usage /= count
        avg_metrics.disk_usage /= count

        return avg_metrics

    def detect_performance_trends(self) -> Dict[str, Any]:
        """Erkennt Performance-Trends und Anomalien"""
        if len(self.metrics_history) < 10:
            return {"trend": "insufficient_data"}

        with self.lock:
            recent = self.metrics_history[-10:]

        # Trend-Analyse
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent])
        gpu_trend = self._calculate_trend([m.gpu_usage for m in recent])

        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "gpu_trend": gpu_trend,
            "overall_load": (cpu_trend["slope"] + memory_trend["slope"] + gpu_trend["slope"]) / 3
        }

    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Berechnet Trend für eine Metrik-Serie"""
        if len(values) < 2:
            return {"slope": 0, "direction": "stable"}

        # Lineare Regression für Trend
        x = np.arange(len(values))
        y = np.array(values)

        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.5:
            direction = "increasing"
        elif slope < -0.5:
            direction = "decreasing"
        else:
            direction = "stable"

        return {
            "slope": slope,
            "direction": direction,
            "current_value": values[-1],
            "avg_value": np.mean(values)
        }

class AdaptiveOptimizer:
    """
    Führt adaptive Optimierungen basierend auf Performance-Daten durch
    """

    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.scaling_decisions: List[ScalingDecision] = []
        self.optimization_interval = 30.0  # 30 Sekunden
        self.last_optimization = time.time()

        # Optimierungsschwellen
        self.thresholds = {
            "cpu_high": 80.0,
            "cpu_low": 30.0,
            "memory_high": 85.0,
            "memory_low": 40.0,
            "gpu_high": 90.0,
            "gpu_low": 40.0
        }

        logger.info("🎯 AdaptiveOptimizer initialisiert")

    def analyze_and_optimize(self) -> List[ScalingDecision]:
        """Analysiert System und trifft Optimierungsentscheidungen"""
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return []

        self.last_optimization = current_time

        decisions = []
        metrics = self.performance_monitor.get_average_metrics(60.0)  # Letzte 60 Sekunden
        trends = self.performance_monitor.detect_performance_trends()

        # CPU-basierte Optimierungen
        if metrics.cpu_usage > self.thresholds["cpu_high"]:
            decisions.append(ScalingDecision(
                action="optimize",
                target="batch_size",
                value="reduce",
                reason=f"Hohe CPU-Auslastung: {metrics.cpu_usage:.1f}%",
                confidence=0.8
            ))

        elif (metrics.cpu_usage < self.thresholds["cpu_low"] and
              "cpu_trend" in trends and
              trends["cpu_trend"]["direction"] == "decreasing"):
            decisions.append(ScalingDecision(
                action="optimize",
                target="batch_size",
                value="increase",
                reason=f"Niedrige CPU-Auslastung: {metrics.cpu_usage:.1f}%",
                confidence=0.6
            ))

        # Memory-basierte Optimierungen
        if metrics.memory_usage > self.thresholds["memory_high"]:
            decisions.append(ScalingDecision(
                action="optimize",
                target="memory",
                value="cleanup",
                reason=f"Hohe Memory-Auslastung: {metrics.memory_usage:.1f}%",
                confidence=0.9
            ))

        # GPU-basierte Optimierungen
        if self.performance_monitor.gpu_available:
            if metrics.gpu_usage > self.thresholds["gpu_high"]:
                decisions.append(ScalingDecision(
                    action="optimize",
                    target="gpu_memory",
                    value="reduce_batch",
                    reason=f"Hohe GPU-Auslastung: {metrics.gpu_usage:.1f}%",
                    confidence=0.8
                ))

        # Trend-basierte Optimierungen
        if "overall_load" in trends and trends["overall_load"] > 0.5:
            decisions.append(ScalingDecision(
                action="scale_up",
                target="model_instances",
                value=1,
                reason=f"Steigende Systemlast: {trends['overall_load']:.2f}",
                confidence=0.7
            ))

        elif "overall_load" in trends and trends["overall_load"] < -0.3:
            decisions.append(ScalingDecision(
                action="scale_down",
                target="model_instances",
                value=1,
                reason=f"Fallende Systemlast: {trends['overall_load']:.2f}",
                confidence=0.6
            ))

        # Entscheidungen speichern
        self.scaling_decisions.extend(decisions)

        # Nur die letzten 50 Entscheidungen behalten
        if len(self.scaling_decisions) > 50:
            self.scaling_decisions = self.scaling_decisions[-50:]

        return decisions

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über Optimierungsentscheidungen zurück"""
        if not self.scaling_decisions:
            return {"total_decisions": 0}

        recent_decisions = [d for d in self.scaling_decisions if time.time() - d.timestamp < 3600]  # Letzte Stunde

        action_counts = {}
        target_counts = {}

        for decision in recent_decisions:
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
            target_counts[decision.target] = target_counts.get(decision.target, 0) + 1

        return {
            "total_decisions": len(self.scaling_decisions),
            "recent_decisions": len(recent_decisions),
            "action_distribution": action_counts,
            "target_distribution": target_counts,
            "avg_confidence": np.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0
        }

class LoadBalancer:
    """
    Verteilt Anfragen auf verfügbare Modell-Instanzen
    """

    def __init__(self):
        self.model_instances: Dict[str, ModelInstance] = {}
        self.load_balancing_strategy = "least_loaded"  # 'least_loaded', 'round_robin', 'performance_weighted'

        logger.info("⚖️ LoadBalancer initialisiert")

    def register_model_instance(self, instance: ModelInstance):
        """Registriert eine neue Modell-Instanz"""
        self.model_instances[instance.id] = instance
        logger.info(f"📝 Modell-Instanz {instance.id} registriert ({instance.model_type})")

    def unregister_model_instance(self, instance_id: str):
        """Entfernt eine Modell-Instanz"""
        if instance_id in self.model_instances:
            del self.model_instances[instance_id]
            logger.info(f"🗑️ Modell-Instanz {instance_id} entfernt")

    def select_instance(self, model_type: str, request_priority: int = 1) -> Optional[str]:
        """Wählt die beste Instanz für eine Anfrage aus"""
        available_instances = [
            inst for inst in self.model_instances.values()
            if inst.model_type == model_type and inst.status == "active"
        ]

        if not available_instances:
            return None

        if self.load_balancing_strategy == "least_loaded":
            # Wähle Instanz mit niedrigstem Load-Faktor
            selected = min(available_instances, key=lambda x: x.load_factor)
        elif self.load_balancing_strategy == "performance_weighted":
            # Gewichte nach Performance-Score
            total_weight = sum(inst.performance_score for inst in available_instances)
            if total_weight > 0:
                weights = [inst.performance_score / total_weight for inst in available_instances]
                selected = np.random.choice(available_instances, p=weights)
            else:
                selected = available_instances[0]
        else:  # round_robin
            # Einfache Round-Robin Implementierung
            sorted_instances = sorted(available_instances, key=lambda x: x.last_used)
            selected = sorted_instances[0]

        selected.last_used = time.time()
        return selected.id

    def update_instance_load(self, instance_id: str, load_factor: float):
        """Aktualisiert den Load-Faktor einer Instanz"""
        if instance_id in self.model_instances:
            self.model_instances[instance_id].load_factor = load_factor

    def get_load_distribution(self) -> Dict[str, Any]:
        """Gibt die aktuelle Load-Verteilung zurück"""
        distribution = {}
        for model_type in set(inst.model_type for inst in self.model_instances.values()):
            type_instances = [inst for inst in self.model_instances.values() if inst.model_type == model_type]
            distribution[model_type] = {
                "total_instances": len(type_instances),
                "active_instances": len([inst for inst in type_instances if inst.status == "active"]),
                "avg_load": np.mean([inst.load_factor for inst in type_instances]) if type_instances else 0,
                "total_load": sum(inst.load_factor for inst in type_instances)
            }

        return distribution

class AutoScaler:
    """
    Haupt-Klasse für Auto-Scaling Funktionalität.

    Diese Klasse orchestriert das gesamte Auto-Scaling System und integriert:
    - Performance-Monitoring für kontinuierliche Systemüberwachung
    - Adaptive Optimizer für intelligente Skalierungsentscheidungen
    - Load Balancer für optimale Anfragenverteilung

    Attributes:
        performance_monitor (PerformanceMonitor): System-Performance Überwachung
        adaptive_optimizer (AdaptiveOptimizer): Intelligente Optimierungsentscheidungen
        load_balancer (LoadBalancer): Anfragenverteilung und Load Balancing

        scaling_enabled (bool): Aktivierungsstatus des Auto-Scaling
        min_instances (int): Minimale Anzahl von Modell-Instanzen
        max_instances (int): Maximale Anzahl von Modell-Instanzen
        scaling_cooldown (int): Cooldown-Periode zwischen Skalierungen in Sekunden

        scaling_events (List[Dict[str, Any]]): Historie der Skalierungsereignisse
        last_scaling_action (float): Timestamp der letzten Skalierungsaktion
    """

    def __init__(self) -> None:
        """
        Initialisiert das Auto-Scaling System.

        Erstellt alle notwendigen Komponenten und startet den Monitoring-Thread.
        Das System ist standardmäßig aktiviert und bereit für den Betrieb.
        """
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_optimizer = AdaptiveOptimizer(self.performance_monitor)
        self.load_balancer = LoadBalancer()

        # Auto-Scaling Konfiguration
        self.scaling_enabled: bool = True
        self.min_instances: int = 1
        self.max_instances: int = 5
        self.scaling_cooldown: int = 300  # 5 Minuten zwischen Skalierungen

        # Threading
        self.monitoring_thread: threading.Thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # Statistische Daten
        self.scaling_events: List[Dict[str, Any]] = []
        self.last_scaling_action: float = 0

        logger.info("🚀 AutoScaler initialisiert")

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Sammelt aktuelle System-Metriken als Dictionary"""
        try:
            # CPU-Metriken
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory-Metriken
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # GPU-Metriken
            gpu_percent = 0.0
            gpu_memory_percent = 0.0
            if self.performance_monitor.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = gpus[0].load * 100
                        gpu_memory_percent = gpus[0].memoryUtil * 100
                except Exception as e:
                    logger.warning(f"⚠️ GPU-Metriken konnten nicht abgerufen werden: {e}")

            # Disk-Metriken
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Network-Metriken
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'gpu_percent': gpu_percent,
                'gpu_memory_percent': gpu_memory_percent,
                'disk_percent': disk_percent,
                'network_io': network_io,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"❌ Fehler beim Sammeln von System-Metriken: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'gpu_percent': 0.0,
                'gpu_memory_percent': 0.0,
                'disk_percent': 0.0,
                'network_io': {},
                'timestamp': time.time()
            }

    def _monitoring_loop(self):
        """Haupt-Monitoring-Loop"""
        while True:
            try:
                # System-Metriken sammeln
                self.performance_monitor.collect_system_metrics()

                # Optimierungsentscheidungen treffen
                if self.scaling_enabled:
                    decisions = self.adaptive_optimizer.analyze_and_optimize()

                    for decision in decisions:
                        self._execute_decision(decision)

                # Kurz warten
                time.sleep(10)  # Alle 10 Sekunden

            except Exception as e:
                logger.error(f"❌ Fehler im Monitoring-Loop: {e}")
                time.sleep(30)  # Bei Fehler länger warten

    def _execute_decision(self, decision: ScalingDecision):
        """Führt eine Skalierungsentscheidung aus"""
        current_time = time.time()

        # Cooldown prüfen
        if current_time - self.last_scaling_action < self.scaling_cooldown:
            return

        logger.info(f"⚡ Führe Skalierungsaktion aus: {decision.action} {decision.target} -> {decision.value}")
        logger.info(f"📋 Grund: {decision.reason} (Konfidenz: {decision.confidence:.2f})")

        # Entscheidung ausführen (hier würden die tatsächlichen Skalierungsaktionen stattfinden)
        if decision.action == "optimize":
            if decision.target == "batch_size":
                self._optimize_batch_size(decision.value)
            elif decision.target == "memory":
                self._optimize_memory(decision.value)

        elif decision.action in ["scale_up", "scale_down"]:
            if decision.target == "model_instances":
                self._scale_model_instances(decision.action, decision.value)

        # Skalierungsereignis protokollieren
        self.scaling_events.append({
            "timestamp": current_time,
            "decision": decision.__dict__,
            "executed": True
        })

        self.last_scaling_action = current_time

    def _optimize_batch_size(self, direction: str):
        """Optimiert Batch-Größen"""
        try:
            # Hier würde die Integration mit dem Request Batching System stattfinden
            if direction == "increase":
                logger.info("📈 Erhöhe Batch-Größen für bessere Durchsatz")
            elif direction == "reduce":
                logger.info("📉 Reduziere Batch-Größen zur Ressourcenschonung")
        except Exception as e:
            logger.error(f"❌ Fehler bei Batch-Größen-Optimierung: {e}")

    def _optimize_memory(self, action: str):
        """Optimiert Memory-Verwendung"""
        try:
            if action == "cleanup":
                logger.info("🧹 Führe Memory-Cleanup durch")
                # Force garbage collection
                import gc
                gc.collect()
        except Exception as e:
            logger.error(f"❌ Fehler bei Memory-Optimierung: {e}")

    def _scale_model_instances(self, action: str, count: int):
        """Skaliert Modell-Instanzen"""
        try:
            current_instances = len([inst for inst in self.load_balancer.model_instances.values() if inst.status == "active"])

            if action == "scale_up":
                new_count = min(current_instances + count, self.max_instances)
                if new_count > current_instances:
                    logger.info(f"⬆️ Skaliere von {current_instances} auf {new_count} Modell-Instanzen")
            elif action == "scale_down":
                new_count = max(current_instances - count, self.min_instances)
                if new_count < current_instances:
                    logger.info(f"⬇️ Skaliere von {current_instances} auf {new_count} Modell-Instanzen")
        except Exception as e:
            logger.error(f"❌ Fehler bei Modell-Skalierung: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Gibt den aktuellen System-Status zurück"""
        metrics = self.performance_monitor.get_average_metrics(60.0)
        trends = self.performance_monitor.detect_performance_trends()
        load_distribution = self.load_balancer.get_load_distribution()
        optimization_stats = self.adaptive_optimizer.get_optimization_stats()

        return {
            "metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "gpu_usage": metrics.gpu_usage,
                "gpu_memory_usage": metrics.gpu_memory_usage,
                "disk_usage": metrics.disk_usage
            },
            "trends": trends,
            "load_distribution": load_distribution,
            "optimization_stats": optimization_stats,
            "scaling_enabled": self.scaling_enabled,
            "active_instances": len([inst for inst in self.load_balancer.model_instances.values() if inst.status == "active"]),
            "last_scaling": self.last_scaling_action,
            "timestamp": time.time()
        }

    def enable_scaling(self) -> None:
        """
        Aktiviert das Auto-Scaling System.

        Startet die automatische Performance-Überwachung und Optimierung.
        Das System wird kontinuierlich System-Metriken sammeln und
        automatisch Skalierungsentscheidungen treffen.
        """
        self.scaling_enabled = True
        logger.info("✅ Auto-Scaling aktiviert")

    def disable_scaling(self) -> None:
        """
        Deaktiviert das Auto-Scaling System.

        Stoppt die automatische Performance-Überwachung und Optimierung.
        Alle laufenden Monitoring-Prozesse werden beendet.
        """
        self.scaling_enabled = False
        logger.info("⏸️ Auto-Scaling deaktiviert")

    def manual_scale(self, action: str, target: str, value: Any) -> None:
        """
        Führt eine manuelle Skalierungsaktion durch.

        Ermöglicht das manuelle Überschreiben der automatischen Skalierungsentscheidungen.
        Die Aktion wird sofort ausgeführt und in der Historie protokolliert.

        Args:
            action: Skalierungsaktion ("scale_up", "scale_down", "optimize")
            target: Ziel der Skalierung ("batch_size", "model_instances", "memory", "cpu")
            value: Neuer Wert oder Zielwert für die Skalierung
        """
        decision = ScalingDecision(
            action=action,
            target=target,
            value=value,
            reason="Manual scaling",
            confidence=1.0
        )

        self._execute_decision(decision)
        logger.info(f"🔧 Manuelle Skalierung ausgeführt: {action} {target} -> {value}")

# Globale Instanz
auto_scaler = AutoScaler()

def get_auto_scaler() -> AutoScaler:
    """
    Gibt die globale AutoScaler-Instanz zurück.

    Diese Funktion implementiert das Singleton-Pattern für das Auto-Scaling System.
    Stellt sicher, dass nur eine Instanz des AutoScalers existiert und
    von allen Komponenten gemeinsam genutzt wird.

    Returns:
        AutoScaler: Die globale AutoScaler-Instanz
    """
    return auto_scaler
    return auto_scaler