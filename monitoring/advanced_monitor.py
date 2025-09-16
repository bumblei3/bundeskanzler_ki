#!/usr/bin/env python3
"""
Advanced Monitoring System für Bundeskanzler-KI
Umfassendes Monitoring, Logging und Alerting
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil


@dataclass
class MetricEntry:
    """Single metric entry"""

    timestamp: datetime
    metric_name: str
    value: Any
    tags: Dict[str, str] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "tags": self.tags or {},
        }


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    duration_minutes: int = 5
    enabled: bool = True
    callback: Optional[Callable] = None


class AdvancedMonitor:
    """
    Advanced monitoring system with metrics collection, alerting, and reporting
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.metrics_dir = self.project_root / "monitoring" / "metrics"
        self.logs_dir = self.project_root / "logs"

        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, datetime] = {}

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None

        # Configure advanced logging
        self._setup_logging()

        # System metrics
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "gpu_memory": 0.0,
        }

        # Application metrics
        self.app_metrics = {
            "requests_total": 0,
            "requests_per_minute": 0,
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "cache_hit_rate": 0.0,
            "confidence_scores": [],
        }

        # Load configuration
        self._load_config()

        logging.info("🔍 Advanced Monitoring System initialisiert")

    def _setup_logging(self):
        """Setup advanced logging configuration"""
        # Main application logger
        self.logger = logging.getLogger("bundeskanzler_ki")
        self.logger.setLevel(logging.DEBUG)

        # Metrics logger
        self.metrics_logger = logging.getLogger("metrics")
        self.metrics_logger.setLevel(logging.INFO)

        # Error logger
        self.error_logger = logging.getLogger("errors")
        self.error_logger.setLevel(logging.ERROR)

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s"
        )

        simple_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

        # Create handlers
        # Main log file
        main_handler = logging.FileHandler(self.logs_dir / "bundeskanzler_ki.log")
        main_handler.setFormatter(detailed_formatter)
        main_handler.setLevel(logging.DEBUG)

        # Error log file
        error_handler = logging.FileHandler(self.logs_dir / "errors.log")
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)

        # Metrics log file
        metrics_handler = logging.FileHandler(self.logs_dir / "metrics.log")
        metrics_handler.setFormatter(simple_formatter)
        metrics_handler.setLevel(logging.INFO)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)

        # Add handlers to loggers
        self.logger.addHandler(main_handler)
        self.logger.addHandler(console_handler)

        self.error_logger.addHandler(error_handler)
        self.error_logger.addHandler(console_handler)

        self.metrics_logger.addHandler(metrics_handler)

        # Ensure no duplicate logs
        self.logger.propagate = False
        self.error_logger.propagate = False
        self.metrics_logger.propagate = False

    def _load_config(self):
        """Load monitoring configuration"""
        config_file = self.project_root / "monitoring" / "config.json"

        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Load alert rules
                for rule_config in config.get("alert_rules", []):
                    self.add_alert_rule(AlertRule(**rule_config))

                self.logger.info(
                    f"Monitoring Konfiguration geladen: {len(self.alert_rules)} Alert Rules"
                )

            except Exception as e:
                self.logger.warning(f"Fehler beim Laden der Monitoring-Konfiguration: {e}")
        else:
            # Create default configuration
            self._create_default_config()

    def _create_default_config(self):
        """Create default monitoring configuration"""
        default_config = {
            "alert_rules": [
                {
                    "name": "High CPU Usage",
                    "metric_name": "cpu_usage",
                    "condition": "gt",
                    "threshold": 80.0,
                    "duration_minutes": 5,
                },
                {
                    "name": "High Memory Usage",
                    "metric_name": "memory_usage",
                    "condition": "gt",
                    "threshold": 85.0,
                    "duration_minutes": 3,
                },
                {
                    "name": "High Error Rate",
                    "metric_name": "error_rate",
                    "condition": "gt",
                    "threshold": 5.0,
                    "duration_minutes": 2,
                },
                {
                    "name": "Low Confidence Score",
                    "metric_name": "avg_confidence",
                    "condition": "lt",
                    "threshold": 50.0,
                    "duration_minutes": 10,
                },
            ],
            "collection_interval": 30,
            "retention_days": 7,
        }

        config_file = self.project_root / "monitoring" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        # Load the default rules
        for rule_config in default_config["alert_rules"]:
            self.add_alert_rule(AlertRule(**rule_config))

    def add_metric(self, name: str, value: Any, tags: Dict[str, str] = None):
        """Add a metric entry"""
        entry = MetricEntry(timestamp=datetime.now(), metric_name=name, value=value, tags=tags)

        self.metrics[name].append(entry)

        # Log metric
        self.metrics_logger.info(f"METRIC {name}={value} {tags or ''}")

        # Check alert rules
        self._check_alerts(name, value)

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules.append(rule)
        self.logger.info(f"Alert Rule hinzugefügt: {rule.name}")

    def _check_alerts(self, metric_name: str, value: Any):
        """Check if any alert rules are triggered"""
        for rule in self.alert_rules:
            if not rule.enabled or rule.metric_name != metric_name:
                continue

            triggered = False

            if rule.condition == "gt" and float(value) > rule.threshold:
                triggered = True
            elif rule.condition == "lt" and float(value) < rule.threshold:
                triggered = True
            elif rule.condition == "eq" and float(value) == rule.threshold:
                triggered = True
            elif rule.condition == "ne" and float(value) != rule.threshold:
                triggered = True

            if triggered:
                self._trigger_alert(rule, value)

    def _trigger_alert(self, rule: AlertRule, current_value: Any):
        """Trigger an alert"""
        alert_key = rule.name
        now = datetime.now()

        # Check if alert is already active
        if alert_key in self.active_alerts:
            # Check if duration threshold is met
            if now - self.active_alerts[alert_key] >= timedelta(minutes=rule.duration_minutes):
                self._send_alert(rule, current_value)
        else:
            # New alert
            self.active_alerts[alert_key] = now

            # Immediate alert for critical metrics
            if rule.metric_name in ["error_rate", "memory_usage"] and rule.threshold > 90:
                self._send_alert(rule, current_value)

    def _send_alert(self, rule: AlertRule, current_value: Any):
        """Send alert notification"""
        alert_message = f"🚨 ALERT: {rule.name} - {rule.metric_name}={current_value} (Threshold: {rule.threshold})"

        # Log alert
        self.error_logger.error(alert_message)

        # Execute callback if provided
        if rule.callback:
            try:
                rule.callback(rule, current_value)
            except Exception as e:
                self.error_logger.error(f"Alert callback failed: {e}")

        # Save alert to file
        self._save_alert(rule, current_value, alert_message)

    def _save_alert(self, rule: AlertRule, current_value: Any, message: str):
        """Save alert to alerts log"""
        alert_entry = {
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "current_value": current_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "message": message,
        }

        alerts_file = self.logs_dir / "alerts.jsonl"
        with open(alerts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert_entry, ensure_ascii=False) + "\n")

    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.add_metric("cpu_usage", cpu_percent, {"type": "system"})

            # Memory Usage
            memory = psutil.virtual_memory()
            self.add_metric("memory_usage", memory.percent, {"type": "system"})
            self.add_metric("memory_available_gb", memory.available / (1024**3), {"type": "system"})

            # Disk Usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.add_metric("disk_usage", disk_percent, {"type": "system"})

            # GPU Memory (if available)
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    self.add_metric("gpu_memory_usage", gpu_memory_percent, {"type": "gpu"})
                    self.add_metric("gpu_utilization", gpu.load * 100, {"type": "gpu"})
            except ImportError:
                pass  # GPU monitoring not available

            # Process-specific metrics
            process = psutil.Process()
            self.add_metric(
                "process_memory_mb",
                process.memory_info().rss / (1024**2),
                {"type": "process"},
            )
            self.add_metric("process_cpu_percent", process.cpu_percent(), {"type": "process"})

        except Exception as e:
            self.error_logger.error(f"Error collecting system metrics: {e}")

    def track_request(self, start_time: float, success: bool, confidence: float = None):
        """Track application request metrics"""
        response_time = time.time() - start_time

        self.add_metric("response_time_ms", response_time * 1000, {"type": "application"})
        self.add_metric("request_success", 1 if success else 0, {"type": "application"})

        if confidence is not None:
            self.add_metric("confidence_score", confidence * 100, {"type": "application"})

        # Update aggregated metrics
        self.app_metrics["requests_total"] += 1

        # Calculate error rate (last 100 requests)
        recent_requests = list(self.metrics["request_success"])[-100:]
        if recent_requests:
            error_rate = (1 - sum(r.value for r in recent_requests) / len(recent_requests)) * 100
            self.add_metric("error_rate", error_rate, {"type": "application"})

    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.logger.info(f"Continuous monitoring gestartet (Interval: {interval}s)")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("Continuous monitoring gestoppt")

    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                self.error_logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)

    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        summary = {}

        for metric_name, entries in self.metrics.items():
            recent_entries = [e for e in entries if e.timestamp >= cutoff_time]

            if recent_entries:
                values = [e.value for e in recent_entries if isinstance(e.value, (int, float))]

                if values:
                    summary[metric_name] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1],
                    }

        return summary

    def export_metrics(self, format: str = "json", hours: int = 24) -> str:
        """Export metrics in specified format"""
        summary = self.get_metrics_summary(hours)

        if format.lower() == "json":
            return json.dumps(summary, indent=2, ensure_ascii=False)
        elif format.lower() == "prometheus":
            return self._export_prometheus_format(summary)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_prometheus_format(self, summary: Dict) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        timestamp = int(time.time() * 1000)

        for metric_name, stats in summary.items():
            # Clean metric name for Prometheus
            clean_name = metric_name.replace(" ", "_").replace("-", "_").lower()

            lines.append(f"# HELP {clean_name} {metric_name}")
            lines.append(f"# TYPE {clean_name} gauge")
            lines.append(f"{clean_name}_avg {stats['avg']} {timestamp}")
            lines.append(f"{clean_name}_min {stats['min']} {timestamp}")
            lines.append(f"{clean_name}_max {stats['max']} {timestamp}")
            lines.append(f"{clean_name}_latest {stats['latest']} {timestamp}")

        return "\n".join(lines)

    def generate_health_check(self) -> Dict:
        """Generate comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
        }

        try:
            # System health
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            health["checks"]["system"] = {
                "cpu_ok": cpu_usage < 80,
                "memory_ok": memory_usage < 85,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
            }

            # Application health
            recent_errors = len(
                [
                    e
                    for e in self.metrics.get("request_success", [])
                    if e.timestamp >= datetime.now() - timedelta(minutes=5) and e.value == 0
                ]
            )

            health["checks"]["application"] = {
                "recent_errors": recent_errors,
                "errors_ok": recent_errors < 5,
                "total_requests": self.app_metrics["requests_total"],
            }

            # Active alerts
            active_alerts_count = len(self.active_alerts)
            health["checks"]["alerts"] = {
                "active_alerts": active_alerts_count,
                "alerts_ok": active_alerts_count == 0,
            }

            # Overall status
            all_checks_ok = all(
                check.get("cpu_ok", True)
                and check.get("memory_ok", True)
                and check.get("errors_ok", True)
                and check.get("alerts_ok", True)
                for check in health["checks"].values()
            )

            health["status"] = "healthy" if all_checks_ok else "unhealthy"

        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            self.error_logger.error(f"Health check failed: {e}")

        return health

    def cleanup_old_metrics(self, days: int = 7):
        """Clean up old metric entries"""
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        for metric_name, entries in self.metrics.items():
            original_count = len(entries)
            # Keep only recent entries
            recent_entries = deque([e for e in entries if e.timestamp >= cutoff_time], maxlen=1000)
            self.metrics[metric_name] = recent_entries
            cleaned_count += original_count - len(recent_entries)

        self.logger.info(f"Metric cleanup: {cleaned_count} alte Einträge entfernt")
        return cleaned_count


# Context manager for request tracking
class RequestTracker:
    """Context manager for tracking request metrics"""

    def __init__(self, monitor: AdvancedMonitor, operation: str = "request"):
        self.monitor = monitor
        self.operation = operation
        self.start_time = None
        self.success = True
        self.confidence = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.monitor.error_logger.error(f"Request failed: {exc_val}")

        self.monitor.track_request(self.start_time, self.success, self.confidence)

    def set_confidence(self, confidence: float):
        """Set confidence score for this request"""
        self.confidence = confidence


# Global monitor instance
_global_monitor = None


def get_monitor() -> AdvancedMonitor:
    """Get global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AdvancedMonitor()
    return _global_monitor


def monitor_request(operation: str = "request") -> RequestTracker:
    """Create request tracker context manager"""
    return RequestTracker(get_monitor(), operation)


if __name__ == "__main__":
    # Demo monitoring
    monitor = AdvancedMonitor()

    print("🔍 Starting monitoring demo...")

    # Start continuous monitoring
    monitor.start_monitoring(interval=5)

    # Simulate some application activity
    import random

    for i in range(10):
        with monitor_request(f"test_request_{i}") as tracker:
            # Simulate work
            time.sleep(random.uniform(0.1, 0.5))

            # Simulate confidence score
            confidence = random.uniform(0.6, 0.9)
            tracker.set_confidence(confidence)

            # Simulate occasional errors
            if random.random() < 0.1:
                raise Exception("Simulated error")

    time.sleep(10)  # Let monitoring collect some data

    # Show results
    print("\n📊 Monitoring Results:")
    print("=" * 50)

    summary = monitor.get_metrics_summary(hours=1)
    for metric, stats in summary.items():
        print(f"{metric}: avg={stats['avg']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    health = monitor.generate_health_check()
    print(f"\n🏥 Health Status: {health['status']}")

    monitor.stop_monitoring()
