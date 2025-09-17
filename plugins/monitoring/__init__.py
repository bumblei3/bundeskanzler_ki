#!/usr/bin/env python3
"""
Monitoring-Plugin für Bundeskanzler KI
Überwacht System-Performance und Plugin-Aktivitäten
"""

import time
import psutil
import threading
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import deque
from core.plugin_system import HookPlugin, PluginMetadata

class MonitoringPlugin(HookPlugin):
    """
    Plugin zur Überwachung von System und Plugins

    Sammelt Metriken über CPU, Speicher, Plugin-Ausführung und Systemzustand.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="monitoring",
            version="1.0.0",
            description="Überwacht System-Performance und Plugin-Aktivitäten",
            author="Bundeskanzler KI Team",
            license="MIT",
            tags=["monitoring", "metrics", "performance", "system"],
            dependencies=["psutil"]
        )

    def initialize(self) -> None:
        """Initialisiert das Plugin"""
        self.logger.info("Monitoring-Plugin initialisiert")

        # Metriken-Speicher
        self.metrics_history = deque(maxlen=1000)
        self.plugin_metrics = {}
        self.system_metrics = {}

        # Monitoring-Konfiguration
        self.monitoring_interval = self._config.settings.get('interval', 30)  # Sekunden
        self.max_history_size = self._config.settings.get('max_history', 1000)

        # Monitoring-Thread
        self._monitoring_thread = None
        self._stop_monitoring = False

        # Starte Monitoring
        if self._config.auto_start:
            self.start_monitoring()

    def shutdown(self) -> None:
        """Beendet das Plugin"""
        self.logger.info("Monitoring-Plugin beendet")
        self.stop_monitoring()

    def start_monitoring(self) -> None:
        """Startet die kontinuierliche Überwachung"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring bereits aktiv")
            return

        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("Monitoring gestartet")

    def stop_monitoring(self) -> None:
        """Stoppt die kontinuierliche Überwachung"""
        if self._monitoring_thread:
            self._stop_monitoring = True
            self._monitoring_thread.join(timeout=5)
            self.logger.info("Monitoring gestoppt")

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Gibt aktuelle System-Metriken zurück

        Returns:
            Dictionary mit System-Metriken
        """
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'timestamp': datetime.now().isoformat()
        }

    def get_plugin_metrics(self) -> Dict[str, Any]:
        """
        Gibt Plugin-Metriken zurück

        Returns:
            Dictionary mit Plugin-Metriken
        """
        return dict(self.plugin_metrics)

    def get_recent_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        Gibt Metriken der letzten Stunden zurück

        Args:
            hours: Anzahl der Stunden für die Historie

        Returns:
            Liste der Metriken
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = []

        for metric in self.metrics_history:
            if datetime.fromisoformat(metric['timestamp']) > cutoff_time:
                recent_metrics.append(metric)

        return recent_metrics

    def log_plugin_execution(self, plugin_name: str, method_name: str, execution_time: float) -> None:
        """
        Protokolliert die Ausführung eines Plugins

        Args:
            plugin_name: Name des Plugins
            method_name: Name der ausgeführten Methode
            execution_time: Ausführungszeit in Sekunden
        """
        if plugin_name not in self.plugin_metrics:
            self.plugin_metrics[plugin_name] = {
                'total_executions': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'last_execution': None,
                'methods': {}
            }

        metrics = self.plugin_metrics[plugin_name]
        metrics['total_executions'] += 1
        metrics['total_time'] += execution_time
        metrics['average_time'] = metrics['total_time'] / metrics['total_executions']
        metrics['last_execution'] = datetime.now().isoformat()

        if method_name not in metrics['methods']:
            metrics['methods'][method_name] = {
                'executions': 0,
                'total_time': 0.0,
                'average_time': 0.0
            }

        method_metrics = metrics['methods'][method_name]
        method_metrics['executions'] += 1
        method_metrics['total_time'] += execution_time
        method_metrics['average_time'] = method_metrics['total_time'] / method_metrics['executions']

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Erstellt einen Performance-Bericht

        Returns:
            Dictionary mit Performance-Statistiken
        """
        report = {
            'system_health': self._assess_system_health(),
            'plugin_performance': {},
            'recommendations': []
        }

        # Plugin-Performance analysieren
        for plugin_name, metrics in self.plugin_metrics.items():
            report['plugin_performance'][plugin_name] = {
                'total_executions': metrics['total_executions'],
                'average_execution_time': metrics['average_time'],
                'efficiency_score': self._calculate_efficiency_score(metrics)
            }

        # Empfehlungen generieren
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def on_request_start(self, request_data: Dict[str, Any]) -> None:
        """Hook für Anfrage-Start"""
        request_data['monitoring_start_time'] = time.time()
        self.logger.debug("Anfrage gestartet")

    def on_request_end(self, request_data: Dict[str, Any], response: Any) -> None:
        """Hook für Anfrage-Ende"""
        if 'monitoring_start_time' in request_data:
            execution_time = time.time() - request_data['monitoring_start_time']
            self.logger.debug(f"Anfrage beendet in {execution_time:.3f}s")

            # Metrik speichern
            metric = {
                'type': 'request',
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'request_type': request_data.get('type', 'unknown')
            }
            self.metrics_history.append(metric)

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Hook für Fehler"""
        error_metric = {
            'type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        self.metrics_history.append(error_metric)
        self.logger.warning(f"Fehler aufgezeichnet: {error}")

    def on_system_start(self) -> None:
        """Hook für System-Start"""
        self.logger.info("System-Monitoring aktiviert")

    def on_system_shutdown(self) -> None:
        """Hook für System-Shutdown"""
        self.logger.info("System-Monitoring deaktiviert")

    def _monitoring_loop(self) -> None:
        """Haupt-Monitoring-Schleife"""
        while not self._stop_monitoring:
            try:
                # System-Metriken sammeln
                system_metrics = self.get_system_metrics()
                self.system_metrics = system_metrics

                # Metrik zur Historie hinzufügen
                metric = {
                    'type': 'system',
                    'cpu_percent': system_metrics['cpu_percent'],
                    'memory_percent': system_metrics['memory_percent'],
                    'timestamp': system_metrics['timestamp']
                }
                self.metrics_history.append(metric)

                # Auf hohe Auslastung prüfen
                if system_metrics['cpu_percent'] > 90:
                    self.logger.warning("Hohe CPU-Auslastung erkannt")
                if system_metrics['memory_percent'] > 90:
                    self.logger.warning("Hohe Speicher-Auslastung erkannt")

            except Exception as e:
                self.logger.error(f"Fehler im Monitoring-Loop: {e}")

            time.sleep(self.monitoring_interval)

    def _assess_system_health(self) -> str:
        """Bewertet die Systemgesundheit"""
        if not self.system_metrics:
            return "unknown"

        cpu = self.system_metrics.get('cpu_percent', 0)
        memory = self.system_metrics.get('memory_percent', 0)

        if cpu > 90 or memory > 90:
            return "critical"
        elif cpu > 70 or memory > 70:
            return "warning"
        else:
            return "healthy"

    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Berechnet einen Effizienz-Score für ein Plugin"""
        avg_time = metrics.get('average_time', 0)
        executions = metrics.get('total_executions', 0)

        if executions == 0:
            return 0.0

        # Höhere Scores für schnellere, häufig verwendete Plugins
        base_score = 100 / (1 + avg_time)  # 0-100 basierend auf Geschwindigkeit
        usage_bonus = min(executions / 10, 50)  # Bonus für häufige Verwendung

        return min(base_score + usage_bonus, 100)

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generiert Empfehlungen basierend auf dem Bericht"""
        recommendations = []

        system_health = report.get('system_health', 'unknown')
        if system_health == 'critical':
            recommendations.append("System überlastet - Ressourcen überprüfen")
        elif system_health == 'warning':
            recommendations.append("Systemauslastung hoch - Optimierung empfohlen")

        for plugin_name, perf in report.get('plugin_performance', {}).items():
            if perf['average_execution_time'] > 5.0:
                recommendations.append(f"Plugin {plugin_name} ist langsam - Optimierung erwägen")
            if perf['efficiency_score'] < 30:
                recommendations.append(f"Plugin {plugin_name} hat niedrige Effizienz - Überprüfung empfohlen")

        return recommendations