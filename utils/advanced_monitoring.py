"""
Erweitertes Monitoring-System für Bundeskanzler KI
Beinhaltet Performance-Metriken, Logging und System-Überwachung
"""

import logging
import time
import psutil
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
import statistics

logger = logging.getLogger(__name__)

class AdvancedMonitoringSystem:
    """
    Umfassendes Monitoring-System für die Bundeskanzler KI mit:
    - Performance-Metriken
    - Antwort-Qualitäts-Tracking
    - System-Health-Monitoring
    - Nutzer-Feedback-Analyse
    - Echtzeit-Dashboards
    """

    def __init__(self, log_dir: str = "monitoring_logs", max_history: int = 1000):
        self.log_dir = log_dir
        self.max_history = max_history

        # Metriken-Speicher
        self.performance_metrics = deque(maxlen=max_history)
        self.response_metrics = deque(maxlen=max_history)
        self.system_health = deque(maxlen=max_history)
        self.user_feedback = deque(maxlen=max_history)

        # Aggregierte Statistiken
        self.hourly_stats = defaultdict(list)
        self.daily_stats = defaultdict(list)

        # Aktive Sessions
        self.active_sessions = {}
        self.session_counter = 0

        # Monitoring-Thread
        self.monitoring_active = False
        self.monitor_thread = None

        # Erstelle Log-Verzeichnis
        os.makedirs(log_dir, exist_ok=True)

        # Lade historische Daten
        self._load_historical_data()

    def start_monitoring(self):
        """Startet das kontinuierliche Monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Advanced Monitoring System gestartet")

    def stop_monitoring(self):
        """Stoppt das Monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self._save_historical_data()
        logger.info("⏹️ Advanced Monitoring System gestoppt")

    def _monitoring_loop(self):
        """Haupt-Monitoring-Loop"""
        while self.monitoring_active:
            try:
                # Sammle System-Metriken
                system_metrics = self._collect_system_metrics()
                self.system_health.append(system_metrics)

                # Berechne aggregierte Statistiken
                self._update_aggregated_stats()

                # Automatische Bereinigung alter Daten
                self._cleanup_old_data()

                time.sleep(30)  # Alle 30 Sekunden

            except Exception as e:
                logger.error(f"Fehler im Monitoring-Loop: {e}")

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Sammelt umfassende System-Metriken"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'active_processes': len(psutil.pids()),
            'system_load': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }

    def log_performance_metric(self, operation: str, duration: float, success: bool = True,
                             metadata: Optional[Dict] = None):
        """Loggt eine Performance-Metrik"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_ms': duration * 1000,
            'success': success,
            'metadata': metadata or {}
        }

        self.performance_metrics.append(metric)

        # Log bei langsamen Operationen
        if duration > 5.0:  # > 5 Sekunden
            logger.warning(f"🐌 Langsame Operation: {operation} ({duration:.2f}s)")
        elif not success:
            logger.error(f"❌ Fehlgeschlagene Operation: {operation}")

    def log_response_metric(self, question: str, response: str, response_time: float,
                          quality_score: float, confidence_score: float,
                          model_type: str = "unknown", user_id: str = "anonymous"):
        """Loggt eine Antwort-Metrik"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'question': question,
            'response_length': len(response),
            'response_time_ms': response_time * 1000,
            'quality_score': quality_score,
            'confidence_score': confidence_score,
            'model_type': model_type,
            'question_length': len(question),
            'response_quality_category': self._categorize_quality(quality_score)
        }

        self.response_metrics.append(metric)

    def _categorize_quality(self, score: float) -> str:
        """Kategorisiert die Antwortqualität"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "acceptable"
        else:
            return "poor"

    def log_user_feedback(self, user_id: str, question: str, response: str,
                         rating: int, feedback_text: str = "", metadata: Optional[Dict] = None):
        """Loggt Nutzer-Feedback"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'question': question,
            'response': response,
            'rating': rating,
            'feedback_text': feedback_text,
            'metadata': metadata or {}
        }

        self.user_feedback.append(feedback)

        # Log bei niedrigen Bewertungen
        if rating <= 2:
            logger.warning(f"👎 Niedrige Bewertung ({rating}/5) für Frage: {question[:50]}...")

    def start_session(self, user_id: str, session_type: str = "interactive") -> str:
        """Startet eine neue Session"""
        session_id = f"{user_id}_{int(time.time())}_{self.session_counter}"
        self.session_counter += 1

        self.active_sessions[session_id] = {
            'user_id': user_id,
            'session_type': session_type,
            'start_time': datetime.now(),
            'interactions': 0,
            'total_response_time': 0,
            'avg_quality_score': 0
        }

        return session_id

    def end_session(self, session_id: str):
        """Beendet eine Session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['end_time'] = datetime.now()
            session['duration'] = (session['end_time'] - session['start_time']).total_seconds()

            # Berechne finale Metriken
            if session['interactions'] > 0:
                session['avg_response_time'] = session['total_response_time'] / session['interactions']

            logger.info(f"📊 Session {session_id} beendet: {session['interactions']} Interaktionen, "
                       f"Dauer: {session['duration']:.1f}s")

            del self.active_sessions[session_id]

    def update_session_metrics(self, session_id: str, response_time: float, quality_score: float):
        """Aktualisiert Session-Metriken"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['interactions'] += 1
            session['total_response_time'] += response_time
            # Laufender Durchschnitt für Qualität
            session['avg_quality_score'] = (
                (session['avg_quality_score'] * (session['interactions'] - 1)) + quality_score
            ) / session['interactions']

    def get_system_health_report(self) -> Dict[str, Any]:
        """Gibt einen System-Health-Report zurück"""
        if not self.system_health:
            return {"status": "no_data"}

        recent_metrics = list(self.system_health)[-10:]  # Letzte 10 Messungen

        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'cpu_avg_percent': statistics.mean(m['cpu_percent'] for m in recent_metrics),
            'memory_avg_percent': statistics.mean(m['memory_percent'] for m in recent_metrics),
            'memory_used_mb': recent_metrics[-1]['memory_used_mb'],
            'disk_usage_percent': recent_metrics[-1]['disk_usage_percent'],
            'active_sessions': len(self.active_sessions),
            'total_metrics_logged': len(self.performance_metrics) + len(self.response_metrics)
        }

    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """Gibt einen Performance-Report für die letzten N Stunden zurück"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filtere Performance-Metriken
        recent_perf = [m for m in self.performance_metrics
                      if datetime.fromisoformat(m['timestamp']) > cutoff_time]

        # Filtere Response-Metriken
        recent_resp = [m for m in self.response_metrics
                      if datetime.fromisoformat(m['timestamp']) > cutoff_time]

        report = {
            'time_range_hours': hours,
            'total_operations': len(recent_perf),
            'successful_operations': sum(1 for m in recent_perf if m['success']),
            'failed_operations': sum(1 for m in recent_perf if not m['success']),
            'avg_response_time_ms': statistics.mean(m['response_time_ms'] for m in recent_resp) if recent_resp else 0,
            'avg_quality_score': statistics.mean(m['quality_score'] for m in recent_resp) if recent_resp else 0,
            'total_responses': len(recent_resp)
        }

        # Berechne Erfolgsrate
        if recent_perf:
            report['success_rate'] = report['successful_operations'] / report['total_operations']
        else:
            report['success_rate'] = 0

        return report

    def get_user_satisfaction_report(self) -> Dict[str, Any]:
        """Gibt einen Bericht über Nutzer-Zufriedenheit zurück"""
        if not self.user_feedback:
            return {"status": "no_feedback"}

        ratings = [f['rating'] for f in self.user_feedback]

        return {
            'total_feedback': len(self.user_feedback),
            'avg_rating': statistics.mean(ratings),
            'median_rating': statistics.median(ratings),
            'rating_distribution': {
                '1_star': sum(1 for r in ratings if r == 1),
                '2_star': sum(1 for r in ratings if r == 2),
                '3_star': sum(1 for r in ratings if r == 3),
                '4_star': sum(1 for r in ratings if r == 4),
                '5_star': sum(1 for r in ratings if r == 5)
            },
            'recent_feedback': list(self.user_feedback)[-5:]  # Letzte 5 Feedbacks
        }

    def _update_aggregated_stats(self):
        """Aktualisiert stündliche und tägliche aggregierte Statistiken"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Performance-Stats pro Stunde
        if self.performance_metrics:
            recent_perf = list(self.performance_metrics)[-100:]  # Letzte 100
            self.hourly_stats[current_hour.isoformat()].append({
                'avg_duration_ms': statistics.mean(m['duration_ms'] for m in recent_perf),
                'success_rate': sum(1 for m in recent_perf if m['success']) / len(recent_perf)
            })

        # Response-Stats pro Tag
        if self.response_metrics:
            recent_resp = list(self.response_metrics)[-200:]  # Letzte 200
            self.daily_stats[current_day.isoformat()].append({
                'avg_quality': statistics.mean(m['quality_score'] for m in recent_resp),
                'avg_response_time': statistics.mean(m['response_time_ms'] for m in recent_resp),
                'total_responses': len(recent_resp)
            })

    def _cleanup_old_data(self):
        """Bereinigt alte aggregierte Daten"""
        cutoff_hour = datetime.now() - timedelta(days=7)  # Behalte 7 Tage
        cutoff_day = datetime.now() - timedelta(days=30)  # Behalte 30 Tage

        # Bereinige stündliche Stats
        to_remove = [k for k in self.hourly_stats.keys()
                    if datetime.fromisoformat(k) < cutoff_hour]
        for k in to_remove:
            del self.hourly_stats[k]

        # Bereinige tägliche Stats
        to_remove = [k for k in self.daily_stats.keys()
                    if datetime.fromisoformat(k) < cutoff_day]
        for k in to_remove:
            del self.daily_stats[k]

    def _load_historical_data(self):
        """Lädt historische Monitoring-Daten"""
        try:
            history_file = os.path.join(self.log_dir, "monitoring_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Lade historische Metriken (begrenze auf max_history)
                if 'performance_metrics' in data:
                    self.performance_metrics.extend(data['performance_metrics'][-self.max_history:])
                if 'response_metrics' in data:
                    self.response_metrics.extend(data['response_metrics'][-self.max_history:])
                if 'user_feedback' in data:
                    self.user_feedback.extend(data['user_feedback'][-self.max_history:])

                logger.info(f"📚 Historische Monitoring-Daten geladen: {len(self.performance_metrics)} Performance, "
                          f"{len(self.response_metrics)} Response, {len(self.user_feedback)} Feedback")

        except Exception as e:
            logger.warning(f"Fehler beim Laden historischer Daten: {e}")

    def _save_historical_data(self):
        """Speichert Monitoring-Daten für Persistenz"""
        try:
            data = {
                'performance_metrics': list(self.performance_metrics),
                'response_metrics': list(self.response_metrics),
                'user_feedback': list(self.user_feedback),
                'hourly_stats': dict(self.hourly_stats),
                'daily_stats': dict(self.daily_stats),
                'last_saved': datetime.now().isoformat()
            }

            history_file = os.path.join(self.log_dir, "monitoring_history.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info("💾 Monitoring-Daten gespeichert")

        except Exception as e:
            logger.error(f"Fehler beim Speichern der Monitoring-Daten: {e}")

    def export_report(self, report_type: str = "full", file_path: Optional[str] = None) -> str:
        """
        Exportiert einen detaillierten Report

        Args:
            report_type: "full", "performance", "health", "satisfaction"
            file_path: Optionaler Pfad für Export-Datei

        Returns:
            JSON-String des Reports
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'report_type': report_type
        }

        if report_type in ["full", "health"]:
            report['system_health'] = self.get_system_health_report()

        if report_type in ["full", "performance"]:
            report['performance_1h'] = self.get_performance_report(hours=1)
            report['performance_24h'] = self.get_performance_report(hours=24)

        if report_type in ["full", "satisfaction"]:
            report['user_satisfaction'] = self.get_user_satisfaction_report()

        if report_type == "full":
            report['active_sessions'] = len(self.active_sessions)
            report['total_metrics'] = {
                'performance': len(self.performance_metrics),
                'response': len(self.response_metrics),
                'feedback': len(self.user_feedback)
            }

        # Export als JSON
        json_report = json.dumps(report, ensure_ascii=False, indent=2)

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_report)
            logger.info(f"📄 Report exportiert nach: {file_path}")

        return json_report