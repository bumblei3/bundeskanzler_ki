#!/usr/bin/env python3
"""
🚀 Bundeskanzler KI - Einfaches Debugging System
===============================================

Einfaches aber effektives Debugging-System für die Bundeskanzler KI.

Features:
- 📝 Strukturiertes Logging
- 📊 Performance-Monitoring
- 🏥 Health-Checks
- 🔧 Debug-Modi

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import logging
import logging.handlers
import time
import threading
from datetime import datetime
from pathlib import Path
import psutil
import requests

# Konfiguration
DEBUG_CONFIG = {
    "log_level": "INFO",
    "log_to_file": True,
    "log_directory": "logs",
    "performance_monitoring": True,
    "debug_modes": {
        "api_debug": False,
        "web_debug": False,
        "ki_debug": False,
        "performance_debug": False
    }
}

class DebugSystem:
    """Einfaches Debugging-System"""

    def __init__(self):
        self.logger = self._setup_logger()
        self.performance_data = []
        self.debug_modes = DEBUG_CONFIG["debug_modes"].copy()

        if DEBUG_CONFIG["performance_monitoring"]:
            self.monitoring_thread = threading.Thread(target=self._performance_monitor, daemon=True)
            self.monitoring_thread.start()

    def _setup_logger(self):
        """Konfiguriert den Logger"""
        logger = logging.getLogger("bundeskanzler_debug")
        logger.setLevel(getattr(logging, DEBUG_CONFIG["log_level"]))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Handler entfernen falls vorhanden
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Konsolen-Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Datei-Handler
        if DEBUG_CONFIG["log_to_file"]:
            log_dir = Path(DEBUG_CONFIG["log_directory"])
            log_dir.mkdir(exist_ok=True)

            log_file = log_dir / "debug.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _performance_monitor(self):
        """Überwacht System-Performance"""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                perf_data = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3)
                }

                self.performance_data.append(perf_data)

                # Alert bei hoher Auslastung
                if cpu_percent > 90:
                    self.logger.warning(f"🚨 Hohe CPU-Auslastung: {cpu_percent:.1f}%")
                if memory.percent > 80:
                    self.logger.warning(f"🚨 Hohe Memory-Auslastung: {memory.percent:.1f}%")

                # Nur letzte 100 Einträge behalten
                if len(self.performance_data) > 100:
                    self.performance_data = self.performance_data[-100:]

            except Exception as e:
                self.logger.error(f"Performance-Monitoring Fehler: {e}")

            time.sleep(10)

    def enable_debug_mode(self, mode: str):
        """Aktiviert einen Debug-Modus"""
        if mode in self.debug_modes:
            self.debug_modes[mode] = True
            self.logger.info(f"Debug-Modus {mode} aktiviert")
        else:
            self.logger.warning(f"Unbekannter Debug-Modus: {mode}")

    def disable_debug_mode(self, mode: str):
        """Deaktiviert einen Debug-Modus"""
        if mode in self.debug_modes:
            self.debug_modes[mode] = False
            self.logger.info(f"Debug-Modus {mode} deaktiviert")

    def is_debug_enabled(self, mode: str) -> bool:
        """Überprüft, ob ein Debug-Modus aktiviert ist"""
        return self.debug_modes.get(mode, False)

    def log_api_request(self, method: str, url: str, status_code: int, duration: float):
        """Log für API-Requests"""
        if self.is_debug_enabled("api_debug"):
            level = logging.INFO if status_code < 400 else logging.WARNING
            self.logger.log(level, f"API {method} {url} -> {status_code} ({duration:.3f}s)")

    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log für Performance-Messungen mit zusätzlichen Parametern"""
        message = f"Performance: {operation} took {duration:.3f}s"
        if kwargs:
            extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} [{extra_info}]"

        if self.is_debug_enabled("performance_debug"):
            self.logger.info(message)
        else:
            # Immer Performance-Daten sammeln, auch wenn Debug ausgeschaltet
            self.performance_data.append({
                "operation": operation,
                "duration": duration,
                "timestamp": datetime.now(),
                "extra": kwargs
            })

    def check_api_health(self) -> dict:
        """Health-Check für die KI-API"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "message": "API verfügbar"}
            else:
                return {"status": "unhealthy", "message": f"API Status {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"API Fehler: {str(e)}"}

    def get_performance_stats(self) -> dict:
        """Gibt Performance-Statistiken zurück"""
        if not self.performance_data:
            return {"error": "Keine Performance-Daten verfügbar"}

        latest = self.performance_data[-1]
        avg_cpu = sum(d["cpu_percent"] for d in self.performance_data[-10:]) / min(10, len(self.performance_data))

        return {
            "current": latest,
            "average_cpu": avg_cpu,
            "data_points": len(self.performance_data)
        }

    def debug(self, message: str, **kwargs):
        """Debug-Log mit zusätzlichen Parametern"""
        if kwargs:
            extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} [{extra_info}]"
        self.logger.debug(message)

    def info(self, message: str, **kwargs):
        """Info-Log mit zusätzlichen Parametern"""
        if kwargs:
            extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} [{extra_info}]"
        self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """Warning-Log mit zusätzlichen Parametern"""
        if kwargs:
            extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} [{extra_info}]"
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Error-Log mit zusätzlichen Parametern"""
        if kwargs:
            extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} [{extra_info}]"
        self.logger.error(message)

# Globale Instanz
debug_system = DebugSystem()

if __name__ == "__main__":
    # Beispiel-Verwendung
    debug_system.info("Debug-System initialisiert")

    # API-Health-Check
    health = debug_system.check_api_health()
    debug_system.info(f"API Health: {health}")

    # Performance-Stats
    stats = debug_system.get_performance_stats()
    debug_system.info(f"Performance Stats: {stats}")

    print("✅ Einfaches Debugging-System bereit!")
    print("📊 Performance-Monitoring aktiv")
    print("📝 Logs werden in logs/debug.log gespeichert")