#!/usr/bin/env python3
"""
Verbessertes Cleanup-Skript für ordnungsgemäßes Beenden der Bundeskanzler KI
Verhindert Keyboard-Interrupt-Probleme durch ordnungsgemäßes Monitoring-Cleanup
"""

import signal
import sys
import logging
import threading
import time
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)

class GracefulShutdown:
    """Stellt ordnungsgemäßes Beenden der Anwendung sicher"""

    def __init__(self):
        self.monitoring_system = None
        self.model = None
        self.threads: List[threading.Thread] = []
        self.cleanup_callbacks: List[Callable] = []
        self._shutdown_event = threading.Event()
        self._cleanup_done = False  # Verhindert rekursive Cleanup-Aufrufe
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Richtet Signal-Handler für ordnungsgemäßes Beenden ein"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Signal-Handler für ordnungsgemäßes Beenden"""
        logger.info(f"🛑 Signal {signum} empfangen - Starte ordnungsgemäßes Beenden...")
        self._shutdown_event.set()

        # Führe Cleanup durch (nur einmal)
        if not self._cleanup_done:
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"⚠️ Fehler beim Cleanup: {e}")

        # Beende alle Threads
        self._stop_all_threads()

        # Warte kurz um Cleanup zu vollenden
        time.sleep(0.1)

        logger.info("✅ Programm ordnungsgemäß beendet")
        sys.exit(0)

    def register_monitoring(self, monitoring_system):
        """Registriert das Monitoring-System für Cleanup"""
        self.monitoring_system = monitoring_system

    def register_model(self, model):
        """Registriert das Modell für Cleanup"""
        self.model = model

    def register_thread(self, thread: threading.Thread):
        """Registriert einen Thread für ordnungsgemäßes Beenden"""
        self.threads.append(thread)

    def add_cleanup_callback(self, callback: Callable):
        """Fügt einen Cleanup-Callback hinzu"""
        self.cleanup_callbacks.append(callback)

    def _stop_all_threads(self):
        """Stoppt alle registrierten Threads"""
        for thread in self.threads:
            if thread.is_alive():
                logger.info(f"🛑 Beende Thread: {thread.name}")
                thread.join(timeout=2.0)  # Warte max 2 Sekunden

    def cleanup(self):
        """Führt ordnungsgemäßes Cleanup durch"""
        global _graceful_shutdown_completed

        if self._cleanup_done:
            logger.debug("🧹 Cleanup bereits durchgeführt - überspringe")
            return

        self._cleanup_done = True
        _graceful_shutdown_completed = True  # Setze globale Flag
        logger.info("🧹 Starte ordnungsgemäßes Cleanup...")

        try:
            # Führe alle Cleanup-Callbacks aus
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"⚠️ Cleanup-Callback Fehler: {e}")

            # Stoppe Monitoring-System
            if self.monitoring_system:
                logger.info("📊 Beende Monitoring-System...")
                try:
                    self.monitoring_system.stop_monitoring()
                    if hasattr(self.monitoring_system, '__exit__'):
                        self.monitoring_system.__exit__(None, None, None)
                except Exception as e:
                    logger.error(f"⚠️ Monitoring Cleanup Fehler: {e}")

            # Cleanup Modell - aber vermeide rekursiven Aufruf
            if self.model:
                logger.info("🧠 Cleanup Modell...")
                try:
                    # Verwende nur direkte Cleanup-Methoden, nicht __del__
                    if hasattr(self.model, 'cleanup') and callable(getattr(self.model, 'cleanup')):
                        self.model.cleanup()
                    # Vermeide __del__ Aufruf um Rekursion zu verhindern
                except Exception as e:
                    logger.error(f"⚠️ Modell Cleanup Fehler: {e}")

        except Exception as e:
            logger.error(f"⚠️ Allgemeiner Cleanup-Fehler: {e}")

    def wait_for_shutdown(self):
        """Wartet auf Shutdown-Signal"""
        self._shutdown_event.wait()

def is_graceful_shutdown_completed() -> bool:
    """Prüft ob bereits ein Graceful Shutdown durchgeführt wurde"""
    global _graceful_shutdown_completed
    return _graceful_shutdown_completed# Globale Instanz
_shutdown_handler = None
_graceful_shutdown_completed = False  # Globale Flag für anderen Cleanup-Code

def get_shutdown_handler() -> GracefulShutdown:
    """Gibt globale Shutdown-Handler-Instanz zurück"""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdown()
    return _shutdown_handler

def setup_graceful_shutdown(monitoring_system=None, model=None):
    """Richtet ordnungsgemäßes Shutdown ein"""
    handler = get_shutdown_handler()
    if monitoring_system:
        handler.register_monitoring(monitoring_system)
    if model:
        handler.register_model(model)
    return handler

# Automatisches Cleanup beim Modul-Import
import atexit

def _cleanup_at_exit():
    """Cleanup beim Programmende"""
    try:
        handler = get_shutdown_handler()
        if not handler._cleanup_done:
            logger.info("🧹 Führe finalen Cleanup durch...")
            handler.cleanup()
            logger.info("✅ Finaler Cleanup abgeschlossen")
        else:
            logger.debug("🧹 Cleanup bereits durchgeführt - überspringe atexit-Handler")
    except Exception as e:
        logger.warning(f"⚠️ Fehler beim finalen Cleanup: {e}")

atexit.register(_cleanup_at_exit)