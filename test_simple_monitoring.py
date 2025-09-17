#!/usr/bin/env python3
"""
Einfacher Test für Monitoring-System Shutdown
"""

import sys
import os
import logging
import time
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_simple_monitoring():
    """Testet einfaches Monitoring ohne komplexe Modelle"""
    logger.info("🧪 Teste einfaches Monitoring-System...")

    try:
        # Import des Monitoring-Systems
        from core.local_monitoring import LocalMonitoringSystem

        # Erstelle Monitoring-System
        logger.info("🎯 Erstelle Monitoring-System...")
        monitoring = LocalMonitoringSystem()

        # Starte Monitoring
        logger.info("📊 Starte Monitoring...")
        monitoring.start_monitoring(interval_seconds=2.0)

        # Simuliere Arbeit
        logger.info("🔄 Simuliere Arbeit für 10 Sekunden...")
        for i in range(20):
            logger.info(f"📊 Arbeitsschritt {i+1}/20")
            time.sleep(0.5)

        # Stoppe Monitoring
        logger.info("🛑 Stoppe Monitoring...")
        monitoring.stop_monitoring()

        logger.info("✅ Einfacher Monitoring-Test erfolgreich")
        return True

    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        return False

def main():
    """Hauptfunktion"""
    logger.info("🚀 Einfacher Monitoring-Test")
    logger.info("=" * 40)

    if test_simple_monitoring():
        logger.info("✅ Test erfolgreich abgeschlossen")
        return 0
    else:
        logger.error("❌ Test fehlgeschlagen")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)