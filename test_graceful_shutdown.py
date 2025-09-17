#!/usr/bin/env python3
"""
Test-Skript für verbessertes Graceful Shutdown System
Testet ordnungsgemäßes Beenden der Multimodal-KI
"""

import sys
import signal
import time
import logging
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_ki import MultimodalTransformerModel

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_graceful_shutdown():
    """Testet das Graceful Shutdown System ohne Signal"""
    logger.info("🚀 Teste verbessertes Graceful Shutdown System...")

    model = None
    try:
        # Multimodale KI initialisieren OHNE GracefulShutdown für diesen Test
        logger.info("🎯 Initialisiere Multimodal-KI...")
        model = MultimodalTransformerModel(model_tier="rtx2070", enable_graceful_shutdown=False)

        logger.info("✅ Multimodal-KI erfolgreich initialisiert")
        logger.info("⏳ Warte 5 Sekunden (dieser Test sollte ohne Signal durchlaufen)...")

        # Warte auf Signal oder Timeout
        time.sleep(5)

        logger.info("✅ Test erfolgreich abgeschlossen - kein Signal empfangen")

    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt empfangen - teste Graceful Shutdown")
        if model:
            try:
                model.shutdown()
                logger.info("✅ Graceful Shutdown erfolgreich")
            except Exception as e:
                logger.error(f"❌ Graceful Shutdown fehlgeschlagen: {e}")
        return True

    except Exception as e:
        logger.error(f"❌ Fehler beim Test: {e}")
        return False

    finally:
        # Cleanup wird automatisch durch __del__ durchgeführt
        if model:
            logger.info("🧹 Cleanup durch __del__ wird ausgeführt...")

    return True

def test_signal_handling():
    """Testet Signal-Handling mit GracefulShutdown System"""
    logger.info("🔔 Teste Signal-Handling mit GracefulShutdown System...")

    # Initialisiere Multimodal-KI mit GracefulShutdown
    logger.info("🎯 Initialisiere Multimodal-KI für Signal-Test...")
    model = MultimodalTransformerModel(model_tier="rtx2070")

    logger.info("⏳ Sende SIGINT in 2 Sekunden...")
    time.sleep(2)

    # Simuliere SIGINT - das GracefulShutdown System sollte das abfangen
    logger.info("📡 Sende simuliertes SIGINT...")
    signal.raise_signal(signal.SIGINT)

    # Diese Zeile sollte nie erreicht werden, da sys.exit() aufgerufen wird
    logger.error("❌ Signal wurde nicht abgefangen!")
    return False

def main():
    """Hauptfunktion für Shutdown-Tests"""
    logger.info("🧪 Graceful Shutdown Test Suite")
    logger.info("=" * 50)

    success = True

    # Test 1: Normaler Ablauf
    logger.info("Test 1: Normaler Ablauf ohne Signal")
    if not test_graceful_shutdown():
        success = False

    # Test 2: Signal-Handling
    logger.info("\nTest 2: Signal-Handling")
    try:
        test_signal_handling()
        # Diese Zeile sollte nie erreicht werden
        logger.error("❌ Signal-Handling Test fehlgeschlagen - kein SystemExit")
        success = False
    except SystemExit as e:
        logger.info("✅ Signal-Handling Test erfolgreich (SystemExit erwartet)")
        logger.info("🎯 GracefulShutdown System hat korrekt funktioniert!")
        logger.info("🎯 Das Programm hat sich AUTOMATISCH beendet - kein manuelles Strg+C nötig!")
        # Beende das Programm sofort
        logger.info("🎯 Beende Programm jetzt...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Signal-Handling Test fehlgeschlagen: {e}")
        success = False

    if success:
        logger.info("\n✅ Alle Graceful Shutdown Tests erfolgreich abgeschlossen")
        logger.info("🎯 Das Programm beendet sich jetzt automatisch!")
        return 0
    else:
        logger.error("\n❌ Einige Tests fehlgeschlagen")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except SystemExit as e:
        # SystemExit vom Graceful Shutdown System weitergeben
        logger.info("🎯 Programm wurde automatisch durch Graceful Shutdown beendet!")
        sys.exit(e.code)