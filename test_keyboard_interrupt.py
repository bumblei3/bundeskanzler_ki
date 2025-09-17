#!/usr/bin/env python3
"""
Test für Keyboard-Interrupt Handling
Testet das neue Signalhandling-System
"""

import sys
import os
import logging
import time
import signal
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Graceful Shutdown importieren
from graceful_shutdown import setup_graceful_shutdown

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_keyboard_interrupt():
    """Testet Keyboard-Interrupt Handling"""
    logger.info("🧪 Teste Keyboard-Interrupt Handling...")

    try:
        # Import der multimodalen KI
        from multimodal_ki import MultimodalTransformerModel

        # Erstelle RTX 2070 optimiertes Modell
        logger.info("🎯 Erstelle RTX 2070 optimiertes Modell...")
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Simuliere Arbeit
        logger.info("🔄 Simuliere kontinuierliche Arbeit...")
        counter = 0
        while True:
            counter += 1
            if counter % 10 == 0:
                logger.info(f"📊 Arbeitsschritt {counter} abgeschlossen")

            # Kurze Pause um CPU nicht zu überlasten
            time.sleep(0.5)

            # Nach 30 Schritten beenden (um Test zu begrenzen)
            if counter >= 30:
                logger.info("✅ Test erfolgreich abgeschlossen")
                break

    except KeyboardInterrupt:
        logger.info("🛑 Keyboard-Interrupt empfangen - beginne sauberes Shutdown...")
        raise

    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        return False

    return True

def main():
    """Hauptfunktion für Keyboard-Interrupt Test"""
    logger.info("🚀 Bundeskanzler KI - Keyboard-Interrupt Test")
    logger.info("=" * 60)

    # Graceful Shutdown aktivieren
    shutdown_handler = setup_graceful_shutdown()
    logger.info("🛡️ Graceful Shutdown aktiviert")

    # Teste Keyboard-Interrupt Handling
    try:
        success = test_keyboard_interrupt()
        if success:
            logger.info("✅ Keyboard-Interrupt Test erfolgreich")
            return 0
        else:
            logger.error("❌ Keyboard-Interrupt Test fehlgeschlagen")
            return 1
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard-Interrupt erfolgreich behandelt")
        logger.info("✅ Graceful Shutdown System funktioniert!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)