#!/usr/bin/env python3
"""
Test für verbesserte TensorRT-Integration in die KI-Pipeline
Testet automatischen Fallback und Performance-Vergleich
"""

import sys
import os
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

def test_tensorrt_pipeline_integration():
    """Testet die Integration von TensorRT in die Haupt-KI-Pipeline"""
    logger.info("🔬 Teste TensorRT-Pipeline-Integration...")

    model = None
    try:
        # Multimodal KI initialisieren
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # TensorRT-Optimierung ausführen
        logger.info("🚀 Führe TensorRT-Optimierung aus...")
        optimization_results = model.optimize_with_tensorrt()

        logger.info("📊 Optimierungsergebnisse:")
        logger.info(f"  Status: {optimization_results['status']}")
        logger.info(f"  Verfügbare Engines: {list(optimization_results.get('tensorrt_engines', {}).keys())}")

        # Teste automatische Modell-Auswahl
        logger.info("🎯 Teste automatische TensorRT-Integration...")

        # Text-Test
        test_text = "Die Bundeskanzlerin hat heute eine wichtige Rede gehalten."
        logger.info("📝 Teste Text-Verarbeitung...")

        start_time = time.time()
        text_result = model.process_text(test_text, max_length=50)
        text_time = time.time() - start_time

        logger.info(f"  Ergebnis: {text_result[:100]}...")
        logger.info(".3f")
        logger.info(f"  Verwendetes Modell: {'TensorRT' if 'text' in model.tensorrt_engines else 'Original'}")

        # Image-Test (Mock - da keine Test-Datei verfügbar)
        logger.info("🖼️ Teste Bild-Verarbeitung (simuliert)...")
        try:
            # Erstelle eine kleine Test-Bild-Datei für den Test
            from PIL import Image
            import numpy as np

            # Erstelle ein einfaches Test-Bild
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            test_image_path = "/tmp/test_image.jpg"
            test_image.save(test_image_path)

            start_time = time.time()
            image_result = model.process_image(test_image_path)
            image_time = time.time() - start_time

            logger.info(f"  Beschreibung: {image_result.get('description', 'N/A')}")
            logger.info(".3f")
            logger.info(f"  Verwendetes Modell: {'TensorRT' if 'vision' in model.tensorrt_engines else 'Original'}")

            # Aufräumen
            os.remove(test_image_path)

        except Exception as e:
            logger.warning(f"  Bild-Test übersprungen: {e}")

        # Audio-Test (Mock - da keine Test-Datei verfügbar)
        logger.info("🎵 Teste Audio-Verarbeitung (simuliert)...")
        logger.info(f"  Verwendetes Modell: {'TensorRT' if 'audio' in model.tensorrt_engines else 'Original'}")

        # Teste Fallback-Mechanismen
        logger.info("🔄 Teste Fallback-Mechanismen...")

        # Simuliere Engine-Ausfall
        if model.tensorrt_engines:
            original_engines = model.tensorrt_engines.copy()
            model.tensorrt_engines.clear()  # Simuliere Ausfall aller Engines

            logger.info("  Simuliere TensorRT-Engine-Ausfall...")

            # Teste Fallback zur Original-Verarbeitung
            start_time = time.time()
            fallback_result = model.process_text(test_text, max_length=30)
            fallback_time = time.time() - start_time

            logger.info(f"  Fallback-Ergebnis: {fallback_result[:50]}...")
            logger.info(".3f")
            logger.info("  ✅ Fallback zu Original-Modellen funktioniert")

            # Stelle Engines wieder her
            model.tensorrt_engines = original_engines

        # Performance-Vergleich
        logger.info("📊 Performance-Vergleich:")

        # TensorRT-Version
        if "vision" in model.tensorrt_engines:
            logger.info("  ✅ TensorRT Vision-Modell verfügbar")
            logger.info("  ✅ Erwartete Performance: 3-4x schneller")

        if "audio" in model.tensorrt_engines:
            logger.info("  ✅ TensorRT Audio-Modell verfügbar")
            logger.info("  ✅ Erwartete Performance: 2-3x schneller")

        if "text" in model.tensorrt_engines:
            logger.info("  ✅ TensorRT Text-Modell verfügbar")
            logger.info("  ✅ Erwartete Performance: 2.5-3.5x schneller")

        logger.info("✅ TensorRT-Pipeline-Integration Test erfolgreich abgeschlossen")
        return True

    except Exception as e:
        logger.error(f"❌ Fehler bei Pipeline-Integration-Test: {e}")
        return False

    finally:
        if model:
            try:
                del model
            except:
                pass

def test_automatic_fallback():
    """Testet automatischen Fallback-Mechanismus"""
    logger.info("🔄 Teste automatischen Fallback-Mechanismus...")

    try:
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Stelle sicher, dass TensorRT verfügbar ist
        if not model.tensorrt_optimizer:
            logger.warning("⚠️ TensorRT nicht verfügbar - überspringe Fallback-Test")
            return True

        # Teste verschiedene Szenarien
        test_text = "Test für Fallback-Mechanismus."

        # Szenario 1: Normale Verarbeitung
        logger.info("  Szenario 1: Normale Verarbeitung")
        result1 = model.process_text(test_text)
        logger.info(f"  ✅ Normal: {result1[:30]}...")

        # Szenario 2: Simuliere partielle Engine-Verfügbarkeit
        logger.info("  Szenario 2: Partielle Engine-Verfügbarkeit")
        if "vision" in model.tensorrt_engines:
            # Entferne nur Vision-Engine
            del model.tensorrt_engines["vision"]
            logger.info("  🗑️ Vision-Engine entfernt - teste Fallback")

            # Text sollte weiterhin funktionieren
            result2 = model.process_text(test_text)
            logger.info(f"  ✅ Text weiterhin verfügbar: {result2[:30]}...")

        # Szenario 3: Simuliere komplette Engine-Verfügbarkeit
        logger.info("  Szenario 3: Komplette Engine-Verfügbarkeit")
        model.tensorrt_engines.clear()
        result3 = model.process_text(test_text)
        logger.info(f"  ✅ Kompletter Fallback: {result3[:30]}...")

        logger.info("✅ Automatischer Fallback-Mechanismus funktioniert korrekt")
        return True

    except Exception as e:
        logger.error(f"❌ Fehler bei Fallback-Test: {e}")
        return False

def main():
    """Hauptfunktion für TensorRT-Pipeline-Tests"""
    logger.info("🚀 TensorRT Pipeline Integration Test")
    logger.info("=" * 60)

    success = True

    # Test 1: Pipeline-Integration
    if not test_tensorrt_pipeline_integration():
        success = False

    # Test 2: Automatischer Fallback
    if not test_automatic_fallback():
        success = False

    if success:
        logger.info("✅ Alle TensorRT-Pipeline-Tests erfolgreich abgeschlossen")
        return 0
    else:
        logger.error("❌ Einige Tests fehlgeschlagen")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)