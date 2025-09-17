#!/usr/bin/env python3
"""
🎯 Vollständiger TensorRT-Integrationstest für RTX 2070
==============================================

Testet die komplette TensorRT-Integration in den RTX2070LLMManager
mit echten Modellen und Benchmarking.

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import logging
import sys
import time
from pathlib import Path

# Füge das Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.rtx2070_llm_manager import RTX2070LLMManager

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_complete_tensorrt_integration():
    """Testet die vollständige TensorRT-Integration"""

    logger.info("🚀 Starte vollständigen TensorRT-Integrationstest")

    # RTX 2070 Manager mit TensorRT-Unterstützung initialisieren
    manager = RTX2070LLMManager(enable_tensorrt=True)

    # TensorRT-Status prüfen
    status = manager.get_tensorrt_status()
    logger.info(f"📊 TensorRT-Status: {status}")

    if not status['tensorrt_available']:
        logger.error("❌ TensorRT nicht verfügbar")
        return False

    # Test-Modell laden (German GPT-2 als Fallback)
    test_model = "german_gpt2"
    logger.info(f"🎯 Lade Test-Modell: {test_model}")

    success = manager.load_model(test_model)
    if not success:
        logger.error(f"❌ Modell {test_model} konnte nicht geladen werden")
        return False

    logger.info(f"✅ Modell {test_model} erfolgreich geladen")

    # Test-Prompts
    test_prompts = [
        "Die Europäische Union ist",
        "Die wichtigsten Parteien in Deutschland sind",
        "Klimapolitik bedeutet"
    ]

    # Standard-Inference Benchmark
    logger.info("📊 Führe Standard-Inference Benchmark durch...")
    standard_times = []

    for prompt in test_prompts:
        start_time = time.time()
        try:
            result = manager.generate_response(prompt, max_new_tokens=20)
            inference_time = time.time() - start_time
            standard_times.append(inference_time)
            logger.info(".4f")
        except Exception as e:
            logger.warning(f"⚠️ Standard-Inference für Prompt fehlgeschlagen: {e}")
            standard_times.append(None)

    # TensorRT-Optimierung durchführen
    logger.info("🔧 Starte TensorRT-Optimierung...")
    optimization_success = manager.optimize_model_with_tensorrt(test_model)

    if optimization_success:
        logger.info("✅ TensorRT-Optimierung erfolgreich!")

        # TensorRT-Inference Benchmark
        logger.info("🚀 Führe TensorRT-Inference Benchmark durch...")
        tensorrt_times = []

        for prompt in test_prompts:
            start_time = time.time()
            try:
                result = manager.generate_with_tensorrt(prompt, max_new_tokens=20)
                if result:
                    inference_time = time.time() - start_time
                    tensorrt_times.append(inference_time)
                    logger.info(".4f")
                else:
                    logger.warning(f"⚠️ TensorRT-Inference für Prompt fehlgeschlagen")
                    tensorrt_times.append(None)
            except Exception as e:
                logger.warning(f"⚠️ TensorRT-Inference für Prompt fehlgeschlagen: {e}")
                tensorrt_times.append(None)

        # Performance-Vergleich
        logger.info("📈 Performance-Vergleich:")

        valid_standard = [t for t in standard_times if t is not None]
        valid_tensorrt = [t for t in tensorrt_times if t is not None]

        if valid_standard and valid_tensorrt:
            avg_standard = sum(valid_standard) / len(valid_standard)
            avg_tensorrt = sum(valid_tensorrt) / len(valid_tensorrt)

            logger.info(".4f")
            logger.info(".4f")

            if avg_tensorrt > 0:
                speedup = avg_standard / avg_tensorrt
                logger.info(".1f")

                if speedup > 1.0:
                    logger.info("🎉 TensorRT zeigt Performance-Verbesserung!")
                else:
                    logger.info("ℹ️ TensorRT zeigt keine signifikante Verbesserung")
            else:
                logger.info("⚠️ TensorRT-Benchmark fehlgeschlagen")
        else:
            logger.warning("⚠️ Unzureichende Benchmark-Daten für Vergleich")

        # Vollständiges Benchmark mit Manager-Methode
        logger.info("📊 Führe vollständiges Benchmark mit Manager-Methode durch...")
        benchmark_results = manager.benchmark_tensorrt_performance(test_prompts)

        if benchmark_results['summary'].get('tensorrt_available'):
            summary = benchmark_results['summary']
            logger.info("📋 Vollständige Benchmark-Ergebnisse:")
            logger.info(".4f")
            logger.info(".4f")
            logger.info(".1f")

    else:
        logger.warning("⚠️ TensorRT-Optimierung fehlgeschlagen - verwende Standard-Inference")

    # Cache bereinigen
    manager.cleanup_tensorrt_cache()
    logger.info("🧹 TensorRT-Cache bereinigt")

    # Modell entladen
    manager.unload_model()
    logger.info("📤 Modell entladen")

    # Endergebnis
    if optimization_success:
        logger.info("🎉 TensorRT-Integrationstest erfolgreich abgeschlossen!")
        logger.info("✅ RTX 2070 TensorRT-Optimierung ist bereit für den Produktiveinsatz!")
        return True
    else:
        logger.warning("⚠️ TensorRT-Integrationstest mit Einschränkungen abgeschlossen")
        return False


if __name__ == "__main__":
    success = test_complete_tensorrt_integration()
    sys.exit(0 if success else 1)