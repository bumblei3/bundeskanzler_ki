#!/usr/bin/env python3
"""
🎯 TensorRT-Optimierung für RTX 2070 Modelle
===========================================

Optimiert die verfügbaren Modelle (Mistral, Llama, German GPT-2) mit TensorRT
für maximale Performance auf der RTX 2070 GPU.

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

from core.rtx2070_llm_manager import RTX2070LLMManager, RTX2070_MODELS

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tensorrt_optimization():
    """Testet TensorRT-Optimierung für alle verfügbaren Modelle"""

    logger.info("🚀 Starte TensorRT-Optimierungstest für RTX 2070 Modelle")

    # RTX 2070 Manager initialisieren
    manager = RTX2070LLMManager(enable_tensorrt=True)

    # TensorRT-Status prüfen
    status = manager.get_tensorrt_status()
    logger.info(f"📊 TensorRT-Status: {status}")

    if not status['tensorrt_available']:
        logger.error("❌ TensorRT nicht verfügbar - beende Test")
        return False

    # Test-Prompts für Benchmarking
    test_prompts = [
        "Was ist die aktuelle Klimapolitik Deutschlands?",
        "Erklären Sie die Europäische Union kurz.",
        "Was sind die wichtigsten politischen Parteien in Deutschland?",
        "Wie funktioniert das deutsche Wahlsystem?"
    ]

    results = {}

    # Jedes verfügbare Modell testen
    for model_key, model_config in RTX2070_MODELS.items():
        logger.info(f"🎯 Optimiere Modell: {model_config.name} ({model_key})")

        try:
            # Modell laden
            success = manager.load_model(model_key)
            if not success:
                logger.warning(f"⚠️ Modell {model_key} konnte nicht geladen werden")
                continue

            logger.info(f"✅ Modell {model_key} erfolgreich geladen")

            # TensorRT-Optimierung durchführen
            logger.info(f"🔧 Starte TensorRT-Optimierung für {model_key}...")
            optimization_success = manager.optimize_model_with_tensorrt(model_key)

            if optimization_success:
                logger.info(f"✅ TensorRT-Optimierung für {model_key} erfolgreich!")

                # Performance-Benchmark durchführen
                logger.info(f"📊 Führe Performance-Benchmark für {model_key} durch...")
                benchmark_results = manager.benchmark_tensorrt_performance(test_prompts)

                results[model_key] = {
                    'optimization_success': True,
                    'benchmark': benchmark_results
                }

                # Speichere Benchmark-Ergebnisse
                if benchmark_results['summary'].get('tensorrt_available'):
                    logger.info(f"📈 {model_key} Speedup: {benchmark_results['summary']['avg_speedup']:.1f}x")

            else:
                logger.warning(f"⚠️ TensorRT-Optimierung für {model_key} fehlgeschlagen")
                results[model_key] = {
                    'optimization_success': False,
                    'error': 'Optimization failed'
                }

            # Modell entladen um VRAM freizugeben
            manager.unload_model()

        except Exception as e:
            logger.error(f"❌ Fehler bei Modell {model_key}: {e}")
            results[model_key] = {
                'optimization_success': False,
                'error': str(e)
            }

    # Zusammenfassung ausgeben
    logger.info("📋 TensorRT-Optimierung Zusammenfassung:")
    logger.info("=" * 50)

    successful_optimizations = 0
    total_speedup = 0
    speedup_count = 0

    for model_key, result in results.items():
        model_name = RTX2070_MODELS[model_key].name
        if result['optimization_success']:
            successful_optimizations += 1
            if 'benchmark' in result and result['benchmark']['summary'].get('tensorrt_available'):
                speedup = result['benchmark']['summary']['avg_speedup']
                total_speedup += speedup
                speedup_count += 1
                logger.info(f"✅ {model_name}: Optimiert (Speedup: {speedup:.1f}x)")
            else:
                logger.info(f"✅ {model_name}: Optimiert (Benchmark nicht verfügbar)")
        else:
            logger.info(f"❌ {model_name}: Fehler - {result.get('error', 'Unbekannter Fehler')}")

    if speedup_count > 0:
        avg_speedup = total_speedup / speedup_count
        logger.info(f"📊 Durchschnittlicher Speedup: {avg_speedup:.1f}x")
        logger.info(f"🎯 Erfolgreiche Optimierungen: {successful_optimizations}/{len(RTX2070_MODELS)}")

    # TensorRT-Cache bereinigen
    manager.cleanup_tensorrt_cache()

    return successful_optimizations > 0


if __name__ == "__main__":
    success = test_tensorrt_optimization()
    sys.exit(0 if success else 1)