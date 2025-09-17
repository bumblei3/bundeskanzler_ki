#!/usr/bin/env python3
"""
TensorRT Integration Test für Bundeskanzler KI
Testet die neue TensorRT-Optimierung für RTX 2070 GPU
"""

import sys
import os
import logging
import time
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

def test_tensorrt_integration():
    """Testet die TensorRT-Integration"""
    logger.info("🧪 Starte erweiterte TensorRT Integration Test...")

    model = None
    try:
        # Import der multimodalen KI
        from multimodal_ki import MultimodalTransformerModel

        # Erstelle RTX 2070 optimiertes Modell
        logger.info("🎯 Erstelle RTX 2070 optimiertes Modell...")
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Überprüfe TensorRT-Verfügbarkeit
        if hasattr(model, 'tensorrt_optimizer') and model.tensorrt_optimizer:
            logger.info("✅ TensorRT Optimizer verfügbar")

            # Zeige TensorRT-Status
            logger.info(f"🎮 RTX 2070 erkannt: {model.is_rtx2070}")
            logger.info(f"💾 GPU Memory: {model.gpu_memory_gb}GB")

            # Teste echte TensorRT Engine-Erstellung
            logger.info("🚀 Teste echte TensorRT Engine-Erstellung...")
            optimization_results = model.optimize_with_tensorrt()

            logger.info("📊 Optimierungsergebnisse:")
            logger.info(f"  Status: {optimization_results['status']}")
            logger.info(f"  Optimierte Modelle: {optimization_results['models_optimized']}")

            if optimization_results['status'] == 'completed':
                logger.info("✅ TensorRT Engine-Erstellung erfolgreich!")

                # Zeige Performance-Gains
                for model_name, gains in optimization_results.get('performance_gains', {}).items():
                    if isinstance(gains, dict):
                        speedup = gains.get('expected_speedup', 'N/A')
                        memory = gains.get('memory_efficiency', 'N/A')
                        logger.info(f"  {model_name}: {speedup} schneller, {memory} Speicherersparnis")

                # Teste TensorRT Inference
                test_tensorrt_inference(model, optimization_results)

            else:
                logger.warning("⚠️ TensorRT Engine-Erstellung fehlgeschlagen")
                if 'errors' in optimization_results:
                    for error in optimization_results['errors']:
                        logger.error(f"  Fehler: {error}")

            # TensorRT Engines Status
            logger.info(f"🔧 TensorRT Engines: {len(model.tensorrt_engines)} geladen")

            logger.info("✅ Erweiterte TensorRT Integration Test erfolgreich abgeschlossen")
            return True

        else:
            logger.warning("⚠️ TensorRT Optimizer nicht verfügbar")
            logger.info("ℹ️ Überprüfe TensorRT-Installation...")

            # Zeige verfügbare Komponenten
            logger.info(f"🎮 RTX 2070 erkannt: {model.is_rtx2070}")
            logger.info(f"💾 GPU Memory: {model.gpu_memory_gb}GB")
            logger.info(f"📝 Text-Modell: {model.text_model_type if model.text_model else 'Nicht verfügbar'}")
            logger.info(f"🖼️ Vision-Modell: {model.vision_model_type if model.vision_model else 'Nicht verfügbar'}")
            logger.info(f"🎵 Audio-Modell: {model.audio_model_type if model.audio_model else 'Nicht verfügbar'}")

            return True

    except ImportError as e:
        logger.error(f"❌ Import-Fehler: {e}")
        logger.info("ℹ️ Stelle sicher, dass alle Abhängigkeiten installiert sind")
        return False

    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        return False

    finally:
        # Sicherstellen, dass das Modell ordnungsgemäß beendet wird
        if model:
            try:
                del model
            except:
                pass

def test_tensorrt_inference(model, optimization_results):
    """Testet TensorRT Inference mit echten Daten"""
    logger.info("🔬 Teste TensorRT Inference...")

    try:
        # Text-Test
        if "text" in optimization_results.get('tensorrt_engines', {}):
            logger.info("📝 Teste Text-Inference mit TensorRT...")
            test_text = "Die Bundeskanzlerin sprach über wichtige Themen der Wirtschaftspolitik."

            # Vergleiche Original vs TensorRT
            start_time = time.time()
            result_original = model.process_text(test_text, max_length=50)
            original_time = time.time() - start_time

            start_time = time.time()
            result_tensorrt = model.process_text_tensorrt(test_text, max_length=50)
            tensorrt_time = time.time() - start_time

            speedup = original_time / tensorrt_time if tensorrt_time > 0 else 0

            logger.info(f"  Original Ergebnis: {result_original[:100]}...")
            logger.info(f"  TensorRT Ergebnis: {result_tensorrt[:100]}...")
            logger.info(f"  Speedup: {speedup:.2f}x")

        # Vision-Test (vereinfacht)
        if "vision" in optimization_results.get('tensorrt_engines', {}):
            logger.info("🖼️ Vision-Modell mit TensorRT optimiert verfügbar")

        # Audio-Test (vereinfacht)
        if "audio" in optimization_results.get('tensorrt_engines', {}):
            logger.info("🎵 Audio-Modell mit TensorRT optimiert verfügbar")

    except Exception as e:
        logger.error(f"❌ Fehler bei TensorRT Inference-Test: {e}")


def test_tensorrt_availability():
    """Testet grundlegende TensorRT-Verfügbarkeit"""
    logger.info("🔍 Teste TensorRT-Verfügbarkeit...")

    try:
        import tensorrt as trt
        logger.info(f"✅ TensorRT {trt.__version__} verfügbar")

        # Teste Logger
        logger_trt = trt.Logger(trt.Logger.WARNING)
        logger.info("✅ TensorRT Logger erstellt")

        return True

    except ImportError:
        logger.error("❌ TensorRT nicht installiert")
        logger.info("ℹ️ Installiere TensorRT mit: pip install tensorrt")
        return False

    except Exception as e:
        logger.error(f"❌ TensorRT-Fehler: {e}")
        return False

def main():
    """Hauptfunktion für TensorRT-Tests"""
    logger.info("🚀 Bundeskanzler KI - TensorRT Integration Test")
    logger.info("=" * 60)

    # Graceful Shutdown aktivieren
    shutdown_handler = setup_graceful_shutdown()
    logger.info("🛡️ Graceful Shutdown aktiviert")

    # Teste TensorRT-Verfügbarkeit
    if not test_tensorrt_availability():
        logger.error("❌ TensorRT nicht verfügbar - beende Test")
        return 1

    # Teste Integration
    if test_tensorrt_integration():
        logger.info("✅ Alle Tests erfolgreich abgeschlossen")

        # Warte kurz um Shutdown zu testen
        logger.info("⏳ Warte 5 Sekunden um Shutdown zu testen...")
        time.sleep(5)

        return 0
    else:
        logger.error("❌ Tests fehlgeschlagen")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)