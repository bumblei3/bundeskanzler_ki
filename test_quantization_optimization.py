#!/usr/bin/env python3
"""
Test-Script für erweiterte Quantisierungs-Optimierung
Testet die neuen Quantization Optimizer Features
"""

import sys
import os
import logging

# Füge aktuelles Verzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_quantization_optimizer():
    """Testet den Quantization Optimizer"""
    try:
        from core.quantization_optimizer import QuantizationOptimizer

        logger.info("🧪 Teste Quantization Optimizer...")

        # Optimizer initialisieren
        optimizer = QuantizationOptimizer()
        logger.info(f"✅ Optimizer initialisiert - GPU: {optimizer.gpu_memory_gb:.1f}GB")

        # Teste optimale Konfiguration für verschiedene Modelle
        test_models = ["gpt2", "gpt2-medium", "siglip-base", "whisper-base"]

        for model in test_models:
            config = optimizer.get_optimal_quantization_config(model)
            logger.info(f"📊 {model}: {config.quantization_type} Quantisierung empfohlen")

        # Performance-Report
        report = optimizer.get_performance_report()
        logger.info(f"📈 Performance-Report: {report}")

        return True

    except Exception as e:
        logger.error(f"❌ Quantization Optimizer Test fehlgeschlagen: {e}")
        return False

def test_multimodal_model_with_quantization():
    """Testet das multimodale Modell mit Quantization Optimizer"""
    try:
        from multimodal_ki import MultimodalTransformerModel

        logger.info("🧪 Teste MultimodalTransformerModel mit Quantization...")

        # Modell mit RTX 2070 Modus erstellen
        model = MultimodalTransformerModel(model_tier="rtx2070")
        logger.info(f"✅ Modell erstellt - Typ: {model.text_model_type}")

        # Performance-Report abrufen
        report = model.get_quantization_performance_report()
        logger.info(f"📈 Quantization Performance: {report}")

        # Quantisierung optimieren
        optimized = model.optimize_quantization_settings()
        logger.info(f"🔧 Quantisierung optimiert: {optimized}")

        return True

    except Exception as e:
        logger.error(f"❌ Multimodal Model Test fehlgeschlagen: {e}")
        return False

def main():
    """Hauptfunktion für Tests"""
    logger.info("🚀 Starte Quantisierungs-Optimierung Tests...")

    # Test 1: Quantization Optimizer
    test1_result = test_quantization_optimizer()
    logger.info(f"Test 1 (Quantization Optimizer): {'✅ BESTANDEN' if test1_result else '❌ FEHLGESCHLAGEN'}")

    # Test 2: Multimodal Model mit Quantization
    test2_result = test_multimodal_model_with_quantization()
    logger.info(f"Test 2 (Multimodal Model): {'✅ BESTANDEN' if test2_result else '❌ FEHLGESCHLAGEN'}")

    # Zusammenfassung
    if test1_result and test2_result:
        logger.info("🎉 Alle Tests bestanden! Quantisierungs-Optimierung ist bereit für den Einsatz.")
        return 0
    else:
        logger.warning("⚠️ Einige Tests fehlgeschlagen. Überprüfe die Implementierung.")
        return 1

if __name__ == "__main__":
    sys.exit(main())