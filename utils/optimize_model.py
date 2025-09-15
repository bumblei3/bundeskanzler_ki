#!/usr/bin/env python3
"""
Modell-Optimierung für Bundeskanzler-KI
Führt alle Optimierungen durch und zeigt Ergebnisse an
"""

import sys
import os
import logging
import numpy as np
import time
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Projekt-Pfad hinzufügen
sys.path.insert(0, '/home/tobber/bkki_venv')

try:
    from model_optimizer import ModelOptimizer
except ImportError as e:
    logger.error(f"Fehler beim Importieren: {e}")
    sys.exit(1)

def create_sample_data():
    """Erstellt Beispieldaten für Training und Test"""
    logger.info("🔄 Erstelle Beispieldaten...")

    # Vereinfachte Text-Daten für Demo
    vocab_size = 1000
    max_length = 100
    num_samples = 1000

    # Dummy-Daten generieren
    x_train = np.random.randint(0, vocab_size, size=(num_samples, max_length))
    y_train = np.random.randint(0, vocab_size, size=(num_samples, vocab_size))

    x_test = np.random.randint(0, vocab_size, size=(200, max_length))
    y_test = np.random.randint(0, vocab_size, size=(200, vocab_size))

    logger.info(f"✅ Beispieldaten erstellt: {num_samples} Train, {len(x_test)} Test Samples")
    return x_train, y_train, x_test, y_test

def train_base_model(x_train, y_train):
    """Trainiert ein Basis-Modell falls keines vorhanden ist"""
    model_path = "bundeskanzler_ki_model.keras"

    # Lösche altes Modell falls vorhanden (wegen GPU-Problemen)
    if os.path.exists(model_path):
        logger.info("🗑️  Lösche altes Modell wegen Kompatibilitätsproblemen...")
        os.remove(model_path)

    logger.info("🔄 Trainiere neues Basis-Modell...")

    try:
        # TensorFlow-Modell direkt erstellen (vereinfacht für Demo)
        import tensorflow as tf

        # GPU-Probleme vermeiden - CPU-only für Training
        tf.config.set_visible_devices([], 'GPU')

        vocab_size = 1000
        embedding_dim = 128
        rnn_units = 256

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.GRU(rnn_units, return_sequences=True),
            tf.keras.layers.GRU(rnn_units),
            tf.keras.layers.Dense(vocab_size)
        ])

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy']
        )

        # Kurzes Training für Demo
        logger.info("⚠️  Hinweis: Für vollständige Optimierung echte Trainingsdaten verwenden")
        model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)

        # Modell speichern
        model.save(model_path)
        logger.info(f"✅ Basis-Modell gespeichert: {model_path}")

        return model_path

    except Exception as e:
        logger.error(f"Fehler beim Training des Basis-Modells: {e}")
        return None

def run_optimization():
    """Führt die vollständige Modell-Optimierung durch"""
    logger.info("🚀 Starte Modell-Optimierung für Bundeskanzler-KI")

    try:
        # Beispieldaten erstellen
        x_train, y_train, x_test, y_test = create_sample_data()

        # Basis-Modell trainieren
        model_path = train_base_model(x_train, y_train)
        if not model_path:
            logger.error("Konnte kein Basis-Modell erstellen")
            return

        # Modell-Optimierer initialisieren
        optimizer = ModelOptimizer(model_path)

        # Vollständige Optimierung durchführen
        report = optimizer.optimize_all(x_train, y_train, x_test, y_test)

        # Ergebnisse anzeigen
        print("\n" + "="*60)
        print("📊 MODELL-OPTIMIERUNG ERGEBNISSE")
        print("="*60)

        print(f"\n📁 Original-Modell: {model_path}")
        print(f"🔧 Optimierungen angewendet: {', '.join(optimizer.optimized_models.keys())}")

        print(f"\n📈 Benchmark-Ergebnisse:")
        print("-" * 40)

        for model_name, results in report['benchmark_results'].items():
            print(f"\n🔹 {model_name.upper()}:")
            if 'error' in results:
                print(f"   ❌ Fehler: {results['error']}")
            else:
                if 'inference_time' in results:
                    print(".3f")
                if 'accuracy' in results:
                    print(".3f")
                if 'model_size' in results:
                    size_mb = results['model_size'] / (1024 * 1024)
                    print(".2f")
                if 'parameters' in results:
                    print(f"   📏 Parameter: {results['parameters']:,}")
                if 'type' in results:
                    print(f"   🔧 Typ: {results['type']}")

        if report['recommendations']:
            print(f"\n💡 EMPFEHLUNGEN:")
            print("-" * 40)
            for rec in report['recommendations']:
                print(f"   • {rec}")

        print(f"\n📄 Detaillierter Bericht: model_optimization_report.json")

        print("\n" + "="*60)
        print("✅ Modell-Optimierung erfolgreich abgeschlossen!")
        print("="*60)

    except Exception as e:
        logger.error(f"Fehler bei der Modell-Optimierung: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_optimization()