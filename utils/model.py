import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# GPU-Konfiguration für optimale Performance
def configure_gpu():
    """Konfiguriert GPU für TensorFlow mit Memory-Growth"""
    try:
        # GPU Memory Growth aktivieren (verhindert Out-of-Memory)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU konfiguriert: {len(gpus)} GPU(s) verfügbar")
            return True
        else:
            print("⚠️  Keine GPU gefunden, verwende CPU")
            return False
    except Exception as e:
        print(f"⚠️  GPU-Konfiguration fehlgeschlagen: {e}")
        return False


# GPU beim Import konfigurieren
gpu_available = configure_gpu()


def benchmark_gpu_performance(model, test_data, batch_size=32):
    """Misst GPU-Performance für Inferenz"""
    if not gpu_available:
        print("⚠️  GPU-Benchmark übersprungen (keine GPU verfügbar)")
        return

    print("🚀 Starte GPU-Performance-Benchmark...")

    # Test-Daten vorbereiten
    test_sequences = test_data[
        : min(100, len(test_data))
    ]  # Max 100 Samples für Benchmark

    # CPU-Zeit messen
    start_time = time.time()
    with tf.device("/CPU:0"):
        cpu_predictions = model.predict(
            test_sequences, batch_size=batch_size, verbose=0
        )
    cpu_time = time.time() - start_time

    # GPU-Zeit messen
    start_time = time.time()
    with tf.device("/GPU:0"):
        gpu_predictions = model.predict(
            test_sequences, batch_size=batch_size, verbose=0
        )
    gpu_time = time.time() - start_time

    # Ergebnisse anzeigen
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    print(f"📊 CPU-Zeit: {cpu_time:.2f}s")
    print(f"🚀 GPU-Zeit: {gpu_time:.2f}s")
    print(f"⚡ Speedup: {speedup:.1f}x")
    return {
        "cpu_time": cpu_time,
        "gpu_time": gpu_time,
        "speedup": speedup,
        "samples": len(test_sequences),
    }


def build_model(tokenizer, maxlen, embedding_size=256, lstm_output_size=128):
    """Erstellt ein neuronales Netzwerk-Modell mit GPU-Unterstützung"""
    with tf.device("/GPU:0" if gpu_available else "/CPU:0"):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=len(tokenizer.word_index) + 1,
                    output_dim=embedding_size,
                    input_length=maxlen,
                ),
                tf.keras.layers.GRU(
                    lstm_output_size,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    return_sequences=False,
                ),
                tf.keras.layers.Dense(
                    len(tokenizer.word_index) + 1, activation="softmax"
                ),
            ]
        )

        # Optimierte Kompilierung für GPU
        if gpu_available:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            optimizer = "adam"

        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        print(f"✅ Modell erstellt auf: {'GPU' if gpu_available else 'CPU'}")
        return model


def load_or_train_model(tokenizer, X, Y, maxlen, args):
    model_path = "bundeskanzler_ki_model.keras"
    if os.path.exists(model_path):
        print("Lade vorhandenes Modell...")
        model = tf.keras.models.load_model(model_path)
    else:
        model = build_model(tokenizer, maxlen)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=2, restore_best_weights=True
        )
        model.fit(
            X, Y, batch_size=args.batch_size, epochs=args.epochs, callbacks=[early_stop]
        )
        model.save(model_path)
        print(f"Modell gespeichert unter {model_path}")

    # GPU-Performance Benchmark nach Laden/Training
    if gpu_available and len(X) > 0:
        benchmark_gpu_performance(
            model, X[: min(100, len(X))], batch_size=args.batch_size
        )

    return model
