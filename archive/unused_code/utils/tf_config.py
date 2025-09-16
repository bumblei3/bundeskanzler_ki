"""
TensorFlow Konfigurationsmodul für die Bundeskanzler-KI.
Stellt zentrale Konfigurationseinstellungen für TensorFlow bereit.
"""

import os

import tensorflow as tf


def configure_tensorflow():
    """
    Konfiguriert TensorFlow mit optimalen Einstellungen für CPU-Training.
    - Minimiert Logging-Ausgaben
    - Optimiert Thread-Nutzung
    - Deaktiviert GPU-bezogene Warnungen
    """
    # Logging-Konfiguration
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nur ERROR-Level Logging

    # CPU-Optimierungen
    tf.config.set_visible_devices([], "GPU")  # Deaktiviere GPU
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    # Precision und Memory
    tf.keras.mixed_precision.set_global_policy("float32")


def get_model_config():
    """
    Liefert die Standardkonfiguration für das Modell.
    Returns:
        dict: Modellkonfiguration mit allen wichtigen Parametern
    """
    return {
        "embedding_dim": 16,
        "lstm_units": 8,
        "dense_units": 8,
        "dropout_rate": 0.6,
        "l1_reg": 0.01,
        "l2_reg": 0.02,
        "learning_rate": 0.00001,
        "clip_norm": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
    }


# Automatische Konfiguration beim Import
configure_tensorflow()
