"""
GPU-Management und Konfiguration für die Bundeskanzler-KI.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf


class GPUManager:
    def __init__(self, memory_limit: Optional[int] = None):
        """
        Initialisiert den GPU-Manager.

        Args:
            memory_limit: Optionale Speicherbegrenzung in MB
        """
        self.gpus = tf.config.list_physical_devices("GPU")
        self.memory_limit = memory_limit
        self.initialized = False
        self.distribution_strategy = None
        self._setup_logging()

    def _setup_logging(self):
        """Initialisiert Logging für GPU-Management."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("GPUManager")

    def setup_gpus(self) -> None:
        """Konfiguriert verfügbare GPUs für optimale Nutzung."""
        if self.initialized:
            return

        if not self.gpus:
            logging.warning("Keine GPUs gefunden. Verwende CPU.")
            self.distribution_strategy = tf.distribute.OneDeviceStrategy(
                device="/cpu:0"
            )
            return

        try:
            # GPU-Speicher nach Bedarf allokieren
            for gpu in self.gpus:
                if self.memory_limit:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=self.memory_limit
                            )
                        ],
                    )
                tf.config.experimental.set_memory_growth(gpu, True)

            # Wähle optimale Distribution Strategy
            if len(self.gpus) > 1:
                self.distribution_strategy = tf.distribute.MirroredStrategy()
                logging.info(f"Multi-GPU Training aktiviert mit {len(self.gpus)} GPUs")
            else:
                self.distribution_strategy = tf.distribute.OneDeviceStrategy(
                    device="/gpu:0"
                )
                logging.info("Single-GPU Training aktiviert")

            logging.info(
                f"GPU-Konfiguration erfolgreich. Verfügbare GPUs: {len(self.gpus)}"
            )
            for gpu in self.gpus:
                logging.info(f"  - {gpu.device_type}: {gpu.name}")

            self.initialized = True

        except RuntimeError as e:
            logging.error(f"GPU-Konfiguration fehlgeschlagen: {str(e)}")
            self.distribution_strategy = tf.distribute.OneDeviceStrategy(
                device="/cpu:0"
            )
            logging.warning("Fallback auf CPU-Training")

    def enable_mixed_precision(self) -> None:
        """
        Aktiviert Mixed Precision Training für bessere Performance.
        """
        if self.gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logging.info("Mixed Precision Training (float16) aktiviert")
        else:
            logging.warning("Mixed Precision nicht verfügbar auf CPU")

    def enable_xla(self) -> None:
        """
        Aktiviert XLA (Accelerated Linear Algebra) Optimierungen.
        """
        tf.config.optimizer.set_jit(True)
        logging.info("XLA Optimierungen aktiviert")

    def optimize_for_training(self) -> None:
        """Optimiert die GPU-Konfiguration für Training."""
        if not self.gpus:
            return

        try:
            # Automatische Mixed Precision aktivieren
            self.enable_mixed_precision()

            # XLA JIT Kompilierung aktivieren
            self.enable_xla()

            # Kernel-Caching für Training optimieren
            tf.config.optimizer.set_experimental_options(
                {
                    "layout_optimizer": True,
                    "constant_folding": True,
                    "shape_optimization": True,
                    "remapping": True,
                    "auto_mixed_precision": True,
                    "pin_to_host_optimization": True,
                    "scoped_allocator_optimization": True,
                }
            )

            logging.info("GPU für Training optimiert")

        except Exception as e:
            logging.warning(f"GPU-Optimierung für Training fehlgeschlagen: {str(e)}")

    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """
        Bestimmt die optimale Batch-Größe basierend auf verfügbarer Hardware.
        """
        if not self.gpus:
            return base_batch_size
        return base_batch_size * len(self.gpus)

    def get_memory_stats(self) -> Optional[Dict]:
        """
        Gibt Statistiken über GPU-Speichernutzung zurück.
        """
        if not self.gpus:
            return None

        try:
            stats = {}
            for gpu in self.gpus:
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                stats[gpu.name] = {
                    "peak": memory_info["peak"] / (1024 * 1024),  # MB
                    "current": memory_info["current"] / (1024 * 1024),  # MB
                }
            return stats
        except:
            return None


def get_optimal_training_config() -> Dict[str, Any]:
    """
    Ermittelt optimale Trainingsparameter basierend auf System-Ressourcen.
    """
    config = {
        "batch_size": 32,  # Standard Batch-Size
        "mixed_precision": False,
        "xla_acceleration": False,
    }

    try:
        # Prüfe verfügbaren GPU-Speicher
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Aktiviere Mixed Precision für bessere Performance
            config["mixed_precision"] = True
            # Aktiviere XLA für bessere Performance
            config["xla_acceleration"] = True

            # Passe Batch-Size an verfügbaren Speicher an
            memory_limit = (
                tf.config.get_logical_device_configuration(gpus[0])[0].memory_limit
                if tf.config.get_logical_device_configuration(gpus[0])
                else None
            )

            if memory_limit:
                # Grobe Schätzung: Mehr Speicher = größere Batch-Size
                if memory_limit > 8 * (1024**3):  # > 8GB
                    config["batch_size"] = 64
                elif memory_limit > 4 * (1024**3):  # > 4GB
                    config["batch_size"] = 48
    except Exception as e:
        logging.error(f"Fehler bei der Konfigurationsermittlung: {str(e)}")

    return config
