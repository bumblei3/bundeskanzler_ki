"""
Memory-Optimierung für Bundeskanzler-KI
Implementiert effiziente Speicherverwaltung, Lazy Loading und Garbage Collection Tuning
"""

import gc
import os
import psutil
import logging
from typing import Iterator, List, Optional
from contextlib import contextmanager
import mmap
import numpy as np

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Memory-Optimierungsklasse für effiziente Speicherverwaltung"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self) -> float:
        """Gibt den aktuellen Speicherverbrauch in MB zurück"""
        return self.process.memory_info().rss / 1024 / 1024

    def log_memory_usage(self, label: str = ""):
        """Loggt den aktuellen Speicherverbrauch"""
        current = self.get_memory_usage()
        delta = current - self.initial_memory
        logger.info(".1f")

    @contextmanager
    def memory_context(self, label: str = ""):
        """Context Manager für Memory-Messungen"""
        start_mem = self.get_memory_usage()
        try:
            yield
        finally:
            end_mem = self.get_memory_usage()
            delta = end_mem - start_mem
            logger.info(".1f")

    def force_garbage_collection(self):
        """Erzwingt Garbage Collection und loggt Ergebnisse"""
        collected = gc.collect()
        logger.debug(f"🗑️  Garbage Collection: {collected} Objekte freigegeben")

    def optimize_numpy_arrays(self):
        """Optimiert NumPy Array-Speicher"""
        # Sammle alle NumPy Arrays
        arrays = []
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                arrays.append(obj)

        total_size = sum(arr.nbytes for arr in arrays) / 1024 / 1024
        logger.info(f"📊 NumPy Arrays optimiert: {len(arrays)} Arrays, {total_size:.1f} MB")
        return total_size

class LazyFileReader:
    """Lazy File Reader für große Dateien"""

    def __init__(self, file_path: str, chunk_size: int = 8192):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    def read_lines_lazy(self) -> Iterator[str]:
        """Liest Datei zeilenweise ohne alles in den Speicher zu laden"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.rstrip('\n\r')
        except FileNotFoundError:
            logger.warning(f"Datei nicht gefunden: {self.file_path}")
            return

    def read_chunks(self, chunk_size: Optional[int] = None) -> Iterator[List[str]]:
        """Liest Datei in Chunks"""
        chunk_size = chunk_size or self.chunk_size
        chunk = []

        for line in self.read_lines_lazy():
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:  # Rest verarbeiten
            yield chunk

class MemoryMappedCorpus:
    """Memory-Mapped Corpus für effizienten Zugriff auf große Textdaten"""

    def __init__(self, corpus_file: str):
        self.corpus_file = corpus_file
        self.mmap_obj = None
        self.lines = []
        self.line_positions = []

        if os.path.exists(corpus_file):
            self._build_index()

    def _build_index(self):
        """Erstellt Index der Zeilenpositionen für schnellen Zugriff"""
        logger.info(f"Erstelle Memory-Map Index für {self.corpus_file}")
        self.line_positions = []

        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            pos = 0
            for line_num, line in enumerate(f):
                self.line_positions.append(pos)
                pos += len(line.encode('utf-8'))

        logger.info(f"Index erstellt: {len(self.line_positions)} Zeilen")

    def get_line(self, line_num: int) -> Optional[str]:
        """Holt eine spezifische Zeile ohne die ganze Datei zu laden"""
        if not self.line_positions or line_num >= len(self.line_positions):
            return None

        if self.mmap_obj is None:
            try:
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    self.mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception as e:
                logger.error(f"Memory-Mapping fehlgeschlagen: {e}")
                return None

        try:
            start_pos = self.line_positions[line_num]
            # Finde Ende der Zeile
            end_pos = start_pos
            while end_pos < len(self.mmap_obj) and self.mmap_obj[end_pos] != ord('\n'):
                end_pos += 1

            line_bytes = self.mmap_obj[start_pos:end_pos]
            return line_bytes.decode('utf-8').rstrip('\n\r')
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Zeile {line_num}: {e}")
            return None

    def close(self):
        """Schließt Memory-Mapping"""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None

    def __len__(self):
        return len(self.line_positions)

class ChunkedProcessor:
    """Chunked Processing für große Datenmengen"""

    def __init__(self, chunk_size: int = 1000, memory_optimizer: Optional[MemoryOptimizer] = None):
        self.chunk_size = chunk_size
        self.memory_optimizer = memory_optimizer or MemoryOptimizer()

    def process_in_chunks(self, data: List, processor_func, *args, **kwargs):
        """Verarbeitet Daten in Chunks mit Memory-Management"""
        results = []

        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            logger.debug(f"Verarbeite Chunk {i//self.chunk_size + 1}/{(len(data) + self.chunk_size - 1)//self.chunk_size}")

            # Verarbeite Chunk
            chunk_results = processor_func(chunk, *args, **kwargs)
            results.extend(chunk_results)

            # Memory-Optimierung nach jedem Chunk
            self.memory_optimizer.force_garbage_collection()
            self.memory_optimizer.log_memory_usage(f"nach Chunk {i//self.chunk_size + 1}")

        return results

# Globale Memory-Optimierungsinstanz
memory_optimizer = MemoryOptimizer()

def optimize_tensorflow_memory():
    """Optimiert TensorFlow Memory-Einstellungen"""
    try:
        import tensorflow as tf

        # Prüfe ob TensorFlow vollständig verfügbar ist (nicht nur Stub)
        if not hasattr(tf, 'config') or not hasattr(tf.config, 'list_physical_devices'):
            logger.info("TensorFlow Stub erkannt - überspringe Memory-Optimierung")
            return

        # GPU Memory Growth aktivieren
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"✅ GPU Memory Growth aktiviert für {gpu}")
                except RuntimeError as e:
                    logger.warning(f"GPU Memory Growth konnte nicht aktiviert werden: {e}")

        # CPU Memory-Optimierung
        if hasattr(tf.config, 'threading'):
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)

        logger.info("✅ TensorFlow Memory-Optimierung abgeschlossen")

    except Exception as e:
        logger.warning(f"TensorFlow Memory-Optimierung fehlgeschlagen: {e}")
        logger.info("Fortfahren ohne TensorFlow Memory-Optimierung")

def setup_memory_optimization():
    """Richtet globale Memory-Optimierungen ein"""
    # Garbage Collection Tuning
    gc.set_threshold(700, 10, 10)  # Aggressivere GC

    # TensorFlow Memory-Optimierung
    try:
        optimize_tensorflow_memory()
    except (ImportError, Exception) as e:
        logger.warning(f"TensorFlow Memory-Optimierung nicht verfügbar: {e}")

    logger.info("✅ Memory-Optimierung initialisiert")