"""
Kontextprozessor für die Bundeskanzler-KI.
Implementiert erweiterte Kontextverarbeitung mit Memory-Netzwerk Integration.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from memory_network import MemoryNetwork


class ContextProcessor:
    """
    Verarbeitet und verwaltet Kontext-Informationen für die Bundeskanzler-KI.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        key_dim: int = 512,
        num_heads: int = 8,
        max_context_window: int = 50,
    ):
        """
        Initialisiert den Kontextprozessor.

        Args:
            memory_size: Größe des Gedächtnisses
            key_dim: Dimensionalität der Schlüssel/Werte
            num_heads: Anzahl der Attention-Heads
            max_context_window: Maximale Anzahl zu merkender Kontexte
        """
        self.memory = MemoryNetwork(
            memory_size=memory_size, key_dim=key_dim, num_heads=num_heads
        )
        self.max_context_window = max_context_window
        self.context_buffer = []
        self.context_importances = []

    def process_context(
        self, text: str, embeddings: tf.Tensor, metadata: Optional[Dict] = None
    ) -> tf.Tensor:
        """
        Verarbeitet neuen Kontext und integriert ihn ins Gedächtnis.

        Args:
            text: Eingabetext
            embeddings: Text-Embeddings [batch_size, seq_len, dim]
            metadata: Optionale Metadaten (Datum, Quelle, etc.)

        Returns:
            Kontextualisierte Embeddings
        """
        # Berechne Kontext-Wichtigkeit
        importance = self._calculate_importance(text, metadata)

        # Buffer-Management
        if len(self.context_buffer) >= self.max_context_window:
            self.context_buffer.pop(0)
            self.context_importances.pop(0)
        self.context_buffer.append(embeddings)
        self.context_importances.append(importance)

        # Aktualisiere Gedächtnis
        batch_keys = tf.concat(self.context_buffer, axis=0)
        batch_values = batch_keys  # Verwende Embeddings als Werte
        self.memory.update_memory(
            keys=batch_keys,
            values=batch_values,
            importance=tf.concat(self.context_importances, axis=0),
        )

        # Kontextualisiere aktuelle Eingabe
        contextualized_embeddings, _ = self.memory(embeddings)
        return contextualized_embeddings

    def _calculate_importance(
        self, text: str, metadata: Optional[Dict] = None
    ) -> tf.Tensor:
        """
        Berechnet die Wichtigkeit des Kontexts.

        Args:
            text: Eingabetext
            metadata: Optionale Metadaten

        Returns:
            Wichtigkeits-Score [batch_size]
        """
        # Basis-Wichtigkeit
        importance = 1.0

        if metadata:
            # Zeitliche Relevanz (neuere Einträge wichtiger)
            if "date" in metadata and metadata["date"] is not None:
                current_date = datetime.now()
                time_diff = (current_date - metadata["date"]).days
                temporal_factor = np.exp(-time_diff / 365)  # Jahres-Skala
                importance *= temporal_factor
            if "source" in metadata:
                source_weights = {
                    "pressemitteilung": 1.2,
                    "rede": 1.5,
                    "interview": 1.3,
                    "social_media": 0.8,
                }
                importance *= source_weights.get(metadata["source"], 1.0)

            # Thematische Relevanz
            if "topics" in metadata and metadata["topics"]:
                topic_weights = {
                    "wirtschaft": 1.4,
                    "außenpolitik": 1.3,
                    "innenpolitik": 1.2,
                    "klimaschutz": 1.3,
                    "digitalisierung": 1.2,
                }
                topic_importance = max(
                    topic_weights.get(topic, 1.0) for topic in metadata["topics"]
                )
                importance *= topic_importance

        return tf.constant([importance])
