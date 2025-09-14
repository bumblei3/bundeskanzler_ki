"""
Memory-Netzwerk für die Bundeskanzler-KI.
Implementiert ein erweitertes Memory-Network für Kontext-Verarbeitung.
"""
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class MemoryNetwork:
    """
    Memory-Netzwerk zur Verarbeitung und Speicherung von Kontext-Informationen.
    """
    
    def __init__(self, memory_size: int = 1000, embedding_dim: int = 256):
        """
        Initialisiert das Memory-Netzwerk.
        
        Args:
            memory_size: Maximale Anzahl der Speichereinträge
            embedding_dim: Dimension der Embedding-Vektoren
        """
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        
        # Initialisiere Memory als TensorFlow Variable
        self.memory = tf.Variable(
            tf.zeros([memory_size, embedding_dim], dtype=tf.float32),
            trainable=False,
            name='memory_storage'
        )
        
        # Zeitstempel-Speicher für Verfallssteuerung
        self.timestamps = tf.Variable(
            tf.zeros([memory_size], dtype=tf.float32),
            trainable=False,
            name='memory_timestamps'
        )
        
        # Aktuelle Position im Memory
        self.current_position = tf.Variable(0, trainable=False, dtype=tf.int32)
        
        # Importance-Scores für Memory-Einträge
        self.importance = tf.Variable(
            tf.zeros([memory_size], dtype=tf.float32),
            trainable=False,
            name='memory_importance'
        )

    def store(self, embedding: tf.Tensor, importance: float = 1.0) -> None:
        """
        Speichert einen neuen Embedding-Vektor im Memory.
        
        Args:
            embedding: Embedding-Vektor der gespeichert werden soll
            importance: Wichtigkeits-Score des Eintrags (0.0 bis 1.0)
        """
        # Reshape embedding falls nötig
        embedding = tf.reshape(embedding, [self.embedding_dim])
        
        # Aktuelle Position im ringförmigen Speicher
        position = self.current_position.assign(
            (self.current_position + 1) % self.memory_size
        )
        
        # Speichere Embedding und Metadaten
        self.memory[position].assign(embedding)
        self.timestamps[position].assign(float(datetime.now().timestamp()))
        self.importance[position].assign(float(importance))

    def query(self, query_embedding: tf.Tensor, k: int = 5) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Findet die k ähnlichsten Einträge zu einem Query-Embedding.
        
        Args:
            query_embedding: Query-Vektor
            k: Anzahl der zurückzugebenden Einträge
            
        Returns:
            Tuple aus (Memory-Einträge, Ähnlichkeits-Scores)
        """
        # Reshape query falls nötig
        query_embedding = tf.reshape(query_embedding, [1, self.embedding_dim])
        
        # Normalisiere Embeddings
        query_norm = tf.nn.l2_normalize(query_embedding, axis=1)
        memory_norm = tf.nn.l2_normalize(self.memory, axis=1)

        # Berechne Kosinus-Ähnlichkeit
        similarity = tf.matmul(query_norm, tf.transpose(memory_norm))

        # Gewichte mit Importance und Time-Decay
        current_time = float(datetime.now().timestamp())
        time_weights = tf.exp(
            -(current_time - self.timestamps) / (24.0 * 3600.0)  # 24h Decay
        )
        
        weighted_similarity = similarity * self.importance * time_weights
        
        # Finde Top-k Einträge
        values, indices = tf.nn.top_k(tf.squeeze(weighted_similarity), k=k)
        
        return tf.gather(self.memory, indices), values

    def update_importance(self, index: int, new_importance: float) -> None:
        """
        Aktualisiert den Importance-Score eines Memory-Eintrags.
        
        Args:
            index: Index des zu aktualisierenden Eintrags
            new_importance: Neuer Importance-Score (0.0 bis 1.0)
        """
        self.importance[index].assign(float(new_importance))

    def clear_old_entries(self, max_age_days: float = 30.0) -> None:
        """
        Löscht Einträge die älter als max_age_days sind.
        
        Args:
            max_age_days: Maximales Alter in Tagen
        """
        current_time = float(datetime.now().timestamp())
        max_age = max_age_days * 24.0 * 3600.0  # Konvertiere zu Sekunden
        
        old_indices = tf.where(
            (current_time - self.timestamps) > max_age
        )
        
        # Setze alte Einträge auf Null
        if tf.size(old_indices) > 0:
            old_indices_flat = tf.reshape(old_indices, [-1])
            zeros_mem = tf.zeros([tf.size(old_indices_flat), self.embedding_dim], dtype=tf.float32)
            zeros_imp = tf.zeros([tf.size(old_indices_flat)], dtype=tf.float32)
            self.memory.assign(tf.tensor_scatter_nd_update(self.memory, tf.expand_dims(old_indices_flat, 1), zeros_mem))
            self.importance.assign(tf.tensor_scatter_nd_update(self.importance, tf.expand_dims(old_indices_flat, 1), zeros_imp))