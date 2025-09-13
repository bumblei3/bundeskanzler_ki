"""
Memory-Netzwerk für erweitertes Kontextverständnis.
Implementiert ein Key-Value Gedächtnis mit Attention-Mechanismus für verbesserte
Langzeit-Kontextverarbeitung in der Bundeskanzler-KI.

Attributes:
    memory_size: Anzahl der Gedächtniszellen
    key_dim: Dimensionalität der Schlüssel und Werte
    num_heads: Anzahl der Attention-Heads
    attention: Multi-Head Attention Layer
    layernorm: Layer-Normalisierung
    dropout: Dropout für Regularisierung
"""
from typing import Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MemoryNetwork(layers.Layer):
    """
    Implementiert ein differenzierbares Memory-Netzwerk mit Attention-Mechanismus.
    """
    def __init__(
        self,
        memory_size: int,
        key_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(MemoryNetwork, self).__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.num_heads = num_heads
        
        # Memory-Komponenten mit korrekter Shape für Attention
        self.memory_keys = self.add_weight(
            name="memory_keys",
            shape=(1, memory_size, key_dim),  # [1, memory_size, key_dim] für Attention
            initializer="glorot_uniform",
            trainable=True
        )
        self.memory_values = self.add_weight(
            name="memory_values",
            shape=(1, memory_size, key_dim),  # [1, memory_size, key_dim] für Attention
            initializer="glorot_uniform",
            trainable=True
        )
        
        # Attention-Mechanismus
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        
        # Normalisierung und Dropout
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
        
    @tf.function
    def call(
        self,
        queries: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Führt Memory-Lookup und Update durch.
        
        Args:
            queries: Input-Tensor für Memory-Lookup [batch_size, seq_len, dim]
            training: Training-Modus Flag
            
        Returns:
            Tuple aus:
                - memory_output: Verarbeitete Memory-Ausgabe [batch_size, seq_len, dim]
                - attention_weights: Attention-Gewichte [batch_size, num_heads, seq_len, memory_size]
        """
        # Memory-Lookup mit Attention (broadcast zu batch_size)
        batch_size = tf.shape(queries)[0]
        keys = tf.broadcast_to(
            self.memory_keys,
            [batch_size, *self.memory_keys.shape[1:]]
        )
        values = tf.broadcast_to(
            self.memory_values,
            [batch_size, *self.memory_values.shape[1:]]
        )
        
        memory_output, attention_weights = self.attention(
            queries,
            keys,
            values,
            return_attention_scores=True
        )
        
        # Residual-Verbindung und Normalisierung
        memory_output = self.dropout(memory_output, training=training)
        memory_output = self.layernorm(queries + memory_output)
        
        return memory_output, attention_weights
    
    def update_memory(
        self,
        keys: tf.Tensor,
        values: tf.Tensor,
        importance: Optional[tf.Tensor] = None
    ) -> None:
        """
        Aktualisiert den Gedächtnisinhalt.
        
        Args:
            keys: Neue Schlüssel für das Gedächtnis
            values: Neue Werte für das Gedächtnis
            importance: Optionale Wichtigkeits-Gewichte
        """
        if importance is None:
            importance = tf.ones_like(keys[..., 0])
            
        # Importance-gewichtetes Update
        importance = tf.expand_dims(importance, -1)
        update_mask = tf.cast(
            tf.random.uniform((1, self.memory_size)) < 0.1,  # Angepasste Shape
            tf.float32
        )
        update_mask = tf.expand_dims(update_mask, -1)  # [1, memory_size, 1]
        
        # Zufälliges Update von 10% der Gedächtniszellen
        self.memory_keys.assign(
            self.memory_keys * (1 - update_mask) +
            tf.expand_dims(keys, 1) * update_mask  # [batch, memory_size, dim]
        )
        self.memory_values.assign(
            self.memory_values * (1 - update_mask) +
            tf.expand_dims(values, 1) * update_mask
        )
    
    def get_memory_state(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Gibt den aktuellen Gedächtniszustand zurück.
        
        Returns:
            Tuple aus Memory-Keys und Memory-Values
        """
        return self.memory_keys, self.memory_values