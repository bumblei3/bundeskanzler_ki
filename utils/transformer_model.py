"""
Modellarchitektur für die Bundeskanzler-KI.
Implementiert ein optimiertes LSTM-Modell für Textverarbeitung.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers
import tf_config

class ModelBuilder:
    """
    Builder-Klasse für die Modellarchitektur.
    Ermöglicht saubere Konstruktion des Modells mit allen Konfigurationsoptionen.
    """
    def __init__(self, maxlen, vocab_size, output_size):
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.config = tf_config.get_model_config()
    
    def build_embedding_layer(self, inputs):
        """Erstellt die Embedding-Schicht mit Regularisierung."""
        embedding = layers.Embedding(
            self.vocab_size,
            self.config['embedding_dim'],
            embeddings_regularizer=regularizers.l1_l2(
                l1=self.config['l1_reg'],
                l2=self.config['l2_reg']
            ),
            embeddings_constraint=tf.keras.constraints.MaxNorm(2)
        )(inputs)
        return layers.Dropout(self.config['dropout_rate'])(embedding)
    
    def build_lstm_layer(self, inputs):
        """Erstellt die LSTM-Schicht mit Regularisierung."""
        return layers.LSTM(
            self.config['lstm_units'],
            kernel_regularizer=regularizers.l1_l2(
                l1=self.config['l1_reg'],
                l2=self.config['l2_reg']
            ),
            recurrent_regularizer=regularizers.l2(self.config['l2_reg']),
            bias_regularizer=regularizers.l2(self.config['l2_reg']),
            activity_regularizer=regularizers.l1(self.config['l1_reg']),
            kernel_constraint=tf.keras.constraints.MaxNorm(2),
            recurrent_constraint=tf.keras.constraints.UnitNorm()
        )(inputs)
    
    def build_dense_layer(self, inputs):
        """Erstellt die Dense-Schicht mit Regularisierung."""
        dense = layers.Dense(
            self.config['dense_units'],
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(
                l1=self.config['l1_reg'],
                l2=self.config['l2_reg']
            ),
            activity_regularizer=regularizers.l1(self.config['l1_reg']),
            kernel_constraint=tf.keras.constraints.MaxNorm(2)
        )(inputs)
        dropout = layers.Dropout(self.config['dropout_rate'])(dense)
        return layers.BatchNormalization()(dropout)

def create_transformer_model(maxlen, vocab_size, output_size):
    """
    Erstellt das Modell mit der optimierten Architektur.
    
    Args:
        maxlen: Maximale Sequenzlänge
        vocab_size: Größe des Vokabulars
        output_size: Anzahl der Ausgabeklassen
        
    Returns:
        tf.keras.Model: Kompiliertes Modell
    """
    builder = ModelBuilder(maxlen, vocab_size, output_size)
    
    # Modellarchitektur
    inputs = layers.Input(shape=(maxlen,))
    x = builder.build_embedding_layer(inputs)
    x = builder.build_lstm_layer(x)
    x = builder.build_dense_layer(x)
    
    # Ausgabeschicht
    outputs = layers.Dense(
        output_size,
        activation='softmax',
        kernel_regularizer=regularizers.l1_l2(
            l1=builder.config['l1_reg'],
            l2=builder.config['l2_reg']
        ),
        activity_regularizer=regularizers.l1(builder.config['l1_reg']),
        kernel_constraint=tf.keras.constraints.MaxNorm(2)
    )(x)
    
    # Modell erstellen
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=builder.config['learning_rate'],
        clipnorm=builder.config['clip_norm'],
        beta_1=builder.config['beta_1'],
        beta_2=builder.config['beta_2'],
        epsilon=builder.config['epsilon']
    )
    
    # Modell kompilieren
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_training_callbacks(model, config=None):
    """
    Erstellt optimierte Callbacks für das Training.
    
    Args:
        model: Das Modell, für das die Callbacks erstellt werden
        config: Optional, Konfigurationswerte (verwendet sonst Standardwerte)
        
    Returns:
        list: Liste von Keras Callbacks
    """
    if config is None:
        config = tf_config.get_model_config()
        
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=config['learning_rate'] / 1000,  # Dynamische minimale Lernrate
            min_delta=0.0001
        ),
        tf.keras.callbacks.LambdaCallback(
            on_batch_begin=lambda batch, logs: tf.clip_by_global_norm(
                [layer.trainable_variables for layer in model.layers],
                config['clip_norm']
            )
        )
    ]

def train_transformer(model, X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=None):
    """
    Trainiert das Modell mit optimierten Parametern.
    
    Args:
        model: Das zu trainierende Modell
        X_train: Trainingsdaten
        y_train: Trainingslabels
        batch_size: Größe der Batches
        epochs: Anzahl der Epochen
        validation_split: Anteil der Validierungsdaten
        callbacks: Optional, Liste von Callbacks
        
    Returns:
        history: Trainingsverlauf
    """
    if callbacks is None:
        callbacks = get_training_callbacks()
    
    return model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks
    )