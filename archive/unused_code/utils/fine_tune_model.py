#!/usr/bin/env python3
"""
Fine-tuning Script für Bundeskanzler KI
Optimiert das vortrainierte Modell auf den erweiterten Regierungs-Corpus
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

# Projekt-Pfad hinzufügen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports aus dem Hauptprogramm
from bundeskanzler_ki import (
    build_model,
    detect_language,
    load_or_train_model,
    preprocess,
    preprocess_corpus,
    train_model,
)
from corpus_manager import CorpusManager

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ModelFineTuner:
    """
    Fine-tuning Klasse für das Bundeskanzler KI Modell
    """

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.corpus_manager = CorpusManager()

        # GPU Memory Growth aktivieren
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"✅ GPU Memory Growth aktiviert für {len(gpus)} GPU(s)")

    def _load_config(self) -> dict:
        """Lädt die Konfiguration"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"⚠️ Konnte Konfiguration nicht laden: {e}")
            return {
                "model": {"maxlen": 100, "vocab_size": 10000},
                "training": {"batch_size": 32, "epochs": 10, "learning_rate": 1e-5},
            }

    def load_base_model(self, model_path: str = None):
        """Lädt das vortrainierte Modell oder erstellt ein neues"""
        try:
            if model_path and os.path.exists(model_path):
                logging.info(f"🚀 Lade Basis-Modell von {model_path}...")
                self.model = tf.keras.models.load_model(model_path)
                logging.info("✅ Basis-Modell erfolgreich geladen")
            else:
                logging.info("🏗️ Erstelle neues Modell für Fine-tuning...")
                # Erstelle ein Sequence-to-Sequence Modell für Fine-tuning
                maxlen = self.config.get("model", {}).get("maxlen", 100)
                vocab_size = self.config.get("model", {}).get("vocab_size", 10000)

                self.model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Embedding(
                            input_dim=vocab_size, output_dim=128, input_length=maxlen
                        ),
                        tf.keras.layers.LSTM(128, return_sequences=True),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.LSTM(64, return_sequences=True),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(vocab_size, activation="softmax"),
                    ]
                )
                logging.info("✅ Neues Modell erstellt")
            return True
        except Exception as e:
            logging.error(f"❌ Fehler beim Laden/Erstellen des Modells: {e}")
            import traceback

            traceback.print_exc()
            return False

    def prepare_fine_tuning_data(self):
        """Bereitet die Fine-tuning Daten vor"""
        logging.info("📚 Lade erweiterten Corpus für Fine-tuning...")

        # Lade den erweiterten Corpus direkt
        try:
            with open("corpus.json", "r", encoding="utf-8") as f:
                corpus_data = json.load(f)
                corpus = [entry["text"] for entry in corpus_data["entries"]]
                logging.info(f"📊 Corpus geladen: {len(corpus)} Einträge")
        except Exception as e:
            logging.error(f"❌ Fehler beim Laden des Corpus: {e}")
            # Fallback: Verwende Standard-Corpus
            corpus = [
                "Deutschland setzt sich für ambitionierte Klimaschutz ein",
                "Die Bundesregierung fördert erneuerbare Energien",
                "Deutschland investiert in Zukunftstechnologien wie KI",
                "Die digitale Transformation wird vorangetrieben",
            ]
            logging.info(f"📊 Fallback-Corpus verwendet: {len(corpus)} Einträge")

        # Erstelle Trainingsdaten für Fine-tuning
        # Für Sequence-to-Sequence Training brauchen wir Input-Output Paare
        inputs = []
        outputs = []

        # Teile lange Texte in kleinere Sequenzen oder verwende ganze Texte
        maxlen = self.config.get("model", {}).get("maxlen", 100)

        for text in corpus:
            words = text.split()
            if len(words) >= 5:  # Mindestens 5 Wörter
                if len(words) <= maxlen:
                    # Kurzer Text: Verwende als Ganzes
                    inputs.append(text)
                    outputs.append(text)  # Autoencoder-Style Training
                else:
                    # Langer Text: Teile in überlappende Sequenzen
                    for i in range(0, len(words) - 10, 5):  # Überlappung von 5 Wörtern
                        chunk = " ".join(words[i : i + 10])
                        inputs.append(chunk)
                        # Ziel: Die nächsten Wörter vorhersagen
                        if i + 10 < len(words):
                            target_chunk = " ".join(words[i + 1 : i + 11])
                            outputs.append(target_chunk)
                        else:
                            outputs.append(chunk)  # Fallback

        logging.info(f"🧪 Erstelle {len(inputs)} Trainingssequenzen")

        # Tokenizer aus dem Hauptprogramm verwenden oder neu erstellen
        try:
            from bundeskanzler_ki import tokenizer

            self.tokenizer = tokenizer
        except:
            # Neuen Tokenizer erstellen
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=self.config.get("model", {}).get("vocab_size", 10000),
                oov_token="<OOV>",
            )
            self.tokenizer.fit_on_texts(inputs + outputs)

        # Sequenzen erstellen
        input_sequences = self.tokenizer.texts_to_sequences(inputs)
        output_sequences = self.tokenizer.texts_to_sequences(outputs)

        # Padding
        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            input_sequences, maxlen=maxlen, padding="post"
        )
        output_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            output_sequences, maxlen=maxlen, padding="post"
        )

        # Für Autoencoder-Style Training: Input und Output sind gleich
        X_train, X_val, y_train, y_val = train_test_split(
            input_sequences, input_sequences, test_size=0.2, random_state=42
        )

        logging.info(
            f"✅ Fine-tuning Daten vorbereitet: {len(X_train)} Train, {len(X_val)} Validation"
        )

        return X_train, X_val, y_train, y_val

    def fine_tune_model(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        learning_rate: float = 1e-5,
        epochs: int = 5,
        batch_size: int = 16,
    ):
        """Fine-tuning des Modells"""
        logging.info("🎯 Starte Fine-tuning...")

        # Kompiliere Modell mit niedrigerem Learning Rate für Fine-tuning
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'fine_tuned_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras',
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
        ]

        # Fine-tuning
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logging.info("✅ Fine-tuning abgeschlossen")
        return history

    def save_fine_tuned_model(self, output_path: str):
        """Speichert das fine-tuned Modell"""
        try:
            self.model.save(output_path)
            logging.info(f"💾 Fine-tuned Modell gespeichert: {output_path}")

            # Speichere auch Tokenizer
            import pickle

            tokenizer_path = output_path.replace(".keras", "_tokenizer.pkl")
            with open(tokenizer_path, "wb") as f:
                pickle.dump(self.tokenizer, f)
            logging.info(f"💾 Tokenizer gespeichert: {tokenizer_path}")

        except Exception as e:
            logging.error(f"❌ Fehler beim Speichern: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning für Bundeskanzler KI")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Pfad zum vortrainierten Modell (optional)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="fine_tuned_model.keras",
        help="Pfad für das fine-tuned Modell",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning Rate für Fine-tuning",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Anzahl der Fine-tuning Epochen")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch Size für Fine-tuning")

    args = parser.parse_args()

    # Fine-tuner initialisieren
    fine_tuner = ModelFineTuner()

    # Basis-Modell laden/erstellen
    if not fine_tuner.load_base_model(args.model_path):
        return

    # Fine-tuning Daten vorbereiten
    X_train, X_val, y_train, y_val = fine_tuner.prepare_fine_tuning_data()

    # Fine-tuning durchführen
    history = fine_tuner.fine_tune_model(
        X_train,
        X_val,
        y_train,
        y_val,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Modell speichern
    fine_tuner.save_fine_tuned_model(args.output_path)

    # Zusammenfassung
    print("\n" + "=" * 50)
    print("🎉 FINE-TUNING ABGESCHLOSSEN!")
    print("=" * 50)
    print(f"📊 Finale Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"🎯 Finale Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"💾 Modell gespeichert: {args.output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
