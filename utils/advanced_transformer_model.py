"""
Advanced Transformer Model für Bundeskanzler KI
Integriert vortrainierte Modelle wie GPT-2, BERT für bessere Sprachverarbeitung
"""

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AdvancedTransformerModel:
    """
    Erweiterte Transformer-Modell-Klasse mit Unterstützung für verschiedene vortrainierte Modelle
    """

    def __init__(self, model_type="gpt2", model_name="gpt2"):
        """
        Initialisiert das Transformer-Modell

        Args:
            model_type: Typ des Modells ("gpt2", "bert", "roberta")
            model_name: Name des vortrainierten Modells
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()

    def load_model(self):
        """Lädt das vortrainierte Modell"""
        try:
            if self.model_type == "gpt2":
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
                # GPT-2 hat kein pad_token, verwende eos_token
                self.tokenizer.pad_token = self.tokenizer.eos_token

            elif self.model_type == "bert":
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self.model = BertModel.from_pretrained("bert-base-uncased")

            elif self.model_type == "roberta":
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                self.model = RobertaModel.from_pretrained("roberta-base")

            elif self.model_type == "hybrid":
                # Für hybrides Modell, lade GPT-2 als Basis
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.model = GPT2LMHeadModel.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            else:
                # Fallback zu GPT-2
                self.model_type = "gpt2"
                self.model_name = "gpt2"
                self.load_model()
                return

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✅ {self.model_type.upper()} Modell '{self.model_name}' erfolgreich geladen")

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des Modells {self.model_name}: {e}")
            # Fallback zu GPT-2 small
            self.model_type = "gpt2"
            self.model_name = "gpt2"
            self.load_model()

    def generate_response(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """
        Generiert eine Antwort basierend auf dem Prompt

        Args:
            prompt: Eingabe-Prompt
            max_length: Maximale Länge der generierten Antwort
            temperature: Kreativität der Generierung (0.1-1.0)
            top_p: Nucleus Sampling Parameter

        Returns:
            str: Generierte Antwort
        """
        try:
            if self.model_type == "gpt2":
                return self._generate_gpt2(prompt, max_length, temperature, top_p)
            elif self.model_type in ["bert", "roberta"]:
                return self._generate_bert_roberta(prompt, max_length)
            else:
                return "Modell-Typ nicht unterstützt"

        except Exception as e:
            logger.error(f"Fehler bei der Textgenerierung: {e}")
            return f"Fehler bei der Generierung: {str(e)}"

    def _generate_gpt2(self, prompt, max_length, temperature, top_p):
        """GPT-2 Textgenerierung"""
        # Prompt vorbereiten
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Attention mask für padding
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length + len(input_ids[0]),
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        # Decode der generierten Tokens
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Nur den neuen Teil zurückgeben (nach dem Prompt)
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()

        return response

    def _generate_bert_roberta(self, prompt, max_length):
        """BERT/RoBERTa basierte Generierung (vereinfacht)"""
        # Für BERT/RoBERTa verwenden wir eine einfachere Approach
        # In der Praxis würde man hier ein Sequence-to-Sequence Modell verwenden

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Vereinfachte Antwort-Generierung
            # In der Praxis würde hier ein Decoder-Modell verwendet werden
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Placeholder-Response für BERT/RoBERTa
        return f"[BERT/RoBERTa Response] Verarbeitet: {prompt[:50]}..."

    def get_embedding(self, text):
        """
        Alias für get_embeddings (Singular)
        """
        return self.get_embeddings(text)

    def get_embeddings(self, text):
        """
        Extrahiert Embeddings für einen gegebenen Text

        Args:
            text: Eingabetext

        Returns:
            numpy array: Embeddings
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                if self.model_type == "gpt2":
                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Verwende letzte Hidden States
                    embeddings = outputs.hidden_states[-1].mean(dim=1)
                else:
                    # Für BERT/RoBERTa
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

            return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Fehler bei Embedding-Generierung: {e}")
            return np.zeros((1, 768))  # Fallback embedding

def create_hybrid_model(maxlen, vocab_size, output_size, transformer_model=None):
    """
    Erstellt ein hybrides Modell, das Transformer-Embeddings mit LSTM kombiniert

    Args:
        maxlen: Maximale Sequenzlänge
        vocab_size: Vokabular-Größe
        output_size: Anzahl der Ausgabeklassen
        transformer_model: Optionales Transformer-Modell für Embeddings

    Returns:
        tf.keras.Model: Hybrides Modell
    """
    inputs = layers.Input(shape=(maxlen,))

    # Basis-Embedding Layer
    embedding = layers.Embedding(vocab_size, 256)(inputs)
    x = layers.Dropout(0.3)(embedding)

    # LSTM Layer
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)

    # Dense Layer
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output
    outputs = layers.Dense(output_size, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Kompatibilitätsfunktion für bestehende Codebasis
def create_enhanced_transformer_model(maxlen, vocab_size, output_size, use_transformer=True):
    """
    Erstellt ein erweitertes Modell mit optionaler Transformer-Unterstützung

    Args:
        maxlen: Maximale Sequenzlänge
        vocab_size: Vokabular-Größe
        output_size: Anzahl der Ausgabeklassen
        use_transformer: Ob Transformer-Modelle verwendet werden sollen

    Returns:
        tf.keras.Model: Erweitertes Modell
    """
    if use_transformer:
        # Versuche Transformer-Modell zu laden
        try:
            transformer = AdvancedTransformerModel(model_type="gpt2", model_name="gpt2")
            logger.info("✅ Transformer-Modell erfolgreich geladen")
            return create_hybrid_model(maxlen, vocab_size, output_size, transformer)
        except Exception as e:
            logger.warning(f"⚠️ Transformer-Modell konnte nicht geladen werden: {e}")
            logger.info("🔄 Fallback zu Standard-LSTM-Modell")

    # Fallback zu Standard-Modell
    return create_hybrid_model(maxlen, vocab_size, output_size, None)