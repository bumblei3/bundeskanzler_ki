"""
Retrieval-Augmented Generation (RAG) System für die Bundeskanzler-KI
Kombiniert semantische Suche mit generativer KI für kontextuelle Antworten
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer


class RAGSystem:
    """
    Retrieval-Augmented Generation System für kontextuelle Antworten
    Verwendet FAISS für effiziente Vektor-Suche und Sentence Transformers für Embeddings
    """

    def __init__(
        self,
        corpus_path: str = None,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        models_path: str = None,
    ):
        """
        Initialisiert das RAG-System

        Args:
            corpus_path: Pfad zur Corpus-Datei
            embedding_model: Name des Sentence Transformer Modells
            models_path: Pfad zum models/ Verzeichnis (für Tests)
        """
        # Dynamischer Pfad basierend auf dem aktuellen Skript-Verzeichnis
        if corpus_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            corpus_path = os.path.join(project_root, "data", "corpus.json")

        if models_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            models_path = os.path.join(project_root, "models")

        self.corpus_path = corpus_path
        self.models_path = models_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.index = None
        self.corpus_entries = []
        self.embeddings = None

        # Konfiguration
        self.config = {
            "top_k": 5,  # Anzahl der abzurufenden Dokumente
            "similarity_threshold": 0.3,  # Mindest-Ähnlichkeit für Retrieval
            "max_context_length": 1000,  # Maximale Länge des Kontexts
            "embedding_dimension": 384,  # Dimension der Embeddings (für MiniLM)
        }

        self._initialize_system()

    def _initialize_system(self):
        """Initialisiert das RAG-System"""
        logging.info("🚀 Initialisiere RAG-System...")

        try:
            # Lade Embedding-Modell
            logging.info(f"📥 Lade Embedding-Modell: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logging.info("✅ Embedding-Modell geladen")

            # Lade und verarbeite Corpus
            self._load_corpus()

            # Erstelle oder lade Vektor-Index
            self._setup_vector_index()

            logging.info("✅ RAG-System erfolgreich initialisiert")

        except Exception as e:
            logging.error(f"❌ Fehler bei der Initialisierung des RAG-Systems: {e}")
            raise

    def _load_corpus(self):
        """Lädt den Corpus aus der JSON-Datei"""
        logging.info(f"📚 Lade Corpus aus {self.corpus_path}")

        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Handle both formats: array of entries or object with entries
                if isinstance(data, list):
                    self.corpus_entries = data
                else:
                    self.corpus_entries = data.get("entries", [])

            logging.info(f"✅ Corpus geladen: {len(self.corpus_entries)} Einträge")

            # Erstelle Text-Liste für Embeddings
            self.corpus_texts = [entry["text"] for entry in self.corpus_entries]

        except Exception as e:
            logging.error(f"❌ Fehler beim Laden des Corpus: {e}")
            raise

    def _setup_vector_index(self):
        """Erstellt oder lädt den FAISS Vektor-Index"""
        # Verwende den konfigurierten models_path
        index_path = os.path.join(self.models_path, "rag_index.faiss")
        embeddings_path = os.path.join(self.models_path, "rag_embeddings.pkl")

        if os.path.exists(index_path) and os.path.exists(embeddings_path):
            # Lade vorhandenen Index
            logging.info("📥 Lade vorhandenen FAISS-Index...")
            self.index = faiss.read_index(index_path)

            with open(embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)

            logging.info("✅ Vorhandener Index geladen")
        else:
            # Erstelle neuen Index
            logging.info("🏗️ Erstelle neuen FAISS-Index...")
            self._create_vector_index()
            logging.info("✅ Neuer Index erstellt und gespeichert")

    def _create_vector_index(self):
        """Erstellt einen neuen FAISS Vektor-Index"""
        # Erstelle Embeddings für alle Corpus-Einträge
        logging.info("🔄 Erstelle Embeddings für Corpus...")
        self.embeddings = self.embedding_model.encode(
            self.corpus_texts, show_progress_bar=True, convert_to_numpy=True
        )

        # Normalisiere Embeddings für Kosinus-Ähnlichkeit
        faiss.normalize_L2(self.embeddings)

        # Erstelle FAISS Index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(
            dimension
        )  # Inner Product für normalisierte Vektoren = Kosinus-Ähnlichkeit

        # Füge Embeddings zum Index hinzu
        self.index.add(self.embeddings)

        # Speichere Index und Embeddings in models/ Verzeichnis
        # Verwende den konfigurierten models_path
        os.makedirs(self.models_path, exist_ok=True)

        index_path = os.path.join(self.models_path, "rag_index.faiss")
        embeddings_path = os.path.join(self.models_path, "rag_embeddings.pkl")

        faiss.write_index(self.index, index_path)

        with open(embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)

        logging.info(
            f"💾 Index gespeichert: {len(self.corpus_texts)} Dokumente, Dimension {dimension}"
        )

    def retrieve_relevant_documents(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Ruft relevante Dokumente für eine Query ab

        Args:
            query: Suchanfrage
            top_k: Anzahl der abzurufenden Dokumente (überschreibt Config)

        Returns:
            Liste der relevanten Dokumente mit Scores
        """
        if top_k is None:
            top_k = self.config["top_k"]

        # Erstelle Embedding für die Query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Suche nach ähnlichen Dokumenten
        scores, indices = self.index.search(query_embedding, top_k)

        # Sammle Ergebnisse
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.config["similarity_threshold"]:
                result = {
                    "document": self.corpus_entries[idx],
                    "score": float(score),
                    "text": self.corpus_texts[idx],
                    "index": int(idx),
                }
                results.append(result)

        return results

    def generate_context(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Generiert einen Kontext-String aus relevanten Dokumenten

        Args:
            query: Original-Query
            relevant_docs: Relevante Dokumente

        Returns:
            Kontext-String für das Modell
        """
        if not relevant_docs:
            return f"Query: {query}\n\nKein relevanter Kontext gefunden."

        context_parts = [f"Query: {query}\n"]

        # Sortiere Dokumente nach Relevanz
        sorted_docs = sorted(relevant_docs, key=lambda x: x["score"], reverse=True)

        context_parts.append("Relevante Informationen:")
        total_length = 0

        for i, doc in enumerate(sorted_docs, 1):
            doc_text = doc["text"]
            doc_length = len(doc_text)

            # Prüfe, ob das Hinzufügen die maximale Länge überschreiten würde
            if total_length + doc_length > self.config["max_context_length"]:
                break

            context_parts.append(f"\n{i}. {doc_text}")
            context_parts.append(f"   (Relevanz: {doc['score']:.3f})")

            total_length += doc_length

        return "\n".join(context_parts)

    def rag_answer(
        self, query: str, generation_model=None, generation_tokenizer=None
    ) -> Dict[str, Any]:
        """
        Generiert eine RAG-basierte Antwort

        Args:
            query: Die Frage des Benutzers
            generation_model: Optionales Generierungsmodell (z.B. fine-tuned Modell)
            generation_tokenizer: Optionales Tokenizer für das Generierungsmodell

        Returns:
            Dictionary mit Antwort, Kontext und Metadaten
        """
        logging.info(f"🔍 RAG-Abfrage: {query}")

        # 1. Retrieval: Finde relevante Dokumente
        relevant_docs = self.retrieve_relevant_documents(query)

        # 2. Context Building: Erstelle Kontext aus relevanten Dokumenten
        context = self.generate_context(query, relevant_docs)

        # 3. Generation: Generiere Antwort basierend auf Kontext
        if generation_model and generation_tokenizer:
            # Verwende bereitgestelltes Modell
            answer = self._generate_with_model(context, generation_model, generation_tokenizer)
        else:
            # Fallback: Verwende einfache Extraktion
            answer = self._extract_answer_from_context(context, relevant_docs)

        result = {
            "query": query,
            "answer": answer,
            "context": context,
            "relevant_documents": relevant_docs,
            "num_documents": len(relevant_docs),
            "timestamp": datetime.now().isoformat(),
            "method": "model_generation" if generation_model else "context_extraction",
        }

        logging.info(f"✅ RAG-Antwort generiert: {len(relevant_docs)} Dokumente verwendet")
        return result

    def _generate_with_model(self, context: str, model, tokenizer) -> str:
        """
        Generiert eine Antwort mit einem bereitgestellten Modell

        Args:
            context: Kontext-String
            model: TensorFlow/Keras Modell
            tokenizer: Keras Tokenizer

        Returns:
            Generierte Antwort
        """
        try:
            # Tokenisiere Kontext
            input_sequence = tokenizer.texts_to_sequences([context])
            input_padded = tf.keras.preprocessing.sequence.pad_sequences(
                input_sequence, maxlen=self.config.get("maxlen", 100), padding="post"
            )

            # Generiere Antwort
            prediction = model.predict(input_padded, verbose=0)
            predicted_sequence = tf.argmax(prediction, axis=-1)[0]

            # Konvertiere zurück zu Text
            predicted_sequence = predicted_sequence.numpy()
            reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

            predicted_text = []
            for token in predicted_sequence:
                if token > 0 and token in reverse_word_index:
                    word = reverse_word_index[token]
                    if word not in ["<OOV>", "<UNK>"]:
                        predicted_text.append(word)
                if len(predicted_text) >= 50:  # Begrenze Länge
                    break

            return " ".join(predicted_text).strip()

        except Exception as e:
            logging.warning(f"⚠️ Fehler bei der Modell-Generierung: {e}")
            return "Fehler bei der Antwortgenerierung."

    def _extract_answer_from_context(
        self, context: str, relevant_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Extrahiert eine verbesserte, strukturierte Antwort aus dem Kontext

        Args:
            context: Kontext-String
            relevant_docs: Relevante Dokumente

        Returns:
            Strukturierte Antwort
        """
        if not relevant_docs:
            return "Ich konnte keine relevanten politischen Informationen zu Ihrer Frage finden."

        # Sammle Informationen aus allen relevanten Dokumenten
        topics = []
        key_points = []
        sources = []

        for doc in relevant_docs[:3]:  # Verwende Top 3 Dokumente
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            # Extrahiere Topic
            topic = metadata.get("topic", "allgemein")
            if topic not in topics:
                topics.append(topic)

            # Extrahiere Schlüsselinformationen
            sentences = text.split(".")
            for sentence in sentences[:2]:  # Erste 2 Sätze pro Dokument
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 150:
                    key_points.append(sentence)

            # Sammle Quellen
            source = metadata.get("source", "unbekannt")
            if source not in sources:
                sources.append(source)

        # Erstelle strukturierte Antwort basierend auf Topics
        if "klima" in topics:
            return self._generate_climate_answer(key_points, sources)
        elif any(topic in ["politik", "regierung", "bundestag"] for topic in topics):
            return self._generate_politics_answer(key_points, sources)
        elif "energie" in topics or "wirtschaft" in topics:
            return self._generate_economy_answer(key_points, sources)
        else:
            return self._generate_general_answer(key_points, sources)

    def _generate_climate_answer(self, key_points: List[str], sources: List[str]) -> str:
        """Generiert strukturierte Klimapolitik-Antwort"""
        if not key_points:
            return "📋 **Allgemeine Information:** Deutschland verfolgt ambitionierte Klimaziele bis 2045."

        response = "🌍 **Klima-Analyse:**\n\n"
        for i, point in enumerate(key_points[:4], 1):
            response += f"{i}. {point}.\n"

        if sources:
            response += f"\n♻️ **Quelle:** Basiert auf {len(sources)} Klima-Dokumenten"
        return response

    def _generate_politics_answer(self, key_points: List[str], sources: List[str]) -> str:
        """Generiert strukturierte Politik-Antwort"""
        if not key_points:
            return "📋 **Allgemeine Information:** Deutschland setzt sich für stabile politische Rahmenbedingungen ein."

        response = "🏛️ **Politische Analyse:**\n\n"
        for i, point in enumerate(key_points[:4], 1):
            response += f"{i}. {point}.\n"

        if sources:
            response += f"\n📊 **Quelle:** Basiert auf {len(sources)} politischen Quellen"
        return response

    def _generate_economy_answer(self, key_points: List[str], sources: List[str]) -> str:
        """Generiert strukturierte Wirtschafts-Antwort"""
        if not key_points:
            return "📋 **Allgemeine Information:** Deutschland fördert nachhaltiges Wirtschaftswachstum."

        response = "💼 **Wirtschaftliche Analyse:**\n\n"
        for i, point in enumerate(key_points[:4], 1):
            response += f"{i}. {point}.\n"

        if sources:
            response += f"\n📈 **Quelle:** Basiert auf {len(sources)} wirtschaftlichen Dokumenten"
        return response

    def _generate_general_answer(self, key_points: List[str], sources: List[str]) -> str:
        """Generiert allgemeine strukturierte Antwort"""
        if not key_points:
            return "📋 **Allgemeine Information:** Deutschland will bis 2045 klimaneutral werden. Alle Sektoren müssen ihren Beitrag leisten."

        response = "📋 **Allgemeine Information:**\n\n"
        for i, point in enumerate(key_points[:3], 1):
            response += f"{i}. {point}.\n"

        if sources:
            response += f"\n📚 **Quelle:** Basiert auf {len(sources)} Dokumenten"
        return response

    def update_corpus(self, new_entries: List[Dict[str, Any]]):
        """
        Aktualisiert den Corpus mit neuen Einträgen und rebuildet den Index

        Args:
            new_entries: Neue Corpus-Einträge
        """
        logging.info(f"🔄 Aktualisiere Corpus mit {len(new_entries)} neuen Einträgen")

        # Füge neue Einträge hinzu
        self.corpus_entries.extend(new_entries)
        self.corpus_texts.extend([entry["text"] for entry in new_entries])

        # Rebuild Index
        self._create_vector_index()

        logging.info("✅ Corpus und Index aktualisiert")

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über das RAG-System zurück"""
        return {
            "total_documents": len(self.corpus_entries),
            "embedding_dimension": (self.embeddings.shape[1] if self.embeddings is not None else 0),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "config": self.config,
        }


# Standalone-Funktionen für einfache Verwendung
def initialize_rag_system(corpus_path: str = "corpus.json") -> RAGSystem:
    """
    Initialisiert ein RAG-System

    Args:
        corpus_path: Pfad zur Corpus-Datei

    Returns:
        Initialisiertes RAGSystem
    """
    return RAGSystem(corpus_path=corpus_path)


def rag_query(
    query: str, rag_system: RAGSystem, generation_model=None, generation_tokenizer=None
) -> Dict[str, Any]:
    """
    Führt eine RAG-Abfrage aus

    Args:
        query: Suchanfrage
        rag_system: RAGSystem Instanz
        generation_model: Optionales Generierungsmodell
        generation_tokenizer: Optionales Tokenizer

    Returns:
        RAG-Ergebnis
    """
    return rag_system.rag_answer(query, generation_model, generation_tokenizer)


if __name__ == "__main__":
    # Test des RAG-Systems
    logging.basicConfig(level=logging.INFO)

    print("🧪 Teste RAG-System...")

    try:
        # Initialisiere System
        rag = initialize_rag_system()

        # Test-Abfrage
        result = rag_query("Was ist die Klimapolitik der Bundesregierung?", rag)

        print(f"📝 Antwort: {result['answer']}")
        print(f"📊 Relevante Dokumente: {result['num_documents']}")
        print(f"🎯 Methode: {result['method']}")

        # Statistiken
        stats = rag.get_statistics()
        print(
            f"📈 Statistiken: {stats['total_documents']} Dokumente, {stats['embedding_dimension']}D Embeddings"
        )

        print("✅ RAG-System funktioniert!")

    except Exception as e:
        print(f"❌ Fehler beim Testen: {e}")
        import traceback

        traceback.print_exc()
