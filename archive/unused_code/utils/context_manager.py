"""
Kontext-Manager für die Bundeskanzler-KI.
Implementiert erweitertes Kontextverständnis und Themenerkennung.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


class ContextManager:
    def __init__(
        self,
        window_size: int = 5,
        max_context_length: int = 1000,
        min_context_similarity: float = 0.3,
    ):
        """
        Initialisiert den Kontext-Manager.

        Args:
            window_size: Anzahl der vorherigen Interaktionen im Kontext
            max_context_length: Maximale Länge des Gesamtkontexts
            min_context_similarity: Minimale Ähnlichkeit für Kontextrelevanz
        """
        self.window_size = window_size
        self.max_context_length = max_context_length
        self.min_context_similarity = min_context_similarity

        # Kontext-Speicher
        self.interaction_history = deque(maxlen=window_size)
        self.context_embeddings = {}

        # Thematischer Kontext
        self.current_topic = None
        self.topic_history = deque(maxlen=window_size)

    def add_interaction(
        self,
        query: str,
        response: str,
        embeddings: np.ndarray,
        topic: Optional[str] = None,
    ) -> None:
        """
        Fügt eine neue Interaktion zum Kontext hinzu.

        Args:
            query: Nutzereingabe
            response: System-Antwort
            embeddings: Vektor-Embeddings der Interaktion
            topic: Erkanntes Thema (optional)
        """
        interaction = {
            "query": query,
            "response": response,
            "embeddings": embeddings,
            "topic": topic,
            "timestamp": tf.timestamp(),
        }

        self.interaction_history.append(interaction)
        if topic:
            self.topic_history.append(topic)
            self._update_current_topic()

    def get_relevant_context(self, query_embeddings: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Findet relevanten Kontext basierend auf der aktuellen Anfrage.

        Args:
            query_embeddings: Embeddings der aktuellen Anfrage
            top_k: Anzahl der relevantesten Kontexte

        Returns:
            Liste der relevantesten vorherigen Interaktionen
        """
        if not self.interaction_history:
            return []

        # Ähnlichkeiten berechnen
        similarities = []
        for interaction in self.interaction_history:
            similarity = self._compute_similarity(query_embeddings, interaction["embeddings"])
            similarities.append((similarity, interaction))

        # Nach Ähnlichkeit sortieren
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Nur relevante Kontexte zurückgeben
        relevant_contexts = []
        for similarity, interaction in similarities[:top_k]:
            if similarity >= self.min_context_similarity:
                relevant_contexts.append(interaction)

        return relevant_contexts

    def get_current_topic(self) -> Optional[str]:
        """Gibt das aktuelle Hauptthema zurück."""
        return self.current_topic

    def get_topic_context(self) -> List[str]:
        """Gibt die thematische Entwicklung zurück."""
        return list(self.topic_history)

    def _update_current_topic(self) -> None:
        """Aktualisiert das aktuelle Hauptthema basierend auf der Historie."""
        if not self.topic_history:
            self.current_topic = None
            return

        # Häufigste Themen finden
        from collections import Counter

        topic_counts = Counter(self.topic_history)

        # Aktuellstes häufigstes Thema wählen
        most_common = topic_counts.most_common()
        if most_common:
            self.current_topic = most_common[0][0]

    def _compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Embedding-Vektoren.
        """
        # Normalisierung
        norm1 = np.linalg.norm(embeddings1)
        norm2 = np.linalg.norm(embeddings2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Kosinus-Ähnlichkeit
        return np.dot(embeddings1, embeddings2) / (norm1 * norm2)

    def summarize_context(self) -> Dict:
        """
        Erstellt eine Zusammenfassung des aktuellen Kontexts.

        Returns:
            Dict mit Kontextinformationen
        """
        summary = {
            "current_topic": self.current_topic,
            "topic_history": list(self.topic_history),
            "interaction_count": len(self.interaction_history),
            "context_window": self.window_size,
            "last_interaction": None,
        }

        if self.interaction_history:
            last = self.interaction_history[-1]
            summary["last_interaction"] = {
                "query": last["query"],
                "response": last["response"],
                "topic": last["topic"],
                "timestamp": float(last["timestamp"]),
            }

        return summary
