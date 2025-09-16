"""
Enhanced Context Processor für die Bundeskanzler KI.
Stellt erweiterte Kontextverarbeitung mit Memory-Integration bereit.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class MemoryItem:
    """Repräsentiert ein einzelnes Memory-Element"""

    def __init__(
        self,
        content: str,
        embedding: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
        importance: float = 1.0,
    ):
        self.content = content
        self.embedding = embedding
        self.timestamp = timestamp or datetime.now()
        self.importance = importance
        self.access_count = 0
        self.last_accessed = self.timestamp
        self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das MemoryItem zu einem Dictionary"""
        return {
            "content": self.content,
            "embedding": (self.embedding.tolist() if self.embedding is not None else None),
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Erstellt ein MemoryItem aus einem Dictionary"""
        embedding = np.array(data["embedding"]) if data.get("embedding") else None
        timestamp = datetime.fromisoformat(data["timestamp"])
        last_accessed = datetime.fromisoformat(data["last_accessed"])

        item = cls(
            content=data["content"],
            embedding=embedding,
            timestamp=timestamp,
            importance=data.get("importance", 1.0),
        )
        item.access_count = data.get("access_count", 0)
        item.last_accessed = last_accessed
        item.tags = data.get("tags", [])
        return item


class HierarchicalMemory:
    """Hierarchisches Memory-System mit mehreren Ebenen"""

    def __init__(self, base_path: str = "./memory", levels: int = 3):
        self.base_path = base_path
        self.levels = levels
        self.memory_levels = {}
        self.short_term_memory = []
        self.short_term_capacity = 50
        self.long_term_memory = []

        # Erstelle Memory-Verzeichnisse für jede Ebene
        for level in range(levels):
            level_path = os.path.join(base_path, f"level_{level}")
            os.makedirs(level_path, exist_ok=True)
            self.memory_levels[level] = {
                "path": level_path,
                "items": [],
                "max_items": 100 * (level + 1),  # Höhere Ebenen haben mehr Kapazität
            }

        self._load_memory()

    def _load_memory(self):
        """Lädt gespeicherte Memory-Daten"""
        for level, level_data in self.memory_levels.items():
            memory_file = os.path.join(level_data["path"], "memory.json")
            try:
                if os.path.exists(memory_file):
                    with open(memory_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        level_data["items"] = [
                            MemoryItem.from_dict(item_data) for item_data in data.get("items", [])
                        ]
            except Exception as e:
                print(f"Warnung: Konnte Memory Level {level} nicht laden: {e}")
                level_data["items"] = []

    def _save_memory(self, level: int):
        """Speichert Memory-Daten für eine Ebene"""
        level_data = self.memory_levels[level]
        memory_file = os.path.join(level_data["path"], "memory.json")

        try:
            data = {
                "level": level,
                "items": [item.to_dict() for item in level_data["items"]],
                "last_updated": datetime.now().isoformat(),
            }
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warnung: Konnte Memory Level {level} nicht speichern: {e}")

    def add_item(self, item: MemoryItem, level: int = 0):
        """Fügt ein MemoryItem zu einer bestimmten Ebene hinzu"""
        if level not in self.memory_levels:
            raise ValueError(f"Ungültige Ebene: {level}")

        level_data = self.memory_levels[level]
        level_data["items"].append(item)

        # Begrenze Anzahl der Items pro Ebene
        if len(level_data["items"]) > level_data["max_items"]:
            # Entferne älteste Items mit niedrigster Importance
            level_data["items"].sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
            level_data["items"] = level_data["items"][: level_data["max_items"]]

        self._save_memory(level)

    def get_items(self, level: int, limit: Optional[int] = None) -> List[MemoryItem]:
        """Ruft MemoryItems von einer Ebene ab"""
        if level not in self.memory_levels:
            return []

        items = self.memory_levels[level]["items"]
        if limit:
            items = items[:limit]

        # Aktualisiere access_count und last_accessed
        for item in items:
            item.access_count += 1
            item.last_accessed = datetime.now()

        return items

    def promote_item(self, item: MemoryItem, from_level: int, to_level: int):
        """Befördert ein Item zu einer höheren Ebene"""
        if from_level not in self.memory_levels or to_level not in self.memory_levels:
            return

        # Entferne von der niedrigeren Ebene
        self.memory_levels[from_level]["items"] = [
            i for i in self.memory_levels[from_level]["items"] if i != item
        ]

        # Füge zur höheren Ebene hinzu
        self.add_item(item, to_level)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über das hierarchische Memory zurück"""
        total_memories = sum(len(level_data["items"]) for level_data in self.memory_levels.values())

        stats = {
            "total_levels": self.levels,
            "base_path": self.base_path,
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "total_memories": total_memories,
            "level_stats": {},
        }

        for level, level_data in self.memory_levels.items():
            stats["level_stats"][level] = {
                "item_count": len(level_data["items"]),
                "max_items": level_data["max_items"],
                "path": level_data["path"],
            }

        return stats

    def add_memory(
        self,
        content: str,
        embedding: Optional[np.ndarray] = None,
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Fügt ein neues Memory-Element hinzu"""
        item = MemoryItem(content=content, embedding=embedding, importance=importance)
        if tags:
            item.tags = tags
        if metadata:
            item.metadata = metadata
        self.short_term_memory.append(item)
        self.add_item(item, level=0)

    def retrieve_memories(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemoryItem]:
        """Ruft relevante Memories basierend auf Embedding-Ähnlichkeit ab"""
        all_items = []
        for level_data in self.memory_levels.values():
            all_items.extend(level_data["items"])

        # Berechne Ähnlichkeiten und sortiere
        similarities = []
        for item in all_items:
            if item.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, item.embedding)
                similarities.append((item, similarity))

        # Sortiere nach Ähnlichkeit und return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in similarities[:top_k]]

    def _consolidate_memories(self):
        """Konsolidiert Memories von Short-Term zu Long-Term Memory"""
        # Verschiebe alte oder wichtige Memories von short_term zu long_term
        to_consolidate = []
        for item in self.short_term_memory:
            # Konsolidiere Items die älter als 1 Tag sind oder hohe Wichtigkeit haben
            age_days = (datetime.now() - item.timestamp).days
            if age_days > 1 or item.importance > 0.8:
                to_consolidate.append(item)

        for item in to_consolidate:
            self.long_term_memory.append(item)
            self.short_term_memory.remove(item)

    def search_by_tags(self, tags: List[str]) -> List[MemoryItem]:
        """Sucht Memory-Items nach Tags"""
        results = []
        for level_data in self.memory_levels.values():
            for item in level_data["items"]:
                if any(tag in item.tags for tag in tags):
                    results.append(item)
        return results

    def search_by_content(self, query: str, fuzzy: bool = False) -> List[MemoryItem]:
        """Sucht Memory-Items nach Inhalt"""
        results = []
        query_lower = query.lower()
        for level_data in self.memory_levels.values():
            for item in level_data["items"]:
                if fuzzy:
                    # Einfache Fuzzy-Suche: enthält alle Wörter
                    query_words = query_lower.split()
                    content_lower = item.content.lower()
                    if all(word in content_lower for word in query_words):
                        results.append(item)
                else:
                    if query_lower in item.content.lower():
                        results.append(item)
        return results

    def forget_old_memories(self, days: int = 30, max_age_days: Optional[int] = None):
        """Vergisst alte Memories"""
        cutoff_days = max_age_days if max_age_days is not None else days
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=cutoff_days)
        forgotten_count = 0
        for level_data in self.memory_levels.values():
            old_count = len(level_data["items"])
            level_data["items"] = [item for item in level_data["items"] if item.timestamp > cutoff]
            forgotten_count += old_count - len(level_data["items"])
        return forgotten_count

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Berechnet Kosinus-Ähnlichkeit zwischen zwei Vektoren"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0.0

    def _calculate_time_weight(self, timestamp: datetime) -> float:
        """Berechnet Zeitgewicht basierend auf Alter"""
        if isinstance(timestamp, MemoryItem):
            timestamp = timestamp.timestamp
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        # Jüngere Items haben höheres Gewicht
        return 1.0 / (1.0 + age_hours / 24)  # Abnahme über Tage


class AdaptiveMemoryManager:
    """Adaptiver Memory-Manager der Items zwischen Ebenen verschiebt"""

    def __init__(self, hierarchical_memory: HierarchicalMemory):
        self.hierarchical_memory = hierarchical_memory
        self.access_threshold = 10  # Mindestzugriffe für Beförderung
        self.importance_threshold = 0.7  # Mindestimportance für Beförderung

    def update_item_access(self, item: MemoryItem):
        """Aktualisiert Zugriffsstatistiken eines Items"""
        item.access_count += 1
        item.last_accessed = datetime.now()

    def should_promote(self, item: MemoryItem, current_level: int) -> bool:
        """Entscheidet, ob ein Item befördert werden sollte"""
        if current_level >= self.hierarchical_memory.levels - 1:
            return False  # Maximale Ebene erreicht

        return (
            item.access_count >= self.access_threshold
            and item.importance >= self.importance_threshold
        )

    def adapt_memory(self):
        """Passe Memory-Struktur adaptiv an"""
        for level in range(self.hierarchical_memory.levels - 1):  # Alle außer der höchsten Ebene
            items_to_promote = []

            for item in self.hierarchical_memory.memory_levels[level]["items"]:
                if self.should_promote(item, level):
                    items_to_promote.append(item)

            # Befördere Items zur nächsten Ebene
            for item in items_to_promote:
                self.hierarchical_memory.promote_item(item, level, level + 1)

    def adaptive_importance_calculation(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        user_feedback: Optional[float] = None,
    ) -> float:
        """Berechnet adaptive Importance basierend auf verschiedenen Faktoren"""
        importance = 0.5  # Basis-Wichtigkeit

        # Content-basierte Faktoren
        if content:
            # Längere Inhalte sind tendenziell wichtiger
            content_length_factor = min(len(content) / 1000, 1.0)  # Max bei 1000 Zeichen
            importance += content_length_factor * 0.2

            # Schlüsselwörter erhöhen Wichtigkeit
            keywords = ["politik", "regierung", "gesetz", "wichtig", "dringend"]
            keyword_count = sum(1 for keyword in keywords if keyword in content.lower())
            importance += min(keyword_count * 0.1, 0.3)

        # Context-basierte Faktoren
        if context:
            if context.get("source") == "official":
                importance += 0.2
            if "urgency" in context:
                importance += context["urgency"] * 0.3

        # User feedback
        if user_feedback is not None:
            importance += user_feedback * 0.2

        return min(importance, 1.0)

    def suggest_memory_cleanup(self) -> Dict[str, Any]:
        """Schlägt Memory-Bereinigungen vor"""
        suggestions = {}

        # Finde Memories mit niedriger Wichtigkeit
        low_importance_items = []
        for level_data in self.hierarchical_memory.memory_levels.values():
            for item in level_data["items"]:
                if item.importance < self.importance_threshold:
                    low_importance_items.append(item)

        if low_importance_items:
            suggestions["low_importance"] = {
                "count": len(low_importance_items),
                "items": low_importance_items[:5],  # Max 5 Beispiele
            }

        return suggestions

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über die Memory-Adaption zurück"""
        return {
            "access_threshold": self.access_threshold,
            "importance_threshold": self.importance_threshold,
            "memory_stats": self.hierarchical_memory.get_memory_stats(),
        }


class EnhancedContextProcessor:
    """
    Erweiterter Kontextprozessor mit Memory-Integration.
    """

    def __init__(self, memory_path: str = "./api_memory", embedding_dim: int = 512):
        """
        Initialisiert den Enhanced Context Processor.

        Args:
            memory_path: Pfad zum Memory-Verzeichnis
            embedding_dim: Dimensionalität der Embeddings
        """
        self.memory_path = memory_path
        self.embedding_dim = embedding_dim
        self.context_memory = []
        self.memory_file = os.path.join(memory_path, "context_memory.json")

        # Erstelle Memory-Verzeichnis falls es nicht existiert
        os.makedirs(memory_path, exist_ok=True)

        # Lade bestehende Memory-Daten
        self._load_memory()

    def _load_memory(self):
        """Lädt gespeicherte Memory-Daten."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.context_memory = data.get("contexts", [])
        except Exception as e:
            print(f"Warnung: Konnte Memory nicht laden: {e}")
            self.context_memory = []

    def _save_memory(self):
        """Speichert Memory-Daten."""
        try:
            data = {
                "contexts": self.context_memory,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warnung: Konnte Memory nicht speichern: {e}")

    def add_context(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fügt neuen Kontext zum Memory hinzu.

        Args:
            text: Kontext-Text
            embedding: Optionale Embedding-Daten
            metadata: Zusätzliche Metadaten
        """
        context_entry = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "embedding": embedding.tolist() if embedding is not None else None,
        }

        self.context_memory.append(context_entry)

        # Behalte nur die letzten 1000 Einträge
        if len(self.context_memory) > 1000:
            self.context_memory = self.context_memory[-1000:]

        self._save_memory()

    def get_relevant_context(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Findet relevanten Kontext basierend auf der Query.

        Args:
            query: Suchanfrage
            max_results: Maximale Anzahl der Ergebnisse

        Returns:
            Liste der relevanten Kontext-Einträge
        """
        if not self.context_memory:
            return []

        # Einfache textbasierte Suche (kann später durch semantische Suche ersetzt werden)
        query_lower = query.lower()
        relevant_contexts = []

        for context in reversed(self.context_memory):  # Neueste zuerst
            if query_lower in context["text"].lower():
                relevant_contexts.append(context)
                if len(relevant_contexts) >= max_results:
                    break

        return relevant_contexts

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über das Memory zurück.

        Returns:
            Dictionary mit Memory-Statistiken
        """
        return {
            "total_contexts": len(self.context_memory),
            "memory_path": self.memory_path,
            "embedding_dim": self.embedding_dim,
            "last_updated": datetime.now().isoformat(),
        }

    def clear_memory(self) -> None:
        """Löscht alle Memory-Daten."""
        self.context_memory = []
        self._save_memory()
