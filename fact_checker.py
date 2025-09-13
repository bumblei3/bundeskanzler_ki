"""
Fact Checker für die Bundeskanzler-KI.
Implementiert Faktenprüfung, Quellenverifizierung und zeitliche Aktualität.
"""
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import datetime
import json
import numpy as np

class FactChecker:
    """
    Überprüft Fakten und Aussagen auf Richtigkeit und Aktualität.
    """
    def __init__(
        self,
        facts_database: str = "facts_db.json",
        confidence_threshold: float = 0.8,
        max_age_days: int = 30
    ):
        self.confidence_threshold = confidence_threshold
        self.max_age_days = max_age_days
        self.facts_db = self._load_facts_database(facts_database)
        
    def _load_facts_database(self, database_path: str) -> Dict:
        """Lädt die Faktendatenbank."""
        try:
            with open(database_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Erstelle leere Datenbank falls nicht vorhanden
            return {
                "facts": {},
                "sources": {},
                "last_updated": str(datetime.datetime.now())
            }
            
    def verify_statement(
        self,
        statement: str,
        embeddings: tf.Tensor
    ) -> Tuple[bool, float, List[str]]:
        """
        Überprüft eine Aussage auf Faktengenauigkeit.
        
        Args:
            statement: Zu überprüfende Aussage
            embeddings: Einbettungen der Aussage
            
        Returns:
            Tuple aus:
            - Wahrheitswert
            - Konfidenz
            - Liste von Quellen
        """
        # Ähnlichkeitssuche in der Faktendatenbank
        most_similar = self._find_similar_facts(embeddings)
        
        # Konfidenz berechnen
        confidence = self._calculate_confidence(most_similar)
        
        # Zeitliche Aktualität prüfen
        is_current = self._check_temporal_validity(most_similar)
        
        # Quellen sammeln
        sources = self._collect_sources(most_similar)
        
        return is_current and confidence >= self.confidence_threshold, confidence, sources
        
    def _find_similar_facts(
        self,
        embeddings: tf.Tensor,
        top_k: int = 5
    ) -> List[Dict]:
        """Findet ähnliche Fakten in der Datenbank."""
        facts = []
        for fact_id, fact in self.facts_db["facts"].items():
            similarity = tf.keras.losses.cosine_similarity(
                embeddings,
                tf.constant(fact["embeddings"])
            )
            facts.append((fact_id, float(similarity)))
            
        # Top-K ähnlichste Fakten
        facts.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                **self.facts_db["facts"][fact_id],
                "similarity": sim
            }
            for fact_id, sim in facts[:top_k]
        ]
        
    def _calculate_confidence(self, similar_facts: List[Dict]) -> float:
        """Berechnet die Konfidenz basierend auf ähnlichen Fakten."""
        if not similar_facts:
            return 0.0
            
        # Gewichteter Durchschnitt der Ähnlichkeiten
        weights = np.array([fact["similarity"] for fact in similar_facts])
        weights = np.exp(weights) / np.sum(np.exp(weights))  # Softmax
        
        confidence = np.sum(weights * np.array([
            fact.get("confidence", 0.5) for fact in similar_facts
        ]))
        
        return float(confidence)
        
    def _check_temporal_validity(self, similar_facts: List[Dict]) -> bool:
        """Überprüft die zeitliche Gültigkeit der Fakten."""
        if not similar_facts:
            return False
            
        now = datetime.datetime.now()
        
        for fact in similar_facts:
            fact_date = datetime.datetime.fromisoformat(fact.get("date", "2000-01-01"))
            if (now - fact_date).days > self.max_age_days:
                return False
                
        return True
        
    def _collect_sources(self, similar_facts: List[Dict]) -> List[str]:
        """Sammelt relevante Quellen für die Fakten."""
        sources = set()
        
        for fact in similar_facts:
            fact_sources = fact.get("sources", [])
            for source_id in fact_sources:
                if source_id in self.facts_db["sources"]:
                    source_info = self.facts_db["sources"][source_id]
                    sources.add(f"{source_info['title']} ({source_info['date']})")
                    
        return list(sources)
        
    def add_fact(
        self,
        statement: str,
        embeddings: tf.Tensor,
        sources: List[str],
        confidence: float = 1.0
    ) -> str:
        """
        Fügt einen neuen Fakt zur Datenbank hinzu.
        
        Args:
            statement: Der neue Fakt
            embeddings: Einbettungen des Fakts
            sources: Quellen-IDs
            confidence: Konfidenzwert
            
        Returns:
            ID des neuen Fakts
        """
        fact_id = f"fact_{len(self.facts_db['facts'])}"
        
        self.facts_db["facts"][fact_id] = {
            "statement": statement,
            "embeddings": embeddings.numpy().tolist(),
            "sources": sources,
            "confidence": confidence,
            "date": str(datetime.datetime.now())
        }
        
        return fact_id