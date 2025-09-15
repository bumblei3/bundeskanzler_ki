"""
Adaptive Response Manager für die Bundeskanzler-KI.
Steuert die Komplexität und Personalisierung von Antworten.
"""
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

from typing import Dict, List, Optional, Tuple
import numpy as np

class AdaptiveResponseManager:
    """
    Verwaltet die adaptive Anpassung von Antworten basierend auf Nutzerkontext.
    """
    def __init__(
        self,
        complexity_levels: int = 5,
        min_tokens: int = 20,
        max_tokens: int = 500
    ):
        self.complexity_levels = complexity_levels
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.user_profiles = {}
        
    def get_complexity_params(
        self,
        user_id: str,
        context: Dict
    ) -> Dict[str, float]:
        """
        Bestimmt Parameter für die Antwortkomplexität.
        
        Args:
            user_id: ID des Nutzers
            context: Aktueller Konversationskontext
            
        Returns:
            Parameter für die Antwortgenerierung
        """
        # Nutzerprofil abrufen oder erstellen
        profile = self._get_or_create_profile(user_id)
        
        # Komplexität basierend auf Kontext anpassen
        base_complexity = profile["base_complexity"]
        context_complexity = self._analyze_context_complexity(context)
        
        # Dynamische Anpassung
        target_complexity = self._adjust_complexity(
            base_complexity,
            context_complexity,
            profile["learning_rate"]
        )
        
        return {
            "target_complexity": target_complexity,
            "min_length": self._scale_length(target_complexity, "min"),
            "max_length": self._scale_length(target_complexity, "max"),
            "vocab_level": self._get_vocab_level(target_complexity),
            "structure_complexity": self._get_structure_complexity(target_complexity)
        }
        
    def update_user_profile(
        self,
        user_id: str,
        interaction_data: Dict
    ) -> None:
        """
        Aktualisiert das Nutzerprofil basierend auf Interaktionen.
        
        Args:
            user_id: ID des Nutzers
            interaction_data: Daten aus der Interaktion
        """
        profile = self._get_or_create_profile(user_id)
        
        # Erfolgsmetriken auswerten
        success = interaction_data.get("success", 0.5)
        comprehension = interaction_data.get("comprehension", 0.5)
        
        # Profil anpassen
        if success < 0.3:  # Zu komplex
            profile["base_complexity"] *= 0.9
        elif success > 0.7 and comprehension > 0.7:  # Zu einfach
            profile["base_complexity"] = min(
                1.0,
                profile["base_complexity"] * 1.1
            )
            
        # Lernrate anpassen
        profile["learning_rate"] = max(
            0.1,
            min(1.0, profile["learning_rate"] + (success - 0.5) * 0.1)
        )
        
        # Interaktionshistorie aktualisieren
        profile["history"].append(interaction_data)
        if len(profile["history"]) > 100:
            profile["history"] = profile["history"][-100:]
            
    def _get_or_create_profile(self, user_id: str) -> Dict:
        """Erstellt oder holt ein Nutzerprofil."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "base_complexity": 0.5,
                "learning_rate": 0.5,
                "history": []
            }
        return self.user_profiles[user_id]
        
    def _analyze_context_complexity(self, context: Dict) -> float:
        """Analysiert die Komplexität des aktuellen Kontexts."""
        factors = [
            len(context.get("current_topic", "")),
            len(context.get("recent_messages", [])),
            context.get("technical_terms", 0),
            context.get("question_complexity", 0.5)
        ]
        return np.mean([f/100 if isinstance(f, int) else f for f in factors])
        
    def _adjust_complexity(
        self,
        base: float,
        context: float,
        learning_rate: float
    ) -> float:
        """Passt die Komplexität dynamisch an."""
        target = base * (1 - learning_rate) + context * learning_rate
        return max(0.1, min(1.0, target))
        
    def _scale_length(self, complexity: float, bound_type: str) -> int:
        """Skaliert die Antwortlänge basierend auf Komplexität."""
        if bound_type == "min":
            return int(self.min_tokens * (1 + complexity))
        else:
            return int(self.max_tokens * complexity)
            
    def _get_vocab_level(self, complexity: float) -> float:
        """Bestimmt das Vokabularniveau."""
        return complexity
        
    def _get_structure_complexity(self, complexity: float) -> Dict[str, float]:
        """Bestimmt Parameter für die Satzstruktur."""
        return {
            "max_clause_depth": 1 + int(3 * complexity),
            "conjunction_probability": 0.2 + 0.6 * complexity,
            "technical_term_ratio": complexity * 0.3
        }