#!/usr/bin/env python3
"""
🧠 Nutzerprofil-System für Bundeskanzler-KI
============================================

Erstellt und verwaltet individuelle Nutzerprofile für personalisierte Antworten:
- Interessen und Präferenzen speichern
- Nutzerverhalten analysieren
- Personalisierte Antworten generieren
- Feedback für kontinuierliche Verbesserung

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Repräsentiert ein Nutzerprofil"""

    user_id: str
    created_at: str
    last_active: str
    preferences: Dict[str, Any]
    interests: Dict[str, float]  # Thema -> Interesse-Score (0-1)
    interaction_history: List[Dict[str, Any]]
    feedback_scores: Dict[str, float]
    personalization_level: str  # "basic", "intermediate", "advanced"

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das Profil zu einem Dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Erstellt ein Profil aus einem Dictionary"""
        return cls(**data)


class UserProfileManager:
    """
    Verwaltet Nutzerprofile für personalisierte KI-Antworten
    """

    def __init__(self, profiles_dir: str = "data/user_profiles"):
        """
        Initialisiert den UserProfileManager

        Args:
            profiles_dir: Verzeichnis für Nutzerprofile
        """
        self.profiles_dir = profiles_dir
        self.profiles_cache = {}  # user_id -> UserProfile
        self._ensure_profiles_dir()

        logger.info(f"🧠 UserProfileManager initialisiert: {profiles_dir}")

    def _ensure_profiles_dir(self):
        """Stellt sicher, dass das Profile-Verzeichnis existiert"""
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
            logger.info(f"📁 Profile-Verzeichnis erstellt: {self.profiles_dir}")

    def _get_profile_path(self, user_id: str) -> str:
        """Gibt den Pfad für ein Nutzerprofil zurück"""
        # Verwende Hash für Dateinamen (für Datenschutz)
        hashed_id = hashlib.md5(user_id.encode()).hexdigest()
        return os.path.join(self.profiles_dir, f"{hashed_id}.json")

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """
        Holt ein bestehendes Profil oder erstellt ein neues

        Args:
            user_id: Eindeutige Nutzer-ID

        Returns:
            UserProfile-Instanz
        """
        # Cache prüfen
        if user_id in self.profiles_cache:
            return self.profiles_cache[user_id]

        # Datei prüfen
        profile_path = self._get_profile_path(user_id)
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = UserProfile.from_dict(data)
                self.profiles_cache[user_id] = profile
                logger.debug(f"✅ Profil geladen: {user_id}")
                return profile
            except Exception as e:
                logger.warning(f"❌ Fehler beim Laden des Profils {user_id}: {e}")

        # Neues Profil erstellen
        now = datetime.now().isoformat()
        profile = UserProfile(
            user_id=user_id,
            created_at=now,
            last_active=now,
            preferences={
                "language": "de",
                "detail_level": "medium",  # "low", "medium", "high"
                "response_style": "formal",  # "formal", "casual", "technical"
                "notification_preferences": {"email_updates": False, "feedback_requests": True},
            },
            interests={
                "klima": 0.5,
                "wirtschaft": 0.5,
                "soziales": 0.5,
                "digital": 0.5,
                "europa": 0.5,
                "sicherheit": 0.5,
                "bildung": 0.5,
                "gesundheit": 0.5,
            },
            interaction_history=[],
            feedback_scores={
                "overall_satisfaction": 0.0,
                "relevance": 0.0,
                "accuracy": 0.0,
                "helpfulness": 0.0,
            },
            personalization_level="basic",
        )

        self.profiles_cache[user_id] = profile
        self._save_profile(profile)
        logger.info(f"🆕 Neues Profil erstellt: {user_id}")

        return profile

    def _save_profile(self, profile: UserProfile):
        """Speichert ein Profil auf der Festplatte"""
        try:
            profile_path = self._get_profile_path(profile.user_id)
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"💾 Profil gespeichert: {profile.user_id}")
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern des Profils {profile.user_id}: {e}")

    def update_interaction(
        self,
        user_id: str,
        question: str,
        answer: str,
        confidence: float,
        theme: str,
        feedback: Optional[Dict[str, Any]] = None,
    ):
        """
        Aktualisiert das Nutzerprofil basierend auf einer Interaktion

        Args:
            user_id: Nutzer-ID
            question: Die gestellte Frage
            answer: Die erhaltene Antwort
            confidence: Konfidenz der Antwort
            theme: Erkanntes Thema
            feedback: Optionales Feedback des Nutzers
        """
        profile = self.get_or_create_profile(user_id)

        # Interaktion zur Historie hinzufügen
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "theme": theme,
            "feedback": feedback or {},
        }

        profile.interaction_history.append(interaction)

        # Begrenze Historie auf letzte 100 Interaktionen
        if len(profile.interaction_history) > 100:
            profile.interaction_history = profile.interaction_history[-100:]

        # Aktualisiere Interessen basierend auf Thema
        self._update_interests(profile, theme, confidence)

        # Aktualisiere Feedback-Scores
        if feedback:
            self._update_feedback_scores(profile, feedback)

        # Aktualisiere Personalisierungslevel
        self._update_personalization_level(profile)

        # Aktualisiere letzte Aktivität
        profile.last_active = datetime.now().isoformat()

        # Speichere Änderungen
        self._save_profile(profile)

        logger.debug(f"🔄 Profil aktualisiert: {user_id}")

    def _update_interests(self, profile: UserProfile, theme: str, confidence: float):
        """
        Aktualisiert die Interessen basierend auf der Interaktion

        Args:
            profile: Das zu aktualisierende Profil
            theme: Das erkannte Thema
            confidence: Konfidenz der Antwort
        """
        if theme in profile.interests:
            # Erhöhe Interesse für erfolgreiche Interaktionen
            if confidence > 0.5:
                profile.interests[theme] = min(1.0, profile.interests[theme] + 0.1)
            # Reduziere Interesse für niedrige Konfidenz
            elif confidence < 0.3:
                profile.interests[theme] = max(0.0, profile.interests[theme] - 0.05)

    def _update_feedback_scores(self, profile: UserProfile, feedback: Dict[str, Any]):
        """
        Aktualisiert die Feedback-Scores

        Args:
            profile: Das zu aktualisierende Profil
            feedback: Feedback-Daten
        """
        # Gleitender Durchschnitt für Feedback-Scores
        alpha = 0.1  # Lernrate

        for key in profile.feedback_scores:
            if key in feedback:
                current_score = profile.feedback_scores[key]
                new_score = feedback[key]
                profile.feedback_scores[key] = current_score * (1 - alpha) + new_score * alpha

    def _update_personalization_level(self, profile: UserProfile):
        """
        Aktualisiert das Personalisierungslevel basierend auf Aktivität

        Args:
            profile: Das zu aktualisierende Profil
        """
        interaction_count = len(profile.interaction_history)
        avg_feedback = sum(profile.feedback_scores.values()) / len(profile.feedback_scores)

        if interaction_count > 50 and avg_feedback > 0.7:
            profile.personalization_level = "advanced"
        elif interaction_count > 20 and avg_feedback > 0.5:
            profile.personalization_level = "intermediate"
        else:
            profile.personalization_level = "basic"

    def get_personalized_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Gibt personalisierte Empfehlungen für den Nutzer

        Args:
            user_id: Nutzer-ID

        Returns:
            Dictionary mit Empfehlungen
        """
        profile = self.get_or_create_profile(user_id)

        # Top-Interessen ermitteln
        top_interests = sorted(profile.interests.items(), key=lambda x: x[1], reverse=True)[:3]

        # Empfehlungen basierend auf Interessen und Feedback
        recommendations = {
            "suggested_topics": [interest[0] for interest in top_interests],
            "preferred_detail_level": profile.preferences.get("detail_level", "medium"),
            "response_style": profile.preferences.get("response_style", "formal"),
            "personalization_level": profile.personalization_level,
            "interaction_count": len(profile.interaction_history),
            "avg_feedback": sum(profile.feedback_scores.values()) / len(profile.feedback_scores),
        }

        return recommendations

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Gibt Einblicke in das Nutzerverhalten

        Args:
            user_id: Nutzer-ID

        Returns:
            Dictionary mit Nutzer-Einblicken
        """
        profile = self.get_or_create_profile(user_id)

        # Thema-Häufigkeit analysieren
        theme_counts = defaultdict(int)
        for interaction in profile.interaction_history:
            theme_counts[interaction.get("theme", "unbekannt")] += 1

        # Zeitliche Analyse
        recent_interactions = [
            i
            for i in profile.interaction_history
            if datetime.fromisoformat(i["timestamp"]) > datetime.now() - timedelta(days=7)
        ]

        insights = {
            "total_interactions": len(profile.interaction_history),
            "recent_interactions": len(recent_interactions),
            "favorite_themes": dict(
                sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "avg_confidence": sum(i.get("confidence", 0) for i in profile.interaction_history)
            / max(1, len(profile.interaction_history)),
            "feedback_trends": profile.feedback_scores,
            "personalization_level": profile.personalization_level,
        }

        return insights

    def export_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Exportiert ein Nutzerprofil für Analyse oder Backup

        Args:
            user_id: Nutzer-ID

        Returns:
            Profil-Daten oder None wenn nicht gefunden
        """
        try:
            profile = self.get_or_create_profile(user_id)
            return profile.to_dict()
        except Exception as e:
            logger.error(f"❌ Fehler beim Exportieren des Profils {user_id}: {e}")
            return None

    def delete_profile(self, user_id: str) -> bool:
        """
        Löscht ein Nutzerprofil

        Args:
            user_id: Nutzer-ID

        Returns:
            True wenn erfolgreich gelöscht
        """
        try:
            # Cache entfernen
            if user_id in self.profiles_cache:
                del self.profiles_cache[user_id]

            # Datei entfernen
            profile_path = self._get_profile_path(user_id)
            if os.path.exists(profile_path):
                os.remove(profile_path)

            logger.info(f"🗑️ Profil gelöscht: {user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Löschen des Profils {user_id}: {e}")
            return False


# Convenience Functions
def get_user_profile_manager(profiles_dir: str = "data/user_profiles") -> UserProfileManager:
    """
    Erstellt oder gibt den UserProfileManager zurück

    Args:
        profiles_dir: Verzeichnis für Nutzerprofile

    Returns:
        UserProfileManager-Instanz
    """
    return UserProfileManager(profiles_dir)


if __name__ == "__main__":
    # Test des UserProfileManagers
    print("🧠 Testing UserProfileManager...")

    manager = UserProfileManager()

    # Test-Profil erstellen
    profile = manager.get_or_create_profile("test_user_123")

    print(f"✅ Profil erstellt: {profile.user_id}")
    print(f"📊 Interessen: {profile.interests}")
    print(f"🎯 Personalisierungslevel: {profile.personalization_level}")

    # Test-Interaktion hinzufügen
    manager.update_interaction(
        user_id="test_user_123",
        question="Was ist die aktuelle Klimapolitik?",
        answer="Deutschland setzt sich für Klimaneutralität bis 2045 ein.",
        confidence=0.85,
        theme="klima",
        feedback={"relevance": 0.9, "helpfulness": 0.8},
    )

    # Empfehlungen abrufen
    recommendations = manager.get_personalized_recommendations("test_user_123")
    print(f"💡 Empfehlungen: {recommendations}")

    # Einblicke abrufen
    insights = manager.get_user_insights("test_user_123")
    print(f"📈 Einblicke: {insights}")

    print("✅ UserProfileManager Test abgeschlossen!")
