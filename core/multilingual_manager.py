#!/usr/bin/env python3
"""
Erweiterter Multilingual Manager
===============================
Unterstützt mehr Sprachen mit verbesserter Erkennung
"""

import logging
import re
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class Language(Enum):
    """Unterstützte Sprachen"""
    DE = "de"
    EN = "en"
    FR = "fr"
    ES = "es"
    IT = "it"

class ExtendedMultilingualManager:
    """Erweiterter Multilingual Manager mit mehr Sprachen"""

    def __init__(self):
        self.supported_languages = [Language.DE, Language.EN, Language.FR, Language.ES, Language.IT]

        # Sprachspezifische Keywords für bessere Erkennung
        self.language_keywords = {
            Language.DE: [
                'der', 'die', 'das', 'ist', 'und', 'mit', 'für', 'auf', 'von', 'zu',
                'deutschland', 'bundeskanzler', 'regierung', 'politik', 'europa',
                'energiewende', 'klimaschutz', 'nachhaltigkeit', 'digitalisierung'
            ],
            Language.EN: [
                'the', 'is', 'and', 'with', 'for', 'on', 'from', 'to',
                'germany', 'chancellor', 'government', 'policy', 'europe',
                'energy', 'climate', 'sustainability', 'digitalization'
            ],
            Language.FR: [
                'le', 'la', 'les', 'est', 'et', 'avec', 'pour', 'sur', 'de', 'à',
                'allemagne', 'chancelier', 'gouvernement', 'politique', 'europe',
                'énergie', 'climat', 'durabilité', 'numérisation'
            ],
            Language.ES: [
                'el', 'la', 'los', 'las', 'es', 'y', 'con', 'para', 'en', 'de',
                'alemania', 'canciller', 'gobierno', 'política', 'europa',
                'energía', 'clima', 'sostenibilidad', 'digitalización'
            ],
            Language.IT: [
                'il', 'la', 'i', 'le', 'è', 'e', 'con', 'per', 'in', 'di',
                'germania', 'cancelliere', 'governo', 'politica', 'europa',
                'energia', 'clima', 'sostenibilità', 'digitalizzazione'
            ]
        }

        logger.info(f"🌍 Extended MultilingualManager initialisiert - {len(self.supported_languages)} Sprachen unterstützt")

    def detect_language(self, text: str) -> Tuple[Language, float]:
        """Verbesserte Spracherkennung mit mehr Sprachen"""
        text_lower = text.lower()
        scores = {}

        # Berechne Score für jede Sprache
        for lang in self.supported_languages:
            score = 0
            keywords = self.language_keywords[lang]

            # Zähle Keyword-Matches
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1

            # Zusätzliche Heuristiken
            if lang == Language.DE:
                # Deutsche Umlaute und ß
                if any(char in text_lower for char in ['ä', 'ö', 'ü', 'ß']):
                    score += 2
                # Typische deutsche Wortendungen
                if re.search(r'\b\w+(ung|keit|tion|ment|ismus)\b', text_lower):
                    score += 1

            elif lang == Language.FR:
                # Französische Akzente
                if any(char in text_lower for char in ['é', 'è', 'ê', 'à', 'â', 'ô', 'û', 'ç']):
                    score += 2

            elif lang == Language.ES:
                # Spanische Sonderzeichen
                if any(char in text_lower for char in ['ñ', 'á', 'é', 'í', 'ó', 'ú']):
                    score += 2

            elif lang == Language.IT:
                # Italienische Endungen
                if re.search(r'\b\w+(zione|mento|ismo|ità)\b', text_lower):
                    score += 1

            scores[lang] = score

        # Finde Sprache mit höchstem Score
        best_lang = max(scores, key=scores.get)
        confidence = min(scores[best_lang] / 5.0, 1.0)  # Normalisiere auf 0-1

        # Mindestvertrauen für unbekannte Sprachen
        if confidence < 0.3:
            best_lang = Language.EN
            confidence = 0.5

        logger.info(f"🌍 Sprache erkannt: {best_lang.value} (Vertrauen: {confidence:.2f})")
        return best_lang, confidence

    def translate_text(self, text: str, target_lang: Language) -> Optional[str]:
        """Einfache Übersetzung (Platzhalter ohne externe APIs)"""
        logger.warning(f"🌍 Übersetzung nach {target_lang.value} nicht verfügbar - DeepL wurde entfernt")
        return None

    def get_supported_languages(self) -> list:
        """Gibt Liste der unterstützten Sprachen zurück"""
        return [lang.value for lang in self.supported_languages]

def get_multilingual_manager():
    """Factory-Funktion für erweiterten Manager"""
    return ExtendedMultilingualManager()
