#!/usr/bin/env python3
"""
Einfacher Multilingual Manager ohne DeepL
========================================
"""

import logging
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)

class Language(Enum):
    """Unterstützte Sprachen"""
    DE = "de"
    EN = "en"

class SimpleMultilingualManager:
    """Einfacher Multilingual Manager"""

    def __init__(self):
        self.supported_languages = [Language.DE, Language.EN]
        logger.info("🌍 Simple MultilingualManager initialisiert")

    def detect_language(self, text: str) -> tuple:
        """Einfache Spracherkennung"""
        # Sehr einfache Erkennung - kann verbessert werden
        if any(word in text.lower() for word in ['der', 'die', 'das', 'ist', 'und']):
            return Language.DE, 0.8
        else:
            return Language.EN, 0.6

    def translate_text(self, text: str, target_lang):
        """Einfache Übersetzung (nur Platzhalter)"""
        logger.warning("Übersetzung nicht verfügbar - DeepL wurde entfernt")
        return None

def get_multilingual_manager():
    """Factory-Funktion"""
    return SimpleMultilingualManager()
