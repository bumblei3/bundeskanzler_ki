#!/usr/bin/env python3
"""
Text-Verbesserungs-Plugin für Bundeskanzler KI
Verbessert die Qualität und Korrektheit von Textausgaben
"""

import re
from typing import Dict, Any
from core.plugin_system import TextProcessingPlugin, PluginMetadata

class TextImprovementPlugin(TextProcessingPlugin):
    """
    Plugin zur Verbesserung von Textausgaben

    Dieses Plugin verbessert Grammatik, Rechtschreibung und Stil von KI-Textausgaben.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="text_improvement",
            version="1.0.0",
            description="Verbessert Grammatik, Rechtschreibung und Stil von Textausgaben",
            author="Bundeskanzler KI Team",
            license="MIT",
            tags=["text", "improvement", "grammar", "style"],
            dependencies=[]
        )

    def initialize(self) -> None:
        """Initialisiert das Plugin"""
        self.logger.info("Text-Verbesserungs-Plugin initialisiert")
        self._common_mistakes = {
            r'\b(ist|war|wird)\s+sehr\s+(gut|schlecht|wichtig)\b': r'\1 \2',
            r'\b(und|oder|aber)\s+,\s+': r'\1 ',
            r'\s+([.,!?])': r'\1',
            r'([.,!?])\s*([.,!?])': r'\1',
        }

    def shutdown(self) -> None:
        """Beendet das Plugin"""
        self.logger.info("Text-Verbesserungs-Plugin beendet")

    def process_text(self, text: str, **kwargs) -> str:
        """
        Verbessert den gegebenen Text

        Args:
            text: Der zu verbessernde Text
            **kwargs: Zusätzliche Parameter

        Returns:
            Der verbesserte Text
        """
        if not text:
            return text

        improved_text = text

        # Grundlegende Textverbesserungen
        improved_text = self._fix_common_mistakes(improved_text)
        improved_text = self._improve_punctuation(improved_text)
        improved_text = self._capitalize_sentences(improved_text)
        improved_text = self._fix_spacing(improved_text)

        # Erweiterte Verbesserungen basierend auf Konfiguration
        if self._config.settings.get('advanced_improvements', True):
            improved_text = self._advanced_improvements(improved_text)

        self.logger.debug(f"Text verbessert: {len(text)} -> {len(improved_text)} Zeichen")
        return improved_text

    def _fix_common_mistakes(self, text: str) -> str:
        """Behebt häufige Rechtschreib- und Grammatikfehler"""
        for pattern, replacement in self._common_mistakes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _improve_punctuation(self, text: str) -> str:
        """Verbessert die Interpunktion"""
        # Fügt fehlende Punkte am Ende von Sätzen hinzu
        text = re.sub(r'([a-z])\s*$', r'\1.', text)
        # Entfernt überflüssige Leerzeichen vor Satzzeichen
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text

    def _capitalize_sentences(self, text: str) -> str:
        """Stellt sicher, dass Sätze mit Großbuchstaben beginnen"""
        sentences = re.split(r'([.!?]\s*)', text)
        capitalized_sentences = []

        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Eigentlicher Satz-Text
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
            capitalized_sentences.append(sentence)

        return ''.join(capitalized_sentences)

    def _fix_spacing(self, text: str) -> str:
        """Korrigiert Leerzeichen und Abstände"""
        # Entfernt mehrfache Leerzeichen
        text = re.sub(r'\s+', ' ', text)
        # Entfernt Leerzeichen am Anfang und Ende
        text = text.strip()
        return text

    def _advanced_improvements(self, text: str) -> str:
        """Führt erweiterte Textverbesserungen durch"""
        # Verbessert Wortwahl für formelle Kommunikation
        improvements = {
            r'\b(hi|hey|halo)\b': 'Guten Tag',
            r'\b(thx|thanks|danke)\b': 'Vielen Dank',
            r'\b(pls|please|bitte)\b': 'Bitte',
            r'\b(ok|okay)\b': 'Einverstanden',
        }

        for pattern, replacement in improvements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text