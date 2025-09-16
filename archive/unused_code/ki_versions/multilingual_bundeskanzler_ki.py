"""
Multilingual Bundeskanzler KI - Mehrsprachige Unterstützung
Erweitert die Bundeskanzler KI um Unterstützung für Deutsch, Englisch und Französisch.
"""

import logging
import os

# Bestehende KI importieren
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.append("/home/tobber/bkki_venv/core")
from bundeskanzler_ki import detect_language as legacy_detect_language
from bundeskanzler_ki import (
    generate_response,
    preprocess,
    setup_model,
)

# Mehrsprachige Services importieren
from language_detection import (
    detect_language,
    get_supported_languages,
    is_language_supported,
)

# Multimodale KI für erweiterte Features
from multimodal_ki import MultimodalTransformerModel
from translation_service import TranslationService

# Debug-System
from webgui_ki import DebugLevel, DebugSystem


class MultilingualBundeskanzlerKI:
    """
    Mehrsprachige Version der Bundeskanzler KI.
    Unterstützt Deutsch, Englisch und Französisch mit automatischer Übersetzung.
    """

    def __init__(self, model_tier: str = "rtx2070", debug: bool = True):
        """
        Initialisiert die mehrsprachige KI.

        Args:
            model_tier: Modell-Tier ('rtx2070', 'advanced', 'basic', 'premium')
            debug: Debug-System aktivieren
        """
        self.model_tier = model_tier
        self.debug = debug

        # Debug-System initialisieren
        self.debug_system = DebugSystem() if debug else None

        # Translation Service initialisieren
        self.translation_service = TranslationService()

        # Unterstützte Sprachen
        self.supported_languages = get_supported_languages()

        # Multimodales Modell für erweiterte Features
        self.multimodal_model = None

        # Logging
        self.logger = logging.getLogger(__name__)

        if self.debug_system:
            self.debug_system.log(DebugLevel.INFO, "Multilingual Bundeskanzler KI initialisiert")
            self.debug_system.log(
                DebugLevel.INFO,
                f"Unterstützte Sprachen: {list(self.supported_languages.keys())}",
            )

    def initialize_multimodal_model(self):
        """Initialisiert das multimodale Modell für erweiterte Features."""
        try:
            if self.debug_system:
                self.debug_system.log(
                    DebugLevel.INFO,
                    f"Initialisiere multimodales Modell (Tier: {self.model_tier})",
                )

            self.multimodal_model = MultimodalTransformerModel(model_tier=self.model_tier)

            if self.debug_system:
                self.debug_system.log(
                    DebugLevel.SUCCESS, "Multimodales Modell erfolgreich initialisiert"
                )

        except Exception as e:
            error_msg = f"Fehler beim Initialisieren des multimodalen Modells: {e}"
            self.logger.error(error_msg)
            if self.debug_system:
                self.debug_system.log(DebugLevel.ERROR, error_msg)

    def detect_language(self, text: str) -> str:
        """
        Erkennt die Sprache des Eingabetextes.

        Args:
            text: Eingabetext

        Returns:
            Sprachcode ('de', 'en', 'fr', etc.) oder 'unknown'
        """
        try:
            detected_lang = detect_language(text)

            if self.debug_system:
                self.debug_system.log(DebugLevel.INFO, f"Sprache erkannt: {detected_lang}")

            return detected_lang

        except Exception as e:
            error_msg = f"Fehler bei der Spracherkennung: {e}"
            self.logger.error(error_msg)
            if self.debug_system:
                self.debug_system.log(DebugLevel.ERROR, error_msg)
            return "unknown"

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Übersetzt Text zwischen Sprachen.

        Args:
            text: Zu übersetzender Text
            source_lang: Quellsprache
            target_lang: Zielsprache

        Returns:
            Übersetzter Text
        """
        try:
            if source_lang == target_lang:
                return text

            translated = self.translation_service.translate(text, source_lang, target_lang)

            if self.debug_system:
                self.debug_system.log(
                    DebugLevel.INFO,
                    f"Übersetzung: {source_lang} -> {target_lang} ({len(text)} -> {len(translated)} Zeichen)",
                )

            return translated

        except Exception as e:
            error_msg = f"Fehler bei der Übersetzung {source_lang}->{target_lang}: {e}"
            self.logger.error(error_msg)
            if self.debug_system:
                self.debug_system.log(DebugLevel.ERROR, error_msg)
            return text  # Fallback: Originaltext zurückgeben

    def process_multilingual_query(
        self, query: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verarbeitet eine mehrsprachige Anfrage.

        Strategie:
        1. Sprache erkennen
        2. Falls nicht Deutsch: nach Deutsch übersetzen
        3. Antwort in Deutsch generieren
        4. Falls nötig: Antwort in Originalsprache zurückübersetzen

        Args:
            query: Die Anfrage des Benutzers
            user_id: Optionale Benutzer-ID für personalisierte Antworten

        Returns:
            Dictionary mit Antwort, Sprache und Metadaten
        """
        start_time = time.time()

        try:
            # 1. Sprache erkennen
            detected_lang = self.detect_language(query)

            if self.debug_system:
                self.debug_system.log(
                    DebugLevel.INFO, f"Verarbeite Anfrage in Sprache: {detected_lang}"
                )

            # 2. Übersetze nach Deutsch falls nötig
            if detected_lang != "de" and detected_lang != "unknown":
                query_de = self.translate_text(query, detected_lang, "de")
                if self.debug_system:
                    self.debug_system.log(
                        DebugLevel.INFO,
                        f"Anfrage übersetzt nach Deutsch: {len(query_de)} Zeichen",
                    )
            else:
                query_de = query

            # 3. Antwort in Deutsch generieren
            if self.multimodal_model:
                # Verwende multimodales Modell falls verfügbar
                response_de = self.multimodal_model.process_text(query_de)
            else:
                # Fallback auf Legacy-KI
                response_de = generate_response(query_de, user_id=user_id)

            # 4. Übersetze Antwort zurück in Originalsprache falls nötig
            if detected_lang != "de" and detected_lang != "unknown":
                response_original = self.translate_text(response_de, "de", detected_lang)
                if self.debug_system:
                    self.debug_system.log(
                        DebugLevel.INFO,
                        f"Antwort übersetzt nach {detected_lang}: {len(response_original)} Zeichen",
                    )
            else:
                response_original = response_de

            # 5. Metadaten sammeln
            processing_time = time.time() - start_time
            result = {
                "response": response_original,
                "detected_language": detected_lang,
                "original_query": query,
                "german_query": query_de,
                "german_response": response_de,
                "processing_time": processing_time,
                "supported_languages": list(self.supported_languages.keys()),
                "translation_used": detected_lang != "de",
            }

            if self.debug_system:
                self.debug_system.log(
                    DebugLevel.SUCCESS,
                    f"Anfrage erfolgreich verarbeitet in {processing_time:.2f}s",
                )

            return result

        except Exception as e:
            error_msg = f"Fehler bei der mehrsprachigen Verarbeitung: {e}"
            self.logger.error(error_msg)

            if self.debug_system:
                self.debug_system.log(DebugLevel.ERROR, error_msg)

            # Fallback: Versuche Legacy-Verarbeitung
            try:
                fallback_response = generate_response(query, user_id=user_id)
                return {
                    "response": fallback_response,
                    "detected_language": "unknown",
                    "error": str(e),
                    "fallback_used": True,
                }
            except Exception as fallback_error:
                return {
                    "response": "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.",
                    "detected_language": "unknown",
                    "error": str(e),
                    "fallback_error": str(fallback_error),
                }

    def get_supported_languages_info(self) -> Dict[str, str]:
        """
        Gibt Informationen über unterstützte Sprachen zurück.

        Returns:
            Dictionary mit Sprachcodes und Namen
        """
        return self.supported_languages.copy()

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Gibt Debug-Informationen zurück.

        Returns:
            Debug-Informationen als Dictionary
        """
        if not self.debug_system:
            return {"debug_disabled": True}

        return {
            "messages": self.debug_system.messages,
            "api_calls": self.debug_system.api_calls,
            "message_count": len(self.debug_system.messages),
            "api_call_count": len(self.debug_system.api_calls),
        }


# Globale Instanz für einfachen Zugriff
_multilingual_ki_instance = None


def get_multilingual_ki(
    model_tier: str = "rtx2070", debug: bool = True
) -> MultilingualBundeskanzlerKI:
    """
    Gibt eine globale Instanz der mehrsprachigen KI zurück.

    Args:
        model_tier: Modell-Tier
        debug: Debug-System aktivieren

    Returns:
        MultilingualBundeskanzlerKI Instanz
    """
    global _multilingual_ki_instance

    if _multilingual_ki_instance is None:
        _multilingual_ki_instance = MultilingualBundeskanzlerKI(model_tier=model_tier, debug=debug)

    return _multilingual_ki_instance


def multilingual_query(
    query: str, user_id: Optional[str] = None, model_tier: str = "rtx2070"
) -> Dict[str, Any]:
    """
    Convenience-Funktion für mehrsprachige Anfragen.

    Args:
        query: Die Anfrage des Benutzers
        user_id: Optionale Benutzer-ID
        model_tier: Modell-Tier

    Returns:
        Antwort-Dictionary
    """
    ki = get_multilingual_ki(model_tier=model_tier)
    return ki.process_multilingual_query(query, user_id=user_id)


if __name__ == "__main__":
    # Test der mehrsprachigen KI
    print("🚀 Teste Multilingual Bundeskanzler KI...")

    ki = get_multilingual_ki()

    # Test-Anfragen in verschiedenen Sprachen
    test_queries = [
        "Was ist die Klimapolitik der Bundesregierung?",
        "What is the climate policy of the German government?",
        "Quelle est la politique climatique du gouvernement fédéral allemand?",
    ]

    for query in test_queries:
        print(f"\n📝 Anfrage: {query}")
        result = ki.process_multilingual_query(query)
        print(f"🌐 Erkannte Sprache: {result['detected_language']}")
        print(f"✅ Antwort: {result['response'][:100]}...")
        print(f"⏱️ Verarbeitungszeit: {result['processing_time']:.2f}s")

    print("\n🎉 Multilingual KI Test abgeschlossen!")
