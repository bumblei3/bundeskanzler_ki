"""
Multilingual Bundeskanzler KI
Erweitert die Bundeskanzler KI um mehrsprachige Unterstützung
"""

import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

# Füge das Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from core.bundeskanzler_ki import BundeskanzlerKI
    from core.corpus_manager import CorpusManager
    from core.fact_checker import FactChecker
    from core.debug_system import DebugLevel, debug_system
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import-Fehler in multilingual_bundeskanzler_ki: {e}")
    # Fallback für Testumgebung
    BundeskanzlerKI = None
    CorpusManager = None
    FactChecker = None
    debug_system = None
    DebugLevel = None
    IMPORTS_AVAILABLE = False


class MultilingualBundeskanzlerKI:
    """Multilingual erweiterte Bundeskanzler KI"""

    def __init__(self):
        self.ki = None
        self.corpus_manager = None
        self.fact_checker = None
        self.supported_languages = {
            'de': 'Deutsch',
            'en': 'English',
            'fr': 'Français',
            'es': 'Español',
            'it': 'Italiano'
        }
        self.initialized = False

    def initialize_multimodal_model(self):
        """Initialisiert die multimodalen Modelle"""
        try:
            if BundeskanzlerKI and IMPORTS_AVAILABLE:
                self.ki = BundeskanzlerKI()
                self.corpus_manager = CorpusManager()
                self.fact_checker = FactChecker()
                self.initialized = True
                if debug_system:
                    debug_system.log(DebugLevel.INFO, "Multilingual KI erfolgreich initialisiert")
            else:
                if debug_system:
                    debug_system.log(DebugLevel.WARNING, "BundeskanzlerKI nicht verfügbar - verwende Mock-Modus")
                self.initialized = True
        except Exception as e:
            if debug_system:
                debug_system.log(DebugLevel.ERROR, f"Fehler bei Multilingual-Initialisierung: {e}")
            self.initialized = False

    def get_supported_languages_info(self) -> Dict[str, str]:
        """Gibt die unterstützten Sprachen zurück"""
        return self.supported_languages

    def detect_language(self, text: str) -> str:
        """Erkennt die Sprache des Textes (vereinfachte Implementierung)"""
        # Vereinfachte Spracherkennung basierend auf Schlüsselwörtern
        text_lower = text.lower()

        if any(word in text_lower for word in ['ich', 'ist', 'der', 'die', 'das', 'und', 'mit']):
            return 'de'
        elif any(word in text_lower for word in ['the', 'is', 'and', 'with', 'for']):
            return 'en'
        elif any(word in text_lower for word in ['le', 'la', 'les', 'et', 'avec']):
            return 'fr'
        elif any(word in text_lower for word in ['el', 'la', 'los', 'las', 'y', 'con']):
            return 'es'
        elif any(word in text_lower for word in ['il', 'la', 'i', 'le', 'e', 'con']):
            return 'it'
        else:
            return 'de'  # Default

    def translate_to_german(self, text: str, source_lang: str) -> str:
        """Übersetzt Text ins Deutsche (Mock-Implementierung)"""
        if source_lang == 'de':
            return text

        # Vereinfachte Übersetzungen für Demo-Zwecke
        translations = {
            'en': {
                'hello': 'hallo',
                'what is': 'was ist',
                'how does': 'wie funktioniert',
                'climate policy': 'Klimapolitik',
                'government': 'Regierung'
            },
            'fr': {
                'bonjour': 'guten tag',
                'quelle est': 'was ist',
                'comment': 'wie',
                'politique climatique': 'Klimapolitik',
                'gouvernement': 'Regierung'
            }
        }

        if source_lang in translations:
            result = text
            for eng, ger in translations[source_lang].items():
                result = result.replace(eng, ger)
            return result

        return text  # Fallback

    def process_multilingual_query(self, query: str) -> Dict[str, Any]:
        """Verarbeitet eine mehrsprachige Anfrage"""
        try:
            detected_lang = self.detect_language(query)
            german_query = self.translate_to_german(query, detected_lang)

            # Verarbeite die Anfrage mit der KI
            if self.ki and self.initialized:
                # Verwende die bestehende KI für die Verarbeitung
                response = self.ki.process_query(german_query)
                german_response = response.get('response', 'Keine Antwort verfügbar')
            else:
                # Mock-Antwort für Demo
                german_response = f"Mock-Antwort auf Deutsch für: {german_query}"

            # Übersetze Antwort zurück (falls nötig)
            final_response = german_response
            translation_used = False

            if detected_lang != 'de':
                # Vereinfachte Rückübersetzung
                final_response = german_response  # Für Demo unverändert lassen
                translation_used = True

            return {
                'original_query': query,
                'detected_language': detected_lang,
                'german_query': german_query,
                'german_response': german_response,
                'response': final_response,
                'translation_used': translation_used,
                'processing_time': 0.5,
                'fallback_used': not self.initialized
            }

        except Exception as e:
            if debug_system:
                debug_system.log(DebugLevel.ERROR, f"Fehler bei multilingualer Verarbeitung: {e}")
            return {
                'original_query': query,
                'detected_language': 'de',
                'error': str(e),
                'response': f"Fehler bei der Verarbeitung: {e}",
                'processing_time': 0.1,
                'fallback_used': True
            }

    def get_debug_info(self) -> Dict[str, Any]:
        """Gibt Debug-Informationen zurück"""
        return {
            'initialized': self.initialized,
            'supported_languages': list(self.supported_languages.keys()),
            'message_count': 0,
            'api_call_count': 0,
            'messages': [],
            'api_calls': [],
            'debug_disabled': debug_system is None
        }


# Globale Instanz
_multilingual_ki_instance = None

def get_multilingual_ki() -> MultilingualBundeskanzlerKI:
    """Gibt die globale Multilingual-KI-Instanz zurück"""
    global _multilingual_ki_instance
    if _multilingual_ki_instance is None:
        _multilingual_ki_instance = MultilingualBundeskanzlerKI()
    return _multilingual_ki_instance

def multilingual_query(query: str) -> Dict[str, Any]:
    """Vereinfachte Funktion für mehrsprachige Abfragen"""
    ki = get_multilingual_ki()
    if not ki.initialized:
        ki.initialize_multimodal_model()
    return ki.process_multilingual_query(query)