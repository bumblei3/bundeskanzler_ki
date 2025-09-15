from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import logging

SUPPORTED_LANGUAGES = {
    'de': 'Deutsch',
    'en': 'English',
    'fr': 'Français',
    'es': 'Español',
    'it': 'Italiano'
}

def detect_language(text: str) -> str:
    """
    Erkennt die Sprache eines Textes und gibt den Sprachcode zurück.
    
    Args:
        text: Der zu analysierende Text
        
    Returns:
        str: Sprachcode (z.B. 'de', 'en') oder 'unknown'
        
    Raises:
        ValueError: Wenn die Sprache nicht erkannt werden konnte
    """
    try:
        # Hole alle möglichen Sprachen mit Wahrscheinlichkeiten
        langs = detect_langs(text)
        
        # Finde die wahrscheinlichste Sprache
        best_match = max(langs, key=lambda x: x.prob)
        
        # Prüfe ob die Sprache unterstützt wird und die Konfidenz hoch genug ist
        if best_match.lang in SUPPORTED_LANGUAGES and best_match.prob > 0.8:
            return best_match.lang
            
        return 'unknown'
        
    except LangDetectException as e:
        raise ValueError(f"Konnte Sprache nicht erkennen: {str(e)}")

def get_supported_languages() -> dict[str, str]:
    """Gibt ein Dictionary mit unterstützten Sprachen zurück."""
    return SUPPORTED_LANGUAGES.copy()

def is_language_supported(lang_code: str) -> bool:
    """Prüft ob eine Sprache unterstützt wird."""
    return lang_code in SUPPORTED_LANGUAGES