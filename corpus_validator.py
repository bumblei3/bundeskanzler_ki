"""
Korpus-Validator für die Bundeskanzler-KI.
Validiert den Trainingskorpus auf Qualität und Konsistenz.
"""

import logging
from typing import Dict, List, Any
from collections import Counter

logger = logging.getLogger(__name__)


class CorpusValidator:
    """Validator für Korpus-Daten."""

    def __init__(self):
        """Initialisiert den Korpus-Validator."""
        self.min_sentence_length = 10
        self.max_sentence_length = 500
        self.required_languages = ["de", "en"]

    def validate_corpus(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Validiert eine Liste von Sätzen.

        Args:
            sentences: Liste der zu validierenden Sätze

        Returns:
            Dict mit Validierungsergebnissen
        """
        results = {
            "total_sentences": len(sentences),
            "valid_sentences": 0,
            "invalid_sentences": 0,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        # Grundlegende Validierungen
        for i, sentence in enumerate(sentences):
            is_valid, errors = self._validate_sentence(sentence, i)
            if is_valid:
                results["valid_sentences"] += 1
            else:
                results["invalid_sentences"] += 1
                results["errors"].extend(errors)

        # Statistiken berechnen
        results["statistics"] = self._calculate_statistics(sentences)

        # Qualitätsbewertung
        results["quality_score"] = self._calculate_quality_score(results)

        return results

    def _validate_sentence(self, sentence: str, index: int) -> tuple[bool, List[str]]:
        """
        Validiert einen einzelnen Satz.

        Args:
            sentence: Der zu validierende Satz
            index: Index des Satzes in der Liste

        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []

        # Längenvalidierung
        if len(sentence) < self.min_sentence_length:
            errors.append(f"Satz {index}: Zu kurz ({len(sentence)} Zeichen, min {self.min_sentence_length})")

        if len(sentence) > self.max_sentence_length:
            errors.append(f"Satz {index}: Zu lang ({len(sentence)} Zeichen, max {self.max_sentence_length})")

        # Grundlegende Formatvalidierung
        if not sentence.strip():
            errors.append(f"Satz {index}: Leerer Satz")

        # Prüfung auf sinnvollen Inhalt
        if sentence.count('.') > 5:
            errors.append(f"Satz {index}: Zu viele Punkte (möglicherweise fehlerhaft)")

        # Prüfung auf wiederholte Wörter
        words = sentence.lower().split()
        word_counts = Counter(words)
        repeated_words = [word for word, count in word_counts.items() if count > 3]
        if repeated_words:
            errors.append(f"Satz {index}: Wiederholte Wörter: {repeated_words}")

        return len(errors) == 0, errors

    def _calculate_statistics(self, sentences: List[str]) -> Dict[str, Any]:
        """Berechnet Statistiken für den Korpus."""
        if not sentences:
            return {}

        lengths = [len(s) for s in sentences]
        words_per_sentence = [len(s.split()) for s in sentences]

        return {
            "avg_sentence_length": sum(lengths) / len(lengths),
            "min_sentence_length": min(lengths),
            "max_sentence_length": max(lengths),
            "avg_words_per_sentence": sum(words_per_sentence) / len(words_per_sentence),
            "total_words": sum(words_per_sentence),
            "unique_sentences": len(set(sentences)),
        }

    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Berechnet einen Qualitäts-Score basierend auf den Validierungsergebnissen."""
        if results["total_sentences"] == 0:
            return 0.0

        valid_ratio = results["valid_sentences"] / results["total_sentences"]

        # Bonus für gute Statistiken
        stats = results.get("statistics", {})
        avg_length = stats.get("avg_sentence_length", 0)

        # Ideale Satzlänge: 50-150 Zeichen
        length_score = 1.0
        if avg_length < 50:
            length_score = avg_length / 50
        elif avg_length > 150:
            length_score = max(0.5, 1.0 - (avg_length - 150) / 200)

        # Kombinierter Score
        quality_score = (valid_ratio * 0.8) + (length_score * 0.2)

        return round(quality_score, 2)


def print_validation_report(results: Dict[str, Any]) -> None:
    """
    Gibt einen formatierten Validierungsbericht aus.

    Args:
        results: Validierungsergebnisse von validate_corpus
    """
    print("\n" + "="*60)
    print("📊 KORPUS-VALIDIERUNGSBERICHT")
    print("="*60)

    print(f"📝 Gesamtanzahl Sätze: {results['total_sentences']}")
    print(f"✅ Gültige Sätze: {results['valid_sentences']}")
    print(f"❌ Ungültige Sätze: {results['invalid_sentences']}")
    print(".2f")

    # Statistiken
    if results.get("statistics"):
        stats = results["statistics"]
        print("\n📈 STATISTIKEN:")
        print(f"   Durchschnittliche Satzlänge: {stats['avg_sentence_length']:.1f} Zeichen")
        print(f"   Durchschnittliche Wörter pro Satz: {stats['avg_words_per_sentence']:.1f}")
        print(f"   Gesamtanzahl Wörter: {stats['total_words']}")
        print(f"   Einzigartige Sätze: {stats['unique_sentences']}")

    # Fehler anzeigen
    if results.get("errors"):
        print("\n❌ FEHLER:")
        for error in results["errors"][:10]:  # Nur die ersten 10 Fehler anzeigen
            print(f"   • {error}")
        if len(results["errors"]) > 10:
            print(f"   ... und {len(results['errors']) - 10} weitere Fehler")

    # Warnungen
    if results.get("warnings"):
        print("\n⚠️  WARNUNGEN:")
        for warning in results["warnings"]:
            print(f"   • {warning}")

    print("\n" + "="*60)