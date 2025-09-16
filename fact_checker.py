"""
Fact-Checker für die Bundeskanzler-KI.
Überprüft Fakten, validiert Antworten und bewertet Quellen.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class FactChecker:
    """Fact-Checker für Faktenüberprüfung und Quellenevaluation."""

    def __init__(self):
        """Initialisiert den Fact-Checker."""
        self.source_credibility_file = "source_credibility.json"
        self.source_credibility: Dict[str, float] = {}
        self.trusted_sources: List[str] = []
        self.fact_database: Dict[str, Dict[str, Any]] = {}

        self._load_source_credibility()
        self._initialize_trusted_sources()

    def _load_source_credibility(self) -> None:
        """Lädt die Quellencredibility-Datenbank."""
        if os.path.exists(self.source_credibility_file):
            try:
                with open(self.source_credibility_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.source_credibility = data.get("credibility", {})
                    self.trusted_sources = data.get("trusted_sources", [])
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Quellencredibility: {e}")
                self._initialize_default_credibility()
        else:
            self._initialize_default_credibility()

    def _initialize_default_credibility(self) -> None:
        """Initialisiert Standard-Credibility-Werte."""
        self.source_credibility = {
            "bundesregierung.de": 0.95,
            "bundestag.de": 0.95,
            "destatis.de": 0.90,
            "wikipedia.org": 0.70,
            "zeit.de": 0.85,
            "spiegel.de": 0.80,
            "faz.net": 0.80,
            "sueddeutsche.de": 0.80,
            "tagesschau.de": 0.90,
            "ard.de": 0.85,
            "zdf.de": 0.85,
        }

        self.trusted_sources = [
            "bundesregierung.de",
            "bundestag.de",
            "destatis.de",
            "tagesschau.de",
        ]

    def _initialize_trusted_sources(self) -> None:
        """Initialisiert vertrauenswürdige Quellen basierend auf Credibility."""
        if not self.trusted_sources:
            # Quellen mit Credibility > 0.85 als vertrauenswürdig einstufen
            self.trusted_sources = [
                domain for domain, credibility in self.source_credibility.items()
                if credibility > 0.85
            ]

    def check_fact(self, statement: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Überprüft eine Aussage auf Fakten.

        Args:
            statement: Die zu überprüfende Aussage
            context: Optionaler Kontext für die Überprüfung

        Returns:
            Dict mit Überprüfungsergebnissen
        """
        result = {
            "statement": statement,
            "verdict": "unknown",  # true, false, partially_true, unknown
            "confidence": 0.5,
            "explanation": "",
            "sources": [],
            "warnings": [],
        }

        # Einfache regelbasierte Überprüfung
        if self._is_obviously_false(statement):
            result["verdict"] = "false"
            result["confidence"] = 0.8
            result["explanation"] = "Aussage widerspricht bekannten Fakten"
        elif self._is_likely_true(statement):
            result["verdict"] = "true"
            result["confidence"] = 0.7
            result["explanation"] = "Aussage stimmt mit bekannten Fakten überein"
        else:
            result["verdict"] = "unknown"
            result["confidence"] = 0.3
            result["explanation"] = "Aussage konnte nicht eindeutig überprüft werden"

        # Quellen hinzufügen, falls verfügbar
        if context:
            sources = self._extract_sources_from_context(context)
            result["sources"] = sources

        return result

    def _is_obviously_false(self, statement: str) -> bool:
        """Prüft, ob eine Aussage offensichtlich falsch ist."""
        false_indicators = [
            "die erde ist eine scheibe",
            "der mond ist aus käse",
            "covid-19 ist ein hoax",
            "der holocaust hat nie stattgefunden",
        ]

        statement_lower = statement.lower()
        return any(indicator in statement_lower for indicator in false_indicators)

    def _is_likely_true(self, statement: str) -> bool:
        """Prüft, ob eine Aussage wahrscheinlich wahr ist."""
        true_indicators = [
            "berlin ist die hauptstadt deutschlands",
            "deutschland ist in der eu",
            "der bundeskanzler führt die bundesregierung",
            "deutschland hat 16 bundesländer",
        ]

        statement_lower = statement.lower()
        return any(indicator in statement_lower for indicator in true_indicators)

    def _extract_sources_from_context(self, context: str) -> List[Dict[str, Any]]:
        """Extrahiert Quellen aus dem Kontext."""
        sources = []
        # Einfache URL-Extraktion
        words = context.split()
        for word in words:
            if word.startswith("http"):
                try:
                    domain = urlparse(word).netloc
                    credibility = self.source_credibility.get(domain, 0.5)
                    sources.append({
                        "url": word,
                        "domain": domain,
                        "credibility": credibility,
                        "trusted": domain in self.trusted_sources,
                    })
                except:
                    continue
        return sources

    def validate_response(self, response: str, user_query: str) -> Dict[str, Any]:
        """
        Validiert eine KI-Antwort auf Konsistenz und Fakten.

        Args:
            response: Die zu validierende Antwort
            user_query: Die ursprüngliche Benutzeranfrage

        Returns:
            Dict mit Validierungsergebnissen
        """
        validation = {
            "is_valid": True,
            "score": 0.8,
            "issues": [],
            "suggestions": [],
        }

        # Längenvalidierung
        if len(response) < 10:
            validation["issues"].append("Antwort ist zu kurz")
            validation["score"] -= 0.2

        if len(response) > 2000:
            validation["issues"].append("Antwort ist zu lang")
            validation["score"] -= 0.1

        # Relevanz zur Anfrage prüfen
        if not self._is_relevant_to_query(response, user_query):
            validation["issues"].append("Antwort ist nicht relevant zur Anfrage")
            validation["score"] -= 0.3

        # Faktenüberprüfung
        fact_check = self.check_fact(response)
        if fact_check["verdict"] == "false":
            validation["issues"].append("Antwort enthält falsche Informationen")
            validation["score"] -= 0.4

        # Politische Neutralität prüfen
        if self._has_political_bias(response):
            validation["issues"].append("Antwort zeigt politische Voreingenommenheit")
            validation["score"] -= 0.2

        validation["is_valid"] = validation["score"] >= 0.6

        return validation

    def _is_relevant_to_query(self, response: str, query: str) -> bool:
        """Prüft, ob die Antwort relevant zur Anfrage ist."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Mindestens 20% Überlappung der Wörter
        overlap = len(query_words.intersection(response_words))
        return overlap / len(query_words) >= 0.2 if query_words else True

    def _has_political_bias(self, response: str) -> bool:
        """Prüft auf politische Voreingenommenheit."""
        bias_indicators = [
            "extrem links", "extrem rechts", "verschwörung",
            "die medien lügen", "fake news", "deep state",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in bias_indicators)

    def update_source_credibility(self, domain: str, credibility: float) -> None:
        """
        Aktualisiert die Credibility einer Quelle.

        Args:
            domain: Die Domain der Quelle
            credibility: Neue Credibility (0.0-1.0)
        """
        self.source_credibility[domain] = max(0.0, min(1.0, credibility))

        # Trusted sources aktualisieren
        if credibility > 0.85 and domain not in self.trusted_sources:
            self.trusted_sources.append(domain)
        elif credibility <= 0.85 and domain in self.trusted_sources:
            self.trusted_sources.remove(domain)

        # Speichern
        self._save_source_credibility()

    def _save_source_credibility(self) -> None:
        """Speichert die Quellencredibility-Datenbank."""
        data = {
            "credibility": self.source_credibility,
            "trusted_sources": self.trusted_sources,
        }

        try:
            with open(self.source_credibility_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Quellencredibility: {e}")

    def get_source_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über Quellen zurück."""
        return {
            "total_sources": len(self.source_credibility),
            "trusted_sources": len(self.trusted_sources),
            "avg_credibility": sum(self.source_credibility.values()) / len(self.source_credibility) if self.source_credibility else 0,
            "high_credibility_sources": len([c for c in self.source_credibility.values() if c > 0.85]),
        }