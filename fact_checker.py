"""
Faktenprüfung und Quellenvalidierung für die Bundeskanzler KI
Implementiert Faktenvalidierung, Bias-Erkennung und Quellenbewertung
"""

import re
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class FactCheckResult:
    """Ergebnis einer Faktenprüfung"""
    statement: str
    confidence_score: float  # 0.0 - 1.0
    sources: List[Dict[str, Any]]
    bias_score: float  # -1.0 (links) bis +1.0 (rechts)
    verification_status: str  # "verified", "partially_verified", "unverified", "contradicted"
    explanation: str
    timestamp: datetime

@dataclass
class SourceCredibility:
    """Bewertung der Glaubwürdigkeit einer Quelle"""
    domain: str
    credibility_score: float  # 0.0 - 1.0
    political_bias: float  # -1.0 bis +1.0
    fact_checking_rating: float  # 0.0 - 1.0
    last_updated: datetime
    category: str  # "government", "media", "academic", "ngo", etc.

class FactChecker:
    """
    Umfassende Faktenprüfung für politische Aussagen
    """

    def __init__(self, credibility_db_path: str = "source_credibility.json"):
        self.credibility_db_path = credibility_db_path
        self.source_credibility = self._load_source_credibility()

        # Politische Bias-Indikatoren (vereinfacht)
        self.bias_indicators = {
            'left_bias': [
                'sozial', 'gerechtigkeit', 'umwelt', 'klima', 'gleichheit',
                'migration', 'asyl', 'antirassismus', 'inklusion', 'diversität'
            ],
            'right_bias': [
                'sicherheit', 'ordnung', 'tradition', 'nation', 'grenze',
                'wirtschaftsfreiheit', 'eigenverantwortung', 'leistung', 'disziplin'
            ]
        }

        # Vertrauenswürdige Quellen für Cross-Referencing
        self.trusted_sources = [
            "bundesregierung.de",
            "bundestag.de",
            "destatis.de",
            "bmi.bund.de",
            "bmwi.de",
            "bmz.de",
            "auswaertiges-amt.de",
            "zeit.de",
            "faz.net",
            "spiegel.de",
            "welt.de",
            "tagesschau.de",
            "dw.com"
        ]

    def _load_source_credibility(self) -> Dict[str, SourceCredibility]:
        """Lädt die Quellen-Glaubwürdigkeitsdatenbank"""
        try:
            with open(self.credibility_db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    domain: SourceCredibility(**info)
                    for domain, info in data.items()
                }
        except FileNotFoundError:
            logger.warning(f"Credibility database not found: {self.credibility_db_path}")
            return self._create_default_credibility_db()

    def _create_default_credibility_db(self) -> Dict[str, SourceCredibility]:
        """Erstellt eine Standard-Quellen-Glaubwürdigkeitsdatenbank"""
        default_sources = {
            "bundesregierung.de": {
                "domain": "bundesregierung.de",
                "credibility_score": 0.95,
                "political_bias": 0.0,
                "fact_checking_rating": 0.9,
                "last_updated": datetime.now().isoformat(),
                "category": "government"
            },
            "bundestag.de": {
                "domain": "bundestag.de",
                "credibility_score": 0.95,
                "political_bias": 0.0,
                "fact_checking_rating": 0.9,
                "last_updated": datetime.now().isoformat(),
                "category": "government"
            },
            "destatis.de": {
                "domain": "destatis.de",
                "credibility_score": 0.98,
                "political_bias": 0.0,
                "fact_checking_rating": 0.95,
                "last_updated": datetime.now().isoformat(),
                "category": "government"
            },
            "zeit.de": {
                "domain": "zeit.de",
                "credibility_score": 0.85,
                "political_bias": -0.2,
                "fact_checking_rating": 0.8,
                "last_updated": datetime.now().isoformat(),
                "category": "media"
            },
            "faz.net": {
                "domain": "faz.net",
                "credibility_score": 0.8,
                "political_bias": 0.1,
                "fact_checking_rating": 0.75,
                "last_updated": datetime.now().isoformat(),
                "category": "media"
            },
            "spiegel.de": {
                "domain": "spiegel.de",
                "credibility_score": 0.75,
                "political_bias": -0.3,
                "fact_checking_rating": 0.7,
                "last_updated": datetime.now().isoformat(),
                "category": "media"
            },
            "welt.de": {
                "domain": "welt.de",
                "credibility_score": 0.7,
                "political_bias": 0.4,
                "fact_checking_rating": 0.65,
                "last_updated": datetime.now().isoformat(),
                "category": "media"
            }
        }

        # Speichere die Standarddatenbank
        with open(self.credibility_db_path, 'w', encoding='utf-8') as f:
            json.dump(default_sources, f, indent=2, ensure_ascii=False)

        return {
            domain: SourceCredibility(**info)
            for domain, info in default_sources.items()
        }

    def check_fact(self, statement: str, context: Optional[Dict[str, Any]] = None) -> FactCheckResult:
        """
        Führt eine umfassende Faktenprüfung durch
        """
        statement = statement.strip()

        # 1. Bias-Analyse
        bias_score = self._analyze_bias(statement)

        # 2. Quellen-Suche und -Validierung
        sources = self._find_sources(statement)

        # 3. Confidence-Score berechnen
        confidence_score = self._calculate_confidence(statement, sources, bias_score)

        # 4. Verifikationsstatus bestimmen
        verification_status = self._determine_verification_status(confidence_score, sources)

        # 5. Erklärung generieren
        explanation = self._generate_explanation(verification_status, confidence_score, bias_score, sources)

        return FactCheckResult(
            statement=statement,
            confidence_score=confidence_score,
            sources=sources,
            bias_score=bias_score,
            verification_status=verification_status,
            explanation=explanation,
            timestamp=datetime.now()
        )

    def _analyze_bias(self, statement: str) -> float:
        """
        Analysiert politische Bias in der Aussage
        Returns: -1.0 (stark links) bis +1.0 (stark rechts)
        """
        statement_lower = statement.lower()

        left_score = 0
        right_score = 0

        # Zähle Bias-Indikatoren
        for word in self.bias_indicators['left_bias']:
            left_score += statement_lower.count(word)

        for word in self.bias_indicators['right_bias']:
            right_score += statement_lower.count(word)

        total_indicators = left_score + right_score

        if total_indicators == 0:
            return 0.0  # Neutral

        # Normalisiere auf -1.0 bis +1.0
        bias_ratio = (right_score - left_score) / total_indicators
        return max(-1.0, min(1.0, bias_ratio))

    def _find_sources(self, statement: str) -> List[Dict[str, Any]]:
        """
        Sucht relevante Quellen für die Faktenprüfung
        """
        sources = []

        # Extrahiere Schlüsselwörter für die Suche
        keywords = self._extract_keywords(statement)

        # Simuliere Quellen-Suche (in Produktion: echte API-Aufrufe)
        for keyword in keywords[:3]:  # Begrenze auf 3 Schlüsselwörter
            mock_sources = self._get_mock_sources(keyword)
            sources.extend(mock_sources)

        # Filtere und sortiere Quellen nach Relevanz
        sources = sorted(sources, key=lambda x: x.get('relevance_score', 0), reverse=True)
        return sources[:5]  # Maximal 5 Quellen

    def _extract_keywords(self, statement: str) -> List[str]:
        """Extrahiert Schlüsselwörter aus der Aussage"""
        # Entferne Stoppwörter und extrahiere wichtige Begriffe
        stopwords = {'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines',
                    'und', 'oder', 'aber', 'doch', 'weil', 'da', 'als', 'wie', 'so', 'dass',
                    'ist', 'sind', 'war', 'waren', 'wird', 'werden', 'hat', 'haben', 'hatte'}

        words = re.findall(r'\b\w+\b', statement.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]

        # Priorisiere politische und wirtschaftliche Begriffe
        priority_terms = {'regierung', 'bundeskanzler', 'politik', 'wirtschaft', 'europa',
                         'klima', 'energie', 'digital', 'bildung', 'gesundheit', 'migration'}

        priority_keywords = [kw for kw in keywords if kw in priority_terms]
        other_keywords = [kw for kw in keywords if kw not in priority_terms]

        return priority_keywords + other_keywords[:3]

    def _get_mock_sources(self, keyword: str) -> List[Dict[str, Any]]:
        """Gibt simulierte Quellen zurück (für Demo-Zwecke)"""
        # In Produktion würde hier eine echte Suche stattfinden
        mock_sources = [
            {
                "title": f"Regierungspolitik zu {keyword.title()}",
                "url": f"https://bundesregierung.de/{keyword}",
                "domain": "bundesregierung.de",
                "credibility_score": 0.95,
                "relevance_score": 0.9,
                "publication_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "summary": f"Offizielle Informationen der Bundesregierung zu {keyword}."
            },
            {
                "title": f"Analyse: {keyword.title()} in Deutschland",
                "url": f"https://zeit.de/{keyword}",
                "domain": "zeit.de",
                "credibility_score": 0.85,
                "relevance_score": 0.7,
                "publication_date": (datetime.now() - timedelta(days=15)).isoformat(),
                "summary": f"Journalistische Analyse zu {keyword}."
            }
        ]

        # Filtere nach verfügbaren Domains
        return [source for source in mock_sources if source['domain'] in self.source_credibility]

    def _calculate_confidence(self, statement: str, sources: List[Dict], bias_score: float) -> float:
        """
        Berechnet einen Confidence-Score für die Aussage
        """
        if not sources:
            return 0.3  # Niedrige Confidence ohne Quellen

        # Basis-Confidence aus Quellen-Glaubwürdigkeit
        credibility_scores = [s.get('credibility_score', 0.5) for s in sources]
        avg_credibility = sum(credibility_scores) / len(credibility_scores)

        # Anzahl der Quellen berücksichtigen
        source_count_factor = min(1.0, len(sources) / 3.0)

        # Bias berücksichtigen (neutrale Aussagen sind vertrauenswürdiger)
        bias_penalty = abs(bias_score) * 0.2

        confidence = (avg_credibility * 0.6) + (source_count_factor * 0.3) + (0.1 - bias_penalty)

        return max(0.0, min(1.0, confidence))

    def _determine_verification_status(self, confidence: float, sources: List[Dict]) -> str:
        """Bestimmt den Verifikationsstatus"""
        if confidence >= 0.8 and len(sources) >= 2:
            return "verified"
        elif confidence >= 0.6 and len(sources) >= 1:
            return "partially_verified"
        elif confidence >= 0.4:
            return "unverified"
        else:
            return "contradicted"

    def _generate_explanation(self, status: str, confidence: float,
                            bias_score: float, sources: List[Dict]) -> str:
        """Generiert eine Erklärung für das Faktenprüfungsergebnis"""
        explanations = {
            "verified": f"Diese Aussage ist gut belegt (Konfidenz: {confidence:.1f}). "
                       f"Sie wurde von {len(sources)} vertrauenswürdigen Quellen bestätigt.",
            "partially_verified": f"Diese Aussage ist teilweise belegt (Konfidenz: {confidence:.1f}). "
                                f"Es gibt einige Quellen, aber weitere Überprüfung wäre ratsam.",
            "unverified": f"Diese Aussage konnte nicht ausreichend überprüft werden (Konfidenz: {confidence:.1f}). "
                         f"Es fehlen verlässliche Quellen oder Daten.",
            "contradicted": f"Diese Aussage widerspricht verfügbaren Informationen (Konfidenz: {confidence:.1f}). "
                           f"Bitte überprüfen Sie die Fakten."
        }

        explanation = explanations.get(status, "Faktenprüfung nicht möglich.")

        # Bias-Information hinzufügen
        if abs(bias_score) > 0.3:
            bias_direction = "links" if bias_score < 0 else "rechts"
            explanation += f" Die Aussage zeigt eine {bias_direction}e Tendenz (Bias-Score: {bias_score:.2f})."

        return explanation

    def validate_response(self, response: str, user_query: str) -> Dict[str, Any]:
        """
        Validiert eine KI-Antwort auf Fakten und Bias
        """
        # Teile die Antwort in einzelne Aussagen auf
        statements = self._split_into_statements(response)

        validation_results = []
        overall_confidence = 0.0
        overall_bias = 0.0

        for statement in statements:
            if len(statement.strip()) > 10:  # Nur substantielle Aussagen prüfen
                result = self.check_fact(statement)
                validation_results.append({
                    "statement": statement,
                    "confidence": result.confidence_score,
                    "bias": result.bias_score,
                    "status": result.verification_status,
                    "explanation": result.explanation
                })
                overall_confidence += result.confidence_score
                overall_bias += result.bias_score

        if validation_results:
            overall_confidence /= len(validation_results)
            overall_bias /= len(validation_results)

        return {
            "overall_confidence": overall_confidence,
            "overall_bias": overall_bias,
            "statement_validations": validation_results,
            "recommendations": self._generate_recommendations(overall_confidence, overall_bias)
        }

    def _split_into_statements(self, text: str) -> List[str]:
        """Teilt Text in einzelne Aussagen auf"""
        # Einfache Aufteilung an Satzenden
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_recommendations(self, confidence: float, bias: float) -> List[str]:
        """Generiert Verbesserungsvorschläge"""
        recommendations = []

        if confidence < 0.7:
            recommendations.append("Fügen Sie mehr spezifische Fakten und Quellen hinzu.")
            recommendations.append("Vermeiden Sie vage Formulierungen.")

        if abs(bias) > 0.4:
            recommendations.append("Bemühen Sie sich um ausgewogenere Formulierungen.")
            recommendations.append("Berücksichtigen Sie verschiedene Perspektiven.")

        if confidence < 0.5:
            recommendations.append("Diese Antwort sollte mit offiziellen Quellen überprüft werden.")

        return recommendations if recommendations else ["Die Antwort scheint gut fundiert zu sein."]