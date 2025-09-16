"""
Erweiterte Antwortqualitäts-Optimierung für Bundeskanzler KI
Beinhaltet Prompt-Engineering, Kontext-Management und Faktenchecking
"""

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ResponseQualityOptimizer:
    """
    Optimiert die Antwortqualität durch verschiedene Techniken:
    - Prompt-Engineering für politische Genauigkeit
    - Kontext-Management für Konversationen
    - Faktenchecking gegen vertrauenswürdige Quellen
    - Confidence-Scoring für Antworten
    """

    def __init__(self, corpus_manager=None):
        self.corpus_manager = corpus_manager
        self.conversation_context = defaultdict(list)  # user_id -> list of messages
        self.fact_database = self._load_fact_database()
        self.political_prompts = self._load_political_prompts()

    def _load_fact_database(self) -> Dict[str, Any]:
        """Lädt die Fakten-Datenbank für Faktenchecking"""
        fact_db = {
            "klima": {
                "klimaneutralität": "Deutschland strebt Klimaneutralität bis 2045 an",
                "kohleausstieg": "Kohleausstieg bis spätestens 2038, Ziel 2030",
                "erneuerbare_energien": "Ausbau erneuerbarer Energien durch EEG",
            },
            "wirtschaft": {
                "mindestlohn": "Mindestlohn wird regelmäßig angepasst",
                "vollbeschäftigung": "Ziel: Vollbeschäftigung und faire Löhne",
                "mittelstand": "Förderung von Mittelstand und Start-ups",
            },
            "europa": {
                "eu_haushalt": "Deutschland ist Nettozahler im EU-Haushalt",
                "wertegemeinschaft": "EU basiert auf Demokratie, Rechtsstaatlichkeit und Menschenrechten",
                "green_deal": "European Green Deal: Klimaneutralität bis 2050",
            },
        }
        return fact_db

    def _load_political_prompts(self) -> Dict[str, str]:
        """Lädt optimierte Prompts für politische Genauigkeit"""
        return {
            "generative_base": """Du bist die Bundeskanzler-KI, ein hilfreicher und sachkundiger KI-Assistent für politische Fragen.

Regeln für Antworten:
- Antworte immer auf Deutsch, es sei denn die Frage ist auf Englisch
- Sei sachlich, präzise und politisch korrekt
- Verwende aktuelle Daten und Fakten (Stand: September 2025)
- Erkläre komplexe politische Zusammenhänge verständlich
- Gib klare Handlungsempfehlungen wo möglich
- Bleibe neutral und objektiv
- Wenn du unsicher bist, sage das deutlich

Frage: {question}

Bitte gib eine fundierte, politisch korrekte Antwort:""",
            "policy_specific": """Als Bundeskanzler-KI beantworte diese politische Frage besonders sorgfältig:

Thema: {topic}
Kontext: {context}

Regeln:
- Verwende nur verifizierte Informationen
- Zitiere relevante Gesetze oder Programme
- Erkläre die Position der Bundesregierung
- Berücksichtige europäische und internationale Aspekte
- Gib konkrete Beispiele oder Zahlen

Frage: {question}

Antwort:""",
            "follow_up": """Diese Frage ist Teil einer laufenden Konversation.

Bisheriger Kontext:
{conversation_history}

Neue Frage: {question}

Bitte gib eine kohärente Antwort, die den bisherigen Kontext berücksichtigt:""",
        }

    def optimize_prompt(self, question: str, context: Optional[Dict] = None) -> str:
        """
        Optimiert den Prompt basierend auf der Frage und dem Kontext

        Args:
            question: Die Benutzerfrage
            context: Optionaler Kontext (conversation_history, topic, etc.)

        Returns:
            Optimierter Prompt für das Modell
        """
        if not context:
            context = {}

        # Bestimme den Fragetyp und das Thema
        question_type = self._classify_question(question)
        topic = self._extract_topic(question)

        # Wähle den passenden Prompt-Typ
        if context.get("conversation_history"):
            prompt_template = self.political_prompts["follow_up"]
            context["conversation_history"] = self._format_conversation_history(
                context["conversation_history"]
            )
        elif topic and topic in self.fact_database:
            prompt_template = self.political_prompts["policy_specific"]
            context["topic"] = topic
        else:
            prompt_template = self.political_prompts["generative_base"]

        # Fülle den Prompt aus
        try:
            prompt = prompt_template.format(question=question, **context)
        except KeyError as e:
            logger.warning(f"Fehlender Kontext-Key: {e}, verwende Basis-Prompt")
            prompt = self.political_prompts["generative_base"].format(question=question)

        return prompt

    def _classify_question(self, question: str) -> str:
        """Klassifiziert den Typ der Frage"""
        question = question.lower()

        if any(word in question for word in ["wie", "warum", "weshalb", "wieso"]):
            return "explanatory"
        elif any(word in question for word in ["was", "wer", "wann", "wo"]):
            return "factual"
        elif any(word in question for word in ["sollte", "müsste", "könnte"]):
            return "opinion"
        elif any(word in question for word in ["plan", "strategie", "zukunft"]):
            return "strategic"
        else:
            return "general"

    def _extract_topic(self, question: str) -> Optional[str]:
        """Extrahiert das politische Thema aus der Frage"""
        question = question.lower()

        topic_keywords = {
            "klima": ["klima", "umwelt", "kohle", "energie", "erneuerbar"],
            "wirtschaft": [
                "wirtschaft",
                "arbeit",
                "unternehmen",
                "mittelstand",
                "export",
            ],
            "europa": ["eu", "europa", "europäisch", "kommission", "parlament"],
            "soziales": ["sozial", "rente", "armut", "bildung", "gesundheit"],
            "sicherheit": ["sicherheit", "verteidigung", "armee", "cyber", "polizei"],
            "digital": ["digital", "internet", "ki", "technologie", "daten"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in question for keyword in keywords):
                return topic

        return None

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Formatiert den Konversationsverlauf für den Prompt"""
        formatted = []
        for msg in history[-3:]:  # Nur die letzten 3 Nachrichten
            role = "Nutzer" if msg.get("role") == "user" else "KI"
            content = msg.get("content", "")[:200]  # Begrenze Länge
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def add_conversation_context(self, user_id: str, message: Dict):
        """Fügt eine Nachricht zum Konversationskontext hinzu"""
        self.conversation_context[user_id].append(message)

        # Begrenze Kontext auf letzte 10 Nachrichten
        if len(self.conversation_context[user_id]) > 10:
            self.conversation_context[user_id] = self.conversation_context[user_id][-10:]

    def get_conversation_context(self, user_id: str) -> List[Dict]:
        """Gibt den Konversationskontext für einen Benutzer zurück"""
        return self.conversation_context.get(user_id, [])

    def fact_check_response(self, response: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Führt ein Faktenchecking der Antwort durch

        Returns:
            Dict mit confidence_score, verified_facts, unverified_claims
        """
        result = {
            "confidence_score": 0.8,  # Basis-Confidence
            "verified_facts": [],
            "unverified_claims": [],
            "suggestions": [],
        }

        if not topic or topic not in self.fact_database:
            result["confidence_score"] = 0.6  # Reduzierte Confidence ohne Thema
            result["suggestions"].append(
                "Antwort ohne spezifisches Thema - bitte Thema spezifizieren"
            )
            return result

        facts = self.fact_database[topic]
        response_lower = response.lower()

        verified_count = 0
        for fact_key, fact_value in facts.items():
            if fact_key.lower() in response_lower or fact_value.lower() in response_lower:
                result["verified_facts"].append({"fact": fact_value, "verified": True})
                verified_count += 1

        # Berechne Confidence basierend auf verifizierten Fakten
        if verified_count > 0:
            result["confidence_score"] = min(0.95, 0.7 + (verified_count * 0.1))
        else:
            result["confidence_score"] = 0.5
            result["suggestions"].append("Keine verifizierten Fakten in der Antwort gefunden")

        return result

    def calculate_response_quality_score(self, response: str, question: str) -> Dict[str, Any]:
        """
        Berechnet einen umfassenden Qualitäts-Score für die Antwort

        Returns:
            Dict mit verschiedenen Qualitätsmetriken
        """
        quality_metrics = {
            "overall_score": 0.0,
            "criteria": {
                "relevance": 0.0,
                "accuracy": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "helpfulness": 0.0,
            },
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
        }

        # Relevanz prüfen
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        overlap = len(question_words.intersection(response_words))
        relevance_score = min(1.0, overlap / max(len(question_words), 1))
        quality_metrics["criteria"]["relevance"] = relevance_score

        if relevance_score > 0.3:
            quality_metrics["strengths"].append("Hohe Relevanz zur Frage")
        else:
            quality_metrics["weaknesses"].append("Geringe Relevanz zur Frage")
            quality_metrics["improvement_suggestions"].append(
                "Antwort direkter auf die Frage beziehen"
            )

        # Vollständigkeit prüfen
        if len(response.split()) > 20:
            quality_metrics["criteria"]["completeness"] = 0.9
            quality_metrics["strengths"].append("Umfassende Antwort")
        elif len(response.split()) > 10:
            quality_metrics["criteria"]["completeness"] = 0.7
        else:
            quality_metrics["criteria"]["completeness"] = 0.4
            quality_metrics["weaknesses"].append("Antwort zu knapp")
            quality_metrics["improvement_suggestions"].append("Antwort mit mehr Details ausfüllen")

        # Klarheit prüfen
        if response.count(".") > 2 and len(response) > 50:
            quality_metrics["criteria"]["clarity"] = 0.8
            quality_metrics["strengths"].append("Klare und strukturierte Antwort")
        else:
            quality_metrics["criteria"]["clarity"] = 0.6
            quality_metrics["improvement_suggestions"].append("Antwort strukturierter gestalten")

        # Hilfsbereitschaft prüfen
        helpful_indicators = [
            "ich helfe",
            "wir können",
            "es gibt",
            "möglich",
            "empfehle",
        ]
        helpful_score = sum(1 for indicator in helpful_indicators if indicator in response.lower())
        quality_metrics["criteria"]["helpfulness"] = min(1.0, helpful_score * 0.2)

        if helpful_score > 0:
            quality_metrics["strengths"].append("Hilfreiche Handlungsempfehlungen")
        else:
            quality_metrics["improvement_suggestions"].append(
                "Konkrete Handlungsempfehlungen hinzufügen"
            )

        # Faktenchecking für Accuracy
        topic = self._extract_topic(question)
        fact_check = self.fact_check_response(response, topic)
        quality_metrics["criteria"]["accuracy"] = fact_check["confidence_score"]

        if fact_check["verified_facts"]:
            quality_metrics["strengths"].append(
                f"{len(fact_check['verified_facts'])} Fakten verifiziert"
            )
        if fact_check["suggestions"]:
            quality_metrics["improvement_suggestions"].extend(fact_check["suggestions"])

        # Gesamt-Score berechnen
        weights = {
            "relevance": 0.25,
            "accuracy": 0.3,
            "completeness": 0.2,
            "clarity": 0.15,
            "helpfulness": 0.1,
        }
        overall_score = sum(quality_metrics["criteria"][k] * v for k, v in weights.items())
        quality_metrics["overall_score"] = round(overall_score, 2)

        return quality_metrics

    def enhance_response(
        self, response: str, question: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Verbessert eine Antwort und gibt Qualitätsmetriken zurück

        Returns:
            Dict mit enhanced_response, quality_metrics, suggestions
        """
        # Qualitätsanalyse
        quality_metrics = self.calculate_response_quality_score(response, question)

        # Verbesserungsvorschläge basierend auf Qualitätsanalyse
        enhanced_response = response

        if quality_metrics["overall_score"] < 0.7:
            # Verbesserungen anwenden
            if quality_metrics["criteria"]["completeness"] < 0.6:
                # Antwort erweitern
                topic = self._extract_topic(question)
                if topic and topic in self.fact_database:
                    facts = self.fact_database[topic]
                    fact_text = " Zusätzlich: " + list(facts.values())[0]
                    if fact_text not in enhanced_response:
                        enhanced_response += fact_text

            if quality_metrics["criteria"]["helpfulness"] < 0.5:
                # Handlungsempfehlung hinzufügen
                enhanced_response += " Für weitere Informationen steht Ihnen die Bundesregierung gerne zur Verfügung."

        return {
            "original_response": response,
            "enhanced_response": enhanced_response,
            "quality_metrics": quality_metrics,
            "improvements_applied": enhanced_response != response,
        }
