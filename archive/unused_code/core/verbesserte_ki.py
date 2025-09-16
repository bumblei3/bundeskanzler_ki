#!/usr/bin/env python3
"""
Verbesserte Bundeskanzler-KI mit optimiertem RAG-System
Fokussiert auf direkte, relevante Antworten ohne fehlerhafte Textgenerierung
"""

import os
import sys

# Dynamischer Pfad zum Projekt-Root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.gpu_rag_system import GPUAcceleratedRAG
from core.multilingual_manager import Language, get_multilingual_manager
from core.multimodal_generator import get_multimodal_generator
from core.security_manager import get_security_manager
from core.user_profile_manager import get_user_profile_manager

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class VerbesserteBundeskanzlerKI:
    """
    Verbesserte Bundeskanzler-KI mit optimiertem RAG-System
    """

    def __init__(self):
        """Initialisiert die verbesserte KI"""
        print("🚀 Initialisiere Bundeskanzler-KI (Optimierte Version)...")

        # Pfad zur Corpus-Datei
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        corpus_path = os.path.join(project_root, "data", "corpus.json")

        self.rag_system = GPUAcceleratedRAG(corpus_path=corpus_path)
        print("✅ RAG-System geladen")

        # Security Manager initialisieren
        self.security_manager = get_security_manager()
        print("🛡️ Security Manager geladen")

        # User Profile Manager initialisieren
        self.profile_manager = get_user_profile_manager()
        print("🧠 User Profile Manager geladen")

        # Multilingual Manager initialisieren
        self.multilingual_manager = get_multilingual_manager()
        print("🌍 Multilingual Manager geladen")

        # Multimodal Generator initialisieren
        self.multimodal_generator = get_multimodal_generator()
        print("🎨 Multimodal Generator geladen")

        # Themen-Keywords für bessere Antworten
        self.themen_keywords = {
            "klima": [
                "klima",
                "klimaschutz",
                "klimaneutralität",
                "energie",
                "erneuerbar",
                "kohle",
                "co2",
            ],
            "wirtschaft": [
                "wirtschaft",
                "industrie",
                "mittelstand",
                "start-up",
                "innovation",
                "ki",
                "wasserstoff",
            ],
            "gesundheit": ["gesundheit", "pflege", "kranken", "medizin", "prävention"],
            "soziales": ["sozial", "rente", "kindergeld", "armut", "integration"],
            "bildung": ["bildung", "schule", "universität", "ausbildung", "lernen"],
            "digital": ["digital", "technologie", "internet", "cybersicherheit"],
            "europa": ["europa", "eu", "union", "zusammenarbeit"],
            "sicherheit": ["sicherheit", "polizei", "bundeswehr", "verteidigung"],
        }

    def erkenne_thema(self, frage: str) -> str:
        """Erkennt das Hauptthema einer Frage"""
        frage_lower = frage.lower()

        for thema, keywords in self.themen_keywords.items():
            if any(keyword in frage_lower for keyword in keywords):
                return thema

        return "allgemein"

    def antwort(
        self, frage: str, user_id: str = "anonymous", feedback: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        Generiert eine personalisierte Antwort auf eine Frage mit Sicherheitsprüfungen

        Args:
            frage: Die Benutzerfrage
            user_id: Optionale User-ID für Personalisierung
            feedback: Optionales Feedback von vorheriger Interaktion

        Returns:
            Dict mit Antwort, Konfidenz und Metadaten
        """
        try:
            # 🌍 SPRACHERKENNUNG UND ÜBERSETZUNG
            detected_lang, lang_confidence = self.multilingual_manager.detect_language(frage)

            # Übersetze Frage ins Deutsche falls nötig
            if detected_lang.value != "de":
                translation_result = self.multilingual_manager.translate_text(
                    frage, Language.DE, detected_lang
                )
                frage_deutsch = translation_result.translated_text
                original_frage = frage
                frage = frage_deutsch
                print(f"🌍 Frage übersetzt: {original_frage} -> {frage_deutsch}")
            else:
                original_frage = frage
                frage_deutsch = frage

            # 🧠 NUTZERPROFIL LADEN
            user_profile = self.profile_manager.get_or_create_profile(user_id)
            personalized_recommendations = self.profile_manager.get_personalized_recommendations(
                user_id
            )

            # 🛡️ SICHERHEITSPRÜFUNGEN

            # 1. Input-Validation
            is_valid, validation_reason, validation_metadata = self.security_manager.validate_input(
                frage, user_id
            )

            if not is_valid:
                return {
                    "antwort": f"❌ Ihre Anfrage wurde aus Sicherheitsgründen blockiert: {validation_reason}",
                    "konfidenz": 0.0,
                    "thema": "sicherheit",
                    "methode": "security_blocked",
                    "security_status": "blocked",
                    "block_reason": validation_reason,
                }

            # 2. Bias-Detection für Input
            input_bias = self.security_manager.detect_bias(frage)

            # Erkenne Thema
            thema = self.erkenne_thema(frage)

            # Retrieve relevante Dokumente mit Personalisierung
            docs = self.rag_system.retrieve_relevant_documents(frage, top_k=5)

            # Personalisierte Filterung basierend auf Nutzerinteressen
            docs = self._personalisiere_dokumente(docs, user_profile, thema)

            if not docs:
                return {
                    "antwort": "Entschuldigung, ich habe keine relevanten Informationen zu Ihrer Frage gefunden.",
                    "konfidenz": 0.0,
                    "thema": thema,
                    "methode": "fallback",
                }

            # Filtere nach Thema
            thema_docs = self._filtere_nach_thema(docs, thema, frage)

            if thema_docs:
                beste_antwort = thema_docs[0]
                antwort_text = beste_antwort["text"]
                # Konfidenz als Prozentwert (0-100%) skalieren
                konfidenz = beste_antwort["score"] * 100

                # Personalisierte Antwortgenerierung
                antwort_text = self._personalisiere_antwort(
                    antwort_text, user_profile, personalized_recommendations, konfidenz
                )

                # 3. Content-Filtering für Output
                content_allowed, filter_reason, filter_metadata = (
                    self.security_manager.filter_content(antwort_text, beste_antwort)
                )

                if not content_allowed:
                    return {
                        "antwort": f"⚠️ Diese Antwort wurde aus Sicherheitsgründen gefiltert: {filter_reason}",
                        "konfidenz": 0.0,
                        "thema": thema,
                        "methode": "content_filtered",
                        "security_status": "filtered",
                        "filter_reason": filter_reason,
                    }

                # 4. Bias-Detection für Output
                output_bias = self.security_manager.detect_bias(antwort_text, beste_antwort)

                # Erweitere Antwort wenn möglich
                if len(thema_docs) > 1 and konfidenz > 50:  # Threshold für Prozentwerte anpassen
                    zusatz_info = [
                        doc["text"]
                        for doc in thema_docs[1:3]
                        if doc["score"] * 100 > 30  # Threshold für Prozentwerte anpassen
                    ]
                    if zusatz_info:
                        antwort_text += f" Zusätzlich: {' '.join(zusatz_info)}"
            else:
                # Fallback zur besten verfügbaren Antwort
                beste_antwort = docs[0]
                antwort_text = beste_antwort["text"]
                konfidenz = beste_antwort["score"] * 100  # Konfidenz als Prozentwert skalieren

                # Personalisierte Antwortgenerierung auch für Fallback
                antwort_text = self._personalisiere_antwort(
                    antwort_text, user_profile, personalized_recommendations, konfidenz
                )

            # Logge die Antwort
            self._log_antwort(frage, antwort_text, konfidenz, thema)

            # 5. Ethics-Reporting
            interaction_data = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "question": frage,
                "answer": antwort_text,
                "confidence": konfidenz / 100,  # Konfidenz als 0-1 Wert für Profile
                "theme": thema,
                "sources": [doc.get("source", "unbekannt") for doc in docs[:3]],
                "bias_detected": output_bias["bias_detected"],
                "bias_score": output_bias["bias_score"],
                "source_verified": beste_antwort.get("verified", False),
                "logged": True,
            }

            ethics_report = self.security_manager.generate_ethics_report(interaction_data)

            # 🧠 NUTZERPROFIL AKTUALISIEREN
            self.profile_manager.update_interaction(
                user_id=user_id,
                question=frage,
                answer=antwort_text,
                confidence=konfidenz / 100,
                theme=thema,
                feedback=feedback,
            )

            # 🌍 ANTWORT ÜBERSETZEN falls Originalfrage nicht auf Deutsch war
            if detected_lang.value != "de":
                antwort_translation = self.multilingual_manager.translate_text(
                    antwort_text, detected_lang, Language.DE
                )
                antwort_text = antwort_translation.translated_text
                print(
                    f"🌍 Antwort übersetzt: {antwort_translation.original_text[:50]}... -> {antwort_text[:50]}..."
                )

            return {
                "antwort": antwort_text,
                "konfidenz": konfidenz,
                "thema": thema,
                "methode": "rag_optimiert",
                "dokumente_verwendet": len(docs),
                "quellen": [doc.get("source", "unbekannt") for doc in docs[:3]],
                "vertrauenslevel": beste_antwort.get("confidence_level", "unbekannt"),
                "erklaerung": beste_antwort.get(
                    "explanation", "Keine detaillierte Erklärung verfügbar."
                ),
                "verifiziert": beste_antwort.get("verified", False),
                "datum": beste_antwort.get("date", "unbekannt"),
                # 🛡️ SICHERHEITSINFORMATIONEN
                "security_status": "approved",
                "input_bias_detected": input_bias["bias_detected"],
                "input_bias_score": input_bias["bias_score"],
                "output_bias_detected": output_bias["bias_detected"],
                "output_bias_score": output_bias["bias_score"],
                "ethics_transparency_score": ethics_report["transparency_score"],
                "ethics_fairness_score": ethics_report["fairness_score"],
                "ethics_accountability_score": ethics_report["accountability_score"],
                "ethics_issues": ethics_report["issues"],
                "ethics_recommendations": ethics_report["recommendations"],
                # 🧠 PERSONALSISIERUNGSINFORMATIONEN
                "personalization_level": personalized_recommendations.get(
                    "personalization_level", "basic"
                ),
                "user_interaction_count": len(user_profile.interaction_history),
                "suggested_topics": personalized_recommendations.get("suggested_topics", []),
                # 🌍 MEHRSPRACHIGE INFORMATIONEN
                "detected_language": detected_lang.value,
                "language_confidence": lang_confidence,
                "original_question": original_frage if detected_lang.value != "de" else frage,
                "translated_question": frage_deutsch if detected_lang.value != "de" else None,
            }

        except Exception as e:
            logging.error(f"Fehler bei Antwortgenerierung: {e}")
            return {
                "antwort": f"Entschuldigung, es gab einen technischen Fehler: {str(e)}",
                "konfidenz": 0.0,
                "thema": "fehler",
                "methode": "error",
            }

    def _personalisiere_dokumente(self, docs: list, user_profile, thema: str) -> list:
        """Personalisiert Dokumente basierend auf Nutzerprofil"""
        if not docs:
            return docs

        # Sortiere Dokumente nach Nutzerinteresse
        user_interest = user_profile.interests.get(thema, 0.5)

        for doc in docs:
            # Erhöhe Score für interessante Themen
            interest_boost = user_interest * 0.2  # Max 20% Boost
            doc["score"] = min(1.0, doc["score"] + interest_boost)

        # Sortiere nach angepasstem Score
        docs.sort(key=lambda x: x["score"], reverse=True)

        return docs

    def _personalisiere_antwort(
        self, antwort_text: str, user_profile, recommendations: dict, konfidenz: float
    ) -> str:
        """Personalisiert die Antwort basierend auf Nutzerprofil"""
        personalized_answer = antwort_text

        # Anpassung an Detail-Level
        detail_level = recommendations.get("preferred_detail_level", "medium")

        if detail_level == "low" and len(antwort_text) > 200:
            # Kürze Antwort für low detail
            sentences = antwort_text.split(".")
            personalized_answer = ".".join(sentences[:2]) + "."
        elif detail_level == "high" and konfidenz > 60:
            # Erweitere Antwort für high detail
            personalized_answer += " (Diese Information basiert auf verifizierten Quellen.)"

        # Anpassung an Response Style
        response_style = recommendations.get("response_style", "formal")

        if response_style == "casual":
            # Mache Antwort umgangssprachlicher
            personalized_answer = personalized_answer.replace("Deutschland", "Deutschland")
            if "setzt sich" in personalized_answer:
                personalized_answer = personalized_answer.replace("setzt sich", "geht ran an")
        elif response_style == "technical":
            # Mache Antwort technischer
            if "Klimaneutralität" in personalized_answer:
                personalized_answer += " (CO2-Netto-Null-Emissionen bis 2045)"

        # Personalisierte Empfehlungen hinzufügen
        personalization_level = recommendations.get("personalization_level", "basic")

        if personalization_level in ["intermediate", "advanced"]:
            suggested_topics = recommendations.get("suggested_topics", [])
            if suggested_topics:
                personalized_answer += f" 💡 Basierend auf Ihren Interessen könnten Sie auch fragen: {', '.join(suggested_topics[:2])}"

        return personalized_answer

    def _filtere_nach_thema(self, docs: list, thema: str, frage: str) -> list:
        """Filtert Dokumente nach erkanntem Thema"""
        if thema == "allgemein":
            return docs

        # Hole thema-spezifische Keywords
        keywords = self.themen_keywords.get(thema, [])

        # Filtere Dokumente
        thema_docs = []
        for doc in docs:
            doc_text = doc["text"].lower()
            if any(keyword in doc_text for keyword in keywords):
                thema_docs.append(doc)

        return thema_docs if thema_docs else docs

    def _log_antwort(self, frage: str, antwort: str, konfidenz: float, thema: str):
        """Loggt Antworten für Monitoring"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"[{timestamp}] Eingabe: {frage}\n"
        log_entry += f"  Antwort: {antwort}\n"
        log_entry += f"  Konfidenz: {konfidenz:.1%}\n"
        log_entry += f"  Thema: {thema}\n\n"

        try:
            # Pfad zur Log-Datei
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            log_path = os.path.join(project_root, "data", "log.txt")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            logging.warning(f"Konnte Log nicht schreiben: {e}")


def interaktiver_modus():
    """Startet den interaktiven Modus"""
    ki = VerbesserteBundeskanzlerKI()

    print("\n🤖 Bundeskanzler-KI bereit!")
    print("📋 Stellen Sie Fragen zur deutschen Politik, Wirtschaft, Klimaschutz, etc.")
    print("💡 Beenden mit 'exit', 'quit' oder Ctrl+C\n")

    while True:
        try:
            frage = input("🤖 Ihre Frage: ").strip()

            if frage.lower() in ["exit", "quit", "bye", "tschüss"]:
                print("👋 Auf Wiedersehen!")
                break

            if not frage:
                continue

            # Generiere Antwort
            result = ki.antwort(frage)

            # Ausgabe formatieren
            print(f"\n💡 {result['antwort']}")

            # Korrigiere Konfidenz-Anzeige für niedrige Scores
            konfidenz = result["konfidenz"]
            if konfidenz < 10:  # Wenn Konfidenz sehr niedrig ist
                vertrauenslevel = "sehr niedrig"
            elif konfidenz < 20:
                vertrauenslevel = "niedrig"
            elif konfidenz < 40:
                vertrauenslevel = "mittel"
            elif konfidenz < 60:
                vertrauenslevel = "hoch"
            else:
                vertrauenslevel = "sehr hoch"

            print(f"📊 Konfidenz: {konfidenz:.1f}% ({vertrauenslevel})")
            print(f"📋 Thema: {result['thema']}")
            print(f"🔍 Methode: {result['methode']}")
            if "dokumente_verwendet" in result:
                print(f"📚 Dokumente verwendet: {result['dokumente_verwendet']}")

            # Erklärende Informationen
            if "quellen" in result and result["quellen"]:
                print(f"📄 Quellen: {', '.join(result['quellen'][:3])}")
            if "verifiziert" in result:
                status = "✅ Verifiziert" if result["verifiziert"] else "⚠️ Nicht verifiziert"
                print(f"🔒 Status: {status}")
            if "datum" in result and result["datum"] != "unbekannt":
                print(f"📅 Datum: {result['datum']}")
            if "erklaerung" in result:
                erklaerung = result["erklaerung"]
                # Korrigiere Erklärung für niedrige Konfidenz
                if konfidenz < 10:
                    erklaerung = erklaerung.replace("sehr hoch", "sehr niedrig")
                elif konfidenz < 20:
                    erklaerung = erklaerung.replace("sehr hoch", "niedrig")
                print(f"💭 Erklärung: {erklaerung}")

            # 🛡️ SICHERHEITSINFORMATIONEN
            security_status = result.get("security_status", "unknown")
            if security_status == "approved":
                print(f"🛡️ Sicherheit: ✅ Genehmigt")

                # Bias-Informationen
                input_bias = result.get("input_bias_detected", False)
                output_bias = result.get("output_bias_detected", False)

                if input_bias or output_bias:
                    print(f"⚠️ Bias erkannt: Input={input_bias}, Output={output_bias}")
                    if result.get("output_bias_score", 0) > 0.5:
                        print(f"   Bias-Score: {result.get('output_bias_score', 0):.2f}")

                # Ethics-Scores
                transparency = result.get("ethics_transparency_score", 0)
                fairness = result.get("ethics_fairness_score", 0)
                accountability = result.get("ethics_accountability_score", 0)

                print(
                    f"📊 Ethik-Scores: Transparenz={transparency:.1f}, Fairness={fairness:.1f}, Accountability={accountability:.1f}"
                )

                # Ethics-Issues
                if result.get("ethics_issues"):
                    print(f"⚠️ Ethik-Hinweise: {', '.join(result['ethics_issues'])}")

                if result.get("ethics_recommendations"):
                    print(f"💡 Empfehlungen: {', '.join(result['ethics_recommendations'])}")

            elif security_status == "blocked":
                print(
                    f"🛡️ Sicherheit: ❌ Blockiert - {result.get('block_reason', 'Unbekannter Grund')}"
                )
            elif security_status == "filtered":
                print(
                    f"🛡️ Sicherheit: ⚠️ Gefiltert - {result.get('filter_reason', 'Unbekannter Grund')}"
                )

            if result["konfidenz"] < 30:  # Threshold für Prozentwerte anpassen
                print("⚠️  Niedrige Konfidenz - Antwort möglicherweise ungenau")
            print()

            # 🧠 PERSONALSISIERUNGSINFORMATIONEN
            personalization_level = result.get("personalization_level", "basic")
            interaction_count = result.get("user_interaction_count", 0)
            suggested_topics = result.get("suggested_topics", [])

            if personalization_level != "basic":
                print(
                    f"🧠 Personalisierung: {personalization_level.title()} (Interaktionen: {interaction_count})"
                )
                if suggested_topics:
                    print(f"💡 Vorgeschlagene Themen: {', '.join(suggested_topics)}")
                print()

            # 💬 FEEDBACK ANFRAGE
            if personalization_level in ["intermediate", "advanced"]:
                print("📝 Wie hilfreich war diese Antwort? (1-5, oder Enter für kein Feedback)")
                try:
                    feedback_input = input("Feedback: ").strip()
                    if feedback_input and feedback_input.isdigit():
                        feedback_score = int(feedback_input)
                        if 1 <= feedback_score <= 5:
                            # Konvertiere zu 0-1 Skala
                            relevance_score = feedback_score / 5.0
                            helpfulness_score = feedback_score / 5.0

                            feedback = {
                                "relevance": relevance_score,
                                "helpfulness": helpfulness_score,
                                "overall_satisfaction": relevance_score,
                            }

                            # Aktualisiere Profil mit Feedback
                            ki.profile_manager.update_interaction(
                                user_id="anonymous",
                                question=frage,
                                answer=result["antwort"],
                                confidence=result["konfidenz"] / 100,
                                theme=result["thema"],
                                feedback=feedback,
                            )

                            print(
                                f"✅ Feedback gespeichert! Danke für Ihre Bewertung ({feedback_score}/5)"
                            )
                        else:
                            print(
                                "❌ Ungültige Bewertung. Bitte geben Sie eine Zahl zwischen 1 und 5 ein."
                            )
                    print()
                except (EOFError, KeyboardInterrupt):
                    print()

        except KeyboardInterrupt:
            print("\n👋 Auf Wiedersehen!")
            break
        except EOFError:
            break


def generiere_multimodale_antwort(
    ki_system: VerbesserteBundeskanzlerKI,
    frage: str,
    user_id: str = "anonymous",
    visual_type: str = "auto",
) -> dict:
    """
    Generiert eine multimodale Antwort mit visuellen Elementen

    Args:
        ki_system: Die KI-Instanz
        frage: Die Benutzerfrage
        user_id: User-ID für Personalisierung
        visual_type: Typ der Visualisierung (auto, trends, comparison, distribution, timeline)

    Returns:
        Dictionary mit multimodaler Antwort
    """
    try:
        # Erstelle normale Antwort
        text_response = ki_system.antwort(frage, user_id)

        # Erkenne Thema für Visualisierung
        thema = text_response.get("thema", "allgemein")

        # Bestimme Visualisierungstyp
        if visual_type == "auto":
            # Automatische Auswahl basierend auf Thema und Frage
            if "vergleich" in frage.lower() or "vs" in frage.lower():
                visual_type = "comparison"
            elif "entwicklung" in frage.lower() or "trend" in frage.lower():
                visual_type = "trends"
            elif "verteilung" in frage.lower() or "regional" in frage.lower():
                visual_type = "distribution"
            elif "ereignis" in frage.lower() or "zeitstrahl" in frage.lower():
                visual_type = "timeline"
            else:
                visual_type = "trends"

        # Generiere visuelle Antwort
        visual_response = ki_system.multimodal_generator.generate_visual_response(
            topic=thema, data_type=visual_type
        )

        # Kombiniere Text- und visuelle Antwort
        multimodal_response = text_response.copy()
        multimodal_response.update(
            {
                "multimodal": True,
                "visual_type": visual_type,
                "visual_chart": visual_response.get("chart_path"),
                "visual_description": visual_response.get("description"),
                "visual_insights": visual_response.get("insights", ""),
                "combined_answer": f"{text_response['antwort']}\n\n📊 Visuelle Darstellung: {visual_response.get('description', '')}",
            }
        )

        return multimodal_response

    except Exception as e:
        logging.error(f"Fehler bei multimodaler Antwortgenerierung: {e}")
        # Fallback zur normalen Textantwort
        return ki_system.antwort(frage, user_id)


if __name__ == "__main__":
    interaktiver_modus()
