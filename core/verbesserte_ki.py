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

from core.rag_system import RAGSystem

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

        self.rag_system = RAGSystem(corpus_path=corpus_path)
        print("✅ RAG-System geladen")

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

    def antwort(self, frage: str) -> dict:
        """
        Generiert eine optimierte Antwort auf eine Frage

        Args:
            frage: Die Benutzerfrage

        Returns:
            Dict mit Antwort, Konfidenz und Metadaten
        """
        try:
            # Erkenne Thema
            thema = self.erkenne_thema(frage)

            # Retrieve relevante Dokumente
            docs = self.rag_system.retrieve_relevant_documents(frage, top_k=5)

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
                konfidenz = beste_antwort["score"]

                # Erweitere Antwort wenn möglich
                if len(thema_docs) > 1 and konfidenz > 0.6:
                    zusatz_info = [
                        doc["text"] for doc in thema_docs[1:3] if doc["score"] > 0.5
                    ]
                    if zusatz_info:
                        antwort_text += f" Zusätzlich: {' '.join(zusatz_info)}"
            else:
                # Fallback zur besten verfügbaren Antwort
                beste_antwort = docs[0]
                antwort_text = beste_antwort["text"]
                konfidenz = beste_antwort["score"]

            # Logge die Antwort
            self._log_antwort(frage, antwort_text, konfidenz, thema)

            return {
                "antwort": antwort_text,
                "konfidenz": konfidenz,
                "thema": thema,
                "methode": "rag_optimiert",
                "dokumente_verwendet": len(docs),
            }

        except Exception as e:
            logging.error(f"Fehler bei Antwortgenerierung: {e}")
            return {
                "antwort": f"Entschuldigung, es gab einen technischen Fehler: {str(e)}",
                "konfidenz": 0.0,
                "thema": "fehler",
                "methode": "error",
            }

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
            print(f"📊 Konfidenz: {result['konfidenz']:.1%}")
            print(f"📋 Thema: {result['thema']}")
            if result["konfidenz"] < 0.5:
                print("⚠️  Niedrige Konfidenz - Antwort möglicherweise ungenau")
            print()

        except KeyboardInterrupt:
            print("\n👋 Auf Wiedersehen!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    interaktiver_modus()
