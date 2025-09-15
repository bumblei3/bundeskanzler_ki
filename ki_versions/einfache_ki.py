#!/usr/bin/env python3
"""
Einfacher Test für die Bundeskanzler-KI ohne komplexe Generierung
Nutzt nur das RAG-System für direkte, relevante Antworten
"""

import os
import sys

# Dynamischer Pfad zum Projekt-Root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import logging

from core.rag_system import RAGSystem

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def einfache_ki_antwort(frage: str) -> str:
    """
    Gibt eine einfache, direkte Antwort basierend auf dem RAG-System
    """
    try:
        # Initialisiere RAG-System mit korrektem Pfad
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        corpus_path = os.path.join(project_root, "data", "corpus.json")

        rag = RAGSystem(corpus_path=corpus_path)

        # Führe RAG-Abfrage durch
        result = rag.retrieve_relevant_documents(frage, top_k=3)

        if result and len(result) > 0:
            # Nimm die relevantesten Dokumente
            relevant_docs = result[:3]

            # Kombiniere die Antworten zu einer kohärenten Antwort
            antwort_teile = []
            for doc in relevant_docs:
                if doc["score"] > 0.5:  # Nur sehr relevante Dokumente
                    antwort_teile.append(doc["text"])

            if antwort_teile:
                antwort = " ".join(antwort_teile)
                return f"📋 Basierend auf der Wissensbasis: {antwort}"
            else:
                # Fallback zu weniger relevanten Dokumenten
                antwort = (
                    relevant_docs[0]["text"]
                    if relevant_docs
                    else "Keine relevanten Informationen gefunden."
                )
                return f"📋 Basierend auf der Wissensbasis: {antwort}"
        else:
            return "❌ Keine relevanten Informationen in der Wissensbasis gefunden."

    except Exception as e:
        return f"❌ Fehler bei der Antwortgenerierung: {e}"


if __name__ == "__main__":
    print("🚀 Einfache Bundeskanzler-KI (RAG-basiert)")
    print("=" * 50)

    while True:
        try:
            frage = input("\n🤖 Ihre Frage: ").strip()

            if frage.lower() in ["exit", "quit", "bye"]:
                print("👋 Auf Wiedersehen!")
                break

            if not frage:
                continue

            print(f"\n💭 Suche Antwort für: {frage}")
            antwort = einfache_ki_antwort(frage)
            print(f"\n🎯 {antwort}")

        except KeyboardInterrupt:
            print("\n👋 Auf Wiedersehen!")
            break
        except EOFError:
            break
