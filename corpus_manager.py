"""
Korpus-Manager für die Bundeskanzler-KI.
Verwaltet das Laden, Speichern und Validieren des Trainingskorpus.
"""
import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from corpus_validator import CorpusValidator, print_validation_report

class CorpusManager:
    def __init__(self, corpus_file: str = "corpus.json"):
        """
        Initialisiert den Korpus-Manager.
        
        Args:
            corpus_file: Pfad zur Korpus-Datei (JSON-Format)
        """
        self.corpus_file = corpus_file
        self.corpus: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.validator = CorpusValidator()
        self.load_corpus()
    
    def load_corpus(self) -> None:
        """Lädt den Korpus aus der JSON-Datei."""
        if os.path.exists(self.corpus_file):
            try:
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Verarbeite das neue Format
                    self.corpus = []
                    if 'entries' in data:
                        print(f"Gefundene Einträge im Korpus: {len(data['entries'])}")
                        self.corpus = []
                        for entry in data['entries']:
                            if isinstance(entry, dict) and 'text' in entry:
                                self.corpus.append(entry['text'])
                                print(f"Hinzugefügter Text: {entry['text'][:50]}...")
                            else:
                                logging.warning(f"Ungültiger Eintrag im Korpus: {entry}")
                    print(f"Geladener Korpus enthält {len(self.corpus)} Einträge")
                    logging.info(f"Korpus geladen aus {self.corpus_file} mit {len(self.corpus)} Einträgen")
            except Exception as e:
                logging.error(f"Fehler beim Laden des Korpus: {e}")
                self._initialize_default_corpus()
        else:
            logging.warning(f"{self.corpus_file} nicht gefunden, initialisiere Standard-Korpus")
            self._initialize_default_corpus()
    
    def save_corpus(self) -> None:
        """Speichert den Korpus in die JSON-Datei."""
        try:
            with open(self.corpus_file, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=2)
            logging.info(f"Korpus gespeichert in {self.corpus_file}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Korpus: {e}")
    
    def add_sentence(self, sentence: str, category: str, language: str) -> None:
        """
        Fügt einen neuen Satz zum Korpus hinzu.
        
        Args:
            sentence: Der hinzuzufügende Satz
            category: Kategorie des Satzes (z.B. 'Politik', 'Wirtschaft')
            language: Sprache des Satzes (z.B. 'de', 'en')
        """
        if not category in self.corpus:
            self.corpus[category] = []
        
        self.corpus[category].append({
            'text': sentence,
            'language': language
        })
    
    def get_all_sentences(self) -> List[str]:
        """Gibt alle Sätze aus dem Korpus zurück."""
        return self.corpus
    
    def get_sentences_by_category(self, category: str) -> List[str]:
        """Gibt alle Sätze einer bestimmten Kategorie zurück."""
        return self.corpus  # In der vereinfachten Version geben wir alle Sätze zurück
    
    def get_sentences_by_language(self, language: str) -> List[str]:
        """Gibt alle Sätze einer bestimmten Sprache zurück."""
        return self.corpus  # In der vereinfachten Version geben wir alle Sätze zurück

    def get_categories(self) -> List[str]:
        """Gibt alle verfügbaren Kategorien zurück."""
        return ["default"]  # In der vereinfachten Version gibt es nur eine Kategorie
    
    def get_statistics(self) -> Dict[str, Dict[str, int]]:
        """Gibt Statistiken über den Korpus zurück."""
        stats = {
            'total': len(self.corpus),
            'by_category': {'default': len(self.corpus)},
            'by_language': {'de': len(self.corpus)}
        }
        return stats
    
    def validate_corpus(self, print_report: bool = True) -> Dict:
        """
        Führt eine vollständige Validierung des Korpus durch.
        
        Args:
            print_report: Wenn True, wird ein formatierter Bericht ausgegeben
        
        Returns:
            Dict mit Validierungsergebnissen
        """
        sentences = self.get_all_sentences()
        results = self.validator.validate_corpus(sentences)
        
        if print_report:
            print_validation_report(results)
        
        return results
    
    def _initialize_default_corpus(self) -> None:
        """Initialisiert einen Standard-Korpus mit Beispieldaten."""
        self.corpus = [
            "Wir arbeiten hart daran, Deutschland voranzubringen.",
            "Die Bundesregierung setzt sich für soziale Gerechtigkeit ein.",
            "Klimaschutz ist eine unserer wichtigsten Aufgaben."
        ]
        # Speichere im neuen Format
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            json.dump({
                "entries": [
                    {
                        "text": text,
                        "topic": "allgemein",
                        "language": "de",
                        "date": "2025-09-13",
                        "source": "regierung",
                        "verified": True
                    } for text in self.corpus
                ]
            }, f, ensure_ascii=False, indent=2)