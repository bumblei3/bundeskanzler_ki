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
                    self.corpus = defaultdict(list)
                    if 'entries' in data:
                        print(f"Gefundene Einträge im Korpus: {len(data['entries'])}")
                        for entry in data['entries']:
                            if isinstance(entry, dict) and 'text' in entry:
                                cat = entry.get('topic', 'default')
                                self.corpus[cat].append({
                                    'text': entry['text'],
                                    'language': entry.get('language', 'de')
                                })
                                print(f"Hinzugefügter Text: {entry['text'][:50]}...")
                            else:
                                logging.warning(f"Ungültiger Eintrag im Korpus: {entry}")
                    print(f"Geladener Korpus enthält {sum(len(v) for v in self.corpus.values())} Einträge")
                    logging.info(f"Korpus geladen aus {self.corpus_file} mit {sum(len(v) for v in self.corpus.values())} Einträgen")
            except Exception as e:
                logging.error(f"Fehler beim Laden des Korpus: {e}")
                self._initialize_default_corpus()
        else:
            logging.warning(f"{self.corpus_file} nicht gefunden, initialisiere Standard-Korpus")
            self._initialize_default_corpus()
    
    def save_corpus(self) -> None:
        """Speichert den Korpus in die JSON-Datei (als Eintragsliste)."""
        try:
            entries = []
            for cat, items in self.corpus.items():
                for item in items:
                    entry = {
                        'text': item['text'],
                        'topic': cat,
                        'language': item.get('language', 'de'),
                        'date': '2025-09-13',
                        'source': 'regierung',
                        'verified': True
                    }
                    entries.append(entry)
            with open(self.corpus_file, 'w', encoding='utf-8') as f:
                json.dump({'entries': entries}, f, ensure_ascii=False, indent=2)
            logging.info(f"Korpus gespeichert in {self.corpus_file}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Korpus: {e}")
    
    def add_sentence(self, sentence: str, category: str, language: str) -> None:
        """
        Fügt einen neuen Satz zum Korpus hinzu.
        """
        if not category in self.corpus:
            self.corpus[category] = []
        self.corpus[category].append({
            'text': sentence,
            'language': language
        })
    
    def get_all_sentences(self) -> List[str]:
        """Gibt alle Sätze aus dem Korpus als Liste von Strings zurück."""
        result = []
        for items in self.corpus.values():
            for item in items:
                result.append(item['text'])
        return result
    
    def get_sentences_by_category(self, category: str) -> List[str]:
        """Gibt alle Sätze einer bestimmten Kategorie zurück."""
        return [item['text'] for item in self.corpus.get(category, [])]
    
    def get_sentences_by_language(self, language: str) -> List[str]:
        """Gibt alle Sätze einer bestimmten Sprache zurück."""
        result = []
        for items in self.corpus.values():
            for item in items:
                if item.get('language', 'de') == language:
                    result.append(item['text'])
        return result

    def get_categories(self) -> List[str]:
        """Gibt alle verfügbaren Kategorien zurück."""
        return list(self.corpus.keys())
    
    def get_statistics(self) -> Dict[str, Dict[str, int]]:
        """Gibt Statistiken über den Korpus zurück."""
        stats = {
            'total': sum(len(items) for items in self.corpus.values()),
            'by_category': {cat: len(items) for cat, items in self.corpus.items()},
            'by_language': {}
        }
        lang_count = {}
        for items in self.corpus.values():
            for item in items:
                lang = item.get('language', 'de')
                lang_count[lang] = lang_count.get(lang, 0) + 1
        stats['by_language'] = lang_count
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
        """Initialisiert einen Standard-Korpus mit Beispieldaten als Dictionary."""
        self.corpus = defaultdict(list)
        beispielsaetze = [
            "Wir arbeiten hart daran, Deutschland voranzubringen.",
            "Die Bundesregierung setzt sich für soziale Gerechtigkeit ein.",
            "Klimaschutz ist eine unserer wichtigsten Aufgaben."
        ]
        for text in beispielsaetze:
            self.corpus['allgemein'].append({'text': text, 'language': 'de'})
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
                    } for text in beispielsaetze
                ]
            }, f, ensure_ascii=False, indent=2)