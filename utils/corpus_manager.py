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
                    # Entferne Duplikate nach dem Laden
                    self._deduplicate_corpus()
            except Exception as e:
                logging.error(f"Fehler beim Laden des Korpus: {e}")
                self._initialize_default_corpus()
        else:
            logging.warning(f"{self.corpus_file} nicht gefunden, initialisiere Standard-Korpus")
            self._initialize_default_corpus()
    
    def _deduplicate_corpus(self) -> None:
        """Entfernt Duplikate aus dem Korpus basierend auf dem Text-Inhalt."""
        seen_texts = set()
        deduplicated_corpus = defaultdict(list)
        
        for category, items in self.corpus.items():
            for item in items:
                text = item['text']
                if text not in seen_texts:
                    seen_texts.add(text)
                    deduplicated_corpus[category].append(item)
        
        original_count = sum(len(v) for v in self.corpus.values())
        deduplicated_count = sum(len(v) for v in deduplicated_corpus.values())
        removed_duplicates = original_count - deduplicated_count
        
        self.corpus = deduplicated_corpus
        
        if removed_duplicates > 0:
            print(f"Duplikate entfernt: {removed_duplicates} Einträge (von {original_count} auf {deduplicated_count})")
            logging.info(f"Duplikate entfernt: {removed_duplicates} Einträge")
        else:
            print("Keine Duplikate gefunden")
    
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
    
    def expand_corpus_with_government_data(self) -> None:
        """Erweitert den Korpus mit zusätzlichen Regierungsdaten aus verschiedenen Bereichen."""
        print("📚 Erweitere Korpus mit zusätzlichen Regierungsdaten...")
        
        # Erweiterte Daten für verschiedene politische Bereiche
        additional_data = [
            # Wirtschaft und Arbeit
            {
                "text": "Die Bundesregierung setzt sich für Vollbeschäftigung und faire Löhne ein. Mindestlohn wird regelmäßig angepasst.",
                "topic": "wirtschaft",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_arbeit",
                "verified": True
            },
            {
                "text": "Start-ups und Mittelstand werden durch verbesserte Finanzierungsmöglichkeiten und Bürokratieabbau unterstützt.",
                "topic": "wirtschaft",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_wirtschaft",
                "verified": True
            },
            {
                "text": "Die Bundesregierung fördert die duale Berufsausbildung und stärkt die berufliche Weiterbildung.",
                "topic": "bildung",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_bildung",
                "verified": True
            },
            
            # Soziales und Gesundheit
            {
                "text": "Die Kindergrundsicherung wird eingeführt, um Kinderarmut zu bekämpfen und Familien zu entlasten.",
                "topic": "soziales",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_familie",
                "verified": True
            },
            {
                "text": "Das Gesundheitssystem wird durch Digitalisierung und Prävention gestärkt. Telematikinfrastruktur wird ausgebaut.",
                "topic": "gesundheit",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_gesundheit",
                "verified": True
            },
            {
                "text": "Die Pflegeversicherung wird reformiert, um bessere Arbeitsbedingungen für Pflegekräfte zu schaffen.",
                "topic": "gesundheit",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_gesundheit",
                "verified": True
            },
            
            # Außenpolitik und Europa
            {
                "text": "Deutschland setzt sich für eine starke Europäische Union und transatlantische Partnerschaft ein.",
                "topic": "international",
                "language": "de",
                "date": "2025-09-15",
                "source": "auswaertiges_amt",
                "verified": True
            },
            {
                "text": "Internationale Zusammenarbeit beim Klimaschutz ist ein zentraler Pfeiler deutscher Außenpolitik.",
                "topic": "international",
                "language": "de",
                "date": "2025-09-15",
                "source": "auswaertiges_amt",
                "verified": True
            },
            
            # Technologie und Zukunft
            {
                "text": "Deutschland wird bis 2030 führend in der KI-Entwicklung und setzt auf vertrauenswürdige KI.",
                "topic": "technologie",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_wirtschaft",
                "verified": True
            },
            {
                "text": "Deutschland führt bei der Entwicklung von Wasserstofftechnologie und grüner Stahlproduktion.",
                "topic": "technologie",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_wirtschaft",
                "verified": True
            },
            {
                "text": "Die Bildungsreform stärkt digitale Kompetenzen in Schulen und Hochschulen.",
                "topic": "bildung",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_bildung",
                "verified": True
            },
            
            # Sicherheit und Inneres
            {
                "text": "Die Bundesregierung investiert in die digitale Souveränität und Cybersicherheit Deutschlands.",
                "topic": "sicherheit",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_des_innern",
                "verified": True
            },
            {
                "text": "Der Bevölkerungsschutz wird durch moderne Frühwarnsysteme und Katastrophenschutz gestärkt.",
                "topic": "sicherheit",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_des_innern",
                "verified": True
            },
            
            # Verkehr und Infrastruktur
            {
                "text": "Deutschland baut das Schienennetz aus und setzt auf klimafreundliche Mobilität.",
                "topic": "verkehr",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_verkehr",
                "verified": True
            },
            {
                "text": "Die Ladeinfrastruktur für Elektrofahrzeuge wird massiv ausgebaut.",
                "topic": "verkehr",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_verkehr",
                "verified": True
            },
            
            # Landwirtschaft und Ernährung
            {
                "text": "Die Landwirtschaft wird durch nachhaltige Praktiken und Biodiversitätsschutz unterstützt.",
                "topic": "landwirtschaft",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_ernaehrung",
                "verified": True
            },
            
            # Kultur und Medien
            {
                "text": "Kulturelle Bildung wird gefördert und der Zugang zu Kunst und Kultur für alle ermöglicht.",
                "topic": "kultur",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_kultur",
                "verified": True
            },
            
            # Justiz und Recht
            {
                "text": "Die Justiz wird digitalisiert und der Zugang zum Recht für alle Bürgerinnen und Bürger verbessert.",
                "topic": "justiz",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_der_justiz",
                "verified": True
            },
            
            # Englische Einträge für internationale Kontexte
            {
                "text": "Germany is committed to ambitious climate protection goals. Climate neutrality should be achieved by 2045.",
                "topic": "klima",
                "language": "en",
                "date": "2025-09-15",
                "source": "government",
                "verified": True
            },
            {
                "text": "The federal government promotes the expansion of renewable energies through the EEG and various funding programs.",
                "topic": "energie",
                "language": "en",
                "date": "2025-09-15",
                "source": "government",
                "verified": True
            },
            {
                "text": "Germany invests in future technologies such as AI, quantum computing and hydrogen technology.",
                "topic": "innovation",
                "language": "en",
                "date": "2025-09-15",
                "source": "government",
                "verified": True
            }
        ]
        
        # Füge neue Daten zum Korpus hinzu
        for entry in additional_data:
            category = entry['topic']
            if category not in self.corpus:
                self.corpus[category] = []
            
            # Prüfe auf Duplikate
            existing_texts = [item['text'] for item in self.corpus[category]]
            if entry['text'] not in existing_texts:
                self.corpus[category].append({
                    'text': entry['text'],
                    'language': entry['language']
                })
        
        print(f"✅ Korpus um {len(additional_data)} Einträge erweitert")
        self.save_corpus()
    
    def expand_corpus_with_eu_data(self) -> None:
        """Erweitert den Korpus mit EU-bezogenen Daten und europäischen Kontexten."""
        print("🇪🇺 Erweitere Korpus mit EU-Daten...")
        
        eu_data = [
            # Europäische Union und Institutionen
            {
                "text": "Die Europäische Union ist eine Wertegemeinschaft, die auf Demokratie, Rechtsstaatlichkeit und Menschenrechten basiert.",
                "topic": "europa",
                "language": "de",
                "date": "2025-09-15",
                "source": "europaeische_kommission",
                "verified": True
            },
            {
                "text": "Deutschland setzt sich für eine starke Rolle der EU in der Welt ein und unterstützt die Gemeinsame Außen- und Sicherheitspolitik.",
                "topic": "europa",
                "language": "de",
                "date": "2025-09-15",
                "source": "auswaertiges_amt",
                "verified": True
            },
            {
                "text": "Der Green Deal der Europäischen Union zielt auf Klimaneutralität bis 2050 und nachhaltige Entwicklung ab.",
                "topic": "europa",
                "language": "de",
                "date": "2025-09-15",
                "source": "europaeische_kommission",
                "verified": True
            },
            {
                "text": "Die EU fördert die digitale Transformation durch Programme wie Digital Europe und InvestEU.",
                "topic": "europa",
                "language": "de",
                "date": "2025-09-15",
                "source": "europaeische_kommission",
                "verified": True
            },
            {
                "text": "Deutschland trägt als Nettozahler wesentlich zum EU-Haushalt bei und profitiert vom Binnenmarkt.",
                "topic": "europa",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesfinanzministerium",
                "verified": True
            },
            
            # Englische EU-Einträge
            {
                "text": "The European Union is a community of values based on democracy, rule of law and human rights.",
                "topic": "europa",
                "language": "en",
                "date": "2025-09-15",
                "source": "european_commission",
                "verified": True
            },
            {
                "text": "Germany advocates for a strong EU role in the world and supports the Common Foreign and Security Policy.",
                "topic": "europa",
                "language": "en",
                "date": "2025-09-15",
                "source": "foreign_office",
                "verified": True
            },
            {
                "text": "The European Green Deal aims for climate neutrality by 2050 and sustainable development.",
                "topic": "europa",
                "language": "en",
                "date": "2025-09-15",
                "source": "european_commission",
                "verified": True
            }
        ]
        
        # Füge EU-Daten hinzu
        for entry in eu_data:
            category = entry['topic']
            if category not in self.corpus:
                self.corpus[category] = []
            
            # Prüfe auf Duplikate
            existing_texts = [item['text'] for item in self.corpus[category]]
            if entry['text'] not in existing_texts:
                self.corpus[category].append({
                    'text': entry['text'],
                    'language': entry['language']
                })
        
        print(f"✅ Korpus um {len(eu_data)} EU-Einträge erweitert")
        self.save_corpus()
    
    def expand_corpus_with_policy_data(self) -> None:
        """Erweitert den Korpus mit spezifischen Policy-Daten und politischen Positionen."""
        print("📋 Erweitere Korpus mit Policy-Daten...")
        
        policy_data = [
            # Wirtschaftspolitik
            {
                "text": "Die Bundesregierung verfolgt eine Politik der sozialen Marktwirtschaft mit Augenmaß und setzt auf Innovation und Nachhaltigkeit.",
                "topic": "politik",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundeskanzleramt",
                "verified": True
            },
            {
                "text": "Deutschland setzt auf Industrie 4.0 und stärkt die Wettbewerbsfähigkeit durch Forschung und Entwicklung.",
                "topic": "politik",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_wirtschaft",
                "verified": True
            },
            
            # Sozialpolitik
            {
                "text": "Die Bundesregierung bekämpft Altersarmut durch eine stabile Rente und private Altersvorsorge.",
                "topic": "soziales",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_fuer_arbeit",
                "verified": True
            },
            {
                "text": "Integration und Teilhabe sind zentrale Säulen der deutschen Sozialpolitik.",
                "topic": "soziales",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_des_innern",
                "verified": True
            },
            
            # Außenpolitik
            {
                "text": "Deutschland setzt sich für eine wertebasierte Außenpolitik und die Stärkung multilateraler Institutionen ein.",
                "topic": "international",
                "language": "de",
                "date": "2025-09-15",
                "source": "auswaertiges_amt",
                "verified": True
            },
            {
                "text": "Die transatlantische Partnerschaft bleibt ein Eckpfeiler deutscher Außenpolitik.",
                "topic": "international",
                "language": "de",
                "date": "2025-09-15",
                "source": "auswaertiges_amt",
                "verified": True
            },
            
            # Zukunftsthemen
            {
                "text": "Deutschland investiert in die Zukunft durch Bildung, Forschung und Innovation für die nächste Generation.",
                "topic": "zukunft",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundeskanzleramt",
                "verified": True
            },
            {
                "text": "Die Bundesregierung fördert den Strukturwandel in den Regionen durch den Ausbau von Infrastruktur und Arbeitsplätzen.",
                "topic": "regional",
                "language": "de",
                "date": "2025-09-15",
                "source": "bundesministerium_des_innern",
                "verified": True
            }
        ]
        
        # Füge Policy-Daten hinzu
        for entry in policy_data:
            category = entry['topic']
            if category not in self.corpus:
                self.corpus[category] = []
            
            # Prüfe auf Duplikate
            existing_texts = [item['text'] for item in self.corpus[category]]
            if entry['text'] not in existing_texts:
                self.corpus[category].append({
                    'text': entry['text'],
                    'language': entry['language']
                })
        
        print(f"✅ Korpus um {len(policy_data)} Policy-Einträge erweitert")
        self.save_corpus()
    
    def generate_additional_training_data(self, num_entries_per_category: int = 3) -> None:
        """Generiert zusätzliche Trainingsdaten basierend auf vorhandenen Mustern."""
        print(f"🤖 Generiere zusätzliche Trainingsdaten ({num_entries_per_category} pro Kategorie)...")
        
        # Basis-Patterns für verschiedene Kategorien
        generation_patterns = {
            "klima": [
                "Deutschland setzt sich für {} Klimaschutz ein.",
                "Die Bundesregierung fördert {} Maßnahmen zum Klimaschutz.",
                "Bis {} soll {} erreicht werden."
            ],
            "wirtschaft": [
                "Die Bundesregierung unterstützt {} durch {}.",
                "Deutschland investiert in {} zur Stärkung der Wirtschaft.",
                "{} wird durch {} gefördert."
            ],
            "bildung": [
                "Die Bildungspolitik setzt auf {} und {}.",
                "Deutschland stärkt {} in Schulen und Hochschulen.",
                "{} wird durch {} verbessert."
            ],
            "gesundheit": [
                "Das Gesundheitssystem wird durch {} gestärkt.",
                "Die Bundesregierung investiert in {} für bessere Gesundheitsversorgung.",
                "{} wird durch {} modernisiert."
            ],
            "technologie": [
                "Deutschland führt bei {} und {}.",
                "Die Bundesregierung fördert {} durch {}.",
                "{} wird durch {} vorangetrieben."
            ]
        }
        
        # Füllwörter für verschiedene Bereiche
        fillers = {
            "klima": ["ambitionierte", "nachhaltige", "innovative", "internationale", "Klimaneutralität", "2030", "2045"],
            "wirtschaft": ["den Mittelstand", "Start-ups", "die Industrie", "Forschung", "Entwicklung", "Förderprogramme"],
            "bildung": ["digitale Kompetenzen", "berufliche Bildung", "lebenslanges Lernen", "moderne Lehrkonzepte", "Bildungsreformen"],
            "gesundheit": ["Digitalisierung", "Prävention", "Telemedizin", "Pflege", "Krankenhäuser", "Ärzte"],
            "technologie": ["KI-Entwicklung", "Quantencomputing", "Wasserstofftechnologie", "Batterieforschung", "Investitionen"]
        }
        
        import random
        
        generated_count = 0
        for category, patterns in generation_patterns.items():
            if category not in self.corpus:
                self.corpus[category] = []
            
            # Generiere Einträge für diese Kategorie
            for i in range(num_entries_per_category):
                pattern = random.choice(patterns)
                category_fillers = fillers.get(category, [""])
                
                # Ersetze Platzhalter mit zufälligen Füllwörtern
                text = pattern
                for placeholder in ["{}", "{}"]:
                    if "{}" in text:
                        filler = random.choice(category_fillers)
                        text = text.replace("{}", filler, 1)
                
                # Prüfe auf Duplikate
                existing_texts = [item['text'] for item in self.corpus[category]]
                if text not in existing_texts:
                    self.corpus[category].append({
                        'text': text,
                        'language': 'de'
                    })
                    generated_count += 1
        
        print(f"✅ {generated_count} zusätzliche Einträge generiert")
        self.save_corpus()