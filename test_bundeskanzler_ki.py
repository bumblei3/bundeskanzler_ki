"""
Integrationstest für die Bundeskanzler-KI mit allen Komponenten.
"""

import torch
from pattern_language_detector import PatternLanguageDetector
from translation_service import TranslationService
from transfer_learning import TransferLearningModel
import logging
from typing import List, Dict, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BundeskanzlerKITester:
    """Testet alle Komponenten der Bundeskanzler-KI."""
    
    def __init__(self):
        """Initialisiert alle Komponenten für den Test."""
        logger.info("Initialisiere Testumgebung...")
        
        self.language_detector = PatternLanguageDetector()
        self.translator = TranslationService()
        self.transfer_model = TransferLearningModel(num_labels=4)
        
        # Themenkategorien für Tests
        self.politik_kategorien = [
            "Außenpolitik",
            "Wirtschaft",
            "Klimaschutz",
            "Digitalisierung",
            "Soziales",
            "Gesundheit"
        ]
        
        # Testfälle für verschiedene Sprachen
        self.test_inputs = {
            'de': [
                "Wie steht die Bundesregierung zum Ausbau erneuerbarer Energien?",
                "Was sind die aktuellen Maßnahmen zur Digitalisierung der Verwaltung?",
                "Welche Position vertritt Deutschland in der EU-Außenpolitik?"
            ],
            'en': [
                "What is the German government's position on renewable energy?",
                "What are the current measures for digitalizing administration?",
                "What position does Germany take in EU foreign policy?"
            ],
            'fr': [
                "Quelle est la position du gouvernement allemand sur les énergies renouvelables?",
                "Quelles sont les mesures actuelles pour la numérisation de l'administration?",
                "Quelle position l'Allemagne adopte-t-elle dans la politique étrangère de l'UE?"
            ]
        }
        
        logger.info("Testumgebung initialisiert")
    
    def test_language_detection(self) -> None:
        """Testet die Spracherkennung mit verschiedenen Eingaben."""
        logger.info("\n=== Teste Spracherkennung ===")
        
        for sprache, texte in self.test_inputs.items():
            print(f"\nTests für {sprache}:")
            for text in texte:
                erkannte_sprache, konfidenz = self.language_detector.detect_language(text)
                print(f"\nText: {text}")
                print(f"Erwartete Sprache: {sprache}")
                print(f"Erkannte Sprache: {erkannte_sprache}")
                print(f"Konfidenz: {konfidenz:.2f}")
    
    def test_translation(self) -> None:
        """Testet die Übersetzungsfunktionen."""
        logger.info("\n=== Teste Übersetzungen ===")
        
        # Test: Übersetzung in alle unterstützten Sprachen
        de_text = self.test_inputs['de'][0]
        zielsprachen = ['en', 'fr', 'it', 'es']
        
        print(f"\nAusgangstext (de): {de_text}")
        for ziel in zielsprachen:
            übersetzung = self.translator.translate(de_text, 'de', ziel)
            print(f"\nÜbersetzung ({ziel}): {übersetzung}")
            
            # Rückübersetzung zum Vergleich
            rück = self.translator.translate(übersetzung, ziel, 'de')
            print(f"Rückübersetzung (de): {rück}")
    
    def test_zero_shot(self) -> None:
        """Testet Zero-shot Klassifikation."""
        logger.info("\n=== Teste Zero-shot Klassifikation ===")
        
        test_text = """
        Die Bundesregierung plant massive Investitionen in den Klimaschutz
        und will Deutschland bis 2045 klimaneutral machen. Dafür sollen
        erneuerbare Energien stark ausgebaut und die Industrie bei der
        Transformation unterstützt werden.
        """
        
        ergebnisse = self.transfer_model.zero_shot_classify(
            test_text,
            self.politik_kategorien
        )
        
        print(f"\nAnalysierter Text:\n{test_text}\n")
        print("Thematische Einordnung:")
        for kategorie, score in ergebnisse:
            print(f"- {kategorie}: {score:.2f}")
    
    def test_few_shot(self) -> None:
        """Testet Few-shot Learning."""
        logger.info("\n=== Teste Few-shot Learning ===")
        
        # Beispiele für verschiedene Politikbereiche
        support_examples = [
            ("Deutschland setzt sich für stärkere EU-Integration ein.", 0),  # Außenpolitik
            ("Die Wirtschaft erholt sich nach der Krise.", 1),              # Wirtschaft
            ("Neue Solarpflicht für Gebäude beschlossen.", 2),             # Klimaschutz
            ("Verwaltung wird durch KI effizienter.", 3),                  # Digitalisierung
        ]
        
        test_queries = [
            "Bundeskanzler trifft EU-Kommissionspräsidentin in Brüssel.",
            "Inflation sinkt auf niedrigsten Stand seit zwei Jahren.",
            "Offshore-Windparks werden massiv ausgebaut.",
            "Behördengänge künftig komplett online möglich."
        ]
        
        print("\nKlassifikation von Testtexten:")
        for query in test_queries:
            klasse, konf = self.transfer_model.few_shot_classify(
                query,
                support_examples
            )
            print(f"\nText: {query}")
            print(f"Erkannte Kategorie: {klasse}")
            print(f"Konfidenz: {konf:.2f}")
    
    def run_all_tests(self) -> None:
        """Führt alle Tests nacheinander aus."""
        start_time = time.time()
        
        try:
            self.test_language_detection()
            self.test_translation()
            self.test_zero_shot()
            self.test_few_shot()
            
            duration = time.time() - start_time
            logger.info(f"\nAlle Tests abgeschlossen in {duration:.2f} Sekunden")
            
        except Exception as e:
            logger.error(f"Fehler während der Tests: {str(e)}")
        finally:
            # Aufräumen
            self.translator.unload_models()


if __name__ == '__main__':
    tester = BundeskanzlerKITester()
    tester.run_all_tests()
