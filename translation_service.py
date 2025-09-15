"""
Neuronaler Übersetzungsdienst für die Bundeskanzler-KI basierend auf MarianMT.
"""

from transformers import MarianMTModel, MarianTokenizer
import torch
import os
import time
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Übersetzungsdienst basierend auf MarianMT für mehrsprachige Kommunikation.
    Unterstützt Übersetzungen zwischen Deutsch, Englisch, Französisch, Italienisch und Spanisch.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialisiert den Übersetzungsdienst.
        
        Args:
            cache_dir: Optional. Verzeichnis zum Zwischenspeichern der Modelle.
        """
        self.cache_dir = cache_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sprachpaar zu Modell-Mapping
        self.model_map = {
            ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
            ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
            ('fr', 'de'): 'Helsinki-NLP/opus-mt-fr-de',
            ('de', 'fr'): 'Helsinki-NLP/opus-mt-de-fr',
            ('it', 'de'): 'Helsinki-NLP/opus-mt-it-de',
            ('de', 'it'): 'Helsinki-NLP/opus-mt-de-it',
            ('es', 'de'): 'Helsinki-NLP/opus-mt-es-de',
            ('de', 'es'): 'Helsinki-NLP/opus-mt-de-es'
        }
        
        self.loaded_models: Dict[Tuple[str, str], Tuple[MarianMTModel, MarianTokenizer]] = {}
        logger.info(f"Übersetzungsdienst initialisiert. Verwende Gerät: {self.device}")
    
    def _load_model(self, source_lang: str, target_lang: str) -> Tuple[MarianMTModel, MarianTokenizer]:
        """
        Lädt ein Übersetzungsmodell für ein spezifisches Sprachpaar.
        
        Args:
            source_lang: Quellsprache (ISO-Code)
            target_lang: Zielsprache (ISO-Code)
            
        Returns:
            Tuple aus Modell und Tokenizer
        """
        model_key = (source_lang, target_lang)
        if model_key not in self.model_map:
            raise ValueError(f"Keine Unterstützung für Übersetzung von {source_lang} nach {target_lang}")
        
        if model_key not in self.loaded_models:
            model_name = self.model_map[model_key]
            logger.info(f"Lade Modell {model_name}...")
            
            try:
                tokenizer = MarianTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                model = MarianMTModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                ).to(self.device)
                
                self.loaded_models[model_key] = (model, tokenizer)
                logger.info(f"Modell {model_name} erfolgreich geladen")
                
            except Exception as e:
                logger.error(f"Fehler beim Laden des Modells {model_name}: {str(e)}")
                raise
        
        return self.loaded_models[model_key]
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        max_length: int = 512
    ) -> str:
        """
        Übersetzt einen Text von einer Sprache in eine andere.
        
        Args:
            text: Zu übersetzender Text
            source_lang: Quellsprache (ISO-Code)
            target_lang: Zielsprache (ISO-Code)
            batch_size: Größe der Batch-Verarbeitung
            max_length: Maximale Sequenzlänge
            
        Returns:
            Übersetzter Text
        """
        if not text.strip():
            return ""
            
        try:
            model, tokenizer = self._load_model(source_lang, target_lang)
            
            # Teile langen Text in Sätze
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
            
            translated_batches = []
            start_time = time.time()
            
            for batch in batches:
                # Tokenisiere Eingabe
                encoded = tokenizer.prepare_seq2seq_batch(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Generiere Übersetzung
                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        max_length=max_length,
                        num_beams=4,
                        length_penalty=0.6,
                        early_stopping=True
                    )
                
                # Dekodiere Ausgabe
                translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translated_batches.extend(translated)
            
            duration = time.time() - start_time
            logger.info(f"Übersetzung abgeschlossen in {duration:.2f} Sekunden")
            
            return '. '.join(translated_batches)
            
        except Exception as e:
            logger.error(f"Fehler bei der Übersetzung: {str(e)}")
            raise

    def unload_models(self):
        """Entlädt alle geladenen Modelle um Speicher freizugeben."""
        self.loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Alle Modelle entladen")


# Beispielverwendung
if __name__ == '__main__':
    translator = TranslationService()
    
    test_texts = {
        'de': "Die künstliche Intelligenz verändert die Art und Weise, wie wir arbeiten und leben.",
        'en': "Artificial intelligence is changing the way we work and live.",
        'fr': "L'intelligence artificielle change notre façon de travailler et de vivre.",
        'it': "L'intelligenza artificiale sta cambiando il nostro modo di lavorare e vivere.",
        'es': "La inteligencia artificial está cambiando nuestra forma de trabajar y vivir."
    }
    
    # Teste Übersetzungen nach Deutsch
    print("\nÜbersetzungen nach Deutsch:")
    for source_lang, text in test_texts.items():
        if source_lang != 'de':
            try:
                translation = translator.translate(text, source_lang, 'de')
                print(f"\nOriginal ({source_lang}): {text}")
                print(f"Übersetzung (de): {translation}")
            except Exception as e:
                print(f"Fehler bei {source_lang}-de: {str(e)}")
    
    # Teste Übersetzungen von Deutsch
    print("\nÜbersetzungen von Deutsch:")
    german_text = test_texts['de']
    for target_lang in ['en', 'fr', 'it', 'es']:
        try:
            translation = translator.translate(german_text, 'de', target_lang)
            print(f"\nOriginal (de): {german_text}")
            print(f"Übersetzung ({target_lang}): {translation}")
        except Exception as e:
            print(f"Fehler bei de-{target_lang}: {str(e)}")
    
    # Speicher freigeben
    translator.unload_models()