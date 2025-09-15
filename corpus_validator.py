"""
Validator für den Trainingskorpus der Bundeskanzler-KI.
Prüft Duplikate, Qualität und Konsistenz der Sätze.
"""
import re
from typing import List, Dict, Set, Tuple
from collections import Counter
import logging
from language_detection import detect_language, get_supported_languages

class CorpusValidator:
    def __init__(self, min_length: int = 20, max_length: int = 200):
        """
        Initialisiert den Korpus-Validator.
        
        Args:
            min_length: Minimale Satzlänge
            max_length: Maximale Satzlänge
        """
        self.min_length = min_length
        self.max_length = max_length
        self.supported_languages = get_supported_languages()
    
    def validate_sentence(self, sentence: str) -> Tuple[bool, List[str]]:
        """
        Validiert einen einzelnen Satz.
        
        Returns:
            Tuple aus (ist_valid, liste_der_probleme)
        """
        problems = []
        
        # Längenprüfung
        if len(sentence) < self.min_length:
            problems.append(f"Satz zu kurz (min. {self.min_length} Zeichen)")
        elif len(sentence) > self.max_length:
            problems.append(f"Satz zu lang (max. {self.max_length} Zeichen)")
        
        # Grundlegende Formatierung
        if not sentence[0].isupper():
            problems.append("Satz beginnt nicht mit Großbuchstaben")
        if not sentence[-1] in '.!?':
            problems.append("Satz endet nicht mit Satzzeichen")
        
        # Sprachprüfung
        lang = detect_language(sentence)
        if lang == 'unknown':
            problems.append("Sprache konnte nicht zuverlässig erkannt werden")
        elif lang not in self.supported_languages:
            problems.append(f"Nicht unterstützte Sprache: {lang}")
        
        return len(problems) == 0, problems
    
    def find_duplicates(self, sentences: List[str]) -> Dict[str, List[int]]:
        """Findet Duplikate im Korpus."""
        seen = {}
        duplicates = {}
        
        for idx, sentence in enumerate(sentences):
            norm_sentence = self._normalize_sentence(sentence)
            if norm_sentence in seen:
                if norm_sentence not in duplicates:
                    duplicates[norm_sentence] = [seen[norm_sentence]]
                duplicates[norm_sentence].append(idx)
            else:
                seen[norm_sentence] = idx
        
        return duplicates
    
    def find_similar_sentences(self, sentences: List[str], threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Findet ähnliche Sätze im Korpus.
        Verwendet Levenshtein-Distanz für den Vergleich.
        """
        from difflib import SequenceMatcher
        similar = []
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = SequenceMatcher(None, 
                    self._normalize_sentence(sentences[i]),
                    self._normalize_sentence(sentences[j])
                ).ratio()
                
                if similarity >= threshold:
                    similar.append((i, j, similarity))
        
        return similar
    
    def check_language_consistency(self, sentences: List[str]) -> Dict[str, List[int]]:
        """Prüft die Sprachkonsistenz der Sätze."""
        inconsistencies = {}
        
        for idx, sentence in enumerate(sentences):
            try:
                lang = detect_language(sentence)
                if lang not in self.supported_languages:
                    if lang not in inconsistencies:
                        inconsistencies[lang] = []
                    inconsistencies[lang].append(idx)
            except Exception as e:
                if 'error' not in inconsistencies:
                    inconsistencies['error'] = []
                inconsistencies['error'].append(idx)
        
        return inconsistencies
    
    def validate_corpus(self, sentences: List[str]) -> Dict:
        """
        Führt eine vollständige Validierung des Korpus durch.
        
        Returns:
            Dict mit Validierungsergebnissen
        """
        results = {
            'total_sentences': len(sentences),
            'valid_sentences': 0,
            'invalid_sentences': [],
            'duplicates': {},
            'similar_sentences': [],
            'language_inconsistencies': {},
            'statistics': {
                'by_language': Counter(),
                'avg_length': 0,
                'problems': Counter()
            }
        }
        
        # Einzelsatzvalidierung
        total_length = 0
        for idx, sentence in enumerate(sentences):
            is_valid, problems = self.validate_sentence(sentence)
            if is_valid:
                results['valid_sentences'] += 1
            else:
                results['invalid_sentences'].append({
                    'index': idx,
                    'sentence': sentence,
                    'problems': problems
                })
                for problem in problems:
                    results['statistics']['problems'][problem] += 1
            
            # Sprachstatistik
            try:
                lang = detect_language(sentence)
                results['statistics']['by_language'][lang] += 1
            except:
                pass
            
            total_length += len(sentence)
        
        # Duplikate
        results['duplicates'] = self.find_duplicates(sentences)
        
        # Ähnliche Sätze
        results['similar_sentences'] = self.find_similar_sentences(sentences)
        
        # Sprachkonsistenz
        results['language_inconsistencies'] = self.check_language_consistency(sentences)
        
        # Durchschnittliche Länge
        if sentences:
            results['statistics']['avg_length'] = total_length / len(sentences)
        
        return results
    
    def _normalize_sentence(self, sentence: str) -> str:
        """Normalisiert einen Satz für Vergleiche."""
        return re.sub(r'\s+', ' ', sentence.lower().strip())

def print_validation_report(results: Dict) -> None:
    """Gibt einen formatierten Validierungsbericht aus."""
    logging.info("=== Korpus-Validierungsbericht ===")
    logging.info(f"Gesamtzahl Sätze: {results['total_sentences']}")
    logging.info(f"Valide Sätze: {results['valid_sentences']} ({results['valid_sentences']/results['total_sentences']*100:.1f}%)")
    
    if results['invalid_sentences']:
        logging.info("\nProblematische Sätze:")
        for invalid in results['invalid_sentences']:
            logging.info(f"  [{invalid['index']}] {invalid['sentence']}")
            for problem in invalid['problems']:
                logging.info(f"    - {problem}")
    
    if results['duplicates']:
        logging.info("\nGefundene Duplikate:")
        for sentence, indices in results['duplicates'].items():
            logging.info(f"  Satz: {sentence}")
            logging.info(f"  Gefunden an Positionen: {indices}")
    
    if results['similar_sentences']:
        logging.info("\nÄhnliche Sätze:")
        for i, j, similarity in results['similar_sentences']:
            logging.info(f"  Ähnlichkeit {similarity:.2f}:")
            logging.info(f"    1: {results['sentences'][i]}")
            logging.info(f"    2: {results['sentences'][j]}")
    
    if results['language_inconsistencies']:
        logging.info("\nSprachinkonsistenzen:")
        for lang, indices in results['language_inconsistencies'].items():
            logging.info(f"  {lang}: {len(indices)} Sätze")
    
    logging.info("\nStatistiken:")
    logging.info(f"  Durchschnittliche Satzlänge: {results['statistics']['avg_length']:.1f} Zeichen")
    logging.info("\n  Sprachen:")
    for lang, count in results['statistics']['by_language'].items():
        logging.info(f"    {lang}: {count} Sätze")
    
    if results['statistics']['problems']:
        logging.info("\n  Häufigste Probleme:")
        for problem, count in results['statistics']['problems'].most_common():
            logging.info(f"    {problem}: {count}x")