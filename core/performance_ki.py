#!/usr/bin/env python3
"""
Performance-optimierte Bundeskanzler-KI mit Caching und schnellerer Initialisierung
"""

import sys
import os
import json
import pickle
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor

# Dynamischer Pfad zum Projekt-Root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from core.rag_system import RAGSystem
import logging
from datetime import datetime

# Konfiguriere optimiertes Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class PerformanceOptimizedKI:
    """
    Performance-optimierte Bundeskanzler-KI mit intelligentem Caching
    """
    
    def __init__(self, enable_cache: bool = True):
        """Initialisiert die optimierte KI mit Caching"""
        start_time = time.time()
        print("⚡ Initialisiere Performance-Optimierte KI...")
        
        self.enable_cache = enable_cache
        self.cache_file = os.path.join(project_root, 'models', 'response_cache.pkl')
        self.stats_file = os.path.join(project_root, 'models', 'performance_stats.json')
        
        # Performance-Metriken
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        
        # Lade vorhandene Stats
        self._load_stats()
        
        # Lade Response-Cache
        self.response_cache = self._load_cache() if enable_cache else {}
        
        # Initialisiere RAG-System in separatem Thread für bessere Performance
        with ThreadPoolExecutor(max_workers=2) as executor:
            rag_future = executor.submit(self._init_rag_system)
            themes_future = executor.submit(self._init_themes)
            
            self.rag_system = rag_future.result()
            self.themen_keywords = themes_future.result()
        
        self.last_confidence = 0.0
        
        init_time = time.time() - start_time
        print(f"✅ Optimierte KI geladen in {init_time:.2f}s")
        
        # Update Stats
        if hasattr(self, 'stats'):
            self.stats['last_init_time'] = init_time
            self._save_stats()
    
    def _init_rag_system(self) -> RAGSystem:
        """Initialisiert RAG-System"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        corpus_path = os.path.join(project_root, 'data', 'corpus.json')
        return RAGSystem(corpus_path=corpus_path)
    
    def _init_themes(self) -> Dict[str, List[str]]:
        """Initialisiert Themen-Keywords"""
        return {
            'klima': ['klima', 'klimaschutz', 'klimaneutralität', 'energie', 'erneuerbar', 'kohle', 'co2', 'wind', 'solar'],
            'wirtschaft': ['wirtschaft', 'industrie', 'mittelstand', 'start-up', 'innovation', 'ki', 'wasserstoff', 'export'],
            'gesundheit': ['gesundheit', 'pflege', 'kranken', 'medizin', 'prävention', 'corona', 'impf'],
            'soziales': ['sozial', 'rente', 'kindergeld', 'armut', 'integration', 'migration', 'wohnen'],
            'bildung': ['bildung', 'schule', 'universität', 'ausbildung', 'lernen', 'forschung'],
            'digital': ['digital', 'technologie', 'internet', 'cybersicherheit', 'datenschutz', 'ai'],
            'außenpolitik': ['europa', 'eu', 'nato', 'international', 'diplomatie', 'handel'],
            'sicherheit': ['sicherheit', 'polizei', 'terror', 'kriminalität', 'bundeswehr']
        }
    
    def _load_cache(self) -> Dict:
        """Lädt Response-Cache"""
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}
    
    def _save_cache(self):
        """Speichert Response-Cache"""
        if not self.enable_cache:
            return
        
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.response_cache, f)
        except Exception as e:
            logging.warning(f"Cache speichern fehlgeschlagen: {e}")
    
    def _load_stats(self):
        """Lädt Performance-Statistiken"""
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_stats(self):
        """Speichert Performance-Statistiken"""
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Stats speichern fehlgeschlagen: {e}")
    
    @lru_cache(maxsize=128)
    def erkenne_thema(self, frage: str) -> str:
        """Erkennt das Hauptthema einer Frage (cached)"""
        frage_lower = frage.lower()
        
        thema_scores = {}
        for thema, keywords in self.themen_keywords.items():
            score = sum(1 for keyword in keywords if keyword in frage_lower)
            if score > 0:
                thema_scores[thema] = score
        
        if thema_scores:
            return max(thema_scores, key=thema_scores.get)
        return 'allgemein'
    
    def _get_cache_key(self, frage: str) -> str:
        """Erstellt Cache-Schlüssel für Frage"""
        return frage.lower().strip()
    
    def antwort(self, frage: str) -> str:
        """
        Generiert optimierte Antwort mit Caching
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Cache-Check
        cache_key = self._get_cache_key(frage)
        if self.enable_cache and cache_key in self.response_cache:
            self.stats['cache_hits'] += 1
            cached_response = self.response_cache[cache_key]
            self.last_confidence = cached_response.get('confidence', 0.0)
            
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)
            
            logging.info(f"Cache-Hit für Frage: {frage[:50]}...")
            return cached_response['answer']
        
        # Cache-Miss - generiere neue Antwort
        self.stats['cache_misses'] += 1
        
        try:
            # RAG-Retrieval
            docs = self.rag_system.retrieve_relevant_documents(frage, top_k=5)
            
            if not docs:
                response = "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer Frage finden."
                self.last_confidence = 0.0
            else:
                # Themen-basierte Filterung
                thema = self.erkenne_thema(frage)
                
                # Filtere nach Thema wenn möglich
                if thema != 'allgemein':
                    thema_docs = [doc for doc in docs 
                                if any(keyword in doc['text'].lower() 
                                     for keyword in self.themen_keywords[thema])]
                    if thema_docs:
                        docs = thema_docs
                
                # Beste Antwort auswählen
                beste_doc = docs[0]
                response = beste_doc['text']
                self.last_confidence = beste_doc['score']
            
            # Cache aktualisieren
            if self.enable_cache:
                self.response_cache[cache_key] = {
                    'answer': response,
                    'confidence': self.last_confidence,
                    'timestamp': datetime.now().isoformat(),
                    'theme': thema if 'thema' in locals() else 'unknown'
                }
                
                # Cache periodisch speichern (alle 10 Anfragen)
                if self.stats['total_queries'] % 10 == 0:
                    self._save_cache()
            
        except Exception as e:
            logging.error(f"Fehler bei Antwort-Generierung: {e}")
            response = f"Es tut mir leid, bei der Verarbeitung Ihrer Frage ist ein Fehler aufgetreten: {str(e)}"
            self.last_confidence = 0.0
        
        # Performance-Metriken aktualisieren
        response_time = time.time() - start_time
        self._update_avg_response_time(response_time)
        
        # Stats periodisch speichern
        if self.stats['total_queries'] % 5 == 0:
            self._save_stats()
        
        return response
    
    def _update_avg_response_time(self, response_time: float):
        """Aktualisiert durchschnittliche Antwortzeit"""
        current_avg = self.stats['avg_response_time']
        total_queries = self.stats['total_queries']
        
        if total_queries == 1:
            self.stats['avg_response_time'] = response_time
        else:
            self.stats['avg_response_time'] = (current_avg * (total_queries - 1) + response_time) / total_queries
    
    def get_performance_stats(self) -> Dict:
        """Gibt Performance-Statistiken zurück"""
        cache_hit_rate = 0.0
        if self.stats['total_queries'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_queries'] * 100
        
        return {
            **self.stats,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'cache_size': len(self.response_cache),
            'memory_usage': f"{sys.getsizeof(self.response_cache) / 1024:.1f} KB"
        }
    
    def clear_cache(self):
        """Leert den Response-Cache"""
        self.response_cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logging.info("Response-Cache geleert")
    
    def optimize_cache(self, max_size: int = 1000):
        """Optimiert Cache-Größe durch Entfernen alter Einträge"""
        if len(self.response_cache) <= max_size:
            return
        
        # Sortiere nach Timestamp und behalte neueste
        items = list(self.response_cache.items())
        items.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
        
        self.response_cache = dict(items[:max_size])
        self._save_cache()
        
        logging.info(f"Cache auf {max_size} Einträge optimiert")


# Alias für Kompatibilität
VerbesserteBundeskanzlerKI = PerformanceOptimizedKI

if __name__ == "__main__":
    # Performance-Test
    ki = PerformanceOptimizedKI()
    
    test_fragen = [
        "Was ist die Klimapolitik der Bundesregierung?",
        "Wie steht es um den Kohleausstieg in Deutschland?",
        "Welche Wirtschaftspolitik verfolgt die Regierung?"
    ]
    
    print("\n🧪 Performance-Test...")
    print("="*50)
    
    for i, frage in enumerate(test_fragen, 1):
        start = time.time()
        antwort = ki.antwort(frage)
        end = time.time()
        
        print(f"\n{i}. Frage: {frage}")
        print(f"   Antwort: {antwort[:100]}...")
        print(f"   Zeit: {end-start:.3f}s | Konfidenz: {ki.last_confidence:.1%}")
    
    print(f"\n📊 Performance-Statistiken:")
    stats = ki.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")