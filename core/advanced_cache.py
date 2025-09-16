#!/usr/bin/env python3
"""
Advanced Cache System für Bundeskanzler-KI
Caching für Modelle, Embeddings und Responses
"""

import json
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib


class CacheManager:
    """Manager für verschiedene Cache-Typen"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.caches = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache(self, name: str) -> 'Cache':
        """Holt oder erstellt einen Cache"""
        if name not in self.caches:
            self.caches[name] = Cache(name, self.cache_dir)
        return self.caches[name]
    
    def get_stats(self) -> Dict[str, int]:
        """Gibt Cache-Statistiken zurück"""
        total_stats = self.stats.copy()
        for cache in self.caches.values():
            cache_stats = cache.get_stats()
            for key, value in cache_stats.items():
                total_stats[key] = total_stats.get(key, 0) + value
        return total_stats
    
    def clear_all(self):
        """Leert alle Caches"""
        for cache in self.caches.values():
            cache.clear()
        self.stats = {k: 0 for k in self.stats}


class Cache:
    """Einfacher Cache mit TTL-Unterstützung"""
    
    def __init__(self, name: str, cache_dir: str, default_ttl: int = 3600):
        self.name = name
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.cache_file = os.path.join(cache_dir, f"{name}.json")
        self.data = self._load_cache()
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
    
    def _load_cache(self) -> Dict[str, Any]:
        """Lädt Cache aus Datei"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_cache(self):
        """Speichert Cache in Datei"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Holt Wert aus Cache"""
        if key in self.data:
            entry = self.data[key]
            if self._is_expired(entry):
                del self.data[key]
                self.stats["deletes"] += 1
                self._save_cache()
                self.stats["misses"] += 1
                return default
            self.stats["hits"] += 1
            return entry["value"]
        self.stats["misses"] += 1
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Setzt Wert in Cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        self.data[key] = {
            "value": value,
            "expires": time.time() + ttl,
            "created": time.time()
        }
        self.stats["sets"] += 1
        self._save_cache()
    
    def delete(self, key: str) -> bool:
        """Löscht Wert aus Cache"""
        if key in self.data:
            del self.data[key]
            self.stats["deletes"] += 1
            self._save_cache()
            return True
        return False
    
    def clear(self):
        """Leert Cache"""
        self.data = {}
        self.stats = {k: 0 for k in self.stats}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Überprüft ob Eintrag abgelaufen ist"""
        return time.time() > entry.get("expires", 0)
    
    def get_stats(self) -> Dict[str, int]:
        """Gibt Statistiken zurück"""
        return self.stats.copy()


# Globale Instanzen
cache_manager = CacheManager()

def get_cache_stats() -> Dict[str, int]:
    """Gibt globale Cache-Statistiken zurück"""
    return cache_manager.get_stats()

def initialize_caches():
    """Initialisiert alle Caches"""
    # Hier können verschiedene Caches initialisiert werden
    cache_manager.get_cache("embeddings")
    cache_manager.get_cache("responses")
    cache_manager.get_cache("models")

# Bei Import automatisch initialisieren
initialize_caches()
