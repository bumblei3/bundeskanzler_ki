#!/usr/bin/env python3
"""
Smart Cache Invalidation System für Bundeskanzler-KI
Automatische Invalidierung von Cache-Einträgen bei Datenänderungen
"""

import logging
import re
import time
from typing import Dict, List, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class InvalidationTrigger(Enum):
    """Arten von Invalidierung-Triggers"""
    DATA_UPDATE = "data_update"
    MODEL_CHANGE = "model_change"
    CONFIG_CHANGE = "config_change"
    TIME_BASED = "time_based"
    DEPENDENCY_CHANGE = "dependency_change"


@dataclass
class InvalidationPattern:
    """Pattern für Cache-Invalidierung"""
    pattern: str  # Regex pattern für Keys
    trigger: InvalidationTrigger
    condition: Callable[[Dict[str, Any]], bool] = lambda x: True
    priority: int = 1
    cascade: bool = False  # Invalidiert auch abhängige Keys


@dataclass
class CacheDependency:
    """Cache-Abhängigkeit für Cascade-Invalidierung"""
    key: str
    depends_on: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


class SmartInvalidationManager:
    """
    Intelligentes Cache-Invalidierungssystem
    Überwacht Datenänderungen und invalidiert betroffene Cache-Einträge
    """

    def __init__(self):
        self.patterns: List[InvalidationPattern] = []
        self.dependencies: Dict[str, CacheDependency] = {}
        self.last_invalidation_time: Dict[str, float] = {}
        self.invalidation_stats = {
            "total_invalidations": 0,
            "pattern_matches": 0,
            "cascade_invalidations": 0,
            "time_based_cleanups": 0
        }

    def add_invalidation_pattern(self, pattern: InvalidationPattern):
        """Fügt Invalidierungspattern hinzu"""
        self.patterns.append(pattern)
        # Sortiere nach Priorität (höher = wichtiger)
        self.patterns.sort(key=lambda p: p.priority, reverse=True)
        logger.info(f"➕ Invalidierungspattern hinzugefügt: {pattern.pattern}")

    def add_dependency(self, key: str, depends_on: Set[str]):
        """Fügt Cache-Abhängigkeit hinzu"""
        if key not in self.dependencies:
            self.dependencies[key] = CacheDependency(key=key)

        dep = self.dependencies[key]
        dep.depends_on.update(depends_on)

        # Registriere umgekehrte Abhängigkeiten
        for parent_key in depends_on:
            if parent_key not in self.dependencies:
                self.dependencies[parent_key] = CacheDependency(key=parent_key)
            self.dependencies[parent_key].dependents.add(key)

    def trigger_invalidation(self, trigger: InvalidationTrigger,
                           metadata: Dict[str, Any] = None) -> List[str]:
        """
        Löst Invalidierung basierend auf Trigger aus

        Returns:
            Liste der invalidierten Keys
        """
        invalidated_keys = []
        metadata = metadata or {}

        logger.info(f"🚨 Invalidierung Trigger: {trigger.value}")

        # Finde matching patterns
        for pattern in self.patterns:
            if pattern.trigger == trigger and pattern.condition(metadata):
                matching_keys = self._find_matching_keys(pattern.pattern)
                if matching_keys:
                    invalidated_keys.extend(matching_keys)
                    self.invalidation_stats["pattern_matches"] += len(matching_keys)

                    # Cascade-Invalidierung falls aktiviert
                    if pattern.cascade:
                        cascade_keys = self._get_cascade_keys(matching_keys)
                        invalidated_keys.extend(cascade_keys)
                        self.invalidation_stats["cascade_invalidations"] += len(cascade_keys)

        # Entferne Duplikate
        invalidated_keys = list(set(invalidated_keys))
        self.invalidation_stats["total_invalidations"] += len(invalidated_keys)

        # Aktualisiere Timestamps
        current_time = time.time()
        for key in invalidated_keys:
            self.last_invalidation_time[key] = current_time

        logger.info(f"🗑️ {len(invalidated_keys)} Cache-Einträge invalidiert")
        return invalidated_keys

    def _find_matching_keys(self, pattern: str) -> List[str]:
        """Findet Keys die zu Pattern matchen"""
        # In einer echten Implementierung würde hier die Cache-Datenbank durchsucht
        # Für Demo-Zwecke simulieren wir einige Keys
        matching_keys = []

        # Simuliere verschiedene Key-Patterns
        if pattern == "embedding:*":
            matching_keys = [f"embedding:{i}" for i in range(10)]
        elif pattern == "model:*":
            matching_keys = ["model:response_cache", "model:embedding_cache"]
        elif pattern == "config:*":
            matching_keys = ["config:api_settings", "config:model_config"]
        elif pattern == "user:*":
            matching_keys = [f"user:{i}" for i in range(5)]

        return matching_keys

    def _get_cascade_keys(self, keys: List[str]) -> List[str]:
        """Findet abhängige Keys für Cascade-Invalidierung"""
        cascade_keys = []

        for key in keys:
            if key in self.dependencies:
                dep = self.dependencies[key]
                cascade_keys.extend(dep.dependents)

        return list(set(cascade_keys))

    def schedule_time_based_invalidation(self, interval_seconds: int = 3600):
        """Plant zeitbasierte Invalidierung"""
        import threading

        def time_based_cleanup():
            while True:
                time.sleep(interval_seconds)
                self._perform_time_based_cleanup()

        thread = threading.Thread(target=time_based_cleanup, daemon=True)
        thread.start()
        logger.info(f"⏰ Zeitbasierte Invalidierung gestartet (Intervall: {interval_seconds}s)")

    def _perform_time_based_cleanup(self):
        """Führt zeitbasierte Cache-Bereinigung durch"""
        current_time = time.time()
        expired_keys = []

        # Finde Keys die länger als 24h nicht invalidiert wurden
        max_age = 24 * 3600  # 24 Stunden
        for key, last_time in self.last_invalidation_time.items():
            if current_time - last_time > max_age:
                expired_keys.append(key)

        if expired_keys:
            logger.info(f"🕐 Zeitbasierte Bereinigung: {len(expired_keys)} Keys")
            self.invalidation_stats["time_based_cleanups"] += len(expired_keys)

            # Entferne aus Tracking
            for key in expired_keys:
                del self.last_invalidation_time[key]

    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Gibt Invalidierungsstatistiken zurück"""
        return self.invalidation_stats.copy()

    def reset_stats(self):
        """Setzt Statistiken zurück"""
        self.invalidation_stats = {k: 0 for k in self.invalidation_stats}


# Vordefinierte Invalidierungspatterns für Bundeskanzler-KI
DEFAULT_INVALIDATION_PATTERNS = [
    # Model-bezogene Invalidierung
    InvalidationPattern(
        pattern="model:*",
        trigger=InvalidationTrigger.MODEL_CHANGE,
        priority=10,
        cascade=True
    ),

    # Embedding-Invalidierung bei Datenänderungen
    InvalidationPattern(
        pattern="embedding:*",
        trigger=InvalidationTrigger.DATA_UPDATE,
        condition=lambda meta: meta.get('data_type') == 'corpus',
        priority=8
    ),

    # Response-Cache Invalidierung bei Model-Updates
    InvalidationPattern(
        pattern="response:*",
        trigger=InvalidationTrigger.MODEL_CHANGE,
        priority=9,
        cascade=True
    ),

    # Konfigurations-Invalidierung
    InvalidationPattern(
        pattern="config:*",
        trigger=InvalidationTrigger.CONFIG_CHANGE,
        priority=7
    ),

    # User-spezifische Invalidierung
    InvalidationPattern(
        pattern="user:*",
        trigger=InvalidationTrigger.DATA_UPDATE,
        condition=lambda meta: meta.get('update_type') == 'profile',
        priority=5
    )
]


class CacheInvalidationService:
    """
    Service für Cache-Invalidierung mit Integration in bestehende Systeme
    """

    def __init__(self):
        self.manager = SmartInvalidationManager()

        # Lade Standard-Patterns
        for pattern in DEFAULT_INVALIDATION_PATTERNS:
            self.manager.add_invalidation_pattern(pattern)

        # Starte zeitbasierte Bereinigung
        self.manager.schedule_time_based_invalidation()

    def invalidate_on_data_change(self, data_type: str, entity_id: str = None) -> List[str]:
        """Invalidiert Cache bei Datenänderungen"""
        metadata = {
            "data_type": data_type,
            "entity_id": entity_id,
            "timestamp": time.time()
        }

        return self.manager.trigger_invalidation(
            InvalidationTrigger.DATA_UPDATE,
            metadata
        )

    def invalidate_on_model_update(self, model_name: str, version: str = None) -> List[str]:
        """Invalidiert Cache bei Model-Updates"""
        metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": time.time()
        }

        return self.manager.trigger_invalidation(
            InvalidationTrigger.MODEL_CHANGE,
            metadata
        )

    def invalidate_on_config_change(self, config_section: str) -> List[str]:
        """Invalidiert Cache bei Konfigurationsänderungen"""
        metadata = {
            "config_section": config_section,
            "timestamp": time.time()
        }

        return self.manager.trigger_invalidation(
            InvalidationTrigger.CONFIG_CHANGE,
            metadata
        )

    def add_custom_invalidation_rule(self, pattern: str, trigger: InvalidationTrigger,
                                   condition: Callable = None, priority: int = 1):
        """Fügt benutzerdefinierte Invalidierungsregel hinzu"""
        invalidation_pattern = InvalidationPattern(
            pattern=pattern,
            trigger=trigger,
            condition=condition or (lambda x: True),
            priority=priority
        )

        self.manager.add_invalidation_pattern(invalidation_pattern)

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        return {
            "invalidation_stats": self.manager.get_invalidation_stats(),
            "patterns_count": len(self.manager.patterns),
            "dependencies_count": len(self.manager.dependencies),
            "tracked_keys_count": len(self.manager.last_invalidation_time)
        }