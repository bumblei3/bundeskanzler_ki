#!/usr/bin/env python3
"""
Test-Script für Advanced Redis-basiertes Caching-System
Demonstriert Hybrid Cache, Smart Invalidation und Redis-Integration
"""

import sys
import os
import logging
import numpy as np
import time
from pathlib import Path

# Füge aktuelles Verzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_redis_cache_basic():
    """Testet grundlegende Redis Cache Funktionalität"""
    try:
        from core.redis_cache import RedisCacheManager, RedisCacheConfig

        logger.info("🔴 Teste Redis Cache Basic...")

        # Erstelle Redis Cache
        config = RedisCacheConfig(host="localhost", port=6379, db=1)  # Test-DB
        redis_cache = RedisCacheManager(config, namespace="test")

        if not redis_cache.health_check():
            logger.warning("⚠️ Redis nicht verfügbar - überspringe Redis Tests")
            return False

        # Test-Daten
        test_data = {
            "string": "Hello Redis!",
            "number": 42,
            "list": [1, 2, 3, "test"],
            "dict": {"key": "value", "nested": {"data": True}},
            "numpy_array": np.random.rand(10).astype(np.float32)
        }

        # Speichere Daten
        for key, value in test_data.items():
            success = redis_cache.set(f"test_{key}", value, ttl=300)  # 5min TTL
            if success:
                logger.info(f"✅ Gespeichert: {key}")
            else:
                logger.error(f"❌ Speichern fehlgeschlagen: {key}")
                return False

        # Lade Daten
        for key, original_value in test_data.items():
            loaded_value = redis_cache.get(f"test_{key}")
            if loaded_value is not None:
                # Spezielle Überprüfung für numpy arrays
                if isinstance(original_value, np.ndarray):
                    if np.allclose(loaded_value, original_value):
                        logger.info(f"✅ Geladen: {key} (numpy array)")
                    else:
                        logger.error(f"❌ Numpy array mismatch: {key}")
                        return False
                else:
                    if loaded_value == original_value:
                        logger.info(f"✅ Geladen: {key}")
                    else:
                        logger.error(f"❌ Wert mismatch: {key}")
                        return False
            else:
                logger.error(f"❌ Laden fehlgeschlagen: {key}")
                return False

        # Teste TTL
        ttl = redis_cache.get_ttl("test_string")
        if ttl > 0 and ttl <= 300:
            logger.info(f"✅ TTL funktioniert: {ttl}s verbleibend")
        else:
            logger.warning(f"⚠️ TTL ungewöhnlich: {ttl}")

        # Zeige Statistiken
        stats = redis_cache.get_stats()
        logger.info("📊 Redis Cache Statistiken:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        # Aufräumen
        redis_cache.clear_namespace()

        return True

    except Exception as e:
        logger.error(f"❌ Redis Cache Basic Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smart_invalidation():
    """Testet Smart Cache Invalidation"""
    try:
        from core.smart_invalidation import CacheInvalidationService, InvalidationTrigger, InvalidationPattern

        logger.info("🎯 Teste Smart Invalidation...")

        service = CacheInvalidationService()

        # Füge benutzerdefinierte Regel hinzu
        service.add_custom_invalidation_rule(
            pattern="custom:*",
            trigger=InvalidationTrigger.DATA_UPDATE,
            condition=lambda meta: meta.get('source') == 'test',
            priority=5
        )

        # Teste verschiedene Invalidierungen
        test_cases = [
            ("invalidate_on_data_change", {"data_type": "user", "entity_id": "123"}),
            ("invalidate_on_model_update", {"model_name": "bert", "version": "2.0"}),
            ("invalidate_on_config_change", {"config_section": "api"}),
        ]

        total_invalidated = 0
        for method_name, metadata in test_cases:
            method = getattr(service, method_name)
            invalidated = method(**metadata) if metadata else method()
            logger.info(f"🚨 {method_name}: {len(invalidated)} Keys invalidiert")
            total_invalidated += len(invalidated)

        # Zeige Statistiken
        stats = service.get_stats()
        logger.info("📊 Invalidation Statistiken:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"❌ Smart Invalidation Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_cache():
    """Testet Hybrid Cache (lokal + Redis)"""
    try:
        from core.intelligent_cache import HybridIntelligentCache
        from core.redis_cache import RedisCacheConfig

        logger.info("🔄 Teste Hybrid Cache...")

        # Erstelle Hybrid Cache
        redis_config = RedisCacheConfig(host="localhost", port=6379, db=2)  # Test-DB
        hybrid_cache = HybridIntelligentCache(
            name="test_hybrid",
            redis_config=redis_config,
            local_cache_size_mb=50
        )

        # Test-Daten mit Embeddings
        test_entries = [
            ("text_1", "Das ist ein Test-Text", np.random.rand(384).astype(np.float32)),
            ("text_2", "Ein weiterer Test-Text", np.random.rand(384).astype(np.float32)),
            ("data_1", {"type": "metadata", "size": 100}, None),
        ]

        # Speichere Daten
        for key, value, embedding in test_entries:
            success = hybrid_cache.set(key, value, ttl=600, embedding=embedding)
            if success:
                logger.info(f"✅ Hybrid gespeichert: {key}")
            else:
                logger.error(f"❌ Hybrid speichern fehlgeschlagen: {key}")
                return False

        # Lade Daten
        for key, original_value, _ in test_entries:
            loaded_value = hybrid_cache.get(key)
            if loaded_value == original_value:
                logger.info(f"✅ Hybrid geladen: {key}")
            else:
                logger.error(f"❌ Hybrid laden fehlgeschlagen: {key}")
                return False

        # Teste semantische Suche
        query_embedding = np.random.rand(384).astype(np.float32)
        similar_result = hybrid_cache.get("nonexistent", query_embedding=query_embedding)
        if similar_result is not None:
            logger.info("🎯 Semantische Suche funktioniert")
        else:
            logger.info("ℹ️ Kein semantischer Treffer (normal)")

        # Teste Smart Invalidation
        invalidated = hybrid_cache.trigger_invalidation("data_update", {"data_type": "test"})
        logger.info(f"🚨 Hybrid Invalidation: {len(invalidated)} Keys")

        # Zeige Statistiken
        stats = hybrid_cache.get_stats()
        logger.info("📊 Hybrid Cache Statistiken:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        # Aufräumen
        hybrid_cache.clear()

        return True

    except Exception as e:
        logger.error(f"❌ Hybrid Cache Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_integration():
    """Testet Integration mit bestehendem System"""
    try:
        from core.intelligent_cache import get_hybrid_cache, get_hybrid_cache_stats

        logger.info("🔗 Teste Cache Integration...")

        # Hole verschiedene Caches
        embedding_cache = get_hybrid_cache("embeddings")
        response_cache = get_hybrid_cache("responses")
        model_cache = get_hybrid_cache("models")

        # Test-Daten für verschiedene Cache-Typen
        test_data = {
            "embeddings": [
                ("embed_1", np.random.rand(768).astype(np.float32), np.random.rand(384).astype(np.float32)),
                ("embed_2", np.random.rand(768).astype(np.float32), np.random.rand(384).astype(np.float32)),
            ],
            "responses": [
                ("response_1", {"text": "Test response", "confidence": 0.95}, None),
                ("response_2", {"text": "Another response", "confidence": 0.87}, None),
            ],
            "models": [
                ("model_config", {"layers": 12, "heads": 8, "hidden_size": 768}, None),
            ]
        }

        # Speichere in verschiedenen Caches
        for cache_name, cache, entries in [("embeddings", embedding_cache, test_data["embeddings"]),
                                          ("responses", response_cache, test_data["responses"]),
                                          ("models", model_cache, test_data["models"])]:
            logger.info(f"💾 Speichere in {cache_name} Cache...")
            for key, value, embedding in entries:
                cache.set(key, value, ttl=3600, embedding=embedding)

        # Lade aus verschiedenen Caches
        for cache_name, cache, entries in [("embeddings", embedding_cache, test_data["embeddings"]),
                                          ("responses", response_cache, test_data["responses"]),
                                          ("models", model_cache, test_data["models"])]:
            logger.info(f"📖 Lade aus {cache_name} Cache...")
            for key, original_value, _ in entries:
                loaded_value = cache.get(key)
                # Spezielle Überprüfung für numpy arrays
                if isinstance(original_value, np.ndarray):
                    if loaded_value is not None and np.allclose(loaded_value, original_value):
                        logger.info(f"✅ {cache_name}: {key} (numpy)")
                    else:
                        logger.error(f"❌ {cache_name}: {key} (numpy)")
                        return False
                elif loaded_value == original_value:
                    logger.info(f"✅ {cache_name}: {key}")
                else:
                    logger.error(f"❌ {cache_name}: {key}")
                    return False

        # Zeige Gesamtstatistiken
        stats = get_hybrid_cache_stats()
        logger.info("📊 Gesamt-Cache-Statistiken:")
        for key, value in stats.items():
            if key != "cache_details":  # Details separat
                logger.info(f"   {key}: {value}")

        # Zeige Details für jeden Cache
        for cache_name, cache_stats in stats.get("cache_details", {}).items():
            logger.info(f"   📈 {cache_name}: {cache_stats.get('local_entries', 0)} Einträge")

        return True

    except Exception as e:
        logger.error(f"❌ Cache Integration Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_comparison():
    """Vergleicht Performance von lokalem vs Hybrid Cache"""
    try:
        from core.intelligent_cache import IntelligentCache, HybridIntelligentCache
        from core.redis_cache import RedisCacheConfig

        logger.info("⚡ Performance-Vergleich...")

        # Erstelle Caches
        local_cache = IntelligentCache("perf_test_local", max_size_mb=100)
        redis_config = RedisCacheConfig(host="localhost", port=6379, db=3)
        hybrid_cache = HybridIntelligentCache("perf_test_hybrid", redis_config=redis_config)

        # Test-Daten
        test_data = []
        for i in range(100):
            test_data.append((f"key_{i}", f"value_{i}", np.random.rand(384).astype(np.float32)))

        # Performance-Test: Schreiben
        logger.info("📝 Performance-Test: Schreiben...")

        # Lokaler Cache
        start_time = time.time()
        for key, value, embedding in test_data:
            local_cache.set(key, value, embedding=embedding)
        local_write_time = time.time() - start_time

        # Hybrid Cache
        start_time = time.time()
        for key, value, embedding in test_data:
            hybrid_cache.set(key, value, embedding=embedding)
        hybrid_write_time = time.time() - start_time

        logger.info(".3f")
        logger.info(".3f")

        # Performance-Test: Lesen
        logger.info("📖 Performance-Test: Lesen...")

        # Lokaler Cache
        start_time = time.time()
        for key, _, _ in test_data:
            local_cache.get(key)
        local_read_time = time.time() - start_time

        # Hybrid Cache
        start_time = time.time()
        for key, _, _ in test_data:
            hybrid_cache.get(key)
        hybrid_read_time = time.time() - start_time

        logger.info(".3f")
        logger.info(".3f")

        # Aufräumen
        local_cache.clear()
        hybrid_cache.clear()

        return True

    except Exception as e:
        logger.error(f"❌ Performance-Vergleich fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion für Advanced Cache Tests"""
    print("🚀 Advanced Redis Cache Test Suite")
    print("=" * 50)

    tests = [
        ("Redis Cache Basic", test_redis_cache_basic),
        ("Smart Invalidation", test_smart_invalidation),
        ("Hybrid Cache", test_hybrid_cache),
        ("Cache Integration", test_cache_integration),
        ("Performance Comparison", performance_comparison),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' fehlgeschlagen: {e}")
            results.append((test_name, False))

    # Zusammenfassung
    print("\n" + "="*60)
    print("📋 TEST-ZUSAMMENFASSUNG")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n📊 Ergebnis: {passed}/{total} Tests erfolgreich")

    if passed == total:
        print("🎉 Alle Advanced Cache Tests erfolgreich!")
        print("✅ Redis-Integration aktiv")
        print("✅ Smart Invalidation funktioniert")
        print("✅ Hybrid Cache bereit")
    elif passed >= total * 0.8:
        print("👍 Meisten Tests erfolgreich. Advanced Caching teilweise aktiv.")
    else:
        print("⚠️ Einige Tests fehlgeschlagen. Fallback auf lokales Caching.")

    print("\n💡 Cache-Features:")
    print("   • Redis-basierte Distributed Caches")
    print("   • Smart Invalidation bei Datenänderungen")
    print("   • Hybrid Cache mit automatischem Fallback")
    print("   • Semantische Ähnlichkeitssuche")
    print("   • Automatische Synchronisation")


if __name__ == "__main__":
    main()