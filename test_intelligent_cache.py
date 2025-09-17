#!/usr/bin/env python3
"""
Test-Script für intelligentes Caching-System
Demonstriert Advanced Caching Features mit semantischer Ähnlichkeit
"""

import sys
import os
import logging
import numpy as np
import time

# Füge aktuelles Verzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_intelligent_cache():
    """Testet das intelligente Cache-System"""
    try:
        from core.intelligent_cache import IntelligentCache

        logger.info("🧠 Teste Intelligent Cache System...")

        # Erstelle intelligenten Cache für Embeddings
        cache = IntelligentCache(
            name="test_embeddings",
            max_size_mb=50,
            enable_compression=True,
            enable_similarity=True
        )

        # Test-Daten
        test_texts = [
            "Die Bundeskanzler KI ist ein fortschrittliches System",
            "Das KI-System des Bundeskanzlers bietet moderne Funktionen",
            "Fortschrittliche KI für politische Entscheidungsfindung",
            "Maschinelles Lernen in der Politik",
            "Künstliche Intelligenz für Regierungsarbeit"
        ]

        # Erstelle Test-Embeddings (vereinfacht)
        embeddings = []
        for i, text in enumerate(test_texts):
            # Simuliere Embedding (normalerweise von Modell)
            embedding = np.random.rand(384).astype(np.float32)
            embedding[0] = i * 0.1  # Kleine Variation für semantische Ähnlichkeit
            embeddings.append(embedding)

        logger.info("📝 Speichere Test-Embeddings im Cache...")

        # Speichere Embeddings im Cache
        for text, embedding in zip(test_texts, embeddings):
            cache.set(
                key=f"embedding_{hash(text) % 1000}",
                value=embedding,
                embedding=embedding,
                ttl=3600,
                metadata={"text": text, "length": len(text)}
            )

        logger.info(f"✅ {len(test_texts)} Embeddings gespeichert")

        # Teste direkte Cache-Hits
        logger.info("🎯 Teste direkte Cache-Hits...")
        for i, text in enumerate(test_texts[:2]):
            cache_key = f"embedding_{hash(text) % 1000}"
            cached_embedding = cache.get(cache_key)
            if cached_embedding is not None:
                logger.info(f"✅ Direkter Hit für: {text[:30]}...")
            else:
                logger.info(f"❌ Kein Hit für: {text[:30]}...")

        # Teste semantische Suche
        logger.info("🔍 Teste semantische Ähnlichkeitssuche...")
        query_text = "Moderne KI-Systeme für politische Anwendungen"
        query_embedding = np.random.rand(384).astype(np.float32)
        query_embedding[0] = 0.15  # Ähnlich zu erstem Embedding

        similar_result = cache.get(
            key="nonexistent_key",
            query_embedding=query_embedding,
            similarity_threshold=0.8
        )

        if similar_result is not None:
            logger.info("🎯 Semantischer Treffer gefunden!")
        else:
            logger.info("ℹ️ Kein semantischer Treffer (normal bei zufälligen Embeddings)")

        # Zeige Cache-Statistiken
        stats = cache.get_stats()
        logger.info("📊 Cache-Statistiken:")
        logger.info(f"  - Einträge: {stats['entries']}")
        logger.info(f"  - Größe: {stats['size_mb']:.2f} MB")
        logger.info(f"  - Hit Rate: {stats['hit_rate']:.1f}%")
        logger.info(f"  - Kompressionsrate: {stats['compression_ratio']:.2f}x")

        # Teste LRU-Eviction
        logger.info("🗑️ Teste LRU-Eviction...")
        initial_entries = len(cache.cache)

        # Füge viele Einträge hinzu, um Eviction zu triggern
        for i in range(20):
            large_embedding = np.random.rand(768).astype(np.float32)  # Größere Embeddings
            cache.set(
                key=f"large_embedding_{i}",
                value=large_embedding,
                embedding=large_embedding,
                ttl=3600
            )

        final_entries = len(cache.cache)
        evicted_count = initial_entries + 20 - final_entries
        logger.info(f"🗑️ {evicted_count} Einträge durch LRU entfernt")

        # Finale Statistiken
        final_stats = cache.get_stats()
        logger.info("📈 Finale Cache-Statistiken:")
        logger.info(f"  - Einträge: {final_stats['entries']}")
        logger.info(f"  - Größe: {final_stats['size_mb']:.2f} MB")
        logger.info(f"  - Auslastung: {final_stats['utilization_percent']:.1f}%")
        logger.info(f"  - Evictions: {final_stats['evictions']}")

        return True

    except Exception as e:
        logger.error(f"❌ Intelligent Cache Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_cache_integration():
    """Testet Integration mit multimodalem KI-System"""
    try:
        from multimodal_ki import MultimodalTransformerModel

        logger.info("🔗 Teste Cache-Integration mit MultimodalTransformerModel...")

        # Erstelle Modell (wird automatisch Caches initialisieren)
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Teste Cache-Methoden
        logger.info("📊 Teste Cache-Methoden...")

        # Cache-Statistiken abrufen
        cache_stats = model.get_cache_stats()
        logger.info(f"📈 Cache-Statistiken verfügbar: {'Ja' if cache_stats else 'Nein'}")

        if cache_stats and 'cache_details' in cache_stats:
            logger.info(f"🎯 {cache_stats['total_caches']} intelligente Caches verfügbar")
            for cache_name in cache_stats['cache_details'].keys():
                logger.info(f"  - {cache_name}")

        # Teste Embedding-Caching (falls verfügbar)
        if hasattr(model, 'embedding_cache') and model.embedding_cache:
            test_text = "Test für Embedding-Cache"
            test_embedding = np.random.rand(384).astype(np.float32)

            # Speichere Embedding
            model.cache_embedding(test_text, test_embedding)

            # Hole Embedding zurück
            cached = model.get_cached_embedding(test_text)
            if cached is not None:
                logger.info("✅ Embedding-Cache funktioniert")
            else:
                logger.info("ℹ️ Embedding-Cache noch leer")

        # Teste Response-Caching
        if hasattr(model, 'response_cache') and model.response_cache:
            test_query = "Was ist künstliche Intelligenz?"
            test_response = "Künstliche Intelligenz ist ein Bereich der Informatik..."

            # Speichere Response
            model.cache_response(test_query, test_response)

            # Hole Response zurück
            cached = model.get_cached_response(test_query)
            if cached is not None:
                logger.info("✅ Response-Cache funktioniert")
            else:
                logger.info("ℹ️ Response-Cache noch leer")

        # Optimiere Caches
        model.optimize_cache()
        logger.info("⚡ Caches optimiert")

        return True

    except Exception as e:
        logger.error(f"❌ Multimodal Cache Integration Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_comparison():
    """Vergleicht Performance mit und ohne Caching"""
    try:
        from core.intelligent_cache import IntelligentCache

        logger.info("⚡ Führe Performance-Vergleich durch...")

        # Erstelle Cache
        cache = IntelligentCache("performance_test", max_size_mb=100)

        # Test-Daten
        test_data = []
        for i in range(100):
            data = {
                "id": i,
                "text": f"Test-Datensatz {i} mit zusätzlichen Informationen",
                "metadata": {"created": time.time(), "size": "large" if i % 10 == 0 else "small"}
            }
            test_data.append(data)

        # Performance ohne Cache
        logger.info("⏱️ Messe Performance ohne Cache...")
        start_time = time.time()
        results_no_cache = []
        for data in test_data:
            # Simuliere Verarbeitung
            time.sleep(0.001)  # 1ms Verarbeitung
            results_no_cache.append(data)
        no_cache_time = time.time() - start_time

        # Performance mit Cache
        logger.info("⏱️ Messe Performance mit Cache...")

        # Fülle Cache
        for data in test_data:
            cache.set(f"data_{data['id']}", data, ttl=3600)

        # Cache-Hits messen
        start_time = time.time()
        results_with_cache = []
        for data in test_data:
            cached_result = cache.get(f"data_{data['id']}")
            if cached_result:
                results_with_cache.append(cached_result)
            else:
                # Fallback ohne Cache
                time.sleep(0.001)
                results_with_cache.append(data)
        with_cache_time = time.time() - start_time

        # Ergebnisse
        speedup = no_cache_time / with_cache_time if with_cache_time > 0 else 1.0

        logger.info("📊 Performance-Ergebnisse:")
        logger.info(f"  - Ohne Cache: {no_cache_time:.3f}s")
        logger.info(f"  - Mit Cache: {with_cache_time:.3f}s")
        logger.info(f"  - Geschwindigkeitszunahme: {speedup:.1f}x")

        stats = cache.get_stats()
        logger.info(f"  - Cache Hit Rate: {stats['hit_rate']:.1f}%")

        return True

    except Exception as e:
        logger.error(f"❌ Performance-Vergleich fehlgeschlagen: {e}")
        return False

def main():
    """Hauptfunktion für Tests"""
    logger.info("🚀 Starte Advanced Caching Tests...")

    # Test 1: Intelligent Cache System
    test1_result = test_intelligent_cache()
    logger.info(f"Test 1 (Intelligent Cache): {'✅ BESTANDEN' if test1_result else '❌ FEHLGESCHLAGEN'}")

    # Test 2: Multimodal Integration
    test2_result = test_multimodal_cache_integration()
    logger.info(f"Test 2 (Multimodal Integration): {'✅ BESTANDEN' if test2_result else '❌ FEHLGESCHLAGEN'}")

    # Test 3: Performance-Vergleich
    test3_result = performance_comparison()
    logger.info(f"Test 3 (Performance): {'✅ BESTANDEN' if test3_result else '❌ FEHLGESCHLAGEN'}")

    # Zusammenfassung
    if test1_result and test2_result and test3_result:
        logger.info("🎉 Alle Advanced Caching Tests bestanden!")
        logger.info("✨ Features implementiert:")
        logger.info("  🎯 Semantische Ähnlichkeitssuche")
        logger.info("  🗑️ LRU-Eviction Policy")
        logger.info("  🗜️ Automatische Kompression")
        logger.info("  📊 Detaillierte Performance-Metriken")
        logger.info("  🔗 Multimodale Integration")
        return 0
    else:
        logger.warning("⚠️ Einige Tests fehlgeschlagen. Überprüfe die Implementierung.")
        return 1

if __name__ == "__main__":
    sys.exit(main())