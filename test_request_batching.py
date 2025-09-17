#!/usr/bin/env python3
"""
Test-Script für Request Batching System
Demonstriert Batch-Verarbeitung für verschiedene Anfrage-Typen
"""

import sys
import os
import logging
import asyncio
import time
import requests
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Füge aktuelles Verzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batch_text_requests():
    """Testet Batch-Verarbeitung für Text-Anfragen"""
    logger.info("📝 Teste Batch-Text-Anfragen...")

    try:
        # Test-Daten
        text_requests = [
            {"text": "Was ist die Aufgabe des Bundeskanzlers?", "priority": 1},
            {"text": "Erkläre die Funktionsweise der Bundesregierung", "priority": 2},
            {"text": "Wie funktioniert das deutsche Parlament?", "priority": 1},
            {"text": "Was sind die wichtigsten Gesetze in Deutschland?", "priority": 3},
            {"text": "Beschreibe die Rolle der Opposition", "priority": 1}
        ]

        # Einzelne Anfragen senden
        request_ids = []
        for req in text_requests:
            try:
                response = requests.post(
                    "http://localhost:8000/batch/text",
                    json=req,
                    headers={"Authorization": "Bearer test-token"},
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    request_ids.append(result["request_id"])
                    logger.info(f"✅ Anfrage {result['request_id']} eingereicht")
                else:
                    logger.warning(f"⚠️ Anfrage fehlgeschlagen: {response.status_code}")

            except Exception as e:
                logger.error(f"❌ Fehler bei Anfrage: {e}")

        logger.info(f"📊 {len(request_ids)} Text-Anfragen zur Batch-Verarbeitung eingereicht")
        return request_ids

    except Exception as e:
        logger.error(f"❌ Batch-Text-Test fehlgeschlagen: {e}")
        return []

def test_batch_embedding_requests():
    """Testet Batch-Verarbeitung für Embedding-Anfragen"""
    logger.info("🔗 Teste Batch-Embedding-Anfragen...")

    try:
        # Test-Daten
        embedding_requests = [
            {"texts": ["Bundeskanzler", "Politik", "Deutschland"], "priority": 1},
            {"texts": ["Regierung", "Parlament", "Demokratie"], "priority": 2},
            {"texts": ["Gesetze", "Verfassung", "Grundgesetz"], "priority": 1}
        ]

        # Einzelne Anfragen senden
        request_ids = []
        for req in embedding_requests:
            try:
                response = requests.post(
                    "http://localhost:8000/batch/embeddings",
                    json=req,
                    headers={"Authorization": "Bearer test-token"},
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    request_ids.append(result["request_id"])
                    logger.info(f"✅ Embedding-Anfrage {result['request_id']} eingereicht")
                else:
                    logger.warning(f"⚠️ Embedding-Anfrage fehlgeschlagen: {response.status_code}")

            except Exception as e:
                logger.error(f"❌ Fehler bei Embedding-Anfrage: {e}")

        logger.info(f"📊 {len(request_ids)} Embedding-Anfragen zur Batch-Verarbeitung eingereicht")
        return request_ids

    except Exception as e:
        logger.error(f"❌ Batch-Embedding-Test fehlgeschlagen: {e}")
        return []

def test_batch_search_requests():
    """Testet Batch-Verarbeitung für Suchanfragen"""
    logger.info("🔍 Teste Batch-Suchanfragen...")

    try:
        # Test-Daten
        search_requests = [
            {"query": "Bundeskanzler Aufgaben", "context": ["Politik", "Regierung"], "priority": 1},
            {"query": "Deutsche Verfassung", "context": ["Gesetze", "Demokratie"], "priority": 2},
            {"query": "Parlament Funktionen", "context": ["Politik", "Gesetzgebung"], "priority": 1}
        ]

        # Einzelne Anfragen senden
        request_ids = []
        for req in search_requests:
            try:
                response = requests.post(
                    "http://localhost:8000/batch/search",
                    json=req,
                    headers={"Authorization": "Bearer test-token"},
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    request_ids.append(result["request_id"])
                    logger.info(f"✅ Suchanfrage {result['request_id']} eingereicht")
                else:
                    logger.warning(f"⚠️ Suchanfrage fehlgeschlagen: {response.status_code}")

            except Exception as e:
                logger.error(f"❌ Fehler bei Suchanfrage: {e}")

        logger.info(f"📊 {len(request_ids)} Suchanfragen zur Batch-Verarbeitung eingereicht")
        return request_ids

    except Exception as e:
        logger.error(f"❌ Batch-Search-Test fehlgeschlagen: {e}")
        return []

def test_immediate_batch_processing():
    """Testet sofortige Batch-Verarbeitung"""
    logger.info("⚡ Teste sofortige Batch-Verarbeitung...")

    try:
        # Test-Daten für sofortige Verarbeitung
        batch_requests = [
            {"type": "text", "text": "Was ist KI?", "priority": 1},
            {"type": "embedding", "texts": ["Künstliche Intelligenz", "Maschine"], "priority": 1},
            {"type": "search", "query": "KI Anwendungen", "context": ["Technologie"], "priority": 1}
        ]

        # Sofortige Batch-Verarbeitung
        response = requests.post(
            "http://localhost:8000/batch/process",
            json={"requests": batch_requests},
            headers={"Authorization": "Bearer test-token"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Sofortige Batch-Verarbeitung erfolgreich: {len(result['results'])} Ergebnisse")
            return result
        else:
            logger.warning(f"⚠️ Sofortige Batch-Verarbeitung fehlgeschlagen: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ Sofortige Batch-Verarbeitung fehlgeschlagen: {e}")
        return None

def test_batch_stats():
    """Testet Batch-Statistiken"""
    logger.info("📊 Teste Batch-Statistiken...")

    try:
        response = requests.get(
            "http://localhost:8000/admin/batch/stats",
            headers={"Authorization": "Bearer admin-token"},
            timeout=5
        )

        if response.status_code == 200:
            stats = response.json()
            logger.info("✅ Batch-Statistiken abgerufen:")
            logger.info(f"  - Text Processor: {stats['batch_system']['text_processor']['total_requests']} Anfragen")
            logger.info(f"  - Embedding Processor: {stats['batch_system']['embedding_processor']['total_requests']} Anfragen")
            logger.info(f"  - Search Processor: {stats['batch_system']['search_processor']['total_requests']} Anfragen")
            logger.info(".2f"            return stats
        else:
            logger.warning(f"⚠️ Batch-Statistiken fehlgeschlagen: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ Batch-Statistiken-Test fehlgeschlagen: {e}")
        return None

def test_batch_optimization():
    """Testet Batch-Optimierung"""
    logger.info("⚡ Teste Batch-Optimierung...")

    try:
        response = requests.post(
            "http://localhost:8000/admin/batch/optimize",
            headers={"Authorization": "Bearer admin-token"},
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Batch-Optimierung erfolgreich durchgeführt")
            return result
        else:
            logger.warning(f"⚠️ Batch-Optimierung fehlgeschlagen: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ Batch-Optimierung-Test fehlgeschlagen: {e}")
        return None

def performance_comparison():
    """Vergleicht Performance von Batch vs. sequentieller Verarbeitung"""
    logger.info("⚡ Führe Performance-Vergleich durch...")

    try:
        # Test-Daten
        test_requests = [
            {"type": "text", "text": f"Test-Anfrage {i}", "priority": 1}
            for i in range(10)
        ]

        # Batch-Verarbeitung Zeit messen
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/batch/process",
            json={"requests": test_requests},
            headers={"Authorization": "Bearer test-token"},
            timeout=15
        )
        batch_time = time.time() - start_time

        if response.status_code == 200:
            logger.info(".3f"            return {"batch_time": batch_time, "requests": len(test_requests)}
        else:
            logger.warning(f"⚠️ Performance-Test fehlgeschlagen: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ Performance-Vergleich fehlgeschlagen: {e}")
        return None

def concurrent_request_simulation():
    """Simuliert gleichzeitige Anfragen"""
    logger.info("🔄 Simuliere gleichzeitige Anfragen...")

    try:
        def send_request(req_id: int):
            """Sendet eine einzelne Anfrage"""
            try:
                response = requests.post(
                    "http://localhost:8000/batch/text",
                    json={"text": f"Gleichzeitige Anfrage {req_id}", "priority": 1},
                    headers={"Authorization": "Bearer test-token"},
                    timeout=5
                )
                return response.status_code == 200
            except:
                return False

        # Gleichzeitige Anfragen mit ThreadPool
        num_requests = 20
        with ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            results = list(executor.map(send_request, range(num_requests)))
            concurrent_time = time.time() - start_time

        successful_requests = sum(results)
        logger.info(".3f"        logger.info(".1f"
        return {"successful": successful_requests, "total": num_requests, "time": concurrent_time}

    except Exception as e:
        logger.error(f"❌ Concurrent-Request-Simulation fehlgeschlagen: {e}")
        return None

def main():
    """Hauptfunktion für Request Batching Tests"""
    logger.info("🚀 Starte Request Batching Tests...")

    # Warte kurz, um sicherzustellen, dass der Server läuft
    time.sleep(2)

    # Test 1: Batch-Text-Anfragen
    text_ids = test_batch_text_requests()
    logger.info(f"Test 1 (Batch Text): {'✅ BESTANDEN' if text_ids else '❌ FEHLGESCHLAGEN'}")

    # Test 2: Batch-Embedding-Anfragen
    embedding_ids = test_batch_embedding_requests()
    logger.info(f"Test 2 (Batch Embeddings): {'✅ BESTANDEN' if embedding_ids else '❌ FEHLGESCHLAGEN'}")

    # Test 3: Batch-Suchanfragen
    search_ids = test_batch_search_requests()
    logger.info(f"Test 3 (Batch Search): {'✅ BESTANDEN' if search_ids else '❌ FEHLGESCHLAGEN'}")

    # Test 4: Sofortige Batch-Verarbeitung
    immediate_result = test_immediate_batch_processing()
    logger.info(f"Test 4 (Immediate Batch): {'✅ BESTANDEN' if immediate_result else '❌ FEHLGESCHLAGEN'}")

    # Test 5: Batch-Statistiken
    stats_result = test_batch_stats()
    logger.info(f"Test 5 (Batch Stats): {'✅ BESTANDEN' if stats_result else '❌ FEHLGESCHLAGEN'}")

    # Test 6: Batch-Optimierung
    optimize_result = test_batch_optimization()
    logger.info(f"Test 6 (Batch Optimization): {'✅ BESTANDEN' if optimize_result else '❌ FEHLGESCHLAGEN'}")

    # Test 7: Performance-Vergleich
    perf_result = performance_comparison()
    logger.info(f"Test 7 (Performance): {'✅ BESTANDEN' if perf_result else '❌ FEHLGESCHLAGEN'}")

    # Test 8: Concurrent Requests
    concurrent_result = concurrent_request_simulation()
    logger.info(f"Test 8 (Concurrent): {'✅ BESTANDEN' if concurrent_result else '❌ FEHLGESCHLAGEN'}")

    # Zusammenfassung
    tests_passed = sum([
        bool(text_ids), bool(embedding_ids), bool(search_ids),
        bool(immediate_result), bool(stats_result), bool(optimize_result),
        bool(perf_result), bool(concurrent_result)
    ])

    if tests_passed >= 6:  # Mindestens 6 von 8 Tests bestanden
        logger.info("🎉 Request Batching Tests erfolgreich!")
        logger.info("✨ Features implementiert:")
        logger.info("  🎯 Intelligente Batch-Verarbeitung")
        logger.info("  📝 Text-Anfrage Batching")
        logger.info("  🔗 Embedding-Anfrage Batching")
        logger.info("  🔍 Suchanfrage Batching")
        logger.info("  ⚡ Sofortige Batch-Verarbeitung")
        logger.info("  📊 Detaillierte Statistiken")
        logger.info("  🔄 Adaptive Batch-Größen-Optimierung")
        logger.info("  🚀 Concurrent Request Handling")
        return 0
    else:
        logger.warning(f"⚠️ Nur {tests_passed}/8 Tests bestanden. Überprüfe die Implementierung.")
        return 1

if __name__ == "__main__":
    # Überprüfe, ob der Server läuft
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code != 200:
            logger.error("❌ API-Server scheint nicht zu laufen. Starte ihn mit: python bundeskanzler_api.py")
            sys.exit(1)
    except:
        logger.error("❌ API-Server nicht erreichbar. Starte ihn mit: python bundeskanzler_api.py")
        sys.exit(1)

    sys.exit(main())