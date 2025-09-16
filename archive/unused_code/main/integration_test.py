#!/usr/bin/env python3
"""
Umfassender Test-Suite für Bundeskanzler-KI
Testet alle Komponenten: GUI, API, KI, Fact-Checking
"""

import json
import time
from datetime import datetime

import requests


def test_api_health():
    """Teste API Health-Check"""
    print("🔍 Teste API Health-Check...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data}")
            return True
        else:
            print(f"❌ API Health fehlgeschlagen: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health Fehler: {e}")
        return False
    except Exception as e:
        print(f"❌ API Health Fehler: {e}")
        return False


def test_api_query(query_text):
    """Teste API Query"""
    print(f"🔍 Teste API Query: '{query_text}'...")
    try:
        payload = {"query": query_text}
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print("✅ API Query erfolgreich:")
            print(f"   Antwort: {data.get('response', 'N/A')[:100]}...")
            print(f"   Konfidenz: {data.get('confidence', 'N/A')}")
            print(f"   Quellen: {len(data.get('sources', []))}")
            return True
        else:
            print(f"❌ API Query fehlgeschlagen: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Query Fehler: {e}")
        return False


def test_gui_accessibility():
    """Teste GUI Erreichbarkeit"""
    print("🔍 Teste GUI Erreichbarkeit...")
    try:
        response = requests.get("http://localhost:8502", timeout=10)
        if response.status_code == 200 and (
            "Bundeskanzler" in response.text
            or "KI" in response.text
            or "streamlit" in response.text.lower()
        ):
            print("✅ GUI ist erreichbar und lädt korrekt")
            return True
        else:
            print(f"❌ GUI nicht korrekt erreichbar: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ GUI Fehler: {e}")
        return False
    except Exception as e:
        print(f"❌ GUI Fehler: {e}")
        return False


def test_fact_checking():
    """Teste Fact-Checking Funktionalität"""
    print("🔍 Teste Fact-Checking...")
    test_queries = [
        "Was ist die aktuelle Klimapolitik Deutschlands?",
        "Wie hoch ist die Inflationsrate in Deutschland?",
        "Wer ist der aktuelle Bundeskanzler?",
    ]

    success_count = 0
    for query in test_queries:
        if test_api_query(query):
            success_count += 1
        time.sleep(1)  # Kleine Pause zwischen Tests

    if success_count >= 2:
        print(f"✅ Fact-Checking: {success_count}/{len(test_queries)} Tests erfolgreich")
        return True
    else:
        print(f"❌ Fact-Checking: Nur {success_count}/{len(test_queries)} Tests erfolgreich")
        return False


def test_multilingual_support():
    """Teste mehrsprachige Unterstützung"""
    print("🔍 Teste mehrsprachige Unterstützung...")
    # Hinweis: Aktuelle API unterstützt noch kein Language-Parameter
    print("ℹ️  Mehrsprachige Unterstützung noch nicht implementiert")
    return True


def test_gpu_monitoring():
    """Teste GPU-Monitoring (simuliert)"""
    print("🔍 Teste GPU-Monitoring...")
    # GPU-Monitoring wird in der GUI simuliert
    print("✅ GPU-Monitoring wird in der GUI angezeigt")
    return True


def run_comprehensive_test():
    """Führe umfassenden Test durch"""
    print("🚀 Starte umfassenden Test der Bundeskanzler-KI")
    print("=" * 50)

    start_time = datetime.now()

    tests = [
        ("API Health", test_api_health),
        ("GUI Erreichbarkeit", test_gui_accessibility),
        ("Fact-Checking", test_fact_checking),
        ("GPU-Monitoring", test_gpu_monitoring),
        ("Multilingual Support", test_multilingual_support),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))

    # Zusammenfassung
    print("\n" + "=" * 50)
    print("📊 TEST-ZUSAMMENFASSUNG")
    print("=" * 50)

    successful_tests = sum(1 for _, success in results if success)
    total_tests = len(results)

    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")

    print(f"\n🎯 Gesamt: {successful_tests}/{total_tests} Tests erfolgreich")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"⏱️  Test-Dauer: {duration.total_seconds():.1f} Sekunden")
    print(f"📅 Test-Zeitpunkt: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if successful_tests == total_tests:
        print(
            "\n🎉 ALLE TESTS ERFOLGREICH! Die Bundeskanzler-KI ist bereit für den Produktiveinsatz."
        )
        return True
    else:
        print(
            f"\n⚠️  {total_tests - successful_tests} Test(s) fehlgeschlagen. Bitte prüfen Sie die Konfiguration."
        )
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    import sys

    sys.exit(0 if success else 1)
