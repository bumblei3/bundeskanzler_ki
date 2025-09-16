#!/usr/bin/env python3
"""
🧪 Faktencheck-System Test für Bundeskanzler-KI
===============================================

Testet die Integration des Faktencheck-Systems in die RTX 2070 KI
"""

import os
import sys
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.rtx2070_bundeskanzler_ki import get_rtx2070_bundeskanzler_ki


def test_fact_check_integration():
    """Testet die Faktencheck-Integration"""
    print("🧪 Faktencheck-Integration Test")
    print("=" * 50)

    # KI initialisieren
    ki = get_rtx2070_bundeskanzler_ki()

    # System-Info prüfen
    system_info = ki.get_system_info()
    print(
        f"🔍 Faktencheck verfügbar: {system_info.get('components_status', {}).get('fact_check_system', False)}"
    )
    print(f"📊 Faktencheck-Info: {system_info.get('fact_check_info', {})}")

    # Test-Queries
    test_queries = [
        "Was ist die aktuelle Klimapolitik Deutschlands?",
        "Wie hoch ist das Budget der Bundesregierung 2024?",
        "Wie funktioniert die Energiewende in Deutschland?",
        "Was sind die Ziele der Bundesregierung für 2030?",
    ]

    print("\n🔬 Teste faktencheck-validierte Queries:")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 30)

        try:
            # Normale Query
            normal_result = ki.process_query(query)
            print("📝 Normale Antwort:")
            print(f"   {normal_result.get('response', '')[:100]}...")

            # Faktencheck-validierte Query
            fact_check_result = ki.process_query_with_fact_check(query)
            print("🔍 Faktencheck-Antwort:")
            print(f"   {fact_check_result.get('response', '')[:100]}...")

            # Faktencheck-Details
            fact_check = fact_check_result.get("fact_check")
            if fact_check:
                print("📊 Faktencheck-Ergebnisse:")
                print(f"   Konfidenz: {fact_check.get('overall_confidence', 0):.2f}")
                print(f"   Quellen: {fact_check.get('sources_used', 0)}")
                print(
                    f"   Korrekte Aussagen: {fact_check.get('accurate_statements', 0)}/{fact_check.get('total_statements', 0)}"
                )
            else:
                print("⚠️  Faktencheck nicht verfügbar")

        except Exception as e:
            print(f"❌ Fehler bei Query {i}: {e}")

        except Exception as e:
            print(f"❌ Fehler bei Query {i}: {e}")

    print("\n✅ Faktencheck-Integration Test abgeschlossen!")


if __name__ == "__main__":
    test_fact_check_integration()
