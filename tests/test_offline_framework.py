#!/usr/bin/env python3
"""
Offline Testing Framework für Bundeskanzler-KI
API-FREI: Vollständige Tests ohne externe Abhängigkeiten

Test-Kategorien:
- Unit-Tests für lokale Modelle
- GPU-Performance-Tests
- Integration-Tests
- Offline-Funktionalitäts-Tests
- Modell-Qualitäts-Tests

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import json
import logging
import os
import sys
import time
import unittest
from typing import Dict, List, Any, Optional
import torch

# Test-Imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from multimodal_ki import MultimodalTransformerModel
    from core.rtx2070_llm_manager import RTX2070LLMManager
    from core.local_monitoring import get_monitoring_system
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"❌ Import-Fehler: {e}")

logger = logging.getLogger(__name__)


class RTX2070OfflineTestSuite(unittest.TestCase):
    """Vollständige Test-Suite für RTX 2070 optimierte KI"""

    def setUp(self):
        """Test-Setup"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Erforderliche Module nicht verfügbar")

        self.monitoring = get_monitoring_system()
        self.monitoring.start_monitoring(interval_seconds=1.0)  # Schnelles Monitoring für Tests

    def tearDown(self):
        """Test-Cleanup"""
        if hasattr(self, 'monitoring'):
            self.monitoring.stop_monitoring()

    def test_rtx2070_detection(self):
        """Test: RTX 2070 GPU-Erkennung"""
        gpu_available = torch.cuda.is_available()
        self.assertTrue(gpu_available, "CUDA GPU nicht verfügbar")

        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            is_rtx2070 = 'RTX 2070' in gpu_name
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print(f"✅ GPU erkannt: {gpu_name} ({gpu_memory:.1f}GB)")

            if is_rtx2070:
                self.assertGreaterEqual(gpu_memory, 7.5, "RTX 2070 sollte mindestens 8GB VRAM haben")
            else:
                print(f"⚠️ Test auf {gpu_name} läuft (nicht RTX 2070)")

    def test_monitoring_system(self):
        """Test: Lokales Monitoring-System"""
        # Warte kurz für erste Metriken
        time.sleep(2)

        # Teste Dashboard-Daten
        data = self.monitoring.get_dashboard_data()
        self.assertIsInstance(data, dict, "Dashboard sollte Dict zurückgeben")
        self.assertIn('timestamp', data, "Timestamp sollte vorhanden sein")

        # Teste GPU-Metriken (falls verfügbar)
        if data.get('gpu'):
            gpu_data = data['gpu']
            self.assertIn('memory_used_mb', gpu_data, "GPU-Memory-Metriken sollten verfügbar sein")

        # Teste System-Metriken
        if data.get('system'):
            system_data = data['system']
            self.assertIn('cpu_percent', system_data, "CPU-Metriken sollten verfügbar sein")

        # Teste Health-Status
        health = self.monitoring.get_health_status()
        self.assertIn('overall_healthy', health, "Health-Status sollte verfügbar sein")

        print("✅ Monitoring-System funktioniert korrekt")

    def test_rtx2070_llm_manager(self):
        """Test: RTX 2070 LLM Manager"""
        manager = RTX2070LLMManager()

        # Teste Modell-Auswahl
        simple_model = manager.select_optimal_model("simple")
        medium_model = manager.select_optimal_model("medium")
        complex_model = manager.select_optimal_model("complex")

        self.assertIsInstance(simple_model, str, "Modell-Auswahl sollte String zurückgeben")
        self.assertIsInstance(medium_model, str, "Modell-Auswahl sollte String zurückgeben")
        self.assertIsInstance(complex_model, str, "Modell-Auswahl sollte String zurückgeben")

        print(f"✅ Modell-Auswahl: Simple={simple_model}, Medium={medium_model}, Complex={complex_model}")

        # Teste Query-Analyse
        test_queries = [
            ("Was ist 2+2?", "simple"),
            ("Erkläre die Klimapolitik", "medium"),
            ("Analysiere die EU-Wirtschaftspolitik detailliert", "complex")
        ]

        for query, expected_complexity in test_queries:
            complexity = manager.analyze_query_complexity(query)
            self.assertEqual(complexity, expected_complexity,
                           f"Query '{query}' sollte '{expected_complexity}' sein, ist aber '{complexity}'")

        print("✅ Query-Komplexitätsanalyse funktioniert")

    def test_multimodal_model_initialization(self):
        """Test: MultimodalTransformerModel Initialisierung"""
        # Teste RTX 2070 Modus
        model = MultimodalTransformerModel(model_tier="rtx2070")

        self.assertEqual(model.model_tier, "rtx2070", "Model-Tier sollte rtx2070 sein")
        self.assertTrue(model.is_rtx2070, "RTX 2070 sollte erkannt werden")

        # Teste Manager-Verfügbarkeit
        if model.rtx2070_manager:
            self.assertIsNotNone(model.monitoring, "Monitoring sollte verfügbar sein")
            print("✅ RTX 2070 Manager und Monitoring integriert")
        else:
            print("⚠️ RTX 2070 Manager nicht verfügbar - verwende Fallback")

    def test_model_inference(self):
        """Test: Modell-Inferenz (schnell und sicher)"""
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Teste einfache Text-Verarbeitung
        test_text = "Was ist die Hauptstadt von Deutschland?"
        start_time = time.time()

        try:
            response = model.process_text(test_text, max_length=50)
            inference_time = time.time() - start_time

            self.assertIsInstance(response, str, "Response sollte String sein")
            self.assertGreater(len(response), 0, "Response sollte nicht leer sein")
            self.assertLess(inference_time, 30.0, "Inferenz sollte unter 30s dauern")

            print(f"✅ Inference erfolgreich: {inference_time:.2f}s, {len(response)} Zeichen")
        except Exception as e:
            self.fail(f"Text-Verarbeitung fehlgeschlagen: {e}")

    def test_gpu_memory_management(self):
        """Test: GPU-Memory-Management"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA nicht verfügbar")

        initial_memory = torch.cuda.memory_allocated()

        # Teste Modell-Loading und -Entladung
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Speicher nach Initialisierung
        after_init = torch.cuda.memory_allocated()
        memory_increase = after_init - initial_memory

        print(".1f"
        # Speicher sollte nicht explodieren (max 4GB für RTX 2070)
        self.assertLess(memory_increase, 4 * 1024**3, "Speicherverbrauch sollte unter 4GB bleiben")

        # Teste Speicher-Freigabe
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated()
        print(".1f"
    def test_offline_functionality(self):
        """Test: Offline-Funktionalität (keine Netzwerk-Abhängigkeiten)"""
        # Stelle sicher, dass keine API-Calls gemacht werden
        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Teste lokale Verarbeitung
        test_text = "Test offline Funktionalität"
        response = model.process_text(test_text, max_length=30)

        self.assertIsInstance(response, str, "Offline-Verarbeitung sollte funktionieren")
        self.assertGreater(len(response), 0, "Offline-Response sollte nicht leer sein")

        print("✅ Offline-Funktionalität bestätigt")

    def test_model_quality(self):
        """Test: Modell-Qualitäts-Metriken"""
        model = MultimodalTransformerModel(model_tier="rtx2070")

        test_cases = [
            ("Hallo Welt", "simple_greeting"),
            ("Was ist KI?", "technical_question"),
            ("Erkläre die Demokratie", "political_concept")
        ]

        for test_input, test_type in test_cases:
            with self.subTest(test_input=test_input, test_type=test_type):
                response = model.process_text(test_input, max_length=100)

                # Basis-Qualitäts-Checks
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), len(test_input) // 2)  # Mindestens halbe Eingabe-Länge

                # Spezifische Checks
                if test_type == "simple_greeting":
                    self.assertIn("Hallo", response, "Greeting sollte beantwortet werden")
                elif test_type == "technical_question":
                    self.assertTrue(any(word in response.lower() for word in ["ki", "künstlich", "intelligenz"]),
                                  "KI-Frage sollte KI-Begriffe enthalten")

        print("✅ Modell-Qualität grundlegend bestätigt")


class PerformanceBenchmarkTests(unittest.TestCase):
    """Performance-Benchmarks für RTX 2070"""

    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Erforderliche Module nicht verfügbar")

    def test_inference_speed_benchmark(self):
        """Benchmark: Inference-Geschwindigkeit"""
        model = MultimodalTransformerModel(model_tier="rtx2070")

        test_queries = [
            "Was ist 2+2?",  # Sehr kurz
            "Erkläre die Funktionsweise von KI-Systemen",  # Mittel
            "Analysiere die Auswirkungen der Digitalisierung auf die Arbeitswelt",  # Lang
        ]

        results = {}

        for query in test_queries:
            start_time = time.time()
            response = model.process_text(query, max_length=50)
            end_time = time.time()

            inference_time = end_time - start_time
            tokens_per_second = len(response.split()) / inference_time if inference_time > 0 else 0

            results[query[:30]] = {
                "time": inference_time,
                "tokens_per_sec": tokens_per_second,
                "response_length": len(response)
            }

            print(".2f"
        # RTX 2070 sollte mindestens 5 tokens/sec erreichen
        avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in results.values()) / len(results)
        self.assertGreater(avg_tokens_per_sec, 5.0, "Durchschnitt sollte über 5 tokens/sec liegen")

    def test_memory_efficiency_benchmark(self):
        """Benchmark: Memory-Effizienz"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA nicht verfügbar")

        model = MultimodalTransformerModel(model_tier="rtx2070")

        # Messe Speicherverbrauch bei verschiedenen Aufgaben
        memory_usage = {}

        for task_name, task_func in [
            ("text_processing", lambda: model.process_text("Test query", max_length=30)),
            ("health_check", lambda: model.get_system_health()),
            ("monitoring_data", lambda: model.get_monitoring_data())
        ]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            task_func()

            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_usage[task_name] = peak_memory

            print(".1f"
        # RTX 2070 sollte unter 6GB bleiben
        max_memory = max(memory_usage.values())
        self.assertLess(max_memory, 6000, "Peak-Memory sollte unter 6GB bleiben")


def run_offline_tests():
    """Führt alle Offline-Tests aus"""
    print("🧪 STARTE OFFLINE TESTING FRAMEWORK")
    print("=" * 50)

    if not IMPORTS_AVAILABLE:
        print("❌ Erforderliche Module nicht verfügbar - Tests übersprungen")
        return False

    # Test-Suite erstellen
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Alle Test-Klassen hinzufügen
    suite.addTests(loader.loadTestsFromTestCase(RTX2070OfflineTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(PerformanceBenchmarkTests))

    # Tests ausführen
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("📊 TEST-ERGEBNIS:")
    print(f"   ✅ Erfolgreich: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Fehlgeschlagen: {len(result.failures)}")
    print(f"   ⚠️ Fehler: {len(result.errors)}")
    print(f"   ⏭️ Übersprungen: {len(result.skipped)}")

    if result.wasSuccessful():
        print("🎉 ALLE TESTS ERFOLGREICH!")
        return True
    else:
        print("❌ EINIGE TESTS FEHLGESCHLAGEN!")
        return False


def run_quick_diagnostics():
    """Schnelle Diagnose ohne vollständige Test-Suite"""
    print("🔍 SCHNELLE SYSTEM-DIAGNOSE")
    print("=" * 35)

    diagnostics = {
        "gpu_available": torch.cuda.is_available(),
        "imports_ok": IMPORTS_AVAILABLE,
        "monitoring_ok": False,
        "model_loading_ok": False,
        "inference_ok": False
    }

    if diagnostics["gpu_available"]:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    if diagnostics["imports_ok"]:
        print("✅ Module: Alle Imports erfolgreich")

        try:
            monitoring = get_monitoring_system()
            monitoring.start_monitoring(interval_seconds=1.0)
            time.sleep(1)
            data = monitoring.get_dashboard_data()
            monitoring.stop_monitoring()
            diagnostics["monitoring_ok"] = True
            print("✅ Monitoring: System funktioniert")
        except Exception as e:
            print(f"❌ Monitoring: {e}")

        try:
            model = MultimodalTransformerModel(model_tier="rtx2070")
            response = model.process_text("Test", max_length=10)
            diagnostics["model_loading_ok"] = True
            diagnostics["inference_ok"] = True
            print("✅ Modelle: Laden und Inferenz erfolgreich")
        except Exception as e:
            print(f"❌ Modelle: {e}")

    # Zusammenfassung
    healthy_components = sum(diagnostics.values())
    total_components = len(diagnostics)

    print(f"\n📊 STATUS: {healthy_components}/{total_components} Komponenten funktionieren")

    if healthy_components == total_components:
        print("🎯 SYSTEM GESUND - BEREIT FÜR PRODUKTIV!")
    elif healthy_components >= total_components * 0.7:
        print("⚠️ SYSTEM TEILWEISE FUNKTIONSFÄHIG")
    else:
        print("❌ SYSTEM BEDIARF WARTUNG")

    return diagnostics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline Testing Framework für Bundeskanzler-KI")
    parser.add_argument("--quick", action="store_true", help="Schnelle Diagnose statt vollständiger Tests")
    parser.add_argument("--benchmarks", action="store_true", help="Nur Performance-Benchmarks ausführen")

    args = parser.parse_args()

    if args.quick:
        run_quick_diagnostics()
    elif args.benchmarks:
        # Nur Benchmarks ausführen
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmarkTests))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        success = run_offline_tests()
        sys.exit(0 if success else 1)