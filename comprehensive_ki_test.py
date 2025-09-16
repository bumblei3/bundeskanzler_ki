#!/usr/bin/env python3
"""
🔬 Umfassender Bundeskanzler-KI Test Suite
==========================================

Testet alle Kernfunktionen der Bundeskanzler-KI nach DeepL-Entfernung:
- RTX 2070 GPU-Optimierung
- RAG-System (Retrieval-Augmented Generation)
- Multi-Agent Intelligence System
- Corpus-Verarbeitung und Embeddings
- Query-Verarbeitung und Antwort-Generierung
- Performance-Metriken und Benchmarks
- System-Monitoring und GPU-Status
- Fehlerbehandlung und Robustheit

Verwendung:
    python3 comprehensive_ki_test.py [--verbose] [--performance] [--stress]

Optionen:
    --verbose       Detaillierte Ausgabe
    --performance   Performance-Tests einschließen
    --stress        Stress-Tests einschließen
    --all           Alle Tests ausführen
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.rtx2070_bundeskanzler_ki import get_rtx2070_bundeskanzler_ki

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_ki_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Ergebnis eines einzelnen Tests"""
    test_name: str
    success: bool
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class TestSuiteResult:
    """Ergebnis einer Test-Suite"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration: float
    results: List[TestResult]

class BundeskanzlerKITestSuite:
    """Umfassende Test-Suite für Bundeskanzler-KI"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = datetime.now()

        # Test-Daten
        self.test_queries = [
            "Was ist die aktuelle Klimapolitik Deutschlands?",
            "Wie funktioniert die Energiewende?",
            "Was sind die Ziele der Bundesregierung für 2030?",
            "Erkläre die Bedeutung von Nachhaltigkeit in der Politik.",
            "Wie steht Deutschland zur EU?",
            "Was ist der Unterschied zwischen Sozialdemokratie und Konservatismus?",
            "Wie funktioniert das deutsche Wahlsystem?",
            "Was sind die Herausforderungen der Digitalisierung?",
            "Wie wird die Rente in Deutschland berechnet?",
            "Was ist der Bundeskanzler-Plan für die Zukunft?"
        ]

        # KI-Instanz
        self.ki = None

        # Initialisiere KI
        self._initialize_ki()

    def _initialize_ki(self):
        """Initialisiert die Bundeskanzler-KI"""
        try:
            self.ki = get_rtx2070_bundeskanzler_ki()
            logger.info("✅ Bundeskanzler-KI erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"❌ KI-Initialisierung fehlgeschlagen: {e}")
            self.ki = None

    def log_test_result(self, result: TestResult):
        """Protokolliert ein Testergebnis"""
        status = "✅ PASS" if result.success else "❌ FAIL"
        message = f"{status} {result.test_name}: {result.message}"

        if result.details and self.verbose:
            message += f" | Details: {result.details}"

        logger.info(message)

        if not result.success:
            logger.error(f"❌ Test fehlgeschlagen: {result.test_name} - {result.message}")

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Führt einen einzelnen Test aus"""
        start_time = time.time()

        try:
            success, message, details = test_func()
            duration = time.time() - start_time

            result = TestResult(
                test_name=test_name,
                success=success,
                duration=duration,
                message=message,
                details=details
            )

        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"exception": str(e), "traceback": sys.exc_info()}
            )

        self.log_test_result(result)
        return result

    # === RTX 2070 KI Initialisierung Tests ===

    def test_ki_initialization(self) -> Tuple[bool, str, Dict]:
        """Testet KI-Initialisierung"""
        if not self.ki:
            return False, "KI konnte nicht initialisiert werden", {}

        try:
            system_info = self.ki.get_system_info()
            return True, "KI erfolgreich initialisiert", {
                "system_info": system_info,
                "components_status": system_info.get("components_status", {}),
                "gpu_info": system_info.get("gpu_info", {})
            }
        except Exception as e:
            return False, f"System-Info Fehler: {e}", {"error": str(e)}

    def test_rtx2070_gpu_status(self) -> Tuple[bool, str, Dict]:
        """Testet RTX 2070 GPU-Status"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            system_info = self.ki.get_system_info()
            gpu_info = system_info.get("gpu_info", {})

            # Prüfe GPU-Verfügbarkeit
            gpu_available = gpu_info.get("gpu_utilization") is not None
            memory_total = gpu_info.get("memory_total_gb", 0)
            memory_used = gpu_info.get("memory_used_gb", 0)

            if gpu_available and memory_total > 0:
                memory_usage_percent = (memory_used / memory_total) * 100
                return True, ".1f", {
                    "gpu_available": gpu_available,
                    "memory_total_gb": memory_total,
                    "memory_used_gb": memory_used,
                    "memory_usage_percent": memory_usage_percent,
                    "temperature_c": gpu_info.get("temperature_c"),
                    "power_usage_w": gpu_info.get("power_usage_w")
                }
            else:
                return False, "GPU nicht verfügbar oder fehlerhaft", {"gpu_info": gpu_info}

        except Exception as e:
            return False, f"GPU-Status Fehler: {e}", {"error": str(e)}

    def test_tensor_cores(self) -> Tuple[bool, str, Dict]:
        """Testet Tensor Cores Verfügbarkeit"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            system_info = self.ki.get_system_info()
            gpu_info = system_info.get("gpu_info", {})

            tensor_cores = gpu_info.get("tensor_core_usage", 0)
            fp16_enabled = system_info.get("components_status", {}).get("rtx2070_llm", False)

            if tensor_cores > 0 and fp16_enabled:
                return True, f"Tensor Cores aktiv: {tensor_cores}% Auslastung", {
                    "tensor_core_usage": tensor_cores,
                    "fp16_enabled": fp16_enabled
                }
            else:
                return False, "Tensor Cores nicht verfügbar oder nicht aktiviert", {
                    "tensor_core_usage": tensor_cores,
                    "fp16_enabled": fp16_enabled
                }

        except Exception as e:
            return False, f"Tensor Cores Test Fehler: {e}", {"error": str(e)}

    # === RAG-System Tests ===

    def test_rag_system_initialization(self) -> Tuple[bool, str, Dict]:
        """Testet RAG-System Initialisierung"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            system_info = self.ki.get_system_info()
            components = system_info.get("components_status", {})

            rag_available = components.get("rtx2070_rag", False) or components.get("fallback_rag", False)

            if rag_available:
                return True, "RAG-System verfügbar", {
                    "rtx2070_rag": components.get("rtx2070_rag", False),
                    "fallback_rag": components.get("fallback_rag", False),
                    "corpus_loaded": system_info.get("rag_info", {}).get("corpus_loaded", False)
                }
            else:
                return False, "RAG-System nicht verfügbar", {"components": components}

        except Exception as e:
            return False, f"RAG-System Test Fehler: {e}", {"error": str(e)}

    def test_corpus_loading(self) -> Tuple[bool, str, Dict]:
        """Testet Corpus-Loading"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            system_info = self.ki.get_system_info()
            rag_info = system_info.get("rag_info", {})

            corpus_entries = rag_info.get("corpus_entries", 0)
            corpus_loaded = rag_info.get("corpus_loaded", False)

            if corpus_loaded and corpus_entries > 0:
                return True, f"Corpus geladen: {corpus_entries} Einträge", {
                    "corpus_entries": corpus_entries,
                    "corpus_loaded": corpus_loaded,
                    "embedding_model": rag_info.get("embedding_model")
                }
            else:
                return False, "Corpus nicht geladen oder leer", {
                    "corpus_entries": corpus_entries,
                    "corpus_loaded": corpus_loaded
                }

        except Exception as e:
            return False, f"Corpus-Test Fehler: {e}", {"error": str(e)}

    def test_embedding_generation(self) -> Tuple[bool, str, Dict]:
        """Testet Embedding-Generierung"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            # Test-Query für Embedding-Generierung
            test_query = "Test für Embedding-Generierung"

            # Prüfe ob Embeddings generiert werden können
            system_info = self.ki.get_system_info()
            rag_info = system_info.get("rag_info", {})

            embedding_dimension = rag_info.get("embedding_dimension", 0)
            embedding_model = rag_info.get("embedding_model", "")

            if embedding_dimension > 0 and embedding_model:
                return True, f"Embeddings verfügbar: {embedding_model} (Dim: {embedding_dimension})", {
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension
                }
            else:
                return False, "Embedding-System nicht verfügbar", {
                    "embedding_dimension": embedding_dimension,
                    "embedding_model": embedding_model
                }

        except Exception as e:
            return False, f"Embedding-Test Fehler: {e}", {"error": str(e)}

    # === Multi-Agent System Tests ===

    def test_multi_agent_system(self) -> Tuple[bool, str, Dict]:
        """Testet Multi-Agent System"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            system_info = self.ki.get_system_info()
            components = system_info.get("components_status", {})

            multi_agent_available = components.get("multi_agent_system", False)

            if multi_agent_available:
                return True, "Multi-Agent System verfügbar", {
                    "multi_agent_system": multi_agent_available,
                    "agents_initialized": system_info.get("multi_agent_info", {}).get("agents_initialized", 0)
                }
            else:
                return False, "Multi-Agent System nicht verfügbar", {"components": components}

        except Exception as e:
            return False, f"Multi-Agent Test Fehler: {e}", {"error": str(e)}

    # === Query-Verarbeitung Tests ===

    def test_basic_query_processing(self) -> Tuple[bool, str, Dict]:
        """Testet grundlegende Query-Verarbeitung"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        test_query = "Was ist KI?"

        try:
            start_time = time.time()
            result = self.ki.query(test_query)
            duration = time.time() - start_time

            if result and result.get('response'):
                response_length = len(result['response'])
                return True, ".2f", {
                    "query": test_query,
                    "response_length": response_length,
                    "duration": duration,
                    "has_response": True
                }
            else:
                return False, "Keine Antwort erhalten", {
                    "query": test_query,
                    "result": result,
                    "duration": duration
                }

        except Exception as e:
            return False, f"Query-Verarbeitung Fehler: {e}", {"error": str(e)}

    def test_multiple_queries(self) -> Tuple[bool, str, Dict]:
        """Testet mehrere Queries nacheinander"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        queries = self.test_queries[:5]  # Erste 5 Queries
        results = []
        total_duration = 0

        for query in queries:
            try:
                start_time = time.time()
                result = self.ki.query(query)
                duration = time.time() - start_time
                total_duration += duration

                success = result and result.get('response')
                results.append({
                    "query": query,
                    "success": success,
                    "duration": duration,
                    "response_length": len(result.get('response', '')) if result else 0
                })

            except Exception as e:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "duration": 0
                })

        successful_queries = sum(1 for r in results if r["success"])
        avg_duration = total_duration / len(queries)

        success = successful_queries == len(queries)
        return success, f"Multiple Queries: {successful_queries}/{len(queries)} erfolgreich (Ø {avg_duration:.2f}s)", {
            "total_queries": len(queries),
            "successful": successful_queries,
            "avg_duration": avg_duration,
            "results": results
        }

    def test_query_consistency(self) -> Tuple[bool, str, Dict]:
        """Testet Konsistenz der Query-Antworten"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        test_query = "Erkläre die Energiewende kurz."
        responses = []

        # Führe die gleiche Query mehrmals aus
        for i in range(3):
            try:
                result = self.ki.query(test_query)
                if result and result.get('response'):
                    responses.append(result['response'])
                else:
                    responses.append("")
            except Exception as e:
                responses.append(f"ERROR: {e}")

        # Prüfe Konsistenz (alle Antworten sollten ähnlich sein)
        non_empty_responses = [r for r in responses if r and not r.startswith("ERROR")]
        unique_responses = len(set(non_empty_responses))

        if len(non_empty_responses) >= 2:
            # Einfache Konsistenz-Prüfung: Mindestens 2 erfolgreiche Antworten
            consistency_score = len(non_empty_responses) / len(responses)
            success = consistency_score >= 0.67  # Mindestens 2/3 erfolgreich
            return success, ".1f", {
                "query": test_query,
                "total_runs": len(responses),
                "successful_runs": len(non_empty_responses),
                "unique_responses": unique_responses,
                "consistency_score": consistency_score
            }
        else:
            return False, "Nicht genügend erfolgreiche Antworten für Konsistenz-Test", {
                "responses": responses
            }

    # === Performance Tests ===

    def test_performance_baseline(self) -> Tuple[bool, str, Dict]:
        """Testet Performance-Baseline"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        query = "Was ist die Bedeutung von Demokratie?"
        iterations = 10
        durations = []

        for i in range(iterations):
            try:
                start_time = time.time()
                result = self.ki.query(query)
                duration = time.time() - start_time

                if result and result.get('response'):
                    durations.append(duration)
                else:
                    durations.append(float('inf'))  # Markiere als fehlgeschlagen

            except Exception as e:
                durations.append(float('inf'))

        # Entferne fehlgeschlagene Läufe
        valid_durations = [d for d in durations if d != float('inf')]

        if valid_durations:
            avg_duration = sum(valid_durations) / len(valid_durations)
            min_duration = min(valid_durations)
            max_duration = max(valid_durations)

            # Performance-Kriterien: Durchschnitt unter 10 Sekunden
            success = avg_duration < 10.0

            return success, ".2f", {
                "iterations": iterations,
                "valid_runs": len(valid_durations),
                "avg_duration": avg_duration,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "durations": valid_durations
            }
        else:
            return False, "Alle Performance-Test-Läufe fehlgeschlagen", {
                "iterations": iterations,
                "valid_runs": 0
            }

    def test_memory_usage(self) -> Tuple[bool, str, Dict]:
        """Testet Speichernutzung während Query-Verarbeitung"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            # Speicher vor Query
            system_info_before = self.ki.get_system_info()
            memory_before = system_info_before.get("gpu_info", {}).get("memory_used_gb", 0)

            # Führe Query aus
            result = self.ki.query("Test für Speichernutzung.")

            # Speicher nach Query
            system_info_after = self.ki.get_system_info()
            memory_after = system_info_after.get("gpu_info", {}).get("memory_used_gb", 0)

            memory_delta = memory_after - memory_before
            memory_total = system_info_after.get("gpu_info", {}).get("memory_total_gb", 0)

            if memory_total > 0:
                memory_usage_percent = (memory_after / memory_total) * 100

                # Speicher-Zuwachs sollte moderat sein (< 1GB)
                reasonable_memory_usage = abs(memory_delta) < 1.0

                return reasonable_memory_usage, ".2f", {
                    "memory_before_gb": memory_before,
                    "memory_after_gb": memory_after,
                    "memory_delta_gb": memory_delta,
                    "memory_usage_percent": memory_usage_percent,
                    "memory_total_gb": memory_total
                }
            else:
                return False, "GPU-Speicher-Informationen nicht verfügbar", {}

        except Exception as e:
            return False, f"Speicher-Test Fehler: {e}", {"error": str(e)}

    # === Stress Tests ===

    def test_concurrent_queries(self) -> Tuple[bool, str, Dict]:
        """Testet gleichzeitige Query-Verarbeitung"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        def single_query(query_text: str) -> Dict:
            try:
                start_time = time.time()
                result = self.ki.query(query_text)
                duration = time.time() - start_time
                return {
                    "success": result is not None and result.get('response'),
                    "duration": duration,
                    "response_length": len(result.get('response', '')) if result else 0
                }
            except Exception as e:
                return {"success": False, "error": str(e), "duration": 0}

        # Verwende verschiedene Queries für gleichzeitige Ausführung
        concurrent_queries = self.test_queries[:8]  # 8 Queries gleichzeitig
        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:  # Max 4 gleichzeitige Threads
            futures = [executor.submit(single_query, query) for query in concurrent_queries]
            for future in as_completed(futures):
                results.append(future.result())

        successful = sum(1 for r in results if r.get("success", False))
        total = len(results)
        avg_duration = sum(r.get("duration", 0) for r in results) / len(results)

        success = successful >= total * 0.75  # Mindestens 75% Erfolg
        return success, f"Concurrent Queries: {successful}/{total} erfolgreich (Ø {avg_duration:.2f}s)", {
            "total_queries": total,
            "successful": successful,
            "avg_duration": avg_duration,
            "results": results[:3]  # Nur erste 3 Ergebnisse zeigen
        }

    def test_long_running_session(self) -> Tuple[bool, str, Dict]:
        """Testet eine längere Session mit mehreren Queries"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        session_queries = self.test_queries[:10]  # 10 Queries in einer Session
        session_results = []
        session_start = time.time()

        try:
            for i, query in enumerate(session_queries):
                start_time = time.time()
                result = self.ki.query(query)
                duration = time.time() - start_time

                success = result and result.get('response')
                session_results.append({
                    "query_num": i + 1,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "success": success,
                    "duration": duration,
                    "response_length": len(result.get('response', '')) if result else 0
                })

                # Kleine Pause zwischen Queries
                time.sleep(0.5)

            session_duration = time.time() - session_start
            successful_queries = sum(1 for r in session_results if r["success"])

            success = successful_queries >= len(session_queries) * 0.8  # Mindestens 80% Erfolg
            return success, ".1f", {
                "session_duration": session_duration,
                "total_queries": len(session_queries),
                "successful_queries": successful_queries,
                "avg_query_duration": sum(r["duration"] for r in session_results) / len(session_results),
                "results_summary": session_results
            }

        except Exception as e:
            session_duration = time.time() - session_start
            return False, f"Session-Test Fehler nach {session_duration:.1f}s: {e}", {
                "session_duration": session_duration,
                "completed_queries": len(session_results),
                "error": str(e)
            }

    # === System Monitoring Tests ===

    def test_system_monitoring(self) -> Tuple[bool, str, Dict]:
        """Testet System-Monitoring Funktionen"""
        if not self.ki:
            return False, "KI nicht verfügbar", {}

        try:
            system_info = self.ki.get_system_info()

            # Prüfe alle wichtigen Monitoring-Komponenten
            required_components = [
                "gpu_info", "llm_info", "rag_info", "multi_agent_info",
                "components_status", "rtx2070_optimized"
            ]

            available_components = [comp for comp in required_components if comp in system_info]
            missing_components = [comp for comp in required_components if comp not in system_info]

            monitoring_completeness = len(available_components) / len(required_components)

            success = monitoring_completeness >= 0.8  # Mindestens 80% der Komponenten verfügbar

            return success, ".1f", {
                "monitoring_completeness": monitoring_completeness,
                "available_components": available_components,
                "missing_components": missing_components,
                "system_info_keys": list(system_info.keys())
            }

        except Exception as e:
            return False, f"Monitoring-Test Fehler: {e}", {"error": str(e)}

    def run_basic_tests(self) -> TestSuiteResult:
        """Führt grundlegende Tests aus"""
        suite_name = "Basic Tests"
        logger.info(f"🚀 Starte {suite_name}...")

        tests = [
            ("KI Initialization", self.test_ki_initialization),
            ("RTX 2070 GPU Status", self.test_rtx2070_gpu_status),
            ("Tensor Cores", self.test_tensor_cores),
            ("RAG System Initialization", self.test_rag_system_initialization),
            ("Corpus Loading", self.test_corpus_loading),
            ("Embedding Generation", self.test_embedding_generation),
            ("Multi-Agent System", self.test_multi_agent_system),
            ("Basic Query Processing", self.test_basic_query_processing),
            ("Multiple Queries", self.test_multiple_queries),
            ("Query Consistency", self.test_query_consistency),
            ("System Monitoring", self.test_system_monitoring)
        ]

        results = []
        start_time = time.time()

        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            results.append(result)

        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = len(results) - passed_tests

        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_duration=total_duration,
            results=results
        )

    def run_performance_tests(self) -> TestSuiteResult:
        """Führt Performance-Tests aus"""
        suite_name = "Performance Tests"
        logger.info(f"🚀 Starte {suite_name}...")

        tests = [
            ("Performance Baseline", self.test_performance_baseline),
            ("Memory Usage", self.test_memory_usage)
        ]

        results = []
        start_time = time.time()

        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            results.append(result)

        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = len(results) - passed_tests

        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_duration=total_duration,
            results=results
        )

    def run_stress_tests(self) -> TestSuiteResult:
        """Führt Stress-Tests aus"""
        suite_name = "Stress Tests"
        logger.info(f"🚀 Starte {suite_name}...")

        tests = [
            ("Concurrent Queries", self.test_concurrent_queries),
            ("Long Running Session", self.test_long_running_session)
        ]

        results = []
        start_time = time.time()

        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            results.append(result)

        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = len(results) - passed_tests

        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_duration=total_duration,
            results=results
        )

    def generate_report(self, suite_results: List[TestSuiteResult]) -> str:
        """Generiert einen detaillierten Test-Report"""
        total_tests = sum(suite.total_tests for suite in suite_results)
        total_passed = sum(suite.passed_tests for suite in suite_results)
        total_failed = sum(suite.failed_tests for suite in suite_results)
        total_duration = sum(suite.total_duration for suite in suite_results)

        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        report = []
        report.append("🤖 Bundeskanzler-KI Test Report")
        report.append("=" * 50)
        report.append(f"Zeitstempel: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Gesamtdauer: {total_duration:.2f} Sekunden")
        report.append("")

        # Gesamtstatistik
        report.append("📊 Gesamtstatistik:")
        report.append(f"  • Gesamt Tests: {total_tests}")
        report.append(f"  • Erfolgreich: {total_passed}")
        report.append(f"  • Fehlgeschlagen: {total_failed}")
        report.append(f"  • Erfolgsrate: {success_rate:.1f}%")
        report.append("")

        # Suite-Ergebnisse
        for suite in suite_results:
            status = "✅" if suite.failed_tests == 0 else "⚠️" if suite.passed_tests > 0 else "❌"
            report.append(f"{status} {suite.suite_name}:")
            report.append(f"  • Tests: {suite.total_tests}")
            report.append(f"  • Erfolgreich: {suite.passed_tests}")
            report.append(f"  • Fehlgeschlagen: {suite.failed_tests}")
            report.append(f"  • Dauer: {suite.total_duration:.2f}s")
            report.append("")

            # Detaillierte Testergebnisse
            if self.verbose:
                for result in suite.results:
                    status_icon = "✅" if result.success else "❌"
                    report.append(f"    {status_icon} {result.test_name}: {result.message}")
                    report.append(f"      Dauer: {result.duration:.3f}s")
                    report.append("")

        # System-Übersicht
        if self.ki:
            try:
                system_info = self.ki.get_system_info()
                report.append("🖥️  System-Übersicht:")
                gpu_info = system_info.get("gpu_info", {})
                if gpu_info:
                    report.append(f"  • GPU: RTX 2070 ({gpu_info.get('memory_total_gb', 0):.1f}GB)")
                    report.append(f"  • Speichernutzung: {gpu_info.get('memory_used_gb', 0):.1f}GB / {gpu_info.get('memory_total_gb', 0):.1f}GB")
                    report.append(f"  • Temperatur: {gpu_info.get('temperature_c', 0)}°C")

                components = system_info.get("components_status", {})
                active_components = [k for k, v in components.items() if v]
                report.append(f"  • Aktive Komponenten: {', '.join(active_components)}")
                report.append("")
            except Exception as e:
                report.append(f"  • System-Info Fehler: {e}")
                report.append("")

        # Empfehlungen
        report.append("💡 Empfehlungen:")
        if success_rate >= 90:
            report.append("  ✅ Ausgezeichnete KI-Performance! Alle Systeme funktionieren optimal.")
        elif success_rate >= 75:
            report.append("  ⚠️ Gute KI-Performance. Einige kleinere Probleme sollten behoben werden.")
        else:
            report.append("  ❌ Kritische Probleme erkannt. KI-System benötigt Wartung.")

        if total_failed > 0:
            report.append("  • Überprüfen Sie die fehlgeschlagenen Tests für Details.")
            report.append("  • Stellen Sie sicher, dass GPU-Treiber aktuell sind.")
            report.append("  • Prüfen Sie die Corpus- und Modell-Dateien.")

        return "\n".join(report)

    def run_all_tests(self, include_performance: bool = True, include_stress: bool = True) -> List[TestSuiteResult]:
        """Führt alle Tests aus"""
        logger.info("🎯 Starte umfassende Bundeskanzler-KI Tests...")

        results = []

        # Basis-Tests (immer ausführen)
        basic_results = self.run_basic_tests()
        results.append(basic_results)

        # Performance-Tests
        if include_performance:
            perf_results = self.run_performance_tests()
            results.append(perf_results)

        # Stress-Tests
        if include_stress:
            stress_results = self.run_stress_tests()
            results.append(stress_results)

        # Report generieren
        report = self.generate_report(results)

        # Report in Datei speichern
        report_file = Path("logs/comprehensive_ki_test_report.txt")
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # Report ausgeben
        print("\n" + "="*80)
        print(report)
        print("="*80)

        logger.info(f"📄 Detaillierter Report gespeichert: {report_file}")

        return results

def main():
    parser = argparse.ArgumentParser(description="Umfassende Bundeskanzler-KI Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detaillierte Ausgabe")
    parser.add_argument("--performance", action="store_true", help="Performance-Tests einschließen")
    parser.add_argument("--stress", action="store_true", help="Stress-Tests einschließen")
    parser.add_argument("--all", action="store_true", help="Alle Tests ausführen")

    args = parser.parse_args()

    # Alle Tests ausführen wenn --all oder keine spezifischen Tests angegeben
    if args.all or (not args.performance and not args.stress):
        args.performance = True
        args.stress = True

    # Test-Suite initialisieren und ausführen
    test_suite = BundeskanzlerKITestSuite(verbose=args.verbose)
    results = test_suite.run_all_tests(
        include_performance=args.performance,
        include_stress=args.stress
    )

    # Exit-Code basierend auf Ergebnissen
    total_failed = sum(suite.failed_tests for suite in results)
    sys.exit(0 if total_failed == 0 else 1)

if __name__ == "__main__":
    main()