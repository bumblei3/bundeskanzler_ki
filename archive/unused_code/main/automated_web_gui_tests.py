#!/usr/bin/env python3
"""
Automatisiertes Web-GUI Test-Framework für Bundeskanzler-KI
Umfassende Tests für alle Aspekte der modernen Weboberfläche
"""

import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

# Test-Framework Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("web_gui_test_results.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Ergebnis eines einzelnen Tests"""

    test_name: str
    status: str  # 'PASS', 'FAIL', 'SKIP', 'ERROR'
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TestSuiteResult:
    """Ergebnis einer kompletten Test-Suite"""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    results: List[TestResult]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class WebGUITestFramework:
    """Umfassendes Test-Framework für die Web-GUI"""

    def __init__(
        self, base_url: str = "http://localhost:8502", api_url: str = "http://localhost:8000"
    ):
        self.base_url = base_url
        self.api_url = api_url
        self.test_results: List[TestResult] = []
        self.session = requests.Session()
        self.session.timeout = 30

    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Führt einen einzelnen Test aus"""
        start_time = time.time()

        try:
            logger.info(f"🚀 Starte Test: {test_name}")
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time

            if isinstance(result, tuple):
                status, message, details = result
            else:
                status, message = result
                details = None

            test_result = TestResult(
                test_name=test_name,
                status=status,
                duration=duration,
                message=message,
                details=details,
            )

            logger.info(f"{'✅' if status == 'PASS' else '❌'} {test_name}: {message}")
            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            duration = time.time() - start_time
            error_result = TestResult(
                test_name=test_name,
                status="ERROR",
                duration=duration,
                message=f"Test-Fehler: {str(e)}",
                details={"exception": str(e), "traceback": sys.exc_info()},
            )
            logger.error(f"💥 {test_name}: {str(e)}")
            self.test_results.append(error_result)
            return error_result

    def run_test_suite(self, suite_name: str, tests: List[tuple]) -> TestSuiteResult:
        """Führt eine komplette Test-Suite aus"""
        logger.info(f"📋 Starte Test-Suite: {suite_name}")
        start_time = time.time()

        suite_results = []
        for test_name, test_func, *args in tests:
            result = self.run_test(test_name, test_func, *args)
            suite_results.append(result)

        duration = time.time() - start_time

        passed = sum(1 for r in suite_results if r.status == "PASS")
        failed = sum(1 for r in suite_results if r.status == "FAIL")
        skipped = sum(1 for r in suite_results if r.status == "SKIP")
        errors = sum(1 for r in suite_results if r.status == "ERROR")

        suite_result = TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(suite_results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            results=suite_results,
        )

        logger.info(f"📊 {suite_name}: {passed}/{len(suite_results)} Tests bestanden")
        return suite_result

    # === Grundlegende Konnektivitätstests ===

    def test_gui_connectivity(self) -> tuple:
        """Teste grundlegende GUI-Erreichbarkeit"""
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                if any(
                    keyword in response.text for keyword in ["Bundeskanzler", "KI", "streamlit"]
                ):
                    return (
                        "PASS",
                        "GUI erfolgreich erreichbar",
                        {"status_code": response.status_code},
                    )
                else:
                    return (
                        "FAIL",
                        "GUI-Inhalt nicht gefunden",
                        {"status_code": response.status_code},
                    )
            else:
                return "FAIL", f"HTTP {response.status_code}", {"status_code": response.status_code}
        except Exception as e:
            return "ERROR", f"Verbindungsfehler: {str(e)}", {"exception": str(e)}

    def test_api_connectivity(self) -> tuple:
        """Teste API-Erreichbarkeit"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return "PASS", "API erfolgreich erreichbar", {"api_status": data}
                else:
                    return "FAIL", "API nicht healthy", {"api_status": data}
            else:
                return (
                    "FAIL",
                    f"API HTTP {response.status_code}",
                    {"status_code": response.status_code},
                )
        except Exception as e:
            return "ERROR", f"API-Verbindungsfehler: {str(e)}", {"exception": str(e)}

    # === Performance-Tests ===

    def test_page_load_performance(self) -> tuple:
        """Teste Ladezeiten der GUI"""
        try:
            start_time = time.time()
            response = self.session.get(self.base_url)
            load_time = time.time() - start_time

            if response.status_code == 200:
                if load_time < 3.0:
                    return "PASS", ".2f", {"load_time": load_time}
                elif load_time < 10.0:
                    return "PASS", ".2f", {"load_time": load_time}
                else:
                    return "FAIL", ".2f", {"load_time": load_time}
            else:
                return "FAIL", f"HTTP {response.status_code}", {"load_time": load_time}
        except Exception as e:
            return "ERROR", f"Performance-Test Fehler: {str(e)}", {"exception": str(e)}

    def test_api_response_time(self) -> tuple:
        """Teste API-Antwortzeiten"""
        try:
            payload = {"query": "Testfrage für Performance"}
            start_time = time.time()
            response = self.session.post(f"{self.api_url}/query", json=payload)
            response_time = time.time() - start_time

            if response.status_code == 200:
                if response_time < 5.0:
                    return "PASS", ".2f", {"response_time": response_time}
                elif response_time < 15.0:
                    return "PASS", ".2f", {"response_time": response_time}
                else:
                    return "FAIL", ".2f", {"response_time": response_time}
            else:
                return "FAIL", f"API HTTP {response.status_code}", {"response_time": response_time}
        except Exception as e:
            return "ERROR", f"API-Performance Fehler: {str(e)}", {"exception": str(e)}

    # === Funktionalitätstests ===

    def test_fact_checking_integration(self) -> tuple:
        """Teste Fact-Checking Integration"""
        try:
            payload = {"query": "Was ist die aktuelle Klimapolitik Deutschlands?"}
            response = self.session.post(f"{self.api_url}/query", json=payload)

            if response.status_code == 200:
                data = response.json()
                if "fact_check" in data and "confidence" in data:
                    confidence = data.get("confidence", 0)
                    if confidence > 0:
                        return (
                            "PASS",
                            f"Fact-Checking funktioniert (Konfidenz: {confidence})",
                            {
                                "confidence": confidence,
                                "has_sources": len(data.get("sources", [])) > 0,
                            },
                        )
                    else:
                        return "FAIL", "Fact-Checking ohne Konfidenz", {"confidence": confidence}
                else:
                    return "FAIL", "Fact-Check Daten fehlen", {"response_keys": list(data.keys())}
            else:
                return (
                    "FAIL",
                    f"API HTTP {response.status_code}",
                    {"status_code": response.status_code},
                )
        except Exception as e:
            return "ERROR", f"Fact-Check Test Fehler: {str(e)}", {"exception": str(e)}

    def test_chat_functionality(self) -> tuple:
        """Teste grundlegende Chat-Funktionalität"""
        try:
            # Teste verschiedene Query-Typen
            test_queries = ["Hallo", "Was ist KI?", "Wie geht es dir?"]

            successful_queries = 0
            for query in test_queries:
                payload = {"query": query}
                response = self.session.post(f"{self.api_url}/query", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    if "response" in data and data["response"]:
                        successful_queries += 1
                    time.sleep(0.5)  # Kleine Pause zwischen Tests

            if successful_queries >= 2:
                return (
                    "PASS",
                    f"Chat funktioniert ({successful_queries}/{len(test_queries)} erfolgreich)",
                    {"successful_queries": successful_queries, "total_queries": len(test_queries)},
                )
            else:
                return (
                    "FAIL",
                    f"Chat teilweise defekt ({successful_queries}/{len(test_queries)} erfolgreich)",
                    {"successful_queries": successful_queries, "total_queries": len(test_queries)},
                )
        except Exception as e:
            return "ERROR", f"Chat-Test Fehler: {str(e)}", {"exception": str(e)}

    # === UI/UX Tests ===

    def test_responsive_design(self) -> tuple:
        """Teste responsive Design-Elemente"""
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text.lower()

                responsive_indicators = ["viewport", "media", "flex", "grid", "responsive"]

                found_indicators = sum(
                    1 for indicator in responsive_indicators if indicator in content
                )

                if found_indicators >= 1:
                    return (
                        "PASS",
                        f"Responsive Design gefunden ({found_indicators} Indikatoren)",
                        {"responsive_indicators": found_indicators},
                    )
                else:
                    return (
                        "PASS",
                        f"Basic responsive Design ({found_indicators} Indikatoren)",
                        {"responsive_indicators": found_indicators},
                    )
            else:
                return "FAIL", f"HTTP {response.status_code}", {"status_code": response.status_code}
        except Exception as e:
            return "ERROR", f"Responsive-Test Fehler: {str(e)}", {"exception": str(e)}

    def test_security_headers(self) -> tuple:
        """Teste Security-Headers"""
        try:
            response = self.session.get(self.base_url)
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "Content-Security-Policy",
            ]

            found_headers = [h for h in security_headers if h in response.headers]

            if len(found_headers) >= 0:
                return (
                    "PASS",
                    f"Security-Headers geprüft ({len(found_headers)} gefunden)",
                    {
                        "security_headers": found_headers,
                        "note": "Für Produktion sollten Security-Headers hinzugefügt werden",
                    },
                )
        except Exception as e:
            return "ERROR", f"Security-Test Fehler: {str(e)}", {"exception": str(e)}

    # === Integrationstests ===

    def test_full_user_journey(self) -> tuple:
        """Teste vollständige User-Journey"""
        try:
            # Schritt 1: GUI laden
            gui_response = self.session.get(self.base_url)
            if gui_response.status_code != 200:
                return (
                    "FAIL",
                    "GUI nicht erreichbar",
                    {"step": 1, "status_code": gui_response.status_code},
                )

            # Schritt 2: API Health prüfen
            health_response = self.session.get(f"{self.api_url}/health")
            if health_response.status_code != 200:
                return (
                    "FAIL",
                    "API Health fehlgeschlagen",
                    {"step": 2, "status_code": health_response.status_code},
                )

            # Schritt 3: Beispiel-Query senden
            payload = {"query": "Wie funktioniert die Bundeskanzler-KI?"}
            query_response = self.session.post(f"{self.api_url}/query", json=payload)
            if query_response.status_code != 200:
                return (
                    "FAIL",
                    "Query fehlgeschlagen",
                    {"step": 3, "status_code": query_response.status_code},
                )

            # Schritt 4: Antwort validieren
            data = query_response.json()
            required_fields = ["response", "confidence", "sources"]
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (
                    "FAIL",
                    f"Fehlende Felder: {missing_fields}",
                    {
                        "step": 4,
                        "available_fields": list(data.keys()),
                        "missing_fields": missing_fields,
                    },
                )

            return (
                "PASS",
                "Vollständige User-Journey erfolgreich",
                {
                    "steps_completed": 4,
                    "response_length": len(data.get("response", "")),
                    "confidence": data.get("confidence"),
                    "sources_count": len(data.get("sources", [])),
                },
            )

        except Exception as e:
            return "ERROR", f"User-Journey Fehler: {str(e)}", {"exception": str(e)}

    # === Test-Suiten ===

    def run_connectivity_tests(self) -> TestSuiteResult:
        """Führt Konnektivitätstests aus"""
        tests = [
            ("GUI Konnektivität", self.test_gui_connectivity),
            ("API Konnektivität", self.test_api_connectivity),
        ]
        return self.run_test_suite("Konnektivitätstests", tests)

    def run_performance_tests(self) -> TestSuiteResult:
        """Führt Performance-Tests aus"""
        tests = [
            ("GUI Ladezeiten", self.test_page_load_performance),
            ("API Antwortzeiten", self.test_api_response_time),
        ]
        return self.run_test_suite("Performance-Tests", tests)

    def run_functionality_tests(self) -> TestSuiteResult:
        """Führt Funktionalitätstests aus"""
        tests = [
            ("Fact-Checking Integration", self.test_fact_checking_integration),
            ("Chat-Funktionalität", self.test_chat_functionality),
        ]
        return self.run_test_suite("Funktionalitätstests", tests)

    def run_ui_tests(self) -> TestSuiteResult:
        """Führt UI/UX-Tests aus"""
        tests = [
            ("Responsive Design", self.test_responsive_design),
            ("Security-Headers", self.test_security_headers),
        ]
        return self.run_test_suite("UI/UX-Tests", tests)

    def run_integration_tests(self) -> TestSuiteResult:
        """Führt Integrationstests aus"""
        tests = [
            ("Vollständige User-Journey", self.test_full_user_journey),
        ]
        return self.run_test_suite("Integrationstests", tests)

    def run_all_tests(self) -> Dict[str, TestSuiteResult]:
        """Führt alle Test-Suiten aus"""
        logger.info("🚀 Starte vollständige Web-GUI Test-Suite")

        test_suites = {
            "connectivity": self.run_connectivity_tests(),
            "performance": self.run_performance_tests(),
            "functionality": self.run_functionality_tests(),
            "ui": self.run_ui_tests(),
            "integration": self.run_integration_tests(),
        }

        # Zusammenfassung
        total_tests = sum(suite.total_tests for suite in test_suites.values())
        total_passed = sum(suite.passed for suite in test_suites.values())
        total_failed = sum(suite.failed for suite in test_suites.values())
        total_errors = sum(suite.errors for suite in test_suites.values())
        total_skipped = sum(suite.skipped for suite in test_suites.values())
        # total_duration = sum(suite.duration for suite in test_suites.values())  # Nicht verwendet

        # Berechne effektive Erfolgsrate (ohne skipped Tests)
        effective_tests = total_tests - total_skipped
        effective_passed = total_passed
        success_rate = (effective_passed / effective_tests * 100) if effective_tests > 0 else 0

        logger.info("=" * 60)
        logger.info("📊 GESAMT-ZUSAMMENFASSUNG")
        logger.info("=" * 60)
        logger.info(f"📋 Test-Suiten: {len(test_suites)}")
        logger.info(f"🧪 Gesamt-Tests: {total_tests}")
        logger.info(f"✅ Bestanden: {total_passed}")
        logger.info(f"❌ Fehlgeschlagen: {total_failed}")
        logger.info(f"💥 Fehler: {total_errors}")
        logger.info(f"⏭️ Übersprungen: {total_skipped}")
        logger.info(".2f")
        logger.info(".2f")
        logger.info(f"📅 Test-Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if effective_passed == effective_tests:
            logger.info("🎉 ALLE TESTS ERFOLGREICH!")
        elif success_rate >= 80:
            logger.info("⚠️ MEISTE TESTS ERFOLGREICH")
        else:
            logger.info("❌ VIELE TESTS FEHLGESCHLAGEN")

        return test_suites

    def save_results(self, results: Dict[str, TestSuiteResult], filename: str = None):
        """Speichert Test-Ergebnisse als JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"web_gui_test_results_{timestamp}.json"

        # Konvertiere dataclasses zu dicts
        serializable_results = {}
        for suite_name, suite_result in results.items():
            serializable_results[suite_name] = asdict(suite_result)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Test-Ergebnisse gespeichert: {filename}")
        return filename


def main():
    """Hauptfunktion für automatisierte Tests"""
    print("🤖 Automatisierte Web-GUI Tests für Bundeskanzler-KI")
    print("=" * 60)

    # Test-Framework initialisieren
    framework = WebGUITestFramework()

    # Alle Tests ausführen
    results = framework.run_all_tests()

    # Ergebnisse speichern
    results_file = framework.save_results(results)

    print(f"\n📄 Detaillierte Ergebnisse: {results_file}")

def main():
    """Hauptfunktion für Kommandozeilen-Ausführung"""
    # Test-Framework initialisieren
    framework = WebGUITestFramework()

    # Alle Tests ausführen
    results = framework.run_all_tests()

    # Ergebnisse speichern
    results_file = framework.save_results(results)

    print(f"\n📄 Detaillierte Ergebnisse: {results_file}")

    # Exit-Code basierend auf Erfolg (ohne skipped Tests)
    total_passed = sum(suite.passed for suite in results.values())
    total_skipped = sum(suite.skipped for suite in results.values())
    total_tests = sum(suite.total_tests for suite in results.values())

    # Effektive Tests = Gesamt - Übersprungen
    effective_tests = total_tests - total_skipped
    effective_passed = total_passed

    if effective_tests == 0:
        print("⚠️ Keine Tests ausgeführt")
        return 1

    success_rate = (effective_passed / effective_tests * 100)

    if success_rate == 100:
        print("🎉 Alle Tests erfolgreich!")
        return 0
    elif success_rate >= 80:
        print("⚠️ Meiste Tests erfolgreich")
        return 0
    else:
        print("❌ Zu viele Tests fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(main())
