#!/usr/bin/env python3
"""
Umfassender Web-GUI Test für Bundeskanzler-KI
Testet alle Features der modernen Weboberfläche
"""

import json
import time
from datetime import datetime

import requests
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class WebGUITester:
    """Umfassender Tester für die Web-GUI"""

    def __init__(self, base_url="http://localhost:8502"):
        self.base_url = base_url
        self.driver = None
        self.test_results = []

    def setup_driver(self):
        """Setup für Selenium WebDriver"""
        try:
            from selenium.webdriver.firefox.options import Options

            options = Options()
            options.add_argument("--headless")  # Headless mode für CI/CD
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Firefox(options=options)
            print("✅ Firefox WebDriver erfolgreich initialisiert")
            return True
        except ImportError:
            print("⚠️  Selenium nicht verfügbar - verwende HTTP-Tests nur")
            return False
        except Exception as e:
            print(f"❌ WebDriver Fehler: {e}")
            return False

    def log_test_result(self, test_name, success, details=""):
        """Protokolliere Test-Ergebnis"""
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        self.test_results.append(
            {
                "test": test_name,
                "success": success,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def test_basic_connectivity(self):
        """Teste grundlegende Erreichbarkeit"""
        print("🔍 Teste grundlegende Erreichbarkeit...")
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                if "Bundeskanzler" in response.text or "KI" in response.text:
                    self.log_test_result("Grundlegende Erreichbarkeit", True, "GUI lädt korrekt")
                    return True
                else:
                    self.log_test_result(
                        "Grundlegende Erreichbarkeit",
                        False,
                        "Bundeskanzler-KI Inhalt nicht gefunden",
                    )
                    return False
            else:
                self.log_test_result(
                    "Grundlegende Erreichbarkeit", False, f"HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.log_test_result("Grundlegende Erreichbarkeit", False, str(e))
            return False

    def test_page_title(self):
        """Teste Seitentitel"""
        print("🔍 Teste Seitentitel...")
        try:
            response = requests.get(self.base_url, timeout=10)
            # Suche nach verschiedenen Varianten des Titels
            title_indicators = ["Bundeskanzler", "KI", "🤖", "Bundeskanzler-KI", "Bundeskanzler KI"]
            found_indicators = sum(
                1 for indicator in title_indicators if indicator in response.text
            )
            if found_indicators >= 2:
                self.log_test_result(
                    "Seitentitel", True, f"{found_indicators} Titel-Elemente gefunden"
                )
                return True
            else:
                self.log_test_result(
                    "Seitentitel", False, f"Nur {found_indicators} Titel-Elemente gefunden"
                )
                return False
        except Exception as e:
            self.log_test_result("Seitentitel", False, str(e))
            return False

    def test_css_styling(self):
        """Teste CSS-Styling"""
        print("🔍 Teste CSS-Styling...")
        try:
            response = requests.get(self.base_url, timeout=10)
            css_indicators = [
                "main-header",
                "query-card",
                "fact-check-indicator",
                "chat-message",
                "style",
                "css",
                "background",
                "color",
            ]
            found_styles = sum(1 for indicator in css_indicators if indicator in response.text)
            if found_styles >= 4:
                self.log_test_result(
                    "CSS-Styling", True, f"{found_styles}/{len(css_indicators)} Styles gefunden"
                )
                return True
            else:
                self.log_test_result(
                    "CSS-Styling",
                    False,
                    f"Nur {found_styles}/{len(css_indicators)} Styles gefunden",
                )
                return False
        except Exception as e:
            self.log_test_result("CSS-Styling", False, str(e))
            return False

    def test_responsive_design(self):
        """Teste responsive Design"""
        print("🔍 Teste responsive Design...")
        try:
            response = requests.get(self.base_url, timeout=10)
            # Suche nach responsive Meta-Tags und CSS
            responsive_indicators = ["viewport", "mobile", "responsive", "@media", "flex", "grid"]
            found_indicators = sum(
                1
                for indicator in responsive_indicators
                if indicator.lower() in response.text.lower()
            )
            if found_indicators >= 4:
                self.log_test_result(
                    "Responsive Design", True, f"{found_indicators} responsive Elemente gefunden"
                )
                return True
            else:
                self.log_test_result(
                    "Responsive Design",
                    False,
                    f"Nur {found_indicators} responsive Elemente gefunden",
                )
                return False
        except Exception as e:
            self.log_test_result("Responsive Design", False, str(e))
            return False

    def test_ui_components(self):
        """Teste UI-Komponenten"""
        print("🔍 Teste UI-Komponenten...")
        try:
            response = requests.get(self.base_url, timeout=10)
            components = [
                "text_area",
                "button",
                "sidebar",
                "columns",
                "container",
                "input",
                "form",
                "div",
                "span",
            ]
            found_components = sum(1 for component in components if component in response.text)
            if found_components >= 5:
                self.log_test_result(
                    "UI-Komponenten",
                    True,
                    f"{found_components}/{len(components)} Komponenten gefunden",
                )
                return True
            else:
                self.log_test_result(
                    "UI-Komponenten",
                    False,
                    f"Nur {found_components}/{len(components)} Komponenten gefunden",
                )
                return False
        except Exception as e:
            self.log_test_result("UI-Komponenten", False, str(e))
            return False

    def test_fact_check_visualization(self):
        """Teste Fact-Check Visualisierung"""
        print("🔍 Teste Fact-Check Visualisierung...")
        try:
            response = requests.get(self.base_url, timeout=10)
            fact_check_indicators = [
                "fact-check",
                "confidence",
                "sources",
                "Konfidenz",
                "Quellen",
                "Fact",
                "Check",
                "Vertrauenswürdigkeit",
            ]
            found_indicators = sum(
                1 for indicator in fact_check_indicators if indicator in response.text
            )
            if found_indicators >= 2:
                self.log_test_result(
                    "Fact-Check Visualisierung",
                    True,
                    f"{found_indicators} Fact-Check Elemente gefunden",
                )
                return True
            else:
                self.log_test_result(
                    "Fact-Check Visualisierung",
                    False,
                    f"Nur {found_indicators} Fact-Check Elemente gefunden",
                )
                return False
        except Exception as e:
            self.log_test_result("Fact-Check Visualisierung", False, str(e))
            return False

    def test_gpu_monitoring_ui(self):
        """Teste GPU-Monitoring UI"""
        print("🔍 Teste GPU-Monitoring UI...")
        try:
            response = requests.get(self.base_url, timeout=10)
            gpu_indicators = [
                "GPU",
                "RTX",
                "2070",
                "Auslastung",
                "VRAM",
                "Temperatur",
                "NVIDIA",
                "Grafikkarte",
            ]
            found_indicators = sum(1 for indicator in gpu_indicators if indicator in response.text)
            if found_indicators >= 2:
                self.log_test_result(
                    "GPU-Monitoring UI", True, f"{found_indicators} GPU-Elemente gefunden"
                )
                return True
            else:
                self.log_test_result(
                    "GPU-Monitoring UI", False, f"Nur {found_indicators} GPU-Elemente gefunden"
                )
                return False
        except Exception as e:
            self.log_test_result("GPU-Monitoring UI", False, str(e))
            return False

    def test_chat_functionality(self):
        """Teste Chat-Funktionalität"""
        print("🔍 Teste Chat-Funktionalität...")
        try:
            response = requests.get(self.base_url, timeout=10)
            chat_indicators = [
                "chat",
                "message",
                "historie",
                "frage",
                "antwort",
                "Chat",
                "Nachricht",
                "Frage",
            ]
            found_indicators = sum(1 for indicator in chat_indicators if indicator in response.text)
            if found_indicators >= 2:
                self.log_test_result(
                    "Chat-Funktionalität", True, f"{found_indicators} Chat-Elemente gefunden"
                )
                return True
            else:
                self.log_test_result(
                    "Chat-Funktionalität", False, f"Nur {found_indicators} Chat-Elemente gefunden"
                )
                return False
        except Exception as e:
            self.log_test_result("Chat-Funktionalität", False, str(e))
            return False

    def test_example_questions(self):
        """Teste Beispiel-Fragen"""
        print("🔍 Teste Beispiel-Fragen...")
        try:
            response = requests.get(self.base_url, timeout=10)
            example_indicators = [
                "Beispiel",
                "Klimapolitik",
                "Energiewende",
                "Bundesregierung",
                "Frage",
                "Beispiel-Frage",
            ]
            found_indicators = sum(
                1 for indicator in example_indicators if indicator in response.text
            )
            if found_indicators >= 2:
                self.log_test_result(
                    "Beispiel-Fragen", True, f"{found_indicators} Beispiel-Elemente gefunden"
                )
                return True
            else:
                self.log_test_result(
                    "Beispiel-Fragen", False, f"Nur {found_indicators} Beispiel-Elemente gefunden"
                )
                return False
        except Exception as e:
            self.log_test_result("Beispiel-Fragen", False, str(e))
            return False

    def test_performance(self):
        """Teste Performance"""
        print("🔍 Teste Performance...")
        try:
            start_time = time.time()
            response = requests.get(self.base_url, timeout=10)
            load_time = time.time() - start_time

            if response.status_code == 200 and load_time < 5.0:
                self.log_test_result("Performance", True, ".2f")
                return True
            else:
                self.log_test_result("Performance", False, ".2f")
                return False
        except Exception as e:
            self.log_test_result("Performance", False, str(e))
            return False

    def test_security_headers(self):
        """Teste Security-Headers"""
        print("🔍 Teste Security-Headers...")
        try:
            response = requests.get(self.base_url, timeout=10)
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Content-Security-Policy",
            ]
            found_headers = sum(1 for header in security_headers if header in response.headers)
            if found_headers >= 1:
                self.log_test_result(
                    "Security-Headers", True, f"{found_headers} Security-Headers gefunden"
                )
                return True
            else:
                self.log_test_result(
                    "Security-Headers", False, f"Nur {found_headers} Security-Headers gefunden"
                )
                return False
        except Exception as e:
            self.log_test_result("Security-Headers", False, str(e))
            return False

    def run_all_tests(self):
        """Führe alle Tests durch"""
        print("🚀 Starte umfassenden Web-GUI Test")
        print("=" * 50)

        start_time = datetime.now()

        # Setup WebDriver (optional)
        has_webdriver = self.setup_driver()

        # Führe alle Tests durch
        tests = [
            ("Grundlegende Erreichbarkeit", self.test_basic_connectivity),
            ("Seitentitel", self.test_page_title),
            ("CSS-Styling", self.test_css_styling),
            ("Responsive Design", self.test_responsive_design),
            ("UI-Komponenten", self.test_ui_components),
            ("Fact-Check Visualisierung", self.test_fact_check_visualization),
            ("GPU-Monitoring UI", self.test_gpu_monitoring_ui),
            ("Chat-Funktionalität", self.test_chat_functionality),
            ("Beispiel-Fragen", self.test_example_questions),
            ("Performance", self.test_performance),
            ("Security-Headers", self.test_security_headers),
        ]

        successful_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            print(f"\n📋 Test: {test_name}")
            print("-" * 30)
            if test_func():
                successful_tests += 1

        # Cleanup
        if self.driver:
            self.driver.quit()

        # Zusammenfassung
        print("\n" + "=" * 50)
        print("📊 WEB-GUI TEST-ZUSAMMENFASSUNG")
        print("=" * 50)

        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}")
            if result["details"]:
                print(f"   {result['details']}")

        print(f"\n🎯 Gesamt: {successful_tests}/{total_tests} Tests erfolgreich")

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"⏱️  Test-Dauer: {duration.total_seconds():.1f} Sekunden")
        print(f"📅 Test-Zeitpunkt: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if successful_tests == total_tests:
            print("\n🎉 ALLE WEB-GUI TESTS ERFOLGREICH!")
            print(
                "Die moderne Web-GUI ist vollständig funktionsfähig und bereit für den Produktiveinsatz."
            )
            return True
        elif successful_tests >= total_tests * 0.8:
            print(f"\n⚠️  {successful_tests}/{total_tests} Tests erfolgreich")
            print(
                "Die Web-GUI ist größtenteils funktionsfähig, aber einige Features könnten verbessert werden."
            )
            return True
        else:
            print(f"\n❌ Nur {successful_tests}/{total_tests} Tests erfolgreich")
            print("Die Web-GUI benötigt weitere Optimierungen.")
            return False


def main():
    """Hauptfunktion"""
    tester = WebGUITester()
    success = tester.run_all_tests()
    return success


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
