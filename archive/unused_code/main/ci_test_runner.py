#!/usr/bin/env python3
"""
CI/CD Web-GUI Test Runner für Bundeskanzler-KI
Automatisierte Tests für Continuous Integration
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class CITestRunner:
    """CI/CD Test-Runner für Web-GUI Tests"""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or os.getcwd())
        self.results_dir = self.base_dir / "test_results"
        self.results_dir.mkdir(exist_ok=True)

    def setup_environment(self):
        """Richtet die Testumgebung ein"""
        print("🔧 Richte Testumgebung ein...")

        # Aktiviere Virtual Environment
        venv_path = self.base_dir / "bin" / "activate"
        if venv_path.exists():
            print("✅ Virtual Environment gefunden")
            return True
        else:
            print("❌ Virtual Environment nicht gefunden")
            return False

    def start_services(self):
        """Startet die erforderlichen Services"""
        print("🚀 Starte Services...")

        # Starte API im Hintergrund
        try:
            api_process = subprocess.Popen(
                [sys.executable, "simple_api.py"],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            print("✅ API-Service gestartet")
            return api_process
        except Exception as e:
            print(f"❌ Fehler beim Starten der API: {e}")
            return None

    def wait_for_services(self, timeout: int = 30):
        """Wartet bis Services bereit sind"""
        import time

        import requests

        print("⏳ Warte auf Services...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Teste API
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✅ API bereit")
                    return True
            except:
                pass

            time.sleep(2)

        print("❌ Services nicht bereit innerhalb des Timeouts")
        return False

    def run_tests(self, test_type: str = "all"):
        """Führt Tests aus"""
        print(f"🧪 Führe {test_type} Tests aus...")

        test_commands = {
            "unit": [sys.executable, "integration_test.py"],
            "web_gui": [sys.executable, "automated_web_gui_tests.py"],
            "performance": [sys.executable, "web_gui_test.py"],
            "all": [sys.executable, "automated_web_gui_tests.py"],
        }

        if test_type not in test_commands:
            print(f"❌ Unbekannter Test-Typ: {test_type}")
            return False

        try:
            result = subprocess.run(
                test_commands[test_type],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 Minuten Timeout
            )

            # Speichere Ausgabe
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"test_output_{test_type}_{timestamp}.txt"

            with open(output_file, "w") as f:
                f.write(f"=== STDOUT ===\n{result.stdout}\n")
                f.write(f"=== STDERR ===\n{result.stderr}\n")
                f.write(f"=== RETURN CODE ===\n{result.returncode}\n")

            if result.returncode == 0:
                print("✅ Tests erfolgreich abgeschlossen")
                return True
            else:
                print("❌ Tests fehlgeschlagen")
                print("STDOUT:", result.stdout[-500:])  # Letzte 500 Zeichen
                print("STDERR:", result.stderr[-500:])
                return False

        except subprocess.TimeoutExpired:
            print("❌ Tests durch Timeout abgebrochen")
            return False
        except Exception as e:
            print(f"❌ Fehler beim Ausführen der Tests: {e}")
            return False

    def generate_report(self, test_results: dict):
        """Generiert einen Test-Bericht"""
        print("📊 Generiere Test-Bericht...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"test_report_{timestamp}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "test_run": test_results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cwd": str(self.base_dir),
            },
            "services_status": {
                "api_available": self.check_service("http://localhost:8000/health"),
                "gui_available": self.check_service("http://localhost:8502"),
            },
        }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 Bericht gespeichert: {report_file}")
        return report_file

    def check_service(self, url: str) -> bool:
        """Prüft ob ein Service verfügbar ist"""
        try:
            import requests

            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def cleanup(self, api_process):
        """Räumt Testumgebung auf"""
        print("🧹 Räume auf...")

        if api_process:
            try:
                api_process.terminate()
                api_process.wait(timeout=10)
                print("✅ API-Service beendet")
            except:
                api_process.kill()
                print("⚠️ API-Service erzwungen beendet")

    def run_ci_pipeline(self, test_type: str = "all"):
        """Führt die komplette CI-Pipeline aus"""
        print("🚀 Starte CI/CD Pipeline für Web-GUI Tests")
        print("=" * 60)

        success = True
        results = {}

        try:
            # Schritt 1: Umgebung einrichten
            print("\n📋 Schritt 1: Umgebung einrichten")
            if not self.setup_environment():
                success = False
                results["setup"] = "failed"
            else:
                results["setup"] = "success"

            if success:
                # Schritt 2: Services starten
                print("\n📋 Schritt 2: Services starten")
                api_process = self.start_services()
                if not api_process:
                    success = False
                    results["services"] = "failed"
                else:
                    results["services"] = "success"

                    # Schritt 3: Auf Services warten
                    print("\n📋 Schritt 3: Services bereitstellen")
                    if not self.wait_for_services():
                        success = False
                        results["wait"] = "failed"
                    else:
                        results["wait"] = "success"

                        # Schritt 4: Tests ausführen
                        print(f"\n📋 Schritt 4: Tests ausführen ({test_type})")
                        if not self.run_tests(test_type):
                            success = False
                            results["tests"] = "failed"
                        else:
                            results["tests"] = "success"

                # Cleanup
                self.cleanup(api_process)

        except Exception as e:
            print(f"💥 Unerwarteter Fehler in CI-Pipeline: {e}")
            success = False
            results["error"] = str(e)

        # Bericht generieren
        report_file = self.generate_report(results)

        # Zusammenfassung
        print("\n" + "=" * 60)
        print("📊 CI/CD PIPELINE ZUSAMMENFASSUNG")
        print("=" * 60)

        for step, status in results.items():
            icon = "✅" if status == "success" else "❌"
            print(f"{icon} {step.capitalize()}: {status}")

        print(f"\n📄 Detaillierter Bericht: {report_file}")

        if success:
            print("🎉 CI/CD Pipeline erfolgreich abgeschlossen!")
            return 0
        else:
            print("❌ CI/CD Pipeline fehlgeschlagen!")
            return 1


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="CI/CD Web-GUI Test Runner")
    parser.add_argument(
        "--test-type",
        choices=["unit", "web_gui", "performance", "all"],
        default="all",
        help="Welche Tests ausführen",
    )
    parser.add_argument("--base-dir", default=None, help="Basis-Verzeichnis für Tests")

    args = parser.parse_args()

    runner = CITestRunner(args.base_dir)
    exit_code = runner.run_ci_pipeline(args.test_type)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
