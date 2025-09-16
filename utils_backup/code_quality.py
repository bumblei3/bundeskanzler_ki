#!/usr/bin/env python3
"""
Code Quality Tools für Bundeskanzler-KI
Automatisierte Linting, Formatting und Type-Checking
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CodeQualityManager:
    """
    Verwaltet Code-Qualitäts-Tools und -Metriken
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.report_dir = self.project_root / "reports" / "quality"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def install_quality_tools(self) -> bool:
        """Installiert notwendige Code-Qualitäts-Tools"""
        tools = [
            "black",
            "isort",
            "pylint",
            "mypy",
            "flake8",
            "bandit",  # Security linting
            "safety",  # Dependency vulnerability checking
        ]

        print("🔧 Installiere Code-Qualitäts-Tools...")

        try:
            for tool in tools:
                print(f"   📦 Installiere {tool}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", tool],
                    capture_output=True,
                    text=True,
                    check=True,
                )

            print("✅ Alle Tools erfolgreich installiert")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler bei Installation: {e}")
            return False

    def format_code(self, paths: List[str] = None) -> Dict[str, bool]:
        """Formatiert Code mit Black und isort"""
        if paths is None:
            paths = ["core/", "utils/", "web/", "ki_versions/", "tests/"]

        results = {}

        print("🎨 Formatiere Code...")

        # Black Formatting
        try:
            cmd = [sys.executable, "-m", "black"] + paths
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            results["black"] = result.returncode == 0

            if result.returncode == 0:
                print("   ✅ Black Formatting erfolgreich")
            else:
                print(f"   ❌ Black Fehler: {result.stderr}")

        except Exception as e:
            print(f"   ❌ Black Fehler: {e}")
            results["black"] = False

        # isort Import Sorting
        try:
            cmd = [sys.executable, "-m", "isort"] + paths
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            results["isort"] = result.returncode == 0

            if result.returncode == 0:
                print("   ✅ isort Import-Sortierung erfolgreich")
            else:
                print(f"   ❌ isort Fehler: {result.stderr}")

        except Exception as e:
            print(f"   ❌ isort Fehler: {e}")
            results["isort"] = False

        return results

    def lint_code(self, paths: List[str] = None) -> Dict[str, Dict]:
        """Führt Linting mit verschiedenen Tools durch"""
        if paths is None:
            paths = ["core/", "utils/", "web/", "ki_versions/"]

        results = {}

        print("🔍 Analysiere Code-Qualität...")

        # Pylint
        results["pylint"] = self._run_pylint(paths)

        # Flake8
        results["flake8"] = self._run_flake8(paths)

        # MyPy Type Checking
        results["mypy"] = self._run_mypy(paths)

        # Bandit Security Linting
        results["bandit"] = self._run_bandit(paths)

        return results

    def _run_pylint(self, paths: List[str]) -> Dict:
        """Führt Pylint-Analyse durch"""
        try:
            cmd = [sys.executable, "-m", "pylint", "--output-format=json"] + paths
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            # Parse JSON output
            if result.stdout:
                issues = json.loads(result.stdout)
                score = self._extract_pylint_score(result.stderr)

                print(f"   📊 Pylint Score: {score}/10")
                print(f"   📋 Issues gefunden: {len(issues)}")

                # Speichere detaillierten Report
                report_file = self.report_dir / "pylint_report.json"
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(issues, f, indent=2, ensure_ascii=False)

                return {
                    "success": True,
                    "score": score,
                    "issues_count": len(issues),
                    "issues": issues,
                }
            else:
                print("   ✅ Pylint: Keine Issues gefunden")
                return {"success": True, "score": 10.0, "issues_count": 0, "issues": []}

        except Exception as e:
            print(f"   ❌ Pylint Fehler: {e}")
            return {"success": False, "error": str(e)}

    def _run_flake8(self, paths: List[str]) -> Dict:
        """Führt Flake8-Analyse durch"""
        try:
            cmd = [sys.executable, "-m", "flake8", "--format=json"] + paths
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.stdout:
                # Flake8 JSON format parsing
                issues = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            issues.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

                print(f"   📋 Flake8 Issues: {len(issues)}")

                return {"success": True, "issues_count": len(issues), "issues": issues}
            else:
                print("   ✅ Flake8: Keine Issues gefunden")
                return {"success": True, "issues_count": 0, "issues": []}

        except Exception as e:
            print(f"   ❌ Flake8 Fehler: {e}")
            return {"success": False, "error": str(e)}

    def _run_mypy(self, paths: List[str]) -> Dict:
        """Führt MyPy Type-Checking durch"""
        try:
            cmd = [
                sys.executable,
                "-m",
                "mypy",
                "--json-report",
                str(self.report_dir),
            ] + paths
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            # MyPy gibt Fehler über stderr aus
            error_count = result.stderr.count("error:") if result.stderr else 0

            print(f"   🔍 MyPy Type Errors: {error_count}")

            return {
                "success": result.returncode == 0,
                "error_count": error_count,
                "output": result.stderr,
            }

        except Exception as e:
            print(f"   ❌ MyPy Fehler: {e}")
            return {"success": False, "error": str(e)}

    def _run_bandit(self, paths: List[str]) -> Dict:
        """Führt Bandit Security-Analyse durch"""
        try:
            cmd = [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                "-f",
                "json",
                "-o",
                str(self.report_dir / "bandit_report.json"),
            ] + paths
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            # Lade Bandit Report
            report_file = self.report_dir / "bandit_report.json"
            if report_file.exists():
                with open(report_file, "r", encoding="utf-8") as f:
                    report = json.load(f)

                high_issues = len(
                    [i for i in report.get("results", []) if i.get("issue_severity") == "HIGH"]
                )
                medium_issues = len(
                    [i for i in report.get("results", []) if i.get("issue_severity") == "MEDIUM"]
                )

                print(
                    f"   🔒 Bandit Security Issues: {len(report.get('results', []))} (High: {high_issues}, Medium: {medium_issues})"
                )

                return {
                    "success": True,
                    "total_issues": len(report.get("results", [])),
                    "high_severity": high_issues,
                    "medium_severity": medium_issues,
                    "report": report,
                }
            else:
                print("   ✅ Bandit: Keine Security Issues gefunden")
                return {"success": True, "total_issues": 0}

        except Exception as e:
            print(f"   ❌ Bandit Fehler: {e}")
            return {"success": False, "error": str(e)}

    def _extract_pylint_score(self, stderr: str) -> float:
        """Extrahiert Pylint Score aus stderr"""
        for line in stderr.split("\n"):
            if "Your code has been rated at" in line:
                try:
                    score_str = line.split("rated at ")[1].split("/")[0]
                    return float(score_str)
                except (IndexError, ValueError):
                    pass
        return 0.0

    def check_dependencies(self) -> Dict:
        """Prüft Dependencies auf Sicherheitslücken"""
        print("🔒 Prüfe Dependency-Sicherheit...")

        try:
            cmd = [sys.executable, "-m", "safety", "check", "--json"]
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                print(f"   🚨 Sicherheitslücken gefunden: {len(vulnerabilities)}")

                # Speichere Report
                report_file = self.report_dir / "safety_report.json"
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(vulnerabilities, f, indent=2, ensure_ascii=False)

                return {
                    "success": True,
                    "vulnerabilities_count": len(vulnerabilities),
                    "vulnerabilities": vulnerabilities,
                }
            else:
                print("   ✅ Keine Sicherheitslücken in Dependencies gefunden")
                return {"success": True, "vulnerabilities_count": 0}

        except Exception as e:
            print(f"   ❌ Safety Check Fehler: {e}")
            return {"success": False, "error": str(e)}

    def generate_quality_report(self) -> Dict:
        """Generiert umfassenden Qualitätsbericht"""
        print("\n📊 Generiere Code-Qualitätsbericht...")
        print("=" * 50)

        # Format Code
        format_results = self.format_code()

        # Lint Code
        lint_results = self.lint_code()

        # Check Dependencies
        security_results = self.check_dependencies()

        # Zusammenfassung
        report = {
            "timestamp": str(subprocess.check_output(["date"], text=True).strip()),
            "formatting": format_results,
            "linting": lint_results,
            "security": security_results,
            "summary": self._generate_summary(format_results, lint_results, security_results),
        }

        # Speichere Gesamtbericht
        report_file = self.report_dir / "quality_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📋 Vollständiger Bericht gespeichert: {report_file}")

        return report

    def _generate_summary(
        self, format_results: Dict, lint_results: Dict, security_results: Dict
    ) -> Dict:
        """Generiert Zusammenfassung der Qualitätsprüfung"""
        total_issues = 0
        critical_issues = 0

        # Sammle Issues
        for tool_name, tool_result in lint_results.items():
            if isinstance(tool_result, dict) and "issues_count" in tool_result:
                total_issues += tool_result["issues_count"]

                if tool_name == "bandit" and "high_severity" in tool_result:
                    critical_issues += tool_result["high_severity"]

        if "vulnerabilities_count" in security_results:
            critical_issues += security_results["vulnerabilities_count"]

        # Quality Score berechnen
        pylint_score = lint_results.get("pylint", {}).get("score", 0)
        quality_score = max(0, pylint_score - (total_issues * 0.1) - (critical_issues * 0.5))

        return {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "quality_score": round(quality_score, 2),
            "pylint_score": pylint_score,
            "security_vulnerabilities": security_results.get("vulnerabilities_count", 0),
            "recommendation": self._get_recommendation(quality_score, critical_issues),
        }

    def _get_recommendation(self, quality_score: float, critical_issues: int) -> str:
        """Gibt Empfehlung basierend auf Quality Score"""
        if critical_issues > 0:
            return "🚨 KRITISCH: Security Issues müssen sofort behoben werden!"
        elif quality_score >= 8.0:
            return "🟢 AUSGEZEICHNET: Code-Qualität ist sehr hoch"
        elif quality_score >= 6.0:
            return "🟡 GUT: Kleinere Verbesserungen empfohlen"
        elif quality_score >= 4.0:
            return "🟠 MITTEL: Mehrere Issues sollten behoben werden"
        else:
            return "🔴 SCHLECHT: Umfassende Code-Überarbeitung notwendig"


def main():
    """Hauptfunktion für Code-Qualitätsprüfung"""
    quality_manager = CodeQualityManager()

    # Installiere Tools (falls nicht vorhanden)
    print("🚀 Bundeskanzler-KI Code Quality Check")
    print("=" * 50)

    if not quality_manager.install_quality_tools():
        print("❌ Tool-Installation fehlgeschlagen")
        return

    # Führe vollständige Qualitätsprüfung durch
    report = quality_manager.generate_quality_report()

    # Zeige Zusammenfassung
    summary = report["summary"]
    print(f"\n📊 QUALITÄTSBERICHT ZUSAMMENFASSUNG")
    print("=" * 50)
    print(f"🎯 Quality Score: {summary['quality_score']}/10")
    print(f"📋 Total Issues: {summary['total_issues']}")
    print(f"🚨 Critical Issues: {summary['critical_issues']}")
    print(f"🔒 Security Vulnerabilities: {summary['security_vulnerabilities']}")
    print(f"💡 Empfehlung: {summary['recommendation']}")


if __name__ == "__main__":
    main()
