#!/usr/bin/env python3
"""
Web-GUI Test Dashboard
Interaktives Dashboard für Test-Ergebnisse
"""

import glob
import json
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


class TestDashboard:
    """Dashboard für Test-Ergebnisse"""

    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def load_recent_results(self, days: int = 7) -> list:
        """Lädt die letzten Test-Ergebnisse"""
        # Suche in beiden möglichen Verzeichnissen
        patterns = [
            self.results_dir / "web_gui_test_results_*.json",
            Path(".") / "web_gui_test_results_*.json",
            Path(".") / "test_results" / "web_gui_test_results_*.json",
        ]

        result_files = []
        for pattern in patterns:
            result_files.extend(glob.glob(str(pattern)))

        # Entferne Duplikate
        result_files = list(set(result_files))

        results = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for file_path in result_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Extrahiere Timestamp aus Dateiname
                filename = Path(file_path).name
                timestamp_str = filename.replace("web_gui_test_results_", "").replace(".json", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if timestamp >= cutoff_date:
                    data["timestamp"] = timestamp
                    data["file_path"] = file_path
                    results.append(data)

            except Exception as e:
                print(f"Fehler beim Laden {file_path}: {e}")

        # Sortiere nach Timestamp (neueste zuerst)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    def create_summary_stats(self, results: list) -> dict:
        """Erstellt Zusammenfassungsstatistiken"""
        if not results:
            return {}

        # Sammle alle Test-Suites
        all_suites = {}
        for result in results:
            for suite_name, suite_data in result.items():
                if suite_name not in ["timestamp", "file_path"]:
                    if suite_name not in all_suites:
                        all_suites[suite_name] = []
                    all_suites[suite_name].append(suite_data)

        # Berechne Statistiken pro Suite
        stats = {}
        for suite_name, suite_results in all_suites.items():
            total_tests = sum(r["total_tests"] for r in suite_results)
            total_passed = sum(r["passed"] for r in suite_results)
            total_failed = sum(r["failed"] for r in suite_results)
            total_errors = sum(r["errors"] for r in suite_results)
            avg_duration = sum(r["duration"] for r in suite_results) / len(suite_results)

            stats[suite_name] = {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "errors": total_errors,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "avg_duration": avg_duration,
                "runs": len(suite_results),
            }

        return stats

    def create_trend_chart(self, results: list):
        """Erstellt Trend-Chart für Test-Ergebnisse"""
        if not results:
            return None

        # Sammle Daten für Trend
        timestamps = []
        success_rates = []
        total_tests = []
        durations = []

        for result in results:
            timestamps.append(result["timestamp"])

            # Berechne Gesamt-Erfolg
            total_passed = 0
            total_count = 0

            for suite_name, suite_data in result.items():
                if suite_name not in ["timestamp", "file_path"]:
                    total_passed += suite_data["passed"]
                    total_count += suite_data["total_tests"]

            success_rate = (total_passed / total_count * 100) if total_count > 0 else 0
            success_rates.append(success_rate)
            total_tests.append(total_count)

            # Durchschnittliche Dauer
            durations_sum = sum(
                suite_data["duration"]
                for suite_name, suite_data in result.items()
                if suite_name not in ["timestamp", "file_path"]
            )
            durations_count = sum(
                1 for suite_name in result.keys() if suite_name not in ["timestamp", "file_path"]
            )
            avg_duration = durations_sum / durations_count if durations_count > 0 else 0
            durations.append(avg_duration)

        # Erstelle Plotly Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Erfolgsrate
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=success_rates,
                name="Erfolgsrate (%)",
                mode="lines+markers",
                line=dict(color="green"),
            ),
            secondary_y=False,
        )

        # Test-Anzahl
        fig.add_trace(
            go.Bar(
                x=timestamps, y=total_tests, name="Test-Anzahl", opacity=0.6, marker_color="blue"
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Test-Ergebnisse Trend",
            xaxis_title="Zeitpunkt",
            yaxis_title="Erfolgsrate (%)",
            yaxis2_title="Anzahl Tests",
        )

        return fig

    def create_suite_comparison(self, stats: dict):
        """Erstellt Vergleichs-Chart für Test-Suites"""
        if not stats:
            return None

        suites = list(stats.keys())
        success_rates = [stats[suite]["success_rate"] for suite in suites]
        durations = [stats[suite]["avg_duration"] for suite in suites]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=suites, y=success_rates, name="Erfolgsrate (%)", marker_color="lightgreen"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=suites,
                y=durations,
                name="Durchschn. Dauer (s)",
                mode="lines+markers",
                line=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Test-Suite Vergleich",
            xaxis_title="Test-Suite",
            yaxis_title="Erfolgsrate (%)",
            yaxis2_title="Dauer (Sekunden)",
        )

        return fig

    def run_dashboard(self):
        """Führt das Dashboard aus"""
        st.set_page_config(page_title="Web-GUI Test Dashboard", page_icon="🧪", layout="wide")

        st.title("🧪 Web-GUI Test Dashboard")
        st.markdown("**Automatisierte Tests für Bundeskanzler-KI Weboberfläche**")

        # Lade Daten
        results = self.load_recent_results()

        if not results:
            st.warning("Keine Test-Ergebnisse gefunden. Führen Sie zuerst Tests aus.")
            return

        # Übersicht
        col1, col2, col3, col4 = st.columns(4)

        latest = results[0]
        total_suites = len([k for k in latest.keys() if k not in ["timestamp", "file_path"]])
        total_tests = sum(
            latest[suite]["total_tests"]
            for suite in latest.keys()
            if suite not in ["timestamp", "file_path"]
        )
        total_passed = sum(
            latest[suite]["passed"]
            for suite in latest.keys()
            if suite not in ["timestamp", "file_path"]
        )

        with col1:
            st.metric("Test-Suiten", total_suites)
        with col2:
            st.metric("Gesamt-Tests", total_tests)
        with col3:
            st.metric("Bestanden", total_passed)
        with col4:
            success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            st.metric("Erfolgsrate", ".1f")

        # Trend-Chart
        st.subheader("📈 Test-Ergebnisse Trend")
        trend_chart = self.create_trend_chart(results)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)
        else:
            st.info("Nicht genügend Daten für Trend-Analyse")

        # Suite-Vergleich
        st.subheader("📊 Test-Suite Vergleich")
        stats = self.create_summary_stats(results)
        if stats:
            suite_chart = self.create_suite_comparison(stats)
            if suite_chart:
                st.plotly_chart(suite_chart, use_container_width=True)

            # Detaillierte Statistiken
            st.subheader("📋 Detaillierte Statistiken")

            for suite_name, suite_stats in stats.items():
                with st.expander(f"📁 {suite_name.replace('_', ' ').title()}", expanded=False):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Tests", suite_stats["total_tests"])
                        st.metric("Bestanden", suite_stats["passed"])

                    with col2:
                        st.metric("Fehlgeschlagen", suite_stats["failed"])
                        st.metric("Fehler", suite_stats["errors"])

                    with col3:
                        st.metric("Erfolgsrate", ".1f")
                        st.metric("∅ Dauer", ".2f")

        # Letzte Test-Ergebnisse
        st.subheader("📄 Letzte Test-Ergebnisse")
        selected_result = st.selectbox(
            "Wählen Sie einen Test-Lauf:",
            options=[
                f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ({r['file_path']})"
                for r in results
            ],
            index=0,
        )

        if selected_result:
            selected_idx = [
                f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ({r['file_path']})"
                for r in results
            ].index(selected_result)
            selected_data = results[selected_idx]

            st.json(selected_data)

        # Test-Ausführung
        st.subheader("🚀 Tests ausführen")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Alle Tests ausführen", type="primary"):
                with st.spinner("Führe Tests aus..."):
                    # Hier würde der Test ausgeführt werden
                    st.success("Tests würden hier ausgeführt werden")
                    st.rerun()

        with col2:
            if st.button("📊 Nur GUI-Tests"):
                with st.spinner("Führe GUI-Tests aus..."):
                    # Hier würden GUI-Tests ausgeführt werden
                    st.success("GUI-Tests würden hier ausgeführt werden")
                    st.rerun()


def main():
    """Hauptfunktion"""
    dashboard = TestDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
