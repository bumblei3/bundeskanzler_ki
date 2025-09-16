#!/usr/bin/env python3
"""
🎨 Multimodales System für Bundeskanzler-KI
==============================================

Generiert visuelle Antworten:
- Diagramme für politische Daten
- Charts für Statistiken
- Infografiken für komplexe Themen
- Zeitliche Entwicklungen
- Vergleichsdiagramme

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)


class MultimodalResponseGenerator:
    """
    Generiert multimodale Antworten für die Bundeskanzler-KI
    """

    def __init__(self):
        """
        Initialisiert den multimodalen Generator
        """
        # Setze Seaborn-Style für bessere Diagramme
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 14

        # Deutsche Farbpalette für politische Themen
        self.colors = {
            "klima": "#2E86AB",  # Blau für Klima
            "wirtschaft": "#A23B72",  # Violett für Wirtschaft
            "soziales": "#F18F01",  # Orange für Soziales
            "bildung": "#C73E1D",  # Rot für Bildung
            "digital": "#3A7D44",  # Grün für Digital
            "europa": "#FFD23F",  # Gelb für Europa
            "sicherheit": "#1B1B1E",  # Schwarz für Sicherheit
            "gesundheit": "#E63946",  # Rot für Gesundheit
        }

        logger.info("🎨 MultimodalResponseGenerator initialisiert")

    def generate_visual_response(
        self, topic: str, data_type: str = "trends", time_range: str = "5_years"
    ) -> Dict[str, Any]:
        """
        Generiert eine visuelle Antwort basierend auf Thema und Datentyp

        Args:
            topic: Das Thema (klima, wirtschaft, etc.)
            data_type: Typ der Visualisierung (trends, comparison, distribution)
            time_range: Zeitraum für Daten

        Returns:
            Dictionary mit Visualisierungsdaten und Metadaten
        """
        try:
            if data_type == "trends":
                return self._generate_trend_chart(topic, time_range)
            elif data_type == "comparison":
                return self._generate_comparison_chart(topic)
            elif data_type == "distribution":
                return self._generate_distribution_chart(topic)
            elif data_type == "timeline":
                return self._generate_timeline_chart(topic)
            else:
                return self._generate_default_visualization(topic)

        except Exception as e:
            logger.error(f"Fehler bei visueller Generierung: {e}")
            return self._generate_error_visualization()

    def _generate_trend_chart(self, topic: str, time_range: str) -> Dict[str, Any]:
        """
        Generiert ein Trend-Diagramm für ein Thema
        """
        # Simulierte Daten basierend auf Thema
        data = self._get_topic_data(topic, time_range)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Erstelle Zeitreihe
        dates = pd.date_range(start="2019-01-01", periods=len(data), freq="YS")
        values = np.array(data) + np.random.normal(0, 0.1, len(data))  # Leichter Rauschen

        # Trendlinie berechnen
        z = np.polyfit(range(len(values)), values, 2)
        p = np.poly1d(z)

        # Plot
        ax.plot(
            dates,
            values,
            "o-",
            color=self.colors.get(topic, "#2E86AB"),
            linewidth=3,
            markersize=8,
            alpha=0.8,
            label=f"{topic.title()} Entwicklung",
        )

        # Trendlinie
        ax.plot(
            dates,
            p(range(len(values))),
            "--",
            color=self.colors.get(topic, "#2E86AB"),
            linewidth=2,
            alpha=0.6,
            label="Trend",
        )

        # Formatierung
        ax.set_title(f"Entwicklung {topic.title()} (2019-2024)", fontsize=18, pad=20)
        ax.set_xlabel("Jahr", fontsize=14)
        ax.set_ylabel("Indexwert", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # X-Achse formatieren
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Speichere Diagramm
        chart_path = f"/tmp/{topic}_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "type": "trend_chart",
            "topic": topic,
            "chart_path": chart_path,
            "description": f"Trend-Diagramm zeigt die Entwicklung von {topic.title()} über die letzten Jahre",
            "data_points": len(data),
            "time_range": time_range,
            "insights": self._analyze_trend_data(data, topic),
        }

    def _generate_comparison_chart(self, topic: str) -> Dict[str, Any]:
        """
        Generiert ein Vergleichsdiagramm
        """
        # Vergleichsdaten für verschiedene Bereiche
        categories = ["Deutschland", "EU-Durchschnitt", "OECD-Durchschnitt", "Weltweit"]
        values = np.random.uniform(60, 95, len(categories))

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.bar(
            categories, values, color=self.colors.get(topic, "#2E86AB"), alpha=0.8, width=0.6
        )

        # Werte über den Balken anzeigen
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_title(
            f"Vergleich: {topic.title()} - Deutschland vs. International", fontsize=18, pad=20
        )
        ax.set_ylabel("Prozentwert", fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")

        # Speichere Diagramm
        chart_path = f"/tmp/{topic}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "type": "comparison_chart",
            "topic": topic,
            "chart_path": chart_path,
            "description": f"Vergleichsdiagramm zeigt {topic.title()} im internationalen Kontext",
            "categories": categories,
            "values": values.tolist(),
            "insights": self._analyze_comparison_data(values, categories, topic),
        }

    def _generate_distribution_chart(self, topic: str) -> Dict[str, Any]:
        """
        Generiert ein Verteilungsdiagramm
        """
        # Simulierte Verteilungsdaten
        regions = ["Nord", "Süd", "Ost", "West", "Berlin"]
        values = np.random.uniform(20, 80, len(regions))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Kreisdiagramm
        ax1.pie(
            values,
            labels=regions,
            autopct="%1.1f%%",
            colors=plt.cm.Set3(np.linspace(0, 1, len(regions))),
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax1.set_title(f"Regionale Verteilung: {topic.title()}", fontsize=16)

        # Balkendiagramm
        bars = ax2.barh(regions, values, color=self.colors.get(topic, "#2E86AB"), alpha=0.8)
        ax2.set_xlabel("Prozentwert", fontsize=14)
        ax2.set_title(f"Regionale Unterschiede", fontsize=16)
        ax2.grid(True, alpha=0.3, axis="x")

        # Werte neben den Balken
        for bar, value in zip(bars, values):
            ax2.text(
                value + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=12,
            )

        plt.tight_layout()

        # Speichere Diagramm
        chart_path = f"/tmp/{topic}_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "type": "distribution_chart",
            "topic": topic,
            "chart_path": chart_path,
            "description": f"Verteilungsdiagramm zeigt regionale Unterschiede in {topic.title()}",
            "regions": regions,
            "values": values.tolist(),
            "insights": self._analyze_distribution_data(values, regions, topic),
        }

    def _generate_timeline_chart(self, topic: str) -> Dict[str, Any]:
        """
        Generiert ein Zeitstrahl-Diagramm mit wichtigen Ereignissen
        """
        # Wichtige Ereignisse basierend auf Thema
        events = self._get_topic_events(topic)

        fig, ax = plt.subplots(figsize=(16, 10))

        # Zeitstrahl zeichnen
        ax.axhline(y=0, color="black", linewidth=2, alpha=0.7)

        # Ereignisse plotten
        for i, event in enumerate(events):
            date = datetime.strptime(event["date"], "%Y-%m-%d")
            y_pos = (-1) ** i * 0.5  # Abwechselnd oben/unten

            # Punkt für Ereignis
            ax.scatter(
                date,
                y_pos,
                s=100,
                color=self.colors.get(topic, "#2E86AB"),
                edgecolor="white",
                linewidth=2,
                zorder=5,
            )

            # Linie zum Punkt
            ax.plot(
                [date, date],
                [0, y_pos],
                color=self.colors.get(topic, "#2E86AB"),
                linewidth=2,
                alpha=0.7,
            )

            # Text für Ereignis
            ha = "left" if y_pos > 0 else "right"
            va = "bottom" if y_pos > 0 else "top"
            ax.text(
                date,
                y_pos + (0.1 if y_pos > 0 else -0.1),
                event["title"],
                ha=ha,
                va=va,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Formatierung
        ax.set_title(f"Zeitstrahl: Wichtige Ereignisse in {topic.title()}", fontsize=18, pad=30)
        ax.set_xlabel("Datum", fontsize=14)
        ax.set_xlim(datetime(2019, 1, 1), datetime(2025, 12, 31))
        ax.set_ylim(-1, 1)

        # Y-Achse ausblenden
        ax.get_yaxis().set_visible(False)

        # X-Achse formatieren
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Speichere Diagramm
        chart_path = f"/tmp/{topic}_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "type": "timeline_chart",
            "topic": topic,
            "chart_path": chart_path,
            "description": f"Zeitstrahl zeigt wichtige Ereignisse in {topic.title()}",
            "events": events,
            "insights": f"Zeigt {len(events)} wichtige Ereignisse im Zeitraum 2019-2025",
        }

    def _generate_default_visualization(self, topic: str) -> Dict[str, Any]:
        """
        Generiert eine Standard-Visualisierung
        """
        return self._generate_trend_chart(topic, "5_years")

    def _generate_error_visualization(self) -> Dict[str, Any]:
        """
        Generiert eine Fehler-Visualisierung
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Einfache Fehler-Nachricht
        ax.text(
            0.5,
            0.5,
            "Visualisierung\nnicht verfügbar",
            ha="center",
            va="center",
            fontsize=20,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        chart_path = f"/tmp/error_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "type": "error_chart",
            "topic": "error",
            "chart_path": chart_path,
            "description": "Fehler bei der Visualisierungsgenerierung",
            "error": True,
        }

    def _get_topic_data(self, topic: str, time_range: str) -> List[float]:
        """
        Gibt simulierte Daten für ein Thema zurück
        """
        base_data = {
            "klima": [65, 68, 72, 75, 78, 82],
            "wirtschaft": [85, 83, 87, 89, 91, 88],
            "soziales": [70, 72, 74, 76, 78, 80],
            "bildung": [75, 77, 79, 81, 83, 85],
            "digital": [60, 65, 70, 75, 80, 85],
            "europa": [78, 80, 82, 84, 86, 88],
            "sicherheit": [82, 84, 83, 85, 87, 89],
            "gesundheit": [88, 86, 89, 91, 90, 92],
        }

        return base_data.get(topic, [70, 72, 74, 76, 78, 80])

    def _get_topic_events(self, topic: str) -> List[Dict[str, str]]:
        """
        Gibt wichtige Ereignisse für ein Thema zurück
        """
        events_data = {
            "klima": [
                {"date": "2019-12-01", "title": "Klimapaket beschlossen"},
                {"date": "2020-06-15", "title": "EU Green Deal"},
                {"date": "2021-11-01", "title": "COP26 Glasgow"},
                {"date": "2022-07-20", "title": "Hitzewelle Europa"},
                {"date": "2023-12-01", "title": "Klimaziel 2045"},
                {"date": "2024-11-15", "title": "COP29 Baku"},
            ],
            "wirtschaft": [
                {"date": "2020-03-01", "title": "COVID-19 Wirtschaftshilfen"},
                {"date": "2021-06-01", "title": "Konjunkturpaket"},
                {"date": "2022-03-01", "title": "Energiekosten-Hilfen"},
                {"date": "2023-01-15", "title": "Wachstumschancengesetz"},
                {"date": "2024-06-01", "title": "Inflationsbekämpfung"},
            ],
        }

        return events_data.get(
            topic,
            [
                {"date": "2020-01-01", "title": "Jahr 2020"},
                {"date": "2021-01-01", "title": "Jahr 2021"},
                {"date": "2022-01-01", "title": "Jahr 2022"},
                {"date": "2023-01-01", "title": "Jahr 2023"},
                {"date": "2024-01-01", "title": "Jahr 2024"},
            ],
        )

    def _analyze_trend_data(self, data: List[float], topic: str) -> str:
        """
        Analysiert Trend-Daten und gibt Insights zurück
        """
        if len(data) < 2:
            return "Nicht genügend Daten für Trend-Analyse"

        trend = (data[-1] - data[0]) / len(data)
        direction = "steigend" if trend > 0 else "fallend" if trend < 0 else "stabil"

        return f"Der Trend zeigt eine {direction}e Entwicklung mit einer durchschnittlichen jährlichen Veränderung von {trend:.2f} Punkten."

    def _analyze_comparison_data(
        self, values: np.ndarray, categories: List[str], topic: str
    ) -> str:
        """
        Analysiert Vergleichsdaten
        """
        germany_idx = categories.index("Deutschland")
        germany_value = values[germany_idx]
        avg_others = np.mean(np.delete(values, germany_idx))

        if germany_value > avg_others:
            return f"Deutschland liegt {germany_value - avg_others:.1f} Punkte über dem Durchschnitt der Vergleichsländer."
        else:
            return f"Deutschland liegt {avg_others - germany_value:.1f} Punkte unter dem Durchschnitt der Vergleichsländer."

    def _analyze_distribution_data(self, values: np.ndarray, regions: List[str], topic: str) -> str:
        """
        Analysiert Verteilungsdaten
        """
        max_region = regions[np.argmax(values)]
        min_region = regions[np.argmin(values)]
        spread = np.max(values) - np.min(values)

        return f"Die größte Konzentration zeigt sich in {max_region}, die geringste in {min_region}. Die Spanne beträgt {spread:.1f} Punkte."


# Convenience Functions
def get_multimodal_generator() -> MultimodalResponseGenerator:
    """
    Erstellt oder gibt den MultimodalResponseGenerator zurück

    Returns:
        MultimodalResponseGenerator-Instanz
    """
    return MultimodalResponseGenerator()


if __name__ == "__main__":
    # Test des MultimodalResponseGenerator
    print("🎨 Testing MultimodalResponseGenerator...")

    generator = MultimodalResponseGenerator()

    # Test verschiedener Visualisierungen
    topics = ["klima", "wirtschaft", "bildung"]

    for topic in topics:
        print(f"\n📊 Generiere Visualisierung für: {topic}")

        # Trend-Chart
        trend_result = generator.generate_visual_response(topic, "trends")
        print(f"  ✅ Trend-Chart: {trend_result['chart_path']}")
        print(f"  📝 Beschreibung: {trend_result['description']}")

        # Vergleichs-Chart
        comparison_result = generator.generate_visual_response(topic, "comparison")
        print(f"  ✅ Vergleichs-Chart: {comparison_result['chart_path']}")
        print(f"  📝 Insights: {comparison_result['insights']}")

    print("\n✅ MultimodalResponseGenerator Test abgeschlossen!")
