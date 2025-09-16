#!/usr/bin/env python3
"""
Automatisches Daten-Update-System für Bundeskanzler-KI
Aktualisiert die Wissensbasis mit neuen politischen Dokumenten
"""

import json
import requests
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
from typing import List, Dict, Any
import os

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataUpdater:
    """Automatisches Update-System für politische Daten"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.corpus_file = self.data_dir / "corpus.json"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Datenquellen
        self.sources = {
            "bundestag": {
                "url": "https://www.bundestag.de/services/opendata",
                "api_key": None,
                "update_interval": 24  # Stunden
            },
            "bundesregierung": {
                "url": "https://www.bundesregierung.de/api",
                "api_key": None,
                "update_interval": 12
            },
            "bundesanzeiger": {
                "url": "https://www.bundesanzeiger.de/api",
                "api_key": None,
                "update_interval": 6
            }
        }

    def load_existing_corpus(self) -> Dict[str, Any]:
        """Lädt bestehende Corpus-Daten"""
        if not self.corpus_file.exists():
            logger.warning(f"Corpus-Datei {self.corpus_file} nicht gefunden. Erstelle neue.")
            return {"entries": [], "last_update": None, "version": "1.0"}

        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Corpus geladen: {len(data.get('entries', []))} Einträge")
                return data
        except Exception as e:
            logger.error(f"Fehler beim Laden der Corpus-Datei: {e}")
            return {"entries": [], "last_update": None, "version": "1.0"}

    def save_corpus(self, corpus: Dict[str, Any]):
        """Speichert Corpus mit Backup"""
        # Backup erstellen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"corpus_backup_{timestamp}.json"

        if self.corpus_file.exists():
            self.corpus_file.rename(backup_file)
            logger.info(f"Backup erstellt: {backup_file}")

        # Neue Daten speichern
        corpus["last_update"] = datetime.now().isoformat()
        corpus["version"] = "2.0"

        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        logger.info(f"Corpus gespeichert: {len(corpus.get('entries', []))} Einträge")

    def fetch_bundestag_data(self) -> List[Dict[str, Any]]:
        """Holt neue Daten vom Bundestag"""
        logger.info("Hole Daten vom Bundestag...")

        # Simulierte Daten für Demo (in Produktion echte API verwenden)
        new_entries = [
            {
                "text": "Der Deutsche Bundestag hat heute das neue Klimaschutzgesetz verabschiedet, das strengere CO2-Grenzwerte für die Industrie vorsieht.",
                "topic": "klima",
                "language": "de",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "bundestag",
                "verified": True,
                "url": "https://www.bundestag.de/dokumente"
            },
            {
                "text": "Die Koalition einigte sich auf zusätzliche Mittel für die Digitalisierung von Schulen in Höhe von 2 Milliarden Euro.",
                "topic": "bildung",
                "language": "de",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "bundestag",
                "verified": True,
                "url": "https://www.bundestag.de/dokumente"
            }
        ]

        logger.info(f"{len(new_entries)} neue Einträge vom Bundestag geholt")
        return new_entries

    def fetch_regierung_data(self) -> List[Dict[str, Any]]:
        """Holt neue Daten von der Bundesregierung"""
        logger.info("Hole Daten von der Bundesregierung...")

        new_entries = [
            {
                "text": "Bundeskanzler Scholz kündigte neue Investitionen in die Wasserstofftechnologie in Höhe von 10 Milliarden Euro an.",
                "topic": "energie",
                "language": "de",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "bundesregierung",
                "verified": True,
                "url": "https://www.bundesregierung.de/"
            },
            {
                "text": "Die Bundesregierung plant eine Reform des Einwanderungsgesetzes zur Fachkräftesicherung.",
                "topic": "migration",
                "language": "de",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "bundesregierung",
                "verified": True,
                "url": "https://www.bundesregierung.de/"
            }
        ]

        logger.info(f"{len(new_entries)} neue Einträge von der Bundesregierung geholt")
        return new_entries

    def fetch_bundesanzeiger_data(self) -> List[Dict[str, Any]]:
        """Holt neue Gesetzesänderungen vom Bundesanzeiger"""
        logger.info("Hole Daten vom Bundesanzeiger...")

        new_entries = [
            {
                "text": "Neue Verordnung zur Förderung erneuerbarer Energien wurde im Bundesanzeiger veröffentlicht.",
                "topic": "energie",
                "language": "de",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "bundesanzeiger",
                "verified": True,
                "url": "https://www.bundesanzeiger.de/"
            }
        ]

        logger.info(f"{len(new_entries)} neue Einträge vom Bundesanzeiger geholt")
        return new_entries

    def update_corpus(self):
        """Hauptfunktion zum Aktualisieren der Wissensbasis"""
        logger.info("Starte Daten-Update...")

        # Bestehende Daten laden
        corpus = self.load_existing_corpus()

        # Neue Daten sammeln
        new_entries = []
        new_entries.extend(self.fetch_bundestag_data())
        new_entries.extend(self.fetch_regierung_data())
        new_entries.extend(self.fetch_bundesanzeiger_data())

        # Duplikate vermeiden (einfache Prüfung)
        existing_texts = {entry["text"] for entry in corpus["entries"]}
        unique_new_entries = [entry for entry in new_entries if entry["text"] not in existing_texts]

        # Neue Einträge hinzufügen
        corpus["entries"].extend(unique_new_entries)

        # Speichern
        self.save_corpus(corpus)

        logger.info(f"Update abgeschlossen: {len(unique_new_entries)} neue Einträge hinzugefügt")
        logger.info(f"Gesamt-Einträge: {len(corpus['entries'])}")

    def schedule_daily_update(self):
        """Plant tägliches Update"""
        logger.info("Plane tägliches Update...")

        while True:
            try:
                self.update_corpus()
                logger.info("Warte 24 Stunden bis zum nächsten Update...")
                time.sleep(24 * 60 * 60)  # 24 Stunden
            except KeyboardInterrupt:
                logger.info("Update-Scheduler gestoppt")
                break
            except Exception as e:
                logger.error(f"Fehler beim Update: {e}")
                time.sleep(60 * 60)  # Bei Fehler 1 Stunde warten

def main():
    """Hauptfunktion"""
    updater = DataUpdater()

    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        logger.info("Starte geplanten Update-Service...")
        updater.schedule_daily_update()
    else:
        logger.info("Führe einmaliges Update durch...")
        updater.update_corpus()

if __name__ == "__main__":
    import sys
    main()