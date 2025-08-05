# Bundeskanzler-KI

Dieses Projekt ist eine interaktive Textklassifikation und Antwort-Generierung für politische Aussagen (deutsch/englisch) mit einem GRU-Modell in TensorFlow/Keras.

## Features
- Korpus aus Datei (`corpus.txt`) oder Standardkorpus
- Batch-Inferenz aus Datei (`input.txt`)
- Interaktiver Modus mit Top-N Antworten
- Konfigurierbare Parameter über Kommandozeile (z.B. Top-N, Batchgröße, Epochen)
- Logging aller Interaktionen in `log.txt`
- Stoppwort-Entfernung, Lemmatization/Stemming, einfache Spracherkennung
- Modell wird automatisch gespeichert und geladen
- Erweiterte Validierung und Feedback-Analyse
- Automatisierte Tests für alle Module
- Streamlit-Dashboard und PDF-Report für Validierungsergebnisse

## Installation
1. Python 3.8+ und pip installieren
2. Abhängigkeiten installieren:
   ```bash
   pip install tensorflow numpy nltk pandas scikit-learn streamlit reportlab pillow matplotlib langdetect
   ```

## Nutzung
### Interaktiver Modus
```bash
python bundeskanzler_ki.py
```

### Batch-Inferenz
Erstelle eine Datei `input.txt` mit je einer Eingabe pro Zeile:
```bash
python bundeskanzler_ki.py --input input.txt
```

### Parameter anpassen
```bash
python bundeskanzler_ki.py --top_n 5 --batch_size 16 --epochs 30 --corpus mein_korpus.txt --log mein_log.txt
```

## Korpus erweitern
Füge eigene Sätze in die Datei `corpus.txt` (eine Zeile pro Satz).


## Logging
Alle Anfragen und Antworten werden in `log.txt` gespeichert.

## Validierung & Reporting
- Validierungsergebnisse werden als JSON exportiert (`validation_results.json`).
- Streamlit-Dashboard (`validation_dashboard.py`) für visuelle Analyse.
- PDF-Report (`validation_report.py`) für Dokumentation.

## Hinweise
- Das Modell wird nach dem ersten Training als `bundeskanzler_ki_model.keras` gespeichert und beim nächsten Start geladen.
- Für deutsche und englische Sätze geeignet.
- Automatisierte Tests: `python3 -m unittest discover`

## Lizenz
MIT
