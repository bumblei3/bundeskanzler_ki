
# Bundeskanzler KI

Ein KI-System für politische Fragen, mit hierarchischem Memory, FastAPI-Backend und Web-GUI. Fokus: echte, kontextbasierte Antworten und robuste Testabdeckung.


## 🎯 Projektübersicht

Die Bundeskanzler KI beantwortet politische Fragen auf Deutsch, speichert Erinnerungen (Memories) und bietet eine API sowie ein Web-Interface. Kein Training oder Fine-Tuning nötig – Fokus liegt auf Memory-Management, Kontext und Testbarkeit.


## 🚀 Setup & Start

### Voraussetzungen
- Python 3.12+
- Virtuelle Umgebung (venv, bereits enthalten)

### Installation & Start

```bash
# 1. Virtuelle Umgebung aktivieren
source bin/activate

# 2. (Falls nötig) Abhängigkeiten installieren
pip install -r requirements.txt

# 3. API starten
python -m uvicorn bundeskanzler_api:app --host 0.0.0.0 --port 8000 --reload
```

### Web-GUI (Streamlit)

```bash
streamlit run bundeskanzler_ki.py
```


## 🧪 Testen

```bash
# Alle Kern-Tests ausführen
python comprehensive_test.py

# Einzelne API-Tests (Beispiel)
pytest tests/test_bundeskanzler_api.py -v
```


## 🧠 Memory-System

Hierarchisches Memory mit Kurz- und Langzeitgedächtnis, automatischer Konsolidierung, semantischer Suche und persistenter Speicherung.


**Features:**
- Kurz- und Langzeitgedächtnis
- Semantische Suche (Cosinus-Ähnlichkeit)
- Adaptive Wichtigkeit & Vergessenskurven
- Tag-basierte Indexierung
- Persistente Speicherung


## 📦 Abhängigkeiten (Auszug)
- numpy, pandas, scikit-learn
- fastapi, uvicorn, streamlit
- pytest


## 🚀 Features

- Intelligente, kontextbasierte Antworten auf politische Fragen
- Hierarchisches Memory mit Deduplication
- REST-API (FastAPI)
- Web-GUI (Streamlit)
- Vollständige Testabdeckung


## 🔧 Konfiguration

Die wichtigsten Einstellungen (z. B. API-Port, Memory-Pfade) sind direkt im Code konfigurierbar. Kein Training oder Modell-Tuning nötig.


## 📋 Geplante Erweiterungen

- [ ] Erweiterte Quellenanbindung
- [ ] Mehrsprachigkeit
- [ ] Faktenprüfung & Quellenangabe
- [ ] Verbesserte Web-GUI


## 🤝 Beitragen

Beiträge willkommen! Bitte per Pull Request oder Issue.


## 📝 Lizenz

MIT-Lizenz


## 🙏 Danksagung

Danke an alle Mitwirkenden und die Open-Source-Community!


## 📬 Kontakt

Fragen oder Feedback? Einfach ein Issue erstellen.

---

**Hinweis:** Das Projekt ist aktiv in Entwicklung. Feedback ist willkommen!
