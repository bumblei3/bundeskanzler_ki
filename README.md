# 🤖 Bundeskanzler KI - Intelligente politische Beratung mit RAG-System

Eine fortschrittliche KI für politische Fragen und Beratung mit semantischer Suche (RAG), optimiert für GPU-Beschleunigung und professionelle Anwendungen.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Projektübersicht

Die Bundeskanzler KI ist ein hochmodernes KI-System für politische Fragen, Beratung und Analyse. Sie kombiniert Retrieval-Augmented Generation (RAG) mit fortschrittlichen Sprachmodellen für präzise, vertrauenswürdige Antworten zu deutschen politischen Themen.

### 🚀 **Kernfeatures (2025)**

#### 🧠 **Intelligente KI-Systeme**
- 🎯 **Verbesserte KI**: RAG-basiert mit 60-75% Vertrauenswerten
- 🔍 **Semantische Suche**: FAISS-Index mit 75 politischen Dokumenten
- 🤖 **Multiple Versionen**: Original, Verbesserte, Einfache, Multimodale KI
- 📱 **Web-Interface**: Benutzerfreundliche Streamlit-Oberfläche

#### ⚡ **GPU-Optimierte Performance**
- 🚀 **CUDA-Beschleunigung**: RTX 2070 optimiert
- 🧠 **Sentence Transformers**: paraphrase-multilingual-MiniLM-L12-v2
- 📊 **FAISS-Indexierung**: Ultraschnelle semantische Suche
- 🔧 **TensorFlow Integration**: Optimierte Modell-Pipeline

#### 🎮 **Benutzerfreundlichkeit**
- 🚀 **Start-Script**: Interaktives Menü mit 8 Optionen
- 📊 **Logging-System**: Detaillierte Performance-Metriken
- 🌐 **API-Server**: RESTful API für Integration
- 🔧 **Admin-Panel**: Live-Monitoring und Systemstatistiken

## 🏗️ **System-Architektur**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web-Interface │    │     FastAPI     │    │   Start-Script  │
│    (Streamlit)  │◄──►│      API        │◄──►│  (Interactive)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Verbesserte KI     │
                    │  (RAG-basiert)      │
                    │                     │
                    │ • 74% Vertrauen     │
                    │ • FAISS Suche       │
                    │ • GPU-optimiert     │
                    │ • 75 Dokumente      │
                    └─────────────────────┘
```

## 🚀 **Schnellstart**

### 📋 **Voraussetzungen**
- **Python**: 3.12+ (Virtual Environment empfohlen)
- **GPU**: NVIDIA GPU mit CUDA-Support (optional, aber empfohlen)
- **RAM**: 8GB+ empfohlen
- **Speicher**: 10GB+ freier Festplattenspeicher

### ⚡ **Installation & Start**

```bash
# 1. Repository klonen
git clone https://github.com/bumblei3/bundeskanzler_ki.git
cd bundeskanzler_ki

# 2. Virtual Environment aktivieren
source bin/activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Interaktives Start-Menu verwenden
./start_ki.sh
```

### 🎮 **Start-Optionen**

Das Start-Script bietet folgende Optionen:

1. **🎯 Verbesserte KI (EMPFOHLEN)** - Beste Performance mit RAG
2. **🌐 Web-Interface (Streamlit)** - Benutzerfreundliche GUI
3. **📡 API Server** - RESTful API für Integration
4. **🔧 Original KI (Interaktiv)** - Classic Version
5. **🧪 Einfache KI (Test)** - Minimal Version für Tests
6. **📊 Status & Logs** - System-Monitoring
7. **🧹 Cache bereinigen** - Performance-Optimierung
8. **❌ Beenden** - Programm verlassen

### 🎯 **Direkter Start (Empfohlen)**

```bash
# Beste Performance - Verbesserte KI starten
python3 core/verbesserte_ki.py

# Web-Interface starten
streamlit run web/webgui_ki.py --server.port 8501

# API-Server starten
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 📊 **Performance-Metriken (September 2025)**

### 🎯 **Verbesserte KI (Empfohlen)**
- **Klimapolitik**: 74.0% Vertrauen
- **Wirtschaftspolitik**: 74.8% Vertrauen
- **Gesundheitspolitik**: 72.7% Vertrauen
- **Bildungspolitik**: 60.7% Vertrauen
- **Europa-Politik**: 74.4% Vertrauen

### 🔍 **RAG-System**
- **Dokumente**: 75 politische Einträge
- **Ähnlichkeits-Scores**: 0.6-0.8
- **Suchzeit**: <1 Sekunde
- **GPU-Beschleunigung**: CUDA aktiviert

### ⚡ **System-Performance**
- **Ladezeit**: ~8-15 Sekunden
- **Antwortzeit**: ~2-5 Sekunden
- **GPU-Speicher**: ~774MB (RTX 2070)
- **Speicher-Effizienz**: Optimiert für 8GB VRAM

## 📁 **Projekt-Struktur**

```
bkki_venv/
├── 🎯 HAUPTKOMPONENTEN
│   ├── core/
│   │   ├── verbesserte_ki.py      # ⭐ EMPFOHLEN - Beste KI (74% Vertrauen)
│   │   ├── bundeskanzler_ki.py    # Original KI-System
│   │   ├── bundeskanzler_api.py   # FastAPI Backend
│   │   └── rag_system.py          # RAG-Suche (75 Dokumente)
│   │
│   ├── ki_versions/               # Alternative KI-Versionen
│   │   ├── einfache_ki.py         # Minimal-Version
│   │   ├── multimodal_ki.py       # Multimodale Features
│   │   └── multilingual_ki.py     # Mehrsprachig (experimentell)
│   │
│   ├── 🌐 WEB & API
│   │   ├── web/
│   │   │   ├── webgui_ki.py       # Streamlit Interface
│   │   │   └── admin_panel.py     # Admin-Bereich
│   │   └── api_memory/            # API-Speicher
│   │
│   ├── 📊 DATEN & KONFIGURATION
│   │   ├── data/
│   │   │   ├── corpus.json        # 75 Wissensbasis-Einträge
│   │   │   ├── config.yaml        # Systemkonfiguration
│   │   │   └── log.txt           # Performance-Logs
│   │   └── config/
│   │       └── model_config.yaml  # Modell-Einstellungen
│   │
│   ├── 🧪 TESTS & UTILS
│   │   ├── tests/                 # Test-Suite
│   │   ├── utils/                 # Hilfsfunktionen
│   │   └── monitoring/            # System-Überwachung
│   │
│   └── 📋 DOKUMENTATION
│       ├── README.md              # Diese Datei
│       ├── SYSTEM_TEST_BERICHT.md # Test-Ergebnisse
│       └── docs/                  # Erweiterte Dokumentation
```

## 💻 **Nutzung**

### 🎯 **Verbesserte KI (Terminal)**
```bash
python3 core/verbesserte_ki.py
```
```
🚀 Bundeskanzler-KI (Verbesserte RAG-Version)
==================================================

Beispiel-Fragen:
• Was ist die Klimapolitik der Bundesregierung?
• Welche Wirtschaftsreformen sind geplant?
• Wie steht Deutschland zur EU-Politik?

🤖 Ihre Frage: Was ist die Klimapolitik der Bundesregierung?
💡 Deutschland setzt sich für innovative Klimaschutzmaßnahmen ein...
📊 Konfidenz: 74.0% | 🔍 Quellen: 3 Dokumente | ⏱️ Zeit: 2.3s
```

### 🌐 **Web-Interface**
```bash
streamlit run web/webgui_ki.py
```
- **URL**: http://localhost:8501
- **Features**: GUI, Chat-Verlauf, Datei-Upload
- **Admin-Login**: `admin` / `admin123!`

### 📡 **API-Nutzung**
```bash
# API starten
uvicorn core/bundeskanzler_api:app --host 0.0.0.0 --port 8000

# Anfrage senden
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Was ist die Klimapolitik der Bundesregierung?"}'
```

### 🔍 **RAG-System direkt nutzen**
```python
from core.rag_system import RAGSystem

# RAG initialisieren
rag = RAGSystem()

# Dokumente suchen
results = rag.retrieve_relevant_documents("Klimapolitik", top_k=5)
for doc in results:
    print(f"Text: {doc['text']}")
    print(f"Score: {doc['score']:.2%}")
```

## 🔧 **Entwicklung**

### 📦 **Hauptabhängigkeiten**
```txt
# KI & ML
tensorflow>=2.8.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0

# Web & API
fastapi>=0.100.0
streamlit>=1.25.0
uvicorn>=0.20.0

# Datenverarbeitung
pandas>=2.0.0
numpy>=1.24.0
datasets>=2.16.1

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
psutil>=5.9.0
```

### 🧪 **Tests ausführen**
```bash
# Alle Tests
python3 -m pytest tests/

# Spezifische Tests
python3 tests/test_integration.py

# Performance-Tests
python3 comprehensive_test.py
```

### 🔍 **Debug-Modus**
```bash
# Detaillierte Logs
export DEBUG=true
python3 core/verbesserte_ki.py

# RAG-System debuggen
python3 -c "from core.rag_system import RAGSystem; rag = RAGSystem(); print(rag.get_corpus_stats())"
```

## 📈 **Systemstatus**

### ✅ **Vollständig Funktional**
- ✅ **Verbesserte KI** (60-75% Vertrauen)
- ✅ **RAG-System** (75 Dokumente, GPU-optimiert)
- ✅ **Einfache KI** (Repariert und getestet)
- ✅ **Start-Script** (8 Optionen, interaktiv)

### ⚠️ **Experimentell**
- ⚠️ **Multimodale KI** (Lädt erfolgreich, weitere Tests ausstehend)
- ⚠️ **Multilingual KI** (Import-Abhängigkeiten fehlen)

### 🎯 **Empfehlung**
Verwenden Sie die **Verbesserte KI** (`core/verbesserte_ki.py`) für optimale Performance und Zuverlässigkeit.

## 🤝 **Beitragen**

1. **Fork** das Repository
2. **Feature-Branch** erstellen (`git checkout -b feature/AmazingFeature`)
3. **Änderungen committen** (`git commit -m 'Add some AmazingFeature'`)
4. **Branch pushen** (`git push origin feature/AmazingFeature`)
5. **Pull Request** öffnen

## 📄 **Lizenz**

Dieses Projekt steht unter der MIT-Lizenz - siehe [LICENSE](LICENSE) für Details.

## 🆘 **Support**

- **GitHub Issues**: [Probleme melden](https://github.com/bumblei3/bundeskanzler_ki/issues)
- **Dokumentation**: `docs/` Verzeichnis
- **Tests**: Siehe `SYSTEM_TEST_BERICHT.md`
- **FAQ**: Siehe `docs/ANLEITUNG.md`

## 🏆 **Danksagungen**

- **TensorFlow** für GPU-Optimierung
- **Sentence Transformers** für semantische Suche
- **FAISS** für Vektordatenbank
- **Streamlit** für Web-Interface
- **FastAPI** für REST API

---

**Entwickelt mit ❤️ für die deutsche politische Bildung**

*Letztes Update: 15. September 2025*