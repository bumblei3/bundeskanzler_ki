# 🤖 Bundeskanzler KI - RTX 2070 GPU-optimiert

Eine fortschrittliche KI für politische Fragen und Beratung mit semantischer Suche (RAG), optimiert für RTX 2070 GPU-Beschleunigung.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-100%25-brightgreen.svg)](OPEN_SOURCE_MANIFEST.md)

## 🌟 **100% Open-Source**

Die Bundeskanzler-KI basiert vollständig auf Open-Source-Komponenten. Alle Technologien sind unter freien Lizenzen verfügbar.

**Detailliert:** [OPEN_SOURCE_MANIFEST.md](OPEN_SOURCE_MANIFEST.md)

## 🎯 **Kernfeatures**

- 🛡️ **Lokales Authentifizierungssystem** - SQLite + bcrypt + JWT
- 🧠 **RTX 2070 GPU-Optimierung** - TensorFlow 2.20 + PyTorch 2.8
- 🔍 **Advanced RAG System** - Semantische Suche mit FAISS
- 🌍 **5 Sprachen** - Deutsch, Englisch, Französisch, Spanisch, Italienisch
- ✅ **Fact-Checking** - Automatische Validierung
- 📱 **Web-Interface** - Streamlit-basierte Oberfläche
- 🧪 **100% Test-Abdeckung** - Vollständige Test-Suite
- ⚡ **Request Batching System** - GPU-optimierte Batch-Verarbeitung
- 🎨 **Multimodale KI** - Text, Bilder, Audio, Video-Unterstützung
- 🚀 **Intelligent Caching** - Mehrstufiges Cache-System
- 📊 **Monitoring & Analytics** - Umfassende System-Metriken

## 🏗️ **System-Architektur**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web-Interface │    │   Advanced API  │    │   GPU-Start     │
│    (Streamlit)  │◄──►│   (FastAPI)     │◄──►│   Script        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │ Authentifizierungs- │
                    │ system (SQLite)     │
                    │                     │
                    │ • User-Registrierung│
                    │ • bcrypt-Hashing    │
                    │ • JWT-Tokens        │
                    │ • Rollen-Management │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │ RTX 2070 KI-System  │
                    │ (GPU-optimiert)     │
                    │                     │
                    │ • TensorFlow 2.20   │
                    │ • PyTorch 2.8       │
                    │ • FAISS GPU-Index   │
                    │ • Multi-Agent       │
                    │ • RAG-System 2.0    │
                    │ • Fact-Checking     │
                    │ • 5 Sprachen        │
                    │ • Request Batching  │
                    │ • Multimodal KI     │
                    │ • Intelligent Cache │
                    └─────────────────────┘
```

## 🚀 **Schnellstart**

### 📋 **Voraussetzungen**
- Python 3.12+
- NVIDIA RTX 2070 oder besser
- 8GB+ RAM
- CUDA 12.0+

### ⚡ **Installation**
```bash
# Repository klonen
git clone https://github.com/bumblei3/bundeskanzler_ki.git
cd bundeskanzler_ki

# Virtual Environment
python3 -m venv bkki_venv
source bkki_venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Datenbank initialisieren
python core/bundeskanzler_api.py
```

### 🎮 **Startoptionen**

#### **GPU-optimiert (Empfohlen)**
```bash
./start_gpu.sh interactive
```

#### **API-Server**
```bash
python core/bundeskanzler_api.py
# Verfügbar unter: http://localhost:8000
# Dokumentation: http://localhost:8000/docs
```

#### **Web-Interface**
```bash
streamlit run web/webgui_ki.py --server.port 8501
# Verfügbar unter: http://localhost:8501
```

## 🔌 **API-Nutzung**

### **Authentifizierung**
```bash
# Admin-Login (Standard: admin/admin123!)
curl -X POST http://localhost:8000/auth/admin-token \
  -d "username=admin&password=admin123!"

# User-Registrierung
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass123!", "email": "user@example.com"}'
```

### **Batch-Verarbeitung**
```bash
# Text-Batch-Verarbeitung
curl -X POST http://localhost:8000/batch/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Was ist Demokratie?", "priority": 1}'

# Embedding-Batch-Verarbeitung
curl -X POST http://localhost:8000/batch/embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Politik", "Regierung", "Demokratie"], "priority": 1}'

# Such-Batch-Verarbeitung
curl -X POST http://localhost:8000/batch/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Bundeskanzler Aufgaben", "context": ["Politik"], "priority": 1}'
```

## 📊 **Performance**

### **Wichtige Endpunkte**
- `GET /health` - Systemstatus
- `POST /auth/admin-token` - Admin-Login
- `POST /auth/token` - User-Login
- `POST /auth/register` - User-Registrierung
- `POST /chat` - KI-Gespräch
- `POST /batch/text` - Batch-Text-Verarbeitung
- `POST /batch/embeddings` - Batch-Embedding-Generierung
- `POST /batch/search` - Batch-Suchanfragen
- `POST /batch/immediate` - Sofortige Batch-Verarbeitung
- `GET /admin/system-stats` - System-Metriken
- `GET /admin/batch/stats` - Batch-System-Statistiken

## 📊 **Performance**

- **Test-Erfolgsrate**: 100% (8/8 Request Batching Tests)
- **Query-Verarbeitung**: ~0.17 Sekunden (Einzeln), ~0.507s (Batch 10)
- **Batch-Durchsatz**: 326.3 Anfragen/Sekunde
- **Konfidenz-Score**: 52.3%
- **GPU-Auslastung**: RTX 2070 mit 6.8GB VRAM
- **CUDA-Status**: Aktiv
- **Sprachen**: 5 (DE, EN, FR, ES, IT)
- **Batch-Größe**: RTX 2070 optimiert (8 Requests)
- **Cache-Hit-Rate**: >85% (Intelligent Caching)

## 🧪 **Tests**

```bash
# Vollständige System-Verifizierung
python verify_system.py

# Alle Tests ausführen
python comprehensive_test.py

# Spezifische Tests
python -m pytest tests/ -v
```

## 📖 **Dokumentation**

- [OPEN_SOURCE_MANIFEST.md](OPEN_SOURCE_MANIFEST.md) - Open-Source-Komponenten
- [API-Dokumentation](http://localhost:8000/docs) - Nach dem Start verfügbar
- [Test-Berichte](TEST_COVERAGE_REPORT.md) - Test-Ergebnisse
- [Architektur-Roadmap](NEXT_GENERATION_ROADMAP.md) - Zukünftige Entwicklungen
- [Request Batching Guide](test_request_batching.py) - Batch-System Dokumentation
- [RTX 2070 Optimierung](RTX_2070_OPTIMIZATION_ROADMAP.md) - GPU-Optimierungen
- [Multimodal KI](multimodal_ki.py) - Multimodale Features

## 🆘 **Support**

Bei Problemen:
1. Überprüfe die Logs: `tail -f logs/api.log`
2. GPU-Status: `nvidia-smi`
3. Tests ausführen: `python comprehensive_test.py`

## 📄 **Lizenz**

Apache License 2.0 - Siehe [LICENSE](LICENSE) für Details.

---

**Entwickelt mit ❤️ für die deutsche politische Bildung**

*Letztes Update: 17. September 2025*
