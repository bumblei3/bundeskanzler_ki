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

### **KI-Abfrage**
```bash
# Mit Token authentifizieren
TOKEN="your_jwt_token"
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Was ist die aktuelle Klimapolitik Deutschlands?"}'
```

### **Wichtige Endpunkte**
- `GET /health` - Systemstatus
- `POST /auth/admin-token` - Admin-Login
- `POST /auth/token` - User-Login
- `POST /auth/register` - User-Registrierung
- `POST /chat` - KI-Gespräch
- `GET /admin/system-stats` - System-Metriken

## 📊 **Performance**

- **Test-Erfolgsrate**: 100%
- **Query-Verarbeitung**: ~0.17 Sekunden
- **Konfidenz-Score**: 52.3%
- **GPU-Auslastung**: RTX 2070 mit 6.8GB VRAM
- **CUDA-Status**: Aktiv
- **Sprachen**: 5 (DE, EN, FR, ES, IT)

## 🧪 **Tests**

```bash
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

## 🆘 **Support**

Bei Problemen:
1. Überprüfe die Logs: `tail -f logs/api.log`
2. GPU-Status: `nvidia-smi`
3. Tests ausführen: `python comprehensive_test.py`

## 📄 **Lizenz**

Apache License 2.0 - Siehe [LICENSE](LICENSE) für Details.

---

**Entwickelt mit ❤️ für die deutsche politische Bildung**

*Letztes Update: 16. September 2025*
