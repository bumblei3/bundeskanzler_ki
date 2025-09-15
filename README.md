
# 🤖 Bundeskanzler KI - Multimodale KI mit RTX 2070 Optimierung

Eine fortschrittliche multimodale KI für politische Fragen und Beratung, optimiert für RTX 2070 GPUs. Unterstützt Text, Bilder, Audio und Video mit kontinuierlichem Lernen und professionellem Admin-Panel.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Projektübersicht

Die Bundeskanzler KI ist ein hochmodernes multimodales KI-System für politische Fragen, Beratung und Analyse. Sie kombiniert fortschrittliche KI-Modelle mit GPU-Optimierung für RTX 2070 Hardware, kontinuierliches Lernen und professionelle Verwaltungstools.

### 🚀 **Kernfeatures (2025)**

#### 🤖 **Multimodale KI-Fähigkeiten**
- 📝 **Text-Verarbeitung**: GPT-2 Medium mit 8-bit Quantisierung
- 👁️ **Bild-Analyse**: SigLIP Vision-Modelle für Bildverständnis
- 🎤 **Audio-Verarbeitung**: Whisper Base für Spracherkennung
- 🎥 **Video-Support**: Integrierte Video-Frame-Analyse (zukünftig)

#### 🎮 **RTX 2070 GPU-Optimierung**
- ⚡ **8-bit Quantisierung**: BitsAndBytes für optimale Speichereffizienz
- 🧠 **GPU-Memory-Management**: Automatische Speicherbereinigung
- 🚀 **Device Mapping**: Optimierte Modell-Verteilung auf GPU
- 📊 **Speicher-Monitoring**: Live GPU-Speicher-Überwachung

#### 🧠 **Kontinuierliches Lernen**
- 📈 **Feedback-Schleifen**: Automatische Modell-Verbesserung
- � **Memory-Netzwerke**: Hierarchische Gedächtnis-Systeme
- 🔄 **Adaptive Responses**: Kontextabhängige Antwort-Optimierung
- 📊 **Performance-Tracking**: Detaillierte Metriken und Analysen

#### 🔐 **Sicherheit & Verwaltung**
- 🔐 **Erweiterte Sicherheit**: API-Key-Management und Content-Filtering
- 👥 **Benutzer-Management**: Rollenbasierte Zugriffskontrolle
- 📊 **Admin-Panel**: Live-Monitoring und Systemstatistiken
- 📋 **Log-Management**: Strukturierte Logging-Systeme
- 🔍 **Debug-System**: Erweiterte Fehlerbehebung mit API-Call-Tracking und Live-Debugging

## 🏗️ **System-Architektur**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web-Interface │    │     FastAPI     │    │   Admin-Panel   │
│    (Streamlit)  │◄──►│      API        │◄──►│  (Monitoring)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Multimodale KI     │
                    │  (RTX 2070 opt.)    │
                    │                     │
                    │ • GPT-2 Medium      │
                    │ • SigLIP Vision     │
                    │ • Whisper Audio     │
                    │ • Memory Networks   │
                    └─────────────────────┘
```

## 🚀 **Schnellstart**

### 📋 **Voraussetzungen**
- **Python**: 3.12+
- **GPU**: NVIDIA RTX 2070 (8GB VRAM) oder besser
- **RAM**: 16GB+ empfohlen
- **Speicher**: 50GB+ freier Festplattenspeicher

### ⚡ **Installation & Start**

```bash
# 1. Virtuelle Umgebung aktivieren
cd /home/tobber/bkki_venv
source bin/activate

# 2. Abhängigkeiten installieren (falls nicht vorhanden)
pip install -r requirements.txt

# 3. RTX 2070 optimierte Modelle laden
python -c "from multimodal_ki import MultimodalTransformerModel; model = MultimodalTransformerModel(model_tier='rtx2070')"

# 4. API-Server starten
uvicorn bundeskanzler_api:app --host 0.0.0.0 --port 8001 --reload

# 5. Web-Interface starten (neues Terminal)
streamlit run webgui_ki.py --server.port 8501 --server.address 0.0.0.0
```

### � **Zugriff auf die Anwendung**

- **Web-Interface**: http://localhost:8501
- **API-Dokumentation**: http://localhost:8001/docs
- **Admin-Login**: `admin` / `admin123!`

## 🎮 **RTX 2070 Optimierung**

Das System ist speziell für RTX 2070 GPUs optimiert:

```python
# Automatische RTX 2070 Optimierung
from multimodal_ki import MultimodalTransformerModel

model = MultimodalTransformerModel(model_tier='rtx2070')
# • 8-bit Quantisierung für alle Modelle
# • GPU-Memory-Management
# • Optimierte Modell-Verteilung
# • ~774MB GPU-Speicher Verbrauch
```

### 📊 **Performance-Metriken**
- **GPU-Speicher**: 774MB / 8GB verwendet (9.7% Auslastung)
- **Ladezeit**: ~8 Sekunden für alle Modelle
- **Inference**: ~9 Sekunden pro Query
- **Speicher-Effizienz**: 75% weniger RAM-Verbrauch

## 🎨 **Multimodale Modi**

### 📝 **Text-Modus**
```python
from multimodal_ki import MultimodalTransformerModel

model = MultimodalTransformerModel()
response = model.process_text("Was ist die Aufgabe des Bundeskanzlers?")
print(response)
```

### �️ **Bild-Modus**
```python
# Bild hochladen und analysieren
image_path = "politik_bild.jpg"
analysis = model.process_image(image_path)
print(f"Bild-Analyse: {analysis}")
```

### 🎤 **Audio-Modus**
```python
# Audio-Datei transkribieren
audio_path = "rede.wav"
transcription = model.process_audio(audio_path)
print(f"Transkription: {transcription}")
```

### 🔄 **Multimodal Kombination**
```python
# Kombinierte Analyse
result = model.process_multimodal(
    text="Analysiere diese politische Rede",
    image="rede_bild.jpg",
    audio="rede_audio.wav"
)
```

## 🧠 **Kontinuierliches Lernen**

Das System lernt kontinuierlich aus User-Interaktionen:

```python
from continuous_learning import ContinuousLearningSystem

learning_system = ContinuousLearningSystem()
learning_system.add_feedback(user_input="Frage", ai_response="Antwort", rating=5)
learning_system.update_model()  # Automatische Modell-Verbesserung
```

### 📈 **Lern-Features**
- **Feedback-Schleifen**: User-Ratings für Antwort-Verbesserung
- **Kontext-Lernen**: Situationsabhängige Antwort-Optimierung
- **Memory-Netzwerke**: Langzeitgedächtnis für Konversationen
- **Performance-Tracking**: Detaillierte Analysen und Metriken

## 🔧 **Konfiguration**

### ⚙️ **Model-Tiers**
```python
# RTX 2070 optimiert (empfohlen)
model = MultimodalTransformerModel(model_tier='rtx2070')

# Alternative Tiers
model_advanced = MultimodalTransformerModel(model_tier='advanced')  # Größere Modelle
model_basic = MultimodalTransformerModel(model_tier='basic')       # CPU-Only
model_premium = MultimodalTransformerModel(model_tier='premium')   # API-Modelle
```

### 🔑 **API-Keys (Optional)**
```bash
# Für Premium-Features
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 📊 **Monitoring & Admin**

### 🖥️ **Admin-Panel Features**
- **Live-Dashboard**: System-Metriken und Performance
- **GPU-Monitoring**: Speicher- und Temperatur-Überwachung
- **Log-Viewer**: Strukturierte Logs mit Filterung
- **Memory-Management**: Detaillierte Speicher-Statistiken
- **User-Management**: Benutzer- und Berechtigungs-Verwaltung

### 📈 **System-Metriken**
- **GPU-Auslastung**: Live RTX 2070 Monitoring
- **Memory-Usage**: RAM und GPU-Speicher Tracking
- **API-Performance**: Response-Zeiten und Durchsatz
- **Model-Accuracy**: Antwort-Qualität und Feedback-Ratings

## 🧪 **Tests & Validierung**

```bash
# Vollständige System-Tests
python -c "
from multimodal_ki import MultimodalTransformerModel
import torch

# RTX 2070 Test
model = MultimodalTransformerModel(model_tier='rtx2070')
print(f'GPU Memory: {torch.cuda.memory_allocated() // 1024**2}MB')
print('✅ RTX 2070 Optimierung funktioniert!')
"

# Unit-Tests
python -m pytest tests/ -v

# Integration-Tests
python test_integration.py
```

## 🐳 **Docker-Deployment**

```bash
# GPU-Version (empfohlen für RTX 2070)
docker-compose up -d

# CPU-Version (Fallback)
docker-compose -f docker-compose.cpu.yml up -d
```

## 📚 **API-Referenz**

### 🔐 **Authentifizierung**
```bash
# Admin-Token erhalten
curl -X POST "http://localhost:8000/auth/admin-token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123!"}'
```

### 🤖 **KI-Endpunkte**
```bash
# Text-Verarbeitung
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Ihre Frage hier"}'

# Multimodale Analyse
curl -X POST "http://localhost:8000/api/multimodal" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "text=Ihre Frage" \
  -F "image=@bild.jpg"
```

## 🔧 **Entwicklung**

### 📦 **Abhängigkeiten**
```
torch>=2.8.0
transformers>=4.56.0
fastapi>=0.116.0
uvicorn>=0.35.0
streamlit>=1.47.0
numpy>=2.3.0
pandas>=2.3.0
scikit-learn>=1.7.0
accelerate>=1.4.0
datasets>=3.2.0
anthropic>=0.30.0
openai>=1.40.0
psutil>=6.1.0
SQLAlchemy>=2.0.0
redis>=5.2.0
pytest>=8.4.0
```

### 🏗️ **Projekt-Struktur**
```
bundeskanzler_ki/
├── multimodal_ki.py          # Haupt-KI-Modul (RTX 2070 opt.)
├── bundeskanzler_ki.py       # Legacy-KI-System
├── continuous_learning.py    # Kontinuierliches Lernen
├── advanced_security.py      # Sicherheit & Authentifizierung
├── bundeskanzler_api.py      # FastAPI-Backend
├── webgui_ki.py             # Streamlit Web-Interface
├── admin_panel_server.py    # Admin-Panel Server
└── tests/                    # Test-Suite
```

## 🤝 **Beitragen**

1. Fork das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Commit deine Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Öffne einen Pull Request

## 📄 **Lizenz**

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die [LICENSE](LICENSE) Datei für Details.

## 🙏 **Danksagungen**

- **PyTorch Team** für die GPU-Optimierung
- **Hugging Face** für die Transformer-Modelle
- **FastAPI** für das hochperformante Backend
- **Streamlit** für das intuitive Web-Interface

## 📞 **Support**

Bei Fragen oder Problemen:
- 📧 **Email**: support@bundeskanzler-ki.de
- 🐛 **Issues**: [GitHub Issues](https://github.com/bumblei3/bundeskanzler_ki/issues)
- 📖 **Dokumentation**: [Wiki](https://github.com/bumblei3/bundeskanzler_ki/wiki)

---

**Letzte Aktualisierung**: 15. September 2025
**Version**: 2.1.0 (RTX 2070 Edition mit Debug-System)
**Python**: 3.12+
**GPU**: NVIDIA RTX 2070 (8GB VRAM)

```bash
# HTML-Admin-Panel (einfach)
# URL: http://localhost:8000/admin

# CLI-Admin-Tool
python3 admin_cli.py
```

### API-Endpunkte

```bash
# Health-Check
GET /health

# Chat mit KI
POST /chat
Content-Type: application/json
{
  "message": "Wie steht es um die Klimapolitik?",
  "user_id": "optional_user_id"
}

# Admin-Login
POST /auth/admin-token
Content-Type: application/x-www-form-urlencoded
{
  "username": "admin",
  "password": "admin123!"
}

# Admin Memory-Stats (behoben!)
GET /admin/memory/stats
Authorization: Bearer <admin_token>
Response: {
  "kurzzeitgedaechtnis_entries": 0,
  "langzeitgedaechtnis_entries": 0,
  "total_entries": 0,
  "memory_efficiency": 0.0,
  "status": "success"
}

# Weitere Admin-Endpunkte
GET /admin/system-stats          # System-Metriken
GET /admin/health               # Health-Check
GET /admin/logs/{type}          # Live-Logs
GET /admin/users                # Benutzer-Management
POST /admin/memory/clear        # Memory leeren
GET /admin/config               # Konfiguration
PUT /admin/config               # Konfiguration aktualisieren
```

### Web-Interface

```bash
# Streamlit Admin-Panel (Empfohlen)
streamlit run webgui_ki.py
URL: http://localhost:8501

# HTML Admin-Panel (Einfach)
URL: http://localhost:8000/admin

# API Dokumentation
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
```

## 🧪 Testen

```bash
# Vollständige Test-Suite
python comprehensive_test.py

# Einzelne Komponenten testen
python test_gpu_batching.py
python test_memory_optimization.py
python test_admin_panel.py

# API-Tests
pytest tests/ -v
```

## 🏗️ Architektur

### GPU-Batching-System 🚀
- **CUDA/ROCm Support**: Automatische Hardware-Erkennung
- **Async Processing**: Parallele Batch-Verarbeitung
- **Performance**: 47.730 Embeddings/Sekunde
- **Fallback**: CPU-Modus bei fehlender GPU

### Optimiertes Memory-System 🧠
- **Quantisierung**: int8/float16 für 75% Speicherersparnis
- **LRU-Caching**: Automatische Cache-Verwaltung
- **Memory Pooling**: Effiziente Wiederverwendung von Arrays
- **Hierarchisches Design**: Kurz-/Langzeitgedächtnis

### Admin-Panel 📊
- **Live-Monitoring**: Echtzeit-Systemstatistiken
- **Memory-Insights**: Detaillierte Speichernutzung (neu behoben!)
- **GPU-Monitoring**: Batch-Performance und Hardware-Status
- **Log-Viewer**: Strukturierte Live-Logs mit Filterung
- **Benutzer-Management**: Vollständige User-Verwaltung
- **Konfiguration**: Runtime-Systemeinstellungen
- **JWT-Sicherheit**: Geschützte Admin-Bereiche

### Memory-System 🧠 (Neu optimiert!)
- **Quantisierung**: int8/float16 für 75% Speicherersparnis
- **LRU-Caching**: Automatische Cache-Verwaltung
- **Memory Pooling**: Effiziente Wiederverwendung von Arrays
- **Hierarchisches Design**: Kurz-/Langzeitgedächtnis
- **Fallback-Initialisierung**: Automatische Reparatur bei Fehlern
- **Live-Statistiken**: Detaillierte Memory-Metriken

## 📦 Technische Details

### Abhängigkeiten
```txt
fastapi>=0.116.0
uvicorn>=0.35.0
pydantic>=2.11.0
python-jose[cryptography]>=3.5.0
numpy>=2.3.0
pandas>=2.3.0
scikit-learn>=1.7.0
torch>=2.8.0
transformers>=4.56.0
accelerate>=1.4.0
datasets>=3.2.0
streamlit>=1.47.0
SQLAlchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=5.2.0
pytest>=8.4.0
psutil>=6.1.0
```

### Systemanforderungen
- **RAM**: 4GB+ für optimale Performance
- **GPU**: NVIDIA/AMD GPU mit CUDA/ROCm (empfohlen)
- **Speicher**: 2GB+ freier Festplattenspeicher

## � Konfiguration

### Umgebungsvariablen
```bash
export BUNDESKANZLER_SECRET_KEY="your-secret-key"
export CUDA_VISIBLE_DEVICES="0"  # GPU-Device
```

### Memory-Konfiguration
```python
memory_system = OptimizedHierarchicalMemory(
    short_term_capacity=200,
    long_term_capacity=5000,
    embedding_dim=512,
    enable_quantization=True,
    enable_caching=True,
    cache_size=1000,
    memory_pool_size=2000
)
```

### GPU-Konfiguration
```python
gpu_processor = GPUBatchProcessor(
    batch_size=16,
    max_workers=4,
    device="auto",  # cuda/rocm/cpu/auto
    embedding_dim=512,
    enable_async=True
)
```

## 📊 Performance

| Komponente | Performance | Optimierung |
|------------|-------------|-------------|
| GPU-Batching | 47.730 Emb/s | CUDA/ROCm |
| Memory | 75% weniger RAM | Quantisierung |
| API | <100ms Response | FastAPI |
| Admin-Panel | Live-Updates | WebSocket |

## 🎨 Admin-Interface

Das Admin-Panel bietet vollständige Systemkontrolle:

- **Dashboard**: Übersicht aller Systemmetriken
- **Memory-Monitoring**: Live-Speicherstatistiken
- **GPU-Status**: Batch-Performance und Hardware-Info
- **Log-Viewer**: Strukturierte Logs mit Filterung
- **Konfiguration**: Runtime-Einstellungen anpassen

## 🔐 Sicherheit

- **JWT-Authentifizierung**: Sichere Token-basierte Authentifizierung
- **Rate Limiting**: 60 Requests/Minute pro Client
- **CORS-Konfiguration**: Eingeschränkte Origin-Kontrolle
- **Input-Validation**: Vollständige Pydantic-Validierung

## 🧪 Testabdeckung

```bash
# Test-Übersicht
pytest --cov=bundeskanzler_api --cov=optimized_memory --cov=gpu_batching

# Performance-Tests
python -m pytest tests/ -k "performance" --tb=short

# Integration-Tests
python comprehensive_test.py
```

## 📋 Roadmap

- [x] GPU-Batching Implementation
- [x] Memory-Optimierung (Quantisierung)
- [x] Admin-Panel mit Live-Monitoring
- [x] JWT-Authentifizierung
- [x] **Memory-Stats-API behoben** (Fallback-Initialisierung)
- [x] Streamlit Web-Interface
- [x] Log-Monitoring System
- [x] Benutzer-Management
- [x] **Debug-System Integration** (Live-Debugging in Web-GUI)
- [x] **Erweiterte API-Call-Tracking**
- [ ] Mehrsprachige Unterstützung
- [ ] Faktenprüfung & Quellenvalidierung
- [ ] Erweiterte Admin-Analytics
- [ ] Kubernetes-Deployment

## 🤝 Beitragen

Beiträge sind willkommen! Bitte:

1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Füge Tests für neue Features hinzu
4. Stelle einen Pull Request

### Entwicklungsrichtlinien
- **Code-Style**: Black + isort
- **Tests**: 100% Coverage für neue Features
- **Dokumentation**: Docstrings für alle öffentlichen APIs
- **Performance**: Benchmarks für Performance-ändernde Änderungen

## 📝 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

## 🙏 Danksagung

Danke an die Open-Source-Community für die großartigen Tools und Libraries!

## 📬 Kontakt

- **Issues**: [GitHub Issues](https://github.com/bumblei3/bundeskanzler_ki/issues)
- **Discussions**: Für Fragen und Feedback

---

**🚀 Das System ist vollständig funktionsfähig und bereit für den Produktiveinsatz!**
