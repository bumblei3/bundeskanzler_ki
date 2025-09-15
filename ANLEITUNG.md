# 🤖 Bundeskanzler KI - Multimodale Anleitung (RTX 2070 Edition)

**Version 2.0.0** - Optimiert für NVIDIA RTX 2070 GPUs mit 8GB VRAM

## 🚀 **Schnellstart (RTX 2070 optimiert)**

### 📋 **System-Anforderungen**
- **GPU**: NVIDIA RTX 2070 (8GB VRAM) oder besser
- **RAM**: 16GB+ empfohlen
- **Python**: 3.12+
- **Speicher**: 50GB+ freier Festplattenspeicher

### ⚡ **Installation & Start**

```bash
# 1. Arbeitsverzeichnis
cd /home/tobber/bkki_venv

# 2. Virtuelle Umgebung aktivieren
source bin/activate

# 3. RTX 2070 optimierte Modelle laden (erster Start dauert ~8 Sekunden)
python -c "from multimodal_ki import MultimodalTransformerModel; model = MultimodalTransformerModel(model_tier='rtx2070')"

# 4. API-Server starten
uvicorn bundeskanzler_api:app --host 0.0.0.0 --port 8001 --reload

# 5. Web-Interface starten (NEUES TERMINAL)
streamlit run webgui_ki.py --server.port 8501 --server.address 0.0.0.0
```

### 🌐 **Zugriff**
- **Web-Interface**: http://localhost:8501
- **API-Dokumentation**: http://localhost:8001/docs
- **Admin-Login**: `admin` / `admin123!`

## 🎮 **RTX 2070 Optimierung verstehen**

Das System ist speziell für Ihre RTX 2070 optimiert:

```python
# Automatische RTX 2070 Optimierung
from multimodal_ki import MultimodalTransformerModel

model = MultimodalTransformerModel(model_tier='rtx2070')
# ✅ 8-bit Quantisierung für alle Modelle
# ✅ GPU-Memory-Management (~774MB Verbrauch)
# ✅ Optimierte Modell-Verteilung
# ✅ Automatische Speicherbereinigung
```

### 📊 **Performance-Metriken**
- **GPU-Speicher**: 774MB / 8GB verwendet (9.7% Auslastung)
- **Ladezeit**: ~8 Sekunden für alle Modelle
- **Inference**: ~9 Sekunden pro Query
- **Stromverbrauch**: Optimiert für RTX 2070

## 🎨 **Multimodale Modi ausprobieren**

### 📝 **1. Text-Modus (Hauptfunktion)**

```bash
# Im Web-Interface (empfohlen)
# 1. Gehe zu: http://localhost:8501
# 2. Login: admin / admin123!
# 3. Wähle "Text-Chat" Modus
# 4. Frage stellen: "Was ist die Aufgabe des Bundeskanzlers?"
```

**Curl-Beispiel:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Erkläre die Bundesregierung"}'
```

### 👁️ **2. Bild-Analyse Modus**

```bash
# Web-Interface
# 1. Gehe zu "Multimodal" Tab
# 2. Wähle Bild hochladen
# 3. Frage: "Analysiere dieses politische Plakat"
```

**API-Beispiel:**
```bash
curl -X POST "http://localhost:8000/api/multimodal" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "text=Analysiere dieses Bild" \
  -F "image=@politik_plakat.jpg"
```

### 🎤 **3. Audio-Transkription Modus**

```bash
# Web-Interface
# 1. Gehe zu "Audio" Tab
# 2. Audio-Datei hochladen (.wav, .mp3)
# 3. Automatische Transkription mit Whisper
```

**Unterstützte Formate:**
- WAV, MP3, M4A, FLAC
- Deutsche und englische Sprache
- Bis zu 30 Minuten Audio

### 🔄 **4. Kombinierter Multimodal-Modus**

```python
# Erweiterte Analyse
from multimodal_ki import MultimodalTransformerModel

model = MultimodalTransformerModel(model_tier='rtx2070')

result = model.process_multimodal(
    text="Analysiere diese politische Rede",
    image="rede_plakat.jpg",      # Optional
    audio="rede_audio.wav"        # Optional
)

print(f"Multimodale Analyse: {result}")
```

## 🧠 **Kontinuierliches Lernen**

Das System lernt aus jeder Interaktion:

```python
# Feedback geben (im Web-Interface)
# 1. Nach jeder Antwort: Thumbs up/down klicken
# 2. System lernt automatisch aus Bewertungen
# 3. Modell-Verbesserung über Nacht
```

### 📈 **Lern-Features**
- **User-Feedback**: Jede Antwort kann bewertet werden
- **Kontext-Lernen**: System merkt sich Gesprächskontext
- **Performance-Tracking**: Live-Metriken im Admin-Panel
- **Automatische Optimierung**: Modell verbessert sich kontinuierlich

## 📊 **Admin-Panel Features**

### 🖥️ **Dashboard**
- **Live-Metriken**: GPU-Speicher, RAM, CPU
- **System-Status**: Modell-Ladezustand, API-Status
- **Performance-Graphen**: Response-Zeiten, Durchsatz

### 👥 **Benutzer-Management**
- **User-Verwaltung**: Hinzufügen/Löschen von Benutzern
- **Berechtigungen**: Rollenbasierte Zugriffe
- **Session-Management**: Aktive Sessions überwachen

### 📋 **Log-Monitoring**
- **Live-Logs**: Echtzeit-Logging mit Filterung
- **Fehler-Analyse**: Detaillierte Fehlerberichte
- **Performance-Logs**: Langsame Queries identifizieren

### 💾 **Memory-Management**
- **GPU-Monitoring**: RTX 2070 Speicher-Überwachung
- **Memory-Optimierung**: Automatische Speicherbereinigung
- **Cache-Management**: Temporäre Dateien verwalten

## 🔧 **Erweiterte Konfiguration**

### ⚙️ **Model-Tiers**

```python
# RTX 2070 optimiert (empfohlen für Ihre Hardware)
model = MultimodalTransformerModel(model_tier='rtx2070')

# Alternative Tiers
model_advanced = MultimodalTransformerModel(model_tier='advanced')  # Größere Modelle
model_basic = MultimodalTransformerModel(model_tier='basic')       # CPU-Only
model_premium = MultimodalTransformerModel(model_tier='premium')   # GPT-4/Claude APIs
```

### 🔑 **API-Keys für Premium-Features**

```bash
# Optional für GPT-4/Claude Integration
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"

# Dann Premium-Modus verwenden
model = MultimodalTransformerModel(model_tier='premium')
```

### 🎛️ **System-Konfiguration**

```yaml
# config.yaml bearbeiten für erweiterte Einstellungen
gpu_memory_limit: 0.9          # 90% GPU-Speicher verwenden
max_concurrent_requests: 5     # Gleichzeitige Anfragen
log_level: INFO                # Logging-Level
cache_ttl: 3600               # Cache-Gültigkeit in Sekunden
```

## 🧪 **Tests & Troubleshooting**

### ✅ **System-Tests**

```bash
# Vollständige System-Validierung
python -c "
from multimodal_ki import MultimodalTransformerModel
import torch

print('🧪 RTX 2070 System-Test...')

# Modell laden
model = MultimodalTransformerModel(model_tier='rtx2070')

# GPU-Check
gpu_mem = torch.cuda.memory_allocated() // 1024**2
print(f'✅ GPU-Speicher: {gpu_mem}MB verwendet')

# Inference-Test
response = model.process_text('Test')
print(f'✅ Inference funktioniert: {len(response)} Zeichen')

print('🎉 RTX 2070 Optimierung erfolgreich!')
"
```

### 🔍 **Problembehandlung**

#### **GPU-Speicher Fehler**
```bash
# Bei CUDA out-of-memory:
# 1. System neu starten
# 2. Weniger Tabs im Browser offen lassen
# 3. GPU-Speicher in config.yaml reduzieren
gpu_memory_limit: 0.8  # Auf 80% reduzieren
```

#### **Lange Ladezeiten**
```bash
# Modelle werden beim ersten Start heruntergeladen
# Das kann 5-10 Minuten dauern bei langsamer Internetverbindung
# Geduld haben - danach geht es schnell!
```

#### **Audio funktioniert nicht**
```bash
# Zusätzliche Audio-Abhängigkeiten installieren
pip install librosa soundfile
# FFmpeg für Audio-Konvertierung sicherstellen
sudo apt-get install ffmpeg
```

## 🐳 **Docker-Deployment (RTX 2070)**

```bash
# GPU-Version für RTX 2070 (empfohlen)
docker-compose up -d

# CPU-Fallback (langsamer)
docker-compose -f docker-compose.cpu.yml up -d

# Logs prüfen
docker-compose logs -f
```

## 📚 **API-Referenz**

### 🔐 **Authentifizierung**
```bash
# Admin-Token erhalten
curl -X POST "http://localhost:8000/auth/admin-token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123!"}'

# Token für weitere Requests verwenden
TOKEN="your_token_here"
```

### 🤖 **Haupt-Endpunkte**

#### **Text-Chat**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Ihre politische Frage"}'
```

#### **Multimodale Analyse**
```bash
curl -X POST "http://localhost:8000/api/multimodal" \
  -H "Authorization: Bearer $TOKEN" \
  -F "text=Ihre Analyse-Anfrage" \
  -F "image=@bild.jpg" \
  -F "audio=@audio.wav"
```

#### **Feedback geben**
```bash
curl -X POST "http://localhost:8000/api/feedback" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Frage",
    "ai_response": "Antwort",
    "rating": 5,
    "session_id": "session_123"
  }'
```

## 📞 **Support & Hilfe**

### 🆘 **Häufige Probleme**

**Q: System startet nicht?**
A: Virtuelle Umgebung aktivieren: `source bin/activate`

**Q: GPU wird nicht erkannt?**
A: NVIDIA-Treiber prüfen: `nvidia-smi`

**Q: Zu wenig Speicher?**
A: GPU-Memory-Limit in config.yaml reduzieren

**Q: Audio funktioniert nicht?**
A: FFmpeg installieren: `sudo apt-get install ffmpeg`

### 📧 **Kontakt**
- **Issues**: [GitHub Issues](https://github.com/bumblei3/bundeskanzler_ki/issues)
- **Dokumentation**: [Wiki](https://github.com/bumblei3/bundeskanzler_ki/wiki)
- **Email**: support@bundeskanzler-ki.de

## 🎯 **Nächste Schritte**

1. **System starten** und erste Tests durchführen
2. **Multimodale Modi** ausprobieren (Text, Bild, Audio)
3. **Admin-Panel** erkunden und System überwachen
4. **Feedback geben** um kontinuierliches Lernen zu aktivieren
5. **Konfiguration anpassen** für Ihre Bedürfnisse

## 📈 **Performance-Tipps**

- **GPU-Speicher**: ~774MB Basis-Verbrauch - genug für gleichzeitige Sessions
- **RAM-Optimierung**: System verwendet ~7GB RAM bei voller Auslastung
- **Cache**: Automatische Bereinigung verhindert Speicherlecks
- **Updates**: Regelmäßige Modell-Updates durch kontinuierliches Lernen

---

**Bundeskanzler KI v2.0.0** - RTX 2070 Edition
**Optimiert für**: NVIDIA RTX 2070 (8GB VRAM)
**Python**: 3.12+
**Letzte Aktualisierung**: September 2025

## 💬 Verschiedene Wege zum Ausprobieren

### 🎯 Methode 1: Admin-Panel (Empfohlen!)
```bash
streamlit run webgui_ki.py
```
**Features:**
- � **Dashboard**: Live-System-Metriken
- 👥 **Benutzer-Management**: User verwalten
- � **Log-Viewer**: Live-Logs mit Filterung
- 💾 **Memory-Management**: Statistiken und Verwaltung (neu behoben!)
- ⚙️ **Konfiguration**: System-Einstellungen

### 🌐 Methode 2: Web-Browser (Swagger UI)
1. API starten (siehe oben)
2. Browser öffnen: http://localhost:8000/docs
3. **Features:**
   - 📖 Automatische API-Dokumentation
   - 🧪 Direkte API-Tests im Browser
   - 🔐 Authentifizierung testen

### 🔧 Methode 3: curl Kommandos
```bash
# Health Check
curl http://localhost:8000/health

# Admin-Login
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/admin-token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123!" | jq -r .access_token)

# Memory-Stats (neu behoben!)
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/memory/stats

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Wie steht es um die Klimapolitik?"}'
```

### 🧪 Methode 4: Vollständiger Test
```bash
python direct_test.py
```

## 💡 Was du ausprobieren kannst

### 🗣️ Chat-Themen
- **Klimapolitik**: "Wie steht es um den Klimaschutz in Deutschland?"
- **Wirtschaft**: "Wie entwickelt sich die deutsche Wirtschaft?"
- **Digitalisierung**: "Welche Digitalisierungsmaßnahmen plant die Regierung?"
- **Energiewende**: "Können Sie mir zur Energiewende etwas sagen?"
- **Soziales**: "Was unternimmt die Regierung für soziale Gerechtigkeit?"

### 🧠 Memory-System
```bash
# In der Demo:
/memory Deutschland plant bis 2030 eine führende Rolle in der KI-Entwicklung
/search KI Deutschland
/stats
```

### 🔗 Webhooks testen
```bash
curl -X POST "http://localhost:8000/webhook/news_update" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Neue KI-Strategie angekündigt",
    "source": "Bundespressekonferenz",
    "timestamp": "2025-09-14T12:00:00"
  }'
```

## 🎯 Demo-Beispiele

### Beispiel-Konversation:
```
Sie: Hallo! Wie geht es Deutschland?
🤖: Vielen Dank für Ihre Anfrage. Deutschland steht vor wichtigen 
    Herausforderungen und Chancen...

Sie: /memory Deutschland investiert 10 Milliarden in KI-Forschung
✅ Erinnerung hinzugefügt!

Sie: Was plant Deutschland für Künstliche Intelligenz?
🤖: Deutschland hat eine umfassende KI-Strategie entwickelt...

Sie: /search KI Forschung
🔍 Gefunden 1 Ergebnisse:
   1. Deutschland investiert 10 Milliarden in KI-Forschung...
```

## 🔐 Anmeldedaten

### Admin-Panel (Streamlit)
- **Username**: `admin`
- **Password**: `admin123!`

### API-Zugang (für Entwickler)
- **Username**: `bundeskanzler`
- **Password**: `ki2025`

## 📊 URLs

### Admin-Panel
- **Streamlit GUI**: http://localhost:8501 (Empfohlen!)
- **HTML Admin-Panel**: http://localhost:8000/admin (Einfach)

### API-Endpunkte
- **API Base**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/health

### Admin-API (erfordert Token)
- **Login**: `POST /auth/admin-token`
- **Memory-Stats**: `GET /admin/memory/stats` ✅ (Neu behoben!)
- **System-Stats**: `GET /admin/system-stats`
- **Logs**: `GET /admin/logs/{type}`
- **Users**: `GET /admin/users`

## 🚨 Problemlösung

### API startet nicht?
```bash
# Prüfe ob Port belegt ist
lsof -i :8000

# Stoppe andere uvicorn Prozesse
pkill -f uvicorn

# Starte neu
python -m uvicorn bundeskanzler_api:app --host 0.0.0.0 --port 8000 --reload
```

### Demo verbindet nicht?
- ✅ Stelle sicher, dass die API läuft (siehe oben)
- ✅ Warte 5-10 Sekunden nach API-Start
- ✅ Prüfe dass kein Firewall die Verbindung blockiert

### Authentifizierung schlägt fehl?
- ✅ Verwende exakt: `bundeskanzler` / `ki2025`
- ✅ Prüfe dass die API vollständig gestartet ist

## 🎉 Viel Spaß beim Ausprobieren!

Die Bundeskanzler KI ist bereit für deine Fragen! 🤖